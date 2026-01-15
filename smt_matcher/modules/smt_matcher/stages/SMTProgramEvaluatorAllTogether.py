from __future__ import annotations

import re, time
from typing import Dict, Any, List, Tuple

import dspy, z3
from smt_core.utils.z3_helpers import _whole_program

import pprint

def _pretty(res: Dict[str, Any]):
    pprint.pprint(res, width=100, sort_dicts=False)

# ────────────────────────────────────────────────────────────────────
#  tiny helpers — keep *outside* the class so any method can use them
# ────────────────────────────────────────────────────────────────────
_LOC_RE = re.compile(r"line\s+(\d+),\s*column\s+(\d+)", re.I)

def _extract_loc(msg: str) -> Tuple[int | None, int | None]:
    """(line, col) <- “… line 42 column 7 …”  — both None if not present."""
    if m := _LOC_RE.search(msg):
        return int(m.group(1)), int(m.group(2))
    return None, None

def _snippet(src: str, line: int | None, span: int = 2) -> str | None:
    if line is None:
        return None
    lines = src.splitlines()
    lo, hi = max(1, line - span), min(len(lines), line + span)
    out = []
    for i in range(lo, hi + 1):
        prefix = ">>" if i == line else "  "
        out.append(f"{prefix} {i:4d} | {lines[i-1]}")
    return "\n".join(out)

# ─────────────────── strategy / outcome history helpers ─────────────
def _push_strategy(context: dict, comment: str, outcome: str) -> None:
    hist = context.setdefault("strategy_outcome_history", [])
    hist.append({"timestamp": time.time(), "comment": comment, "outcome": outcome})

def _format_strategy_map(hist: List[dict], limit: int = 5) -> str:
    if not hist:
        return "_none yet_"
    rows = [f"* {h['comment']}  ⇒  {h['outcome']}" for h in hist[-limit:][::-1]]
    return "\n".join(rows)

def show_labels(s: z3.Solver) -> None:
    for i, a in enumerate(s.assertions(), 1):
        sexp = a.sexpr()
        # labelled if it still contains a :named (rare) *or*
        # if it is an implication whose antecedent is a symbol
        has_label = " :named " in sexp or sexp.lstrip().startswith("(=> ")
        print(f"[{i:02}] labelled={has_label} {sexp}")

def refine_label_status(s: z3.Solver, all_R: set[str], core: set[str]) -> dict[str, list[str]]:
    sat, unsat, unknown = set(), set(core), set()

    for lbl in sorted(all_R - core):
        s.push()
        try:
            # create the Bool in *this* solver’s context
            s.add(z3.Bool(lbl, ctx=s.ctx))     # ← change here
            chk = s.check()
            if chk == z3.sat:
                sat.add(lbl)
            elif chk == z3.unsat:
                unsat.add(lbl)
            else:                              # still unknown
                unknown.add(lbl)
        finally:
            s.pop()

    return {
        "sat":     sorted(sat),
        "unsat":   sorted(unsat),
        "unknown": sorted(unknown)
    }



def _inject_unsat_core_option(src: str) -> str:
    """
    Ensure the SMT-LIB script contains
        (set-option :produce-unsat-cores true)
    immediately after the first `(set-logic …)` line.
    Does nothing if the option is already present.
    """
    if ":produce-unsat-cores" in src:
        return src

    lines = src.splitlines()
    insertion = "(set-option :produce-unsat-cores true)"
    if lines and lines[0].lstrip().startswith("(set-logic"):
        return "\n".join([lines[0], insertion, *lines[1:]])
    # fall-back: prepend if there is no set-logic line up front
    return insertion + "\n" + src


def test_label_individually(smt_text: str, label: str) -> str:
    tctx = z3.Context()
    ts   = z3.Solver(ctx=tctx)
    ts.set(unsat_core=False)                # no need for cores here
    ts.from_string(smt_text)

    # *disable* every label by adding (not L) for L ≠ label
    for a in ts.assertions():
        if a.num_args() == 2:
            L = a.arg(0).decl().name()
            if L != label:
                ts.add(z3.Not(z3.Bool(L, ctx=tctx)))

    # now assert the one label we are testing
    ts.add(z3.Bool(label, ctx=tctx))

    r = ts.check()
    return "sat" if r == z3.sat else "unsat" if r == z3.unsat else "unknown"



# ────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ────────────────────────────────────────────────────────────────────
class SMTProgramEvaluator(dspy.Module):
    """
    Evaluate the generated SMT-LIB program on one or more *concrete* patient
    assignments supplied in `context["patient_var_values"]`.

    Results (per-patient or aggregate) are stored under `context["eval_result"]`.
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine          # symmetry only

    # ---------- literal helpers -------------------------------------
    @staticmethod
    def _literal(sort: str, value):
        """Convert Python value -> SMT literal.
        Supports wrapper dicts like {"value": 45, "evidence": "..."}."""
        if value is None:
            return None
        if isinstance(value, dict):              # NEW
            return SMTProgramEvaluator._literal(sort, value.get("value"))
        s = sort.lower()
        if s == "bool":
            return "true" if str(value).lower() in {"true", "1", "t", "yes"} else "false"
        if s == "string":
            return f'"{value}"'
        if s in {"int", "integer", "real"}:
            return str(value)
        return str(value)                        # assume enum/datatype ctor

    # ---------- build concrete-value assertion block ----------------
    def _assert_block(self, vals: Dict[str, Any], vindex: Dict[str, Dict]) -> str:
        lines = []
        for v, val in vals.items():
            raw_val = val.get("value") if isinstance(val, dict) else val
            if raw_val is None:
                continue                    # skip “unknown” variables
            vinfo = vindex.get(v)
            if vinfo is None:
                continue                      # skip undeclared sym
            lit = self._literal(vinfo.get("type", ""), val)
            if lit is None:
                continue
            lines.append(f"(assert (! (= {v} {lit}) :named patient_{v}))")
        return "\n".join(lines)

    # ---------- SAFE solver wrapper  --------------------------------
    @staticmethod
    def _solve(smt_prog: str) -> Dict[str, Any]:
        ctx = z3.Context()
        s   = z3.Solver(ctx=ctx)
        # always request unsat cores; harmless if the problem becomes SAT
        s.set(unsat_core=True, ignore_labels=False)
        # s.set(':produce-unsat-cores', True)  

        # (1) parse
        # try:
        #     smt_prog = _inject_unsat_core_option(smt_prog)
        #     # print(f"smt_prog \n {smt_prog}")
        #     s.from_string(smt_prog)
            # show_labels(s)

        try:
            smt_prog = _inject_unsat_core_option(smt_prog)
            s.from_string(smt_prog)

            # ── NEW ── collect every Boolean label that begins with "R"
            all_R = {
                a.arg(0).decl().name()
                for a in s.assertions()
                if a.num_args() == 2               # looks like (=> label φ)
                and a.arg(0).decl().arity() == 0
                and a.arg(0).decl().range().kind() == z3.Z3_BOOL_SORT
                and a.arg(0).decl().name().startswith("R")
            }
        except z3.Z3Exception as e:
            msg = str(e)
            line, col = _extract_loc(msg)
            return {
                "status":  "error",
                "message": msg,
                "line":    line,
                "column":  col,
                "snippet": _snippet(smt_prog, line),
            }

        # (2) check
        st  = s.check()
        res = {"status": str(st)}

        # ---------- SAT -------------------------------------------------
        if st == z3.sat:
            mdl = s.model()
            res["model"] = {d.name(): str(mdl[d]) for d in mdl.decls()}
            res["label_status"] = {
                "sat":     sorted(all_R),
                "unsat":   [],
                "unknown": []
            }

        # ---------- UNSAT ----------------------------------------------
        elif st == z3.unsat:
            core = {c.sexpr() for c in s.unsat_core()}
            res["unsat_core"] = sorted(core)

            label_status = {"sat": [], "unsat": [], "unknown": []}
            for lbl in all_R:
                if lbl in core:
                    label_status["unsat"].append(lbl)
                else:
                    outcome = test_label_individually(smt_prog, lbl)
                    label_status[outcome].append(lbl)
            res["label_status"] = {k: sorted(v) for k, v in label_status.items()}


        # ---------- UNKNOWN --------------------------------------------
        else:
            res["reason_unknown"] = s.reason_unknown()
            res["label_status"] = {
                "sat":     [],
                "unsat":   [],
                "unknown": sorted(all_R)
            }

        return res

    # ---------- evaluate a single patient ---------------------------
    def _evaluate_one(self,
                    vals: Dict[str, Any],
                    vindex: Dict[str, Dict],
                    base_prog: str) -> Dict[str, Any]:

        block = self._assert_block(vals, vindex)
        if not block:
            return {"status": "unknown", "message": "no concrete values"}

        combined = f"{base_prog}\n\n;;; ---- patient values ----\n{block}\n"

        # ── NEW: print the *exact* SMT-LIB we hand to Z3 ───────────────
        print("   ── SMT program sent to solver ──")
        print(combined)
        print("   ────────────────────────────────")
        # --------------------------------------------------------------

        return self._solve(combined)

    # ---------- public entry point ----------------------------------
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("► SMTProgramEvaluator: enforcing patient-specific values")
        
        base_prog  = _whole_program(context)

        
        vindex     = context.get("variable_index", {})
        pv         = context.get("patient_var_values", {})

        # single vs multi-patient detection
        multi = (
            isinstance(pv, dict)
            and pv
            and isinstance(next(iter(pv.values())), dict)
            and not (set(pv.keys()) <= set(vindex.keys()))
        )

        if not multi:
            result = self._evaluate_one(pv, vindex, base_prog)
            context["eval_result"] = result
            print("   full solver result:")
            _pretty(result)
            print()
            return context

        # batch mode
        all_results = {}
        for pid, vals in pv.items():
            res = self._evaluate_one(vals, vindex, base_prog)
            all_results[pid] = res
            print(f"   [{pid}]")
            _pretty(res)

        context["eval_result"] = all_results
        print()
        return context