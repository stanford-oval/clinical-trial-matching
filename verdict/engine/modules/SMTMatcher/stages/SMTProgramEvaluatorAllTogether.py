from __future__ import annotations

import copy
import math
import pprint
import re
from typing import Dict, Any, Tuple, Optional, List, Set

import dspy
import z3

from ...utils.utils import _whole_program
from ...utils.mbench import get_mbench, mbench_enabled


_LOC_RE = re.compile(r"line\s+(\d+),\s*column\s+(\d+)", re.I)
_SAFE_FN_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_ENTRY_KEYS = {"value", "evidence", "explanation", "range", "assessment"}


# === DEFERABLE-ATOM PROJECTION ===
# Atoms whose name matches any of the patterns below have their LLM-mined
# values dropped before solve. Z3 then leaves them free in the model — i.e.
# the prescreen-doctrine NULL semantics applied at the SMT-LIB level.
#
# Three families of deferable atoms (matched by uniform name rule, no
# per-failure-mode patches):
#
#   * Data-availability suffixes (chart cannot establish trial-site
#     documentation; visit can): _is_documented_now, _is_available, etc.
#   * Methodology / detection / criteria-application qualifiers (visit can
#     re-confirm via the named modality / criteria-set):
#     @@detected_on_*, @@confirmed_by_*, @@diagnosed_by_*, @@by_<modality>,
#     @@biopsy_proven, @@histologically_confirmed, @@diagnosis_according_to_*,
#     @@according_to_*, @@defined_according_to_*, @@per_*_criteria, @@meets_*,
#     @@performed_from_*, @@performed_at_*, @@in_*_leads, @@minimum_duration_*,
#     @@presented_within_*, @@within_*_days, _outcome_is_*
#   * Negation-marker qualifiers (auxiliary-polarity safe; the compiler may
#     emit either polarity — leaving these free lets Z3 pick consistently):
#     @@absent_*, @@without_*, @@no_*, @@off_*, @@free_*, @@negative_*,
#     @@inactive_*, @@uncontrolled, @@unable_*, @@negated_*
#   * Aggregate-style atoms (definitional; the program defines them as a sum
#     of constituents — let the constituents drive the value):
#     _count$, _total$, _sum$, _aggregate, _index$, _overall_score,
#     _signs_count, _criteria_count, _criteria_met, _criteria_satisfied,
#     _count_value_recorded
_DEFERABLE_PATTERNS = [
    # Cat 1 — data-availability
    r"_is_documented_now$", r"_is_known_now$", r"_is_recorded_now$",
    r"_data_available_now$", r"_documentation_present_now$",
    r"_is_available$", r"_is_signed_now$",
    # Cat 1 — methodology / detection / criteria-application qualifiers
    r"@@detected_(on|by|with|using)_", r"@@confirmed_by_", r"@@diagnosed_by_",
    r"@@by_(ct|mri|biopsy|histology|imaging|ultrasound|x_ray|pet|microscopy|ekg|ecg)",
    r"@@biopsy_proven", r"@@histologically?_confirmed", r"@@histology_confirmed",
    r"@@diagnosis_according_to_", r"@@according_to_", r"@@defined_according_to_",
    r"@@per_.*_criteria", r"@@meets_",
    r"@@performed_from_", r"@@performed_at_", r"@@in_\d+_or_more_contiguous_leads",
    r"@@aatscored",
    r"_outcome_is_(normal|abnormal)$",
    r"@@minimum_duration_", r"@@presented_within_", r"@@within_\d+_days?",
    # Cat 2 — negation-marker qualifiers
    r"@@absent_", r"@@without_", r"@@no_", r"@@off_", r"@@free_",
    r"@@negative_", r"@@inactive_", r"@@uncontrolled", r"@@unable_",
    r"@@negated_", r"@@not_present", r"@@not_documented",
    # Cat 4 — aggregate-style atoms
    r"_count$", r"_total$", r"_sum$", r"_aggregat",
    r"_index$", r"_overall_score", r"_signs_count", r"_criteria_count",
    r"_criteria_met", r"_criteria_satisfied", r"_count_value_recorded",
]
_DEFERABLE_RE = re.compile("|".join(_DEFERABLE_PATTERNS))


def _is_deferable_atom(atom_name: str) -> bool:
    """True if the atom's mined value should be dropped before solve."""
    return bool(_DEFERABLE_RE.search(atom_name or ""))


def _project_deferable(pv: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Return (filtered_pv, n_dropped). Drops any LLM-mined value for atoms
    whose name matches a deferable pattern. Z3 then leaves these atoms free.

    Operates on both the shallow shape ({atom: scalar}) and the rich shape
    ({atom: {value, evidence, ...}}) — only the value field is rewritten to
    None in the rich shape; the assessment/evidence are preserved for
    downstream reporting. Multi-patient nesting (one level: {pid: {...}}) is
    handled recursively.
    """
    if not isinstance(pv, dict): return pv, 0
    # Detect multi-patient nesting: values are dicts whose own keys look like atoms.
    if pv and all(isinstance(v, dict) for v in pv.values()):
        # Heuristic: if first inner dict has _ENTRY_KEYS members, this is a single
        # patient with rich-shape values. Otherwise it's multi-patient.
        first_v = next(iter(pv.values()))
        if not (set(first_v.keys()) & _ENTRY_KEYS):
            # multi-patient: recurse
            new_pv = {}
            total_dropped = 0
            for pid, sub in pv.items():
                new_sub, n = _project_deferable(sub)
                new_pv[pid] = new_sub
                total_dropped += n
            return new_pv, total_dropped
    # Single-patient: filter directly
    out = {}
    n_dropped = 0
    for atom, info in pv.items():
        if _is_deferable_atom(atom):
            n_dropped += 1
            if isinstance(info, dict):
                # preserve metadata, just null the value
                new_info = dict(info)
                new_info["value"] = None
                new_info["_deferable_projected"] = True
                out[atom] = new_info
            else:
                # shallow scalar form — drop entirely (None as the value)
                out[atom] = None
        else:
            out[atom] = info
    return out, n_dropped

_DECL_CONST_RE = re.compile(r"\(declare-const\s+([^\s()]+)\s+([^\s()]+)\)")
_DECL_FUN0_RE = re.compile(r"\(declare-fun\s+([^\s()]+)\s+\(\s*\)\s+([^\s()]+)\)")


def _declared_sorts_from_smt(src: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _DECL_CONST_RE.finditer(src or ""):
        out[m.group(1)] = m.group(2)
    for m in _DECL_FUN0_RE.finditer(src or ""):
        out[m.group(1)] = m.group(2)
    return out


def _safe_fname(s: str) -> str:
    s = (s or "unspecified").strip()
    s = _SAFE_FN_RE.sub("_", s)
    return s[:200] if len(s) > 200 else s


def _extract_loc(msg: str) -> Tuple[int | None, int | None]:
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


def _inject_unsat_core_option(src: str) -> str:
    if ":produce-unsat-cores" in src:
        return src
    lines = src.splitlines()
    ins = "(set-option :produce-unsat-cores true)"
    if lines and lines[0].lstrip().startswith("(set-logic"):
        return "\n".join([lines[0], ins, *lines[1:]])
    return ins + "\n" + src


def test_label_individually(smt_text: str, label: str) -> str:
    tctx = z3.Context()
    ts = z3.Solver(ctx=tctx)
    ts.set(unsat_core=False)
    ts.from_string(smt_text)
    for a in ts.assertions():
        if a.num_args() == 2:
            L = a.arg(0).decl().name()
            if L != label:
                ts.add(z3.Not(z3.Bool(L, ctx=tctx)))
    ts.add(z3.Bool(label, ctx=tctx))
    r = ts.check()
    return "sat" if r == z3.sat else "unsat" if r == z3.unsat else "unknown"


def _looks_like_entry_dict(x: Any) -> bool:
    return isinstance(x, dict) and any(k in x for k in _ENTRY_KEYS)


def _looks_like_var_map(x: Any) -> bool:
    if not isinstance(x, dict) or not x:
        return False
    for vv in list(x.values())[:50]:
        if _looks_like_entry_dict(vv):
            return True
    return False


def _is_multi_patient(pv: Any) -> bool:
    if not isinstance(pv, dict) or not pv:
        return False
    if any(_looks_like_entry_dict(v) for v in pv.values()):
        return False
    return any(_looks_like_var_map(v) for v in pv.values())


def _safe_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _is_uninterpreted_const(e: z3.ExprRef) -> bool:
    try:
        return (
            z3.is_const(e)
            and e.num_args() == 0
            and e.decl().kind() == z3.Z3_OP_UNINTERPRETED
        )
    except Exception:
        return False


def _z3_symbol_name(expr: z3.ExprRef) -> Optional[str]:
    try:
        if _is_uninterpreted_const(expr):
            return expr.decl().name()
    except Exception:
        return None
    return None


def _extract_named_assertion_bodies(smt_prog: str) -> Dict[str, z3.ExprRef]:
    """
    Parse SMT and recover {assertion_tag -> assertion_body} for named assertions.

    Expected named assertion shape after z3 parsing:
        (=> TAG BODY)
    where TAG is the named label introduced by :named.
    """
    out: Dict[str, z3.ExprRef] = {}
    try:
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        s.from_string(_inject_unsat_core_option(smt_prog))
        for a in s.assertions():
            try:
                if a.num_args() == 2:
                    lhs = a.arg(0)
                    rhs = a.arg(1)
                    lbl = _z3_symbol_name(lhs)
                    if lbl:
                        out[lbl] = rhs
            except Exception:
                continue
    except Exception:
        return {}
    return out


def _extract_defined_symbol_from_body(body: z3.ExprRef) -> Optional[str]:
    """
    Recognize simple definitional assertions like:
        (= X expr)
        (= expr X)
    and return X when X is a plain symbol.
    """
    try:
        if body.decl().kind() != z3.Z3_OP_EQ or body.num_args() != 2:
            return None

        a = body.arg(0)
        b = body.arg(1)

        a_sym = _z3_symbol_name(a)
        b_sym = _z3_symbol_name(b)

        if a_sym is not None and (b_sym is None or a.sexpr() != b.sexpr()):
            return a_sym
        if b_sym is not None and (a_sym is None or a.sexpr() != b.sexpr()):
            return b_sym
    except Exception:
        return None
    return None


def _extract_aux_definitions(base_prog: str) -> Dict[str, str]:
    """
    Return {assertion_tag -> defined_symbol} for definitions considered repairable.

    Current policy:
    - only tags containing AUXILIARY or DEF are considered
    - only equality-style definitions are considered
    """
    named = _extract_named_assertion_bodies(base_prog)
    out: Dict[str, str] = {}

    for tag, body in named.items():
        tag_u = str(tag).upper()
        if "AUXILIARY" not in tag_u and "DEF" not in tag_u:
            continue
        defined = _extract_defined_symbol_from_body(body)
        if defined:
            out[tag] = defined

    return out


def _patient_label_to_var(label: str) -> Optional[str]:
    """
    Map patient assertion labels back to their variable names.

    Supports:
      patient_x
      patient_x__eq
      patient_x__min
      patient_x__max
    """
    s = str(label or "")
    if not s.startswith("patient_"):
        return None
    rest = s[len("patient_"):]

    for suffix in ("__eq", "__min", "__max"):
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return rest or None


def _drop_patient_vars(vals: Dict[str, Any], vars_to_drop: Set[str]) -> Dict[str, Any]:
    """
    Remove direct patient constraints on selected vars while preserving others.
    Works for both rich-entry dicts and flat values.
    """
    out = copy.deepcopy(vals or {})

    for v in vars_to_drop:
        if v not in out:
            continue

        entry = out[v]
        if isinstance(entry, dict):
            entry["value"] = None
            entry["range"] = None
            prev = str(entry.get("assessment") or "").strip()
            note = "Dropped direct patient assignment because AUX/DEF assertion defines this variable."
            entry["assessment"] = f"{prev} | {note}" if prev else note
        else:
            out[v] = None

    return out


class SMTProgramEvaluator(dspy.Module):
    """
    Whole-program evaluator.

    IMPORTANT:
    - Does NOT rely on context["variable_index"].
    - Sorts are derived from the final SMT program declarations.
    - Missing bools stay missing; no default-false behavior.
    - Numeric range asserts are supported.
    - Strict numeric bounds are supported via:
        {"min": ..., "max": ..., "min_strict": bool, "max_strict": bool}

    NEW:
    - Optional unsat repair pass:
      If the unsat core contains an AUXILIARY/DEF assertion that definitionally
      determines a symbol X, and the core also contains a patient assertion on X,
      drop the direct patient assertion on X and re-solve. This lets the defining
      variables decide X.
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @staticmethod
    def _literal(sort: str, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return SMTProgramEvaluator._literal(sort, value.get("value"))
        s = (sort or "").lower()
        if s == "bool":
            return "true" if str(value).lower() in {"true", "1", "t", "yes"} else "false"
        if s == "string":
            return f'"{value}"'
        if s in {"int", "integer"}:
            return str(int(value))
        if s == "real":
            try:
                f = float(value)
                return f"{int(f)}.0" if f.is_integer() else str(f)
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _range_literals(sort: str, r: Any) -> Tuple[Optional[str], Optional[str], bool, bool]:
        if not isinstance(r, dict):
            return None, None, False, False

        s = (sort or "").lower()
        mn = _safe_num(r.get("min"))
        mx = _safe_num(r.get("max"))
        mn_strict = bool(r.get("min_strict", False))
        mx_strict = bool(r.get("max_strict", False))

        if s in {"int", "integer"}:
            mn_i = None if mn is None else int(math.ceil(mn))
            mx_i = None if mx is None else int(math.floor(mx))

            if mn_i is not None and mn_strict:
                mn_i += 1
                mn_strict = False
            if mx_i is not None and mx_strict:
                mx_i -= 1
                mx_strict = False

            min_lit = None if mn_i is None else str(mn_i)
            max_lit = None if mx_i is None else str(mx_i)
            return min_lit, max_lit, False, False

        if s == "real":
            def _r_lit(x: Optional[float]) -> Optional[str]:
                if x is None:
                    return None
                return f"{int(x)}.0" if float(x).is_integer() else str(float(x))

            return _r_lit(mn), _r_lit(mx), mn_strict, mx_strict

        return None, None, False, False

    def _assert_block(
        self,
        vals: Dict[str, Any],
        declared_sorts: Dict[str, str],
        *,
        enable_numeric_ranges: bool,
    ) -> str:
        if _looks_like_entry_dict(vals):
            return ""

        lines = []
        for v in sorted((vals or {}).keys()):
            val = (vals or {}).get(v)

            sort = declared_sorts.get(v)
            if not sort:
                continue

            raw_val = val.get("value") if isinstance(val, dict) else val

            if raw_val is not None:
                lit = self._literal(sort, raw_val)
                if lit is None:
                    continue
                lines.append(f"(assert (! (= {v} {lit}) :named patient_{v}))")
                continue

            if enable_numeric_ranges and isinstance(val, dict):
                r = val.get("range")
                min_lit, max_lit, min_strict, max_strict = self._range_literals(sort, r)

                if (
                    min_lit is not None
                    and max_lit is not None
                    and min_lit == max_lit
                    and not min_strict
                    and not max_strict
                ):
                    lines.append(f"(assert (! (= {v} {min_lit}) :named patient_{v}__eq))")
                    continue

                if min_lit is not None:
                    cmp_op = ">" if min_strict else ">="
                    lines.append(f"(assert (! ({cmp_op} {v} {min_lit}) :named patient_{v}__min))")
                if max_lit is not None:
                    cmp_op = "<" if max_strict else "<="
                    lines.append(f"(assert (! ({cmp_op} {v} {max_lit}) :named patient_{v}__max))")

        return "\n".join(lines)

    @staticmethod
    def _solve(smt_prog: str) -> Dict[str, Any]:
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        s.set(unsat_core=True, ignore_labels=False)
        try:
            smt_prog = _inject_unsat_core_option(smt_prog)
            s.from_string(smt_prog)
            all_R = {
                a.arg(0).decl().name()
                for a in s.assertions()
                if a.num_args() == 2
                and a.arg(0).decl().arity() == 0
                and a.arg(0).decl().range().kind() == z3.Z3_BOOL_SORT
                and a.arg(0).decl().name().startswith("R")
            }
        except z3.Z3Exception as e:
            msg = str(e)
            line, col = _extract_loc(msg)
            return {
                "status": "error",
                "message": msg,
                "line": line,
                "column": col,
                "snippet": _snippet(smt_prog, line),
            }

        st = s.check()
        res = {"status": str(st)}

        if st == z3.sat:
            mdl = s.model()
            res["model"] = {d.name(): str(mdl[d]) for d in mdl.decls()}
            res["label_status"] = {"sat": sorted(all_R), "unsat": [], "unknown": []}

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

        else:
            res["reason_unknown"] = s.reason_unknown()
            res["label_status"] = {"sat": [], "unsat": [], "unknown": sorted(all_R)}

        return res

    def _repair_unsat_by_aux_defs(
        self,
        result: Dict[str, Any],
        vals: Dict[str, Any],
        declared_sorts: Dict[str, str],
        base_prog: str,
        *,
        allow_bare: bool,
        verbose: bool,
        enable_numeric_ranges: bool,
        max_rounds: int = 3,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        import sys

        def _dbg(msg: str):
            if verbose:
                print(msg, file=sys.stderr)

        current_vals = copy.deepcopy(vals or {})
        current_prog = (
            f"{base_prog}\n\n;;; ---- patient values ----\n"
            f"{self._assert_block(current_vals, declared_sorts, enable_numeric_ranges=enable_numeric_ranges)}\n"
        )

        aux_defs = _extract_aux_definitions(base_prog)
        if not aux_defs:
            return result, current_prog

        original_result = copy.deepcopy(result)
        dropped_all: Set[str] = set()
        last_result = result
        last_prog = current_prog

        for round_idx in range(max_rounds):
            if last_result.get("status") != "unsat":
                break

            core = set(last_result.get("unsat_core") or [])
            if not core:
                break

            patient_core_vars = {
                v for v in (_patient_label_to_var(lbl) for lbl in core) if v
            }

            vars_to_drop: Set[str] = set()
            trigger_tags: List[str] = []

            for tag, defined_var in aux_defs.items():
                if tag in core and defined_var in patient_core_vars:
                    vars_to_drop.add(defined_var)
                    trigger_tags.append(tag)

            if not vars_to_drop:
                break

            _dbg(
                f"[SMTProgramEvaluator] AUX/DEF repair round {round_idx + 1}: "
                f"dropping direct patient assignments for {sorted(vars_to_drop)} "
                f"triggered by {sorted(trigger_tags)}"
            )

            current_vals = _drop_patient_vars(current_vals, vars_to_drop)
            dropped_all.update(vars_to_drop)

            repaired_block = self._assert_block(
                current_vals,
                declared_sorts,
                enable_numeric_ranges=enable_numeric_ranges,
            )

            if not repaired_block and allow_bare:
                repaired_prog = base_prog
            elif not repaired_block and not allow_bare:
                repaired_prog = base_prog
            else:
                repaired_prog = f"{base_prog}\n\n;;; ---- patient values ----\n{repaired_block}\n"

            repaired_result = self._solve(repaired_prog)
            repaired_result["repair_applied"] = True
            repaired_result["repair_type"] = "drop_patient_values_for_aux_or_def_defined_symbols"
            repaired_result["repair_round"] = round_idx + 1
            repaired_result["dropped_patient_vars_due_to_aux_defs"] = sorted(dropped_all)
            repaired_result["original_status_before_repair"] = original_result.get("status")
            repaired_result["original_unsat_core_before_repair"] = list(original_result.get("unsat_core") or [])
            repaired_result["repair_trigger_tags_last_round"] = sorted(trigger_tags)

            last_result = repaired_result
            last_prog = repaired_prog

        return last_result, last_prog

    def _evaluate_one(
        self,
        vals: Dict[str, Any],
        declared_sorts: Dict[str, str],
        base_prog: str,
        *,
        allow_bare: bool,
        verbose: bool,
        enable_numeric_ranges: bool,
        enable_aux_def_repair: bool,
        aux_def_repair_max_rounds: int,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        import sys

        def _dbg(msg):
            if verbose:
                print(msg, file=sys.stderr)

        block = self._assert_block(vals, declared_sorts, enable_numeric_ranges=enable_numeric_ranges)

        if not block and allow_bare:
            _dbg("[SMTProgramEvaluator] No patient values/ranges; evaluating bare SMT.")
            prog = base_prog
            return self._solve(prog), prog

        if not block and not allow_bare:
            return {"status": "unknown", "message": "no concrete values"}, base_prog

        combined = f"{base_prog}\n\n;;; ---- patient values ----\n{block}\n"
        _dbg("[SMTProgramEvaluator] Solving combined program (SMT + patient assignments/ranges).")
        result = self._solve(combined)

        if enable_aux_def_repair and result.get("status") == "unsat":
            repaired_result, repaired_prog = self._repair_unsat_by_aux_defs(
                result,
                vals,
                declared_sorts,
                base_prog,
                allow_bare=allow_bare,
                verbose=verbose,
                enable_numeric_ranges=enable_numeric_ranges,
                max_rounds=aux_def_repair_max_rounds,
            )
            return repaired_result, repaired_prog

        return result, combined

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import sys

        def _dbg(msg):
            if context.get("VERBOSE"):
                print(msg, file=sys.stderr)

        _dbg("► SMTProgramEvaluator: enforcing patient-specific values")

        mb = get_mbench(context)
        log_enabled = mbench_enabled(context)
        save_prog = bool(context.get("SAVE_EVAL_SMT_PROGRAM", True))

        base_prog = _whole_program(context) or ""
        declared_sorts = _declared_sorts_from_smt(base_prog)

        enable_numeric_ranges = bool(context.get("ENABLE_NUMERIC_RANGE_ASSERTS", True))
        prefer_rich = bool(context.get("EVAL_USE_RICH_PATIENT_VALUES", enable_numeric_ranges))

        enable_aux_def_repair = bool(context.get("ENABLE_AUX_DEF_UNSAT_REPAIR", True))
        aux_def_repair_max_rounds = int(context.get("AUX_DEF_UNSAT_REPAIR_MAX_ROUNDS", 3) or 3)

        pv_rich = context.get("patient_var_values_rich", None)
        pv_flat = context.get("patient_var_values", {}) or {}

        pv = pv_rich if (prefer_rich and isinstance(pv_rich, dict) and pv_rich) else pv_flat

        # Project deferable atoms to NULL (drop their mined value) so Z3 leaves
        # them free during solve. Two ways to compute the projection set:
        #   (a) An LLM-supplied list passed in via context["DEFERABLE_ATOMS_LIST"]
        #       (preferred — generalizable; the LLM applies the prescreen-NULL
        #       principle to this trial's atoms).
        #   (b) Fallback regex name-rule via _project_deferable() (a leaky proxy
        #       for the principle, kept for ablation / no-LLM mode).
        #
        # Default OFF — opt in via PROJECT_DEFERABLE_ATOMS=True. When the LLM
        # list is present, prefer it.
        if context.get("PROJECT_DEFERABLE_ATOMS", False):
            llm_list = context.get("DEFERABLE_ATOMS_LIST")
            if isinstance(llm_list, (list, set)):
                allowed = set(llm_list)
                if isinstance(pv, dict):
                    new_pv = {}; n_dropped = 0
                    for atom, info in pv.items():
                        if atom in allowed:
                            n_dropped += 1
                            if isinstance(info, dict):
                                new_info = dict(info); new_info["value"] = None
                                new_info["_deferable_projected"] = True
                                new_pv[atom] = new_info
                            else:
                                new_pv[atom] = None
                        else:
                            new_pv[atom] = info
                    pv = new_pv
                    if n_dropped and bool(context.get("VERBOSE")):
                        print(f"  [DEFERABLE/LLM] dropped {n_dropped} mined values", flush=True)
            else:
                pv, _n = _project_deferable(pv)
                if _n and bool(context.get("VERBOSE")):
                    print(f"  [DEFERABLE/regex-fallback] dropped {_n} mined values", flush=True)

        allow_bare = bool(context.get("WHOLE_PROGRAM", False))
        side = _safe_fname(str(context.get("inc_exc", "unknown")))

        multi = _is_multi_patient(pv)

        def _log_prog(pid: str, prog: str | None):
            if not prog:
                return
            if log_enabled:
                fname = f"{_safe_fname(pid)}__{side}__program_with_values.smt2"
                mb.log_text("SMTProgramEvaluator", fname, prog)

        if not multi:
            result, prog = self._evaluate_one(
                pv,
                declared_sorts,
                base_prog,
                allow_bare=allow_bare,
                verbose=bool(context.get("VERBOSE")),
                enable_numeric_ranges=enable_numeric_ranges,
                enable_aux_def_repair=enable_aux_def_repair,
                aux_def_repair_max_rounds=aux_def_repair_max_rounds,
            )
            context["eval_result"] = result
            if save_prog:
                context["eval_smt_program"] = prog or ""

            pid_for_log = context.get("patient_id", "patient")
            _log_prog(pid_for_log, prog)

            _dbg("   full solver result:")
            if context.get("VERBOSE"):
                pprint.pprint(result, width=100, sort_dicts=False, stream=sys.stderr)
                print(file=sys.stderr)

            return context

        all_results: Dict[str, Any] = {}
        all_progs: Dict[str, str] = {}

        for pid, vals in pv.items():
            res, prog = self._evaluate_one(
                vals,
                declared_sorts,
                base_prog,
                allow_bare=allow_bare,
                verbose=bool(context.get("VERBOSE")),
                enable_numeric_ranges=enable_numeric_ranges,
                enable_aux_def_repair=enable_aux_def_repair,
                aux_def_repair_max_rounds=aux_def_repair_max_rounds,
            )
            all_results[pid] = res
            if save_prog:
                all_progs[pid] = prog or ""
            _log_prog(pid, prog)

            _dbg(f"   [{pid}]")
            if context.get("VERBOSE"):
                pprint.pprint(res, width=100, sort_dicts=False, stream=sys.stderr)

        context["eval_result"] = all_results
        if save_prog:
            context["eval_smt_program"] = all_progs
        _dbg("")
        return context