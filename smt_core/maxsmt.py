r"""MaxSMT accountability artifacts — Steps 2-6 of the VERDICT algorithm.

Implements the published formulation (VERDICT, EMNLP update):

    S      the patient constraints: /\\_i {c_i = v_i} for every OBSERVED or
           IMPUTED condition.
    Step 2 (z, gamma) = SMT(phi_t /\\ S).  d = ELIGIBLE iff z = SAT.
    Step 3 delta_E = {s in S | MAXSMT(phi_t /\\ S, W).s = FALSE}
           i.e. which asserted patient facts the solver had to give up to reach
           ELIGIBLE.  delta_E = {} iff d = ELIGIBLE.
    Step 4 rho = /\\_i {c_i = MAXSMT(phi_t /\\ S, W).c_i | q_i = UNRESOLVED}
           the *assumptions*: the values the solver assigns to conditions the
           chart never settled.
    Step 5 delta_I = {x in S /\\ rho | MAXSMT(-phi_t /\\ S /\\ rho, W').x = FALSE}
           delta_I = {} iff d = INELIGIBLE.
    Step 6 delta = delta_I if d = ELIGIBLE else delta_E.

Terminology note. Up to the submitted version rho was called *residual
constraints* and held unresolved conditions as constraints ({crcl >= 60}).
It is now *assumptions* and holds the witness the solver assigns
({crcl = 72}). The two are not interchangeable: one is a requirement, the
other one satisfying value of it.

Verbalizer rule (Step 7). A witness is arbitrary within the satisfying
region, so a rationale must report the *requirement* phi_t imposes on a
numeric condition, not the witness. `requirement_for()` recovers that bound
from the program text; `Artifacts.for_verbalizer()` applies it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

OBSERVED, IMPUTED, UNRESOLVED = "OBSERVED", "IMPUTED", "UNRESOLVED"
#: the submitted paper called IMPUTED conditions ASSUMED; accepted as an alias
ASSUMED = "ASSUMED"
ELIGIBLE, INELIGIBLE = "eligible", "ineligible"

#: Which published formulation to run.
RESIDUAL = "residual"   # submitted: rho = the UNRESOLVED conditions themselves
MAXSMT = "maxsmt"       # update:    rho = the values MAXSMT assigns to them
VERSIONS = (RESIDUAL, MAXSMT)


# --------------------------------------------------------------------------
# Paper mapping
#
# Every artifact this module produces maps to a symbol in the paper. Keep this
# table next to the code so the two cannot drift.
#
SYMBOLS = {
    "d": {
        "symbol": "d", "field": "decision",
        "meaning": "eligibility decision, ELIGIBLE iff SMT(phi_t /\\ S) is SAT",
        "step": "Step 2", "versions": (RESIDUAL, MAXSMT)},
    "gamma": {
        "symbol": "gamma", "field": "trace",
        "meaning": "decision trace: the derivation of d from the inputs",
        "step": "Step 2", "versions": (RESIDUAL, MAXSMT)},
    "rho": {
        "symbol": "rho", "field": "assumptions",
        "meaning": {
            RESIDUAL: "residual constraints: the conditions of phi_t that remain "
                      "UNRESOLVED after evidence and policy (a requirement)",
            MAXSMT:   "assumptions: the value MAXSMT assigns to each UNRESOLVED "
                      "condition (a witness, arbitrary within the satisfying region)",
        },
        "step": {RESIDUAL: "Step 3", MAXSMT: "Step 4"},
        "versions": (RESIDUAL, MAXSMT)},
    "delta": {
        "symbol": "delta", "field": "pivotal",
        "meaning": "pivotal conditions: what would have to change to flip d",
        "step": {RESIDUAL: "Step 4", MAXSMT: "Step 6"},
        "versions": (RESIDUAL, MAXSMT)},
    "delta_E": {
        "symbol": "delta_E", "field": "delta_e",
        "meaning": "conditions to change to render the decision ELIGIBLE; "
                   "empty iff d = ELIGIBLE",
        "step": "Step 3", "versions": (MAXSMT,)},
    "delta_I": {
        "symbol": "delta_I", "field": "delta_i",
        "meaning": "conditions to change to render the decision INELIGIBLE; "
                   "empty iff d = INELIGIBLE",
        "step": "Step 5", "versions": (MAXSMT,)},
}

#: which published PDF each version corresponds to
PAPER_OF = {RESIDUAL: "submitted", MAXSMT: "update"}


@dataclass
class Condition:
    """One condition c_i with its resolved value v_i and status q_i."""
    name: str
    value: Any = None                    # v_i; None when UNRESOLVED
    status: str = UNRESOLVED             # q_i
    policy: Optional[str] = None         # a_i, the policy realization if IMPUTED

    @property
    def resolved(self) -> bool:
        return self.status in (OBSERVED, IMPUTED, ASSUMED)


@dataclass
class Artifacts:
    """The accountability artifacts of one decision."""
    decision: str                                   # d
    trace: str                                      # gamma
    assumptions: Dict[str, Any] = field(default_factory=dict)   # rho
    pivotal: List[str] = field(default_factory=list)            # delta
    delta_e: List[str] = field(default_factory=list)
    delta_i: List[str] = field(default_factory=list)
    status: str = "ok"
    version: str = MAXSMT           # which formulation produced these


    def to_paper(self, *, include_glossary: bool = True) -> Dict[str, Any]:
        """Export the artifacts keyed by their symbol in the paper.

            >>> a.to_paper()["rho"]["value"]
            {'crcl': 60.0}

        Symbols that do not exist in this version (delta_E / delta_I under
        RESIDUAL) are omitted rather than emitted empty, so the export cannot
        be mistaken for "computed and found empty".
        """
        out: Dict[str, Any] = {
            "version": self.version,
            "paper": PAPER_OF[self.version],
            "status": self.status,
        }
        for sym, spec in SYMBOLS.items():
            if self.version not in spec["versions"]:
                continue
            entry: Dict[str, Any] = {"value": getattr(self, spec["field"]),
                                     "field": spec["field"]}
            if include_glossary:
                meaning = spec["meaning"]
                step = spec["step"]
                entry["meaning"] = (meaning[self.version]
                                    if isinstance(meaning, dict) else meaning)
                entry["step"] = (step[self.version]
                                 if isinstance(step, dict) else step)
            out[sym] = entry
        return out

    def describe(self) -> str:
        """One-screen human summary with the paper symbols attached."""
        d = self.to_paper()
        lines = [f"VERDICT artifacts  [{d['version']} = {d['paper']} paper]"]
        for sym in ("d", "gamma", "rho", "delta", "delta_E", "delta_I"):
            if sym not in d:
                continue
            lines.append(f"  {sym:<8s} ({d[sym]['step']:<7s}) {d[sym]['value']}")
        return "\n".join(lines)

    def assumptions_report(self, phi_lines: Sequence[str], *,
                           labels: Optional[Dict[str, str]] = None
                           ) -> List[Dict[str, Any]]:
        """rho as statements a person can act on, one per assumption.

        Each record says what was taken for granted, that the chart did not
        supply it, whether it decided the outcome, and what to check. The
        numeric `witness` is carried but never phrased as a finding: under
        MAXSMT it is an arbitrary point in the satisfying region, so
        "creatinine clearance 60" would invent a lab result.

            >>> r = a.assumptions_report(phi)[0]
            >>> r["statement"]
            'Assumed: crcl is at least 60.'
            >>> r["basis"]
            'Not found in the chart.'
        """
        labels = labels or {}
        out: List[Dict[str, Any]] = []
        for name, witness in self.assumptions.items():
            req = requirement_for(name, phi_lines)
            # RESIDUAL already stores the requirement as the value
            if req is None and isinstance(witness, str) and witness[:1] in "<>=":
                req = witness
                witness = None
            label = labels.get(name) or humanize(name)
            pivotal = name in self.pivotal
            rec = {
                "condition": name,
                "label": label,
                "requirement": req,
                "statement": "Assumed: " + _sentence(label, req, witness) + ".",
                "basis": "Not found in the chart.",
                "pivotal": pivotal,
                "action": ("Verify before acting -- this assumption decided the "
                           f"outcome: {label}."
                           if pivotal else f"Confirm {label} when convenient."),
            }
            if witness is not None and req is not None:
                #: solver witness, retained for audit; not a finding
                rec["witness"] = witness
            elif witness is not None:
                rec["value"] = witness       # boolean: the witness IS the fact
            out.append(rec)
        return out

    def render_assumptions(self, phi_lines: Sequence[str], *,
                           labels: Optional[Dict[str, str]] = None) -> str:
        """The report as plain text, pivotal assumptions first."""
        recs = self.assumptions_report(phi_lines, labels=labels)
        if not recs:
            return "No assumptions: every condition was resolved from the chart."
        recs.sort(key=lambda r: (not r["pivotal"], r["label"]))
        lines = []
        for r in recs:
            mark = "!" if r["pivotal"] else "-"
            lines.append(f"{mark} {r['statement']} {r['basis']}")
            lines.append(f"    {r['action']}")
        return "\n".join(lines)

    def for_verbalizer(self, phi_lines: Sequence[str]) -> Dict[str, Any]:
        """Step 7 view: numeric conditions reported as requirements, not witnesses.

        The solver's witness for an unresolved numeric condition is arbitrary
        within the satisfying region, so surfacing it ("creatinine clearance
        72") would invent a fact. We surface the requirement instead
        ("creatinine clearance >= 60").
        """
        out_rho = {}
        for name, witness in self.assumptions.items():
            req = requirement_for(name, phi_lines)
            out_rho[name] = {"requirement": req, "witness": witness} if req \
                else {"value": witness}
        return {
            "decision": self.decision,
            "trace": self.trace,
            "assumptions": out_rho,
            "pivotal": [{"condition": c, "requirement": requirement_for(c, phi_lines)}
                        for c in self.pivotal],
        }


# --------------------------------------------------------------------------
# Making an assumption useful to a person
#
# `rho` is an SMT fact: `crcl = 60.0`, or at best `crcl >= 60`. Neither tells
# a clinician what was taken for granted, whether it came from the chart, or
# what to check before acting. These helpers turn one into a sentence that
# does, using only the program text -- no model call, so the rendering cannot
# introduce a claim the solver did not make.

#: comparator -> how a person says it
_PHRASING = {">=": "at least", "<=": "at most", ">": "greater than",
             "<": "less than", "=": "equal to"}

#: name fragments that carry no meaning for a reader
_NOISE = ("patient_", "_value_recorded_now", "_recorded_now", "_inthehistory",
          "_in_years", "has_finding_of_", "_flag", "_status")


def humanize(name: str) -> str:
    """A readable label for an SMT variable name.

        >>> humanize("patient_age_value_recorded_now_in_years")
        'age'
        >>> humanize("__THRESH__::patient_crcl::gt::60")
        'crcl'
    """
    if name.startswith("__THRESH__::"):
        parts = name.split("::")
        name = parts[1] if len(parts) > 1 else name
    for frag in _NOISE:
        name = name.replace(frag, " ")
    return " ".join(name.replace("_", " ").split()) or name


def _sentence(label: str, requirement: Optional[str],
              witness: Any = None) -> str:
    """'crcl', '>= 60' -> 'crcl is at least 60'.

    For a boolean the witness carries the polarity, and getting it backwards
    would state the opposite of what was assumed: a program asserting
    `(not on_warfarin)` has witness False, i.e. the patient was assumed NOT
    to be on warfarin. With no witness (RESIDUAL stores none) the polarity is
    genuinely unknown, so say that rather than guess.
    """
    if requirement:
        op, _, val = requirement.partition(" ")
        return f"{label} is {_PHRASING.get(op, op)} {val}".rstrip()
    if witness is True:
        return f"{label} is present"
    if witness is False:
        return f"{label} is absent"
    return f"{label} is not established by the chart"

_CMP = r"(>=|<=|>|<|=)"


def requirement_for(name: str, phi_lines: Sequence[str]) -> Optional[str]:
    """Recover the numeric bound phi_t imposes on `name`, e.g. '>= 60'.

    Returns None for non-numeric conditions, where the witness IS the fact.
    """
    esc = re.escape(name)
    for ln in phi_lines:
        m = re.search(rf"\(\s*{_CMP}\s+\|?{esc}\|?\s+([-\d.]+)\s*\)", ln)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        m = re.search(rf"\(\s*{_CMP}\s+([-\d.]+)\s+\|?{esc}\|?\s*\)", ln)
        if m:                                  # reversed operand order
            flip = {">=": "<=", "<=": ">=", ">": "<", "<": ">", "=": "="}
            return f"{flip[m.group(1)]} {m.group(2)}"
    return None


def _z3():
    """Import z3, distinguishing 'absent' from 'shadowed by something else'.

    A bare `import z3` can succeed against an unrelated namespace package (no
    __file__, no attributes), which then fails much later with a confusing
    AttributeError. Probe for the API we actually need.
    """
    try:
        import z3
    except ImportError as e:                    # pragma: no cover
        raise ImportError(
            "MaxSMT needs the solver: pip install 'z3-solver>=4.12'") from e
    except Exception as e:                      # pragma: no cover
        # z3-solver is installed but its shared library will not load. The
        # usual cause is a wheel built for a newer OS than the one running
        # (macOS especially), and it raises Z3Exception, not ImportError.
        raise ImportError(
            f"z3-solver is installed but its native library failed to load "
            f"({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}). "
            f"This is usually a wheel built for a newer OS than yours. Try "
            f"`pip install --force-reinstall --no-binary :all: z3-solver`, or "
            f"install z3 from your package manager.") from e
    missing = [a for a in ("parse_smt2_string", "Optimize", "Solver")
               if not hasattr(z3, a)]
    if missing:                                 # pragma: no cover
        raise ImportError(
            f"the importable 'z3' is not z3-solver (missing {missing}; "
            f"resolved to {getattr(z3, '__file__', None)!r}). Something else "
            f"named z3 is shadowing it -- pip install 'z3-solver>=4.12' into "
            f"this interpreter.")
    return z3


def _decl_lines(conds: Sequence[Condition], phi_lines: Sequence[str]) -> List[str]:
    """Declare any condition the program text does not already declare."""
    out = []
    for c in conds:
        if any(re.search(rf"\(declare-(const|fun)\s+\|?{re.escape(c.name)}\|?\s+", ln)
               for ln in phi_lines):
            continue
        sort = "Real" if isinstance(c.value, (int, float)) and not isinstance(
            c.value, bool) else "Bool"
        out.append(f"(declare-const |{c.name}| {sort})")
    return out


def _lit(c: Condition) -> str:
    """The SMT literal asserting c_i = v_i."""
    if isinstance(c.value, bool):
        return f"|{c.name}|" if c.value else f"(not |{c.name}|)"
    return f"(= |{c.name}| {c.value})"


def solve(phi_lines: Sequence[str], conditions: Sequence[Condition],
          version: str = MAXSMT) -> Artifacts:
    r"""Run the accountability computation and return the artifacts.

    Args:
        phi_lines:  the trial program phi_t, as SMT-LIB lines (hard constraints).
        conditions: every c_i with its status; resolved ones form S.
        version:    MAXSMT (updated paper, default) or RESIDUAL (submitted).

    The two versions differ in what rho means and how delta is found:

    RESIDUAL (submitted, Steps 3-4)
        rho   = {c_i | q_i = UNRESOLVED}                  the conditions themselves
        delta = MAXSAT(phi /\ S, W)  .s = FALSE          if INELIGIBLE
                MAXSAT(-(phi /\ S), W).s = FALSE          if ELIGIBLE

    MAXSMT (update, Steps 3-6)
        delta_E = MAXSMT(phi /\ S, W).s = FALSE
        rho     = {c_i = MAXSMT(phi /\ S, W).c_i | q_i = UNRESOLVED}
        delta_I = MAXSMT(-phi /\ S /\ rho, W').x = FALSE
        delta   = delta_I if ELIGIBLE else delta_E
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    z3 = _z3()
    phi_lines = list(phi_lines)
    resolved = [c for c in conditions if c.resolved]
    unresolved = [c for c in conditions if not c.resolved]
    prelude = "\n".join(list(phi_lines) + _decl_lines(conditions, phi_lines))

    def parse(extra: str = "", negate_phi: bool = False):
        """Build a z3 goal from the program text (+ extra assertions)."""
        src = prelude + "\n" + extra
        try:
            asserts = z3.parse_smt2_string(src)
        except z3.Z3Exception as e:
            raise ValueError(f"cannot parse SMT program: {e}") from e
        return list(asserts)

    # ---- Step 2: the decision itself -------------------------------------
    hard = parse()
    S_exprs = parse("\n".join(f"(assert {_lit(c)})" for c in resolved))[len(hard):]
    s = z3.Solver()
    s.add(*hard, *S_exprs)
    z = s.check()
    if z == z3.unknown:
        return Artifacts(decision=INELIGIBLE, trace="solver returned unknown",
                         status="unknown", version=version)
    decision = ELIGIBLE if z == z3.sat else INELIGIBLE
    trace = "SAT under the asserted patient constraints" if z == z3.sat \
        else "UNSAT under the asserted patient constraints"

    if version == RESIDUAL:
        return _solve_residual(z3, hard, S_exprs, resolved, unresolved,
                               decision, trace, phi_lines)

    # ---- Steps 3-4: MAXSMT(phi /\ S, W) ----------------------------------
    opt = z3.Optimize()
    opt.add(*hard)
    for c, e in zip(resolved, S_exprs):
        opt.add_soft(e, 1, "S")
    delta_e: List[str] = []
    assumptions: Dict[str, Any] = {}
    if opt.check() == z3.sat:
        m = opt.model()
        for c, e in zip(resolved, S_exprs):            # Step 3
            if not z3.is_true(m.eval(e, model_completion=True)):
                delta_e.append(c.name)
        for c in unresolved:                            # Step 4
            for d in m.decls():
                if d.name() == c.name:
                    assumptions[c.name] = _py(m[d])
                    break
    else:
        trace += "; trial constraints unsatisfiable in isolation"

    # ---- Step 5: MAXSMT(-phi /\ S /\ rho, W') ----------------------------
    delta_i: List[str] = []
    rho_lines = "\n".join(
        f"(assert (= |{k}| {_smt_val(v)}))" for k, v in assumptions.items())
    try:
        all_hard = parse(rho_lines)
        rho_exprs = all_hard[len(hard):]
        opt2 = z3.Optimize()
        opt2.add(z3.Not(z3.And(*hard)) if hard else z3.BoolVal(False))
        soft2 = list(S_exprs) + list(rho_exprs)
        names2 = [c.name for c in resolved] + list(assumptions)
        for e in soft2:
            opt2.add_soft(e, 1, "Sp")
        if opt2.check() == z3.sat:
            m2 = opt2.model()
            for nm, e in zip(names2, soft2):
                if not z3.is_true(m2.eval(e, model_completion=True)):
                    delta_i.append(nm)
    except Exception:                                   # pragma: no cover
        trace += "; delta_I unavailable"

    # ---- Step 6 ----------------------------------------------------------
    pivotal = delta_i if decision == ELIGIBLE else delta_e
    return Artifacts(decision=decision, trace=trace, assumptions=assumptions,
                     pivotal=pivotal, delta_e=delta_e, delta_i=delta_i,
                     version=MAXSMT)


def _solve_residual(z3, hard, S_exprs, resolved, unresolved, decision, trace,
                    phi_lines) -> Artifacts:
    r"""The submitted formulation (Steps 3-4).

    rho is the *set of unresolved conditions*, not values assigned to them, so
    it is reported with the requirement phi imposes where one exists. delta is
    a single MAXSAT call whose formula depends on the decision.
    """
    rho = {c.name: requirement_for(c.name, phi_lines) for c in unresolved}

    opt = z3.Optimize()
    if decision == INELIGIBLE:                       # MAXSAT(phi /\ S, W)
        opt.add(*hard)
    else:                                            # MAXSAT(-(phi /\ S), W)
        conj = z3.And(*hard, *S_exprs) if (hard or S_exprs) else z3.BoolVal(True)
        opt.add(z3.Not(conj))
    for e in S_exprs:
        opt.add_soft(e, 1, "S")

    pivotal: List[str] = []
    if opt.check() == z3.sat:
        m = opt.model()
        for c, e in zip(resolved, S_exprs):
            if not z3.is_true(m.eval(e, model_completion=True)):
                pivotal.append(c.name)
    else:
        trace += "; MAXSAT infeasible"
    return Artifacts(decision=decision, trace=trace, assumptions=rho,
                     pivotal=pivotal, delta_e=[], delta_i=[],
                     version=RESIDUAL)


def _py(v) -> Any:
    """z3 model value -> plain Python."""
    z3 = _z3()
    if z3.is_true(v):
        return True
    if z3.is_false(v):
        return False
    try:
        if v.is_int():
            return v.as_long()
        return float(v.as_decimal(6).rstrip("?"))
    except Exception:
        return str(v)


def _smt_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
