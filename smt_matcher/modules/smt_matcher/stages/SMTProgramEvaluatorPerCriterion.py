# modules/stages/SingleRequirementEvaluator.py
from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple, Optional, Set, DefaultDict
from collections import defaultdict

import z3
from z3.z3util import get_vars  # extract variable refs from an AST
import dspy  # type: ignore – provided by surrounding project

from smt_core.utils.z3_helpers import _whole_program
from ...utils.mbench import get_mbench


# ─────────────────────────────────────────────────────────────── helpers ──

_LOC_RE = re.compile(r"line\s+(\d+),\s*column\s+(\d+)", re.I)

# Debug configuration: log only for requirements whose label starts with these prefixes
DEBUG_REQUIREMENT_PREFIXES = ("REQ8",)


def _is_debug_label(lbl: str) -> bool:
    return any(lbl.startswith(pfx) for pfx in DEBUG_REQUIREMENT_PREFIXES)


def _extract_loc(msg: str) -> Tuple[int | None, int | None]:
    m = _LOC_RE.search(msg)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


# ─────────────────────── 3-valued entailment helper ──────────────────────

def _status_three_valued(decls: str, patient_block: str, body: str) -> str:
    """
    Return 'sat' / 'unsat' / 'unknown' under Kleene 3-valued semantics:

      sat     – φ is forced true       (¬φ UNSAT)
      unsat   – φ is forced false      (φ UNSAT)
      unknown – both φ and ¬φ satisfiable
      conflict– both φ and ¬φ UNSAT (inconsistent declarations/values)
    """
    # Check UNSAT of ¬φ
    s1 = z3.Solver()
    s1.from_string("\n".join([decls, patient_block, f"(assert (not {body}))"]))
    not_phi_unsat = (s1.check() == z3.unsat)

    # Check UNSAT of φ
    s2 = z3.Solver()
    s2.from_string("\n".join([decls, patient_block, f"(assert {body})"]))
    phi_unsat = (s2.check() == z3.unsat)

    if phi_unsat and not_phi_unsat:
        return "conflict"
    elif phi_unsat:
        return "unsat"
    elif not_phi_unsat:
        return "sat"
    else:
        return "unknown"


def _declarations_only(src: str) -> str:
    """
    Keep only declarations/logic/options/defines and strip all (assert …) blocks,
    preserving (declare-datatypes …) sections.
    """
    keep_prefixes = (
        "(set-logic",
        "(set-option",
        "(declare-const",
        "(declare-fun",
        "(declare-sort)",
        "(declare-sort",
        "(define-fun",
        "(define-sort",
    )

    out: List[str] = []
    in_dtype = False
    skip_depth = 0

    for ln in src.splitlines():
        s = ln.lstrip()

        if skip_depth:
            skip_depth += ln.count("(") - ln.count(")")
            continue

        if s.startswith("(assert"):
            # Skip entire assert S-expression
            skip_depth = ln.count("(") - ln.count(")")
            continue

        if s.startswith("(declare-datatypes"):
            in_dtype = True
            out.append(ln)
            continue

        if in_dtype:
            out.append(ln)
            if s.endswith("))"):
                in_dtype = False
            continue

        if s.startswith(keep_prefixes):
            out.append(ln)

    return "\n".join(out)


def _inject_unsat_option(src: str) -> str:
    """Put (set-option :produce-unsat-cores true) right after (set-logic …)."""
    if ":produce-unsat-cores" in src:
        return src
    lines = src.splitlines()
    ins = "(set-option :produce-unsat-cores true)"
    if lines and lines[0].lstrip().startswith("(set-logic"):
        return "\n".join([lines[0], ins, *lines[1:]])
    return ins + "\n" + src


# ───────────────────────────── requirement extractor ─────────────────────

def _extract_requirements(src: str) -> List[Tuple[str, str, Any]]:
    """
    Return [(label, body_sexpr, body_ast)] for every *component* criterion in the SMT text.

    Z3 parses named asserts as an internal (=> label body).
    We now keep labels that start with 'REQ' and contain '_COMPONENT'.
    """
    prog = _inject_unsat_option(src)
    reqs: List[Tuple[str, str, Any]] = []
    for e in z3.parse_smt2_string(prog):
        # Z3 encodes (assert (! φ :named LBL)) as (=> LBL φ)
        if e.num_args() == 2 and e.decl().kind() == z3.Z3_OP_IMPLIES:
            try:
                lbl = e.arg(0).decl().name()
            except Exception:
                continue
            # NEW: keep only constraint (component) labels like REQ5_COMPONENT0_...
            if lbl.startswith("REQ") and "_COMPONENT" in lbl:
                body_ast = e.arg(1)
                reqs.append((lbl, body_ast.sexpr(), body_ast))
    return reqs

# ───────── numeric composition recognition & boolean-definition mapping ───

_NUMERIC_KINDS = {
    z3.Z3_OP_LT, z3.Z3_OP_LE, z3.Z3_OP_GT, z3.Z3_OP_GE
}
_BOOL_LOGIC_KINDS = {z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_NOT}
_ARITH_KINDS = {  # basic arithmetic over Int/Real
    z3.Z3_OP_ADD, z3.Z3_OP_SUB, z3.Z3_OP_UMINUS, z3.Z3_OP_MUL,
    z3.Z3_OP_DIV, z3.Z3_OP_IDIV, z3.Z3_OP_REM, z3.Z3_OP_MOD,
    z3.Z3_OP_TO_REAL, z3.Z3_OP_TO_INT, z3.Z3_OP_POWER,
}


def _is_arith_sort(ast) -> bool:
    try:
        k = ast.sort().kind()
        return k in (z3.Z3_INT_SORT, z3.Z3_REAL_SORT)
    except Exception:
        return False


def _is_arith_term(ast) -> bool:
    """
    Accept numerals, uninterpreted Int/Real consts, and arithmetic operator trees.
    """
    try:
        if z3.is_const(ast) and _is_arith_sort(ast):
            return True
        dk = ast.decl().kind()
        if dk in _ARITH_KINDS:
            return all(_is_arith_term(c) for c in ast.children())
        # Numerals – guard anyway:
        return _is_arith_sort(ast)
    except Exception:
        return False


def _is_numeric_comparison(ast) -> bool:
    """
    True iff ast is a numeric comparator between arithmetic terms.
    Also allow numeric equality (= t1 t2) where both sides are arith terms.
    """
    try:
        dk = ast.decl().kind()
    except Exception:
        return False

    if dk in _NUMERIC_KINDS:
        a, b = ast.children()
        return _is_arith_term(a) and _is_arith_term(b)

    if dk == z3.Z3_OP_EQ and ast.num_args() == 2:
        a, b = ast.children()
        if _is_arith_term(a) and _is_arith_term(b):
            return True

    return False


def _is_numeric_pred_tree(ast) -> bool:
    """
    True iff ast is built only from:
      - numeric comparisons (incl. numeric '='), and
      - boolean connectives (and/or/not) over such comparisons.
    """
    try:
        dk = ast.decl().kind()
    except Exception:
        return False

    if _is_numeric_comparison(ast):
        return True
    if dk in _BOOL_LOGIC_KINDS:
        return all(_is_numeric_pred_tree(c) for c in ast.children())
    return False


def _is_bool_symbol(ast) -> bool:
    try:
        # Uninterpreted Bool const, not 'true'/'false'
        return z3.is_const(ast) and z3.is_bool(ast) and ast.decl().kind() == z3.Z3_OP_UNINTERPRETED
    except Exception:
        return False


def _numeric_var_names(ast) -> List[str]:
    """
    Collect names of Int/Real variables referenced by this AST.
    """
    try:
        out: Set[str] = set()
        for v in get_vars(ast):
            try:
                if v.sort().kind() in (z3.Z3_INT_SORT, z3.Z3_REAL_SORT):
                    out.add(v.decl().name())
            except Exception:
                pass
        return sorted(out)
    except Exception:
        return []


def _are_all_numeric_inputs_missing(ast, vals: Dict[str, Any]) -> bool:
    """
    Return True iff every numeric (Int/Real) variable referenced by `ast` has no value in `vals`.
    """
    names = _numeric_var_names(ast)
    if not names:
        return False
    for n in names:
        raw = vals.get(n, None)
        if isinstance(raw, dict):
            raw = raw.get("value")
        if raw is not None:
            return False
    return True


def _collect_numeric_def_bool_map(full_src: str) -> Dict[str, List[Any]]:
    """
    Build a mapping BoolVar -> [numeric_pred_subtree_ast, ...] by scanning **all** asserts,
    accepting compositional numeric predicates and both directions (real↔bool) in (=, iff, =>).

    Examples that add entries:
      (= B (and (< x 3) (>= y 2)))
      (= (and (< x 3) (>= y 2)) B)
      (iff B (or (< x 1) (> y 9)))
      (iff (or (< x 1) (> y 9)) B)
      (=> (and (< x 3) (>= y 2)) B)
      (=> B (and (< x 3) (>= y 2)))   # now supported
    """
    mapping: DefaultDict[str, List[Any]] = defaultdict(list)
    prog = _inject_unsat_option(full_src)

    try:
        for e in z3.parse_smt2_string(prog):
            # For named asserts, Z3 returns (=> label body); for plain asserts it can return body directly.
            body_ast = e.arg(1) if (e.num_args() == 2 and e.decl().kind() == z3.Z3_OP_IMPLIES) else e

            dk = body_ast.decl().kind() if hasattr(body_ast, "decl") else None

            # (= a b) — both directions
            if dk == z3.Z3_OP_EQ and body_ast.num_args() == 2:
                a, b = body_ast.children()
                if _is_bool_symbol(a) and _is_numeric_pred_tree(b):
                    mapping[a.decl().name()].append(b)
                if _is_bool_symbol(b) and _is_numeric_pred_tree(a):
                    mapping[b.decl().name()].append(a)
                continue

            # (iff a b) — both directions
            if dk == z3.Z3_OP_IFF and body_ast.num_args() == 2:
                a, b = body_ast.children()
                if _is_bool_symbol(a) and _is_numeric_pred_tree(b):
                    mapping[a.decl().name()].append(b)
                if _is_bool_symbol(b) and _is_numeric_pred_tree(a):
                    mapping[b.decl().name()].append(a)
                continue

            # (=> a b) — both directions
            if dk == z3.Z3_OP_IMPLIES and body_ast.num_args() == 2:
                a, b = body_ast.children()
                if _is_bool_symbol(b) and _is_numeric_pred_tree(a):   # (=> NUM B)
                    mapping[b.decl().name()].append(a)
                if _is_bool_symbol(a) and _is_numeric_pred_tree(b):   # (=> B NUM)
                    mapping[a.decl().name()].append(b)
                continue

            # Also scan inside (and/or/not) to catch nested (= …), (iff …), (=> …)
            if dk in _BOOL_LOGIC_KINDS:
                stack = list(body_ast.children())
                while stack:
                    node = stack.pop()
                    try:
                        ndk = node.decl().kind()
                    except Exception:
                        continue
                    if ndk in _BOOL_LOGIC_KINDS:
                        stack.extend(node.children())
                        continue
                    if ndk == z3.Z3_OP_EQ and node.num_args() == 2:
                        a, b = node.children()
                        if _is_bool_symbol(a) and _is_numeric_pred_tree(b):
                            mapping[a.decl().name()].append(b)
                        if _is_bool_symbol(b) and _is_numeric_pred_tree(a):
                            mapping[b.decl().name()].append(a)
                    elif ndk == z3.Z3_OP_IFF and node.num_args() == 2:
                        a, b = node.children()
                        if _is_bool_symbol(a) and _is_numeric_pred_tree(b):
                            mapping[a.decl().name()].append(b)
                        if _is_bool_symbol(b) and _is_numeric_pred_tree(a):
                            mapping[b.decl().name()].append(a)
                    elif ndk == z3.Z3_OP_IMPLIES and node.num_args() == 2:
                        a, b = node.children()
                        if _is_bool_symbol(b) and _is_numeric_pred_tree(a):
                            mapping[b.decl().name()].append(a)
                        if _is_bool_symbol(a) and _is_numeric_pred_tree(b):
                            mapping[a.decl().name()].append(b)
        return dict(mapping)
    except Exception:
        return dict(mapping)


# ───────────────────────────── values & printing helpers ──────────────────

def _literal(sort: str, value):
    if value is None:
        return None
    if isinstance(value, dict):
        return _literal(sort, value.get("value"))

    s = (sort or "").lower()
    if s == "bool":
        return "true" if str(value).lower() in {"true", "1", "t", "yes"} else "false"
    if s == "string":
        return f'"{value}"'
    if s in {"int", "integer", "real"}:
        return str(value)
    return str(value)


def _assert_block(vals: Dict[str, Any], vindex: Dict[str, Dict]) -> str:
    """
    Build (assert (= var lit)) lines only for variables that have explicit values.
    """
    lines: List[str] = []
    for v, val in vals.items():
        raw_val = val.get("value") if isinstance(val, dict) else val
        if raw_val is None:
            continue
        vinfo = vindex.get(v)
        if vinfo is None:
            continue
        lit = _literal(vinfo.get("type", ""), val)
        if lit is None:
            continue
        lines.append(f"(assert (= {v} {lit}))")
    return "\n".join(lines)

_REQ_IDX = re.compile(r"^REQ(\d+)", re.I)

def _req_index(lbl: str, fallback: int) -> int:
    m = _REQ_IDX.match(lbl)
    return int(m.group(1)) if m else fallback

# # Extract the group index from labels like R3_A0_FOO → 3
# _R_IDX = re.compile(r"^R(\d+)", re.I)


# def _req_index(lbl: str, fallback: int) -> int:
#     m = _R_IDX.match(lbl)
#     return int(m.group(1)) if m else fallback


def _aggregate_group_status(statuses: List[str]) -> str:
    if any(s == "unsat" for s in statuses):
        return "unsat"
    if statuses and all(s == "sat" for s in statuses):
        return "sat"
    return "unknown"


def _group_per_requirement(per_req: Dict[str, str]) -> Dict[str, str]:
    buckets: Dict[int, List[str]] = {}
    for lbl, st in per_req.items():
        idx = _req_index(lbl, -1)
        buckets.setdefault(idx, []).append(st)

    grouped: Dict[str, str] = {}
    for idx, sts in buckets.items():
        grouped[f"REQ{idx}"] = _aggregate_group_status(sts)
    return grouped



def _relevant_var_names(body_ast) -> List[str]:
    try:
        return sorted({v.decl().name() for v in get_vars(body_ast)})
    except Exception:
        return []


def _overall_status_from(per_req: Dict[str, str]) -> str:
    sts = list(per_req.values())
    if any(s == "unsat" for s in sts):
        return "unsat"
    if sts and all(s == "sat" for s in sts):
        return "sat"
    return "unknown"


# ──────────────────────────── NEW: pattern helpers ────────────────────────
def _is_implies_numeric_to_bool(ast) -> Tuple[bool, Optional[Any], Optional[Any]]:
    """
    Detect (=> NUMPRED BoolVar). Return (True, NUMPRED, BoolVarAST) if matched.
    """
    try:
        if ast.decl().kind() == z3.Z3_OP_IMPLIES and ast.num_args() == 2:
            a, b = ast.children()
            if _is_numeric_pred_tree(a) and _is_bool_symbol(b):
                return True, a, b
    except Exception:
        pass
    return False, None, None


# ───────────────────────────────────────────────────────── core runner ────

def _run_for_patient(
    decls: str,
    vals: Dict[str, Any],
    vindex: Dict[str, Dict],
    requirements: List[Tuple[str, str, Any]],
    numeric_def_map: Dict[str, List[Any]],
    *,
    pid: str | None = None,
    mb=None,
    side: Optional[str] = None,               # kept for API compat, unused
    ref_gpt4: Optional[List[str]] = None,     # kept for API compat, unused
    ref_expert: Optional[List[str]] = None,   # kept for API compat, unused
    ref_criteria: Optional[List[str]] = None, # kept for API compat, unused
) -> Dict[str, Any]:
    """
    Evaluate every requirement for ONE patient and return:

    {
      "per_requirement": {label: "sat"/"unsat"/"unknown"/"conflict", …},
      "overall_status": "sat" | "unsat" | "unknown",
      "per_requirement_grouped": { "R0": "...", ... }
    }

    Fallback rule (existing):
      If a requirement is 'unknown', and there exists a Bool var B in that requirement
      which is mapped to one or more *numeric predicate trees*, and B is missing in vals,
      and for EVERY such mapped numeric tree all its numeric inputs are missing in vals,
      then we default B = false (once) and re-evaluate.

    Fallback rule (NEW — “forget null real if Bool is present”):
      If requirement is of the form (=> NUMPRED B), B has a concrete value in vals,
      and ALL numeric inputs of NUMPRED are missing in vals, then re-evaluate the
      requirement with an additional (assert (not NUMPRED)) to vacuously satisfy it.
    """
    patient_block = _assert_block(vals, vindex)

    per_req: Dict[str, str] = {}

    for i, (lbl, body_s, body_ast) in enumerate(requirements):
        st = _status_three_valued(decls, patient_block, body_s)

        # DEBUG (pre-default)
        if _is_debug_label(lbl):
            rel_vars_dbg = _relevant_var_names(body_ast)
            cand = [v for v in rel_vars_dbg if v in numeric_def_map]
            # Numeric subtree stats per candidate B
            subtree_dbg: Dict[str, Any] = {}
            for v in cand:
                trees = numeric_def_map.get(v, [])
                per_tree_stats = []
                for t in trees:
                    nvars = _numeric_var_names(t)
                    per_tree_stats.append({
                        "numeric_vars": nvars,
                        "all_missing": all(
                            (vals.get(n, {}).get("value") if isinstance(vals.get(n), dict) else vals.get(n)) is None
                            for n in nvars
                        ) if nvars else False
                    })
                subtree_dbg[v] = per_tree_stats

            rel_vals_dbg = {v: vals.get(v, None) for v in rel_vars_dbg}
            print(f"[DEBUG:R8] Processing {lbl}:")
            print(f"[DEBUG:R8]   body: {body_s}")
            print(f"[DEBUG:R8]   initial_status: {st}")
            print(f"[DEBUG:R8]   relevant_vars: {rel_vars_dbg}")
            print(f"[DEBUG:R8]   numeric_defined_candidates: {cand}")
            print(f"[DEBUG:R8]   numeric_subtree_stats: {subtree_dbg}")
            print(f"[DEBUG:R8]   current_values (subset): {{"
                  f"{', '.join(f'{v}: {rel_vals_dbg[v]}' for v in cand)}"
                  f"}}")

            if mb and pid:
                mb.log_json(
                    "SingleRequirementEvaluator",
                    f"{pid}/{lbl}/debug_pre",
                    {
                        "label": lbl,
                        "body": body_s,
                        "initial_status": st,
                        "relevant_vars": rel_vars_dbg,
                        "numeric_defined_candidates": cand,
                        "numeric_subtree_stats": subtree_dbg,
                        "current_values_subset": {v: rel_vals_dbg[v] for v in cand},
                    },
                )

        # Fallback — existing “default missing Bool to false if all numeric inputs missing”
        to_default_bools: List[str] = []
        if st == "unknown":
            rel_vars = _relevant_var_names(body_ast)
            for v in rel_vars:
                trees = numeric_def_map.get(v, [])
                if not trees:
                    continue
                # Has to be Bool (by design we map only Bool symbols, but double-check vindex if present)
                vinfo = vindex.get(v, {})
                vtype = vinfo.get("type", "").lower()
                if vtype and vtype != "bool":
                    continue
                # Missing?
                raw_val = vals.get(v, None)
                if isinstance(raw_val, dict):
                    raw_val = raw_val.get("value")
                if raw_val is not None:
                    continue  # already set, do not default

                # Only default if EVERY mapped numeric subtree has all numeric inputs missing
                if trees and all(_are_all_numeric_inputs_missing(t, vals) for t in trees):
                    to_default_bools.append(v)

            if to_default_bools:
                defaults_block = "\n".join(f"(assert (= {v} false))" for v in to_default_bools)
                st2 = _status_three_valued(decls, "\n".join([patient_block, defaults_block]), body_s)
                if st2 != "unknown":
                    st = st2

        # Fallback — NEW: vacuously satisfy (=> NUMPRED B) if B present and all numeric inputs missing
        forced_numeric_false: List[str] = []
        if st == "unknown":
            matched, numpred, bool_ast = _is_implies_numeric_to_bool(body_ast)
            if matched and numpred is not None and bool_ast is not None:
                bname = bool_ast.decl().name()
                bval = vals.get(bname, None)
                if isinstance(bval, dict):
                    bval = bval.get("value")
                # Bool present, but numeric inputs all missing → assert (not NUMPRED) for classification
                if bval is not None and _are_all_numeric_inputs_missing(numpred, vals):
                    defaults_block2 = f"(assert (not {numpred.sexpr()}))"
                    st3 = _status_three_valued(decls, "\n".join([patient_block, defaults_block2]), body_s)
                    if st3 != "unknown":
                        st = st3
                        forced_numeric_false.append(numpred.sexpr())

        # DEBUG (post-default)
        if _is_debug_label(lbl):
            print(f"[DEBUG:R8]   defaulted_to_false: {to_default_bools}")
            if forced_numeric_false:
                print(f"[DEBUG:R8]   numeric_antecedent_forced_false: {forced_numeric_false}")  # NEW
            print(f"[DEBUG:R8]   final_status: {st}")
            if mb and pid:
                mb.log_json(
                    "SingleRequirementEvaluator",
                    f"{pid}/{lbl}/debug_post",
                    {
                        "label": lbl,
                        "defaulted_to_false": to_default_bools,
                        "numeric_antecedent_forced_false": forced_numeric_false,  # NEW
                        "final_status": st,
                    },
                )

        per_req[lbl] = st

        # Persist one "decisive" program per requirement
        if st == "sat":
            program_src = "\n".join([decls, f"(assert (not {body_s}))"])
        else:
            program_src = "\n".join([decls, f"(assert {body_s})"])

        if mb and pid:
            subdir = f"{pid}/{lbl}"
            mb.log_text("SingleRequirementEvaluator", f"{subdir}/program.smt2", program_src)
            rel_vars = _relevant_var_names(body_ast)
            rel_vals = {v: vals.get(v, None) for v in rel_vars}
            mb.log_json("SingleRequirementEvaluator", f"{subdir}/values", rel_vals)
            mb.log_json(
                "SingleRequirementEvaluator",
                f"{subdir}/outcome",
                {"label": lbl, "group": f"R{_req_index(lbl, i)}", "status": st},
            )

    grouped = _group_per_requirement(per_req)
    overall = _overall_status_from(per_req)

    # Extra debug for R8 group
    if "R8" in grouped:
        print(f"[DEBUG:R8]   grouped_status_for_R8: {grouped['R8']}")
        if mb and pid:
            mb.log_json(
                "SingleRequirementEvaluator",
                f"{pid}/R8/debug_grouped",
                {"grouped_status_for_R8": grouped["R8"], "overall_after_run": overall},
            )

    out = {
        "per_requirement": per_req,
        "overall_status": overall,
        "per_requirement_grouped": grouped,
    }
    return out


# ───────────────────────────────────────────────────── main evaluator ────

class SingleRequirementEvaluator(dspy.Module):
    """
    Check each inclusion/exclusion criterion separately; write minimal per-requirement artifacts.
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mb = get_mbench(context)
        full_prog = _whole_program(context)
        vindex = context.get("variable_index", {})
        pv = context.get("patient_var_values", {})

        # prepare pieces
        decls = _declarations_only(full_prog)
        requirements = _extract_requirements(full_prog)            # [(label, sexpr, ast)]
        numeric_def_map = _collect_numeric_def_bool_map(full_prog) # Bool -> [numeric_pred_ast, …]

        # single-patient
        if not _is_multi_patient(pv, vindex):
            result = _run_for_patient(
                decls, pv, vindex, requirements, numeric_def_map,
                pid="single", mb=mb, side=None, ref_gpt4=None, ref_expert=None, ref_criteria=None
            )
            # Store canonical result
            context["eval_result"] = result
            # Mirror to commonly-read top-level keys
            context["per_requirement"] = result.get("per_requirement", {})
            context["per_requirement_grouped"] = result.get("per_requirement_grouped", {})
            context["overall_status_single_requirement_eval"] = result.get("overall_status")
            # Optional legacy/alias used by some pipelines:
            context["eval_result_grouped"] = result.get("per_requirement_grouped", {})
            return context

        # multi-patient
        all_results: Dict[str, Any] = {}
        for pid, vals in pv.items():
            res = _run_for_patient(
                decls, vals, vindex, requirements, numeric_def_map,
                pid=pid, mb=mb, side=None, ref_gpt4=None, ref_expert=None, ref_criteria=None
            )
            all_results[pid] = res

        context["eval_result"] = all_results
        # Convenience: expose grouped results by patient for downstreams
        context["eval_result_grouped"] = {
            pid: res.get("per_requirement_grouped", {})
            for pid, res in all_results.items()
        }
        # If some consumers still expect single-patient-style keys and you often run 1 patient:
        if len(all_results) == 1:
            only = next(iter(all_results.values()))
            context["per_requirement"] = only.get("per_requirement", {})
            context["per_requirement_grouped"] = only.get("per_requirement_grouped", {})
            context["overall_status_single_requirement_eval"] = only.get("overall_status")
        return context


# helper ────────────────────────────────────────────────────────────────

def _is_multi_patient(pv: Any, vindex: dict) -> bool:
    return (
        isinstance(pv, dict)
        and pv
        and isinstance(next(iter(pv.values())), dict)
        and not (set(pv.keys()) <= set(vindex.keys()))
    )


# Utility pretty-printer (optional)

def print_eval(res: Dict[str, Any]):
    print("=== trial decision ===")
    print("overall:", res["overall_status"])
    for k, v in res["per_requirement"].items():
        print(f"{k:6s} : {v}")
