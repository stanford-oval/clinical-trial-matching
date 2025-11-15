#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smt_projector.py  –  Signature-closure QE projector with definition orientation
+ DB-friendly clause emitter WITH per-clause SAT filtering
+ Always-on UNSAT-core pruning (remove ONLY contradicting assertions)
+ Append SAT-filtered CNF OR-constraint_clauses (stems-only) to final projected SMT
+ OR cleanups — dedup, subsumption, and removal of constraint_clauses made true by unit literals

Performance improvements
───────────────────────
- Reuse solvers (push/pop) for SAT checks and entailment (no per-clause Solver()).
- Bounded CNF expansion to avoid exponential explosion (MAX_CLAUSE_PRODUCT, MAX_OR_ARITY).
- Wall-clock time budget for OR-clause SAT filtering (TIME_BUDGET_S).
- Optional numeric-QE cost guard to skip heavy tactics on large terms.
- Faster subsumption via length-bucketing.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import argparse, json, logging, re, sys, time

from z3 import (  # type: ignore
    Solver, And, Or, Not, Implies,
    Bool, BoolVal, parse_smt2_file, Z3Exception, unsat, unknown, sat, simplify,
    is_const, is_app, is_app_of, is_quantifier, is_true, is_false,
    is_int_value, is_rational_value, is_algebraic_value,
    Exists, Tactic, substitute, ExprRef,
    Z3_OP_EQ, Z3_OP_LABEL, Z3_OP_LABEL_LIT,
)

# ─────────────────────────────────────────────
# Performance caps / budgets
# ─────────────────────────────────────────────
MAX_CLAUSE_PRODUCT = 50_000   # max combinations when distributing OR over AND
MAX_OR_ARITY       = 16       # skip OR nodes wider than this during CNF
TIME_BUDGET_S      = 15.0     # wall-clock time budget for OR-clause SAT filtering (per file)
QE_NUMERIC_TERM_COST_CAP = 5000  # skip numeric QE if term is too big (optional guard)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
log = logging.getLogger("smt_projector")
if not log.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)
import traceback

_PARSE_POS_RE = re.compile(r"[Ll]ine\s+(\d+)\s*,?\s*[Cc]ol(?:umn)?\s+(\d+)")

def _write_parse_error_diagnostics(smt_path: str, err: Exception, dbg_dir: Optional[Path]) -> dict:
    """
    Write a human-friendly parse error report with line/column and a few context lines.
    Returns a metadata dict to include in the summary.
    """
    meta = {
        "error": str(err)[:500],
        "line": None, "col": None,
        "context_file": None,
        "traceback": None,
    }
    try:
        txt = Path(smt_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        txt = ""

    m = _PARSE_POS_RE.search(str(err))
    line = int(m.group(1)) if m else None
    col  = int(m.group(2)) if m and m.lastindex and m.lastindex >= 2 else None
    meta["line"], meta["col"] = line, col

    # Build a short context: ±3 lines around the error
    lines = txt.splitlines()
    def _clip(i): return max(1, i)
    start = _clip((line or 1) - 3)
    end   = min(len(lines), (line or 1) + 3)

    report = []
    report.append(f"SMT file: {smt_path}")
    report.append(f"Error: {type(err).__name__}: {str(err)}")
    if line is not None:
        report.append(f"At: line {line}, column {col or 1}")
    report.append("")
    report.append("Context (±3 lines):")
    for i in range(start, end + 1):
        prefix = ">>" if i == (line or -1) else "  "
        body = lines[i-1] if 1 <= i <= len(lines) else ""
        caret = ""
        if i == line and col and col >= 1:
            caret = "\n" + " " * (col + 3) + "^"
        report.append(f"{prefix} {i:6d}  {body}{caret}")

    # Quick sanity check helpers (often catches silly issues fast)
    if txt:
        # 1) Unbalanced parens
        bal = 0
        for ch in txt:
            if ch == "(": bal += 1
            elif ch == ")": bal -= 1
        if bal != 0:
            report.append("")
            report.append(f"[hint] Unbalanced parentheses detected (balance={bal}).")

        # 2) Bad control chars (rare, but happens in generated SMT)
        if "\x00" in txt:
            report.append("[hint] NUL byte found in file; remove or replace it.")

    # Save file
    if dbg_dir is not None:
        dbg_dir.mkdir(parents=True, exist_ok=True)
        outp = dbg_dir / "zz_parse_error_context.txt"
        outp.write_text("\n".join(report) + "\n", encoding="utf-8")
        meta["context_file"] = str(outp)

    # Also capture a short traceback for logs (optional)
    meta["traceback"] = "".join(traceback.format_exception_only(type(err), err)).strip()
    log.error("[parse] %s line=%s col=%s (ctx=%s)",
              Path(smt_path).name, line, col, meta.get("context_file"))
    return meta

# ─────────────────────────────────────────────
# Debug-dump helper
# ─────────────────────────────────────────────

def _dump_expr(tag: str, e: ExprRef, dbg_dir: Optional[Path]):
    if dbg_dir is None: return
    dbg_dir.mkdir(parents=True, exist_ok=True)
    idx = _dump_expr.counter; _dump_expr.counter += 1
    p = dbg_dir / f"{idx:02d}_{tag}.smt2"
    s = e.sexpr()
    p.write_text(s, encoding="utf-8")
    log.info(f"[dbg] dumped {tag:<17} → {p.name:<22} (chars={len(s):,}, vars={len(_collect_vars(e))})")
_dump_expr.counter = 0

# ─────────────────────────────────────────────
# AST helpers
# ─────────────────────────────────────────────

def _is_num(x: ExprRef) -> bool:
    return bool(is_int_value(x) or is_rational_value(x) or is_algebraic_value(x))

def _name(x: ExprRef) -> str:
    try: return x.decl().name()
    except Exception: return str(x)

def _strip_labels(e: ExprRef) -> ExprRef:
    while is_app_of(e, Z3_OP_LABEL) or is_app_of(e, Z3_OP_LABEL_LIT):
        e = e.arg(0)
    return e

def _collect_vars(e: ExprRef) -> List[ExprRef]:
    seen, out = set(), []
    def go(t: ExprRef):
        t = _strip_labels(t)
        if is_const(t) and t.decl().arity() == 0 and not is_true(t) and not is_false(t) and not _is_num(t):
            if t not in seen:
                seen.add(t); out.append(t)
        if is_quantifier(t):
            go(t.body()); return
        if is_app(t):
            for i in range(t.num_args()):
                go(t.arg(i))
    go(e); return out

def _free_names(e: ExprRef) -> Set[str]:
    return {_name(v) for v in _collect_vars(e)}

# ─────────────────────────────────────────────
# Parse sorts
# ─────────────────────────────────────────────

def _decl_sorts(txt:str)->Dict[str,str]:
    out={}
    for m in re.finditer(r"\(declare-const\s+(\S+)\s+(\S+)\)",txt):
        out[m[1]]=m[2]
    for m in re.finditer(r"\(declare-fun\s+(\S+)\s+\([^\)]*\)\s+(\S+)\)",txt):
        out[m[1]]=m[2]
    return out

# ─────────────────────────────────────────────
# Definitional equalities: (= v φ) with v not in Free(φ)
# ─────────────────────────────────────────────

def _def_eq_of(e: ExprRef) -> Optional[Tuple[ExprRef,ExprRef]]:
    e = _strip_labels(e)
    if not is_app_of(e, Z3_OP_EQ): return None
    a0, a1 = e.arg(0), e.arg(1)
    for v, phi in ((a0, a1), (a1, a0)):
        if is_const(v) and v.decl().arity()==0 and not _is_num(v):
            if _name(v) not in _free_names(phi):
                return (v, phi)
    return None

def _collect_def_eqs(assertions: List[ExprRef]) -> List[Tuple[ExprRef,ExprRef]]:
    out=[]
    for e in assertions:
        d=_def_eq_of(e)
        if d is not None: out.append(d)
    return out

def _def_closure(initial_keep: Set[str], def_eqs: List[Tuple[ExprRef,ExprRef]]) -> Set[str]:
    keep = set(initial_keep)
    changed = True
    while changed:
        changed = False
        for v, phi in def_eqs:
            if _name(v) in keep:
                for fn in _free_names(phi):
                    if fn not in keep:
                        keep.add(fn); changed = True
    return keep

# ─────────────────────────────────────────────
# Healing (passthrough)
# ─────────────────────────────────────────────

def _heal_to_temp(p: str) -> Tuple[str, Dict[str,int]]:
    fp = Path(p)
    healed = fp if fp.name.endswith(".healed.smt2") else fp.with_suffix(".healed.smt2")
    healed.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
    return str(healed), {"auto_decls_added":0,"fixed_decl_lines":0}

# ─────────────────────────────────────────────
# Definition orientation & pruning
# ─────────────────────────────────────────────

def _orient_and_prune(assertions: List[ExprRef],
                      keep_closed: Set[str],
                      def_eqs: List[Tuple[ExprRef,ExprRef]]) -> List[ExprRef]:
    """
    Keep (= v φ) as-is. For any top-level (⇒ ψ v) where v has a definitional equality,
    drop the implication. This avoids cycles.
    """
    defined_names = {_name(v) for v,_ in def_eqs}

    out: List[ExprRef] = []
    dropped = 0
    for e in assertions:
        ee = _strip_labels(e)

        if _def_eq_of(ee) is not None:
            out.append(e); continue

        if is_app(ee) and ee.decl().name() == '=>':
            rhs = ee.arg(1)
            if is_const(rhs) and _name(rhs) in defined_names:
                dropped += 1
                continue

        out.append(e)

    if dropped:
        log.info("[info] pruned %d attr→base implication(s) targeting defined symbols", dropped)
    return out

# ─────────────────────────────────────────────
# Definitional preservation for non-kept defs
# ─────────────────────────────────────────────

def _preserve_defs_for_nonkept(assertions: List[ExprRef],
                               keep_closed: Set[str],
                               sorts: Dict[str, str]) -> List[ExprRef]:
    out: List[ExprRef] = []
    for e in assertions:
        d = _def_eq_of(_strip_labels(e))
        if d is None:
            out.append(e)
            continue

        v, phi = d
        v_name = _name(v)
        v_sort = sorts.get(v_name, str(v.sort()))  # 兜底用 z3 的 sort

        if v_name in keep_closed:
            # 保留定义等式
            out.append(e)
        else:
            if v_sort == "Bool":
                # 只有 Bool 才能安全改写成蕴含
                out.append(Implies(v, phi))
                out.append(Implies(phi, v))
            else:
                # 非 Bool（Int/Real 等）保持等式不变，便于 QE 消去
                out.append(e)
    return out

# ─────────────────────────────────────────────
# UNSAT-core pruning
# ─────────────────────────────────────────────

def _prune_by_unsat_core(assertions: List[ExprRef], timeout_ms: int) -> Tuple[List[ExprRef], Dict[str, int]]:
    if not assertions:
        return assertions, {"rounds": 0, "pruned_total": 0, "last_core_size": 0, "final_status": "unknown"}

    kept = list(assertions)
    rounds = 0
    pruned_total = 0
    last_core_size = 0
    final_status = "unknown"

    while True:
        s = Solver(); s.set(timeout=timeout_ms)
        labels = []
        for i, e in enumerate(kept):
            lbl = Bool(f"__uc_{rounds}_{i}")
            s.assert_and_track(e, lbl)
            labels.append(lbl)

        r = s.check()
        if r == sat or r == unknown:
            final_status = "sat" if r == sat else "unknown"
            break

        core = set(s.unsat_core())
        last_core_size = len(core)
        to_remove = {kept[i] for i, lbl in enumerate(labels) if lbl in core}
        if not to_remove:
            final_status = "unsat_no_core"
            break

        kept = [e for e in kept if e not in to_remove]
        pruned_total += len(to_remove)
        rounds += 1

    if pruned_total:
        log.warning("[warn] UNSAT detected; pruned %d assertion(s) in %d round(s) (last core=%d, final=%s)",
                    pruned_total, rounds, last_core_size, final_status)
    else:
        log.info("[ok] input assertions are %s; no UNSAT-core pruning needed", final_status)

    return kept, {
        "rounds": rounds,
        "pruned_total": pruned_total,
        "last_core_size": last_core_size,
        "final_status": final_status,
    }

# ─────────────────────────────────────────────
# QE helpers
# ─────────────────────────────────────────────

def _apply_qe_any(F: ExprRef, drop: List[ExprRef], strict: bool) -> Optional[ExprRef]:
    strict_ls=['qe','qe2','qe_rec','nlqe','qe-light']
    light_ls =['qe2','qe_rec','qe','nlqe','qe-light']
    for t in (strict_ls if strict else light_ls):
        try:
            return Tactic(t)(Exists(drop, F)).as_expr()
        except Exception:
            pass
    return None

def _unsat(phi: ExprRef) -> bool:
    s=Solver(); s.add(phi)
    return s.check()==unsat

def _bool_qe_shannon(F:ExprRef, drops:List[ExprRef], cap:int)->Optional[ExprRef]:
    if not drops: return F
    if len(drops)>cap: return None
    G=F
    for b in drops:
        G=simplify(Or(substitute(G,(b,BoolVal(True))),
                      substitute(G,(b,BoolVal(False)))))
    return G

def _term_cost(e: ExprRef) -> int:
    """Cheap structural size for QE guard."""
    cnt = 0; stack = [e]; seen=set()
    while stack:
        t = stack.pop()
        if t in seen: continue
        seen.add(t); cnt += 1
        if is_app(t):
            for i in range(t.num_args()): stack.append(t.arg(i))
    return cnt

# ─────────────────────────────────────────────
# Clause helpers (DB-friendly export path)
# ─────────────────────────────────────────────

def _is_bool_sort(e: ExprRef) -> bool:
    return str(e.sort()) == "Bool"

def _is_bool_atom_like(e: ExprRef) -> bool:
    e = _strip_labels(e)
    return (_is_bool_sort(e) and is_const(e) and e.decl().arity() == 0)

def _is_numeric_atom(e: ExprRef) -> bool:
    e = _strip_labels(e)
    if not is_app(e): return False
    nm = e.decl().name()
    return nm in (">=","<=","<",">","=") and not (_is_bool_sort(e.arg(0)) or _is_bool_sort(e.arg(1)))

# NEW: normalize negated comparators to positive numeric atoms

def _as_positive_numeric_atom(e: ExprRef) -> Optional[ExprRef]:
    """Return a positive numeric comparator for e if possible.
       - If e is a numeric comparator, return it (simplified).
       - If e is (not (<=/>=/>/< ...)), flip the relation and return the positive atom.
       - If e is (not (= ...)) or non-numeric, return None.
    """
    e = _strip_labels(e)
    if is_app(e):
        nm = e.decl().name()
        if nm == "not":
            a = _strip_labels(e.arg(0))
            if is_app(a):
                am = a.decl().name()
                if am in ("<=", ">=", "<", ">"):
                    x, y = a.arg(0), a.arg(1)
                    if am == "<=":  return simplify(x >  y, arith_lhs=True)
                    if am == ">=":  return simplify(x <  y, arith_lhs=True)
                    if am == "<":   return simplify(x >= y, arith_lhs=True)
                    if am == ">":   return simplify(x <= y, arith_lhs=True)
                # ¬(= …) is not a single comparator; skip
            return None
        elif nm in (">=", "<=", "<", ">", "="):
            if not (_is_bool_sort(e.arg(0)) or _is_bool_sort(e.arg(1))):
                return simplify(e, arith_lhs=True)
    return None


def _nnf(e: ExprRef) -> ExprRef:
    e = _strip_labels(e)
    if _is_bool_atom_like(e) or _is_numeric_atom(e):
        return e
    if is_app(e):
        nm = e.decl().name()
        if nm == "not":
            a = _strip_labels(e.arg(0))
            if _is_bool_atom_like(a) or _is_numeric_atom(a):
                return Not(a)
            if is_app(a):
                an = a.decl().name()
                if an == "not":
                    return _nnf(a.arg(0))
                if an == "and":
                    return Or(*(_nnf(Not(a.arg(i))) for i in range(a.num_args())))
                if an == "or":
                    return And(*(_nnf(Not(a.arg(i))) for i in range(a.num_args())))
                if an == "=>":
                    return And(_nnf(a.arg(0)), _nnf(Not(a.arg(1))))
                if an == "=" and _is_bool_sort(a.arg(0)) and _is_bool_sort(a.arg(1)):
                    A, B = a.arg(0), a.arg(1)
                    return Or(And(_nnf(A), _nnf(Not(B))), And(_nnf(Not(A)), _nnf(B)))
            return Not(_nnf(a))
        if nm == "and":
            return And(*(_nnf(e.arg(i)) for i in range(e.num_args())))
        if nm == "or":
            return Or(*(_nnf(e.arg(i)) for i in range(e.num_args())))
        if nm == "=>":
            return Or(_nnf(Not(e.arg(0))), _nnf(e.arg(1)))
        if nm == "=" and _is_bool_sort(e.arg(0)) and _is_bool_sort(e.arg(1)):
            A, B = e.arg(0), e.arg(1)
            return And(Or(_nnf(Not(A)), _nnf(B)), Or(_nnf(Not(B)), _nnf(A)))
    if is_app(e):
        return e.decl()(*(_nnf(e.arg(i)) for i in range(e.num_args())))
    return e


def _distribute_or_bounded(constraint_clauses_L: List[List[ExprRef]],
                           constraint_clauses_R: List[List[ExprRef]],
                           product_cap: int) -> Optional[List[List[ExprRef]]]:
    if len(constraint_clauses_L) * len(constraint_clauses_R) > product_cap:
        return None
    out: List[List[ExprRef]] = []
    for cl in constraint_clauses_L:
        for cr in constraint_clauses_R:
            out.append(cl + cr)
    return out


def _cnf_constraint_clauses_from_nnf_bounded(e: ExprRef,
                                  product_cap: int,
                                  or_arity_cap: int) -> Optional[List[List[ExprRef]]]:
    e = _strip_labels(e)

    if _is_bool_atom_like(e) or _is_numeric_atom(e):
        return [[e]]

    if is_app(e) and e.decl().name() == "not":
        a = _strip_labels(e.arg(0))
        if _is_bool_atom_like(a) or _is_numeric_atom(a):
            return [[Not(a)]]

    if is_app(e) and e.decl().name() == "and":
        out: List[List[ExprRef]] = []
        for i in range(e.num_args()):
            sub = _cnf_constraint_clauses_from_nnf_bounded(e.arg(i), product_cap, or_arity_cap)
            if sub is None:
                return None
            out.extend(sub)
        return out

    if is_app(e) and e.decl().name() == "or":
        ar = e.num_args()
        if ar > or_arity_cap:
            return None
        acc: List[List[ExprRef]] = [[]]  # start with empty clause
        for i in range(ar):
            sub = _cnf_constraint_clauses_from_nnf_bounded(e.arg(i), product_cap, or_arity_cap)
            if sub is None:
                return None
            nxt = _distribute_or_bounded(acc, sub, product_cap)
            if nxt is None:
                return None
            acc = nxt
        return [c for c in acc if c]

    return [[e]]


def _filter_clause_to_vocab(cl: List[ExprRef], allowed: Set[str]) -> List[ExprRef]:
    out: List[ExprRef] = []
    for lit in cl:
        if all(n in allowed for n in _free_names(lit)):
            out.append(lit)
    return out


def _to_clause_expr(clause: List[ExprRef]) -> ExprRef:
    lits=[_strip_labels(l) for l in clause]
    return lits[0] if len(lits)==1 else Or(*lits)

# ─────────────────────────────────────────────
# CNF OR-clause emitter (dedup/subsumption/units-aware)
# ─────────────────────────────────────────────

def _emit_or_constraint_clauses_from_proj(
    Proj: ExprRef, *,
    timeout_ms: int = 3000,
    allowed_names: Optional[Set[str]] = None,
    entailed_true_sexprs: Optional[Set[str]] = None,   # literals known true as units (including negatives)
) -> Tuple[str, Dict[str, int]]:
    """
    Emit clean CNF OR-constraint_clauses from Proj with bounded CNF:
      • NNF → bounded CNF (no let, no embedded and, no Bool '=')
      • Filter literals to allowed_names (stems/numerics)
      • Drop constraint_clauses that are trivially true given entailed units (positive or negative)
      • Keep only constraint_clauses with length ≥ 2
      • SAT-filter each clause against Proj (keep on sat/unknown) using a single reused solver
      • Deduplicate & subsumption-minimize
      • Wall-clock TIME_BUDGET_S guard for the SAT-filter loop
    """
    stats = {"checked": 0, "kept": 0, "dropped": 0, "proj_unsat": 0}
    start_time = time.time()

    # Baseline SAT check on Proj
    S0 = Solver(); S0.set(timeout=timeout_ms); S0.add(Proj)
    if S0.check() == unsat:
        stats["proj_unsat"] = 1
        return "", stats

    # NNF → bounded CNF
    nnf_proj = _nnf(Proj)
    constraint_clauses_opt = _cnf_constraint_clauses_from_nnf_bounded(nnf_proj, MAX_CLAUSE_PRODUCT, MAX_OR_ARITY)
    if constraint_clauses_opt is None:
        log.info("[info] CNF expansion skipped (budget exceeded: cap=%d, or_arity_cap=%d)",
                 MAX_CLAUSE_PRODUCT, MAX_OR_ARITY)
        return "", stats
    all_constraint_clauses = constraint_clauses_opt

    # 1) Filter by vocabulary + dedup literals + drop tautologies / unit-true constraint_clauses
    def _canon_lit(lit: ExprRef) -> str:
        return " ".join(_strip_labels(lit).sexpr().split())

    filtered: List[List[ExprRef]] = []
    for cl in all_constraint_clauses:
        cl2 = _filter_clause_to_vocab(cl, allowed_names) if allowed_names is not None else cl
        if not cl2:
            continue

        # Drop trivially true ORs if any literal is already entailed true
        if entailed_true_sexprs:
            if any(_canon_lit(l) in entailed_true_sexprs for l in cl2):
                continue

        # Dedup literals within a clause
        seen, lits = set(), []
        for lit in cl2:
            s = _canon_lit(lit)
            if s in seen:
                continue
            seen.add(s); lits.append(lit)

        # Drop tautologies p ∨ ¬p for Bool atoms
        pos_atoms, neg_atoms = set(), set()
        for lit in lits:
            ll = _strip_labels(lit)
            if is_app(ll) and ll.decl().name()=="not" and _is_bool_atom_like(ll.arg(0)):
                neg_atoms.add(_name(ll.arg(0)))
            elif _is_bool_atom_like(ll):
                pos_atoms.add(_name(ll))
        if any(a in pos_atoms for a in neg_atoms):
            continue

        if len(lits) >= 2:
            filtered.append(lits)

    # 2) SAT-filter candidates with a single reused solver
    S_proj = Solver(); S_proj.set(timeout=timeout_ms); S_proj.add(Proj)

    def _sat_with_clause(cexpr: ExprRef):
        S_proj.push()
        S_proj.add(cexpr)
        r = S_proj.check()
        S_proj.pop()
        return r

    sat_kept: List[List[ExprRef]] = []
    for cl in filtered:
        if time.time() - start_time > TIME_BUDGET_S:
            log.info("[info] OR-clause SAT-filter loop truncated by time budget (%.1fs)", TIME_BUDGET_S)
            break
        cexpr = _to_clause_expr(cl)
        r = _sat_with_clause(cexpr)
        stats["checked"] += 1
        if r == unsat:
            stats["dropped"] += 1
            continue
        sat_kept.append(cl)
        stats["kept"] += 1

    # 3) Clause-level dedup and subsumption (length-bucketed)
    def canon_clause(cl: List[ExprRef]) -> Tuple[str, ...]:
        return tuple(sorted(" ".join(_strip_labels(l).sexpr().split()) for l in cl))

    uniq_map: Dict[Tuple[str,...], List[ExprRef]] = {}
    for cl in sat_kept:
        uniq_map[canon_clause(cl)] = cl

    uniq_keys = list(uniq_map.keys())
    by_len: Dict[int, List[Tuple[str,...]]] = {}
    for k in uniq_keys:
        by_len.setdefault(len(k), []).append(k)

    kept_keys: List[Tuple[str, ...]] = []
    seen_keys: Set[Tuple[str, ...]] = set()
    max_len = max(by_len) if by_len else 0
    for L in sorted(by_len):
        for k in by_len[L]:
            if k in seen_keys: continue
            kept_keys.append(k)
            sk = set(k)
            for LL in range(L+1, max_len+1):
                for h in by_len.get(LL, []):
                    if h in seen_keys: continue
                    if sk.issubset(set(h)):
                        seen_keys.add(h)

    # 4) Render
    out_lines: List[str] = []
    for k in kept_keys:
        cl = uniq_map[k]
        body = " ".join(_strip_labels(l).sexpr() for l in cl)
        out_lines.append(f"(assert (or {body}))")

    return ("\n".join(out_lines) + ("\n" if out_lines else "")), stats

# ─────────────────────────────────────────────
# Projection
# ─────────────────────────────────────────────

def _proj_with_qe(base:List[ExprRef], keep:List[str], *, sorts:Dict[str,str],
                  strict:bool, cap:int, dbg:Optional[Path])->ExprRef:
    F=And(*base); _dump_expr("pre_QE",F,dbg)

    vars=_collect_vars(F); name2={_name(v):v for v in vars}
    keep_syms={name2[n] for n in keep if n in name2}
    drops=[v for v in vars if v not in keep_syms]

    drop_b,drop_n=[],[]
    for v in drops:
        typ=sorts.get(_name(v),str(v.sort()))
        if typ=="Bool": drop_b.append(v)
        elif typ in("Int","Real"): drop_n.append(v)

    tmp=_bool_qe_shannon(F,drop_b,cap)
    G=tmp if tmp is not None else F
    _dump_expr("after_BoolQE",G,dbg)

    # Optional numeric-QE guard to avoid heavy tactics on large terms
    if drop_n and _term_cost(G) < QE_NUMERIC_TERM_COST_CAP:
        nxt=_apply_qe_any(G,drop_n,strict)
        if nxt is not None: G=nxt
    _dump_expr("after_NumQE",G,dbg)

    return simplify(G,arith_lhs=True,som=True,ctx_simplify=True)

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

class ProjectConfig:
    def __init__(self,*,use_qe_strict=False,emit_binary_implicates=False,
                 timeout_ms=3000,max_bool_shannon=20,
                 debug_dir:Optional[Union[str,Path]]=None,
                 db_constraint_clauses_out:Optional[Union[str,Path]]=None):
        self.use_qe_strict=use_qe_strict
        self.emit_binary_implicates=emit_binary_implicates
        self.timeout_ms=timeout_ms
        self.max_bool_shannon=max_bool_shannon
        self.debug_dir=Path(debug_dir) if debug_dir else None
        self.db_constraint_clauses_out=Path(db_constraint_clauses_out) if db_constraint_clauses_out else None

def project_constraints_for_file(
    smt_path:str, canon_bools:List[str], *, canon_nums:Optional[List[str]]=None,
    cfg:Optional[ProjectConfig]=None, emit_projected_smt=True
)->Tuple[Dict[str,Any],str]:

    cfg=cfg or ProjectConfig()
    healed,heal_meta=_heal_to_temp(smt_path)

    # Parse
    s = Solver(); s.set(timeout=cfg.timeout_ms)
    try:
        parsed = parse_smt2_file(healed)
        s.add(*parsed if isinstance(parsed, list) else list(parsed))
    except Z3Exception as e:
        # Write diagnostics next to other dumps, if debug_dir is configured
        perr = _write_parse_error_diagnostics(healed, e, cfg.debug_dir)
        summary = {
            "file": Path(smt_path).name,
            "assertion_count": 0,
            "sat_status": "unknown",
            "parse_error": True,
            "parse_error_message": perr.get("error", ""),
            "parse_error_line": perr.get("line"),
            "parse_error_col": perr.get("col"),
            "parse_error_context_file": perr.get("context_file"),
        }
        return summary, ";; parse error\n"

    raw=list(s.assertions())
    txt=Path(healed).read_text(encoding="utf-8")
    sorts=_decl_sorts(txt)

    # Always-on UNSAT-core pruning
    raw, uc_meta = _prune_by_unsat_core(raw, cfg.timeout_ms)

    # Restrict keep-set strictly to the provided minified_canon vocabulary.
    # Filter by declared sorts so Bool/Int/Real不会混淆。
    canon_bools = [n for n in (canon_bools or []) if sorts.get(n) == "Bool"]

    if canon_nums is None:
        # 不再默认“全部 Int/Real”，而是必须显式提供 numeric 白名单
        canon_nums = []
    else:
        canon_nums = [n for n in canon_nums if sorts.get(n) in ("Int", "Real")]

    keep0 = set(canon_bools) | set(canon_nums)


    # Definitional closure
    def_eqs=_collect_def_eqs(raw)
    keep_closed=_def_closure(keep0,def_eqs)
    added=sorted(keep_closed-keep0)
    if added:
        log.info("[info] definitional-closure enlarged keep-set by %d: %s",
                 len(added), ", ".join(added[:20]) + (" ..." if len(added)>20 else ""))

    # Orient & preserve
    oriented=_orient_and_prune(raw, keep_closed, def_eqs)
    de = _preserve_defs_for_nonkept(oriented, keep_closed, sorts)
    _dump_expr("after_DEE",And(*de),cfg.debug_dir)

    # Projection
    Proj=_proj_with_qe(
        de,sorted(keep_closed),sorts=sorts,strict=cfg.use_qe_strict,
        cap=cfg.max_bool_shannon,dbg=cfg.debug_dir
    )
    _dump_expr("post_QE",Proj,cfg.debug_dir)

    # DB-friendly clause file (optional) — reuse one solver inside
    db_stats = {"checked":0,"kept":0,"dropped":0,"proj_unsat":0}
    try:
        db_text, db_stats = _emit_db_friendly_clause_smt(
            Proj, timeout_ms=cfg.timeout_ms, require_sat=True
        )
        if cfg.db_constraint_clauses_out is not None:
            cfg.db_constraint_clauses_out.parent.mkdir(parents=True, exist_ok=True)
            cfg.db_constraint_clauses_out.write_text(db_text, encoding="utf-8")
            log.info("[ok] wrote DB-friendly constraint_clauses → %s (kept=%d, dropped=%d, checked=%d)",
                     str(cfg.db_constraint_clauses_out), db_stats["kept"], db_stats["dropped"], db_stats["checked"])
        elif cfg.debug_dir is not None:
            db_path = cfg.debug_dir / "zz_db_constraint_clauses.smt2"
            db_path.write_text(db_text, encoding="utf-8")
            log.info("[ok] wrote DB-friendly constraint_clauses → %s (kept=%d, dropped=%d, checked=%d)",
                     db_path.name, db_stats["kept"], db_stats["dropped"], db_stats["checked"])
    except Exception as e:
        log.error("[warn] failed writing DB-friendly constraint_clauses: %s", e)

    # Reused solver for entailment checks
    S_entail = Solver(); S_entail.set(timeout=cfg.timeout_ms); S_entail.add(Proj)
    def _entailed(phi: ExprRef) -> bool:
        S_entail.push()
        S_entail.add(Not(phi))
        r = S_entail.check()
        S_entail.pop()
        return r == unsat

    # Backbone on kept Booleans
    must_t=[n for n in canon_bools if _entailed(Bool(n))]
    must_f=[n for n in canon_bools if _entailed(Not(Bool(n)))]
    incon=[n for n in canon_bools if n not in must_t and n not in must_f]

    # Numeric atoms on kept numerics (negation-aware)
    def _num_atoms():
        allow=set(canon_nums); atoms: List[ExprRef] = []

        def allowed_vars_only(expr: ExprRef) -> bool:
            return all(_name(v) in allow for v in _collect_vars(expr))

        def walk(e: ExprRef):
            # Try to view e as a (possibly normalized) numeric atom
            cand = _as_positive_numeric_atom(e)
            if cand is not None and allowed_vars_only(cand):
                atoms.append(cand)
            # Continue traversal regardless
            e = _strip_labels(e)
            if is_app(e):
                for i in range(e.num_args()):
                    walk(e.arg(i))

        walk(Proj)
        seen,out=set(),[]
        for a in atoms:
            sx=" ".join(a.sexpr().split())
            if sx not in seen and _entailed(a):
                seen.add(sx); out.append(a)
        return out

    num_units=_num_atoms()
    sat_status="unsat" if _unsat(Proj) else "sat_or_unknown"

    summary={
        "file":Path(smt_path).name,
        "assertion_count":len(raw),
        "sat_status":sat_status,
        "canon_bool_count":len(canon_bools),
        "canon_numeric_count":len(canon_nums),
        "must_true":must_t,
        "must_false":must_f,
        "inconclusive_bools":incon,
        "numeric_unit_count":len(num_units),
        "closure_added": added,
        "inference_auto_decls_added":heal_meta["auto_decls_added"],
        "inference_fixed_decl_lines":heal_meta["fixed_decl_lines"],
        # DB clause SAT-filter stats
        "db_sat_filter_enabled": True,
        "db_constraint_clauses_checked": db_stats["checked"],
        "db_constraint_clauses_kept": db_stats["kept"],
        "db_constraint_clauses_dropped": db_stats["dropped"],
        "db_proj_unsat": bool(db_stats["proj_unsat"]),
        # UNSAT-core pruning stats
        "unsat_core_pruning_rounds": uc_meta["rounds"],
        "unsat_core_pruned_total": uc_meta["pruned_total"],
        "unsat_core_last_size": uc_meta["last_core_size"],
        "unsat_core_final_status": uc_meta["final_status"],
    }

    if not emit_projected_smt:
        return summary, ""

    # Allowed vocabulary = STEMS ONLY (canonical bools + numerics)
    allowed_names: Set[str] = set(canon_bools) | set(canon_nums)

    # Build set of entailed-true literals to omit trivial ORs (include negatives!)
    entailed_true_sexprs: Set[str] = set()
    for n in must_t:
        entailed_true_sexprs.add(n)  # Bool atom true
    for n in must_f:
        entailed_true_sexprs.add(" ".join(Not(Bool(n)).sexpr().split()))  # add the negated atom
    for a in num_units:
        entailed_true_sexprs.add(" ".join(a.sexpr().split()))

    # Emit projected SMT (units + stem-only CNF OR-constraint_clauses)
    out=[]
    out.append(";; ===================== PROJECTED CANON (QE) =====================")
    out.append(";; Each clause below is entailed by Proj = ∃W.F over the canonical vocabulary.")
    out.append(";; Units appear first for clarity; OR-constraint_clauses (SAT-filtered) follow.")
    out.append(";; ==================================================================")

    if must_t or must_f:
        out.append("\n;; Unit Bool constraints (entailed)")
    out.extend(f"(assert {n})" for n in must_t)
    out.extend(f"(assert (not {n}))" for n in must_f)

    if num_units:
        out.append("\n;; Numeric unit constraints (entailed)")
        seen=set()
        for a in num_units:
            sx=" ".join(a.sexpr().split())
            if sx in seen: continue
            seen.add(sx); out.append(f"(assert {a.sexpr()})")

    # Append SAT-filtered CNF OR-constraint_clauses (STEMS ONLY)
    or_text, or_stats = _emit_or_constraint_clauses_from_proj(
        Proj,
        timeout_ms=cfg.timeout_ms,
        allowed_names=allowed_names,
        entailed_true_sexprs=entailed_true_sexprs,
    )
    if or_text:
        out.append("\n;; Additional OR-constraint_clauses (SAT-filtered)")
        out.append(or_text.rstrip())

    out.append("")
    smt_blob = "\n".join(out)

    # Extend summary with OR-clause stats
    summary.update({
        "ored_constraint_clauses_checked": or_stats.get("checked", 0),
        "ored_constraint_clauses_kept":    or_stats.get("kept", 0),
        "ored_proj_unsat":      bool(or_stats.get("proj_unsat", 0)),
    })

    return summary, smt_blob

# ─────────────────────────────────────────────
# DB-friendly clause emitter (aligned with CNF pipeline, handles negative units)
# ─────────────────────────────────────────────

def _emit_db_friendly_clause_smt(
    Proj: ExprRef, *, timeout_ms: int = 3000, require_sat: bool = True
) -> Tuple[str, Dict[str,int]]:
    stats = {"checked": 0, "kept": 0, "dropped": 0, "proj_unsat": 0}

    s0 = Solver(); s0.set(timeout=timeout_ms); s0.add(Proj)
    r0 = s0.check()
    if r0 == unsat:
        stats["proj_unsat"] = 1
        log.warning("[warn] projected formula is UNSAT; no DB-friendly constraint_clauses emitted")
        return "", stats

    # Reuse entailment solver to gather unit-entailed literals (positive and negative)
    S_entail = Solver(); S_entail.set(timeout=timeout_ms); S_entail.add(Proj)

    def _entailed(phi: ExprRef) -> bool:
        S_entail.push()
        S_entail.add(Not(phi))
        r = S_entail.check()
        S_entail.pop()
        return r == unsat

    entailed_true_sexprs: Set[str] = set()

    # Collect Bool (including negative) and numeric unit literals entailed by Proj
    def _collect_units() -> Tuple[List[ExprRef], List[ExprRef]]:
        bool_units: List[ExprRef] = []
        num_units: List[ExprRef] = []

        def walk(e: ExprRef):
            # NUMERIC: normalize negated comparators to positive atoms
            cand = _as_positive_numeric_atom(e)
            if cand is not None and _entailed(cand):
                num_units.append(simplify(cand, arith_lhs=True))
            # BOOL atoms (both signs)
            ee = _strip_labels(e)
            if _is_bool_atom_like(ee):
                if _entailed(ee): bool_units.append(ee)
                elif _entailed(Not(ee)): bool_units.append(Not(ee))
            # Recurse into children for all apps (including 'not')
            if is_app(ee):
                for i in range(ee.num_args()):
                    walk(ee.arg(i))

        walk(Proj)
        return bool_units, num_units

    b_units, n_units = _collect_units()
    for b in b_units:
        entailed_true_sexprs.add(" ".join(_strip_labels(b).sexpr().split()))
    for n in n_units:
        entailed_true_sexprs.add(" ".join(simplify(n, arith_lhs=True).sexpr().split()))

    # Build OR-constraint_clauses with the same pipeline (no vocab restriction here)
    or_text, or_stats = _emit_or_constraint_clauses_from_proj(
        Proj, timeout_ms=timeout_ms, allowed_names=None, entailed_true_sexprs=entailed_true_sexprs
    )
    stats.update({
        "checked": or_stats["checked"],
        "kept": or_stats["kept"],
        "dropped": or_stats["dropped"],
        "proj_unsat": or_stats["proj_unsat"],
    })

    # Units first, then OR-constraint_clauses — AND-of-asserts overall
    out_lines: List[str] = []
    for b in b_units:
        out_lines.append(f"(assert { _strip_labels(b).sexpr() })")
    seen_num = set()
    for n in n_units:
        s = " ".join(simplify(n, arith_lhs=True).sexpr().split())
        if s in seen_num: continue
        seen_num.add(s)
        out_lines.append(f"(assert {simplify(n, arith_lhs=True).sexpr()})")
    if or_text:
        out_lines.append(or_text.rstrip())

    return ("\n".join(out_lines) + ("\n" if out_lines else "")), stats

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _cli() -> int:
    # declare globals BEFORE any reads/writes
    global TIME_BUDGET_S, MAX_CLAUSE_PRODUCT, MAX_OR_ARITY, QE_NUMERIC_TERM_COST_CAP

    ap = argparse.ArgumentParser(
        description=(
            "Signature-closure QE projector (def-oriented) + DB clause emitter (SAT filtered) "
            "+ stem-only CNF OR-constraint_clauses in final SMT (dedup/subsumption/unit-triviality removal) "
            "with performance bounds"
        )
    )
    ap.add_argument("smt_file")
    ap.add_argument("--bools", nargs="*", default=[], help="Canonical Bool names to KEEP (stems only)")
    ap.add_argument("--nums", nargs="*", default=None,
                    help="Canonical numeric names to KEEP (default: all Int/Real)")
    ap.add_argument("--strict-qe", action="store_true", help="Prefer heavy 'qe' tactic")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--max-bool-shannon", type=int, default=20)
    ap.add_argument("--debug-dir", default=None, help="Directory to dump intermediate SMT2 files")
    ap.add_argument("--db-constraint_clauses-out", default=None,
                    help=("Write DB-friendly constraint_clauses (units/OR-constraint_clauses) to this SMT2 file; "
                          "if omitted, will write zz_db_constraint_clauses.smt2 into --debug-dir (if provided)"))
    ap.add_argument("--no-smt-out", action="store_true")
    ap.add_argument("--json", default=None, help="Write JSON summary here")

    # optional tuning flags — defaults read current globals
    ap.add_argument("--time-budget-s", type=float, default=TIME_BUDGET_S,
                    help="Wall-clock budget for OR-clause SAT filtering")
    ap.add_argument("--max-clause-product", type=int, default=MAX_CLAUSE_PRODUCT,
                    help="Max product during CNF distribution")
    ap.add_argument("--max-or-arity", type=int, default=MAX_OR_ARITY,
                    help="Max OR arity allowed during CNF")
    ap.add_argument("--qe-numeric-term-cap", type=int, default=QE_NUMERIC_TERM_COST_CAP,
                    help="Skip numeric QE if term cost exceeds this")

    args = ap.parse_args()

    # update globals from flags
    TIME_BUDGET_S = float(args.time_budget_s)
    MAX_CLAUSE_PRODUCT = int(args.max_clause_product)
    MAX_OR_ARITY = int(args.max_or_arity)
    QE_NUMERIC_TERM_COST_CAP = int(args.qe_numeric_term_cap)

    cfg = ProjectConfig(use_qe_strict=args.strict_qe,
                        timeout_ms=args.timeout_ms,
                        max_bool_shannon=args.max_bool_shannon,
                        debug_dir=args.debug_dir,
                        db_constraint_clauses_out=args.db_constraint_clauses_out)

    summary, smt_out = project_constraints_for_file(
        args.smt_file,
        canon_bools=args.bools,
        canon_nums=args.nums,
        cfg=cfg,
        emit_projected_smt=not args.no_smt_out
    )

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log.info(f"[ok] wrote JSON → {args.json}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if smt_out:
        sys.stdout.write("\n" + smt_out)
    return 0

if __name__=="__main__":
    raise SystemExit(_cli())
