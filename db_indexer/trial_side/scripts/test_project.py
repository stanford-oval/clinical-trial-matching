#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
project_one.py — run QE projection for a single SMT file with one canon JSON.
All outputs go to ./test_projection_result by default.

Usage:
  python project_one.py --smt path/to/NCTxxxx.smt2 \
    --canon path/to/xxx_canonical_variables.json \
    [--outdir ./test_projection_result] [--strict-qe] [--timeout-ms 4000]

What you get in --outdir:
  - <stem>_projected.smt2        : projected SMT
  - <stem>_summary.json          : projector JSON summary
  - <stem>_diagnostics.txt       : numeric-focused human-readable report
  - _debug/                       : projector intermediate dumps (if enabled)
  - db_constraint_clauses.smt2               : DB-friendly units/OR-constraint_clauses (optional)
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, re, sys
from typing import List, Dict, Set

# Adjust import path if your layout differs
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str((_Path("../../..") / "TrialGPT-SMT" / "irsrc" / "trial_side").resolve()))
from smt_projector import ProjectConfig as QEConfig, project_constraints_for_file  # type: ignore

DECL_CONST_RE = re.compile(r"\(declare-const\s+(\S+)\s+(\S+)\)")
DECL_FUN_RE   = re.compile(r"\(declare-fun\s+(\S+)\s+\([^\)]*\)\s+(\S+)\)")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-trial SMT QE projector (numeric diagnostics).")
    ap.add_argument("--smt", required=True, help="Path to the SMT2 file")
    ap.add_argument("--canon", required=True, help="Path to canonical_variables*.json (list of variable names)")
    ap.add_argument("--outdir", default="./test_projection_result", help="Output directory (default: ./test_projection_result)")
    ap.add_argument("--strict-qe", action="store_true", help="Prefer heavy 'qe' tactic")
    ap.add_argument("--timeout-ms", type=int, default=4000, help="Solver timeout (ms)")
    ap.add_argument("--no-debug-dumps", action="store_true", help="Disable projector debug dumps")
    ap.add_argument("--emit-db-constraint_clauses", action="store_true", help="Also write DB-friendly constraint_clauses to outdir/db_constraint_clauses.smt2")
    return ap.parse_args()

def load_canon_vars(p: Path) -> List[str]:
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen, out = set(), []
    for x in arr:
        if isinstance(x, str):
            s = x.strip()
            if s and s not in seen:
                out.append(s); seen.add(s)
    return out

def decl_sorts(txt: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in DECL_CONST_RE.finditer(txt):
        out[m.group(1)] = m.group(2)
    for m in DECL_FUN_RE.finditer(txt):
        out[m.group(1)] = m.group(2)
    return out

def extract_numeric_unit_asserts(projected_smt: str, canon_numeric: Set[str]) -> List[str]:
    lines = []
    for ln in projected_smt.splitlines():
        s = ln.strip()
        if not s.startswith("(assert"):
            continue
        if any(v in s for v in canon_numeric):
            if any(tok in s for tok in (">=", "<=", "<", ">", " = ")):
                lines.append(ln)
    return lines

def main() -> None:
    args = parse_args()
    smt_path = Path(args.smt).resolve()
    canon_path = Path(args.canon).resolve()
    outdir = Path(args.outdir).resolve()

    if not smt_path.exists():
        print(f"[error] SMT file not found: {smt_path}", file=sys.stderr); sys.exit(2)
    if not canon_path.exists():
        print(f"[error] canon JSON not found: {canon_path}", file=sys.stderr); sys.exit(2)

    outdir.mkdir(parents=True, exist_ok=True)
    debug_dir = None if args.no_debug_dumps else (outdir / "_debug")
    db_constraint_clauses_out = (outdir / "db_constraint_clauses.smt2") if args.emit_db_constraint_clauses else None

    canon_vars = load_canon_vars(canon_path)
    if not canon_vars:
        print(f"[error] canon JSON appears empty or invalid: {canon_path}", file=sys.stderr); sys.exit(2)

    smt_text = smt_path.read_text(encoding="utf-8", errors="replace")
    sorts = decl_sorts(smt_text)
    declared_bools  = {n for n, srt in sorts.items() if srt == "Bool"}
    declared_nums   = {n for n, srt in sorts.items() if srt in ("Int", "Real")}
    canon_set       = set(canon_vars)
    canon_as_bools  = sorted(canon_set & declared_bools)
    canon_as_nums   = sorted(canon_set & declared_nums)
    canon_unknown   = sorted([n for n in canon_set if n not in sorts])

    cfg = QEConfig(
        use_qe_strict=args.strict_qe,
        timeout_ms=args.timeout_ms,
        debug_dir=str(debug_dir) if debug_dir else None,
        db_constraint_clauses_out=str(db_constraint_clauses_out) if db_constraint_clauses_out else None,
    )

    # Pass canon to BOTH bools and nums; projector will type-filter internally.
    summary, projected = project_constraints_for_file(
        str(smt_path),
        canon_bools=canon_vars,
        canon_nums=canon_vars,
        cfg=cfg,
        emit_projected_smt=True,
    )

    # Write outputs
    stem = smt_path.stem
    out_smt = outdir / f"{stem}_projected.smt2"
    out_json = outdir / f"{stem}_summary.json"
    out_diag = outdir / f"{stem}_diagnostics.txt"
    out_smt.write_text(projected, encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Build diagnostics text
    numeric_unit_lines = extract_numeric_unit_asserts(projected, set(canon_as_nums))
    diag = []
    diag.append("========== PROJECTION SUMMARY ==========")
    diag.append(f"file: {summary.get('file')}")
    diag.append(f"sat_status: {summary.get('sat_status')}")
    diag.append(f"assertion_count: {summary.get('assertion_count')}")
    diag.append(f"canon_bool_count (kept): {summary.get('canon_bool_count')}  "
                f"canon_numeric_count (kept): {summary.get('canon_numeric_count')}")
    diag.append(f"unsat_core_pruned_total: {summary.get('unsat_core_pruned_total')} "
                f"(rounds={summary.get('unsat_core_pruning_rounds')}, final={summary.get('unsat_core_final_status')})")
    diag.append(f"numeric_unit_count (summary): {summary.get('numeric_unit_count')}")
    diag.append(f"OR-constraint_clauses kept: {summary.get('ored_constraint_clauses_kept')} / checked: {summary.get('ored_constraint_clauses_checked')}")
    diag.append(f"must_true booleans (first 10): {(summary.get('must_true') or [])[:10]}")
    diag.append(f"must_false booleans (first 10): {(summary.get('must_false') or [])[:10]}")
    diag.append("")
    diag.append("========== VOCAB DIAGNOSTICS ==========")
    diag.append(f"Declared Bool ∩ canon: {len(canon_as_bools)}")
    diag.append(f"Declared Numeric ∩ canon: {len(canon_as_nums)}")
    if canon_as_nums:
        diag.append(f"Numeric names kept (first 15): {canon_as_nums[:15]}")
    if canon_unknown:
        diag.append(f"[warn] {len(canon_unknown)} canon names not declared in SMT (up to 15): {canon_unknown[:15]}")
    age_like = [n for n in canon_as_nums if "age" in n]
    diag.append(f"Age-like numeric vars present: {age_like or 'NONE'}")
    diag.append("")
    diag.append("========== NUMERIC UNIT ASSERTS (from projected SMT) ==========")
    if numeric_unit_lines:
        diag.extend(numeric_unit_lines)
    else:
        diag.append("(none found)")
    if summary.get("numeric_unit_count", 0) == 0:
        diag.append("")
        diag.append("========== WHY NUMERIC UNITS MIGHT BE MISSING ==========")
        diag.append("- The numeric variable names aren’t in the SMT (see [warn] above).")
        diag.append("- The variable exists but is not Int/Real (check its declare-const sort).")
        diag.append("- The numeric constraints aren’t globally entailed (gated by dropped Booleans).")
        diag.append("- allowed_names filtering removed them (only canon_bools ∪ canon_nums survive).")
        diag.append("- QE numeric term cost guard skipped elimination (try --strict-qe and higher --timeout-ms).")
        diag.append("- OR-constraint_clauses may be redundant given units and were SAT-filtered out.")

    out_diag.write_text("\n".join(diag) + "\n", encoding="utf-8")

    print(f"[ok] projected SMT     → {out_smt}")
    print(f"[ok] summary JSON       → {out_json}")
    print(f"[ok] diagnostics TXT    → {out_diag}")
    if debug_dir:
        print(f"[ok] debug dumps dir    → {debug_dir}")
    if db_constraint_clauses_out:
        print(f"[ok] DB constraint_clauses SMT     → {db_constraint_clauses_out}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted.", file=sys.stderr)
        sys.exit(130)
