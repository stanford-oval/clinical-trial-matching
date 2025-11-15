#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expand_outcome_implications.py

Reads JSONL from --in (or stdin), writes JSONL to --out (or stdout).
For each record, it:
  1) Upgrades legacy alias procedure outcome vars (no timeframe) to canonical form
     when a timeframe is known/recoverable.
  2) Emits outcome- & timeframe-preserving hypernym entailments for any canonical
     procedure-outcome source var.
  3) Dedupes the 'entailed' list.

Input row (minimal):
  {
    "source_variable_name": "patient_has_undergone_ultrasonography_of_abdomen_now_outcome_is_abnormal",
    "source_value": true,
    // optional:
    "timeframe": "now",
    "entailed": [{"variable": "...", "value": true}, ...]
  }

Output row: same schema, with 'entailed' expanded and alias names upgraded when possible.

Env:
  IMP_PROC_OUTCOME_HYPERNYMS=/path/to/hypernyms.json   (optional)
"""

from __future__ import annotations
import argparse, json, re, sys
from typing import Any, Dict, List

from implications_procedure_outcomes import (
    outcome_hypernym_entailments,
    upgrade_or_identity,
)

# Try to recover timeframe from anywhere reasonable in the row
_TF_RX = re.compile(r"(now|inthehistory|inthepast\d+days)$", re.I)

def _recover_tf(row: Dict[str, Any]) -> str|None:
    # Prefer explicit row['timeframe']
    tf = (row.get("timeframe") or "").strip().lower()
    if tf and _TF_RX.search(tf):
        return tf
    # Next try parsing from the var itself
    vn = (row.get("source_variable_name") or "").strip()
    m = _TF_RX.search(vn)
    if m:
        return m.group(1).lower()
    return None

def _dedupe_entailed(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items or []:
        v = (it.get("variable") or "").strip()
        val = bool(it.get("value"))
        key = f"{v}|{int(val)}"
        if v and key not in seen:
            out.append({"variable": v, "value": val})
            seen.add(key)
    return out

def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    vn = (row.get("source_variable_name") or "").strip()
    sv = bool(row.get("source_value", False))

    # Best-effort TF (helps upgrade legacy aliases if any)
    tf = _recover_tf(row)

    # 1) Upgrade the *source* var to canonical if it's a legacy alias
    upgraded_vn = upgrade_or_identity(vn, known_tf=tf)
    row["source_variable_name"] = upgraded_vn

    # 2) Start with any existing entailed
    entailed: List[Dict[str, Any]] = list(row.get("entailed") or [])

    # 3) If canonical outcome-bearing, add hypernym entailments preserving tf/outcome
    pack = outcome_hypernym_entailments(upgraded_vn)
    if pack:
        for v in pack["entailed"]:
            entailed.append({"variable": v, "value": sv})

    # 4) Also, if existing entailed entries are legacy aliases, try to upgrade them
    fixed_entailed: List[Dict[str, Any]] = []
    for it in entailed:
        name = (it.get("variable") or "").strip()
        val  = bool(it.get("value"))
        fixed_name = upgrade_or_identity(name, known_tf=tf)
        fixed_entailed.append({"variable": fixed_name, "value": val})

    # 5) Deduplicate
    row["entailed"] = _dedupe_entailed(fixed_entailed)

    return row

def main():
    ap = argparse.ArgumentParser(description="Outcome/timeframe-preserving expansion for procedure variables.")
    ap.add_argument("--in",  dest="inp",  type=str, default="-", help="Input JSONL (default: stdin)")
    ap.add_argument("--out", dest="out", type=str, default="-", help="Output JSONL (default: stdout)")
    args = ap.parse_args()

    fin  = sys.stdin  if args.inp == "-" else open(args.inp, "r", encoding="utf-8")
    fout = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")

    try:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            out_row = process_row(row)
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    finally:
        if fin  is not sys.stdin:  fin.close()
        if fout is not sys.stdout: fout.close()

if __name__ == "__main__":
    main()
