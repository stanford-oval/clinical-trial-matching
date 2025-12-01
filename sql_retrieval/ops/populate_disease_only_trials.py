#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
populate_disease_only_trials.py — create placeholder rows in `trials` for disease-only NCTs.

- Scans disease tables (disease_constraint_atoms, disease_constraint_alternatives) for trial ids.
- Compares to trials.nct_id and inserts missing trials.
- If trials.*_trial_side_id columns are NOT NULL, creates empty trial_constraint_sides entries.

Usage:
  python populate_disease_only_trials.py --db path/to/db.sqlite [--dli disease_constraint_atoms] [--daa disease_constraint_alternatives] [--dry-run] [--verbose]
"""

import argparse, sqlite3, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set

def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    return bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())

def pragma_table_info(cur: sqlite3.Cursor, table: str):
    # cid, name, type, notnull, dflt_value, pk
    return cur.execute(f"PRAGMA table_info({table})").fetchall()

def pick_col(cur: sqlite3.Cursor, table: str, candidates: List[str]) -> Optional[str]:
    cols = {r[1] for r in pragma_table_info(cur, table)} if table_exists(cur, table) else set()
    for c in candidates:
        if c in cols:
            return c
    return None

def normalize_nct(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return x.strip().upper()

def discover_disease_ncts(cur: sqlite3.Cursor, dli: Optional[str], daa: Optional[str]) -> Set[str]:
    ncts: Set[str] = set()
    parts: List[Tuple[str, str]] = []

    if daa and table_exists(cur, daa):
        c_trial = pick_col(cur, daa, ["trial_id","nct_id"])
        if c_trial:
            parts.append((daa, c_trial))
    if dli and table_exists(cur, dli):
        c_trial = pick_col(cur, dli, ["trial_id","nct_id"])
        if c_trial:
            parts.append((dli, c_trial))

    for tbl, col in parts:
        for (nct,) in cur.execute(f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL"):
            n = normalize_nct(nct)
            if n:
                ncts.add(n)
    return ncts

def fetch_trials_ncts(cur: sqlite3.Cursor, trials_nct_col: str) -> Set[str]:
    have: Set[str] = set()
    for (nct,) in cur.execute(f"SELECT DISTINCT {trials_nct_col} FROM trials"):
        n = normalize_nct(nct)
        if n:
            have.add(n)
    return have

def ensure_empty_trial_side(cur: sqlite3.Cursor, kind_value: str) -> int:
    """
    Creates an empty row in trial_constraint_sides and returns its id.
    Tries to satisfy minimal NOT NULLs with simple defaults.
    """
    cols = pragma_table_info(cur, "trial_constraint_sides")
    colnames = [c[1] for c in cols]
    notnull = {c[1] for c in cols if c[3] == 1 and c[5] == 0}  # NOT NULL and not PK
    # Common columns we might fill
    values: Dict[str, object] = {}
    if "kind" in colnames:
        values["kind"] = kind_value
    if "title" in colnames and "title" in notnull:
        values["title"] = f"{kind_value} (auto)"
    # Build insert
    keys = ", ".join(values.keys())
    qms  = ", ".join(["?"]*len(values))
    sql = f"INSERT INTO trial_constraint_sides ({keys}) VALUES ({qms})"
    cur.execute(sql, tuple(values.values()))
    return cur.lastrowid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dli", default="disease_constraint_atoms")
    ap.add_argument("--daa", default="disease_constraint_alternatives")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Basic checks
    if not table_exists(cur, "trials"):
        print("[error] missing `trials` table", file=sys.stderr); sys.exit(2)

    trials_cols = pragma_table_info(cur, "trials")
    trials_colnames = {c[1] for c in trials_cols}
    trials_nct_col = "nct_id" if "nct_id" in trials_colnames else None
    if not trials_nct_col:
        print("[error] `trials` has no nct_id column", file=sys.stderr); sys.exit(2)

    # Identify side-id columns & their nullability
    side_cols = [c for c in trials_cols if c[1] in ("inclusion_trial_side_id","exclusion_trial_side_id","assumed_trial_side_id")]
    side_notnull = {c[1] for c in side_cols if c[3] == 1}  # NOT NULL
    need_sides = bool(side_notnull)

    # Collect disease NCTs and diff
    disease_ncts = discover_disease_ncts(cur, args.dli, args.daa)
    existing_ncts = fetch_trials_ncts(cur, trials_nct_col)
    missing_ncts = sorted(n for n in disease_ncts - existing_ncts)

    if args.verbose:
        print(f"[info] disease NCTs: {len(disease_ncts)}; trials NCTs: {len(existing_ncts)}; missing: {len(missing_ncts)}")

    if not missing_ncts:
        print("[ok] no missing disease-only trials")
        return

    if args.dry_run:
        print("[dry-run] would insert placeholders for:", ", ".join(missing_ncts[:50]), (" ..." if len(missing_ncts) > 50 else ""))
        return

    try:
        conn.execute("BEGIN")
        # Pre-create empty side rows if required
        side_ids: Dict[str,int] = {}
        if need_sides:
            if not table_exists(cur, "trial_constraint_sides"):
                raise RuntimeError("trials.*_trial_side_id is NOT NULL but `trial_constraint_sides` table is missing")
            # create one empty of each kind, reuse for all inserts
            for kind, col in [("inclusion","inclusion_trial_side_id"),
                              ("exclusion","exclusion_trial_side_id"),
                              ("inclusion","assumed_trial_side_id")]:
                if col in side_notnull:
                    side_ids[col] = ensure_empty_trial_side(cur, "inclusion" if "inclusion" in col else "exclusion")

        # Build INSERT statement dynamically
        insert_cols = [trials_nct_col]
        insert_vals = ["?"]
        for col in ("inclusion_trial_side_id","exclusion_trial_side_id","assumed_trial_side_id"):
            if col in trials_colnames:
                insert_cols.append(col)
                insert_vals.append("?")

        sql = f"INSERT INTO trials ({', '.join(insert_cols)}) VALUES ({', '.join(insert_vals)})"

        for nct in missing_ncts:
            params: List[object] = [nct]
            for col in ("inclusion_trial_side_id","exclusion_trial_side_id","assumed_trial_side_id"):
                if col in trials_colnames:
                    if col in side_notnull:
                        params.append(side_ids[col])
                    else:
                        params.append(None)
            cur.execute(sql, params)

        conn.commit()
        print(f"[ok] inserted {len(missing_ncts)} placeholder trial(s).")

    except Exception as e:
        conn.rollback()
        print(f"[error] failed to insert placeholders: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
