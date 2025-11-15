#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_disease_ingest.py

Given a single *_disease_link_filter_summary.json file, show:

  1) What rows trial_disease_to_sqlite.py *would* upsert
     (trial_id, disease, stem, stem_var, conceptId, etc.)
  2) Which effective trial_ids it will fan out to (cohorts).
  3) Whether those rows already exist in disease_constraint_atoms in the DB.

This is read-only: it does NOT write anything to the DB.
"""

from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple

# We reuse the real ingestion helpers to stay in sync.
import trial_disease_to_sqlite as tds


def preview_rows(json_path: Path, preproc_dir: Path, prefer_side: str) -> Tuple[str, str, List[Tuple]]:
    """Load JSON and run the same extract_rows + find_effective_trial_ids."""
    blob = tds.load_summary(json_path)
    parent_tid, generated, rows = tds.extract_rows(blob)
    eff_ids = tds.find_effective_trial_ids(parent_tid, preproc_dir, prefer_side)
    return parent_tid, generated, eff_ids, list(rows)


def check_existing(conn: sqlite3.Connection, trial_ids: List[str]):
    """Fetch existing rows in disease_constraint_atoms for the given trial_ids."""
    if not trial_ids:
        return []

    q_marks = ",".join(["?"] * len(trial_ids))
    sql = f"""
      SELECT trial_id, disease, stem, stem_var, conceptId
      FROM disease_constraint_atoms
      WHERE trial_id IN ({q_marks})
      ORDER BY trial_id, disease, stem
    """
    cur = conn.execute(sql, trial_ids)
    return cur.fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test whether a *_disease_link_filter_summary.json file can be effectively ingested."
    )
    ap.add_argument("--json", required=True,
                    help="Path to ONE *_disease_link_filter_summary.json file")
    ap.add_argument("--db", default="../../build/trial.db",
                    help="Path to SQLite DB (default: ../../build/trial.db)")
    ap.add_argument("--preproc-dir", default="mbench/req_mbench/preproc_logs",
                    help="Dir with {trial_id}_{side}.pre.normalized.json (same as trial_disease_to_sqlite)")
    ap.add_argument("--prefer-side", choices=["inclusion", "exclusion"], default="inclusion")
    args = ap.parse_args()

    json_path = Path(args.json).expanduser().resolve()
    preproc_dir = Path(args.preproc_dir).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()

    if not json_path.is_file():
        print(f"[ERROR] JSON file not found: {json_path}")
        return 2

    if not db_path.is_file():
        print(f"[ERROR] DB file not found: {db_path}")
        return 2

    print(f"=== JSON: {json_path.name} ===")
    parent_tid, generated, eff_ids, rows = preview_rows(json_path, preproc_dir, args.prefer_side)

    print(f"parent_trial_id: {parent_tid}")
    print(f"generated:       {generated or '(none)'}")
    print(f"effective_ids:   {', '.join(eff_ids)}")
    print()

    if not rows:
        print("[WARN] extract_rows() produced NO disease rows.")
        print("       (No final_selected_concept_by_disease / linked_result concepts?)")
        return 0

    print("=== Rows that WOULD be upserted (per effective trial_id) ===")
    for tid_eff in eff_ids:
        print(f"\n-- effective trial_id = {tid_eff} --")
        for (_parent_tid, gen, disease, var_name, stem, stem_var,
             conceptId, preferred_term, fsn, ctype, definition, best_match_term) in rows:
            print(f"  disease   : {disease}")
            print(f"  conceptId : {conceptId or '(none)'}")
            print(f"  stem      : {stem}")
            print(f"  stem_var  : {stem_var}")
            print(f"  var_name  : {var_name}")
            print(f"  term      : {preferred_term or fsn or '(no name)'}")
            print("  ---")

    # Now check what's actually in the DB
    print("\n=== Checking existing rows in disease_constraint_atoms ===")
    conn = sqlite3.connect(str(db_path))
    existing = check_existing(conn, eff_ids)
    conn.close()

    if not existing:
        print("[INFO] No rows currently in disease_constraint_atoms for these trial_ids.")
        print("       After running trial_disease_to_sqlite.py, you SHOULD expect to see rows like the preview above.")
        return 0

    print("Found rows:")
    for trial_id, disease, stem, stem_var, conceptId in existing:
        print(f"  trial_id  : {trial_id}")
        print(f"  disease   : {disease}")
        print(f"  stem      : {stem}")
        print(f"  stem_var  : {stem_var}")
        print(f"  conceptId : {conceptId}")
        print("  ---")

    # Optional: quick diff — which preview stems are missing in DB for each effective_id
    preview_keys = {(tid_eff, r[4]) for tid_eff in eff_ids for r in rows}  # (trial_id, stem)
    existing_keys = {(r[0], r[2]) for r in existing}

    missing = sorted(preview_keys - existing_keys)
    if missing:
        print("\n[WARN] Stems that WOULD be ingested but are NOT currently present in DB:")
        for tid_eff, stem in missing:
            print(f"  trial_id={tid_eff} stem={stem}")
    else:
        print("\n[OK] Every previewed (trial_id, stem) appears to already exist in disease_constraint_atoms.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
