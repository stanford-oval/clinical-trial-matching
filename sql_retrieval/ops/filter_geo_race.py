#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trial_geo_race_filter.py
───────────────────────────────────────────────────────────────────────────────
Filter trials by a patient's geography & race against trial-side *allowed* lists.

Policy
- Start from ALL trials by default (SELECT id FROM trials).
- A trial is ELIMINATED iff, for ANY dimension ∈ {birth_country, resident_country, race}:
    * the patient has ≥1 value for that dimension, AND
    * the trial lists ≥1 allowed value for that dimension, AND
    * NONE of the patient's values match any allowed value for the trial.
- If the patient lacks a value for a dimension, that dimension is ignored.
- If a trial has no allowed list for a dimension, that dimension is ignored.
- Race matches by numeric race_id when available, OR by case-insensitive race_name.

DB expectations (created by your ingesters):
  patient_birth_country(patient_id TEXT, country_index INTEGER, ...)
  patient_resident_country(patient_id TEXT, country_index INTEGER, ...)
  patient_race(patient_id TEXT, race_id INTEGER, race_name TEXT, ...)
  trial_birth_country_allowed(trial_id TEXT, country_index INTEGER, ...)
  trial_resident_country_allowed(trial_id TEXT, country_index INTEGER, ...)
  trial_race_allowed(trial_id TEXT, race_id INTEGER, race_name TEXT, ...)
  trials(id INTEGER PRIMARY KEY, nct_id TEXT, ...)

Usage examples
  # Start from ALL trials:
  python trial_geo_race_filter.py --db build/trial.db --patient sigir-20141

  # Start from specific trials (comma list of trials.id):
  python trial_geo_race_filter.py --db build/trial.db --patient sigir-20141 --trials 1,2,3

  # Start from @file (one trials.id per line):
  python trial_geo_race_filter.py --db build/trial.db --patient sigir-20141 --trials @/path/to/ids.txt

  # Show elimination reasons per trial:
  python trial_geo_race_filter.py --db build/trial.db --patient sigir-20141 --explain
"""

from __future__ import annotations
import argparse, csv, json, sqlite3, sys
from pathlib import Path
from typing import Dict, List, Tuple

def _read_trials_arg(arg: str) -> List[int]:
    if not arg:
        return []
    if arg.startswith("@"):
        p = arg[1:]
        with open(p, "r", encoding="utf-8") as f:
            return [int(x.strip()) for x in f if x.strip()]
    return [int(x) for x in arg.split(",") if x.strip()]

def _ensure_tmp_candidates(conn: sqlite3.Connection, trials_arg: str) -> None:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tmp_candidates")
    cur.execute("CREATE TEMP TABLE tmp_candidates(id INTEGER PRIMARY KEY)")
    ids = _read_trials_arg(trials_arg) if trials_arg else []
    if ids:
        cur.executemany("INSERT INTO tmp_candidates(id) VALUES (?)", [(int(t),) for t in ids])
    else:
        # Start from ALL trials.id
        cur.execute("INSERT INTO tmp_candidates(id) SELECT id FROM trials")
    conn.commit()

def filter_trials_by_geo_race(conn: sqlite3.Connection, patient_id: str) -> Tuple[List[int], List[Tuple[int,str]]]:
    """
    Returns (survivor_trial_ids, eliminated_with_reason_rows)
    eliminated_with_reason_rows: list of (trial_id, reason) where reason ∈ {'birth_country','resident_country','race'}
    """
    sql = r"""
    WITH itc AS (  -- candidate trials with nct_id
      SELECT t.id AS trial_id, t.nct_id
      FROM trials t
      JOIN tmp_candidates c ON c.id = t.id
    ),

    -- Violations if patient has data AND trial has allowed list AND no overlap
    viol_birth AS (
      SELECT i.trial_id
      FROM itc i
      WHERE EXISTS (SELECT 1 FROM patient_birth_country p WHERE p.patient_id = :patient)
        AND EXISTS (SELECT 1 FROM trial_birth_country_allowed a WHERE a.trial_id = i.nct_id)
        AND NOT EXISTS (
          SELECT 1
          FROM patient_birth_country p
          JOIN trial_birth_country_allowed a
                ON a.trial_id = i.nct_id
               AND a.country_index = p.country_index
          WHERE p.patient_id = :patient
        )
    ),

    viol_resident AS (
      SELECT i.trial_id
      FROM itc i
      WHERE EXISTS (SELECT 1 FROM patient_resident_country p WHERE p.patient_id = :patient)
        AND EXISTS (SELECT 1 FROM trial_resident_country_allowed a WHERE a.trial_id = i.nct_id)
        AND NOT EXISTS (
          SELECT 1
          FROM patient_resident_country p
          JOIN trial_resident_country_allowed a
                ON a.trial_id = i.nct_id
               AND a.country_index = p.country_index
          WHERE p.patient_id = :patient
        )
    ),

    -- Race match predicate: by id when present, OR by case-insensitive name
    viol_race AS (
      SELECT i.trial_id
      FROM itc i
      WHERE EXISTS (SELECT 1 FROM patient_race pr WHERE pr.patient_id = :patient)
        AND EXISTS (SELECT 1 FROM trial_race_allowed tr WHERE tr.trial_id = i.nct_id)
        AND NOT EXISTS (
          SELECT 1
          FROM patient_race pr
          JOIN trial_race_allowed tr
            ON tr.trial_id = i.nct_id
           AND (
                 (pr.race_id IS NOT NULL AND tr.race_id IS NOT NULL AND tr.race_id = pr.race_id)
                 OR (
                      pr.race_name IS NOT NULL AND tr.race_name IS NOT NULL
                      AND UPPER(TRIM(pr.race_name)) = UPPER(TRIM(tr.race_name))
                    )
               )
          WHERE pr.patient_id = :patient
        )
    ),

    elim AS (
      SELECT trial_id, 'birth_country'    AS reason FROM viol_birth
      UNION ALL
      SELECT trial_id, 'resident_country' AS reason FROM viol_resident
      UNION ALL
      SELECT trial_id, 'race'             AS reason FROM viol_race
    )

    SELECT i.trial_id,
           COALESCE(e.reason, '') AS reason
    FROM itc i
    LEFT JOIN elim e ON e.trial_id = i.trial_id
    ORDER BY i.trial_id, e.reason
    """
    cur = conn.cursor()
    rows = cur.execute(sql, {"patient": patient_id}).fetchall()

    survivors: List[int] = []
    eliminated: List[Tuple[int,str]] = []

    # Rows come as (trial_id, reason-or-empty)
    current_id = None
    has_violation = False
    reasons_for_current: List[str] = []

    # We'll walk grouped-by trial_id
    for trial_id, reason in rows:
        if current_id is None:
            current_id = trial_id
            has_violation = False
            reasons_for_current = []

        if trial_id != current_id:
            if has_violation:
                for r in reasons_for_current:
                    eliminated.append((current_id, r))
            else:
                survivors.append(current_id)
            # reset for new id
            current_id = trial_id
            has_violation = False
            reasons_for_current = []

        if reason:
            has_violation = True
            reasons_for_current.append(reason)

    # Flush last
    if current_id is not None:
        if has_violation:
            for r in reasons_for_current:
                eliminated.append((current_id, r))
        else:
            survivors.append(current_id)

    return survivors, eliminated

def main():
    ap = argparse.ArgumentParser(description="Filter trials by patient geography & race against trial-side allowed lists.")
    ap.add_argument("--db", required=True, help="Path to SQLite DB (e.g., build/trial.db)")
    ap.add_argument("--patient", required=True, help="Patient id (matches patient_* tables)")
    ap.add_argument("--trials", default="", help="Optional comma list of trials.id or @file path. Default: ALL trials.")
    ap.add_argument("--out", default="", help="Optional output directory to write survivors.json/csv and eliminated.csv")
    ap.add_argument("--quiet", action="store_true", help="If set, print survivors trial_ids only (one per line)")
    ap.add_argument("--explain", action="store_true", help="If set, also print/show elimination reasons per trial")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"[error] DB not found: {db_path}", file=sys.stderr)
        sys.exit(2)

    try:
        conn = sqlite3.connect(str(db_path))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    _ensure_tmp_candidates(conn, args.trials)
    survivors, eliminated = filter_trials_by_geo_race(conn, args.patient)

    if args.quiet and not args.explain and not args.out:
        for tid in survivors:
            print(tid)
        return

    payload = {
        "patient_id": args.patient,
        "n_candidates": (len(survivors) + len({tid for tid,_ in eliminated})),
        "n_survivors": len(survivors),
        "survivor_trial_ids": survivors,
    }

    if args.explain:
        # Group eliminated reasons per trial
        elim_map: Dict[int, List[str]] = {}
        for tid, reason in eliminated:
            elim_map.setdefault(tid, []).append(reason)
        # Sort and deduplicate reason lists
        elim_rows = [
            {"trial_id": tid, "reasons": sorted(set(rs))}
            for tid, rs in sorted(elim_map.items(), key=lambda kv: kv[0])
        ]
        payload["eliminated"] = elim_rows

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # survivors.json
        with open(out_dir / "survivors.json", "w", encoding="utf-8") as f:
            json.dump({"trial_ids": survivors}, f, ensure_ascii=False, indent=2)
        # survivors.csv
        with open(out_dir / "survivors.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["trial_id"]); w.writerows([[tid] for tid in survivors])
        # eliminated.csv (when explain)
        if args.explain:
            with open(out_dir / "eliminated.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["trial_id","reason"])
                for tid, reason in eliminated:
                    w.writerow([tid, reason])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)
