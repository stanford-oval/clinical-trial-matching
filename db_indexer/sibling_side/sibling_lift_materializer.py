#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibling_lift_materializer.py

After ingesting sibling alternatives into concept_accepted_alternatives,
use this script to:

  1. Materialize constraint_lifted_atoms and constraint_literal_alternatives
     using ontology_lifter_clause.build_lifted_for_trials.
  2. Keep ONLY sibling rows in constraint_literal_alternatives whose base_var
     corresponds to a base_variable present in the siblings aggregate folder
     (matching up to timeframe via base_var_stem).

Effectively replaces the "After running this, call ontology_lifter.py --lift"
step with an in-module call, plus the base_var-based filter you asked for.

CLI:
  python sibling_lift_materializer.py \
      --db ../../build/trial.db \
      --siblings-root ../../build/siblings

Optional:
  --trial-id 123              # limit lifting + filtering to a specific trial_constraint_sides.id
  --skip-lift                 # only run the base_var filter on existing data
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Optional, Set, Tuple

# Reuse lifting logic + timeframe stripper from clause ontology lifter
from ontology_lifter_clause import ensure_schema, build_lifted_for_trials, _strip_timeframe_once


# ────────────────────────────────────────────────────────────────
# Helpers: collect base_variables + stems from siblings aggregates
# ────────────────────────────────────────────────────────────────


def collect_sibling_base_vars_and_stems(siblings_root: str) -> Tuple[Set[str], Set[str]]:
    """
    Recursively walk siblings_root and read every 'aggregate.json' file.

    We look for either:
      - item["base_variable_name"], or
      - item["variable_name"]

    and treat those as base variables.

    Returns:
      (base_vars, base_stems)
    where base_stems are computed with _strip_timeframe_once, so that
    base_var_stem in constraint_literal_alternatives can be matched robustly.
    """
    root = Path(siblings_root)
    if not root.exists():
        raise SystemExit(f"[ERROR] siblings_root not found: {root}")

    base_vars: Set[str] = set()
    base_stems: Set[str] = set()

    for agg_path in root.rglob("aggregate.json"):
        try:
            js = json.loads(agg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {agg_path}: {e}")
            continue

        items = js.get("items") if isinstance(js, dict) else None
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue
            # be flexible on key name: both older and newer aggregate formats
            v = it.get("base_variable_name") or it.get("base_variable") or it.get("variable_name")
            if isinstance(v, str):
                v_clean = v.strip()
                if not v_clean:
                    continue
                base_vars.add(v_clean)
                base_stems.add(_strip_timeframe_once(v_clean))

    print(f"[INFO] Collected {len(base_vars)} base variables from siblings aggregates.")
    print(f"[INFO] Collected {len(base_stems)} base variable stems from siblings aggregates.")
    return base_vars, base_stems


# ────────────────────────────────────────────────────────────────
# Core: run ontology lift, then filter sibling rows by base_var_stem
# ────────────────────────────────────────────────────────────────


def lift_from_siblings(
    db_path: str,
    siblings_root: str,
    trial_id: Optional[int] = None,
    do_lift: bool = True,
) -> None:
    """
    1) Ensure schema in DB.
    2) Optionally run ontology_lifter_clause.build_lifted_for_trials (full lift).
    3) Filter constraint_literal_alternatives so that:
         - reason = 'sibling'
         - base_var_stem is one of the base_variable stems from siblings_root
       (matching base_var to base_variable up to timeframe).

    If trial_id is provided (trial_constraint_sides.id, int), both lifting and filtering
    are limited to that trial_constraint_sides.id. Otherwise, all trials are processed.
    """
    sibling_base_vars, sibling_base_stems = collect_sibling_base_vars_and_stems(siblings_root)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        ensure_schema(conn)

        if do_lift:
            print("[INFO] Running ontology lifting (build_lifted_for_trials)...")
            inserted_self = build_lifted_for_trials(
                conn,
                trial_id=trial_id,
                include_assumed=True,
                include_assumed_ancestors_for_relevance=True,
            )
            print(f"[INFO] constraint_lifted_atoms (self) inserted: {inserted_self}")

        # Create a TEMP table with allowed base_var_stems from siblings
        cur.execute("DROP TABLE IF EXISTS tmp_sibling_base_stems")
        cur.execute(
            "CREATE TEMP TABLE tmp_sibling_base_stems (stem TEXT PRIMARY KEY)"
        )
        cur.executemany(
            "INSERT OR IGNORE INTO tmp_sibling_base_stems(stem) VALUES (?)",
            ((s,) for s in sibling_base_stems),
        )

        # Count existing sibling rows (optionally per trial_id)
        if trial_id is not None:
            before = cur.execute(
                """
                SELECT COUNT(*) FROM constraint_literal_alternatives
                WHERE reason='sibling' AND trial_id=?
                """,
                (trial_id,),
            ).fetchone()[0] or 0
        else:
            before = cur.execute(
                "SELECT COUNT(*) FROM constraint_literal_alternatives WHERE reason='sibling'"
            ).fetchone()[0] or 0

        # Delete sibling rows whose base_var_stem is not in the siblings stem set
        if trial_id is not None:
            cur.execute(
                """
                DELETE FROM constraint_literal_alternatives
                WHERE reason = 'sibling'
                  AND trial_id = ?
                  AND (
                      base_var_stem IS NULL
                      OR base_var_stem NOT IN (SELECT stem FROM tmp_sibling_base_stems)
                  )
                """,
                (trial_id,),
            )
        else:
            cur.execute(
                """
                DELETE FROM constraint_literal_alternatives
                WHERE reason = 'sibling'
                  AND (
                      base_var_stem IS NULL
                      OR base_var_stem NOT IN (SELECT stem FROM tmp_sibling_base_stems)
                  )
                """
            )

        conn.commit()

        # Count sibling rows after filter
        if trial_id is not None:
            after = cur.execute(
                """
                SELECT COUNT(*) FROM constraint_literal_alternatives
                WHERE reason='sibling' AND trial_id=?
                """,
                (trial_id,),
            ).fetchone()[0] or 0
        else:
            after = cur.execute(
                "SELECT COUNT(*) FROM constraint_literal_alternatives WHERE reason='sibling'"
            ).fetchone()[0] or 0

        print(
            f"[INFO] sibling rows in constraint_literal_alternatives: "
            f"{after} (kept) / {before} (before base_var_stem filter)"
        )

    finally:
        conn.close()


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Materialize constraint_literal_alternatives & constraint_lifted_atoms "
            "for sibling alternatives, and keep only those whose base_var matches "
            "a base_variable present in the siblings folder (up to timeframe)."
        )
    )
    ap.add_argument(
        "--db",
        default="../../build/trial.db",
        help="SQLite DB path (e.g., ../../build/trial.db or ../../build/trials.sqlite)",
    )
    ap.add_argument(
        "--siblings-root",
        default="../../build/siblings/",
        help="Root directory containing siblings subfolders with aggregate.json files.",
    )
    ap.add_argument(
        "--trial-id",
        type=int,
        default=None,
        help="Optional trial_constraint_sides.id to restrict lifting & filtering; if omitted, all trials are processed.",
    )
    ap.add_argument(
        "--skip-lift",
        action="store_true",
        help="Do NOT call build_lifted_for_trials; only run the sibling base_var_stem filter on existing data.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    lift_from_siblings(
        db_path=args.db,
        siblings_root=args.siblings_root,
        trial_id=args.trial_id,
        do_lift=not args.skip_lift,
    )


if __name__ == "__main__":
    main()
