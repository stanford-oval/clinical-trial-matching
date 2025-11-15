#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibling_positive_lift_materializer.py

After ingesting sibling alternatives into concept_accepted_alternatives,
use this script to:

  1. Materialize positive_constraint_alternatives
     using ontology_lifter.build_lifted_for_positive_constraint_literals.
  2. Keep ONLY sibling rows in positive_constraint_alternatives whose
     base_var corresponds to a base_variable present in the siblings
     aggregate folder (matching up to timeframe via base_var_stem).

This is the positive-literals analogue of sibling_lift_materializer.py
for the clause table.

CLI:
  python sibling_positive_lift_materializer.py \
      --db ../../build/trial.db \
      --siblings-root ../../build/siblings

Optional:
  --skip-lift   # only run the sibling filter on existing data
  --nct-id NCT00000369   # (optional) restrict filter to a single NCT id
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Optional, Set, Tuple

# Reuse lifting logic + timeframe stripper from positive-literal ontology_lifter
from ontology_lifter import ensure_schema, build_lifted_for_positive_constraint_literals, _strip_timeframe_once


# ────────────────────────────────────────────────────────────────
# Helpers: collect base_variables + stems from siblings aggregates
# ────────────────────────────────────────────────────────────────

def collect_sibling_base_vars_and_stems(siblings_root: str) -> Tuple[Set[str], Set[str]]:
    """
    Recursively walk siblings_root and read every 'aggregate.json' file.

    We look for any of:
      - item["base_variable_name"]
      - item["base_variable"]
      - item["variable_name"]

    and treat those as base variables.

    Returns:
      (base_vars, base_stems)
    where base_stems are computed with _strip_timeframe_once, so that
    base_var_stem in positive_constraint_alternatives can be matched
    robustly (up to timeframe).
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
            v = (
                it.get("base_variable_name")
                or it.get("base_variable")
                or it.get("variable_name")
            )
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
# Core: run positive-literal lift, then filter sibling rows
# ────────────────────────────────────────────────────────────────

def lift_positive_from_siblings(
    db_path: str,
    siblings_root: str,
    *,
    do_lift: bool = True,
    nct_id: Optional[str] = None,
) -> None:
    """
    1) Ensure schema in DB.
    2) Optionally run build_lifted_for_positive_constraint_literals().
    3) Filter positive_constraint_alternatives so that:
         - reason = 'sibling'
         - base_var_stem is one of the base_variable stems from siblings_root
       (matching base_var to base_variable up to timeframe).

    If nct_id is provided, both the "before" / "after" counts and the DELETE
    are restricted to that NCT id; otherwise the filter applies globally.
    """
    sibling_base_vars, sibling_base_stems = collect_sibling_base_vars_and_stems(siblings_root)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        ensure_schema(conn)

        if do_lift:
            print("[INFO] Running positive-literal ontology lifting (build_lifted_for_positive_constraint_literals)...")
            self_count = build_lifted_for_positive_constraint_literals(
                conn,
                include_assumed=True,
                include_assumed_ancestors_for_relevance=True,
            )
            print(f"[INFO] positive_constraint_alternatives (self) inserted: {self_count}")

        # TEMP table of allowed base_var_stems from siblings
        cur.execute("DROP TABLE IF EXISTS tmp_sibling_pos_base_stems")
        cur.execute("CREATE TEMP TABLE tmp_sibling_pos_base_stems (stem TEXT PRIMARY KEY)")
        cur.executemany(
            "INSERT OR IGNORE INTO tmp_sibling_pos_base_stems(stem) VALUES (?)",
            ((s,) for s in sibling_base_stems),
        )

        # Count sibling rows before filter
        if nct_id is not None:
            before = cur.execute(
                """
                SELECT COUNT(*)
                FROM positive_constraint_alternatives
                WHERE reason='sibling' AND nct_id=?
                """,
                (nct_id,),
            ).fetchone()[0] or 0
        else:
            before = cur.execute(
                "SELECT COUNT(*) FROM positive_constraint_alternatives WHERE reason='sibling'"
            ).fetchone()[0] or 0

        # Delete sibling rows whose base_var_stem is not in the siblings stem set
        if nct_id is not None:
            cur.execute(
                """
                DELETE FROM positive_constraint_alternatives
                WHERE reason = 'sibling'
                  AND nct_id = ?
                  AND (
                      base_var_stem IS NULL
                      OR base_var_stem NOT IN (SELECT stem FROM tmp_sibling_pos_base_stems)
                  )
                """,
                (nct_id,),
            )
        else:
            cur.execute(
                """
                DELETE FROM positive_constraint_alternatives
                WHERE reason = 'sibling'
                  AND (
                      base_var_stem IS NULL
                      OR base_var_stem NOT IN (SELECT stem FROM tmp_sibling_pos_base_stems)
                  )
                """
            )

        conn.commit()

        # Count sibling rows after filter
        if nct_id is not None:
            after = cur.execute(
                """
                SELECT COUNT(*)
                FROM positive_constraint_alternatives
                WHERE reason='sibling' AND nct_id=?
                """,
                (nct_id,),
            ).fetchone()[0] or 0
        else:
            after = cur.execute(
                "SELECT COUNT(*) FROM positive_constraint_alternatives WHERE reason='sibling'"
            ).fetchone()[0] or 0

        print(
            f"[INFO] sibling rows in positive_constraint_alternatives: "
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
            "Positive-literal sibling ontology lifting: materialize "
            "positive_constraint_alternatives via ontology_lifter, then "
            "keep only sibling rows whose base_var matches a base_variable "
            "present in the siblings folder (up to timeframe)."
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
        "--skip-lift",
        action="store_true",
        help="Do NOT call build_lifted_for_positive_constraint_literals; only run the sibling base_var_stem filter.",
    )
    ap.add_argument(
        "--nct-id",
        default=None,
        help="Optional NCT id (e.g., NCT00000369) to restrict filtering; if omitted, applies globally.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    lift_positive_from_siblings(
        db_path=args.db,
        siblings_root=args.siblings_root,
        do_lift=not args.skip_lift,
        nct_id=args.nct_id,
    )


if __name__ == "__main__":
    main()
