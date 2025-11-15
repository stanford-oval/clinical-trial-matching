#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_sibling_alternatives.py

Ingest SiblingOverlapPipeline aggregates (after postprocessing) into the
SQLite DB so ontology_lifter can treat them as ontology lifting alternatives.

For each aggregate item:

  base_concept_id = item["concept_id"]

  for each yes_example:
      alt_concept_id  = yes_example["conceptId"]
      alt_label       = yes_example["candidate_entity_canonical_form"]
                        (fallbacks to candidate_concept / alt_label / preferred_term)
      hop             = 0
      reason          = --reason (default: "sibling")

Rows are written into concept_accepted_alternatives:

  CREATE TABLE IF NOT EXISTS concept_accepted_alternatives (
    concept_id     TEXT NOT NULL,
    alt_concept_id TEXT NOT NULL,
    hop            INTEGER NOT NULL,
    alt_label      TEXT,
    reason         TEXT,
    decided_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (concept_id, alt_concept_id)
  );

After running this, call ontology_lifter.py --lift to materialize
constraint_literal_alternatives and constraint_lifted_atoms.

Usage example (from repo root):

  python siblingsrc/ingest_sibling_alternatives.py \
      --db ../../build/trial.db \
      --agg-root ../../build/siblings \
      --reason sibling \
      --purge-existing-reason sibling
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("sibling_ingest")

# ───────────────────────────── logging ─────────────────────────────

def _configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ───────────────────────────── DB helpers ─────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS concept_accepted_alternatives (
  concept_id     TEXT NOT NULL,
  alt_concept_id TEXT NOT NULL,
  hop            INTEGER NOT NULL,
  alt_label      TEXT,
  reason         TEXT,
  decided_at     TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (concept_id, alt_concept_id)
);
CREATE INDEX IF NOT EXISTS idx_caa_concept_hop
  ON concept_accepted_alternatives(concept_id, hop);
"""

def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(DDL)
    return conn

# ───────────────────────────── JSON helpers ─────────────────────────────

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        LOGGER.error("Failed to load JSON %s: %s", path, e)
        return None

# ───────────────────────────── core ingestion ─────────────────────────────

def _iter_aggregate_files(agg_root: Path) -> List[Path]:
    """
    Find all <trial_id>/aggregate.json under agg_root.
    """
    if not agg_root.exists() or not agg_root.is_dir():
        LOGGER.error("agg-root does not exist or is not a directory: %s", agg_root)
        return []

    out: List[Path] = []
    for td in sorted(agg_root.iterdir()):
        if not td.is_dir():
            continue
        agg = td / "aggregate.json"
        if agg.exists():
            out.append(agg)
    return out


def _extract_pairs_from_item(
    item: Dict[str, Any],
    reason: str,
) -> List[Tuple[str, str, int, Optional[str], str]]:
    """
    From a single aggregate item, produce rows:

      (concept_id, alt_concept_id, hop, alt_label, reason)

    hop is always 0 for siblings.

    NOTE: alt_label is set to candidate_entity_canonical_form when available,
    so that ontology_lifter._compose_from_label will reconstruct the same
    entity_canonical_form and thus the same candidate_variable_name.
    """
    base_cid = str(item.get("concept_id") or "").strip()
    if not base_cid:
        return []

    yes = item.get("yes_examples") or []
    rows: List[Tuple[str, str, int, Optional[str], str]] = []

    for ex in yes:
        if not isinstance(ex, dict):
            continue

        alt_cid = (
            ex.get("conceptId")
            or ex.get("alt_concept_id")
            or ex.get("candidate_concept_id")
        )
        alt_cid = str(alt_cid or "").strip()
        if not alt_cid:
            continue

        # Prefer the canonical snake_case form, fall back to labely things.
        alt_label = (
            ex.get("candidate_entity_canonical_form")
            or ex.get("candidate_concept")
            or ex.get("alt_label")
            or ex.get("preferred_term")
            or ex.get("concept_name")
        )
        alt_label = str(alt_label).strip() if alt_label is not None else None

        rows.append((base_cid, alt_cid, 0, alt_label, reason))

    return rows


def _ingest_aggregate_file(
    conn: sqlite3.Connection,
    agg_path: Path,
    reason: str,
    *,
    dry_run: bool = False,
) -> int:
    """
    Ingest a single aggregate.json file.

    Returns number of (base, alt) pairs processed.
    """
    obj = _load_json(agg_path)
    if not isinstance(obj, dict):
        LOGGER.warning("Skipping %s (not a JSON object)", agg_path)
        return 0

    items = obj.get("items")
    if not isinstance(items, list):
        LOGGER.warning("Skipping %s (no 'items' list)", agg_path)
        return 0

    pairs: List[Tuple[str, str, int, Optional[str], str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        pairs.extend(_extract_pairs_from_item(it, reason))

    if not pairs:
        LOGGER.info("No sibling pairs found in %s", agg_path)
        return 0

    LOGGER.info(
        "Ingesting %d sibling alternatives from %s", len(pairs), agg_path
    )

    if dry_run:
        for base_cid, alt_cid, hop, alt_label, why in pairs[:10]:
            LOGGER.debug(
                "[dryrun] %s -> %s (hop=%d, label=%r, reason=%s)",
                base_cid, alt_cid, hop, alt_label, why,
            )
        LOGGER.info("[dryrun] (no DB writes performed)")
        return len(pairs)

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO concept_accepted_alternatives
          (concept_id, alt_concept_id, hop, alt_label, reason)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
          hop       = excluded.hop,
          alt_label = COALESCE(excluded.alt_label,
                               concept_accepted_alternatives.alt_label),
          reason    = excluded.reason,
          decided_at = datetime('now')
        """,
        pairs,
    )
    conn.commit()
    return len(pairs)


def ingest_siblings(
    db_path: str,
    agg_root: str,
    *,
    reason: str = "sibling",
    purge_existing_reason: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    conn = _open_db(db_path)
    try:
        if purge_existing_reason:
            LOGGER.info(
                "Purging existing concept_accepted_alternatives rows with reason=%r",
                purge_existing_reason,
            )
            with conn:
                conn.execute(
                    "DELETE FROM concept_accepted_alternatives WHERE reason = ?",
                    (purge_existing_reason,),
                )

        agg_root_path = Path(agg_root)
        files = _iter_aggregate_files(agg_root_path)
        if not files:
            LOGGER.warning("No aggregate.json files found under %s", agg_root_path)
            return

        total_pairs = 0
        for agg_path in files:
            n = _ingest_aggregate_file(conn, agg_path, reason, dry_run=dry_run)
            total_pairs += n

        LOGGER.info("Done. Processed %d sibling (base, alt) pairs.", total_pairs)
    finally:
        conn.close()

# ───────────────────────────── CLI ─────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Ingest SiblingOverlapPipeline aggregates into concept_accepted_alternatives "
            "as hop=0 'sibling' alternatives."
        )
    )
    ap.add_argument(
        "--db",
        default="../../build/trial.db",
        help="SQLite DB path (e.g., ../../build/trial.db)",
    )
    ap.add_argument(
        "--agg-root",
        default="../../build/siblings/",
        help="Root directory of per-trial aggregates (e.g., ../../build/siblings)",
    )
    ap.add_argument(
        "--reason",
        default="sibling",
        help="Reason string to store for sibling alternatives (default: 'sibling')",
    )
    ap.add_argument(
        "--purge-existing-reason",
        default=None,
        help=(
            "If set, delete existing rows in concept_accepted_alternatives where "
            "reason equals this string before ingesting. Useful to refresh "
            "previous sibling ingests (e.g., 'sibling')."
        ),
    )
    ap.add_argument(
        "--dryrun",
        action="store_true",
        help="Parse aggregates and log, but do not write to the DB.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = ap.parse_args()
    _configure_logging(args.log_level)

    ingest_siblings(
        db_path=args.db,
        agg_root=args.agg_root,
        reason=args.reason,
        purge_existing_reason=args.purge_existing_reason,
        dry_run=args.dryrun,
    )

if __name__ == "__main__":
    main()
