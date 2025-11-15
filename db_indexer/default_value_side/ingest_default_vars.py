#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest_default_vars.py
────────────────────────────────────────────────────────────────────────────
Standalone ingester for default variable values.

It scans a directory of JSON files with shape:

{
  "trial_id": "NCT00000369",
  "inc_exc": "inclusion",
  "variable_must_suffice": [
    "patient_has_diagnosis_of_bipolar_i_disorder_now",
    "patient_is_taking_lithium_carbonate_containing_product_now"
  ]
}

For each variable, we:
  • derive base_var and timeframe token (now / inthehistory / inthepastXd / inthefuture…)
  • compute timeframe window in hours (tf_lb_hours, tf_ub_hours)
  • upsert into predicate_catalog
  • insert/replace a row into default_predicate_values with value=0 (false)

If inc_exc is missing, we try to infer from filename; default fallback is inclusion.

CLI:
  python ingest_default_vars.py \
      --defaults-dir /path/to/default_vars \
      --db /path/to/trial.db

To DROP and recreate ONLY default_predicate_values (DELETES DATA IN THAT TABLE ONLY):
  python ingest_default_vars.py \
      --defaults-dir /path/to/default_vars \
      --db /path/to/trial.db \
      --recreate
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

# ============================== Filename helpers ==============================

# Accept NCT ids with optional trailing suffix letters/numbers/_/-
_FILE_RE = re.compile(r"^(NCT\d+[A-Za-z0-9_-]*)_(inclusion|exclusion)_", re.I)


def _trial_meta_from_filename(fname: str) -> Tuple[str, str]:
    m = _FILE_RE.match(fname)
    if not m:
        return ("", "inclusion")
    return (m.group(1), m.group(2).lower())


# ============================== Timeframe parsing ==============================

_TIMEFRAME_TOKEN_PAT = (
    r"(now|inthehistory|inthepast\d+(?:minutes|hours|days|weeks|months|years)"
    r"|inthefuture\d*(?:minutes|hours|days|weeks|months|years)?|inthefuture)"
)
_TF_FINDER_RE = re.compile(r"_" + _TIMEFRAME_TOKEN_PAT + r"(?:_|$)")

_UNIT_HOURS = {
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 24.0 * 7.0,
    "months": 24.0 * 30.0,
    "years": 24.0 * 365.0,
}


def _split_base_timeframe(var_name: str) -> tuple[str, Optional[str]]:
    """
    Split the last timeframe token (if any) out of the var name.
    Returns (base_var, timeframe_token_or_None).

    Example:
      patient_has_x_inthepast7days  -> base=patient_has_x, tf=inthepast7days
      patient_has_x_now             -> base=patient_has_x, tf=now
      patient_has_x                 -> base=patient_has_x, tf=None
    """
    last: Optional[re.Match[str]] = None
    for m in _TF_FINDER_RE.finditer(var_name):
        last = m
    if not last:
        return var_name, None

    tf = last.group(1)
    matched = last.group(0)

    # Remove the matched substring and stitch underscores nicely if needed.
    if matched.endswith("_"):
        base_var = var_name[: last.start()] + "_" + var_name[last.end() :]
    else:
        base_var = var_name[: last.start()] + var_name[last.end() :]
    return base_var, tf


def _tf_window_hours(token: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """
    Map timeframe token → (lb_hours, ub_hours) relative to now.
      now → (0, 0)
      inthehistory → (-∞, 0)
      inthepast7days → (-7*24, 0)
      inthefuture30days → (0, +30*24)
      inthefuture → (0, +∞)
    """
    if token is None:
        return None, None
    if token == "now":
        return 0.0, 0.0
    if token == "inthehistory":
        return -1.0e9, 0.0  # sentinel for -infinity

    m = re.match(r"inthepast(\d+)(minutes|hours|days|weeks|months|years)", token)
    if m:
        n, u = int(m.group(1)), m.group(2)
        return -(n * _UNIT_HOURS[u]), 0.0

    m = re.match(r"inthefuture(\d+)?(minutes|hours|days|weeks|months|years)?", token)
    if m:
        if m.group(1) and m.group(2):
            n, u = int(m.group(1)), m.group(2)
            return 0.0, (n * _UNIT_HOURS[u])
        return 0.0, 1.0e9  # sentinel for +infinity

    return None, None


# ============================== DB schema ==============================

DDL_DEFAULTS = [
    """
    CREATE TABLE IF NOT EXISTS default_predicate_values (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nct_id      TEXT NOT NULL,
        kind        TEXT NOT NULL CHECK (kind IN ('inclusion','exclusion')),
        var_name    TEXT NOT NULL,
        base_var    TEXT,
        timeframe   TEXT,
        tf_lb_hours REAL,
        tf_ub_hours REAL,
        value       INTEGER NOT NULL CHECK (value IN (0,1)) DEFAULT 0,
        source_file TEXT,
        UNIQUE (nct_id, kind, var_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_defvars_nct_kind ON default_predicate_values(nct_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_defvars_base ON default_predicate_values(base_var)",
]

DDL_VAR_CATALOG = [
    """
    CREATE TABLE IF NOT EXISTS predicate_catalog (
        var_name    TEXT PRIMARY KEY,
        base_var    TEXT,
        timeframe   TEXT,
        tf_lb_hours REAL,
        tf_ub_hours REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_predicate_catalog_base ON predicate_catalog(base_var)",
    "CREATE INDEX IF NOT EXISTS idx_predicate_catalog_tf   ON predicate_catalog(timeframe)",
]

# Minimal trials table so we can seed rows if needed (safe if you already have it)
DDL_TRIALS_MIN = [
    """
    CREATE TABLE IF NOT EXISTS trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nct_id TEXT NOT NULL,
        label TEXT UNIQUE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trials_nct ON trials(nct_id)",
]


def ensure_default_vars_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for stmt in DDL_DEFAULTS:
        cur.execute(stmt)
    conn.commit()


def ensure_predicate_catalog_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for stmt in DDL_VAR_CATALOG:
        cur.execute(stmt)
    conn.commit()


def ensure_trials_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for stmt in DDL_TRIALS_MIN:
        cur.execute(stmt)
    conn.commit()


def ensure_schema(conn: sqlite3.Connection) -> None:
    ensure_default_vars_schema(conn)
    ensure_predicate_catalog_schema(conn)
    ensure_trials_schema(conn)


def recreate_default_predicate_values_table(conn: sqlite3.Connection) -> None:
    """
    Drop and recreate ONLY default_predicate_values.
    WARNING: This deletes data in default_predicate_values only.
    """
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS default_predicate_values;")
    conn.commit()
    ensure_default_vars_schema(conn)


# ============================== Upserts ==============================


def upsert_predicate_catalog(conn: sqlite3.Connection, var_name: str) -> None:
    base_var, tf = _split_base_timeframe(var_name)
    tf_lb_h, tf_ub_h = _tf_window_hours(tf)
    conn.execute(
        """
        INSERT INTO predicate_catalog(var_name, base_var, timeframe, tf_lb_hours, tf_ub_hours)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(var_name) DO UPDATE SET
            base_var=excluded.base_var,
            timeframe=excluded.timeframe,
            tf_lb_hours=excluded.tf_lb_hours,
            tf_ub_hours=excluded.tf_ub_hours
        """,
        (var_name, base_var, tf, tf_lb_h, tf_ub_h),
    )


# ============================== Core ingestion ==============================


def _kind_from_json_inc_exc(x: Optional[str]) -> str:
    k = (x or "").strip().lower()
    if k == "inclusion":
        return "inclusion"
    if k == "exclusion":
        return "exclusion"
    return ""


def ingest_default_vars(
    *,
    defaults_dir: Path,
    db_path: Path,
    glob: str = "*.json",
    recreate: bool = False,
) -> None:
    defaults_dir = Path(defaults_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        # Pragmas suitable for bulk writes
        old_iso = conn.isolation_level
        conn.isolation_level = None
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.isolation_level = old_iso

        # Always ensure supporting tables exist
        ensure_predicate_catalog_schema(conn)
        ensure_trials_schema(conn)

        if recreate:
            recreate_default_predicate_values_table(conn)
        else:
            ensure_default_vars_schema(conn)

        files = sorted(defaults_dir.rglob(glob))
        if not files:
            print(f"[info] No default-var JSON files in {defaults_dir}")
            return

        total_rows = 0
        for jf in files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[warn] Failed to parse {jf}: {e}")
                continue

            nct_id = (data.get("trial_id") or "").strip()
            kind = _kind_from_json_inc_exc(data.get("inc_exc"))
            vars_list = data.get("variable_must_suffice") or []

            # Fallbacks from filename
            if not nct_id:
                fn_nct, fn_kind = _trial_meta_from_filename(jf.name)
                nct_id = fn_nct or nct_id
                if not kind and fn_kind:
                    kind = fn_kind

            if not kind:
                kind = "inclusion"  # final fallback

            if not nct_id:
                print(f"[warn] Skip {jf.name}: missing trial_id")
                continue

            # Seed a minimal trials row (harmless if already present)
            conn.execute(
                """
                INSERT INTO trials (nct_id, label)
                VALUES (?, ?)
                ON CONFLICT(label) DO NOTHING
                """,
                (nct_id, f"{nct_id}_merged"),
            )

            rows = []
            for var_name in vars_list:
                if not isinstance(var_name, str) or not var_name:
                    continue
                base_var, tf = _split_base_timeframe(var_name)
                tf_lb_h, tf_ub_h = _tf_window_hours(tf)
                upsert_predicate_catalog(conn, var_name)
                rows.append(
                    (
                        nct_id,
                        kind,
                        var_name,
                        base_var,
                        tf,
                        tf_lb_h,
                        tf_ub_h,
                        0,
                        str(jf.name),
                    )
                )

            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO default_predicate_values
                      (nct_id, kind, var_name, base_var, timeframe, tf_lb_hours, tf_ub_hours, value, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                total_rows += len(rows)
                print(f"[ok] {jf.name}: {len(rows)} vars")

        print(f"[ok] Ingestion complete. Inserted/updated {total_rows} rows.")
    finally:
        conn.close()


# ============================== CLI ==============================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest default variable JSONs into SQLite with timeframe/basevar decoding."
    )
    ap.add_argument(
        "--defaults-dir",
        type=Path,
        required=True,
        help="Directory containing default var JSON files",
    )
    ap.add_argument(
        "--db",
        "--db-path",
        dest="db_path",
        type=Path,
        required=True,
        help="SQLite DB path to create/update",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.json",
        help="Filename glob (default: *.json); searched recursively",
    )
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="DROP and recreate ONLY default_predicate_values before ingesting (DELETES DATA in that table only).",
    )
    args = ap.parse_args()

    ingest_default_vars(
        defaults_dir=args.defaults_dir,
        db_path=args.db_path,
        glob=args.glob,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()