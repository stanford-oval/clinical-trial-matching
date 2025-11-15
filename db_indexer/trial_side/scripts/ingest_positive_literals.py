#!/usr/bin/env python3
# scripts/ingest_positive_constraint_literals.py
from __future__ import annotations

import json, re, sqlite3
from pathlib import Path
import argparse

# ────────────────────────────────────────────────────────────────
# Filename parsing
# NOTE: nct_id now includes any trailing cohort suffix, e.g. NCT00000932a
#       so subcohorted trials are NOT collapsed.
_FILE_RE = re.compile(
    r'^(NCT\d{8})([A-Za-z0-9_-]*?)_(inclusion|exclusion)_program(?:\.assumed)?\.smt2$',
    re.I,
)

def _parse_fname(name: str):
    m = _FILE_RE.match(name or "")
    if not m:
        return None, None, "main"
    nct_base, suffix, kind = m.group(1), (m.group(2) or ""), m.group(3).lower()
    nct = nct_base + suffix  # full id, e.g. NCT00000932a
    variant = (
        "assumed"
        if ".assumed." in name or name.endswith(".assumed.smt2")
        else "main"
    )
    return nct, kind, variant

# ────────────────────────────────────────────────────────────────
# Timeframe parsing (match smt_clause_db_build.py)
_TIMEFRAME_TOKEN_PAT = (
    r'(now|inthehistory|'
    r'inthepast\d+(?:minutes|hours|days|weeks|months|years)|'
    r'inthefuture(\d+)?(?:minutes|hours|days|weeks|months|years)?|'
    r'inthefuture)'
)
_TF_FINDER_RE = re.compile(r"_" + _TIMEFRAME_TOKEN_PAT + r"(?:_|$)")

_UNIT_HOURS = {
    "minutes": 1 / 60,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 168.0,
    "months": 720.0,
    "years": 8760.0,
}


def _split_base_timeframe(var_name: str) -> tuple[str, str | None]:
    last = None
    for m in _TF_FINDER_RE.finditer(var_name or ""):
        last = m
    if not last:
        return var_name, None
    tf = last.group(1)
    matched = last.group(0)
    if matched.endswith("_"):
        base_var = var_name[: last.start()] + "_" + var_name[last.end() :]
    else:
        base_var = var_name[: last.start()] + var_name[last.end() :]
    base_var = re.sub(r"_+", "_", base_var).strip("_")
    return base_var, tf


def _tf_window_hours(token: str | None) -> tuple[float | None, float | None]:
    if token is None:
        return None, None
    if token == "now":
        return 0.0, 0.0
    if token == "inthehistory":
        return -1.0e9, 0.0
    m = re.match(r"inthepast(\d+)(minutes|hours|days|weeks|months|years)", token)
    if m:
        n, u = int(m.group(1)), m.group(2)
        return -(n * _UNIT_HOURS[u]), 0.0
    m = re.match(r"inthefuture(\d+)?(minutes|hours|days|weeks|months|years)?", token)
    if m:
        if m.group(1) and m.group(2):
            n, u = int(m.group(1)), m.group(2)
            return 0.0, (n * _UNIT_HOURS[u])
        return 0.0, 1.0e9
    return None, None


# ────────────────────────────────────────────────────────────────
# DDL (table only). Indexes are created AFTER columns are ensured.
DDL_TABLE_ONLY = """
CREATE TABLE IF NOT EXISTS positive_constraint_literals (
  nct_id     TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind in ('inclusion','exclusion')),
  variant    TEXT NOT NULL CHECK (variant in ('main','assumed')),
  direction  TEXT NOT NULL CHECK (direction in ('sat_on_true','unsat_on_true')),
  var_name   TEXT NOT NULL,
  smt2_file  TEXT,
  -- new columns may be added later via ALTER TABLE
  PRIMARY KEY (nct_id, kind, variant, direction, var_name)
);
"""


def _maybe_add_column(cur: sqlite3.Cursor, table: str, column: str, decl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if column not in cols:
        # BUGFIX: include the column name in ALTER TABLE
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _ensure_columns_and_indexes(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # add columns if missing
    _maybe_add_column(cur, "positive_constraint_literals", "base_var", "TEXT")
    _maybe_add_column(cur, "positive_constraint_literals", "timeframe", "TEXT")
    _maybe_add_column(cur, "positive_constraint_literals", "tf_lb_hours", "REAL")
    _maybe_add_column(cur, "positive_constraint_literals", "tf_ub_hours", "REAL")
    conn.commit()
    # repair any bad/old indexes referencing typos
    bad_idxs = list(
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' "
            "AND (sql LIKE '%base_va,%' OR sql LIKE '%base_va)')"
        ).fetchall()
    )
    for (idx,) in bad_idxs:
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    # create indexes safely
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_poslit_var  ON positive_constraint_literals(var_name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_poslit_base ON positive_constraint_literals(base_var)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_poslit_tf   ON positive_constraint_literals(timeframe)"
    )
    conn.commit()


# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Ingest per_file positive literal JSONs into SQLite "
            "(UPSERT + timeframe aware)."
        )
    )
    ap.add_argument(
        "--db", required=True, help="SQLite path (e.g., ../../build/trial.db)"
    )
    ap.add_argument(
        "--per-file-dir",
        default="../../build/positive_constraint_literals/per_file",
        help="Directory with *.json from find_positive_canon_literals.py",
    )
    ap.add_argument(
        "--direction",
        choices=["sat_on_true", "unsat_on_true", "any"],
        default="sat_on_true",
        help="Filter by direction (or 'any' to ingest both)",
    )
    args = ap.parse_args()

    per = Path(args.per_file_dir)
    files = sorted(per.glob("*.json"))
    if not files:
        print(f"[warn] no per-file JSONs in {per}")
        return

    db_path = Path(args.db).resolve()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(DDL_TABLE_ONLY)  # table only, no fragile indexes yet
        _ensure_columns_and_indexes(conn)  # add columns, then safe indexes

        cur = conn.cursor()
        upserts = 0

        for f in files:
            js = json.loads(f.read_text(encoding="utf-8"))
            js_dir = (js.get("direction") or "").strip()
            if args.direction != "any" and js_dir != args.direction:
                continue

            nct, kind, variant = _parse_fname(js.get("smt2_file", ""))
            if not nct or not kind:
                continue

            for v in (js.get("hits") or []):
                if not isinstance(v, str):
                    continue
                var_name = v.strip()
                if not var_name:
                    continue

                base_var, tf = _split_base_timeframe(var_name)
                lb_h, ub_h = _tf_window_hours(tf)

                cur.execute(
                    """
                  INSERT INTO positive_constraint_literals
                    (nct_id, kind, variant, direction, var_name, smt2_file,
                     base_var, timeframe, tf_lb_hours, tf_ub_hours)
                  VALUES (?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(nct_id,kind,variant,direction,var_name) DO UPDATE SET
                    smt2_file   = excluded.smt2_file,
                    base_var    = COALESCE(excluded.base_var,    positive_constraint_literals.base_var),
                    timeframe   = COALESCE(excluded.timeframe,   positive_constraint_literals.timeframe),
                    tf_lb_hours = COALESCE(excluded.tf_lb_hours, positive_constraint_literals.tf_lb_hours),
                    tf_ub_hours = COALESCE(excluded.tf_ub_hours, positive_constraint_literals.tf_ub_hours)
                """,
                    (
                        nct,
                        kind,
                        variant,
                        js_dir,
                        var_name,
                        js.get("smt2_file", ""),
                        base_var,
                        tf,
                        lb_h,
                        ub_h,
                    ),
                )
                upserts += cur.rowcount or 0

        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(FULL);")
        except Exception:
            pass

        print(f"[db] {db_path}")
        print(
            f"[ok] upserted {upserts} rows into positive_constraint_literals "
            f"(with base/timeframe/intervals)"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()