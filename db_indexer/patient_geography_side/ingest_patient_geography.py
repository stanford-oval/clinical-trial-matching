#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest patient geographic & race data into the existing trial.db.

Expected layout under --geo-root (patient_geographic_build):
  {geo_root}/
    sigir-20141/
      birth_country.csv
      resident_country.csv
      race.csv
    sigir-20142/
      ...
    summary_birth_country.csv        (optional; ignored by default)
    summary_resident_country.csv     (optional; ignored by default)
    summary_race.csv                 (optional; ignored by default)

CSV formats (per patient folder):
  - birth_country.csv:
        "All Country allowed edges:"      # optional banner line to ignore
        note_id,country_index
        sigir-20141,185
    NOTE: May be empty (header only). If present, same format as resident_country.

  - resident_country.csv:
        "All Country allowed edges:"      # optional banner line to ignore
        note_id,country_index
        sigir-20141,185

  - race.csv:
        note_idx,race_id,race_name
        sigir-20141,4,African
    NOTE: The first column header can be 'note_idx' or 'note_id'.

Creates (or upserts into) three tables in the SQLite DB:
    patient_birth_country(patient_id TEXT, country_index INTEGER, source_file TEXT, created_at INTEGER, PRIMARY KEY(patient_id, country_index))
    patient_resident_country(patient_id TEXT, country_index INTEGER, source_file TEXT, created_at INTEGER, PRIMARY KEY(patient_id, country_index))
    patient_race(patient_id TEXT, race_id INTEGER, race_name TEXT, source_file TEXT, created_at INTEGER, PRIMARY KEY(patient_id, race_id))

Usage example:
  python ingest_patient_geography.py \
    --db /path/to/trial.db \
    --geo-root /path/to/patient_geographic_build \
    --recreate  # (optional) drops and recreates ONLY these 3 tables
"""

from __future__ import annotations
import os, sys, csv, argparse, sqlite3
from contextlib import contextmanager
from typing import List, Dict, Optional

# -------------------- DB helpers --------------------
@contextmanager
def db_conn(path: str):
    conn = sqlite3.connect(path, timeout=100, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        yield conn
    finally:
        conn.close()

def drop_geo_tables(conn: sqlite3.Connection):
    conn.executescript("""
    DROP TABLE IF EXISTS patient_birth_country;
    DROP TABLE IF EXISTS patient_resident_country;
    DROP TABLE IF EXISTS patient_race;
    """)

def ensure_geo_schema(conn: sqlite3.Connection, recreate: bool = False):
    if recreate:
        drop_geo_tables(conn)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS patient_birth_country (
      patient_id    TEXT NOT NULL,
      country_index INTEGER NOT NULL,
      source_file   TEXT,
      created_at    INTEGER NOT NULL DEFAULT (strftime('%s','now')),
      PRIMARY KEY (patient_id, country_index)
    );
    CREATE INDEX IF NOT EXISTS idx_birth_country_patient ON patient_birth_country(patient_id);

    CREATE TABLE IF NOT EXISTS patient_resident_country (
      patient_id    TEXT NOT NULL,
      country_index INTEGER NOT NULL,
      source_file   TEXT,
      created_at    INTEGER NOT NULL DEFAULT (strftime('%s','now')),
      PRIMARY KEY (patient_id, country_index)
    );
    CREATE INDEX IF NOT EXISTS idx_resident_country_patient ON patient_resident_country(patient_id);

    CREATE TABLE IF NOT EXISTS patient_race (
      patient_id   TEXT NOT NULL,
      race_id      INTEGER NOT NULL,
      race_name    TEXT,
      source_file  TEXT,
      created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now')),
      PRIMARY KEY (patient_id, race_id)
    );
    CREATE INDEX IF NOT EXISTS idx_race_patient ON patient_race(patient_id);
    """)

# -------------------- Filesystem helpers --------------------
def list_patient_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return sorted(out)

def _safe_int(x: Optional[str]) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None

def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Read a CSV while skipping banner lines like 'All Country allowed edges:' and empty lines.
       Returns list of dict rows using the (first non-banner) header row.
    """
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    # Drop any leading banner/comment lines
    while lines and (lines[0].lower().startswith("all country allowed edges") or lines[0].startswith("#")):
        lines.pop(0)
    if not lines:
        return []
    # Use csv module on the filtered content
    reader = csv.DictReader(lines)
    rows: List[Dict[str, str]] = []
    for r in reader:
        # Normalize keys (strip BOM or whitespace)
        nr = { (k or "").strip().lstrip("\ufeff"): (v or "").strip() for k,v in r.items() }
        rows.append(nr)
    return rows

# -------------------- Ingest logic --------------------
def ingest_birth_country(conn: sqlite3.Connection, geo_root: str) -> int:
    total = 0
    for pid in list_patient_dirs(geo_root):
        fpath = os.path.join(geo_root, pid, "birth_country.csv")
        rows = _read_csv_rows(fpath)
        if not rows:
            continue
        to_ins = []
        for r in rows:
            # Accept 'note_id' or 'note_idx'; prefer directory pid as patient_id
            country_idx = _safe_int(r.get("country_index"))
            if country_idx is None:
                continue
            to_ins.append((pid, country_idx, fpath))
        if to_ins:
            conn.executemany("""
                INSERT INTO patient_birth_country (patient_id, country_index, source_file)
                VALUES (?,?,?)
                ON CONFLICT(patient_id, country_index) DO NOTHING
            """, to_ins)
            total += len(to_ins)
    print(f"[geo] birth_country rows inserted: {total}")
    return total

def ingest_resident_country(conn: sqlite3.Connection, geo_root: str) -> int:
    total = 0
    for pid in list_patient_dirs(geo_root):
        fpath = os.path.join(geo_root, pid, "resident_country.csv")
        rows = _read_csv_rows(fpath)
        if not rows:
            continue
        to_ins = []
        for r in rows:
            country_idx = _safe_int(r.get("country_index"))
            if country_idx is None:
                continue
            to_ins.append((pid, country_idx, fpath))
        if to_ins:
            conn.executemany("""
                INSERT INTO patient_resident_country (patient_id, country_index, source_file)
                VALUES (?,?,?)
                ON CONFLICT(patient_id, country_index) DO NOTHING
            """, to_ins)
            total += len(to_ins)
    print(f"[geo] resident_country rows inserted: {total}")
    return total

def ingest_race(conn: sqlite3.Connection, geo_root: str) -> int:
    total = 0
    for pid in list_patient_dirs(geo_root):
        fpath = os.path.join(geo_root, pid, "race.csv")
        rows = _read_csv_rows(fpath)
        if not rows:
            continue
        to_ins = []
        for r in rows:
            # Column may be 'note_idx' or 'note_id'; ignore and trust folder pid
            race_id = _safe_int(r.get("race_id"))
            race_name = r.get("race_name") or None
            if race_id is None and (race_name is None or race_name == ""):
                # Nothing to insert
                continue
            if race_id is None:
                # Use -1 to comply with NOT NULL PK; still keep the name
                race_id = -1
            to_ins.append((pid, race_id, race_name, fpath))
        if to_ins:
            conn.executemany("""
                INSERT INTO patient_race (patient_id, race_id, race_name, source_file)
                VALUES (?,?,?,?)
                ON CONFLICT(patient_id, race_id) DO UPDATE SET
                  race_name = COALESCE(excluded.race_name, patient_race.race_name),
                  source_file = COALESCE(excluded.source_file, patient_race.source_file)
            """, to_ins)
            total += len(to_ins)
    print(f"[geo] race rows upserted: {total}")
    return total

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Ingest patient geography (birth/resident country) and race into trial.db tables.")
    ap.add_argument("--db", required=True, help="Path to trial.db (SQLite).")
    ap.add_argument("--geo-root", required=True, help="Root path to patient_geographic_build.")
    ap.add_argument("--recreate", action="store_true", default=False, help="Drop and recreate ONLY the new geography tables before ingest.")
    args = ap.parse_args()

    if not os.path.isdir(args.geo_root):
        print(f"[error] geo-root does not exist: {args.geo_root}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)

    with db_conn(args.db) as conn:
        ensure_geo_schema(conn, recreate=args.recreate)
        n1 = ingest_birth_country(conn, args.geo_root)
        n2 = ingest_resident_country(conn, args.geo_root)
        n3 = ingest_race(conn, args.geo_root)
        print(f"[done] inserted birth={n1}, resident={n2}, race={n3} rows.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)
