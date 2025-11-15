#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest disease-prevention facts into SQLite (time-aware explicit intervals),
reading from FINAL files.

Expected layout under --root:
  patient_facts_export/{pid}/disease_prevention.final.jsonl

Additionally, to set is_root:
  patient_coded_results/{pid}/disease_prevention.final.jsonl

Rule:
  is_root = 1 if entity_variable_name exists in the coded file for that patient, else 0.

Example:
  python ingest_disease_prevention.py \
    --db <SATIR_ROOT>/build_sigir/trial.db \
    --root <SATIR_ROOT>/patient_disease_prevention_build \
    --recreate
"""

from __future__ import annotations
import os, sys, json, argparse, sqlite3, re, hashlib
from typing import Iterable, Dict, Any, List, Tuple, Optional, Set
from contextlib import contextmanager
from math import isfinite

# -------------------- Defaults --------------------
DEFAULT_ROOT = "<SATIR_ROOT>/patient_disease_prevention_build"
FACTS_DIRNAME = "patient_facts_export"
CODED_DIRNAME = "patient_coded_results"
DEFAULT_FILENAME = "disease_prevention.final.jsonl"

# ===================================== Timeframe parsing (same style as your facts ingester) =====================================
_TIMEFRAME_TOKEN_PAT = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years))"
)
TF_FINDER_RE = re.compile(_TIMEFRAME_TOKEN_PAT)

_UNIT_HOURS = {
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 24.0 * 7.0,
    "months": 24.0 * 30.0,
    "years": 24.0 * 365.0,
}


def tf_window_hours(token: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if token is None:
        return None, None
    if token == "now":
        return 0.0, 0.0
    if token == "inthehistory":
        return -1.0e9, 0.0
    m = re.match(r"inthepast(\d+)(minutes|hours|days|weeks|months|years)$", token or "")
    if m:
        n, u = int(m.group(1)), m.group(2)
        return -(n * _UNIT_HOURS[u]), 0.0
    m = re.match(r"inthefuture(?:(\d+)(minutes|hours|days|weeks|months|years))?$", token or "")
    if m:
        n = m.group(1)
        u = m.group(2)
        if n and u:
            return 0.0, (int(n) * _UNIT_HOURS[u])
        return 0.0, 1.0e9
    m = re.match(r"foradurationof(\d+)(minutes|hours|days|weeks|months|years)$", token or "")
    if m:
        n, u = int(m.group(1)), m.group(2)
        return -(n * _UNIT_HOURS[u]), 0.0
    return None, None


def split_base_and_timeframe(varname: str) -> tuple[str, Optional[str]]:
    varname = (varname or "").strip()
    m = TF_FINDER_RE.search(varname)
    if not m:
        return varname, None
    tf = m.group(0)
    s, e = m.span()
    base = (varname[:s] + varname[e:]).strip("_")
    base = re.sub(r"__+", "_", base)
    return base, tf


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if isfinite(v) else None
    except Exception:
        return None


def canonical_interval_token(lb: Optional[float], ub: Optional[float], li: Optional[bool], ui: Optional[bool]) -> str:
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "null"
        return ("{:.6f}".format(v)).rstrip("0").rstrip(".") if abs(v) < 1e8 else str(int(v))

    def _ib(b: Optional[bool]) -> str:
        return "1" if (b is None or b is True) else "0"

    return f"interval({_fmt(lb)},{_fmt(ub)},{_ib(li)},{_ib(ui)})"


def resolve_interval_fields(rec: Dict[str, Any], name_for_legacy: str) -> Tuple[str, Optional[str], Optional[float], Optional[float], int, int]:
    """
    Returns:
      (tf_token, timeframe_legacy_token_or_none, lb_hours, ub_hours, lb_inc_int, ub_inc_int)

    Prefers explicit fields:
      start_time_in_hours, end_time_in_hours, start_time_inclusive, end_time_inclusive

    Falls back to legacy tf_token or embedded token in name.
    Defaults to interval(0,0,1,1) if nothing found.
    """
    has_explicit = any(k in rec for k in ("start_time_in_hours", "end_time_in_hours", "start_time_inclusive", "end_time_inclusive"))
    if has_explicit:
        lb = _safe_float(rec.get("start_time_in_hours"))
        ub = _safe_float(rec.get("end_time_in_hours"))
        li = bool(rec.get("start_time_inclusive", True))
        ui = bool(rec.get("end_time_inclusive", True))
        tf_tok = canonical_interval_token(lb, ub, li, ui)
        return tf_tok, None, lb, ub, int(li), int(ui)

    tf_tok = (rec.get("tf_token") or "").strip() or None
    if tf_tok:
        lb, ub = tf_window_hours(tf_tok)
        return tf_tok, tf_tok, lb, ub, 1, 1

    _, tf_tok2 = split_base_and_timeframe(name_for_legacy or "")
    if tf_tok2:
        lb, ub = tf_window_hours(tf_tok2)
        return tf_tok2, tf_tok2, lb, ub, 1, 1

    return "interval(0,0,1,1)", "now", 0.0, 0.0, 1, 1


def _read_value_and_kind(rec: Dict[str, Any]) -> Tuple[Optional[float], str]:
    # prefer 'value' if present; else map 'extracted_value' + 'type'
    if "value" in rec and rec["value"] is not None:
        v = rec["value"]
        kind = (rec.get("kind") or rec.get("type") or "bool").strip().lower()
    else:
        ev = rec.get("extracted_value", None)
        t = (rec.get("type") or rec.get("kind") or "Bool").strip().lower()
        kind = "num" if t in {"int", "float", "number", "num"} else "bool"
        if ev is None:
            v = None
        elif kind == "bool":
            v = 1.0 if bool(ev) else 0.0
        else:
            try:
                v = float(ev)
            except Exception:
                v = None
    return (None if v is None else float(v)), ("num" if kind == "num" else "bool")


# ===================================== IO helpers =====================================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def list_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


# ===================================== DB plumbing =====================================
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


def drop_tables(conn: sqlite3.Connection):
    conn.executescript("""
    DROP TABLE IF EXISTS patient_prevention_constraints;
    """)


def ensure_schema(conn: sqlite3.Connection, recreate: bool):
    if recreate:
        drop_tables(conn)

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS patient_prevention_constraints (
      row_key TEXT PRIMARY KEY,

      patient_id TEXT NOT NULL,
      entity_var TEXT NOT NULL,

      is_root    INTEGER NOT NULL DEFAULT 0,

      kind       TEXT NOT NULL CHECK (kind IN ('bool','num')),
      value      REAL,

      prevent_source TEXT,
      class          TEXT,
      hop            INTEGER,
      derivation_stage TEXT,
      derivation_rule  TEXT,

      source_variable_name    TEXT,
      source_concept_id       TEXT,
      derived_from_variable   TEXT,
      derived_from_concept_id TEXT,
      fact_id                TEXT,

      concept_id           TEXT,
      preferred_term       TEXT,
      fully_specified_name TEXT,
      snomed_type          TEXT,
      span_match           TEXT,
      entity_variable_meaning TEXT,

      tf_token TEXT NOT NULL,
      timeframe TEXT,
      tf_lb_hours REAL,
      tf_ub_hours REAL,
      tf_lb_inclusive INTEGER NOT NULL DEFAULT 1,
      tf_ub_inclusive INTEGER NOT NULL DEFAULT 1,

      round INTEGER NOT NULL DEFAULT 9999,

      raw_json TEXT,
      created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    """)

    try:
        conn.execute("ALTER TABLE patient_prevention_constraints ADD COLUMN is_root INTEGER NOT NULL DEFAULT 0;")
    except sqlite3.OperationalError:
        pass

    conn.executescript("""
    CREATE INDEX IF NOT EXISTS idx_dpp_pat     ON patient_prevention_constraints(patient_id);
    CREATE INDEX IF NOT EXISTS idx_dpp_var     ON patient_prevention_constraints(entity_var);
    CREATE INDEX IF NOT EXISTS idx_dpp_concept ON patient_prevention_constraints(concept_id);
    CREATE INDEX IF NOT EXISTS idx_dpp_source  ON patient_prevention_constraints(prevent_source);
    CREATE INDEX IF NOT EXISTS idx_dpp_bounds  ON patient_prevention_constraints(entity_var, tf_lb_hours, tf_ub_hours);
    CREATE INDEX IF NOT EXISTS idx_dpp_is_root ON patient_prevention_constraints(is_root);
    """)


# ===================================== Field extraction =====================================
def _first_present(rec: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = rec.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def get_entity_var(rec: Dict[str, Any]) -> Optional[str]:
    return _first_present(rec, ["entity_variable_name", "entity_var", "target_variable_name", "new_variable_name"])


def extract_concept_fields(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    mapping = rec.get("mapping") or {}
    concept_id = _first_present(rec, ["conceptId", "derived_conceptId"]) or _first_present(mapping, ["conceptId"])
    preferred = _first_present(rec, ["preferred_term", "derived_entity_term"]) or _first_present(mapping, ["preferred_term"])
    fsn = _first_present(rec, ["fully_specified_name"]) or _first_present(mapping, ["fully_specified_name"])
    snomed_type = _first_present(mapping, ["type"])
    return concept_id, preferred, fsn, snomed_type


def sha1_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ===================================== Root vars (coded results) =====================================
def load_root_entity_vars(root: str, coded_dirname: str, pid: str, filename: str) -> Set[str]:
    """
    Load entity_variable_name set from:
      {root}/{coded_dirname}/{pid}/{filename}
    If file doesn't exist, returns empty set.
    """
    coded_path = os.path.join(root, coded_dirname, pid, filename)
    out: Set[str] = set()
    if not os.path.isfile(coded_path):
        return out
    for rec in iter_jsonl(coded_path):
        ev = get_entity_var(rec)
        if ev:
            out.add(ev)
    return out


# ===================================== Ingestion =====================================
def ingest_disease_prevention(
    root: str,
    facts_dirname: str,
    filename: str,
    conn: sqlite3.Connection,
    coded_dirname: str = CODED_DIRNAME,
    batch_sz: int = 20000
) -> int:
    facts_root = os.path.join(root, facts_dirname)
    if not os.path.isdir(facts_root):
        print(f"[disease_prevention] MISSING facts root: {facts_root}")
        return 0

    patients = list_dirs(facts_root)
    if not patients:
        print(f"[disease_prevention] no patients under {facts_root}")
        return 0

    total = 0
    batch: List[Tuple] = []

    ins_sql = """
    INSERT INTO patient_prevention_constraints
      (row_key, patient_id, entity_var, is_root, kind, value,
       prevent_source, class, hop, derivation_stage, derivation_rule,
       source_variable_name, source_concept_id, derived_from_variable, derived_from_concept_id, fact_id,
       concept_id, preferred_term, fully_specified_name, snomed_type, span_match, entity_variable_meaning,
       tf_token, timeframe, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive, round,
       raw_json)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(row_key) DO UPDATE SET
      is_root = excluded.is_root
    """

    for pid in patients:
        fpath = os.path.join(facts_root, pid, filename)
        if not os.path.isfile(fpath):
            continue

        root_vars = load_root_entity_vars(root, coded_dirname, pid, filename)

        for rec in iter_jsonl(fpath):
            pid_eff = (rec.get("patient_id") or pid)
            entity_var = get_entity_var(rec) or ""
            if not entity_var:
                continue

            is_root = 1 if entity_var in root_vars else 0

            tf_token, timeframe, lb_h, ub_h, lb_inc, ub_inc = resolve_interval_fields(rec, entity_var)
            value, kind = _read_value_and_kind(rec)

            prevent_source = (rec.get("prevent_source") or "").strip() or None
            clazz = (rec.get("class") or "").strip() or None

            hop = rec.get("hop")
            try:
                hop = int(hop) if hop is not None else None
            except Exception:
                hop = None

            deriv_stage = (rec.get("derivation_stage") or "").strip() or None
            deriv_rule = (rec.get("derivation_rule") or "").strip() or None

            src_var = (rec.get("source_variable_name") or "").strip() or None
            src_cid = (rec.get("source_conceptId") or "").strip() or None
            dfrom_var = (rec.get("derived_from_variable") or "").strip() or None
            dfrom_cid = (rec.get("derived_from_conceptId") or "").strip() or None
            fact_id = rec.get("fact_id")
            fact_id = str(fact_id) if fact_id is not None else None

            concept_id, preferred, fsn, snomed_type = extract_concept_fields(rec)

            span_match = (rec.get("span_match") or "").strip() or None
            meaning = (rec.get("entity_variable_meaning") or "").strip() or None

            rnd = rec.get("round", 9999)
            try:
                rnd = int(rnd)
            except Exception:
                rnd = 9999

            rec2 = dict(rec)
            rec2["patient_id"] = pid_eff
            rec2["entity_variable_name"] = entity_var
            raw_json = json.dumps(rec2, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            row_key = sha1_key(raw_json)

            batch.append((
                row_key, pid_eff, entity_var, is_root, kind, value,
                prevent_source, clazz, hop, deriv_stage, deriv_rule,
                src_var, src_cid, dfrom_var, dfrom_cid, fact_id,
                concept_id, preferred, fsn, snomed_type, span_match, meaning,
                tf_token, timeframe, lb_h, ub_h, lb_inc, ub_inc, rnd,
                raw_json
            ))

            if len(batch) >= batch_sz:
                conn.executemany(ins_sql, batch)
                total += len(batch)
                batch.clear()

    if batch:
        conn.executemany(ins_sql, batch)
        total += len(batch)
        batch.clear()

    print(f"[disease_prevention] inserted (and upsert-updated is_root where needed): {total}")
    return total


# ===================================== CLI =====================================
def main():
    ap = argparse.ArgumentParser(description="Ingest disease-prevention FINAL JSONL facts into SQLite with explicit time intervals.")
    ap.add_argument("--db", required=True, help="SQLite file path to create/update.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Root containing patient_facts_export/{pid}/disease_prevention.final.jsonl")
    ap.add_argument("--facts-dirname", default=FACTS_DIRNAME, help="Directory name under root holding patient subdirs.")
    ap.add_argument("--coded-dirname", default=CODED_DIRNAME, help="Directory name under root holding coded results subdirs.")
    ap.add_argument("--filename", default=DEFAULT_FILENAME, help="JSONL filename inside each patient directory.")
    ap.add_argument("--recreate", action="store_true", help="Drop and recreate tables before ingesting.")
    args = ap.parse_args()

    print("[config]")
    print("  db           :", args.db)
    print("  root         :", args.root)
    print("  facts_dirname:", args.facts_dirname)
    print("  coded_dirname:", args.coded_dirname)
    print("  filename     :", args.filename)
    print("  recreate     :", args.recreate)

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)

    with db_conn(args.db) as conn:
        ensure_schema(conn, recreate=args.recreate)
        ingest_disease_prevention(
            root=args.root,
            facts_dirname=args.facts_dirname,
            filename=args.filename,
            conn=conn,
            coded_dirname=args.coded_dirname,
        )

    print("[done] disease prevention ingestion complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)