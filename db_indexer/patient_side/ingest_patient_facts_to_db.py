#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-aware facts ingestion (explicit intervals + demographics file), tailored for:
  - Inclusion facts from patient_build_inclusion/.../inclusion/canonical.final.jsonl
  - Exclusion facts from patient_build_exclusion/.../exclusion/canonical.final.jsonl
  - Demographics from patient_build_inclusion/.../patient_coded_results/{pid}/demographics.jsonl

Creates SQLite tables:
  patient_inclusion_constraints, patient_exclusion_constraints, patient_demographic_constraints
and a union view: vw_fact_timeframes

NEW (root labeling):
  - Adds concept_id TEXT and is_root INTEGER to facts_{inclusion,exclusion}
  - For inclusion facts (and "other demographics as facts"), set:
      is_root = 1 if concept_id is found in:
        {--root-fact-root}/patient_coded_results/{pid}_inclusion/canonical.jsonl OR
        {--root-fact-root}/patient_coded_results/{pid}_inclusion/diagnosis.jsonl
      else is_root = 0
  - Exclusion facts default is_root=0 (kept symmetric; you can extend later)

Demographics behavior:
  - Age + sex variables (patient_age_value_recorded_in_*, patient_sex_is_*)
    go into patient_demographic_constraints (with normalized age units).
  - All other variables in demographics.jsonl (e.g., patient_is_pregnant,
    patient_is_inpatient, etc.) are ingested as time-aware facts into
    patient_inclusion_constraints, with root labeling if concept_id is present.

NEW (progress logging):
  - Logs progress while building root concept map
  - Logs per-patient and per-row progress during inclusion/exclusion fact ingest
  - Logs per-patient and per-row progress during demographics ingest
  - All progress prints are flushed immediately for live terminal visibility
"""

from __future__ import annotations
import os, sys, json, argparse, glob, sqlite3, re, time
from typing import Iterable, Dict, Any, List, Tuple, Optional
from contextlib import contextmanager
from math import isfinite


DEFAULT_INCL_ROOT = "<SATIR_ROOT>/patient_build_inclusion"
DEFAULT_EXCL_ROOT = "<SATIR_ROOT>/patient_build_exclusion"

FACTS_DIRNAME_INCL = "patient_facts_export"
FACTS_DIRNAME_EXCL = "patient_facts_export"
DEMO_DIRNAME       = "patient_coded_results"


_TIMEFRAME_TOKEN_PAT = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years))"
)
TF_FINDER_RE = re.compile(_TIMEFRAME_TOKEN_PAT)

_UNIT_HOURS = {
    "minutes": 1.0/60.0,
    "hours":   1.0,
    "days":    24.0,
    "weeks":   24.0*7.0,
    "months":  24.0*30.0,
    "years":   24.0*365.0,
}


def _fmt_n(n: int) -> str:
    return f"{n:,}"


def _progress(msg: str) -> None:
    print(msg, flush=True)


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
        n = m.group(1); u = m.group(2)
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


# ===================================== Root conceptId lookup =====================================
def _norm_pid(pid: Any) -> Optional[str]:
    if pid is None:
        return None
    s = str(pid).strip()
    if not s:
        return None
    if s.endswith("_inclusion"):
        s = s[:-10]
    elif s.endswith("_exclusion"):
        s = s[:-10]
    return s or None


def _norm_concept_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _read_concept_ids_from_jsonl(path: str) -> set[str]:
    out: set[str] = set()
    for rec in iter_jsonl(path):
        cid = _norm_concept_id(rec.get("conceptId") or rec.get("concept_id"))
        if cid:
            out.add(cid)
    return out


def _root_patient_dir(root_fact_root: str, pid: str) -> str:
    pid0 = _norm_pid(pid)
    return os.path.join(root_fact_root, "patient_coded_results", f"{pid0}_inclusion")


def load_root_concept_ids_for_patient(root_fact_root: Optional[str], pid: str) -> set[str]:
    if not root_fact_root:
        return set()

    pdir = _root_patient_dir(root_fact_root, pid)
    if not os.path.isdir(pdir):
        return set()

    out: set[str] = set()

    c1 = os.path.join(pdir, "canonical.jsonl")
    c2 = os.path.join(pdir, "diagnosis.jsonl")

    if os.path.isfile(c1):
        out |= _read_concept_ids_from_jsonl(c1)
    if os.path.isfile(c2):
        out |= _read_concept_ids_from_jsonl(c2)

    return out


def build_patient_root_concept_map(
    root_fact_root: Optional[str],
    progress_every: int = 100
) -> dict[str, set[str]]:
    if not root_fact_root:
        return {}

    coded_root = os.path.join(root_fact_root, "patient_coded_results")
    if not os.path.isdir(coded_root):
        _progress(f"[root] coded_results not found under root_fact_root: {coded_root}")
        return {}

    dirs = list_dirs(coded_root)
    total = len(dirs)
    _progress(f"[root] scanning {total} patient dirs under {coded_root}")

    out: dict[str, set[str]] = {}
    t0 = time.time()

    for i, d in enumerate(dirs, 1):
        pdir = os.path.join(coded_root, d)

        pid_norm = _norm_pid(d)
        if not pid_norm:
            continue

        s: set[str] = set()
        c1 = os.path.join(pdir, "canonical.jsonl")
        c2 = os.path.join(pdir, "diagnosis.jsonl")

        if os.path.isfile(c1):
            s |= _read_concept_ids_from_jsonl(c1)
        if os.path.isfile(c2):
            s |= _read_concept_ids_from_jsonl(c2)

        if s:
            out[pid_norm] = s

        if i == 1 or i % progress_every == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            _progress(
                f"[root] progress {i}/{total} dirs "
                f"({i/total:.1%}); loaded {len(out)} patients with roots; "
                f"{rate:.1f} dirs/s"
            )

    _progress(f"[root] loaded root concept sets for {len(out)} patients from {coded_root}")
    return out


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


def make_schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS patient_inclusion_constraints (
  patient_id TEXT NOT NULL,
  base_var   TEXT NOT NULL,
  tf_token   TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('bool','num')),
  value      REAL,
  source     TEXT,
  round      INTEGER NOT NULL,

  concept_id TEXT,
  is_root    INTEGER NOT NULL DEFAULT 0,

  timeframe        TEXT,
  tf_lb_hours      REAL,
  tf_ub_hours      REAL,
  tf_lb_inclusive  INTEGER NOT NULL DEFAULT 1,
  tf_ub_inclusive  INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
  PRIMARY KEY (patient_id, base_var, tf_token, kind, round)
);

CREATE TABLE IF NOT EXISTS patient_exclusion_constraints (
  patient_id TEXT NOT NULL,
  base_var   TEXT NOT NULL,
  tf_token   TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('bool','num')),
  value      REAL,
  source     TEXT,
  round      INTEGER NOT NULL,

  concept_id TEXT,
  is_root    INTEGER NOT NULL DEFAULT 0,

  timeframe        TEXT,
  tf_lb_hours      REAL,
  tf_ub_hours      REAL,
  tf_lb_inclusive  INTEGER NOT NULL DEFAULT 1,
  tf_ub_inclusive  INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
  PRIMARY KEY (patient_id, base_var, tf_token, kind, round)
);

CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_var     ON patient_inclusion_constraints(base_var, tf_token);
CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_pat     ON patient_inclusion_constraints(patient_id);
CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_rnd     ON patient_inclusion_constraints(round);
CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_base_tf ON patient_inclusion_constraints(base_var, timeframe);
CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_bounds  ON patient_inclusion_constraints(base_var, tf_lb_hours, tf_ub_hours);
CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_concept ON patient_inclusion_constraints(concept_id);
CREATE INDEX IF NOT EXISTS patient_inclusion_constraints_idx_isroot  ON patient_inclusion_constraints(is_root);

CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_var     ON patient_exclusion_constraints(base_var, tf_token);
CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_pat     ON patient_exclusion_constraints(patient_id);
CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_rnd     ON patient_exclusion_constraints(round);
CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_base_tf ON patient_exclusion_constraints(base_var, timeframe);
CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_bounds  ON patient_exclusion_constraints(base_var, tf_lb_hours, tf_ub_hours);
CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_concept ON patient_exclusion_constraints(concept_id);
CREATE INDEX IF NOT EXISTS patient_exclusion_constraints_idx_isroot  ON patient_exclusion_constraints(is_root);

CREATE TABLE IF NOT EXISTS patient_demographic_constraints (
  patient_id  TEXT NOT NULL,
  tf_token    TEXT NOT NULL,
  round       INTEGER NOT NULL,
  source      TEXT,
  age_years   REAL,
  age_months  REAL,
  age_days    REAL,
  sex         TEXT,
  timeframe        TEXT,
  tf_lb_hours      REAL,
  tf_ub_hours      REAL,
  tf_lb_inclusive  INTEGER NOT NULL DEFAULT 1,
  tf_ub_inclusive  INTEGER NOT NULL DEFAULT 1,
  created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now')),
  PRIMARY KEY (patient_id, tf_token, round)
);
CREATE INDEX IF NOT EXISTS demo_idx_pat_tf   ON patient_demographic_constraints(patient_id, tf_token);

DROP VIEW IF EXISTS vw_fact_timeframes;
CREATE VIEW vw_fact_timeframes AS
  SELECT 'inclusion' AS side, patient_id, base_var, concept_id, is_root,
         timeframe, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive, kind, value, round
  FROM patient_inclusion_constraints
  UNION ALL
  SELECT 'exclusion' AS side, patient_id, base_var, concept_id, is_root,
         timeframe, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive, kind, value, round
  FROM patient_exclusion_constraints;
"""


def drop_tables(conn: sqlite3.Connection):
    conn.executescript("""
    DROP VIEW IF EXISTS vw_fact_timeframes;
    DROP TABLE IF EXISTS patient_inclusion_constraints;
    DROP TABLE IF EXISTS patient_exclusion_constraints;
    DROP TABLE IF EXISTS patient_demographic_constraints;
    """)


def ensure_schema(conn: sqlite3.Connection, recreate: bool):
    if recreate:
        drop_tables(conn)
    conn.executescript(make_schema_sql())


AGE_BASE_RX = re.compile(r"^patient_age_value_recorded_in_(years|months|days)$")
SEX_BASE_RX = re.compile(r"^patient_sex_is_(.+)$")


def normalize_sex(token: str) -> str:
    t = (token or "").strip().lower()
    if t in {"m","male","man","masculine","biological_male"}:
        return "male"
    if t in {"f","female","woman","feminine","biological_female"}:
        return "female"
    if t in {"intersex"}:
        return "intersex"
    if t in {"unknown","undisclosed","na","n/a","none"}:
        return "unknown"
    return (token or "").strip()


def _is_demographic_base_var(base_var: str) -> bool:
    if not base_var:
        return False
    if base_var.startswith("patient_sex_is_"):
        return True
    if base_var.startswith("patient_age_value_recorded_in_"):
        return True
    return False


def upsert_facts_into_table(conn: sqlite3.Connection, table: str, rows: List[Tuple]):
    filtered = [r for r in rows if not _is_demographic_base_var(r[1] or "")]
    if not filtered:
        return

    conn.executemany(f"""
    INSERT INTO {table}
      (patient_id, base_var, tf_token, kind, value, source, round,
       concept_id, is_root,
       timeframe, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive)
    VALUES (?,?,?,?,?,?,?,
            ?,?,
            ?,?,?,?,?)
    ON CONFLICT(patient_id, base_var, tf_token, kind, round) DO UPDATE SET
      value           = excluded.value,
      source          = COALESCE(excluded.source, {table}.source),
      concept_id      = COALESCE(excluded.concept_id, {table}.concept_id),
      is_root         = excluded.is_root,
      timeframe       = excluded.timeframe,
      tf_lb_hours     = excluded.tf_lb_hours,
      tf_ub_hours     = excluded.tf_ub_hours,
      tf_lb_inclusive = excluded.tf_lb_inclusive,
      tf_ub_inclusive = excluded.tf_ub_inclusive
    """, filtered)


def upsert_demographics(conn: sqlite3.Connection, rows: List[Tuple]):
    if not rows:
        return
    conn.executemany("""
    INSERT INTO patient_demographic_constraints
      (patient_id, tf_token, round, source, age_years, age_months, age_days, sex,
       timeframe, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(patient_id, tf_token, round) DO UPDATE SET
      source          = COALESCE(excluded.source,      patient_demographic_constraints.source),
      age_years       = COALESCE(excluded.age_years,   patient_demographic_constraints.age_years),
      age_months      = COALESCE(excluded.age_months,  patient_demographic_constraints.age_months),
      age_days        = COALESCE(excluded.age_days,    patient_demographic_constraints.age_days),
      sex             = COALESCE(excluded.sex,         patient_demographic_constraints.sex),
      timeframe       = COALESCE(excluded.timeframe,   patient_demographic_constraints.timeframe),
      tf_lb_hours     = COALESCE(excluded.tf_lb_hours, patient_demographic_constraints.tf_lb_hours),
      tf_ub_hours     = COALESCE(excluded.tf_ub_hours, patient_demographic_constraints.tf_ub_hours),
      tf_lb_inclusive = COALESCE(excluded.tf_lb_inclusive, patient_demographic_constraints.tf_lb_inclusive),
      tf_ub_inclusive = COALESCE(excluded.tf_ub_inclusive, patient_demographic_constraints.tf_ub_inclusive)
    """, rows)


def _safe_float(x: Optional[float]) -> Optional[float]:
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


def resolve_interval_fields(rec: Dict[str, Any], base_name: str) -> Tuple[str, Optional[str], Optional[float], Optional[float], int, int]:
    has_explicit = any(k in rec for k in ("start_time_in_hours","end_time_in_hours","start_time_inclusive","end_time_inclusive"))
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

    _, tf_tok2 = split_base_and_timeframe(base_name or "")
    if tf_tok2:
        lb, ub = tf_window_hours(tf_tok2)
        return tf_tok2, tf_tok2, lb, ub, 1, 1

    return "interval(0,0,1,1)", "now", 0.0, 0.0, 1, 1


def _read_value_and_kind(rec: Dict[str, Any]) -> Tuple[Optional[float], str]:
    if "value" in rec and rec["value"] is not None:
        v = rec["value"]
        kind = (rec.get("kind") or rec.get("type") or "bool").strip().lower()
    else:
        ev = rec.get("extracted_value", None)
        t  = (rec.get("type") or rec.get("kind") or "Bool").strip().lower()
        kind = "num" if t in {"int","float","number","num"} else "bool"
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


def ingest_side_facts(
    root: str,
    facts_dirname: str,
    side: str,
    conn: sqlite3.Connection,
    final_only: bool,
    root_map: Optional[dict[str, set[str]]] = None,
    batch_sz: int = 20000,
    progress_every_patients: int = 25,
    progress_every_rows: int = 100000
) -> int:
    assert side in ("inclusion", "exclusion")
    side_total = 0

    facts_root = os.path.join(root, facts_dirname)
    if not os.path.isdir(facts_root):
        _progress(f"[facts:{side}] MISSING facts root: {facts_root}")
        return 0

    patients = list_dirs(facts_root)
    total_patients = len(patients)
    if not patients:
        _progress(f"[facts:{side}] no patients under {facts_root}")
        return 0

    _progress(f"[facts:{side}] scanning {total_patients} patient dirs under {facts_root}")

    facts_batch: List[Tuple] = []
    seen_rows = 0
    t0 = time.time()

    for pidx, pid in enumerate(patients, 1):
        sdir = os.path.join(facts_root, pid, side)
        if not os.path.isdir(sdir):
            continue

        cpath = os.path.join(sdir, "canonical.final.jsonl")
        if os.path.isfile(cpath):
            patterns = [cpath]
            rnd_for = lambda _f: 9999
        else:
            if final_only:
                patterns = [os.path.join(sdir, "facts.round9999.jsonl")]
            else:
                patterns = sorted(glob.glob(os.path.join(sdir, "facts.round*.jsonl")))

            def _rnd(f):
                try:
                    base = os.path.basename(f)
                    return int(base.split("facts.round", 1)[1].split(".jsonl", 1)[0])
                except Exception:
                    return 9999
            rnd_for = _rnd

        if pidx == 1 or pidx % progress_every_patients == 0 or pidx == total_patients:
            elapsed = time.time() - t0
            _progress(
                f"[facts:{side}] patient {pidx}/{total_patients} "
                f"({pidx/total_patients:.1%}); seen_rows={_fmt_n(seen_rows)} "
                f"upserted={_fmt_n(side_total)} elapsed={elapsed:.1f}s"
            )

        for fpath in patterns:
            rnd = rnd_for(fpath)
            for rec in iter_jsonl(fpath):
                seen_rows += 1

                pid_eff = rec.get("patient_id") or pid
                nm = (rec.get("base_var") or rec.get("entity_variable_name") or rec.get("source_variable_name") or "").strip()
                if not nm:
                    continue

                base_var, _ = split_base_and_timeframe(nm)
                tf_token, timeframe, lb_h, ub_h, lb_inc, ub_inc = resolve_interval_fields(rec, nm)

                if _is_demographic_base_var(base_var):
                    continue

                value, kind = _read_value_and_kind(rec)
                src = rec.get("source")

                concept_id = _norm_concept_id(rec.get("conceptId") or rec.get("concept_id"))

                is_root = 0
                if side == "inclusion" and concept_id and root_map is not None:
                    pid_eff_norm = _norm_pid(pid_eff)
                    pid_norm = _norm_pid(pid)
                    pid_roots = root_map.get(pid_eff_norm) or root_map.get(pid_norm) or set()
                    if concept_id in pid_roots:
                        is_root = 1

                facts_batch.append(
                    (pid_eff, base_var, tf_token, kind, value, src, rnd,
                     concept_id, is_root,
                     timeframe, lb_h, ub_h, lb_inc, ub_inc)
                )

                if seen_rows % progress_every_rows == 0:
                    elapsed = time.time() - t0
                    rate = seen_rows / elapsed if elapsed > 0 else 0.0
                    _progress(
                        f"[facts:{side}] rows={_fmt_n(seen_rows)} "
                        f"upserted={_fmt_n(side_total)} batch={_fmt_n(len(facts_batch))} "
                        f"rate={rate:.1f} rows/s"
                    )

                if len(facts_batch) >= batch_sz:
                    upsert_facts_into_table(conn, f"facts_{side}", facts_batch)
                    side_total += len(facts_batch)
                    facts_batch.clear()

    if facts_batch:
        upsert_facts_into_table(conn, f"facts_{side}", facts_batch)
        side_total += len(facts_batch)
        facts_batch.clear()

    elapsed = time.time() - t0
    _progress(f"[facts:{side}] upserted rows (non-demographics): {side_total} in {elapsed:.1f}s")
    return side_total


def ingest_demographics_jsonl(
    inclusion_root: str,
    conn: sqlite3.Connection,
    root_map: Optional[dict[str, set[str]]] = None,
    batch_sz: int = 20000,
    progress_every_patients: int = 25,
    progress_every_rows: int = 50000
) -> int:
    total_demo = 0
    total_fact = 0
    seen_rows = 0
    t0 = time.time()

    coded_root = os.path.join(inclusion_root, "patient_coded_results")
    facts_root = os.path.join(inclusion_root, "patient_facts_export")

    demo_batch: List[Tuple] = []
    fact_batch: List[Tuple] = []

    def ingest_one(pid: str, fpath: str):
        nonlocal total_demo, total_fact, demo_batch, fact_batch, seen_rows
        for rec in iter_jsonl(fpath):
            seen_rows += 1

            nm = (rec.get("entity_variable_name") or rec.get("base_var") or "").strip()
            if not nm:
                continue
            base_var, _ = split_base_and_timeframe(nm)
            tf_token, timeframe, lb_h, ub_h, lb_inc, ub_inc = resolve_interval_fields(rec, nm)

            pid_eff = rec.get("patient_id") or pid
            rnd = int(rec.get("round", 9999))
            src = rec.get("source")

            m_age = AGE_BASE_RX.match(base_var)
            m_sex = SEX_BASE_RX.match(base_var)

            if m_age or m_sex:
                val, _kind = _read_value_and_kind(rec)
                ay = am = ad = None
                sex_val = None

                if m_age and val is not None:
                    unit = m_age.group(1)
                    if unit == "years":
                        ay = val
                    elif unit == "months":
                        am = val
                    elif unit == "days":
                        ad = val

                if m_sex:
                    if val and val != 0.0:
                        sex_val = normalize_sex(m_sex.group(1))

                demo_batch.append(
                    (pid_eff, tf_token, rnd, src,
                     ay, am, ad, sex_val,
                     timeframe, lb_h, ub_h, lb_inc, ub_inc)
                )
                if len(demo_batch) >= batch_sz:
                    upsert_demographics(conn, demo_batch)
                    total_demo += len(demo_batch)
                    demo_batch.clear()
            else:
                value, kind = _read_value_and_kind(rec)
                concept_id = _norm_concept_id(rec.get("conceptId") or rec.get("concept_id"))

                is_root = 0
                if concept_id and root_map is not None:
                    pid_eff_norm = _norm_pid(pid_eff)
                    pid_norm = _norm_pid(pid)
                    pid_roots = root_map.get(pid_eff_norm) or root_map.get(pid_norm) or set()
                    if concept_id in pid_roots:
                        is_root = 1

                fact_batch.append(
                    (pid_eff, base_var, tf_token, kind, value, src, rnd,
                     concept_id, is_root,
                     timeframe, lb_h, ub_h, lb_inc, ub_inc)
                )
                if len(fact_batch) >= batch_sz:
                    upsert_facts_into_table(conn, "patient_inclusion_constraints", fact_batch)
                    total_fact += len(fact_batch)
                    fact_batch.clear()

            if seen_rows % progress_every_rows == 0:
                elapsed = time.time() - t0
                rate = seen_rows / elapsed if elapsed > 0 else 0.0
                _progress(
                    f"[demo] rows={_fmt_n(seen_rows)} age/sex_upserted={_fmt_n(total_demo)} "
                    f"other_fact_upserted={_fmt_n(total_fact)} rate={rate:.1f} rows/s"
                )

    ingested_any = False

    if os.path.isdir(coded_root):
        patients = list_dirs(coded_root)
        total_patients = len(patients)
        _progress(f"[demo] scanning {total_patients} patient dirs under {coded_root}")
        for i, pid in enumerate(patients, 1):
            fpath = os.path.join(coded_root, pid, "demographics.jsonl")
            if i == 1 or i % progress_every_patients == 0 or i == total_patients:
                _progress(f"[demo] patient {i}/{total_patients} ({i/total_patients:.1%})")
            if os.path.isfile(fpath):
                ingested_any = True
                ingest_one(pid, fpath)
    else:
        _progress(f"[demo] coded_results not found, will try facts_export fallback: {coded_root}")

    if not ingested_any and os.path.isdir(facts_root):
        patients = list_dirs(facts_root)
        total_patients = len(patients)
        _progress(f"[demo] fallback scanning {total_patients} patient dirs under {facts_root}")
        for i, pid in enumerate(patients, 1):
            fpath = os.path.join(facts_root, pid, "inclusion", "demographics.jsonl")
            if i == 1 or i % progress_every_patients == 0 or i == total_patients:
                _progress(f"[demo] fallback patient {i}/{total_patients} ({i/total_patients:.1%})")
            if os.path.isfile(fpath):
                ingested_any = True
                ingest_one(pid, fpath)

    if not ingested_any:
        _progress(f"[demo] No demographics files found under either:")
        _progress(f"       - {coded_root}/*/demographics.jsonl")
        _progress(f"       - {facts_root}/*/inclusion/demographics.jsonl")
        return 0

    if demo_batch:
        upsert_demographics(conn, demo_batch)
        total_demo += len(demo_batch)
        demo_batch.clear()

    if fact_batch:
        upsert_facts_into_table(conn, "patient_inclusion_constraints", fact_batch)
        total_fact += len(fact_batch)
        fact_batch.clear()

    _progress(f"[demo] upserted rows (age/sex): {total_demo}")
    _progress(f"[demo->facts] upserted rows (other demographics as facts): {total_fact}")
    return total_demo + total_fact


DAYS_PER_YEAR   = 365.25
MONTHS_PER_YEAR = 12.0
DAYS_PER_MONTH  = DAYS_PER_YEAR / MONTHS_PER_YEAR


def _fill_age_units(ay: Optional[float], am: Optional[float], ad: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    def _sf(x: Optional[float]) -> Optional[float]:
        try:
            v = float(x)
            return v if isfinite(v) else None
        except Exception:
            return None

    ay = _sf(ay); am = _sf(am); ad = _sf(ad)
    if ad is not None:
        am = ad / DAYS_PER_MONTH if am is None else am
        ay = ad / DAYS_PER_YEAR  if ay is None else ay
    elif am is not None:
        ad = am * DAYS_PER_MONTH if ad is None else ad
        ay = am / MONTHS_PER_YEAR if ay is None else ay
    elif ay is not None:
        am = ay * MONTHS_PER_YEAR if am is None else am
        ad = ay * DAYS_PER_YEAR   if ad is None else ad
    return ay, am, ad


def normalize_demographics_units(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("""SELECT patient_id, tf_token, round, source, age_years, age_months, age_days, sex,
                          timeframe, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
                   FROM patient_demographic_constraints""")
    rows = list(cur.fetchall())
    out: List[Tuple] = []
    for pid, tf_token, rnd, src, ay, am, ad, sex, timeframe, lb, ub, li, ui in rows:
        n_ay, n_am, n_ad = _fill_age_units(ay, am, ad)
        if (n_ay != ay) or (n_am != am) or (n_ad != ad):
            out.append((pid, tf_token, rnd, src, n_ay, n_am, n_ad, sex, timeframe, lb, ub, li, ui))
    if out:
        upsert_demographics(conn, out)
    _progress(f"[demo] normalized age units for {len(out)} rows")
    return len(out)


def main():
    ap = argparse.ArgumentParser(
        description="Ingest time-aware facts (inclusion/exclusion) and demographics using explicit intervals and canonical.final.jsonl layout."
    )
    ap.add_argument("--db", default="../../build/trial.db", help="SQLite file path to create/update.")
    ap.add_argument("--inclusion-root", default=DEFAULT_INCL_ROOT, help="Root containing patient_facts_export and patient_coded_results.")
    ap.add_argument("--exclusion-root", default=DEFAULT_EXCL_ROOT, help="Root containing patient_facts_export.")
    ap.add_argument(
        "--root-fact-root",
        default="../../patient_build_sigir_inclusion_root_fact",
        help="Root containing patient_coded_results/{pid}_inclusion/canonical.jsonl and diagnosis.jsonl for is_root labeling (optional)."
    )

    ap.add_argument(
        "--final-only",
        action="store_true",
        help="Only ingest facts.round9999.jsonl if canonical.final.jsonl absent."
    )
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate tables before ingesting."
    )

    args = ap.parse_args()

    _progress("[config]")
    _progress(f"  db             : {args.db}")
    _progress(f"  inclusion_root : {args.inclusion_root}")
    _progress(f"  exclusion_root : {args.exclusion_root}")
    _progress(f"  root_fact_root : {args.root_fact_root}")
    _progress(f"  final_only     : {args.final_only}")
    _progress(f"  recreate       : {args.recreate}")

    db_dir = os.path.dirname(os.path.abspath(args.db))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    root_map = build_patient_root_concept_map(args.root_fact_root.strip() or None)
    _progress(f"[root] patients with root concepts: {len(root_map)}")

    with db_conn(args.db) as conn:
        ensure_schema(conn, recreate=args.recreate)

        _progress("[phase] ingest inclusion facts")
        ingest_side_facts(
            args.inclusion_root, FACTS_DIRNAME_INCL, "inclusion",
            conn, final_only=args.final_only, root_map=root_map
        )

        _progress("[phase] ingest exclusion facts")
        ingest_side_facts(
            args.exclusion_root, FACTS_DIRNAME_EXCL, "exclusion",
            conn, final_only=args.final_only, root_map=root_map
        )

        _progress("[phase] ingest demographics")
        ingest_demographics_jsonl(args.inclusion_root, conn, root_map=root_map)

        _progress("[phase] normalize demographics")
        normalize_demographics_units(conn)

    _progress("[done] ingestion complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr, flush=True)
        sys.exit(130)