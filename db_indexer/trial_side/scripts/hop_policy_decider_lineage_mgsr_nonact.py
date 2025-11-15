#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hop_policy_decider_lineage_mgsr_nonact.py — flattened keep-list decider (index-driven, case_id echo)
(Positive-literals NONACT edition) — TWO BRANCHES

Reads:
  main:       positive_constraint_literals_nonact
  prevention: positive_constraint_literals_prevention_nonact

Writes (NO CONFLICT with act tables):
  main:
    concept_accepted_alternatives_nonact
    positive_constraint_alternatives_expanded_nonact
  prevention:
    concept_accepted_alternatives_prevention_nonact
    positive_constraint_alternatives_expanded_prevention_nonact

Artifacts (NO CONFLICT with act artifacts):
  main:       mbench/hop_decider_logs_nonact
  prevention: mbench/hop_decider_logs_prevention_nonact
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# ────────────────────────────────────────────────────────────────
# NONACT INPUT TABLES (per your note)
# ────────────────────────────────────────────────────────────────
TABLE_MAIN = "positive_constraint_literals_nonact"
TABLE_PREV = "positive_constraint_literals_prevention_nonact"

# ────────────────────────────────────────────────────────────────
# BRANCH CONFIG (NONACT outputs + artifacts)
# ────────────────────────────────────────────────────────────────
BRANCHES = {
    "main": {
        "poslit_table": TABLE_MAIN,
        "caa_table": "concept_accepted_alternatives_nonact",
        "expanded_table": "positive_constraint_alternatives_expanded_nonact",
        "artifact_dir": Path("mbench/hop_decider_logs_nonact"),
    },
    "prevention": {
        "poslit_table": TABLE_PREV,
        "caa_table": "concept_accepted_alternatives_prevention_nonact",
        "expanded_table": "positive_constraint_alternatives_expanded_prevention_nonact",
        "artifact_dir": Path("mbench/hop_decider_logs_prevention_nonact"),
    },
}

_ID_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _safe_ident(x: str, what: str) -> str:
    x = (x or "").strip()
    if not _ID_RX.match(x):
        raise ValueError(f"Unsafe {what}: {x!r}")
    return x

# ────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────
BATCH_SIZE      = 10
WORKERS         = 24
RATE_PER_MIN    = 60
LINEAGE_WORKERS = 1
LOG_LEVEL       = os.getenv("HOP_DECIDER_LOG_LEVEL", "INFO")

# ────────────────────────────────────────────────────────────────
# Logging (per-branch logger; avoids basicConfig collisions)
# ────────────────────────────────────────────────────────────────
def _make_logger(name: str, artifact_dir: Path) -> logging.Logger:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    # reset handlers each time to avoid duplicates across runs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(artifact_dir / "hop_policy_decider.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ────────────────────────────────────────────────────────────────
# Engine wiring (lazy)
# ────────────────────────────────────────────────────────────────
def _make_engine():
    endpoint = os.environ.get("OPENAI_ENDPOINT", "").strip().lower()
    if endpoint.endswith("gpt-5"):
        from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
        model_name = "gpt-5"
    elif endpoint.endswith("gpt-4o"):
        from smt_core.inference_engine import AzureInferenceEngine  # type: ignore
        model_name = "gpt-4o"
    elif endpoint.endswith("gpt-4.1"):
        from smt_core.inference_engine import AzureInferenceEngine  # type: ignore
        model_name = "gpt-4.1"
    else:
        raise EnvironmentError("OPENAI_ENDPOINT must end with 'gpt-4o', 'gpt-4.1', or 'gpt-5'.")
    engine = AzureInferenceEngine(
        endpoint=os.environ.get("OPENAI_ENDPOINT",""),
        api_key_env_var="OPENAI_API_KEY",
        model_name=model_name
    )
    return engine, model_name

# ────────────────────────────────────────────────────────────────
# Prompt loader
# ────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE_CACHE: Optional[str] = None

def _default_prompt_path() -> Path:
    return (Path(__file__).resolve().parent / "prompts" / "decider.prompt")

def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE_CACHE
    if _PROMPT_TEMPLATE_CACHE is not None:
        return _PROMPT_TEMPLATE_CACHE

    env_path = os.environ.get("DECIDER_PROMPT_PATH")
    path = Path(env_path).expanduser() if env_path else _default_prompt_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found at {path}. "
            "Create it (or set DECIDER_PROMPT_PATH). The file must include a {cases} token."
        )
    txt = path.read_text(encoding="utf-8")
    if "{cases}" not in txt:
        raise ValueError(f"Prompt template {path} must include a {{cases}} placeholder.")
    _PROMPT_TEMPLATE_CACHE = txt
    return txt

# ────────────────────────────────────────────────────────────────
# Snowstorm client + caches
# ────────────────────────────────────────────────────────────────
SNOWSTORM_BASE   = os.getenv("SNOWSTORM_BASE", "http://localhost:8080").rstrip("/")
SNOWSTORM_BRANCH = os.getenv("SNOWSTORM_BRANCH", "MAIN")
SNOWSTORM_FORM   = os.getenv("SNOWSTORM_FORM", "inferred")
HTTP_TIMEOUT_S   = float(os.getenv("SNOW_TIMEOUT", "6.0"))

_PARENTS_CACHE: Dict[str, List[dict]]    = {}
_ANCESTORS_CACHE: Dict[str, List[dict]]  = {}
_LABEL_CACHE: Dict[str, str]             = {}
_PARENTS_IDS_CACHE: Dict[str, List[str]] = {}
_FSN_CACHE: Dict[str, str]               = {}

def _http_get_json(url: str) -> dict | list | None:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=HTTP_TIMEOUT_S)
        if r.status_code == 200:
            return r.json()
    except Exception:
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as res:
                if res.status == 200:
                    return _json.loads(res.read().decode("utf-8"))
        except Exception:
            return None
    return None

def _concept_json(cid: str) -> Optional[dict]:
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{cid}"
    js = _http_get_json(url)
    return js if isinstance(js, dict) else None

def _fsn_for(cid: str) -> Optional[str]:
    if cid in _FSN_CACHE:
        return _FSN_CACHE[cid]
    js = _concept_json(cid)
    fsn = (js or {}).get("fsn", {}) or {}
    term = fsn.get("term")
    if term:
        _FSN_CACHE[cid] = term
        return term
    return None

def _label_for(cid: str) -> Optional[str]:
    if not cid:
        return None
    if cid in _LABEL_CACHE:
        return _LABEL_CACHE[cid]
    js = _concept_json(cid)
    if isinstance(js, dict):
        pt  = (js.get("pt") or {}).get("term")
        fsn = (js.get("fsn") or {}).get("term")
        lab = js.get("preferredTerm") or pt or fsn or js.get("term")
        if lab:
            _LABEL_CACHE[cid] = lab
            return lab
    return None

def _enrich_label(raw_mini: dict) -> dict:
    cid  = str(raw_mini.get("conceptId") or raw_mini.get("id") or "").strip()
    pt   = (raw_mini.get("pt") or {}).get("term")
    fsn  = (raw_mini.get("fsn") or {}).get("term")
    pref = raw_mini.get("preferredTerm") or raw_mini.get("preferred_term") or raw_mini.get("term") or pt or fsn
    row  = {"conceptId": cid, "preferred_term": pref, "fully_specified_name": fsn}
    if cid and not row.get("preferred_term"):
        lab = _label_for(cid)
        if lab:
            row["preferred_term"] = lab
    return row

def parents_cached(cid: str) -> List[dict]:
    if cid in _PARENTS_CACHE:
        return _PARENTS_CACHE[cid]
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{cid}/parents?form={SNOWSTORM_FORM}"
    js  = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"):
                out.append(row)
    _PARENTS_CACHE[cid] = out
    return out

def ancestors_cached(cid: str) -> List[dict]:
    if cid in _ANCESTORS_CACHE:
        return _ANCESTORS_CACHE[cid]
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{cid}/ancestors?form={SNOWSTORM_FORM}"
    js  = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"):
                out.append(row)
    _ANCESTORS_CACHE[cid] = out
    return out

def _parent_ids(cid: str) -> List[str]:
    if cid in _PARENTS_IDS_CACHE:
        return _PARENTS_IDS_CACHE[cid]
    ids = [str(m.get("conceptId")) for m in (parents_cached(cid) or []) if m.get("conceptId")]
    _PARENTS_IDS_CACHE[cid] = ids
    return ids

def compute_upward_hops(start_cid: str, target_ids: List[str]) -> Dict[str,int]:
    targets = set(target_ids); found: Dict[str,int] = {}; seen = {start_cid}
    q: deque[Tuple[str,int]] = deque([(start_cid,0)])
    while q and len(found) < len(targets):
        cur, h = q.popleft()
        for p in _parent_ids(cur):
            if p in seen:
                continue
            seen.add(p); hop = h+1
            if p in targets and p not in found:
                found[p] = hop
            q.append((p, hop))
    return found

# ────────────────────────────────────────────────────────────────
# Helpers: schema detection
# ────────────────────────────────────────────────────────────────
def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    table = _safe_ident(table, "table")
    cur = conn.cursor()
    try:
        rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return False
    for r in rows:
        if len(r) > 1 and str(r[1]) == col:
            return True
    return False

def _join_key_expr(conn: sqlite3.Connection, poslit_table: str) -> str:
    if _table_has_column(conn, poslit_table, "orig_var_name"):
        return "COALESCE(pl.orig_var_name, pl.var_name)"
    return "pl.var_name"

# ────────────────────────────────────────────────────────────────
# DDL (table-aware)
# ────────────────────────────────────────────────────────────────
def _ddl_caa(caa_table: str) -> str:
    caa_table = _safe_ident(caa_table, "caa_table")
    return f"""
    DROP TABLE IF EXISTS {caa_table};
    CREATE TABLE {caa_table} (
      concept_id     TEXT NOT NULL,
      alt_concept_id TEXT NOT NULL,
      hop            INTEGER NOT NULL,
      alt_label      TEXT,
      reason         TEXT,
      decided_at     TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (concept_id, alt_concept_id)
    );
    CREATE INDEX IF NOT EXISTS idx_{caa_table}_concept_hop ON {caa_table}(concept_id, hop);
    """

def _open(db_path: str, caa_table: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(_ddl_caa(caa_table))
    return conn

# Concepts iterator (positive_constraint_literals → predicate_to_concept)
def _iter_concepts(conn: sqlite3.Connection, poslit_table: str, include_assumed: bool) -> Iterable[str]:
    poslit_table = _safe_ident(poslit_table, "poslit_table")
    cur = conn.cursor()
    join_key = _join_key_expr(conn, poslit_table)
    if include_assumed:
        q = f"""
            SELECT DISTINCT v2.concept_id
            FROM {poslit_table} pl
            JOIN predicate_to_concept v2 ON v2.var_name = {join_key}
            WHERE pl.kind='inclusion' OR pl.variant='assumed'
        """
    else:
        q = f"""
            SELECT DISTINCT v2.concept_id
            FROM {poslit_table} pl
            JOIN predicate_to_concept v2 ON v2.var_name = {join_key}
            WHERE pl.kind='inclusion'
        """
    for (cid,) in cur.execute(q):
        if cid:
            yield str(cid)

# ────────────────────────────────────────────────────────────────
# Lineage & case structures
# ────────────────────────────────────────────────────────────────
@dataclass
class ConceptCase:
    concept_id: str
    lineage: Dict[str, List[Dict[str,str]]]  # hop -> [{"concept_id": "...", "label": "..."}]

def _lineage_grouped_by_hop(cid: str) -> Dict[str, List[Dict[str,str]]]:
    minis_par = parents_cached(cid) or []
    minis_anc = ancestors_cached(cid) or []
    minis_all = minis_par + minis_anc

    ids = [str(m.get("conceptId")) for m in minis_all if m.get("conceptId")]
    hop_map = compute_upward_hops(cid, ids)

    grouped_raw: Dict[str, Dict[str, str]] = {}
    for m in minis_all:
        aid = str(m.get("conceptId") or "")
        if not aid:
            continue
        h = hop_map.get(aid)
        if not isinstance(h, int) or h <= 0:
            continue
        label = (m.get("preferred_term") or m.get("fully_specified_name") or "").strip()
        bucket = grouped_raw.setdefault(str(h), {})
        if aid not in bucket:
            bucket[aid] = label

    grouped: Dict[str, List[Dict[str,str]]] = {}
    for hk, mp in grouped_raw.items():
        grouped[hk] = sorted(
            [{"concept_id": k, "label": v} for k, v in mp.items()],
            key=lambda x: ((x.get("label") or "").lower(), x.get("concept_id") or "")
        )
    return grouped

def _collect_case(cid: str) -> ConceptCase:
    return ConceptCase(concept_id=cid, lineage=_lineage_grouped_by_hop(cid))

def _decide_all_ancestors(conn: sqlite3.Connection, caa_table: str, cids: List[str]) -> int:
    caa_table = _safe_ident(caa_table, "caa_table")
    n_upserts = 0
    with conn:
        for base_cid in cids:
            lineage = _lineage_grouped_by_hop(base_cid) or {}
            for hk, lst in lineage.items():
                try:
                    hop = int(hk)
                except Exception:
                    continue
                if hop <= 0:
                    continue
                for m in lst:
                    alt_cid = str(m.get("concept_id") or "").strip()
                    if not alt_cid or alt_cid == base_cid:
                        continue
                    alt_label = (m.get("label") or None)
                    conn.execute(
                        f"""
                        INSERT INTO {caa_table}
                          (concept_id, alt_concept_id, hop, alt_label, reason)
                        VALUES (?,?,?,?,?)
                        ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
                          hop       = excluded.hop,
                          alt_label = excluded.alt_label,
                          reason    = excluded.reason,
                          decided_at= datetime('now')
                        """,
                        (base_cid, alt_cid, hop, alt_label, "all-ancestors")
                    )
                    n_upserts += 1
    return n_upserts

# ────────────────────────────────────────────────────────────────
# Parsing & engine glue
# ────────────────────────────────────────────────────────────────
def _unwrap_code_fence(txt: str) -> str:
    s = (txt or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    return m.group(1).strip() if m else s

_IDX_RX = re.compile(r"^\s*(\d+)\s*\.\s*([a-zA-Z])\s*$")
def _normalize_idx(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = _IDX_RX.match(s)
    if not m:
        return None
    return f"{int(m.group(1))}.{m.group(2).lower()}"

def _parse_keep_list_order_agnostic(raw: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    try:
        data = json.loads(_unwrap_code_fence(raw))
    except json.JSONDecodeError:
        logger.warning("[parse] keep-list JSON decode failed; returning empty")
        return []
    if not isinstance(data, list):
        logger.warning("[parse] keep-list expected list; returning empty")
        return []
    out: List[Dict[str, Any]] = []
    for ent in data:
        cidtok = (ent.get("case_id") or "").strip()
        keeps = []
        for k in (ent.get("keep") or []):
            if not isinstance(k, dict):
                continue
            keeps.append({
                "alt_idx": _normalize_idx(k.get("alt_idx")),
                "reason": (k.get("reason") or "").strip() or "keep-list",
            })
        out.append({"case_id": cidtok, "keep": keeps})
    return out

def _engine_call(engine, prompt: str, logger: logging.Logger) -> str:
    t0 = time.time()
    out = engine(prompt)
    dt = (time.time() - t0) * 1000
    try:
        first = out[0]
        s = str(first[0]) if isinstance(first, (list, tuple)) else str(first)
    except Exception:
        s = str(out)
    logger.info(f"[engine] response in {dt:.1f} ms (chars={len(s)})")
    return s

class _RateLimiter:
    def __init__(self, per_min: int):
        self.capacity = max(per_min, 1)
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last_refill = time.time()
    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    add = (elapsed * self.capacity) / 60.0
                    if add >= 1:
                        self.tokens = min(self.capacity, self.tokens + int(add))
                        self.last_refill = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
            time.sleep(0.05)

# ────────────────────────────────────────────────────────────────
# Var naming helpers (timeframe stripping)
# ────────────────────────────────────────────────────────────────
_TIMEFRAME_TOKEN_RX = re.compile(
    r"(?:(?<=^)|(?<=_))"
    r"(?:(?:now|inthehistory|inthefuture)|"
    r"(?:inthepast|inthefuture)\d+(?:minute(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?)|"
    r"foradurationof\d+(?:minute(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?))"
    r"(?=(?:_|$))"
)

def _semantic_tag(cid: str) -> Optional[str]:
    fsn = _fsn_for(cid) or ""
    m = re.search(r"\(([^)]+)\)\s*$", fsn)
    return (m.group(1).strip().lower() if m else None) or None

def _slug_entity(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _choose_prefix(cid: str, label: str) -> str:
    tag = (_semantic_tag(cid) or "").lower()
    lab = (label or "").lower()
    if "suspicion" in lab or "suspected" in lab:
        return "patient_has_suspicion_of"
    if tag in {"disorder", "disease"}:
        return "patient_has_diagnosis_of"
    if tag in {"finding", "clinical finding"}:
        return "patient_has_finding_of"
    if tag == "procedure":
        return "patient_has_undergone"
    if tag in {"product", "medicinal product", "pharmaceutical / biologic product"}:
        return "patient_is_taking"
    if tag == "substance":
        return "patient_is_exposed_to"
    if any(w in lab for w in ("carcinoma", "neoplasm", "malignant", "infection", "pneumocystis")):
        return "patient_has_diagnosis_of"
    return "patient_has_finding_of"

def _build_var_for_concept(cid: str, timeframe: str, *, prevention: bool = False) -> str:
    lab = _label_for(cid) or ""
    entity = _slug_entity(lab) or f"concept_{cid}"
    tf = timeframe if timeframe else "now"
    if prevention:
        return f"patient_wants_to_prevent_{entity}_{tf}"
    prefix = _choose_prefix(cid, lab)
    return f"{prefix}_{entity}_{tf}"

def _strip_timeframe_once(stem: str) -> str:
    if not stem:
        return stem
    out = _TIMEFRAME_TOKEN_RX.sub("", stem, count=1)
    out = re.sub(r"_+", "_", out).strip("_")
    return out

# ────────────────────────────────────────────────────────────────
# Expanded table builder (table-aware)
# ────────────────────────────────────────────────────────────────
def _create_poslit_expanded(conn: sqlite3.Connection, expanded_table: str) -> None:
    expanded_table = _safe_ident(expanded_table, "expanded_table")
    conn.executescript(f"""
    DROP TABLE IF EXISTS {expanded_table};
    CREATE TABLE {expanded_table} (
      nct_id        TEXT,
      kind          TEXT CHECK (kind in ('inclusion','exclusion')),
      variant       TEXT CHECK (variant in ('main','assumed')),
      direction     TEXT CHECK (direction in ('sat_on_true','unsat_on_true')),
      var_name      TEXT,
      timeframe     TEXT,
      alt_concept_id  TEXT NOT NULL,
      hop             INTEGER NOT NULL,
      reason          TEXT,
      decided_at      TEXT,
      lifted_var      TEXT,
      base_var        TEXT,
      base_var_stem   TEXT,
      lifted_var_stem TEXT,
      PRIMARY KEY (nct_id,kind,variant,direction,var_name,alt_concept_id)
    );
    CREATE INDEX IF NOT EXISTS idx_{expanded_table}_join ON {expanded_table}(alt_concept_id, timeframe);
    """)

def _fetch_alt_map(conn: sqlite3.Connection, caa_table: str) -> Dict[str, List[sqlite3.Row]]:
    caa_table = _safe_ident(caa_table, "caa_table")
    q = f"""
    SELECT concept_id, alt_concept_id, hop, COALESCE(reason,'') AS reason,
           decided_at
    FROM {caa_table}
    ORDER BY concept_id, hop, alt_concept_id
    """
    mp: Dict[str, List[sqlite3.Row]] = {}
    for r in conn.execute(q):
        mp.setdefault(str(r["concept_id"]), []).append(r)
    return mp

def _iter_poslits(conn: sqlite3.Connection, poslit_table: str, include_assumed: bool) -> List[sqlite3.Row]:
    poslit_table = _safe_ident(poslit_table, "poslit_table")
    where = "pl.kind='inclusion'" + (" OR pl.variant='assumed'" if include_assumed else "")
    join_key = _join_key_expr(conn, poslit_table)

    q = f"""
      SELECT pl.nct_id, pl.kind, pl.variant, pl.direction, pl.var_name,
             COALESCE(pl.timeframe,'now') AS timeframe,
             COALESCE(pl.base_var, pl.var_name) AS base_var,
             v2.concept_id AS base_concept_id
      FROM {poslit_table} pl
      LEFT JOIN predicate_to_concept v2 ON v2.var_name = {join_key}
      WHERE {where}
      ORDER BY pl.nct_id, pl.kind, pl.variant, pl.var_name
    """
    return list(conn.execute(q))

def _rebuild_expanded(
    conn: sqlite3.Connection,
    artifact_dir: Path,
    *,
    include_assumed: bool,
    poslit_table: str,
    caa_table: str,
    expanded_table: str,
    prevention: bool,
) -> None:
    _create_poslit_expanded(conn, expanded_table)
    rows = _iter_poslits(conn, poslit_table, include_assumed=include_assumed)
    alt_map = _fetch_alt_map(conn, caa_table)
    out_rows: List[Dict[str, Any]] = []

    expanded_table = _safe_ident(expanded_table, "expanded_table")
    cur = conn.cursor()
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S")

    for r in rows:
        nct_id     = str(r["nct_id"] or "")
        kind       = str(r["kind"] or "")
        variant    = str(r["variant"] or "")
        direction  = str(r["direction"] or "")
        var_name   = str(r["var_name"] or "")
        timeframe  = str(r["timeframe"] or "now")
        base_cid   = str(r["base_concept_id"] or "")
        base_var   = str(r["base_var"] or var_name)

        if not base_cid:
            continue

        base_var_stem = _strip_timeframe_once(base_var)

        # self
        lifted_var_self = _build_var_for_concept(base_cid, timeframe, prevention=prevention)
        lifted_var_self_stem = _strip_timeframe_once(lifted_var_self)

        cur.execute(f"""
            INSERT OR REPLACE INTO {expanded_table}
              (nct_id,kind,variant,direction,var_name,timeframe,
               alt_concept_id,hop,reason,decided_at,
               lifted_var,base_var,base_var_stem,lifted_var_stem)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (nct_id, kind, variant, direction, var_name, timeframe,
              base_cid, 0, "self", now_iso,
              lifted_var_self, base_var, base_var_stem, lifted_var_self_stem))

        out_rows.append({
            "nct_id": nct_id, "kind": kind, "variant": variant, "direction": direction,
            "var_name": var_name, "timeframe": timeframe,
            "alt_concept_id": base_cid, "hop": 0, "reason": "self", "decided_at": now_iso,
            "lifted_var": lifted_var_self, "base_var": base_var,
            "base_var_stem": base_var_stem, "lifted_var_stem": lifted_var_self_stem
        })

        # accepted alternatives
        for alt in alt_map.get(base_cid, []):
            alt_cid = str(alt["alt_concept_id"])
            hop     = int(alt["hop"])
            reason  = str(alt["reason"] or "")
            decided = str(alt["decided_at"] or now_iso)

            lifted_var = _build_var_for_concept(alt_cid, timeframe, prevention=prevention)
            lifted_var_stem = _strip_timeframe_once(lifted_var)

            cur.execute(f"""
                INSERT OR REPLACE INTO {expanded_table}
                  (nct_id,kind,variant,direction,var_name,timeframe,
                   alt_concept_id,hop,reason,decided_at,
                   lifted_var,base_var,base_var_stem,lifted_var_stem)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (nct_id, kind, variant, direction, var_name, timeframe,
                  alt_cid, hop, reason, decided,
                  lifted_var, base_var, base_var_stem, lifted_var_stem))

            out_rows.append({
                "nct_id": nct_id, "kind": kind, "variant": variant, "direction": direction,
                "var_name": var_name, "timeframe": timeframe,
                "alt_concept_id": alt_cid, "hop": hop, "reason": reason, "decided_at": decided,
                "lifted_var": lifted_var, "base_var": base_var,
                "base_var_stem": base_var_stem, "lifted_var_stem": lifted_var_stem
            })

    conn.commit()
    _write_json(artifact_dir / "expanded_rows.json", out_rows)
    csv_path = artifact_dir / "expanded_rows.csv"
    if out_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

# ────────────────────────────────────────────────────────────────
# Prompt cases + core execution
# ────────────────────────────────────────────────────────────────
def _alpha(n: int) -> str:
    return chr(ord('a') + n)

def _run_one_branch(
    *,
    db_path: str,
    branch: str,
    include_assumed: bool,
    keep_all_ancestors: bool,
    use_llm: bool,
) -> None:
    cfg = BRANCHES[branch]
    poslit_table = cfg["poslit_table"]
    caa_table = cfg["caa_table"]
    expanded_table = cfg["expanded_table"]
    artifact_dir = cfg["artifact_dir"]
    prevention = (branch == "prevention")

    # fresh artifacts each run (branch-specific dirs, so no cross-script conflicts)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger = _make_logger(f"keep_list_decider_nonact.{branch}", artifact_dir)
    logger.info(f"[start:{branch}] DB={db_path}")
    logger.info(f"[tables:{branch}] poslit={poslit_table} caa={caa_table} expanded={expanded_table}")
    logger.info(f"[opts:{branch}] include_assumed={include_assumed} keep_all_ancestors={keep_all_ancestors} use_llm={use_llm}")

    conn = _open(db_path, caa_table)

    cids = list(_iter_concepts(conn, poslit_table, include_assumed=include_assumed))
    logger.info(f"[concepts:{branch}] total={len(cids)}")

    meta = {
        "db": db_path,
        "branch": branch,
        "poslit_table": poslit_table,
        "caa_table": caa_table,
        "expanded_table": expanded_table,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "rate_per_min": RATE_PER_MIN,
        "lineage_workers": LINEAGE_WORKERS,
        "include_assumed": include_assumed,
        "keep_all_ancestors": keep_all_ancestors,
        "use_llm": use_llm,
        "OPENAI_ENDPOINT": os.environ.get("OPENAI_ENDPOINT", ""),
        "SNOWSTORM_BASE": SNOWSTORM_BASE,
        "SNOWSTORM_BRANCH": SNOWSTORM_BRANCH,
        "SNOWSTORM_FORM": SNOWSTORM_FORM,
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not cids:
        logger.warning(f"[concepts:{branch}] nothing to decide; still materializing empty expanded table")
        _rebuild_expanded(conn, artifact_dir, include_assumed=include_assumed,
                          poslit_table=poslit_table, caa_table=caa_table, expanded_table=expanded_table,
                          prevention=prevention)
        conn.close()
        logger.info(f"[done:{branch}]")
        return

    if keep_all_ancestors:
        up = _decide_all_ancestors(conn, caa_table, cids)
        logger.info(f"[db:{branch}] all-ancestors upserts={up}")
    else:
        if not use_llm:
            raise RuntimeError("keep_all_ancestors is False but use_llm is also False. Choose one mode.")
        engine, model_name = _make_engine()
        logger.info(f"[llm:{branch}] model={model_name}")

        batches = [cids[i:i+BATCH_SIZE] for i in range(0, len(cids), BATCH_SIZE)]
        limiter = _RateLimiter(RATE_PER_MIN)

        def _prepare_batch(idx: int, ids: List[str]) -> Tuple[Path, str]:
            bdir = artifact_dir / f"batch_{idx+1:04d}"
            bdir.mkdir(parents=True, exist_ok=True)

            if LINEAGE_WORKERS > 1:
                cases: List[ConceptCase] = []
                with ThreadPoolExecutor(max_workers=LINEAGE_WORKERS) as ex_lin:
                    futs = [ex_lin.submit(_collect_case, cid) for cid in ids]
                    for fu in as_completed(futs):
                        cases.append(fu.result())
            else:
                cases = [_collect_case(cid) for cid in ids]

            compact: List[Dict[str, Any]] = []
            keymaps_by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
            base_by_case: Dict[str, str] = {}

            for i, c in enumerate(cases):
                case_id_token = f"c{idx+1:04d}_{i:04d}"
                root_label = _label_for(c.concept_id) or ""
                ancestors_by_hop: Dict[str, List[Dict[str, str]]] = {}
                keymap: Dict[str, Dict[str, Any]] = {}

                for hk in sorted((k for k in c.lineage.keys() if k.isdigit() and int(k) <= 6), key=lambda x: int(x)):
                    h = int(hk)
                    entries = []
                    for j, m in enumerate(c.lineage[hk]):
                        aid = str(m.get("concept_id") or "").strip()
                        lab = (m.get("label") or "").strip()
                        if not aid:
                            continue
                        idx_str = f"{h}.{_alpha(j)}"
                        keymap[idx_str] = {"concept_id": aid, "hop": h, "label": lab}
                        entries.append({"idx": idx_str, "label": lab})
                    if entries:
                        ancestors_by_hop[str(h)] = entries

                compact.append({"case_id": case_id_token, "root_label": root_label, "ancestors_by_hop": ancestors_by_hop})
                keymaps_by_case[case_id_token] = keymap
                base_by_case[case_id_token] = c.concept_id

            _write_json(bdir / "keymaps_by_case.json", keymaps_by_case)
            _write_json(bdir / "base_by_case.json", base_by_case)

            tmpl = _load_prompt_template()
            prompt = tmpl.replace("{cases}", json.dumps(compact, ensure_ascii=False, indent=2))
            _write_json(bdir / "prompt.json", {"batch_index": idx+1, "prompt_chars": len(prompt), "prompt": prompt})
            return bdir, prompt

        prepared = [_prepare_batch(i, ids) for i, ids in enumerate(batches)]

        def _call_engine(prompt: str) -> str:
            limiter.acquire()
            tries, delay = 0, 0.6
            while True:
                try:
                    return _engine_call(engine, prompt, logger)
                except Exception:
                    tries += 1
                    if tries >= 5:
                        raise
                    time.sleep(delay + random.uniform(0, 0.3))
                    delay *= 2.0

        caa_table_safe = _safe_ident(caa_table, "caa_table")

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            fut_to_meta = {ex.submit(_call_engine, prompt): (bdir, prompt)
                           for (bdir, prompt) in prepared}
            for fu in as_completed(fut_to_meta):
                bdir, _prompt = fut_to_meta[fu]
                raw = fu.result()
                (bdir / "response.txt").write_text(str(raw), encoding="utf-8")
                parsed = _parse_keep_list_order_agnostic(raw, logger)
                _write_json(bdir / "response_parsed.json", parsed)

                keymaps_by_case = json.loads((bdir / "keymaps_by_case.json").read_text(encoding="utf-8"))
                base_by_case    = json.loads((bdir / "base_by_case.json").read_text(encoding="utf-8"))

                up = 0
                with conn:
                    for ent in parsed:
                        case_id = ent.get("case_id") or ""
                        if case_id not in keymaps_by_case or case_id not in base_by_case:
                            continue
                        base_cid = base_by_case[case_id]
                        km = keymaps_by_case[case_id]
                        seen: set[str] = set()
                        for k in (ent.get("keep") or []):
                            idx = _normalize_idx(k.get("alt_idx"))
                            if not idx or idx not in km:
                                continue
                            hit = km[idx]
                            alt_cid = str(hit.get("concept_id") or "")
                            if not alt_cid or alt_cid in seen:
                                continue
                            seen.add(alt_cid)
                            hop = int(hit.get("hop") or 0)
                            alt_label = hit.get("label")
                            reason = k.get("reason") or "keep-list"
                            conn.execute(
                                f"""
                                INSERT INTO {caa_table_safe}
                                  (concept_id, alt_concept_id, hop, alt_label, reason)
                                VALUES (?,?,?,?,?)
                                ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
                                  hop=excluded.hop,
                                  alt_label=excluded.alt_label,
                                  reason=excluded.reason,
                                  decided_at=datetime('now')
                                """,
                                (base_cid, alt_cid, hop, alt_label, reason)
                            )
                            up += 1
                logger.info(f"[db:{branch}] {bdir.name}: upserts={up}")

    logger.info(f"[expand:{branch}] materializing expanded table...")
    _rebuild_expanded(conn, artifact_dir,
                      include_assumed=include_assumed,
                      poslit_table=poslit_table,
                      caa_table=caa_table,
                      expanded_table=expanded_table,
                      prevention=prevention)
    conn.close()
    logger.info(f"[done:{branch}]")

def main():
    ap = argparse.ArgumentParser(
        description="NONACT keep-list decider over positive_constraint_literals_*_nonact (runs BOTH main + prevention by default)."
    )
    ap.add_argument("--db", default="../../../build/trial.db", help="SQLite DB path (e.g., ../../build/trial.db)")
    ap.add_argument("--only", choices=["both", "main", "prevention"], default="both",
                    help="Which branch(es) to run (default: both)")

    # Proper boolean flags with defaults:
    ap.add_argument("--include-assumed", action=argparse.BooleanOptionalAction, default=True,
                    help="Include rows where variant='assumed' (default: True)")
    ap.add_argument("--keep-all-ancestors", action=argparse.BooleanOptionalAction, default=False,
                    help="If True, skip LLM calls and keep all ancestors (default: True)")
    ap.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=False,
                    help="Use LLM keep-list (only used if --keep-all-ancestors is False)")

    args = ap.parse_args()
    db_path = str(Path(args.db).expanduser().resolve())

    branches = ["main", "prevention"] if args.only == "both" else [args.only]
    for br in branches:
        _run_one_branch(
            db_path=db_path,
            branch=br,
            include_assumed=bool(args.include_assumed),
            keep_all_ancestors=bool(args.keep_all_ancestors),
            use_llm=bool(args.use_llm),
        )

if __name__ == "__main__":
    main()