#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disease_ontology_lifter.py — keep-list–only acceptance + concurrent decider (NAMESPACED)
WITH TABLE SUFFIX SUPPORT (e.g., _nonact)

Table naming (base):
  - ns = "main" (or empty): base tables
      disease_predicate_concepts
      disease_constraint_lifted_atoms
      disease_constraint_alternatives
      concept_accepted_alternatives
  - ns in {"prevention","prevent","prev"}: suffix "_prevent"
      disease_predicate_concepts_prevent
      ...
  - otherwise: suffix "_<ns>"

NEW:
  - --table-suffix "_nonact" appends to ALL tables produced/used by this script:
      disease_predicate_concepts_nonact
      disease_predicate_concepts_prevent_nonact
      ...

Legacy cleanup:
  - Drops old v1 namespaced tables with __main/__prevent* each run.
"""

from __future__ import annotations
import argparse
import os, re, sqlite3, sys, json, time, shutil, random, threading
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ────────────────────────── normalization ──────────────────────────
_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")  # strip trailing “ ( … )” semantic tag

def _strip_tag(term: Optional[str]) -> str:
    return _TAG_RE.sub("", term or "").strip()

def _to_var(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

import re as _re
def _parse_timeframe_from_varname(vn: str) -> str:
    m = _re.search(r"_(inthehistory|now|inthepast[0-9a-z]+|[0-9a-z]+ago|[0-9a-z]+slater)(?:@@|$)", vn)
    return m.group(1) if m else ""

def _compose_var_name(form: str, timeframe: str, *, prefix: str = "patient_has_finding_of") -> str:
    name = f"{prefix}_{form}"
    if timeframe:
        name += f"_{timeframe}"
    return _re.sub(r"_+", "_", name).strip("_")

_TIMEFRAME_TAIL_RE = re.compile(r"_(inthehistory|now|inthepast[0-9a-z]+|[0-9a-z]+ago|[0-9a-z]+slater)$", re.I)
def _strip_timeframe_segment(vn: str) -> str:
    return _TIMEFRAME_TAIL_RE.sub("", vn or "")

# ────────────────────────── Snowstorm client (cached) ──────────────────────────
SNOWSTORM_BASE   = os.getenv("SNOWSTORM_BASE", "http://localhost:8080").rstrip("/")
SNOWSTORM_BRANCH = os.getenv("SNOWSTORM_BRANCH", "MAIN")
SNOWSTORM_FORM   = os.getenv("SNOWSTORM_FORM", "inferred")  # inferred | stated
HTTP_TIMEOUT_S   = float(os.getenv("SNOW_TIMEOUT", "6.0"))

_PARENTS_CACHE: Dict[str, List[dict]]    = {}
_ANCESTORS_CACHE: Dict[str, List[dict]]  = {}
_LABEL_CACHE: Dict[str, str]             = {}
_PARENTS_IDS_CACHE: Dict[str, List[str]] = {}

def _http_get_json(url: str):
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

def _mini_to_row(raw: dict) -> dict:
    cid = str(raw.get("conceptId") or raw.get("id") or "").strip()
    pt  = (raw.get("pt") or {}).get("term")
    fsn = (raw.get("fsn") or {}).get("term")
    pref = raw.get("preferredTerm") or raw.get("preferred_term") or raw.get("term") or pt or fsn
    return {"conceptId": cid, "preferred_term": pref, "fully_specified_name": fsn}

def _label_for(cid: str) -> Optional[str]:
    if not cid:
        return None
    if cid in _LABEL_CACHE:
        return _LABEL_CACHE[cid]
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{cid}"
    js = _http_get_json(url)
    if isinstance(js, dict):
        pt  = (js.get("pt") or {}).get("term")
        fsn = (js.get("fsn") or {}).get("term")
        lab = js.get("preferredTerm") or pt or fsn or js.get("term")
        if lab:
            _LABEL_CACHE[cid] = lab
            return lab
    return None

def _enrich_label(raw_mini: dict) -> dict:
    row = dict(_mini_to_row(raw_mini) or {})
    cid = row.get("conceptId")
    if cid and not row.get("preferred_term"):
        lab = _label_for(cid)
        if lab:
            row["preferred_term"] = lab
    return row

def parents_cached(concept_id: str) -> List[dict]:
    if concept_id in _PARENTS_CACHE:
        return _PARENTS_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{concept_id}/parents?form={SNOWSTORM_FORM}"
    js = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"):
                out.append(row)
    _PARENTS_CACHE[concept_id] = out
    return out

def ancestors_cached(concept_id: str) -> List[dict]:
    if concept_id in _ANCESTORS_CACHE:
        return _ANCESTORS_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{concept_id}/ancestors?form={SNOWSTORM_FORM}"
    js = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"):
                out.append(row)
    _ANCESTORS_CACHE[concept_id] = out
    return out

def _parent_ids(cid: str) -> List[str]:
    if cid in _PARENTS_IDS_CACHE:
        return _PARENTS_IDS_CACHE[cid]
    ids = [str(m.get("conceptId")) for m in (parents_cached(cid) or []) if m.get("conceptId")]
    _PARENTS_IDS_CACHE[cid] = ids
    return ids

def compute_upward_hops(start_cid: str, target_ids: List[str]) -> Dict[str, int]:
    targets = set(target_ids)
    found: Dict[str, int] = {}
    seen = {start_cid}
    q: deque[Tuple[str,int]] = deque([(start_cid, 0)])
    while q and len(found) < len(targets):
        cur, h = q.popleft()
        for p in _parent_ids(cur):
            if p in seen:
                continue
            seen.add(p)
            hop = h + 1
            if p in targets and p not in found:
                found[p] = hop
            q.append((p, hop))
    return found

# ────────────────────────── safe SQL identifiers ──────────────────────────
_ID_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
def _safe_ident(x: str, what: str) -> str:
    x = (x or "").strip()
    if not _ID_RX.match(x):
        raise ValueError(f"Unsafe {what}: {x!r}")
    return x

# ────────────────────────── table suffix support ──────────────────────────
TABLE_SUFFIX = ""  # set from CLI

def _norm_suffix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if s.startswith("_") else "_" + s

def _apply_suffix(tbl: str) -> str:
    suf = _norm_suffix(TABLE_SUFFIX)
    if not suf:
        return tbl
    if tbl.endswith(suf):
        return tbl
    return tbl + suf

def _is_prevention_ns(ns: str) -> bool:
    return (ns or "").lower() in {"prevention", "prevent", "prev"}

def _t(ns: str, base: str) -> str:
    ns_l = (ns or "").strip().lower()
    if ns_l in {"", "main"}:
        out = base
    elif ns_l in {"prevention", "prevent", "prev"}:
        out = f"{base}_prevent"
    else:
        out = f"{base}_{ns_l}"
    return _apply_suffix(out)

# ────────────────────────── legacy table cleanup ──────────────────────────
_LEGACY_NS_SUFFIXES = ["__main", "__prevention", "__prevent", "__prev"]
_LEGACY_BASE_TABLES = [
    "disease_predicate_concepts",
    "disease_constraint_lifted_atoms",
    "disease_constraint_alternatives",
    "concept_accepted_alternatives",
]

def drop_legacy_namespaced_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for base in _LEGACY_BASE_TABLES:
        for suf in _LEGACY_NS_SUFFIXES:
            tbl = f"{base}{suf}"
            cur.execute(f'DROP TABLE IF EXISTS "{tbl}"')
    conn.commit()

# ────────────────────────── DDL generators ──────────────────────────
def ddl_for(ns: str) -> str:
    dv2c = _t(ns, "disease_predicate_concepts")
    dlm  = _t(ns, "disease_constraint_lifted_atoms")
    daa  = _t(ns, "disease_constraint_alternatives")
    caa  = _t(ns, "concept_accepted_alternatives")

    return f"""
    CREATE TABLE IF NOT EXISTS {dv2c} (
      trial_id   TEXT NOT NULL,
      var_name   TEXT NOT NULL,
      concept_id TEXT NOT NULL,
      PRIMARY KEY (trial_id, var_name)
    );
    CREATE INDEX IF NOT EXISTS idx_{dv2c}_trial ON {dv2c}(trial_id);
    CREATE INDEX IF NOT EXISTS idx_{dv2c}_concept ON {dv2c}(concept_id);

    CREATE TABLE IF NOT EXISTS {dlm} (
      trial_id           TEXT NOT NULL,
      var_name           TEXT NOT NULL,
      lifted_var         TEXT NOT NULL,
      hop                INTEGER NOT NULL,
      base_var           TEXT,
      timeframe          TEXT,
      lifted_concept_id  TEXT,
      PRIMARY KEY (trial_id, var_name, lifted_var)
    );
    CREATE INDEX IF NOT EXISTS idx_{dlm}_trial ON {dlm}(trial_id);
    CREATE INDEX IF NOT EXISTS idx_{dlm}_lifted_cid ON {dlm}(lifted_concept_id);

    CREATE TABLE IF NOT EXISTS {daa} (
      trial_id       TEXT NOT NULL,
      var_name       TEXT NOT NULL,
      var_name_notime TEXT,
      stem_var       TEXT,
      alt_concept_id TEXT NOT NULL,
      hop            INTEGER NOT NULL,
      reason         TEXT,
      alt_var_name   TEXT,
      alt_var_name_notime TEXT,
      decided_at     TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (trial_id, var_name, alt_concept_id)
    );
    CREATE INDEX IF NOT EXISTS idx_{daa}_trial_var ON {daa}(trial_id, var_name);
    CREATE INDEX IF NOT EXISTS idx_{daa}_altvar ON {daa}(alt_var_name);
    CREATE INDEX IF NOT EXISTS idx_{daa}_altvar_notime ON {daa}(alt_var_name_notime);
    CREATE INDEX IF NOT EXISTS idx_{daa}_var_notime ON {daa}(var_name_notime);
    CREATE INDEX IF NOT EXISTS idx_{daa}_stem_var ON {daa}(stem_var);

    CREATE TABLE IF NOT EXISTS {caa} (
      concept_id     TEXT NOT NULL,
      alt_concept_id TEXT NOT NULL,
      hop            INTEGER NOT NULL,
      alt_label      TEXT,
      reason         TEXT,
      decided_at     TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (concept_id, alt_concept_id)
    );
    CREATE INDEX IF NOT EXISTS idx_{caa}_concept_hop ON {caa}(concept_id, hop);
    """

def _open(db_path: str, ns: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        drop_legacy_namespaced_tables(conn)
        conn.executescript(ddl_for(ns))
    return conn

# ────────────────────────── map rebuild ──────────────────────────
def rebuild_disease_predicate_concepts(
    conn: sqlite3.Connection,
    *,
    ns: str,
    source_table: str,
    trial_id: Optional[str] = None,
    also_predicate_to_concept: bool = False,
    prevention: Optional[bool] = None,
) -> int:
    if prevention is None:
        prevention = _is_prevention_ns(ns)

    dv2c = _t(ns, "disease_predicate_concepts")
    source_table = _safe_ident(source_table, "source table")

    cur = conn.cursor()
    like_pattern = "patient_wants_to_prevent_%" if prevention else "patient_has_finding_of_%"

    if trial_id:
        rows = cur.execute(
            f"SELECT trial_id, var_name, conceptId FROM {source_table} "
            f"WHERE trial_id=? AND conceptId IS NOT NULL AND conceptId<>'' "
            f"AND var_name LIKE ?",
            (trial_id, like_pattern),
        ).fetchall()
    else:
        rows = cur.execute(
            f"SELECT trial_id, var_name, conceptId FROM {source_table} "
            f"WHERE conceptId IS NOT NULL AND conceptId<>'' "
            f"AND var_name LIKE ?",
            (like_pattern,),
        ).fetchall()

    inserted = 0
    with conn:
        for r in rows:
            tid, vn, cid = str(r[0]), str(r[1]), str(r[2])
            conn.execute(
                f"INSERT OR REPLACE INTO {dv2c}(trial_id,var_name,concept_id) VALUES (?,?,?)",
                (tid, vn, cid)
            )
            inserted += 1

        if also_predicate_to_concept and rows:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS predicate_to_concept (var_name TEXT PRIMARY KEY, concept_id TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT OR REPLACE INTO predicate_to_concept(var_name, concept_id) VALUES (?,?)",
                [(str(r[1]), str(r[2])) for r in rows]
            )
    return inserted

# ────────────────────────── lifting ──────────────────────────
def _ancestors_with_hops(concept_id: str):
    minis_par = parents_cached(concept_id) or []
    minis_anc = ancestors_cached(concept_id) or []
    minis_all = minis_par + minis_anc
    ids = [str(m.get("conceptId")) for m in minis_all if m.get("conceptId")]
    hop_map = compute_upward_hops(concept_id, ids)
    minis_all.sort(
        key=lambda m: (
            hop_map.get(str(m.get("conceptId")), 1_000_000),
            (_strip_tag(m.get("preferred_term") or m.get("fully_specified_name") or "")).lower(),
        )
    )
    return minis_all, hop_map

def lift_disease_items(
    conn: sqlite3.Connection,
    *,
    ns: str,
    trial_id: Optional[str] = None,
    max_lift_per_item: int = 200,
    prevention: bool = False,
) -> int:
    dv2c = _t(ns, "disease_predicate_concepts")
    dlm  = _t(ns, "disease_constraint_lifted_atoms")

    cur = conn.cursor()
    if trial_id:
        rows = cur.execute(
            f"SELECT trial_id, var_name, concept_id FROM {dv2c} WHERE trial_id=?",
            (trial_id,)
        ).fetchall()
    else:
        rows = cur.execute(
            f"SELECT trial_id, var_name, concept_id FROM {dv2c}"
        ).fetchall()

    prefix = "patient_wants_to_prevent" if prevention else "patient_has_finding_of"

    inserted_total = 0
    with conn:
        for tid, var_name, base_cid in rows:
            tid = str(tid); var_name = str(var_name); base_cid = str(base_cid)
            tf = _parse_timeframe_from_varname(var_name)

            before = conn.total_changes
            conn.execute(
                f"""INSERT OR IGNORE INTO {dlm}
                    (trial_id,var_name,lifted_var,hop,base_var,timeframe,lifted_concept_id)
                    VALUES (?,?,?,?,?,?,?)""",
                (tid, var_name, var_name, 0, var_name, tf, base_cid)
            )
            inserted_total += max(conn.total_changes - before, 0)

            if not base_cid:
                continue

            minis, hops = _ancestors_with_hops(base_cid)
            count_for_item = 0
            for m in minis:
                aid = str(m.get("conceptId") or "")
                if not aid:
                    continue
                hop = int(hops.get(aid, 999))
                if hop <= 0:
                    continue

                label = _strip_tag(m.get("preferred_term") or m.get("fully_specified_name") or "")
                form  = _to_var(label)
                if not form:
                    continue

                lifted_var = _compose_var_name(form, tf, prefix=prefix)

                before = conn.total_changes
                conn.execute(
                    f"""INSERT OR IGNORE INTO {dlm}
                        (trial_id,var_name,lifted_var,hop,base_var,timeframe,lifted_concept_id)
                        VALUES (?,?,?,?,?,?,?)""",
                    (tid, var_name, lifted_var, hop, var_name, tf, aid)
                )
                delta = max(conn.total_changes - before, 0)
                inserted_total += delta
                count_for_item += delta

                if count_for_item >= max_lift_per_item:
                    break
    return inserted_total

# ────────────────────────── acceptance ──────────────────────────
def populate_disease_constraint_alternatives(conn: sqlite3.Connection, *, ns: str, trial_id: Optional[str]=None) -> int:
    dlm = _t(ns, "disease_constraint_lifted_atoms")
    dv2c = _t(ns, "disease_predicate_concepts")
    daa = _t(ns, "disease_constraint_alternatives")
    caa = _t(ns, "concept_accepted_alternatives")

    cur = conn.cursor()
    mgsr_exists = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (caa,),
    ).fetchone())

    params: List[Any] = []
    where_parts = ["1=1"]
    if trial_id:
        where_parts.append("llm.trial_id = ?")
        params.append(trial_id)

    if not mgsr_exists:
        sql = f"""
        INSERT OR REPLACE INTO {daa}
          (trial_id, var_name, var_name_notime, stem_var, alt_concept_id, hop, reason, alt_var_name, alt_var_name_notime)
        SELECT
          llm.trial_id,
          llm.var_name,
          CASE
            WHEN llm.var_name LIKE '%_inthehistory'
              THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
            ELSE llm.var_name
          END AS var_name_notime,
          CASE
            WHEN llm.var_name LIKE '%_inthehistory'
              THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
            ELSE llm.var_name
          END AS stem_var,
          llm.lifted_concept_id,
          llm.hop,
          'self',
          llm.var_name,
          CASE
            WHEN llm.var_name LIKE '%_inthehistory'
              THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
            ELSE llm.var_name
          END AS alt_var_name_notime
        FROM {dlm} llm
        WHERE {" AND ".join(where_parts)}
          AND llm.hop = 0
          AND llm.lifted_concept_id IS NOT NULL
        """
        cur.execute(sql, params)
        conn.commit()
        return cur.rowcount or 0

    sql = f"""
    INSERT OR REPLACE INTO {daa}
      (trial_id, var_name, var_name_notime, stem_var, alt_concept_id, hop, reason, alt_var_name, alt_var_name_notime)

    SELECT
      llm.trial_id,
      llm.var_name,
      CASE
        WHEN llm.var_name LIKE '%_inthehistory'
          THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
        ELSE llm.var_name
      END AS var_name_notime,
      CASE
        WHEN llm.var_name LIKE '%_inthehistory'
          THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
        ELSE llm.var_name
      END AS stem_var,
      llm.lifted_concept_id,
      llm.hop,
      'self',
      llm.var_name,
      CASE
        WHEN llm.var_name LIKE '%_inthehistory'
          THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
        ELSE llm.var_name
      END AS alt_var_name_notime
    FROM {dlm} llm
    WHERE {" AND ".join(where_parts)}
      AND llm.hop = 0
      AND llm.lifted_concept_id IS NOT NULL

    UNION ALL

    SELECT
      llm.trial_id,
      llm.var_name,
      CASE
        WHEN llm.var_name LIKE '%_inthehistory'
          THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
        ELSE llm.var_name
      END AS var_name_notime,
      CASE
        WHEN llm.var_name LIKE '%_inthehistory'
          THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
        ELSE llm.var_name
      END AS stem_var,
      llm.lifted_concept_id,
      llm.hop,
      'keep-list',
      llm.lifted_var,
      CASE
        WHEN llm.lifted_var LIKE '%_inthehistory'
          THEN substr(llm.lifted_var, 1, length(llm.lifted_var) - length('_inthehistory'))
        ELSE llm.lifted_var
      END AS alt_var_name_notime
    FROM {dlm} llm
    JOIN {dv2c} dv2c
           ON dv2c.trial_id = llm.trial_id
          AND dv2c.var_name = llm.var_name
    JOIN {caa} caa
           ON caa.concept_id = dv2c.concept_id
          AND caa.alt_concept_id = llm.lifted_concept_id
    WHERE {" AND ".join(where_parts)}
      AND llm.hop > 0
      AND llm.lifted_concept_id IS NOT NULL

    UNION ALL

    SELECT
      llm.trial_id,
      llm.lifted_var AS var_name,
      CASE
        WHEN llm.lifted_var LIKE '%_inthehistory'
          THEN substr(llm.lifted_var, 1, length(llm.lifted_var) - length('_inthehistory'))
        ELSE llm.lifted_var
      END AS var_name_notime,
      CASE
        WHEN llm.var_name LIKE '%_inthehistory'
          THEN substr(llm.var_name, 1, length(llm.var_name) - length('_inthehistory'))
        ELSE llm.var_name
      END AS stem_var,
      llm.lifted_concept_id,
      llm.hop,
      'keep-list',
      llm.lifted_var AS alt_var_name,
      CASE
        WHEN llm.lifted_var LIKE '%_inthehistory'
          THEN substr(llm.lifted_var, 1, length(llm.lifted_var) - length('_inthehistory'))
        ELSE llm.lifted_var
      END AS alt_var_name_notime
    FROM {dlm} llm
    JOIN {dv2c} dv2c
           ON dv2c.trial_id = llm.trial_id
          AND dv2c.var_name = llm.var_name
    JOIN {caa} caa
           ON caa.concept_id = dv2c.concept_id
          AND caa.alt_concept_id = llm.lifted_concept_id
    WHERE {" AND ".join(where_parts)}
      AND llm.hop > 0
      AND llm.lifted_concept_id IS NOT NULL
    """
    cur.execute(sql, params + params + params)
    conn.commit()
    return cur.rowcount or 0

# ────────────────────────── decider (same behavior as your version) ──────────────────────────
_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "").strip().lower()
_MODEL_NAME = None
if _ENDPOINT.endswith("gpt-5"):
    _MODEL_NAME = "gpt-5"
elif _ENDPOINT.endswith("gpt-4o"):
    _MODEL_NAME = "gpt-4o"
elif _ENDPOINT.endswith("gpt-4.1"):
    _MODEL_NAME = "gpt-4.1"

PROMPT_KEEP_LIST = """# === TASK ===
You will choose, for each requirement concept, the set of ancestor concepts that should be accepted for inclusion matching.
# (template omitted here for brevity; use --decider-prompt in practice)
{cases}
"""

def _distinct_disease_concepts(conn: sqlite3.Connection, source_table: str, trial_id: Optional[str]) -> List[str]:
    source_table = _safe_ident(source_table, "source table")
    cur = conn.cursor()
    if trial_id:
        q = f"SELECT DISTINCT conceptId FROM {source_table} WHERE trial_id=? AND conceptId IS NOT NULL AND conceptId<>''"
        return [str(r[0]) for r in cur.execute(q, (trial_id,)).fetchall()]
    q = f"SELECT DISTINCT conceptId FROM {source_table} WHERE conceptId IS NOT NULL AND conceptId<>''"
    return [str(r[0]) for r in cur.execute(q).fetchall()]

def _lineage_by_hop(cid: str) -> Dict[str, List[Dict[str,str]]]:
    minis = (parents_cached(cid) or []) + (ancestors_cached(cid) or [])
    ids = [str(m.get("conceptId")) for m in minis if m.get("conceptId")]
    hop_map = compute_upward_hops(cid, ids)
    out: Dict[str, List[Dict[str,str]]] = {}
    for m in minis:
        aid = str(m.get("conceptId") or "")
        if not aid:
            continue
        h = hop_map.get(aid, None)
        if not isinstance(h, int) or h <= 0:
            continue
        lab = _strip_tag(m.get("preferred_term") or m.get("fully_specified_name") or "")
        out.setdefault(str(h), []).append({"concept_id": aid, "label": lab})
    for k in list(out.keys()):
        out[k].sort(key=lambda x: ((x.get("label") or "").lower(), x.get("concept_id") or ""))
    return out

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

def _parse_keep_list_order_agnostic(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(_unwrap_code_fence(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for ent in data:
        cidtok = (ent.get("case_id") or "").strip()
        root_label = ent.get("root_label") or None
        keeps = []
        for k in (ent.get("keep") or []):
            if not isinstance(k, dict):
                continue
            keeps.append({
                "alt_idx": _normalize_idx(k.get("alt_idx")),
                "alt_label": (k.get("alt_label") or None),
                "reason": (k.get("reason") or "keep-list"),
            })
        out.append({"case_id": cidtok, "root_label": root_label, "keep": keeps})
    return out

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

def run_disease_keep_list_decider(
    conn: sqlite3.Connection,
    *,
    ns: str,
    source_table: str,
    trial_id: Optional[str],
    prompt_path: Optional[str]=None,
    batch_size: int=10,
    save_io_dir: Optional[str]=None,
    overwrite: str="purge",
    decider_workers: int=4,
    decider_rate_per_min: int=60,
    lineage_workers: int=1
) -> int:
    if _MODEL_NAME is None:
        print("[decider] OPENAI_ENDPOINT must end with gpt-4o/gpt-4.1/gpt-5 to run decider. Skipping.", file=sys.stderr)
        return 0

    if _MODEL_NAME == "gpt-5":
        from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
    else:
        from smt_core.inference_engine import AzureInferenceEngine  # type: ignore

    engine = AzureInferenceEngine(endpoint=os.getenv("OPENAI_ENDPOINT",""),
                                  api_key_env_var="OPENAI_API_KEY",
                                  model_name=_MODEL_NAME)

    concepts = _distinct_disease_concepts(conn, source_table, trial_id)
    if not concepts:
        print(f"[decider:{ns}] no disease concepts in {source_table}; skipping.", file=sys.stderr)
        return 0

    root = Path(save_io_dir) if save_io_dir else Path(f"mbench/disease_hop_decider_logs__{ns}{_norm_suffix(TABLE_SUFFIX)}")
    if overwrite == "purge" and root.exists():
        shutil.rmtree(root)
    elif overwrite == "fail" and root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Artifacts dir exists and not empty: {root}. Use --decider-overwrite keep|purge|fail.")
    root.mkdir(parents=True, exist_ok=True)

    template = Path(prompt_path).read_text(encoding="utf-8") if prompt_path else PROMPT_KEEP_LIST

    caa = _t(ns, "concept_accepted_alternatives")
    with conn:
        conn.execute(f"DELETE FROM {caa}")

    batches = [concepts[i:i+batch_size] for i in range(0, len(concepts), batch_size)]
    limiter = _RateLimiter(decider_rate_per_min)

    def _build_case(cid: str):
        lin = _lineage_by_hop(cid)
        return {"concept_id": cid,
                "lineage": {h: lst for h, lst in lin.items() if h.isdigit() and int(h) <= 6 and lst}}

    def _build_cases(batch: List[str]):
        out = []
        if lineage_workers > 1:
            with ThreadPoolExecutor(max_workers=lineage_workers) as ex:
                futs = [ex.submit(_build_case, cid) for cid in batch]
                for fu in as_completed(futs):
                    out.append(fu.result())
        else:
            out = [_build_case(cid) for cid in batch]
        return out

    def _call_engine(prompt: str, out_dir: Path):
        limiter.acquire()
        tries, delay = 0, 0.6
        while True:
            try:
                raw = engine(prompt)[0]
                (out_dir / "response.txt").write_text(str(raw), encoding="utf-8")
                return raw
            except Exception:
                tries += 1
                if tries >= 5:
                    raise
                time.sleep(delay + random.uniform(0, 0.3))
                delay *= 2.0

    prepared = []
    for i, b in enumerate(batches):
        bdir = root / f"batch_{i+1:04d}"
        bdir.mkdir(parents=True, exist_ok=True)

        cases_full = _build_cases(b)
        compact_cases: List[Dict[str, Any]] = []
        keymaps_by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
        base_by_case: Dict[str, str] = {}

        def _alpha(n: int) -> str:
            return chr(ord('a') + n)

        for j, c in enumerate(cases_full):
            base_cid = c["concept_id"]
            root_label = _label_for(base_cid) or ""
            case_id = f"c{i+1:04d}_{j:04d}"
            keymap: Dict[str, Dict[str, Any]] = {}
            ancestors_by_hop: Dict[str, List[Dict[str, str]]] = {}

            for hk in sorted(c["lineage"].keys(), key=lambda x: int(x)):
                h = int(hk)
                if h > 6:
                    continue
                entries = []
                for k, m in enumerate(c["lineage"][hk]):
                    aid = str(m.get("concept_id") or "").strip()
                    lab = (m.get("label") or "").strip()
                    if not aid:
                        continue
                    idx_tok = f"{h}.{_alpha(k)}"
                    keymap[idx_tok] = {"concept_id": aid, "hop": h, "label": lab}
                    entries.append({"idx": idx_tok, "label": lab})
                if entries:
                    ancestors_by_hop[str(h)] = entries

            compact_cases.append({
                "case_id": case_id,
                "root_label": root_label,
                "ancestors_by_hop": ancestors_by_hop,
            })
            keymaps_by_case[case_id] = keymap
            base_by_case[case_id] = base_cid

        (bdir / "keymaps_by_case.json").write_text(json.dumps(keymaps_by_case, ensure_ascii=False, indent=2), encoding="utf-8")
        (bdir / "base_by_case.json").write_text(json.dumps(base_by_case, ensure_ascii=False, indent=2), encoding="utf-8")

        prompt = template.replace("{cases}", json.dumps(compact_cases, ensure_ascii=False, indent=2))
        (bdir / "prompt.json").write_text(
            json.dumps({"offset": i*batch_size, "cases": compact_cases, "prompt": prompt},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        prepared.append((bdir, prompt))

    upserts = 0
    with ThreadPoolExecutor(max_workers=decider_workers) as ex:
        fut2dir = {ex.submit(_call_engine, prompt, bdir): bdir for (bdir, prompt) in prepared}
        for fu in as_completed(list(fut2dir.keys())):
            bdir = fut2dir[fu]
            raw = fu.result()
            parsed = _parse_keep_list_order_agnostic(raw)
            (bdir / "response_parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

            keymaps_by_case = json.loads((bdir / "keymaps_by_case.json").read_text(encoding="utf-8"))
            base_by_case    = json.loads((bdir / "base_by_case.json").read_text(encoding="utf-8"))

            caa = _t(ns, "concept_accepted_alternatives")
            with conn:
                for ent in parsed:
                    cidtok = ent.get("case_id") or ""
                    keeps  = ent.get("keep") or []
                    if cidtok not in keymaps_by_case or cidtok not in base_by_case:
                        continue
                    base_cid = base_by_case[cidtok]
                    keymap   = keymaps_by_case[cidtok]
                    seen: set[str] = set()
                    for k in keeps:
                        idx = _normalize_idx(k.get("alt_idx"))
                        if not idx or idx not in keymap:
                            continue
                        hit = keymap[idx]
                        alt_cid = str(hit.get("concept_id") or "")
                        if not alt_cid or alt_cid in seen:
                            continue
                        seen.add(alt_cid)
                        hop = int(hit.get("hop") or 0)
                        alt_label = hit.get("label")
                        reason = k.get("reason") or "keep-list"
                        conn.execute(
                            f"""INSERT INTO {caa}
                                (concept_id, alt_concept_id, hop, alt_label, reason)
                                VALUES (?,?,?,?,?)
                                ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
                                hop=excluded.hop,
                                alt_label=excluded.alt_label,
                                reason=excluded.reason,
                                decided_at=datetime('now')""",
                            (base_cid, alt_cid, hop, alt_label, reason)
                        )
                        upserts += 1
    return upserts

def decide_all_ancestors_for_diseases(
    conn: sqlite3.Connection,
    *,
    ns: str,
    source_table: str,
    trial_id: Optional[str] = None,
) -> int:
    caa = _t(ns, "concept_accepted_alternatives")
    source_table = _safe_ident(source_table, "source table")
    cur = conn.cursor()

    concepts = _distinct_disease_concepts(conn, source_table, trial_id)
    if not concepts:
        print(f"[all-ancestors:{ns}] no disease concepts in {source_table}; nothing to do.", file=sys.stderr)
        return 0

    with conn:
        conn.execute(f"DELETE FROM {caa}")

        upserts = 0
        for base_cid in concepts:
            minis, hop_map = _ancestors_with_hops(base_cid)
            for m in minis:
                aid = str(m.get("conceptId") or "")
                if not aid or aid == base_cid:
                    continue
                hop = int(hop_map.get(aid, 0))
                if hop <= 0:
                    continue
                label = _strip_tag(m.get("preferred_term") or m.get("fully_specified_name") or "")
                conn.execute(
                    f"""
                    INSERT INTO {caa}
                      (concept_id, alt_concept_id, hop, alt_label, reason)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
                      hop       = excluded.hop,
                      alt_label = excluded.alt_label,
                      reason    = excluded.reason,
                      decided_at= datetime('now')
                    """,
                    (base_cid, aid, hop, label, "all-ancestors"),
                )
                upserts += 1
    return upserts

# ────────────────────────── CLI ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Ontology lifting + keep-list–only acceptance for diseases (with table suffix).")
    ap.add_argument("--db", default="../../build/trial.db", help="SQLite DB (e.g., ../../build/trial.db)")
    ap.add_argument("--trial", type=str, default=None, help="Only process this NCT trial_id for lifting")
    ap.add_argument("--source-table", default="disease_constraint_atoms", help="Ingestion table to read from when rebuilding map")
    ap.add_argument("--ns", default="main", help="Namespace: main or prevention (writes *_prevent)")

    ap.add_argument("--table-suffix", type=str, default="",
                    help="Append suffix to ALL tables written/read by this script (e.g., _nonact).")

    ap.add_argument("--rebuild-map", dest="rebuild_map", action="store_true", default=True)
    ap.add_argument("--no-rebuild", dest="rebuild_map", action="store_false")

    ap.add_argument("--overwrite-map", dest="overwrite_map", action="store_true", default=True)
    ap.add_argument("--no-overwrite-map", dest="overwrite_map", action="store_false")

    ap.add_argument("--overwrite-lifts", dest="overwrite_lifts", action="store_true", default=True)
    ap.add_argument("--no-overwrite", dest="overwrite_lifts", action="store_false")

    ap.add_argument("--decider", dest="decider", action="store_true", default=True)
    ap.add_argument("--no-decider", dest="decider", action="store_false")

    ap.add_argument("--keep-all-ancestors", action="store_true", default=False)

    ap.add_argument("--decider-batch-size", type=int, default=10)
    ap.add_argument("--decider-save-dir", type=str, default=None)
    ap.add_argument("--decider-overwrite", choices=["keep","purge","fail"], default="purge")
    ap.add_argument("--decider-prompt", type=str, default="./prompts/decider.prompt")
    ap.add_argument("--decider-workers", type=int, default=8)
    ap.add_argument("--decider-rate-per-min", type=int, default=60)
    ap.add_argument("--lineage-workers", type=int, default=1)

    ap.add_argument("--lift", dest="lift", action="store_true", default=True)
    ap.add_argument("--no-lift", dest="lift", action="store_false")
    ap.add_argument("--max-lift-per-item", type=int, default=int(os.getenv("MAX_LIFT_PER_ITEM", "200")))

    ap.add_argument("--also-predicate_to_concept", action="store_true",
                    help="Also upsert disease vars into global predicate_to_concept")

    args = ap.parse_args()

    global TABLE_SUFFIX
    TABLE_SUFFIX = args.table_suffix or ""

    ns = _safe_ident(args.ns, "namespace")
    source_table = _safe_ident(args.source_table, "source table")
    prevention_branch = _is_prevention_ns(ns)

    conn = _open(args.db, ns)

    has_src = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (source_table,)
    ).fetchone())
    if not has_src:
        print(f"[disease:{ns}] Missing source table {source_table!r}. Run trial_disease_to_sqlite.py first with the SAME --db path.", file=sys.stderr)
        sys.exit(2)

    dv2c = _t(ns, "disease_predicate_concepts")
    dlm  = _t(ns, "disease_constraint_lifted_atoms")
    daa  = _t(ns, "disease_constraint_alternatives")

    try:
        if args.rebuild_map:
            if args.overwrite_map:
                if args.trial:
                    conn.execute(f"DELETE FROM {dv2c} WHERE trial_id=?", (args.trial,))
                else:
                    conn.execute(f"DELETE FROM {dv2c}")

                conn.execute("CREATE TABLE IF NOT EXISTS predicate_to_concept (var_name TEXT PRIMARY KEY, concept_id TEXT NOT NULL)")

                if args.trial:
                    conn.execute(
                        f"DELETE FROM predicate_to_concept WHERE var_name IN (SELECT var_name FROM {source_table} WHERE trial_id=?)",
                        (args.trial,)
                    )
                else:
                    if prevention_branch:
                        conn.execute("DELETE FROM predicate_to_concept WHERE var_name LIKE 'patient_wants_to_prevent_%_inthehistory'")
                    else:
                        conn.execute("DELETE FROM predicate_to_concept WHERE var_name LIKE 'patient_has_finding_of_%_inthehistory'")
                conn.commit()

            n = rebuild_disease_predicate_concepts(
                conn,
                ns=ns,
                source_table=source_table,
                trial_id=args.trial,
                also_predicate_to_concept=args.also_predicate_to_concept,
                prevention=prevention_branch,
            )
            print(f"[disease:{ns}] {dv2c} upserts: {n}")

        if args.keep_all_ancestors:
            up = decide_all_ancestors_for_diseases(conn, ns=ns, source_table=source_table, trial_id=args.trial)
            print(f"[all-ancestors:{ns}] {_t(ns,'concept_accepted_alternatives')} upserts: {up}")
        elif args.decider:
            up = run_disease_keep_list_decider(
                conn,
                ns=ns,
                source_table=source_table,
                trial_id=args.trial,
                prompt_path=args.decider_prompt,
                batch_size=args.decider_batch_size,
                save_io_dir=args.decider_save_dir,
                overwrite=args.decider_overwrite,
                decider_workers=args.decider_workers,
                decider_rate_per_min=args.decider_rate_per_min,
                lineage_workers=args.lineage_workers,
            )
            print(f"[decider:{ns}] {_t(ns,'concept_accepted_alternatives')} upserts: {up}")
        else:
            print(f"[decider:{ns}] decider disabled and --keep-all-ancestors not set; using self-only acceptance.", file=sys.stderr)

        if args.lift:
            if args.overwrite_lifts:
                conn.execute(f"DELETE FROM {dlm}")
                conn.execute(f"DELETE FROM {daa}")
                conn.commit()

            ins = lift_disease_items(
                conn,
                ns=ns,
                trial_id=args.trial,
                max_lift_per_item=args.max_lift_per_item,
                prevention=prevention_branch,
            )
            print(f"[disease:{ns}] {dlm} inserted: {ins}")
            acc = populate_disease_constraint_alternatives(conn, ns=ns, trial_id=args.trial)
            print(f"[disease:{ns}] {daa} upserts: {acc}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()