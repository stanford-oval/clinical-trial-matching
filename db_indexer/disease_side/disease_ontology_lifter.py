#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disease_ontology_lifter.py — keep-list–only acceptance + concurrent decider (DISEASE-SCOPED)

Fresh overwrite policy (default):
  - overwrite_map: deletes disease_predicate_concepts (and optional predicate_to_concept disease entries)
  - overwrite_lifts: deletes disease_constraint_lifted_atoms, disease_constraint_alternatives,
                     and disease_concept_constraint_alternatives
  - decider (if enabled): repopulates disease_concept_constraint_alternatives

Important change:
  - keep-list table is now DISEASE-SCOPED: disease_concept_constraint_alternatives
    (so you can safely rerun without clobbering other domains).

Acceptance materializes:
  (1) self under BASE var  (alt_var_name = var_name)
  (2) kept ancestors under BASE var (audit; alt_var_name = lifted_var)
  (3) kept ancestors under ALT var (var_name = lifted_var; alt_var_name = lifted_var)

Also persists:
  - var_name_notime, alt_var_name_notime
  - stem_var (canonical base alias, no timeframe), sourced from disease_constraint_atoms.stem_var
"""

from __future__ import annotations
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
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

# timeframe helpers
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

# ────────────────────────── DB schema ──────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS disease_predicate_concepts (
  trial_id   TEXT NOT NULL,
  var_name   TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  PRIMARY KEY (trial_id, var_name)
);
CREATE INDEX IF NOT EXISTS idx_dv2c_trial ON disease_predicate_concepts(trial_id);
CREATE INDEX IF NOT EXISTS idx_dv2c_concept ON disease_predicate_concepts(concept_id);

CREATE TABLE IF NOT EXISTS disease_constraint_lifted_atoms (
  trial_id           TEXT NOT NULL,
  var_name           TEXT NOT NULL,
  lifted_var         TEXT NOT NULL,
  hop                INTEGER NOT NULL,
  base_var           TEXT,
  timeframe          TEXT,
  lifted_concept_id  TEXT,
  PRIMARY KEY (trial_id, var_name, lifted_var)
);
CREATE INDEX IF NOT EXISTS idx_dlm_trial ON disease_constraint_lifted_atoms(trial_id);
CREATE INDEX IF NOT EXISTS idx_dlm_lifted_cid ON disease_constraint_lifted_atoms(lifted_concept_id);

CREATE TABLE IF NOT EXISTS disease_constraint_alternatives (
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
CREATE INDEX IF NOT EXISTS idx_daa_trial_var ON disease_constraint_alternatives(trial_id, var_name);

-- ✅ disease-scoped keep-list
CREATE TABLE IF NOT EXISTS disease_concept_constraint_alternatives (
  concept_id     TEXT NOT NULL,
  alt_concept_id TEXT NOT NULL,
  hop            INTEGER NOT NULL,
  alt_label      TEXT,
  reason         TEXT,
  decided_at     TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (concept_id, alt_concept_id)
);
CREATE INDEX IF NOT EXISTS idx_dcaa_concept_hop ON disease_concept_constraint_alternatives(concept_id, hop);
"""

def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_alternatives)").fetchall()]
    if "alt_var_name" not in cols:
        cur.execute("ALTER TABLE disease_constraint_alternatives ADD COLUMN alt_var_name TEXT")
    if "alt_var_name_notime" not in cols:
        cur.execute("ALTER TABLE disease_constraint_alternatives ADD COLUMN alt_var_name_notime TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daa_altvar_notime ON disease_constraint_alternatives(alt_var_name_notime)")
    if "var_name_notime" not in cols:
        cur.execute("ALTER TABLE disease_constraint_alternatives ADD COLUMN var_name_notime TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daa_var_notime ON disease_constraint_alternatives(var_name_notime)")
    if "stem_var" not in cols:
        cur.execute("ALTER TABLE disease_constraint_alternatives ADD COLUMN stem_var TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daa_stem_var ON disease_constraint_alternatives(stem_var)")
        cur.execute("""
            UPDATE disease_constraint_alternatives
               SET stem_var = COALESCE(var_name_notime, var_name)
             WHERE stem_var IS NULL OR stem_var = '';
        """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_daa_altvar ON disease_constraint_alternatives(alt_var_name)")
    conn.commit()

def _open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(DDL)
    _ensure_schema_migrations(conn)
    return conn

# ────────────────────────── map rebuild ──────────────────────────
def rebuild_disease_predicate_concepts(conn: sqlite3.Connection, trial_id: Optional[str] = None, also_predicate_to_concept: bool = False) -> int:
    cur = conn.cursor()
    if trial_id:
        q = """SELECT trial_id, var_name, stem_var, conceptId
               FROM disease_constraint_atoms
               WHERE trial_id=? AND conceptId IS NOT NULL AND conceptId<>''"""
        rows = cur.execute(q, (trial_id,)).fetchall()
    else:
        q = """SELECT trial_id, var_name, stem_var, conceptId
               FROM disease_constraint_atoms
               WHERE conceptId IS NOT NULL AND conceptId<>''"""
        rows = cur.execute(q).fetchall()

    inserted = 0
    with conn:
        for r in rows:
            tid = str(r[0])
            vn = str(r[1])
            cid = str(r[3])
            conn.execute(
                "INSERT OR REPLACE INTO disease_predicate_concepts(trial_id,var_name,concept_id) VALUES (?,?,?)",
                (tid, vn, cid)
            )
            inserted += 1

        if also_predicate_to_concept:
            conn.execute("CREATE TABLE IF NOT EXISTS predicate_to_concept (var_name TEXT PRIMARY KEY, concept_id TEXT NOT NULL)")
            pairs: List[Tuple[str, str]] = []
            for r in rows:
                vn = str(r[1])
                stem_v = str(r[2] or "")
                cid = str(r[3])
                if vn:
                    pairs.append((vn, cid))
                if stem_v:
                    pairs.append((stem_v, cid))
            conn.executemany("INSERT OR REPLACE INTO predicate_to_concept(var_name, concept_id) VALUES (?,?)", pairs)

    return inserted

# ────────────────────────── lifting ──────────────────────────
def _ancestors_with_hops(concept_id: str):
    minis_all = (parents_cached(concept_id) or []) + (ancestors_cached(concept_id) or [])
    ids = [str(m.get("conceptId")) for m in minis_all if m.get("conceptId")]
    hop_map = compute_upward_hops(concept_id, ids)
    minis_all.sort(
        key=lambda m: (
            hop_map.get(str(m.get("conceptId")), 1_000_000),
            (_strip_tag(m.get("preferred_term") or m.get("fully_specified_name") or "")).lower(),
        )
    )
    return minis_all, hop_map

def lift_disease_items(conn: sqlite3.Connection, *, trial_id: Optional[str]=None, max_lift_per_item: int = 200) -> int:
    cur = conn.cursor()
    if trial_id:
        rows = cur.execute("""SELECT trial_id, var_name, concept_id FROM disease_predicate_concepts WHERE trial_id=?""", (trial_id,)).fetchall()
    else:
        rows = cur.execute("""SELECT trial_id, var_name, concept_id FROM disease_predicate_concepts""").fetchall()

    inserted_total = 0
    with conn:
        for tid, var_name, base_cid in rows:
            tid = str(tid)
            var_name = str(var_name)
            base_cid = str(base_cid)
            tf = _parse_timeframe_from_varname(var_name)

            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO disease_constraint_lifted_atoms
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

                lifted_var = _compose_var_name(form, tf, prefix="patient_has_finding_of")

                before = conn.total_changes
                conn.execute(
                    """INSERT OR IGNORE INTO disease_constraint_lifted_atoms
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
def populate_disease_constraint_alternatives(conn: sqlite3.Connection, *, trial_id: Optional[str]=None, use_keep_list: bool=True) -> int:
    """
    If use_keep_list=False: insert ONLY self rows (hop=0).
    If use_keep_list=True: insert self + kept ancestors (base + alt var forms).
    stem_var is pulled from disease_constraint_atoms.stem_var for the BASE var.
    """
    cur = conn.cursor()
    params: List[Any] = []
    where_parts = ["1=1"]
    if trial_id:
        where_parts.append("llm.trial_id = ?")
        params.append(trial_id)

    if not use_keep_list:
        sql = f"""
        INSERT OR REPLACE INTO disease_constraint_alternatives
          (trial_id, var_name, var_name_notime, stem_var, alt_concept_id, hop, reason, alt_var_name, alt_var_name_notime)
        SELECT
          llm.trial_id,
          llm.var_name,
          dli.stem_var AS var_name_notime,
          dli.stem_var AS stem_var,
          llm.lifted_concept_id,
          llm.hop,
          'self',
          llm.var_name,
          dli.stem_var AS alt_var_name_notime
        FROM disease_constraint_lifted_atoms llm
        JOIN disease_constraint_atoms dli
          ON dli.trial_id = llm.trial_id
         AND dli.var_name = llm.var_name
        WHERE {" AND ".join(where_parts)}
          AND llm.hop = 0
          AND llm.lifted_concept_id IS NOT NULL
        """
        cur.execute(sql, params)
        conn.commit()
        return cur.rowcount or 0

    sql = f"""
    INSERT OR REPLACE INTO disease_constraint_alternatives
      (trial_id, var_name, var_name_notime, stem_var, alt_concept_id, hop, reason, alt_var_name, alt_var_name_notime)

    -- 1) self
    SELECT
      llm.trial_id,
      llm.var_name,
      dli.stem_var AS var_name_notime,
      dli.stem_var AS stem_var,
      llm.lifted_concept_id,
      llm.hop,
      'self',
      llm.var_name,
      dli.stem_var AS alt_var_name_notime
    FROM disease_constraint_lifted_atoms llm
    JOIN disease_constraint_atoms dli
      ON dli.trial_id = llm.trial_id
     AND dli.var_name = llm.var_name
    WHERE {" AND ".join(where_parts)}
      AND llm.hop = 0
      AND llm.lifted_concept_id IS NOT NULL

    UNION ALL

    -- 2) kept ancestors under BASE var
    SELECT
      llm.trial_id,
      llm.var_name,
      dli.stem_var AS var_name_notime,
      dli.stem_var AS stem_var,
      llm.lifted_concept_id,
      llm.hop,
      'keep-list',
      llm.lifted_var AS alt_var_name,
      CASE
        WHEN llm.lifted_var LIKE '%_inthehistory'
          THEN substr(llm.lifted_var, 1, length(llm.lifted_var) - length('_inthehistory'))
        ELSE llm.lifted_var
      END AS alt_var_name_notime
    FROM disease_constraint_lifted_atoms llm
    JOIN disease_constraint_atoms dli
      ON dli.trial_id = llm.trial_id
     AND dli.var_name = llm.var_name
    JOIN disease_predicate_concepts dv2c
      ON dv2c.trial_id = llm.trial_id
     AND dv2c.var_name = llm.var_name
    JOIN disease_concept_constraint_alternatives dcaa
      ON dcaa.concept_id = dv2c.concept_id
     AND dcaa.alt_concept_id = llm.lifted_concept_id
    WHERE {" AND ".join(where_parts)}
      AND llm.hop > 0
      AND llm.lifted_concept_id IS NOT NULL

    UNION ALL

    -- 3) kept ancestors under ALT var (var_name = lifted_var)
    SELECT
      llm.trial_id,
      llm.lifted_var AS var_name,
      CASE
        WHEN llm.lifted_var LIKE '%_inthehistory'
          THEN substr(llm.lifted_var, 1, length(llm.lifted_var) - length('_inthehistory'))
        ELSE llm.lifted_var
      END AS var_name_notime,
      dli.stem_var AS stem_var,
      llm.lifted_concept_id,
      llm.hop,
      'keep-list',
      llm.lifted_var AS alt_var_name,
      CASE
        WHEN llm.lifted_var LIKE '%_inthehistory'
          THEN substr(llm.lifted_var, 1, length(llm.lifted_var) - length('_inthehistory'))
        ELSE llm.lifted_var
      END AS alt_var_name_notime
    FROM disease_constraint_lifted_atoms llm
    JOIN disease_constraint_atoms dli
      ON dli.trial_id = llm.trial_id
     AND dli.var_name = llm.var_name
    JOIN disease_predicate_concepts dv2c
      ON dv2c.trial_id = llm.trial_id
     AND dv2c.var_name = llm.var_name
    JOIN disease_concept_constraint_alternatives dcaa
      ON dcaa.concept_id = dv2c.concept_id
     AND dcaa.alt_concept_id = llm.lifted_concept_id
    WHERE {" AND ".join(where_parts)}
      AND llm.hop > 0
      AND llm.lifted_concept_id IS NOT NULL
    """
    cur.execute(sql, params + params + params)
    conn.commit()
    return cur.rowcount or 0

# ────────────────────────── decider ──────────────────────────
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

# === Goal ===
Return a concise allow-list of clinically relevant ancestors per concept.

# === Inputs ===
For each case below, you are given ONLY the ancestor labels grouped by hop,
each ancestor being identified by an index like "1.a", "2.c", etc.

Each case:
{
  "case_id": "c<batch>_<index>",
  "root_label": "<preferred term of the requirement>",
  "ancestors_by_hop": { ... }
}

# === Deny-list (case-insensitive substring) ===
["clinical finding","disorder","disease","finding by site","symptom","procedure","event","abnormal morphology","physical object","situation","observable entity","body structure","organism","substance"]

# === Guidance ===
- Prefer shallow ancestors that preserve intent.
- Avoid overly generic ancestors.
- Output indices only; do not invent indices.

# === Output (JSON array only; no prose) ===
[
  {
    "case_id":"...",
    "root_label":"...",
    "keep":[{"alt_idx":"1.a","alt_label":"...","reason":"<=25 words"}, ...]
  },
  ...
]

# === CASES ===
{cases}
"""

def _distinct_disease_concepts(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    q = "SELECT DISTINCT conceptId FROM disease_constraint_atoms WHERE conceptId IS NOT NULL AND conceptId<>''"
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
        if not isinstance(ent, dict):
            continue
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

def run_disease_keep_list_decider(conn: sqlite3.Connection, *,
                                  prompt_path: Optional[str]=None,
                                  batch_size: int=10,
                                  save_io_dir: Optional[str]=None,
                                  overwrite: str="purge",
                                  decider_workers: int=4,
                                  decider_rate_per_min: int=60,
                                  lineage_workers: int=1) -> int:
    if _MODEL_NAME is None:
        print("[decider] OPENAI_ENDPOINT must end with gpt-4o or gpt-5 to run decider. Skipping.", file=sys.stderr)
        return 0

    if _MODEL_NAME == "gpt-5":
        from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
    else:
        from smt_core.inference_engine import AzureInferenceEngine  # type: ignore

    engine = AzureInferenceEngine(endpoint=os.getenv("OPENAI_ENDPOINT",""),
                                  api_key_env_var="OPENAI_API_KEY",
                                  model_name=_MODEL_NAME)

    concepts = _distinct_disease_concepts(conn)
    if not concepts:
        print("[decider] no disease concepts in disease_constraint_atoms; skipping.", file=sys.stderr)
        return 0

    root = Path(save_io_dir) if save_io_dir else Path("mbench/disease_hop_decider_logs")
    if overwrite == "purge" and root.exists():
        shutil.rmtree(root)
    elif overwrite == "fail" and root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Artifacts dir exists and not empty: {root}. Use --decider-overwrite keep|purge|fail.")
    root.mkdir(parents=True, exist_ok=True)

    template = Path(prompt_path).read_text(encoding="utf-8") if prompt_path else PROMPT_KEEP_LIST

    # ✅ Fresh run semantics for DISEASE keep-list table
    with conn:
        conn.execute("DELETE FROM disease_concept_constraint_alternatives")

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

    upserts = 0
    batches = [concepts[i:i+batch_size] for i in range(0, len(concepts), batch_size)]

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

            compact_cases.append({"case_id": case_id, "root_label": root_label, "ancestors_by_hop": ancestors_by_hop})
            keymaps_by_case[case_id] = keymap
            base_by_case[case_id] = base_cid

        (bdir / "keymaps_by_case.json").write_text(json.dumps(keymaps_by_case, ensure_ascii=False, indent=2), encoding="utf-8")
        (bdir / "base_by_case.json").write_text(json.dumps(base_by_case, ensure_ascii=False, indent=2), encoding="utf-8")

        prompt = template.replace("{cases}", json.dumps(compact_cases, ensure_ascii=False, indent=2))
        (bdir / "prompt.json").write_text(json.dumps({"cases": compact_cases, "prompt": prompt}, ensure_ascii=False, indent=2), encoding="utf-8")
        prepared.append((bdir, prompt))

    with ThreadPoolExecutor(max_workers=decider_workers) as ex:
        futs = [ex.submit(_call_engine, prompt, bdir) for (bdir, prompt) in prepared]
        for fu, (bdir, _prompt) in zip(as_completed(futs), prepared):
            raw = fu.result()
            parsed = _parse_keep_list_order_agnostic(raw)
            (bdir / "response_parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

            keymaps_by_case = json.loads((bdir / "keymaps_by_case.json").read_text(encoding="utf-8"))
            base_by_case    = json.loads((bdir / "base_by_case.json").read_text(encoding="utf-8"))

            up = 0
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
                            """INSERT INTO disease_concept_constraint_alternatives
                               (concept_id, alt_concept_id, hop, alt_label, reason)
                               VALUES (?,?,?,?,?)
                               ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
                                 hop=excluded.hop,
                                 alt_label=excluded.alt_label,
                                 reason=excluded.reason,
                                 decided_at=datetime('now')""",
                            (base_cid, alt_cid, hop, alt_label, reason)
                        )
                        up += 1
            upserts += up

    return upserts

# ────────────────────────── CLI ──────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ontology lifting + keep-list–only acceptance for coded disease list items (fresh overwrite by default).")
    ap.add_argument("--db", default="../../build/trial.db", help="SQLite DB (e.g., ../../build/trial.db)")
    ap.add_argument("--trial", type=str, default=None, help="Only process this NCT trial_id for lifting")

    # Map rebuild
    ap.add_argument("--rebuild-map", dest="rebuild_map", action="store_true", default=True)
    ap.add_argument("--no-rebuild", dest="rebuild_map", action="store_false")

    # Overwrite controls (default ON)
    ap.add_argument("--overwrite-map", dest="overwrite_map", action="store_true", default=True)
    ap.add_argument("--no-overwrite-map", dest="overwrite_map", action="store_false")
    ap.add_argument("--overwrite-lifts", dest="overwrite_lifts", action="store_true", default=True)
    ap.add_argument("--no-overwrite", dest="overwrite_lifts", action="store_false")

    # Decider controls (default ON)
    ap.add_argument("--decider", dest="decider", action="store_true", default=True)
    ap.add_argument("--no-decider", dest="decider", action="store_false")
    ap.add_argument("--decider-batch-size", type=int, default=10)
    ap.add_argument("--decider-save-dir", type=str, default=None)
    ap.add_argument("--decider-overwrite", choices=["keep","purge","fail"], default="purge")
    ap.add_argument("--decider-prompt", type=str, default=None)
    ap.add_argument("--decider-workers", type=int, default=8)
    ap.add_argument("--decider-rate-per-min", type=int, default=60)
    ap.add_argument("--lineage-workers", type=int, default=1)

    ap.add_argument("--lift", dest="lift", action="store_true", default=True)
    ap.add_argument("--no-lift", dest="lift", action="store_false")
    ap.add_argument("--max-lift-per-item", type=int, default=int(os.getenv("MAX_LIFT_PER_ITEM", "200")))
    ap.add_argument("--also-predicate_to_concept", action="store_true",
                    help="Also upsert disease vars (both var_name and stem_var) into global predicate_to_concept")

    args = ap.parse_args()
    conn = _open(args.db)

    has_dli = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_atoms'").fetchone())
    if not has_dli:
        print("[disease] Missing table 'disease_constraint_atoms'. Run trial_disease_to_sqlite.py first with the SAME --db path.", file=sys.stderr)
        sys.exit(2)

    try:
        if args.rebuild_map:
            if args.overwrite_map:
                if args.trial:
                    conn.execute("DELETE FROM disease_predicate_concepts WHERE trial_id=?", (args.trial,))
                else:
                    conn.execute("DELETE FROM disease_predicate_concepts")
                conn.commit()

            n = rebuild_disease_predicate_concepts(conn, trial_id=args.trial, also_predicate_to_concept=args.also_predicate_to_concept)
            print(f"[disease] disease_predicate_concepts upserts: {n}")

        # ✅ Always clear disease keep-list table when doing an overwrite-lifts run,
        # so stale keep-lists never leak into acceptance.
        if args.overwrite_lifts:
            conn.execute("DELETE FROM disease_concept_constraint_alternatives")
            conn.execute("DELETE FROM disease_constraint_lifted_atoms")
            conn.execute("DELETE FROM disease_constraint_alternatives")
            conn.commit()

        if args.decider:
            up = run_disease_keep_list_decider(
                conn,
                prompt_path=args.decider_prompt,
                batch_size=args.decider_batch_size,
                save_io_dir=args.decider_save_dir,
                overwrite=args.decider_overwrite,
                decider_workers=args.decider_workers,
                decider_rate_per_min=args.decider_rate_per_min,
                lineage_workers=args.lineage_workers,
            )
            print(f"[decider] disease_concept_constraint_alternatives upserts: {up}")

        if args.lift:
            ins = lift_disease_items(conn, trial_id=args.trial, max_lift_per_item=args.max_lift_per_item)
            print(f"[disease] disease_constraint_lifted_atoms inserted: {ins}")

            acc = populate_disease_constraint_alternatives(conn, trial_id=args.trial, use_keep_list=bool(args.decider))
            print(f"[disease] disease_constraint_alternatives upserts: {acc}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
