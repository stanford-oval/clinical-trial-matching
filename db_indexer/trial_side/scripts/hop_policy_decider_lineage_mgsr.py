#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hop_policy_decider_lineage_mgsr.py — flattened keep-list decider (index-driven, case_id echo)
(Positive-literals edition)

Behavior:
  - Collects base concepts from positive_constraint_literals (+ predicate_to_concept).
  - Builds keep-list decisions via index-driven prompt (e.g., "1.a", "2.c").
  - Prompt contains NO concept ids and NO feature blobs.
  - Each case includes an opaque case_id and a root_label for readability.
  - LLM returns indices + echoes case_id/root_label; we resolve indices → SNOMED concept_ids
    via a per-case keymap from lineage.
  - Only indices are trusted; LLM-provided labels are ignored when persisting.
  - Purges concept_accepted_alternatives before writing new decisions.
  - Multithreaded LLM calls + token-bucket rate limiting.
  - Optional threaded lineage building (constants below).
  - After deciding, materializes
      positive_constraint_alternatives_expanded
    keyed by (nct_id,kind,variant,direction,var_name,timeframe).

New:
  • External prompt template loaded from prompts/decider.prompt (or DECIDER_PROMPT_PATH).
  • Lineage de-duplicated per hop.
  • Robust timeframe token stripping.
  • Order-agnostic resolution via per-case case_id (no dependence on output order).
  • Keymap stores BOTH concept_id and label; persisted alt_label comes from keymap (never from LLM output).
  • Optional --keep-all-ancestors mode: skip LLM, keep all lineage ancestors as accepted alternatives.

Usage:
  python scripts/hop_policy_decider_lineage_mgsr.py --db /path/to/trial.db [--include-assumed] [--keep-all-ancestors]
"""

from __future__ import annotations

import json, os, re, sqlite3, time, logging, sys, shutil, random, threading
import csv
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────
BATCH_SIZE      = 10   # concepts per LLM call
WORKERS         = 24   # concurrent LLM requests
RATE_PER_MIN    = 60   # global LLM requests/min across workers
LINEAGE_WORKERS = 1    # concurrent Snowstorm fetchers per batch (be gentle)
ARTIFACT_DIR    = Path("mbench/hop_decider_logs")
LOG_LEVEL       = "INFO"

# ────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────
LOGGER = logging.getLogger("keep_list_decider")
def _configure_logging():
    lvl = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    log_path = ARTIFACT_DIR / "hop_policy_decider.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w", encoding="utf-8")]
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)

# ────────────────────────────────────────────────────────────────
# Engine wiring (lazy, so --keep-all-ancestors works without OPENAI_ENDPOINT)
# ────────────────────────────────────────────────────────────────

def _make_engine():
    """
    Initialize AzureInferenceEngine based on OPENAI_ENDPOINT suffix.
    Returns (engine, model_name).
    """
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
    engine = AzureInferenceEngine(endpoint=os.environ.get("OPENAI_ENDPOINT",""),
                                  api_key_env_var="OPENAI_API_KEY",
                                  model_name=model_name)
    return engine, model_name

# ────────────────────────────────────────────────────────────────
# Prompt loader (external file)
# ────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE_CACHE: Optional[str] = None

def _default_prompt_path() -> Path:
    return (Path(__file__).resolve().parent / "prompts" / "decider.prompt")

def _load_prompt_template() -> str:
    """
    Load the prompt template from:
      - DECIDER_PROMPT_PATH env var (if set), else
      - prompts/decider.prompt next to this script.
    Must contain "{cases}" placeholder.
    """
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
    LOGGER.info(f"[prompt] loaded template from {path}")
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
            if p in seen: continue
            seen.add(p); hop = h+1
            if p in targets and p not in found:
                found[p] = hop
            q.append((p, hop))
    return found

# ────────────────────────────────────────────────────────────────
# DB helpers
# ────────────────────────────────────────────────────────────────
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
CREATE INDEX IF NOT EXISTS idx_caa_concept_hop ON concept_accepted_alternatives(concept_id, hop);
"""

def _open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(DDL)
    return conn

# Concepts iterator (positive_constraint_literals → predicate_to_concept)
def _iter_concepts(conn: sqlite3.Connection, include_assumed: bool) -> Iterable[str]:
    """
    Source base concepts from already-known positive literals:
      - Always include kind='inclusion'
      - If include_assumed=True, also include rows where variant='assumed'
    """
    cur = conn.cursor()
    if include_assumed:
        q = """
            SELECT DISTINCT v2.concept_id
            FROM positive_constraint_literals pl
            JOIN predicate_to_concept v2 ON v2.var_name = pl.var_name
            WHERE pl.kind='inclusion' OR pl.variant='assumed'
        """
    else:
        q = """
            SELECT DISTINCT v2.concept_id
            FROM positive_constraint_literals pl
            JOIN predicate_to_concept v2 ON v2.var_name = pl.var_name
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
    """Parents + ancestors, grouped by hop with de-dup per hop."""
    minis_par = parents_cached(cid) or []
    minis_anc = ancestors_cached(cid) or []
    minis_all = minis_par + minis_anc

    ids = [str(m.get("conceptId")) for m in minis_all if m.get("conceptId")]
    hop_map = compute_upward_hops(cid, ids)

    grouped_raw: Dict[str, Dict[str, str]] = {}  # hop -> {concept_id: label}
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

def _collect_case(conn: sqlite3.Connection, cid: str) -> ConceptCase:
    return ConceptCase(concept_id=cid, lineage=_lineage_grouped_by_hop(cid))

def _decide_all_ancestors(conn: sqlite3.Connection, cids: List[str]) -> int:
    """
    Populate concept_accepted_alternatives by keeping *all* ancestors
    from Snowstorm lineage for each base concept, without calling the LLM.

    - Uses _lineage_grouped_by_hop to get hop distances.
    - Reason is always 'all-ancestors'.
    """
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
                        """
                        INSERT INTO concept_accepted_alternatives
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

def _parse_keep_list_order_agnostic(raw: str) -> List[Dict[str, Any]]:
    """
    Expect a JSON array of elements like:
      { "case_id": "c0001_0000", "root_label": "...", "keep": [ {"alt_idx":"1.a","reason":"..."} ... ] }
    Order does NOT matter (we match by case_id).
    Tolerates legacy entries missing case_id by ignoring them (logged).
    """
    try:
        data = json.loads(_unwrap_code_fence(raw))
    except json.JSONDecodeError:
        LOGGER.warning("[parse] keep-list JSON decode failed; returning empty")
        return []
    if not isinstance(data, list):
        LOGGER.warning("[parse] keep-list expected a list; returning empty")
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
                "alt_key": (k.get("alt_key") or "").strip() or None,                # legacy opaque key
                "alt_concept_id": (k.get("alt_concept_id") or "").strip() or None,  # legacy direct id
                "reason": (k.get("reason") or "").strip() or None,
            })
        out.append({"case_id": cidtok, "keep": keeps})
    return out

def _engine_call(engine, prompt: str) -> str:
    t0 = time.time()
    out = engine(prompt)
    dt = (time.time() - t0) * 1000
    try:
        first = out[0]
        s = str(first[0]) if isinstance(first, (list, tuple)) else str(first)
    except Exception:
        s = str(out)
    LOGGER.info(f"[engine] response in {dt:.1f} ms (chars={len(s)})")
    return s

# ────────────────────────────────────────────────────────────────
# Concurrency helpers
# ────────────────────────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, per_min: int):
        self.capacity = max(per_min, 1); self.tokens = self.capacity
        self.lock = threading.Lock(); self.last_refill = time.time()
    def acquire(self):
        while True:
            with self.lock:
                now = time.time(); elapsed = now - self.last_refill
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

def _build_var_for_concept(cid: str, timeframe: str) -> str:
    lab = _label_for(cid) or ""
    entity = _slug_entity(lab) or f"concept_{cid}"
    prefix = _choose_prefix(cid, lab)
    tf = timeframe if timeframe else "now"
    return f"{prefix}_{entity}_{tf}"

def _strip_timeframe_once(stem: str) -> str:
    if not stem:
        return stem
    out = _TIMEFRAME_TOKEN_RX.sub("", stem, count=1)
    out = re.sub(r"_+", "_", out).strip("_")
    return out

# ────────────────────────────────────────────────────────────────
# Expanded table builder (positive literals)
# ────────────────────────────────────────────────────────────────
def _create_poslit_expanded(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TABLE IF EXISTS positive_constraint_alternatives_expanded;
    CREATE TABLE positive_constraint_alternatives_expanded (
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
    CREATE INDEX IF NOT EXISTS idx_pos_caae_join ON positive_constraint_alternatives_expanded(alt_concept_id, timeframe);
    """)

def _fetch_alt_map(conn: sqlite3.Connection) -> Dict[str, List[sqlite3.Row]]:
    q = """
    SELECT concept_id, alt_concept_id, hop, COALESCE(reason,'') AS reason,
           decided_at
    FROM concept_accepted_alternatives
    ORDER BY concept_id, hop, alt_concept_id
    """
    mp: Dict[str, List[sqlite3.Row]] = {}
    for r in conn.execute(q):
        mp.setdefault(str(r["concept_id"]), []).append(r)
    return mp

def _iter_poslits(conn: sqlite3.Connection, include_assumed: bool) -> List[sqlite3.Row]:
    where = "pl.kind='inclusion'" + (" OR pl.variant='assumed'" if include_assumed else "")
    q = f"""
      SELECT pl.nct_id, pl.kind, pl.variant, pl.direction, pl.var_name,
             COALESCE(pl.timeframe,'now') AS timeframe,
             COALESCE(pl.base_var, pl.var_name) AS base_var,
             v2.concept_id AS base_concept_id
      FROM positive_constraint_literals pl
      LEFT JOIN predicate_to_concept v2 ON v2.var_name = pl.var_name
      WHERE {where}
      ORDER BY pl.nct_id, pl.kind, pl.variant, pl.var_name
    """
    return list(conn.execute(q))

def _rebuild_expanded(conn: sqlite3.Connection, artifact_dir: Path, *, include_assumed: bool) -> None:
    _create_poslit_expanded(conn)
    rows = _iter_poslits(conn, include_assumed=include_assumed)
    alt_map = _fetch_alt_map(conn)
    out_rows: List[Dict[str, Any]] = []

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
            # Cannot expand without a resolvable base concept id
            continue

        base_var_stem = _strip_timeframe_once(base_var)

        # self
        lifted_var_self = _build_var_for_concept(base_cid, timeframe)
        lifted_var_self_stem = _strip_timeframe_once(lifted_var_self)

        cur.execute("""
            INSERT OR REPLACE INTO positive_constraint_alternatives_expanded
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

            lifted_var = _build_var_for_concept(alt_cid, timeframe)
            lifted_var_stem = _strip_timeframe_once(lifted_var)

            cur.execute("""
                INSERT OR REPLACE INTO positive_constraint_alternatives_expanded
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
# Compact prompt prep (index-driven + case_id) + core execution
# ────────────────────────────────────────────────────────────────
def _alpha(n: int) -> str:
    return chr(ord('a') + n)  # 0->a, 1->b, ...

def decide_and_persist(db_path: str, *, include_assumed: bool=False, keep_all_ancestors: bool=False) -> None:
    # artifacts + logging
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_logging()
    LOGGER.info(f"[start] DB={db_path}")
    LOGGER.info(f"[env] OPENAI_ENDPOINT={os.environ.get('OPENAI_ENDPOINT','')}")
    LOGGER.info(f"[env] SNOWSTORM={SNOWSTORM_BASE} BRANCH={SNOWSTORM_BRANCH} FORM={SNOWSTORM_FORM}")
    LOGGER.info(f"[opts] include_assumed={include_assumed} keep_all_ancestors={keep_all_ancestors}")

    # DB
    conn = _open(db_path)
    cur = conn.cursor()

    # concepts
    cids = list(_iter_concepts(conn, include_assumed=include_assumed))
    LOGGER.info(f"[concepts] total={len(cids)} (include_assumed={include_assumed})")
    if not cids:
        LOGGER.warning("[concepts] nothing to decide; exiting")
        conn.close()
        return

    # purge existing accepted_alternatives
    with conn:
        cur.execute("DELETE FROM concept_accepted_alternatives")

    # Mode selection
    if keep_all_ancestors:
        # ── ancestor-only mode (no LLM) ─────────────────────────
        meta = {
            "db": db_path,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "rate_per_min": RATE_PER_MIN,
            "lineage_workers": LINEAGE_WORKERS,
            "model_name": None,
            "include_assumed": include_assumed,
            "keep_all_ancestors": True,
            "mode": "all_ancestors",
        }
        (ARTIFACT_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        LOGGER.info("[mode] keep_all_ancestors=True — skipping LLM and accepting all lineage ancestors.")
        n_upserts = _decide_all_ancestors(conn, cids)
        LOGGER.info(f"[db] all-ancestors mode: upserts={n_upserts}")

    else:
        # ── existing LLM-driven mode ────────────────────────────
        engine, model_name = _make_engine()

        meta = {
            "db": db_path,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "rate_per_min": RATE_PER_MIN,
            "lineage_workers": LINEAGE_WORKERS,
            "model_name": model_name,
            "include_assumed": include_assumed,
            "keep_all_ancestors": False,
            "mode": "llm_keep_list",
        }
        (ARTIFACT_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # batching
        batches = [cids[i:i+BATCH_SIZE] for i in range(0, len(cids), BATCH_SIZE)]
        limiter = _RateLimiter(RATE_PER_MIN)

        def _prepare_batch(idx: int, ids: List[str]) -> Tuple[Path, str, List[ConceptCase]]:
            bdir = ARTIFACT_DIR / f"batch_{idx+1:04d}"
            bdir.mkdir(parents=True, exist_ok=True)

            # assemble cases
            if LINEAGE_WORKERS > 1:
                cases: List[ConceptCase] = []
                with ThreadPoolExecutor(max_workers=LINEAGE_WORKERS) as ex_lin:
                    futs = [ex_lin.submit(_collect_case, conn, cid) for cid in ids]
                    for fu in as_completed(futs):
                        cases.append(fu.result())
            else:
                cases = [_collect_case(conn, cid) for cid in ids]

            # build compact prompt cases (no IDs), with per-case keymap {idx -> (concept_id, hop, label)}
            compact: List[Dict[str, Any]] = []
            keymaps_by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
            base_by_case: Dict[str, str] = {}

            for i, c in enumerate(cases):
                case_id_token = f"c{idx+1:04d}_{i:04d}"
                root_label = _label_for(c.concept_id) or ""
                ancestors_by_hop: Dict[str, List[Dict[str, str]]] = {}
                keymap: Dict[str, Dict[str, Any]] = {}

                # compact lineage to ≤6 hops
                for hk in sorted((k for k in c.lineage.keys() if k.isdigit() and int(k) <= 6), key=lambda x: int(x)):
                    h = int(hk)
                    entries = []
                    for j, m in enumerate(c.lineage[hk]):
                        aid = str(m.get("concept_id") or "").strip()
                        lab = (m.get("label") or "").strip()
                        if not aid:
                            continue
                        idx_str = f"{h}.{_alpha(j)}"  # "1.a"
                        keymap[idx_str] = {"concept_id": aid, "hop": h, "label": lab}
                        entries.append({"idx": idx_str, "label": lab})
                    if entries:
                        ancestors_by_hop[str(h)] = entries

                compact.append({
                    "case_id": case_id_token,
                    "root_label": root_label,
                    "ancestors_by_hop": ancestors_by_hop
                })
                keymaps_by_case[case_id_token] = keymap
                base_by_case[case_id_token] = c.concept_id

            # persist artifacts
            _write_json(bdir / "cases_full.json", [
                {"concept_id": c.concept_id, "lineage": c.lineage} for c in cases
            ])
            _write_json(bdir / "keymaps_by_case.json", keymaps_by_case)
            _write_json(bdir / "base_by_case.json", base_by_case)

            # build prompt
            tmpl = _load_prompt_template()
            prompt = tmpl.replace("{cases}", json.dumps(compact, ensure_ascii=False, indent=2))
            _write_json(bdir / "prompt.json", {
                "batch_index": idx+1, "offset": idx*BATCH_SIZE, "size": len(ids),
                "prompt_chars": len(prompt), "prompt": prompt
            })
            _write_json(bdir / "hierarchies.json", {c.concept_id: c.lineage for c in cases})
            return bdir, prompt, cases

        prepared = [_prepare_batch(i, ids) for i, ids in enumerate(batches)]

        # engine call with rate limit & retry
        def _call_engine(prompt: str) -> str:
            limiter.acquire()
            tries, delay = 0, 0.6
            while True:
                try:
                    return _engine_call(engine, prompt)
                except Exception:
                    tries += 1
                    if tries >= 5:
                        raise
                    time.sleep(delay + random.uniform(0,0.3))
                    delay *= 2.0

        # run threaded — map futures back to batch metadata
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            fut_to_meta = {ex.submit(_call_engine, prompt): (bdir, prompt, cases)
                           for (bdir, prompt, cases) in prepared}
            for fu in as_completed(fut_to_meta):
                bdir, _prompt, cases = fut_to_meta[fu]
                raw = fu.result()
                (bdir / "response.txt").write_text(str(raw), encoding="utf-8")
                decisions = _parse_keep_list_order_agnostic(raw)

                # load case mappings
                keymaps_by_case = json.loads((bdir / "keymaps_by_case.json").read_text(encoding="utf-8"))
                base_by_case    = json.loads((bdir / "base_by_case.json").read_text(encoding="utf-8"))

                # normalize: map alt_idx to (alt_concept_id, alt_label) per case_id
                normalized_by_case: Dict[str, List[Dict[str, Any]]] = {}
                for ent in decisions:
                    cidtok = ent.get("case_id") or ""
                    if not cidtok or cidtok not in keymaps_by_case or cidtok not in base_by_case:
                        LOGGER.warning(f"[parse] unknown/missing case_id in response: {cidtok!r}")
                        continue
                    km = keymaps_by_case[cidtok]
                    keeps_out, seen = [], set()
                    for k in (ent.get("keep") or []):
                        aid, lbl = None, None
                        idx_norm = _normalize_idx(k.get("alt_idx"))
                        if idx_norm and idx_norm in km:
                            hit = km[idx_norm]
                            aid = str(hit.get("concept_id") or "")
                            lbl = (hit.get("label") or None)
                        elif k.get("alt_key") and k["alt_key"] in km:
                            hit = km[k["alt_key"]]
                            aid = str(hit.get("concept_id") or "")
                            lbl = (hit.get("label") or None)
                        elif k.get("alt_concept_id"):
                            aid = k["alt_concept_id"]
                            lbl = _label_for(aid) or None  # optional fallback

                        if not aid or aid in seen:
                            continue
                        seen.add(aid)
                        keeps_out.append({
                            "alt_concept_id": aid,
                            "alt_label": lbl,
                            "reason": (k.get("reason") or "keep-list"),
                        })
                    normalized_by_case[cidtok] = keeps_out

                _write_json(bdir / "response.json", {"parsed_decisions": normalized_by_case})

                # helper to find hop via full lineage (already compacted to ≤6 for prompt)
                case_by_base: Dict[str, ConceptCase] = {c.concept_id: c for c in cases}
                def _hop_of(base_cid: str, alt_cid: str) -> int:
                    lin = case_by_base.get(base_cid).lineage if case_by_base.get(base_cid) else {}
                    for hk, lst in lin.items():
                        try:
                            h = int(hk)
                        except:
                            continue
                        if any(x.get("concept_id") == alt_cid for x in lst):
                            return h
                    return 0

                # persist (order-agnostic by case_id)
                n_alts = 0
                with conn:
                    for cidtok, keeps in normalized_by_case.items():
                        base_cid = base_by_case.get(cidtok)
                        if not base_cid:
                            continue
                        for k in keeps:
                            alt_cid = k.get("alt_concept_id") or ""
                            why     = k.get("reason") or "keep-list"
                            lbl     = k.get("alt_label")
                            h       = _hop_of(base_cid, alt_cid)
                            conn.execute(
                                """INSERT INTO concept_accepted_alternatives
                                   (concept_id, alt_concept_id, hop, alt_label, reason)
                                   VALUES (?,?,?,?,?)
                                   ON CONFLICT(concept_id, alt_concept_id) DO UPDATE SET
                                     hop=excluded.hop,
                                     alt_label=excluded.alt_label,
                                     reason=excluded.reason,
                                     decided_at=datetime('now')""",
                                (base_cid, alt_cid, int(h), lbl, why)
                            )
                            n_alts += 1
                LOGGER.info(f"[db] {bdir.name}: upserts={n_alts}")

                # snapshot
                ids = [c.concept_id for c in cases]
                rows = list(conn.execute(
                    f"SELECT concept_id, alt_concept_id, hop, alt_label, reason, decided_at "
                    f"FROM concept_accepted_alternatives WHERE concept_id IN ({','.join(['?']*len(ids))}) "
                    f"ORDER BY concept_id, hop, alt_concept_id", ids))
                _write_json(bdir / "accepted_alternatives.json", [
                    dict(concept_id=r[0], alt_concept_id=r[1], hop=int(r[2]), alt_label=r[3],
                         reason=r[4], decided_at=r[5]) for r in rows
                ])

    # Expanded vars & stems table (positive literals) — shared by both modes
    LOGGER.info("[expand] building base/lifted vars for positive literals and timeframe-stripped stems …")
    _rebuild_expanded(conn, ARTIFACT_DIR, include_assumed=include_assumed)

    conn.close()
    LOGGER.info("[done] keep-list decisions persisted and expanded rows materialized.")

# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Flattened keep-list decider (index-driven + case_id) over positive_constraint_literals.")
    ap.add_argument("--db", required=True, help="SQLite DB path (e.g., ../../build/trial.db)")
    ap.add_argument("--include-assumed", action="store_true", default=True,
                    help="Also include rows where variant='assumed' from positive_constraint_literals (for relevance).")
    ap.add_argument("--keep-all-ancestors", action="store_true",
                    help="Skip LLM calls and keep all ancestors from lineage as accepted alternatives.")
    args = ap.parse_args()
    decide_and_persist(
        args.db,
        include_assumed=args.include_assumed,
        keep_all_ancestors=args.keep_all_ancestors,
    )

if __name__ == "__main__":
    main()