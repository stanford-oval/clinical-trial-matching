#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ontology_lifter.py — inclusion + assumed ontology lifting (decider-first, keep-list only)

Assumes concept_accepted_alternatives (keep-list) is already populated by the decider.

Writes:
- predicate_to_concept(var_name PRIMARY KEY, concept_id TEXT)

- constraint_lifted_atoms(
    trial_id, clause_id, literal_index,
    var_name, lifted_var, hop,
    base_var, timeframe,
    lifted_concept_id, base_var_stem, lifted_var_stem,
    usage_scope NULL|'relevance_only',
    PRIMARY KEY (trial_id, clause_id, literal_index, lifted_var)
  )

- constraint_literal_alternatives(
    trial_id, clause_id, literal_index, timeframe,
    alt_concept_id, hop, reason, decided_at,
    lifted_var, base_var,
    base_var_stem, lifted_var_stem,
    usage_scope NULL|'relevance_only',
    PRIMARY KEY (trial_id, clause_id, literal_index, timeframe, alt_concept_id)
  )

Behavior:
- For each positive literal on trial_constraint_sides:
    * Always insert a self row (hop=0) in both tables (when we can map a concept).
    * Inclusion: also add ancestors with usage_scope=NULL (normal).
    * Assumed: optionally add ancestors tagged usage_scope='relevance_only' (relevance-only).

CLI:
  --db <path>                               (required)
  --trial <id>                              (optional; limit to one trial_constraint_sides.id)
  --rebuild-predicate_to_concept [--overwrite-predicate_to_concept]
  --canon-dir <dir> | --minified-canon-dir <dir>
  --lift [--overwrite-lifts]
  --include-assumed
  --assumed-ancestors-for-relevance
  --dryrun
"""

from __future__ import annotations

import os, re, sqlite3, json, sys
from typing import Dict, Iterable, List, Optional, Tuple
from collections import deque
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# Snowstorm client (cached)
# ────────────────────────────────────────────────────────────────
SNOWSTORM_BASE   = os.getenv("SNOWSTORM_BASE", "http://localhost:8080").rstrip("/")
SNOWSTORM_BRANCH = os.getenv("SNOWSTORM_BRANCH", "MAIN")
SNOWSTORM_FORM   = os.getenv("SNOWSTORM_FORM", "inferred")
HTTP_TIMEOUT_S   = float(os.getenv("SNOW_TIMEOUT", "6.0"))

_PARENTS_CACHE: Dict[str, List[dict]]   = {}
_ANCESTORS_CACHE: Dict[str, List[dict]] = {}
_LABEL_CACHE: Dict[str, str]            = {}

def _http_get_json(url: str) -> dict | list | None:
    try:
        import requests  # type: ignore
        resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:
                if r.status == 200:
                    return _json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    return None

def mini_to_row(raw: dict) -> dict:
    cid = str(raw.get("conceptId") or raw.get("id") or "").strip()
    pt  = (raw.get("pt") or {}).get("term")
    fsn = (raw.get("fsn") or {}).get("term")
    pref = raw.get("preferredTerm") or raw.get("preferred_term") or raw.get("term") or pt or fsn
    return {"conceptId": cid, "preferred_term": pref, "fully_specified_name": fsn}

def get_concept_label(concept_id: str) -> Optional[str]:
    if not concept_id: return None
    if concept_id in _LABEL_CACHE: return _LABEL_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{concept_id}"
    js = _http_get_json(url)
    if isinstance(js, dict):
        pt  = (js.get("pt") or {}).get("term")
        fsn = (js.get("fsn") or {}).get("term")
        lab = js.get("preferredTerm") or pt or fsn or js.get("term")
        if lab:
            _LABEL_CACHE[concept_id] = lab
            return lab
    return None

def _enrich_label(raw_mini: dict) -> dict:
    row = dict(mini_to_row(raw_mini) or {})
    cid = str(row.get("conceptId") or "").strip()
    if not row.get("preferred_term") and cid:
        lab = get_concept_label(cid)
        if lab: row["preferred_term"] = lab
    return row

def parents_cached(branch: str, form: str, concept_id: str) -> List[dict]:
    if concept_id in _PARENTS_CACHE: return _PARENTS_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{branch}/concepts/{concept_id}/parents?form={form}"
    js = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"): out.append(row)
    _PARENTS_CACHE[concept_id] = out
    return out

def ancestors_cached(branch: str, form: str, concept_id: str) -> List[dict]:
    if concept_id in _ANCESTORS_CACHE: return _ANCESTORS_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{branch}/concepts/{concept_id}/ancestors?form={form}"
    js = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"): out.append(row)
    _ANCESTORS_CACHE[concept_id] = out
    return out

# ────────────────────────────────────────────────────────────────
# Hop computation (diagnostics only)
# ────────────────────────────────────────────────────────────────
_PARENTS_IDS_CACHE: Dict[str, List[str]] = {}

def _parent_ids(cid: str) -> List[str]:
    if cid in _PARENTS_IDS_CACHE: return _PARENTS_IDS_CACHE[cid]
    ids = [str(m.get("conceptId")) for m in (parents_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, cid) or []) if m.get("conceptId")]
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
            if p in targets and p not in found: found[p] = hop
            q.append((p, hop))
    return found

# ────────────────────────────────────────────────────────────────
# Templates (spec-complete)
# ────────────────────────────────────────────────────────────────
TEMPLATES = {
    "findings": [
        "patient_has_diagnosis_of_{entity_canonical_form}_{timeframe}",
        "patient_has_finding_of_{entity_canonical_form}_{timeframe}",
        "patient_has_symptoms_of_{entity_canonical_form}_{timeframe}",
        "patient_has_clinical_signs_of_{entity_canonical_form}_{timeframe}",
        "patient_has_suspicion_of_{entity_canonical_form}_{timeframe}",
    ],
    "procedures": [
        "patient_has_undergone_{entity_canonical_form}_{timeframe}",
        "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_positive",
        "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_negative",
        "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_normal",
        "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_abnormal",
        "patient_is_undergoing_{entity_canonical_form}_{timeframe}",
        "patient_will_undergo_{entity_canonical_form}_{timeframe}",
        "patient_can_undergo_{entity_canonical_form}_{timeframe}",
    ],
    "observable_entities_numeric": [
        "patient_{entity_canonical_form}_value_recorded_{timeframe}_withunit_{unit}",
    ],
    "observable_entities_status": [
        "patients_{entity_canonical_form}_is_positive_{timeframe}",
        "patients_{entity_canonical_form}_is_negative_{timeframe}",
        "patients_{entity_canonical_form}_is_normal_{timeframe}",
        "patients_{entity_canonical_form}_is_abnormal_{timeframe}",
    ],
    "product": [
        "patient_is_taking_{entity_canonical_form}_{timeframe}",
        "patient_has_taken_{entity_canonical_form}_{timeframe}",
        "patient_has_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
        "patient_has_intolerance_to_{entity_canonical_form}_{timeframe}",
        "patient_has_allergy_to_{entity_canonical_form}_{timeframe}",
        "patient_has_nonimmune_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
    ],
    "substance": [
        "patient_is_exposed_to_{entity_canonical_form}_{timeframe}",
        "patient_has_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
        "patient_has_intolerance_to_{entity_canonical_form}_{timeframe}",
        "patient_has_allergy_to_{entity_canonical_form}_{timeframe}",
        "patient_has_nonimmune_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
    ],
}
ALL_TEMPLATES: List[str] = [tpl for group in TEMPLATES.values() for tpl in group]

import re as _re
def _canonize_entity_form(term: Optional[str]) -> str:
    s = (term or "").strip().lower()
    s = _re.sub(r"[^a-z0-9]+", "_", s)
    s = _re.sub(r"_+", "_", s)
    return s.strip("_")

def _template_to_regex(tpl: str):
    pat = _re.escape(tpl)
    pat = pat.replace(_re.escape("{entity_canonical_form}"), r"(?P<entity>[a-z0-9_]+)")
    pat = pat.replace(_re.escape("{timeframe}"), r"(?P<tf>[a-z0-9_]+)")
    pat = pat.replace(_re.escape("{unit}"), r"(?P<unit>[a-z0-9_]+)")
    return _re.compile(rf"^{pat}(?:@@[a-z0-9_]+)*$", _re.I)

_TEMPLATE_REGEXES = [(tpl, _template_to_regex(tpl)) for tpl in ALL_TEMPLATES]
_QUAL_RX = _re.compile(r"(?:@@[a-z0-9_]+)+$", _re.I)
def _split_qual_suffix(vn: str) -> tuple[str, str]:
    m = _QUAL_RX.search(vn or "")
    if not m: return vn, ""
    return vn[:m.start()], m.group(0)

def _detect_template_from_varname(varname: str) -> Optional[Tuple[str, Dict[str, str]]]:
    vn = (varname or "").strip()
    base_vn, qual = _split_qual_suffix(vn)
    if not (base_vn.startswith("patient_") or base_vn.startswith("patients_")):
        return None
    for tpl, rgx in _TEMPLATE_REGEXES:
        m = rgx.match(base_vn)
        if m:
            g = m.groupdict()
            if not g.get("tf"): return None
            g["qual_suffix"] = qual
            return tpl, {"entity": g.get("entity",""), "tf": g.get("tf",""), "unit": g.get("unit",""), "qual_suffix": qual}
    return None

def _compose_var_name(tpl: str, entity_form: str, timeframe: str, unit: str, *, qual_suffix: str = "") -> str:
    name = (tpl or "").replace("{entity_canonical_form}", entity_form).replace("{timeframe}", timeframe).replace("{unit}", unit or "")
    name = _re.sub(r"__+", "_", name).strip("_")
    return (name + (qual_suffix or "")).strip("_")

# ────────────────────────────────────────────────────────────────
# Timeframe-stripped stems
# ────────────────────────────────────────────────────────────────
_TIMEFRAME_TOKEN_RX = re.compile(
    r"(?:^|_)(?:"
    r"now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years)"
    r")(?:_|$)"
)

def _strip_timeframe_once(stem: str) -> str:
    """Remove exactly ONE timeframe token per schema; clean underscores."""
    if not stem:
        return stem
    m = _TIMEFRAME_TOKEN_RX.search(stem)
    if not m:
        return stem
    start, end = m.span()
    out = stem[:start] + stem[end:]
    out = re.sub(r"_+", "_", out).strip("_")
    return out

# ────────────────────────────────────────────────────────────────
# Canon/minified loaders + predicate_to_concept
# ────────────────────────────────────────────────────────────────
def _unique_preserve_order(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set(); out: List[str] = []
    for x in seq:
        if x not in seen: out.append(x); seen.add(x)
    return out

def _iter_canon_files(canon_dir: Optional[str]) -> Iterable[Path]:
    if not canon_dir: return []
    p = Path(canon_dir)
    if not p.exists(): return []
    yield from sorted(p.glob("*.json"))

def _load_predicate_to_concept_from_canon(canon_dir: Optional[str]) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for f in _iter_canon_files(canon_dir):
        try:
            js = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = None
        if isinstance(js, dict):
            for key in ("canonical_variables", "new_canonical_variable_declarations"):
                v = js.get(key)
                if isinstance(v, list):
                    items = v; break
        elif isinstance(js, list):
            items = js
        if not items: continue
        for it in items:
            if not isinstance(it, dict): continue
            vn = it.get("entity_variable_name"); cid = it.get("concept_id")
            if isinstance(vn, str) and vn.strip() and isinstance(cid, (str, int)):
                results.append((vn.strip(), str(cid)))
    # de-dup by first occurrence
    seen: set[str] = set(); dedup: List[Tuple[str, str]] = []
    for vn, cid in results:
        if vn not in seen:
            dedup.append((vn, cid)); seen.add(vn)
    return dedup

def _load_minified_names(minified_dir: Optional[str]) -> List[str]:
    if not minified_dir: return []
    base = Path(minified_dir); out: List[str] = []
    if not base.exists(): return out
    for f in sorted(base.glob("*_entity_variable_names.json")):
        try:
            arr = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(arr, list):
                out.extend([x for x in arr if isinstance(x, str) and x.strip()])
        except Exception: pass
    flat = base / "_all_entity_variable_names.txt"
    if flat.exists():
        try:
            out.extend([ln.strip() for ln in flat.read_text(encoding="utf-8").splitlines() if ln.strip()])
        except Exception: pass
    return _unique_preserve_order(out)

def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.execute("""CREATE TABLE IF NOT EXISTS predicate_to_concept (var_name TEXT PRIMARY KEY, concept_id TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS constraint_lifted_atoms (
                     trial_id INTEGER NOT NULL, clause_id INTEGER NOT NULL, literal_index INTEGER NOT NULL,
                     var_name TEXT NOT NULL, lifted_var TEXT NOT NULL, hop INTEGER NOT NULL,
                     base_var TEXT, timeframe TEXT,
                     lifted_concept_id TEXT,
                     base_var_stem TEXT, lifted_var_stem TEXT,
                     usage_scope TEXT,
                     PRIMARY KEY (trial_id, clause_id, literal_index, lifted_var))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS constraint_literal_alternatives (
          trial_id       INTEGER NOT NULL,
          clause_id      INTEGER NOT NULL,
          literal_index  INTEGER NOT NULL,
          timeframe      TEXT    NOT NULL,
          alt_concept_id TEXT    NOT NULL,
          hop            INTEGER NOT NULL,
          reason         TEXT,
          decided_at     TEXT DEFAULT (datetime('now')),
          lifted_var     TEXT,
          base_var       TEXT,
          base_var_stem  TEXT,
          lifted_var_stem TEXT,
          usage_scope    TEXT,
          PRIMARY KEY (trial_id, clause_id, literal_index, timeframe, alt_concept_id)
    )""")
    # Back-compat: make sure new columns exist if tables pre-existed
    def _ensure_column(table: str, column: str, decl: str):
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    _ensure_column("constraint_lifted_atoms", "lifted_concept_id", "TEXT")
    _ensure_column("constraint_lifted_atoms", "base_var_stem", "TEXT")
    _ensure_column("constraint_lifted_atoms", "lifted_var_stem", "TEXT")
    _ensure_column("constraint_lifted_atoms", "usage_scope", "TEXT")
    _ensure_column("constraint_literal_alternatives", "lifted_var", "TEXT")
    _ensure_column("constraint_literal_alternatives", "base_var", "TEXT")
    _ensure_column("constraint_literal_alternatives", "base_var_stem", "TEXT")
    _ensure_column("constraint_literal_alternatives", "lifted_var_stem", "TEXT")
    _ensure_column("constraint_literal_alternatives", "usage_scope", "TEXT")
    conn.commit()

    # Convenience view for relevance experiments
    cur.execute("DROP VIEW IF EXISTS vw_assumed_relevance_lifts")
    cur.execute("""
      CREATE VIEW IF NOT EXISTS vw_assumed_relevance_lifts AS
      SELECT laa.*
      FROM constraint_literal_alternatives laa
      JOIN trial_constraint_clauses tsc ON tsc.trial_id = laa.trial_id AND tsc.clause_id = laa.clause_id
      JOIN trial_constraint_sides t ON t.id = tsc.trial_id
      WHERE t.variant='assumed' AND laa.usage_scope='relevance_only'
    """)
    conn.commit()

def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)

# ────────────────────────────────────────────────────────────────
# predicate_to_concept population
# ────────────────────────────────────────────────────────────────
_RX_SCT_PREFIXED = re.compile(r"(?:^|_)(?:sctid|snomed|sct|sctid-)(?:[:_])?(?P<id>\d{5,18})(?:_|$)", re.I)
_RX_RAW_SCTID    = re.compile(r"(?:^|_)(?P<id>\d{6,18})(?:_|$)")

def _extract_concept_id_from_entity(entity_form: str) -> Optional[str]:
    s = (entity_form or "").lower()
    m = _RX_SCT_PREFIXED.search(s)
    if m: return m.group("id")
    m = _RX_RAW_SCTID.search(s)
    if m: return m.group("id")
    return None

def _detect_template_and_groups(var_name: str) -> Optional[Tuple[str, Dict[str,str]]]:
    return _detect_template_from_varname(var_name)

def rebuild_predicate_to_concept(conn: sqlite3.Connection,
                        *, trial_id: Optional[int]=None,
                        canon_dir: Optional[str]=None,
                        minified_canon_dir: Optional[str]=None) -> int:
    cur = conn.cursor()
    ensure_schema(conn)

    pairs = _load_predicate_to_concept_from_canon(canon_dir)
    if pairs:
        before = cur.execute("SELECT COUNT(*) FROM predicate_to_concept").fetchone()[0] or 0
        cur.executemany("INSERT OR IGNORE INTO predicate_to_concept(var_name, concept_id) VALUES (?,?)", pairs)
        conn.commit()
        after = cur.execute("SELECT COUNT(*) FROM predicate_to_concept").fetchone()[0] or 0
        return max(0, after - before)

    min_names = set(_load_minified_names(minified_canon_dir))
    inserted = 0

    def _iter_positive_vars(cur: sqlite3.Cursor, tid: Optional[int]) -> Iterable[str]:
        if tid is None:
            q = """SELECT DISTINCT cl.var_name
                   FROM trial_constraint_clauses tsc
                   JOIN constraint_clause_atoms cl ON cl.clause_id=tsc.clause_id
                   WHERE cl.is_neg=0"""
            params = ()
        else:
            q = """SELECT DISTINCT cl.var_name
                   FROM trial_constraint_clauses tsc
                   JOIN constraint_clause_atoms cl ON cl.clause_id=tsc.clause_id
                   WHERE tsc.trial_id=? AND cl.is_neg=0"""
            params = (tid,)
        for (vn,) in cur.execute(q, params):
            yield vn

    for var_name in _iter_positive_vars(cur, trial_id):
        if min_names and var_name not in min_names: continue
        det = _detect_template_and_groups(var_name)
        if not det: continue
        _, groups = det
        cid = _extract_concept_id_from_entity(groups.get("entity", ""))
        if not cid: continue
        cur.execute("INSERT OR IGNORE INTO predicate_to_concept(var_name, concept_id) VALUES (?,?)", (var_name, cid))
        if cur.rowcount: inserted += 1
    conn.commit(); return inserted

# ────────────────────────────────────────────────────────────────
# Lifting helpers
# ────────────────────────────────────────────────────────────────
def _compose_from_label(base_var: str, alt_label: str, timeframe_fallback: str="") -> Optional[str]:
    det = _detect_template_from_varname(base_var)
    if not det: return None
    tpl, g = det
    tf  = g.get("tf") or timeframe_fallback or ""
    unit = g.get("unit") or ""
    ent = _canonize_entity_form(alt_label)
    if not ent or not tf: return None
    return _compose_var_name(tpl, ent, tf, unit, qual_suffix=g.get("qual_suffix",""))

def _hop_between(base_cid: str, alt_cid: str) -> int:
    minis = (parents_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, base_cid) or []) + \
            (ancestors_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, base_cid) or [])
    ids = [str(m.get("conceptId")) for m in minis if m.get("conceptId")]
    hop_map = compute_upward_hops(base_cid, ids)
    return int(hop_map.get(str(alt_cid), 0))

def _fetch_kept_ancestors(cur: sqlite3.Cursor, *, base_cid: str, base_var: str, tf: str) -> List[Tuple[str,str,str,int,str,str]]:
    """
    Returns list of (alt_cid, alt_label, reason, hop, lifted_var, lifted_var_stem)
    """
    out: List[Tuple[str,str,str,int,str,str]] = []
    for alt in cur.execute("""
        SELECT caa.alt_concept_id, COALESCE(caa.alt_label,''), COALESCE(caa.reason,'keep-list') AS reason
        FROM concept_accepted_alternatives caa
        WHERE caa.concept_id = ?
        ORDER BY caa.hop, caa.alt_concept_id
    """, (base_cid,)).fetchall():
        alt_cid, alt_label, reason = str(alt[0]), str(alt[1]), str(alt[2])
        lifted_var = _compose_from_label(base_var, alt_label, timeframe_fallback=tf)
        if not lifted_var:
            continue
        lifted_var_stem = _strip_timeframe_once(lifted_var)
        hop = _hop_between(base_cid, alt_cid)  # diagnostic ordering only
        out.append((alt_cid, alt_label, reason, hop, lifted_var, lifted_var_stem))
    return out

# ────────────────────────────────────────────────────────────────
# Trial iteration & lifting
# ────────────────────────────────────────────────────────────────
def _iter_positive_constraint_literals_for_trial(cur: sqlite3.Cursor, trial_id: int):
    q = """
    SELECT t.kind, t.variant, c.id AS clause_id, cl.literal_index, cl.var_name,
           vc.base_var, vc.timeframe
    FROM trial_constraint_clauses tsc
    JOIN trial_constraint_sides t  ON t.id = tsc.trial_id
    JOIN constraint_clauses c      ON c.id = tsc.clause_id
    JOIN constraint_clause_atoms cl ON cl.clause_id = c.id AND cl.is_neg = 0
    LEFT JOIN predicate_catalog vc ON vc.var_name = cl.var_name
    WHERE tsc.trial_id = ?
    ORDER BY c.id, cl.literal_index
    """
    for row in cur.execute(q, (trial_id,)):
        kind, variant, cid, idx, vname, base_var, tf = row
        yield (str(kind), str(variant), int(cid), int(idx), str(vname),
               (base_var if base_var is not None else None),
               (tf if tf is not None else ""))

def build_lifted_for_trials(conn: sqlite3.Connection, *,
                            trial_id: Optional[int]=None,
                            include_assumed: bool = True,
                            include_assumed_ancestors_for_relevance: bool = True) -> int:
    """
    - Always inserts hop=0 self rows for inclusion and assumed.
    - Inclusion: materializes keep-list ancestors (usage_scope=NULL).
    - Assumed: if include_assumed_ancestors_for_relevance=True, materializes keep-list
      ancestors with usage_scope='relevance_only'.
    Returns number of self rows inserted into constraint_lifted_atoms.
    """
    ensure_schema(conn)
    cur = conn.cursor()
    keep_exists = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='concept_accepted_alternatives'").fetchone())
    if not keep_exists:
        print("[lifter] ERROR: concept_accepted_alternatives (keep-list) not found. Run the decider first.", file=sys.stderr)
        return 0

    if trial_id is None:
        trial_ids = [r[0] for r in cur.execute("SELECT id FROM trial_constraint_sides").fetchall()]
    else:
        trial_ids = [int(trial_id)]

    total_self = 0

    for tid in trial_ids:
        rows = list(_iter_positive_constraint_literals_for_trial(cur, tid))
        if not rows:
            continue

        insert_vals_llm = []

        for kind, variant, clause_id, lit_idx, var_name, base_var, tf in rows:
            det = _detect_template_from_varname(var_name)
            tf_from_name = det[1]["tf"] if det else ""
            tf_eff = (tf or tf_from_name or "")

            base_effective = base_var or var_name
            base_var_stem = _strip_timeframe_once(base_effective)
            lifted_var_stem = _strip_timeframe_once(var_name)

            base_cid_row = cur.execute("SELECT concept_id FROM predicate_to_concept WHERE var_name=?", (var_name,)).fetchone()
            base_cid = (base_cid_row[0] if base_cid_row else None)

            # (1) SELF → constraint_lifted_atoms  (usage_scope=NULL)
            insert_vals_llm.append((tid, clause_id, lit_idx, var_name, var_name, 0,
                                    base_var, tf_eff, base_cid, base_var_stem, lifted_var_stem, None))

            # (2) SELF → constraint_literal_alternatives (only if we can map concept)
            if base_cid:
                cur.execute("""
                    INSERT OR REPLACE INTO constraint_literal_alternatives
                      (trial_id, clause_id, literal_index, timeframe,
                       alt_concept_id, hop, reason, lifted_var, base_var,
                       base_var_stem, lifted_var_stem, usage_scope)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?, NULL)
                """, (tid, clause_id, lit_idx, tf_eff, base_cid, 0, "self", var_name, var_name,
                      base_var_stem, lifted_var_stem))

            # (3) Ancestors
            if base_cid:
                kept = _fetch_kept_ancestors(cur, base_cid=base_cid, base_var=base_effective, tf=tf_eff)
                if kind == "inclusion":
                    for alt_cid, alt_label, reason, hop, lifted_var, lifted_stem in kept:
                        cur.execute("""
                            INSERT OR REPLACE INTO constraint_literal_alternatives
                              (trial_id, clause_id, literal_index, timeframe,
                               alt_concept_id, hop, reason, lifted_var, base_var,
                               base_var_stem, lifted_var_stem, usage_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, NULL)
                        """, (tid, clause_id, lit_idx, tf_eff, alt_cid, hop, reason,
                              lifted_var, base_effective, base_var_stem, lifted_stem))
                elif include_assumed and (variant == "assumed") and include_assumed_ancestors_for_relevance:
                    for alt_cid, alt_label, reason, hop, lifted_var, lifted_stem in kept:
                        cur.execute("""
                            INSERT OR REPLACE INTO constraint_literal_alternatives
                              (trial_id, clause_id, literal_index, timeframe,
                               alt_concept_id, hop, reason, lifted_var, base_var,
                               base_var_stem, lifted_var_stem, usage_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, 'relevance_only')
                        """, (tid, clause_id, lit_idx, tf_eff, alt_cid, hop, reason or 'keep-list',
                              lifted_var, base_effective, base_var_stem, lifted_stem))

        if insert_vals_llm:
            cur.executemany("""
                INSERT OR IGNORE INTO constraint_lifted_atoms
                   (trial_id,clause_id,literal_index,var_name,lifted_var,hop,base_var,timeframe,
                    lifted_concept_id,base_var_stem,lifted_var_stem,usage_scope)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, insert_vals_llm)
            total_self += cur.rowcount or 0

        conn.commit()

    return total_self

# ────────────────────────────────────────────────────────────────
# Backwards-compatible wrapper (keeps old CLI working)
# ────────────────────────────────────────────────────────────────
def build_lifted_for_inclusion(conn: sqlite3.Connection, *, trial_id: Optional[int]=None) -> int:
    """
    Legacy shim: performs inclusion-only behavior (self + ancestors; no assumed).
    """
    return build_lifted_for_trials(conn,
                                   trial_id=trial_id,
                                   include_assumed=False,
                                   include_assumed_ancestors_for_relevance=False)

# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def _open_core(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row; return conn

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ontology lifting (inclusion + optional assumed; keep-list only).")
    ap.add_argument("--db", required=True, help="SQLite DB (e.g., ../../build/trial.db)")
    ap.add_argument("--trial", type=int, default=None, help="Only process this trial_constraint_sides.id")
    ap.add_argument("--rebuild-predicate_to_concept", action="store_true",
                    help="(Re)populate predicate_to_concept (from --canon-dir if provided; else fallback to DB extractor).")
    ap.add_argument("--overwrite-predicate_to_concept", action="store_true",
                    help="Delete predicate_to_concept before rebuilding (use with --rebuild-predicate_to_concept).")
    ap.add_argument("--canon-dir", type=str, default=None, help="Directory of canon JSON files.")
    ap.add_argument("--minified-canon-dir", type=str, default=None, help="Directory of minified_canon files.")
    ap.add_argument("--lift", action="store_true", help="Materialize constraint_lifted_atoms and constraint_literal_alternatives")
    ap.add_argument("--overwrite-lifts", action="store_true",
                    help="Delete constraint_lifted_atoms and constraint_literal_alternatives before lifting.")
    ap.add_argument("--include-assumed", action="store_true", default = True,
                    help="Also process trial_constraint_sides where variant='assumed'.")
    ap.add_argument("--assumed-ancestors-for-relevance", action="store_true",  default = True,
                    help="For assumed sides, also write ancestor lifts with usage_scope='relevance_only'.")
    ap.add_argument("--dryrun", action="store_true", help="Run but rollback DB writes")
    args = ap.parse_args()

    conn = _open_core(args.db)
    try:
        ensure_schema(conn)

        if args.rebuild_predicate_to_concept:
            if args.overwrite_predicate_to_concept:
                conn.execute("DELETE FROM predicate_to_concept")
                conn.commit()
            n = rebuild_predicate_to_concept(conn,
                                    trial_id=args.trial,
                                    canon_dir=args.canon_dir,
                                    minified_canon_dir=args.minified_canon_dir)
            print(f"[lifter] predicate_to_concept inserted: {n}")

        if args.lift:
            if args.overwrite_lifts:
                conn.execute("DELETE FROM constraint_lifted_atoms")
                conn.execute("DELETE FROM constraint_literal_alternatives")
                conn.commit()
            if args.dryrun:
                print("[lifter] dryrun...")
                conn.isolation_level = "DEFERRED"; conn.execute("BEGIN")
                _ = build_lifted_for_trials(conn,
                                            trial_id=args.trial,
                                            include_assumed=args.include_assumed,
                                            include_assumed_ancestors_for_relevance=args.assumed_ancestors_for_relevance)
                conn.execute("ROLLBACK"); print("[lifter] dryrun complete.")
            else:
                ins = build_lifted_for_trials(conn,
                                              trial_id=args.trial,
                                              include_assumed=args.include_assumed,
                                              include_assumed_ancestors_for_relevance=args.assumed_ancestors_for_relevance)
                print(f"[lifter] constraint_lifted_atoms (self) inserted: {ins}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
