#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ontology_lifter.py — inclusion + assumed ontology lifting (decider-style construction, keep-list only)

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

Important change vs old version:
- NO longer reconstructs lifted vars by regex-parsing base_var.
- Instead mirrors hop_policy_decider_lineage_mgsr.py:
    * concept_id -> label
    * explicit timeframe
    * semantic prefix choice
    * then strip timeframe for *_stem fields

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

import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────
# Snowstorm client (cached)
# ────────────────────────────────────────────────────────────────
SNOWSTORM_BASE = os.getenv("SNOWSTORM_BASE", "http://localhost:8080").rstrip("/")
SNOWSTORM_BRANCH = os.getenv("SNOWSTORM_BRANCH", "MAIN")
SNOWSTORM_FORM = os.getenv("SNOWSTORM_FORM", "inferred")
HTTP_TIMEOUT_S = float(os.getenv("SNOW_TIMEOUT", "6.0"))

_PARENTS_CACHE: Dict[str, List[dict]] = {}
_ANCESTORS_CACHE: Dict[str, List[dict]] = {}
_LABEL_CACHE: Dict[str, str] = {}
_FSN_CACHE: Dict[str, str] = {}
_PARENTS_IDS_CACHE: Dict[str, List[str]] = {}


def _http_get_json(url: str) -> dict | list | None:
    try:
        import requests  # type: ignore
        resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        try:
            import urllib.request
            import json as _json
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:
                if r.status == 200:
                    return _json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    return None


def _concept_json(concept_id: str) -> Optional[dict]:
    if not concept_id:
        return None
    url = f"{SNOWSTORM_BASE}/browser/{SNOWSTORM_BRANCH}/concepts/{concept_id}"
    js = _http_get_json(url)
    return js if isinstance(js, dict) else None


def mini_to_row(raw: dict) -> dict:
    cid = str(raw.get("conceptId") or raw.get("id") or "").strip()
    pt = (raw.get("pt") or {}).get("term")
    fsn = (raw.get("fsn") or {}).get("term")
    pref = raw.get("preferredTerm") or raw.get("preferred_term") or raw.get("term") or pt or fsn
    return {"conceptId": cid, "preferred_term": pref, "fully_specified_name": fsn}


def get_concept_label(concept_id: str) -> Optional[str]:
    if not concept_id:
        return None
    if concept_id in _LABEL_CACHE:
        return _LABEL_CACHE[concept_id]
    js = _concept_json(concept_id)
    if isinstance(js, dict):
        pt = (js.get("pt") or {}).get("term")
        fsn = (js.get("fsn") or {}).get("term")
        lab = js.get("preferredTerm") or pt or fsn or js.get("term")
        if lab:
            _LABEL_CACHE[concept_id] = lab
            return lab
    return None


def _fsn_for(concept_id: str) -> Optional[str]:
    if not concept_id:
        return None
    if concept_id in _FSN_CACHE:
        return _FSN_CACHE[concept_id]
    js = _concept_json(concept_id)
    if isinstance(js, dict):
        fsn = (js.get("fsn") or {}).get("term")
        if fsn:
            _FSN_CACHE[concept_id] = fsn
            return fsn
    return None


def _enrich_label(raw_mini: dict) -> dict:
    row = dict(mini_to_row(raw_mini) or {})
    cid = str(row.get("conceptId") or "").strip()
    if not row.get("preferred_term") and cid:
        lab = get_concept_label(cid)
        if lab:
            row["preferred_term"] = lab
    return row


def parents_cached(branch: str, form: str, concept_id: str) -> List[dict]:
    if concept_id in _PARENTS_CACHE:
        return _PARENTS_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{branch}/concepts/{concept_id}/parents?form={form}"
    js = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"):
                out.append(row)
    _PARENTS_CACHE[concept_id] = out
    return out


def ancestors_cached(branch: str, form: str, concept_id: str) -> List[dict]:
    if concept_id in _ANCESTORS_CACHE:
        return _ANCESTORS_CACHE[concept_id]
    url = f"{SNOWSTORM_BASE}/browser/{branch}/concepts/{concept_id}/ancestors?form={form}"
    js = _http_get_json(url)
    out: List[dict] = []
    if isinstance(js, list):
        for m in js:
            row = _enrich_label(m)
            if row.get("conceptId"):
                out.append(row)
    _ANCESTORS_CACHE[concept_id] = out
    return out


# ────────────────────────────────────────────────────────────────
# Hop computation (diagnostics only)
# ────────────────────────────────────────────────────────────────
def _parent_ids(cid: str) -> List[str]:
    if cid in _PARENTS_IDS_CACHE:
        return _PARENTS_IDS_CACHE[cid]
    ids = [
        str(m.get("conceptId"))
        for m in (parents_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, cid) or [])
        if m.get("conceptId")
    ]
    _PARENTS_IDS_CACHE[cid] = ids
    return ids


def compute_upward_hops(start_cid: str, target_ids: List[str]) -> Dict[str, int]:
    targets = set(target_ids)
    found: Dict[str, int] = {}
    seen = {start_cid}
    q: deque[Tuple[str, int]] = deque([(start_cid, 0)])
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


# ────────────────────────────────────────────────────────────────
# Timeframe stripping + decider-style var reconstruction
# ────────────────────────────────────────────────────────────────
_TIMEFRAME_TOKEN_RX = re.compile(
    r"(?:(?<=^)|(?<=_))"
    r"(?:(?:now|inthehistory|inthefuture)|"
    r"(?:inthepast|inthefuture)\d+(?:minute(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?)|"
    r"foradurationof\d+(?:minute(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?))"
    r"(?=(?:_|$))"
)

_QUAL_RX = re.compile(r"(?:@@[a-z0-9_]+)+$", re.I)


def _split_qual_suffix(vn: str) -> tuple[str, str]:
    m = _QUAL_RX.search(vn or "")
    if not m:
        return vn, ""
    return vn[:m.start()], m.group(0)


def _extract_qual_suffix(vn: str) -> str:
    _, qual = _split_qual_suffix(vn or "")
    return qual


def _strip_timeframe_once(stem: str) -> str:
    if not stem:
        return stem
    base, qual = _split_qual_suffix(stem)
    out = _TIMEFRAME_TOKEN_RX.sub("", base, count=1)
    out = re.sub(r"_+", "_", out).strip("_")
    return out + qual


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


def _build_var_for_concept(cid: str, timeframe: str, *, qual_suffix: str = "") -> str:
    lab = get_concept_label(cid) or ""
    entity = _slug_entity(lab) or f"concept_{cid}"
    tf = timeframe if timeframe else "now"
    prefix = _choose_prefix(cid, lab)
    return f"{prefix}_{entity}_{tf}{qual_suffix or ''}"


# ────────────────────────────────────────────────────────────────
# Canon/minified loaders + predicate_to_concept
# ────────────────────────────────────────────────────────────────
def _unique_preserve_order(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _iter_canon_files(canon_dir: Optional[str]) -> Iterable[Path]:
    if not canon_dir:
        return []
    p = Path(canon_dir)
    if not p.exists():
        return []
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
                    items = v
                    break
        elif isinstance(js, list):
            items = js
        if not items:
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            vn = it.get("entity_variable_name")
            cid = it.get("concept_id")
            if isinstance(vn, str) and vn.strip() and isinstance(cid, (str, int)):
                results.append((vn.strip(), str(cid)))

    seen: set[str] = set()
    dedup: List[Tuple[str, str]] = []
    for vn, cid in results:
        if vn not in seen:
            dedup.append((vn, cid))
            seen.add(vn)
    return dedup


def _load_minified_names(minified_dir: Optional[str]) -> List[str]:
    if not minified_dir:
        return []
    base = Path(minified_dir)
    out: List[str] = []
    if not base.exists():
        return out
    for f in sorted(base.glob("*_entity_variable_names.json")):
        try:
            arr = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(arr, list):
                out.extend([x for x in arr if isinstance(x, str) and x.strip()])
        except Exception:
            pass
    flat = base / "_all_entity_variable_names.txt"
    if flat.exists():
        try:
            out.extend([ln.strip() for ln in flat.read_text(encoding="utf-8").splitlines() if ln.strip()])
        except Exception:
            pass
    return _unique_preserve_order(out)


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predicate_to_concept (
            var_name TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS constraint_lifted_atoms (
            trial_id INTEGER NOT NULL,
            clause_id INTEGER NOT NULL,
            literal_index INTEGER NOT NULL,
            var_name TEXT NOT NULL,
            lifted_var TEXT NOT NULL,
            hop INTEGER NOT NULL,
            base_var TEXT,
            timeframe TEXT,
            lifted_concept_id TEXT,
            base_var_stem TEXT,
            lifted_var_stem TEXT,
            usage_scope TEXT,
            PRIMARY KEY (trial_id, clause_id, literal_index, lifted_var)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS constraint_literal_alternatives (
            trial_id INTEGER NOT NULL,
            clause_id INTEGER NOT NULL,
            literal_index INTEGER NOT NULL,
            timeframe TEXT NOT NULL,
            alt_concept_id TEXT NOT NULL,
            hop INTEGER NOT NULL,
            reason TEXT,
            decided_at TEXT DEFAULT (datetime('now')),
            lifted_var TEXT,
            base_var TEXT,
            base_var_stem TEXT,
            lifted_var_stem TEXT,
            usage_scope TEXT,
            PRIMARY KEY (trial_id, clause_id, literal_index, timeframe, alt_concept_id)
        )
    """)

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

    cur.execute("DROP VIEW IF EXISTS vw_assumed_relevance_lifts")
    cur.execute("""
      CREATE VIEW IF NOT EXISTS vw_assumed_relevance_lifts AS
      SELECT laa.*
      FROM constraint_literal_alternatives laa
      JOIN trial_constraint_clauses tsc
        ON tsc.trial_id = laa.trial_id AND tsc.clause_id = laa.clause_id
      JOIN trial_constraint_sides t
        ON t.id = tsc.trial_id
      WHERE t.variant='assumed' AND laa.usage_scope='relevance_only'
    """)
    conn.commit()


# ────────────────────────────────────────────────────────────────
# predicate_to_concept population
# ────────────────────────────────────────────────────────────────
_RX_SCT_PREFIXED = re.compile(r"(?:^|_)(?:sctid|snomed|sct|sctid-)(?:[:_])?(?P<id>\d{5,18})(?:_|$)", re.I)
_RX_RAW_SCTID = re.compile(r"(?:^|_)(?P<id>\d{6,18})(?:_|$)")


def _extract_concept_id_from_varname(var_name: str) -> Optional[str]:
    s = (var_name or "").lower()
    m = _RX_SCT_PREFIXED.search(s)
    if m:
        return m.group("id")
    m = _RX_RAW_SCTID.search(s)
    if m:
        return m.group("id")
    return None


def rebuild_predicate_to_concept(
    conn: sqlite3.Connection,
    *,
    trial_id: Optional[int] = None,
    canon_dir: Optional[str] = None,
    minified_canon_dir: Optional[str] = None,
) -> int:
    cur = conn.cursor()
    ensure_schema(conn)

    pairs = _load_predicate_to_concept_from_canon(canon_dir)
    if pairs:
        before = cur.execute("SELECT COUNT(*) FROM predicate_to_concept").fetchone()[0] or 0
        cur.executemany("INSERT OR IGNORE INTO predicate_to_concept(var_name, concept_id) VALUES (?, ?)", pairs)
        conn.commit()
        after = cur.execute("SELECT COUNT(*) FROM predicate_to_concept").fetchone()[0] or 0
        return max(0, after - before)

    min_names = set(_load_minified_names(minified_canon_dir))
    inserted = 0

    def _iter_positive_vars(cur0: sqlite3.Cursor, tid: Optional[int]) -> Iterable[str]:
        if tid is None:
            q = """
                SELECT DISTINCT cl.var_name
                FROM trial_constraint_clauses tsc
                JOIN constraint_clause_atoms cl ON cl.clause_id=tsc.clause_id
                WHERE cl.is_neg=0
            """
            params = ()
        else:
            q = """
                SELECT DISTINCT cl.var_name
                FROM trial_constraint_clauses tsc
                JOIN constraint_clause_atoms cl ON cl.clause_id=tsc.clause_id
                WHERE tsc.trial_id=? AND cl.is_neg=0
            """
            params = (tid,)
        for (vn,) in cur0.execute(q, params):
            yield vn

    for var_name in _iter_positive_vars(cur, trial_id):
        if min_names and var_name not in min_names:
            continue
        cid = _extract_concept_id_from_varname(var_name)
        if not cid:
            continue
        cur.execute("INSERT OR IGNORE INTO predicate_to_concept(var_name, concept_id) VALUES (?, ?)", (var_name, cid))
        if cur.rowcount:
            inserted += 1

    conn.commit()
    return inserted


# ────────────────────────────────────────────────────────────────
# Lifting helpers
# ────────────────────────────────────────────────────────────────
def _hop_between(base_cid: str, alt_cid: str) -> int:
    minis = (
        (parents_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, base_cid) or []) +
        (ancestors_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, base_cid) or [])
    )
    ids = [str(m.get("conceptId")) for m in minis if m.get("conceptId")]
    hop_map = compute_upward_hops(base_cid, ids)
    return int(hop_map.get(str(alt_cid), 0))


def _fetch_kept_ancestors(
    cur: sqlite3.Cursor,
    *,
    base_cid: str,
    tf: str,
    qual_suffix: str,
) -> List[Tuple[str, str, int, str, str]]:
    """
    Returns list of:
      (alt_cid, reason, hop, lifted_var, lifted_var_stem)
    """
    out: List[Tuple[str, str, int, str, str]] = []
    for alt in cur.execute("""
        SELECT caa.alt_concept_id, COALESCE(caa.reason,'keep-list') AS reason
        FROM concept_accepted_alternatives caa
        WHERE caa.concept_id = ?
        ORDER BY caa.hop, caa.alt_concept_id
    """, (base_cid,)).fetchall():
        alt_cid = str(alt[0])
        reason = str(alt[1])
        lifted_var = _build_var_for_concept(alt_cid, tf, qual_suffix=qual_suffix)
        lifted_var_stem = _strip_timeframe_once(lifted_var)
        hop = _hop_between(base_cid, alt_cid)
        out.append((alt_cid, reason, hop, lifted_var, lifted_var_stem))
    return out


# ────────────────────────────────────────────────────────────────
# Trial iteration & lifting
# ────────────────────────────────────────────────────────────────
def _iter_positive_constraint_literals_for_trial(cur: sqlite3.Cursor, trial_id: int):
    q = """
    SELECT
        t.kind,
        t.variant,
        c.id AS clause_id,
        cl.literal_index,
        cl.var_name,
        vc.base_var,
        vc.timeframe
    FROM trial_constraint_clauses tsc
    JOIN trial_constraint_sides t ON t.id = tsc.trial_id
    JOIN constraint_clauses c ON c.id = tsc.clause_id
    JOIN constraint_clause_atoms cl ON cl.clause_id = c.id AND cl.is_neg = 0
    LEFT JOIN predicate_catalog vc ON vc.var_name = cl.var_name
    WHERE tsc.trial_id = ?
    ORDER BY c.id, cl.literal_index
    """
    for row in cur.execute(q, (trial_id,)):
        kind, variant, cid, idx, vname, base_var, tf = row
        yield (
            str(kind),
            str(variant),
            int(cid),
            int(idx),
            str(vname),
            (base_var if base_var is not None else None),
            (tf if tf is not None else ""),
        )


def build_lifted_for_trials(
    conn: sqlite3.Connection,
    *,
    trial_id: Optional[int] = None,
    include_assumed: bool = True,
    include_assumed_ancestors_for_relevance: bool = True,
) -> int:
    """
    - Always inserts hop=0 self rows for inclusion and assumed.
    - Inclusion: materializes keep-list ancestors (usage_scope=NULL).
    - Assumed: if include_assumed_ancestors_for_relevance=True, materializes keep-list
      ancestors with usage_scope='relevance_only'.

    Returns number of rows inserted into constraint_lifted_atoms.
    """
    ensure_schema(conn)
    cur = conn.cursor()

    keep_exists = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='concept_accepted_alternatives'"
    ).fetchone())
    if not keep_exists:
        print(
            "[lifter] ERROR: concept_accepted_alternatives (keep-list) not found. Run the decider first.",
            file=sys.stderr,
        )
        return 0

    if trial_id is None:
        trial_ids = [r[0] for r in cur.execute("SELECT id FROM trial_constraint_sides").fetchall()]
    else:
        trial_ids = [int(trial_id)]

    total_inserted_llm = 0

    for tid in trial_ids:
        rows = list(_iter_positive_constraint_literals_for_trial(cur, tid))
        if not rows:
            continue

        insert_vals_llm = []

        for kind, variant, clause_id, lit_idx, var_name, base_var, tf in rows:
            tf_eff = (tf or "now").strip() or "now"
            qual_suffix = _extract_qual_suffix(var_name)

            base_effective = base_var or var_name
            base_var_stem = _strip_timeframe_once(base_effective)

            base_cid_row = cur.execute(
                "SELECT concept_id FROM predicate_to_concept WHERE var_name=?",
                (var_name,)
            ).fetchone()
            base_cid = (base_cid_row[0] if base_cid_row else None)

            # If no concept mapping, still keep self row in constraint_lifted_atoms using original var_name
            if base_cid:
                self_lifted_var = _build_var_for_concept(base_cid, tf_eff, qual_suffix=qual_suffix)
                self_lifted_var_stem = _strip_timeframe_once(self_lifted_var)
            else:
                self_lifted_var = var_name
                self_lifted_var_stem = _strip_timeframe_once(var_name)

            insert_vals_llm.append((
                tid, clause_id, lit_idx,
                var_name, self_lifted_var, 0,
                base_effective, tf_eff,
                base_cid, base_var_stem, self_lifted_var_stem,
                None
            ))

            # Self row in constraint_literal_alternatives only if concept exists
            if base_cid:
                cur.execute("""
                    INSERT OR REPLACE INTO constraint_literal_alternatives
                      (trial_id, clause_id, literal_index, timeframe,
                       alt_concept_id, hop, reason, lifted_var, base_var,
                       base_var_stem, lifted_var_stem, usage_scope)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?, NULL)
                """, (
                    tid, clause_id, lit_idx, tf_eff,
                    base_cid, 0, "self",
                    self_lifted_var, base_effective,
                    base_var_stem, self_lifted_var_stem
                ))

            # Ancestors
            if base_cid:
                kept = _fetch_kept_ancestors(
                    cur,
                    base_cid=base_cid,
                    tf=tf_eff,
                    qual_suffix=qual_suffix,
                )

                if kind == "inclusion":
                    for alt_cid, reason, hop, lifted_var, lifted_stem in kept:
                        cur.execute("""
                            INSERT OR REPLACE INTO constraint_literal_alternatives
                              (trial_id, clause_id, literal_index, timeframe,
                               alt_concept_id, hop, reason, lifted_var, base_var,
                               base_var_stem, lifted_var_stem, usage_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, NULL)
                        """, (
                            tid, clause_id, lit_idx, tf_eff,
                            alt_cid, hop, reason,
                            lifted_var, base_effective,
                            base_var_stem, lifted_stem
                        ))

                elif include_assumed and (variant == "assumed") and include_assumed_ancestors_for_relevance:
                    for alt_cid, reason, hop, lifted_var, lifted_stem in kept:
                        cur.execute("""
                            INSERT OR REPLACE INTO constraint_literal_alternatives
                              (trial_id, clause_id, literal_index, timeframe,
                               alt_concept_id, hop, reason, lifted_var, base_var,
                               base_var_stem, lifted_var_stem, usage_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, 'relevance_only')
                        """, (
                            tid, clause_id, lit_idx, tf_eff,
                            alt_cid, hop, reason or "keep-list",
                            lifted_var, base_effective,
                            base_var_stem, lifted_stem
                        ))

        if insert_vals_llm:
            cur.executemany("""
                INSERT OR IGNORE INTO constraint_lifted_atoms
                  (trial_id, clause_id, literal_index, var_name, lifted_var, hop,
                   base_var, timeframe, lifted_concept_id, base_var_stem,
                   lifted_var_stem, usage_scope)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, insert_vals_llm)
            total_inserted_llm += cur.rowcount or 0

        conn.commit()

    return total_inserted_llm


# ────────────────────────────────────────────────────────────────
# Backwards-compatible wrapper
# ────────────────────────────────────────────────────────────────
def build_lifted_for_inclusion(conn: sqlite3.Connection, *, trial_id: Optional[int] = None) -> int:
    return build_lifted_for_trials(
        conn,
        trial_id=trial_id,
        include_assumed=False,
        include_assumed_ancestors_for_relevance=False,
    )


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def _open_core(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Ontology lifting (inclusion + optional assumed; keep-list only)."
    )
    ap.add_argument("--db", required=True, help="SQLite DB (e.g., ../../build/trial.db)")
    ap.add_argument("--trial", type=int, default=None, help="Only process this trial_constraint_sides.id")

    ap.add_argument(
        "--rebuild-predicate_to_concept",
        action="store_true",
        help="(Re)populate predicate_to_concept (from --canon-dir if provided; else fallback to varname extractor).",
    )
    ap.add_argument(
        "--overwrite-predicate_to_concept",
        action="store_true",
        help="Delete predicate_to_concept before rebuilding (use with --rebuild-predicate_to_concept).",
    )
    ap.add_argument("--canon-dir", type=str, default=None, help="Directory of canon JSON files.")
    ap.add_argument("--minified-canon-dir", type=str, default=None, help="Directory of minified_canon files.")

    ap.add_argument(
        "--lift",
        action="store_true",
        default=True,
        help="Materialize constraint_lifted_atoms and constraint_literal_alternatives",
    )
    ap.add_argument(
        "--overwrite-lifts",
        action="store_true",
        default=True,
        help="Delete constraint_lifted_atoms and constraint_literal_alternatives before lifting.",
    )
    ap.add_argument(
        "--include-assumed",
        action="store_true",
        default=True,
        help="Also process trial_constraint_sides where variant='assumed'.",
    )
    ap.add_argument(
        "--assumed-ancestors-for-relevance",
        action="store_true",
        default=True,
        help="For assumed sides, also write ancestor lifts with usage_scope='relevance_only'.",
    )
    ap.add_argument("--dryrun", action="store_true", help="Run but rollback DB writes")
    args = ap.parse_args()

    conn = _open_core(args.db)
    try:
        ensure_schema(conn)

        if args.rebuild_predicate_to_concept:
            if args.overwrite_predicate_to_concept:
                conn.execute("DELETE FROM predicate_to_concept")
                conn.commit()
            n = rebuild_predicate_to_concept(
                conn,
                trial_id=args.trial,
                canon_dir=args.canon_dir,
                minified_canon_dir=args.minified_canon_dir,
            )
            print(f"[lifter] predicate_to_concept inserted: {n}")

        if args.lift:
            if args.overwrite_lifts:
                conn.execute("DELETE FROM constraint_lifted_atoms")
                conn.execute("DELETE FROM constraint_literal_alternatives")
                conn.commit()

            if args.dryrun:
                print("[lifter] dryrun...")
                conn.isolation_level = "DEFERRED"
                conn.execute("BEGIN")
                _ = build_lifted_for_trials(
                    conn,
                    trial_id=args.trial,
                    include_assumed=args.include_assumed,
                    include_assumed_ancestors_for_relevance=args.assumed_ancestors_for_relevance,
                )
                conn.execute("ROLLBACK")
                print("[lifter] dryrun complete.")
            else:
                ins = build_lifted_for_trials(
                    conn,
                    trial_id=args.trial,
                    include_assumed=args.include_assumed,
                    include_assumed_ancestors_for_relevance=args.assumed_ancestors_for_relevance,
                )
                print(f"[lifter] constraint_lifted_atoms inserted: {ins}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()