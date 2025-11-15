#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ontology_lifter.py — inclusion + assumed ontology lifting (decider-first, keep-list only)
TWO BRANCHES
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import argparse

# ────────────────────────────────────────────────────────────────
# BRANCH CONFIG
# ────────────────────────────────────────────────────────────────
BRANCHES = {
    "main": {
        "caa_table": "concept_accepted_alternatives",
        "llm_table": "constraint_lifted_atoms",
        "laa_table": "constraint_literal_alternatives",
    },
    "prevention": {
        "caa_table": "concept_accepted_alternatives_prevention",
        "llm_table": "constraint_lifted_atoms_prevention",
        "laa_table": "constraint_literal_alternatives_prevention",
    },
}
_ID_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _safe_ident(x: str, what: str) -> str:
    x = (x or "").strip()
    if not _ID_RX.match(x):
        raise ValueError(f"Unsafe {what}: {x!r}")
    return x

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
    if not concept_id:
        return None
    if concept_id in _LABEL_CACHE:
        return _LABEL_CACHE[concept_id]
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
# Hop computation
# ────────────────────────────────────────────────────────────────
_PARENTS_IDS_CACHE: Dict[str, List[str]] = {}

def _parent_ids(cid: str) -> List[str]:
    if cid in _PARENTS_IDS_CACHE:
        return _PARENTS_IDS_CACHE[cid]
    ids = [str(m.get("conceptId")) for m in (parents_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, cid) or []) if m.get("conceptId")]
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
# Templates (schema-aware: add prevention)
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
    # ✅ NEW: prevention template
    "prevention": [
        "patient_wants_to_prevent_{entity_canonical_form}_{timeframe}",
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
    if not m:
        return vn, ""
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
            if not g.get("tf"):
                return None
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
# Schema (table-aware)
# ────────────────────────────────────────────────────────────────
def ensure_schema(conn: sqlite3.Connection, llm_table: str, laa_table: str) -> None:
    llm_table = _safe_ident(llm_table, "llm_table")
    laa_table = _safe_ident(laa_table, "laa_table")

    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.execute("""CREATE TABLE IF NOT EXISTS predicate_to_concept (var_name TEXT PRIMARY KEY, concept_id TEXT NOT NULL)""")

    cur.execute(f"""CREATE TABLE IF NOT EXISTS {llm_table} (
        trial_id INTEGER NOT NULL, clause_id INTEGER NOT NULL, literal_index INTEGER NOT NULL,
        var_name TEXT NOT NULL, lifted_var TEXT NOT NULL, hop INTEGER NOT NULL,
        base_var TEXT, timeframe TEXT,
        lifted_concept_id TEXT,
        base_var_stem TEXT, lifted_var_stem TEXT,
        usage_scope TEXT,
        PRIMARY KEY (trial_id, clause_id, literal_index, lifted_var)
    )""")

    cur.execute(f"""CREATE TABLE IF NOT EXISTS {laa_table} (
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
    conn.commit()

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def _compose_from_label(base_var: str, alt_label: str, timeframe_fallback: str="", *, prevention: bool=False) -> Optional[str]:
    """
    main: use detected template from base_var
    prevention: force patient_wants_to_prevent_{entity}_{timeframe} but preserve @@qual from base_var if present
    """
    ent = _canonize_entity_form(alt_label)
    if not ent:
        return None

    det = _detect_template_from_varname(base_var)
    qual_suffix = det[1].get("qual_suffix","") if det else ""

    tf = ""
    if det:
        tf = det[1].get("tf","") or ""
    if not tf:
        tf = timeframe_fallback or ""
    if not tf:
        return None

    if prevention:
        return f"patient_wants_to_prevent_{ent}_{tf}{qual_suffix}"

    if not det:
        return None
    tpl, g = det
    unit = g.get("unit") or ""
    return _compose_var_name(tpl, ent, tf, unit, qual_suffix=qual_suffix)

def _hop_between(base_cid: str, alt_cid: str) -> int:
    minis = (parents_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, base_cid) or []) + \
            (ancestors_cached(SNOWSTORM_BRANCH, SNOWSTORM_FORM, base_cid) or [])
    ids = [str(m.get("conceptId")) for m in minis if m.get("conceptId")]
    hop_map = compute_upward_hops(base_cid, ids)
    return int(hop_map.get(str(alt_cid), 0))

def _fetch_kept_ancestors(
    cur: sqlite3.Cursor,
    *,
    caa_table: str,
    base_cid: str,
    base_var: str,
    tf: str,
    prevention: bool,
) -> List[Tuple[str,str,str,int,str,str]]:
    caa_table = _safe_ident(caa_table, "caa_table")
    out: List[Tuple[str,str,str,int,str,str]] = []
    for alt in cur.execute(f"""
        SELECT caa.alt_concept_id, COALESCE(caa.alt_label,''), COALESCE(caa.reason,'keep-list') AS reason
        FROM {caa_table} caa
        WHERE caa.concept_id = ?
        ORDER BY caa.hop, caa.alt_concept_id
    """, (base_cid,)).fetchall():
        alt_cid, alt_label, reason = str(alt[0]), str(alt[1]), str(alt[2])
        lifted_var = _compose_from_label(base_var, alt_label, timeframe_fallback=tf, prevention=prevention)
        if not lifted_var:
            continue
        lifted_var_stem = _strip_timeframe_once(lifted_var)
        hop = _hop_between(base_cid, alt_cid)
        out.append((alt_cid, alt_label, reason, hop, lifted_var, lifted_var_stem))
    return out

def _strip_qual(v: str) -> str:
    stem, _q = _split_qual_suffix(v or "")
    return stem

def _lookup_concept_id(cur: sqlite3.Cursor, var_name: str, base_effective: str) -> Optional[str]:
    """
    Try several keys against predicate_to_concept:
      1) var_name
      2) var_name without @@qual
      3) base_effective
      4) base_effective without @@qual
    """
    candidates = []
    if var_name:
        candidates.append(var_name)
        candidates.append(_strip_qual(var_name))
    if base_effective and base_effective not in candidates:
        candidates.append(base_effective)
        candidates.append(_strip_qual(base_effective))

    seen = set()
    for k in candidates:
        if not k or k in seen:
            continue
        seen.add(k)
        row = cur.execute("SELECT concept_id FROM predicate_to_concept WHERE var_name=?", (k,)).fetchone()
        if row and row[0]:
            return str(row[0])
    return None

# ────────────────────────────────────────────────────────────────
# Trial iteration & lifting (table-aware outputs)
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

def build_lifted_for_trials(
    conn: sqlite3.Connection,
    *,
    llm_table: str,
    laa_table: str,
    caa_table: str,
    trial_id: Optional[int]=None,
    include_assumed: bool=True,
    include_assumed_ancestors_for_relevance: bool=True,
    prevention: bool=False,
) -> int:
    llm_table = _safe_ident(llm_table, "llm_table")
    laa_table = _safe_ident(laa_table, "laa_table")
    caa_table = _safe_ident(caa_table, "caa_table")

    ensure_schema(conn, llm_table, laa_table)

    cur = conn.cursor()
    keep_exists = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (caa_table,),
    ).fetchone())
    if not keep_exists:
        print(f"[lifter] ERROR: keep-list table {caa_table!r} not found. Run decider first.", file=sys.stderr)
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

            base_cid = _lookup_concept_id(cur, var_name, base_effective)

            # (1) SELF → constraint_lifted_atoms
            insert_vals_llm.append((tid, clause_id, lit_idx, var_name, var_name, 0,
                                    base_var, tf_eff, base_cid, base_var_stem, lifted_var_stem, None))

            # (2) SELF → constraint_literal_alternatives (if mapped)
            if base_cid:
                cur.execute(f"""
                    INSERT OR REPLACE INTO {laa_table}
                      (trial_id, clause_id, literal_index, timeframe,
                       alt_concept_id, hop, reason, lifted_var, base_var,
                       base_var_stem, lifted_var_stem, usage_scope)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?, NULL)
                """, (tid, clause_id, lit_idx, tf_eff, base_cid, 0, "self", var_name, var_name,
                      base_var_stem, lifted_var_stem))

            # (3) Ancestors via keep-list
            if base_cid:
                kept = _fetch_kept_ancestors(
                    cur,
                    caa_table=caa_table,
                    base_cid=base_cid,
                    base_var=base_effective,
                    tf=tf_eff,
                    prevention=prevention,
                )
                if kind == "inclusion":
                    for alt_cid, _alt_label, reason, hop, lifted_var, lifted_stem in kept:
                        cur.execute(f"""
                            INSERT OR REPLACE INTO {laa_table}
                              (trial_id, clause_id, literal_index, timeframe,
                               alt_concept_id, hop, reason, lifted_var, base_var,
                               base_var_stem, lifted_var_stem, usage_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, NULL)
                        """, (tid, clause_id, lit_idx, tf_eff, alt_cid, hop, reason,
                              lifted_var, base_effective, base_var_stem, lifted_stem))
                elif include_assumed and (variant == "assumed") and include_assumed_ancestors_for_relevance:
                    for alt_cid, _alt_label, reason, hop, lifted_var, lifted_stem in kept:
                        cur.execute(f"""
                            INSERT OR REPLACE INTO {laa_table}
                              (trial_id, clause_id, literal_index, timeframe,
                               alt_concept_id, hop, reason, lifted_var, base_var,
                               base_var_stem, lifted_var_stem, usage_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, 'relevance_only')
                        """, (tid, clause_id, lit_idx, tf_eff, alt_cid, hop, reason or "keep-list",
                              lifted_var, base_effective, base_var_stem, lifted_stem))

        if insert_vals_llm:
            cur.executemany(f"""
                INSERT OR IGNORE INTO {llm_table}
                   (trial_id,clause_id,literal_index,var_name,lifted_var,hop,base_var,timeframe,
                    lifted_concept_id,base_var_stem,lifted_var_stem,usage_scope)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, insert_vals_llm)
            total_self += cur.rowcount or 0

        conn.commit()

    return total_self

# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def _open_core(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row; return conn

def main():
    ap = argparse.ArgumentParser(description="Ontology lifting (two branches: main + prevention).")
    ap.add_argument("--db", default="../../../build/trial.db",help="SQLite DB (e.g., ../../build/trial.db)")
    ap.add_argument("--trial", type=int, default=None, help="Only process this trial_constraint_sides.id")

    ap.add_argument("--only", choices=["both", "main", "prevention"], default="both",
                    help="Which branch(es) to run (default: both)")

    ap.add_argument("--lift", action="store_true", default=True,
                    help="Materialize lifted tables (default: True)")
    ap.add_argument("--overwrite-lifts", action="store_true", default=True,
                    help="Delete lifted outputs before lifting (default: True)")

    ap.add_argument("--include-assumed", action="store_true", default=True,
                    help="Also process assumed variant trial_constraint_sides (default: True)")
    ap.add_argument("--assumed-ancestors-for-relevance", action="store_true", default=True,
                    help="For assumed sides, also write ancestors with usage_scope='relevance_only' (default: True)")

    ap.add_argument("--dryrun", action="store_true", help="Run but rollback DB writes")
    args = ap.parse_args()

    db_path = str(Path(args.db).expanduser().resolve())
    branches = ["main", "prevention"] if args.only == "both" else [args.only]

    conn = _open_core(db_path)
    try:
        for br in branches:
            cfg = BRANCHES[br]
            caa_table = cfg["caa_table"]
            llm_table = cfg["llm_table"]
            laa_table = cfg["laa_table"]
            prevention = (br == "prevention")

            print(f"\n=== LIFT BRANCH: {br} ===")
            print(f"keep-list: {caa_table}")
            print(f"outputs:   {llm_table}, {laa_table}")

            ensure_schema(conn, llm_table, laa_table)

            if args.lift:
                if args.overwrite_lifts:
                    conn.execute(f"DELETE FROM {llm_table}")
                    conn.execute(f"DELETE FROM {laa_table}")
                    conn.commit()

                if args.dryrun:
                    print("[lifter] dryrun...")
                    conn.isolation_level = "DEFERRED"
                    conn.execute("BEGIN")
                    _ = build_lifted_for_trials(
                        conn,
                        llm_table=llm_table,
                        laa_table=laa_table,
                        caa_table=caa_table,
                        trial_id=args.trial,
                        include_assumed=args.include_assumed,
                        include_assumed_ancestors_for_relevance=args.assumed_ancestors_for_relevance,
                        prevention=prevention,
                    )
                    conn.execute("ROLLBACK")
                    print("[lifter] dryrun complete.")
                else:
                    ins = build_lifted_for_trials(
                        conn,
                        llm_table=llm_table,
                        laa_table=laa_table,
                        caa_table=caa_table,
                        trial_id=args.trial,
                        include_assumed=args.include_assumed,
                        include_assumed_ancestors_for_relevance=args.assumed_ancestors_for_relevance,
                        prevention=prevention,
                    )
                    print(f"[lifter:{br}] constraint_lifted_atoms (self) inserted: {ins}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
