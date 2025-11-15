#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_disease_ingest_trial.py

Debug why a trial (default: NCT00443313a) appears to have nothing ingested.

Checks:
  1) Find matching *_disease_link_filter_summary.json files
  2) Show trial_id from filename and from JSON
  3) Show all diseases and raw/canonicalized categories
  4) Simulate extraction logic for:
       - main ingest (treat,other,not relevant)
       - prevention ingest (prevent,other)
  5) Inspect SQLite tables:
       - disease_constraint_atoms{suffix}
       - disease_constraint_atoms_prevent{suffix}
       - disease_predicate_concepts{suffix}
       - disease_predicate_concepts_prevent{suffix}
       - disease_constraint_lifted_atoms{suffix}
       - disease_constraint_lifted_atoms_prevent{suffix}
       - disease_constraint_alternatives{suffix}
       - disease_constraint_alternatives_prevent{suffix}

Example:
  python probe_disease_ingest_trial.py \
    --trial NCT00443313a \
    --input-dir ../../build/disease_filtered_categorized \
    --db ../../build/trial.db \
    --table-suffix _nonact
"""

from __future__ import annotations
import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PAREN_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_TRIAL_ID_TOKEN_RE = re.compile(r"^NCT\d+[a-z]?$", re.IGNORECASE)

# --------------------------
# copied / aligned helpers
# --------------------------

def norm_suffix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if s.startswith("_") else "_" + s

def with_suffix(name: str, suf: str) -> str:
    suf = norm_suffix(suf)
    if not suf:
        return name
    if name.endswith(suf):
        return name
    return name + suf

def _norm_cat(s: str) -> str:
    return (s or "").strip().lower()

def to_var_snake(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PAREN_TAIL_RE.sub("", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def trial_id_from_filename(fp: Path) -> Optional[str]:
    tok = fp.name.split("_", 1)[0].strip()
    if _TRIAL_ID_TOKEN_RE.match(tok):
        return tok
    return None

def normalize_category_list(cat_val: Any) -> List[str]:
    if isinstance(cat_val, str):
        s = _norm_cat(cat_val)
        return [s] if s else []
    if isinstance(cat_val, list):
        out: List[str] = []
        for x in cat_val:
            if isinstance(x, str):
                s = _norm_cat(x)
                if s:
                    out.append(s)
        return out
    return []

_CATEGORY_SYNONYMS: Dict[str, str] = {
    "pther": "other",
    "clinical address": "clinically address",
    "clinically addresses": "clinically address",
    "prevention": "prevent",
    "not clinically relevant": "not of clinical interest",
}

_CATEGORY_CANONICALIZE: Dict[str, str] = {
    "clinically address": "treat",
    "prevent": "prevent",
    "other": "other",
    "not of clinical interest": "not relevant",
    "not relevant": "not relevant",
}

def canonicalize_categories(cat_list: List[str]) -> List[str]:
    out: List[str] = []
    for c in cat_list:
        c2 = _CATEGORY_SYNONYMS.get(c, c)
        c3 = _CATEGORY_CANONICALIZE.get(c2, c2)
        if c3:
            out.append(c3)

    seen = set()
    deduped: List[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped

def choose_primary_category(cat_list: List[str], allowed_categories: set[str]) -> str:
    is_prevent = any(c == "prevent" or c.startswith("prevent") for c in cat_list)
    if is_prevent:
        return "prevent"
    if allowed_categories:
        for c in cat_list:
            if c in allowed_categories:
                return c
    return cat_list[0] if cat_list else ""

def extract_rows_for_allowed(blob: Dict[str, Any], allowed_categories: set[str]) -> List[Tuple]:
    concepts = blob.get("final_selected_concept_by_disease") or {}
    if not isinstance(concepts, dict):
        return []

    trial_id = (blob.get("trial_id") or "").strip() or "UNKNOWN_TRIAL_ID"
    generated = (blob.get("generated") or "").strip() or None

    rows: List[Tuple] = []
    for disease_name, concept in concepts.items():
        if not isinstance(concept, dict):
            continue

        disease = str(disease_name)
        raw_cats = normalize_category_list(concept.get("category", ""))
        cat_list = canonicalize_categories(raw_cats)

        if allowed_categories and not any(c in allowed_categories for c in cat_list):
            continue

        is_prevent = any(c == "prevent" or c.startswith("prevent") for c in cat_list)
        category_for_db = choose_primary_category(cat_list, allowed_categories)

        conceptId = (concept.get("conceptId") or "")
        preferred_term = (concept.get("preferred_term") or "")
        fully_specified_name = (concept.get("fully_specified_name") or "")
        ctype = (concept.get("type") or "")
        definition = (concept.get("definition") or "")
        best_match_term = (concept.get("best_match_term") or "")

        stem_src = preferred_term or fully_specified_name or disease
        stem = to_var_snake(stem_src)

        base_var_name = f"patient_has_finding_of_{stem}_inthehistory"
        base_stem_var = f"patient_has_finding_of_{stem}"
        rows.append((
            trial_id, generated or "", disease,
            base_var_name, stem, base_stem_var, category_for_db,
            conceptId, preferred_term, fully_specified_name, ctype, definition, best_match_term
        ))

        if is_prevent:
            prev_var_name = f"patient_wants_to_prevent_{stem}_inthehistory"
            prev_stem_var = f"patient_wants_to_prevent_{stem}"
            rows.append((
                trial_id, generated or "", disease,
                prev_var_name, stem, prev_stem_var, category_for_db,
                conceptId, preferred_term, fully_specified_name, ctype, definition, best_match_term
            ))

    return rows

# --------------------------
# sqlite helpers
# --------------------------

def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return row is not None

def count_rows(conn: sqlite3.Connection, table: str, trial_id: str) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE trial_id=?",
        (trial_id,)
    ).fetchone()[0]

def sample_rows(conn: sqlite3.Connection, table: str, trial_id: str, limit: int = 10) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        f"SELECT * FROM {table} WHERE trial_id=? ORDER BY var_name LIMIT ?",
        (trial_id, limit)
    )
    return cur.fetchall()

def distinct_trial_ids_like(conn: sqlite3.Connection, table: str, prefix: str) -> List[str]:
    return [
        r[0] for r in conn.execute(
            f"SELECT DISTINCT trial_id FROM {table} WHERE trial_id LIKE ? ORDER BY trial_id",
            (prefix + "%",)
        ).fetchall()
    ]

# --------------------------
# reporting
# --------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_kv(k: str, v: Any) -> None:
    print(f"{k:<28} {v}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="NCT00443313a")
    ap.add_argument("--input-dir", default="../../build/disease_filtered_categorized")
    ap.add_argument("--db", default="../../build/trial.db")
    ap.add_argument("--table-suffix", default="_nonact")
    ap.add_argument("--show-all-diseases", action="store_true")
    args = ap.parse_args()

    trial = args.trial.strip()
    input_dir = Path(args.input_dir).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    suf = norm_suffix(args.table_suffix)

    tbl_main = with_suffix("disease_constraint_atoms", suf)
    tbl_prev = with_suffix("disease_constraint_atoms_prevent", suf)

    dv2c_main = with_suffix("disease_predicate_concepts", suf)
    dv2c_prev = with_suffix("disease_predicate_concepts_prevent", suf)

    dlm_main = with_suffix("disease_constraint_lifted_atoms", suf)
    dlm_prev = with_suffix("disease_constraint_lifted_atoms_prevent", suf)

    daa_main = with_suffix("disease_constraint_alternatives", suf)
    daa_prev = with_suffix("disease_constraint_alternatives_prevent", suf)

    print_header("CONFIG")
    print_kv("trial", trial)
    print_kv("input_dir", input_dir)
    print_kv("db", db_path)
    print_kv("table_suffix", suf or "(none)")
    print_kv("tbl_main", tbl_main)
    print_kv("tbl_prev", tbl_prev)

    # 1) file discovery
    print_header("1) SOURCE FILE DISCOVERY")
    hits = sorted(input_dir.rglob(f"{trial}_disease_link_filter_summary.json"))
    if not hits:
        # relaxed search
        relaxed = sorted(input_dir.rglob(f"*{trial}*disease_link_filter_summary.json"))
        if relaxed:
            print(f"No exact match. Found relaxed matches ({len(relaxed)}):")
            for fp in relaxed:
                print(" -", fp)
        else:
            print("No matching disease summary JSON found.")
            return 1
    else:
        print(f"Found {len(hits)} exact match file(s):")
        for fp in hits:
            print(" -", fp)

    # use first exact hit
    fp = hits[0]
    blob = json.loads(fp.read_text(encoding="utf-8"))

    json_trial_id = blob.get("trial_id")
    filename_trial_id = trial_id_from_filename(fp)

    print_header("2) TRIAL ID CHECK")
    print_kv("file", fp)
    print_kv("filename_trial_id", filename_trial_id)
    print_kv("json_trial_id_before_override", json_trial_id)

    # simulate real ingestion behavior
    blob_ingest = dict(blob)
    if filename_trial_id:
        blob_ingest["trial_id"] = filename_trial_id
    print_kv("json_trial_id_after_override", blob_ingest.get("trial_id"))

    concepts = blob_ingest.get("final_selected_concept_by_disease") or {}
    if not isinstance(concepts, dict):
        concepts = {}

    print_header("3) CONCEPT / CATEGORY INSPECTION")
    print_kv("num_diseases_in_json", len(concepts))

    if not concepts:
        print("No final_selected_concept_by_disease content. That alone explains empty ingestion.")
    else:
        shown = 0
        for disease_name, concept in concepts.items():
            if not isinstance(concept, dict):
                continue
            raw_cats = normalize_category_list(concept.get("category", ""))
            canon_cats = canonicalize_categories(raw_cats)
            conceptId = concept.get("conceptId", "")
            pref = concept.get("preferred_term", "")
            fsn = concept.get("fully_specified_name", "")
            stem_src = pref or fsn or disease_name
            stem = to_var_snake(stem_src)

            if args.show_all_diseases or shown < 50:
                print("-" * 60)
                print_kv("disease", disease_name)
                print_kv("raw_categories", raw_cats)
                print_kv("canonical_categories", canon_cats)
                print_kv("conceptId", conceptId)
                print_kv("preferred_term", pref)
                print_kv("fully_specified_name", fsn)
                print_kv("stem", stem)
                shown += 1

        if (not args.show_all_diseases) and len(concepts) > shown:
            print(f"... truncated {len(concepts) - shown} additional disease entries")

    print_header("4) SIMULATED EXTRACTION")
    allowed_main = {"treat", "other", "not relevant"}
    allowed_prev = {"prevent", "other"}

    rows_main = extract_rows_for_allowed(blob_ingest, allowed_main)
    rows_prev = extract_rows_for_allowed(blob_ingest, allowed_prev)

    print_kv("simulated main rows", len(rows_main))
    print_kv("simulated prevention rows", len(rows_prev))

    if rows_main:
        print("\nSample main rows:")
        for r in rows_main[:10]:
            print({
                "trial_id": r[0],
                "disease": r[2],
                "var_name": r[3],
                "category": r[6],
                "conceptId": r[7],
            })

    if rows_prev:
        print("\nSample prevention rows:")
        for r in rows_prev[:10]:
            print({
                "trial_id": r[0],
                "disease": r[2],
                "var_name": r[3],
                "category": r[6],
                "conceptId": r[7],
            })

    if not rows_main and not rows_prev:
        print("\nNothing would be extracted even before SQLite.")
        print("Most likely causes:")
        print("  - all categories filtered out after canonicalization")
        print("  - malformed / missing final_selected_concept_by_disease")
        print("  - disease entries are not dicts")
        print("  - filename trial id mismatch led you to inspect wrong trial key")

    print_header("5) SQLITE TABLE CHECKS")
    if not db_path.exists():
        print(f"DB does not exist: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for tbl in [tbl_main, tbl_prev, dv2c_main, dv2c_prev, dlm_main, dlm_prev, daa_main, daa_prev]:
            exists = table_exists(conn, tbl)
            print(f"{tbl:<45} exists={exists}")
            if exists and tbl.startswith("disease_"):
                try:
                    n = count_rows(conn, tbl, trial)
                    print(f"  rows for {trial}: {n}")
                except Exception as e:
                    print(f"  row count failed: {e}")

        print_header("6) SAMPLE DB ROWS")

        for tbl in [tbl_main, tbl_prev, dv2c_main, dv2c_prev, dlm_main, dlm_prev, daa_main, daa_prev]:
            if not table_exists(conn, tbl):
                continue
            try:
                n = count_rows(conn, tbl, trial)
            except Exception:
                continue
            if n <= 0:
                continue
            print(f"\n--- {tbl} ({n} rows for {trial}) ---")
            rows = sample_rows(conn, tbl, trial, limit=5)
            for r in rows:
                print(dict(r))

        print_header("7) NEARBY TRIAL IDS IN DB")
        for tbl in [tbl_main, tbl_prev]:
            if not table_exists(conn, tbl):
                continue
            ids = distinct_trial_ids_like(conn, tbl, "NCT00443313")
            print(f"{tbl}: {ids if ids else 'none'}")

    finally:
        conn.close()

    print_header("8) INTERPRETATION GUIDE")
    print("Case A: source JSON not found")
    print("  -> wrong input-dir or filename token")
    print("\nCase B: source JSON found, but num_diseases_in_json = 0")
    print("  -> upstream disease categorizer produced nothing")
    print("\nCase C: diseases present, but simulated main/prevention rows = 0")
    print("  -> category mismatch after canonicalization")
    print("     check values like 'prevent', 'other', 'not of clinical interest', etc.")
    print("\nCase D: simulated rows > 0, but ingest tables have 0")
    print("  -> wrong DB path, wrong table suffix, or ingest not run on this DB")
    print("\nCase E: ingest tables have rows, but predicate_to_concept/lifted/accepted are empty")
    print("  -> lifter not run, wrong suffix/ns, or conceptId missing / map rebuild issue")
    print("\nCase F: rows landed under nearby trial id like NCT00443313 instead of NCT00443313a")
    print("  -> filename/json trial_id mismatch or parent/cohort confusion")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())