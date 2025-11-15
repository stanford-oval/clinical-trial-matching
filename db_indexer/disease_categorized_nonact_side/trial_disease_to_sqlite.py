#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trial_disease_to_sqlite.py
────────────────────────────────────────────────────────────────────────────
Load *_disease_link_filter_summary.json files and write selected rows into a single
SQLite table (default: `disease_constraint_atoms`).

Subcohort-first ingestion (default):
  • Recursive discovery (rglob).
  • Default trial_id from filename token:
        NCT00002466a_disease_link_filter_summary.json → trial_id=NCT00002466a
  • NO parent→effective fan-out unless --cohort-fanout.

Legacy cohort-aware fan-out (optional):
  • If --cohort-fanout, expand parent trial_id → effective_trial_ids from preproc logs.

MULTI-LABEL CATEGORIES:
  • concept["category"] may be str or list[str]
  • ingest row if ANY category matches allowlist (--categories) after canonicalization
  • canonicalization:
      clinically address → treat
      prevent/prevention → prevent
      other (and typo pther) → other
      not of clinical interest → not relevant

Row emission policy:
  • Main ingest tables (e.g., disease_constraint_atoms*):
      patient_has_finding_of_<stem>_inthehistory
  • Prevention ingest tables (e.g., disease_constraint_atoms_prevent*):
      patient_wants_to_prevent_<stem>_inthehistory

This means prefix is determined by INGESTION CONTEXT (target table), not by whether
the original concept category is "prevent".

Idempotent upserts by (trial_id, var_name).

Fresh tables by default:
  • --recreate default ON (drop table + user indexes, recreate).
"""

from __future__ import annotations
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional, Set

_PAREN_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_TRIAL_ID_TOKEN_RE = re.compile(r"^NCT\d+[a-z]?$", re.IGNORECASE)

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

def choose_primary_category(cat_list: List[str], allowed_categories: Set[str]) -> str:
    # deterministic: prefer prevent > first matched-in-allowlist > first available > ""
    is_prevent = any(c == "prevent" or c.startswith("prevent") for c in cat_list)
    if is_prevent:
        return "prevent"
    if allowed_categories:
        for c in cat_list:
            if c in allowed_categories:
                return c
    return cat_list[0] if cat_list else ""

def _read_preproc_normalized(fp: Path) -> Optional[Dict[str, Any]]:
    try:
        if fp.is_file():
            return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def find_effective_trial_ids(
    trial_id: str,
    preproc_dir: Path,
    prefer_side: str = "inclusion",
) -> List[str]:
    prefer_side = "inclusion" if prefer_side not in ("inclusion", "exclusion") else prefer_side
    other_side = "exclusion" if prefer_side == "inclusion" else "inclusion"

    def _fp(side: str) -> Path:
        return preproc_dir / f"{trial_id}_{side}.pre.normalized.json"

    for side in (prefer_side, other_side):
        blob = _read_preproc_normalized(_fp(side))
        if isinstance(blob, dict):
            eff = blob.get("effective_trial_ids")
            if isinstance(eff, list) and eff:
                out = [str(x) for x in eff if isinstance(x, str) and x.strip()]
                if out:
                    return out
    return [trial_id]

def ddl_for(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id TEXT NOT NULL,
        generated TEXT,
        disease TEXT NOT NULL,
        var_name TEXT NOT NULL,
        stem TEXT NOT NULL,
        stem_var TEXT,
        category TEXT,
        conceptId TEXT,
        preferred_term TEXT,
        fully_specified_name TEXT,
        type TEXT,
        definition TEXT,
        best_match_term TEXT,
        UNIQUE(trial_id, var_name)
    );
    """

def upsert_sql_for(table: str) -> str:
    return f"""
    INSERT INTO {table} (
        trial_id, generated, disease, var_name, stem, stem_var, category,
        conceptId, preferred_term, fully_specified_name, type, definition, best_match_term
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(trial_id, var_name) DO UPDATE SET
        generated=excluded.generated,
        disease=excluded.disease,
        stem=excluded.stem,
        stem_var=excluded.stem_var,
        category=excluded.category,
        conceptId=excluded.conceptId,
        preferred_term=excluded.preferred_term,
        fully_specified_name=excluded.fully_specified_name,
        type=excluded.type,
        definition=excluded.definition,
        best_match_term=excluded.best_match_term;
    """

def indexes_for(table: str) -> List[str]:
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{table}_trial ON {table}(trial_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_var ON {table}(var_name);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_stem ON {table}(stem);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_stem_var ON {table}(stem_var);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_category ON {table}(category);",
    ]

def _drop_table_everything(conn: sqlite3.Connection, table: str) -> None:
    cur = conn.cursor()
    for (idx, sql) in cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    ).fetchall():
        if sql is None:
            continue
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()

def iter_json_files(input_dir: Path) -> Iterable[Path]:
    preferred = sorted(input_dir.rglob("*_disease_link_filter_summary.json"))
    if preferred:
        yield from preferred
    else:
        yield from sorted(input_dir.rglob("*.json"))

def load_summary(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def extract_rows(
    blob: Dict[str, Any],
    allowed_categories: Set[str],
    *,
    force_prevent_prefix: bool = False,
) -> Tuple[str, str, List[Tuple]]:
    trial_id = (blob.get("trial_id") or "").strip() or "UNKNOWN_TRIAL_ID"
    generated = (blob.get("generated") or "").strip() or None

    concepts = blob.get("final_selected_concept_by_disease") or {}
    if not isinstance(concepts, dict):
        concepts = {}

    rows: List[Tuple] = []
    for disease_name, concept in concepts.items():
        if not isinstance(concept, dict):
            continue

        disease = str(disease_name)

        cat_list = normalize_category_list(concept.get("category", ""))
        cat_list = canonicalize_categories(cat_list)

        if allowed_categories and not any(c in allowed_categories for c in cat_list):
            continue

        category_for_db = choose_primary_category(cat_list, allowed_categories)

        conceptId = (concept.get("conceptId") or "")
        preferred_term = (concept.get("preferred_term") or "")
        fully_specified_name = (concept.get("fully_specified_name") or "")
        ctype = (concept.get("type") or "")
        definition = (concept.get("definition") or "")
        best_match_term = (concept.get("best_match_term") or "")

        stem_src = preferred_term or fully_specified_name or disease
        stem = to_var_snake(stem_src)

        if force_prevent_prefix:
            base_var_name = f"patient_wants_to_prevent_{stem}_inthehistory"
            base_stem_var = f"patient_wants_to_prevent_{stem}"
        else:
            base_var_name = f"patient_has_finding_of_{stem}_inthehistory"
            base_stem_var = f"patient_has_finding_of_{stem}"

        rows.append((
            trial_id, (generated or ""), disease,
            base_var_name, stem, base_stem_var, category_for_db,
            conceptId, preferred_term, fully_specified_name, ctype, definition, best_match_term,
        ))

    return trial_id, (generated or ""), rows

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="../../build/disease",
                    help="Dir with *_disease_link_filter_summary.json files (supports nested subdirs)")
    ap.add_argument("--db", default="../../build/trial.db",
                    help="SQLite db path (default: trial.db)")

    ap.add_argument("--preproc-dir", default="mbench/req_mbench/preproc_logs",
                    help="Directory with {trial_id}_{side}.pre.normalized.json (only used with --cohort-fanout)")
    ap.add_argument("--prefer-side", choices=["inclusion", "exclusion"], default="inclusion",
                    help="Which side’s preproc file to prefer when both exist (only used with --cohort-fanout)")
    ap.add_argument("--cohort-fanout", action="store_true", default=False,
                    help="(legacy) Expand parent trial_id to effective_trial_ids via preproc logs and ingest for each effective id")

    ap.add_argument("--trial-id-from-filename", dest="trial_id_from_filename",
                    action="store_true", default=True,
                    help="(default ON) Override/ensure blob['trial_id'] from filename token (e.g., NCT...a_...)")
    ap.add_argument("--no-trial-id-from-filename", dest="trial_id_from_filename",
                    action="store_false",
                    help="Trust JSON trial_id field as-is (do not override from filename)")

    ap.add_argument("--table", default="disease_constraint_atoms",
                    help="Target table name for ingestion")

    ap.add_argument("--categories", default="treat,other,prevent",
                    help="Comma-separated allowlist of CANONICAL categories to ingest (after canonicalization).")

    ap.add_argument("--overwrite", choices=["all", "by-trial", "keep", "drop"], default="all",
                    help="Overwrite mode when --no-recreate.")
    ap.add_argument("--recreate", dest="recreate", action="store_true", default=True,
                    help="(default ON) DROP table + user indexes and recreate target table before ingesting.")
    ap.add_argument("--no-recreate", dest="recreate", action="store_false",
                    help="Do not drop the table at start; use --overwrite behavior instead.")

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"[!] Not a directory: {input_dir}", file=sys.stderr)
        return 2

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    preproc_dir = Path(args.preproc_dir).expanduser().resolve()

    table = (args.table or "disease_constraint_atoms").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        print(f"[!] Unsafe table name: {table}", file=sys.stderr)
        return 2

    allowed = {c.strip().lower() for c in (args.categories or "").split(",") if c.strip()}

    # Prevention tables always emit patient_wants_to_prevent_% vars.
    force_prevent_prefix = table.startswith("disease_constraint_atoms_prevent")

    total_files = 0
    total_upserts = 0
    total_filtered_out = 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        if args.recreate:
            print(f"[info] fresh mode: dropping table {table}")
            _drop_table_everything(conn, table)

        if (not args.recreate) and args.overwrite == "drop":
            conn.execute(f"DROP TABLE IF EXISTS {table};")
            conn.commit()

        conn.execute(ddl_for(table))

        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if "stem_var" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN stem_var TEXT;")
            if force_prevent_prefix:
                conn.execute(f"""
                    UPDATE {table}
                       SET stem_var = 'patient_wants_to_prevent_' || stem
                     WHERE stem_var IS NULL OR stem_var = '';
                """)
            else:
                conn.execute(f"""
                    UPDATE {table}
                       SET stem_var = 'patient_has_finding_of_' || stem
                     WHERE stem_var IS NULL OR stem_var = '';
                """)
            conn.commit()

        if "category" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN category TEXT;")
            conn.commit()

        for idx_sql in indexes_for(table):
            conn.execute(idx_sql)

        if (not args.recreate) and args.overwrite == "all":
            conn.execute(f"DELETE FROM {table}")
            conn.commit()

        UPSERT_SQL = upsert_sql_for(table)

        for fp in iter_json_files(input_dir):
            try:
                blob = load_summary(fp)

                if args.trial_id_from_filename:
                    tid_fn = trial_id_from_filename(fp)
                    if tid_fn:
                        blob["trial_id"] = tid_fn

                parent_tid, generated, rows = extract_rows(
                    blob,
                    allowed,
                    force_prevent_prefix=force_prevent_prefix,
                )

                if args.cohort_fanout:
                    effective_ids = find_effective_trial_ids(parent_tid, preproc_dir, args.prefer_side)
                else:
                    effective_ids = [parent_tid]

                if (not args.recreate) and args.overwrite == "by-trial":
                    conn.executemany(
                        f"DELETE FROM {table} WHERE trial_id = ?",
                        [(tid_eff,) for tid_eff in effective_ids]
                    )

                if (not rows) and isinstance(blob.get("final_selected_concept_by_disease"), dict) and blob["final_selected_concept_by_disease"]:
                    total_filtered_out += 1

                batch: List[Tuple] = []
                for tid_eff in effective_ids:
                    for (_tid_in_row, gen, disease, var_name, stem, stem_var, category,
                         conceptId, preferred_term, fsn, ctype, definition, best_match_term) in rows:
                        batch.append((
                            tid_eff, gen, disease, var_name, stem, stem_var, category,
                            conceptId, preferred_term, fsn, ctype, definition, best_match_term
                        ))

                if batch:
                    cur = conn.executemany(UPSERT_SQL, batch)
                    conn.commit()
                    up = (cur.rowcount or 0)
                else:
                    up = 0

                total_files += 1
                total_upserts += up

                eff_note = "" if effective_ids == [parent_tid] else f" (fanout: {', '.join(effective_ids)})"
                try:
                    rel = fp.relative_to(input_dir)
                except Exception:
                    rel = fp
                prefix_note = "prevent-prefix" if force_prevent_prefix else "finding-prefix"
                print(f"[✓] {rel} → {table}: upserts={up} [{prefix_note}]{eff_note}")
            except Exception as e:
                print(f"[✗] {fp}: {e}", file=sys.stderr)

    print(f"\nDone. Table: {table}")
    print(f"Files processed: {total_files}, rows upserted: {total_upserts}")
    if total_filtered_out:
        print(f"[info] files where all diseases were filtered out by category: {total_filtered_out}")
    print(f"SQLite DB: {db_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())