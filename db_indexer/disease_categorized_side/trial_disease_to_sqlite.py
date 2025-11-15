#!/usr/bin/env python3
"""
trial_disease_to_sqlite.py
────────────────────────────────────────────────────────────────────────────
Load *_disease_link_filter_summary.json files and write selected rows into a single
SQLite table (default: `disease_constraint_atoms`).

NEW (subcohort-first ingestion):
  • Input may contain per-trial subdirectories:
        <input_dir>/<trial_id>/<trial_id>_disease_link_filter_summary.json
    so file discovery is RECURSIVE (rglob).
  • Default behavior ingests "as-is" per file:
        one file → one trial_id namespace
    i.e., NO parent→effective fan-out by default.
  • Default trial_id is taken from the FILENAME token (robust for NCT...a/b/c):
        NCT00002466a_disease_link_filter_summary.json → trial_id=NCT00002466a

Legacy cohort-aware fan-out (optional):
  • If --cohort-fanout is set, we expand parent trial_id to effective_trial_ids
    found in preproc logs (legacy behavior).

MULTI-LABEL CATEGORIES (UPDATED):
  • concept["category"] may be:
      - str: "treat"
      - list[str]: ["Clinically address", "Other", ...]
  • We ingest a disease if ANY category matches the allowlist (--categories).
  • We emit prevention-projected row if ANY category is prevention-like.
  • We canonicalize human labels into stable internal categories:
      clinically address → treat
      prevent/prevention → prevent
      other (and typo pther) → other
      not of clinical interest → not relevant

For each disease concept, we emit:
  - var_name: patient_has_finding_of_<normalized_stem>_inthehistory
              AND, for prevention categories, ALSO:
              patient_wants_to_prevent_<normalized_stem>_inthehistory
  - stem: <normalized_stem>
  - stem_var:
      • patient_has_finding_of_<normalized_stem>
      • AND, for prevention categories, ALSO: patient_wants_to_prevent_<normalized_stem>
  - disease, conceptId, preferred_term, fully_specified_name, type, definition, best_match_term
  - category (a single primary label, chosen deterministically)
  - trial_id (from filename by default; or from JSON if disabled), generated

Idempotent: upserts by (trial_id, var_name).

Fresh tables by default:
  • Default is --recreate (drop table + user indexes, then recreate) so each run
    produces a clean, fresh table. Use --no-recreate to keep schema/data and apply
    overwrite rules instead.
"""

from __future__ import annotations
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────
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
    """
    Extract the leading token before '_' and validate it looks like NCT... or NCT...a/b/c.
      NCT00002466a_disease_link_filter_summary.json -> NCT00002466a
    """
    tok = fp.name.split("_", 1)[0].strip()
    if _TRIAL_ID_TOKEN_RE.match(tok):
        return tok
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Category handling (multi-label + canonicalization)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_category_list(cat_val: Any) -> List[str]:
    """
    Accept category as str or list[str]. Return normalized list[str].
    """
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

# Fix common label typos / variants (defensive)
_CATEGORY_SYNONYMS: Dict[str, str] = {
    # typos from LLM output
    "pther": "other",
    # minor variants
    "clinical address": "clinically address",
    "clinically addresses": "clinically address",
    "prevention": "prevent",
    "not clinically relevant": "not of clinical interest",
}

# Map human labels -> stable ingestion taxonomy
_CATEGORY_CANONICALIZE: Dict[str, str] = {
    "clinically address": "treat",
    "prevent": "prevent",
    "other": "other",
    "not of clinical interest": "not relevant",
    "not relevant": "not relevant",
}

def canonicalize_categories(cat_list: List[str]) -> List[str]:
    """
    Apply synonyms + canonicalization, return deduped list preserving order.
    """
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
    """
    Store a single primary category in DB, deterministic:
      prefer prevent > first matched-in-allowlist > first available > "".
    """
    is_prevent = any(c == "prevent" or c.startswith("prevent") for c in cat_list)
    if is_prevent:
        return "prevent"
    if allowed_categories:
        for c in cat_list:
            if c in allowed_categories:
                return c
    return cat_list[0] if cat_list else ""

# ─────────────────────────────────────────────────────────────────────────────
# Cohort helpers (legacy optional)
# ─────────────────────────────────────────────────────────────────────────────
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
    """
    Legacy: expand parent trial_id -> effective_trial_ids from preproc outputs.
    If not found, returns [trial_id].
    """
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

# ─────────────────────────────────────────────────────────────────────────────
# SQLite schema generators (table-aware)
# ─────────────────────────────────────────────────────────────────────────────
def ddl_for(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id TEXT NOT NULL,
        generated TEXT,
        disease TEXT NOT NULL,
        var_name TEXT NOT NULL,
        stem TEXT NOT NULL,
        stem_var TEXT,                     -- patient_has_finding_of_{{stem}} (no timeframe)
        category TEXT,                     -- treat / prevent / other / not relevant
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
    """
    Strong drop: remove user-created indexes + table.

    NOTE:
      Do NOT try to drop SQLite's internal auto-indexes backing UNIQUE/PK.
      Those have sql IS NULL in sqlite_master and will be removed with DROP TABLE.
    """
    cur = conn.cursor()
    for (idx, sql) in cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    ).fetchall():
        if sql is None:
            continue  # internal auto-index
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def iter_json_files(input_dir: Path) -> Iterable[Path]:
    """
    Recursive to support:
        build/disease_categorized/<trial>/<file>.json

    Preferred filename:
        *_disease_link_filter_summary.json

    If none found, fall back to all *.json (useful if you changed naming).
    """
    preferred = sorted(input_dir.rglob("*_disease_link_filter_summary.json"))
    if preferred:
        yield from preferred
    else:
        yield from sorted(input_dir.rglob("*.json"))

def load_summary(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def extract_rows(blob: Dict[str, Any], allowed_categories: Set[str]) -> Tuple[str, str, List[Tuple]]:
    """
    Uses categorized structure:
      final_selected_concept_by_disease = {
        disease_name: { ..., category: str OR list[str] }
      }

    Returns:
      trial_id, generated, rows list of tuples in the shape:
        (trial_id, generated, disease, var_name, stem, stem_var, category,
         conceptId, preferred_term, fully_specified_name, ctype, definition, best_match_term)

    NOTE:
      For prevention-like categories, we emit BOTH:
        - patient_has_finding_of_<stem>_inthehistory / stem_var=patient_has_finding_of_<stem>
        - patient_wants_to_prevent_<stem>_inthehistory / stem_var=patient_wants_to_prevent_<stem>
    """
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

        # MULTI-LABEL: normalize + canonicalize to stable categories
        cat_list = normalize_category_list(concept.get("category", ""))
        cat_list = canonicalize_categories(cat_list)

        # FILTER: ingest if ANY category matches allowlist
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

        # Always emit the "has_finding" row
        base_var_name = f"patient_has_finding_of_{stem}_inthehistory"
        base_stem_var = f"patient_has_finding_of_{stem}"

        rows.append((
            trial_id,
            (generated or ""),
            disease,
            base_var_name,
            stem,
            base_stem_var,
            category_for_db,
            conceptId,
            preferred_term,
            fully_specified_name,
            ctype,
            definition,
            best_match_term,
        ))

        # If ANY category is prevent-like, ALSO emit the prevention-projected row
        if is_prevent:
            prev_var_name = f"patient_wants_to_prevent_{stem}_inthehistory"
            prev_stem_var = f"patient_wants_to_prevent_{stem}"

            rows.append((
                trial_id,
                (generated or ""),
                disease,
                prev_var_name,
                stem,
                prev_stem_var,
                category_for_db,
                conceptId,
                preferred_term,
                fully_specified_name,
                ctype,
                definition,
                best_match_term,
            ))

    return trial_id, (generated or ""), rows

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="../../build/disease",
                    help="Dir with *_disease_link_filter_summary.json files (supports nested subdirs)")
    ap.add_argument("--db", default="../../build/trial.db",
                    help="SQLite db path (default: trial.db)")

    # Legacy cohort fan-out
    ap.add_argument("--preproc-dir", default="mbench/req_mbench/preproc_logs",
                    help="Directory with {trial_id}_{side}.pre.normalized.json (only used with --cohort-fanout)")
    ap.add_argument("--prefer-side", choices=["inclusion", "exclusion"], default="inclusion",
                    help="Which side’s preproc file to prefer when both exist (only used with --cohort-fanout)")
    ap.add_argument("--cohort-fanout", action="store_true", default=False,
                    help="(legacy) Expand parent trial_id to effective_trial_ids via preproc logs and ingest for each effective id")

    # Trial id selection
    ap.add_argument("--trial-id-from-filename", dest="trial_id_from_filename",
                    action="store_true", default=True,
                    help="(default ON) Override/ensure blob['trial_id'] from filename token (e.g., NCT...a_...)")
    ap.add_argument("--no-trial-id-from-filename", dest="trial_id_from_filename",
                    action="store_false",
                    help="Trust JSON trial_id field as-is (do not override from filename)")

    # Target table
    ap.add_argument("--table", default="disease_constraint_atoms",
                    help="Target table name for ingestion")

    ap.add_argument("--categories", default="treat,other,prevent",
                    help=(
                        "Comma-separated allowlist of CANONICAL categories to ingest "
                        "(exact match after lowercasing). "
                        "Canonicalization maps: clinically address→treat, pther→other, etc."
                    ))

    # Overwrite / freshening controls
    ap.add_argument(
        "--overwrite",
        choices=["all", "by-trial", "keep", "drop"],
        default="all",
        help=(
            "Overwrite target table (used when --no-recreate): "
            "'all' purge everything; "
            "'by-trial' purge per ingested trial_id; "
            "'keep' no purge; "
            "'drop' DROP and recreate the table."
        ),
    )
    ap.add_argument(
        "--recreate",
        dest="recreate",
        action="store_true",
        default=True,
        help="(default ON) DROP table + user indexes and recreate target table before ingesting.",
    )
    ap.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        help="Do not drop the table at start; use --overwrite behavior instead.",
    )

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

    total_files = 0
    total_upserts = 0
    total_filtered_out = 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        if args.recreate:
            print(f"[info] fresh mode: dropping table {table}")
            _drop_table_everything(conn, table)

        # legacy drop mode (only if not recreate)
        if (not args.recreate) and args.overwrite == "drop":
            conn.execute(f"DROP TABLE IF EXISTS {table};")
            conn.commit()

        conn.execute(ddl_for(table))

        # --- MIGRATIONS: ensure stem_var + category columns exist before indexes ---
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if "stem_var" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN stem_var TEXT;")
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

        # global purge (only if not recreate)
        if (not args.recreate) and args.overwrite == "all":
            conn.execute(f"DELETE FROM {table}")
            conn.commit()

        UPSERT_SQL = upsert_sql_for(table)

        for fp in iter_json_files(input_dir):
            try:
                blob = load_summary(fp)

                # Default: prefer filename-derived trial_id (robust for subcohorts)
                if args.trial_id_from_filename:
                    tid_fn = trial_id_from_filename(fp)
                    if tid_fn:
                        blob["trial_id"] = tid_fn

                parent_tid, generated, rows = extract_rows(blob, allowed)

                # Choose ingested trial_ids:
                #   default: per-file (no fan-out)
                #   legacy: cohort fan-out via preproc effective_trial_ids
                if args.cohort_fanout:
                    effective_ids = find_effective_trial_ids(parent_tid, preproc_dir, args.prefer_side)
                else:
                    effective_ids = [parent_tid]

                # by-trial purge (only if not recreate)
                if (not args.recreate) and args.overwrite == "by-trial":
                    conn.executemany(
                        f"DELETE FROM {table} WHERE trial_id = ?",
                        [(tid_eff,) for tid_eff in effective_ids]
                    )

                # If diseases exist but all filtered out
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
                rel = fp.relative_to(input_dir) if fp.is_relative_to(input_dir) else fp
                print(f"[✓] {rel} → {table}: upserts={up}{eff_note}")
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