#!/usr/bin/env python3
"""
trial_disease_to_sqlite.py
────────────────────────────────────────────────────────────────────────────
Load *_disease_link_filter_summary.json files and write all rows into a single
SQLite table named `disease_constraint_atoms`.

Subcohort/effective-id aware:
  • If the input JSON's trial_id is ALREADY an effective ID (e.g., NCT...a, NCT...b),
    we upsert ONLY for that trial_id (no fan-out).
  • If the input JSON's trial_id is a parent ID (e.g., NCT02004509) and preproc logs
    provide effective_trial_ids, we fan-out (upsert one copy per effective id).
  • If no cohort info exists, we upsert once for the parent trial_id (legacy behavior).

For each disease concept, we emit:
  - var_name: patient_has_finding_of_<normalized_stem>_inthehistory
  - stem: <normalized_stem>
  - stem_var: patient_has_finding_of_<normalized_stem>  (NO timeframe)
  - disease, conceptId, preferred_term, fully_specified_name, type, definition, best_match_term
  - trial_id (effective or parent), generated

Idempotent: upserts by (trial_id, var_name).
"""

from __future__ import annotations
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────
_PAREN_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_EFFECTIVE_TRIAL_RE = re.compile(r"^(NCT\d+)([a-z]+)$", re.I)  # NCT + digits + subcohort suffix letters

def to_var_snake(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PAREN_TAIL_RE.sub("", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def is_effective_trial_id(tid: str) -> bool:
    tid = (tid or "").strip()
    return bool(_EFFECTIVE_TRIAL_RE.match(tid))

# ─────────────────────────────────────────────────────────────────────────────
# Cohort helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_json_if_exists(fp: Path) -> Optional[Dict[str, Any]]:
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
    If trial_id is already effective (NCT...a/b/..), return [trial_id].
    Else, try to read preproc {trial_id}_{side}.pre.normalized.json and return effective_trial_ids.
    Fallback: [trial_id].
    """
    trial_id = (trial_id or "").strip()
    if not trial_id:
        return ["UNKNOWN_TRIAL_ID"]

    # ✅ Key rule: if already subcohort/effective id, do NOT fan out
    if is_effective_trial_id(trial_id):
        return [trial_id]

    if not preproc_dir.is_dir():
        return [trial_id]

    prefer_side = "inclusion" if prefer_side not in ("inclusion", "exclusion") else prefer_side
    other_side = "exclusion" if prefer_side == "inclusion" else "inclusion"

    def _fp(side: str) -> Path:
        return preproc_dir / f"{trial_id}_{side}.pre.normalized.json"

    for side in (prefer_side, other_side):
        blob = _read_json_if_exists(_fp(side))
        if isinstance(blob, dict):
            eff = blob.get("effective_trial_ids")
            if isinstance(eff, list) and eff:
                out = [str(x).strip() for x in eff if isinstance(x, str) and str(x).strip()]
                if out:
                    return out

    return [trial_id]

# ─────────────────────────────────────────────────────────────────────────────
# SQLite schema
# ─────────────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS disease_constraint_atoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    generated TEXT,
    disease TEXT NOT NULL,
    var_name TEXT NOT NULL,
    stem TEXT NOT NULL,
    stem_var TEXT,                     -- patient_has_finding_of_{stem} (no timeframe)
    conceptId TEXT,
    preferred_term TEXT,
    fully_specified_name TEXT,
    type TEXT,
    definition TEXT,
    best_match_term TEXT,
    UNIQUE(trial_id, var_name)
);
"""

UPSERT_SQL = """
INSERT INTO disease_constraint_atoms (
    trial_id, generated, disease, var_name, stem, stem_var,
    conceptId, preferred_term, fully_specified_name, type, definition, best_match_term
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(trial_id, var_name) DO UPDATE SET
    generated=excluded.generated,
    disease=excluded.disease,
    stem=excluded.stem,
    stem_var=excluded.stem_var,
    conceptId=excluded.conceptId,
    preferred_term=excluded.preferred_term,
    fully_specified_name=excluded.fully_specified_name,
    type=excluded.type,
    definition=excluded.definition,
    best_match_term=excluded.best_match_term;
"""

BASE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_dli_trial ON disease_constraint_atoms(trial_id);",
    "CREATE INDEX IF NOT EXISTS idx_dli_var ON disease_constraint_atoms(var_name);",
    "CREATE INDEX IF NOT EXISTS idx_dli_stem ON disease_constraint_atoms(stem);",
]
STEM_VAR_INDEX = "CREATE INDEX IF NOT EXISTS idx_dli_stem_var ON disease_constraint_atoms(stem_var);"

# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def iter_json_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*_disease_link_filter_summary.json"))

def load_summary(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def extract_rows(blob: Dict[str, Any]) -> Tuple[str, Optional[str], List[Tuple]]:
    """
    Returns:
      source_trial_id, generated, rows
    Each row tuple matches UPSERT_SQL order (minus trial_id, generated).
    """
    trial_id = (blob.get("trial_id") or "").strip() or "UNKNOWN_TRIAL_ID"
    generated = (blob.get("generated") or "").strip() or None

    concepts = blob.get("final_selected_concept_by_disease") or blob.get("linked_result") or {}
    rows: List[Tuple] = []

    if not isinstance(concepts, dict):
        return trial_id, generated, rows

    for disease_name, concept in concepts.items():
        disease = str(disease_name)

        conceptId = ""
        preferred_term = ""
        fully_specified_name = ""
        ctype = ""
        definition = ""
        best_match_term = ""

        if isinstance(concept, dict):
            conceptId = str(concept.get("conceptId") or "").strip()
            preferred_term = str(concept.get("preferred_term") or "").strip()
            fully_specified_name = str(concept.get("fully_specified_name") or "").strip()
            ctype = str(concept.get("type") or "").strip()
            definition = str(concept.get("definition") or "").strip()
            best_match_term = str(concept.get("best_match_term") or "").strip()

        stem_src = preferred_term or fully_specified_name or disease
        stem = to_var_snake(stem_src)
        var_name = f"patient_has_finding_of_{stem}_inthehistory"
        stem_var = f"patient_has_finding_of_{stem}"

        rows.append((
            disease, var_name, stem, stem_var,
            conceptId, preferred_term, fully_specified_name, ctype, definition, best_match_term
        ))

    return trial_id, generated, rows

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="../../build/disease",
                    help="Dir with *_disease_link_filter_summary.json files")
    ap.add_argument("--db", default="../../build/trial.db",
                    help="SQLite db path (default: trial.db)")
    ap.add_argument("--preproc-dir", default="mbench/req_mbench/preproc_logs",
                    help="Directory with {trial_id}_{side}.pre.normalized.json from the preprocessor (only used if input JSON is parent-level)")
    ap.add_argument("--prefer-side", choices=["inclusion", "exclusion"], default="inclusion",
                    help="Which side’s preproc file to prefer when both exist")
    ap.add_argument("--overwrite", choices=["all", "by-trial", "keep"], default="all",
                    help="Overwrite disease_constraint_atoms: 'all' purge everything; 'by-trial' purge per effective id; 'keep' no purge.")

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"[!] Not a directory: {input_dir}", file=sys.stderr)
        return 2

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    preproc_dir = Path(args.preproc_dir).expanduser().resolve()

    total_files = 0
    total_rows_targeted = 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Create table if brand-new
        conn.execute(CREATE_TABLE_SQL)

        # --- MIGRATION FIRST: ensure stem_var column exists before any indexes ---
        cols = [r[1] for r in conn.execute("PRAGMA table_info(disease_constraint_atoms)").fetchall()]
        if "stem_var" not in cols:
            conn.execute("ALTER TABLE disease_constraint_atoms ADD COLUMN stem_var TEXT;")
            # Backfill from existing rows
            conn.execute("""
                UPDATE disease_constraint_atoms
                   SET stem_var = 'patient_has_finding_of_' || stem
                 WHERE stem_var IS NULL OR stem_var = '';
            """)
            conn.commit()

        # Now create indexes (safe even if they already exist)
        for idx_sql in BASE_INDEXES:
            conn.execute(idx_sql)
        conn.execute(STEM_VAR_INDEX)

        # global purge
        if args.overwrite == "all":
            conn.execute("DELETE FROM disease_constraint_atoms")
            conn.commit()

        for fp in iter_json_files(input_dir):
            try:
                blob = load_summary(fp)
                source_tid, generated, rows = extract_rows(blob)

                if not rows:
                    print(f"[i] {fp.name}: no rows (empty or malformed concept map)")
                    total_files += 1
                    continue

                # Expand to effective ids ONLY if source_tid is parent-level
                effective_ids = find_effective_trial_ids(source_tid, preproc_dir, args.prefer_side)

                # by-trial purge
                if args.overwrite == "by-trial":
                    conn.executemany(
                        "DELETE FROM disease_constraint_atoms WHERE trial_id = ?",
                        [(tid_eff,) for tid_eff in effective_ids]
                    )

                # fan-out upserts
                batch: List[Tuple] = []
                for tid_eff in effective_ids:
                    for (disease, var_name, stem, stem_var,
                         conceptId, preferred_term, fsn, ctype, definition, best_match_term) in rows:
                        batch.append((
                            tid_eff, generated, disease, var_name, stem, stem_var,
                            conceptId, preferred_term, fsn, ctype, definition, best_match_term
                        ))

                conn.executemany(UPSERT_SQL, batch)
                conn.commit()

                total_files += 1
                total_rows_targeted += len(batch)

                if is_effective_trial_id(source_tid):
                    note = " (effective-id input; no fan-out)"
                elif effective_ids != [source_tid]:
                    note = f" (fan-out cohorts: {', '.join(effective_ids)})"
                else:
                    note = ""

                print(f"[✓] {fp.name}: targeted_rows={len(batch)}{note}")

            except Exception as e:
                print(f"[✗] {fp.name}: {e}", file=sys.stderr)

    print(f"\nDone. Files processed: {total_files}, targeted upserts: {total_rows_targeted}")
    print(f"SQLite DB: {db_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
