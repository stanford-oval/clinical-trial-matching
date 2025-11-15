#!/usr/bin/env python3
from __future__ import annotations
import argparse, sqlite3, json
from pathlib import Path
from typing import List, Tuple, Optional

def _iter_canon_files(canon_dir: Path):
    yield from sorted(canon_dir.glob("*.json"))

def _load_predicate_to_concept_from_canon(canon_dir: Path) -> List[Tuple[str, str]]:
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

    # de-dup by first occurrence
    seen = set()
    dedup: List[Tuple[str, str]] = []
    for vn, cid in results:
        if vn not in seen:
            seen.add(vn)
            dedup.append((vn, cid))
    return dedup

def main():
    ap = argparse.ArgumentParser(description="Rebuild predicate_to_concept from canon JSON files.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--canon-dir", required=True)
    ap.add_argument("--minified-canon-dir", required=False)  # kept for interface compatibility (unused here)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    canon_dir = Path(args.canon_dir).expanduser().resolve()
    if not canon_dir.is_dir():
        raise SystemExit(f"[!] Not a dir: {canon_dir}")

    pairs = _load_predicate_to_concept_from_canon(canon_dir)
    if not pairs:
        raise SystemExit(f"[!] No predicate_to_concept pairs found in canon dir: {canon_dir}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS predicate_to_concept (var_name TEXT PRIMARY KEY, concept_id TEXT NOT NULL)")
        if args.overwrite:
            conn.execute("DELETE FROM predicate_to_concept")
        conn.executemany("INSERT OR REPLACE INTO predicate_to_concept(var_name, concept_id) VALUES (?,?)", pairs)
        conn.commit()
        print(f"[ok] predicate_to_concept rows: {len(pairs)}  db={db_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
