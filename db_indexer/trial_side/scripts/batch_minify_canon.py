#!/usr/bin/env python3
# batch_minify_canon.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Optional
import json

# --- Config (edit if needed) ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling stage modules
from smt_core.buildroot import build_root as _build_root

ROOT = _build_root().parent

IN_DIR = ROOT / "build" / "canon_expanded"
OUT_DIR = ROOT / "build" / "minified_canon"
GLOB = "*.json"
WRITE_INDEX = True
WRITE_FLAT_TXT = True
OUT_SUFFIX = "_entity_variable_names.json"  # output filename = <input.stem> + this suffix

# --- Helpers ---

CONTAINER_KEYS: Sequence[str] = (
    "canonical_variables",
    "new_canonical_variable_declarations",  # future-friendly
)

def _unique_preserve_order(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _extract_from_obj(obj: Any) -> List[str]:
    """
    Returns a de-duplicated, order-preserving list of entity_variable_name values
    from a parsed JSON object shaped like:
      { "canonical_variables": [ {...}, ... ] }
    or directly a list of dicts.
    """
    items: Optional[List[Any]] = None

    if isinstance(obj, dict):
        for k in CONTAINER_KEYS:
            v = obj.get(k)
            if isinstance(v, list):
                items = v
                break
        if items is None and isinstance(obj.get("entity_variable_name"), str):
            items = [obj]
    elif isinstance(obj, list):
        items = obj

    if not items:
        return []

    names: List[str] = []
    for it in items:
        if isinstance(it, dict):
            v = it.get("entity_variable_name")
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
    return _unique_preserve_order(names)

def extract_entity_variable_names_from_file(path: Path, encoding: str = "utf-8") -> List[str]:
    return _extract_from_obj(json.loads(path.read_text(encoding=encoding)))

def minify_one_file(in_path: Path, out_dir: Path, *, encoding: str = "utf-8") -> List[str]:
    names = extract_entity_variable_names_from_file(in_path, encoding=encoding)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}{OUT_SUFFIX}"
    out_path.write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding=encoding)
    return names

def batch_minify(in_dir: Path, out_dir: Path, glob: str = "*.json") -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for p in sorted(in_dir.glob(glob)):
        try:
            names = minify_one_file(p, out_dir)
            mapping[p.name] = names
            print(f"[ok] {p.name}  ->  {p.stem}{OUT_SUFFIX}  (vars={len(names)})")
        except Exception as e:
            print(f"[warn] Failed {p.name}: {e}")
    return mapping

# --- Run ---
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = batch_minify(IN_DIR, OUT_DIR, GLOB)

    if WRITE_INDEX:
        (OUT_DIR / "_index.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if WRITE_FLAT_TXT:
        flat = _unique_preserve_order(n for names in results.values() for n in names)
        (OUT_DIR / "_all_entity_variable_names.txt").write_text(
            "\n".join(flat) + "\n", encoding="utf-8"
        )

    print(f"\n[done] Wrote {len(results)} file(s) to {OUT_DIR}")
