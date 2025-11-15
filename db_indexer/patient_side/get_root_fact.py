#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_sigir_into_canonical.py

Merge src lines into dst, dedupe by entity_variable_name (keep existing dst first).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    items.append(obj)
                else:
                    print(f"[WARN] {path}:{lineno} is not a JSON object, skipped.", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno} JSON decode error: {e}. Skipped.", file=sys.stderr)
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def normalize_patient_id(raw: str) -> Tuple[str, str]:
    """
    Returns (base_pid, src_pid_dirname)
      - base_pid: used for dst folder name
      - src_pid_dirname: used for src folder name
    If raw endswith "_inclusion", base_pid is stripped; else src dirname = raw + "_inclusion".
    """
    raw = raw.strip()
    if raw.endswith("_inclusion"):
        base = raw[: -len("_inclusion")]
        return base, raw
    return raw, f"{raw}_inclusion"


def get_varname(obj: Dict[str, Any]) -> Optional[str]:
    # primary key per your requirement
    v = obj.get("entity_variable_name")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def merge_one_patient(
    base_pid: str,
    src_pid_dir: str,
    src_root: Path,
    dst_root: Path,
    src_files: List[str],
    dst_file: str,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    Returns (kept_total, added_count, skipped_dup_count)
    """
    src_dir = src_root / src_pid_dir
    dst_dir = dst_root / src_pid_dir
    dst_path = dst_dir / dst_file

    dst_items = read_jsonl(dst_path)

    var2idx: Dict[str, int] = {}
    added = 0
    skipped_dup = 0

    # Build index from existing dst (prefer keeping these)
    for i, obj in enumerate(dst_items):
        var = get_varname(obj)
        if var is not None and var not in var2idx:
            var2idx[var] = i

    # Merge sources
    for rel in src_files:
        src_path = src_dir / rel
        src_items = read_jsonl(src_path)
        if not src_items:
            # file missing or empty is OK
            continue

        for obj in src_items:
            var = get_varname(obj)
            if var is None:
                # No varname: append (can't dedupe by varname)
                dst_items.append(obj)
                added += 1
                continue

            if var in var2idx:
                skipped_dup += 1
                continue

            var2idx[var] = len(dst_items)
            dst_items.append(obj)
            added += 1

    if dry_run:
        return (len(dst_items), added, skipped_dup)

    write_jsonl(dst_path, dst_items)
    return (len(dst_items), added, skipped_dup)


def iter_patient_ids(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"patient txt not found: {txt_path}")
    out: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--patient-txt", required=True, type=Path, help="txt file with patient ids (one per line)")
    p.add_argument(
        "--src-root",
        type=Path,
        default=Path("<SATIR_ROOT>/patient_build_sigir_inclusion_216/patient_coded_results"),
    )
    p.add_argument(
        "--dst-root",
        type=Path,
        default=Path("<SATIR_ROOT>/patient_build_sigir_inclusion_216/patient_coded_results"),
    )
    p.add_argument(
        "--src-files",
        nargs="+",
        default=["diagnosis.jsonl", "embedding_search_other_candidate_canonical.jsonl"],
        help="source jsonl files under each <pid>_inclusion folder",
    )
    p.add_argument(
        "--dst-file",
        default="canonical.jsonl",
        help="destination jsonl file name under each <pid> folder",
    )
    p.add_argument("--dry-run", action="store_true", help="do not write, just print stats")
    args = p.parse_args()

    patient_ids = iter_patient_ids(args.patient_txt)
    if not patient_ids:
        print("[WARN] No patient ids found in txt.", file=sys.stderr)
        return 0

    total_added = 0
    total_skipped = 0
    total_patients = 0

    for raw_pid in patient_ids:
        base_pid, src_pid_dir = normalize_patient_id(raw_pid)
        kept_total, added, skipped = merge_one_patient(
            base_pid=base_pid,
            src_pid_dir=src_pid_dir,
            src_root=args.src_root,
            dst_root=args.dst_root,
            src_files=args.src_files,
            dst_file=args.dst_file,
            dry_run=args.dry_run,
        )
        total_patients += 1
        total_added += added
        total_skipped += skipped

        action = "DRYRUN" if args.dry_run else "WROTE"
        print(f"[{action}] {raw_pid} -> dst={src_pid_dir}: kept_total={kept_total}, added={added}, skipped_dup={skipped}")

    print(f"[DONE] patients={total_patients}, total_added={total_added}, total_skipped_dup={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())