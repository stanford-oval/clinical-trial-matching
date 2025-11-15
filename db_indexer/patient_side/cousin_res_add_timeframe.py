#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cousin_res_add_timeframe

对每个 patient_id（从 txt 每行一个读取），处理文件：
  <root>/<patient_id>/<subdir>/canonical_embedding_new_variables.jsonl

对 JSONL 每一行（一个 JSON object），新增四个字段：
  start_time_in_hours
  end_time_in_hours
  start_time_inclusive
  end_time_inclusive

这四个字段的值只从下面四个字段复制（只用 largest，不用 smallest）：
  largest_timewindow_start_time_in_hours
  largest_timewindow_end_time_in_hours
  largest_timewindow_start_time_inclusive
  largest_timewindow_end_time_inclusive

输出到：
  <root>/<patient_id>/<subdir>/canonical_embedding_new_variables_with_timeframe.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


LARGEST_KEYS = {
    "start_time_in_hours": "largest_timewindow_start_time_in_hours",
    "end_time_in_hours": "largest_timewindow_end_time_in_hours",
    "start_time_inclusive": "largest_timewindow_start_time_inclusive",
    "end_time_inclusive": "largest_timewindow_end_time_inclusive",
}


def read_patient_ids(txt_path: Path) -> List[str]:
    ids: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def add_timeframe_from_largest(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    把 largest_timewindow_* 复制到 start/end timeframe 字段。
    返回 (obj, missing_keys)
    """
    missing = [src for src in LARGEST_KEYS.values() if src not in obj]
    if missing:
        return obj, missing

    for dst, src in LARGEST_KEYS.items():
        obj[dst] = obj[src]
    return obj, []


def process_one_file(in_path: Path, out_path: Path, strict: bool = False) -> Tuple[int, int]:
    """
    返回 (written_lines, unchanged_lines_due_to_missing_largest)
    strict=True: JSON 解析失败或缺 key 直接报错退出
    strict=False: 缺 key 的行保留原样并警告
    """
    written = 0
    unchanged = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(out_path.parent),
        prefix=out_path.name + ".tmp.",
    ) as tmp:
        tmp_path = Path(tmp.name)

        for ln, line in enumerate(fin, start=1):
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    raise ValueError("JSON line is not an object")
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Bad JSON at {in_path}:{ln}: {e}") from e
                print(f"[WARN] Bad JSON, keep unchanged: {in_path}:{ln} ({e})", file=sys.stderr)
                tmp.write(line if line.endswith("\n") else (line + "\n"))
                written += 1
                continue

            obj, missing = add_timeframe_from_largest(obj)
            if missing:
                if strict:
                    raise RuntimeError(f"Missing keys {missing} at {in_path}:{ln}")
                unchanged += 1
                print(f"[WARN] Missing largest_timewindow keys {missing}, keep unchanged: {in_path}:{ln}", file=sys.stderr)

            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    tmp_path.replace(out_path)
    return written, unchanged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient-ids-txt", required=True, type=Path, help="txt 文件：每行一个 patient id")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("<SATIR_ROOT>/patient_build_inclusion_with_cousins/patient_facts_export_inclusion"),
        help="根目录：包含 <patient_id>/<subdir>/... 的那一层",
    )
    ap.add_argument("--subdir", default="inclusion", help="通常是 inclusion / exclusion")
    ap.add_argument("--in-name", default="canonical_embedding_new_variables.jsonl")
    ap.add_argument("--out-name", default="canonical_embedding_new_variables_with_timeframe.jsonl")
    ap.add_argument("--strict", action="store_true", help="遇到 bad JSON 或缺 largest keys 直接报错退出")
    args = ap.parse_args()

    patient_ids = read_patient_ids(args.patient_ids_txt)
    if not patient_ids:
        print(f"[ERROR] No patient ids found in {args.patient_ids_txt}", file=sys.stderr)
        return 2

    total_written = 0
    total_unchanged = 0
    missing_files = 0

    for pid in patient_ids:
        in_path = args.root / pid / args.subdir / args.in_name
        out_path = args.root / pid / args.subdir / args.out_name

        if not in_path.exists():
            missing_files += 1
            print(f"[WARN] Missing input file: {in_path}", file=sys.stderr)
            continue

        written, unchanged = process_one_file(in_path, out_path, strict=args.strict)
        total_written += written
        total_unchanged += unchanged
        print(f"[OK] {pid}: wrote {written} lines -> {out_path} (unchanged_missing_largest={unchanged})")

    print(
        f"[DONE] patients={len(patient_ids)} missing_files={missing_files} "
        f"total_lines_written={total_written} total_lines_unchanged_missing_largest={total_unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())