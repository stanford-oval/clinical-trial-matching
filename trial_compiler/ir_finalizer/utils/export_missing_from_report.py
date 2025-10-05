#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python export_missing_from_report.py \
  --report-jsonl ../build/ir_strict_final/export_report.jsonl \
  --out-dir ../build/ir_missing_strict_ok \
  --copy-all-tried
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r']
    for ch in bad:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export rows with exported=false from export_report.jsonl for manual inspection.")
    ap.add_argument("--report-jsonl", required=True, help="Path to export_report.jsonl from export_strict_fallback.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--copy-all-tried", action="store_true", help="Copy every tried path that exists (stage3/stage2/stage1/input).")
    args = ap.parse_args()

    report = Path(args.report_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    missing = 0
    copied = 0

    with report.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: Dict[str, Any] = json.loads(line)
            if row.get("exported") is True:
                continue

            eff = row.get("effective_trial_id")
            side = row.get("side")
            tried = row.get("tried", [])
            if not eff or not side:
                continue

            missing += 1
            pair_dir = out_dir / f"{eff}_{side}"
            ensure_dir(pair_dir)

            # Write inspection metadata
            (pair_dir / "meta.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

            # Copy input (always if exists in tried list)
            for t in tried:
                label = t.get("label")
                path = t.get("path")
                if not label or not path:
                    continue
                src = Path(path)
                if not src.exists() or not src.is_file():
                    continue

                if (not args.copy_all_tried) and label != "input":
                    continue

                dst_name = f"{label}__{safe_name(t.get('errors', ['unknown'])[0] if t.get('errors') else 'unknown')}__{src.name}"
                dst = pair_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1

    print(f"[DONE] missing_pairs={missing} copied_files={copied} out_dir={out_dir}")


if __name__ == "__main__":
    main()
