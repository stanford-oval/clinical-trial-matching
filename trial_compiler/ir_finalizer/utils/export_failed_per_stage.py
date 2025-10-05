#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_failed_per_stage.py

Export strict-invalid / quarantined SMT files per stage for inspection.

For each stage (stage0_meaning, stage1_polarity, stage2_repair, stage3_logic), it can export:
  - merged_ir strict failures: stage*/merged_ir/*.smt2 that fail strict compliance
  - raw_out strict failures: stage*/raw_out/*.smt2 that fail strict compliance
  - quarantined files:       stage*/quarantine_invalid/*  (copied as-is)

Outputs:
  out_dir/
    stage0_meaning/{merged_ir_strict_fail, raw_out_strict_fail, quarantine_invalid}/...
    stage1_polarity/...
    stage2_repair/...
    stage3_logic/...
    export_failed_report.jsonl
    export_failed_summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from trial_compiler.ir_finalizer.orchestrators.orchestrate_ir_fixes import (  # type: ignore
    STRICT_GUARDRAILS_DEFAULT,
    ensure_dir,
    find_latest_work_dir,
    is_nonempty_smt,
    validate_smt_file,
)

STAGES = [
    "stage0_meaning",
    "stage1_polarity",
    "stage2_repair",
    "stage3_logic",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export per-stage strict-invalid / quarantined SMT files for inspection.")
    p.add_argument("--ir-dir", default="../build/ir", help="Original canonical IR dir (used only to auto-find latest work dir).")
    p.add_argument("--work-dir", default=None, help="ir_orchestrated_* work dir. If omitted, auto-picks newest under parent of --ir-dir.")
    p.add_argument("--out-dir", required=True, help="Output directory for exported failures.")
    p.add_argument("--skip-assert-checks", action="store_true", help="Match orchestrator mode: disable assert/:named checks.")
    p.add_argument("--clean-out-dir", action="store_true", help="If set, clear stage* subfolders under --out-dir before exporting.")
    p.add_argument("--include-merged-ir", action="store_true", default=True, help="Export strict-invalid files from merged_ir (default: on).")
    p.add_argument("--include-raw-out", action="store_true", default=True, help="Export strict-invalid files from raw_out (default: on).")
    p.add_argument("--include-quarantine", action="store_true", default=True, help="Export all files from quarantine_invalid (default: on).")
    p.add_argument("--max-errors-per-file", type=int, default=20, help="Cap errors recorded per file in the report.")
    return p.parse_args()


def _sanitize(s: str, max_len: int = 120) -> str:
    # make safe-ish for filenames
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r', '[', ']', '{', '}', '(', ')', ',', "'"]
    for ch in bad:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s[:max_len].strip("_") or "unknown"


def strict_ok_with_errs(p: Path, *, strict_cfg) -> Tuple[bool, List[str]]:
    if not p.exists():
        return False, ["missing"]
    if not is_nonempty_smt(p):
        return False, ["empty_or_whitespace"]
    ok, errs = validate_smt_file(p, cfg=strict_cfg)
    return ok, [str(e) for e in errs]


def copy_unique(src: Path, dst_dir: Path, new_name: str) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / new_name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stem = dst.stem
    suf = dst.suffix
    for i in range(1, 10_000):
        cand = dst_dir / f"{stem}__{i}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
    raise RuntimeError(f"Too many collisions copying {src} -> {dst_dir}")


def export_dir_strict_failures(
    *,
    stage_name: str,
    source_label: str,
    src_dir: Path,
    dst_dir: Path,
    strict_cfg,
    report_path: Path,
    reason_counter: Counter,
    file_counter: Counter,
    max_errors_per_file: int,
) -> int:
    if not src_dir.exists():
        return 0

    n = 0
    for p in sorted(src_dir.glob("*.smt2")):
        if not p.is_file():
            continue
        ok, errs = strict_ok_with_errs(p, strict_cfg=strict_cfg)
        if ok:
            continue

        primary = errs[0] if errs else "unknown_error"
        reason_counter[f"{stage_name}:{source_label}:{primary}"] += 1
        file_counter[f"{stage_name}:{source_label}:files_strict_invalid"] += 1

        dst_name = f"{_sanitize(primary)}__{p.name}"
        copied_to = copy_unique(p, dst_dir, dst_name)

        row = {
            "stage": stage_name,
            "source": source_label,
            "path": str(p),
            "strict_ok": False,
            "errors": errs[:max_errors_per_file],
            "copied_to": str(copied_to),
        }
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    return n


def export_quarantine(
    *,
    stage_name: str,
    src_dir: Path,
    dst_dir: Path,
    strict_cfg,
    report_path: Path,
    reason_counter: Counter,
    file_counter: Counter,
    max_errors_per_file: int,
) -> int:
    if not src_dir.exists():
        return 0

    n = 0
    for p in sorted(src_dir.glob("*")):
        if not p.is_file():
            continue
        # We try to validate if it's a .smt2; otherwise just copy
        errs: List[str] = []
        strict_ok = None
        if p.suffix == ".smt2":
            ok, verrs = strict_ok_with_errs(p, strict_cfg=strict_cfg)
            strict_ok = ok
            errs = verrs
            primary = errs[0] if errs else "unknown_error"
            reason_counter[f"{stage_name}:quarantine:{primary}"] += 1
        file_counter[f"{stage_name}:quarantine:files_copied"] += 1

        copied_to = copy_unique(p, dst_dir, p.name)
        row = {
            "stage": stage_name,
            "source": "quarantine_invalid",
            "path": str(p),
            "strict_ok": strict_ok,
            "errors": errs[:max_errors_per_file],
            "copied_to": str(copied_to),
        }
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    return n


def main() -> None:
    args = parse_args()

    strict_cfg = STRICT_GUARDRAILS_DEFAULT
    if args.skip_assert_checks:
        strict_cfg = replace(strict_cfg, check_asserts_and_named=False)

    in_ir_dir = Path(args.ir_dir).resolve()
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
    else:
        latest = find_latest_work_dir(in_ir_dir.parent)
        if latest is None:
            raise FileNotFoundError(f"Could not auto-find ir_orchestrated_* under: {in_ir_dir.parent}")
        work_dir = latest.resolve()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    report_path = out_dir / "export_failed_report.jsonl"
    summary_path = out_dir / "export_failed_summary.json"
    if report_path.exists():
        report_path.unlink()

    if args.clean_out_dir:
        for s in STAGES:
            d = out_dir / s
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    reason_counter: Counter[str] = Counter()
    file_counter: Counter[str] = Counter()

    exported_total = 0

    for stage in STAGES:
        stage_dir = work_dir / stage
        merged = stage_dir / "merged_ir"
        raw_out = stage_dir / "raw_out"
        quarantine = stage_dir / "quarantine_invalid"

        stage_out = out_dir / stage
        ensure_dir(stage_out)

        if args.include_merged_ir:
            exported_total += export_dir_strict_failures(
                stage_name=stage,
                source_label="merged_ir",
                src_dir=merged,
                dst_dir=stage_out / "merged_ir_strict_fail",
                strict_cfg=strict_cfg,
                report_path=report_path,
                reason_counter=reason_counter,
                file_counter=file_counter,
                max_errors_per_file=args.max_errors_per_file,
            )

        if args.include_raw_out:
            exported_total += export_dir_strict_failures(
                stage_name=stage,
                source_label="raw_out",
                src_dir=raw_out,
                dst_dir=stage_out / "raw_out_strict_fail",
                strict_cfg=strict_cfg,
                report_path=report_path,
                reason_counter=reason_counter,
                file_counter=file_counter,
                max_errors_per_file=args.max_errors_per_file,
            )

        if args.include_quarantine:
            exported_total += export_quarantine(
                stage_name=stage,
                src_dir=quarantine,
                dst_dir=stage_out / "quarantine_invalid",
                strict_cfg=strict_cfg,
                report_path=report_path,
                reason_counter=reason_counter,
                file_counter=file_counter,
                max_errors_per_file=args.max_errors_per_file,
            )

    summary = {
        "work_dir": str(work_dir),
        "out_dir": str(out_dir),
        "skip_assert_checks": bool(args.skip_assert_checks),
        "include_merged_ir": bool(args.include_merged_ir),
        "include_raw_out": bool(args.include_raw_out),
        "include_quarantine": bool(args.include_quarantine),
        "exported_items_total": exported_total,
        "file_counters": dict(file_counter),
        "top_reasons": reason_counter.most_common(50),
        "report_jsonl": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[EXPORT FAILED DONE]")
    print(f"  work_dir: {work_dir}")
    print(f"  out_dir:  {out_dir}")
    print(f"  exported items: {exported_total}")
    print(f"  report: {report_path}")
    print(f"  summary: {summary_path}")
    print("", flush=True)


if __name__ == "__main__":
    main()
