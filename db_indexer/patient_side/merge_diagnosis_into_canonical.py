#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_diagnosis_into_canonical.py
Append diagnosis.jsonl (with extracted_value forced to true) into canonical.jsonl
for each patient folder, while backing up the original canonical.jsonl.

Usage:
  python merge_diagnosis_into_canonical.py --root /path/to/patient_coded_results

Options:
  --dry-run           : Show what would happen, without writing.
  --backup-suffix S   : Custom backup suffix (default: ".before_diag.<timestamp>.jsonl")
  --verbose           : Print per-folder actions.
  --string-true       : Write extracted_value as the string "true" instead of boolean true.
                        (default: boolean true)
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ensure_trailing_newline(s: str) -> str:
    return s if s.endswith("\n") else (s + "\n")

def backup_file(src: Path, suffix: str | None, dry_run: bool = False, verbose: bool = False) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if suffix is None:
        backup_name = src.stem + f".before_diag.{ts}" + "".join(src.suffixes)
    else:
        backup_name = src.stem + f"{suffix}" + "".join(src.suffixes)

    backup_path = src.with_name(backup_name)
    if dry_run:
        if verbose:
            eprint(f"[dry-run] Would back up: {src.name}  ->  {backup_path.name}")
        return backup_path

    shutil.copy2(src, backup_path)
    if verbose:
        eprint(f"[backup] {src.name} -> {backup_path.name}")
    return backup_path

def transform_diag_line_to_true(line: str, as_string: bool) -> str | None:
    """
    Convert a diagnosis JSONL line so that extracted_value is set to true (or "true").
    Returns the transformed JSON line (with newline), or None if the input line is empty/invalid.
    """
    s = line.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        # If it's not valid JSON, skip emitting (but count as read)
        return None

    # Force extracted_value to boolean True or string "true"
    obj["extracted_value"] = ("true" if as_string else True)

    # Write compact but deterministic JSON
    out = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return ensure_trailing_newline(out)

def append_diagnosis_as_true(src: Path, dst: Path, as_string: bool, dry_run: bool = False) -> tuple[int, int]:
    """
    Append all lines from diagnosis.jsonl to canonical.jsonl, forcing extracted_value to true.
    Returns (lines_appended, lines_transformed).
    """
    if not src.exists():
        return (0, 0)

    appended = 0
    transformed = 0

    if dry_run:
        # Count lines; still attempt to parse to count transformables
        with src.open("r", encoding="utf-8") as f:
            for raw in f:
                appended += 1
                if transform_diag_line_to_true(raw, as_string) is not None:
                    transformed += 1
        return (appended, transformed)

    with src.open("r", encoding="utf-8") as fin, dst.open("a", encoding="utf-8") as fout:
        for raw in fin:
            appended += 1
            out = transform_diag_line_to_true(raw, as_string)
            if out is not None:
                fout.write(out)
                transformed += 1
            else:
                # If unparsable, just append the original line as-is to avoid data loss
                fout.write(ensure_trailing_newline(raw.rstrip("\n")))
    return (appended, transformed)

def process_patient_dir(
    patient_dir: Path,
    dry_run: bool = False,
    backup_suffix: str | None = None,
    verbose: bool = False,
    string_true: bool = False,
) -> tuple[int, int, int]:
    """
    Process one patient folder.
    Returns (lines_appended, lines_transformed, status_code)
      status_code: 0 success, 1 skipped (missing canonical), 2 no diagnosis, 3 error
    """
    canonical = patient_dir / "canonical.jsonl"
    diagnosis = patient_dir / "diagnosis.jsonl"

    if not canonical.exists():
        if verbose:
            eprint(f"[skip] No canonical.jsonl in {patient_dir.name}")
        return (0, 0, 1)

    if not diagnosis.exists():
        if verbose:
            eprint(f"[skip] No diagnosis.jsonl in {patient_dir.name}")
        return (0, 0, 2)

    try:
        backup_file(canonical, backup_suffix, dry_run=dry_run, verbose=verbose)
    except Exception as e:
        eprint(f"[error] Backup failed for {patient_dir.name}: {e}")
        return (0, 0, 3)

    try:
        appended, transformed = append_diagnosis_as_true(diagnosis, canonical, as_string=string_true, dry_run=dry_run)
        if verbose:
            eprint(f"[ok] {patient_dir.name}: appended {appended} line(s); transformed {transformed} diagnosis line(s) to extracted_value=true")
        return (appended, transformed, 0)
    except Exception as e:
        eprint(f"[error] Append failed for {patient_dir.name}: {e}")
        return (0, 0, 3)

def main():
    ap = argparse.ArgumentParser(description="Append diagnosis.jsonl (forced extracted_value=true) into canonical.jsonl across patient subfolders.")
    ap.add_argument("--root", default="../../patient_build/patient_coded_results",
                    help="Path to folder containing per-patient subfolders.")
    ap.add_argument("--dry-run", action="store_true", help="Report actions without modifying files.")
    ap.add_argument("--backup-suffix", default=None,
                    help="Custom backup suffix (e.g., '.orig'). If not set, uses '.before_diag.<timestamp>'.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs to stderr.")
    ap.add_argument("--string-true", action="store_true",
                    help='Write extracted_value as the string "true" instead of boolean true.')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        eprint(f"[fatal] Root not found or not a directory: {root}")
        sys.exit(2)

    total_dirs = 0
    total_appended = 0
    total_transformed = 0
    skipped_missing_canonical = 0
    skipped_no_diag = 0
    errored = 0

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        total_dirs += 1
        appended, transformed, status = process_patient_dir(
            p,
            dry_run=args.dry_run,
            backup_suffix=args.backup_suffix,
            verbose=args.verbose,
            string_true=args.string_true,
        )
        total_appended += appended
        total_transformed += transformed
        if status == 1:
            skipped_missing_canonical += 1
        elif status == 2:
            skipped_no_diag += 1
        elif status == 3:
            errored += 1

    eprint("──────────────── Summary ────────────────")
    eprint(f"Root: {root}")
    eprint(f"Patient folders scanned : {total_dirs}")
    eprint(f"Total lines appended    : {total_appended}")
    eprint(f"Diagnosis transformed   : {total_transformed}")
    eprint(f"Skipped (no canonical)  : {skipped_missing_canonical}")
    eprint(f"Skipped (no diagnosis)  : {skipped_no_diag}")
    eprint(f"Errors                  : {errored}")
    if args.dry_run:
        eprint("Mode: DRY RUN (no files modified)")

if __name__ == "__main__":
    main()
