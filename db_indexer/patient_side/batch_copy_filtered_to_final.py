#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any


CAN_EXC_KEY = "can_be_used_for_exclusion"
DIAG_PREFIX = "patient_has_diagnosis_of_"


def discover_patients(uncertain_export_root: Path, side: str, src_uncertain_filename: str) -> List[str]:
    """
    Auto-discover patients by scanning:
      <uncertain_export_root>/<patient>/<side>/<src_uncertain_filename>
    Only include patients that have the source uncertain file.
    """
    if not uncertain_export_root.exists():
        return []
    out: List[str] = []
    for pdir in uncertain_export_root.iterdir():
        if not pdir.is_dir():
            continue
        patient = pdir.name
        candidate = uncertain_export_root / patient / side / src_uncertain_filename
        if candidate.is_file():
            out.append(patient)
    return sorted(out)


def is_diag_row(obj: Dict[str, Any]) -> bool:
    n = obj.get("entity_variable_name")
    return isinstance(n, str) and n.strip().lower().startswith(DIAG_PREFIX)


def copy_uncertain_filtered(
    patient: str,
    side: str,
    uncertain_export_root: Path,
    dst_export_root: Path,
    src_uncertain_filename: str,
    dst_uncertain_filename: str,
    *,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    src_path = uncertain_export_root / patient / side / src_uncertain_filename
    dst_path = dst_export_root / patient / side / dst_uncertain_filename

    if not src_path.is_file():
        print(f"[skip][uncertain] {patient}: missing src: {src_path}")
        return False

    if dst_path.exists() and not overwrite:
        print(f"[skip][uncertain] {patient}: dst exists (use --overwrite): {dst_path}")
        return False

    if dry_run:
        print(f"[dry-run][uncertain] {patient}: {src_path} -> {dst_path}")
        return True

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"[ok][uncertain] {patient}: {src_path} -> {dst_path}")
    return True


def patch_and_copy_certain(
    patient: str,
    side: str,
    certain_export_root: Path,
    dst_export_root: Path,
    certain_patient_dir_pattern: str,
    src_certain_filename: str,
    dst_certain_filename: str,
    *,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    """
    Read CERTAIN jsonl, set can_be_used_for_exclusion=true for diagnosis rows,
    write to destination as <dst_certain_filename>.
    """
    try:
        certain_patient_dir = certain_patient_dir_pattern.format(patient=patient, side=side)
    except Exception as e:
        raise SystemExit(f"[err] bad --certain-patient-dir-pattern: {certain_patient_dir_pattern} ({e})")

    src_path = certain_export_root / certain_patient_dir / side / src_certain_filename
    dst_path = dst_export_root / patient / side / dst_certain_filename

    if not src_path.is_file():
        print(f"[skip][certain]   {patient}: missing src: {src_path}")
        return False

    if dst_path.exists() and not overwrite:
        print(f"[skip][certain]   {patient}: dst exists (use --overwrite): {dst_path}")
        return False

    if dry_run:
        print(f"[dry-run][certain]   {patient}: {src_path} -> {dst_path} (diag set {CAN_EXC_KEY}=true)")
        return True

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp_write")

    total = 0
    diag_marked = 0
    written = 0

    with src_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            total += 1
            if is_diag_row(obj):
                obj[CAN_EXC_KEY] = True
                diag_marked += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    os.replace(tmp_path, dst_path)
    print(f"[ok][certain]   {patient}: read={total}, diag_marked={diag_marked}, wrote={written} -> {dst_path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "CLI-only paths: copy filtered UNCERTAIN to inferred_fact.jsonl, "
            "and patch CERTAIN diagnosis rows with can_be_used_for_exclusion=true to canonical.jsonl."
        )
    )

    # Required: side (folder name)
    ap.add_argument("--side", required=True, choices=["inclusion", "exclusion"])

    # Required: export roots (directories)
    ap.add_argument("--uncertain-export-root", required=True,
                    help="UNCERTAIN export root (contains <patient>/<side>/...).")
    ap.add_argument("--certain-export-root", required=True,
                    help="CERTAIN export root (contains <certain_patient_dir>/<side>/...).")
    ap.add_argument("--dst-export-root", required=True,
                    help="Destination export root (will write <patient>/<side>/...).")

    # Required: filenames
    ap.add_argument("--uncertain-src-filename", required=True,
                    help="UNCERTAIN source filename under <patient>/<side>/")
    ap.add_argument("--uncertain-dst-filename", required=True,
                    help="UNCERTAIN destination filename under <patient>/<side>/ (e.g., inferred_fact.jsonl)")
    ap.add_argument("--certain-src-filename", required=True,
                    help="CERTAIN source filename under <certain_patient_dir>/<side>/ (e.g., canonical.jsonl)")
    ap.add_argument("--certain-dst-filename", required=True,
                    help="CERTAIN destination filename under <patient>/<side>/ (e.g., canonical.jsonl)")

    # Required: mapping for certain directory naming
    ap.add_argument("--certain-patient-dir-pattern", required=True,
                    help="Pattern for certain patient dir (use {patient} and {side}). Example: {patient}_{side}")

    # Optional: patient list
    ap.add_argument(
        "--patient",
        action="append",
        default=None,
        help="Patient id to process; repeatable. If omitted, auto-discover from uncertain export root.",
    )

    # Optional: behaviors
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing/copying.")
    args = ap.parse_args()

    side = args.side
    uncertain_export_root = Path(args.uncertain_export_root).expanduser().resolve()
    certain_export_root = Path(args.certain_export_root).expanduser().resolve()
    dst_export_root = Path(args.dst_export_root).expanduser().resolve()

    # Patients: explicit list or auto-discover
    if args.patient:
        patients = args.patient
    else:
        patients = discover_patients(uncertain_export_root, side, args.uncertain_src_filename)

    if not patients:
        raise SystemExit("[err] No patients to process (none specified, and auto-discovery found none).")

    print("[info] side:", side)
    print("[info] uncertain_export_root:", uncertain_export_root)
    print("[info] certain_export_root  :", certain_export_root)
    print("[info] dst_export_root      :", dst_export_root)
    print("[info] patients:", len(patients))
    print("[info] overwrite:", bool(args.overwrite), "dry_run:", bool(args.dry_run))
    print()

    ok_uncertain = 0
    ok_certain = 0

    for patient in patients:
        if copy_uncertain_filtered(
            patient=patient,
            side=side,
            uncertain_export_root=uncertain_export_root,
            dst_export_root=dst_export_root,
            src_uncertain_filename=args.uncertain_src_filename,
            dst_uncertain_filename=args.uncertain_dst_filename,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        ):
            ok_uncertain += 1

        if patch_and_copy_certain(
            patient=patient,
            side=side,
            certain_export_root=certain_export_root,
            dst_export_root=dst_export_root,
            certain_patient_dir_pattern=args.certain_patient_dir_pattern,
            src_certain_filename=args.certain_src_filename,
            dst_certain_filename=args.certain_dst_filename,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        ):
            ok_certain += 1

    print(f"\n[done] uncertain_copied={ok_uncertain}/{len(patients)}, certain_patched_written={ok_certain}/{len(patients)}")


if __name__ == "__main__":
    main()
