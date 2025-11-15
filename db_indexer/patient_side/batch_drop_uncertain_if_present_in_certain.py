#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_drop_uncertain_if_present_in_certain.py

For each patient:
  - Load CERTAIN jsonl, collect normalized entity_variable_name set
  - Stream UNCERTAIN jsonl:
      if entity_variable_name appears in certain -> DROP the row
      else keep
  - Write filtered uncertain to a new file (default) or overwrite in place.

Matching:
  - normalized(entity_variable_name) = strip + lowercase
  - match on name only (no timeframe/qualifier stripping)

Default roots are set to your example paths but can be overridden.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, List


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _get_entity_name(obj: Dict[str, Any]) -> Optional[str]:
    n = obj.get("entity_variable_name")
    if isinstance(n, str) and n.strip():
        return n.strip()
    return None


def load_certain_name_set(certain_path: Path) -> Set[str]:
    s: Set[str] = set()
    with certain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            n = _get_entity_name(obj)
            if not n:
                continue
            s.add(_normalize_name(n))
    return s


def default_out_path(in_path: Path, tag: str) -> Path:
    # e.g. canonical.xxx.jsonl -> canonical.xxx.filtered_against_certain.jsonl
    return in_path.with_name(in_path.stem + f".{tag}" + in_path.suffix)


def safe_filter_file(
    patient: str,
    uncertain_path: Path,
    certain_path: Path,
    *,
    out_path: Optional[Path],
    inplace: bool,
    backup: bool,
    out_tag: str,
    dry_run: bool,
) -> Tuple[int, int, int, Optional[Path]]:
    """
    Returns: (total_read, dropped, kept, written_path_or_None)
    """
    if not uncertain_path.is_file():
        print(f"[skip] {patient}: uncertain not found: {uncertain_path}")
        return (0, 0, 0, None)

    if not certain_path.is_file():
        print(f"[skip] {patient}: certain not found: {certain_path}")
        return (0, 0, 0, None)

    if dry_run:
        print(f"[dry-run] {patient}")
        print(f"  uncertain: {uncertain_path}")
        print(f"  certain  : {certain_path}")
        return (0, 0, 0, None)

    certain_names = load_certain_name_set(certain_path)

    # decide output file
    if inplace:
        tmp_out = uncertain_path.with_name(uncertain_path.name + ".tmp_drop")
        final_out = uncertain_path
    else:
        final_out = out_path if out_path is not None else default_out_path(uncertain_path, out_tag)
        tmp_out = final_out

    # backup
    if inplace and backup:
        bak = uncertain_path.with_suffix(uncertain_path.suffix + ".bak")
        bak.parent.mkdir(parents=True, exist_ok=True)
        bak.write_bytes(uncertain_path.read_bytes())
        print(f"[{patient}] ✓ backup saved: {bak}")

    total = dropped = kept = 0
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    with uncertain_path.open("r", encoding="utf-8") as fin, tmp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                # malformed line -> skip
                continue
            if not isinstance(obj, dict):
                continue

            n = _get_entity_name(obj)
            if n and _normalize_name(n) in certain_names:
                dropped += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    if inplace:
        os.replace(tmp_out, final_out)

    print(f"[{patient}] read={total}, dropped={dropped}, kept={kept} -> {final_out}")
    return (total, dropped, kept, final_out)


def discover_patients(uncertain_export_root: Path) -> List[str]:
    if not uncertain_export_root.exists():
        return []
    return sorted(p.name for p in uncertain_export_root.iterdir() if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch drop UNCERTAIN rows whose entity_variable_name appears in CERTAIN (patient_facts_export)."
    )

    # Patients
    ap.add_argument(
        "--patient",
        action="append",
        default=None,
        help="Patient id to process. Repeatable. If omitted, process ALL patients under uncertain export root.",
    )

    # Side and path patterns
    ap.add_argument("--side", choices=["inclusion", "exclusion"], default="inclusion")
    ap.add_argument(
        "--uncertain-build-root",
        default="<SATIR_ROOT>/patient_build_test_res_inclusion",
        help="Uncertain build root (contains patient_facts_export/...).",
    )
    ap.add_argument(
        "--certain-build-root",
        default="<SATIR_ROOT>/patient_build_sigir_inclusion",
        help="Certain build root (contains patient_facts_export/...).",
    )
    ap.add_argument(
        "--uncertain-filename",
        default="canonical.patched.notdealtwith_complete.jsonl",
        help="Filename under <uncertain_export>/<patient>/<side>/",
    )
    ap.add_argument(
        "--certain-filename",
        default="canonical.jsonl",
        help="Filename under <certain_export>/<certain_patient_dir>/<side>/",
    )
    ap.add_argument(
        "--certain-patient-dir-pattern",
        default="{patient}_{side}",
        help="Pattern for certain patient dir name under patient_facts_export. Default: {patient}_{side}",
    )

    # Output controls
    ap.add_argument(
        "--out-tag",
        default="filtered_against_certain",
        help="Output tag appended to uncertain stem when not inplace.",
    )
    ap.add_argument("--inplace", action="store_true", help="Overwrite uncertain file in place (safe replace).")
    ap.add_argument("--backup", action="store_true", help="With --inplace, create .bak before overwrite.")
    ap.add_argument("--dry-run", action="store_true", help="Only print resolved paths; do not write.")
    args = ap.parse_args()

    side = args.side

    uncertain_build_root = Path(args.uncertain_build_root).expanduser().resolve()
    certain_build_root = Path(args.certain_build_root).expanduser().resolve()

    uncertain_export_root = uncertain_build_root / "patient_facts_export"
    certain_export_root = certain_build_root / "patient_facts_export"

    patients = args.patient if args.patient else discover_patients(uncertain_export_root)
    if not patients:
        raise SystemExit(f"[err] no patients found (or specified) under: {uncertain_export_root}")

    print("[info] side:", side)
    print("[info] uncertain_export_root:", uncertain_export_root)
    print("[info] certain_export_root  :", certain_export_root)
    print("[info] patients:", len(patients))

    for patient in patients:
        uncertain_path = uncertain_export_root / patient / side / args.uncertain_filename
        certain_patient_dir = args.certain_patient_dir_pattern.format(patient=patient, side=side)
        certain_path = certain_export_root / certain_patient_dir / side / args.certain_filename

        safe_filter_file(
            patient=patient,
            uncertain_path=uncertain_path,
            certain_path=certain_path,
            out_path=None,  # default computed next to uncertain
            inplace=bool(args.inplace),
            backup=bool(args.backup),
            out_tag=str(args.out_tag),
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()
