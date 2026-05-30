#!/usr/bin/env python3
"""
Exhaustive equivalence test: runs the refactored constraint retrieval
against the original pipeline for ALL patients, ALL three mode combinations
from run_selected_compose_and_why.sh, and verifies byte-for-byte identity.

Mode combinations tested:
  1. all, prevent, nonact
  2. all, prevent, act
  3. ccr, prevent, act

Usage:
    source .env
    python tests/test_exhaustive_equivalence.py
    python tests/test_exhaustive_equivalence.py --modes act  # just act mode
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
OLD_DIR = Path(os.environ.get("SATIR_ORIGINAL_OPS", str(ROOT.parent / "TrialGPT-SMT" / "irsrc" / "ops")))
OLD_DB = Path(os.environ.get("SATIR_ORIGINAL_DB", str(ROOT.parent / "TrialGPT-SMT" / "build" / "trial.db")))
NEW_DB = ROOT / "build" / "trial.db"

MODE_COMBOS = [
    # (important_mode, enable_prevention, alt_mode)
    ("all", True, "nonact"),
    ("all", True, "act"),
    ("ccr", True, "act"),
]


def get_all_patients(db_path: Path) -> List[str]:
    conn = sqlite3.connect(str(db_path))
    # Use old table name for original DB, new for refactored
    try:
        patients = [r[0] for r in conn.execute(
            "SELECT DISTINCT patient_id FROM facts_inclusion "
            "UNION SELECT DISTINCT patient_id FROM facts_exclusion "
            "UNION SELECT DISTINCT patient_id FROM patient_demographics "
            "ORDER BY 1"
        ).fetchall()]
    except Exception:
        patients = [r[0] for r in conn.execute(
            "SELECT DISTINCT patient_id FROM patient_inclusion_constraints "
            "UNION SELECT DISTINCT patient_id FROM patient_exclusion_constraints "
            "UNION SELECT DISTINCT patient_id FROM patient_demographic_constraints "
            "ORDER BY 1"
        ).fetchall()]
    conn.close()
    return patients


def build_args(mode: str, prevent: bool, alt: str) -> List[str]:
    args = [
        "--scope", "any",
        "--parallel", "1",
        "--important-mode", mode,
        "--alt-mode", alt,
    ]
    if prevent:
        args.append("--enable-prevention-hits")
    return args


def suffix(mode: str, prevent: bool, alt: str) -> str:
    pt = "prevent" if prevent else "noprevent"
    return f"__{mode}__{pt}__{alt}"


def run_original(patient_id: str, mode: str, prevent: bool, alt: str, out_dir: str) -> Path:
    args = build_args(mode, prevent, alt)
    cmd = [
        sys.executable, "compose_trial_eval.py",
        "--db", str(OLD_DB), "--out", out_dir,
        "--patient", patient_id,
    ] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(OLD_DIR))
    if r.returncode != 0:
        raise RuntimeError(f"Original failed: {r.stderr[-500:]}")
    s = suffix(mode, prevent, alt)
    return Path(out_dir) / f"retrieved_mappings{s}" / "json" / f"{patient_id}{s}.json"


def run_refactored(patient_id: str, mode: str, prevent: bool, alt: str, out_dir: str) -> Path:
    args = build_args(mode, prevent, alt)
    cmd = [
        sys.executable, "-m", "sql_retrieval.ops.constraint_retrieval",
        "--db", str(NEW_DB), "--out", out_dir,
        "--patient", patient_id,
    ] + args
    r = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    if r.returncode != 0:
        raise RuntimeError(f"Refactored failed: {r.stderr[-500:]}")
    s = suffix(mode, prevent, alt)
    return Path(out_dir) / f"retrieved_mappings{s}" / "json" / f"{patient_id}{s}.json"


def normalize(data: dict) -> str:
    data["trials"] = sorted(data["trials"], key=lambda t: t["trial_id"])
    return json.dumps(data, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=None,
                    help="Filter to specific alt_modes (e.g., 'act' 'nonact')")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="Specific patient IDs (default: all)")
    ap.add_argument("--max-patients", type=int, default=None)
    args = ap.parse_args()

    combos = MODE_COMBOS
    if args.modes:
        combos = [(m, p, a) for m, p, a in combos if a in args.modes]

    patients = args.patients or get_all_patients(OLD_DB)
    if args.max_patients:
        patients = patients[:args.max_patients]

    print(f"Exhaustive Equivalence Test")
    print(f"  Patients:    {len(patients)}")
    print(f"  Combos:      {len(combos)}")
    print(f"  Total tests: {len(patients) * len(combos)}")
    print()

    passed = 0
    failed = 0
    failures: List[Tuple[str, str, str, str, str]] = []

    for mode, prevent, alt in combos:
        tag = suffix(mode, prevent, alt)
        print(f"--- Mode: {tag} ---")

        for i, pid in enumerate(patients):
            try:
                with tempfile.TemporaryDirectory() as td_old, \
                     tempfile.TemporaryDirectory() as td_new:

                    old_path = run_original(pid, mode, prevent, alt, td_old)
                    new_path = run_refactored(pid, mode, prevent, alt, td_new)

                    if not old_path.exists():
                        print(f"  [{i+1}/{len(patients)}] {pid}: OLD missing output")
                        failed += 1
                        failures.append((pid, mode, alt, "old_missing", ""))
                        continue

                    if not new_path.exists():
                        print(f"  [{i+1}/{len(patients)}] {pid}: NEW missing output")
                        failed += 1
                        failures.append((pid, mode, alt, "new_missing", ""))
                        continue

                    old_data = json.loads(old_path.read_text())
                    new_data = json.loads(new_path.read_text())

                    old_norm = normalize(old_data)
                    new_norm = normalize(new_data)

                    if old_norm == new_norm:
                        n = len(old_data["trials"])
                        print(f"  [{i+1}/{len(patients)}] {pid}: IDENTICAL ({n} trials)")
                        passed += 1
                    else:
                        old_ids = {t["trial_id"] for t in old_data["trials"]}
                        new_ids = {t["trial_id"] for t in new_data["trials"]}
                        diff = len(old_ids ^ new_ids)
                        detail = f"old={len(old_ids)} new={len(new_ids)} diff={diff}"
                        print(f"  [{i+1}/{len(patients)}] {pid}: MISMATCH ({detail})")
                        failed += 1
                        failures.append((pid, mode, alt, "mismatch", detail))

            except Exception as e:
                print(f"  [{i+1}/{len(patients)}] {pid}: ERROR ({e})")
                failed += 1
                failures.append((pid, mode, alt, "error", str(e)[:200]))

    print()
    print(f"{'='*60}")
    print(f"  PASSED: {passed}/{passed+failed}")
    print(f"  FAILED: {failed}/{passed+failed}")
    if failures:
        print(f"  Failures:")
        for pid, mode, alt, kind, detail in failures:
            print(f"    {pid} [{mode}/{alt}]: {kind} {detail}")
    print(f"{'='*60}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
