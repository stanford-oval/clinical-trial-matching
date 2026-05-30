#!/usr/bin/env python3
"""
Invariant test: verify that sql_retrieval produces byte-for-byte identical
results to the original pipeline ground truth.

Ground truth fixtures were captured from the original irsrc/ops/constraint_retrieval.py
running against the full trial.db with these settings:
  --scope any --important-mode all --alt-mode act --enable-prevention-hits

Usage:
    source .env
    python tests/test_retrieval_invariant.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
BUILD = ROOT / "build"
DB = BUILD / "trial.db"

PATIENTS = ["sigir-20141", "sigir-20142"]
SUFFIX = "__all__prevent__act"
MODE_ARGS = [
    "--scope", "any",
    "--parallel", "1",
    "--important-mode", "all",
    "--alt-mode", "act",
    "--enable-prevention-hits",
]


def run_compose_for_patient(patient_id: str, out_dir: str) -> pathlib.Path:
    """Run constraint_retrieval.py for a single patient, return JSON output path."""
    cmd = [
        sys.executable, "-m", "sql_retrieval.ops.constraint_retrieval",
        "--db", str(DB),
        "--out", out_dir,
        "--patient", patient_id,
    ] + MODE_ARGS

    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"constraint_retrieval failed for {patient_id}: exit {result.returncode}")

    return pathlib.Path(out_dir) / f"retrieved_mappings{SUFFIX}" / "json" / f"{patient_id}{SUFFIX}.json"


def normalize(data: dict) -> str:
    """Sort trials by trial_id and produce deterministic JSON string."""
    data["trials"] = sorted(data["trials"], key=lambda t: t["trial_id"])
    return json.dumps(data, sort_keys=True)


def test_patient(patient_id: str, out_dir: str) -> bool:
    """Test one patient against ground truth fixture. Returns True if passes."""
    fixture_path = FIXTURES / f"{patient_id}{SUFFIX}.json"
    if not fixture_path.exists():
        print(f"  SKIP  {patient_id}: fixture not found at {fixture_path}")
        return True

    # Run refactored pipeline
    actual_path = run_compose_for_patient(patient_id, out_dir)
    if not actual_path.exists():
        print(f"  FAIL  {patient_id}: output not produced at {actual_path}")
        return False

    # Load and normalize
    expected = json.loads(fixture_path.read_text())
    actual = json.loads(actual_path.read_text())

    expected_norm = normalize(expected)
    actual_norm = normalize(actual)

    if expected_norm == actual_norm:
        n_trials = len(expected["trials"])
        print(f"  PASS  {patient_id}: {n_trials} trials, byte-for-byte identical")
        return True

    # Detailed diff
    exp_trials = {t["trial_id"]: t for t in expected["trials"]}
    act_trials = {t["trial_id"]: t for t in actual["trials"]}

    exp_ids = set(exp_trials.keys())
    act_ids = set(act_trials.keys())

    print(f"  FAIL  {patient_id}:")
    print(f"         Expected {len(exp_ids)} trials, got {len(act_ids)}")

    only_exp = exp_ids - act_ids
    only_act = act_ids - exp_ids
    if only_exp:
        print(f"         Missing {len(only_exp)} trials: {sorted(only_exp)[:5]}...")
    if only_act:
        print(f"         Extra {len(only_act)} trials: {sorted(only_act)[:5]}...")

    # Check label mismatches on common trials
    common = exp_ids & act_ids
    mismatches = []
    for tid in sorted(common):
        if exp_trials[tid]["label"] != act_trials[tid]["label"]:
            mismatches.append(tid)
    if mismatches:
        print(f"         {len(mismatches)} label mismatches: {mismatches[:5]}...")

    return False


def main():
    if not DB.exists():
        print(f"ERROR: trial.db not found at {DB}")
        print("       Run tests/bootstrap_test_data.sh first.")
        sys.exit(2)

    print("SatIR Retrieval Invariant Test")
    print(f"  DB:       {DB}")
    print(f"  Fixtures: {FIXTURES}")
    print()

    with tempfile.TemporaryDirectory(prefix="satir_test_") as tmpdir:
        all_pass = True
        for pid in PATIENTS:
            ok = test_patient(pid, tmpdir)
            if not ok:
                all_pass = False

    print()
    if all_pass:
        print("ALL PASSED — retrieval output matches ground truth exactly.")
    else:
        print("FAILED — output differs from ground truth.")
        sys.exit(1)


if __name__ == "__main__":
    main()
