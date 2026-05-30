#!/usr/bin/env python3
"""
Benchmark SatIR constraint retrieval across all patients.

Runs the full retrieval pipeline (satisfy → contradict → gap) in-process
for all patients in the DB and reports per-patient and aggregate timing.

Usage:
    source .env
    python tests/benchmark_retrieval.py
    python tests/benchmark_retrieval.py --warmup 2 --runs 3
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "build" / "trial.db"


def get_all_patients(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute("""
        SELECT DISTINCT patient_id FROM patient_inclusion_constraints
        UNION SELECT DISTINCT patient_id FROM patient_exclusion_constraints
        UNION SELECT DISTINCT patient_id FROM patient_demographic_constraints
        ORDER BY 1
    """).fetchall()]


def run_pipeline(tep, conn, pid, important_table, daa_main):
    """Run full retrieval pipeline for one patient. Returns (time_s, n_trials)."""
    t0 = time.perf_counter()
    d = tep.satisfy_disease_constraints(
        conn, pid, daa_table=daa_main,
        require_root_for_hops=True, important_table=important_table)
    p = tep.satisfy_positive_literal_constraints(
        conn, pid, require_root_for_hops=True,
        important_table=important_table, alt_mode="act")
    pv = tep.satisfy_prevention_constraints(
        conn, pid, require_root_for_hops=True, alt_mode="act")
    u = sorted(set(d) | set(p) | set(pv))
    s = tep.check_constraint_contradictions(conn, pid, u, scope="any")
    g = tep.evaluate_constraint_satisfaction_gap(
        conn, pid, scope="any", candidate_trial_ids=u,
        important_table=important_table)
    elapsed = time.perf_counter() - t0
    return elapsed, len(u)


def main():
    ap = argparse.ArgumentParser(description="Benchmark SatIR retrieval")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs (discarded)")
    ap.add_argument("--runs", type=int, default=1, help="Timed runs to average")
    args = ap.parse_args()

    from sql_retrieval.ops import constraint_primitives as tep

    conn = sqlite3.connect(str(args.db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-64000")

    patients = get_all_patients(conn)
    n_trials = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    important_table = "patient_inclusion_constraints_important_all"
    daa_main = "disease_constraint_alternatives"

    print(f"SatIR Retrieval Benchmark")
    print(f"  DB:       {args.db}")
    print(f"  Patients: {len(patients)}")
    print(f"  Trials:   {n_trials}")
    print(f"  Warmup:   {args.warmup}  Runs: {args.runs}")
    print()

    # Warmup
    for _ in range(args.warmup):
        for pid in patients[:3]:
            run_pipeline(tep, conn, pid, important_table, daa_main)

    # Timed runs
    all_times: List[List[Tuple[str, float, int]]] = []
    for run_i in range(args.runs):
        run_data = []
        t_all = time.perf_counter()
        for pid in patients:
            elapsed, n = run_pipeline(tep, conn, pid, important_table, daa_main)
            run_data.append((pid, elapsed, n))
        wall = time.perf_counter() - t_all
        all_times.append(run_data)
        print(f"  Run {run_i+1}: {wall:.2f}s total, {wall/len(patients)*1000:.1f}ms avg")

    # Aggregate across runs
    best_run = min(all_times, key=lambda rd: sum(t for _, t, _ in rd))
    times = [t for _, t, _ in best_run]
    trials_per = [n for _, _, n in best_run]
    total_wall = sum(times)

    print()
    print(f"{'='*55}")
    print(f"  Best run ({len(patients)} patients × {n_trials} trials):")
    print(f"  Total:    {total_wall:.2f}s")
    print(f"  Average:  {statistics.mean(times)*1000:.1f}ms per patient")
    print(f"  Median:   {statistics.median(times)*1000:.1f}ms per patient")
    print(f"  P95:      {sorted(times)[int(len(times)*0.95)]*1000:.1f}ms")
    print(f"  Min:      {min(times)*1000:.1f}ms")
    print(f"  Max:      {max(times)*1000:.1f}ms")
    print(f"  StdDev:   {statistics.stdev(times)*1000:.1f}ms")
    print(f"  Trials retrieved: {sum(trials_per)} total, {statistics.mean(trials_per):.0f} avg/patient")
    print(f"{'='*55}")

    # Per-patient detail
    print()
    print(f"{'Patient':<20s} {'Time (ms)':>10s} {'Trials':>8s}")
    print("-" * 42)
    for pid, t, n in sorted(best_run, key=lambda x: -x[1]):
        print(f"{pid:<20s} {t*1000:10.1f} {n:8d}")

    conn.close()


if __name__ == "__main__":
    main()
