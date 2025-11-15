#!/usr/bin/env python3
"""
build_disease_pipeline.py
End-to-end driver (always fresh overwrite of disease tables):
  1) Ingest *_disease_link_filter_summary.json → disease_constraint_atoms
     (default: DELETE all rows then upsert)
  2) Rebuild maps + (optional) keep-list decider + lifting → accepted alternatives
     (default: DELETE disease_predicate_concepts / disease_constraint_lifted_atoms / disease_constraint_alternatives
               and disease_concept_constraint_alternatives, then rebuild)

Defaults:
  --input-dir   ../../build/disease
  --preproc-dir mbench/req_mbench/preproc_logs
  --db          ../../build/trial.db
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).parent.resolve()

def which_python() -> str:
    return sys.executable or "python3"

def ensure_exists(p: Path, kind: str = "file", must_exist: bool = True) -> None:
    if must_exist:
        if kind == "file" and not p.is_file():
            print(f"[!] Missing {kind}: {p}", file=sys.stderr)
            sys.exit(2)
        if kind == "dir" and not p.is_dir():
            print(f"[!] Missing {kind}: {p}", file=sys.stderr)
            sys.exit(2)

def run(cmd: List[str]) -> None:
    print("➤", " ".join(cmd), flush=True)
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        print(f"[!] Command failed with exit code {cp.returncode}", file=sys.stderr)
        sys.exit(cp.returncode)

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Disease list → SQLite → lifting pipeline (fresh overwrite each run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared
    ap.add_argument("--db", default="../../build/trial.db", help="SQLite DB path")

    # Step 1: ingest
    ap.add_argument("--input-dir", default="../../build/disease",
                    help="Directory with *_disease_link_filter_summary.json")
    ap.add_argument("--preproc-dir", default="mbench/req_mbench/preproc_logs",
                    help="Preproc dir with {trial_id}_{side}.pre.normalized.json (used only if input JSONs are parent-level)")
    ap.add_argument("--prefer-side", choices=["inclusion", "exclusion"], default="inclusion")

    # Step 2: lifter/decider
    ap.add_argument("--trial", default=None,
                    help="Only process a single trial_id in lifting/acceptance")
    ap.add_argument("--no-decider", action="store_true",
                    help="Skip keep-list decider (accept only self; still overwrites disease tables)")
    ap.add_argument("--decider-batch-size", type=int, default=10)
    ap.add_argument("--decider-workers", type=int, default=24)
    ap.add_argument("--decider-rate-per-min", type=int, default=60)
    ap.add_argument("--decider-save-dir", type=str, default=None)
    ap.add_argument("--decider-overwrite", choices=["keep", "purge", "fail"], default="purge")
    ap.add_argument("--decider-prompt", type=str, default=None)
    ap.add_argument("--lineage-workers", type=int, default=12)

    ap.add_argument("--max-lift-per-item", type=int, default=200,
                    help="Max ancestors per item inserted into disease_constraint_lifted_atoms")
    ap.add_argument("--no-lift", action="store_true",
                    help="Only rebuild maps/decider; skip lifting/acceptance")

    # Utility
    ap.add_argument("--echo-only", action="store_true",
                    help="Print planned commands without executing")
    ap.add_argument("--scripts-dir", default=str(HERE),
                    help="Directory where the two component scripts live")

    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    ingest_py = scripts_dir / "trial_disease_to_sqlite.py"
    lifter_py = scripts_dir / "disease_ontology_lifter.py"

    ensure_exists(ingest_py, "file")
    ensure_exists(lifter_py, "file")

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir).expanduser().resolve()
    ensure_exists(input_dir, "dir")

    preproc_dir = Path(args.preproc_dir).expanduser().resolve()
    if not preproc_dir.is_dir():
        print(f"[i] Warning: preproc dir not found: {preproc_dir} (fan-out for parent-level disease JSONs will be skipped)")

    # Step 1) Ingest disease list items (ALWAYS overwrite disease_constraint_atoms)
    ingest_cmd = [
        which_python(), str(ingest_py),
        "--input-dir", str(input_dir),
        "--db", str(db_path),
        "--preproc-dir", str(preproc_dir),
        "--prefer-side", args.prefer_side,
        "--overwrite", "all",   # ✅ enforce fresh overwrite
    ]

    # Step 2) Rebuild maps + optional decider + lifting + acceptance
    lifter_cmd = [
        which_python(), str(lifter_py),
        "--db", str(db_path),

        # Defaults in lifter already overwrite; we keep defaults, but explicit is ok:
        "--overwrite-map",
        "--overwrite-lifts",

        "--decider" if not args.no_decider else "--no-decider",
        "--decider-batch-size", str(args.decider_batch_size),
        "--decider-workers", str(args.decider_workers),
        "--decider-rate-per-min", str(args.decider_rate_per_min),
        "--decider-overwrite", str(args.decider_overwrite),
        "--lineage-workers", str(args.lineage_workers),
        "--max-lift-per-item", str(args.max_lift_per_item),
    ]
    if args.decider_save_dir:
        lifter_cmd += ["--decider-save-dir", args.decider_save_dir]
    if args.decider_prompt:
        lifter_cmd += ["--decider-prompt", args.decider_prompt]
    if args.no_lift:
        lifter_cmd += ["--no-lift"]
    if args.trial:
        lifter_cmd += ["--trial", args.trial]

    print("\n=== Disease Pipeline (fresh overwrite) ===")
    print(f"DB:           {db_path}")
    print(f"Input dir:    {input_dir}")
    print(f"Preproc dir:  {preproc_dir}")
    print(f"Scripts dir:  {scripts_dir}\n")

    if args.echo_only:
        print("➤ (echo) " + " ".join(ingest_cmd))
        print("➤ (echo) " + " ".join(lifter_cmd))
        return 0

    run(ingest_cmd)
    run(lifter_cmd)

    print("\n✅ Pipeline complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
