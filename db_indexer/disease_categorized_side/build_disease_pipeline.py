#!/usr/bin/env python3
"""
build_disease_pipeline.py
End-to-end driver:
  1) Ingest *_disease_link_filter_summary.json → disease_constraint_atoms
  1b) Ingest *_disease_link_filter_summary.json → disease_constraint_atoms_prevent_other
  2) Rebuild maps + (optional) keep-list decider + lifting → accepted alternatives

Subcohort-first ingestion (NEW DEFAULT):
  - Ingestion is per-file trial_id (e.g., NCT...a/b/c) and defaults to deriving trial_id
    from filename tokens.
  - Legacy parent→effective fan-out is available via --cohort-fanout.

Fresh tables by default:
  - trial_disease_to_sqlite.py defaults to --recreate (drop+recreate tables) so each run
    produces fresh ingestion tables unless explicitly disabled.

Alignment:
  - Step 2 runs TWO lifting branches via disease_ontology_lifter.py:
      * main branch:
          --source-table disease_constraint_atoms
          --ns main
      * prevention branch:
          --source-table disease_constraint_atoms_prevent_other
          --ns prevention

NOTE (UPDATED FOR MULTI-LABEL CATEGORIES):
  - DiseaseCategorizer may now output concept["category"] as List[str].
  - trial_disease_to_sqlite.py ingests a row if ANY category matches the allowlist.
  - trial_disease_to_sqlite.py canonicalizes human labels (e.g., "Clinically address"/"Other")
    to stable internal categories ("treat"/"other"/"prevent"/"not relevant").
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
        description="Disease list → SQLite → lifting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared
    ap.add_argument("--db", default="../../build/trial.db", help="SQLite DB path")

    # Step 1: ingest
    ap.add_argument("--input-dir", default="../../build/disease_filtered_categorized",
                    help="Directory with *_disease_link_filter_summary.json (supports nested subdirs)")
    ap.add_argument("--preproc-dir", default="mbench/req_mbench/preproc_logs",
                    help="Preproc dir with {trial_id}_{side}.pre.normalized.json (used only with --cohort-fanout)")
    ap.add_argument("--prefer-side", choices=["inclusion", "exclusion"], default="inclusion")

    # NEW: legacy fan-out toggle
    ap.add_argument("--cohort-fanout", action="store_true", default=False,
                    help="(legacy) expand parent trial_id to effective_trial_ids via preproc logs during ingestion")

    # Step 2: lifter/decider
    ap.add_argument("--trial", default=None,
                    help="Only process a single trial_id in lifting/acceptance")
    ap.add_argument("--no-decider", action="store_true",
                    help="Skip keep-list decider (accept only self)")
    ap.add_argument("--decider-batch-size", type=int, default=10)
    ap.add_argument("--decider-workers", type=int, default=24)
    ap.add_argument("--decider-rate-per-min", type=int, default=60)
    ap.add_argument("--decider-save-dir", type=str, default=None,
                    help="If set, base directory for decider artifacts; branch suffixes __main/__prevention are appended")
    ap.add_argument("--decider-overwrite", choices=["keep", "purge", "fail"], default="purge")
    ap.add_argument("--decider-prompt", type=str, default="./prompts/decider.prompt")
    ap.add_argument("--lineage-workers", type=int, default=12)

    ap.add_argument("--max-lift-per-item", type=int, default=200,
                    help="Max ancestors per item inserted into disease_constraint_lifted_atoms")
    ap.add_argument("--keep-existing-lifts", action="store_true",
                    help="Do not clear disease_constraint_lifted_atoms / disease_constraint_alternatives before re-running")
    ap.add_argument("--no-lift", action="store_true",
                    help="Only rebuild maps/decider; skip lifting/acceptance")

    # Utility
    ap.add_argument("--echo-only", action="store_true",
                    help="Print planned commands without executing")
    ap.add_argument("--scripts-dir", default=str(HERE),
                    help="Directory where the component scripts live")

    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    ingest_py = scripts_dir / "trial_disease_to_sqlite.py"
    lifter_py = scripts_dir / "disease_ontology_lifter.py"

    ensure_exists(ingest_py, "file")
    ensure_exists(lifter_py, "file")

    # Ensure DB parent exists
    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure input dir exists (ingest requires it)
    input_dir = Path(args.input_dir).expanduser().resolve()
    ensure_exists(input_dir, "dir")

    # Preproc dir can be missing; only needed for --cohort-fanout
    preproc_dir = Path(args.preproc_dir).expanduser().resolve()
    if args.cohort_fanout and (not preproc_dir.is_dir()):
        print(f"[i] Warning: preproc dir not found: {preproc_dir} (fan-out may be skipped)")

    # Base ingest flags (subcohort-first by default)
    ingest_mode_flags: List[str] = []
    if args.cohort_fanout:
        ingest_mode_flags += ["--cohort-fanout"]

    # ───────────────────────────────────────────
    # Step 1a) Ingest into disease_constraint_atoms (treat/other/prevent by default)
    # ───────────────────────────────────────────
    ingest_cmd_main = [
        which_python(), str(ingest_py),
        "--input-dir", str(input_dir),
        "--db", str(db_path),
        "--preproc-dir", str(preproc_dir),
        "--prefer-side", args.prefer_side,
        "--table", "disease_constraint_atoms",
        # NOTE: these are CANONICALIZED categories in trial_disease_to_sqlite.py
        "--categories", "treat",
        # fresh tables default is handled by trial_disease_to_sqlite.py via --recreate default ON
    ] + ingest_mode_flags

    # ───────────────────────────────────────────
    # Step 1b) Ingest prevent into disease_constraint_atoms_prevent_other
    # ───────────────────────────────────────────
    ingest_cmd_prev = [
        which_python(), str(ingest_py),
        "--input-dir", str(input_dir),
        "--db", str(db_path),
        "--preproc-dir", str(preproc_dir),
        "--prefer-side", args.prefer_side,
        "--table", "disease_constraint_atoms_prevent_other",
        "--categories", "prevent",
    ] + ingest_mode_flags

    # ───────────────────────────────────────────
    # Step 2) Run TWO branches via disease_ontology_lifter.py:
    #   - main:       source-table=disease_constraint_atoms,               ns=main
    #   - prevention: source-table=disease_constraint_atoms_prevent_other, ns=prevention
    # ───────────────────────────────────────────
    base_lifter_flags = [
        "--decider" if not args.no_decider else "--no-decider",
        "--decider-batch-size", str(args.decider_batch_size),
        "--decider-workers", str(args.decider_workers),
        "--decider-rate-per-min", str(args.decider_rate_per_min),
        "--lineage-workers", str(args.lineage_workers),
        "--max-lift-per-item", str(args.max_lift_per_item),
    ]
    if args.decider_overwrite:
        base_lifter_flags += ["--decider-overwrite", args.decider_overwrite]
    if args.decider_prompt:
        base_lifter_flags += ["--decider-prompt", args.decider_prompt]
    if args.keep_existing_lifts:
        base_lifter_flags += ["--no-overwrite"]
    if args.no_lift:
        base_lifter_flags += ["--no-lift"]
    if args.trial:
        base_lifter_flags += ["--trial", args.trial]

    main_decider_dir = None
    prev_decider_dir = None
    if args.decider_save_dir:
        base = Path(args.decider_save_dir).expanduser()
        main_decider_dir = str(base.with_name(base.name + "__main"))
        prev_decider_dir = str(base.with_name(base.name + "__prevention"))

    lifter_cmd_main = [
        which_python(), str(lifter_py),
        "--db", str(db_path),
        "--source-table", "disease_constraint_atoms",
        "--ns", "main",
    ] + base_lifter_flags
    if main_decider_dir:
        lifter_cmd_main += ["--decider-save-dir", main_decider_dir]

    lifter_cmd_prev = [
        which_python(), str(lifter_py),
        "--db", str(db_path),
        "--source-table", "disease_constraint_atoms_prevent_other",
        "--ns", "prevention",
    ] + base_lifter_flags
    if prev_decider_dir:
        lifter_cmd_prev += ["--decider-save-dir", prev_decider_dir]

    print("\n=== Disease Pipeline ===")
    print(f"DB:           {db_path}")
    print(f"Input dir:    {input_dir}")
    print(f"Preproc dir:  {preproc_dir}")
    print(f"Scripts dir:  {scripts_dir}")
    print(f"Ingest mode:  {'LEGACY fan-out' if args.cohort_fanout else 'SUBCOHORT per-file (default)'}\n")
    print("Tables (ingest):")
    print("  - disease_constraint_atoms                 (categories: treat, other, prevent)")
    print("  - disease_constraint_atoms_prevent_other   (categories: prevent)\n")
    print("Lifter branches:")
    print("  - main       : source_table=disease_constraint_atoms,               ns=main")
    print("  - prevention : source_table=disease_constraint_atoms_prevent_other, ns=prevention\n")

    if args.echo_only:
        print("➤ (echo) " + " ".join(ingest_cmd_main))
        print("➤ (echo) " + " ".join(ingest_cmd_prev))
        print("➤ (echo) " + " ".join(lifter_cmd_main))
        print("➤ (echo) " + " ".join(lifter_cmd_prev))
        return 0

    run(ingest_cmd_main)
    run(ingest_cmd_prev)
    run(lifter_cmd_main)
    run(lifter_cmd_prev)

    print("\n✅ Pipeline complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())