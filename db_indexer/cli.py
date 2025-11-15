"""CLI entry point for database indexing (trial-side and patient-side)."""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(
        description="Database indexer: index trial or patient canonical outputs into SQLite."
    )
    sub = ap.add_subparsers(dest="side", required=True)

    trial_p = sub.add_parser("trial", help="Run trial-side indexing pipeline")
    trial_p.add_argument("extra_args", nargs=argparse.REMAINDER, help="Args forwarded to runall_trialside.py")

    patient_p = sub.add_parser("patient", help="Run patient-side indexing pipeline")
    patient_p.add_argument("extra_args", nargs=argparse.REMAINDER, help="Args forwarded to patient pipeline")

    args = ap.parse_args()

    if args.side == "trial":
        from db_indexer.trial_side.runall_trialside import main as trial_main
        sys.argv = ["index-db-trial"] + args.extra_args
        trial_main()
    elif args.side == "patient":
        from db_indexer.patient_side.run_multi_pass_pipeline import main as patient_main
        sys.argv = ["index-db-patient"] + args.extra_args
        patient_main()


if __name__ == "__main__":
    main()
