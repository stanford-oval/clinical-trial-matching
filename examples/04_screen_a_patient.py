#!/usr/bin/env python3
"""End to end: find candidate trials for a patient, then decide each one.

Needs a clause database for retrieval (see docs/DATA.md), and stage-1 pair
artifacts for the decision step:

    export VERDICT_PAIR_DATA=/path/to/experiments/53_v2_full
    python examples/04_screen_a_patient.py sigir-20141 --db /path/to/trial.db

Candidates VERDICT cannot evaluate are reported as *undecided*, never as
ineligible -- for a trial matcher, "we did not look" and "does not qualify"
must never be confused.
"""

import argparse
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patient", nargs="?", default="sigir-20141")
    ap.add_argument("--db", help="clause database")
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args()

    try:
        results = pipeline.screen(a.patient, db=a.db, limit=a.limit)
    except RuntimeError as e:
        raise SystemExit(f"Retrieval failed.\n  {e}")

    if not results:
        raise SystemExit("No candidates. Is --db pointing at a clause database?")

    decided = [r for r in results if r.evaluated]
    print(f"{a.patient}: {len(results)} candidates, {len(decided)} decided\n")

    for r in results:
        verdict_str = r.decision.upper() if r.decision else "undecided"
        print(f"  rank {str(r.rank):>3}  {r.nct_id:<16s} {verdict_str}")
        if r.assumptions:
            print(f"        assumed {len(r.assumptions)} condition(s) the chart "
                  f"did not settle")
        if r.pivotal:
            print(f"        would change the answer: {', '.join(r.pivotal[:3])}")

    undecided = len(results) - len(decided)
    if undecided:
        print(f"\n{undecided} undecided -- no stage-1 artifacts for those pairs.")
        print("They are not ineligible; they were not evaluated. See docs/DATA.md.")


if __name__ == "__main__":
    main()
