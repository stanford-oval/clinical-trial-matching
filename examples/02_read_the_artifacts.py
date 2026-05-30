#!/usr/bin/env python3
"""What the system assumed, and what would change its mind.

Needs only z3-solver. No corpus, no API keys:

    pip install 'z3-solver>=4.12'
    python examples/02_read_the_artifacts.py

The trial wants a working eGFR and creatinine clearance of at least 60.
Our patient has a failing eGFR, and the chart says nothing about clearance.
"""

import pathlib
import sys

# Run from a clone without installing: put the repo root on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from smt_core.maxsmt import Condition, OBSERVED, UNRESOLVED, solve
TRIAL = [
    "(declare-const |egfr| Bool)",
    "(declare-const |crcl| Real)",
    "(assert |egfr|)",              # eGFR must be adequate
    "(assert (>= |crcl| 60))",      # clearance at least 60
]

PATIENT = [
    Condition("egfr", False, OBSERVED),    # chart: eGFR is failing
    Condition("crcl", None, UNRESOLVED),   # chart: silent on clearance
]


def main() -> None:
    try:
        a = solve(TRIAL, PATIENT)
    except ImportError as e:      # no solver, or one whose library will not load
        raise SystemExit(f"This example needs a working solver.\n  {e}")

    print(f"Decision: {a.decision.upper()}")
    print(f"  because: {a.trace}\n")

    print("What it had to assume (the chart never said):")
    view = a.for_verbalizer(TRIAL)["assumptions"]
    for name, info in view.items():
        # Report the requirement, never the solver's placeholder number --
        # saying "clearance 72" would invent a lab result.
        req = info.get("requirement")
        print(f"  {name}: assumed to meet {req}" if req
              else f"  {name}: assumed {info.get('value')}")
    if not view:
        print("  (nothing -- the chart answered every criterion)")

    print("\nWhat would change the answer:")
    for c in a.pivotal:
        print(f"  {c}")

    print("\nSame thing, labelled with the paper's symbols:")
    print(a.describe())


if __name__ == "__main__":
    main()
