#!/usr/bin/env python3
"""Decide one patient--trial pair and read the result.

Needs stage-1 pair data:

    export VERDICT_PAIR_DATA=/path/to/experiments/53_v2_full
    python examples/01_decide_a_pair.py

See docs/DATA.md for where that comes from.
"""

import pathlib
import sys

# Run from a clone without installing: put the repo root on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sys
import verdict


def main() -> None:
    available = verdict.pairs()
    if not available:
        sys.exit("No pair data found. Set $VERDICT_PAIR_DATA -- see docs/DATA.md.")

    pair = sys.argv[1] if len(sys.argv) > 1 else available[0]
    print(f"{len(available)} pairs available; using {pair}\n")

    d = verdict.match(pair)          # raises if the pair cannot be loaded
    print(f"Decision: {d.decision.upper()}")
    print(f"  {d.reasoning}\n")

    print("How it got there:")
    for i, step in enumerate(d.audit_trail, 1):
        print(f"  {i}. {step.stage}"
              + (f" -> {step.decision}" if step.decision else ""))

    print("\nCompare the variants on this pair:")
    for name in verdict.systems():
        other = verdict.match(pair, system=name, strict=False)
        print(f"  {name:<10s} {other.decision}")


if __name__ == "__main__":
    main()
