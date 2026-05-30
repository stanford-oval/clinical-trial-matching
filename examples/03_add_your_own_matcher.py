#!/usr/bin/env python3
"""Compare your own matcher against the ones from the paper.

Needs nothing but the package:

    python examples/03_add_your_own_matcher.py

A matcher is any callable taking a pair id and returning a Decision. Register
it and it is available everywhere the built-ins are -- the API, and the
`verdict` command's --system flag.
"""

import pathlib
import sys

# Run from a clone without installing: put the repo root on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import verdict
from matchers.schema import AuditStep, Decision


@verdict.register("always-refer", description="refer everyone to screening")
def always_refer(pair_id: str) -> Decision:
    """A deliberately naive baseline: never rule anyone out.

    Useful as a recall ceiling -- it shows what you give up by deciding at all.
    """
    return Decision(
        pair_id, "always-refer", "eligible",
        "Refers every patient; no criterion was checked.",
        [AuditStep(stage="policy", decision="eligible",
                   rationale="always refer", evidence={})],
    )


def main() -> None:
    print("Registered matchers:\n")
    for name, desc in verdict.systems().items():
        mine = "  <- yours" if name == "always-refer" else ""
        print(f"  {name:<14s} {desc}{mine}")

    d = verdict.match("demo__NCT00000000", system="always-refer")
    print(f"\nYour matcher on a made-up pair: {d.decision.upper()}")
    print(f"  {d.reasoning}")
    print("\nIt is also reachable from the command line:")
    print("  verdict match <pair> --system always-refer")


if __name__ == "__main__":
    main()
