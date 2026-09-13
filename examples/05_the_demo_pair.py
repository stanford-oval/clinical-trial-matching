"""Where two systems disagree, and why.

Runs on the bundled synthetic pair, so it needs no corpus, no LLM endpoint
and no compiled trial program -- only z3.

    python examples/05_the_demo_pair.py
"""
import os
import pathlib
import sys

# Run from a clone without installing: put the repo root on the path.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VERDICT_PAIR_DATA", str(ROOT / "data" / "demo" / "pairs"))

import verdict                                        # noqa: E402
from verdict.artifacts import artifacts_for, conditions_from_pair  # noqa: E402

PAIR = "demo-patient-01__NCT00000000"
LABELS = {
    "egfr_ml_min": "renal function (eGFR)",
    "patient_age_in_years": "age",
    "ecog_performance_status": "performance status",
    "histologically_confirmed_nsclc": "confirmed NSCLC",
}


def rule(title):
    print("\n" + title)
    print("-" * len(title))


rule("1. The note")
import json
raw = json.loads((ROOT / "data/demo/pairs/cmsrc_out/demo-patient-01"
                  / "NCT00000000__full.json").read_text())
print(raw["patient_note"])
print("\ninclusion criteria:", raw["inclusion_criteria"])

rule("2. Two systems, two answers")
for system in ("smt-only", "lm-only"):
    d = verdict.match(PAIR, system=system)
    print(f"  {system:10s} {str(d.decision).upper():12s} {d.reasoning[:78]}")

rule("3. What the chart settled, and what it did not")
phi, conds = conditions_from_pair(PAIR, "inclusion")
for c in conds:
    label = LABELS.get(c.name, c.name)
    shown = "--" if c.value is None else c.value
    print(f"  {label:26s} {c.status:11s} {shown}")

rule("4. The disagreement, located")
a = artifacts_for(PAIR, side="inclusion")
print(a.render_assumptions(phi, labels=LABELS))
print("\nThe two answers differ over exactly one condition. The solver will not")
print("supply a value the chart never gave; the language model will. The tool's")
print("job is to say which condition that was, and what it would have to be.")
