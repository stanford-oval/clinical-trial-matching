#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(".")  # or Path("path/to/out")
ROOT = OUT_DIR / "retrieved_mappings" / "merged_json"

LABELS = ["all_satisfied", "unsatisfied_inclusion", "explicit_contradiction"]

total_counts = defaultdict(int)
num_patients = 0

for path in sorted(ROOT.glob("*.json")):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trials = data.get("trials", [])
    if not trials:
        continue

    num_patients += 1
    per_patient_counts = {label: 0 for label in LABELS}
    for t in trials:  # each t is one canonical NCT
        label = t.get("label", "explicit_contradiction")
        if label in per_patient_counts:
            per_patient_counts[label] += 1

    for label in LABELS:
        total_counts[label] += per_patient_counts[label]

if num_patients == 0:
    print("No merged_json files found under", ROOT)
    raise SystemExit

print(f"# patients: {num_patients}")
print("Average # canonical NCTs per patient in each label:")
for label in LABELS:
    avg = total_counts[label] / num_patients
    print(f"  {label}: {avg:.3f}")
