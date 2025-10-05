#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter

ALL_CATEGORIES = [
    # Procedures
    # "Alleviates complication (procedure)",
    # "pre-treatment preparation (procedure)",
    # "other clinically relevant factors (procedure)",
    "relevant (procedure)",
    "not relevant (procedure)",

    # Findings
    "treatment target (finding)",
    "prevention target (finding)",
    "alleviate complication (finding)",
    "other clinically relevant factors (finding)",
    "not relevant (finding)",

    # Other (substance/product)
    # "Alleviates complication (substance/product)",
    # "pre-treatment preparation (substance/product)",
    # "other clinically relevant factors (substance/product)",
    "relevant (substance/product)",
    "not relevant (substance/product)",

    # other deterministically filtered literals
    "not relevant (numeric)",
    "not relevant (demographic)",
    "not relevant",
]


def load_categories_from_file(path: Path):
    """Extract disease → category from a single categorized JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # if "final_selected_concept_by_disease" not in data:
    #     return []

    categories = []
    #for disease, info in data["final_selected_concept_by_disease"].items():
    
    #for disease, info in data["representatives"].items():
    for info in data["representatives"]:
        category = info.get("category")
        if category is not None:
            categories.append(category)
    return categories


def calc_stats(categorized_dir: Path):
    """Calculate category counts and proportions."""
    all_categories = []

    json_files = sorted(categorized_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No categorized JSON files found in: {categorized_dir}")

    for file in json_files:
        categories = load_categories_from_file(file)
        all_categories.extend(categories)

    counter = Counter(all_categories)
    total = sum(counter.values())

    print("=" * 60)
    print(f"Categorized Disease Counts in: {categorized_dir}")
    print("=" * 60)
    print(f"Total diseases categorized: {total}")
    print()

    #for cat in ["treat", "prevent", "others"]:
    for cat in ALL_CATEGORIES:
        count = counter.get(cat, 0)
        prop = count / total if total > 0 else 0
        print(f"{cat:<10}: {count:>5}   ({prop:>6.2%})")

    print("=" * 60)
    return counter, total


def main():
    #categorized_dir = Path("../../../build/disease_test_categorized").resolve()
    categorized_dir = Path("../../../build/positive_constraint_literals_categorized/per_file").resolve()
    calc_stats(categorized_dir)


if __name__ == "__main__":
    main()
