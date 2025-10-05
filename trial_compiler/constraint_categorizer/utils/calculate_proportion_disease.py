#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter

CATEGORY = [
    "treat",
    "prevent",
    "alleviate complications",
    "other",
    "not relevant",
]

def load_categories_from_file(path: Path):
    """Extract disease → category from a single categorized JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "final_selected_concept_by_disease" not in data:
        return []

    categories = []
    for disease, info in data["final_selected_concept_by_disease"].items():
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

    for cat in CATEGORY:
        count = counter.get(cat, 0)
        prop = count / total if total > 0 else 0
        print(f"{cat:<10}: {count:>5}   ({prop:>6.2%})")

    print("=" * 60)
    return counter, total


def main():
    categorized_dir = Path("../../../build/disease_test_categorized").resolve()
    #categorized_dir = Path("../../../build/positive_constraint_literals_categorized/per_file").resolve()
    calc_stats(categorized_dir)


if __name__ == "__main__":
    main()
