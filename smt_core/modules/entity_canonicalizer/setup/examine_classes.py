#!/usr/bin/env python3
"""List unique classes (and counts) in an exact-terms JSONL file.

Each line of the input file must be a JSON object with at least a
"class" field, e.g.:
    {"term": "suicide precautions", "class": "procedure"}

Usage:
    python list_classes.py exact_terms.jsonl

Outputs tab‑separated class names and their counts, ordered by
descending frequency.
"""

import json
import sys
from collections import Counter
from pathlib import Path


def collect_classes(path: Path) -> Counter:
    """Read *path* and count occurrences of each class value."""
    counter = Counter()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] line {lineno}: JSON decode error – {exc}", file=sys.stderr)
                continue
            cls = obj.get("class")
            if cls is None:
                print(f"[warn] line {lineno}: missing 'class' field", file=sys.stderr)
                continue
            counter[cls] += 1
    return counter


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print("Usage: python list_classes.py <exact_terms.jsonl>", file=sys.stderr)
        sys.exit(1)

    path = Path(argv[1])
    if not path.is_file():
        print(f"Error: '{path}' is not a file or cannot be read", file=sys.stderr)
        sys.exit(1)

    counts = collect_classes(path)
    if not counts:
        print("No class data found.")
        return

    # Print results sorted by descending frequency, then alphabetically.
    for cls, cnt in counts.most_common():
        print(f"{cls}\t{cnt}")


if __name__ == "__main__":
    main(sys.argv)
