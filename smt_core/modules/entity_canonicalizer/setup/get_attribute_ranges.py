#!/usr/bin/env python3
"""
get_attribute_ranges.py
──────────────────────────
Fetch every concept that is permitted as the **value** of a specific
SNOMED CT attribute and write the conceptIds to a .txt file.

• Reads the international Attribute-Range reference set (723562003)
  from a TSV/CSV export **or** queries it on-the-fly from Snowstorm’s
  MRCM endpoint.
• Parses the `rangeConstraint` ECL, including expressions that contain
  “OR”.
• Paginates through the Snowstorm `/MAIN/concepts` endpoint so every
  concept is returned (the default 50-row limit is bypassed).
• Produces a text file with one conceptId per line — easy to reuse in
  downstream scripts.

Example
-------
cd <SATIR_ROOT>/src/modules/EntityCanonicalizer/setup
$ python get_attribute_ranges.py \
        --attr-id 405814001 \
    

Author: <you>
"""

#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from typing import Iterable, List, Set, Optional   # ← added Optional

import requests
import os
import pathlib

# ───────────────────────────── configuration
BASE_URL   = "http://localhost:8080"
BRANCH     = "MAIN"
LIMIT      = 1000            # max=10 000 in Snowstorm; tweak if needed
RANGE_REFSET = "723562003"    # Attribute-Range reference set UUID

# ───────────────────────────── helpers
HEADERS = {"Accept": "application/json"}

def fetch_attribute_range_rows(attr_id: str,
                               mrcm_file: Optional[str] = None) -> List[str]:
    """
    Return every *active* rangeConstraint ECL expression for `attr_id`.

    If `mrcm_file` is given we read it locally (faster); otherwise we hit
    Snowstorm’s MRCM API.
    """
    if mrcm_file:
        constraints: list[str] = []
        with open(mrcm_file, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter='\t') \
                     if mrcm_file.lower().endswith((".tsv", ".txt")) \
                     else csv.DictReader(fh)
            for row in reader:
                if (row.get("active") == "1"
                        and row.get("refsetId") == RANGE_REFSET
                        and row.get("referencedComponentId") == attr_id):
                    constraints.append(row["rangeConstraint"])
        if constraints:
            return constraints
        sys.exit(f"[ERROR] No active rows for attribute {attr_id} in {mrcm_file}")

    # --- live query fallback ---
    url = f"{BASE_URL}/mrcm/{BRANCH}/attribute-ranges/{attr_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        sys.exit(f"[ERROR] No active rows for attribute {attr_id} via API")

    return [row["rangeConstraint"] for row in items
            if str(row.get("active", 0)) == "1"]


def split_ecl_by_or(ecl: str) -> List[str]:
    """
    Split a `rangeConstraint` into separate ECL fragments, handling:
      • multiple OR operators,
      • nested parentheses,
      • display names containing the word "or" (ignored).

    Example
    -------
    '<< 260245000 |Finding value| OR << 263714004 |Colors| OR << 308916002 |Environment or location|'
        → ['<<260245000', '<<263714004', '<<308916002']
    """
    # --- 1) Strip display names completely -----------------------------------
    # Everything between the first and next '|' is a human-readable term; remove it
    cleaned = re.sub(r"\|[^|]*\|", "", ecl)

    # --- 2) Scan the string, splitting on OR at depth 0 ----------------------
    parts, buf, depth = [], [], 0
    tokens = cleaned.split()   # tokenise by whitespace

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # update parenthesis depth BEFORE testing for OR
        depth += tok.count('(') - tok.count(')')

        # Boolean OR must be at depth 0 and spelled exactly "OR"
        if depth == 0 and tok.upper() == "OR":
            # flush buffer to parts
            if buf:
                parts.append("".join(buf).strip())
                buf.clear()
            i += 1
            continue

        buf.append(tok)
        i += 1

    # flush any remaining fragment
    if buf:
        parts.append("".join(buf).strip())

    # --- 3) Final clean-up ----------------------------------------------------
    out = []
    for p in parts:
        # remove *internal* whitespace so '<< 123' -> '<<123'
        p = re.sub(r"\s+", "", p)
        # drop single outer parentheses, if any
        if p.startswith("(") and p.endswith(")"):
            p = p[1:-1]
        if p:
            out.append(p)

    return out


def fetch_concepts_for_ecl(ecl: str) -> Iterable[str]:
    """
    Yield every conceptId that matches `ecl`, using Snowstorm’s
    `searchAfter` cursor so we’re not limited by the 10 000-row
    `offset + pageSize` ceiling.
    """
    params = {"ecl": ecl, "limit": LIMIT}
    search_after = None

    while True:
        if search_after is not None:
            # Snowstorm expects the token as a JSON array string, e.g. ["123456"]
            params["searchAfter"] = search_after
        else:
            params.pop("searchAfter", None)    # first page

        resp = requests.get(f"{BASE_URL}/{BRANCH}/concepts",
                            params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data   = resp.json()
        items  = data.get("items", [])
        if not items:
            break

        # emit conceptIds
        for item in items:
            yield item["conceptId"]

        # prepare the cursor for the next loop
        search_after = data.get("searchAfter")
        if not search_after:
            break


def export_attribute_ranges(attr_id: str,
                            outfile: str,
                            mrcm_file: Optional[str] = None) -> None:
    """Main orchestration."""
    print(f"→ Resolving ranges for attribute {attr_id}…")
    constraints = fetch_attribute_range_rows(attr_id, mrcm_file)
    if not constraints:
        sys.exit("[ERROR] No rangeConstraint found.")

    concept_ids: Set[str] = set()
    print("constraints: ", constraints)
    for rc in constraints:
        print("split_ecl_by_or(rc): ", split_ecl_by_or(rc))
        for frag in split_ecl_by_or(rc):
            print("frag: ", frag)
            print(f"   • Fetching concepts matching ECL: {frag}")
            concept_ids.update(fetch_concepts_for_ecl(frag))

    print(f"→ Writing {len(concept_ids):,} conceptIds to {outfile}")
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(concept_ids)) + "\n")


# ───────────────────────────── CLI entry-point
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Export the permitted range concepts for an attribute.")
    ap.add_argument("--attr-id", required=True,
                    help="ConceptId of the attribute (e.g. 405814001)")
    # ap.add_argument("--out", required=True,
    #                 help="Path of the output .txt file")
    # ap.add_argument("--mrcm-file",
    #                 help="Local Attribute-Range TSV/CSV (optional)")
    args = ap.parse_args()

    setup_dir = pathlib.Path(os.environ.get("ENTITY_SETUP_DIR",
                                        pathlib.Path(__file__).resolve().parent))
    out_path = str(setup_dir / "attribute_ranges" / f"{args.attr_id}_ranges.txt")
    export_attribute_ranges(args.attr_id, out_path, os.environ.get("SNOMED_MRCM_RANGE_FILE", ""))
