#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Set


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def trial_id_from_filename(filename: str) -> str:
    # e.g. "NCT00000369_disease_link_filter_summary.json" -> "NCT00000369"
    return filename.split("_", 1)[0]


def extend_trial(
    disease_summary_path: Path,
    agg_root: Path,
) -> Dict[str, Any]:
    """Return modified JSON for a single trial (doesn't write it)."""
    summary = load_json(disease_summary_path)
    fname = disease_summary_path.name
    trial_id = trial_id_from_filename(fname)

    aggregate_path = agg_root / trial_id / "aggregate_disease.json"
    if not aggregate_path.is_file():
        print(f"[WARN] No aggregate_disease.json for {trial_id}, skipping siblings.")
        return summary

    aggregate = load_json(aggregate_path)

    final_sel: Dict[str, Any] = summary.setdefault(
        "final_selected_concept_by_disease", {}
    )
    linked_result: Dict[str, Any] = summary.setdefault("linked_result", {})

    # Track which diseases we already have (to avoid duplicates)
    existing_disease_names: Set[str] = set(final_sel.keys())

    added = 0

    for item in aggregate.get("items", []):
        yes_examples = item.get("yes_examples", [])
        for ex in yes_examples:
            cand_label = ex.get("candidate_concept")
            cand_id = ex.get("conceptId")

            if not cand_label or not cand_id:
                continue

            if cand_label in existing_disease_names:
                # Already present in final_selected_concept_by_disease
                continue

            # Build a minimal SNOMED-like record for the sibling concept
            sibling_record = {
                "disease": cand_label,
                "conceptId": str(cand_id),
                "preferred_term": cand_label,
                "fully_specified_name": cand_label,
                "type": "Clinical finding",
                "definition": None,
                "best_match_term": cand_label,
            }

            # Add to final_selected_concept_by_disease
            final_sel[cand_label] = sibling_record

            # Optionally also mirror into linked_result so shape stays similar
            if cand_label not in linked_result:
                linked_result[cand_label] = {
                    "conceptId": str(cand_id),
                    "preferred_term": cand_label,
                    "fully_specified_name": cand_label,
                    "type": "Clinical finding",
                    "definition": None,
                    "best_match_term": cand_label,
                    "match_reason": "sibling_from_yes_examples",
                }

            existing_disease_names.add(cand_label)
            added += 1

    print(
        f"[INFO] {trial_id}: added {added} sibling diseases "
        f"(total now {len(final_sel)})"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Add yes_example sibling diseases into build/disease_siblinged."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../../",
        help="Path to TrialGPT-SMT repo root (default: current dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log but do not write any files.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    disease_build_dir = root / "build" / "disease"
    agg_root = root / "siblingdsrc" / "sibling_logs_disease"
    out_dir = root / "build" / "disease_siblinged"

    if not disease_build_dir.is_dir():
        raise SystemExit(f"Missing directory: {disease_build_dir}")
    if not agg_root.is_dir():
        raise SystemExit(f"Missing directory: {agg_root}")

    print(f"[ROOT] {root}")
    print(f"[IN ] disease JSONs: {disease_build_dir}")
    print(f"[IN ] sibling logs : {agg_root}")
    print(f"[OUT] siblinged    : {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(disease_build_dir):
        if not fname.endswith("_disease_link_filter_summary.json"):
            continue

        in_path = disease_build_dir / fname
        summary_extended = extend_trial(in_path, agg_root)

        if args.dry_run:
            continue

        out_path = out_dir / fname
        save_json(summary_extended, out_path)


if __name__ == "__main__":
    main()
