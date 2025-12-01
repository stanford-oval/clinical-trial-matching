#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
migrate_patient_labels_to_nct.py

Rewrite clean_eval/patient_labels to use NCT IDs as trial_id, so that
run_eval_precision.py → run_all_judges.py can look up trials in the SIGIR corpus
(which is keyed by NCT ID, not internal integer trial IDs).

Assumptions:
  - compose_trial_eval.py was run with some base output root, e.g.:
        --out OPS_ROOT

  - Under that root, you have:
        OPS_ROOT/retrieved_mappings/json/{patient_id}.json
        OPS_ROOT/clean_eval/patient_labels/{patient_id}.json   (original, int IDs)

This script:
  * Reads retrieved_mappings/json/{patient}.json
      {
        "patient_id": "...",
        "trials": [
          {
            "rank": 1,
            "status": "survivor" | "eliminated",
            "label": "all_satisfied" | "unsatisfied_inclusion" | "explicit_contradiction",
            "trial_id": 2033,
            "nct_id": "NCT00465907",
            ...
          },
          ...
        ]
      }

  * Builds a new patient_labels JSON for each patient:
      {
        "patient_id": "...",
        "trials": [
          { "trial_id": "NCT00465907", "rank": 1, "label": "all_satisfied" },
          ...
        ]
      }

    If nct_id is missing for some row, it falls back to str(trial_id).

By default, the rewritten files go under:
  <root>/clean_eval/patient_labels_nct/{patient_id}.json

If you pass --overwrite, they instead overwrite:
  <root>/clean_eval/patient_labels/{patient_id}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def migrate_patient(
    patient_id: str,
    retrieved_json_path: Path,
    out_dir: Path,
) -> None:
    """Convert retrieved_mappings JSON → patient_labels JSON (NCT-based)."""
    data = load_json(retrieved_json_path)
    trials = data.get("trials", [])

    out_trials: List[Dict[str, Any]] = []
    for row in trials:
        # Prefer NCT ID; fallback to internal trial_id string if somehow missing
        trial_key = row.get("nct_id") or str(row.get("trial_id"))
        label = row.get("label", "explicit_contradiction")
        rank = row.get("rank")

        if rank is None:
            # If rank wasn't saved (shouldn't happen with current compose),
            # fall back to positional index (1-based).
            rank = len(out_trials) + 1

        out_trials.append(
            {
                "trial_id": trial_key,
                "rank": int(rank),
                "label": label,
            }
        )

    out_obj = {
        "patient_id": data.get("patient_id", patient_id),
        "trials": out_trials,
    }

    out_path = out_dir / f"{patient_id}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] wrote NCT-based labels for patient {patient_id} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Migrate clean_eval/patient_labels to use NCT IDs as trial_id "
            "by joining with retrieved_mappings/json."
        )
    )
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help=(
            "Base root where compose_trial_eval outputs live. "
            "Expected layout: "
            "<root>/retrieved_mappings/json and <root>/clean_eval/patient_labels."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If set, overwrite <root>/clean_eval/patient_labels/*.json in-place. "
            "Otherwise write to <root>/clean_eval/patient_labels_nct/."
        ),
    )
    ap.add_argument(
        "--patients",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of patient_ids to migrate. "
            "If omitted, all patients found in retrieved_mappings/json are used."
        ),
    )

    args = ap.parse_args()
    root = Path(args.root).resolve()

    retrieved_json_dir = root / "retrieved_mappings" / "json"
    original_labels_dir = root / "clean_eval" / "patient_labels"

    if not retrieved_json_dir.exists():
        raise SystemExit(f"[err] retrieved_mappings/json not found under {retrieved_json_dir}")
    if not original_labels_dir.exists():
        print(f"[warn] original patient_labels dir does not exist: {original_labels_dir} "
              f"(continuing; we'll still use retrieved_mappings/json as source of truth)")

    if args.overwrite:
        out_dir = original_labels_dir
        print(f"[info] Overwriting existing patient_labels in: {out_dir}")
    else:
        out_dir = root / "clean_eval" / "patient_labels_nct"
        print(f"[info] Writing NCT-based labels to: {out_dir}")

    # Determine which patients to process
    if args.patients:
        patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        # Infer from retrieved_mappings/json/*.json filenames
        patient_ids = sorted(
            p.stem for p in retrieved_json_dir.glob("*.json")
        )

    if not patient_ids:
        raise SystemExit("[err] No patients found to migrate.")

    print(f"[info] Will migrate {len(patient_ids)} patient(s): {', '.join(patient_ids)}")

    for pid in patient_ids:
        rj = retrieved_json_dir / f"{pid}.json"
        if not rj.exists():
            print(f"[warn] missing retrieved_mappings/json for patient {pid}: {rj}; skipping")
            continue
        try:
            migrate_patient(pid, rj, out_dir)
        except Exception as e:
            print(f"[err] failed to migrate patient {pid}: {e}")


if __name__ == "__main__":
    main()
