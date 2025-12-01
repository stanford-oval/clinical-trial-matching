#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count distinct patient-trial false-negative pairs across modes from win artifacts."
    )
    ap.add_argument(
        "--wins-root",
        type=Path,
        required=True,
        help="Root output dir from find_trialgpt_top200_wins.py",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: wins-root)",
    )
    args = ap.parse_args()

    wins_root = args.wins_root
    out_dir = args.out or wins_root
    copied_root = wins_root / "copied_trialgpt_mbench"
    csv_fallback = wins_root / "trialgpt_top200_re_true_smt_not_all_satisfied.csv"

    pair_to_modes: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    pair_to_conflicts: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    found = False

    if copied_root.exists():
        for p in copied_root.rglob("smt_conflict_summary.json"):
            obj = load_json(p)
            if not isinstance(obj, dict):
                continue

            patient_id = str(obj.get("patient_id", "")).strip()
            trial_id = str(obj.get("trial_id", "")).strip()
            mode = str(obj.get("mode", "")).strip()
            conflict = str(obj.get("smt_conflict_category", "")).strip()

            if not patient_id or not trial_id or not mode:
                continue

            pair = (patient_id, trial_id)
            pair_to_modes[pair].add(mode)
            if conflict:
                pair_to_conflicts[pair].add(conflict)
            found = True

    if not found and csv_fallback.exists():
        with csv_fallback.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = str(row.get("patient_id", "")).strip()
                trial_id = str(row.get("trial_id", "")).strip()
                mode = str(row.get("mode", "")).strip()
                conflict = str(row.get("smt_conflict_category", "")).strip()

                if not patient_id or not trial_id or not mode:
                    continue

                pair = (patient_id, trial_id)
                pair_to_modes[pair].add(mode)
                if conflict:
                    pair_to_conflicts[pair].add(conflict)
                found = True

    if not found:
        raise RuntimeError("No artifacts found.")

    rows: List[Dict[str, Any]] = []
    mode_combo_counts: Dict[str, int] = defaultdict(int)

    for (patient_id, trial_id), modes in sorted(pair_to_modes.items()):
        mode_list = sorted(modes)
        mode_combo = "|".join(mode_list)
        mode_combo_counts[mode_combo] += 1

        rows.append({
            "patient_id": patient_id,
            "trial_id": trial_id,
            "num_modes": len(mode_list),
            "modes": mode_combo,
            "conflict_categories": "|".join(sorted(pair_to_conflicts[(patient_id, trial_id)])),
        })

    summary_rows = [
        {"modes": combo, "count": cnt}
        for combo, cnt in sorted(mode_combo_counts.items())
    ]

    write_csv(
        out_dir / "distinct_patient_trial_pairs_across_modes.csv",
        ["patient_id", "trial_id", "num_modes", "modes", "conflict_categories"],
        rows,
    )
    write_csv(
        out_dir / "distinct_patient_trial_pairs_mode_combo_counts.csv",
        ["modes", "count"],
        summary_rows,
    )

    print(f"[ok] wrote {out_dir / 'distinct_patient_trial_pairs_across_modes.csv'}")
    print(f"[ok] wrote {out_dir / 'distinct_patient_trial_pairs_mode_combo_counts.csv'}")
    print(f"[info] distinct_patient_trial_pairs={len(rows)}")


if __name__ == "__main__":
    main()