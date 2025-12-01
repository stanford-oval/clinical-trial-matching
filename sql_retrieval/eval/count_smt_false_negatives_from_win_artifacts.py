#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_smt_false_negatives_from_win_artifacts.py

Count SMT false negatives from the output artifacts created by
find_trialgpt_top200_wins.py.

Definition used here
--------------------
One SMT false negative = one TrialGPT "win" artifact folder, i.e. one copied folder:
    copied_trialgpt_mbench/<mode>/<patient_id>/<trial_id>/

with a corresponding:
    smt_conflict_summary.json

These are exactly the trials where:
- TrialGPT marked relevant=True AND eligible=True
- the trial was within the fixed TrialGPT top-K
- SMT did NOT return label == "all_satisfied"

So counting these folders/summaries gives the number of SMT false negatives.

What this script writes
-----------------------
Under --out (default: same root as input artifacts):
  * false_negatives_by_patient_mode.csv
  * false_negatives_by_mode.csv
  * false_negatives_by_patient.csv
  * false_negative_conflict_breakdown.csv
  * false_negative_summary.json

Input expectations
------------------
Point --wins-root at the output directory produced by find_trialgpt_top200_wins.py,
for example:
    trialgpt_top200_wins_out

This directory may contain:
  * copied_trialgpt_mbench/...
  * trialgpt_top200_re_true_smt_not_all_satisfied.csv

This script prefers the copied artifact folders/summaries, since you asked to use
the resulting artifacts that were created.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_WINS_ROOT = Path("./trialgpt_top200_wins_out")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_csv(path: Path, header: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


# ----------------------------------------------------------------------
# Artifact record
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FNRecord:
    mode: str
    patient_id: str
    trial_id: str
    conflict_category: str
    trialgpt_rank: Optional[int]
    trialgpt_fixed_k_used: Optional[int]
    summary_path: Path


# ----------------------------------------------------------------------
# Load from copied artifact folders
# ----------------------------------------------------------------------

def iter_summary_files(copied_root: Path) -> Iterable[Path]:
    if not copied_root.exists():
        return []
    return copied_root.rglob("smt_conflict_summary.json")


def load_records_from_copied_artifacts(copied_root: Path) -> List[FNRecord]:
    records: List[FNRecord] = []
    seen: Set[Tuple[str, str, str]] = set()

    for summary_path in iter_summary_files(copied_root):
        obj = load_json(summary_path)
        if not isinstance(obj, dict):
            continue

        mode = norm_str(obj.get("mode"))
        patient_id = norm_str(obj.get("patient_id"))
        trial_id = norm_str(obj.get("trial_id"))
        conflict = norm_str(obj.get("smt_conflict_category")) or "unknown"

        tg_rank_raw = obj.get("trialgpt_rank")
        tg_k_raw = obj.get("trialgpt_fixed_k_used")

        tg_rank = int(tg_rank_raw) if isinstance(tg_rank_raw, int) else None
        tg_k = int(tg_k_raw) if isinstance(tg_k_raw, int) else None

        if not mode or not patient_id or not trial_id:
            # Fallback from path: copied_trialgpt_mbench/<mode>/<patient>/<trial>/smt_conflict_summary.json
            try:
                rel = summary_path.relative_to(copied_root)
                parts = rel.parts
                if len(parts) >= 4:
                    mode = mode or parts[0]
                    patient_id = patient_id or parts[1]
                    trial_id = trial_id or parts[2]
            except Exception:
                pass

        if not mode or not patient_id or not trial_id:
            continue

        key = (mode, patient_id, trial_id)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            FNRecord(
                mode=mode,
                patient_id=patient_id,
                trial_id=trial_id,
                conflict_category=conflict,
                trialgpt_rank=tg_rank,
                trialgpt_fixed_k_used=tg_k,
                summary_path=summary_path,
            )
        )

    return records


# ----------------------------------------------------------------------
# Optional fallback: load from main CSV if copied artifacts are unavailable
# ----------------------------------------------------------------------

def load_records_from_csv(csv_path: Path) -> List[FNRecord]:
    if not csv_path.exists():
        return []

    records: List[FNRecord] = []
    seen: Set[Tuple[str, str, str]] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = norm_str(row.get("mode"))
            patient_id = norm_str(row.get("patient_id"))
            trial_id = norm_str(row.get("trial_id"))
            conflict = norm_str(row.get("smt_conflict_category")) or "unknown"

            if not mode or not patient_id or not trial_id:
                continue

            key = (mode, patient_id, trial_id)
            if key in seen:
                continue
            seen.add(key)

            tg_rank = None
            tg_k = None
            try:
                if row.get("trialgpt_rank", "").strip():
                    tg_rank = int(row["trialgpt_rank"])
            except Exception:
                pass
            try:
                if row.get("trialgpt_fixed_k_used", "").strip():
                    tg_k = int(row["trialgpt_fixed_k_used"])
            except Exception:
                pass

            records.append(
                FNRecord(
                    mode=mode,
                    patient_id=patient_id,
                    trial_id=trial_id,
                    conflict_category=conflict,
                    trialgpt_rank=tg_rank,
                    trialgpt_fixed_k_used=tg_k,
                    summary_path=csv_path,
                )
            )

    return records


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def build_outputs(records: List[FNRecord]) -> Dict[str, Any]:
    by_patient_mode: Counter[Tuple[str, str]] = Counter()
    by_mode: Counter[str] = Counter()
    by_patient: Counter[str] = Counter()
    by_conflict: Counter[Tuple[str, str]] = Counter()

    trials_by_patient_mode: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for r in records:
        by_patient_mode[(r.mode, r.patient_id)] += 1
        by_mode[r.mode] += 1
        by_patient[r.patient_id] += 1
        by_conflict[(r.mode, r.conflict_category)] += 1
        trials_by_patient_mode[(r.mode, r.patient_id)].append(r.trial_id)

    rows_patient_mode: List[Dict[str, Any]] = []
    for (mode, patient_id), count in sorted(by_patient_mode.items()):
        trial_ids = sorted(trials_by_patient_mode[(mode, patient_id)])
        rows_patient_mode.append({
            "mode": mode,
            "patient_id": patient_id,
            "smt_false_negatives": count,
            "trial_ids": "|".join(trial_ids),
        })

    rows_mode = [
        {"mode": mode, "smt_false_negatives": count}
        for mode, count in sorted(by_mode.items())
    ]

    rows_patient = [
        {"patient_id": patient_id, "smt_false_negatives": count}
        for patient_id, count in sorted(by_patient.items())
    ]

    rows_conflict = [
        {
            "mode": mode,
            "smt_conflict_category": conflict,
            "count": count,
        }
        for (mode, conflict), count in sorted(by_conflict.items())
    ]

    summary = {
        "num_false_negative_records": len(records),
        "num_modes": len(by_mode),
        "num_patients": len(by_patient),
        "modes": dict(sorted(by_mode.items())),
        "patients_with_false_negatives": sorted(by_patient.keys()),
    }

    return {
        "rows_patient_mode": rows_patient_mode,
        "rows_mode": rows_mode,
        "rows_patient": rows_patient,
        "rows_conflict": rows_conflict,
        "summary": summary,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count SMT false negatives per patient and mode from win artifacts."
    )
    ap.add_argument(
        "--wins-root",
        type=Path,
        default=DEFAULT_WINS_ROOT,
        help="Root output directory created by find_trialgpt_top200_wins.py",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write count outputs (default: --wins-root)",
    )
    ap.add_argument(
        "--prefer-copied-artifacts",
        action="store_true",
        default=True,
        help="Prefer copied_trialgpt_mbench/**/smt_conflict_summary.json over CSV fallback",
    )
    args = ap.parse_args()

    wins_root = args.wins_root
    if not wins_root.exists():
        raise FileNotFoundError(f"--wins-root not found: {wins_root}")

    out_dir = args.out or wins_root
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_root = wins_root / "copied_trialgpt_mbench"
    main_csv = wins_root / "trialgpt_top200_re_true_smt_not_all_satisfied.csv"

    records: List[FNRecord] = []

    if args.prefer_copied_artifacts:
        records = load_records_from_copied_artifacts(copied_root)
        if not records:
            records = load_records_from_csv(main_csv)
    else:
        records = load_records_from_csv(main_csv)
        if not records:
            records = load_records_from_copied_artifacts(copied_root)

    if not records:
        raise RuntimeError(
            "No false-negative records found. "
            f"Checked copied artifacts under {copied_root} and CSV {main_csv}"
        )

    outputs = build_outputs(records)

    write_csv(
        out_dir / "false_negatives_by_patient_mode.csv",
        ["mode", "patient_id", "smt_false_negatives", "trial_ids"],
        outputs["rows_patient_mode"],
    )
    write_csv(
        out_dir / "false_negatives_by_mode.csv",
        ["mode", "smt_false_negatives"],
        outputs["rows_mode"],
    )
    write_csv(
        out_dir / "false_negatives_by_patient.csv",
        ["patient_id", "smt_false_negatives"],
        outputs["rows_patient"],
    )
    write_csv(
        out_dir / "false_negative_conflict_breakdown.csv",
        ["mode", "smt_conflict_category", "count"],
        outputs["rows_conflict"],
    )
    write_json(
        out_dir / "false_negative_summary.json",
        outputs["summary"],
    )

    print(f"[ok] wrote {out_dir / 'false_negatives_by_patient_mode.csv'}")
    print(f"[ok] wrote {out_dir / 'false_negatives_by_mode.csv'}")
    print(f"[ok] wrote {out_dir / 'false_negatives_by_patient.csv'}")
    print(f"[ok] wrote {out_dir / 'false_negative_conflict_breakdown.csv'}")
    print(f"[ok] wrote {out_dir / 'false_negative_summary.json'}")
    print(
        "[info] total_false_negatives="
        f"{outputs['summary']['num_false_negative_records']} "
        f"patients={outputs['summary']['num_patients']} "
        f"modes={outputs['summary']['num_modes']}"
    )


if __name__ == "__main__":
    main()