#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample_physician_validation_union.py

Build a physician-validation sample from the UNION of SMT and TrialGPT
augmented patient_labels outputs.

Sampling goal
-------------
This script is designed for the setup we discussed:

- Main paper results come from the full LLM-judge evaluation.
- Physician review is a separate validation study.
- Sampling frame is the UNION of judged items from:
    * SMT outputs
    * TrialGPT outputs
- Sample unit is a judged patient-trial item:
    * (mode, patient_id, trial_id)
- Stratify by:
    * mode: ccr / all / all-explore
    * final judgment bucket:
        - relevant_eligible
        - relevant_ineligible
        - not_relevant
- By default, allocation is BALANCED:
    * equal total across modes
    * equal total across buckets within each mode
- If a stratum is smaller than its target, we cap at population and
  redistribute leftover quota to other strata with remaining room.

Input layout
------------
SMT root:
  <smt_root>/<mode>/patient_labels/*.json

TrialGPT root:
  <trialgpt_root>/<mode>/patient_labels/*.json

Each JSON is expected to contain:
{
  "patient_id": "...",
  "trials": [
    {
      "trial_id": "...",
      "label": "...",                         # optional
      "rank": 1,                              # optional
      "any_subcohort_relevant": true/false,
      "any_subcohort_eligible": true/false,
      "any_subcohort_relevant_and_eligible": true/false,
      "subcohort_judge_summary": [...]        # optional
    },
    ...
  ]
}

Union row definition
--------------------
One union item = unique (mode, patient_id, trial_id)

For each union item we keep:
- in_smt / in_trialgpt
- source_bucket:
    * smt_only
    * trialgpt_only
    * both
- system-specific judge outputs if present
- a single union judgment bucket, chosen by --union-label-policy

Union-label policy
------------------
By default:
  --union-label-policy prefer_smt

Choices:
- prefer_smt
    Use SMT label if present, else TrialGPT label.
- prefer_trialgpt
    Use TrialGPT label if present, else SMT label.
- require_match
    Keep only items where both systems agree when both present.
    For items present in only one system, use that system's label.
- pooled_system_rows
    Do NOT collapse union rows. Instead sample system-specific rows:
      (mode, patient_id, trial_id, source_system)
    This is statistically cleaner if you want to validate the judge
    as applied separately to each system pipeline.

Outputs
-------
Writes:
- sampled_items.csv
- sampled_items.jsonl
- allocation_summary.csv
- population_summary.csv
- manifest.json

Recommended usage
-----------------
python sample_physician_validation_union.py \
  --smt-root ./smt_retrieval_eval_out \
  --trialgpt-root ./trialgpt_retrieval_eval_out \
  --output-dir ./physician_validation_union_sample \
  --modes ccr,all,all-explore \
  --total-n 96 \
  --allocation balanced \
  --per-mode-equal-total \
  --union-label-policy prefer_smt \
  --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MODES_DEFAULT = ["ccr", "all", "all-explore"]
BUCKETS = ["relevant_eligible", "relevant_ineligible", "not_relevant"]


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class SystemItem:
    mode: str
    patient_id: str
    trial_id: str
    bucket: str
    source_system: str  # "smt" or "trialgpt"

    label: str = ""
    rank: str = ""
    any_subcohort_relevant: str = ""
    any_subcohort_eligible: str = ""
    any_subcohort_relevant_and_eligible: str = ""
    num_subcohorts: str = ""
    subcohort_summary_json: str = ""
    source_file: str = ""


@dataclass(frozen=True)
class UnionItem:
    mode: str
    patient_id: str
    trial_id: str

    source_bucket: str  # smt_only / trialgpt_only / both
    chosen_bucket: str

    in_smt: int
    in_trialgpt: int

    smt_bucket: str = ""
    trialgpt_bucket: str = ""

    smt_label: str = ""
    trialgpt_label: str = ""

    smt_rank: str = ""
    trialgpt_rank: str = ""

    smt_any_subcohort_relevant: str = ""
    smt_any_subcohort_eligible: str = ""
    smt_any_subcohort_relevant_and_eligible: str = ""

    trialgpt_any_subcohort_relevant: str = ""
    trialgpt_any_subcohort_eligible: str = ""
    trialgpt_any_subcohort_relevant_and_eligible: str = ""

    smt_num_subcohorts: str = ""
    trialgpt_num_subcohorts: str = ""

    smt_subcohort_summary_json: str = ""
    trialgpt_subcohort_summary_json: str = ""

    smt_source_file: str = ""
    trialgpt_source_file: str = ""


@dataclass(frozen=True)
class PooledRow:
    mode: str
    patient_id: str
    trial_id: str
    source_system: str
    bucket: str

    label: str = ""
    rank: str = ""
    any_subcohort_relevant: str = ""
    any_subcohort_eligible: str = ""
    any_subcohort_relevant_and_eligible: str = ""
    num_subcohorts: str = ""
    subcohort_summary_json: str = ""
    source_file: str = ""


# =============================================================================
# Generic helpers
# =============================================================================

def parse_modes(modes_arg: str) -> List[str]:
    modes = [m.strip() for m in modes_arg.split(",") if m.strip()]
    if not modes:
        raise ValueError("No modes specified.")
    return modes


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def bool_to_str(x) -> str:
    if x is True:
        return "true"
    if x is False:
        return "false"
    if x is None:
        return ""
    return str(x)


def classify_bucket(trial_obj: dict) -> str:
    rel = trial_obj.get("any_subcohort_relevant")
    elig = trial_obj.get("any_subcohort_eligible")

    if rel is True and elig is True:
        return "relevant_eligible"
    if rel is True:
        return "relevant_ineligible"
    return "not_relevant"


def largest_remainder_allocation(
    weights: Dict[Tuple[str, str], float],
    total_n: int,
) -> Dict[Tuple[str, str], int]:
    if total_n < 0:
        raise ValueError("total_n must be >= 0")

    positive_keys = [k for k, w in weights.items() if w > 0]
    if total_n == 0 or not positive_keys:
        return {k: 0 for k in weights}

    total_w = sum(weights[k] for k in positive_keys)
    raw = {k: (weights[k] / total_w) * total_n for k in positive_keys}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    used = sum(base.values())
    rem = total_n - used

    frac_sorted = sorted(
        positive_keys,
        key=lambda k: (raw[k] - base[k], raw[k], k),
        reverse=True,
    )

    out = {k: 0 for k in weights}
    for k in positive_keys:
        out[k] = base[k]
    for k in frac_sorted[:rem]:
        out[k] += 1
    return out


def rebalance_with_caps(
    alloc: Dict[Tuple[str, str], int],
    caps: Dict[Tuple[str, str], int],
) -> Dict[Tuple[str, str], int]:
    alloc = dict(alloc)
    keys = list(alloc.keys())

    while True:
        overflow = 0
        for k in keys:
            cap = caps.get(k, 0)
            if alloc[k] > cap:
                overflow += alloc[k] - cap
                alloc[k] = cap

        if overflow == 0:
            return alloc

        room_keys = [k for k in keys if alloc[k] < caps.get(k, 0)]
        if not room_keys:
            return alloc

        room_weights = {k: float(caps[k] - alloc[k]) for k in room_keys}
        add = largest_remainder_allocation(room_weights, overflow)

        added_any = False
        for k, v in add.items():
            if v > 0:
                alloc[k] += v
                added_any = True

        if not added_any:
            return alloc


def apply_min_floor(
    alloc: Dict[Tuple[str, str], int],
    caps: Dict[Tuple[str, str], int],
    min_per_nonempty_stratum: int,
) -> Dict[Tuple[str, str], int]:
    alloc = dict(alloc)
    if min_per_nonempty_stratum <= 0:
        return alloc

    nonempty = [k for k, cap in caps.items() if cap > 0]
    for k in nonempty:
        alloc[k] = min(alloc[k], caps[k])

    target_floor_total = sum(min(min_per_nonempty_stratum, caps[k]) for k in nonempty)
    current_total = sum(alloc.values())
    if current_total <= 0:
        return alloc

    if target_floor_total > current_total:
        floor_weights = {k: 1.0 for k in nonempty}
        alloc2 = largest_remainder_allocation(floor_weights, current_total)
        return rebalance_with_caps(alloc2, caps)

    deficits = {}
    for k in nonempty:
        floor_k = min(min_per_nonempty_stratum, caps[k])
        if alloc[k] < floor_k:
            deficits[k] = floor_k - alloc[k]

    total_deficit = sum(deficits.values())
    if total_deficit == 0:
        return alloc

    donors = []
    donor_room = {}
    for k in nonempty:
        floor_k = min(min_per_nonempty_stratum, caps[k])
        extra = alloc[k] - floor_k
        if extra > 0:
            donors.append(k)
            donor_room[k] = extra

    available = sum(donor_room.values())
    shift = min(total_deficit, available)
    if shift <= 0:
        return alloc

    take = largest_remainder_allocation({k: float(v) for k, v in donor_room.items()}, shift)
    for k, v in take.items():
        alloc[k] -= v

    deficits = {}
    for k in nonempty:
        floor_k = min(min_per_nonempty_stratum, caps[k])
        if alloc[k] < floor_k:
            deficits[k] = floor_k - alloc[k]

    for k, v in deficits.items():
        alloc[k] += v

    return rebalance_with_caps(alloc, caps)


# =============================================================================
# Load system items
# =============================================================================

def load_system_items(root: Path, source_system: str, modes: List[str]) -> List[SystemItem]:
    items: List[SystemItem] = []

    for mode in modes:
        labels_dir = root / mode / "patient_labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing directory: {labels_dir}")

        for path in sorted(labels_dir.glob("*.json")):
            obj = read_json(path)
            patient_id = obj.get("patient_id")
            trials = obj.get("trials", [])
            if not isinstance(patient_id, str) or not isinstance(trials, list):
                continue

            for t in trials:
                if not isinstance(t, dict):
                    continue
                trial_id = t.get("trial_id")
                if not isinstance(trial_id, str) or not trial_id:
                    continue

                bucket = classify_bucket(t)
                sub_summary = t.get("subcohort_judge_summary", [])
                num_subcohorts = len(sub_summary) if isinstance(sub_summary, list) else ""

                items.append(
                    SystemItem(
                        mode=mode,
                        patient_id=patient_id,
                        trial_id=trial_id,
                        bucket=bucket,
                        source_system=source_system,
                        label=str(t.get("label", "")) if t.get("label") is not None else "",
                        rank=str(t.get("rank", "")) if t.get("rank") is not None else "",
                        any_subcohort_relevant=bool_to_str(t.get("any_subcohort_relevant")),
                        any_subcohort_eligible=bool_to_str(t.get("any_subcohort_eligible")),
                        any_subcohort_relevant_and_eligible=bool_to_str(
                            t.get("any_subcohort_relevant_and_eligible")
                        ),
                        num_subcohorts=str(num_subcohorts),
                        subcohort_summary_json=json.dumps(sub_summary, ensure_ascii=False),
                        source_file=str(path),
                    )
                )

    return items


# =============================================================================
# Union building
# =============================================================================

def build_union_items(
    smt_items: List[SystemItem],
    trialgpt_items: List[SystemItem],
    union_label_policy: str,
) -> List[UnionItem]:
    smt_map: Dict[Tuple[str, str, str], SystemItem] = {
        (x.mode, x.patient_id, x.trial_id): x for x in smt_items
    }
    tg_map: Dict[Tuple[str, str, str], SystemItem] = {
        (x.mode, x.patient_id, x.trial_id): x for x in trialgpt_items
    }

    all_keys = sorted(set(smt_map.keys()) | set(tg_map.keys()))
    out: List[UnionItem] = []

    for mode, patient_id, trial_id in all_keys:
        s = smt_map.get((mode, patient_id, trial_id))
        t = tg_map.get((mode, patient_id, trial_id))

        in_smt = 1 if s is not None else 0
        in_tg = 1 if t is not None else 0

        if in_smt and in_tg:
            source_bucket = "both"
        elif in_smt:
            source_bucket = "smt_only"
        else:
            source_bucket = "trialgpt_only"

        smt_bucket = s.bucket if s is not None else ""
        tg_bucket = t.bucket if t is not None else ""

        chosen_bucket: Optional[str] = None

        if union_label_policy == "prefer_smt":
            chosen_bucket = smt_bucket if s is not None else tg_bucket
        elif union_label_policy == "prefer_trialgpt":
            chosen_bucket = tg_bucket if t is not None else smt_bucket
        elif union_label_policy == "require_match":
            if s is not None and t is not None:
                if s.bucket != t.bucket:
                    continue
                chosen_bucket = s.bucket
            elif s is not None:
                chosen_bucket = s.bucket
            elif t is not None:
                chosen_bucket = t.bucket
        else:
            raise ValueError(f"Unknown union_label_policy={union_label_policy}")

        if chosen_bucket not in BUCKETS:
            continue

        out.append(
            UnionItem(
                mode=mode,
                patient_id=patient_id,
                trial_id=trial_id,
                source_bucket=source_bucket,
                chosen_bucket=chosen_bucket,
                in_smt=in_smt,
                in_trialgpt=in_tg,

                smt_bucket=smt_bucket,
                trialgpt_bucket=tg_bucket,

                smt_label=s.label if s is not None else "",
                trialgpt_label=t.label if t is not None else "",

                smt_rank=s.rank if s is not None else "",
                trialgpt_rank=t.rank if t is not None else "",

                smt_any_subcohort_relevant=s.any_subcohort_relevant if s is not None else "",
                smt_any_subcohort_eligible=s.any_subcohort_eligible if s is not None else "",
                smt_any_subcohort_relevant_and_eligible=(
                    s.any_subcohort_relevant_and_eligible if s is not None else ""
                ),

                trialgpt_any_subcohort_relevant=t.any_subcohort_relevant if t is not None else "",
                trialgpt_any_subcohort_eligible=t.any_subcohort_eligible if t is not None else "",
                trialgpt_any_subcohort_relevant_and_eligible=(
                    t.any_subcohort_relevant_and_eligible if t is not None else ""
                ),

                smt_num_subcohorts=s.num_subcohorts if s is not None else "",
                trialgpt_num_subcohorts=t.num_subcohorts if t is not None else "",

                smt_subcohort_summary_json=s.subcohort_summary_json if s is not None else "",
                trialgpt_subcohort_summary_json=t.subcohort_summary_json if t is not None else "",

                smt_source_file=s.source_file if s is not None else "",
                trialgpt_source_file=t.source_file if t is not None else "",
            )
        )

    return out


# =============================================================================
# Pooled-system-row mode
# =============================================================================

def build_pooled_rows(
    smt_items: List[SystemItem],
    trialgpt_items: List[SystemItem],
) -> List[PooledRow]:
    rows: List[PooledRow] = []
    for x in smt_items + trialgpt_items:
        rows.append(
            PooledRow(
                mode=x.mode,
                patient_id=x.patient_id,
                trial_id=x.trial_id,
                source_system=x.source_system,
                bucket=x.bucket,
                label=x.label,
                rank=x.rank,
                any_subcohort_relevant=x.any_subcohort_relevant,
                any_subcohort_eligible=x.any_subcohort_eligible,
                any_subcohort_relevant_and_eligible=x.any_subcohort_relevant_and_eligible,
                num_subcohorts=x.num_subcohorts,
                subcohort_summary_json=x.subcohort_summary_json,
                source_file=x.source_file,
            )
        )
    return rows


# =============================================================================
# Counting helpers
# =============================================================================

def counts_from_union_items(items: List[UnionItem], modes: List[str]) -> Dict[Tuple[str, str], int]:
    counts = {(m, b): 0 for m in modes for b in BUCKETS}
    for it in items:
        counts[(it.mode, it.chosen_bucket)] += 1
    return counts


def counts_from_pooled_rows(items: List[PooledRow], modes: List[str]) -> Dict[Tuple[str, str], int]:
    counts = {(m, b): 0 for m in modes for b in BUCKETS}
    for it in items:
        counts[(it.mode, it.bucket)] += 1
    return counts


# =============================================================================
# Allocation
# =============================================================================

def _normalize_allocation_name(allocation: str) -> str:
    if allocation == "balanced":
        return "equal"
    return allocation


def allocate_samples_generic(
    *,
    counts: Dict[Tuple[str, str], int],
    modes: List[str],
    total_n: int,
    allocation: str,
    per_mode_equal_total: bool,
    min_per_nonempty_stratum: int,
) -> Dict[Tuple[str, str], int]:
    allocation = _normalize_allocation_name(allocation)

    caps = dict(counts)
    pop_total = sum(counts.values())
    if pop_total == 0:
        raise RuntimeError("No items available for sampling.")

    total_n = min(total_n, pop_total)

    if per_mode_equal_total:
        mode_counts = {m: sum(counts[(m, b)] for b in BUCKETS) for m in modes}
        nonempty_modes = [m for m in modes if mode_counts[m] > 0]
        if not nonempty_modes:
            raise RuntimeError("No nonempty modes found.")

        mode_alloc = largest_remainder_allocation(
            {(m, "__mode__"): 1.0 for m in nonempty_modes},
            total_n,
        )
        mode_alloc = rebalance_with_caps(
            mode_alloc,
            {(m, "__mode__"): mode_counts[m] for m in nonempty_modes},
        )
        mode_alloc_simple = {m: mode_alloc[(m, "__mode__")] for m in nonempty_modes}

        final_alloc = {(m, b): 0 for m in modes for b in BUCKETS}

        for m in modes:
            mode_total = mode_alloc_simple.get(m, 0)
            if mode_total <= 0:
                continue

            mode_strata = [(m, b) for b in BUCKETS]
            mode_caps = {k: caps[k] for k in mode_strata}

            if allocation == "equal":
                weights = {k: 1.0 if mode_caps[k] > 0 else 0.0 for k in mode_strata}
            elif allocation == "proportional":
                weights = {k: float(mode_caps[k]) for k in mode_strata}
            else:
                raise ValueError(f"Unknown allocation={allocation}")

            alloc_m = largest_remainder_allocation(weights, mode_total)
            alloc_m = rebalance_with_caps(alloc_m, mode_caps)
            alloc_m = apply_min_floor(alloc_m, mode_caps, min_per_nonempty_stratum)
            alloc_m = rebalance_with_caps(alloc_m, mode_caps)

            for k, v in alloc_m.items():
                final_alloc[k] = v

        current = sum(final_alloc.values())
        remainder = total_n - current
        if remainder > 0:
            room = {k: caps[k] - final_alloc[k] for k in final_alloc if caps[k] - final_alloc[k] > 0}
            add = largest_remainder_allocation({k: float(v) for k, v in room.items()}, remainder)
            for k, v in add.items():
                final_alloc[k] += v

        return final_alloc

    else:
        strata = [(m, b) for m in modes for b in BUCKETS]
        if allocation == "equal":
            weights = {k: 1.0 if caps[k] > 0 else 0.0 for k in strata}
        elif allocation == "proportional":
            weights = {k: float(caps[k]) for k in strata}
        else:
            raise ValueError(f"Unknown allocation={allocation}")

        alloc = largest_remainder_allocation(weights, total_n)
        alloc = rebalance_with_caps(alloc, caps)
        alloc = apply_min_floor(alloc, caps, min_per_nonempty_stratum)
        alloc = rebalance_with_caps(alloc, caps)

        current = sum(alloc.values())
        remainder = total_n - current
        if remainder > 0:
            room = {k: caps[k] - alloc[k] for k in alloc if caps[k] - alloc[k] > 0}
            add = largest_remainder_allocation({k: float(v) for k, v in room.items()}, remainder)
            for k, v in add.items():
                alloc[k] += v

        return alloc


# =============================================================================
# Sampling
# =============================================================================

def sample_union_items(items: List[UnionItem], alloc: Dict[Tuple[str, str], int], seed: int) -> List[UnionItem]:
    rng = random.Random(seed)
    by_stratum: Dict[Tuple[str, str], List[UnionItem]] = defaultdict(list)

    for it in items:
        by_stratum[(it.mode, it.chosen_bucket)].append(it)

    sampled: List[UnionItem] = []
    for key, n in alloc.items():
        bucket_items = list(by_stratum.get(key, []))
        if n <= 0 or not bucket_items:
            continue
        if n >= len(bucket_items):
            chosen = bucket_items
        else:
            chosen = rng.sample(bucket_items, n)
        sampled.extend(chosen)

    sampled.sort(key=lambda x: (x.mode, x.chosen_bucket, x.patient_id, x.trial_id))
    return sampled


def sample_pooled_rows(items: List[PooledRow], alloc: Dict[Tuple[str, str], int], seed: int) -> List[PooledRow]:
    rng = random.Random(seed)
    by_stratum: Dict[Tuple[str, str], List[PooledRow]] = defaultdict(list)

    for it in items:
        by_stratum[(it.mode, it.bucket)].append(it)

    sampled: List[PooledRow] = []
    for key, n in alloc.items():
        bucket_items = list(by_stratum.get(key, []))
        if n <= 0 or not bucket_items:
            continue
        if n >= len(bucket_items):
            chosen = bucket_items
        else:
            chosen = rng.sample(bucket_items, n)
        sampled.extend(chosen)

    sampled.sort(key=lambda x: (x.mode, x.bucket, x.source_system, x.patient_id, x.trial_id))
    return sampled


# =============================================================================
# Output rows
# =============================================================================

def build_population_summary_rows_from_counts(
    counts: Dict[Tuple[str, str], int],
    modes: List[str],
) -> List[dict]:
    rows = []
    grand_total = sum(counts.values())

    for m in modes:
        mode_total = sum(counts[(m, b)] for b in BUCKETS)
        for b in BUCKETS:
            n = counts[(m, b)]
            rows.append({
                "mode": m,
                "bucket": b,
                "population_n": n,
                "mode_total": mode_total,
                "mode_bucket_frac": f"{(n / mode_total):.6f}" if mode_total > 0 else "",
                "global_total": grand_total,
                "global_bucket_frac": f"{(n / grand_total):.6f}" if grand_total > 0 else "",
            })
    return rows


def build_allocation_summary_rows_from_counts(
    counts: Dict[Tuple[str, str], int],
    alloc: Dict[Tuple[str, str], int],
    modes: List[str],
) -> List[dict]:
    rows = []
    total_pop = sum(counts.values())
    total_samp = sum(alloc.values())

    for m in modes:
        mode_pop = sum(counts[(m, b)] for b in BUCKETS)
        mode_samp = sum(alloc[(m, b)] for b in BUCKETS)
        for b in BUCKETS:
            pop_n = counts[(m, b)]
            samp_n = alloc[(m, b)]
            rows.append({
                "mode": m,
                "bucket": b,
                "population_n": pop_n,
                "sample_n": samp_n,
                "sampling_rate_within_stratum": f"{(samp_n / pop_n):.6f}" if pop_n > 0 else "",
                "mode_population_n": mode_pop,
                "mode_sample_n": mode_samp,
                "global_population_n": total_pop,
                "global_sample_n": total_samp,
            })
    return rows


def build_sample_rows_union(sampled: List[UnionItem]) -> List[dict]:
    rows: List[dict] = []
    for idx, it in enumerate(sampled, start=1):
        rows.append({
            "sample_id": idx,
            "mode": it.mode,
            "bucket": it.chosen_bucket,
            "patient_id": it.patient_id,
            "trial_id": it.trial_id,

            "source_bucket": it.source_bucket,
            "in_smt": it.in_smt,
            "in_trialgpt": it.in_trialgpt,

            "union_bucket_used_for_sampling": it.chosen_bucket,

            "smt_bucket": it.smt_bucket,
            "trialgpt_bucket": it.trialgpt_bucket,

            "smt_label": it.smt_label,
            "trialgpt_label": it.trialgpt_label,

            "smt_rank": it.smt_rank,
            "trialgpt_rank": it.trialgpt_rank,

            "smt_any_subcohort_relevant": it.smt_any_subcohort_relevant,
            "smt_any_subcohort_eligible": it.smt_any_subcohort_eligible,
            "smt_any_subcohort_relevant_and_eligible": it.smt_any_subcohort_relevant_and_eligible,

            "trialgpt_any_subcohort_relevant": it.trialgpt_any_subcohort_relevant,
            "trialgpt_any_subcohort_eligible": it.trialgpt_any_subcohort_eligible,
            "trialgpt_any_subcohort_relevant_and_eligible": it.trialgpt_any_subcohort_relevant_and_eligible,

            "smt_num_subcohorts": it.smt_num_subcohorts,
            "trialgpt_num_subcohorts": it.trialgpt_num_subcohorts,

            "physician_accepts_llm_judgment": "",
            "physician_relevance": "",
            "physician_eligibility": "",
            "physician_confidence": "",
            "physician_notes": "",
        })
    return rows


def build_sample_rows_pooled(sampled: List[PooledRow]) -> List[dict]:
    rows: List[dict] = []
    for idx, it in enumerate(sampled, start=1):
        rows.append({
            "sample_id": idx,
            "mode": it.mode,
            "bucket": it.bucket,
            "patient_id": it.patient_id,
            "trial_id": it.trial_id,
            "source_system": it.source_system,

            "llm_label_bucket": it.bucket,
            "system_label": it.label,
            "rank": it.rank,
            "llm_any_subcohort_relevant": it.any_subcohort_relevant,
            "llm_any_subcohort_eligible": it.any_subcohort_eligible,
            "llm_any_subcohort_relevant_and_eligible": it.any_subcohort_relevant_and_eligible,
            "num_subcohorts": it.num_subcohorts,

            "physician_accepts_llm_judgment": "",
            "physician_relevance": "",
            "physician_eligibility": "",
            "physician_confidence": "",
            "physician_notes": "",
        })
    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--smt-root", default="./smt_retrieval_eval_out")
    ap.add_argument("--trialgpt-root", default="./trialgpt_retrieval_eval_out")
    ap.add_argument("--output-dir", default="./physician_validation_union_sample")

    ap.add_argument("--modes", default="ccr,all,all-explore")
    ap.add_argument("--total-n", type=int, default=27)

    ap.add_argument(
        "--allocation",
        choices=["proportional", "equal", "balanced"],
        default="balanced",
        help=(
            "Allocation across strata or within mode if --per-mode-equal-total is set. "
            "'balanced' is an alias for 'equal'."
        ),
    )

    ap.add_argument(
        "--per-mode-equal-total",
        dest="per_mode_equal_total",
        action="store_true",
        help="Give each nonempty mode roughly equal total sample size.",
    )
    ap.add_argument(
        "--no-per-mode-equal-total",
        dest="per_mode_equal_total",
        action="store_false",
        help="Do not force equal total sample size across modes.",
    )
    ap.set_defaults(per_mode_equal_total=True)

    ap.add_argument(
        "--min-per-nonempty-stratum",
        type=int,
        default=0,
        help="Optional minimum per nonempty (mode,bucket) stratum.",
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--union-label-policy",
        choices=["prefer_smt", "prefer_trialgpt", "require_match", "pooled_system_rows"],
        default="prefer_smt",
        help=(
            "How to define the judgment label for union sampling. "
            "'pooled_system_rows' samples system-specific rows instead of union-collapsed rows."
        ),
    )

    args = ap.parse_args()

    smt_root = Path(args.smt_root).resolve()
    trialgpt_root = Path(args.trialgpt_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = parse_modes(args.modes)

    smt_items = load_system_items(smt_root, "smt", modes)
    trialgpt_items = load_system_items(trialgpt_root, "trialgpt", modes)

    if args.union_label_policy == "pooled_system_rows":
        pooled_rows = build_pooled_rows(smt_items, trialgpt_items)
        if not pooled_rows:
            raise RuntimeError("No pooled system rows found.")

        counts = counts_from_pooled_rows(pooled_rows, modes)
        alloc = allocate_samples_generic(
            counts=counts,
            modes=modes,
            total_n=args.total_n,
            allocation=args.allocation,
            per_mode_equal_total=bool(args.per_mode_equal_total),
            min_per_nonempty_stratum=int(args.min_per_nonempty_stratum),
        )
        sampled = sample_pooled_rows(pooled_rows, alloc, seed=int(args.seed))

        population_rows = build_population_summary_rows_from_counts(counts, modes)
        allocation_rows = build_allocation_summary_rows_from_counts(counts, alloc, modes)
        sample_rows = build_sample_rows_pooled(sampled)

    else:
        union_items = build_union_items(
            smt_items=smt_items,
            trialgpt_items=trialgpt_items,
            union_label_policy=args.union_label_policy,
        )
        if not union_items:
            raise RuntimeError("No union items found after applying union-label policy.")

        counts = counts_from_union_items(union_items, modes)
        alloc = allocate_samples_generic(
            counts=counts,
            modes=modes,
            total_n=args.total_n,
            allocation=args.allocation,
            per_mode_equal_total=bool(args.per_mode_equal_total),
            min_per_nonempty_stratum=int(args.min_per_nonempty_stratum),
        )
        sampled = sample_union_items(union_items, alloc, seed=int(args.seed))

        population_rows = build_population_summary_rows_from_counts(counts, modes)
        allocation_rows = build_allocation_summary_rows_from_counts(counts, alloc, modes)
        sample_rows = build_sample_rows_union(sampled)

    write_csv(
        output_dir / "population_summary.csv",
        population_rows,
        fieldnames=[
            "mode", "bucket", "population_n", "mode_total", "mode_bucket_frac",
            "global_total", "global_bucket_frac",
        ],
    )

    write_csv(
        output_dir / "allocation_summary.csv",
        allocation_rows,
        fieldnames=[
            "mode", "bucket", "population_n", "sample_n", "sampling_rate_within_stratum",
            "mode_population_n", "mode_sample_n", "global_population_n", "global_sample_n",
        ],
    )

    sample_fieldnames = list(sample_rows[0].keys()) if sample_rows else []
    write_csv(output_dir / "sampled_items.csv", sample_rows, fieldnames=sample_fieldnames)
    write_jsonl(output_dir / "sampled_items.jsonl", sample_rows)

    manifest = {
        "smt_root": str(smt_root),
        "trialgpt_root": str(trialgpt_root),
        "output_dir": str(output_dir),
        "modes": modes,
        "total_n_requested": int(args.total_n),
        "total_n_sampled": len(sample_rows),
        "allocation": args.allocation,
        "allocation_normalized": _normalize_allocation_name(str(args.allocation)),
        "per_mode_equal_total": bool(args.per_mode_equal_total),
        "min_per_nonempty_stratum": int(args.min_per_nonempty_stratum),
        "seed": int(args.seed),
        "union_label_policy": args.union_label_policy,
        "n_smt_rows_loaded": len(smt_items),
        "n_trialgpt_rows_loaded": len(trialgpt_items),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] Wrote:")
    print(f"  {output_dir / 'sampled_items.csv'}")
    print(f"  {output_dir / 'sampled_items.jsonl'}")
    print(f"  {output_dir / 'allocation_summary.csv'}")
    print(f"  {output_dir / 'population_summary.csv'}")
    print(f"  {output_dir / 'manifest.json'}")
    print(f"[INFO] total sampled rows: {len(sample_rows)}")


if __name__ == "__main__":
    main()