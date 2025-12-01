#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_smt_vs_trialgpt_avgcutoff_qrels_annotated_precision.py

Variant-aware comparison of:
  - SMT: selected set = label == all_satisfied
  - Baseline: selected set = top-K, where K = average SMT all_satisfied count

This script supports multiple baseline variants under a single root, e.g.

  trialgpt_retrieval_eval_out/
    <baseline_name>/
      ccr/patient_labels/*.json
      all/patient_labels/*.json
      all-explore/patient_labels/*.json

and also legacy layouts like:

  trialgpt_retrieval_eval_out/
    ccr/patient_labels/*.json
    all/patient_labels/*.json

In the legacy case, variant is recorded as "default".

IMPORTANT
---------
This script does NOT use any judge outputs.

For each patient p:
  F_p = fetched/selected trial set for the system
  Q_p = qrels-labeled trials for that patient with score in {0,1,2}
  A_p = F_p ∩ Q_p

Metrics:
1) Annotated Precision (binary 1/2-positive):
     numerator   = # pairs in A_p with qrels score in {1,2}
     denominator = # pairs in A_p

2) Annotated Weighted Precision (0/1/2):
     numerator   = sum of qrels scores over pairs in A_p
     denominator = # pairs in A_p

3) Annotated Weighted Precision Normalized:
     numerator   = sum of qrels scores over pairs in A_p
     denominator = 2 * # pairs in A_p
   so the metric lies in [0,1].

Outputs
-------
  <out>/per_patient.csv
  <out>/summary_by_variant_mode.csv
  <out>/distributions_by_variant_mode.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple


DEFAULT_TRIALGPT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out")
DEFAULT_SMT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")
DEFAULT_TRIAL_CORPUS = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus_real.jsonl")
DEFAULT_QRELS = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/qrels/test.tsv")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)

KNOWN_MODES = ("chief", "ccr", "all", "all-explore")
MODE_ALIASES = {
    "chief": "chief",
    "ccr": "ccr",
    "all": "all",
    "all-explore": "all-explore",
    "all_explore": "all-explore",
}


# ----------------------------
# Canonical ID helpers
# ----------------------------

def canon_nct(nct: Optional[str], collapse_subsuffix: bool = True) -> Optional[str]:
    if not nct:
        return None
    s = str(nct).strip()
    if not s:
        return None
    m = NCT_BASE_RE.match(s)
    if not m:
        return s.upper()
    base = m.group(1).upper()
    return base if collapse_subsuffix else s.upper()


def infer_patient_id_from_filename(p: Path) -> Optional[str]:
    name = p.name
    if "__" in name:
        return name.split("__", 1)[0]
    stem = p.stem
    return stem.split("__", 1)[0] if "__" in stem else stem


def infer_mode_and_variant_from_path(p: Path, root: Path) -> Tuple[str, str]:
    try:
        rel = p.relative_to(root)
        parts = list(rel.parts)
    except Exception:
        parts = list(p.parts)

    if len(parts) >= 2 and parts[-2] == "patient_labels":
        prefix = parts[:-2]
    else:
        prefix = parts[:-1]

    mode = "unknown"
    mode_idx = None

    # Search from right to left so the nearest mode directory wins.
    for i in range(len(prefix) - 1, -1, -1):
        norm = MODE_ALIASES.get(prefix[i].lower())
        if norm is not None:
            mode = norm
            mode_idx = i
            break

    if mode_idx is None:
        variant = prefix[0] if prefix else "default"
    else:
        variant_parts = prefix[:mode_idx]
        variant = "/".join(variant_parts) if variant_parts else "default"

    return mode, variant


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_allowed_trial_ids_from_corpus(
    corpus_path: Path,
    collapse_subsuffix: bool,
) -> Set[str]:
    allowed: Set[str] = set()
    bad = 0

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            raw_id = obj.get("_id")
            if raw_id is None:
                continue

            key = canon_nct(str(raw_id), collapse_subsuffix=collapse_subsuffix)
            if key:
                allowed.add(key)

    if not allowed:
        raise ValueError(f"No valid trial IDs loaded from corpus: {corpus_path}")

    if bad:
        print(f"[warn] skipped {bad} malformed JSONL rows in corpus: {corpus_path}", file=sys.stderr)

    return allowed


# ----------------------------
# Qrels loader
# ----------------------------

def load_qrels_scores(
    qrels_path: Path,
    collapse_subsuffix: bool,
    allowed_trial_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Returns:
      patient_id -> {trial_id -> qrels_score}
    keeping scores 0,1,2.
    If duplicates occur, keep the max score.
    """
    out: Dict[str, Dict[str, int]] = defaultdict(dict)

    with qrels_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        parts = first_line.strip().split("\t")
        has_header = (
            len(parts) >= 3
            and (parts[0].lower().startswith("query") or parts[-1].lower() == "score")
        )

        def process_line(line: str) -> None:
            line = line.strip()
            if not line:
                return

            cols = line.split("\t")
            if len(cols) < 3:
                return

            query_id, corpus_id, score_str = cols[0], cols[1], cols[2]

            try:
                score = int(score_str)
            except ValueError:
                return

            if score not in (0, 1, 2):
                return

            key = canon_nct(corpus_id, collapse_subsuffix=collapse_subsuffix)
            if not key:
                return
            if allowed_trial_ids is not None and key not in allowed_trial_ids:
                return

            prev = out[query_id].get(key)
            if prev is None or score > prev:
                out[query_id][key] = score

        if not has_header:
            process_line(first_line)

        for line in f:
            process_line(line)

    return out


# ----------------------------
# Run parsing
# ----------------------------

def extract_key(tr: Dict[str, Any], collapse_subsuffix: bool) -> Optional[str]:
    for k in ("canonical_nct_id", "nct_id", "trial_nct_id", "nct"):
        if tr.get(k):
            return canon_nct(str(tr[k]), collapse_subsuffix=collapse_subsuffix) or str(tr[k]).upper()
    if tr.get("trial_id") is not None:
        tid = str(tr["trial_id"])
        return canon_nct(tid, collapse_subsuffix=collapse_subsuffix) or tid.upper()
    return None


def extract_ranked_list(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj.get("trials"), list) and obj["trials"]:
        return [x for x in obj["trials"] if isinstance(x, dict)]
    if isinstance(obj.get("canonical_trials"), list) and obj["canonical_trials"]:
        return [x for x in obj["canonical_trials"] if isinstance(x, dict)]
    if isinstance(obj.get("ranked"), list) and obj["ranked"]:
        return [x for x in obj["ranked"] if isinstance(x, dict)]
    return None


def norm_label(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    return t if t else None


@dataclass
class PatientRun:
    patient_id: str
    mode: str
    variant: str
    ranked_keys: List[str]
    label_by_key: Dict[str, Optional[str]]
    source_file: Path


def build_run(
    p: Path,
    collapse_subsuffix: bool,
    dedup: bool,
    root: Path,
    default_variant: str = "default",
) -> Optional[PatientRun]:
    obj = load_json(p)
    if not isinstance(obj, dict):
        return None

    patient_id = obj.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.strip():
        patient_id = infer_patient_id_from_filename(p)
    if not patient_id:
        return None

    mode_from_obj = obj.get("mode") if isinstance(obj.get("mode"), str) else None
    mode_from_path, variant_from_path = infer_mode_and_variant_from_path(p, root)

    norm_mode_from_path = MODE_ALIASES.get(str(mode_from_path).lower(), str(mode_from_path).lower())
    norm_mode_from_obj = (
        MODE_ALIASES.get(str(mode_from_obj).lower(), str(mode_from_obj).lower())
        if mode_from_obj is not None else None
    )

    # Trust directory layout over embedded JSON mode.
    mode = norm_mode_from_path if norm_mode_from_path != "unknown" else (norm_mode_from_obj or "unknown")
    variant = variant_from_path or default_variant

    if norm_mode_from_obj is not None and norm_mode_from_path != "unknown" and norm_mode_from_obj != norm_mode_from_path:
        print(
            f"[warn] mode mismatch; trusting path: file={p} json_mode={norm_mode_from_obj} path_mode={norm_mode_from_path}",
            file=sys.stderr,
        )

        
    trials = extract_ranked_list(obj)
    if not trials:
        return None

    if all(("rank" in t and isinstance(t.get("rank"), (int, float))) for t in trials):
        trials = sorted(trials, key=lambda t: int(t.get("rank", 10**9)))

    ranked_keys: List[str] = []
    label_by_key: Dict[str, Optional[str]] = {}
    seen: Set[str] = set()

    for t in trials:
        key = extract_key(t, collapse_subsuffix=collapse_subsuffix)
        if not key:
            continue
        if dedup and key in seen:
            continue
        seen.add(key)

        ranked_keys.append(key)
        label_by_key[key] = norm_label(t.get("label"))

    return PatientRun(
        patient_id=patient_id,
        mode=mode,
        variant=variant,
        ranked_keys=ranked_keys,
        label_by_key=label_by_key,
        source_file=p,
    ) if ranked_keys else None


def collect_runs(
    root: Path,
    collapse_subsuffix: bool,
    dedup: bool,
    default_variant: str = "default",
) -> Dict[Tuple[str, str, str], PatientRun]:
    out: Dict[Tuple[str, str, str], PatientRun] = {}
    for p in root.rglob("patient_labels/*.json"):
        run = build_run(
            p,
            collapse_subsuffix=collapse_subsuffix,
            dedup=dedup,
            root=root,
            default_variant=default_variant,
        )
        if not run:
            continue
        k = (run.variant, run.patient_id, run.mode)
        prev = out.get(k)
        if prev is None or len(run.ranked_keys) > len(prev.ranked_keys):
            out[k] = run
    return out


def print_run_inventory(label: str, runs: Dict[Tuple[str, str, str], PatientRun]) -> None:
    by_variant_mode: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    for (variant, _patient_id, mode) in runs.keys():
        by_variant_mode[(variant, mode)] += 1

    print(f"[info] {label} run inventory:")
    if not by_variant_mode:
        print("  [info] none")
        return

    for (variant, mode), n in sorted(by_variant_mode.items()):
        print(f"  [info] variant={variant} mode={mode} patients={n}")


# ----------------------------
# Metrics
# ----------------------------

def compute_annotated_metrics(
    fetched_set: Set[str],
    qrels_scores: Dict[str, int],
) -> Dict[str, Any]:
    """
    fetched_set: selected/fetched trials for one patient
    qrels_scores: {trial_id -> qrels_score} for one patient, scores in {0,1,2}

    A = fetched_set ∩ qrels-covered trials

    binary_annotated_precision:
      (# with qrels score in {1,2}) / |A|

    weighted_annotated_precision:
      (sum qrels scores over A) / |A|      range [0,2]

    weighted_annotated_precision_norm:
      (sum qrels scores over A) / (2|A|)   range [0,1]
    """
    qrels_covered = set(qrels_scores.keys())
    annotated = fetched_set & qrels_covered
    denom = len(annotated)

    num_pos12 = sum(1 for tid in annotated if qrels_scores.get(tid, 0) in (1, 2))
    num_score_sum = sum(qrels_scores.get(tid, 0) for tid in annotated)

    binary_prec = (num_pos12 / denom) if denom > 0 else None
    weighted_prec = (num_score_sum / denom) if denom > 0 else None
    weighted_prec_norm = (num_score_sum / (2 * denom)) if denom > 0 else None

    num_qrels0 = sum(1 for tid in annotated if qrels_scores.get(tid, 0) == 0)
    num_qrels1 = sum(1 for tid in annotated if qrels_scores.get(tid, 0) == 1)
    num_qrels2 = sum(1 for tid in annotated if qrels_scores.get(tid, 0) == 2)

    return {
        "annotated_overlap": denom,
        "annotated_qrels0": num_qrels0,
        "annotated_qrels1": num_qrels1,
        "annotated_qrels2": num_qrels2,
        "annotated_num_pos12": num_pos12,
        "annotated_score_sum": num_score_sum,
        "annotated_precision_12": binary_prec,
        "annotated_weighted_precision_012": weighted_prec,
        "annotated_weighted_precision_012_norm": weighted_prec_norm,
    }


# ----------------------------
# Utility
# ----------------------------

def hist_counts(xs: List[int], max_bin: Optional[int] = None) -> Dict[str, int]:
    if not xs:
        return {}
    mx = max(xs)
    if max_bin is None:
        max_bin = mx
    out: Dict[str, int] = {}
    for v in xs:
        if v >= max_bin:
            out[f">={max_bin}"] = out.get(f">={max_bin}", 0) + 1
        else:
            out[str(v)] = out.get(str(v), 0) + 1
    return out


def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _parse_modes_arg(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        norm = MODE_ALIASES.get(t)
        if norm is None:
            raise ValueError(f"Unknown mode in --modes: {t} (known: {', '.join(KNOWN_MODES)})")
        out.append(norm)
    return out or None


def _parse_variants_arg(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    out = [tok.strip() for tok in s.split(",") if tok.strip()]
    return out or None


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare SMT(all_satisfied) vs baseline(top-K avg cutoff) using qrels-covered annotated precision."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--trial-corpus", type=Path, default=DEFAULT_TRIAL_CORPUS)
    ap.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    ap.add_argument("--out", type=Path, default=Path("./avgcutoff_compare_qrels_annotated_precision_out"))

    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")

    ap.add_argument("--avg-cutoff-scope", choices=("per_mode", "global"), default="per_mode")
    ap.add_argument("--avg-cutoff-rounding", choices=("round", "floor", "ceil"), default="round")
    ap.add_argument("--max-hist-bin", type=int, default=60)

    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Optional comma-separated modes filter (subset of chief,ccr,all,all-explore).",
    )
    ap.add_argument(
        "--variants",
        type=str,
        default="",
        help="Optional comma-separated baseline variants to include.",
    )

    args = ap.parse_args()

    if not args.trialgpt_root.exists():
        print(f"[error] trialgpt root not found: {args.trialgpt_root}", file=sys.stderr)
        sys.exit(2)
    if not args.smt_root.exists():
        print(f"[error] smt root not found: {args.smt_root}", file=sys.stderr)
        sys.exit(2)
    if not args.trial_corpus.exists():
        print(f"[error] trial corpus not found: {args.trial_corpus}", file=sys.stderr)
        sys.exit(2)
    if not args.qrels.exists():
        print(f"[error] qrels not found: {args.qrels}", file=sys.stderr)
        sys.exit(2)

    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)
    variant_filter = _parse_variants_arg(args.variants)

    allowed_trial_ids = load_allowed_trial_ids_from_corpus(
        args.trial_corpus,
        collapse_subsuffix=collapse,
    )
    print(f"[info] loaded {len(allowed_trial_ids)} allowed trial IDs from corpus")

    qrels_scores = load_qrels_scores(
        args.qrels,
        collapse_subsuffix=collapse,
        allowed_trial_ids=allowed_trial_ids,
    )
    print(f"[info] loaded qrels scores 0/1/2 for {len(qrels_scores)} patients")

    baseline_runs = collect_runs(
        args.trialgpt_root,
        collapse_subsuffix=collapse,
        dedup=args.dedup_canonical,
        default_variant="default",
    )
    smt_runs_raw = collect_runs(
        args.smt_root,
        collapse_subsuffix=collapse,
        dedup=args.dedup_canonical,
        default_variant="smt",
    )

    print_run_inventory("baseline", baseline_runs)
    print_run_inventory("smt_raw", smt_runs_raw)

    smt_runs: Dict[Tuple[str, str], PatientRun] = {}
    for (_variant, patient_id, mode), run in smt_runs_raw.items():
        k = (patient_id, mode)
        prev = smt_runs.get(k)
        if prev is None or len(run.ranked_keys) > len(prev.ranked_keys):
            smt_runs[k] = run

    if mode_filter is not None:
        baseline_runs = {k: v for k, v in baseline_runs.items() if k[2] in mode_filter}
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}

    if variant_filter is not None:
        variant_filter_set = set(variant_filter)
        baseline_runs = {k: v for k, v in baseline_runs.items() if k[0] in variant_filter_set}

    if not baseline_runs:
        print("[error] no baseline runs found after filtering", file=sys.stderr)
        sys.exit(2)
    if not smt_runs:
        print("[error] no SMT runs found after filtering", file=sys.stderr)
        sys.exit(2)

    # Average SMT all_satisfied cutoff K, shared across variants.
    smt_m_counts_by_mode: Dict[str, List[int]] = {}
    smt_m_counts_global: List[int] = []

    smt_keys_for_cutoff: Set[Tuple[str, str]] = set()
    for (_variant, patient_id, mode) in baseline_runs.keys():
        if (patient_id, mode) in smt_runs:
            smt_keys_for_cutoff.add((patient_id, mode))

    if not smt_keys_for_cutoff and args.require_overlap:
        print("[error] no overlapping SMT/baseline patient-mode pairs found", file=sys.stderr)
        sys.exit(2)

    cutoff_source_keys = smt_keys_for_cutoff if args.require_overlap else set(smt_runs.keys())

    for (patient_id, mode) in cutoff_source_keys:
        smt = smt_runs.get((patient_id, mode))
        if smt is None:
            continue
        m = sum(
            1
            for k in smt.ranked_keys
            if smt.label_by_key.get(k) == "all_satisfied" and k in allowed_trial_ids
        )
        smt_m_counts_by_mode.setdefault(mode, []).append(m)
        smt_m_counts_global.append(m)

    def round_k(x: float) -> int:
        if args.avg_cutoff_rounding == "floor":
            return int(math.floor(x))
        if args.avg_cutoff_rounding == "ceil":
            return int(math.ceil(x))
        return int(round(x))

    avg_k_by_mode: Dict[str, int] = {}
    if args.avg_cutoff_scope == "global":
        if not smt_m_counts_global:
            print("[error] no SMT runs found to compute average cutoff.", file=sys.stderr)
            sys.exit(2)
        k = round_k(statistics.mean(smt_m_counts_global))
        modes_present = {mode for _, _, mode in baseline_runs.keys()}
        for mode in modes_present:
            avg_k_by_mode[mode] = k
    else:
        for mode, xs in smt_m_counts_by_mode.items():
            avg_k_by_mode[mode] = round_k(statistics.mean(xs)) if xs else 0

    if not avg_k_by_mode:
        print("[error] failed to compute any average cutoff K", file=sys.stderr)
        sys.exit(2)

    baseline_keys = set(baseline_runs.keys())
    comparable_keys = {
        (variant, patient_id, mode)
        for (variant, patient_id, mode) in baseline_keys
        if (patient_id, mode) in smt_runs
    }

    keys_all = sorted(comparable_keys)

    if not keys_all:
        print("[error] no comparable baseline/SMT keys found", file=sys.stderr)
        sys.exit(2)

    comparable_counts: DefaultDict[str, int] = defaultdict(int)
    for (_variant, _patient_id, mode) in keys_all:
        comparable_counts[mode] += 1
    for mode, n in sorted(comparable_counts.items()):
        print(f"[info] comparable pairs mode={mode} count={n}")

    per_patient: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    dist: Dict[str, Any] = {}
    by_variant_mode_rows: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for (variant, patient_id, mode) in keys_all:
        smt = smt_runs.get((patient_id, mode))
        baseline = baseline_runs.get((variant, patient_id, mode))
        if smt is None or baseline is None:
            continue

        patient_qrels = qrels_scores.get(patient_id, {})

        smt_selected: Set[str] = {
            k for k in smt.ranked_keys
            if smt.label_by_key.get(k) == "all_satisfied" and k in allowed_trial_ids
        }

        k_avg = max(0, int(avg_k_by_mode.get(mode, 0)))
        baseline_selected: Set[str] = set()
        if k_avg > 0:
            baseline_ranked_in_corpus = [k for k in baseline.ranked_keys if k in allowed_trial_ids]
            baseline_selected = set(baseline_ranked_in_corpus[:k_avg])

        smt_metrics = compute_annotated_metrics(smt_selected, patient_qrels)
        baseline_metrics = compute_annotated_metrics(baseline_selected, patient_qrels)

        overlap = smt_selected & baseline_selected
        smt_not_baseline = smt_selected - baseline_selected
        baseline_not_smt = baseline_selected - smt_selected

        row = {
            "variant": variant,
            "patient_id": patient_id,
            "mode": mode,
            "k_avg": k_avg,

            "num_qrels_covered_for_patient": len(patient_qrels),
            "num_qrels0_for_patient": sum(1 for s in patient_qrels.values() if s == 0),
            "num_qrels1_for_patient": sum(1 for s in patient_qrels.values() if s == 1),
            "num_qrels2_for_patient": sum(1 for s in patient_qrels.values() if s == 2),

            "smt_selected": len(smt_selected),
            "baseline_selected": len(baseline_selected),

            "smt_annotated_overlap": smt_metrics["annotated_overlap"],
            "baseline_annotated_overlap": baseline_metrics["annotated_overlap"],

            "smt_annotated_qrels0": smt_metrics["annotated_qrels0"],
            "smt_annotated_qrels1": smt_metrics["annotated_qrels1"],
            "smt_annotated_qrels2": smt_metrics["annotated_qrels2"],
            "baseline_annotated_qrels0": baseline_metrics["annotated_qrels0"],
            "baseline_annotated_qrels1": baseline_metrics["annotated_qrels1"],
            "baseline_annotated_qrels2": baseline_metrics["annotated_qrels2"],

            "smt_annotated_num_pos12": smt_metrics["annotated_num_pos12"],
            "baseline_annotated_num_pos12": baseline_metrics["annotated_num_pos12"],

            "smt_annotated_score_sum": smt_metrics["annotated_score_sum"],
            "baseline_annotated_score_sum": baseline_metrics["annotated_score_sum"],

            "smt_annotated_precision_12": (
                f"{smt_metrics['annotated_precision_12']:.6f}"
                if smt_metrics["annotated_precision_12"] is not None else ""
            ),
            "baseline_annotated_precision_12": (
                f"{baseline_metrics['annotated_precision_12']:.6f}"
                if baseline_metrics["annotated_precision_12"] is not None else ""
            ),

            "smt_annotated_weighted_precision_012": (
                f"{smt_metrics['annotated_weighted_precision_012']:.6f}"
                if smt_metrics["annotated_weighted_precision_012"] is not None else ""
            ),
            "baseline_annotated_weighted_precision_012": (
                f"{baseline_metrics['annotated_weighted_precision_012']:.6f}"
                if baseline_metrics["annotated_weighted_precision_012"] is not None else ""
            ),

            "smt_annotated_weighted_precision_012_norm": (
                f"{smt_metrics['annotated_weighted_precision_012_norm']:.6f}"
                if smt_metrics["annotated_weighted_precision_012_norm"] is not None else ""
            ),
            "baseline_annotated_weighted_precision_012_norm": (
                f"{baseline_metrics['annotated_weighted_precision_012_norm']:.6f}"
                if baseline_metrics["annotated_weighted_precision_012_norm"] is not None else ""
            ),

            "overlap": len(overlap),
            "smt_not_baseline": len(smt_not_baseline),
            "baseline_not_smt": len(baseline_not_smt),
            "jaccard_selected": (len(overlap) / len(smt_selected | baseline_selected)) if (smt_selected or baseline_selected) else "",

            "smt_source": str(smt.source_file),
            "baseline_source": str(baseline.source_file),
        }
        per_patient.append(row)
        by_variant_mode_rows[(variant, mode)].append(row)

    for (variant, mode), rows in sorted(by_variant_mode_rows.items()):
        def _vals(col: str) -> List[float]:
            return [float(r[col]) for r in rows if r[col] != ""]

        def _sumint(col: str) -> int:
            return sum(int(r[col]) for r in rows)

        summary_rows.append({
            "variant": variant,
            "mode": mode,
            "patients": len(rows),
            "k_avg": rows[0]["k_avg"] if rows else 0,

            "avg_num_qrels_covered_for_patient": (
                f"{statistics.mean([int(r['num_qrels_covered_for_patient']) for r in rows]):.6f}" if rows else ""
            ),

            "avg_smt_selected": f"{statistics.mean([int(r['smt_selected']) for r in rows]):.6f}" if rows else "",
            "avg_baseline_selected": f"{statistics.mean([int(r['baseline_selected']) for r in rows]):.6f}" if rows else "",

            "avg_smt_annotated_overlap": f"{statistics.mean([int(r['smt_annotated_overlap']) for r in rows]):.6f}" if rows else "",
            "avg_baseline_annotated_overlap": f"{statistics.mean([int(r['baseline_annotated_overlap']) for r in rows]):.6f}" if rows else "",

            "macro_smt_annotated_precision_12": (
                f"{statistics.mean(_vals('smt_annotated_precision_12')):.6f}" if _vals("smt_annotated_precision_12") else ""
            ),
            "macro_baseline_annotated_precision_12": (
                f"{statistics.mean(_vals('baseline_annotated_precision_12')):.6f}" if _vals("baseline_annotated_precision_12") else ""
            ),
            "micro_smt_annotated_precision_12": (
                f"{(_sumint('smt_annotated_num_pos12') / _sumint('smt_annotated_overlap')):.6f}"
                if _sumint("smt_annotated_overlap") > 0 else ""
            ),
            "micro_baseline_annotated_precision_12": (
                f"{(_sumint('baseline_annotated_num_pos12') / _sumint('baseline_annotated_overlap')):.6f}"
                if _sumint("baseline_annotated_overlap") > 0 else ""
            ),

            "macro_smt_annotated_weighted_precision_012": (
                f"{statistics.mean(_vals('smt_annotated_weighted_precision_012')):.6f}"
                if _vals("smt_annotated_weighted_precision_012") else ""
            ),
            "macro_baseline_annotated_weighted_precision_012": (
                f"{statistics.mean(_vals('baseline_annotated_weighted_precision_012')):.6f}"
                if _vals("baseline_annotated_weighted_precision_012") else ""
            ),
            "micro_smt_annotated_weighted_precision_012": (
                f"{(_sumint('smt_annotated_score_sum') / _sumint('smt_annotated_overlap')):.6f}"
                if _sumint("smt_annotated_overlap") > 0 else ""
            ),
            "micro_baseline_annotated_weighted_precision_012": (
                f"{(_sumint('baseline_annotated_score_sum') / _sumint('baseline_annotated_overlap')):.6f}"
                if _sumint("baseline_annotated_overlap") > 0 else ""
            ),

            "macro_smt_annotated_weighted_precision_012_norm": (
                f"{statistics.mean(_vals('smt_annotated_weighted_precision_012_norm')):.6f}"
                if _vals("smt_annotated_weighted_precision_012_norm") else ""
            ),
            "macro_baseline_annotated_weighted_precision_012_norm": (
                f"{statistics.mean(_vals('baseline_annotated_weighted_precision_012_norm')):.6f}"
                if _vals("baseline_annotated_weighted_precision_012_norm") else ""
            ),
            "micro_smt_annotated_weighted_precision_012_norm": (
                f"{(_sumint('smt_annotated_score_sum') / (2 * _sumint('smt_annotated_overlap'))):.6f}"
                if _sumint("smt_annotated_overlap") > 0 else ""
            ),
            "micro_baseline_annotated_weighted_precision_012_norm": (
                f"{(_sumint('baseline_annotated_score_sum') / (2 * _sumint('baseline_annotated_overlap'))):.6f}"
                if _sumint("baseline_annotated_overlap") > 0 else ""
            ),
        })

        dist_key = f"{variant}::{mode}"
        dist[dist_key] = {
            "variant": variant,
            "mode": mode,
            "k_avg": rows[0]["k_avg"] if rows else 0,
            "num_qrels_covered_for_patient_hist": hist_counts(
                [int(r["num_qrels_covered_for_patient"]) for r in rows],
                max_bin=args.max_hist_bin,
            ),
            "smt_selected_hist": hist_counts([int(r["smt_selected"]) for r in rows], max_bin=args.max_hist_bin),
            "baseline_selected_hist": hist_counts([int(r["baseline_selected"]) for r in rows], max_bin=args.max_hist_bin),
            "smt_annotated_overlap_hist": hist_counts([int(r["smt_annotated_overlap"]) for r in rows], max_bin=args.max_hist_bin),
            "baseline_annotated_overlap_hist": hist_counts([int(r["baseline_annotated_overlap"]) for r in rows], max_bin=args.max_hist_bin),
        }

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_header = [
        "variant",
        "patient_id",
        "mode",
        "k_avg",

        "num_qrels_covered_for_patient",
        "num_qrels0_for_patient",
        "num_qrels1_for_patient",
        "num_qrels2_for_patient",

        "smt_selected",
        "baseline_selected",

        "smt_annotated_overlap",
        "baseline_annotated_overlap",

        "smt_annotated_qrels0",
        "smt_annotated_qrels1",
        "smt_annotated_qrels2",
        "baseline_annotated_qrels0",
        "baseline_annotated_qrels1",
        "baseline_annotated_qrels2",

        "smt_annotated_num_pos12",
        "baseline_annotated_num_pos12",
        "smt_annotated_score_sum",
        "baseline_annotated_score_sum",

        "smt_annotated_precision_12",
        "baseline_annotated_precision_12",
        "smt_annotated_weighted_precision_012",
        "baseline_annotated_weighted_precision_012",
        "smt_annotated_weighted_precision_012_norm",
        "baseline_annotated_weighted_precision_012_norm",

        "overlap",
        "smt_not_baseline",
        "baseline_not_smt",
        "jaccard_selected",

        "smt_source",
        "baseline_source",
    ]
    write_csv(out_dir / "per_patient.csv", per_header, per_patient)

    summary_header = [
        "variant",
        "mode",
        "patients",
        "k_avg",
        "avg_num_qrels_covered_for_patient",
        "avg_smt_selected",
        "avg_baseline_selected",
        "avg_smt_annotated_overlap",
        "avg_baseline_annotated_overlap",

        "macro_smt_annotated_precision_12",
        "macro_baseline_annotated_precision_12",
        "micro_smt_annotated_precision_12",
        "micro_baseline_annotated_precision_12",

        "macro_smt_annotated_weighted_precision_012",
        "macro_baseline_annotated_weighted_precision_012",
        "micro_smt_annotated_weighted_precision_012",
        "micro_baseline_annotated_weighted_precision_012",

        "macro_smt_annotated_weighted_precision_012_norm",
        "macro_baseline_annotated_weighted_precision_012_norm",
        "micro_smt_annotated_weighted_precision_012_norm",
        "micro_baseline_annotated_weighted_precision_012_norm",
    ]
    write_csv(out_dir / "summary_by_variant_mode.csv", summary_header, summary_rows)

    (out_dir / "distributions_by_variant_mode.json").write_text(
        json.dumps(dist, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] wrote {out_dir / 'per_patient.csv'}")
    print(f"[ok] wrote {out_dir / 'summary_by_variant_mode.csv'}")
    print(f"[ok] wrote {out_dir / 'distributions_by_variant_mode.json'}")
    print("[info] metrics are over qrels-covered fetched pairs only")
    print("[info] no judge outputs are used")
    print(f"[info] avg_cutoff_scope={args.avg_cutoff_scope} avg_cutoff_rounding={args.avg_cutoff_rounding}")
    print(f"[info] collapse_subsuffix={collapse} dedup_canonical={args.dedup_canonical} require_overlap={args.require_overlap}")
    print(f"[info] qrels={args.qrels}")
    print(f"[info] trial_corpus={args.trial_corpus}")
    if mode_filter is not None:
        print(f"[info] mode_filter={mode_filter}")
    if variant_filter is not None:
        print(f"[info] variant_filter={variant_filter}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)