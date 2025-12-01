#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_smt_vs_trialgpt_avgcutoff_with_weighted_qrels.py

Variant-aware comparison of:
  - SMT: selected set = trials with label == all_satisfied
  - Baseline: top-K, where K = average SMT all_satisfied count

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

Qrels recall variants
---------------------
We compute TWO versions of qrels-based recall:

1) only2 recall:
   Gold includes only qrels score == 2

   recall_only2 = (# selected ∩ score2_gold) / (# score2_gold)

2) weighted(1,2) recall:
   qrels score 1 has weight 1
   qrels score 2 has weight 2

   recall_w12 = (sum weights of selected gold trials) / (sum weights of all gold trials)

Optional fused rel/elig restriction
-----------------------------------
By default, recall is computed on the raw selected sets at the cutoff.

If you pass:
  --restrict-to-fused-rel-elig

then selected sets are additionally filtered to trials where fused relevant=True
AND fused eligible=True across SMT+baseline for that patient/mode pair.

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
# Truthiness + 3-valued OR
# ----------------------------

def is_truthy(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"1", "true", "t", "yes", "y"}:
            return True
        if t in {"0", "false", "f", "no", "n"}:
            return False
    return None


def or3(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return None


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
    """
    Infer (mode, variant) from path relative to root.

    Examples
    --------
    Legacy:
      <root>/ccr/patient_labels/x.json
        -> mode='ccr', variant='default'

    New:
      <root>/text-bge/ccr/patient_labels/x.json
        -> mode='ccr', variant='text-bge'

      <root>/foo/bar/all_explore/patient_labels/x.json
        -> mode='all-explore', variant='foo/bar'
    """
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

def load_qrels_weights(
    qrels_path: Path,
    collapse_subsuffix: bool,
    allowed_trial_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Returns:
      patient_id -> {trial_id -> qrels_score}
    keeping only score 1 and 2.
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

        def process_line(line: str):
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

            if score not in (1, 2):
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


def extract_rel_elig(tr: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    rel_keys = ["relevant", "is_relevant", "judge_relevant", "any_subcohort_relevant", "rel"]
    elig_keys = ["eligible", "is_eligible", "judge_eligible", "any_subcohort_eligible", "elig"]

    rel = None
    for k in rel_keys:
        if k in tr:
            rel = is_truthy(tr.get(k))
            if rel is not None:
                break

    elig = None
    for k in elig_keys:
        if k in tr:
            elig = is_truthy(tr.get(k))
            if elig is not None:
                break

    if "relevant_and_eligible" in tr:
        rae = is_truthy(tr.get("relevant_and_eligible"))
        if rae is True:
            if rel is None:
                rel = True
            if elig is None:
                elig = True

    lab = tr.get("label")
    if elig is None and isinstance(lab, str):
        l = lab.strip().lower()
        if l == "all_satisfied":
            elig = True
        elif l in {"unsatisfied_inclusion", "explicit_contradiction"}:
            elig = False

    return rel, elig


@dataclass
class PatientRun:
    patient_id: str
    mode: str
    variant: str
    ranked_keys: List[str]
    label_by_key: Dict[str, Optional[str]]
    rel_by_key: Dict[str, Optional[bool]]
    elig_by_key: Dict[str, Optional[bool]]
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
    rel_by_key: Dict[str, Optional[bool]] = {}
    elig_by_key: Dict[str, Optional[bool]] = {}
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
        r, e = extract_rel_elig(t)
        rel_by_key[key] = r
        elig_by_key[key] = e

    return PatientRun(
        patient_id=patient_id,
        mode=mode,
        variant=variant,
        ranked_keys=ranked_keys,
        label_by_key=label_by_key,
        rel_by_key=rel_by_key,
        elig_by_key=elig_by_key,
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
# Fused rel/elig map
# ----------------------------

def build_fused_rel_elig_map(
    baseline_run: Optional[PatientRun],
    smt_run: Optional[PatientRun],
) -> Dict[str, Tuple[Optional[bool], Optional[bool]]]:
    fused: Dict[str, Tuple[Optional[bool], Optional[bool]]] = {}

    def ingest(run: PatientRun) -> None:
        for key in run.ranked_keys:
            r = run.rel_by_key.get(key)
            e = run.elig_by_key.get(key)
            if key not in fused:
                fused[key] = (r, e)
            else:
                pr, pe = fused[key]
                fused[key] = (or3(pr, r), or3(pe, e))

    if baseline_run is not None:
        ingest(baseline_run)
    if smt_run is not None:
        ingest(smt_run)

    return fused


def is_rel_and_elig_true(
    key: str,
    fused_map: Dict[str, Tuple[Optional[bool], Optional[bool]]],
) -> bool:
    r, e = fused_map.get(key, (None, None))
    return (r is True) and (e is True)


# ----------------------------
# Recall helpers
# ----------------------------

def compute_only2_recall(
    selected: Set[str],
    qrels_scores: Dict[str, int],
) -> Tuple[int, int, Optional[float]]:
    gold2 = {tid for tid, s in qrels_scores.items() if s == 2}
    denom = len(gold2)
    num = len(selected & gold2)
    rec = (num / denom) if denom > 0 else None
    return num, denom, rec


def compute_weighted12_recall(
    selected: Set[str],
    qrels_scores: Dict[str, int],
) -> Tuple[int, int, Optional[float]]:
    denom = 0
    num = 0
    for tid, s in qrels_scores.items():
        w = 2 if s == 2 else 1
        denom += w
        if tid in selected:
            num += w
    rec = (num / denom) if denom > 0 else None
    return num, denom, rec


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
        description="Compare SMT(all_satisfied) vs baseline(top-K avg cutoff) with weighted qrels recall."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--trial-corpus", type=Path, default=DEFAULT_TRIAL_CORPUS)
    ap.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    ap.add_argument("--out", type=Path, default=Path("./avgcutoff_compare_with_weighted_qrels_out"))

    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")

    ap.add_argument("--avg-cutoff-scope", choices=("per_mode", "global"), default="per_mode")
    ap.add_argument("--avg-cutoff-rounding", choices=("round", "floor", "ceil"), default="round")
    ap.add_argument("--max-hist-bin", type=int, default=60)

    ap.add_argument(
        "--restrict-to-fused-rel-elig",
        action="store_true",
        help="Restrict selected sets to fused relevant=True AND eligible=True.",
    )
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

    qrels_scores = load_qrels_weights(
        args.qrels,
        collapse_subsuffix=collapse,
        allowed_trial_ids=allowed_trial_ids,
    )
    print(f"[info] loaded qrels score-1/2 labels for {len(qrels_scores)} patients")

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

    # Average SMT cutoff is based on SMT only, shared across baseline variants.
    smt_m_counts_by_mode: Dict[str, List[int]] = {}
    smt_m_counts_global: List[int] = []
    smt_keys_for_cutoff: Set[Tuple[str, str]] = set()

    for (variant, patient_id, mode) in baseline_runs.keys():
        if (patient_id, mode) not in smt_runs:
            continue
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
    smt_keys_lifted = {
        (variant, patient_id, mode)
        for (variant, patient_id, mode) in baseline_keys
        if (patient_id, mode) in smt_runs
    }

    keys_all = sorted(smt_keys_lifted)

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

        fused_map = build_fused_rel_elig_map(baseline, smt)
        gold_scores = qrels_scores.get(patient_id, {})

        smt_candidates = [
            k for k in smt.ranked_keys
            if smt.label_by_key.get(k) == "all_satisfied" and k in allowed_trial_ids
        ]

        k_avg = max(0, int(avg_k_by_mode.get(mode, 0)))
        baseline_candidates: List[str] = []
        if k_avg > 0:
            baseline_ranked_in_corpus = [k for k in baseline.ranked_keys if k in allowed_trial_ids]
            baseline_candidates = baseline_ranked_in_corpus[:k_avg]

        if args.restrict_to_fused_rel_elig:
            smt_set = {k for k in smt_candidates if is_rel_and_elig_true(k, fused_map)}
            baseline_set = {k for k in baseline_candidates if is_rel_and_elig_true(k, fused_map)}
        else:
            smt_set = set(smt_candidates)
            baseline_set = set(baseline_candidates)

        num_gold_any = len(gold_scores)
        num_gold_only2 = sum(1 for _tid, s in gold_scores.items() if s == 2)

        smt_hit_only2, smt_denom_only2, smt_recall_only2 = compute_only2_recall(smt_set, gold_scores)
        baseline_hit_only2, baseline_denom_only2, baseline_recall_only2 = compute_only2_recall(baseline_set, gold_scores)

        smt_hit_w12, smt_denom_w12, smt_recall_w12 = compute_weighted12_recall(smt_set, gold_scores)
        baseline_hit_w12, baseline_denom_w12, baseline_recall_w12 = compute_weighted12_recall(baseline_set, gold_scores)

        overlap = smt_set & baseline_set
        smt_not_baseline = smt_set - baseline_set
        baseline_not_smt = baseline_set - smt_set

        row = {
            "variant": variant,
            "patient_id": patient_id,
            "mode": mode,
            "k_avg": k_avg,

            "num_gold_qrels_1or2": num_gold_any,
            "num_gold_qrels_2": num_gold_only2,

            "smt_selected": len(smt_set),
            "baseline_selected": len(baseline_set),

            "smt_qrels_hit_only2": smt_hit_only2,
            "baseline_qrels_hit_only2": baseline_hit_only2,
            "smt_qrels_denom_only2": smt_denom_only2,
            "baseline_qrels_denom_only2": baseline_denom_only2,
            "smt_qrels_recall_only2": f"{smt_recall_only2:.6f}" if smt_recall_only2 is not None else "",
            "baseline_qrels_recall_only2": f"{baseline_recall_only2:.6f}" if baseline_recall_only2 is not None else "",

            "smt_qrels_hit_w12": smt_hit_w12,
            "baseline_qrels_hit_w12": baseline_hit_w12,
            "smt_qrels_denom_w12": smt_denom_w12,
            "baseline_qrels_denom_w12": baseline_denom_w12,
            "smt_qrels_recall_w12": f"{smt_recall_w12:.6f}" if smt_recall_w12 is not None else "",
            "baseline_qrels_recall_w12": f"{baseline_recall_w12:.6f}" if baseline_recall_w12 is not None else "",

            "overlap": len(overlap),
            "smt_not_baseline": len(smt_not_baseline),
            "baseline_not_smt": len(baseline_not_smt),
            "jaccard_selected": (len(overlap) / len(smt_set | baseline_set)) if (smt_set or baseline_set) else "",

            "smt_source": str(smt.source_file),
            "baseline_source": str(baseline.source_file),
        }
        per_patient.append(row)
        by_variant_mode_rows[(variant, mode)].append(row)

    for (variant, mode), rows in sorted(by_variant_mode_rows.items()):
        valid_smt_only2 = [float(r["smt_qrels_recall_only2"]) for r in rows if r["smt_qrels_recall_only2"] != ""]
        valid_baseline_only2 = [float(r["baseline_qrels_recall_only2"]) for r in rows if r["baseline_qrels_recall_only2"] != ""]

        valid_smt_w12 = [float(r["smt_qrels_recall_w12"]) for r in rows if r["smt_qrels_recall_w12"] != ""]
        valid_baseline_w12 = [float(r["baseline_qrels_recall_w12"]) for r in rows if r["baseline_qrels_recall_w12"] != ""]

        smt_hit_only2_total = sum(int(r["smt_qrels_hit_only2"]) for r in rows)
        baseline_hit_only2_total = sum(int(r["baseline_qrels_hit_only2"]) for r in rows)
        smt_denom_only2_total = sum(int(r["smt_qrels_denom_only2"]) for r in rows)
        baseline_denom_only2_total = sum(int(r["baseline_qrels_denom_only2"]) for r in rows)

        smt_hit_w12_total = sum(int(r["smt_qrels_hit_w12"]) for r in rows)
        baseline_hit_w12_total = sum(int(r["baseline_qrels_hit_w12"]) for r in rows)
        smt_denom_w12_total = sum(int(r["smt_qrels_denom_w12"]) for r in rows)
        baseline_denom_w12_total = sum(int(r["baseline_qrels_denom_w12"]) for r in rows)

        summary_rows.append({
            "variant": variant,
            "mode": mode,
            "patients": len(rows),
            "patients_with_gold_only2": sum(1 for r in rows if int(r["num_gold_qrels_2"]) > 0),
            "patients_with_gold_w12": sum(1 for r in rows if int(r["num_gold_qrels_1or2"]) > 0),
            "k_avg": rows[0]["k_avg"] if rows else 0,

            "macro_smt_qrels_recall_only2": f"{statistics.mean(valid_smt_only2):.6f}" if valid_smt_only2 else "",
            "macro_baseline_qrels_recall_only2": f"{statistics.mean(valid_baseline_only2):.6f}" if valid_baseline_only2 else "",

            "micro_smt_qrels_recall_only2": f"{(smt_hit_only2_total / smt_denom_only2_total):.6f}" if smt_denom_only2_total > 0 else "",
            "micro_baseline_qrels_recall_only2": f"{(baseline_hit_only2_total / baseline_denom_only2_total):.6f}" if baseline_denom_only2_total > 0 else "",

            "macro_smt_qrels_recall_w12": f"{statistics.mean(valid_smt_w12):.6f}" if valid_smt_w12 else "",
            "macro_baseline_qrels_recall_w12": f"{statistics.mean(valid_baseline_w12):.6f}" if valid_baseline_w12 else "",

            "micro_smt_qrels_recall_w12": f"{(smt_hit_w12_total / smt_denom_w12_total):.6f}" if smt_denom_w12_total > 0 else "",
            "micro_baseline_qrels_recall_w12": f"{(baseline_hit_w12_total / baseline_denom_w12_total):.6f}" if baseline_denom_w12_total > 0 else "",

            "avg_smt_selected": f"{statistics.mean([int(r['smt_selected']) for r in rows]):.6f}" if rows else "",
            "avg_baseline_selected": f"{statistics.mean([int(r['baseline_selected']) for r in rows]):.6f}" if rows else "",
        })

        dist_key = f"{variant}::{mode}"
        dist[dist_key] = {
            "variant": variant,
            "mode": mode,
            "k_avg": rows[0]["k_avg"] if rows else 0,
            "num_gold_qrels_1or2_hist": hist_counts([int(r["num_gold_qrels_1or2"]) for r in rows], max_bin=args.max_hist_bin),
            "num_gold_qrels_2_hist": hist_counts([int(r["num_gold_qrels_2"]) for r in rows], max_bin=args.max_hist_bin),
            "smt_selected_hist": hist_counts([int(r["smt_selected"]) for r in rows], max_bin=args.max_hist_bin),
            "baseline_selected_hist": hist_counts([int(r["baseline_selected"]) for r in rows], max_bin=args.max_hist_bin),
            "smt_qrels_hit_only2_hist": hist_counts([int(r["smt_qrels_hit_only2"]) for r in rows], max_bin=args.max_hist_bin),
            "baseline_qrels_hit_only2_hist": hist_counts([int(r["baseline_qrels_hit_only2"]) for r in rows], max_bin=args.max_hist_bin),
            "smt_qrels_hit_w12_hist": hist_counts([int(r["smt_qrels_hit_w12"]) for r in rows], max_bin=args.max_hist_bin),
            "baseline_qrels_hit_w12_hist": hist_counts([int(r["baseline_qrels_hit_w12"]) for r in rows], max_bin=args.max_hist_bin),
        }

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_header = [
        "variant",
        "patient_id",
        "mode",
        "k_avg",
        "num_gold_qrels_1or2",
        "num_gold_qrels_2",
        "smt_selected",
        "baseline_selected",
        "smt_qrels_hit_only2",
        "baseline_qrels_hit_only2",
        "smt_qrels_denom_only2",
        "baseline_qrels_denom_only2",
        "smt_qrels_recall_only2",
        "baseline_qrels_recall_only2",
        "smt_qrels_hit_w12",
        "baseline_qrels_hit_w12",
        "smt_qrels_denom_w12",
        "baseline_qrels_denom_w12",
        "smt_qrels_recall_w12",
        "baseline_qrels_recall_w12",
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
        "patients_with_gold_only2",
        "patients_with_gold_w12",
        "k_avg",
        "macro_smt_qrels_recall_only2",
        "macro_baseline_qrels_recall_only2",
        "micro_smt_qrels_recall_only2",
        "micro_baseline_qrels_recall_only2",
        "macro_smt_qrels_recall_w12",
        "macro_baseline_qrels_recall_w12",
        "micro_smt_qrels_recall_w12",
        "micro_baseline_qrels_recall_w12",
        "avg_smt_selected",
        "avg_baseline_selected",
    ]
    write_csv(out_dir / "summary_by_variant_mode.csv", summary_header, summary_rows)

    (out_dir / "distributions_by_variant_mode.json").write_text(json.dumps(dist, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_dir / 'per_patient.csv'}")
    print(f"[ok] wrote {out_dir / 'summary_by_variant_mode.csv'}")
    print(f"[ok] wrote {out_dir / 'distributions_by_variant_mode.json'}")
    print("[info] qrels recall variants: only2 and weighted(1,2)")
    print(f"[info] avg_cutoff_scope={args.avg_cutoff_scope} avg_cutoff_rounding={args.avg_cutoff_rounding}")
    print(f"[info] collapse_subsuffix={collapse} dedup_canonical={args.dedup_canonical} require_overlap={args.require_overlap}")
    print(f"[info] restrict_to_fused_rel_elig={args.restrict_to_fused_rel_elig}")
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