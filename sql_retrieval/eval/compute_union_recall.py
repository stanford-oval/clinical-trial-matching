#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_union_recall.py

Compute "recall" against the union of relevant-and-eligible trials found across
all methods, per patient and per mode.

Definition
----------
For each patient_id and mode:
  universe(patient, mode) =
      union over all methods of trials that satisfy:
        - trial id is present in the provided trial corpus
        - fused(relevant) == True
        - fused(eligible) == True
        - and method-specific candidate rule is satisfied

Methods included in the union:
  - SMT: trials labeled all_satisfied
  - each TrialGPT-style baseline folder under --trialgpt-root:
      top-K trials, where K is the average SMT all_satisfied count
      (same cutoff logic as compare_smt_vs_trialgpt_avgcutoff.py)

Then for each method:
  recall_to_union = |method_hits ∩ universe| / |universe|

Since method_hits ⊆ universe by construction, this is simply:
  recall_to_union = |method_hits| / |universe|

Outputs
-------
<out>/per_patient_recall.csv
<out>/summary_recall_by_method_mode.csv
<out>/recall_distributions_by_method_mode.json

Notes
-----
- This is not corpus-level ground-truth recall.
- It is "relative recall against the pooled discoveries of all compared methods".
- If you restrict baselines via --baselines, the denominator changes accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_TRIALGPT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out")
DEFAULT_SMT_ROOT      = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")
DEFAULT_TRIAL_CORPUS  = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus_real.jsonl")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)
KNOWN_MODES = ("chief", "ccr", "all", "all-explore")


# ----------------------------
# Baseline-name normalization
# ----------------------------

_NEW_HYBRID_RE = re.compile(
    r"^newhybrid-(?P<encoder>.+)-(?P<objective>obj|noobj)-(?P<tag>gpt41|gpt5|raw|unk)$",
    re.IGNORECASE,
)
_TEXT_RE = re.compile(r"^text-(?P<encoder>.+)$", re.IGNORECASE)


def _pretty_encoder(enc: str) -> str:
    s = str(enc).strip()
    s = s.replace("_", "-")
    return s


def _pretty_tag(tag: str) -> str:
    t = str(tag).lower().strip()
    if t == "gpt41":
        return "4.1"
    if t == "gpt5":
        return "5"
    if t == "raw":
        return "raw"
    return t


def infer_baseline_metadata(baseline_name: str) -> Dict[str, str]:
    raw = str(baseline_name).strip()

    old_map = {
        "clinicianA": ("clinician", "A", "A", "00_clinician_A"),
        "clinicianB": ("clinician", "B", "B", "00_clinician_B"),
        "clinicianC": ("clinician", "C", "C", "00_clinician_C"),
        "clinicianD": ("clinician", "D", "D", "00_clinician_D"),
        "trialgpt41": ("trialgpt", "base-4.1", "TG-4.1", "10_trialgpt_4.1"),
        "trialgpt5": ("trialgpt", "base-5", "TG-5", "10_trialgpt_5"),
        "TrialGPT-with-Objective-gpt4.1": ("trialgpt-objective", "objective-4.1", "TG-O-4.1", "11_trialgpt_objective_4.1"),
        "TrialGPT-with-Objective-gpt5": ("trialgpt-objective", "objective-5", "TG-O-5", "11_trialgpt_objective_5"),
    }
    if raw in old_map:
        fam, var, disp, sort_key = old_map[raw]
        return {
            "method_raw": raw,
            "method_family": fam,
            "method_variant": var,
            "method_display": disp,
            "method_sort_key": sort_key,
        }

    m = _NEW_HYBRID_RE.match(raw)
    if m:
        enc = _pretty_encoder(m.group("encoder"))
        objective = m.group("objective").lower()
        tag = m.group("tag").lower()
        tag_disp = _pretty_tag(tag)

        if objective == "noobj":
            fam = "newhybrid-noobj"
            var = f"{enc}-{tag}"
            disp = f"NH-{enc}-{tag_disp}"
            sort_key = f"20_newhybrid_noobj_{enc}_{tag}"
        else:
            fam = "newhybrid-obj"
            var = f"{enc}-{tag}"
            disp = f"NH-O-{enc}-{tag_disp}"
            sort_key = f"21_newhybrid_obj_{enc}_{tag}"

        return {
            "method_raw": raw,
            "method_family": fam,
            "method_variant": var,
            "method_display": disp,
            "method_sort_key": sort_key,
        }

    m = _TEXT_RE.match(raw)
    if m:
        enc = _pretty_encoder(m.group("encoder"))
        return {
            "method_raw": raw,
            "method_family": "text",
            "method_variant": enc,
            "method_display": f"TXT-{enc}",
            "method_sort_key": f"30_text_{enc}",
        }

    return {
        "method_raw": raw,
        "method_family": "other",
        "method_variant": raw,
        "method_display": raw,
        "method_sort_key": f"99_other_{raw}",
    }


def smt_metadata() -> Dict[str, str]:
    return {
        "method_raw": "smt",
        "method_family": "smt",
        "method_variant": "all_satisfied",
        "method_display": "SMT",
        "method_sort_key": "00_smt",
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


def infer_mode_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]

    for token in ("all-explore", "all_explore"):
        if any(token == part for part in parts):
            return "all-explore"

    for m in ("chief", "ccr", "all"):
        if any(m == part for part in parts):
            return m

    return "unknown"


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
        for _, line in enumerate(f, 1):
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
    ranked_keys: List[str]
    label_by_key: Dict[str, Optional[str]]
    rel_by_key: Dict[str, Optional[bool]]
    elig_by_key: Dict[str, Optional[bool]]
    source_file: Path


def build_run(p: Path, collapse_subsuffix: bool, dedup: bool) -> Optional[PatientRun]:
    obj = load_json(p)
    if not isinstance(obj, dict):
        return None

    patient_id = obj.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.strip():
        patient_id = infer_patient_id_from_filename(p)
    if not patient_id:
        return None

    mode_from_obj = obj.get("mode") if isinstance(obj.get("mode"), str) else None
    mode_from_path = infer_mode_from_path(p)

    mode = (mode_from_obj or mode_from_path)
    if mode_from_path == "all-explore":
        mode = "all-explore"

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

    if not ranked_keys:
        return None

    return PatientRun(
        patient_id=patient_id,
        mode=mode,
        ranked_keys=ranked_keys,
        label_by_key=label_by_key,
        rel_by_key=rel_by_key,
        elig_by_key=elig_by_key,
        source_file=p,
    )


def collect_runs(root: Path, collapse_subsuffix: bool, dedup: bool) -> Dict[Tuple[str, str], PatientRun]:
    out: Dict[Tuple[str, str], PatientRun] = {}
    for p in root.rglob("patient_labels/*.json"):
        run = build_run(p, collapse_subsuffix=collapse_subsuffix, dedup=dedup)
        if not run:
            continue
        k = (run.patient_id, run.mode)
        prev = out.get(k)
        if prev is None or len(run.ranked_keys) > len(prev.ranked_keys):
            out[k] = run
    return out


def collect_trialgpt_runs_by_baseline(
    root: Path,
    collapse_subsuffix: bool,
    dedup: bool,
    baseline_filter: Optional[Set[str]] = None,
) -> Dict[str, Dict[Tuple[str, str], PatientRun]]:
    out: Dict[str, Dict[Tuple[str, str], PatientRun]] = {}

    if not root.exists():
        return out

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        baseline_name = child.name
        if baseline_filter is not None and baseline_name not in baseline_filter:
            continue

        runs: Dict[Tuple[str, str], PatientRun] = {}
        for p in child.rglob("patient_labels/*.json"):
            run = build_run(p, collapse_subsuffix=collapse_subsuffix, dedup=dedup)
            if not run:
                continue
            k = (run.patient_id, run.mode)
            prev = runs.get(k)
            if prev is None or len(run.ranked_keys) > len(prev.ranked_keys):
                runs[k] = run

        if runs:
            out[baseline_name] = runs

    return out


def build_fused_rel_elig_map(
    tg: Optional[PatientRun],
    smt: Optional[PatientRun],
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

    if tg is not None:
        ingest(tg)
    if smt is not None:
        ingest(smt)
    return fused


def is_rel_and_elig_true(key: str, fused_map: Dict[str, Tuple[Optional[bool], Optional[bool]]]) -> bool:
    r, e = fused_map.get(key, (None, None))
    return (r is True) and (e is True)


def quantiles(xs: List[float], qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> Dict[str, float]:
    if not xs:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    xs2 = sorted(xs)
    n = len(xs2)
    out: Dict[str, float] = {}
    for q in qs:
        if q <= 0:
            out["q00"] = float(xs2[0])
            continue
        if q >= 1:
            out["q100"] = float(xs2[-1])
            continue
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out[f"q{int(q*100):02d}"] = float(xs2[lo])
        else:
            frac = pos - lo
            out[f"q{int(q*100):02d}"] = float(xs2[lo] * (1 - frac) + xs2[hi] * frac)
    return out


def hist_counts_recall(xs: List[float]) -> Dict[str, int]:
    bins = {
        "0.0": 0,
        "(0,0.25]": 0,
        "(0.25,0.5]": 0,
        "(0.5,0.75]": 0,
        "(0.75,1.0)": 0,
        "1.0": 0,
    }
    for x in xs:
        if x == 0.0:
            bins["0.0"] += 1
        elif x <= 0.25:
            bins["(0,0.25]"] += 1
        elif x <= 0.5:
            bins["(0.25,0.5]"] += 1
        elif x <= 0.75:
            bins["(0.5,0.75]"] += 1
        elif x < 1.0:
            bins["(0.75,1.0)"] += 1
        else:
            bins["1.0"] += 1
    return bins


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
        if t not in KNOWN_MODES:
            raise ValueError(f"Unknown mode in --modes: {t} (known: {', '.join(KNOWN_MODES)})")
        out.append(t)
    return out or None


def _parse_baselines_arg(s: str) -> Optional[Set[str]]:
    s = (s or "").strip()
    if not s:
        return None
    out = set()
    for tok in s.split(","):
        t = tok.strip()
        if t:
            out.add(t)
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute per-method recall against the union of relevant-and-eligible trials found across all compared methods."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--trial-corpus", type=Path, default=DEFAULT_TRIAL_CORPUS)
    ap.add_argument("--out", type=Path, default=Path("./union_recall_out"))
    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--avg-cutoff-scope", choices=("per_mode", "global"), default="per_mode")
    ap.add_argument("--avg-cutoff-rounding", choices=("round", "floor", "ceil"), default="round")
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Optional comma-separated modes filter (subset of chief,ccr,all,all-explore).",
    )
    ap.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Optional comma-separated TrialGPT baseline folder names under --trialgpt-root.",
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

    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)
    baseline_filter = _parse_baselines_arg(args.baselines)

    allowed_trial_ids = load_allowed_trial_ids_from_corpus(
        args.trial_corpus,
        collapse_subsuffix=collapse,
    )
    print(f"[info] loaded {len(allowed_trial_ids)} allowed trial IDs from corpus")

    smt_runs = collect_runs(args.smt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)
    tg_runs_by_baseline = collect_trialgpt_runs_by_baseline(
        args.trialgpt_root,
        collapse_subsuffix=collapse,
        dedup=args.dedup_canonical,
        baseline_filter=baseline_filter,
    )

    if not tg_runs_by_baseline:
        print("[error] no TrialGPT baseline runs found", file=sys.stderr)
        sys.exit(2)

    if mode_filter is not None:
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}
        tg_runs_by_baseline = {
            baseline: {k: v for k, v in runs.items() if k[1] in mode_filter}
            for baseline, runs in tg_runs_by_baseline.items()
        }
        tg_runs_by_baseline = {b: r for b, r in tg_runs_by_baseline.items() if r}

    # Same K logic as compare_smt_vs_trialgpt_avgcutoff.py
    keys_all_for_k = sorted(set(smt_runs.keys()))
    smt_m_counts_by_mode: Dict[str, List[int]] = {}
    smt_m_counts_global: List[int] = []

    for (patient_id, mode) in keys_all_for_k:
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
        modes_present = sorted({mode for _, mode in smt_runs.keys()})
        for mode in modes_present:
            avg_k_by_mode[mode] = k
    else:
        for mode, xs in smt_m_counts_by_mode.items():
            avg_k_by_mode[mode] = round_k(statistics.mean(xs)) if xs else 0

    # Union keys across all methods
    all_patient_mode_keys: Set[Tuple[str, str]] = set(smt_runs.keys())
    for runs in tg_runs_by_baseline.values():
        all_patient_mode_keys |= set(runs.keys())

    method_hits_by_key: Dict[str, Dict[Tuple[str, str], Set[str]]] = {}
    method_meta: Dict[str, Dict[str, str]] = {"smt": smt_metadata()}

    # SMT hits
    smt_hits_for_all_keys: Dict[Tuple[str, str], Set[str]] = {}
    for key in sorted(all_patient_mode_keys):
        patient_id, mode = key
        smt = smt_runs.get(key)

        fused_map = build_fused_rel_elig_map(None, smt)

        smt_candidates: List[str] = []
        if smt is not None:
            smt_candidates = [
                k for k in smt.ranked_keys
                if smt.label_by_key.get(k) == "all_satisfied" and k in allowed_trial_ids
            ]

        smt_hits_for_all_keys[key] = {k for k in smt_candidates if is_rel_and_elig_true(k, fused_map)}

    method_hits_by_key["smt"] = smt_hits_for_all_keys

    # Baseline hits
    for baseline_raw, tg_runs in sorted(tg_runs_by_baseline.items()):
        method_meta[baseline_raw] = infer_baseline_metadata(baseline_raw)
        hits_for_this_baseline: Dict[Tuple[str, str], Set[str]] = {}

        for key in sorted(all_patient_mode_keys):
            patient_id, mode = key
            smt = smt_runs.get(key)
            tg = tg_runs.get(key)

            fused_map = build_fused_rel_elig_map(tg, smt)
            k_avg = max(0, int(avg_k_by_mode.get(mode, 0)))

            tg_candidates: List[str] = []
            if tg is not None and k_avg > 0:
                tg_ranked_in_corpus = [k for k in tg.ranked_keys if k in allowed_trial_ids]
                tg_candidates = tg_ranked_in_corpus[:k_avg]

            hits_for_this_baseline[key] = {k for k in tg_candidates if is_rel_and_elig_true(k, fused_map)}

        method_hits_by_key[baseline_raw] = hits_for_this_baseline

    # Universe per patient/mode = union across methods
    universe_by_key: Dict[Tuple[str, str], Set[str]] = {}
    for key in sorted(all_patient_mode_keys):
        uni: Set[str] = set()
        for method_raw, hits_map in method_hits_by_key.items():
            uni |= hits_map.get(key, set())
        universe_by_key[key] = uni

    per_patient_rows: List[Dict[str, Any]] = []

    for method_raw, hits_map in sorted(method_hits_by_key.items(), key=lambda kv: method_meta[kv[0]]["method_sort_key"]):
        meta = method_meta[method_raw]
        for key in sorted(all_patient_mode_keys):
            patient_id, mode = key
            hits = hits_map.get(key, set())
            universe = universe_by_key.get(key, set())

            found_n = len(hits)
            denom_n = len(universe)
            recall = (float(found_n) / float(denom_n)) if denom_n > 0 else ""

            per_patient_rows.append({
                "method_raw": meta["method_raw"],
                "method_family": meta["method_family"],
                "method_variant": meta["method_variant"],
                "method_display": meta["method_display"],
                "method_sort_key": meta["method_sort_key"],
                "patient_id": patient_id,
                "mode": mode,
                "k_avg": max(0, int(avg_k_by_mode.get(mode, 0))),
                "found_relevant_eligible": found_n,
                "union_relevant_eligible": denom_n,
                "recall_to_union": recall,
            })

    by_method_mode: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in per_patient_rows:
        by_method_mode.setdefault((r["method_raw"], r["mode"]), []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    dist: Dict[str, Any] = {}

    for (method_raw, mode), rows in sorted(by_method_mode.items(), key=lambda kv: (method_meta[kv[0][0]]["method_sort_key"], kv[0][1])):
        meta = method_meta[method_raw]
        recalls = [float(r["recall_to_union"]) for r in rows if r["recall_to_union"] != ""]
        founds = [int(r["found_relevant_eligible"]) for r in rows]
        denoms = [int(r["union_relevant_eligible"]) for r in rows]

        mean_recall = statistics.mean(recalls) if recalls else float("nan")
        stdev_recall = statistics.pstdev(recalls) if len(recalls) > 1 else 0.0
        macro_recall = mean_recall
        micro_recall = (sum(founds) / sum(denoms)) if sum(denoms) > 0 else float("nan")

        dist.setdefault(method_raw, {})
        dist[method_raw][mode] = {
            "method_raw": meta["method_raw"],
            "method_family": meta["method_family"],
            "method_variant": meta["method_variant"],
            "method_display": meta["method_display"],
            "method_sort_key": meta["method_sort_key"],
            "k_avg": rows[0]["k_avg"] if rows else 0,
            "patients": len(rows),
            "mean_found_relevant_eligible": statistics.mean(founds) if founds else float("nan"),
            "mean_union_relevant_eligible": statistics.mean(denoms) if denoms else float("nan"),
            "macro_recall_to_union": macro_recall,
            "micro_recall_to_union": micro_recall,
            "recall_distribution": {
                "n_patients": len(recalls),
                "mean": mean_recall,
                "stdev": stdev_recall,
                "quantiles": quantiles(recalls),
                "histogram": hist_counts_recall(recalls),
            },
        }

        summary_rows.append({
            "method_raw": meta["method_raw"],
            "method_family": meta["method_family"],
            "method_variant": meta["method_variant"],
            "method_display": meta["method_display"],
            "method_sort_key": meta["method_sort_key"],
            "mode": mode,
            "patients": len(rows),
            "k_avg": rows[0]["k_avg"] if rows else 0,
            "avg_found_relevant_eligible": statistics.mean(founds) if founds else float("nan"),
            "avg_union_relevant_eligible": statistics.mean(denoms) if denoms else float("nan"),
            "macro_recall_to_union": macro_recall,
            "micro_recall_to_union": micro_recall,
        })

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_header = [
        "method_raw", "method_family", "method_variant", "method_display", "method_sort_key",
        "patient_id", "mode", "k_avg",
        "found_relevant_eligible", "union_relevant_eligible", "recall_to_union",
    ]
    write_csv(out_dir / "per_patient_recall.csv", per_header, per_patient_rows)

    sum_header = [
        "method_raw", "method_family", "method_variant", "method_display", "method_sort_key",
        "mode", "patients", "k_avg",
        "avg_found_relevant_eligible", "avg_union_relevant_eligible",
        "macro_recall_to_union", "micro_recall_to_union",
    ]
    write_csv(out_dir / "summary_recall_by_method_mode.csv", sum_header, summary_rows)

    (out_dir / "recall_distributions_by_method_mode.json").write_text(
        json.dumps(dist, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] wrote {out_dir / 'per_patient_recall.csv'}")
    print(f"[ok] wrote {out_dir / 'summary_recall_by_method_mode.csv'}")
    print(f"[ok] wrote {out_dir / 'recall_distributions_by_method_mode.json'}")

    print("[info] denominator = union of RE&EL trials found across all compared methods, per patient/mode")
    print("[info] method candidates:")
    print("  - SMT: label == all_satisfied")
    print("  - TrialGPT-style baselines: top-K in corpus, where K is avg SMT all_satisfied count")
    print(f"[info] avg_cutoff_scope={args.avg_cutoff_scope} avg_cutoff_rounding={args.avg_cutoff_rounding}")
    print(f"[info] collapse_subsuffix={collapse} dedup_canonical={args.dedup_canonical}")
    print(f"[info] trial_corpus={args.trial_corpus}")

    if mode_filter is not None:
        print(f"[info] mode_filter={mode_filter}")
    if baseline_filter is not None:
        print(f"[info] baseline_filter={sorted(baseline_filter)}")
    else:
        print(f"[info] baselines={sorted(tg_runs_by_baseline.keys())}")

    print("[info] methods included in recall denominator:")
    for method_raw in sorted(method_hits_by_key.keys(), key=lambda x: method_meta[x]["method_sort_key"]):
        meta = method_meta[method_raw]
        print(
            f"  - {method_raw} -> display={meta['method_display']} "
            f"family={meta['method_family']} variant={meta['method_variant']}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)