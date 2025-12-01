#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_tiers_by_label.py — Compare TrialGPT vs SMT by heterogeneous tier buckets.

Always computes:
  - SMT buckets by EXACT labels:
      all_satisfied / unsatisfied_inclusion / explicit_contradiction

TrialGPT can be computed in 1 or 2 ways:
  - interval bucketing (native): use interval_mno == m/n/o
  - SMT-cutoff bucketing: compute SMT per-patient cutoffs (m,n,o) from label counts,
      then bucket TrialGPT ranked list by rank segments within top-o:
        ranks [1..m]     -> "m"
        ranks [m+1..n]   -> "n"
        ranks [n+1..o]   -> "o"

NEW (per request):
  - Metrics (relevant / eligible / relevant_and_eligible) are computed using OR-fusion
    across SMT + TrialGPT for the same (patient, mode, trial-key):
      relevant := OR3(relevant_smt, relevant_tg)
      eligible := OR3(eligible_smt, eligible_tg)
    where OR3 is 3-valued OR over {True, False, None}:
      True if either True; False if both False; None otherwise.
  - Bucket membership is still system-native:
      SMT buckets are by SMT label;
      TrialGPT buckets are by interval_mno or SMT-cutoff rank segments.

Outputs:
  <out>/per_patient.csv
  <out>/aggregate.csv

UPDATED (2026-03-05, ALL-EXPLORE MODE):
  - Recognize mode="all-explore" in directory paths and/or JSON payloads.
  - Optionally filter modes via --modes (comma-separated).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TRIALGPT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out")
DEFAULT_SMT_ROOT      = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)

SMT_LABELS = ("all_satisfied", "unsatisfied_inclusion", "explicit_contradiction")
MNO = ("m", "n", "o")
KNOWN_MODES = ("chief", "ccr", "all", "all-explore")


# ----------------------------
# Helpers: parsing / truthiness
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
    """
    Detect mode from directory components.
    IMPORTANT: check all-explore before all.
    """
    parts = [x.lower() for x in p.parts]

    # NEW: all-explore mode
    for token in ("all-explore", "all_explore"):
        if any(token == part for part in parts):
            return "all-explore"

    for m in ("chief", "ccr", "all"):
        if any(m == part for part in parts):
            return m

    return "unknown"


def infer_prevent_tag_from_filename(p: Path) -> str:
    s = p.name.lower()
    if "__prevent" in s or s.endswith("_prevent.json") or s.endswith("__prevent.json"):
        return "prevent"
    if "__noprevent" in s or s.endswith("_noprevent.json") or s.endswith("__noprevent.json"):
        return "noprevent"
    return "na"


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------
# Trial item extraction
# ----------------------------

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

    # SMT-style fallback mapping for eligibility if needed
    lab = tr.get("label")
    if elig is None and isinstance(lab, str):
        l = lab.strip().lower()
        if l == "all_satisfied":
            elig = True
        elif l in {"unsatisfied_inclusion", "explicit_contradiction"}:
            elig = False

    return rel, elig


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


def norm_interval_mno(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    return t if t in {"m", "n", "o"} else None


@dataclass
class PatientRun:
    patient_id: str
    mode: str
    prevent_tag: str
    # (key, rel, elig, label, interval_mno)
    items: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]]
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

    # Prefer folder all-explore (some JSONs may store "all" but live under all-explore/)
    mode = (mode_from_obj or mode_from_path)
    if mode_from_path == "all-explore":
        mode = "all-explore"

    prevent_tag = (obj.get("prevent_tag") if isinstance(obj.get("prevent_tag"), str) else None) or infer_prevent_tag_from_filename(p)

    trials = extract_ranked_list(obj)
    if not trials:
        return None

    # keep rank order if present
    if all(("rank" in t and isinstance(t.get("rank"), (int, float))) for t in trials):
        trials = sorted(trials, key=lambda t: int(t.get("rank", 10**9)))

    items: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]] = []
    seen: set[str] = set()

    for t in trials:
        key = extract_key(t, collapse_subsuffix=collapse_subsuffix)
        if not key:
            continue
        if dedup and key in seen:
            continue
        seen.add(key)

        rel, elig = extract_rel_elig(t)
        lab = norm_label(t.get("label"))
        mno = norm_interval_mno(t.get("interval_mno"))
        items.append((key, rel, elig, lab, mno))

    return PatientRun(patient_id=patient_id, mode=mode, prevent_tag=prevent_tag, items=items, source_file=p) if items else None


def collect_runs(root: Path, collapse_subsuffix: bool, dedup: bool) -> Dict[Tuple[str, str], PatientRun]:
    out: Dict[Tuple[str, str], PatientRun] = {}
    for p in root.rglob("patient_labels/*.json"):
        run = build_run(p, collapse_subsuffix=collapse_subsuffix, dedup=dedup)
        if not run:
            continue
        k = (run.patient_id, run.mode)
        prev = out.get(k)
        if prev is None or len(run.items) > len(prev.items):
            out[k] = run
    return out


# ----------------------------
# OR-fusion of rel/elig across systems (3-valued OR)
# ----------------------------

def or3(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """
    3-valued OR over {True, False, None}.
      - True if either is True
      - False if both are False
      - None otherwise (unknown remains if no True and not both False)
    """
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return None


def build_fused_rel_elig_map(
    tg: Optional[PatientRun],
    smt: Optional[PatientRun],
) -> Dict[str, Tuple[Optional[bool], Optional[bool]]]:
    """
    Return key -> (fused_rel, fused_elig), where fused is OR across sides.
    If a key exists on only one side, fused == that side's value.
    """
    fused: Dict[str, Tuple[Optional[bool], Optional[bool]]] = {}

    def ingest(run: PatientRun) -> None:
        for key, rel, elig, _, _ in run.items:
            if key not in fused:
                fused[key] = (rel, elig)
            else:
                prev_rel, prev_elig = fused[key]
                fused[key] = (or3(prev_rel, rel), or3(prev_elig, elig))

    if tg is not None:
        ingest(tg)
    if smt is not None:
        ingest(smt)

    return fused


def apply_fused_rel_elig(
    items: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]],
    fused_map: Dict[str, Tuple[Optional[bool], Optional[bool]]],
) -> List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]]:
    """
    Replace (rel, elig) in items with fused values when available.
    Keeps label/interval fields untouched (bucket identity stays system-native).
    """
    out: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]] = []
    for key, rel, elig, lab, mno in items:
        if key in fused_map:
            fr, fe = fused_map[key]
            out.append((key, fr, fe, lab, mno))
        else:
            out.append((key, rel, elig, lab, mno))
    return out


# ----------------------------
# Metrics over a bucket (subset), NOT top-K
# ----------------------------

def metrics_over_items(
    items: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]]
) -> Dict[str, int]:
    found = len(items)

    rel = sum(1 for _, r, _, _, _ in items if r is True)
    elig = sum(1 for _, _, e, _, _ in items if e is True)
    rel_and_elig = sum(1 for _, r, e, _, _ in items if (r is True and e is True))

    rel_unk = sum(1 for _, r, _, _, _ in items if r is None)
    elig_unk = sum(1 for _, _, e, _, _ in items if e is None)

    # label counts (SMT)
    c_all = 0
    c_unsat = 0
    c_contra = 0
    c_other = 0
    for _, _, _, lab, _ in items:
        if lab is None:
            c_other += 1
        elif lab == "all_satisfied":
            c_all += 1
        elif lab == "unsatisfied_inclusion":
            c_unsat += 1
        elif lab == "explicit_contradiction":
            c_contra += 1
        else:
            c_other += 1

    # interval counts (TrialGPT)
    i_m = 0
    i_n = 0
    i_o = 0
    i_unk = 0
    for _, _, _, _, mno in items:
        if mno == "m":
            i_m += 1
        elif mno == "n":
            i_n += 1
        elif mno == "o":
            i_o += 1
        else:
            i_unk += 1

    return {
        "bucket_size": found,
        "relevant": rel,
        "eligible": elig,
        "relevant_and_eligible": rel_and_elig,
        "relevant_unknown": rel_unk,
        "eligible_unknown": elig_unk,

        "label_all_satisfied": c_all,
        "label_unsatisfied_inclusion": c_unsat,
        "label_explicit_contradiction": c_contra,
        "label_other": c_other,

        "interval_m": i_m,
        "interval_n": i_n,
        "interval_o": i_o,
        "interval_unknown": i_unk,
    }


def bucket_items_smt(run: PatientRun, label: str):
    return [it for it in run.items if it[3] == label]


def bucket_items_trialgpt_interval(run: PatientRun, bucket_name: str):
    return [it for it in run.items if it[4] == bucket_name]


def smt_cutoffs_from_run(run: PatientRun) -> Tuple[int, int, int]:
    m = 0
    n = 0
    o = 0
    for _, _, _, lab, _ in run.items:
        if lab == "all_satisfied":
            m += 1
            n += 1
            o += 1
        elif lab == "unsatisfied_inclusion":
            n += 1
            o += 1
        elif lab == "explicit_contradiction":
            o += 1
        else:
            pass
    return m, n, o


def bucket_items_trialgpt_by_smt_cutoffs(
    tg_run: PatientRun,
    m: int,
    n: int,
    o: int,
    bucket_name: str,
) -> List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]]:
    top = tg_run.items[: max(0, o)]
    if bucket_name == "m":
        return top[: max(0, m)]
    if bucket_name == "n":
        return top[max(0, m) : max(0, n)]
    if bucket_name == "o":
        return top[max(0, n) : max(0, o)]
    return []


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
        t = tok.strip()
        if not t:
            continue
        t = t.lower()
        if t not in KNOWN_MODES:
            raise ValueError(f"Unknown mode in --modes: {t} (known: {', '.join(KNOWN_MODES)})")
        out.append(t)
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare TrialGPT vs SMT using heterogeneous tier buckets (labels / interval_mno / SMT-cutoff bucketing)."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--out", type=Path, default=Path("./tier_compare_out"))
    ap.add_argument("--dedup-canonical", action="store_true", help="Dedup within a patient's list by canonical key.")
    ap.add_argument("--keep-subcohort-suffix", action="store_true", help="Do NOT collapse NCT########a -> NCT########")
    ap.add_argument("--require-overlap", action="store_true", help="Only evaluate patient+mode pairs present in BOTH systems.")

    ap.add_argument(
        "--trialgpt-bucketing",
        choices=("interval", "smt_cutoff", "both"),
        default="both",
        help="How to bucket TrialGPT: native interval_mno, SMT per-patient cutoffs, or both (default).",
    )

    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Optional comma-separated modes filter (subset of chief,ccr,all,all-explore). "
             "If set, only those modes are evaluated.",
    )

    args = ap.parse_args()

    if not args.trialgpt_root.exists():
        print(f"[error] trialgpt root not found: {args.trialgpt_root}", file=sys.stderr)
        sys.exit(2)
    if not args.smt_root.exists():
        print(f"[error] smt root not found: {args.smt_root}", file=sys.stderr)
        sys.exit(2)

    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)

    tg_runs = collect_runs(args.trialgpt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)
    smt_runs = collect_runs(args.smt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)

    # optional mode filter
    if mode_filter is not None:
        tg_runs = {k: v for k, v in tg_runs.items() if k[1] in mode_filter}
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}

    keys_all = sorted(set(tg_runs.keys()) | set(smt_runs.keys()))
    if args.require_overlap:
        keys_all = sorted(set(tg_runs.keys()) & set(smt_runs.keys()))

    per_patient: List[Dict[str, Any]] = []

    agg: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    patients_seen: Dict[Tuple[str, str], set] = {}

    def acc(system: str, mode: str, bucket_name: str, met: Dict[str, int], patient_id: str) -> None:
        k = (system, mode, bucket_name)
        if k not in agg:
            agg[k] = {kk: 0 for kk in met.keys()}
        for kk, vv in met.items():
            agg[k][kk] += int(vv)
        patients_seen.setdefault((system, mode), set()).add(patient_id)

    miss_tg = 0
    miss_smt = 0
    tg_smtcutoff_missing_smt = 0  # trialgpt@SMT-cutoff could not run due to missing SMT

    for (patient_id, mode) in keys_all:
        tg = tg_runs.get((patient_id, mode))
        smt = smt_runs.get((patient_id, mode))

        # Build OR-fused (rel, elig) map across both systems for this patient+mode
        fused_map = build_fused_rel_elig_map(tg, smt)

        if tg is None:
            miss_tg += 1
        if smt is None:
            miss_smt += 1

        # 1) SMT label buckets (always when smt run exists)
        if smt is not None:
            for lab in SMT_LABELS:
                items_b = bucket_items_smt(smt, lab)
                items_b = apply_fused_rel_elig(items_b, fused_map)
                met = metrics_over_items(items_b)
                per_patient.append({
                    "patient_id": patient_id,
                    "mode": mode,
                    "system": "smt",
                    "bucket_name": lab,
                    "prevent_tag": smt.prevent_tag,
                    "smt_cutoff_m": "",
                    "smt_cutoff_n": "",
                    "smt_cutoff_o": "",
                    **met,
                    "source_file": str(smt.source_file),
                })
                acc("smt", mode, lab, met, patient_id)

        # 2/3) TrialGPT buckets (if tg exists)
        if tg is not None:
            do_interval = args.trialgpt_bucketing in ("interval", "both")
            do_smtcut  = args.trialgpt_bucketing in ("smt_cutoff", "both")

            # 2) TrialGPT-native interval buckets
            if do_interval:
                for b in MNO:
                    items_b = bucket_items_trialgpt_interval(tg, b)
                    items_b = apply_fused_rel_elig(items_b, fused_map)
                    met = metrics_over_items(items_b)
                    per_patient.append({
                        "patient_id": patient_id,
                        "mode": mode,
                        "system": "trialgpt_interval",
                        "bucket_name": b,
                        "prevent_tag": tg.prevent_tag,
                        "smt_cutoff_m": "",
                        "smt_cutoff_n": "",
                        "smt_cutoff_o": "",
                        **met,
                        "source_file": str(tg.source_file),
                    })
                    acc("trialgpt_interval", mode, b, met, patient_id)

            # 3) TrialGPT bucketed by SMT cutoffs (requires smt)
            if do_smtcut:
                if smt is None:
                    tg_smtcutoff_missing_smt += 1
                else:
                    mm, nn, oo = smt_cutoffs_from_run(smt)
                    for b in MNO:
                        items_b = bucket_items_trialgpt_by_smt_cutoffs(tg, mm, nn, oo, b)
                        items_b = apply_fused_rel_elig(items_b, fused_map)
                        met = metrics_over_items(items_b)
                        per_patient.append({
                            "patient_id": patient_id,
                            "mode": mode,
                            "system": "trialgpt_smt_cutoff",
                            "bucket_name": b,
                            "prevent_tag": tg.prevent_tag,
                            "smt_cutoff_m": mm,
                            "smt_cutoff_n": nn,
                            "smt_cutoff_o": oo,
                            **met,
                            "source_file": str(tg.source_file),
                        })
                        acc("trialgpt_smt_cutoff", mode, b, met, patient_id)

    aggregate: List[Dict[str, Any]] = []
    for (system, mode, bucket_name), met in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        aggregate.append({
            "mode": mode,
            "system": system,
            "bucket_name": bucket_name,
            "patients_evaluated": len(patients_seen.get((system, mode), set())),
            **met,
        })

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "bucket_size",
        "relevant", "eligible", "relevant_and_eligible",
        "relevant_unknown", "eligible_unknown",
        "label_all_satisfied", "label_unsatisfied_inclusion", "label_explicit_contradiction", "label_other",
        "interval_m", "interval_n", "interval_o", "interval_unknown",
    ]

    per_header = [
        "patient_id", "mode", "system", "bucket_name", "prevent_tag",
        "smt_cutoff_m", "smt_cutoff_n", "smt_cutoff_o",
        *metric_cols,
        "source_file",
    ]
    agg_header = ["mode", "system", "bucket_name", "patients_evaluated", *metric_cols]

    write_csv(out_dir / "per_patient.csv", per_header, per_patient)
    write_csv(out_dir / "aggregate.csv", agg_header, aggregate)

    print(f"[ok] wrote {out_dir / 'per_patient.csv'}")
    print(f"[ok] wrote {out_dir / 'aggregate.csv'}")
    print(f"[info] pairs_total={len(keys_all)}  missing_trialgpt={miss_tg}  missing_smt={miss_smt}")
    print(f"[info] dedup_canonical={args.dedup_canonical}  collapse_subsuffix={collapse}")
    print(f"[info] trialgpt_bucketing={args.trialgpt_bucketing}  tg_smtcutoff_missing_smt_pairs={tg_smtcutoff_missing_smt}")
    if mode_filter is not None:
        print(f"[info] mode_filter={mode_filter}")

    # helpful sanity warning
    if per_patient:
        rel_total = sum(int(r["relevant"]) for r in per_patient if isinstance(r.get("relevant"), int))
        rel_unk_total = sum(int(r["relevant_unknown"]) for r in per_patient if isinstance(r.get("relevant_unknown"), int))
        if rel_total == 0 and rel_unk_total > 0:
            print(
                "[warn] relevance appears mostly missing (relevant_unknown > 0). "
                "If your judge stores relevance under a different key, tell me the schema and I’ll add it."
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)