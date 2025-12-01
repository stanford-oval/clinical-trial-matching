#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_tiers_by_label_with_weighted_qrels.py

Tiered comparison between SMT and TrialGPT with qrels-faithful recall.

Always computes SMT buckets by exact labels:
  - m := all_satisfied
  - n := unsatisfied_inclusion
  - o := explicit_contradiction

TrialGPT bucketing options:
  - interval: use interval_mno == m/n/o
  - smt_cutoff: use SMT per-patient cutoffs to segment TrialGPT ranked list
  - both

For each bucketing scheme, compute cumulative selected sets:
  - m
  - m+n
  - m+n+o

And for each cumulative set compute:
  1) only2 recall
  2) weighted(1,2) recall

Outputs:
  <out>/per_patient.csv
  <out>/aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_TRIALGPT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out")
DEFAULT_SMT_ROOT      = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")
DEFAULT_TRIAL_CORPUS  = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus_real.jsonl")
DEFAULT_QRELS         = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/qrels/test.tsv")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)

SMT_LABELS = ("all_satisfied", "unsatisfied_inclusion", "explicit_contradiction")
MNO = ("m", "n", "o")
KNOWN_MODES = ("chief", "ccr", "all", "all-explore")


# ----------------------------
# Helpers: truthiness / OR
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


def load_qrels_weights(
    qrels_path: Path,
    collapse_subsuffix: bool,
    allowed_trial_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, int]]:
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
    mode = (mode_from_obj or mode_from_path)
    if mode_from_path == "all-explore":
        mode = "all-explore"

    trials = extract_ranked_list(obj)
    if not trials:
        return None

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

    return PatientRun(patient_id=patient_id, mode=mode, items=items, source_file=p) if items else None


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
# Fused rel/elig map
# ----------------------------

def build_fused_rel_elig_map(
    tg: Optional[PatientRun],
    smt: Optional[PatientRun],
) -> Dict[str, Tuple[Optional[bool], Optional[bool]]]:
    fused: Dict[str, Tuple[Optional[bool], Optional[bool]]] = {}

    def ingest(run: PatientRun) -> None:
        for key, rel, elig, _, _ in run.items:
            if key not in fused:
                fused[key] = (rel, elig)
            else:
                pr, pe = fused[key]
                fused[key] = (or3(pr, rel), or3(pe, elig))

    if tg is not None:
        ingest(tg)
    if smt is not None:
        ingest(smt)

    return fused


def apply_fused_rel_elig(
    items: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]],
    fused_map: Dict[str, Tuple[Optional[bool], Optional[bool]]],
) -> List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]]:
    out: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]] = []
    for key, rel, elig, lab, mno in items:
        if key in fused_map:
            fr, fe = fused_map[key]
            out.append((key, fr, fe, lab, mno))
        else:
            out.append((key, rel, elig, lab, mno))
    return out


# ----------------------------
# Bucketing
# ----------------------------

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
        return top[max(0, m): max(0, n)]
    if bucket_name == "o":
        return top[max(0, n): max(0, o)]
    return []


# ----------------------------
# Recall helpers
# ----------------------------

def compute_only2_recall(selected: Set[str], qrels_scores: Dict[str, int]) -> Tuple[int, int, Optional[float]]:
    gold2 = {tid for tid, s in qrels_scores.items() if s == 2}
    denom = len(gold2)
    num = len(selected & gold2)
    rec = (num / denom) if denom > 0 else None
    return num, denom, rec


def compute_weighted12_recall(selected: Set[str], qrels_scores: Dict[str, int]) -> Tuple[int, int, Optional[float]]:
    denom = 0
    num = 0
    for tid, s in qrels_scores.items():
        w = 2 if s == 2 else 1
        denom += w
        if tid in selected:
            num += w
    rec = (num / denom) if denom > 0 else None
    return num, denom, rec


def union_keys(
    items: List[Tuple[str, Optional[bool], Optional[bool], Optional[str], Optional[str]]]
) -> Set[str]:
    return {it[0] for it in items}


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


def add_record(
    rows: List[Dict[str, Any]],
    *,
    patient_id: str,
    mode: str,
    system: str,
    cumulative_tier: str,
    selected_set: Set[str],
    qrels_scores: Dict[str, int],
    source_file: str,
) -> None:
    hit_only2, denom_only2, recall_only2 = compute_only2_recall(selected_set, qrels_scores)
    hit_w12, denom_w12, recall_w12 = compute_weighted12_recall(selected_set, qrels_scores)

    rows.append({
        "patient_id": patient_id,
        "mode": mode,
        "system": system,
        "cumulative_tier": cumulative_tier,
        "num_gold_qrels_1or2": len(qrels_scores),
        "num_gold_qrels_2": sum(1 for s in qrels_scores.values() if s == 2),
        "selected_size": len(selected_set),
        "qrels_hit_only2": hit_only2,
        "qrels_denom_only2": denom_only2,
        "qrels_recall_only2": f"{recall_only2:.6f}" if recall_only2 is not None else "",
        "qrels_hit_w12": hit_w12,
        "qrels_denom_w12": denom_w12,
        "qrels_recall_w12": f"{recall_w12:.6f}" if recall_w12 is not None else "",
        "source_file": source_file,
    })


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tiered SMT vs TrialGPT comparison with weighted qrels recall."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--trial-corpus", type=Path, default=DEFAULT_TRIAL_CORPUS)
    ap.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    ap.add_argument("--out", type=Path, default=Path("./tier_compare_with_weighted_qrels_out"))
    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")
    ap.add_argument(
        "--trialgpt-bucketing",
        choices=("interval", "smt_cutoff", "both"),
        default="both",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Optional comma-separated modes filter.",
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

    allowed_trial_ids = load_allowed_trial_ids_from_corpus(
        args.trial_corpus,
        collapse_subsuffix=collapse,
    )
    qrels_scores_by_patient = load_qrels_weights(
        args.qrels,
        collapse_subsuffix=collapse,
        allowed_trial_ids=allowed_trial_ids,
    )

    tg_runs = collect_runs(args.trialgpt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)
    smt_runs = collect_runs(args.smt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)

    if mode_filter is not None:
        tg_runs = {k: v for k, v in tg_runs.items() if k[1] in mode_filter}
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}

    keys_all = sorted(set(tg_runs.keys()) | set(smt_runs.keys()))
    if args.require_overlap:
        keys_all = sorted(set(tg_runs.keys()) & set(smt_runs.keys()))

    per_patient: List[Dict[str, Any]] = []

    for (patient_id, mode) in keys_all:
        tg = tg_runs.get((patient_id, mode))
        smt = smt_runs.get((patient_id, mode))
        gold_scores = qrels_scores_by_patient.get(patient_id, {})
        fused_map = build_fused_rel_elig_map(tg, smt)

        if smt is not None:
            smt_items = apply_fused_rel_elig(smt.items, fused_map)

            smt_m = [it for it in smt_items if it[3] == "all_satisfied" and it[0] in allowed_trial_ids]
            smt_n = [it for it in smt_items if it[3] == "unsatisfied_inclusion" and it[0] in allowed_trial_ids]
            smt_o = [it for it in smt_items if it[3] == "explicit_contradiction" and it[0] in allowed_trial_ids]

            smt_m_set = union_keys(smt_m)
            smt_mn_set = union_keys(smt_m + smt_n)
            smt_mno_set = union_keys(smt_m + smt_n + smt_o)

            add_record(
                per_patient,
                patient_id=patient_id,
                mode=mode,
                system="smt",
                cumulative_tier="m",
                selected_set=smt_m_set,
                qrels_scores=gold_scores,
                source_file=str(smt.source_file),
            )
            add_record(
                per_patient,
                patient_id=patient_id,
                mode=mode,
                system="smt",
                cumulative_tier="m+n",
                selected_set=smt_mn_set,
                qrels_scores=gold_scores,
                source_file=str(smt.source_file),
            )
            add_record(
                per_patient,
                patient_id=patient_id,
                mode=mode,
                system="smt",
                cumulative_tier="m+n+o",
                selected_set=smt_mno_set,
                qrels_scores=gold_scores,
                source_file=str(smt.source_file),
            )

        if tg is not None:
            tg_items = apply_fused_rel_elig(tg.items, fused_map)

            if args.trialgpt_bucketing in ("interval", "both"):
                tg_m = [it for it in tg_items if it[4] == "m" and it[0] in allowed_trial_ids]
                tg_n = [it for it in tg_items if it[4] == "n" and it[0] in allowed_trial_ids]
                tg_o = [it for it in tg_items if it[4] == "o" and it[0] in allowed_trial_ids]

                add_record(
                    per_patient,
                    patient_id=patient_id,
                    mode=mode,
                    system="trialgpt_interval",
                    cumulative_tier="m",
                    selected_set=union_keys(tg_m),
                    qrels_scores=gold_scores,
                    source_file=str(tg.source_file),
                )
                add_record(
                    per_patient,
                    patient_id=patient_id,
                    mode=mode,
                    system="trialgpt_interval",
                    cumulative_tier="m+n",
                    selected_set=union_keys(tg_m + tg_n),
                    qrels_scores=gold_scores,
                    source_file=str(tg.source_file),
                )
                add_record(
                    per_patient,
                    patient_id=patient_id,
                    mode=mode,
                    system="trialgpt_interval",
                    cumulative_tier="m+n+o",
                    selected_set=union_keys(tg_m + tg_n + tg_o),
                    qrels_scores=gold_scores,
                    source_file=str(tg.source_file),
                )

            if args.trialgpt_bucketing in ("smt_cutoff", "both") and smt is not None:
                mm, nn, oo = smt_cutoffs_from_run(smt)

                raw_m = bucket_items_trialgpt_by_smt_cutoffs(tg, mm, nn, oo, "m")
                raw_n = bucket_items_trialgpt_by_smt_cutoffs(tg, mm, nn, oo, "n")
                raw_o = bucket_items_trialgpt_by_smt_cutoffs(tg, mm, nn, oo, "o")

                tg_m = [it for it in apply_fused_rel_elig(raw_m, fused_map) if it[0] in allowed_trial_ids]
                tg_n = [it for it in apply_fused_rel_elig(raw_n, fused_map) if it[0] in allowed_trial_ids]
                tg_o = [it for it in apply_fused_rel_elig(raw_o, fused_map) if it[0] in allowed_trial_ids]

                add_record(
                    per_patient,
                    patient_id=patient_id,
                    mode=mode,
                    system="trialgpt_smt_cutoff",
                    cumulative_tier="m",
                    selected_set=union_keys(tg_m),
                    qrels_scores=gold_scores,
                    source_file=str(tg.source_file),
                )
                add_record(
                    per_patient,
                    patient_id=patient_id,
                    mode=mode,
                    system="trialgpt_smt_cutoff",
                    cumulative_tier="m+n",
                    selected_set=union_keys(tg_m + tg_n),
                    qrels_scores=gold_scores,
                    source_file=str(tg.source_file),
                )
                add_record(
                    per_patient,
                    patient_id=patient_id,
                    mode=mode,
                    system="trialgpt_smt_cutoff",
                    cumulative_tier="m+n+o",
                    selected_set=union_keys(tg_m + tg_n + tg_o),
                    qrels_scores=gold_scores,
                    source_file=str(tg.source_file),
                )

    aggregate_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for row in per_patient:
        k = (row["mode"], row["system"], row["cumulative_tier"])
        if k not in aggregate_map:
            aggregate_map[k] = {
                "mode": row["mode"],
                "system": row["system"],
                "cumulative_tier": row["cumulative_tier"],
                "patients": 0,

                "sum_hit_only2": 0,
                "sum_denom_only2": 0,
                "sum_hit_w12": 0,
                "sum_denom_w12": 0,

                "macro_vals_only2": [],
                "macro_vals_w12": [],
                "sum_selected_size": 0,
            }

        d = aggregate_map[k]
        d["patients"] += 1
        d["sum_hit_only2"] += int(row["qrels_hit_only2"])
        d["sum_denom_only2"] += int(row["qrels_denom_only2"])
        d["sum_hit_w12"] += int(row["qrels_hit_w12"])
        d["sum_denom_w12"] += int(row["qrels_denom_w12"])
        d["sum_selected_size"] += int(row["selected_size"])

        if row["qrels_recall_only2"] != "":
            d["macro_vals_only2"].append(float(row["qrels_recall_only2"]))
        if row["qrels_recall_w12"] != "":
            d["macro_vals_w12"].append(float(row["qrels_recall_w12"]))

    aggregate: List[Dict[str, Any]] = []
    for _, d in sorted(aggregate_map.items()):
        aggregate.append({
            "mode": d["mode"],
            "system": d["system"],
            "cumulative_tier": d["cumulative_tier"],
            "patients": d["patients"],

            "macro_qrels_recall_only2": (
                f"{sum(d['macro_vals_only2']) / len(d['macro_vals_only2']):.6f}"
                if d["macro_vals_only2"] else ""
            ),
            "micro_qrels_recall_only2": (
                f"{d['sum_hit_only2'] / d['sum_denom_only2']:.6f}"
                if d["sum_denom_only2"] > 0 else ""
            ),

            "macro_qrels_recall_w12": (
                f"{sum(d['macro_vals_w12']) / len(d['macro_vals_w12']):.6f}"
                if d["macro_vals_w12"] else ""
            ),
            "micro_qrels_recall_w12": (
                f"{d['sum_hit_w12'] / d['sum_denom_w12']:.6f}"
                if d["sum_denom_w12"] > 0 else ""
            ),

            "avg_selected_size": f"{d['sum_selected_size'] / d['patients']:.6f}" if d["patients"] > 0 else "",
        })

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_header = [
        "patient_id",
        "mode",
        "system",
        "cumulative_tier",
        "num_gold_qrels_1or2",
        "num_gold_qrels_2",
        "selected_size",
        "qrels_hit_only2",
        "qrels_denom_only2",
        "qrels_recall_only2",
        "qrels_hit_w12",
        "qrels_denom_w12",
        "qrels_recall_w12",
        "source_file",
    ]
    agg_header = [
        "mode",
        "system",
        "cumulative_tier",
        "patients",
        "macro_qrels_recall_only2",
        "micro_qrels_recall_only2",
        "macro_qrels_recall_w12",
        "micro_qrels_recall_w12",
        "avg_selected_size",
    ]

    write_csv(out_dir / "per_patient.csv", per_header, per_patient)
    write_csv(out_dir / "aggregate.csv", agg_header, aggregate)

    print(f"[ok] wrote {out_dir / 'per_patient.csv'}")
    print(f"[ok] wrote {out_dir / 'aggregate.csv'}")
    print(f"[info] qrels recall variants: only2 and weighted(1,2)")
    print(f"[info] trialgpt_bucketing={args.trialgpt_bucketing}")
    print(f"[info] collapse_subsuffix={collapse} dedup_canonical={args.dedup_canonical} require_overlap={args.require_overlap}")
    if mode_filter is not None:
        print(f"[info] mode_filter={mode_filter}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)