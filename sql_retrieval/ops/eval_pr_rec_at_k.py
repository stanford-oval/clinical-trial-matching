#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pr_rec_at_k.py

- Evaluates SYSTEM retrievals from retrieved-dir/*.txt
- Optionally evaluates BASELINE retrievals from a JSON mapping:
    { "patient_id": ["NCT....", "NCT....", ...], ... }

- Prints SYSTEM vs BASELINE side-by-side in the same line format.
- Plots curves that compare SYSTEM vs BASELINE, with x-axis = SYSTEM's
  average #trials actually fetched per patient (K≈), so "patients run out of
  trials sooner" is reflected on the x-axis.

- Keeps your original extras for SYSTEM when compose labels exist:
  * label-cutoff metrics (all_satisfied / survivors / all_labels)
  * missing_trials_pos12_all_satisfied.json, etc.
  * clean_eval missing_gold_*.json exports (SYSTEM only)
  * labeled_merged_csv / labeled_csv augmentation (SYSTEM only)

NOTE:
- Baseline label-cutoff metrics require baseline compose merged_csv labels.
  If you don't supply --baseline_compose_merged_csv_dir, baseline label-cutoff
  metrics are shown as N/A and label-cut plots are skipped.
"""

from __future__ import annotations

import argparse, csv, json, sys, re, sqlite3
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

# ---------------------------
# Utilities
# ---------------------------

def canon_nct(x: str) -> str:
    if not x:
        return ""
    x_up = str(x).strip().upper()
    m = re.search(r"(NCT\d{8})", x_up)
    return m.group(1) if m else x_up

# ---------------------------
# Ground truth loaders
# ---------------------------

def _read_gt_one(tsv_path: Path, gt_scores: Dict[str, Dict[str, int]], all_queries: Set[str]) -> None:
    with open(tsv_path, "r", encoding="utf-8") as f:
        snif = f.readline()
        f.seek(0)
        headered = ("query-id" in snif and "corpus-id" in snif and "score" in snif)
        if headered:
            reader = csv.DictReader(f, dialect=csv.excel_tab)
            for r in reader:
                q = (r.get("query-id") or "").strip()
                c = canon_nct(r.get("corpus-id") or "")
                s = int(r.get("score") or 0)
                if q and c:
                    all_queries.add(q)
                    gt_scores[q][c] = s
        else:
            reader = csv.reader(f, dialect=csv.excel_tab)
            for row in reader:
                if len(row) < 3:
                    continue
                q = (row[0] or "").strip()
                c = canon_nct(row[1] or "")
                s = int(row[2] or 0)
                if q and c:
                    all_queries.add(q)
                    gt_scores[q][c] = s

def read_ground_truth_multi(paths: List[Path]) -> Tuple[Dict[str, Dict[str, int]], Set[str]]:
    gt_scores: Dict[str, Dict[str, int]] = defaultdict(dict)
    all_queries: Set[str] = set()
    for p in paths:
        _read_gt_one(p, gt_scores, all_queries)
    return gt_scores, all_queries

# ---------------------------
# Retrieval readers
# ---------------------------

def read_retrievals(clean_dir: Path) -> Dict[str, List[str]]:
    """
    Reads ranked retrieval lists from *.txt and collapses duplicates/subcohorts per base NCT.
    Keeps only first occurrence of each base NCT.
    """
    out: Dict[str, List[str]] = {}
    if not clean_dir or not clean_dir.exists():
        return out

    for p in clean_dir.glob("*.txt"):
        qid = p.stem
        ranked_base: List[str] = []
        seen: Set[str] = set()
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.strip()
                if not tok:
                    continue
                base = canon_nct(tok)
                if not base or base in seen:
                    continue
                seen.add(base)
                ranked_base.append(base)
        out[qid] = ranked_base

    return out

def read_retrievals_json(json_path: Path) -> Dict[str, List[str]]:
    """
    Reads ranked retrieval lists from a JSON file shaped like:
      { "patient_id": ["NCT....", "NCT....", ...], ... }
    Canonicalizes IDs and collapses duplicates (keeps first).
    """
    out: Dict[str, List[str]] = {}
    if not json_path:
        return out
    if not json_path.exists():
        raise FileNotFoundError(f"baseline retrieval json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("baseline retrieval json must be a dict: {patient_id: [NCT...] }")

    for qid, items in data.items():
        qid = str(qid).strip()
        if not qid:
            continue

        if items is None:
            items = []
        if isinstance(items, dict) and "retrieved" in items:
            items = items["retrieved"]
        if not isinstance(items, list):
            continue

        ranked_base: List[str] = []
        seen: Set[str] = set()
        for tok in items:
            base = canon_nct(str(tok))
            if not base or base in seen:
                continue
            seen.add(base)
            ranked_base.append(base)

        out[qid] = ranked_base

    return out

# ---------------------------
# Compose three-way labels
# ---------------------------

def load_threeway_labels_from_merged(merged_csv_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Load compose three-way labels from merged_csv/{patient}.csv.
    labels[patient_id][canonical_nct_id] = label in:
      - all_satisfied
      - unsatisfied_inclusion
      - explicit_contradiction
    """
    labels: Dict[str, Dict[str, str]] = defaultdict(dict)
    if not merged_csv_dir or not merged_csv_dir.exists():
        return labels

    for p in merged_csv_dir.glob("*.csv"):
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                qid = (r.get("patient_id") or "").strip()
                base = canon_nct(r.get("canonical_nct_id", "") or r.get("nct_id", ""))
                lab = (r.get("label") or "").strip()
                if qid and base and lab:
                    labels[qid][base] = lab
    return labels

# ---------------------------
# Allow-list helpers
# ---------------------------

def read_patient_allowlist(path: Path) -> Set[str]:
    suf = path.suffix.lower()
    allow: Set[str] = set()
    if suf == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    allow.add(s)
    elif suf in (".csv", ".tsv"):
        dialect = csv.excel_tab if suf == ".tsv" else csv.excel
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f, dialect=dialect)
            if "patient_id" not in (r.fieldnames or []):
                raise ValueError("Allowlist CSV/TSV must have a 'patient_id' column")
            for row in r:
                s = (row.get("patient_id") or "").strip()
                if s:
                    allow.add(s)
    elif suf in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f] if suf == ".jsonl" else json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    allow.add(item.strip())
                elif isinstance(item, dict) and "patient_id" in item:
                    s = (item["patient_id"] or "").strip()
                    if s:
                        allow.add(s)
        else:
            raise ValueError("JSON allowlist must be a list of strings or objects with 'patient_id'")
    else:
        raise ValueError(f"Unsupported allowlist suffix: {suf}")
    return allow

def read_allow_from_sqlite(db: Path, sql: str) -> Set[str]:
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        cur.execute(sql)
        return {str(r[0]).strip() for r in cur.fetchall() if r and r[0] is not None}
    finally:
        con.close()

# ---------------------------
# Metrics
# ---------------------------

def pr_at_k(
    ranked: List[str],
    relevant_scores: Dict[str, int],
    ks: List[int],
) -> Tuple[Dict[int, Tuple[float, float, int, int]], int]:
    """
    Precision@k + graded Recall@k for a single query.

    For k > L (retrieved length), DO NOT penalize:
      - P@k and R@k plateau at values at L.
    """
    results: Dict[int, Tuple[float, float, int, int]] = {}

    total_rel_score = sum(v for v in relevant_scores.values() if v > 0)
    if total_rel_score <= 0:
        for k in ks:
            results[k] = (0.0, 0.0, 0, 0)
        return results, 0

    relevant_ids = set(relevant_scores.keys())
    L = len(ranked)
    ks_set = set(ks)

    hits = 0
    found_score = 0
    seen: Set[str] = set()

    for i, tid in enumerate(ranked, start=1):
        if tid in relevant_ids and tid not in seen:
            seen.add(tid)
            hits += 1
            found_score += relevant_scores[tid]

        if i in ks_set:
            p = hits / i if i > 0 else 0.0
            r = found_score / total_rel_score
            results[i] = (p, r, hits, found_score)

    if L == 0:
        for k in ks:
            results[k] = (0.0, 0.0, 0, 0)
        return results, total_rel_score

    p_L = hits / L if L > 0 else 0.0
    r_L = found_score / total_rel_score
    plateau = (p_L, r_L, hits, found_score)

    for k in ks:
        if k > L:
            results[k] = plateau
        elif k not in results:
            # Shouldn't normally happen (since we loop i=1..L), but keep safe.
            p = hits / k if k > 0 else 0.0
            r = found_score / total_rel_score
            results[k] = (p, r, hits, found_score)

    return results, total_rel_score

# ---------------------------
# Output writers
# ---------------------------

def write_outputs(policy_name: str,
                  ks: List[int],
                  per_query_rows: List[Dict[str, str]],
                  missing_map: Dict[str, List[str]],
                  outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "retrieved_count", "num_rel"] + \
                 [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks]

    with open(outdir / f"metrics_per_query_{policy_name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_query_rows)

    with open(outdir / f"missing_trials_{policy_name}.json", "w", encoding="utf-8") as f:
        json.dump(missing_map, f, ensure_ascii=False, indent=2)

def write_labeled_clean(retrieved: Dict[str, List[str]],
                        gt_scores: Dict[str, Dict[str, int]],
                        outdir: Path) -> None:
    target = outdir / "labeled_clean"
    target.mkdir(parents=True, exist_ok=True)
    for qid, ranked in retrieved.items():
        with open(target / f"{qid}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, dialect=csv.excel_tab)
            w.writerow(["NCT_ID", "gt_label", "gt_is_pos12", "gt_is_pos2"])
            gt = gt_scores.get(qid, {})
            for nct in ranked:
                lab = int(gt.get(nct, -1))
                w.writerow([nct, lab, 1 if lab in (1,2) else 0, 1 if lab == 2 else 0])

# ---------------------------
# Compose CSV augmentation (SYSTEM only)
# ---------------------------

def label_for(qid: str, nct: str, gt_scores: Dict[str, Dict[str, int]]) -> int:
    qid = (qid or "").strip()
    nct = canon_nct(nct or "")
    return int(gt_scores.get(qid, {}).get(nct, -1))

def augment_merged_csv(merged_csv_dir: Path,
                       gt_scores: Dict[str, Dict[str, int]],
                       outdir: Path,
                       retrieved_kept: Dict[str, List[str]]) -> Tuple[int, int]:
    """
    Read compose merged_csv/{patient}.csv; write labeled_merged_csv/{patient}.csv
    - keep rows whose base NCT is among kept retrieval list (if available)
    - deduplicate per (patient, base NCT), keep first
    Adds: gt_label, gt_is_pos12, gt_is_pos2
    """
    if not merged_csv_dir or not merged_csv_dir.exists():
        return (0, 0)
    target = outdir / "labeled_merged_csv"
    target.mkdir(parents=True, exist_ok=True)

    files = 0
    pos_rows = 0

    for p in merged_csv_dir.glob("*.csv"):
        files += 1
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = (reader.fieldnames or [])
            new_fields = list(fields)
            for extra in ["gt_label", "gt_is_pos12", "gt_is_pos2"]:
                if extra not in new_fields:
                    new_fields.append(extra)

            seen_per_patient: Dict[str, Set[str]] = defaultdict(set)

            for r in reader:
                qid = (r.get("patient_id") or "").strip()

                nct = (r.get("canonical_nct_id") or "").strip()
                if not re.search(r"NCT\d{8}", nct, flags=re.I):
                    sub = (r.get("sub_nct_ids") or "").split(";")
                    nct = next((canon_nct(s) for s in sub if s.strip()), "")
                base = canon_nct(nct)

                if not base:
                    continue

                kept_set = set(retrieved_kept.get(qid, []))
                if kept_set and base not in kept_set:
                    continue

                if base in seen_per_patient[qid]:
                    continue
                seen_per_patient[qid].add(base)

                lab = label_for(qid, base, gt_scores)
                if lab > 0:
                    pos_rows += 1
                r["gt_label"] = str(lab)
                r["gt_is_pos12"] = "1" if lab in (1, 2) else "0"
                r["gt_is_pos2"]  = "1" if lab == 2 else "0"
                if "canonical_nct_id" in r and r["canonical_nct_id"]:
                    r["canonical_nct_id"] = base

                rows.append(r)

        with open(target / p.name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_fields)
            w.writeheader()
            w.writerows(rows)

    return (files, pos_rows)

def augment_detailed_csv(detailed_csv_dir: Path,
                         gt_scores: Dict[str, Dict[str, int]],
                         outdir: Path,
                         retrieved_kept: Dict[str, List[str]]) -> Tuple[int, int]:
    """
    Read compose csv/{patient}.csv; write labeled_csv/{patient}.csv
    - keep rows whose base NCT is among kept retrieval list (if available)
    - deduplicate per (patient, base NCT), keep first
    Adds: gt_label, gt_is_pos12, gt_is_pos2
    """
    if not detailed_csv_dir or not detailed_csv_dir.exists():
        return (0, 0)
    target = outdir / "labeled_csv"
    target.mkdir(parents=True, exist_ok=True)

    files = 0
    pos_rows = 0

    for p in detailed_csv_dir.glob("*.csv"):
        files += 1
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = (reader.fieldnames or [])
            new_fields = list(fields)
            for extra in ["gt_label", "gt_is_pos12", "gt_is_pos2"]:
                if extra not in new_fields:
                    new_fields.append(extra)

            seen_per_patient: Dict[str, Set[str]] = defaultdict(set)

            for r in reader:
                qid = (r.get("patient_id") or "").strip()
                base = canon_nct(r.get("nct_id", ""))

                if not base:
                    continue

                kept_set = set(retrieved_kept.get(qid, []))
                if kept_set and base not in kept_set:
                    continue

                if base in seen_per_patient[qid]:
                    continue
                seen_per_patient[qid].add(base)

                lab = label_for(qid, base, gt_scores)
                if lab > 0:
                    pos_rows += 1
                r["gt_label"] = str(lab)
                r["gt_is_pos12"] = "1" if lab in (1, 2) else "0"
                r["gt_is_pos2"]  = "1" if lab == 2 else "0"
                if "nct_id" in r and r["nct_id"]:
                    r["nct_id"] = base

                rows.append(r)

        with open(target / p.name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_fields)
            w.writeheader()
            w.writerows(rows)

    return (files, pos_rows)

# ---------------------------
# Clean missing-gold exports (SYSTEM only)
# ---------------------------

def export_missing_gold_by_label_cutoff(
    gt_scores: Dict[str, Dict[str, int]],
    compose_labels: Dict[str, Dict[str, str]],
    all_queries: Set[str],
    outdir: Path,
) -> None:
    clean_root = outdir / "clean_eval"
    clean_root.mkdir(parents=True, exist_ok=True)

    ms_all: Dict[str, List[List[object]]] = {}
    ms_surv: Dict[str, List[List[object]]] = {}
    ms_all_labels: Dict[str, List[List[object]]] = {}

    for q in sorted(all_queries):
        gt_for_q = gt_scores.get(q, {})
        labels_q = compose_labels.get(q, {})

        missing_all: List[List[object]] = []
        missing_surv: List[List[object]] = []
        missing_any: List[List[object]] = []

        for nct, score in gt_for_q.items():
            lab = labels_q.get(nct)

            found_all = (lab == "all_satisfied")
            found_surv = lab in ("all_satisfied", "unsatisfied_inclusion")
            found_any = lab in ("all_satisfied", "unsatisfied_inclusion", "explicit_contradiction")

            if not found_all:
                missing_all.append([nct, int(score)])
            if not found_surv:
                missing_surv.append([nct, int(score)])
            if not found_any:
                missing_any.append([nct, int(score)])

        ms_all[q] = missing_all
        ms_surv[q] = missing_surv
        ms_all_labels[q] = missing_any

    with open(clean_root / "missing_gold_all_satisfied.json", "w", encoding="utf-8") as f:
        json.dump(ms_all, f, ensure_ascii=False, indent=2)
    with open(clean_root / "missing_gold_survivors.json", "w", encoding="utf-8") as f:
        json.dump(ms_surv, f, ensure_ascii=False, indent=2)
    with open(clean_root / "missing_gold_all_labels.json", "w", encoding="utf-8") as f:
        json.dump(ms_all_labels, f, ensure_ascii=False, indent=2)

# ---------------------------
# Curves + plotting (x-axis = SYSTEM Kavg)
# ---------------------------

def curves_from_acc(
    ks: List[int],
    macro_acc: Dict[int, Dict[str, Any]],
    global_found: Dict[int, int],
    global_R: int,
    k_sum: Dict[int, int],
    k_nq: Dict[int, int],
) -> Dict[str, List[float]]:
    P, R, Kavg, Nq = [], [], [], []
    for k in ks:
        n = int(macro_acc[k]["n"])
        p = (float(macro_acc[k]["p_sum"]) / n) if n > 0 else 0.0
        r = (float(global_found[k]) / float(global_R)) if global_R > 0 else 0.0
        k_avg = (float(k_sum[k]) / float(k_nq[k])) if k_nq[k] > 0 else 0.0
        P.append(p); R.append(r); Kavg.append(k_avg); Nq.append(float(n))
    return {"P": P, "R": R, "Kavg": Kavg, "Nq": Nq}

def _compress_xy(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    """
    Kavg often plateaus -> repeated x values. Compress by keeping the *last* y for each x.
    Assumes x is non-decreasing.
    """
    out_x, out_y = [], []
    last_x = None
    for xi, yi in zip(x, y):
        if last_x is None or xi != last_x:
            out_x.append(xi)
            out_y.append(yi)
            last_x = xi
        else:
            out_y[-1] = yi
    return out_x, out_y

def plot_compare_runs_kavg_x(
    run_summaries: Dict[str, Dict[str, Dict[str, List[float]]]],
    outdir: Path,
    tag: str,
    x_run_name: str = "SYSTEM",
) -> None:
    """
    Compare curves using x-axis = Kavg from `x_run_name` (SYSTEM).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] plotting unavailable (matplotlib import failed): {e}", file=sys.stderr)
        return

    if x_run_name not in run_summaries:
        raise ValueError(f"x_run_name={x_run_name} not in run_summaries keys={list(run_summaries.keys())}")

    def one_plot(metric: str, what: str, ylabel: str) -> None:
        x = run_summaries[x_run_name][metric]["Kavg"]

        plt.figure()
        for run_name, summ in run_summaries.items():
            y = summ[metric][what]
            if len(y) != len(x):
                raise ValueError(
                    f"Length mismatch for {run_name}/{metric}/{what}: len(y)={len(y)} vs len(x)={len(x)}"
                )
            xx, yy = _compress_xy(x, y)
            plt.plot(xx, yy, marker="o", linewidth=1, label=run_name)

        plt.xlabel(f"Avg # trials fetched per patient (K≈ from {x_run_name})")
        plt.ylabel(ylabel)
        plt.title(f"{metric} {ylabel} vs avg fetched trials ({tag})")
        plt.legend()
        plt.grid(True, which="both", linestyle=":", linewidth=0.5)

        plt.savefig(outdir / f"{tag}_{metric}_{what}_vs_Kavg.png", bbox_inches="tight", dpi=200)
        plt.savefig(outdir / f"{tag}_{metric}_{what}_vs_Kavg.pdf", bbox_inches="tight")
        plt.close()

    one_plot("pos12", "P", "Macro Precision")
    one_plot("pos12", "R", "Global graded Recall")
    one_plot("pos2",  "P", "Macro Precision")
    one_plot("pos2",  "R", "Global graded Recall")

# ---------------------------
# One-run evaluation
# ---------------------------

def run_eval(
    run_name: str,
    retrieved: Dict[str, List[str]],
    gt_scores: Dict[str, Dict[str, int]],
    all_queries: Set[str],
    ks: List[int],
    compose_labels: Dict[str, Dict[str, str]],
    outdir: Path,
) -> Dict[str, Any]:
    rows_pos12: List[Dict[str, str]] = []
    rows_pos2:  List[Dict[str, str]] = []
    miss_pos12: Dict[str, List[str]] = {}
    miss_pos2:  Dict[str, List[str]] = {}

    # extra missing maps under label-restricted subsets
    miss_pos12_all_satisfied: Dict[str, List[str]] = {}
    miss_pos2_all_satisfied:  Dict[str, List[str]] = {}
    miss_pos12_survivors:     Dict[str, List[str]] = {}
    miss_pos2_survivors:      Dict[str, List[str]] = {}

    macro_pos12 = {k: {"p_sum": 0.0, "n": 0} for k in ks}
    macro_pos2  = {k: {"p_sum": 0.0, "n": 0} for k in ks}

    global_R_pos12 = 0
    global_R_pos2  = 0
    global_found_pos12 = {k: 0 for k in ks}
    global_found_pos2  = {k: 0 for k in ks}

    k_sum_pos12 = {k: 0 for k in ks}
    k_nq_pos12  = {k: 0 for k in ks}
    k_sum_pos2  = {k: 0 for k in ks}
    k_nq_pos2   = {k: 0 for k in ks}

    have_labels = bool(compose_labels)

    label_cuts = {
        "all_satisfied": {"all_satisfied"},
        "survivors": {"all_satisfied", "unsatisfied_inclusion"},
        "all_labels": {"all_satisfied", "unsatisfied_inclusion", "explicit_contradiction"},
    }

    macro_pos12_cut = {name: {k: {"p_sum": 0.0, "n": 0} for k in ks} for name in label_cuts}
    macro_pos2_cut  = {name: {k: {"p_sum": 0.0, "n": 0} for k in ks} for name in label_cuts}

    global_R_pos12_cut = {name: 0 for name in label_cuts}
    global_R_pos2_cut  = {name: 0 for name in label_cuts}

    global_found_pos12_cut = {name: {k: 0 for k in ks} for name in label_cuts}
    global_found_pos2_cut  = {name: {k: 0 for k in ks} for name in label_cuts}

    k_sum_pos12_cut = {name: {k: 0 for k in ks} for name in label_cuts}
    k_nq_pos12_cut  = {name: {k: 0 for k in ks} for name in label_cuts}
    k_sum_pos2_cut  = {name: {k: 0 for k in ks} for name in label_cuts}
    k_nq_pos2_cut   = {name: {k: 0 for k in ks} for name in label_cuts}

    for q in sorted(all_queries):
        gt = gt_scores.get(q, {})
        ranked = retrieved.get(q, [])
        L = len(ranked)

        rel12_scores = {cid: s for cid, s in gt.items() if s in (1, 2)}
        rel2_scores  = {cid: s for cid, s in gt.items() if s == 2}
        rel12_ids = set(rel12_scores.keys())
        rel2_ids  = set(rel2_scores.keys())
        ranked_set = set(ranked)

        miss_pos12[q] = sorted(rel12_ids - ranked_set)
        miss_pos2[q]  = sorted(rel2_ids  - ranked_set)

        # label-restricted missing (only meaningful if labels exist)
        if have_labels:
            labels_q = compose_labels.get(q, {})
            ranked_all_satisfied = [tid for tid in ranked if labels_q.get(tid) == "all_satisfied"]
            ranked_survivors = [tid for tid in ranked if labels_q.get(tid) in ("all_satisfied", "unsatisfied_inclusion")]

            set_all = set(ranked_all_satisfied)
            set_surv = set(ranked_survivors)

            miss_pos12_all_satisfied[q] = sorted(rel12_ids - set_all)
            miss_pos2_all_satisfied[q]  = sorted(rel2_ids  - set_all)
            miss_pos12_survivors[q]     = sorted(rel12_ids - set_surv)
            miss_pos2_survivors[q]      = sorted(rel2_ids  - set_surv)
        else:
            miss_pos12_all_satisfied[q] = list(miss_pos12[q])
            miss_pos2_all_satisfied[q]  = list(miss_pos2[q])
            miss_pos12_survivors[q]     = list(miss_pos12[q])
            miss_pos2_survivors[q]      = list(miss_pos2[q])

        # ---- no-cut pos12 ----
        if rel12_scores:
            pr12, total_rel_12 = pr_at_k(ranked, rel12_scores, ks)
            global_R_pos12 += total_rel_12
            for k in ks:
                k_sum_pos12[k] += min(k, L)
                k_nq_pos12[k]  += 1
        else:
            pr12 = {k: (0.0, 0.0, 0, 0) for k in ks}

        row12 = {"query_id": q, "retrieved_count": str(L), "num_rel": str(len(rel12_scores))}
        for k in ks:
            p, r, _, found_score = pr12[k]
            row12[f"P@{k}"] = f"{p:.6f}"
            row12[f"R@{k}"] = f"{r:.6f}"
            if rel12_scores:
                macro_pos12[k]["p_sum"] += p
                macro_pos12[k]["n"] += 1
                global_found_pos12[k] += found_score
        rows_pos12.append(row12)

        # ---- no-cut pos2 ----
        if rel2_scores:
            pr2, total_rel_2 = pr_at_k(ranked, rel2_scores, ks)
            global_R_pos2 += total_rel_2
            for k in ks:
                k_sum_pos2[k] += min(k, L)
                k_nq_pos2[k]  += 1
        else:
            pr2 = {k: (0.0, 0.0, 0, 0) for k in ks}

        row2 = {"query_id": q, "retrieved_count": str(L), "num_rel": str(len(rel2_scores))}
        for k in ks:
            p, r, _, found_score = pr2[k]
            row2[f"P@{k}"] = f"{p:.6f}"
            row2[f"R@{k}"] = f"{r:.6f}"
            if rel2_scores:
                macro_pos2[k]["p_sum"] += p
                macro_pos2[k]["n"] += 1
                global_found_pos2[k] += found_score
        rows_pos2.append(row2)

        # ---- label-cutoffs (only if labels exist) ----
        if have_labels:
            labels_q = compose_labels.get(q, {})
            ranked_by_cut = {
                "all_satisfied": [tid for tid in ranked if labels_q.get(tid) == "all_satisfied"],
                "survivors": [tid for tid in ranked if labels_q.get(tid) in ("all_satisfied", "unsatisfied_inclusion")],
                "all_labels": [tid for tid in ranked if labels_q.get(tid) in ("all_satisfied", "unsatisfied_inclusion", "explicit_contradiction")],
            }

            for cut_name in ["all_satisfied", "survivors", "all_labels"]:
                ranked_cut = ranked_by_cut[cut_name]
                L_cut = len(ranked_cut)

                if rel12_scores:
                    pr12_cut, total_rel_12_cut = pr_at_k(ranked_cut, rel12_scores, ks)
                    global_R_pos12_cut[cut_name] += total_rel_12_cut
                    for k in ks:
                        p, _, _, found_score = pr12_cut[k]
                        macro_pos12_cut[cut_name][k]["p_sum"] += p
                        macro_pos12_cut[cut_name][k]["n"] += 1
                        global_found_pos12_cut[cut_name][k] += found_score
                        k_sum_pos12_cut[cut_name][k] += min(k, L_cut)
                        k_nq_pos12_cut[cut_name][k]  += 1

                if rel2_scores:
                    pr2_cut, total_rel_2_cut = pr_at_k(ranked_cut, rel2_scores, ks)
                    global_R_pos2_cut[cut_name] += total_rel_2_cut
                    for k in ks:
                        p, _, _, found_score = pr2_cut[k]
                        macro_pos2_cut[cut_name][k]["p_sum"] += p
                        macro_pos2_cut[cut_name][k]["n"] += 1
                        global_found_pos2_cut[cut_name][k] += found_score
                        k_sum_pos2_cut[cut_name][k] += min(k, L_cut)
                        k_nq_pos2_cut[cut_name][k]  += 1

    # write per-run outputs
    outdir.mkdir(parents=True, exist_ok=True)
    write_outputs("pos12", ks, rows_pos12, miss_pos12, outdir)
    write_outputs("pos2",  ks, rows_pos2,  miss_pos2,  outdir)

    # label-restricted missing JSONs (written for parity; meaningful only if labels exist)
    with open(outdir / "missing_trials_pos12_all_satisfied.json", "w", encoding="utf-8") as f:
        json.dump(miss_pos12_all_satisfied, f, ensure_ascii=False, indent=2)
    with open(outdir / "missing_trials_pos2_all_satisfied.json", "w", encoding="utf-8") as f:
        json.dump(miss_pos2_all_satisfied, f, ensure_ascii=False, indent=2)
    with open(outdir / "missing_trials_pos12_survivors.json", "w", encoding="utf-8") as f:
        json.dump(miss_pos12_survivors, f, ensure_ascii=False, indent=2)
    with open(outdir / "missing_trials_pos2_survivors.json", "w", encoding="utf-8") as f:
        json.dump(miss_pos2_survivors, f, ensure_ascii=False, indent=2)

    write_labeled_clean(retrieved, gt_scores, outdir)

    no_cut = {
        "pos12": curves_from_acc(ks, macro_pos12, global_found_pos12, global_R_pos12, k_sum_pos12, k_nq_pos12),
        "pos2":  curves_from_acc(ks, macro_pos2,  global_found_pos2,  global_R_pos2,  k_sum_pos2,  k_nq_pos2),
    }

    cuts: Dict[str, Any] = {}
    if have_labels:
        for cut_name in ["all_satisfied", "survivors", "all_labels"]:
            cuts[cut_name] = {
                "pos12": curves_from_acc(ks,
                                         macro_pos12_cut[cut_name],
                                         global_found_pos12_cut[cut_name],
                                         global_R_pos12_cut[cut_name],
                                         k_sum_pos12_cut[cut_name],
                                         k_nq_pos12_cut[cut_name]),
                "pos2":  curves_from_acc(ks,
                                         macro_pos2_cut[cut_name],
                                         global_found_pos2_cut[cut_name],
                                         global_R_pos2_cut[cut_name],
                                         k_sum_pos2_cut[cut_name],
                                         k_nq_pos2_cut[cut_name]),
            }

    return {
        "run_name": run_name,
        "outdir": str(outdir),
        "no_cut": no_cut,
        "cuts": cuts,
        "have_labels": have_labels,
    }

# ---------------------------
# Side-by-side printing
# ---------------------------

def fmt_compare_line(ks: List[int], a: Dict[str, List[float]], b: Dict[str, List[float]] | None) -> str:
    parts = []
    for i, k in enumerate(ks):
        Ka = a["Kavg"][i]; Pa = a["P"][i]; Ra = a["R"][i]; Nqa = int(a["Nq"][i])
        if b is None:
            parts.append(f"K≈{Ka:.1f}: P={Pa:.4f}, R={Ra:.4f}, Nq={Nqa}")
        else:
            Kb = b["Kavg"][i]; Pb = b["P"][i]; Rb = b["R"][i]
            parts.append(f"K≈{Ka:.1f}/{Kb:.1f}: P={Pa:.4f}/{Pb:.4f}, R={Ra:.4f}/{Rb:.4f}, Nq={Nqa}")
    return " | ".join(parts)

def print_side_by_side(
    ks: List[int],
    sys_sum: Dict[str, Any],
    base_sum: Dict[str, Any] | None,
) -> None:
    if base_sum is None:
        print(f"\n==================== {sys_sum['run_name']} ====================")
        print("== Macro P / Global (micro) graded Recall over queries with at least one positive ==")
        print("[pos12: scores {1,2}] ", fmt_compare_line(ks, sys_sum["no_cut"]["pos12"], None))
        print("[pos2 : score {2}  ] ", fmt_compare_line(ks, sys_sum["no_cut"]["pos2"], None))

        if sys_sum.get("have_labels"):
            print("\n== Label-cutoff metrics (Compose three-way labels) ==")
            for cut_name in ["all_satisfied", "survivors", "all_labels"]:
                print(f"[pos12 | {cut_name}] ", fmt_compare_line(ks, sys_sum["cuts"][cut_name]["pos12"], None))
                print(f"[pos2  | {cut_name}] ", fmt_compare_line(ks, sys_sum["cuts"][cut_name]["pos2"], None))
                print("=======================================")
        return

    print(f"\n==================== {sys_sum['run_name']}  vs  {base_sum['run_name']} ====================")
    print("== Macro P / Global (micro) graded Recall over queries with at least one positive ==")
    print("[pos12: scores {1,2}] ",
          fmt_compare_line(ks, sys_sum["no_cut"]["pos12"], base_sum["no_cut"]["pos12"]))
    print("[pos2 : score {2}  ] ",
          fmt_compare_line(ks, sys_sum["no_cut"]["pos2"],  base_sum["no_cut"]["pos2"]))

    print("\n== Label-cutoff metrics (Compose three-way labels) ==")
    for cut_name in ["all_satisfied", "survivors", "all_labels"]:
        sys_has = sys_sum.get("have_labels") and cut_name in sys_sum.get("cuts", {})
        base_has = base_sum.get("have_labels") and cut_name in base_sum.get("cuts", {})

        if sys_has and base_has:
            print(f"[pos12 | {cut_name}] ",
                  fmt_compare_line(ks, sys_sum["cuts"][cut_name]["pos12"], base_sum["cuts"][cut_name]["pos12"]))
            print(f"[pos2  | {cut_name}] ",
                  fmt_compare_line(ks, sys_sum["cuts"][cut_name]["pos2"],  base_sum["cuts"][cut_name]["pos2"]))
        elif sys_has and not base_has:
            print(f"[pos12 | {cut_name}] (baseline N/A: no compose labels for baseline)")
            print("  ", fmt_compare_line(ks, sys_sum["cuts"][cut_name]["pos12"], None))
            print(f"[pos2  | {cut_name}] (baseline N/A: no compose labels for baseline)")
            print("  ", fmt_compare_line(ks, sys_sum["cuts"][cut_name]["pos2"], None))
        else:
            print(f"[{cut_name}] N/A (no compose labels)")
        print("=======================================")

# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ground-truth", required=True, type=Path, action="append",
                    help="One or more GT TSVs (pass multiple --ground-truth flags).")

    ap.add_argument("--retrieved-dir", type=Path, default=Path("retrieved_mappings/clean"),
                    help="SYSTEM: directory of clean retrieved lists (txt).")

    ap.add_argument("--baseline-retrieval-json", type=Path, default="trialgptref/trialgpt_retrieve.json",
                    help="BASELINE: JSON mapping {patient_id: [NCT...,...]}")

    ap.add_argument("--compose_merged_csv_dir", type=Path, default=Path("./retrieved_mappings/merged_csv"),
                    help="SYSTEM: compose merged_csv dir (enables label-cutoffs).")
    ap.add_argument("--compose_detailed_csv_dir", type=Path, default=Path("./retrieved_mappings/csv"),
                    help="SYSTEM: compose detailed csv dir (for labeled_csv augmentation).")

    ap.add_argument("--baseline_compose_merged_csv_dir", type=Path, default=None,
                    help="BASELINE: compose merged_csv dir (optional; enables baseline label-cutoffs).")

    ap.add_argument("--k", default="5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,250,300,350,400,450,500,1000")
    ap.add_argument("--out", type=Path, default=Path("eval_out"))

    # Cohort restriction (matches your original behavior)
    ap.add_argument("--demographics-sqlite", type=Path, default=Path("../../build/trial.db"))
    ap.add_argument("--demographics-sql", type=str,
                    default="SELECT DISTINCT patient_id FROM patient_demographic_constraints")
    ap.add_argument("--patient-allowlist", type=Path, default=None)

    # Plotting
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-label-cuts", action="store_true",
                    help="Also plot label-cutoff curves, only if BOTH runs have labels.")

    args = ap.parse_args()
    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})

    gt_scores, all_queries = read_ground_truth_multi(args.ground_truth)

    # cohort allow
    allow: Set[str] = set()
    if args.demographics_sqlite:
        try:
            allow |= read_allow_from_sqlite(args.demographics_sqlite, args.demographics_sql)
        except Exception as e:
            print(f"[warn] demographics cohort restriction skipped: {e}", file=sys.stderr)
    if args.patient_allowlist:
        allow |= read_patient_allowlist(args.patient_allowlist)

    if allow:
        all_queries = {q for q in all_queries if q in allow}
    elif args.demographics_sqlite or args.patient_allowlist:
        print("[warn] cohort restriction produced 0 patients; no patients will be evaluated", file=sys.stderr)

    # SYSTEM
    system_retrieved = read_retrievals(args.retrieved_dir)
    if allow:
        system_retrieved = {q: r for q, r in system_retrieved.items() if q in allow}

    system_labels = load_threeway_labels_from_merged(Path(args.compose_merged_csv_dir)) if args.compose_merged_csv_dir else {}

    system_sum = run_eval(
        run_name="SYSTEM",
        retrieved=system_retrieved,
        gt_scores=gt_scores,
        all_queries=all_queries,
        ks=ks,
        compose_labels=system_labels,
        outdir=args.out,
    )

    # SYSTEM augmentations + clean_eval exports (keep your original behavior)
    merged_summary = (0, 0)
    detailed_summary = (0, 0)

    if args.compose_merged_csv_dir and Path(args.compose_merged_csv_dir).exists():
        merged_summary = augment_merged_csv(Path(args.compose_merged_csv_dir), gt_scores, args.out, system_retrieved)
        print(f"[system] wrote {args.out/'labeled_merged_csv'}  (files={merged_summary[0]}, rows_labeled>0={merged_summary[1]})")

    if args.compose_detailed_csv_dir and Path(args.compose_detailed_csv_dir).exists():
        detailed_summary = augment_detailed_csv(Path(args.compose_detailed_csv_dir), gt_scores, args.out, system_retrieved)
        print(f"[system] wrote {args.out/'labeled_csv'}        (files={detailed_summary[0]}, rows_labeled>0={detailed_summary[1]})")

    if system_labels:
        export_missing_gold_by_label_cutoff(gt_scores, system_labels, all_queries, args.out)
        print(f"[system] wrote {args.out/'clean_eval'}/*missing_gold*.json")
    else:
        print("[warn] compose three-way labels not available; skipping clean_eval missing_gold exports", file=sys.stderr)

    # BASELINE (optional)
    baseline_sum = None
    baseline_retrieved: Dict[str, List[str]] = {}
    if args.baseline_retrieval_json:
        baseline_retrieved = read_retrievals_json(args.baseline_retrieval_json)
        if allow:
            baseline_retrieved = {q: r for q, r in baseline_retrieved.items() if q in allow}

        baseline_labels: Dict[str, Dict[str, str]] = {}
        if args.baseline_compose_merged_csv_dir:
            baseline_labels = load_threeway_labels_from_merged(Path(args.baseline_compose_merged_csv_dir))

        baseline_sum = run_eval(
            run_name="BASELINE(trialgptref)",
            retrieved=baseline_retrieved,
            gt_scores=gt_scores,
            all_queries=all_queries,
            ks=ks,
            compose_labels=baseline_labels,
            outdir=args.out / "trialgptref_baseline",
        )

    # Side-by-side printing (what you asked for)
    print_side_by_side(ks, system_sum, baseline_sum)

    # Plots (compare SYSTEM vs BASELINE; x-axis = SYSTEM Kavg)
    if not args.no_plot:
        plots_dir = args.out / "plots"
        if baseline_sum is not None:
            run_summaries_no_cut = {
                "SYSTEM": system_sum["no_cut"],
                "BASELINE(trialgptref)": baseline_sum["no_cut"],
            }
            plot_compare_runs_kavg_x(run_summaries_no_cut, plots_dir, tag="no_cut_compare", x_run_name="SYSTEM")
            print(f"[plots] wrote {plots_dir}/*no_cut_compare*.png/.pdf")
        else:
            print("[plots] baseline not provided; skipping compare plots")

        if args.plot_label_cuts:
            if baseline_sum is not None and system_sum.get("have_labels") and baseline_sum.get("have_labels"):
                for cut_name in ["all_satisfied", "survivors", "all_labels"]:
                    run_summaries_cut = {
                        "SYSTEM": system_sum["cuts"][cut_name],
                        "BASELINE(trialgptref)": baseline_sum["cuts"][cut_name],
                    }
                    plot_compare_runs_kavg_x(run_summaries_cut, plots_dir, tag=f"{cut_name}_compare", x_run_name="SYSTEM")
                print(f"[plots] wrote {plots_dir}/*_compare*.png/.pdf")
            else:
                print("[plots] skip label-cut compare plots (need compose labels for BOTH runs)")

    # Output summary
    print("\nWrote:")
    print(f"  {args.out/'metrics_per_query_pos12.csv'}")
    print(f"  {args.out/'missing_trials_pos12.json'}")
    print(f"  {args.out/'metrics_per_query_pos2.csv'}")
    print(f"  {args.out/'missing_trials_pos2.json'}")
    print(f"  {args.out/'missing_trials_pos12_all_satisfied.json'}")
    print(f"  {args.out/'missing_trials_pos2_all_satisfied.json'}")
    print(f"  {args.out/'missing_trials_pos12_survivors.json'}")
    print(f"  {args.out/'missing_trials_pos2_survivors.json'}")
    print(f"  {args.out/'labeled_clean'}")

    if baseline_sum is not None:
        bdir = Path(baseline_sum["outdir"])
        print(f"  {bdir/'metrics_per_query_pos12.csv'}")
        print(f"  {bdir/'missing_trials_pos12.json'}")
        print(f"  {bdir/'metrics_per_query_pos2.csv'}")
        print(f"  {bdir/'missing_trials_pos2.json'}")
        print(f"  {bdir/'labeled_clean'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
