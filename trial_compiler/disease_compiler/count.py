#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 TrialGPT-SMT /build/disease/ 下所有 *_disease_link_filter_summary.json：
1) linked_result 中各 disease 的 match_reason 是否 Exact；列出非 Exact 的明细。
2) final_selected_concept_by_disease 的键数量是否 < linked_result 的键数量；输出差值。

输出：
- linked_match_quality.csv ：trial_id, total_linked, exact_count, non_exact_count, non_exact_details
- final_vs_linked_counts.csv：trial_id, linked_count, final_count, delta(final-linked), reduced(bool)

用法（默认路径即可）：
    python analyze_disease_links.py
或指定目录：
    python analyze_disease_links.py <SATIR_ROOT>/build/disease
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import csv
from typing import Dict, Any, List, Tuple

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def iter_summary_files(base_dir: Path):
    # 匹配 *_disease_link_filter_summary.json
    for p in sorted(base_dir.glob("*_disease_link_filter_summary.json")):
        if p.is_file():
            yield p

def analyze_file(p: Path) -> Tuple[str, Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trial_id = data.get("trial_id") or safe_get(data, ["contextual", "_id"]) or p.stem.split("_")[0]

    linked_result = data.get("linked_result") or {}
    final_selected = data.get("final_selected_concept_by_disease") or {}

    # 统一：键是 disease 字符串；值是包含 match_reason 的 dict
    # linked_result 统计
    total_linked = 0
    exact_count = 0
    non_exact_count = 0
    non_exact_details: List[str] = []

    if isinstance(linked_result, dict):
        for disease, info in linked_result.items():
            total_linked += 1
            reason = ""
            if isinstance(info, dict):
                reason = (info.get("match_reason") or "").strip()
            is_exact = (reason.lower() == "exact")
            if is_exact:
                exact_count += 1
            else:
                non_exact_count += 1
                # 记录 disease 与 reason（为空也给出 "?"）
                non_exact_details.append(f"{disease}:{reason or '?'}")

    # final_selected 统计（只数键，值为 None 的不计入）
    final_count = 0
    if isinstance(final_selected, dict):
        for disease, info in final_selected.items():
            if info is not None:
                final_count += 1

    delta = final_count - total_linked
    reduced = final_count < total_linked

    result = {
        "trial_id": trial_id,
        "total_linked": total_linked,
        "exact_count": exact_count,
        "non_exact_count": non_exact_count,
        "non_exact_details": "; ".join(non_exact_details),
        "linked_count": total_linked,
        "final_count": final_count,
        "delta": delta,
        "reduced": reduced,
        "file": str(p),
        "title": safe_get(data, ["contextual", "title"], ""),
    }
    return trial_id, result

def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})

def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.getenv("SATIR_BUILD", "build")) / "disease"
    if not base_dir.exists():
        print(f"[ERROR] Base dir not found: {base_dir}")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for p in iter_summary_files(base_dir):
        try:
            _, res = analyze_file(p)
            results.append(res)
        except Exception as e:
            print(f"[WARN] Failed to analyze {p}: {e}")

    # 1) linked_match_quality.csv
    linked_rows = []
    for r in results:
        linked_rows.append({
            "trial_id": r["trial_id"],
            "total_linked": r["total_linked"],
            "exact_count": r["exact_count"],
            "non_exact_count": r["non_exact_count"],
            "non_exact_details": r["non_exact_details"],
        })
    write_csv(base_dir / "linked_match_quality.csv",
              linked_rows,
              ["trial_id", "total_linked", "exact_count", "non_exact_count", "non_exact_details"])

    # 2) final_vs_linked_counts.csv
    counts_rows = []
    for r in results:
        counts_rows.append({
            "trial_id": r["trial_id"],
            "linked_count": r["linked_count"],
            "final_count": r["final_count"],
            "delta": r["delta"],
            "reduced": r["reduced"],
        })
    write_csv(base_dir / "final_vs_linked_counts.csv",
              counts_rows,
              ["trial_id", "linked_count", "final_count", "delta", "reduced"])

    # 控制台汇总
    total_trials = len(results)
    trials_with_non_exact = sum(1 for r in results if r["non_exact_count"] > 0)
    trials_reduced = [r for r in results if r["reduced"]]
    print("==== Summary ====")
    print(f"Scanned trials: {total_trials}")
    print(f"Trials with non-Exact matches in linked_result: {trials_with_non_exact}")
    print(f"Trials where final_selected_concept_by_disease < linked_result: {len(trials_reduced)}")
    if trials_reduced:
        print("— Reduced trials (trial_id: delta):")
        for r in sorted(trials_reduced, key=lambda x: (x["delta"], x["trial_id"])):
            print(f"  {r['trial_id']}: {r['delta']} (final={r['final_count']}, linked={r['linked_count']})")

if __name__ == "__main__":
    main()
