#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_rank_curves.py — Plot mean relevance and eligibility (fit-rate) vs rank.

Inputs (choose one):
  --summary-json    # saved stdout JSON from llm_judge.py
  --patient-root    # scan kits and read 1trial/llm_judge/judgment_*.json

Outputs (to --out-dir):
  - rank_curves.png            # dual-axis plot (mean relevance & fit-rate) vs rank
  - relevance_by_rank.png      # mean relevance vs rank
  - fit_rate_by_rank.png       # fit-rate vs rank
  - per_rank_aggregates.csv    # table with counts and rates

Notes:
  - Ranks are inferred from any path segment like 'rank3_*' in kit_dir.
  - Unranked kits are excluded by default (use --include-unranked to include as rank=0).
"""

from __future__ import annotations
import argparse, json, re, csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

RANK_RE = re.compile(r"(?i)\brank(\d+)_")
VALID_OUT = ("fit","mismatch","unclear")

def _canon8(nct: str) -> str:
    m = re.match(r"^(NCT\d{8})", nct or "")
    return m.group(1) if m else (nct or "")

def _infer_rank_from_path(p: str) -> Optional[int]:
    m = RANK_RE.search(p or "")
    return int(m.group(1)) if m else None

def _load_summary_json(fp: Path) -> Dict[str, Any]:
    return json.loads(fp.read_text(encoding="utf-8"))

def _scan_patient_root(patient_root: Path) -> Dict[str, Any]:
    trials: List[Dict[str, Any]] = []
    for jd in patient_root.rglob("1trial/llm_judge/judgment_*.json"):
        try:
            obj = json.loads(jd.read_text(encoding="utf-8"))
            rel = obj.get("relevance", {}) or {}
            elg = obj.get("eligibility", {}) or {}
            trials.append({
                "kit_dir": str(jd.parent.parent.parent),
                "patient_id": obj.get("patient_id"),
                "nct": _canon8(obj.get("nct", "")),
                "saved_path": str(jd),
                "relevance_score": rel.get("score"),
                "eligibility_outcome": elg.get("outcome"),
            })
        except Exception:
            pass
    return {"trials": trials}

def _filter_engine_model(trials: List[Dict[str, Any]], engine: Optional[str], model: Optional[str]) -> List[Dict[str, Any]]:
    if not engine and not model:
        return trials
    out = []
    for row in trials:
        sp = row.get("saved_path") or ""
        m = re.search(r"judgment_([^_]+)_(.+)\.json$", sp)
        e = m.group(1) if m else None
        md = m.group(2) if m else None
        if engine and e != engine: continue
        if model and md != model: continue
        out.append(row)
    return out

def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def _aggregate_by_rank(trials: List[Dict[str, Any]], include_unranked: bool) -> Dict[int, Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"scores": [], "outs": []})
    for t in trials:
        rank = _infer_rank_from_path(t.get("kit_dir",""))
        if rank is None:
            if not include_unranked:
                continue
            rank = 0  # treat unranked as 0 if requested
        s = t.get("relevance_score")
        if isinstance(s, int):
            buckets[rank]["scores"].append(s)
        o = t.get("eligibility_outcome")
        if o in VALID_OUT:
            buckets[rank]["outs"].append(o)
    # finalize
    out: Dict[int, Dict[str, Any]] = {}
    for r, b in buckets.items():
        n = max(len(b["scores"]), len(b["outs"]))
        c = Counter(b["outs"])
        denom = sum(c[o] for o in VALID_OUT) or 1
        out[r] = {
            "n": n,
            "mean_relevance": (sum(b["scores"])/len(b["scores"])) if b["scores"] else None,
            "fit_rate": c["fit"]/denom if denom else None,
            "mismatch_rate": c["mismatch"]/denom if denom else None,
            "unclear_rate": c["unclear"]/denom if denom else None,
        }
    return out

def _write_csv(agg: Dict[int, Dict[str, Any]], fp: Path) -> None:
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank","n","mean_relevance","fit_rate","mismatch_rate","unclear_rate"])
        for r in sorted(agg.keys()):
            a = agg[r]
            w.writerow([r, a["n"], a["mean_relevance"], a["fit_rate"], a["mismatch_rate"], a["unclear_rate"]])

def _plot_dual(ranks: List[int], means: List[float], fits: List[float], out_png: Path) -> None:
    fig, ax1 = plt.subplots()
    ax1.plot(ranks, means, marker="o")
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Mean relevance (0–4)")
    ax2 = ax1.twinx()
    ax2.plot(ranks, fits, marker="s")
    ax2.set_ylabel("Fit rate (0–1)")
    fig.suptitle("Relevance & Eligibility vs Rank")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def _plot_single(x: List[int], y: List[float], ylabel: str, title: str, out_png: Path) -> None:
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Rank")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Two curves over rank: mean relevance and fit-rate.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--summary-json", type=Path, help="Saved stdout JSON from llm_judge.py")
    src.add_argument("--patient-root", type=Path, help="Root with rank*/ kits")
    ap.add_argument("--engine", help="Filter by engine (e.g., gpt-4o, gpt-4.1, gpt-5)")
    ap.add_argument("--model", help="Filter by model (e.g., gpt-4o, gpt-4.1, gpt-5)")
    ap.add_argument("--out-dir", type=Path, default=Path("viz_out"))
    ap.add_argument("--include-unranked", action="store_true", help="Include unranked kits as rank=0")
    args = ap.parse_args()

    summary = _load_summary_json(args.summary_json) if args.summary_json else _scan_patient_root(args.patient_root)
    trials: List[Dict[str, Any]] = list(summary.get("trials") or [])
    if not trials:
        raise SystemExit("No trials found.")

    # engine/model filter
    trials = _filter_engine_model(trials, args.engine, args.model)
    if not trials:
        raise SystemExit("No trials left after engine/model filtering.")

    agg = _aggregate_by_rank(trials, include_unranked=args.include_unranked)
    if not agg:
        raise SystemExit("No rank information found. Ensure kit paths contain 'rankN_' or use --include-unranked.")

    _ensure_out(args.out_dir)
    _write_csv(agg, args.out_dir / "per_rank_aggregates.csv")

    # Prepare sequences (drop ranks lacking either metric)
    ranks = sorted(agg.keys())
    mean_seq = [agg[r]["mean_relevance"] for r in ranks]
    fit_seq = [agg[r]["fit_rate"] for r in ranks]

    # Filter out None entries synchronously to keep alignment
    filtered = [(r, m, f) for r, m, f in zip(ranks, mean_seq, fit_seq) if (m is not None and f is not None)]
    if not filtered:
        raise SystemExit("Not enough data to plot curves (need both relevance scores and eligibility outcomes).")

    ranks_f, means_f, fits_f = zip(*filtered)

    # Dual-axis and single-axis variants
    _plot_dual(list(ranks_f), list(means_f), list(fits_f), args.out_dir / "rank_curves.png")
    _plot_single(list(ranks_f), list(means_f), "Mean relevance (0–4)", "Mean Relevance vs Rank", args.out_dir / "relevance_by_rank.png")
    _plot_single(list(ranks_f), list(fits_f), "Fit rate (0–1)", "Eligibility (Fit Rate) vs Rank", args.out_dir / "fit_rate_by_rank.png")

    print(f"Done. Wrote plots + CSV to: {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
