#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_avgcutoff_report.py

Visualize outputs from compare_smt_vs_trialgpt_avgcutoff.py (rel&elig filtered).

Inputs:
  --in-dir: directory containing per_patient.csv and summary_by_mode.csv
Outputs:
  Writes PNGs into <in-dir>/plots/ by default.

Plots:
  1) Per-mode histograms:
     - smt_found, trialgpt_found, smt_not_tg, tg_not_smt, overlap
  2) Boxplots per mode for the same metrics
  3) Scatter:
     - smt_found vs trialgpt_found (per patient), with y=x line
     - smt_not_tg vs tg_not_smt
  4) Optional: ratio distributions:
     - overlap / smt_found  (SMT hit rate)
     - overlap / trialgpt_found (TrialGPT hit rate)
  5) Summary bar chart from summary_by_mode.csv (means)

Notes:
- Uses matplotlib only (no seaborn), and does not set explicit colors.
- Works even if modes are missing (e.g., only "all").
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


METRICS = ["smt_found", "trialgpt_found", "smt_not_tg", "tg_not_smt", "overlap"]


def safe_div(a: float, b: float) -> float:
    if b == 0 or (isinstance(b, float) and (math.isnan(b))):
        return float("nan")
    return a / b


def savefig(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_histograms(df: pd.DataFrame, out_dir: Path, max_bins: int = 30) -> None:
    for mode, g in df.groupby("mode"):
        for m in METRICS:
            xs = g[m].dropna().astype(int).to_numpy()
            if xs.size == 0:
                continue
            bins = min(max_bins, max(5, len(np.unique(xs))))
            plt.figure()
            plt.hist(xs, bins=bins)
            plt.title(f"Histogram: {m} (mode={mode})")
            plt.xlabel("count")
            plt.ylabel("patients")
            savefig(out_dir / f"hist__{m}__mode_{mode}.png")


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    # One figure per metric, grouped by mode
    modes = sorted(df["mode"].dropna().unique().tolist())
    if not modes:
        return
    for m in METRICS:
        data = [df.loc[df["mode"] == md, m].dropna().astype(float).to_numpy() for md in modes]
        if all(len(x) == 0 for x in data):
            continue
        plt.figure()
        plt.boxplot(data, labels=modes, showfliers=False)
        plt.title(f"Boxplot: {m} by mode")
        plt.xlabel("mode")
        plt.ylabel("count")
        savefig(out_dir / f"box__{m}.png")


def plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    # smt_found vs trialgpt_found
    x = df["smt_found"].astype(float)
    y = df["trialgpt_found"].astype(float)
    ok = x.notna() & y.notna()
    if ok.any():
        plt.figure()
        plt.scatter(x[ok], y[ok], s=10, alpha=0.6)
        mx = max(float(x[ok].max()), float(y[ok].max()))
        plt.plot([0, mx], [0, mx], linewidth=1)
        plt.title("Per-patient: TrialGPT(top-K) vs SMT(all_satisfied) counts")
        plt.xlabel("SMT found (rel&elig)")
        plt.ylabel("TrialGPT found (rel&elig)")
        savefig(out_dir / "scatter__trialgpt_vs_smt.png")

    # smt_not_tg vs tg_not_smt
    x2 = df["smt_not_tg"].astype(float)
    y2 = df["tg_not_smt"].astype(float)
    ok2 = x2.notna() & y2.notna()
    if ok2.any():
        plt.figure()
        plt.scatter(x2[ok2], y2[ok2], s=10, alpha=0.6)
        mx = max(float(x2[ok2].max()), float(y2[ok2].max()))
        plt.plot([0, mx], [0, mx], linewidth=1)
        plt.title("Per-patient asymmetry: SMT\\TG vs TG\\SMT")
        plt.xlabel("SMT not TrialGPT (rel&elig)")
        plt.ylabel("TrialGPT not SMT (rel&elig)")
        savefig(out_dir / "scatter__asymmetry.png")


def plot_ratios(df: pd.DataFrame, out_dir: Path, max_bins: int = 30) -> None:
    # overlap ratios
    df2 = df.copy()
    df2["ratio_overlap_over_smt"] = [
        safe_div(o, s) for o, s in zip(df2["overlap"].astype(float), df2["smt_found"].astype(float))
    ]
    df2["ratio_overlap_over_tg"] = [
        safe_div(o, t) for o, t in zip(df2["overlap"].astype(float), df2["trialgpt_found"].astype(float))
    ]

    for mode, g in df2.groupby("mode"):
        for col, title in [
            ("ratio_overlap_over_smt", "Overlap / SMT_found"),
            ("ratio_overlap_over_tg", "Overlap / TrialGPT_found"),
        ]:
            xs = g[col].dropna().to_numpy()
            if xs.size == 0:
                continue
            bins = min(max_bins, max(5, len(np.unique(np.round(xs, 2)))))
            plt.figure()
            plt.hist(xs, bins=bins)
            plt.title(f"Histogram: {title} (mode={mode})")
            plt.xlabel("ratio")
            plt.ylabel("patients")
            savefig(out_dir / f"hist__{col}__mode_{mode}.png")


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    # Bar chart of mean counts by mode
    if summary.empty:
        return
    summary = summary.copy()
    summary["mode"] = summary["mode"].astype(str)
    summary = summary.sort_values("mode")

    modes = summary["mode"].tolist()
    # We'll create one bar chart per metric (means)
    mapping = {
        "avg_smt_found": "Mean SMT found",
        "avg_trialgpt_found": "Mean TrialGPT found",
        "avg_smt_not_tg": "Mean SMT not TG",
        "avg_tg_not_smt": "Mean TG not SMT",
        "avg_overlap": "Mean overlap",
    }

    for col, title in mapping.items():
        if col not in summary.columns:
            continue
        ys = summary[col].astype(float).to_numpy()
        plt.figure()
        plt.bar(modes, ys)
        plt.title(f"{title} by mode")
        plt.xlabel("mode")
        plt.ylabel("mean count")
        savefig(out_dir / f"bar__{col}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot avg-cutoff SMT vs TrialGPT report.")
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with per_patient.csv and summary_by_mode.csv")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write plots (default: <in-dir>/plots)")
    ap.add_argument("--no-ratios", action="store_true", help="Skip ratio plots")
    ap.add_argument("--max-bins", type=int, default=30, help="Max histogram bins")
    args = ap.parse_args()

    in_dir = args.in_dir
    per_path = in_dir / "per_patient.csv"
    sum_path = in_dir / "summary_by_mode.csv"

    if not per_path.exists():
        raise SystemExit(f"[error] missing {per_path}")
    df = pd.read_csv(per_path)

    # Ensure metric cols exist
    for m in METRICS:
        if m not in df.columns:
            raise SystemExit(f"[error] per_patient.csv missing column: {m}")

    out_dir = args.out_dir or (in_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_histograms(df, out_dir, max_bins=args.max_bins)
    plot_boxplots(df, out_dir)
    plot_scatter(df, out_dir)
    if not args.no_ratios:
        plot_ratios(df, out_dir, max_bins=args.max_bins)

    if sum_path.exists():
        summary = pd.read_csv(sum_path)
        plot_summary_bars(summary, out_dir)

    print(f"[ok] plots written under: {out_dir}")


if __name__ == "__main__":
    main()
