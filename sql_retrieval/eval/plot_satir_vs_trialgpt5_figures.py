#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_satir_vs_trialgpt5_figures.py

Plot SatIR vs TrialGPT-5 figures from the dedicated compare output folder.

Expected input:
  <in-dir>/per_patient.csv

Default input:
  ./satir_vs_trialgpt5_avgcutoff_out/per_patient.csv

Default output:
  ./satir_vs_trialgpt5_avgcutoff_out/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


COLORS = {
    "overlap": "#9DB7D5",
    "satir_only": "#F58518",
    "tg_only": "#54A24B",
    "points": "#4C78A8",
    "diag": "#8E8E8E",
    "grid": "#D9D9D9",
    "text": "#333333",
}


def set_paper_style() -> None:
    mpl.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def light_y_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.grid(False, axis="x")


def savefig_both(out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close()


def savefig_fig_both(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def to_int_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def human_mode_name(mode: str) -> str:
    mapping = {
        "ccr": "chief-complaint-treating",
        "all": "any-condition-treating",
        "all-explore": "any-condition-relevant",
        "ALL": "All modes",
    }
    return mapping.get(str(mode), str(mode))


def mode_slug(mode: str) -> str:
    mapping = {
        "ccr": "chief-complaint-treating",
        "all": "any-condition-treating",
        "all-explore": "any-condition-relevant",
        "ALL": "all-modes",
    }
    return mapping.get(str(mode), str(mode))


def filter_both_sides_present(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "satir_source" in df.columns and "tg_source" in df.columns:
        s = df["satir_source"].fillna("").astype(str).str.strip().str.len() > 0
        t = df["tg_source"].fillna("").astype(str).str.strip().str.len() > 0
        return df.loc[s & t].copy()
    return df.loc[df["satir_found"].notna() & df["trialgpt_found"].notna()].copy()


def ensure_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"satir_found", "trialgpt_found"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] per_patient.csv missing columns: {missing}")

    if "overlap" in df.columns and "satir_not_tg" in df.columns and "tg_not_satir" in df.columns:
        df["overlap"] = df["overlap"].fillna(0).astype(int)
        df["satir_only"] = df["satir_not_tg"].fillna(0).astype(int)
        df["tg_only"] = df["tg_not_satir"].fillna(0).astype(int)
        return df

    df["overlap"] = (
        np.minimum(to_int_series(df, "satir_found"), to_int_series(df, "trialgpt_found"))
    )
    df["satir_only"] = (to_int_series(df, "satir_found") - df["overlap"]).clip(lower=0)
    df["tg_only"] = (to_int_series(df, "trialgpt_found") - df["overlap"]).clip(lower=0)
    return df


def _prepare_composition_df(df: pd.DataFrame, max_patients: int) -> pd.DataFrame:
    df = ensure_components(df)
    if df.empty:
        return df

    df = df.copy()
    df["satir_total"] = (df["overlap"] + df["satir_only"]).astype(int)
    df["tg_total"] = (df["overlap"] + df["tg_only"]).astype(int)
    df["total_union"] = (df["overlap"] + df["satir_only"] + df["tg_only"]).astype(int)
    df["max_side"] = df[["satir_total", "tg_total"]].max(axis=1)

    df = df.sort_values(
        ["max_side", "total_union", "overlap", "satir_total", "tg_total"],
        ascending=[False, False, False, False, False],
    )

    if max_patients > 0 and len(df) > max_patients:
        df = df.head(max_patients).copy()

    return df


def scatter_trialgpt_vs_satir(df: pd.DataFrame, out_base: Path, mode: str) -> None:
    x = pd.to_numeric(df["trialgpt_found"], errors="coerce")
    y = pd.to_numeric(df["satir_found"], errors="coerce")
    ok = x.notna() & y.notna()
    if not ok.any():
        return

    xs = x[ok].to_numpy()
    ys = y[ok].to_numpy()
    mx = max(float(np.max(xs)), float(np.max(ys)), 1.0)

    n_above = int(np.sum(ys > xs))
    n_below = int(np.sum(ys < xs))
    n_equal = int(np.sum(ys == xs))
    n_total = len(xs)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(xs, ys, s=34, alpha=0.82, color=COLORS["points"], linewidths=0.0)
    ax.plot([0, mx], [0, mx], linestyle="--", color=COLORS["diag"], linewidth=1.2)

    ax.set_xlim(0, 1.03 * mx)
    ax.set_ylim(0, 1.03 * mx)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"SatIR vs TrialGPT — {human_mode_name(mode)}", pad=12)
    ax.set_xlabel("# TrialGPT-Retrieved Useful Trials", labelpad=10)
    ax.set_ylabel("# SatIR-Retrieved Useful Trials", labelpad=10)

    ax.text(
        0.98, 0.03,
        f"SatIR>TrialGPT: {n_above}/{n_total}\n"
        f"Equal: {n_equal}/{n_total}\n"
        f"SatIR<TrialGPT: {n_below}/{n_total}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=COLORS["text"],
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(True, axis="both", color=COLORS["grid"], linewidth=0.7, alpha=0.55)
    despine(ax)
    savefig_both(out_base)


def per_patient_composition_paired(df: pd.DataFrame, out_base: Path, mode: str, max_patients: int) -> None:
    df_plot = _prepare_composition_df(df, max_patients=max_patients)
    if df_plot.empty:
        return

    labels = df_plot["patient_id"].astype(str).tolist()
    x = np.arange(len(df_plot), dtype=float)

    overlap = df_plot["overlap"].to_numpy()
    satir_only = df_plot["satir_only"].to_numpy()
    tg_only = df_plot["tg_only"].to_numpy()

    fig_w = max(18.0, len(df_plot) * 0.33)
    fig, ax = plt.subplots(figsize=(fig_w, 7.4))

    bar_w = 0.36
    x_satir = x - bar_w / 2
    x_tg = x + bar_w / 2

    ax.bar(x_satir, overlap, width=bar_w, color=COLORS["overlap"], label="Overlap")
    ax.bar(x_satir, satir_only, width=bar_w, bottom=overlap, color=COLORS["satir_only"], label="SatIR only")

    ax.bar(x_tg, overlap, width=bar_w, color=COLORS["overlap"])
    ax.bar(x_tg, tg_only, width=bar_w, bottom=overlap, color=COLORS["tg_only"], label="TrialGPT only")

    ax.set_title(f"Per-patient retrieved trials — {human_mode_name(mode)}", pad=14)
    ax.set_xlabel("Patient IDs", labelpad=12)
    ax.set_ylabel("Number of useful trials", labelpad=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    light_y_grid(ax)

    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=12)
    ax.tick_params(axis="x", which="major", pad=28, length=0)

    trans = ax.get_xaxis_transform()
    for xs, xt in zip(x_satir, x_tg):
        ax.text(xs, -0.01, "S", transform=trans, ha="center", va="top", fontsize=10)
        ax.text(xt, -0.01, "T", transform=trans, ha="center", va="top", fontsize=10)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.8,
    )

    ax.text(
        0.995, 0.995,
        "S = SatIR, T = TrialGPT",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=COLORS["text"],
    )

    despine(ax)
    savefig_both(out_base)


def tug_of_war_violin(df: pd.DataFrame, out_base: Path, mode: str) -> None:
    if df.empty:
        return

    diffs = (to_int_series(df, "satir_found") - to_int_series(df, "trialgpt_found")).to_numpy()
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return

    max_abs = max(1, int(np.max(np.abs(diffs))))
    pad = max(1, int(round(max_abs * 0.08)))
    lim = max_abs + pad

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    vp = ax.violinplot(
        dataset=[diffs],
        positions=[0],
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        widths=0.82,
    )

    for body in vp.get("bodies", []):
        body.set_facecolor(COLORS["points"])
        body.set_alpha(0.72)

    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in vp:
            vp[key].set_color(COLORS["diag"])
            vp[key].set_linewidth(1.2)

    ax.axvline(0, linestyle="--", linewidth=1.2, color=COLORS["diag"])
    ax.set_xlim(-lim, lim)
    ax.set_yticks([0])
    ax.set_yticklabels(["SatIR − TrialGPT"])
    ax.set_title(f"Per-patient difference — {human_mode_name(mode)}", pad=12)
    ax.set_xlabel("Difference in relevant+eligible count", labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.grid(False, axis="y")

    despine(ax)
    savefig_both(out_base)


def draw_per_patient_composition_paired_display(
    ax: plt.Axes,
    df: pd.DataFrame,
    mode: str,
    max_patients: int,
    y_max: int | None = None,
    show_ylabel: bool = True,
) -> None:
    df_plot = _prepare_composition_df(df, max_patients=max_patients)
    if df_plot.empty:
        ax.set_visible(False)
        return

    x = np.arange(len(df_plot))
    overlap = df_plot["overlap"].to_numpy()
    satir_only = df_plot["satir_only"].to_numpy()
    tg_only = df_plot["tg_only"].to_numpy()

    bar_w = 0.40
    x_satir = x - bar_w / 2
    x_tg = x + bar_w / 2

    ax.bar(x_satir, overlap, width=bar_w, color=COLORS["overlap"], label="Overlap")
    ax.bar(x_satir, satir_only, width=bar_w, bottom=overlap, color=COLORS["satir_only"], label="SatIR only")
    ax.bar(x_tg, overlap, width=bar_w, color=COLORS["overlap"])
    ax.bar(x_tg, tg_only, width=bar_w, bottom=overlap, color=COLORS["tg_only"], label="TrialGPT only")

    ax.set_title(human_mode_name(mode), pad=10)
    ax.set_xlabel("Patients", labelpad=8)
    ax.set_xticks([])

    if show_ylabel:
        ax.set_ylabel("Number of trials", labelpad=8)
    else:
        ax.set_ylabel("")

    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    light_y_grid(ax)
    despine(ax)

    sum_overlap = int(df_plot["overlap"].sum())
    sum_satir_only = int(df_plot["satir_only"].sum())
    sum_tg_only = int(df_plot["tg_only"].sum())

    ax.text(
        0.98, 0.98,
        f"Σ overlap={sum_overlap}\nΣ SatIR-only={sum_satir_only}\nΣ TrialGPT-only={sum_tg_only}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=COLORS["text"],
    )

    ax.text(
        0.02, 0.98,
        "Left: SatIR  |  Right: TrialGPT",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=COLORS["text"],
    )


def paper_triptych_per_patient_paired_bars(
    df: pd.DataFrame,
    out_base: Path,
    max_patients: int = 200,
    mode_order: list[str] | None = None,
) -> None:
    if mode_order is None:
        mode_order = ["ccr", "all", "all-explore"]

    groups = []
    global_max = 0

    for mode in mode_order:
        g = df.loc[df["mode"].astype(str) == mode].copy()
        if g.empty:
            continue
        g2 = _prepare_composition_df(g, max_patients=max_patients)
        if g2.empty:
            continue
        groups.append((mode, g2))
        local_max = int(max(g2["satir_total"].max(), g2["tg_total"].max()))
        global_max = max(global_max, local_max)

    if not groups:
        return

    y_max = max(1, int(np.ceil(global_max * 1.10)))
    fig_w = 4.3 * len(groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(fig_w, 3.4), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for i, (ax, (mode, g)) in enumerate(zip(axes, groups)):
        draw_per_patient_composition_paired_display(
            ax=ax,
            df=g,
            mode=mode,
            max_patients=max_patients,
            y_max=y_max,
            show_ylabel=(i == 0),
        )

    handles = [
        Patch(facecolor=COLORS["overlap"], label="Overlap"),
        Patch(facecolor=COLORS["satir_only"], label="SatIR only"),
        Patch(facecolor=COLORS["tg_only"], label="TrialGPT only"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.10),
        columnspacing=1.8,
        handlelength=1.8,
    )

    savefig_fig_both(fig, out_base)


def paper_scatter_triptych(df: pd.DataFrame, out_base: Path, mode_order: list[str] | None = None) -> None:
    if mode_order is None:
        mode_order = ["ccr", "all", "all-explore"]

    groups = []
    for mode in mode_order:
        g = df.loc[df["mode"].astype(str) == mode].copy()
        if not g.empty:
            groups.append((mode, g))

    if not groups:
        return

    fig_w = 3.6 * len(groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(fig_w, 3.3), sharex=False, sharey=False)
    if len(groups) == 1:
        axes = [axes]

    for i, (ax, (mode, g)) in enumerate(zip(axes, groups)):
        x = pd.to_numeric(g["trialgpt_found"], errors="coerce")
        y = pd.to_numeric(g["satir_found"], errors="coerce")
        ok = x.notna() & y.notna()
        if not ok.any():
            ax.set_visible(False)
            continue

        xs = x[ok].to_numpy()
        ys = y[ok].to_numpy()
        mx = max(float(np.max(xs)), float(np.max(ys)), 1.0)
        n_above = int(np.sum(ys > xs))
        n_total = len(xs)

        ax.scatter(xs, ys, s=28, color=COLORS["points"], alpha=0.82, linewidths=0.0)
        ax.plot([0, mx], [0, mx], linestyle="--", color=COLORS["diag"], linewidth=1.1)
        ax.set_xlim(0, 1.03 * mx)
        ax.set_ylim(0, 1.03 * mx)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(human_mode_name(mode), pad=10)
        ax.set_xlabel("# TrialGPT-Retrieved Useful Trials", labelpad=8)
        if i == 0:
            ax.set_ylabel("# SatIR-Retrieved Useful Trials", labelpad=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(True, axis="both", color=COLORS["grid"], linewidth=0.7, alpha=0.55)
        despine(ax)
        ax.text(
            0.98, 0.03,
            f"SatIR > TrialGPT\nfor {n_above}/{n_total} patients",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=COLORS["text"],
        )

    savefig_fig_both(fig, out_base)


def make_display_figures(df: pd.DataFrame, out_dir: Path, max_patients: int) -> None:
    paper_triptych_per_patient_paired_bars(
        df=df,
        out_base=out_dir / "paper_triptych__per_patient_composition",
        max_patients=max_patients,
    )
    paper_scatter_triptych(
        df=df,
        out_base=out_dir / "paper_scatter_triptych__trialgpt_vs_satir",
    )


def plot_suite(df: pd.DataFrame, out_dir: Path, mode: str, max_patients: int) -> None:
    mslug = mode_slug(mode)

    scatter_trialgpt_vs_satir(
        df,
        out_dir / f"scatter__trialgpt_vs_satir__mode_{mslug}",
        mode=mode,
    )
    per_patient_composition_paired(
        df,
        out_dir / f"per_patient_composition__paired__mode_{mslug}",
        mode=mode,
        max_patients=max_patients,
    )
    tug_of_war_violin(
        df,
        out_dir / f"tug_of_war__violin__satir_minus_trialgpt__mode_{mslug}",
        mode=mode,
    )


def main() -> None:
    set_paper_style()

    ap = argparse.ArgumentParser(
        description="Plot selected figures for SatIR vs TrialGPT-5 from dedicated compare output."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=Path("./satir_vs_trialgpt5_avgcutoff_out"),
        help="Directory containing per_patient.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: <in-dir>/plots)",
    )
    ap.add_argument(
        "--max-patients",
        type=int,
        default=200,
        help="Max patients to show in bar plots per mode (0 = all)",
    )
    ap.add_argument("--also-all", action="store_true")
    ap.add_argument("--make-display-version", action="store_true")
    args = ap.parse_args()

    per_path = args.in_dir / "per_patient.csv"
    if not per_path.exists():
        raise SystemExit(f"[error] missing {per_path}")

    df = pd.read_csv(per_path)

    for c in ["satir_found", "trialgpt_found", "mode"]:
        if c not in df.columns:
            raise SystemExit(f"[error] per_patient.csv missing column: {c}")

    df = filter_both_sides_present(df)

    out_dir = args.out_dir or (args.in_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = sorted(df["mode"].dropna().astype(str).unique().tolist())
    if not modes:
        raise SystemExit("[error] no modes found after filtering")

    for mode in modes:
        g = df.loc[df["mode"].astype(str) == mode].copy()
        if not g.empty:
            plot_suite(g, out_dir, mode=mode, max_patients=args.max_patients)

    if args.also_all:
        plot_suite(df, out_dir, mode="ALL", max_patients=args.max_patients)

    if args.make_display_version:
        make_display_figures(df=df, out_dir=out_dir, max_patients=args.max_patients)

    print(f"[ok] wrote requested plots under: {out_dir}")


if __name__ == "__main__":
    main()