#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_selected_figures.py  (PUBLICATION-READY + DEBUG FRIENDLY)

A polished plotting script for SMT vs TrialGPT retrieval comparisons.

What it produces
----------------
Per mode:
  1) scatter__trialgpt_vs_smt__mode_<mode>.pdf/.png
  2) per_patient_composition__paired__mode_<mode>.pdf/.png
  3) tug_of_war__violin__smt_minus_trialgpt__mode_<mode>.pdf/.png
  4) dist__smt_fetched__mode_<mode>.pdf/.png
  5) per_patient__smt_fetched_composition__mode_<mode>.pdf/.png

Optional:
  --also-all
    also writes combined (all modes) versions of the plots

Paper/display figures:
  --make-display-version
    also writes:
      a) paper_triptych__per_patient_composition.pdf/.png
      b) paper_scatter_triptych__trialgpt_vs_smt.pdf/.png

Key behavior
------------
- Only counts patients where BOTH SMT and TrialGPT are present.
- Tries to infer SMT "fetched" and "relevant" columns from common names.
- Uses a restrained publication-style palette and simplified layout.
- Saves both vector PDF and high-res PNG.

Typical usage
-------------
python plot_selected_figures.py \
    --in-dir ./avgcutoff_compare_out \
    --make-display-version \
    --also-all

Notes
-----
- Scatter orientation: x = TrialGPT relevant+eligible, y = SMT relevant+eligible
- In scatter, points above diagonal mean SMT > TrialGPT
- Per-patient paired bars:
    left bar  = SMT      = overlap + smt_only
    right bar = TrialGPT = overlap + tg_only
  so the lower segment in both bars is the overlap.

Mode naming
-----------
Internal raw mode values are assumed to be:
  - ccr
  - all
  - all-explore

For presentation and filenames, these are renamed to:
  - chief-complaint-treating
  - any-condition-treating
  - any-condition-relevant
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

# ────────────────────────────────────────────────────────────────
# GLOBAL STYLE / COLORS
# ────────────────────────────────────────────────────────────────

COLORS = {
    "overlap": "#9DB7D5",       # lighter blue to let unique segments pop
    "smt_only": "#F58518",      # warm orange
    "tg_only": "#54A24B",       # muted green

    "rel_elig": "#4C78A8",
    "rel_only": "#72B7B2",
    "not_rel": "#B9B9B9",

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
        "patch.linewidth": 0.0,

        "axes.grid": False,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

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


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_int_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=COLORS["text"],
    )


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


# ────────────────────────────────────────────────────────────────
# DATA FILTERING / SHAPING
# ────────────────────────────────────────────────────────────────

def filter_both_sides_present(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only patients where BOTH SMT and TrialGPT have results.
    Preferred signal: non-empty smt_source and tg_source.
    Fallback: require smt_found and trialgpt_found non-null.
    """
    df = df.copy()

    if "smt_source" in df.columns and "tg_source" in df.columns:
        smt_src = df["smt_source"].fillna("").astype(str).str.strip()
        tg_src = df["tg_source"].fillna("").astype(str).str.strip()
        m = smt_src.str.len() > 0
        t = tg_src.str.len() > 0
        return df.loc[m & t].copy()

    return df.loc[df["smt_found"].notna() & df["trialgpt_found"].notna()].copy()


def ensure_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure overlap, smt_only, tg_only exist.
    Preferred input:
      - overlap + smt_not_tg + tg_not_smt
    Fallbacks:
      - overlap only
      - smt_not_tg + tg_not_smt
    """
    df = df.copy()

    for c in {"smt_found", "trialgpt_found"}:
        if c not in df.columns:
            raise SystemExit(f"[error] per_patient.csv missing column: {c}")

    if "overlap" in df.columns and "smt_not_tg" in df.columns and "tg_not_smt" in df.columns:
        df["overlap"] = df["overlap"].fillna(0).astype(int)
        df["smt_only"] = df["smt_not_tg"].fillna(0).astype(int)
        df["tg_only"] = df["tg_not_smt"].fillna(0).astype(int)
        return df

    if "overlap" in df.columns:
        df["overlap"] = df["overlap"].fillna(0).astype(int)
        df["smt_only"] = (to_int_series(df, "smt_found") - df["overlap"]).clip(lower=0)
        df["tg_only"] = (to_int_series(df, "trialgpt_found") - df["overlap"]).clip(lower=0)
        return df

    if "smt_not_tg" in df.columns and "tg_not_smt" in df.columns:
        df["smt_only"] = df["smt_not_tg"].fillna(0).astype(int)
        df["tg_only"] = df["tg_not_smt"].fillna(0).astype(int)
        df["overlap"] = (to_int_series(df, "smt_found") - df["smt_only"]).clip(lower=0)
        return df

    raise SystemExit(
        "[error] Need either "
        "(overlap & smt_not_tg & tg_not_smt), or overlap alone, "
        "or (smt_not_tg & tg_not_smt)."
    )


def _prepare_composition_df(df: pd.DataFrame, max_patients: int) -> pd.DataFrame:
    df = ensure_components(df)
    if df.empty:
        return df

    df = df.copy()
    df["smt_total"] = (df["overlap"] + df["smt_only"]).astype(int)
    df["tg_total"] = (df["overlap"] + df["tg_only"]).astype(int)
    df["total_union"] = (df["overlap"] + df["smt_only"] + df["tg_only"]).astype(int)
    df["max_side"] = df[["smt_total", "tg_total"]].max(axis=1)

    df = df.sort_values(
        ["max_side", "total_union", "overlap", "smt_total", "tg_total"],
        ascending=[False, False, False, False, False],
    )

    if max_patients > 0 and len(df) > max_patients:
        df = df.head(max_patients).copy()

    return df


# ────────────────────────────────────────────────────────────────
# COMMON SMT COLUMN CANDIDATES
# ────────────────────────────────────────────────────────────────

FETCHED_CANDIDATES = [
    "smt_fetched_all_satisfied",
    "smt_all_satisfied",
    "smt_fetched",
    "smt_total_fetched",
    "smt_total",
    "num_all_satisfied",
    "all_satisfied",
    "fetched_all_satisfied",
]

RELEVANT_CANDIDATES = [
    "smt_relevant",
    "smt_found_relevant",
    "relevant_smt",
    "num_relevant",
    "relevant",
]


# ────────────────────────────────────────────────────────────────
# DEBUG / ANALYSIS FIGURES
# ────────────────────────────────────────────────────────────────

def scatter_trialgpt_vs_smt(df: pd.DataFrame, out_base: Path, mode: str) -> None:
    """
    Orientation: x=TrialGPT, y=SMT
    Points above diagonal => SMT > TrialGPT
    """
    x = pd.to_numeric(df["trialgpt_found"], errors="coerce")
    y = pd.to_numeric(df["smt_found"], errors="coerce")
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
    ax.scatter(
        xs,
        ys,
        s=34,
        alpha=0.82,
        color=COLORS["points"],
        linewidths=0.0,
    )
    ax.plot([0, mx], [0, mx], linestyle="--", color=COLORS["diag"], linewidth=1.2)

    ax.set_xlim(0, 1.03 * mx)
    ax.set_ylim(0, 1.03 * mx)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"EMC vs TrialGPT — {human_mode_name(mode)}", pad=12)
    ax.set_xlabel("# TrialGPT-Retrieved Useful Trials", labelpad=10)
    ax.set_ylabel("# EMC-Retrieved Useful Trials", labelpad=10)

    ax.text(
        0.98,
        0.03,
        f"EMC>TrialGPT: {n_above}/{n_total}\n"
        f"Equal: {n_equal}/{n_total}\n"
        f"EMC<TrialGPT: {n_below}/{n_total}",
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


def per_patient_composition_paired(
    df: pd.DataFrame,
    out_base: Path,
    mode: str,
    max_patients: int,
) -> None:
    df_plot = _prepare_composition_df(df, max_patients=max_patients)
    if df_plot.empty:
        return

    labels = (
        df_plot["patient_id"].astype(str).tolist()
        if "patient_id" in df_plot.columns
        else [str(i) for i in range(len(df_plot))]
    )
    x = np.arange(len(df_plot), dtype=float)

    overlap = df_plot["overlap"].to_numpy()
    smt_only = df_plot["smt_only"].to_numpy()
    tg_only = df_plot["tg_only"].to_numpy()

    fig_w = max(18.0, len(df_plot) * 0.33)
    fig, ax = plt.subplots(figsize=(fig_w, 7.4))

    bar_w = 0.36
    x_smt = x - bar_w / 2
    x_tg = x + bar_w / 2

    # SMT bars
    ax.bar(
        x_smt,
        overlap,
        width=bar_w,
        align="center",
        color=COLORS["overlap"],
        label="Overlap",
    )
    ax.bar(
        x_smt,
        smt_only,
        width=bar_w,
        align="center",
        bottom=overlap,
        color=COLORS["smt_only"],
        label="EMC only",
    )

    # TrialGPT bars
    ax.bar(
        x_tg,
        overlap,
        width=bar_w,
        align="center",
        color=COLORS["overlap"],
    )
    ax.bar(
        x_tg,
        tg_only,
        width=bar_w,
        align="center",
        bottom=overlap,
        color=COLORS["tg_only"],
        label="TrialGPT only",
    )

    ax.set_title(f"Per-patient retrieved trials — {human_mode_name(mode)}", pad=14)
    ax.set_xlabel("Patient IDs", labelpad=12)
    ax.set_ylabel("Number of useful trials", labelpad=12)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    light_y_grid(ax)

    # Make the full first/last pair visible and visually centered
    ax.set_xlim(-0.8, len(x) - 0.2)

    # Patient IDs: one label per pair, centered between S/T
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=12)

    ax.tick_params(axis="x", which="major", pad=28, length=0)

    # Draw S/T manually under each bar instead of using minor ticks
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for xs, xt in zip(x_smt, x_tg):
        ax.text(xs, -0.01, "E", transform=trans, ha="center", va="top", fontsize=10)
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
        0.995,
        0.995,
        "E = EMC, T = TrialGPT",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=COLORS["text"],
    )

    despine(ax)
    savefig_both(out_base)


def tug_of_war_violin(df: pd.DataFrame, out_base: Path, mode: str) -> None:
    """
    Violin of diff = EMC - TrialGPT in relevant+eligible count.
    """
    if df.empty:
        return

    diffs = (to_int_series(df, "smt_found") - to_int_series(df, "trialgpt_found")).to_numpy()
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
    ax.set_yticklabels(["EMC − TrialGPT"])

    ax.set_title(f"Per-patient difference — {human_mode_name(mode)}", pad=12)
    ax.set_xlabel("Difference in relevant+eligible count", labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.grid(False, axis="y")

    despine(ax)
    savefig_both(out_base)


def dist_smt_fetched(df: pd.DataFrame, out_base: Path, mode: str) -> None:
    fetched_col = pick_first_existing_col(df, FETCHED_CANDIDATES)
    if fetched_col is None:
        print(f"[warn] skip dist_smt_fetched for {mode} (missing fetched column).")
        return

    fetched = to_int_series(df, fetched_col).to_numpy()
    if fetched.size == 0:
        return

    lo = int(np.min(fetched))
    hi = int(np.max(fetched))
    if lo == hi:
        lo = max(0, lo - 1)
        hi = hi + 1
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(
        fetched,
        bins=bins,
        color=COLORS["points"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title(f"Distribution of EMC fetched trials — {human_mode_name(mode)}", pad=12)
    ax.set_xlabel("Fetched trials", labelpad=10)
    ax.set_ylabel("Patients", labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    light_y_grid(ax)

    despine(ax)
    savefig_both(out_base)


def per_patient_smt_fetched_composition(
    df: pd.DataFrame,
    out_base: Path,
    mode: str,
    max_patients: int,
) -> None:
    """
    Per patient, decompose SMT fetched into:
      - relevant_and_eligible (uses smt_found)
      - relevant_only = relevant - rel&elig
      - not_relevant = fetched - relevant
    """
    fetched_col = pick_first_existing_col(df, FETCHED_CANDIDATES)
    rel_col = pick_first_existing_col(df, RELEVANT_CANDIDATES)

    if fetched_col is None or rel_col is None:
        print(
            f"[warn] skip per_patient_smt_fetched_composition for {mode} "
            f"(fetched_col={fetched_col}, rel_col={rel_col})."
        )
        return

    if "smt_found" not in df.columns:
        print(f"[warn] skip per_patient_smt_fetched_composition for {mode} (missing smt_found).")
        return

    fetched = to_int_series(df, fetched_col)
    relevant = to_int_series(df, rel_col)
    rel_elig = to_int_series(df, "smt_found")

    relevant = np.minimum(relevant, fetched)
    rel_elig = np.minimum(rel_elig, relevant)

    relevant_only = (relevant - rel_elig).clip(lower=0)
    not_relevant = (fetched - relevant).clip(lower=0)

    df2 = df.copy()
    df2["fetched"] = fetched
    df2["rel_elig"] = rel_elig
    df2["rel_only"] = relevant_only
    df2["not_rel"] = not_relevant
    df2 = df2.sort_values(["fetched", "rel_elig", "rel_only"], ascending=[False, False, False])

    if max_patients > 0 and len(df2) > max_patients:
        df2 = df2.head(max_patients).copy()
    if df2.empty:
        return

    labels = (
        df2["patient_id"].astype(str).tolist()
        if "patient_id" in df2.columns
        else [str(i) for i in range(len(df2))]
    )
    x = np.arange(len(df2))

    a = df2["rel_elig"].to_numpy()
    b = df2["rel_only"].to_numpy()
    c = df2["not_rel"].to_numpy()

    fig_w = max(11.5, len(df2) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    ax.bar(x, a, width=0.86, color=COLORS["rel_elig"], label="Relevant + eligible")
    ax.bar(x, b, width=0.86, bottom=a, color=COLORS["rel_only"], label="Relevant only")
    ax.bar(x, c, width=0.86, bottom=a + b, color=COLORS["not_rel"], label="Not relevant")

    ax.set_title(f"EMC fetched composition — {human_mode_name(mode)}", pad=14)
    ax.set_xlabel("Patients (sorted by fetched count)", labelpad=12)
    ax.set_ylabel("Number of trials", labelpad=12)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    light_y_grid(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.8,
    )

    if len(labels) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=12)
    else:
        step = max(1, len(labels) // 22)
        show_idx = np.arange(0, len(labels), step)
        ax.set_xticks(show_idx)
        ax.set_xticklabels([labels[i] for i in show_idx], rotation=90, fontsize=12)

    despine(ax)
    savefig_both(out_base)


# ────────────────────────────────────────────────────────────────
# PAPER FIGURES
# ────────────────────────────────────────────────────────────────

def draw_per_patient_composition_paired_display(
    ax: plt.Axes,
    df: pd.DataFrame,
    mode: str,
    max_patients: int,
    y_max: int | None = None,
    show_ylabel: bool = True,
    panel_label: str | None = None,
) -> None:
    """
    Display version:
    - paired per-patient bars
    - left  bar = SMT      = overlap + smt_only
    - right bar = TrialGPT = overlap + tg_only
    - no patient IDs
    - shared legend handled outside
    """
    df_plot = _prepare_composition_df(df, max_patients=max_patients)
    if df_plot.empty:
        ax.set_visible(False)
        return

    x = np.arange(len(df_plot))
    overlap = df_plot["overlap"].to_numpy()
    smt_only = df_plot["smt_only"].to_numpy()
    tg_only = df_plot["tg_only"].to_numpy()

    bar_w = 0.40
    x_smt = x - bar_w / 2
    x_tg = x + bar_w / 2

    ax.bar(x_smt, overlap, width=bar_w, color=COLORS["overlap"], label="Overlap")
    ax.bar(x_smt, smt_only, width=bar_w, bottom=overlap, color=COLORS["smt_only"], label="EMC only")

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
    sum_smt_only = int(df_plot["smt_only"].sum())
    sum_tg_only = int(df_plot["tg_only"].sum())

    ax.text(
        0.98,
        0.98,
        f"Σ overlap={sum_overlap}\nΣ EMC-only={sum_smt_only}\nΣ TrialGPT-only={sum_tg_only}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=COLORS["text"],
    )

    ax.text(
        0.02,
        0.98,
        "Left: EMC  |  Right: TrialGPT",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=COLORS["text"],
    )

    if panel_label is not None:
        add_panel_label(ax, panel_label)


def paper_triptych_per_patient_paired_bars(
    df: pd.DataFrame,
    out_base: Path,
    max_patients: int = 200,
    mode_order: list[str] | None = None,
) -> None:
    if mode_order is None:
        mode_order = ["ccr", "all", "all-explore"]

    groups: list[tuple[str, pd.DataFrame]] = []
    global_max = 0

    for mode in mode_order:
        g = df.loc[df["mode"].astype(str) == mode].copy()
        if g.empty:
            continue
        g2 = _prepare_composition_df(g, max_patients=max_patients)
        if g2.empty:
            continue
        groups.append((mode, g2))
        local_max = int(max(g2["smt_total"].max(), g2["tg_total"].max()))
        global_max = max(global_max, local_max)

    if not groups:
        print("[warn] skip paper_triptych_per_patient_paired_bars (no eligible groups found).")
        return

    y_max = max(1, int(np.ceil(global_max * 1.10)))
    fig_w = 4.3 * len(groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(fig_w, 3.4), sharey=True)

    if len(groups) == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, (ax, (mode, g)) in enumerate(zip(axes, groups)):
        draw_per_patient_composition_paired_display(
            ax=ax,
            df=g,
            mode=mode,
            max_patients=max_patients,
            y_max=y_max,
            show_ylabel=(i == 0),
            panel_label=panel_labels[i] if i < len(panel_labels) else None,
        )

    handles = [
        Patch(facecolor=COLORS["overlap"], label="Overlap"),
        Patch(facecolor=COLORS["smt_only"], label="EMC only"),
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


def draw_scatter_display(
    ax: plt.Axes,
    df: pd.DataFrame,
    mode: str,
    show_ylabel: bool = True,
    panel_label: str | None = None,
) -> None:
    x = pd.to_numeric(df["trialgpt_found"], errors="coerce")
    y = pd.to_numeric(df["smt_found"], errors="coerce")
    ok = x.notna() & y.notna()
    if not ok.any():
        ax.set_visible(False)
        return

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
    if show_ylabel:
        ax.set_ylabel("# EMC-Retrieved Useful Trials", labelpad=8)
    else:
        ax.set_ylabel("")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.grid(True, axis="both", color=COLORS["grid"], linewidth=0.7, alpha=0.55)
    despine(ax)

    ax.text(
        0.98,
        0.03,
        f"EMC > TrialGPT\nfor {n_above}/{n_total} patients",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=COLORS["text"],
    )

    if panel_label is not None:
        add_panel_label(ax, panel_label)


def paper_scatter_triptych(
    df: pd.DataFrame,
    out_base: Path,
    mode_order: list[str] | None = None,
) -> None:
    if mode_order is None:
        mode_order = ["ccr", "all", "all-explore"]

    groups: list[tuple[str, pd.DataFrame]] = []
    for mode in mode_order:
        g = df.loc[df["mode"].astype(str) == mode].copy()
        if g.empty:
            continue
        groups.append((mode, g))

    if not groups:
        print("[warn] skip paper_scatter_triptych (no eligible groups found).")
        return

    fig_w = 3.6 * len(groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(fig_w, 3.3), sharex=False, sharey=False)
    if len(groups) == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, (ax, (mode, g)) in enumerate(zip(axes, groups)):
        draw_scatter_display(
            ax=ax,
            df=g,
            mode=mode,
            show_ylabel=(i == 0),
            panel_label=panel_labels[i] if i < len(panel_labels) else None,
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
        out_base=out_dir / "paper_scatter_triptych__trialgpt_vs_smt",
    )


# ────────────────────────────────────────────────────────────────
# WRAPPERS
# ────────────────────────────────────────────────────────────────

def plot_suite(df: pd.DataFrame, out_dir: Path, mode: str, max_patients: int) -> None:
    mslug = mode_slug(mode)

    scatter_trialgpt_vs_smt(
        df,
        out_dir / f"scatter__trialgpt_vs_smt__mode_{mslug}",
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
        out_dir / f"tug_of_war__violin__smt_minus_trialgpt__mode_{mslug}",
        mode=mode,
    )
    dist_smt_fetched(
        df,
        out_dir / f"dist__smt_fetched__mode_{mslug}",
        mode=mode,
    )
    per_patient_smt_fetched_composition(
        df,
        out_dir / f"per_patient__smt_fetched_composition__mode_{mslug}",
        mode=mode,
        max_patients=max_patients,
    )


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main() -> None:
    set_paper_style()

    ap = argparse.ArgumentParser(
        description="Plot selected figures per mode for SMT vs TrialGPT report (publication-ready version)."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=Path("./avgcutoff_compare_out"),
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
    ap.add_argument(
        "--also-all",
        action="store_true",
        help="Also output combined (all modes) versions of the plots",
    )
    ap.add_argument(
        "--make-display-version",
        action="store_true",
        help="Also write paper/display figures in addition to debug plots",
    )
    args = ap.parse_args()

    in_dir = args.in_dir
    per_path = in_dir / "per_patient.csv"
    if not per_path.exists():
        raise SystemExit(f"[error] missing {per_path}")

    df = pd.read_csv(per_path)

    for c in ["smt_found", "trialgpt_found", "mode"]:
        if c not in df.columns:
            raise SystemExit(f"[error] per_patient.csv missing column: {c}")

    df = filter_both_sides_present(df)

    out_dir = args.out_dir or (in_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = sorted(df["mode"].dropna().astype(str).unique().tolist())
    if not modes:
        raise SystemExit("[error] no modes found after filtering")

    for mode in modes:
        g = df.loc[df["mode"].astype(str) == mode].copy()
        if g.empty:
            continue
        plot_suite(g, out_dir, mode=mode, max_patients=args.max_patients)

    if args.also_all:
        plot_suite(df, out_dir, mode="ALL", max_patients=args.max_patients)

    if args.make_display_version:
        make_display_figures(df=df, out_dir=out_dir, max_patients=args.max_patients)

    print(f"[ok] wrote requested plots under: {out_dir}")


if __name__ == "__main__":
    main()