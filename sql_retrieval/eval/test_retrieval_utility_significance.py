#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_retrieval_utility_significance.py

Given per_patient.csv with columns:
  - patient_id
  - mode
  - smt_found
  - trialgpt_found

Compute paired significance for retrieval-stage utility:
  1) Wilcoxon signed-rank test on per-patient differences
  2) Paired bootstrap confidence interval for the mean difference
  3) Sign test (optional sanity check)
  4) Basic descriptive counts

Outputs:
  - significance_by_mode.csv
  - significance_by_mode.json

Example:
  python test_retrieval_utility_significance.py \
      --in-csv ./avgcutoff_compare_out/per_patient.csv \
      --out-dir ./avgcutoff_compare_out/significance_out
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, binomtest
except Exception as e:
    raise SystemExit(
        "[error] scipy is required. Install with: pip install scipy\n"
        f"Original import error: {e}"
    )


MODE_ORDER = ["ccr", "all", "all-explore"]


def load_per_patient(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["patient_id", "mode", "smt_found", "trialgpt_found"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns in {path}: {missing}")

    df = df.copy()
    df["mode"] = df["mode"].astype(str)
    df["smt_found"] = pd.to_numeric(df["smt_found"], errors="coerce").fillna(0).astype(float)
    df["trialgpt_found"] = pd.to_numeric(df["trialgpt_found"], errors="coerce").fillna(0).astype(float)
    df["diff"] = df["smt_found"] - df["trialgpt_found"]
    return df


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(samples, 100 * (alpha / 2.0)))
    hi = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def paired_bootstrap_mean_diff(
    diffs: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return {
            "mean_diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_boot_two_sided": float("nan"),
            "p_boot_greater": float("nan"),
        }

    observed = float(np.mean(diffs))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    ci_lo, ci_hi = percentile_ci(boot_means, alpha=alpha)

    # Bootstrap p-values (simple empirical tails)
    # two-sided relative to 0
    p_le_zero = (np.sum(boot_means <= 0.0) + 1.0) / (n_boot + 1.0)
    p_ge_zero = (np.sum(boot_means >= 0.0) + 1.0) / (n_boot + 1.0)
    p_two_sided = min(1.0, 2.0 * min(p_le_zero, p_ge_zero))

    # one-sided alternative: EMC > TrialGPT i.e. mean diff > 0
    p_greater = (np.sum(boot_means <= 0.0) + 1.0) / (n_boot + 1.0)

    return {
        "mean_diff": observed,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_boot_two_sided": float(p_two_sided),
        "p_boot_greater": float(p_greater),
    }


def run_wilcoxon(diffs: np.ndarray, alternative: str = "greater") -> Dict[str, float]:
    """
    Wilcoxon signed-rank on paired differences.
    alternative='greater' tests whether median(diff) > 0.
    """
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return {
            "n_nonzero": 0,
            "statistic": float("nan"),
            "p_value": float("nan"),
        }

    res = wilcoxon(
        nonzero,
        zero_method="wilcox",
        alternative=alternative,
        correction=False,
        method="auto",
    )
    return {
        "n_nonzero": int(len(nonzero)),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def run_sign_test(diffs: np.ndarray, alternative: str = "greater") -> Dict[str, float]:
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    n = pos + neg
    if n == 0:
        return {
            "n_nonzero": 0,
            "n_positive": 0,
            "n_negative": 0,
            "p_value": float("nan"),
        }
    res = binomtest(pos, n=n, p=0.5, alternative=alternative)
    return {
        "n_nonzero": int(n),
        "n_positive": pos,
        "n_negative": neg,
        "p_value": float(res.pvalue),
    }


def summarize_mode(
    g: pd.DataFrame,
    n_boot: int,
    alpha: float,
    seed: int,
) -> Dict[str, float]:
    diffs = g["diff"].to_numpy(dtype=float)
    emc = g["smt_found"].to_numpy(dtype=float)
    tg = g["trialgpt_found"].to_numpy(dtype=float)

    emc_mean = float(np.mean(emc)) if len(emc) else float("nan")
    tg_mean = float(np.mean(tg)) if len(tg) else float("nan")
    mean_diff = float(np.mean(diffs)) if len(diffs) else float("nan")

    improvement_pct = float("nan") if tg_mean == 0 else 100.0 * (emc_mean - tg_mean) / tg_mean

    bootstrap = paired_bootstrap_mean_diff(
        diffs=diffs,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    wilx = run_wilcoxon(diffs, alternative="greater")
    sign = run_sign_test(diffs, alternative="greater")

    return {
        "patients": int(len(g)),
        "emc_mean": emc_mean,
        "trialgpt_mean": tg_mean,
        "mean_diff": mean_diff,
        "improvement_pct": improvement_pct,
        "n_emc_better": int(np.sum(diffs > 0)),
        "n_ties": int(np.sum(diffs == 0)),
        "n_trialgpt_better": int(np.sum(diffs < 0)),
        "wilcoxon_n_nonzero": wilx["n_nonzero"],
        "wilcoxon_statistic": wilx["statistic"],
        "wilcoxon_p_greater": wilx["p_value"],
        "signtest_n_nonzero": sign["n_nonzero"],
        "signtest_n_positive": sign["n_positive"],
        "signtest_n_negative": sign["n_negative"],
        "signtest_p_greater": sign["p_value"],
        "bootstrap_mean_diff": bootstrap["mean_diff"],
        "bootstrap_ci_lo": bootstrap["ci_lo"],
        "bootstrap_ci_hi": bootstrap["ci_hi"],
        "bootstrap_p_two_sided": bootstrap["p_boot_two_sided"],
        "bootstrap_p_greater": bootstrap["p_boot_greater"],
    }


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("./avgcutoff_compare_out/per_patient.csv"),
        help="Path to per_patient.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./avgcutoff_compare_out/significance_out"),
        help="Output directory",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="ccr,all,all-explore",
        help="Comma-separated mode order/filter",
    )
    ap.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_per_patient(args.in_csv)
    mode_order = [m.strip() for m in args.modes.split(",") if m.strip()]

    modes_present = list(df["mode"].dropna().astype(str).unique())
    modes = [m for m in mode_order if m in modes_present] + [m for m in modes_present if m not in mode_order]

    rows: List[Dict[str, float]] = []
    by_mode_json: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        g = df[df["mode"] == mode].copy()
        if g.empty:
            continue
        summary = summarize_mode(
            g=g,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
        )
        row = {"mode": mode, **summary}
        rows.append(row)
        by_mode_json[mode] = row

    out_df = pd.DataFrame(rows)

    csv_path = args.out_dir / "significance_by_mode.csv"
    json_path = args.out_dir / "significance_by_mode.json"
    txt_path = args.out_dir / "significance_summary.txt"

    out_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(by_mode_json, indent=2), encoding="utf-8")

    # Human-readable summary
    lines = []
    for _, r in out_df.iterrows():
        lines.append(
            f"[{r['mode']}] "
            f"EMC mean={r['emc_mean']:.2f}, TrialGPT mean={r['trialgpt_mean']:.2f}, "
            f"diff={r['mean_diff']:.2f}, gain={r['improvement_pct']:.1f}%"
        )
        lines.append(
            f"  wins/ties/losses = {int(r['n_emc_better'])}/{int(r['n_ties'])}/{int(r['n_trialgpt_better'])}"
        )
        lines.append(
            f"  Wilcoxon p(one-sided, EMC>TrialGPT) = {fmt_p(r['wilcoxon_p_greater'])}"
        )
        lines.append(
            f"  Sign test p(one-sided, EMC>TrialGPT) = {fmt_p(r['signtest_p_greater'])}"
        )
        lines.append(
            f"  Bootstrap mean diff 95% CI = [{r['bootstrap_ci_lo']:.3f}, {r['bootstrap_ci_hi']:.3f}], "
            f"p(one-sided) = {fmt_p(r['bootstrap_p_greater'])}"
        )
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {txt_path}")


if __name__ == "__main__":
    main()