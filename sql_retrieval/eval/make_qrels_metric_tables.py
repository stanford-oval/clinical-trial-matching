#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_qrels_metric_tables.py

Generate camera-ready LaTeX tables from the variant-aware summary CSVs produced by:

  1) compare_smt_vs_trialgpt_avgcutoff_with_weighted_qrels.py
     -> summary_by_variant_mode.csv

  2) compare_smt_vs_trialgpt_avgcutoff_qrels_annotated_precision.py
     -> summary_by_variant_mode.csv

This script outputs tables for:
  - qrels recall
  - annotated precision

and for each metric family it outputs:
  - aggregated baseline table  (wide tabular; row-wise best value auto-bolded)
  - raw baseline table         (longtable; no bolded metric cells)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


MODE_ORDER = ["ccr", "all", "all-explore"]

MODE_DISPLAY = {
    "ccr": r"\modeccr",
    "all": r"\modeall",
    "all-explore": r"\modeallexplore",
    "chief": r"\modechief",
    "Overall": "Overall",
}

DEFAULT_COMPACT_BASELINE_DISPLAY = {
    "bioclinical-modernbert": "BioClinMB",
    "bge-m3": "BGE-M3",
    "bmretriever": "BMRet",
    "Clinician-best": "ClinBest",
    "spubmedbert": "S-PubMedB",
    "pubmedbert": "PubMedB",
    "bge": "BGE",
    "TG": "TG",
    "trialgpt41": "TG-4.1",
    "trialgpt5": "TG-5",
    "TrialGPT-with-Objective-gpt4.1": "TG-O-4.1",
    "TrialGPT-with-Objective-gpt5": "TG-O-5",
    "newhybrid-bge-noobj-gpt41": "NH-bge-4.1",
    "newhybrid-bge-noobj-gpt5": "NH-bge-5",
    "newhybrid-bge-obj-gpt41": "NH-O-bge-4.1",
    "newhybrid-bge-obj-gpt5": "NH-O-bge-5",
    "newhybrid-bmretriever-noobj-gpt41": "NH-bmretriever-4.1",
    "newhybrid-bmretriever-noobj-gpt5": "NH-bmretriever-5",
    "newhybrid-bmretriever-obj-gpt41": "NH-O-bmretriever-4.1",
    "newhybrid-bmretriever-obj-gpt5": "NH-O-bmretriever-5",
    "newhybrid-pubmedbert-noobj-gpt41": "NH-pubmedbert-4.1",
    "newhybrid-pubmedbert-noobj-gpt5": "NH-pubmedbert-5",
    "newhybrid-pubmedbert-obj-gpt41": "NH-O-pubmedbert-4.1",
    "newhybrid-pubmedbert-obj-gpt5": "NH-O-pubmedbert-5",
    "newhybrid-spubmedbert-noobj-gpt41": "NH-spubmedbert-4.1",
    "newhybrid-spubmedbert-noobj-gpt5": "NH-spubmedbert-5",
    "newhybrid-spubmedbert-obj-gpt41": "NH-O-spubmedbert-4.1",
    "newhybrid-spubmedbert-obj-gpt5": "NH-O-spubmedbert-5",
    "text-bge_m3": "TXT-BGE-M3",
    "text-bioclinical_modernbert": "TXT-BioClinMB",
}

DEFAULT_DROP_GROUPED_BASELINES = {"bioclinical-modernbert", "bge-m3"}

SUBCOHORT_ROW_RE = re.compile(
    r"^(?P<base>.+?)(?:[\s_-]*\(?\s*(?P<tag>[abcdABCD])\s*\)?)$"
)


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "--"
    return str(int(round(float(x))))


def parse_name_map(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    spec = (spec or "").strip()
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                f"[error] bad --baseline-display-names item: {item!r}. "
                f"Expected baseline=Display Name"
            )
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise SystemExit(
                f"[error] bad --baseline-display-names item: {item!r}. "
                f"Expected baseline=Display Name"
            )
        out[k] = v
    return out


def parse_drop_grouped_baselines(spec: str) -> Set[str]:
    spec = (spec or "").strip()
    if not spec:
        return set(DEFAULT_DROP_GROUPED_BASELINES)
    return {x.strip() for x in spec.split(",") if x.strip()}


def latex_escape_header(s: str) -> str:
    return str(s).replace("_", r"\_")


def latex_escape_text(s: str) -> str:
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


def wrap_header_cell(s: str, max_words_per_line: int = 2) -> str:
    s = latex_escape_header(str(s))
    parts = []
    for chunk in s.split(" "):
        subparts = chunk.split("-")
        for i, sp in enumerate(subparts):
            if sp:
                parts.append(sp)
            if i < len(subparts) - 1:
                parts.append("-")

    lines = []
    cur = []
    word_count = 0

    for p in parts:
        cur.append(p)
        if p != "-":
            word_count += 1
        if word_count >= max_words_per_line:
            lines.append("".join(cur))
            cur = []
            word_count = 0

    if cur:
        lines.append("".join(cur))

    if len(lines) <= 1:
        return s
    return r"\makecell[c]{" + r" \\ ".join(lines) + "}"


def normalize_report_mode(mode: str) -> str:
    s = str(mode).strip()
    m = SUBCOHORT_ROW_RE.match(s)
    if not m:
        return s
    return m.group("base").strip()


def ordered_modes(df: pd.DataFrame, requested_order: List[str]) -> List[str]:
    modes_present = list(df["mode"].dropna().astype(str).unique())
    normalized_present = []
    seen = set()
    for m in modes_present:
        nm = normalize_report_mode(m)
        if nm not in seen:
            normalized_present.append(nm)
            seen.add(nm)

    return [m for m in requested_order if m in normalized_present] + [
        m for m in normalized_present if m not in requested_order
    ]


def ordered_baselines(
    df: pd.DataFrame,
    requested_order: Optional[List[str]],
) -> List[str]:
    baselines_present = list(df["baseline_raw"].dropna().astype(str).unique())

    if requested_order is not None:
        return [b for b in requested_order if b in baselines_present] + [
            b for b in baselines_present if b not in requested_order
        ]

    return sorted(baselines_present)


def build_display_name_map(
    baselines: List[str],
    manual_overrides: Dict[str, str],
) -> Dict[str, str]:
    out = {b: b for b in baselines}
    out.update(manual_overrides)
    return out


def collapse_rows_by_max(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()

    out = summary.copy()
    out["mode"] = out["mode"].map(normalize_report_mode)

    numeric_cols = [c for c in out.columns if c != "mode"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.groupby("mode", as_index=False)[numeric_cols].max()
    return out


def sort_summary_modes(summary: pd.DataFrame, mode_order: List[str]) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()

    mode_rank = {m: i for i, m in enumerate(mode_order + ["Overall"])}
    return (
        summary.sort_values(
            by="mode",
            key=lambda s: s.map(lambda x: mode_rank.get(x, 10**9)),
        )
        .reset_index(drop=True)
    )


def parse_clinician_members(spec: str) -> Set[str]:
    spec = (spec or "").strip()
    if not spec:
        return {"cliniciana", "clinicianb", "clinicianc", "cliniciand"}
    return {x.strip().lower() for x in spec.split(",") if x.strip()}


def strip_aux_suffixes(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"/_input_labels$", "", s)
    return s


def strip_report_prefix(name: str) -> str:
    s = strip_aux_suffixes(name)

    if s.startswith("NH-"):
        s = s[len("NH-"):]
    elif s.startswith("TXT-"):
        s = s[len("TXT-"):]

    if s.startswith("O-"):
        s = s[len("O-"):]

    if s.endswith("-O"):
        s = s[:-len("-O")]

    return s


def canonical_baseline_group_name(
    raw: str,
    clinician_members: Set[str],
    clinician_best_label: str,
) -> str:
    s = strip_aux_suffixes(str(raw).strip())
    sl = s.lower()

    if sl in clinician_members:
        return clinician_best_label

    if sl in {"cliniciana", "clinicianb", "clinicianc", "cliniciand"}:
        return clinician_best_label

    if sl in {"trialgpt41", "trialgpt5"}:
        return "TG"

    if sl in {
        "trialgpt-with-objective-gpt4.1",
        "trialgpt-with-objective-gpt41",
        "trialgpt-with-objective-gpt5",
    }:
        return "TG"

    s = strip_report_prefix(s)
    sl = s.lower()

    if sl in {"text-bge_m3", "bge_m3", "bge-m3"}:
        return "bge-m3"

    if sl in {
        "text-bioclinical_modernbert",
        "bioclinical_modernbert",
        "bioclinical-modernbert",
    }:
        return "bioclinical-modernbert"

    if sl.startswith("newhybrid-"):
        x = sl[len("newhybrid-"):]
        for prefix in ("bge-", "bmretriever-", "pubmedbert-", "spubmedbert-"):
            if x.startswith(prefix):
                fam = prefix[:-1]
                return fam

    if sl in {
        "bge",
        "bmretriever",
        "pubmedbert",
        "spubmedbert",
        "bge-m3",
        "bioclinical-modernbert",
        "tg",
    }:
        return "TG" if sl == "tg" else s

    return s


def collapse_baseline_columns(
    summary: pd.DataFrame,
    raw_baseline_order: List[str],
    clinician_members: Set[str],
    clinician_best_label: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if summary.empty:
        return summary.copy(), []

    out = summary.copy()

    raw_to_group: Dict[str, str] = {}
    for raw in raw_baseline_order:
        raw_to_group[raw] = canonical_baseline_group_name(
            raw=raw,
            clinician_members=clinician_members,
            clinician_best_label=clinician_best_label,
        )

    grouped_order: List[str] = []
    seen = set()
    for raw in raw_baseline_order:
        g = raw_to_group[raw]
        if g not in seen:
            grouped_order.append(g)
            seen.add(g)

    for grouped_name in grouped_order:
        member_raws = [r for r in raw_baseline_order if raw_to_group[r] == grouped_name]
        member_cols = [f"baseline::{r}" for r in member_raws if f"baseline::{r}" in out.columns]

        for c in member_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        if member_cols:
            out[f"baseline::{grouped_name}"] = out[member_cols].max(axis=1, skipna=True)

    old_cols = [f"baseline::{r}" for r in raw_baseline_order if f"baseline::{r}" in out.columns]
    out = out.drop(columns=[c for c in old_cols if c in out.columns])

    return out, grouped_order


def sort_baselines_by_metric(
    summary: pd.DataFrame,
    baseline_order: List[str],
    higher_is_better: bool = True,
) -> List[str]:
    scores = []
    for b in baseline_order:
        col = f"baseline::{b}"
        if col not in summary.columns:
            score = float("-inf") if higher_is_better else float("inf")
        else:
            vals = pd.to_numeric(summary[col], errors="coerce")
            score = vals.mean(skipna=True)
            if pd.isna(score):
                score = float("-inf") if higher_is_better else float("inf")
        scores.append((b, score))

    scores.sort(key=lambda x: (x[1], x[0]), reverse=False)
    return [b for b, _ in scores]


def apply_compact_display_names(display_map: Dict[str, str]) -> Dict[str, str]:
    out = dict(display_map)
    for k, v in DEFAULT_COMPACT_BASELINE_DISPLAY.items():
        if k in out:
            out[k] = v
    return out


def drop_all_empty_baseline_columns(
    summary: pd.DataFrame,
    baseline_order: List[str],
) -> List[str]:
    kept = []
    for b in baseline_order:
        col = f"baseline::{b}"
        if col not in summary.columns:
            continue
        vals = pd.to_numeric(summary[col], errors="coerce")
        if vals.notna().any():
            kept.append(b)
    return kept


def latex_simple_table(
    display_df: pd.DataFrame,
    label: str,
    caption: str,
    table_env: str = "table",
    size_cmd: str = r"\small",
    tabcolsep: int = 5,
    arraystretch: float = 1.18,
    colspec: Optional[str] = None,
    wrap_headers: bool = False,
    wrap_header_words: int = 2,
) -> str:
    cols = list(display_df.columns)

    if colspec is None:
        colspec = "l" + "c" * (len(cols) - 1)

    lines = []
    lines.append(f"\\begin{{{table_env}}}[t]")
    lines.append("\\centering")
    lines.append(size_cmd)
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    if wrap_headers:
        wrapped_cols = []
        for c in cols:
            if c in {"Objective", "Ret./pt.", r"\name{}"}:
                wrapped_cols.append(latex_escape_header(c))
            else:
                wrapped_cols.append(wrap_header_cell(c, max_words_per_line=wrap_header_words))
        lines.append(" & ".join(f"\\textbf{{{c}}}" for c in wrapped_cols) + r" \\")
    else:
        lines.append(
            " & ".join(f"\\textbf{{{latex_escape_header(c)}}}" for c in cols) + r" \\"
        )

    lines.append("\\midrule")

    for _, row in display_df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append(" & ".join(vals) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\end{{{table_env}}}")

    return "\n".join(lines) + "\n"


def latex_longtable(
    display_df: pd.DataFrame,
    label: str,
    caption: str,
    colspec: str,
    size_cmd: str = r"\small",
    tabcolsep: int = 3,
    arraystretch: float = 1.12,
    continued_text: str = r"\textit{Continued on next page}",
    add_midrule_between_objectives: bool = True,
    repeat_objective_only_once: bool = False,
) -> str:
    cols = list(display_df.columns)
    header = " & ".join(f"\\textbf{{{c}}}" for c in cols) + r" \\"

    lines = []
    lines.append(r"\begingroup")
    lines.append(size_cmd)
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
    lines.append("")
    lines.append(f"\\begin{{longtable}}{{{colspec}}}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}\\\\")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("")
    lines.append("\\midrule")
    lines.append(rf"\multicolumn{{{len(cols)}}}{{r}}{{{continued_text}}} \\")
    lines.append("\\endfoot")
    lines.append("")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    prev_objective = None
    for _, row in display_df.iterrows():
        vals = [str(row[c]) for c in cols]
        objective = vals[0]

        if prev_objective is not None and objective != prev_objective and add_midrule_between_objectives:
            lines.append(r"\midrule")

        if repeat_objective_only_once and prev_objective == objective:
            vals[0] = ""

        lines.append(" & ".join(vals) + r" \\")
        prev_objective = objective

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def load_recall_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "variant",
        "mode",
        "k_avg",
        "micro_smt_qrels_recall_only2",
        "micro_baseline_qrels_recall_only2",
        "micro_smt_qrels_recall_w12",
        "micro_baseline_qrels_recall_w12",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns in recall summary {path}: {missing}")

    return pd.DataFrame({
        "baseline_raw": df["variant"].astype(str),
        "mode": df["mode"].astype(str),
        "k_avg": pd.to_numeric(df["k_avg"], errors="coerce"),
        "smt_recall_only2": pd.to_numeric(df["micro_smt_qrels_recall_only2"], errors="coerce"),
        "baseline_recall_only2": pd.to_numeric(df["micro_baseline_qrels_recall_only2"], errors="coerce"),
        "smt_recall_w12": pd.to_numeric(df["micro_smt_qrels_recall_w12"], errors="coerce"),
        "baseline_recall_w12": pd.to_numeric(df["micro_baseline_qrels_recall_w12"], errors="coerce"),
    })


def load_precision_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "variant",
        "mode",
        "k_avg",
        "micro_smt_annotated_precision_12",
        "micro_baseline_annotated_precision_12",
        "micro_smt_annotated_weighted_precision_012_norm",
        "micro_baseline_annotated_weighted_precision_012_norm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns in precision summary {path}: {missing}")

    return pd.DataFrame({
        "baseline_raw": df["variant"].astype(str),
        "mode": df["mode"].astype(str),
        "k_avg": pd.to_numeric(df["k_avg"], errors="coerce"),
        "smt_prec_12": pd.to_numeric(df["micro_smt_annotated_precision_12"], errors="coerce"),
        "baseline_prec_12": pd.to_numeric(df["micro_baseline_annotated_precision_12"], errors="coerce"),
        "smt_prec_w": pd.to_numeric(df["micro_smt_annotated_weighted_precision_012_norm"], errors="coerce"),
        "baseline_prec_w": pd.to_numeric(df["micro_baseline_annotated_weighted_precision_012_norm"], errors="coerce"),
    })


def summarize_metric_wide(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    smt_col: str,
    baseline_col: str,
    include_overall: bool = False,
) -> pd.DataFrame:
    rows = []

    def build_row(g_mode: pd.DataFrame, mode_name: str) -> Dict[str, object]:
        row: Dict[str, object] = {
            "mode": mode_name,
            "k_avg": g_mode["k_avg"].mean() if g_mode["k_avg"].notna().any() else float("nan"),
            "emc_metric": g_mode[smt_col].max(skipna=True) if not g_mode.empty else float("nan"),
        }
        for baseline in baseline_order:
            gb = g_mode[g_mode["baseline_raw"] == baseline]
            row[f"baseline::{baseline}"] = (
                gb[baseline_col].max(skipna=True) if not gb.empty else float("nan")
            )
        return row

    for mode in mode_order:
        g = df[df["mode"].map(normalize_report_mode) == mode].copy()
        if g.empty:
            continue
        per_raw_mode = list(g["mode"].dropna().astype(str).unique())
        for raw_mode in per_raw_mode:
            gm = g[g["mode"] == raw_mode].copy()
            rows.append(build_row(gm, raw_mode))

    if include_overall and not df.empty:
        rows.append(build_row(df, "Overall"))

    return pd.DataFrame(rows)


def summarize_metric_long(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    smt_col: str,
    baseline_col: str,
    include_overall: bool = False,
) -> pd.DataFrame:
    rows = []

    for mode in mode_order:
        g = df[df["mode"].map(normalize_report_mode) == mode].copy()
        if g.empty:
            continue
        for baseline in baseline_order:
            gb = g[g["baseline_raw"] == baseline].copy()
            if gb.empty:
                continue
            rows.append({
                "mode": mode,
                "baseline_raw": baseline,
                "k_avg": gb["k_avg"].max(skipna=True),
                "baseline_metric": gb[baseline_col].max(skipna=True),
                "emc_metric": gb[smt_col].max(skipna=True),
            })

    if include_overall and not df.empty:
        for baseline in baseline_order:
            gb = df[df["baseline_raw"] == baseline].copy()
            if gb.empty:
                continue
            rows.append({
                "mode": "Overall",
                "baseline_raw": baseline,
                "k_avg": gb["k_avg"].mean(skipna=True),
                "baseline_metric": gb[baseline_col].max(skipna=True),
                "emc_metric": gb[smt_col].max(skipna=True),
            })

    return pd.DataFrame(rows)


def build_metric_display_df(
    summary: pd.DataFrame,
    baseline_order: List[str],
    baseline_display: Dict[str, str],
    digits: int,
    include_k: bool,
    k_header: str = "Ret./pt.",
    bold_row_best: bool = True,
    tie_tol: float = 1e-12,
) -> pd.DataFrame:
    out_rows = []
    for _, r in summary.iterrows():
        mode = str(r["mode"])
        row = {"Objective": MODE_DISPLAY.get(mode, mode)}

        if include_k:
            row[k_header] = fmt_int(r["k_avg"])

        numeric_entries = []
        for baseline in baseline_order:
            value = pd.to_numeric(r.get(f"baseline::{baseline}", float("nan")), errors="coerce")
            numeric_entries.append((baseline, value))
        emc_value = pd.to_numeric(r.get("emc_metric", float("nan")), errors="coerce")

        best_value = float("nan")
        if bold_row_best:
            candidates = [v for _, v in numeric_entries if pd.notna(v)]
            if pd.notna(emc_value):
                candidates.append(emc_value)
            if candidates:
                best_value = max(candidates)

        for baseline in baseline_order:
            display_name = baseline_display.get(baseline, baseline)
            value = pd.to_numeric(r.get(f"baseline::{baseline}", float("nan")), errors="coerce")
            cell = fmt(value, digits=digits)
            if (
                bold_row_best
                and pd.notna(value)
                and pd.notna(best_value)
                and abs(float(value) - float(best_value)) <= tie_tol
            ):
                cell = rf"\textbf{{{cell}}}"
            row[display_name] = cell

        emc_cell = fmt(emc_value, digits=digits)
        if (
            bold_row_best
            and pd.notna(emc_value)
            and pd.notna(best_value)
            and abs(float(emc_value) - float(best_value)) <= tie_tol
        ):
            emc_cell = rf"\textbf{{{emc_cell}}}"
        row[r"\name{}"] = emc_cell

        out_rows.append(row)

    cols = ["Objective"]
    if include_k:
        cols.append(k_header)
    cols += [baseline_display.get(b, b) for b in baseline_order]
    cols += [r"\name{}"]

    return pd.DataFrame(out_rows)[cols]


def build_metric_long_display_df(
    long_summary: pd.DataFrame,
    baseline_display: Dict[str, str],
    digits: int,
    k_header: str,
    baseline_metric_header: str,
    emc_metric_header: str,
) -> pd.DataFrame:
    rows = []
    for _, r in long_summary.iterrows():
        mode = str(r["mode"])
        baseline_raw = str(r["baseline_raw"])
        baseline_name = baseline_display.get(baseline_raw, baseline_raw)

        rows.append({
            "Objective": MODE_DISPLAY.get(mode, mode),
            "Baseline": latex_escape_text(baseline_name),
            k_header: fmt_int(r["k_avg"]),
            baseline_metric_header: fmt(r["baseline_metric"], digits=digits),
            emc_metric_header: fmt(r["emc_metric"], digits=digits),
        })

    return pd.DataFrame(rows, columns=[
        "Objective",
        "Baseline",
        k_header,
        baseline_metric_header,
        emc_metric_header,
    ])


def make_family_tables(
    family_name: str,
    df: pd.DataFrame,
    smt_col: str,
    baseline_col: str,
    out_dir: Path,
    mode_order: List[str],
    raw_baseline_order: List[str],
    raw_baseline_display: Dict[str, str],
    clinician_members: Set[str],
    clinician_best_label: str,
    drop_grouped_baselines: Set[str],
    include_overall: bool,
    digits: int,
    include_k: bool,
    retrieved_header: str,
    table_env: str,
    size_cmd: str,
    tabcolsep: int,
    arraystretch: float,
    colspec: Optional[str],
    wrap_headers: bool,
    wrap_header_words: int,
    raw_label: str,
    raw_caption: str,
    agg_label: str,
    agg_caption: str,
    raw_baseline_metric_header: str,
    raw_emc_metric_header: str,
    raw_longtable_colspec: str,
    raw_longtable_size_cmd: str,
    raw_longtable_tabcolsep: int,
    raw_longtable_arraystretch: float,
    raw_repeat_objective_only_once: bool,
) -> None:
    # Raw: long format
    raw_summary_wide = summarize_metric_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        smt_col=smt_col,
        baseline_col=baseline_col,
        include_overall=include_overall,
    )
    raw_summary_wide = collapse_rows_by_max(raw_summary_wide)
    raw_summary_wide = sort_summary_modes(raw_summary_wide, mode_order)

    raw_sorted_baselines = sort_baselines_by_metric(
        summary=raw_summary_wide,
        baseline_order=raw_baseline_order,
        higher_is_better=True,
    )
    raw_sorted_baselines = drop_all_empty_baseline_columns(raw_summary_wide, raw_sorted_baselines)

    raw_display_map = apply_compact_display_names(dict(raw_baseline_display))
    raw_display_map = {b: raw_display_map.get(b, b) for b in raw_sorted_baselines}

    raw_summary_long = summarize_metric_long(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_sorted_baselines,
        smt_col=smt_col,
        baseline_col=baseline_col,
        include_overall=include_overall,
    )

    raw_summary_long["mode"] = raw_summary_long["mode"].map(normalize_report_mode)
    raw_summary_long["mode_rank"] = raw_summary_long["mode"].map(
        {m: i for i, m in enumerate(mode_order + ["Overall"])}
    ).fillna(10**9)
    raw_summary_long["baseline_rank"] = raw_summary_long["baseline_raw"].map(
        {b: i for i, b in enumerate(raw_sorted_baselines)}
    ).fillna(10**9)
    raw_summary_long = raw_summary_long.sort_values(
        ["mode_rank", "baseline_rank", "baseline_raw"]
    ).drop(columns=["mode_rank", "baseline_rank"]).reset_index(drop=True)

    raw_csv_path = out_dir / f"raw_{family_name}_summary.csv"
    raw_summary_long.to_csv(raw_csv_path, index=False)

    raw_df = build_metric_long_display_df(
        long_summary=raw_summary_long,
        baseline_display=raw_display_map,
        digits=digits,
        k_header=retrieved_header,
        baseline_metric_header=raw_baseline_metric_header,
        emc_metric_header=raw_emc_metric_header,
    )

    raw_tex_path = out_dir / f"raw_{family_name}_table.tex"
    raw_tex = latex_longtable(
        display_df=raw_df,
        label=raw_label,
        caption=raw_caption,
        colspec=raw_longtable_colspec,
        size_cmd=raw_longtable_size_cmd,
        tabcolsep=raw_longtable_tabcolsep,
        arraystretch=raw_longtable_arraystretch,
        add_midrule_between_objectives=True,
        repeat_objective_only_once=raw_repeat_objective_only_once,
    )
    raw_tex_path.write_text(raw_tex, encoding="utf-8")

    # Aggregated: wide format
    agg_summary = summarize_metric_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        smt_col=smt_col,
        baseline_col=baseline_col,
        include_overall=include_overall,
    )
    agg_summary = collapse_rows_by_max(agg_summary)

    agg_summary, agg_baseline_order = collapse_baseline_columns(
        summary=agg_summary,
        raw_baseline_order=raw_baseline_order,
        clinician_members=clinician_members,
        clinician_best_label=clinician_best_label,
    )

    agg_baseline_order = sort_baselines_by_metric(
        summary=agg_summary,
        baseline_order=agg_baseline_order,
        higher_is_better=True,
    )
    agg_baseline_order = drop_all_empty_baseline_columns(agg_summary, agg_baseline_order)
    agg_baseline_order = [b for b in agg_baseline_order if b not in drop_grouped_baselines]

    agg_display_map = {b: b for b in agg_baseline_order}
    agg_display_map = apply_compact_display_names(agg_display_map)

    agg_summary = sort_summary_modes(agg_summary, mode_order)

    agg_df = build_metric_display_df(
        summary=agg_summary,
        baseline_order=agg_baseline_order,
        baseline_display=agg_display_map,
        digits=digits,
        include_k=include_k,
        k_header=retrieved_header,
        bold_row_best=True,
    )

    agg_csv_path = out_dir / f"aggregated_{family_name}_summary.csv"
    agg_tex_path = out_dir / f"aggregated_{family_name}_table.tex"
    agg_df.to_csv(agg_csv_path, index=False)
    agg_tex = latex_simple_table(
        display_df=agg_df,
        label=agg_label,
        caption=agg_caption,
        table_env=table_env,
        size_cmd=size_cmd,
        tabcolsep=tabcolsep,
        arraystretch=arraystretch,
        colspec=colspec,
        wrap_headers=wrap_headers,
        wrap_header_words=wrap_header_words,
    )
    agg_tex_path.write_text(agg_tex, encoding="utf-8")

    print(f"[ok] wrote {raw_csv_path}")
    print(f"[ok] wrote {raw_tex_path}")
    print(f"[ok] wrote {agg_csv_path}")
    print(f"[ok] wrote {agg_tex_path}")
    print(f"[info] raw baseline order ({family_name}): {raw_sorted_baselines}")
    print(f"[info] aggregated baseline order ({family_name}): {agg_baseline_order}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--recall-summary-csv",
        type=Path,
        required=True,
        help="Path to summary_by_variant_mode.csv from weighted-qrels recall script.",
    )
    ap.add_argument(
        "--precision-summary-csv",
        type=Path,
        required=True,
        help="Path to summary_by_variant_mode.csv from annotated-precision script.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./metric_table_out"),
        help="Output directory",
    )
    ap.add_argument(
        "--include-overall",
        action="store_true",
        help="Append Overall row",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="ccr,all,all-explore",
        help="Comma-separated mode order/filter",
    )
    ap.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Optional comma-separated baseline raw names in desired order/filter.",
    )
    ap.add_argument(
        "--baseline-display-names",
        type=str,
        default="",
        help="Optional comma-separated mapping baseline_raw=Display Name.",
    )
    ap.add_argument(
        "--drop-grouped-baselines",
        type=str,
        default="bioclinical-modernbert,bge-m3",
        help="Comma-separated grouped baseline names to exclude after grouping.",
    )
    ap.add_argument("--digits", type=int, default=3)

    ap.add_argument("--table-env", type=str, default="table")
    ap.add_argument("--size-cmd", type=str, default=r"\small")
    ap.add_argument("--tabcolsep", type=int, default=3)
    ap.add_argument("--arraystretch", type=float, default=1.16)
    ap.add_argument("--colspec", type=str, default="")
    ap.add_argument(
        "--wrap-headers",
        action="store_true",
        help=r"Wrap long LaTeX header cells with \makecell{...}. Requires \usepackage{makecell}.",
    )
    ap.add_argument(
        "--wrap-header-words",
        type=int,
        default=2,
        help="Approximate number of words/chunks per wrapped header line.",
    )

    ap.add_argument(
        "--raw-longtable-colspec",
        type=str,
        default=r"L{2.2cm}L{4.8cm}C{0.95cm}C{1.15cm}C{1.15cm}",
        help="LaTeX colspec for raw longtable output.",
    )
    ap.add_argument(
        "--raw-longtable-size-cmd",
        type=str,
        default=r"\small",
        help="LaTeX size command for raw longtable output.",
    )
    ap.add_argument(
        "--raw-longtable-tabcolsep",
        type=int,
        default=3,
        help="tabcolsep for raw longtable output.",
    )
    ap.add_argument(
        "--raw-longtable-arraystretch",
        type=float,
        default=1.12,
        help="arraystretch for raw longtable output.",
    )
    ap.add_argument(
        "--raw-repeat-objective-only-once",
        action="store_true",
        help="In raw longtables, show the objective only on the first row of each objective block.",
    )

    ap.add_argument(
        "--clinician-best-members",
        type=str,
        default="clinicianA,clinicianB,clinicianC,clinicianD",
        help="Comma-separated clinician baseline names to collapse into Clinician-best",
    )
    ap.add_argument(
        "--clinician-best-label",
        type=str,
        default="Clinician-best",
        help="Label for collapsed clinician columns",
    )
    ap.add_argument(
        "--retrieved-header",
        type=str,
        default="Ret./pt.",
        help="Header for the shared retrieval budget column.",
    )

    ap.add_argument("--raw-recall-label", type=str, default="tab:qrels_recall_raw")
    ap.add_argument("--agg-recall-label", type=str, default="tab:qrels_recall_agg")
    ap.add_argument("--raw-precision-label", type=str, default="tab:annotated_precision_raw")
    ap.add_argument("--agg-precision-label", type=str, default="tab:annotated_precision_agg")

    ap.add_argument(
        "--raw-recall-caption",
        type=str,
        default=(
            r"Qrels-based weighted recall at the shared retrieval budget. "
            r"Each row compares \name{} with one raw baseline variant for one objective."
        ),
    )
    ap.add_argument(
        "--agg-recall-caption",
        type=str,
        default=(
            r"Qrels-based weighted recall (micro, weighted 1/2) at the shared retrieval budget. "
            r"Baselines are aggregated by family using row-wise maximum within each grouped family, "
            r"and subcohort rows are collapsed by column-wise maximum. "
            r"The highest value in each row is bolded."
        ),
    )
    ap.add_argument(
        "--raw-precision-caption",
        type=str,
        default=(
            r"Annotated weighted precision over qrels-covered fetched pairs "
            r"(micro, normalized to $[0,1]$). "
            r"Each row compares \name{} with one raw baseline variant for one objective."
        ),
    )
    ap.add_argument(
        "--agg-precision-caption",
        type=str,
        default=(
            r"Annotated weighted precision over qrels-covered fetched pairs "
            r"(micro, normalized to $[0,1]$). "
            r"Baselines are aggregated by family using row-wise maximum within each grouped family, "
            r"and subcohort rows are collapsed by column-wise maximum. "
            r"The highest value in each row is bolded."
        ),
    )

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mode_order_requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    baseline_order_requested = (
        [b.strip() for b in args.baselines.split(",") if b.strip()]
        if args.baselines.strip()
        else None
    )
    manual_display_map = parse_name_map(args.baseline_display_names)
    drop_grouped_baselines = parse_drop_grouped_baselines(args.drop_grouped_baselines)
    clinician_members = parse_clinician_members(args.clinician_best_members)

    recall_df = load_recall_summary(args.recall_summary_csv)
    precision_df = load_precision_summary(args.precision_summary_csv)

    merged_for_order = pd.concat(
        [
            recall_df[["baseline_raw", "mode"]],
            precision_df[["baseline_raw", "mode"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    mode_order = ordered_modes(merged_for_order, mode_order_requested)
    raw_baseline_order = ordered_baselines(merged_for_order, baseline_order_requested)
    raw_baseline_display = build_display_name_map(raw_baseline_order, manual_display_map)
    raw_baseline_display = apply_compact_display_names(raw_baseline_display)

    colspec = args.colspec.strip() or None

    make_family_tables(
        family_name="qrels_recall",
        df=recall_df,
        smt_col="smt_recall_w12",
        baseline_col="baseline_recall_w12",
        out_dir=args.out_dir,
        mode_order=mode_order,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
        drop_grouped_baselines=drop_grouped_baselines,
        include_overall=args.include_overall,
        digits=args.digits,
        include_k=True,
        retrieved_header=args.retrieved_header,
        table_env=args.table_env,
        size_cmd=args.size_cmd,
        tabcolsep=args.tabcolsep,
        arraystretch=args.arraystretch,
        colspec=colspec,
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
        raw_label=args.raw_recall_label,
        raw_caption=args.raw_recall_caption,
        agg_label=args.agg_recall_label,
        agg_caption=args.agg_recall_caption,
        raw_baseline_metric_header=r"\makecell{\textbf{Baseline}\\\textbf{recall}}",
        raw_emc_metric_header=r"\makecell{\textbf{\name{}}\\\textbf{recall}}",
        raw_longtable_colspec=args.raw_longtable_colspec,
        raw_longtable_size_cmd=args.raw_longtable_size_cmd,
        raw_longtable_tabcolsep=args.raw_longtable_tabcolsep,
        raw_longtable_arraystretch=args.raw_longtable_arraystretch,
        raw_repeat_objective_only_once=args.raw_repeat_objective_only_once,
    )

    make_family_tables(
        family_name="annotated_precision",
        df=precision_df,
        smt_col="smt_prec_w",
        baseline_col="baseline_prec_w",
        out_dir=args.out_dir,
        mode_order=mode_order,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
        drop_grouped_baselines=drop_grouped_baselines,
        include_overall=args.include_overall,
        digits=args.digits,
        include_k=True,
        retrieved_header=args.retrieved_header,
        table_env=args.table_env,
        size_cmd=args.size_cmd,
        tabcolsep=args.tabcolsep,
        arraystretch=args.arraystretch,
        colspec=colspec,
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
        raw_label=args.raw_precision_label,
        raw_caption=args.raw_precision_caption,
        agg_label=args.agg_precision_label,
        agg_caption=args.agg_precision_caption,
        raw_baseline_metric_header=r"\makecell{\textbf{Baseline}\\\textbf{precision}}",
        raw_emc_metric_header=r"\makecell{\textbf{\name{}}\\\textbf{precision}}",
        raw_longtable_colspec=args.raw_longtable_colspec,
        raw_longtable_size_cmd=args.raw_longtable_size_cmd,
        raw_longtable_tabcolsep=args.raw_longtable_tabcolsep,
        raw_longtable_arraystretch=args.raw_longtable_arraystretch,
        raw_repeat_objective_only_once=args.raw_repeat_objective_only_once,
    )

    print(f"[info] mode order: {mode_order}")
    print(f"[info] raw baseline order seed: {raw_baseline_order}")
    print(f"[info] dropped grouped baselines: {sorted(drop_grouped_baselines)}")
    print(r"[info] subcohort rows a/b/c/d are collapsed by column-wise maximum")
    print(r"[info] aggregated baseline groups are collapsed by row-wise maximum")
    print(r"[info] '/_input_labels' suffix is stripped before grouping")
    print(r"[info] NH-, TXT-, and O- prefixes are removed/folded in grouped baseline names")
    print(r"[info] TG-O is folded into TG")
    print(r"[info] all-empty baseline columns are dropped before rendering")
    print(r"[info] raw longtables have no bolded metric cells")
    print(r"[info] aggregated compact tables bold the row-wise maximum value automatically")
    print(r"[info] raw tables are emitted as longtable; add \usepackage{longtable,booktabs,array,makecell} to your preamble")
    if args.wrap_headers:
        print(r"[info] wrapped headers enabled; ensure your LaTeX preamble includes \usepackage{makecell}")


if __name__ == "__main__":
    main()