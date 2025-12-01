#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_recall_table.py

Generate camera-ready LaTeX recall tables from per_patient_recall.csv
produced by compute_union_recall.py.

Expected input columns:
  required:
    - method_raw OR baseline_raw OR baseline
    - patient_id
    - mode
    - recall_to_union
    - found_relevant_eligible
    - union_relevant_eligible

  optional but preferred:
    - method_display / baseline_display
    - method_family / baseline_family
    - method_variant / baseline_variant

Outputs:
  1) retrieval_recall_macro_summary.csv
  2) retrieval_recall_micro_summary.csv
  3) retrieval_recall_macro_table.tex
  4) retrieval_recall_micro_table.tex

Behavior
--------
1) Mode rows like:
      ccr_a, ccr_b, ccr_c, ccr_d
      all-a, all-b, ...
      all-explore (a), ...
   are collapsed into:
      ccr, all, all-explore
   using column-wise maximum over numeric columns.

2) Baseline columns are collapsed by row-wise maximum:
   - A/B/C/D -> Clinician-best
   - TG-4.1 / TG-5 / TG-O-4.1 / TG-O-5 -> TG
   - NH-bge-4.1 / NH-bge-5 / NH-O-bge-4.1 / NH-O-bge-5 -> bge
   - NH-bmretriever-4.1 / NH-bmretriever-5 / NH-O-bmretriever-4.1 / NH-O-bmretriever-5 -> bmretriever
   - NH-pubmedbert-4.1 / NH-pubmedbert-5 / NH-O-pubmedbert-4.1 / NH-O-pubmedbert-5 -> pubmedbert
   - NH-spubmedbert-4.1 / NH-spubmedbert-5 / NH-O-spubmedbert-4.1 / NH-O-spubmedbert-5 -> spubmedbert
   - TXT-bge-m3 -> bge-m3
   - TXT-bioclinical-modernbert -> bioclinical-modernbert
   - SMT -> SMT (rendered as \name{} in the far-right column)

3) Reported grouped baseline columns are sorted by aggregate performance so
   relatively better columns appear on the right side. Sorting uses the
   collapsed MACRO summary, matching the same ordering mechanism.

4) Default compact header display names are used to keep tables narrow.

5) By default, grouped baselines bioclinical-modernbert and bge-m3 are
   dropped from the reported output columns.
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
}

DEFAULT_DROP_GROUPED_BASELINES = {"bioclinical-modernbert", "bge-m3"}

SUBCOHORT_ROW_RE = re.compile(
    r"^(?P<base>.+?)(?:[\s_-]*\(?\s*(?P<tag>[abcdABCD])\s*\)?)$"
)


def fmt(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{100.0 * float(x):.{digits}f}"


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


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "baseline_raw" not in df.columns and "method_raw" in df.columns:
        df["baseline_raw"] = df["method_raw"]
    if "baseline_raw" not in df.columns and "baseline" in df.columns:
        df["baseline_raw"] = df["baseline"]

    if "baseline_display" not in df.columns and "method_display" in df.columns:
        df["baseline_display"] = df["method_display"]

    if "baseline_sort_key" not in df.columns and "method_sort_key" in df.columns:
        df["baseline_sort_key"] = df["method_sort_key"]

    if "baseline_family" not in df.columns and "method_family" in df.columns:
        df["baseline_family"] = df["method_family"]

    if "baseline_variant" not in df.columns and "method_variant" in df.columns:
        df["baseline_variant"] = df["method_variant"]

    if "baseline" not in df.columns and "baseline_raw" in df.columns:
        df["baseline"] = df["baseline_raw"]

    return df


def load_per_patient(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_input_columns(df)

    required = [
        "baseline_raw",
        "patient_id",
        "mode",
        "recall_to_union",
        "found_relevant_eligible",
        "union_relevant_eligible",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns in {path}: {missing}")

    df = df.copy()
    df["baseline_raw"] = df["baseline_raw"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["recall_to_union"] = pd.to_numeric(df["recall_to_union"], errors="coerce")
    df["found_relevant_eligible"] = pd.to_numeric(df["found_relevant_eligible"], errors="coerce")
    df["union_relevant_eligible"] = pd.to_numeric(df["union_relevant_eligible"], errors="coerce")

    if "baseline_display" not in df.columns:
        df["baseline_display"] = df["baseline_raw"]
    else:
        df["baseline_display"] = df["baseline_display"].fillna(df["baseline_raw"]).astype(str)

    if "baseline_sort_key" not in df.columns:
        df["baseline_sort_key"] = df["baseline_raw"]
    else:
        df["baseline_sort_key"] = df["baseline_sort_key"].fillna(df["baseline_raw"]).astype(str)

    return df


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

    meta = (
        df[["baseline_raw", "baseline_sort_key"]]
        .drop_duplicates()
        .sort_values(["baseline_sort_key", "baseline_raw"])
    )
    return meta["baseline_raw"].tolist()


def build_display_name_map(
    df: pd.DataFrame,
    manual_overrides: Dict[str, str],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    meta = df[["baseline_raw", "baseline_display"]].drop_duplicates()

    for _, row in meta.iterrows():
        raw = str(row["baseline_raw"])
        disp = str(row["baseline_display"])
        out[raw] = disp

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


def summarize_macro_wide(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    include_overall: bool = False,
) -> pd.DataFrame:
    rows = []

    def build_row(g_mode: pd.DataFrame, mode_name: str) -> Dict[str, object]:
        row: Dict[str, object] = {"mode": mode_name}
        for baseline in baseline_order:
            gb = g_mode[g_mode["baseline_raw"] == baseline]
            row[f"baseline::{baseline}"] = (
                gb["recall_to_union"].mean() if not gb.empty else float("nan")
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


def summarize_micro_wide(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    include_overall: bool = False,
) -> pd.DataFrame:
    rows = []

    def build_row(g_mode: pd.DataFrame, mode_name: str) -> Dict[str, object]:
        row: Dict[str, object] = {"mode": mode_name}
        for baseline in baseline_order:
            gb = g_mode[g_mode["baseline_raw"] == baseline]
            if gb.empty:
                row[f"baseline::{baseline}"] = float("nan")
            else:
                num = gb["found_relevant_eligible"].sum(skipna=True)
                den = gb["union_relevant_eligible"].sum(skipna=True)
                row[f"baseline::{baseline}"] = (num / den) if den > 0 else float("nan")
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


def parse_clinician_members(spec: str) -> Set[str]:
    spec = (spec or "").strip()
    if not spec:
        return {"a", "b", "c", "d"}
    return {x.strip().lower() for x in spec.split(",") if x.strip()}


def strip_trailing_model_version(name: str) -> str:
    s = str(name).strip()
    return re.sub(r"-(?:4\.1|5)$", "", s)


def strip_report_prefix(name: str) -> str:
    s = str(name).strip()

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
    display: str,
    clinician_members: Set[str],
    clinician_best_label: str,
) -> str:
    raw_s = str(raw).strip()
    disp_s = str(display).strip()

    if raw_s.lower() == "smt" or disp_s.lower() == "smt":
        return "SMT"

    if raw_s.lower() in clinician_members or disp_s.lower() in clinician_members:
        return clinician_best_label

    base = disp_s if disp_s else raw_s
    base = strip_trailing_model_version(base)
    base = strip_report_prefix(base)
    return base


def collapse_baseline_columns(
    summary: pd.DataFrame,
    raw_baseline_order: List[str],
    raw_baseline_display: Dict[str, str],
    clinician_members: Set[str],
    clinician_best_label: str,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    if summary.empty:
        return summary.copy(), [], {}

    out = summary.copy()

    raw_to_group: Dict[str, str] = {}
    for raw in raw_baseline_order:
        disp = raw_baseline_display.get(raw, raw)
        raw_to_group[raw] = canonical_baseline_group_name(
            raw=raw,
            display=disp,
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

    new_display = {g: g for g in grouped_order}
    return out, grouped_order, new_display


def sort_grouped_baselines_by_macro(
    macro_summary: pd.DataFrame,
    baseline_order: List[str],
) -> List[str]:
    scores = []
    for b in baseline_order:
        if b == "SMT":
            continue
        col = f"baseline::{b}"
        if col not in macro_summary.columns:
            score = float("-inf")
        else:
            vals = pd.to_numeric(macro_summary[col], errors="coerce")
            score = vals.mean(skipna=True)
            if pd.isna(score):
                score = float("-inf")
        scores.append((b, score))

    scores.sort(key=lambda x: (x[1], x[0]))
    return [b for b, _ in scores]


def apply_compact_display_names(display_map: Dict[str, str]) -> Dict[str, str]:
    out = dict(display_map)
    for k, v in DEFAULT_COMPACT_BASELINE_DISPLAY.items():
        if k in out:
            out[k] = v
    return out


def build_display_df(
    summary: pd.DataFrame,
    baseline_order: List[str],
    baseline_display: Dict[str, str],
    mean_digits: int = 2,
    emc_col_name: str = r"\name{}",
) -> pd.DataFrame:
    out_rows = []
    for _, r in summary.iterrows():
        mode = str(r["mode"])
        row = {"Objective": MODE_DISPLAY.get(mode, mode)}

        for baseline in baseline_order:
            display_name = baseline_display.get(baseline, baseline)
            row[display_name] = fmt_pct(
                r.get(f"baseline::{baseline}", float("nan")),
                digits=mean_digits,
            )

        emc_val = r.get("baseline::SMT", float("nan"))
        row[emc_col_name] = rf"\textbf{{{fmt_pct(emc_val, digits=mean_digits)}}}"
        out_rows.append(row)

    cols = ["Objective"] + [baseline_display.get(b, b) for b in baseline_order] + [emc_col_name]
    return pd.DataFrame(out_rows)[cols]


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
            if c in {"Objective", r"\name{}"}:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("./union_recall_out/per_patient_recall.csv"),
        help="Path to per_patient_recall.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./union_recall_out/table_out"),
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
        help=(
            "Optional comma-separated mapping baseline_raw=Display Name. "
            "This overrides baseline_display from the CSV and the built-in compact labels."
        ),
    )
    ap.add_argument(
        "--drop-grouped-baselines",
        type=str,
        default="bioclinical-modernbert,bge-m3",
        help=(
            "Comma-separated grouped baseline names to exclude after grouping. "
            "Default drops bioclinical-modernbert and bge-m3."
        ),
    )
    ap.add_argument("--mean-digits", type=int, default=2)

    ap.add_argument("--macro-label", type=str, default="tab:retrieval_recall_macro")
    ap.add_argument("--micro-label", type=str, default="tab:retrieval_recall_micro")

    ap.add_argument(
        "--macro-caption",
        type=str,
        default=(
            r"Macro recall against the pooled union of relevant-and-eligible trials found by all compared methods. "
            r"Per-patient recall is averaged within each objective. "
            r"Subobjective rows are collapsed by column-wise maximum, and baseline variants are collapsed by row-wise maximum."
        ),
    )
    ap.add_argument(
        "--micro-caption",
        type=str,
        default=(
            r"Micro recall against the pooled union of relevant-and-eligible trials found by all compared methods. "
            r"Micro recall is computed as the ratio of summed useful trials found to the summed pooled union size within each objective. "
            r"Subobjective rows are collapsed by column-wise maximum, and baseline variants are collapsed by row-wise maximum."
        ),
    )

    ap.add_argument("--table-env", type=str, default="table")
    ap.add_argument("--macro-size-cmd", type=str, default=r"\small")
    ap.add_argument("--micro-size-cmd", type=str, default=r"\small")
    ap.add_argument("--macro-tabcolsep", type=int, default=4)
    ap.add_argument("--micro-tabcolsep", type=int, default=4)
    ap.add_argument("--macro-arraystretch", type=float, default=1.14)
    ap.add_argument("--micro-arraystretch", type=float, default=1.14)
    ap.add_argument("--macro-colspec", type=str, default="")
    ap.add_argument("--micro-colspec", type=str, default="")
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
        "--clinician-best-members",
        type=str,
        default="a,b,c,d",
        help="Comma-separated clinician baseline names/display names to collapse into Clinician-best",
    )
    ap.add_argument(
        "--clinician-best-label",
        type=str,
        default="Clinician-best",
        help="Label for collapsed clinician columns",
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

    df = load_per_patient(args.in_csv)

    mode_order = ordered_modes(df, mode_order_requested)
    raw_baseline_order = ordered_baselines(df, baseline_order_requested)
    raw_baseline_display = build_display_name_map(df, manual_display_map)

    macro_summary = summarize_macro_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        include_overall=args.include_overall,
    )
    micro_summary = summarize_micro_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        include_overall=args.include_overall,
    )

    macro_summary = collapse_rows_by_max(macro_summary)
    micro_summary = collapse_rows_by_max(micro_summary)

    clinician_members = parse_clinician_members(args.clinician_best_members)

    macro_summary, macro_baseline_order, macro_baseline_display = collapse_baseline_columns(
        summary=macro_summary,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
    )

    micro_summary, micro_baseline_order, micro_baseline_display = collapse_baseline_columns(
        summary=micro_summary,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
    )

    sorted_grouped_baselines = sort_grouped_baselines_by_macro(
        macro_summary=macro_summary,
        baseline_order=macro_baseline_order,
    )
    sorted_grouped_baselines = [
        b for b in sorted_grouped_baselines
        if b not in drop_grouped_baselines and b != "SMT"
    ]

    macro_baseline_order = sorted_grouped_baselines
    micro_baseline_order = sorted_grouped_baselines

    macro_baseline_display = {b: macro_baseline_display.get(b, b) for b in macro_baseline_order}
    micro_baseline_display = {b: micro_baseline_display.get(b, b) for b in micro_baseline_order}

    macro_baseline_display = apply_compact_display_names(macro_baseline_display)
    micro_baseline_display = apply_compact_display_names(micro_baseline_display)

    macro_summary = sort_summary_modes(macro_summary, mode_order)
    micro_summary = sort_summary_modes(micro_summary, mode_order)

    macro_df = build_display_df(
        summary=macro_summary,
        baseline_order=macro_baseline_order,
        baseline_display=macro_baseline_display,
        mean_digits=args.mean_digits,
        emc_col_name=r"\name{}",
    )
    micro_df = build_display_df(
        summary=micro_summary,
        baseline_order=micro_baseline_order,
        baseline_display=micro_baseline_display,
        mean_digits=args.mean_digits,
        emc_col_name=r"\name{}",
    )

    macro_csv_path = args.out_dir / "retrieval_recall_macro_summary.csv"
    micro_csv_path = args.out_dir / "retrieval_recall_micro_summary.csv"
    macro_tex_path = args.out_dir / "retrieval_recall_macro_table.tex"
    micro_tex_path = args.out_dir / "retrieval_recall_micro_table.tex"

    macro_df.to_csv(macro_csv_path, index=False)
    micro_df.to_csv(micro_csv_path, index=False)

    macro_colspec = args.macro_colspec.strip() or None
    micro_colspec = args.micro_colspec.strip() or None

    macro_tex = latex_simple_table(
        display_df=macro_df,
        label=args.macro_label,
        caption=args.macro_caption,
        table_env=args.table_env,
        size_cmd=args.macro_size_cmd,
        tabcolsep=args.macro_tabcolsep,
        arraystretch=args.macro_arraystretch,
        colspec=macro_colspec,
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )
    micro_tex = latex_simple_table(
        display_df=micro_df,
        label=args.micro_label,
        caption=args.micro_caption,
        table_env=args.table_env,
        size_cmd=args.micro_size_cmd,
        tabcolsep=args.micro_tabcolsep,
        arraystretch=args.micro_arraystretch,
        colspec=micro_colspec,
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )

    macro_tex_path.write_text(macro_tex, encoding="utf-8")
    micro_tex_path.write_text(micro_tex, encoding="utf-8")

    print(f"[ok] wrote {macro_csv_path}")
    print(f"[ok] wrote {micro_csv_path}")
    print(f"[ok] wrote {macro_tex_path}")
    print(f"[ok] wrote {micro_tex_path}")
    print(f"[info] reported mode order: {mode_order}")
    print(f"[info] raw baseline order: {raw_baseline_order}")
    print(f"[info] grouped baseline order after performance sort/drop: {sorted_grouped_baselines}")
    print(f"[info] dropped grouped baselines: {sorted(drop_grouped_baselines)}")
    print("[debug] macro_summary columns:", list(macro_summary.columns))
    print("[debug] micro_summary columns:", list(micro_summary.columns))
    print(r"[info] subcohort rows a/b/c/d are collapsed by column-wise maximum")
    print(r"[info] baseline groups are collapsed by row-wise maximum")
    print(r"[info] NH-, TXT-, and O- prefixes are removed/folded in reported grouped baseline names")
    print(r"[info] TG-O is folded into TG")
    print(r"[info] better grouped baseline columns are placed farther to the right")
    print(r"[info] SMT is rendered as \name{} in the far-right column")
    if args.wrap_headers:
        print(r"[info] wrapped headers enabled; ensure your LaTeX preamble includes \usepackage{makecell}")


if __name__ == "__main__":
    main()