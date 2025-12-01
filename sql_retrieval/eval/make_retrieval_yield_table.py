#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_retrieval_yield_table.py

Generate camera-ready LaTeX tables from per_patient.csv produced by the
variant-aware compare script.

Expected input columns:
  required:
    - baseline OR baseline_raw
    - patient_id
    - mode
    - smt_found
    - trialgpt_found

  optional but preferred:
    - baseline_display
    - baseline_sort_key
    - k_avg

Outputs:
  1) retrieval_utility_summary.csv
  2) retrieval_patientlevel_summary.csv
  3) retrieval_utility_table.tex
  4) retrieval_patientlevel_table.tex

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

3) Reported baseline columns are sorted by aggregate performance so that
   relatively better columns appear on the right side. The sorting score
   is computed from the utility summary after mode-row collapse and
   baseline-column grouping.

4) Default compact header display names are used to keep tables narrow:
   - bioclinical-modernbert -> BioClinMB
   - bge-m3 -> BGE-M3
   - bmretriever -> BMRet
   - Clinician-best -> ClinBest
   - spubmedbert -> S-PubMedB
   - pubmedbert -> PubMedB
   - Retrieved/patient -> Ret./pt.

   These can still be overridden with --baseline-display-names.

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


def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "--"
    return str(int(round(float(x))))


def pick_retrieved_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "k_avg",
        "smt_retrieved",
        "retrieved",
        "smt_total_retrieved",
        "total_retrieved",
        "num_retrieved",
        "cutoff",
        "k",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def load_per_patient(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "baseline_raw" not in df.columns and "baseline" in df.columns:
        df["baseline_raw"] = df["baseline"]
    if "baseline" not in df.columns and "baseline_raw" in df.columns:
        df["baseline"] = df["baseline_raw"]

    required = ["baseline_raw", "patient_id", "mode", "smt_found", "trialgpt_found"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns in {path}: {missing}")

    df = df.copy()
    df["baseline_raw"] = df["baseline_raw"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["smt_found"] = pd.to_numeric(df["smt_found"], errors="coerce").fillna(0).astype(int)
    df["trialgpt_found"] = pd.to_numeric(df["trialgpt_found"], errors="coerce").fillna(0).astype(int)

    if "baseline_display" not in df.columns:
        df["baseline_display"] = df["baseline_raw"]
    else:
        df["baseline_display"] = df["baseline_display"].fillna(df["baseline_raw"]).astype(str)

    if "baseline_sort_key" not in df.columns:
        df["baseline_sort_key"] = df["baseline_raw"]
    else:
        df["baseline_sort_key"] = df["baseline_sort_key"].fillna(df["baseline_raw"]).astype(str)

    retrieved_col = pick_retrieved_column(df)
    if retrieved_col is not None:
        df["retrieved_budget"] = pd.to_numeric(df[retrieved_col], errors="coerce")
        print(f"[info] using retrieved-budget column: {retrieved_col}")
    else:
        df["retrieved_budget"] = float("nan")
        print("[warn] no retrieved-budget column found; Retrieved/patient will show as --")

    df["emc_has_useful"] = (df["smt_found"] >= 1).astype(int)
    df["baseline_has_useful"] = (df["trialgpt_found"] >= 1).astype(int)
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


def summarize_utility_wide(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    include_overall: bool = False,
) -> pd.DataFrame:
    rows = []

    def build_row(g_mode: pd.DataFrame, mode_name: str) -> Dict[str, object]:
        row: Dict[str, object] = {
            "mode": mode_name,
            "retrieved_mean": (
                g_mode["retrieved_budget"].mean()
                if g_mode["retrieved_budget"].notna().any()
                else float("nan")
            ),
            "emc_mean": g_mode["smt_found"].mean() if not g_mode.empty else float("nan"),
        }

        for baseline in baseline_order:
            gb = g_mode[g_mode["baseline_raw"] == baseline]
            row[f"baseline::{baseline}"] = (
                gb["trialgpt_found"].mean() if not gb.empty else float("nan")
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


def summarize_patientlevel_wide(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    include_overall: bool = False,
) -> pd.DataFrame:
    rows = []

    def build_row(g_mode: pd.DataFrame, mode_name: str) -> Dict[str, object]:
        row: Dict[str, object] = {
            "mode": mode_name,
            "emc_served": int(g_mode["emc_has_useful"].sum()),
        }

        for baseline in baseline_order:
            gb = g_mode[g_mode["baseline_raw"] == baseline]
            row[f"baseline::{baseline}"] = (
                int(gb["baseline_has_useful"].sum()) if not gb.empty else 0
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


def sort_grouped_baselines_by_utility(
    utility_summary: pd.DataFrame,
    baseline_order: List[str],
) -> List[str]:
    scores = []
    for b in baseline_order:
        col = f"baseline::{b}"
        if col not in utility_summary.columns:
            score = float("-inf")
        else:
            vals = pd.to_numeric(utility_summary[col], errors="coerce")
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


def build_utility_display_df(
    summary: pd.DataFrame,
    baseline_order: List[str],
    baseline_display: Dict[str, str],
    mean_digits: int = 2,
    retrieved_header: str = "Ret./pt.",
) -> pd.DataFrame:
    out_rows = []
    for _, r in summary.iterrows():
        mode = str(r["mode"])
        row = {
            "Objective": MODE_DISPLAY.get(mode, mode),
            retrieved_header: fmt_int(r["retrieved_mean"]),
        }

        for baseline in baseline_order:
            display_name = baseline_display.get(baseline, baseline)
            row[display_name] = fmt(
                r.get(f"baseline::{baseline}", float("nan")),
                digits=mean_digits,
            )

        row[r"\name{}"] = rf"\textbf{{{fmt(r['emc_mean'], digits=mean_digits)}}}"
        out_rows.append(row)

    cols = (
        ["Objective", retrieved_header]
        + [baseline_display.get(b, b) for b in baseline_order]
        + [r"\name{}"]
    )
    return pd.DataFrame(out_rows)[cols]


def build_patientlevel_display_df(
    summary: pd.DataFrame,
    baseline_order: List[str],
    baseline_display: Dict[str, str],
) -> pd.DataFrame:
    out_rows = []
    for _, r in summary.iterrows():
        mode = str(r["mode"])
        row = {"Objective": MODE_DISPLAY.get(mode, mode)}

        for baseline in baseline_order:
            display_name = baseline_display.get(baseline, baseline)
            val = r.get(f"baseline::{baseline}", 0)
            row[display_name] = 0 if pd.isna(val) else int(round(float(val)))

        emc_val = r["emc_served"]
        row[r"\name{}"] = 0 if pd.isna(emc_val) else int(round(float(emc_val)))
        out_rows.append(row)

    cols = ["Objective"] + [baseline_display.get(b, b) for b in baseline_order] + [r"\name{}"]
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
        default=Path("./avgcutoff_compare_out/table_out"),
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

    ap.add_argument("--utility-label", type=str, default="tab:retrieval_utility_main")
    ap.add_argument("--patientlevel-label", type=str, default="tab:retrieval_patientlevel_main")

    ap.add_argument(
        "--utility-caption",
        type=str,
        default=(
            r"Mean number of retrieved trials per patient that are both relevant and eligible. "
            r"Ret./pt. reports the mean shared retrieval budget per patient. "
            r"Baseline methods are shown as columns, and \name{} is shown in the far-right column."
        ),
    )
    ap.add_argument(
        "--patientlevel-caption",
        type=str,
        default=(
            r"Patient-level coverage over the evaluation set. "
            r"A patient is considered served if at least one useful trial "
            r"(both relevant and eligible) is retrieved. "
            r"Baseline methods are shown as columns, and \name{} is shown in the far-right column."
        ),
    )

    ap.add_argument("--table-env", type=str, default="table")
    ap.add_argument("--utility-size-cmd", type=str, default=r"\small")
    ap.add_argument("--patientlevel-size-cmd", type=str, default=r"\small")
    ap.add_argument("--utility-tabcolsep", type=int, default=2)
    ap.add_argument("--patientlevel-tabcolsep", type=int, default=4)
    ap.add_argument("--utility-arraystretch", type=float, default=1.18)
    ap.add_argument("--patientlevel-arraystretch", type=float, default=1.14)
    ap.add_argument("--utility-colspec", type=str, default="")
    ap.add_argument("--patientlevel-colspec", type=str, default="")
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
    ap.add_argument(
        "--retrieved-header",
        type=str,
        default="Ret./pt.",
        help="Header for the retrieved budget column in the utility table.",
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

    utility_summary = summarize_utility_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        include_overall=args.include_overall,
    )
    patientlevel_summary = summarize_patientlevel_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        include_overall=args.include_overall,
    )

    utility_summary = collapse_rows_by_max(utility_summary)
    patientlevel_summary = collapse_rows_by_max(patientlevel_summary)

    clinician_members = parse_clinician_members(args.clinician_best_members)

    utility_summary, utility_baseline_order, utility_baseline_display = collapse_baseline_columns(
        summary=utility_summary,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
    )

    patientlevel_summary, patient_baseline_order, patient_baseline_display = collapse_baseline_columns(
        summary=patientlevel_summary,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
    )

    sorted_grouped_baselines = sort_grouped_baselines_by_utility(
        utility_summary=utility_summary,
        baseline_order=utility_baseline_order,
    )

    sorted_grouped_baselines = [
        b for b in sorted_grouped_baselines
        if b not in drop_grouped_baselines
    ]

    utility_baseline_order = sorted_grouped_baselines
    patient_baseline_order = sorted_grouped_baselines

    utility_baseline_display = {b: utility_baseline_display.get(b, b) for b in utility_baseline_order}
    patient_baseline_display = {b: patient_baseline_display.get(b, b) for b in patient_baseline_order}

    utility_baseline_display = apply_compact_display_names(utility_baseline_display)
    patient_baseline_display = apply_compact_display_names(patient_baseline_display)

    utility_summary = sort_summary_modes(utility_summary, mode_order)
    patientlevel_summary = sort_summary_modes(patientlevel_summary, mode_order)

    utility_df = build_utility_display_df(
        summary=utility_summary,
        baseline_order=utility_baseline_order,
        baseline_display=utility_baseline_display,
        mean_digits=args.mean_digits,
        retrieved_header=args.retrieved_header,
    )
    patientlevel_df = build_patientlevel_display_df(
        summary=patientlevel_summary,
        baseline_order=patient_baseline_order,
        baseline_display=patient_baseline_display,
    )

    utility_csv_path = args.out_dir / "retrieval_utility_summary.csv"
    patientlevel_csv_path = args.out_dir / "retrieval_patientlevel_summary.csv"
    utility_tex_path = args.out_dir / "retrieval_utility_table.tex"
    patientlevel_tex_path = args.out_dir / "retrieval_patientlevel_table.tex"

    utility_df.to_csv(utility_csv_path, index=False)
    patientlevel_df.to_csv(patientlevel_csv_path, index=False)

    utility_colspec = args.utility_colspec.strip() or None
    patientlevel_colspec = args.patientlevel_colspec.strip() or None

    utility_tex = latex_simple_table(
        display_df=utility_df,
        label=args.utility_label,
        caption=args.utility_caption,
        table_env=args.table_env,
        size_cmd=args.utility_size_cmd,
        tabcolsep=args.utility_tabcolsep,
        arraystretch=args.utility_arraystretch,
        colspec=utility_colspec,
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )
    patientlevel_tex = latex_simple_table(
        display_df=patientlevel_df,
        label=args.patientlevel_label,
        caption=args.patientlevel_caption,
        table_env=args.table_env,
        size_cmd=args.patientlevel_size_cmd,
        tabcolsep=args.patientlevel_tabcolsep,
        arraystretch=args.patientlevel_arraystretch,
        colspec=patientlevel_colspec,
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )

    utility_tex_path.write_text(utility_tex, encoding="utf-8")
    patientlevel_tex_path.write_text(patientlevel_tex, encoding="utf-8")

    print(f"[ok] wrote {utility_csv_path}")
    print(f"[ok] wrote {patientlevel_csv_path}")
    print(f"[ok] wrote {utility_tex_path}")
    print(f"[ok] wrote {patientlevel_tex_path}")
    print(f"[info] reported mode order: {mode_order}")
    print(f"[info] raw baseline order: {raw_baseline_order}")
    print(f"[info] grouped baseline order after performance sort/drop: {sorted_grouped_baselines}")
    print(f"[info] dropped grouped baselines: {sorted(drop_grouped_baselines)}")
    print("[debug] utility_summary columns:", list(utility_summary.columns))
    print("[debug] patientlevel_summary columns:", list(patientlevel_summary.columns))
    print(f"[info] compact utility headers enabled; retrieved header={args.retrieved_header!r}")
    print(r"[info] subcohort rows a/b/c/d are collapsed by column-wise maximum")
    print(r"[info] baseline groups are collapsed by row-wise maximum")
    print(r"[info] NH-, TXT-, and O- prefixes are removed/folded in reported grouped baseline names")
    print(r"[info] TG-O is folded into TG")
    print(r"[info] better grouped baseline columns are placed farther to the right")
    print(r"[info] EMC/\name{} is placed in the far-right column")
    if args.wrap_headers:
        print(r"[info] wrapped headers enabled; ensure your LaTeX preamble includes \usepackage{makecell}")


if __name__ == "__main__":
    main()