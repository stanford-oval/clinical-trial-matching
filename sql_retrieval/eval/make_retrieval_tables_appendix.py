#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_retrieval_tables.py

Generate:
  1) compact main-paper utility tables
  2) expanded appendix utility tables
  3) pairwise patient-level winner tables (SMT vs each baseline)

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

Outputs
-------
Compact main-paper style:
  - retrieval_utility_compact_summary.csv
  - retrieval_utility_compact_table.tex

Expanded appendix style:
  - retrieval_utility_expanded_summary.csv
  - retrieval_utility_expanded_table.tex

Pairwise patient-level appendix:
  - retrieval_pairwise_patientlevel_summary.csv
  - retrieval_pairwise_patientlevel_table.tex

Behavior
--------
1) Mode rows like:
      ccr_a, ccr_b, ccr_c, ccr_d
      all-a, all-b, ...
      all-explore (a), ...
   are collapsed into:
      ccr, all, all-explore
   using column-wise maximum over numeric columns.

2) Compact baseline grouping:
   - A/B/C/D -> Clinician-best
   - TG-4.1 / TG-5 / TG-O-4.1 / TG-O-5 -> TG
   - NH-bge-4.1 / NH-bge-5 / NH-O-bge-4.1 / NH-O-bge-5 -> bge
   - NH-bmretriever-4.1 / NH-bmretriever-5 / NH-O-bmretriever-4.1 / NH-O-bmretriever-5 -> bmretriever
   - NH-pubmedbert-4.1 / NH-pubmedbert-5 / NH-O-pubmedbert-4.1 / NH-O-pubmedbert-5 -> pubmedbert
   - NH-spubmedbert-4.1 / NH-spubmedbert-5 / NH-O-spubmedbert-4.1 / NH-O-spubmedbert-5 -> spubmedbert
   - TXT-bge-m3 -> bge-m3
   - TXT-bioclinical-modernbert -> bioclinical-modernbert

3) Expanded appendix utility table:
   - keeps raw baseline variants separate
   - only normalizes row modes (ccr_a -> ccr, etc.)
   - sorts columns by aggregate performance so better baselines are farther right

4) Pairwise patient-level winner table:
   for each objective and each baseline, reports:
     - SMT wins
     - Baseline wins
     - Ties
     - SMT served
     - Baseline served

5) Default compact header names:
   - bioclinical-modernbert -> BioClinMB
   - bge-m3 -> BGE-M3
   - bmretriever -> BMRet
   - Clinician-best -> ClinBest
   - spubmedbert -> S-PubMedB
   - pubmedbert -> PubMedB
   - Retrieved/patient -> Ret./pt.

6) Pairwise baseline display cleanup:
   - NH- prefixes are removed
   - TXT-bge-m3 -> BGE-M3$^{*}$
   - TXT-bioclinical-modernbert -> BioClinMB$^{*}$
   - A/B/C/D -> Clinician A/B/C/D
   - * indicates embedding retrieval using the full clinical note as the query

Usage
-----
python make_retrieval_tables.py \
  --in-csv ./avgcutoff_compare_out/per_patient.csv \
  --out-dir ./avgcutoff_compare_out/table_out \
  --wrap-headers
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

    df["smt_has_useful"] = (df["smt_found"] >= 1).astype(int)
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
            "smt_mean": g_mode["smt_found"].mean() if not g_mode.empty else float("nan"),
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


def sort_baselines_by_utility(
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

        row[r"\name{}"] = rf"\textbf{{{fmt(r['smt_mean'], digits=mean_digits)}}}"
        out_rows.append(row)

    cols = (
        ["Objective", retrieved_header]
        + [baseline_display.get(b, b) for b in baseline_order]
        + [r"\name{}"]
    )
    return pd.DataFrame(out_rows)[cols]


def summarize_pairwise_patientlevel(
    df: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
    include_overall: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_rows(g_mode: pd.DataFrame, mode_name: str) -> None:
        for baseline in baseline_order:
            gb = g_mode[g_mode["baseline_raw"] == baseline].copy()
            if gb.empty:
                continue

            gb = (
                gb.groupby("patient_id", as_index=False)
                .agg(
                    smt_found=("smt_found", "max"),
                    baseline_found=("trialgpt_found", "max"),
                    smt_has_useful=("smt_has_useful", "max"),
                    baseline_has_useful=("baseline_has_useful", "max"),
                )
            )

            smt_wins = int((gb["smt_found"] > gb["baseline_found"]).sum())
            baseline_wins = int((gb["baseline_found"] > gb["smt_found"]).sum())
            ties = int((gb["baseline_found"] == gb["smt_found"]).sum())
            smt_served = int(gb["smt_has_useful"].sum())
            baseline_served = int(gb["baseline_has_useful"].sum())

            rows.append(
                {
                    "mode": mode_name,
                    "baseline_raw": baseline,
                    "smt_wins": smt_wins,
                    "baseline_wins": baseline_wins,
                    "ties": ties,
                    "smt_served": smt_served,
                    "baseline_served": baseline_served,
                }
            )

    for mode in mode_order:
        g = df[df["mode"].map(normalize_report_mode) == mode].copy()
        if g.empty:
            continue
        per_raw_mode = list(g["mode"].dropna().astype(str).unique())
        for raw_mode in per_raw_mode:
            gm = g[g["mode"] == raw_mode].copy()
            add_rows(gm, raw_mode)

    if include_overall and not df.empty:
        add_rows(df.copy(), "Overall")

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["mode"] = out["mode"].map(normalize_report_mode)
    numeric_cols = ["smt_wins", "baseline_wins", "ties", "smt_served", "baseline_served"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    out = (
        out.groupby(["mode", "baseline_raw"], as_index=False)[numeric_cols]
        .max()
        .reset_index(drop=True)
    )
    return out


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

    if wrap_headers:
        header_cells = [
            wrap_header_cell(c, max_words_per_line=wrap_header_words) for c in cols
        ]
        header_line = " & ".join(f"\\textbf{{{c}}}" for c in header_cells) + r" \\"
    else:
        header_line = (
            " & ".join(f"\\textbf{{{latex_escape_header(c)}}}" for c in cols) + r" \\"
        )

    body_lines = []
    for _, row in display_df.iterrows():
        vals = [str(row[c]) for c in cols]
        body_lines.append(" & ".join(vals) + r" \\")

    lines = []

    if table_env == "longtable":
        lines.append("{")
        lines.append(size_cmd)
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")
        lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
        lines.append(f"\\begin{{longtable}}{{{colspec}}}")
        lines.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
        lines.append("\\toprule")
        lines.append(header_line)
        lines.append("\\midrule")
        lines.append("\\endfirsthead")

        lines.append("\\toprule")
        lines.append(header_line)
        lines.append("\\midrule")
        lines.append("\\endhead")

        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{len(cols)}}}{{r}}{{\\emph{{Continued on next page}}}} \\\\"
        )
        lines.append("\\midrule")
        lines.append("\\endfoot")

        lines.append("\\bottomrule")
        lines.append("\\endlastfoot")

        lines.extend(body_lines)

        lines.append("\\end{longtable}")
        lines.append("}")
    else:
        lines.append(f"\\begin{{{table_env}}}[t]")
        lines.append("\\centering")
        lines.append(size_cmd)
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")
        lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append("\\toprule")
        lines.append(header_line)
        lines.append("\\midrule")
        lines.extend(body_lines)
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\end{{{table_env}}}")

    return "\n".join(lines) + "\n"


def is_single_clinician_label(s: str) -> bool:
    s = str(s).strip()
    return len(s) == 1 and s.lower() in {"a", "b", "c", "d"}


def pretty_pairwise_baseline_name(raw: str, display: str) -> str:
    raw_s = str(raw).strip()
    disp_s = str(display).strip() if display is not None else raw_s

    src = raw_s if raw_s else disp_s

    if is_single_clinician_label(raw_s):
        return f"Clinician {raw_s.upper()}"
    if is_single_clinician_label(disp_s):
        return f"Clinician {disp_s.upper()}"

    if src.startswith("TXT-bge-m3"):
        return r"BGE-M3$^{*}$"
    if src.startswith("TXT-bioclinical-modernbert"):
        return r"BioClinMB$^{*}$"

    if src.startswith("NH-"):
        src = src[len("NH-"):]

    return src


def build_pairwise_display_df(
    summary: pd.DataFrame,
    baseline_display: Dict[str, str],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "Objective",
                "Baseline",
                "SMT wins",
                "Baseline wins",
                "Ties",
                "SMT served",
                "Baseline served",
            ]
        )

    rows = []
    for _, r in summary.iterrows():
        raw_baseline = str(r["baseline_raw"])
        disp_baseline = baseline_display.get(raw_baseline, raw_baseline)
        pretty_baseline = pretty_pairwise_baseline_name(raw_baseline, disp_baseline)

        rows.append(
            {
                "Objective": MODE_DISPLAY.get(str(r["mode"]), str(r["mode"])),
                "Baseline": pretty_baseline,
                "SMT wins": int(r["smt_wins"]),
                "Baseline wins": int(r["baseline_wins"]),
                "Ties": int(r["ties"]),
                "SMT served": int(r["smt_served"]),
                "Baseline served": int(r["baseline_served"]),
            }
        )

    return pd.DataFrame(rows)[
        [
            "Objective",
            "Baseline",
            "SMT wins",
            "Baseline wins",
            "Ties",
            "SMT served",
            "Baseline served",
        ]
    ]


def sort_pairwise_summary(
    pairwise_summary: pd.DataFrame,
    mode_order: List[str],
    baseline_order: List[str],
) -> pd.DataFrame:
    if pairwise_summary.empty:
        return pairwise_summary.copy()

    mode_rank = {m: i for i, m in enumerate(mode_order + ["Overall"])}
    baseline_rank = {b: i for i, b in enumerate(baseline_order)}

    out = pairwise_summary.copy()
    out["_mode_rank"] = out["mode"].map(lambda x: mode_rank.get(x, 10**9))
    out["_baseline_rank"] = out["baseline_raw"].map(lambda x: baseline_rank.get(x, 10**9))
    out = out.sort_values(["_mode_rank", "_baseline_rank", "baseline_raw"]).drop(
        columns=["_mode_rank", "_baseline_rank"]
    )
    return out.reset_index(drop=True)


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
        help="Append Overall row where relevant",
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
            "Overrides baseline_display from CSV and built-in compact labels."
        ),
    )
    ap.add_argument(
        "--drop-grouped-baselines",
        type=str,
        default="bioclinical-modernbert,bge-m3",
        help=(
            "Comma-separated grouped baseline names to exclude from compact output. "
            "Default drops bioclinical-modernbert and bge-m3."
        ),
    )
    ap.add_argument("--mean-digits", type=int, default=2)

    ap.add_argument("--compact-utility-label", type=str, default="tab:retrieval_utility_main")
    ap.add_argument("--expanded-utility-label", type=str, default="tab:retrieval_utility_appendix")
    ap.add_argument("--pairwise-label", type=str, default="tab:retrieval_pairwise_appendix")

    ap.add_argument(
        "--compact-utility-caption",
        type=str,
        default=(
            r"Mean number of retrieved trials per patient that are both relevant and eligible. "
            r"Ret./pt. reports the mean shared retrieval budget per patient. "
            r"Compact grouped baselines are shown as columns, and \name{} is shown in the far-right column."
        ),
    )
    ap.add_argument(
        "--expanded-utility-caption",
        type=str,
        default=(
            r"Expanded appendix version of the retrieval table. "
            r"Raw baseline variants are shown separately rather than being merged into grouped columns. "
            r"Values are the mean number of relevant-and-eligible trials retrieved per patient."
        ),
    )
    ap.add_argument(
        "--pairwise-caption",
        type=str,
        default=(
            r"Pairwise patient-level comparison between \name{} and each baseline. "
            r"For each objective and baseline, we report the number of patients on which \name{} retrieves more useful trials, "
            r"the number on which the baseline retrieves more, the number of ties, and patient-level coverage. "
            r"$^{*}$ indicates embedding retrieval using the full clinical note as the query."
        ),
    )

    ap.add_argument("--table-env", type=str, default="table")
    ap.add_argument("--compact-size-cmd", type=str, default=r"\small")
    ap.add_argument("--expanded-size-cmd", type=str, default=r"\scriptsize")
    ap.add_argument("--pairwise-size-cmd", type=str, default=r"\small")

    ap.add_argument("--compact-tabcolsep", type=int, default=2)
    ap.add_argument("--expanded-tabcolsep", type=int, default=2)
    ap.add_argument("--pairwise-tabcolsep", type=int, default=4)

    ap.add_argument("--compact-arraystretch", type=float, default=1.18)
    ap.add_argument("--expanded-arraystretch", type=float, default=1.14)
    ap.add_argument("--pairwise-arraystretch", type=float, default=1.14)

    ap.add_argument("--compact-colspec", type=str, default="")
    ap.add_argument("--expanded-colspec", type=str, default="")
    ap.add_argument("--pairwise-colspec", type=str, default="")

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
        help="Header for the retrieved budget column in utility tables.",
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

    expanded_utility_summary = summarize_utility_wide(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        include_overall=args.include_overall,
    )
    expanded_utility_summary = collapse_rows_by_max(expanded_utility_summary)
    expanded_utility_summary = sort_summary_modes(expanded_utility_summary, mode_order)

    expanded_baseline_order = sort_baselines_by_utility(
        utility_summary=expanded_utility_summary,
        baseline_order=raw_baseline_order,
    )
    expanded_baseline_display = {b: raw_baseline_display.get(b, b) for b in expanded_baseline_order}

    expanded_utility_df = build_utility_display_df(
        summary=expanded_utility_summary,
        baseline_order=expanded_baseline_order,
        baseline_display=expanded_baseline_display,
        mean_digits=args.mean_digits,
        retrieved_header=args.retrieved_header,
    )

    clinician_members = parse_clinician_members(args.clinician_best_members)

    compact_utility_summary, compact_baseline_order, compact_baseline_display = collapse_baseline_columns(
        summary=expanded_utility_summary,
        raw_baseline_order=raw_baseline_order,
        raw_baseline_display=raw_baseline_display,
        clinician_members=clinician_members,
        clinician_best_label=args.clinician_best_label,
    )

    compact_baseline_order = sort_baselines_by_utility(
        utility_summary=compact_utility_summary,
        baseline_order=compact_baseline_order,
    )
    compact_baseline_order = [b for b in compact_baseline_order if b not in drop_grouped_baselines]

    compact_baseline_display = {b: compact_baseline_display.get(b, b) for b in compact_baseline_order}
    compact_baseline_display = apply_compact_display_names(compact_baseline_display)

    compact_utility_summary = sort_summary_modes(compact_utility_summary, mode_order)

    compact_utility_df = build_utility_display_df(
        summary=compact_utility_summary,
        baseline_order=compact_baseline_order,
        baseline_display=compact_baseline_display,
        mean_digits=args.mean_digits,
        retrieved_header=args.retrieved_header,
    )

    pairwise_summary = summarize_pairwise_patientlevel(
        df=df,
        mode_order=mode_order,
        baseline_order=raw_baseline_order,
        include_overall=args.include_overall,
    )
    pairwise_summary = sort_pairwise_summary(
        pairwise_summary=pairwise_summary,
        mode_order=mode_order,
        baseline_order=expanded_baseline_order,
    )

    pairwise_display_map = {b: raw_baseline_display.get(b, b) for b in raw_baseline_order}
    pairwise_df = build_pairwise_display_df(
        summary=pairwise_summary,
        baseline_display=pairwise_display_map,
    )

    compact_csv_path = args.out_dir / "retrieval_utility_compact_summary.csv"
    expanded_csv_path = args.out_dir / "retrieval_utility_expanded_summary.csv"
    pairwise_csv_path = args.out_dir / "retrieval_pairwise_patientlevel_summary.csv"

    compact_utility_df.to_csv(compact_csv_path, index=False)
    expanded_utility_df.to_csv(expanded_csv_path, index=False)
    pairwise_df.to_csv(pairwise_csv_path, index=False)

    compact_tex_path = args.out_dir / "retrieval_utility_compact_table.tex"
    expanded_tex_path = args.out_dir / "retrieval_utility_expanded_table.tex"
    pairwise_tex_path = args.out_dir / "retrieval_pairwise_patientlevel_table.tex"

    compact_tex = latex_simple_table(
        display_df=compact_utility_df,
        label=args.compact_utility_label,
        caption=args.compact_utility_caption,
        table_env=args.table_env,
        size_cmd=args.compact_size_cmd,
        tabcolsep=args.compact_tabcolsep,
        arraystretch=args.compact_arraystretch,
        colspec=(args.compact_colspec.strip() or None),
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )
    expanded_tex = latex_simple_table(
        display_df=expanded_utility_df,
        label=args.expanded_utility_label,
        caption=args.expanded_utility_caption,
        table_env=args.table_env,
        size_cmd=args.expanded_size_cmd,
        tabcolsep=args.expanded_tabcolsep,
        arraystretch=args.expanded_arraystretch,
        colspec=(args.expanded_colspec.strip() or None),
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )
    pairwise_tex = latex_simple_table(
        display_df=pairwise_df,
        label=args.pairwise_label,
        caption=args.pairwise_caption,
        table_env=args.table_env,
        size_cmd=args.pairwise_size_cmd,
        tabcolsep=args.pairwise_tabcolsep,
        arraystretch=args.pairwise_arraystretch,
        colspec=(args.pairwise_colspec.strip() or None),
        wrap_headers=args.wrap_headers,
        wrap_header_words=args.wrap_header_words,
    )

    compact_tex_path.write_text(compact_tex, encoding="utf-8")
    expanded_tex_path.write_text(expanded_tex, encoding="utf-8")
    pairwise_tex_path.write_text(pairwise_tex, encoding="utf-8")

    print(f"[ok] wrote {compact_csv_path}")
    print(f"[ok] wrote {expanded_csv_path}")
    print(f"[ok] wrote {pairwise_csv_path}")
    print(f"[ok] wrote {compact_tex_path}")
    print(f"[ok] wrote {expanded_tex_path}")
    print(f"[ok] wrote {pairwise_tex_path}")

    print(f"[info] reported mode order: {mode_order}")
    print(f"[info] raw baseline order: {raw_baseline_order}")
    print(f"[info] expanded baseline order after performance sort: {expanded_baseline_order}")
    print(f"[info] compact grouped baseline order after performance sort/drop: {compact_baseline_order}")
    print(f"[info] dropped grouped baselines from compact table: {sorted(drop_grouped_baselines)}")
    print(f"[info] compact utility columns: {list(compact_utility_df.columns)}")
    print(f"[info] expanded utility columns: {list(expanded_utility_df.columns)}")
    print(f"[info] pairwise table columns: {list(pairwise_df.columns)}")
    print(f"[info] compact utility rows: {len(compact_utility_df)}")
    print(f"[info] expanded utility rows: {len(expanded_utility_df)}")
    print(f"[info] pairwise rows: {len(pairwise_df)}")
    print(f"[info] retrieved header = {args.retrieved_header!r}")
    print(r"[info] subcohort rows a/b/c/d are collapsed by column-wise maximum")
    print(r"[info] compact table folds TG-O into TG and removes NH-/TXT-/O- prefixes")
    print(r"[info] expanded table keeps raw baseline variants separate")
    print(r"[info] pairwise table is computed baseline-by-baseline against SMT")
    print(r"[info] pairwise display removes NH- prefixes, maps TXT-* to starred labels, and expands A/B/C/D to Clinician A/B/C/D")
    if args.wrap_headers:
        print(r"[info] wrapped headers enabled; ensure LaTeX preamble includes \usepackage{makecell}")


if __name__ == "__main__":
    main()