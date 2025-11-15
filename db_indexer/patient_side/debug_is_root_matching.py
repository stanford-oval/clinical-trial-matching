#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
debug_is_root_matching.py

Debug why is_root stays 0 by comparing:
  1) inclusion facts being ingested
  2) patient root-source files used to infer is_root

It uses the SAME base_var normalization logic as the ingestion script:
  - extract variable name from base_var / entity_variable_name / source_variable_name
  - strip embedded timeframe token via split_base_and_timeframe(...)

It prints:
  - which files were found
  - root base_var set
  - fact base_vars
  - overlaps
  - missing matches
  - optional per-row detail

Expected layouts:

Facts side:
  {inclusion_root}/patient_facts_export/{pid}/inclusion/canonical.final.jsonl
  fallback:
  {inclusion_root}/patient_facts_export/{pid}/inclusion/facts.round*.jsonl

Root side:
  {root_fact_root}/patient_coded_results/{pid}/canonical.final.jsonl
  {root_fact_root}/patient_coded_results/{pid}/canonical.jsonl
  {root_fact_root}/patient_coded_results/{pid}/diagnosis.jsonl
  {root_fact_root}/patient_coded_results/{pid}/embedding_search_other_candidate_canonical.jsonl

Usage:
  python debug_is_root_matching.py \
    --patient sigir-201418 \
    --inclusion-root <SATIR_ROOT>/patient_build_inclusion \
    --root-fact-root <SATIR_ROOT>/patient_build_sigir_inclusion_root_fact \
    --show-all

Optional:
  --facts-file /exact/path/to/file.jsonl
  --root-pid other_patient_folder_name
"""

from __future__ import annotations

import os
import re
import json
import glob
import argparse
from typing import Dict, Any, Iterable, List, Optional, Tuple, Set


DEFAULT_INCL_ROOT = "<SATIR_ROOT>/patient_build_inclusion"
DEFAULT_ROOT_FACT_ROOT = "<SATIR_ROOT>/patient_build_sigir_inclusion_root_fact"

_TIMEFRAME_TOKEN_PAT = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years))"
)
TF_FINDER_RE = re.compile(_TIMEFRAME_TOKEN_PAT)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, ln in enumerate(f, start=1):
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                print(f"[warn] failed json parse: {path}:{ln_no}: {e}")
                continue
            if isinstance(obj, dict):
                yield obj


def split_base_and_timeframe(varname: str) -> Tuple[str, Optional[str]]:
    varname = (varname or "").strip()
    m = TF_FINDER_RE.search(varname)
    if not m:
        return varname, None
    tf = m.group(0)
    s, e = m.span()
    base = (varname[:s] + varname[e:]).strip("_")
    base = re.sub(r"__+", "_", base)
    return base, tf


def extract_var_name(rec: Dict[str, Any]) -> str:
    return (
        (rec.get("base_var") or "")
        or (rec.get("entity_variable_name") or "")
        or (rec.get("source_variable_name") or "")
    ).strip()


def normalize_base_var_from_rec(rec: Dict[str, Any]) -> str:
    nm = extract_var_name(rec)
    base_var, _ = split_base_and_timeframe(nm)
    return base_var


def find_fact_files(inclusion_root: str, patient: str, final_only: bool) -> List[str]:
    sdir = os.path.join(inclusion_root, "patient_facts_export", patient, "inclusion")
    if not os.path.isdir(sdir):
        return []

    cpath = os.path.join(sdir, "canonical.final.jsonl")
    if os.path.isfile(cpath):
        return [cpath]

    if final_only:
        f = os.path.join(sdir, "facts.round9999.jsonl")
        return [f] if os.path.isfile(f) else []

    return sorted(glob.glob(os.path.join(sdir, "facts.round*.jsonl")))


def find_root_files(root_fact_root: str, root_pid: str) -> List[str]:
    pdir = os.path.join(root_fact_root, "patient_coded_results", root_pid)
    cands = [
        os.path.join(pdir, "canonical.final.jsonl"),
        os.path.join(pdir, "canonical.jsonl"),
        os.path.join(pdir, "diagnosis.jsonl"),
        os.path.join(pdir, "embedding_search_other_candidate_canonical.jsonl"),
    ]
    return [p for p in cands if os.path.isfile(p)]


def summarize_records(files: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fpath in files:
        for rec in iter_jsonl(fpath):
            nm = extract_var_name(rec)
            if not nm:
                continue
            base_var, tf = split_base_and_timeframe(nm)
            rows.append({
                "file": fpath,
                "var_name": nm,
                "base_var": base_var,
                "tf_token_embedded": tf,
                "concept_id": rec.get("conceptId") or rec.get("concept_id"),
                "fact_id": rec.get("fact_id"),
                "source": rec.get("source"),
                "raw": rec,
            })
    return rows


def unique_base_vars(rows: List[Dict[str, Any]]) -> Set[str]:
    return {r["base_var"] for r in rows if r.get("base_var")}


def print_file_block(title: str, files: List[str]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not files:
        print("(none)")
        return
    for p in files:
        print(p)


def print_set_block(title: str, vals: Set[str], limit: int = 200) -> None:
    print("\n" + "-" * 100)
    print(f"{title}  [count={len(vals)}]")
    print("-" * 100)
    for i, v in enumerate(sorted(vals)):
        if i >= limit:
            print(f"... ({len(vals) - limit} more)")
            break
        print(v)


def group_first_source(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        bv = r["base_var"]
        if bv and bv not in out:
            out[bv] = r
    return out


def debug_patient(
    patient: str,
    inclusion_root: str,
    root_fact_root: str,
    root_pid: Optional[str],
    facts_file: Optional[str],
    final_only: bool,
    show_all: bool,
) -> None:
    root_pid_eff = root_pid or patient

    fact_files = [facts_file] if facts_file else find_fact_files(inclusion_root, patient, final_only=final_only)
    root_files = find_root_files(root_fact_root, root_pid_eff)

    print_file_block("[facts files found]", fact_files)
    print_file_block("[root files found]", root_files)

    if not fact_files:
        print("\n[error] no fact files found. Check --patient / --inclusion-root / --facts-file")
        return
    if not root_files:
        print("\n[error] no root files found. Check --root-fact-root and possibly --root-pid")
        return

    fact_rows = summarize_records(fact_files)
    root_rows = summarize_records(root_files)

    fact_set = unique_base_vars(fact_rows)
    root_set = unique_base_vars(root_rows)

    overlap = fact_set & root_set
    fact_only = fact_set - root_set
    root_only = root_set - fact_set

    print_set_block("[fact base_vars]", fact_set)
    print_set_block("[root base_vars]", root_set)
    print_set_block("[overlap base_vars => should become is_root=1]", overlap)
    print_set_block("[fact-only base_vars => currently no root match]", fact_only)
    print_set_block("[root-only base_vars => present in root sources but not in facts]", root_only)

    print("\n" + "=" * 100)
    print("[summary]")
    print("=" * 100)
    print(f"patient          : {patient}")
    print(f"root_pid_used    : {root_pid_eff}")
    print(f"num_fact_rows    : {len(fact_rows)}")
    print(f"num_root_rows    : {len(root_rows)}")
    print(f"fact_base_vars   : {len(fact_set)}")
    print(f"root_base_vars   : {len(root_set)}")
    print(f"overlap_count    : {len(overlap)}")

    fact_first = group_first_source(fact_rows)
    root_first = group_first_source(root_rows)

    print("\n" + "=" * 100)
    print("[examples of overlapping vars]")
    print("=" * 100)
    if not overlap:
        print("(none)")
    else:
        for i, bv in enumerate(sorted(overlap)[:50]):
            fr = fact_first[bv]
            rr = root_first[bv]
            print(f"\nbase_var: {bv}")
            print(f"  fact file : {fr['file']}")
            print(f"  fact var  : {fr['var_name']}")
            print(f"  root file : {rr['file']}")
            print(f"  root var  : {rr['var_name']}")

    if show_all:
        print("\n" + "=" * 100)
        print("[fact rows detail]")
        print("=" * 100)
        for r in fact_rows:
            matched = r["base_var"] in root_set
            print(json.dumps({
                "matched_root": matched,
                "base_var": r["base_var"],
                "var_name": r["var_name"],
                "concept_id": r["concept_id"],
                "file": r["file"],
            }, ensure_ascii=False))

        print("\n" + "=" * 100)
        print("[root rows detail]")
        print("=" * 100)
        for r in root_rows:
            print(json.dumps({
                "base_var": r["base_var"],
                "var_name": r["var_name"],
                "concept_id": r["concept_id"],
                "file": r["file"],
            }, ensure_ascii=False))

    print("\n" + "=" * 100)
    print("[likely diagnoses if overlap_count == 0]")
    print("=" * 100)
    print("1. wrong patient folder on root side")
    print("2. wrong --root-fact-root")
    print("3. facts and root files use different variable naming schemes")
    print("4. expected root vars only exist in some file not currently scanned")
    print("5. canonical.final.jsonl on facts side is not the file you thought was being ingested")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, help="Patient ID on facts side")
    ap.add_argument("--root-pid", default=None, help="Optional different patient folder name on root side")
    ap.add_argument("--inclusion-root", default=DEFAULT_INCL_ROOT)
    ap.add_argument("--root-fact-root", default=DEFAULT_ROOT_FACT_ROOT)
    ap.add_argument("--facts-file", default=None, help="Optional exact facts file to inspect")
    ap.add_argument("--final-only", action="store_true", default=False)
    ap.add_argument("--show-all", action="store_true", default=False)
    args = ap.parse_args()

    debug_patient(
        patient=args.patient,
        inclusion_root=args.inclusion_root,
        root_fact_root=args.root_fact_root,
        root_pid=args.root_pid,
        facts_file=args.facts_file,
        final_only=args.final_only,
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()