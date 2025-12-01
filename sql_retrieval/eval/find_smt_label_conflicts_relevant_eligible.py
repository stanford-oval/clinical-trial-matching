#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_smt_label_conflicts_relevant_eligible.py

Find cases where SMT labels indicate NOT eligible:
  - unsatisfied_inclusion
  - explicit_contradiction

...but the LLM-judge outputs say:
  - any_subcohort_relevant == True
  - any_subcohort_eligible == True
  (and/or any_subcohort_relevant_and_eligible == True, when available)

Writes:
  - smt_label_conflicts_relevant_eligible.csv          (combined)
  - summary.csv                                       (counts by mode, tier, label)
  - OPTIONAL per-label CSVs under <out>/<label>/...   (enabled by default)

Optional extras (foldering + copying artifacts):
  - Copy SMT mbench artifacts for each detected item into:
      <out>/copied_smt_label_conflicts/<label>/<mode>/<patient_id>/<act_tag>/<parent_trial_id>/
    (label-separated subfolders; act/nonact separated when present)
  - In each copied trial folder, write patient/trial listings
  - Copy disease_categorized JSON per subcohort
  - Copy positive_constraint_literals_categorized/per_file JSONs per subcohort (inclusion/exclusion/assumed)

Assumed SMT outputs layout:
  <smt_root>/**/patient_labels/*.json   (recursive scan)

Assumed SMT mbench layout:
  <smt_mbench_root>/<mode>/<patient_id>/<parent_trial_id>/<subcohort_id>/

NEW:
  - Recognize mode="all-explore" from path and/or JSON.
  - Prefer folder all-explore over stale JSON mode="all".
  - Recognize act/nonact from path/filename/JSON.
  - Key runs by (patient_id, mode, act_tag), so act/nonact do not collide.
  - Optional mode filtering via --modes (comma-separated subset of chief,ccr,all,all-explore).

Example:
  python find_smt_label_conflicts_relevant_eligible.py \
    --smt-root /path/to/smt_retrieval_eval_out_rederived \
    --smt-mbench-root /path/to/smt_retrieval_eval_mbench \
    --out smt_label_conflicts_out \
    --copy-artifacts --add-listings --copy-disease --copy-poslits \
    --labels unsatisfied_inclusion,explicit_contradiction \
    --modes all,all-explore \
    --sample-pairs-pct 100 --sample-items-pct 50 --sample-seed 123
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


# ----------------------------
# Defaults (keep in sync with your repo paths if desired)
# ----------------------------

DEFAULT_SMT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")
DEFAULT_SMT_MBENCH_ROOT = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_mbench")

DEFAULT_CORPUS_JSONL = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus.jsonl")
DEFAULT_QUERIES_JSONL = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/queries.jsonl")

DEFAULT_DISEASE_ROOT = Path("<SATIR_ROOT>/build/disease_filtered_categorized")
DEFAULT_POSLIT_ROOT = Path("<SATIR_ROOT>/build/positive_constraint_literals_categorized")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)
TIERS = ("m", "n", "o")
KNOWN_MODES = ("chief", "ccr", "all", "all-explore")

DEFAULT_TARGET_LABELS = ("unsatisfied_inclusion", "explicit_contradiction")


# ----------------------------
# Sampling helpers (deterministic)
# ----------------------------

def _pct_ok(pct: float) -> float:
    try:
        x = float(pct)
    except Exception:
        return 1.0
    if x <= 0:
        return 0.0
    if x >= 100:
        return 1.0
    return x / 100.0


def _stable_hash01(s: str, seed: int) -> float:
    """
    Deterministic [0,1) float from string + seed.
    Stable across Python runs (sha256).
    """
    h = hashlib.sha256(f"{seed}::{s}".encode("utf-8")).hexdigest()
    v = int(h[:13], 16)  # 13 hex chars ~= 52 bits
    return (v % (2**52)) / float(2**52)


# ----------------------------
# Basic utils
# ----------------------------

def canon_nct(nct: Optional[str], collapse_subsuffix: bool = True) -> Optional[str]:
    if not nct:
        return None
    s = str(nct).strip()
    if not s:
        return None
    m = NCT_BASE_RE.match(s)
    if not m:
        return s.upper()
    base = m.group(1).upper()
    return base if collapse_subsuffix else s.upper()


def infer_patient_id_from_filename(p: Path) -> Optional[str]:
    name = p.name
    if "__" in name:
        return name.split("__", 1)[0]
    stem = p.stem
    return stem.split("__", 1)[0] if "__" in stem else stem


def _norm_token(s: str) -> str:
    return str(s).strip().lower().replace("_", "-")


def _path_tokens(p: Path) -> List[str]:
    toks: List[str] = []
    for part in p.parts:
        t = _norm_token(part)
        if t:
            toks.append(t)
    return toks


def _token_has_mode_all_explore(tok: str) -> bool:
    t = _norm_token(tok)
    if t == "all-explore":
        return True
    if "__all__noprevent__nonact" in t:
        return True
    if "clean-eval__all__noprevent__nonact" in t:
        return True
    if "retrieved-mappings__all__noprevent__nonact" in t:
        return True
    return False


def infer_mode_from_path(p: Path) -> str:
    parts = _path_tokens(p)

    for part in parts:
        if _token_has_mode_all_explore(part):
            return "all-explore"

    for m in ("chief", "ccr", "all"):
        if any(part == m for part in parts):
            return m

    for part in parts:
        if "__chief__" in part or part.startswith("chief__") or part.endswith("__chief"):
            return "chief"
        if "__ccr__" in part or part.startswith("ccr__") or part.endswith("__ccr"):
            return "ccr"
        if "__all__" in part or part.startswith("all__") or part.endswith("__all"):
            return "all"

    return "unknown"


def infer_prevent_tag_from_filename(p: Path) -> str:
    s = _norm_token(p.name)
    if "__noprevent" in s or s.endswith("-noprevent.json") or s.endswith("__noprevent.json"):
        return "noprevent"
    if "__prevent" in s or s.endswith("-prevent.json") or s.endswith("__prevent.json"):
        return "prevent"
    return "na"


def infer_prevent_tag_from_path(p: Path) -> str:
    for tok in _path_tokens(p):
        if "__noprevent" in tok or tok == "noprevent" or tok.endswith("__noprevent"):
            return "noprevent"
        if "__prevent" in tok or tok == "prevent" or tok.endswith("__prevent"):
            return "prevent"
    return "na"


def infer_act_tag_from_filename(p: Path) -> str:
    s = _norm_token(p.name)
    if "__nonact" in s or s.endswith("-nonact.json") or s.endswith("__nonact.json"):
        return "nonact"
    if "__act" in s or s.endswith("-act.json") or s.endswith("__act.json"):
        return "act"
    return "na"


def infer_act_tag_from_path(p: Path) -> str:
    for tok in _path_tokens(p):
        if "__nonact" in tok or tok == "nonact" or tok.endswith("__nonact"):
            return "nonact"
        if "__act" in tok or tok == "act" or tok.endswith("__act"):
            return "act"
    return "na"


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def norm_label(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    return t if t else None


def extract_ranked_list(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj.get("trials"), list) and obj["trials"]:
        return [x for x in obj["trials"] if isinstance(x, dict)]
    if isinstance(obj.get("canonical_trials"), list) and obj["canonical_trials"]:
        return [x for x in obj["canonical_trials"] if isinstance(x, dict)]
    if isinstance(obj.get("ranked"), list) and obj["ranked"]:
        return [x for x in obj["ranked"] if isinstance(x, dict)]
    return None


def _parse_modes_arg(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in KNOWN_MODES:
            raise ValueError(f"Unknown mode in --modes: {t} (known: {', '.join(KNOWN_MODES)})")
        out.append(t)
    return out or None


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Item:
    parent_trial_id: str
    rank: int
    label: Optional[str]
    any_rel: Optional[bool]
    any_elig: Optional[bool]
    any_rel_and_elig: Optional[bool]
    subcohorts: List[str]


@dataclass
class Run:
    patient_id: str
    mode: str
    prevent_tag: str
    act_tag: str
    items: List[Item]
    source_file: Path


def build_run(p: Path, collapse_subsuffix: bool, dedup: bool) -> Optional[Run]:
    obj = load_json(p)
    if not isinstance(obj, dict):
        return None

    patient_id = obj.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.strip():
        patient_id = infer_patient_id_from_filename(p)
    if not patient_id:
        return None

    mode_from_obj = obj.get("mode") if isinstance(obj.get("mode"), str) else None
    if isinstance(mode_from_obj, str):
        mode_from_obj = mode_from_obj.strip().lower()

    mode_from_path = infer_mode_from_path(p)

    if mode_from_path == "all-explore":
        mode = "all-explore"
    elif mode_from_obj in KNOWN_MODES:
        mode = mode_from_obj
    else:
        mode = mode_from_path

    prevent_tag_obj = obj.get("prevent_tag") if isinstance(obj.get("prevent_tag"), str) else None
    prevent_tag = (
        (prevent_tag_obj.strip().lower() if isinstance(prevent_tag_obj, str) and prevent_tag_obj.strip() else None)
        or infer_prevent_tag_from_filename(p)
        or infer_prevent_tag_from_path(p)
    )
    if prevent_tag == "na":
        prevent_tag = infer_prevent_tag_from_path(p)

    act_tag_obj = obj.get("act_tag") if isinstance(obj.get("act_tag"), str) else None
    act_tag_file = infer_act_tag_from_filename(p)
    act_tag_path = infer_act_tag_from_path(p)
    if isinstance(act_tag_obj, str) and act_tag_obj.strip():
        act_tag = act_tag_obj.strip().lower()
    elif act_tag_file != "na":
        act_tag = act_tag_file
    else:
        act_tag = act_tag_path

    trials = extract_ranked_list(obj)
    if not trials:
        return None

    if all(("rank" in t and isinstance(t.get("rank"), (int, float))) for t in trials):
        trials = sorted(trials, key=lambda t: int(t.get("rank", 10**9)))

    items: List[Item] = []
    seen: Set[str] = set()

    for idx, t in enumerate(trials, start=1):
        tid = t.get("trial_id")
        if tid is None:
            continue
        tid_raw = str(tid).strip()
        if not tid_raw:
            continue

        key = canon_nct(tid_raw, collapse_subsuffix=collapse_subsuffix) or tid_raw.upper()
        if dedup and key in seen:
            continue
        seen.add(key)

        rank = int(t.get("rank", idx)) if isinstance(t.get("rank"), (int, float)) else idx
        lab = norm_label(t.get("label"))

        any_rel = t.get("any_subcohort_relevant")
        any_elig = t.get("any_subcohort_eligible")
        any_re = t.get("any_subcohort_relevant_and_eligible")

        subs: List[str] = []
        summ = t.get("subcohort_judge_summary")
        if isinstance(summ, list):
            for s in summ:
                if isinstance(s, dict) and isinstance(s.get("subcohort_id"), str):
                    subs.append(s["subcohort_id"])
        if not subs:
            subs = [tid_raw]

        items.append(Item(
            parent_trial_id=key,
            rank=rank,
            label=lab,
            any_rel=any_rel if isinstance(any_rel, bool) else None,
            any_elig=any_elig if isinstance(any_elig, bool) else None,
            any_rel_and_elig=any_re if isinstance(any_re, bool) else None,
            subcohorts=subs,
        ))

    return Run(
        patient_id=patient_id,
        mode=mode,
        prevent_tag=prevent_tag,
        act_tag=act_tag,
        items=items,
        source_file=p,
    ) if items else None


def collect_runs(root: Path, collapse_subsuffix: bool, dedup: bool) -> Dict[Tuple[str, str, str], Run]:
    out: Dict[Tuple[str, str, str], Run] = {}
    for p in root.rglob("patient_labels/*.json"):
        run = build_run(p, collapse_subsuffix=collapse_subsuffix, dedup=dedup)
        if not run:
            continue
        k = (run.patient_id, run.mode, run.act_tag)
        prev = out.get(k)
        if prev is None or len(run.items) > len(prev.items):
            out[k] = run
    return out


# ----------------------------
# SMT tier cutoffs
# ----------------------------

def tier_cutoffs_smt(run: Run) -> Dict[str, int]:
    c_all = sum(1 for it in run.items if it.label == "all_satisfied")
    c_uns = sum(1 for it in run.items if it.label == "unsatisfied_inclusion")
    c_con = sum(1 for it in run.items if it.label == "explicit_contradiction")
    total = len(run.items)
    if (c_all + c_uns + c_con) == 0:
        return {"m": min(10, total), "n": min(25, total), "o": min(50, total)}
    return {"m": min(c_all, total), "n": min(c_all + c_uns, total), "o": min(c_all + c_uns + c_con, total)}


# ----------------------------
# CSV writing
# ----------------------------

def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


# ----------------------------
# corpus/queries indexing + writing
# ----------------------------

def load_jsonl_index(path: Path, id_key: str = "_id") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get(id_key), str):
                out[obj[id_key]] = obj
    return out


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_readme(path: Path, patient_id: str, mode: str, act_tag: str, parent_trial_id: str, tier: str, label: str) -> None:
    act_note = f"- act_tag: {act_tag}\n" if act_tag and act_tag != "na" else ""
    txt = (
        f"SMT label conflict artifact (SMT says {label}, but judge says relevant+eligible)\n"
        f"- patient_id: {patient_id}\n"
        f"- mode: {mode}\n"
        f"{act_note}"
        f"- tier: {tier}\n"
        f"- parent_trial_id: {parent_trial_id}\n"
        f"- smt_label: {label}\n"
        f"\n"
        f"Files added by script (if enabled):\n"
        f"- patient_query.json\n"
        f"- trial_listing.json\n"
        f"- trial_listing__<subcohort_id>.json\n"
        f"- disease/<subcohort_id>_disease_link_filter_summary.json\n"
        f"- positive_constraint_literals/<subcohort_id>_*_program*.smt2.json\n"
        f"- mbench/<subcohort_id>/... (copied)\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


# ----------------------------
# Copy helpers
# ----------------------------

def copy_dir(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> Tuple[bool, str]:
    if not src.exists() or not src.is_dir():
        return False, "missing_src"
    if dst.exists():
        if not overwrite:
            return True, "exists_skip"
        if not dry_run:
            shutil.rmtree(dst)
    if dry_run:
        return True, "would_copy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return True, "copied"


def copy_file(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> Tuple[bool, str]:
    if not src.exists() or not src.is_file():
        return False, "missing_src"
    if dst.exists() and not overwrite:
        return True, "exists_skip"
    if dry_run:
        return True, "would_copy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True, "copied"


def _expand_subcohorts_for_copy(
    *,
    mode: str,
    patient_id: str,
    parent_trial_id: str,
    subs: List[str],
    smt_mbench_root: Path,
    disease_root: Path,
    poslit_root: Path,
) -> List[str]:
    expanded: Set[str] = set(x.strip() for x in subs if isinstance(x, str) and x.strip())

    base = canon_nct(parent_trial_id, collapse_subsuffix=True)
    if not base:
        return sorted(expanded)

    base_up = base.upper()

    mbench_parent = smt_mbench_root / mode / patient_id / parent_trial_id
    if mbench_parent.exists() and mbench_parent.is_dir():
        for child in mbench_parent.iterdir():
            if not child.is_dir():
                continue
            name = child.name.strip()
            if not name:
                continue
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", name.upper()):
                expanded.add(name)

    if disease_root.exists() and disease_root.is_dir():
        for f in disease_root.glob(f"{base_up}[a-z]_disease_link_filter_summary.json"):
            stem = f.name.split("_disease_link_filter_summary.json", 1)[0]
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", stem.upper()):
                expanded.add(stem)

    per_file = poslit_root / "per_file"
    if per_file.exists() and per_file.is_dir():
        for f in per_file.glob(f"{base_up}[a-z]_*program*.smt2.json"):
            prefix = f.name.split("_", 1)[0]
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", prefix.upper()):
                expanded.add(prefix)

    return sorted(expanded)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find SMT items labeled unsatisfied_inclusion / explicit_contradiction where judge says relevant AND eligible."
    )
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--out", type=Path, default=Path("./smt_label_conflicts_out"))
    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--copy-tier", default="m,n,o", help="Comma list of tiers to materialize (m,n,o).")
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Optional comma-separated modes filter (subset of chief,ccr,all,all-explore). "
             "If set, only those modes are evaluated.",
    )

    ap.add_argument(
        "--labels",
        default=",".join(DEFAULT_TARGET_LABELS),
        help="Comma list of SMT labels to target (default: unsatisfied_inclusion,explicit_contradiction).",
    )

    ap.add_argument(
        "--write-per-label-csv",
        action="store_true",
        default=True,
        help="If set, also write <out>/<label>/smt_label_conflicts_relevant_eligible.csv per label.",
    )

    ap.add_argument("--sample-seed", type=int, default=42, help="Seed for deterministic sampling.")
    ap.add_argument("--sample-pairs-pct", type=float, default=100.0,
                    help="Percent of (patient_id,mode,act_tag) runs to scan. 100 = scan all.")
    ap.add_argument("--sample-items-pct", type=float, default=100.0,
                    help="Percent of detected items to keep. 100 = keep all.")

    ap.add_argument("--copy-artifacts", action="store_true", default=True,
                    help="If set, create folders under <out>/copied_smt_label_conflicts/<label>/... and copy extras.")
    ap.add_argument("--smt-mbench-root", type=Path, default=DEFAULT_SMT_MBENCH_ROOT)
    ap.add_argument("--copy-overwrite", action="store_true")
    ap.add_argument("--copy-dry-run", action="store_true")

    ap.add_argument("--add-listings", action="store_true", default=True,
                    help="If set, write patient_query.json and trial_listing.json into each copied folder.")
    ap.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS_JSONL)
    ap.add_argument("--queries-jsonl", type=Path, default=DEFAULT_QUERIES_JSONL)

    ap.add_argument("--copy-disease", action="store_true", default=True,
                    help="Copy <disease-root>/<sub>_disease_link_filter_summary.json into each folder.")
    ap.add_argument("--disease-root", type=Path, default=DEFAULT_DISEASE_ROOT)

    ap.add_argument("--copy-poslits", action="store_true", default=True,
                    help="Copy positive literal categorized files from <poslit-root>/per_file for each subcohort.")
    ap.add_argument("--poslit-root", type=Path, default=DEFAULT_POSLIT_ROOT)

    args = ap.parse_args()

    if not args.smt_root.exists():
        print(f"[error] smt root not found: {args.smt_root}", file=sys.stderr)
        sys.exit(2)

    if args.copy_artifacts and not args.smt_mbench_root.exists():
        print(f"[error] smt mbench root not found: {args.smt_mbench_root}", file=sys.stderr)
        sys.exit(2)

    if args.add_listings:
        if not args.corpus_jsonl.exists():
            print(f"[error] corpus jsonl not found: {args.corpus_jsonl}", file=sys.stderr)
            sys.exit(2)
        if not args.queries_jsonl.exists():
            print(f"[error] queries jsonl not found: {args.queries_jsonl}", file=sys.stderr)
            sys.exit(2)

    if args.copy_disease and not args.disease_root.exists():
        print(f"[error] disease root not found: {args.disease_root}", file=sys.stderr)
        sys.exit(2)

    if args.copy_poslits and not args.poslit_root.exists():
        print(f"[error] poslit root not found: {args.poslit_root}", file=sys.stderr)
        sys.exit(2)

    target_labels = [x.strip().lower() for x in str(args.labels).split(",") if x.strip()]
    if not target_labels:
        target_labels = list(DEFAULT_TARGET_LABELS)

    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)

    smt_runs = collect_runs(args.smt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)

    if mode_filter is not None:
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}

    pair_frac = _pct_ok(args.sample_pairs_pct)
    item_frac = _pct_ok(args.sample_items_pct)

    all_keys = list(smt_runs.keys())
    if pair_frac < 1.0:
        kept_keys = [k for k in all_keys if _stable_hash01(f"{k[0]}::{k[1]}::{k[2]}", args.sample_seed) < pair_frac]
        kept_keys = sorted(kept_keys, key=lambda x: (x[1], x[2], x[0]))
        print(f"[info] pair-sampling: kept {len(kept_keys)}/{len(all_keys)} pairs ({args.sample_pairs_pct:.2f}%)")
        keys = kept_keys
    else:
        keys = sorted(all_keys, key=lambda x: (x[1], x[2], x[0]))

    copy_tiers: Set[str] = {x.strip() for x in args.copy_tier.split(",") if x.strip()}
    copy_tiers = {t for t in copy_tiers if t in set(TIERS)}
    if not copy_tiers:
        copy_tiers = set(TIERS)

    rows: List[Dict[str, Any]] = []
    rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[Tuple[str, str, str, str], int] = {}  # (mode, act_tag, tier, label) -> count
    pairs_eval = 0

    to_materialize: Dict[Tuple[str, str, str, str, str, str], Set[str]] = {}
    # (mode, act_tag, patient_id, tier, parent_trial_id, label) -> subcohort set

    for (patient_id, mode, act_tag) in keys:
        run = smt_runs[(patient_id, mode, act_tag)]
        pairs_eval += 1

        smtK = tier_cutoffs_smt(run)

        for tier in TIERS:
            k = smtK[tier]
            top = run.items[:k]

            for it in top:
                if it.label not in target_labels:
                    continue

                judge_re = (it.any_rel_and_elig is True) or (it.any_rel is True and it.any_elig is True)
                if not judge_re:
                    continue

                if item_frac < 1.0:
                    skey = f"{patient_id}::{mode}::{act_tag}::{tier}::{it.parent_trial_id}::{it.label}"
                    if _stable_hash01(skey, args.sample_seed) >= item_frac:
                        continue

                row = {
                    "patient_id": patient_id,
                    "mode": mode,
                    "act_tag": act_tag,
                    "tier": tier,
                    "parent_trial_id": it.parent_trial_id,
                    "smt_rank": it.rank,
                    "smt_label": it.label,
                    "any_subcohort_relevant": it.any_rel,
                    "any_subcohort_eligible": it.any_elig,
                    "any_subcohort_relevant_and_eligible": it.any_rel_and_elig,
                    "smt_cutoff_k": smtK[tier],
                    "smt_file": str(run.source_file),
                    "num_subcohorts": len(it.subcohorts),
                    "subcohorts": "|".join(it.subcohorts),
                    "reason": "smt_label_negative_but_judge_relevant_and_eligible",
                }
                rows.append(row)
                rows_by_label.setdefault(it.label or "null", []).append(row)

                summary[(mode, act_tag, tier, it.label or "null")] = summary.get((mode, act_tag, tier, it.label or "null"), 0) + 1

                if args.copy_artifacts and tier in copy_tiers:
                    key = (mode, act_tag, patient_id, tier, it.parent_trial_id, it.label or "null")
                    if key not in to_materialize:
                        to_materialize[key] = set()
                    for sub in it.subcohorts:
                        if isinstance(sub, str) and sub.strip():
                            to_materialize[key].add(sub.strip())

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "patient_id", "mode", "act_tag", "tier",
        "parent_trial_id",
        "smt_rank", "smt_label",
        "any_subcohort_relevant", "any_subcohort_eligible", "any_subcohort_relevant_and_eligible",
        "smt_cutoff_k",
        "num_subcohorts", "subcohorts",
        "reason",
        "smt_file",
    ]

    write_csv(out_dir / "smt_label_conflicts_relevant_eligible.csv", header, rows)

    sum_rows = []
    for (mode, act_tag, tier, lab), cnt in sorted(summary.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        sum_rows.append({"mode": mode, "act_tag": act_tag, "tier": tier, "label": lab, "count": cnt})
    write_csv(out_dir / "summary.csv", ["mode", "act_tag", "tier", "label", "count"], sum_rows)

    if args.write_per_label_csv:
        for lab, lab_rows in rows_by_label.items():
            write_csv(out_dir / lab / "smt_label_conflicts_relevant_eligible.csv", header, lab_rows)

    print(f"[ok] wrote {out_dir / 'smt_label_conflicts_relevant_eligible.csv'}")
    print(f"[ok] wrote {out_dir / 'summary.csv'}")
    if args.write_per_label_csv:
        print(f"[ok] wrote per-label CSVs under: {out_dir}/<label>/smt_label_conflicts_relevant_eligible.csv")
    print(f"[info] patient_mode_act_pairs_scanned={pairs_eval}  matches={len(rows)}")
    if mode_filter is not None:
        print(f"[info] mode_filter={mode_filter}")

    if not args.copy_artifacts:
        return

    corpus_index: Dict[str, Dict[str, Any]] = {}
    query_index: Dict[str, Dict[str, Any]] = {}
    if args.add_listings:
        print(f"[info] loading corpus index: {args.corpus_jsonl}")
        corpus_index = load_jsonl_index(args.corpus_jsonl, id_key="_id")
        print(f"[info] loading queries index: {args.queries_jsonl}")
        query_index = load_jsonl_index(args.queries_jsonl, id_key="_id")
        print(f"[info] loaded corpus={len(corpus_index)} queries={len(query_index)}")

    copied_dirs = 0
    skipped_dirs = 0
    missing_dirs = 0

    copied_files = 0
    skipped_files = 0
    missing_files = 0

    uniq_parent: Dict[Tuple[str, str, str, str, str], str] = {}  # -> tier hint
    parent_to_subs: Dict[Tuple[str, str, str, str, str], Set[str]] = {}

    for (mode, act_tag, patient_id, tier, parent_trial_id, lab), subs in to_materialize.items():
        k = (lab, mode, act_tag, patient_id, parent_trial_id)
        uniq_parent.setdefault(k, tier)
        parent_to_subs.setdefault(k, set()).update(subs)

    print(f"[info] materialize: unique (label,mode,act_tag,patient,parent_trial)={len(uniq_parent)}")

    for (lab, mode, act_tag, patient_id, parent_trial_id), tier_hint in sorted(uniq_parent.items()):
        subs = sorted(parent_to_subs.get((lab, mode, act_tag, patient_id, parent_trial_id), set()))

        subs = _expand_subcohorts_for_copy(
            mode=mode,
            patient_id=patient_id,
            parent_trial_id=parent_trial_id,
            subs=subs,
            smt_mbench_root=args.smt_mbench_root,
            disease_root=args.disease_root,
            poslit_root=args.poslit_root,
        )

        if act_tag and act_tag != "na":
            base_dst = out_dir / "copied_smt_label_conflicts" / lab / mode / patient_id / act_tag / parent_trial_id
        else:
            base_dst = out_dir / "copied_smt_label_conflicts" / lab / mode / patient_id / parent_trial_id
        safe_mkdir(base_dst)

        if not args.copy_dry_run:
            write_readme(base_dst / "README.txt", patient_id, mode, act_tag, parent_trial_id, tier_hint, lab)

        if args.add_listings:
            patient_obj = query_index.get(patient_id)
            parent_obj = corpus_index.get(parent_trial_id)

            if args.copy_dry_run:
                if patient_obj is None:
                    print(f"[warn] missing patient in queries.jsonl: {patient_id}")
                if parent_obj is None:
                    print(f"[warn] missing parent trial in corpus.jsonl: {parent_trial_id}")
                print(f"[would_write] {base_dst / 'patient_query.json'} and {base_dst / 'trial_listing.json'} (+ per-subcohort)")
            else:
                write_json(
                    base_dst / "patient_query.json",
                    patient_obj if patient_obj is not None else {"_id": patient_id, "missing": True},
                )
                write_json(
                    base_dst / "trial_listing.json",
                    parent_obj if parent_obj is not None else {"_id": parent_trial_id, "missing": True},
                )

                for sub in subs:
                    sub_obj = corpus_index.get(sub)
                    if sub_obj is not None:
                        write_json(base_dst / f"trial_listing__{sub}.json", sub_obj)

        for sub in subs:
            src = args.smt_mbench_root / mode / patient_id / parent_trial_id / sub
            dst = base_dst / "mbench" / sub
            ok, msg = copy_dir(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
            if not ok and msg == "missing_src":
                missing_dirs += 1
                print(f"[missing_mbench] {src}")
            elif ok and msg == "exists_skip":
                skipped_dirs += 1
            elif ok and msg in {"copied", "would_copy"}:
                copied_dirs += 1
            print(f"[mbench:{msg}] {src} -> {dst}")

        if args.copy_disease:
            for sub in subs:
                src = args.disease_root / f"{sub}_disease_link_filter_summary.json"
                dst = base_dst / "disease" / src.name
                ok, msg = copy_file(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                if not ok and msg == "missing_src":
                    missing_files += 1
                    print(f"[missing_disease] {src}")
                elif ok and msg == "exists_skip":
                    skipped_files += 1
                elif ok and msg in {"copied", "would_copy"}:
                    copied_files += 1
                print(f"[disease:{msg}] {src} -> {dst}")

        if args.copy_poslits:
            per_file = args.poslit_root / "per_file"
            for sub in subs:
                matches = sorted(per_file.glob(f"{sub}_*program*.smt2.json"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_poslits] {per_file} glob={sub}_*program*.smt2.json")
                    continue
                for src in matches:
                    dst = base_dst / "positive_constraint_literals" / src.name
                    ok, msg = copy_file(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                    if not ok and msg == "missing_src":
                        missing_files += 1
                        print(f"[missing_poslit] {src}")
                    elif ok and msg == "exists_skip":
                        skipped_files += 1
                    elif ok and msg in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[poslit:{msg}] {src} -> {dst}")

    print(
        f"[done] materialize dirs: copied={copied_dirs} skipped={skipped_dirs} missing={missing_dirs} | "
        f"files: copied={copied_files} skipped={skipped_files} missing={missing_files}"
    )
    if args.copy_dry_run:
        print("[info] copy-dry-run enabled: no files were copied / written")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)