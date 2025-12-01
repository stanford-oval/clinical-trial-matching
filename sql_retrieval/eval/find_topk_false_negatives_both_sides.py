#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_topk_false_negatives_both_sides.py

Compute BOTH directions of false negatives for each fixed K, using the SAME
positive notion on both sides:

    positive := confirmed relevant-and-eligible

Operational definitions
-----------------------
TrialGPT positive at K:
    - trial appears within top-K
    - relevant == True
    - eligible == True

SMT positive:
    - any_subcohort_relevant_and_eligible == True

This preserves the relevant-and-eligible restriction on BOTH sides.

Outputs
-------
For each K, under:
    <out>/top<K>/

we write:

Detailed rows
-------------
trialgpt_false_negatives.csv
    SMT positive, but TrialGPT top-K is not positive

smt_false_negatives.csv
    TrialGPT top-K positive, but SMT is not positive

Summaries
---------
summary_combined.csv
    One row per mode, with both sides together:
        mode,
        trialgpt_false_negatives,
        smt_false_negatives

trialgpt_false_negatives_summary_by_conflict.csv
smt_false_negatives_summary_by_conflict.csv

Notes
-----
- No artifact copying in this script.
- Canonical trial identity is matched by canonicalized NCT id.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_TRIALGPT_ROOT_BASE = Path(
    "<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out"
)
DEFAULT_SMT_ROOT = Path(
    "<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out"
)
DEFAULT_TOPK_LIST = "300,400,500"

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)
KNOWN_MODES = ("chief", "ccr", "all", "all-explore")


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def is_truthy(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"1", "true", "t", "yes", "y"}:
            return True
        if t in {"0", "false", "f", "no", "n"}:
            return False
    return None


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


def infer_mode_from_path(p: Path) -> str:
    parts = _path_tokens(p)

    for part in parts:
        if part == "all-explore":
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


def extract_rel_elig(tr: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    rel_keys = ["relevant", "is_relevant", "judge_relevant", "any_subcohort_relevant", "rel"]
    elig_keys = ["eligible", "is_eligible", "judge_eligible", "any_subcohort_eligible", "elig"]

    rel = None
    for k in rel_keys:
        if k in tr:
            rel = is_truthy(tr.get(k))
            if rel is not None:
                break

    elig = None
    for k in elig_keys:
        if k in tr:
            elig = is_truthy(tr.get(k))
            if elig is not None:
                break

    if "relevant_and_eligible" in tr:
        rae = is_truthy(tr.get("relevant_and_eligible"))
        if rae is True:
            rel = True if rel is None else rel
            elig = True if elig is None else elig

    lab = tr.get("label")
    if elig is None and isinstance(lab, str):
        l = lab.strip().lower()
        if l == "all_satisfied":
            elig = True
        elif l in {"unsatisfied_inclusion", "explicit_contradiction"}:
            elig = False

    return rel, elig


def extract_trial_key(tr: Dict[str, Any], collapse_subsuffix: bool) -> Optional[str]:
    for k in ("canonical_nct_id", "nct_id", "trial_nct_id", "nct"):
        if tr.get(k):
            return canon_nct(str(tr[k]), collapse_subsuffix=collapse_subsuffix) or str(tr[k]).upper()
    if tr.get("trial_id") is not None:
        tid = str(tr["trial_id"])
        return canon_nct(tid, collapse_subsuffix=collapse_subsuffix) or tid.upper()
    return None


def _parse_modes_arg(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    out: List[str] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in KNOWN_MODES:
            raise ValueError(f"Unknown mode in --modes: {t} (known: {', '.join(KNOWN_MODES)})")
        out.append(t)
    return out or None


def parse_topk_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"All K values must be positive, got: {k}")
        vals.append(k)
    if not vals:
        raise ValueError("No valid K values provided.")
    return vals


def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


# ----------------------------------------------------------------------
# Wrapper-mode normalization
# ----------------------------------------------------------------------

def normalize_wrapper_mode(compose_mode: str, prevent_tag: str, act_tag: str) -> str:
    cm = (compose_mode or "").strip().lower()
    pt = (prevent_tag or "").strip().lower()
    at = (act_tag or "").strip().lower()

    if cm == "all-explore":
        return "all-explore"

    if cm == "all" and pt == "prevent" and at == "nonact":
        return "all-explore"

    if cm in {"chief", "ccr", "all"}:
        return cm

    return cm or "unknown"


# ----------------------------------------------------------------------
# TrialGPT / SMT runs
# ----------------------------------------------------------------------

@dataclass
class TrialGPTItem:
    key: str
    raw_trial_id: str
    rank: int
    relevant: Optional[bool]
    eligible: Optional[bool]
    label: Optional[str]


@dataclass
class TrialGPTRun:
    patient_id: str
    mode: str
    prevent_tag: str
    items: List[TrialGPTItem]
    source_file: Path


def build_trialgpt_run(p: Path, collapse_subsuffix: bool, dedup: bool) -> Optional[TrialGPTRun]:
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

    prevent_tag = (
        (obj.get("prevent_tag").strip().lower() if isinstance(obj.get("prevent_tag"), str) and obj.get("prevent_tag").strip() else None)
        or infer_prevent_tag_from_filename(p)
        or infer_prevent_tag_from_path(p)
    )

    act_tag = (
        (obj.get("act_tag").strip().lower() if isinstance(obj.get("act_tag"), str) and obj.get("act_tag").strip() else None)
        or infer_act_tag_from_filename(p)
        or infer_act_tag_from_path(p)
    )

    raw_mode = "all-explore" if mode_from_path == "all-explore" else (mode_from_obj or mode_from_path)
    mode = normalize_wrapper_mode(raw_mode, prevent_tag, act_tag)

    trials = extract_ranked_list(obj)
    if not trials:
        return None

    if all(("rank" in t and isinstance(t.get("rank"), (int, float))) for t in trials):
        trials = sorted(trials, key=lambda t: int(t.get("rank", 10**9)))

    items: List[TrialGPTItem] = []
    seen: Set[str] = set()

    for idx, t in enumerate(trials, start=1):
        key = extract_trial_key(t, collapse_subsuffix=collapse_subsuffix)
        if not key:
            continue
        if dedup and key in seen:
            continue
        seen.add(key)

        raw_trial_id = str(
            t.get("trial_id")
            or t.get("nct_id")
            or t.get("trial_nct_id")
            or t.get("canonical_nct_id")
            or key
        ).strip()

        rel, elig = extract_rel_elig(t)
        rank = int(t.get("rank", idx)) if isinstance(t.get("rank"), (int, float)) else idx
        lab = norm_label(t.get("label"))

        items.append(
            TrialGPTItem(
                key=key,
                raw_trial_id=raw_trial_id,
                rank=rank,
                relevant=rel,
                eligible=elig,
                label=lab,
            )
        )

    return TrialGPTRun(
        patient_id=patient_id,
        mode=mode,
        prevent_tag=prevent_tag,
        items=items,
        source_file=p,
    ) if items else None


def collect_trialgpt_runs(root: Path, collapse_subsuffix: bool, dedup: bool) -> Dict[Tuple[str, str], TrialGPTRun]:
    out: Dict[Tuple[str, str], TrialGPTRun] = {}
    for p in root.rglob("patient_labels/*.json"):
        run = build_trialgpt_run(p, collapse_subsuffix=collapse_subsuffix, dedup=dedup)
        if not run:
            continue
        k = (run.patient_id, run.mode)
        prev = out.get(k)
        if prev is None or len(run.items) > len(prev.items):
            out[k] = run
    return out


@dataclass
class SMTItem:
    parent_trial_id: str
    raw_trial_id: str
    rank: int
    label: Optional[str]
    any_rel: Optional[bool]
    any_elig: Optional[bool]
    any_rel_and_elig: Optional[bool]
    subcohorts: List[str]


@dataclass
class SMTRun:
    patient_id: str
    mode: str
    prevent_tag: str
    act_tag: str
    items: List[SMTItem]
    source_file: Path


def build_smt_run(p: Path, collapse_subsuffix: bool, dedup: bool) -> Optional[SMTRun]:
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

    raw_mode = "all-explore" if mode_from_path == "all-explore" else (mode_from_obj or mode_from_path)
    mode = normalize_wrapper_mode(raw_mode, prevent_tag, act_tag)

    trials = extract_ranked_list(obj)
    if not trials:
        return None

    if all(("rank" in t and isinstance(t.get("rank"), (int, float))) for t in trials):
        trials = sorted(trials, key=lambda t: int(t.get("rank", 10**9)))

    items: List[SMTItem] = []
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

        items.append(
            SMTItem(
                parent_trial_id=key,
                raw_trial_id=tid_raw,
                rank=rank,
                label=lab,
                any_rel=any_rel if isinstance(any_rel, bool) else None,
                any_elig=any_elig if isinstance(any_elig, bool) else None,
                any_rel_and_elig=any_re if isinstance(any_re, bool) else None,
                subcohorts=subs,
            )
        )

    return SMTRun(
        patient_id=patient_id,
        mode=mode,
        prevent_tag=prevent_tag,
        act_tag=act_tag,
        items=items,
        source_file=p,
    ) if items else None


def collect_smt_runs(root: Path, collapse_subsuffix: bool, dedup: bool) -> Dict[Tuple[str, str, str], SMTRun]:
    out: Dict[Tuple[str, str, str], SMTRun] = {}
    for p in root.rglob("patient_labels/*.json"):
        run = build_smt_run(p, collapse_subsuffix=collapse_subsuffix, dedup=dedup)
        if not run:
            continue
        k = (run.patient_id, run.mode, run.act_tag)
        prev = out.get(k)
        if prev is None or len(run.items) > len(prev.items):
            out[k] = run
    return out


# ----------------------------------------------------------------------
# Conflict annotation
# ----------------------------------------------------------------------

@dataclass
class TrialGPTConflictMatch:
    conflict_label: str
    tg_rank: Optional[int]
    tg_relevant: Optional[bool]
    tg_eligible: Optional[bool]
    tg_label: Optional[str]
    source_file: Optional[Path]


def choose_best_trialgpt_conflict_for_trial(
    trial_key: str,
    tg_run: Optional[TrialGPTRun],
    tg_k: int,
) -> TrialGPTConflictMatch:
    if tg_run is None:
        return TrialGPTConflictMatch(
            conflict_label="no_hit",
            tg_rank=None,
            tg_relevant=None,
            tg_eligible=None,
            tg_label=None,
            source_file=None,
        )

    matches = [x for x in tg_run.items if x.key == trial_key]
    if not matches:
        return TrialGPTConflictMatch(
            conflict_label="no_hit",
            tg_rank=None,
            tg_relevant=None,
            tg_eligible=None,
            tg_label=None,
            source_file=tg_run.source_file,
        )

    best = min(matches, key=lambda x: x.rank)
    rel = best.relevant
    elig = best.eligible
    in_topk = best.rank <= tg_k

    if in_topk:
        if rel is True and elig is True:
            label = "topk_re_and_elig"
        elif rel is False and elig is False:
            label = "topk_not_relevant_and_not_eligible"
        elif rel is False:
            label = "topk_not_relevant"
        elif elig is False:
            label = "topk_not_eligible"
        else:
            label = "topk_unclear"
    else:
        if rel is True and elig is True:
            label = "outside_topk_re_and_elig"
        elif rel is False and elig is False:
            label = "outside_topk_not_relevant_and_not_eligible"
        elif rel is False:
            label = "outside_topk_not_relevant"
        elif elig is False:
            label = "outside_topk_not_eligible"
        else:
            label = "outside_topk_unclear"

    return TrialGPTConflictMatch(
        conflict_label=label,
        tg_rank=best.rank,
        tg_relevant=best.relevant,
        tg_eligible=best.eligible,
        tg_label=best.label,
        source_file=tg_run.source_file,
    )


@dataclass
class SMTConflictMatch:
    conflict_label: str
    smt_rank: Optional[int]
    smt_label: Optional[str]
    smt_any_relevant: Optional[bool]
    smt_any_eligible: Optional[bool]
    smt_any_relevant_and_eligible: Optional[bool]
    smt_num_subcohorts: int
    smt_subcohorts: List[str]
    source_file: Optional[Path]


def choose_best_smt_conflict_for_trial(
    trial_key: str,
    smt_runs_for_pair: List[SMTRun],
) -> SMTConflictMatch:
    matches: List[Tuple[SMTItem, SMTRun]] = []
    for run in smt_runs_for_pair:
        for it in run.items:
            if it.parent_trial_id == trial_key:
                matches.append((it, run))

    if not matches:
        return SMTConflictMatch(
            conflict_label="no_hit",
            smt_rank=None,
            smt_label=None,
            smt_any_relevant=None,
            smt_any_eligible=None,
            smt_any_relevant_and_eligible=None,
            smt_num_subcohorts=0,
            smt_subcohorts=[],
            source_file=None,
        )

    best_item, best_run = min(matches, key=lambda x: x[0].rank)

    if best_item.any_rel_and_elig is True:
        c = "re_and_elig"
    elif best_item.label == "explicit_contradiction":
        c = "explicit_contradiction"
    elif best_item.label == "unsatisfied_inclusion":
        c = "unsatisfied_inclusion"
    elif best_item.label == "all_satisfied":
        c = "eligible_but_not_confirmed_re_and_elig"
    elif best_item.label is None:
        c = "unclear_label"
    else:
        c = f"other_label:{best_item.label}"

    return SMTConflictMatch(
        conflict_label=c,
        smt_rank=best_item.rank,
        smt_label=best_item.label,
        smt_any_relevant=best_item.any_rel,
        smt_any_eligible=best_item.any_elig,
        smt_any_relevant_and_eligibile=best_item.any_rel_and_elig,  # type: ignore[arg-type]
        smt_num_subcohorts=len(best_item.subcohorts),
        smt_subcohorts=list(best_item.subcohorts),
        source_file=best_run.source_file,
    )


# patch typo in dataclass construction by using an explicit helper wrapper
def choose_best_smt_conflict_for_trial_fixed(
    trial_key: str,
    smt_runs_for_pair: List[SMTRun],
) -> SMTConflictMatch:
    matches: List[Tuple[SMTItem, SMTRun]] = []
    for run in smt_runs_for_pair:
        for it in run.items:
            if it.parent_trial_id == trial_key:
                matches.append((it, run))

    if not matches:
        return SMTConflictMatch(
            conflict_label="no_hit",
            smt_rank=None,
            smt_label=None,
            smt_any_relevant=None,
            smt_any_eligible=None,
            smt_any_relevant_and_eligible=None,
            smt_num_subcohorts=0,
            smt_subcohorts=[],
            source_file=None,
        )

    best_item, best_run = min(matches, key=lambda x: x[0].rank)

    if best_item.any_rel_and_elig is True:
        c = "re_and_elig"
    elif best_item.label == "explicit_contradiction":
        c = "explicit_contradiction"
    elif best_item.label == "unsatisfied_inclusion":
        c = "unsatisfied_inclusion"
    elif best_item.label == "all_satisfied":
        c = "eligible_but_not_confirmed_re_and_elig"
    elif best_item.label is None:
        c = "unclear_label"
    else:
        c = f"other_label:{best_item.label}"

    return SMTConflictMatch(
        conflict_label=c,
        smt_rank=best_item.rank,
        smt_label=best_item.label,
        smt_any_relevant=best_item.any_rel,
        smt_any_eligible=best_item.any_elig,
        smt_any_relevant_and_eligible=best_item.any_rel_and_elig,
        smt_num_subcohorts=len(best_item.subcohorts),
        smt_subcohorts=list(best_item.subcohorts),
        source_file=best_run.source_file,
    )


# ----------------------------------------------------------------------
# Per-K analysis
# ----------------------------------------------------------------------

def analyze_one_k(
    *,
    k: int,
    trialgpt_root: Path,
    smt_root: Path,
    out_dir: Path,
    collapse_subsuffix: bool,
    dedup_canonical: bool,
    require_overlap: bool,
    mode_filter: Optional[List[str]],
) -> None:
    if not trialgpt_root.exists():
        raise FileNotFoundError(f"trialgpt root not found for K={k}: {trialgpt_root}")
    if not smt_root.exists():
        raise FileNotFoundError(f"smt root not found: {smt_root}")

    tg_runs = collect_trialgpt_runs(
        trialgpt_root,
        collapse_subsuffix=collapse_subsuffix,
        dedup=dedup_canonical,
    )
    smt_runs = collect_smt_runs(
        smt_root,
        collapse_subsuffix=collapse_subsuffix,
        dedup=dedup_canonical,
    )

    if mode_filter is not None:
        tg_runs = {kk: vv for kk, vv in tg_runs.items() if kk[1] in mode_filter}
        smt_runs = {kk: vv for kk, vv in smt_runs.items() if kk[1] in mode_filter}

    smt_pair_index: Dict[Tuple[str, str], List[SMTRun]] = {}
    for _, run in smt_runs.items():
        smt_pair_index.setdefault((run.patient_id, run.mode), []).append(run)

    keys_tg = set(tg_runs.keys())
    keys_smt = set(smt_pair_index.keys())
    keys_all = sorted(keys_tg & keys_smt) if require_overlap else sorted(keys_tg | keys_smt)

    trialgpt_fn_rows: List[Dict[str, Any]] = []
    smt_fn_rows: List[Dict[str, Any]] = []

    trialgpt_fn_by_mode: Dict[str, int] = {}
    smt_fn_by_mode: Dict[str, int] = {}

    trialgpt_fn_by_mode_conflict: Dict[Tuple[str, str], int] = {}
    smt_fn_by_mode_conflict: Dict[Tuple[str, str], int] = {}

    pairs_eval = 0

    for (patient_id, mode) in keys_all:
        tg = tg_runs.get((patient_id, mode))
        smt_runs_for_pair = smt_pair_index.get((patient_id, mode), [])

        if not smt_runs_for_pair and tg is None:
            continue
        if require_overlap and (tg is None or not smt_runs_for_pair):
            continue

        pairs_eval += 1

        # TrialGPT positives at top-K: relevant AND eligible
        tg_top = tg.items[:k] if tg is not None else []
        tg_top_re_elig_items = [x for x in tg_top if x.relevant is True and x.eligible is True]
        tg_top_re_elig_keys: Set[str] = {x.key for x in tg_top_re_elig_items}

        # SMT positives: confirmed relevant AND eligible
        smt_best_re_elig: Dict[str, Tuple[SMTItem, SMTRun]] = {}
        for smt_run in smt_runs_for_pair:
            for x in smt_run.items:
                if x.any_rel_and_elig is not True:
                    continue
                prev = smt_best_re_elig.get(x.parent_trial_id)
                if prev is None or x.rank < prev[0].rank:
                    smt_best_re_elig[x.parent_trial_id] = (x, smt_run)

        smt_re_elig_keys: Set[str] = set(smt_best_re_elig.keys())

        # --------------------------------------------------------------
        # TrialGPT false negatives:
        # SMT re+elig positive, but TrialGPT top-K not re+elig positive
        # --------------------------------------------------------------
        for trial_key, (smt_item, smt_run) in sorted(
            smt_best_re_elig.items(),
            key=lambda kv: (kv[1][0].rank, kv[0]),
        ):
            if trial_key in tg_top_re_elig_keys:
                continue

            conflict = choose_best_trialgpt_conflict_for_trial(
                trial_key=trial_key,
                tg_run=tg,
                tg_k=k,
            )

            trialgpt_fn_rows.append({
                "patient_id": patient_id,
                "mode": mode,
                "trial_id": trial_key,
                "smt_raw_trial_id": smt_item.raw_trial_id,
                "smt_rank": smt_item.rank,
                "smt_label": smt_item.label or "",
                "smt_any_relevant": smt_item.any_rel,
                "smt_any_eligible": smt_item.any_elig,
                "smt_any_relevant_and_eligible": smt_item.any_rel_and_elig,
                "smt_num_subcohorts": len(smt_item.subcohorts),
                "smt_subcohorts": "|".join(smt_item.subcohorts),
                "trialgpt_conflict_category": conflict.conflict_label,
                "trialgpt_rank": conflict.tg_rank if conflict.tg_rank is not None else "",
                "trialgpt_relevant": conflict.tg_relevant,
                "trialgpt_eligible": conflict.tg_eligible,
                "trialgpt_label": conflict.tg_label or "",
                "trialgpt_fixed_k_used": k,
                "reason": "smt_confirmed_relevant_and_eligible_but_trialgpt_topk_missed",
                "smt_file": str(smt_run.source_file),
                "trialgpt_file": str(conflict.source_file) if conflict.source_file else "",
            })

            trialgpt_fn_by_mode[mode] = trialgpt_fn_by_mode.get(mode, 0) + 1
            trialgpt_fn_by_mode_conflict[(mode, conflict.conflict_label)] = (
                trialgpt_fn_by_mode_conflict.get((mode, conflict.conflict_label), 0) + 1
            )

        # --------------------------------------------------------------
        # SMT false negatives:
        # TrialGPT top-K re+elig positive, but SMT not re+elig positive
        # --------------------------------------------------------------
        for tg_item in sorted(tg_top_re_elig_items, key=lambda x: (x.rank, x.key)):
            if tg_item.key in smt_re_elig_keys:
                continue

            smt_conflict = choose_best_smt_conflict_for_trial_fixed(
                trial_key=tg_item.key,
                smt_runs_for_pair=smt_runs_for_pair,
            )

            smt_fn_rows.append({
                "patient_id": patient_id,
                "mode": mode,
                "trial_id": tg_item.key,
                "trialgpt_raw_trial_id": tg_item.raw_trial_id,
                "trialgpt_rank": tg_item.rank,
                "trialgpt_relevant": tg_item.relevant,
                "trialgpt_eligible": tg_item.eligible,
                "trialgpt_label": tg_item.label or "",
                "trialgpt_fixed_k_used": k,
                "smt_conflict_category": smt_conflict.conflict_label,
                "smt_rank": smt_conflict.smt_rank if smt_conflict.smt_rank is not None else "",
                "smt_label": smt_conflict.smt_label or "",
                "smt_any_relevant": smt_conflict.smt_any_relevant,
                "smt_any_eligible": smt_conflict.smt_any_eligible,
                "smt_any_relevant_and_eligible": smt_conflict.smt_any_relevant_and_eligible,
                "smt_num_subcohorts": smt_conflict.smt_num_subcohorts,
                "smt_subcohorts": "|".join(smt_conflict.smt_subcohorts),
                "reason": "trialgpt_topk_relevant_and_eligible_but_smt_missed",
                "trialgpt_file": str(tg.source_file) if tg is not None else "",
                "smt_file": str(smt_conflict.source_file) if smt_conflict.source_file else "",
            })

            smt_fn_by_mode[mode] = smt_fn_by_mode.get(mode, 0) + 1
            smt_fn_by_mode_conflict[(mode, smt_conflict.conflict_label)] = (
                smt_fn_by_mode_conflict.get((mode, smt_conflict.conflict_label), 0) + 1
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Detailed outputs
    write_csv(
        out_dir / "trialgpt_false_negatives.csv",
        [
            "patient_id", "mode",
            "trial_id", "smt_raw_trial_id",
            "smt_rank", "smt_label",
            "smt_any_relevant", "smt_any_eligible", "smt_any_relevant_and_eligible",
            "smt_num_subcohorts", "smt_subcohorts",
            "trialgpt_conflict_category",
            "trialgpt_rank", "trialgpt_relevant", "trialgpt_eligible", "trialgpt_label",
            "trialgpt_fixed_k_used",
            "reason", "smt_file", "trialgpt_file",
        ],
        trialgpt_fn_rows,
    )

    write_csv(
        out_dir / "smt_false_negatives.csv",
        [
            "patient_id", "mode",
            "trial_id", "trialgpt_raw_trial_id",
            "trialgpt_rank", "trialgpt_relevant", "trialgpt_eligible", "trialgpt_label",
            "trialgpt_fixed_k_used",
            "smt_conflict_category",
            "smt_rank", "smt_label",
            "smt_any_relevant", "smt_any_eligible", "smt_any_relevant_and_eligible",
            "smt_num_subcohorts", "smt_subcohorts",
            "reason", "trialgpt_file", "smt_file",
        ],
        smt_fn_rows,
    )

    # Per-side conflict summaries
    write_csv(
        out_dir / "trialgpt_false_negatives_summary_by_conflict.csv",
        ["mode", "trialgpt_conflict_category", "count"],
        [
            {"mode": m, "trialgpt_conflict_category": c, "count": n}
            for ((m, c), n) in sorted(trialgpt_fn_by_mode_conflict.items())
        ],
    )

    write_csv(
        out_dir / "smt_false_negatives_summary_by_conflict.csv",
        ["mode", "smt_conflict_category", "count"],
        [
            {"mode": m, "smt_conflict_category": c, "count": n}
            for ((m, c), n) in sorted(smt_fn_by_mode_conflict.items())
        ],
    )

    # Combined summary table: both sides together
    modes_all = sorted(set(trialgpt_fn_by_mode.keys()) | set(smt_fn_by_mode.keys()))
    combined_rows = []
    for mode in modes_all:
        combined_rows.append({
            "mode": mode,
            "trialgpt_false_negatives": trialgpt_fn_by_mode.get(mode, 0),
            "smt_false_negatives": smt_fn_by_mode.get(mode, 0),
        })

    write_csv(
        out_dir / "summary_combined.csv",
        ["mode", "trialgpt_false_negatives", "smt_false_negatives"],
        combined_rows,
    )

    print("\n" + "=" * 100)
    print(f"[DONE] K={k}")
    print(f"[INFO] trialgpt_root={trialgpt_root}")
    print(f"[INFO] smt_root={smt_root}")
    print(f"[INFO] pairs_evaluated={pairs_eval}")
    print(f"[INFO] TrialGPT false negatives={len(trialgpt_fn_rows)}")
    print(f"[INFO] SMT false negatives={len(smt_fn_rows)}")
    print(f"[INFO] wrote {out_dir}")
    print("=" * 100)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Find both sides' false negatives at each fixed K, using "
            "confirmed relevant-and-eligible on both sides."
        )
    )

    ap.add_argument(
        "--trialgpt-root-base",
        type=Path,
        default=DEFAULT_TRIALGPT_ROOT_BASE,
        help=(
            "Base TrialGPT root. For each K we read <base>_top<K>, "
            "e.g. ./trialgpt_retrieval_eval_out_top300"
        ),
    )
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--out", type=Path, default=Path("./topk_false_negative_analysis_out"))

    ap.add_argument("--topk-list", type=str, default=DEFAULT_TOPK_LIST)

    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")
    ap.add_argument("--modes", type=str, default="")

    args = ap.parse_args()

    if not args.smt_root.exists():
        print(f"[error] smt root not found: {args.smt_root}", file=sys.stderr)
        sys.exit(2)

    topk_list = parse_topk_list(args.topk_list)
    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)

    print(f"[INFO] trialgpt_root_base={args.trialgpt_root_base}")
    print(f"[INFO] smt_root={args.smt_root}")
    print(f"[INFO] out={args.out}")
    print(f"[INFO] topk_list={topk_list}")
    print(f"[INFO] modes={mode_filter if mode_filter is not None else 'ALL'}")
    print(f"[INFO] dedup_canonical={args.dedup_canonical}")
    print(f"[INFO] collapse_subsuffix={collapse}")
    print(f"[INFO] require_overlap={args.require_overlap}")

    for k in topk_list:
        trialgpt_root = Path(f"{args.trialgpt_root_base}_top{k}")
        per_k_out = args.out / f"top{k}"

        analyze_one_k(
            k=k,
            trialgpt_root=trialgpt_root,
            smt_root=args.smt_root,
            out_dir=per_k_out,
            collapse_subsuffix=collapse,
            dedup_canonical=args.dedup_canonical,
            require_overlap=args.require_overlap,
            mode_filter=mode_filter,
        )

    print("\n" + "=" * 100)
    print("[DONE] Finished all K values.")
    print(f"[DONE] Ks: {topk_list}")
    print(f"[DONE] Output root: {args.out}")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)