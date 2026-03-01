#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_match_from_eval_union.py

Rerun batch matcher on the union of:
  1) SMT eval outputs with label == "all_satisfied"
  2) TrialGPT eval outputs with interval_mno == "m"

This version uses a SINGLE flat process pool over (patient_id, canonical_nct_id)
jobs. It does NOT use nested patient/subcohort parallelism.

Sampling source
---------------
This script reads already-produced eval outputs:

  --smt-eval-root
      default: ../irsrc/eval/smt_retrieval_eval_out

  --trialgpt-eval-root
      default: ../irsrc/eval/trialgpt_retrieval_eval_out

Expected structure:
  <root>/<mode>/patient_labels/*.json

Modes:
  - ccr
  - all
  - all-explore

Mode -> compose/list_to_match run mapping
-----------------------------------------
  ccr         -> (ccr, prevent,   act)
  all         -> (all, prevent,   act)
  all-explore -> (all, noprevent, nonact)

Then for each mapped run, the script reads upstream compose outputs from:

  <list_base>/list_to_match__{mode}__{prevent_tag}__{alt_tag}/
      {patient_id}__{mode}__{prevent_tag}__{alt_tag}.json

and reruns batch matching only on the selected canonical patient-trial pairs.

Selection rule
--------------
For each mode and patient:
  selected canonical trial ids =
      union(
        SMT trials where label == "all_satisfied",
        TrialGPT trials where interval_mno == "m"
      )

Sampling dimensions
-------------------
1) Patient sampling:
   --sample-patients N
   --patient-sample-policy {hash,random,first}
   --patient-sample-seed S

2) Pair sampling among the chosen patients:
   --sample-per-patient K
   --sample-pairs N
   --sample-policy {hash,random,first}
   --sample-seed S

Writes
------
Per-pair outputs:
  <out_root>/<run_key>/<patient_id>/<canonical_nct_id>__parent_or.json
  <out_root>/<run_key>/<patient_id>/<canonical_nct_id>__overall_parent.json

Per-patient aggregate outputs:
  <out_root>/list_to_match_eval__<run_key>/<patient_id>__<mode>__<prevent>__<alt>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm

from match_patient_to_trial import (
    AzureInferenceEngine,
    Config,
    _json_default,
    parent_nct_id,
    run_single_pair,
)


# =============================================================================
# Constants / mappings
# =============================================================================

MODE_TO_RUN: Dict[str, Tuple[str, str, str]] = {
    "ccr": ("ccr", "prevent", "act"),
    "all": ("all", "prevent", "act"),
    "all-explore": ("all", "noprevent", "nonact"),
}

DEFAULT_EVAL_MODES: List[str] = ["ccr", "all", "all-explore"]

SMT_POSITIVE_LABEL = "all_satisfied"
TRIALGPT_POSITIVE_INTERVAL = "m"


# =============================================================================
# Run / path helpers
# =============================================================================

def _run_suffix(mode: str, prevent_tag: str, alt_tag: str) -> str:
    return f"__{mode}__{prevent_tag}__{alt_tag}"


def _run_key(mode: str, prevent_tag: str, alt_tag: str) -> str:
    return f"{mode}__{prevent_tag}__{alt_tag}"


def _bool_from_prevent_tag(prevent_tag: str) -> bool:
    return prevent_tag == "prevent"


def _list_root_for_run(list_base: Path, mode: str, prevent_tag: str, alt_tag: str) -> Path:
    return list_base / f"list_to_match{_run_suffix(mode, prevent_tag, alt_tag)}"


def _list_file_for_run(list_base: Path, patient_id: str, mode: str, prevent_tag: str, alt_tag: str) -> Path:
    suf = _run_suffix(mode, prevent_tag, alt_tag)
    return _list_root_for_run(list_base, mode, prevent_tag, alt_tag) / f"{patient_id}{suf}.json"


def _patient_labels_dir(root: Path, mode: str) -> Path:
    return root / mode / "patient_labels"


# =============================================================================
# Small helpers
# =============================================================================

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _or3(vals: List[Optional[bool]]) -> Optional[bool]:
    vv: List[Optional[bool]] = [(v if isinstance(v, bool) else None) for v in vals]
    if any(v is True for v in vv):
        return True
    if len(vv) > 0 and all(v is False for v in vv):
        return False
    return None


def _stable_pair_key(patient_id: str, canonical_nct_id: str, eval_mode: str) -> str:
    return f"{eval_mode}|{patient_id}|{canonical_nct_id}"


def _stable_patient_key(patient_id: str, eval_mode: str) -> str:
    return f"{eval_mode}|{patient_id}"


def _hash_rank(s: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{s}".encode("utf-8")).hexdigest()


def _sampling_meta(
    args: SimpleNamespace,
    *,
    selected_count: Optional[int],
    candidate_count: Optional[int],
    selected_patients: Optional[int],
    candidate_patients: Optional[int],
) -> Dict[str, Any]:
    return {
        "sample_patients": getattr(args, "sample_patients", None),
        "patient_sample_policy": getattr(args, "patient_sample_policy", "hash"),
        "patient_sample_seed": int(getattr(args, "patient_sample_seed", 0) or 0),
        "sample_pairs": getattr(args, "sample_pairs", None),
        "sample_per_patient": getattr(args, "sample_per_patient", None),
        "sample_policy": getattr(args, "sample_policy", "hash"),
        "sample_seed": int(getattr(args, "sample_seed", 0) or 0),
        "selected_patient_count_for_mode": selected_patients,
        "candidate_patient_count_for_mode": candidate_patients,
        "selected_pair_count_for_mode": selected_count,
        "candidate_pair_count_for_mode": candidate_count,
    }


def _normalize_args_dict_for_run_single_pair(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    args_dict.setdefault("patient_file", None)
    args_dict.setdefault("canonical_only", False)
    args_dict.setdefault("canonical_json", None)
    args_dict.setdefault("labels", None)
    args_dict.setdefault("log_json", None)
    args_dict.setdefault("trial_engine_per_task", False)

    args_dict.setdefault("llm_judge", True)
    args_dict.setdefault("llm_judge_prompt", None)
    args_dict.setdefault("llm_judge_temperature", 0.0)
    args_dict.setdefault("llm_judge_force", False)

    args_dict.setdefault("trialgpt_judge", True)
    args_dict.setdefault("trialgpt_judge_temperature", None)
    args_dict.setdefault("trialgpt_judge_force", False)

    args_dict.setdefault("vv_batch_size", 10)
    args_dict.setdefault("verbose", False)

    args_dict.setdefault("important_mode", None)
    args_dict.setdefault("enable_prevention_hits", None)
    args_dict.setdefault("alt_mode", None)

    return args_dict


def _safe_bool(v: Any) -> Optional[bool]:
    return v if isinstance(v, bool) else None


def _bool_agreement(a: Any, b: Any) -> Optional[bool]:
    aa = _safe_bool(a)
    bb = _safe_bool(b)
    if aa is None or bb is None:
        return None
    return aa == bb


def _threeway_agreement(a: Any, b: Any, c: Any) -> Optional[bool]:
    aa = _safe_bool(a)
    bb = _safe_bool(b)
    cc = _safe_bool(c)
    if aa is None or bb is None or cc is None:
        return None
    return aa == bb == cc


def _extract_llm_parent_fields(llm_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(llm_payload, dict):
        return {
            "llm_eligible": None,
            "llm_eligibility": None,
            "llm_explanation": None,
            "llm_parse_error": None,
            "llm_error": None,
        }

    res = llm_payload.get("result")
    if isinstance(res, dict):
        return {
            "llm_eligible": res.get("eligible"),
            "llm_eligibility": res.get("eligibility"),
            "llm_explanation": res.get("explanation"),
            "llm_parse_error": res.get("parse_error"),
            "llm_error": llm_payload.get("error"),
        }

    return {
        "llm_eligible": None,
        "llm_eligibility": None,
        "llm_explanation": None,
        "llm_parse_error": None,
        "llm_error": llm_payload.get("error"),
    }


def _extract_trialgpt_parent_fields(tg_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(tg_payload, dict):
        return {
            "trialgpt_eligible": None,
            "trialgpt_eligible_strict": None,
            "trialgpt_inclusion_sat_like": None,
            "trialgpt_exclusion_sat_like": None,
            "trialgpt_error": None,
        }

    agg = tg_payload.get("aggregate")
    if isinstance(agg, dict):
        return {
            "trialgpt_eligible": agg.get("eligible"),
            "trialgpt_eligible_strict": agg.get("eligible_strict"),
            "trialgpt_inclusion_sat_like": agg.get("inclusion_sat_like"),
            "trialgpt_exclusion_sat_like": agg.get("exclusion_sat_like"),
            "trialgpt_error": tg_payload.get("error"),
        }

    return {
        "trialgpt_eligible": None,
        "trialgpt_eligible_strict": None,
        "trialgpt_inclusion_sat_like": None,
        "trialgpt_exclusion_sat_like": None,
        "trialgpt_error": tg_payload.get("error"),
    }


def _load_parent_cached_judges(
    pair_dir: Path,
    canonical_nct_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    parent_id = parent_nct_id(canonical_nct_id)

    llm_path = pair_dir / f"{parent_id}__llm_judge.json"
    tg_path = pair_dir / f"{parent_id}__trialgpt_judge.json"

    llm_payload = None
    tg_payload = None

    if llm_path.exists():
        try:
            llm_payload = _load_json(llm_path)
        except Exception:
            llm_payload = {"error": f"failed_to_read: {llm_path}"}

    if tg_path.exists():
        try:
            tg_payload = _load_json(tg_path)
        except Exception:
            tg_payload = {"error": f"failed_to_read: {tg_path}"}

    return llm_payload, tg_payload


# =============================================================================
# Eval-output readers
# =============================================================================

def _list_json_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.json") if p.is_file()])


def _extract_smt_positive_trials(label_obj: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for t in label_obj.get("trials", []) or []:
        if not isinstance(t, dict):
            continue
        tid = t.get("trial_id")
        lab = t.get("label")
        if isinstance(tid, str) and tid and lab == SMT_POSITIVE_LABEL:
            out.add(tid)
    return out


def _extract_trialgpt_positive_trials(label_obj: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for t in label_obj.get("trials", []) or []:
        if not isinstance(t, dict):
            continue
        tid = t.get("trial_id")
        interval = t.get("interval_mno")
        if isinstance(tid, str) and tid and interval == TRIALGPT_POSITIVE_INTERVAL:
            out.add(tid)
    return out


def _load_eval_mode_patient_map(eval_root: Path, mode: str, which: str) -> Dict[str, Set[str]]:
    d = _patient_labels_dir(eval_root, mode)
    files = _list_json_files(d)
    out: Dict[str, Set[str]] = {}

    for p in files:
        obj = _load_json(p)
        pid = obj.get("patient_id")
        if not isinstance(pid, str) or not pid:
            continue

        if which == "smt":
            pos = _extract_smt_positive_trials(obj)
        elif which == "trialgpt":
            pos = _extract_trialgpt_positive_trials(obj)
        else:
            raise ValueError(which)

        out[pid] = pos

    return out


def _build_union_selection_for_mode(
    *,
    eval_mode: str,
    smt_eval_root: Path,
    trialgpt_eval_root: Path,
    requested_patients: Optional[Set[str]],
) -> Dict[str, Set[str]]:
    smt_map = _load_eval_mode_patient_map(smt_eval_root, eval_mode, "smt")
    tg_map = _load_eval_mode_patient_map(trialgpt_eval_root, eval_mode, "trialgpt")

    already_ran_patients = set(smt_map.keys()) | set(tg_map.keys())
    if requested_patients is not None:
        already_ran_patients &= requested_patients

    out: Dict[str, Set[str]] = {}
    for pid in sorted(already_ran_patients):
        out[pid] = set(smt_map.get(pid, set())) | set(tg_map.get(pid, set()))
    return out


# =============================================================================
# Sampling helpers
# =============================================================================

def _sample_patients(
    selected_union: Dict[str, Set[str]],
    *,
    eval_mode: str,
    sample_patients: Optional[int],
    patient_sample_policy: str,
    patient_sample_seed: int,
) -> Tuple[Dict[str, Set[str]], int, int]:
    candidate_patients = sorted(selected_union.keys())
    candidate_count = len(candidate_patients)

    if sample_patients is None or sample_patients <= 0 or candidate_count <= sample_patients:
        return dict(selected_union), candidate_count, candidate_count

    if patient_sample_policy == "first":
        chosen_patients = candidate_patients[:sample_patients]
    elif patient_sample_policy == "random":
        rng = random.Random(patient_sample_seed)
        chosen_patients = rng.sample(candidate_patients, sample_patients)
    else:  # hash
        chosen_patients = sorted(
            candidate_patients,
            key=lambda pid: _hash_rank(_stable_patient_key(pid, eval_mode), patient_sample_seed),
        )[:sample_patients]

    chosen = {pid: selected_union[pid] for pid in chosen_patients}
    return chosen, len(chosen_patients), candidate_count


def _apply_sample_per_patient_union(
    selected_by_patient: Dict[str, Set[str]],
    *,
    eval_mode: str,
    sample_per_patient: Optional[int],
    sample_policy: str,
    sample_seed: int,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    for pid, cids_set in selected_by_patient.items():
        cids = sorted([x for x in cids_set if isinstance(x, str) and x])
        if not cids:
            out[pid] = []
            continue

        if sample_per_patient is None or sample_per_patient <= 0 or len(cids) <= sample_per_patient:
            out[pid] = cids
            continue

        if sample_policy == "first":
            out[pid] = cids[:sample_per_patient]
        elif sample_policy == "random":
            rng = random.Random(sample_seed)
            out[pid] = rng.sample(cids, sample_per_patient)
        else:  # hash
            ranked = sorted(
                cids,
                key=lambda cid: _hash_rank(_stable_pair_key(pid, cid, eval_mode), sample_seed),
            )
            out[pid] = ranked[:sample_per_patient]

    return out


def _sample_global_pairs(
    selected_by_patient: Dict[str, List[str]],
    *,
    eval_mode: str,
    sample_pairs: Optional[int],
    sample_policy: str,
    sample_seed: int,
) -> Tuple[Set[Tuple[str, str]], int, int]:
    all_pairs: List[Tuple[str, str]] = []
    for pid, cids in selected_by_patient.items():
        for cid in cids:
            all_pairs.append((pid, cid))

    candidate_count = len(all_pairs)

    if sample_pairs is None or sample_pairs <= 0 or candidate_count <= sample_pairs:
        return set(all_pairs), candidate_count, candidate_count

    if sample_policy == "first":
        chosen = sorted(all_pairs)[:sample_pairs]
    elif sample_policy == "random":
        rng = random.Random(sample_seed)
        chosen = rng.sample(all_pairs, sample_pairs)
    else:  # hash
        chosen = sorted(
            all_pairs,
            key=lambda x: _hash_rank(_stable_pair_key(x[0], x[1], eval_mode), sample_seed),
        )[:sample_pairs]

    chosen_set = set(chosen)
    return chosen_set, len(chosen_set), candidate_count


# =============================================================================
# Parent aggregation
# =============================================================================

def _agg_parent_or(
    patient_id: str,
    parent_id: str,
    sub_rows: List[Dict[str, Any]],
    *,
    timestamp: str,
    mode: str,
    prevent_tag: str,
    alt_tag: str,
) -> Dict[str, Any]:
    eligible_parent = any(r.get("eligible") is True for r in sub_rows)
    eligible_strict_parent = _or3([r.get("eligible_strict") for r in sub_rows])

    inc_parent = _or3([r.get("inclusion_sat_like") for r in sub_rows])
    exc_parent = _or3([r.get("exclusion_sat_like") for r in sub_rows])

    def _rank_key(r: Dict[str, Any]):
        rk = r.get("rank")
        return (rk is None, rk if isinstance(rk, int) else 10**9, r.get("trial_id") or "")

    eligibles = [r for r in sub_rows if r.get("eligible") is True]
    witness = (
        sorted(eligibles, key=_rank_key)[0]
        if eligibles
        else (sorted(sub_rows, key=_rank_key)[0] if sub_rows else None)
    )

    inc_unsat: List[Any] = []
    exc_unsat: List[Any] = []
    for r in sub_rows:
        inc_unsat.extend(r.get("inclusion_unsat_assertions", []) or [])
        exc_unsat.extend(r.get("exclusion_unsat_assertions", []) or [])

    def _dedup(seq: List[Any]) -> List[Any]:
        seen = set()
        out: List[Any] = []
        for x in seq:
            k = json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else str(x)
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return {
        "patient_id": patient_id,
        "mode": mode,
        "prevent_tag": prevent_tag,
        "alt_mode": alt_tag,
        "parent_trial_id": parent_id,
        "timestamp": timestamp,
        "policy": {
            "parent_is_or_over_subcohorts": True,
            "eligible_parent": "OR over subcohort eligible (optimistic)",
            "eligible_strict_parent": "tri-valued OR over eligible_strict",
            "inclusion_sat_like_parent": "tri-valued OR over inclusion_sat_like",
            "exclusion_sat_like_parent": "tri-valued OR over exclusion_sat_like",
        },
        "eligible_parent": eligible_parent,
        "eligible_strict_parent": eligible_strict_parent,
        "inclusion_sat_like_parent": inc_parent,
        "exclusion_sat_like_parent": exc_parent,
        "witness_subcohort": (witness.get("trial_id") if witness else None),
        "inclusion_unsat_assertions_union": _dedup(inc_unsat),
        "exclusion_unsat_assertions_union": _dedup(exc_unsat),
        "subcohorts": sub_rows,
    }


def _build_parent_overall(
    *,
    patient_id: str,
    canonical_nct_id: str,
    timestamp: str,
    eval_mode: str,
    mode: str,
    prevent_tag: str,
    alt_tag: str,
    parent_or: Dict[str, Any],
    llm_payload: Optional[Dict[str, Any]],
    tg_payload: Optional[Dict[str, Any]],
    sampling: Dict[str, Any],
) -> Dict[str, Any]:
    llm_fields = _extract_llm_parent_fields(llm_payload)
    tg_fields = _extract_trialgpt_parent_fields(tg_payload)

    smt_eligible = parent_or.get("eligible_parent")
    smt_eligible_strict = parent_or.get("eligible_strict_parent")

    llm_eligible = llm_fields.get("llm_eligible")
    tg_eligible = tg_fields.get("trialgpt_eligible")
    tg_eligible_strict = tg_fields.get("trialgpt_eligible_strict")

    return {
        "patient_id": patient_id,
        "trial_id_parent": canonical_nct_id,
        "timestamp": timestamp,
        "eval_mode": eval_mode,
        "mode": mode,
        "prevent_tag": prevent_tag,
        "alt_mode": alt_tag,
        "selection_policy": {
            "eval_mode": eval_mode,
            "union_of": [
                f"SMT label == {SMT_POSITIVE_LABEL}",
                f"TrialGPT interval_mno == {TRIALGPT_POSITIVE_INTERVAL}",
            ],
        },
        "sampling": sampling,
        "smt_parent": {
            "eligible": smt_eligible,
            "eligible_strict": smt_eligible_strict,
            "inclusion_sat_like": parent_or.get("inclusion_sat_like_parent"),
            "exclusion_sat_like": parent_or.get("exclusion_sat_like_parent"),
            "witness_subcohort": parent_or.get("witness_subcohort"),
            "subcohort_count": len(parent_or.get("subcohorts", []) or []),
        },
        "llm_parent": {
            "eligible": llm_eligible,
            "eligibility": llm_fields.get("llm_eligibility"),
            "explanation": llm_fields.get("llm_explanation"),
            "parse_error": llm_fields.get("llm_parse_error"),
            "error": llm_fields.get("llm_error"),
            "cached_trial_id_parent": (
                llm_payload.get("trial_id_parent") if isinstance(llm_payload, dict) else None
            ),
        },
        "trialgpt_parent": {
            "eligible": tg_eligible,
            "eligible_strict": tg_eligible_strict,
            "inclusion_sat_like": tg_fields.get("trialgpt_inclusion_sat_like"),
            "exclusion_sat_like": tg_fields.get("trialgpt_exclusion_sat_like"),
            "error": tg_fields.get("trialgpt_error"),
            "cached_trial_id_parent": (
                tg_payload.get("trial_id_parent") if isinstance(tg_payload, dict) else None
            ),
        },
        "agreement": {
            "smt_vs_llm": _bool_agreement(smt_eligible, llm_eligible),
            "smt_vs_trialgpt": _bool_agreement(smt_eligible, tg_eligible),
            "llm_vs_trialgpt": _bool_agreement(llm_eligible, tg_eligible),
            "three_way": _threeway_agreement(smt_eligible, llm_eligible, tg_eligible),
            "smt_strict_vs_trialgpt_strict": _bool_agreement(smt_eligible_strict, tg_eligible_strict),
        },
        "paths": {
            "parent_or": f"{canonical_nct_id}__parent_or.json",
            "overall_parent": f"{canonical_nct_id}__overall_parent.json",
            "llm_judge_parent": f"{canonical_nct_id}__llm_judge.json",
            "trialgpt_judge_parent": f"{canonical_nct_id}__trialgpt_judge.json",
        },
    }


# =============================================================================
# Per-subcohort worker logic (called serially inside one pair worker)
# =============================================================================

def _run_one_subcohort_proc_worker(
    *,
    patient_id: str,
    canonical_nct_id: str,
    sc: Dict[str, Any],
    cfg_kwargs: Dict[str, str],
    args_dict: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    args = SimpleNamespace(**args_dict)

    nct_id = sc.get("nct_id") or canonical_nct_id
    if not nct_id:
        return None

    sub_status = sc.get("status")
    sub_label = sc.get("label")
    sub_rank = sc.get("rank")

    cfg = Config(
        data_root=Path(cfg_kwargs["data_root"]),
        build_root=Path(cfg_kwargs["build_root"]),
        project_root=Path(cfg_kwargs["project_root"]),
        prompt_root=Path(cfg_kwargs["prompt_root"]),
        prompt_map_json=Path(cfg_kwargs["prompt_map_json"]) if cfg_kwargs.get("prompt_map_json") else None,
        conservative=bool(getattr(args, "conservative", False)),
        miner_mode=str(getattr(args, "miner_mode", "infer") or "infer"),
    )

    engine = AzureInferenceEngine(
        endpoint=cfg.azure_endpoint,
        api_key_env_var="OPENAI_API_KEY",
        model_name=cfg.model_name,
    )

    try:
        out = run_single_pair(
            trial_id=nct_id,
            patient_id=patient_id,
            cfg=cfg,
            engine=engine,
            args=args,
        )
    except FileNotFoundError as e:
        return {
            "trial_id": nct_id,
            "nct_id": nct_id,
            "parent_trial_id": parent_nct_id(nct_id),
            "status": sub_status,
            "label": sub_label,
            "rank": sub_rank,
            "error": str(e),
        }
    except Exception as e:
        return {
            "trial_id": nct_id,
            "nct_id": nct_id,
            "parent_trial_id": parent_nct_id(nct_id),
            "status": sub_status,
            "label": sub_label,
            "rank": sub_rank,
            "error": f"exception: {e}",
        }

    inc_summary = (out.get("inclusion", {}).get("summary") or {})
    exc_summary = (out.get("exclusion", {}).get("summary") or {})

    inc_sat_like = out.get("inclusion", {}).get("sat_like")
    exc_sat_like = out.get("exclusion", {}).get("sat_like")

    llm_eligible = None
    llm_explanation = None
    llm_parse_error = None
    lj = out.get("llm_judge")
    if isinstance(lj, dict):
        res = lj.get("result")
        if isinstance(res, dict):
            llm_eligible = res.get("eligible")
            llm_explanation = res.get("explanation")
            llm_parse_error = res.get("parse_error")

    tg_eligible = None
    tg_eligible_strict = None
    tg_inc = None
    tg_exc = None
    tg_err = None
    tgj = out.get("trialgpt_judge")
    if isinstance(tgj, dict):
        if "error" in tgj:
            tg_err = tgj.get("error")
        agg = tgj.get("aggregate")
        if isinstance(agg, dict):
            tg_eligible = agg.get("eligible")
            tg_eligible_strict = agg.get("eligible_strict")
            tg_inc = agg.get("inclusion_sat_like")
            tg_exc = agg.get("exclusion_sat_like")

    pid_parent = parent_nct_id(nct_id)

    return {
        "trial_id": nct_id,
        "nct_id": nct_id,
        "parent_trial_id": pid_parent,
        "status": sub_status,
        "label": sub_label,
        "rank": sub_rank,
        "eligible": out.get("eligible"),
        "eligible_strict": out.get("eligible_strict"),
        "inclusion_sat_like": inc_sat_like,
        "exclusion_sat_like": exc_sat_like,
        "inclusion_status": inc_summary.get("status"),
        "exclusion_status": exc_summary.get("status"),
        "inclusion_unsat_assertions": inc_summary.get("unsat_assertions", []),
        "exclusion_unsat_assertions": exc_summary.get("unsat_assertions", []),
        "llm_eligible": llm_eligible,
        "llm_explanation": llm_explanation,
        "llm_parse_error": llm_parse_error,
        "trialgpt_eligible": tg_eligible,
        "trialgpt_eligible_strict": tg_eligible_strict,
        "trialgpt_inclusion_sat_like": tg_inc,
        "trialgpt_exclusion_sat_like": tg_exc,
        "trialgpt_error": tg_err,
        "paths": {
            "overall": f"{nct_id}__overall.json",
            "full": f"{nct_id}__full.json",
            "inclusion_stats": f"{nct_id}__inclusion_stats.json",
            "exclusion_stats": f"{nct_id}__exclusion_stats.json",
            "llm_judge_parent": f"{pid_parent}__llm_judge.json",
            "trialgpt_judge_parent": f"{pid_parent}__trialgpt_judge.json",
        },
    }


# =============================================================================
# Flat pair worker
# =============================================================================

def _process_pair_run_core(
    *,
    patient_id: str,
    canonical_nct_id: str,
    list_base: Path,
    compose_mode: str,
    prevent_tag: str,
    alt_tag: str,
    eval_mode: str,
    args: SimpleNamespace,
) -> Dict[str, Any]:
    meta_path = _list_file_for_run(list_base, patient_id, compose_mode, prevent_tag, alt_tag)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"list_to_match file missing for patient={patient_id} run={compose_mode}/{prevent_tag}/{alt_tag}: {meta_path}"
        )

    meta = _load_json(meta_path)
    canonical_trials = list(meta.get("canonical_trials", []) or [])

    target_ct = None
    for ct in canonical_trials:
        if ct.get("canonical_nct_id") == canonical_nct_id:
            target_ct = ct
            break

    if target_ct is None:
        raise FileNotFoundError(
            f"canonical trial missing in list_to_match file for patient={patient_id} "
            f"canonical_nct_id={canonical_nct_id}: {meta_path}"
        )

    canon_status = target_ct.get("status")
    canon_label = target_ct.get("label")
    canon_sub_ids = target_ct.get("sub_nct_ids", []) or []
    subcohorts_in = target_ct.get("subcohorts", []) or []

    cfg_kwargs: Dict[str, str] = {
        "data_root": str(Path(args.data_root).resolve()),
        "build_root": str(Path(args.build_root).resolve()),
        "project_root": str(Path(args.project_root).resolve()),
        "prompt_root": str(Path(args.prompt_root)),
        "prompt_map_json": str(Path(args.prompt_map)) if getattr(args, "prompt_map", None) else "",
    }

    args_dict = dict(vars(args))
    for k, v in list(args_dict.items()):
        if isinstance(v, Path):
            args_dict[k] = str(v)
    args_dict = _normalize_args_dict_for_run_single_pair(args_dict)
    args_dict["important_mode"] = compose_mode
    args_dict["enable_prevention_hits"] = _bool_from_prevent_tag(prevent_tag)
    args_dict["alt_mode"] = alt_tag

    out_root = Path(args.out_root).resolve()
    run_key = _run_key(compose_mode, prevent_tag, alt_tag)
    pair_dir = out_root / run_key / patient_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    sub_results: List[Dict[str, Any]] = []
    for sc in subcohorts_in:
        r = _run_one_subcohort_proc_worker(
            patient_id=patient_id,
            canonical_nct_id=canonical_nct_id,
            sc=sc,
            cfg_kwargs=cfg_kwargs,
            args_dict=args_dict,
        )
        if r:
            sub_results.append(r)

    sub_results.sort(key=lambda x: (x.get("rank") is None, x.get("rank", 10**9), x.get("trial_id") or ""))

    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    parent_or = _agg_parent_or(
        patient_id=patient_id,
        parent_id=canonical_nct_id,
        sub_rows=sub_results,
        timestamp=timestamp,
        mode=compose_mode,
        prevent_tag=prevent_tag,
        alt_tag=alt_tag,
    )

    sampling_obj = _sampling_meta(
        args,
        selected_count=getattr(args, "_selected_pair_count_for_mode", None),
        candidate_count=getattr(args, "_candidate_pair_count_for_mode", None),
        selected_patients=getattr(args, "_selected_patient_count_for_mode", None),
        candidate_patients=getattr(args, "_candidate_patient_count_for_mode", None),
    )

    parent_or["selection_policy"] = {
        "eval_mode": eval_mode,
        "union_of": [
            f"SMT label == {SMT_POSITIVE_LABEL}",
            f"TrialGPT interval_mno == {TRIALGPT_POSITIVE_INTERVAL}",
        ],
    }
    parent_or["sampling"] = sampling_obj

    parent_path = pair_dir / f"{canonical_nct_id}__parent_or.json"
    parent_path.write_text(
        json.dumps(parent_or, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    llm_payload, tg_payload = _load_parent_cached_judges(pair_dir, canonical_nct_id)

    overall_parent = _build_parent_overall(
        patient_id=patient_id,
        canonical_nct_id=canonical_nct_id,
        timestamp=timestamp,
        eval_mode=eval_mode,
        mode=compose_mode,
        prevent_tag=prevent_tag,
        alt_tag=alt_tag,
        parent_or=parent_or,
        llm_payload=llm_payload,
        tg_payload=tg_payload,
        sampling=sampling_obj,
    )

    overall_parent_path = pair_dir / f"{canonical_nct_id}__overall_parent.json"
    overall_parent_path.write_text(
        json.dumps(overall_parent, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    return {
        "patient_id": patient_id,
        "eval_mode": eval_mode,
        "mode": compose_mode,
        "prevent_tag": prevent_tag,
        "alt_mode": alt_tag,
        "source_list_to_match_file": str(meta_path),
        "canonical_trial": {
            "canonical_nct_id": canonical_nct_id,
            "status": canon_status,
            "label": canon_label,
            "sub_nct_ids": canon_sub_ids,
            "parent_or": parent_or,
            "overall_parent": overall_parent,
            "subcohorts": sub_results,
        },
    }


def _process_pair_run_worker(
    patient_id: str,
    canonical_nct_id: str,
    list_base_str: str,
    compose_mode: str,
    prevent_tag: str,
    alt_tag: str,
    eval_mode: str,
    args_dict: Dict[str, Any],
) -> Dict[str, Any]:
    list_base = Path(list_base_str)
    args_dict = _normalize_args_dict_for_run_single_pair(dict(args_dict))
    args = SimpleNamespace(**args_dict)
    return _process_pair_run_core(
        patient_id=patient_id,
        canonical_nct_id=canonical_nct_id,
        list_base=list_base,
        compose_mode=compose_mode,
        prevent_tag=prevent_tag,
        alt_tag=alt_tag,
        eval_mode=eval_mode,
        args=args,
    )


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rerun batch matcher on union(SMT all_satisfied, TrialGPT interval_mno=m) "
            "from already-ran eval patients."
        )
    )

    ap.add_argument("--data-root", default="../dataset/clinical_trial")
    ap.add_argument("--patient-file", default=None)
    ap.add_argument("--build-root", default="../build")
    ap.add_argument("--project-root", default="..")
    ap.add_argument("--prompt-root", default="prompts/")
    ap.add_argument("--prompt-map", default=None)

    ap.add_argument(
        "--miner-mode",
        choices=["infer", "explicit"],
        default="infer",
        help="Choose miner prompt mode for BOTH inclusion and exclusion.",
    )
    ap.add_argument(
        "--conservative",
        action="store_true",
        help="Use conservative miner prompt variant if available.",
    )

    ap.add_argument("--canonical-only", action="store_true")
    ap.add_argument("--canonical-json", default=None)
    ap.add_argument("--vv-batch-size", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--labels", action="append", default=None)
    ap.add_argument("--log-json", default=None)

    ap.add_argument(
        "--list-base",
        type=Path,
        default=None,
        help=(
            "Base directory containing list_to_match__{mode}__{prevent}__{alt}/ directories. "
            "If omitted, defaults to ../irsrc/ops/out_compose relative to this script."
        ),
    )

    ap.add_argument(
        "--smt-eval-root",
        type=Path,
        default=Path("../irsrc/eval/smt_retrieval_eval_out"),
        help="Root containing SMT eval outputs: <root>/<mode>/patient_labels/*.json",
    )
    ap.add_argument(
        "--trialgpt-eval-root",
        type=Path,
        default=Path("../irsrc/eval/trialgpt_retrieval_eval_out"),
        help="Root containing TrialGPT eval outputs: <root>/<mode>/patient_labels/*.json",
    )

    ap.add_argument(
        "--eval-modes",
        nargs="*",
        default=None,
        help="Subset of eval modes to use: ccr all all-explore. Default: all three.",
    )

    ap.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional subset of patient_ids. Restricts to these among already-ran eval patients.",
    )

    ap.add_argument(
        "--sample-patients",
        type=int,
        default=None,
        help="Number of patients to sample per eval mode before pair sampling.",
    )
    ap.add_argument(
        "--patient-sample-policy",
        choices=["hash", "random", "first"],
        default="hash",
        help="Patient sampling policy.",
    )
    ap.add_argument(
        "--patient-sample-seed",
        type=int,
        default=0,
        help="Seed for patient sampling.",
    )

    ap.add_argument(
        "--sample-pairs",
        type=int,
        default=None,
        help="Total number of canonical (patient, trial) pairs to evaluate per eval mode after union selection.",
    )
    ap.add_argument(
        "--sample-per-patient",
        type=int,
        default=None,
        help="Maximum number of canonical trials to evaluate per patient per eval mode before global pair sampling.",
    )
    ap.add_argument(
        "--sample-policy",
        choices=["hash", "random", "first"],
        default="hash",
        help="Pair sampling policy.",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for pair sampling.",
    )

    ap.add_argument(
        "--out-root",
        default="match_out_eval_union",
        help="Base folder for per-pair outputs and per-run list_to_match_eval outputs.",
    )

    ap.add_argument(
        "--skip-missing-run-file",
        action="store_true",
        help="If a patient is missing in compose list_to_match for the mapped run, skip it.",
    )

    ap.add_argument(
        "--parallel",
        type=int,
        default=64,
        help="Number of worker processes over flat (patient, canonical_trial) pair jobs (>=1).",
    )

    # kept only for compatibility; ignored by this flat design
    ap.add_argument(
        "--trial-parallel",
        type=int,
        default=1,
        help="Ignored in flat pair-parallel mode; retained for CLI compatibility.",
    )
    ap.add_argument(
        "--trial-engine-per-task",
        action="store_true",
        help="Legacy compatibility flag.",
    )

    ap.add_argument("--llm-judge", default=True, action="store_true")
    ap.add_argument("--llm-judge-prompt", default=None)
    ap.add_argument("--llm-judge-temperature", type=float, default=0.0)
    ap.add_argument("--llm-judge-force", action="store_true")

    ap.add_argument("--trialgpt-judge", action="store_true", default=True)
    ap.add_argument("--trialgpt-judge-temperature", default=None)
    ap.add_argument("--trialgpt-judge-force", action="store_true")

    args = ap.parse_args(argv)

    script_dir = Path(__file__).resolve().parent

    if args.list_base is None:
        list_base = (script_dir / "../irsrc/ops/out_compose").resolve()
    else:
        list_base = args.list_base.resolve()

    smt_eval_root = args.smt_eval_root.resolve()
    trialgpt_eval_root = args.trialgpt_eval_root.resolve()

    if not list_base.exists():
        raise FileNotFoundError(f"list base not found: {list_base}")
    if not smt_eval_root.exists():
        raise FileNotFoundError(f"SMT eval root not found: {smt_eval_root}")
    if not trialgpt_eval_root.exists():
        raise FileNotFoundError(f"TrialGPT eval root not found: {trialgpt_eval_root}")

    if args.eval_modes:
        eval_modes = [m.strip() for m in args.eval_modes if str(m).strip()]
    else:
        eval_modes = list(DEFAULT_EVAL_MODES)

    for m in eval_modes:
        if m not in MODE_TO_RUN:
            raise ValueError(f"Unknown eval mode: {m}")

    requested_patients: Optional[Set[str]] = None
    if args.patients:
        requested_patients = set(args.patients)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    parallel = max(1, int(args.parallel or 1))
    if int(getattr(args, "trial_parallel", 1) or 1) != 1:
        print(
            "[info] --trial-parallel is ignored in this flat pair-parallel version; "
            "concurrency is controlled only by --parallel.",
            file=sys.stderr,
        )

    for eval_mode in eval_modes:
        compose_mode, prevent_tag, alt_tag = MODE_TO_RUN[eval_mode]
        run_key = _run_key(compose_mode, prevent_tag, alt_tag)
        agg_root = out_root / f"list_to_match_eval__{run_key}"
        agg_root.mkdir(parents=True, exist_ok=True)

        selected_union = _build_union_selection_for_mode(
            eval_mode=eval_mode,
            smt_eval_root=smt_eval_root,
            trialgpt_eval_root=trialgpt_eval_root,
            requested_patients=requested_patients,
        )

        if not selected_union:
            print(f"[warn] No already-ran patients found for eval_mode={eval_mode}", file=sys.stderr)
            continue

        sampled_patients_union, selected_patient_count, candidate_patient_count = _sample_patients(
            selected_union,
            eval_mode=eval_mode,
            sample_patients=args.sample_patients,
            patient_sample_policy=args.patient_sample_policy,
            patient_sample_seed=args.patient_sample_seed,
        )

        selected_by_patient = _apply_sample_per_patient_union(
            sampled_patients_union,
            eval_mode=eval_mode,
            sample_per_patient=args.sample_per_patient,
            sample_policy=args.sample_policy,
            sample_seed=args.sample_seed,
        )

        selected_pairs_for_mode, selected_pair_count, candidate_pair_count = _sample_global_pairs(
            selected_by_patient,
            eval_mode=eval_mode,
            sample_pairs=args.sample_pairs,
            sample_policy=args.sample_policy,
            sample_seed=args.sample_seed,
        )

        if not selected_pairs_for_mode:
            print(f"[warn] No selected pairs for eval_mode={eval_mode}", file=sys.stderr)
            continue

        pair_jobs = sorted(selected_pairs_for_mode)

        print(
            f"[mode={eval_mode}] mapped_run={compose_mode}/{prevent_tag}/{alt_tag} "
            f"candidate_patients={candidate_patient_count} selected_patients={selected_patient_count} "
            f"candidate_pairs={candidate_pair_count} selected_pairs={selected_pair_count} "
            f"parallel_pair_workers={parallel}",
            file=sys.stderr,
        )

        args_dict = dict(vars(args))
        for k, v in list(args_dict.items()):
            if isinstance(v, Path):
                args_dict[k] = str(v)
        args_dict = _normalize_args_dict_for_run_single_pair(args_dict)
        args_dict["_selected_pair_count_for_mode"] = selected_pair_count
        args_dict["_candidate_pair_count_for_mode"] = candidate_pair_count
        args_dict["_selected_patient_count_for_mode"] = selected_patient_count
        args_dict["_candidate_patient_count_for_mode"] = candidate_patient_count

        list_base_str = str(list_base)

        patient_buckets: Dict[str, Dict[str, Any]] = {}

        if parallel == 1:
            with tqdm(
                total=len(pair_jobs),
                desc=f"{eval_mode} pairs",
                unit="pair",
                dynamic_ncols=True,
                file=sys.stderr,
            ) as pbar:
                for pid, canonical_nct_id in pair_jobs:
                    try:
                        pair_out = _process_pair_run_worker(
                            pid,
                            canonical_nct_id,
                            list_base_str,
                            compose_mode,
                            prevent_tag,
                            alt_tag,
                            eval_mode,
                            args_dict,
                        )
                    except FileNotFoundError as e:
                        if not args.skip_missing_run_file:
                            print(f"[warn] {e}", file=sys.stderr)
                        pbar.update(1)
                        continue
                    except Exception as e:
                        print(
                            f"[error] worker failed for patient={pid} canonical_nct_id={canonical_nct_id} "
                            f"eval_mode={eval_mode} mapped_run={compose_mode}/{prevent_tag}/{alt_tag}: {e}",
                            file=sys.stderr,
                        )
                        pbar.update(1)
                        continue

                    bucket = patient_buckets.setdefault(
                        pid,
                        {
                            "patient_id": pid,
                            "eval_mode": eval_mode,
                            "mode": compose_mode,
                            "prevent_tag": prevent_tag,
                            "alt_mode": alt_tag,
                            "source_list_to_match_file": pair_out.get("source_list_to_match_file"),
                            "selection_policy": {
                                "union_of": [
                                    f"SMT label == {SMT_POSITIVE_LABEL}",
                                    f"TrialGPT interval_mno == {TRIALGPT_POSITIVE_INTERVAL}",
                                ]
                            },
                            "sampling": _sampling_meta(
                                SimpleNamespace(**args_dict),
                                selected_count=selected_pair_count,
                                candidate_count=candidate_pair_count,
                                selected_patients=selected_patient_count,
                                candidate_patients=candidate_patient_count,
                            ),
                            "canonical_trials": [],
                        },
                    )
                    bucket["canonical_trials"].append(pair_out["canonical_trial"])
                    pbar.update(1)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as ex:
                futures = {
                    ex.submit(
                        _process_pair_run_worker,
                        pid,
                        canonical_nct_id,
                        list_base_str,
                        compose_mode,
                        prevent_tag,
                        alt_tag,
                        eval_mode,
                        args_dict,
                    ): (pid, canonical_nct_id)
                    for pid, canonical_nct_id in pair_jobs
                }

                with tqdm(
                    total=len(futures),
                    desc=f"{eval_mode} pairs",
                    unit="pair",
                    dynamic_ncols=True,
                    file=sys.stderr,
                ) as pbar:
                    for fut in as_completed(futures):
                        pid, canonical_nct_id = futures[fut]
                        try:
                            pair_out = fut.result()
                        except FileNotFoundError as e:
                            if not args.skip_missing_run_file:
                                print(f"[warn] {e}", file=sys.stderr)
                            pbar.update(1)
                            continue
                        except Exception as e:
                            print(
                                f"[error] worker failed for patient={pid} canonical_nct_id={canonical_nct_id} "
                                f"eval_mode={eval_mode} mapped_run={compose_mode}/{prevent_tag}/{alt_tag}: {e}",
                                file=sys.stderr,
                            )
                            pbar.update(1)
                            continue

                        bucket = patient_buckets.setdefault(
                            pid,
                            {
                                "patient_id": pid,
                                "eval_mode": eval_mode,
                                "mode": compose_mode,
                                "prevent_tag": prevent_tag,
                                "alt_mode": alt_tag,
                                "source_list_to_match_file": pair_out.get("source_list_to_match_file"),
                                "selection_policy": {
                                    "union_of": [
                                        f"SMT label == {SMT_POSITIVE_LABEL}",
                                        f"TrialGPT interval_mno == {TRIALGPT_POSITIVE_INTERVAL}",
                                    ]
                                },
                                "sampling": _sampling_meta(
                                    SimpleNamespace(**args_dict),
                                    selected_count=selected_pair_count,
                                    candidate_count=candidate_pair_count,
                                    selected_patients=selected_patient_count,
                                    candidate_patients=candidate_patient_count,
                                ),
                                "canonical_trials": [],
                            },
                        )
                        bucket["canonical_trials"].append(pair_out["canonical_trial"])
                        pbar.update(1)

        for pid, patient_out in patient_buckets.items():
            patient_out["canonical_trials"].sort(
                key=lambda ct: (
                    ct.get("canonical_nct_id") is None,
                    ct.get("canonical_nct_id") or "",
                )
            )
            out_path = agg_root / f"{pid}{_run_suffix(compose_mode, prevent_tag, alt_tag)}.json"
            out_path.write_text(
                json.dumps(patient_out, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()