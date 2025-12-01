#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_smt_top200_wins.py

SMT positive vs TrialGPT(top-K relevant=True and eligible=True).

Definitions
-----------
SMT positive:
    label == "all_satisfied"
    AND any_subcohort_relevant == True
    AND any_subcohort_eligible == True

TrialGPT positive:
    rank <= K AND relevant == True AND eligible == True

A trial is counted as an SMT win iff:
  * SMT contains the canonical trial in the SMT-positive set
  * TrialGPT top-K does NOT contain that canonical trial with relevant=True and eligible=True

We annotate the TrialGPT side as:
  * no_hit
  * outside_topk_re_and_elig
  * topk_not_relevant
  * topk_not_eligible
  * topk_not_relevant_and_not_eligible
  * outside_topk_not_relevant
  * outside_topk_not_eligible
  * outside_topk_not_relevant_and_not_eligible
  * topk_unclear
  * outside_topk_unclear
  * topk_re_and_elig          (should not appear for wins; retained for sanity)

Outputs
-------
  * smt_positive_trialgpt_not_topk_re_elig.csv
  * summary.csv
  * summary_by_conflict.csv
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


DEFAULT_TRIALGPT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out_top200")
DEFAULT_SMT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)
KNOWN_MODES = ("chief", "ccr", "all", "all-explore")


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


def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


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


def smt_item_is_positive(it: SMTItem) -> bool:
    if it.any_rel_and_elig is True:
        return it.label == "all_satisfied"
    return (
        it.label == "all_satisfied"
        and it.any_rel is True
        and it.any_elig is True
    )


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

        any_rel = is_truthy(t.get("any_subcohort_relevant"))
        any_elig = is_truthy(t.get("any_subcohort_eligible"))
        any_re = is_truthy(t.get("any_subcohort_relevant_and_eligible"))

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
                rank=rank,
                label=lab,
                any_rel=any_rel,
                any_elig=any_elig,
                any_rel_and_elig=any_re,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find SMT positive wins against TrialGPT top-K relevant+eligible."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--out", type=Path, default=Path("./smt_top200_wins_out"))
    ap.add_argument("--trialgpt-k", type=int, default=200)
    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")
    ap.add_argument("--modes", type=str, default="")
    args = ap.parse_args()

    if args.trialgpt_k <= 0:
        raise ValueError("--trialgpt-k must be > 0")
    if not args.trialgpt_root.exists():
        print(f"[error] trialgpt root not found: {args.trialgpt_root}", file=sys.stderr)
        sys.exit(2)
    if not args.smt_root.exists():
        print(f"[error] smt root not found: {args.smt_root}", file=sys.stderr)
        sys.exit(2)

    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)

    tg_runs = collect_trialgpt_runs(args.trialgpt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)
    smt_runs = collect_smt_runs(args.smt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)

    if mode_filter is not None:
        tg_runs = {k: v for k, v in tg_runs.items() if k[1] in mode_filter}
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}

    smt_pair_index: Dict[Tuple[str, str], List[SMTRun]] = {}
    for _, run in smt_runs.items():
        smt_pair_index.setdefault((run.patient_id, run.mode), []).append(run)

    keys_tg = set(tg_runs.keys())
    keys_smt = set(smt_pair_index.keys())
    keys_all = sorted(keys_tg & keys_smt) if args.require_overlap else sorted(keys_tg | keys_smt)

    rows: List[Dict[str, Any]] = []
    summary_by_mode: Dict[str, int] = {}
    summary_by_mode_and_conflict: Dict[Tuple[str, str], int] = {}
    pairs_eval = 0

    for (patient_id, mode) in keys_all:
        tg = tg_runs.get((patient_id, mode))
        smt_runs_for_pair = smt_pair_index.get((patient_id, mode), [])

        if not smt_runs_for_pair:
            continue
        if args.require_overlap and tg is None:
            continue

        pairs_eval += 1

        # tg_top = tg.items[:args.trialgpt_k] if tg is not None else []
        # tg_top_re_elig_keys: Set[str] = {
        #     x.key for x in tg_top
        #     if x.relevant is True and x.eligible is True
        # }
        # tg_top_re_elig_count = len(tg_top_re_elig_keys)
        tg_top = tg.items[:args.trialgpt_k] if tg is not None else []
        tg_top_keys: Set[str] = {x.key for x in tg_top}
        tg_top_count = len(tg_top_keys)

        smt_best_positive: Dict[str, SMTItem] = {}
        for smt_run in smt_runs_for_pair:
            for x in smt_run.items:
                if not smt_item_is_positive(x):
                    continue
                prev = smt_best_positive.get(x.parent_trial_id)
                if prev is None or x.rank < prev.rank:
                    smt_best_positive[x.parent_trial_id] = x

        for trial_key, smt_it in sorted(smt_best_positive.items(), key=lambda kv: kv[1].rank):
            # if trial_key in tg_top_re_elig_keys:
            #     continue
            if trial_key in tg_top_keys:
                continue

            conflict = choose_best_trialgpt_conflict_for_trial(
                trial_key=trial_key,
                tg_run=tg,
                tg_k=args.trialgpt_k,
            )

            rows.append({
                "patient_id": patient_id,
                "mode": mode,
                "trial_id": trial_key,
                "smt_rank": smt_it.rank,
                "smt_label": smt_it.label or "",
                "smt_any_relevant": smt_it.any_rel,
                "smt_any_eligible": smt_it.any_elig,
                "smt_any_relevant_and_eligible": smt_it.any_rel_and_elig,
                "smt_num_subcohorts": len(smt_it.subcohorts),
                "smt_subcohorts": "|".join(smt_it.subcohorts),
                "trialgpt_conflict_category": conflict.conflict_label,
                "trialgpt_rank": conflict.tg_rank if conflict.tg_rank is not None else "",
                "trialgpt_relevant": conflict.tg_relevant,
                "trialgpt_eligible": conflict.tg_eligible,
                "trialgpt_label": conflict.tg_label or "",
                "trialgpt_fixed_k_used": args.trialgpt_k,
                "trialgpt_topk_count": tg_top_count,
                "reason": "trialgpt_missing_topk",
                "smt_file": str(smt_runs_for_pair[0].source_file) if smt_runs_for_pair else "",
                "trialgpt_file": str(conflict.source_file) if conflict.source_file else "",
            })

            summary_by_mode[mode] = summary_by_mode.get(mode, 0) + 1
            summary_by_mode_and_conflict[(mode, conflict.conflict_label)] = (
                summary_by_mode_and_conflict.get((mode, conflict.conflict_label), 0) + 1
            )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "patient_id", "mode",
        "trial_id",
        "smt_rank", "smt_label",
        "smt_any_relevant", "smt_any_eligible", "smt_any_relevant_and_eligible",
        "smt_num_subcohorts", "smt_subcohorts",
        "trialgpt_conflict_category",
        "trialgpt_rank", "trialgpt_relevant", "trialgpt_eligible", "trialgpt_label",
        "trialgpt_fixed_k_used",
        "trialgpt_top_re_elig_count",
        "reason", "smt_file", "trialgpt_file",
    ]
    write_csv(out_dir / "smt_positive_trialgpt_not_topk_re_elig.csv", header, rows)
    write_csv(
        out_dir / "summary.csv",
        ["mode", "count"],
        [{"mode": m, "count": cnt} for (m, cnt) in sorted(summary_by_mode.items())]
    )
    write_csv(
        out_dir / "summary_by_conflict.csv",
        ["mode", "trialgpt_conflict_category", "count"],
        [{"mode": m, "trialgpt_conflict_category": c, "count": cnt}
         for ((m, c), cnt) in sorted(summary_by_mode_and_conflict.items())]
    )

    print(f"[ok] wrote {out_dir / 'smt_positive_trialgpt_not_topk_re_elig.csv'}")
    print(f"[ok] wrote {out_dir / 'summary.csv'}")
    print(f"[ok] wrote {out_dir / 'summary_by_conflict.csv'}")
    print(f"[info] pairs_evaluated={pairs_eval} wins={len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)