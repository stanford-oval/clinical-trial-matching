#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_smt_topk_wins_sweep.py

Sweep SMT-win vs TrialGPT-topK analysis over multiple fixed K cutoffs
(e.g. 200,300,400,500), using the richer conflict-annotation logic.

For each K:
  * Load TrialGPT outputs from trialgpt_retrieval_eval_out_top<K>
  * Compare against SMT outputs
  * Count SMT wins:
        SMT(label == all_satisfied)
        AND TrialGPT top-K does NOT return the same canonical trial with
            relevant=True AND eligible=True
  * Annotate TrialGPT-side conflict category
  * Optionally copy artifacts into per-K output directory

Outputs
-------
Per-K output directory:
    <out-base>_top<K>/

Inside each per-K output dir:
  * smt_all_satisfied_trialgpt_not_topk_re_elig.csv
  * summary.csv
  * summary_by_conflict.csv
  * copied_trialgpt_mbench/...           (if copy enabled)

Aggregate output directory:
    <out-base>_aggregate/

Inside aggregate dir:
  * summary_all_k.csv
  * summary_by_conflict_all_k.csv
  * smt_all_satisfied_trialgpt_not_topk_re_elig_all_k.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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
DEFAULT_TRIALGPT_MBENCH_ROOT_BASE = Path(
    "<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_mbench"
)

DEFAULT_SMT_ROOT = Path(
    "<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out"
)
DEFAULT_SMT_MBENCH_ROOT = Path(
    "<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_mbench"
)

DEFAULT_CORPUS_JSONL = Path(
    "<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus.jsonl"
)
DEFAULT_QUERIES_JSONL = Path(
    "<SATIR_ROOT>/dataset/clinical_trial/sigir/queries.jsonl"
)

DEFAULT_DISEASE_ROOT = Path(
    "<SATIR_ROOT>/build/disease_filtered_categorized"
)
DEFAULT_POSLIT_ROOT = Path(
    "<SATIR_ROOT>/build/positive_constraint_literals_categorized"
)
DEFAULT_DEFAULT_VARS_ROOT = Path(
    "<SATIR_ROOT>/build/default_vars"
)
DEFAULT_PROJECTED_SMT_ROOT = Path(
    "<SATIR_ROOT>/build/canon_projection/_projected_smt"
)

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
        raise ValueError("No valid K values in --topk-list.")
    return vals


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


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
# TrialGPT conflict annotation
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


# ----------------------------------------------------------------------
# Artifact expansion / copy helpers
# ----------------------------------------------------------------------

def _expand_subcohorts_for_copy(
    *,
    mode: str,
    patient_id: str,
    parent_trial_id: str,
    subs: List[str],
    smt_mbench_root: Path,
    disease_root: Path,
    poslit_root: Path,
    default_vars_root: Path,
    projected_smt_root: Path,
) -> List[str]:
    expanded: Set[str] = set(x.strip() for x in subs if isinstance(x, str) and x.strip())

    base = canon_nct(parent_trial_id, collapse_subsuffix=True)
    if not base:
        return sorted(expanded)

    base_up = base.upper()

    mbench_parent_1 = smt_mbench_root / mode / patient_id / parent_trial_id
    if mbench_parent_1.exists() and mbench_parent_1.is_dir():
        for child in mbench_parent_1.iterdir():
            if child.is_dir():
                name = child.name.strip()
                if re.match(rf"^{re.escape(base_up)}[A-Z]$", name.upper()):
                    expanded.add(name)

    mbench_parent_2 = smt_mbench_root / mode / patient_id
    if mbench_parent_2.exists() and mbench_parent_2.is_dir():
        for child in mbench_parent_2.iterdir():
            if child.is_dir():
                name = child.name.strip()
                if re.match(rf"^{re.escape(base_up)}[A-Z]$", name.upper()):
                    expanded.add(name)

    if disease_root.exists():
        for f in disease_root.glob(f"{base_up}[a-z]_disease_link_filter_summary.json"):
            stem = f.name.split("_disease_link_filter_summary.json", 1)[0]
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", stem.upper()):
                expanded.add(stem)

    per_file = poslit_root / "per_file"
    if per_file.exists():
        for f in per_file.glob(f"{base_up}[a-z]_*program*.smt2.json"):
            prefix = f.name.split("_", 1)[0]
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", prefix.upper()):
                expanded.add(prefix)

    if default_vars_root.exists():
        for f in default_vars_root.glob(f"{base_up}[a-z]_*.json"):
            prefix = f.name.split("_", 1)[0]
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", prefix.upper()):
                expanded.add(prefix)

    if projected_smt_root.exists():
        for f in projected_smt_root.glob(f"{base_up}[a-z]_*.smt2"):
            prefix = f.name.split("_", 1)[0]
            if re.match(rf"^{re.escape(base_up)}[A-Z]$", prefix.upper()):
                expanded.add(prefix)

    return sorted(expanded)


def resolve_smt_mbench_src(
    *,
    smt_mbench_root: Path,
    mode: str,
    patient_id: str,
    trial_id: str,
    sub: str,
) -> Optional[Path]:
    candidates = [
        smt_mbench_root / mode / patient_id / trial_id / sub,
        smt_mbench_root / mode / patient_id / sub,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


# ----------------------------------------------------------------------
# Explanation text
# ----------------------------------------------------------------------

def build_trialgpt_conflict_explanation(
    *,
    patient_id: str,
    mode: str,
    trial_id: str,
    smt_rank: int,
    tg_k: int,
    conflict: TrialGPTConflictMatch,
) -> str:
    lines = [
        "SMT win with TrialGPT-side conflict annotation",
        f"patient_id: {patient_id}",
        f"mode: {mode}",
        f"trial_id: {trial_id}",
        f"smt_rank: {smt_rank}",
        f"trialgpt_fixed_k_used: {tg_k}",
        "",
        "Why this is a win:",
        "- SMT labeled this trial as all_satisfied.",
        "- TrialGPT top-K did NOT contain this trial with relevant=True and eligible=True.",
        "",
        f"TrialGPT conflict category: {conflict.conflict_label}",
    ]

    if conflict.conflict_label == "no_hit":
        lines += [
            "- Explanation: TrialGPT did not retrieve this canonical trial at all.",
        ]
    else:
        lines += [
            f"- TrialGPT rank: {conflict.tg_rank}",
            f"- TrialGPT relevant: {conflict.tg_relevant}",
            f"- TrialGPT eligible: {conflict.tg_eligible}",
            f"- TrialGPT label: {conflict.tg_label}",
            f"- TrialGPT source_file: {conflict.source_file}",
        ]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Root resolution
# ----------------------------------------------------------------------

def make_topk_root(base: Path, k: int) -> Path:
    return Path(f"{base}_top{k}")


# ----------------------------------------------------------------------
# Core single-K analysis
# ----------------------------------------------------------------------

def run_single_k(
    *,
    k: int,
    trialgpt_root: Path,
    trialgpt_mbench_root: Path,
    smt_root: Path,
    smt_mbench_root: Path,
    out_dir: Path,
    dedup_canonical: bool,
    keep_subcohort_suffix: bool,
    require_overlap: bool,
    mode_filter: Optional[List[str]],
    copy_mbench: bool,
    copy_overwrite: bool,
    copy_dry_run: bool,
    add_listings: bool,
    copy_disease: bool,
    copy_poslits: bool,
    copy_default_vars: bool,
    copy_projected_smt: bool,
    corpus_jsonl: Path,
    queries_jsonl: Path,
    disease_root: Path,
    poslit_root: Path,
    default_vars_root: Path,
    projected_smt_root: Path,
) -> Dict[str, Any]:
    if not trialgpt_root.exists():
        raise FileNotFoundError(f"trialgpt root not found for top{k}: {trialgpt_root}")
    if not smt_root.exists():
        raise FileNotFoundError(f"smt root not found: {smt_root}")

    if copy_mbench:
        if not trialgpt_mbench_root.exists():
            raise FileNotFoundError(f"trialgpt mbench root not found for top{k}: {trialgpt_mbench_root}")
        if not smt_mbench_root.exists():
            raise FileNotFoundError(f"smt mbench root not found: {smt_mbench_root}")

    if add_listings:
        if not corpus_jsonl.exists():
            raise FileNotFoundError(f"corpus jsonl not found: {corpus_jsonl}")
        if not queries_jsonl.exists():
            raise FileNotFoundError(f"queries jsonl not found: {queries_jsonl}")

    if copy_disease and not disease_root.exists():
        raise FileNotFoundError(f"disease root not found: {disease_root}")
    if copy_poslits and not poslit_root.exists():
        raise FileNotFoundError(f"poslit root not found: {poslit_root}")
    if copy_default_vars and not default_vars_root.exists():
        raise FileNotFoundError(f"default vars root not found: {default_vars_root}")
    if copy_projected_smt and not projected_smt_root.exists():
        raise FileNotFoundError(f"projected smt root not found: {projected_smt_root}")

    collapse = not keep_subcohort_suffix

    tg_runs = collect_trialgpt_runs(trialgpt_root, collapse_subsuffix=collapse, dedup=dedup_canonical)
    smt_runs = collect_smt_runs(smt_root, collapse_subsuffix=collapse, dedup=dedup_canonical)

    if mode_filter is not None:
        tg_runs = {kk: v for kk, v in tg_runs.items() if kk[1] in mode_filter}
        smt_runs = {kk: v for kk, v in smt_runs.items() if kk[1] in mode_filter}

    smt_pair_index: Dict[Tuple[str, str], List[SMTRun]] = {}
    for _, run in smt_runs.items():
        smt_pair_index.setdefault((run.patient_id, run.mode), []).append(run)

    keys_tg = set(tg_runs.keys())
    keys_smt = set(smt_pair_index.keys())
    keys_all = sorted(keys_tg & keys_smt) if require_overlap else sorted(keys_tg | keys_smt)

    rows: List[Dict[str, Any]] = []
    summary_by_mode: Dict[str, int] = {}
    summary_by_mode_and_conflict: Dict[Tuple[str, str], int] = {}
    copy_targets: List[Dict[str, Any]] = []

    pairs_eval = 0

    for (patient_id, mode) in keys_all:
        tg = tg_runs.get((patient_id, mode))
        smt_runs_for_pair = smt_pair_index.get((patient_id, mode), [])

        if not smt_runs_for_pair:
            continue
        if require_overlap and tg is None:
            continue

        pairs_eval += 1

        tg_top = tg.items[:k] if tg is not None else []
        tg_top_re_elig_keys: Set[str] = {
            x.key for x in tg_top
            if x.relevant is True and x.eligible is True
        }
        tg_top_re_elig_count = len(tg_top_re_elig_keys)

        smt_best_all_satisfied: Dict[str, SMTItem] = {}
        smt_source_file_by_trial: Dict[str, Path] = {}
        for smt_run in smt_runs_for_pair:
            for x in smt_run.items:
                if x.label != "all_satisfied":
                    continue
                prev = smt_best_all_satisfied.get(x.parent_trial_id)
                if prev is None or x.rank < prev.rank:
                    smt_best_all_satisfied[x.parent_trial_id] = x
                    smt_source_file_by_trial[x.parent_trial_id] = smt_run.source_file

        for trial_key, smt_it in sorted(smt_best_all_satisfied.items(), key=lambda kv: kv[1].rank):
            if trial_key in tg_top_re_elig_keys:
                continue

            conflict = choose_best_trialgpt_conflict_for_trial(
                trial_key=trial_key,
                tg_run=tg,
                tg_k=k,
            )

            rows.append({
                "topk": k,
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
                "trialgpt_fixed_k_used": k,
                "trialgpt_top_re_elig_count": tg_top_re_elig_count,
                "reason": "trialgpt_missing_topk_re_and_elig",
                "smt_file": str(smt_source_file_by_trial.get(trial_key, "")),
                "trialgpt_file": str(conflict.source_file) if conflict.source_file else "",
            })

            summary_by_mode[mode] = summary_by_mode.get(mode, 0) + 1
            summary_by_mode_and_conflict[(mode, conflict.conflict_label)] = (
                summary_by_mode_and_conflict.get((mode, conflict.conflict_label), 0) + 1
            )

            if copy_mbench:
                copy_targets.append({
                    "patient_id": patient_id,
                    "mode": mode,
                    "trial_id": trial_key,
                    "smt_rank": smt_it.rank,
                    "smt_label": smt_it.label,
                    "smt_subcohorts": list(smt_it.subcohorts),
                    "trialgpt_conflict_label": conflict.conflict_label,
                    "trialgpt_rank": conflict.tg_rank,
                    "trialgpt_relevant": conflict.tg_relevant,
                    "trialgpt_eligible": conflict.tg_eligible,
                    "trialgpt_label": conflict.tg_label,
                    "trialgpt_file": str(conflict.source_file) if conflict.source_file else "",
                    "trialgpt_k": k,
                })

    out_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "topk",
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
    write_csv(out_dir / "smt_all_satisfied_trialgpt_not_topk_re_elig.csv", header, rows)
    write_csv(
        out_dir / "summary.csv",
        ["topk", "mode", "count"],
        [{"topk": k, "mode": m, "count": cnt} for (m, cnt) in sorted(summary_by_mode.items())]
    )
    write_csv(
        out_dir / "summary_by_conflict.csv",
        ["topk", "mode", "trialgpt_conflict_category", "count"],
        [{"topk": k, "mode": m, "trialgpt_conflict_category": c, "count": cnt}
         for ((m, c), cnt) in sorted(summary_by_mode_and_conflict.items())]
    )

    print(f"[ok][top{k}] wrote {out_dir / 'smt_all_satisfied_trialgpt_not_topk_re_elig.csv'}")
    print(f"[ok][top{k}] wrote {out_dir / 'summary.csv'}")
    print(f"[ok][top{k}] wrote {out_dir / 'summary_by_conflict.csv'}")
    print(f"[info][top{k}] pairs_evaluated={pairs_eval} wins={len(rows)}")

    if not copy_mbench:
        return {
            "k": k,
            "pairs_evaluated": pairs_eval,
            "rows": rows,
            "summary_by_mode": summary_by_mode,
            "summary_by_mode_and_conflict": summary_by_mode_and_conflict,
        }

    corpus_index: Dict[str, Dict[str, Any]] = {}
    query_index: Dict[str, Dict[str, Any]] = {}
    if add_listings:
        corpus_index = load_jsonl_index(corpus_jsonl, id_key="_id")
        query_index = load_jsonl_index(queries_jsonl, id_key="_id")

    copied_dirs = 0
    skipped_dirs = 0
    missing_dirs = 0
    copied_files = 0
    skipped_files = 0
    missing_files = 0

    uniq: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    priority = {
        "no_hit": 0,
        "outside_topk_re_and_elig": 1,
        "topk_not_relevant": 2,
        "topk_not_eligible": 3,
        "topk_not_relevant_and_not_eligible": 4,
        "outside_topk_not_relevant": 5,
        "outside_topk_not_eligible": 6,
        "outside_topk_not_relevant_and_not_eligible": 7,
        "topk_unclear": 8,
        "outside_topk_unclear": 9,
        "topk_re_and_elig": 99,
    }

    for x in copy_targets:
        kk = (x["mode"], x["patient_id"], x["trial_id"])
        prev = uniq.get(kk)
        if prev is None:
            uniq[kk] = x
        else:
            merged_subs = set(prev.get("smt_subcohorts", []))
            merged_subs.update(x.get("smt_subcohorts", []))
            prev["smt_subcohorts"] = sorted(merged_subs)

            if priority.get(x.get("trialgpt_conflict_label", "no_hit"), 99) < priority.get(prev.get("trialgpt_conflict_label", "no_hit"), 99):
                prev["trialgpt_conflict_label"] = x.get("trialgpt_conflict_label")
                prev["trialgpt_rank"] = x.get("trialgpt_rank")
                prev["trialgpt_relevant"] = x.get("trialgpt_relevant")
                prev["trialgpt_eligible"] = x.get("trialgpt_eligible")
                prev["trialgpt_label"] = x.get("trialgpt_label")
                prev["trialgpt_file"] = x.get("trialgpt_file")

            if x.get("smt_rank") is not None:
                if prev.get("smt_rank") is None or int(x["smt_rank"]) < int(prev["smt_rank"]):
                    prev["smt_rank"] = x["smt_rank"]
                    prev["smt_label"] = x.get("smt_label")

    print(f"[info][top{k}] unique trial dirs to copy={len(uniq)}")

    for _, meta in sorted(uniq.items(), key=lambda kv: kv[0]):
        mode = meta["mode"]
        patient_id = meta["patient_id"]
        trial_id = meta["trial_id"]

        smt_rank = meta["smt_rank"]
        smt_label = meta.get("smt_label")
        smt_subs = list(meta.get("smt_subcohorts", []))

        trialgpt_conflict_label = meta.get("trialgpt_conflict_label", "no_hit")
        trialgpt_rank = meta.get("trialgpt_rank")
        trialgpt_relevant = meta.get("trialgpt_relevant")
        trialgpt_eligible = meta.get("trialgpt_eligible")
        trialgpt_label = meta.get("trialgpt_label")
        trialgpt_file = meta.get("trialgpt_file", "")
        trialgpt_k = meta["trialgpt_k"]

        base_dst = out_dir / "copied_trialgpt_mbench" / mode / patient_id / trial_id

        src_tg = trialgpt_mbench_root / mode / patient_id / trial_id
        ok, msg = copy_dir(src_tg, base_dst, overwrite=copy_overwrite, dry_run=copy_dry_run)
        print(f"[trialgpt_mbench:{msg}] {src_tg} -> {base_dst}")

        if not ok and msg == "missing_src":
            missing_dirs += 1
            continue
        elif ok and msg == "exists_skip":
            skipped_dirs += 1
        elif ok and msg in {"copied", "would_copy"}:
            copied_dirs += 1

        if add_listings:
            patient_obj = query_index.get(patient_id)
            trial_obj = corpus_index.get(trial_id)

            if copy_dry_run:
                print(f"[would_write] {base_dst / 'patient_query.json'}")
                print(f"[would_write] {base_dst / 'trial_listing.json'}")
            else:
                write_json(
                    base_dst / "patient_query.json",
                    patient_obj if patient_obj is not None else {"_id": patient_id, "missing": True}
                )
                write_json(
                    base_dst / "trial_listing.json",
                    trial_obj if trial_obj is not None else {"_id": trial_id, "missing": True}
                )

        subs: List[str] = _expand_subcohorts_for_copy(
            mode=mode,
            patient_id=patient_id,
            parent_trial_id=trial_id,
            subs=smt_subs,
            smt_mbench_root=smt_mbench_root,
            disease_root=disease_root,
            poslit_root=poslit_root,
            default_vars_root=default_vars_root,
            projected_smt_root=projected_smt_root,
        )

        if not subs:
            subs = [trial_id]

        print(f"[info][top{k}] artifact subcohorts for {mode}/{patient_id}/{trial_id}: {subs}")

        if add_listings and not copy_dry_run:
            for sub in subs:
                sub_obj = corpus_index.get(sub)
                if sub_obj is not None:
                    write_json(base_dst / f"trial_listing__{sub}.json", sub_obj)

        for sub in subs:
            src = resolve_smt_mbench_src(
                smt_mbench_root=smt_mbench_root,
                mode=mode,
                patient_id=patient_id,
                trial_id=trial_id,
                sub=sub,
            )
            dst = base_dst / "smt_mbench" / sub
            if src is None:
                missing_dirs += 1
                print(f"[missing_smt_mbench] no source found for mode={mode} patient={patient_id} trial={trial_id} sub={sub}")
            else:
                ok2, msg2 = copy_dir(src, dst, overwrite=copy_overwrite, dry_run=copy_dry_run)
                if not ok2 and msg2 == "missing_src":
                    missing_dirs += 1
                    print(f"[missing_smt_mbench] {src}")
                elif ok2 and msg2 == "exists_skip":
                    skipped_dirs += 1
                elif ok2 and msg2 in {"copied", "would_copy"}:
                    copied_dirs += 1
                print(f"[smt_mbench:{msg2}] {src} -> {dst}")

        if copy_disease:
            for sub in subs:
                src = disease_root / f"{sub}_disease_link_filter_summary.json"
                dst = base_dst / "disease" / src.name
                ok3, msg3 = copy_file(src, dst, overwrite=copy_overwrite, dry_run=copy_dry_run)
                if not ok3 and msg3 == "missing_src":
                    missing_files += 1
                    print(f"[missing_disease] {src}")
                elif ok3 and msg3 == "exists_skip":
                    skipped_files += 1
                elif ok3 and msg3 in {"copied", "would_copy"}:
                    copied_files += 1
                print(f"[disease:{msg3}] {src} -> {dst}")

        if copy_poslits:
            per_file = poslit_root / "per_file"
            for sub in subs:
                matches = sorted(per_file.glob(f"{sub}_*program*.smt2.json"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_poslits] {per_file} glob={sub}_*program*.smt2.json")
                    continue
                for src in matches:
                    dst = base_dst / "positive_constraint_literals" / src.name
                    ok4, msg4 = copy_file(src, dst, overwrite=copy_overwrite, dry_run=copy_dry_run)
                    if not ok4 and msg4 == "missing_src":
                        missing_files += 1
                        print(f"[missing_poslit] {src}")
                    elif ok4 and msg4 == "exists_skip":
                        skipped_files += 1
                    elif ok4 and msg4 in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[poslit:{msg4}] {src} -> {dst}")

        if copy_default_vars:
            for sub in subs:
                matches = sorted(default_vars_root.glob(f"{sub}_*.json"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_default_vars] {default_vars_root} glob={sub}_*.json")
                    continue
                for src in matches:
                    dst = base_dst / "default_vars" / src.name
                    ok5, msg5 = copy_file(src, dst, overwrite=copy_overwrite, dry_run=copy_dry_run)
                    if not ok5 and msg5 == "missing_src":
                        missing_files += 1
                        print(f"[missing_default_var] {src}")
                    elif ok5 and msg5 == "exists_skip":
                        skipped_files += 1
                    elif ok5 and msg5 in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[default_vars:{msg5}] {src} -> {dst}")

        if copy_projected_smt:
            for sub in subs:
                matches = sorted(projected_smt_root.glob(f"{sub}_*.smt2"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_projected_smt] {projected_smt_root} glob={sub}_*.smt2")
                    continue
                for src in matches:
                    dst = base_dst / "projected_smt" / src.name
                    ok6, msg6 = copy_file(src, dst, overwrite=copy_overwrite, dry_run=copy_dry_run)
                    if not ok6 and msg6 == "missing_src":
                        missing_files += 1
                        print(f"[missing_projected_smt_file] {src}")
                    elif ok6 and msg6 == "exists_skip":
                        skipped_files += 1
                    elif ok6 and msg6 in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[projected_smt:{msg6}] {src} -> {dst}")

        conflict_obj = {
            "topk": k,
            "patient_id": patient_id,
            "mode": mode,
            "trial_id": trial_id,
            "smt_rank": smt_rank,
            "smt_label": smt_label,
            "artifact_subcohorts": subs,
            "trialgpt_conflict_category": trialgpt_conflict_label,
            "trialgpt_rank": trialgpt_rank,
            "trialgpt_relevant": trialgpt_relevant,
            "trialgpt_eligible": trialgpt_eligible,
            "trialgpt_label": trialgpt_label,
            "trialgpt_file": trialgpt_file,
            "trialgpt_fixed_k_used": trialgpt_k,
            "explanation": "SMT found this trial as all_satisfied, but TrialGPT top-K did not return it as relevant-and-eligible.",
        }

        explanation = build_trialgpt_conflict_explanation(
            patient_id=patient_id,
            mode=mode,
            trial_id=trial_id,
            smt_rank=smt_rank,
            tg_k=trialgpt_k,
            conflict=TrialGPTConflictMatch(
                conflict_label=trialgpt_conflict_label,
                tg_rank=trialgpt_rank,
                tg_relevant=trialgpt_relevant,
                tg_eligible=trialgpt_eligible,
                tg_label=trialgpt_label,
                source_file=Path(trialgpt_file) if trialgpt_file else None,
            ),
        )

        if copy_dry_run:
            print(f"[would_write] {base_dst / 'trialgpt_conflict_summary.json'}")
            print(f"[would_write] {base_dst / 'trialgpt_conflict_explanation.txt'}")
        else:
            write_json(base_dst / "trialgpt_conflict_summary.json", conflict_obj)
            write_text(base_dst / "trialgpt_conflict_explanation.txt", explanation)

    print(
        f"[done][top{k}] dirs copied={copied_dirs} skipped={skipped_dirs} missing={missing_dirs} | "
        f"files copied={copied_files} skipped={skipped_files} missing={missing_files}"
    )
    if copy_dry_run:
        print(f"[info][top{k}] copy-dry-run enabled: no files were copied / written")

    return {
        "k": k,
        "pairs_evaluated": pairs_eval,
        "rows": rows,
        "summary_by_mode": summary_by_mode,
        "summary_by_mode_and_conflict": summary_by_mode_and_conflict,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep SMT(label=all_satisfied) wins against TrialGPT top-K relevant+eligible "
                    "for multiple K values, annotate TrialGPT-side misses, and optionally copy artifacts."
    )

    ap.add_argument("--topk-list", type=str, default="200,300,400,500")

    ap.add_argument("--trialgpt-root-base", type=Path, default=DEFAULT_TRIALGPT_ROOT_BASE)
    ap.add_argument("--trialgpt-mbench-root-base", type=Path, default=DEFAULT_TRIALGPT_MBENCH_ROOT_BASE)

    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--smt-mbench-root", type=Path, default=DEFAULT_SMT_MBENCH_ROOT)

    ap.add_argument("--out-base", type=Path, default=Path("./smt_topk_wins_out"))

    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")
    ap.add_argument("--modes", type=str, default="")

    ap.add_argument("--copy-mbench", dest="copy_mbench", action="store_true")
    ap.add_argument("--no-copy-mbench", dest="copy_mbench", action="store_false")
    ap.set_defaults(copy_mbench=True)

    ap.add_argument("--copy-overwrite", action="store_true")
    ap.add_argument("--copy-dry-run", action="store_true")

    ap.add_argument("--add-listings", dest="add_listings", action="store_true")
    ap.add_argument("--no-add-listings", dest="add_listings", action="store_false")
    ap.set_defaults(add_listings=True)

    ap.add_argument("--copy-disease", dest="copy_disease", action="store_true")
    ap.add_argument("--no-copy-disease", dest="copy_disease", action="store_false")
    ap.set_defaults(copy_disease=True)

    ap.add_argument("--copy-poslits", dest="copy_poslits", action="store_true")
    ap.add_argument("--no-copy-poslits", dest="copy_poslits", action="store_false")
    ap.set_defaults(copy_poslits=True)

    ap.add_argument("--copy-default-vars", dest="copy_default_vars", action="store_true")
    ap.add_argument("--no-copy-default-vars", dest="copy_default_vars", action="store_false")
    ap.set_defaults(copy_default_vars=True)

    ap.add_argument("--copy-projected-smt", dest="copy_projected_smt", action="store_true")
    ap.add_argument("--no-copy-projected-smt", dest="copy_projected_smt", action="store_false")
    ap.set_defaults(copy_projected_smt=True)

    ap.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS_JSONL)
    ap.add_argument("--queries-jsonl", type=Path, default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--disease-root", type=Path, default=DEFAULT_DISEASE_ROOT)
    ap.add_argument("--poslit-root", type=Path, default=DEFAULT_POSLIT_ROOT)
    ap.add_argument("--default-vars-root", type=Path, default=DEFAULT_DEFAULT_VARS_ROOT)
    ap.add_argument("--projected-smt-root", type=Path, default=DEFAULT_PROJECTED_SMT_ROOT)

    args = ap.parse_args()

    topk_list = parse_topk_list(args.topk_list)
    mode_filter = _parse_modes_arg(args.modes)

    print(f"[info] topk_list={topk_list}")
    print(f"[info] trialgpt_root_base={args.trialgpt_root_base}")
    print(f"[info] trialgpt_mbench_root_base={args.trialgpt_mbench_root_base}")
    print(f"[info] smt_root={args.smt_root}")
    print(f"[info] out_base={args.out_base}")

    all_rows: List[Dict[str, Any]] = []
    summary_all_k: List[Dict[str, Any]] = []
    summary_by_conflict_all_k: List[Dict[str, Any]] = []

    for k in topk_list:
        print("\n" + "#" * 100)
        print(f"[STAGE] top{k}")
        print("#" * 100)

        trialgpt_root = make_topk_root(args.trialgpt_root_base, k)
        trialgpt_mbench_root = make_topk_root(args.trialgpt_mbench_root_base, k)
        out_dir = make_topk_root(args.out_base, k)

        result = run_single_k(
            k=k,
            trialgpt_root=trialgpt_root,
            trialgpt_mbench_root=trialgpt_mbench_root,
            smt_root=args.smt_root,
            smt_mbench_root=args.smt_mbench_root,
            out_dir=out_dir,
            dedup_canonical=args.dedup_canonical,
            keep_subcohort_suffix=args.keep_subcohort_suffix,
            require_overlap=args.require_overlap,
            mode_filter=mode_filter,
            copy_mbench=args.copy_mbench,
            copy_overwrite=args.copy_overwrite,
            copy_dry_run=args.copy_dry_run,
            add_listings=args.add_listings,
            copy_disease=args.copy_disease,
            copy_poslits=args.copy_poslits,
            copy_default_vars=args.copy_default_vars,
            copy_projected_smt=args.copy_projected_smt,
            corpus_jsonl=args.corpus_jsonl,
            queries_jsonl=args.queries_jsonl,
            disease_root=args.disease_root,
            poslit_root=args.poslit_root,
            default_vars_root=args.default_vars_root,
            projected_smt_root=args.projected_smt_root,
        )

        all_rows.extend(result["rows"])

        for mode, cnt in sorted(result["summary_by_mode"].items()):
            summary_all_k.append({
                "topk": k,
                "mode": mode,
                "count": cnt,
                "pairs_evaluated": result["pairs_evaluated"],
            })

        for (mode, conflict), cnt in sorted(result["summary_by_mode_and_conflict"].items()):
            summary_by_conflict_all_k.append({
                "topk": k,
                "mode": mode,
                "trialgpt_conflict_category": conflict,
                "count": cnt,
            })

    agg_dir = Path(f"{args.out_base}_aggregate")
    agg_dir.mkdir(parents=True, exist_ok=True)

    agg_header = [
        "topk",
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
    write_csv(
        agg_dir / "smt_all_satisfied_trialgpt_not_topk_re_elig_all_k.csv",
        agg_header,
        all_rows,
    )
    write_csv(
        agg_dir / "summary_all_k.csv",
        ["topk", "mode", "count", "pairs_evaluated"],
        summary_all_k,
    )
    write_csv(
        agg_dir / "summary_by_conflict_all_k.csv",
        ["topk", "mode", "trialgpt_conflict_category", "count"],
        summary_by_conflict_all_k,
    )

    print("\n" + "=" * 100)
    print("[DONE] entire top-k sweep complete")
    print(f"[DONE] aggregate rows: {len(all_rows)}")
    print(f"[DONE] wrote: {agg_dir / 'smt_all_satisfied_trialgpt_not_topk_re_elig_all_k.csv'}")
    print(f"[DONE] wrote: {agg_dir / 'summary_all_k.csv'}")
    print(f"[DONE] wrote: {agg_dir / 'summary_by_conflict_all_k.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)