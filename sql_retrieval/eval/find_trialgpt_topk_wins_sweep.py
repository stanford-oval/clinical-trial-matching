#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_trialgpt_topk_wins_sweep.py

Sweep TrialGPT(top-K RE+EL) vs SMT all_satisfied for multiple fixed K values,
creating separate output folders for each K.

Default sweep:
  --trialgpt-ks 300,400,500

Outputs:
  <out>/top300/
  <out>/top400/
  <out>/top500/

Each subfolder contains the same artifacts as the old top-200 script:
  * trialgpt_topK_re_true_smt_not_all_satisfied.csv
  * summary.csv
  * summary_by_conflict.csv
  * copied_trialgpt_mbench/...
  * per-win smt_conflict_summary.json
  * per-win smt_conflict_explanation.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_TRIALGPT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_out_top200")
DEFAULT_SMT_ROOT = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_out")

DEFAULT_TRIALGPT_MBENCH_ROOT = Path("<SATIR_ROOT>/irsrc/eval/trialgpt_retrieval_eval_mbench_top200")
DEFAULT_SMT_MBENCH_ROOT = Path("<SATIR_ROOT>/irsrc/eval/smt_retrieval_eval_mbench")

DEFAULT_CORPUS_JSONL = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus.jsonl")
DEFAULT_QUERIES_JSONL = Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/queries.jsonl")

DEFAULT_DISEASE_ROOT = Path("<SATIR_ROOT>/build/disease_filtered_categorized")
DEFAULT_POSLIT_ROOT = Path("<SATIR_ROOT>/build/positive_constraint_literals_categorized")
DEFAULT_DEFAULT_VARS_ROOT = Path("<SATIR_ROOT>/build/default_vars")
DEFAULT_PROJECTED_SMT_ROOT = Path("<SATIR_ROOT>/build/canon_projection/_projected_smt")
DEFAULT_DB = Path("<SATIR_ROOT>/build/trial.db")

NCT_BASE_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

KNOWN_MODES = ("chief", "ccr", "all", "all-explore")

SENTINEL_NEG_INF = -1e15
SENTINEL_POS_INF = 1e15


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


def _parse_ks_arg(s: str) -> List[int]:
    vals: List[int] = []
    for tok in (s or "").split(","):
        t = tok.strip()
        if not t:
            continue
        k = int(t)
        if k <= 0:
            raise ValueError(f"All K values must be > 0, got {k}")
        vals.append(k)
    if not vals:
        raise ValueError("--trialgpt-ks produced no valid K values")
    return sorted(set(vals))


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


def important_table_for_wrapper_mode(wrapper_mode: str) -> str:
    wm = (wrapper_mode or "").strip().lower()
    if wm == "all-explore":
        wm = "all"
    if wm not in {"chief", "ccr", "all"}:
        raise ValueError(f"Unsupported wrapper mode for important table: {wrapper_mode!r}")
    return f"patient_inclusion_constraints_important_{wm}"


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


@dataclass
class SMTConflictMatch:
    conflict_label: str
    smt_label: Optional[str]
    smt_rank: Optional[int]
    act_tag: Optional[str]
    source_file: Optional[Path]
    subcohorts: List[str]


def choose_best_smt_conflict_for_trial(trial_key: str, smt_runs_for_pair: List[SMTRun]) -> SMTConflictMatch:
    priority = {"explicit_contradiction": 0, "unsatisfied_inclusion": 1}
    best: Optional[Tuple[int, int, SMTConflictMatch]] = None

    for run in smt_runs_for_pair:
        for it in run.items:
            if it.parent_trial_id != trial_key:
                continue
            if it.label not in {"explicit_contradiction", "unsatisfied_inclusion"}:
                continue
            pr = priority[it.label]
            rk = it.rank if isinstance(it.rank, int) else 10**9
            cand = SMTConflictMatch(
                conflict_label=it.label,
                smt_label=it.label,
                smt_rank=it.rank,
                act_tag=run.act_tag,
                source_file=run.source_file,
                subcohorts=list(it.subcohorts),
            )
            tup = (pr, rk, cand)
            if best is None or (tup[0], tup[1]) < (best[0], best[1]):
                best = tup

    if best is not None:
        return best[2]

    return SMTConflictMatch(
        conflict_label="no_hit",
        smt_label=None,
        smt_rank=None,
        act_tag=None,
        source_file=None,
        subcohorts=[],
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
# SQLite / clause helpers
# ----------------------------------------------------------------------

def _safe_ident(name: str) -> str:
    name = (name or "").strip()
    if not IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def table_has(cur: sqlite3.Cursor, table: str) -> bool:
    return bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())


def pick_col(cur: sqlite3.Cursor, table: str, candidates: List[str]) -> Optional[str]:
    rows = cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()
    have = {r[1] for r in rows}
    for c in candidates:
        if c in have:
            return c
    return None


def _materialize(cur: sqlite3.Cursor, name: str, select_sql: str, params: Dict[str, Any]) -> None:
    cur.execute(f"DROP TABLE IF EXISTS tmp_{name}")
    cur.execute(f"CREATE TEMP TABLE tmp_{name} AS {select_sql}", params)


def _scope_pred(alias: str, scope: str) -> str:
    if scope == "any":
        return ""
    return (
        f"AND COALESCE({alias}.tf_lb_hours, {SENTINEL_NEG_INF}) <= 0 "
        f"AND COALESCE({alias}.tf_ub_hours,  {SENTINEL_POS_INF}) >= 0"
    )


def _inc_cols(cur: sqlite3.Cursor, table: str, alias: str) -> Tuple[str, str]:
    try:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
    except sqlite3.Error:
        cols = set()
    lb = f"{alias}.tf_lb_inclusive" if "tf_lb_inclusive" in cols else "1"
    ub = f"{alias}.tf_ub_inclusive" if "tf_ub_inclusive" in cols else "1"
    return lb, ub


def _overlaps_pred_inc(
    fact_alias: str,
    fact_lb_inc_col: str,
    fact_ub_inc_col: str,
    lit_lb: str,
    lit_ub: str,
    lit_lb_inc_col: str,
    lit_ub_inc_col: str,
) -> str:
    flb = f"COALESCE({fact_alias}.tf_lb_hours, {SENTINEL_NEG_INF})"
    fub = f"COALESCE({fact_alias}.tf_ub_hours,  {SENTINEL_POS_INF})"
    llb = f"COALESCE({lit_lb}, {SENTINEL_NEG_INF})"
    lub = f"COALESCE({lit_ub}, {SENTINEL_POS_INF})"

    f_lb_inc = f"COALESCE({fact_alias}.{fact_lb_inc_col}, 1)"
    f_ub_inc = f"COALESCE({fact_alias}.{fact_ub_inc_col}, 1)"
    l_lb_inc = f"COALESCE({lit_lb_inc_col}, 1)"
    l_ub_inc = f"COALESCE({lit_ub_inc_col}, 1)"

    return (
        "("
        f"(({flb} < {lub}) OR ({flb} = {lub} AND {f_lb_inc}=1 AND {l_ub_inc}=1))"
        " AND "
        f"(({fub} > {llb}) OR ({fub} = {llb} AND {f_ub_inc}=1 AND {l_lb_inc}=1))"
        ")"
    )


def _find_trial_row_by_nct(conn: sqlite3.Connection, nct_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT id, nct_id, inclusion_trial_side_id, assumed_trial_side_id, exclusion_trial_side_id
        FROM trials
        WHERE UPPER(nct_id) = UPPER(?)
        LIMIT 1
        """,
        (nct_id,)
    ).fetchone()
    if row is None:
        return None
    return {
        "merged_trial_id": int(row[0]),
        "nct_id": row[1],
        "inclusion_trial_side_id": row[2],
        "assumed_trial_side_id": row[3],
        "exclusion_trial_side_id": row[4],
    }


def _format_num_bound(lb: Any, ub: Any, lb_inc: Any, ub_inc: Any) -> str:
    left = "[" if int(lb_inc or 0) == 1 else "("
    right = "]" if int(ub_inc or 0) == 1 else ")"
    lb_s = "-inf" if lb is None else str(lb)
    ub_s = "+inf" if ub is None else str(ub)
    return f"{left}{lb_s}, {ub_s}{right}"


def _format_time_bounds(lb: Any, ub: Any, lb_inc: Any, ub_inc: Any) -> str:
    if lb is None and ub is None:
        return ""
    left = "[" if int(lb_inc or 0) == 1 else "("
    right = "]" if int(ub_inc or 0) == 1 else ")"
    lb_s = "-inf" if lb is None else str(lb)
    ub_s = "+inf" if ub is None else str(ub)
    return f" @tf{left}{lb_s}, {ub_s}{right}"


def render_clause(conn: sqlite3.Connection, clause_id: int) -> Dict[str, Any]:
    cur = conn.cursor()

    lit_cols = {r[1] for r in cur.execute("PRAGMA table_info(constraint_clause_atoms)").fetchall()} if table_has(cur, "constraint_clause_atoms") else set()
    num_cols = {r[1] for r in cur.execute("PRAGMA table_info(constraint_clause_numeric_range)").fetchall()} if table_has(cur, "constraint_clause_numeric_range") else set()

    members: List[Dict[str, Any]] = []
    texts: List[str] = []

    if table_has(cur, "constraint_clause_atoms"):
        q = """
        SELECT literal_index, is_neg, base_var,
               tf_lb_hours, tf_ub_hours,
               {}
        FROM constraint_clause_atoms
        WHERE clause_id=?
        ORDER BY literal_index
        """.format(
            ", ".join([
                "tf_lb_inclusive" if "tf_lb_inclusive" in lit_cols else "1 AS tf_lb_inclusive",
                "tf_ub_inclusive" if "tf_ub_inclusive" in lit_cols else "1 AS tf_ub_inclusive",
            ])
        )
        for literal_index, is_neg, base_var, tf_lb, tf_ub, tf_lb_inc, tf_ub_inc in cur.execute(q, (clause_id,)).fetchall():
            time_txt = _format_time_bounds(tf_lb, tf_ub, tf_lb_inc, tf_ub_inc)
            txt = f"(not {base_var}){time_txt}" if int(is_neg or 0) == 1 else f"{base_var}{time_txt}"
            members.append({
                "kind": "bool",
                "literal_index": literal_index,
                "is_neg": int(is_neg or 0),
                "base_var": base_var,
                "tf_lb_hours": tf_lb,
                "tf_ub_hours": tf_ub,
                "tf_lb_inclusive": int(tf_lb_inc or 0),
                "tf_ub_inclusive": int(tf_ub_inc or 0),
                "text": txt,
            })
            texts.append(txt)

    if table_has(cur, "constraint_clause_numeric_range"):
        q = """
        SELECT member_index, base_var, lb, ub, lb_inc, ub_inc
        FROM constraint_clause_numeric_range
        WHERE clause_id=?
        ORDER BY member_index
        """
        for member_index, base_var, lb, ub, lb_inc, ub_inc in cur.execute(q, (clause_id,)).fetchall():
            rng = _format_num_bound(lb, ub, lb_inc, ub_inc)
            txt = f"{base_var} in {rng}"
            members.append({
                "kind": "num",
                "member_index": member_index,
                "base_var": base_var,
                "lb": lb,
                "ub": ub,
                "lb_inc": int(lb_inc or 0),
                "ub_inc": int(ub_inc or 0),
                "text": txt,
            })
            texts.append(txt)

    clause_text = "(or " + " ; ".join(texts) + ")" if texts else f"(clause {clause_id})"

    return {
        "clause_id": int(clause_id),
        "clause_text": clause_text,
        "members": members,
    }


def locate_explicit_contradiction_constraint_clauses(
    conn: sqlite3.Connection,
    patient_id: str,
    subcohort_id: str,
    scope: str = "any",
) -> List[Dict[str, Any]]:
    trial = _find_trial_row_by_nct(conn, subcohort_id)
    if trial is None:
        return [{
            "subcohort_id": subcohort_id,
            "status": "missing_trial_row",
            "details": f"trials.nct_id not found for {subcohort_id}",
        }]

    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS tmp_candidates")
    cur.execute("CREATE TEMP TABLE tmp_candidates(id INTEGER PRIMARY KEY)")
    cur.execute("INSERT INTO tmp_candidates(id) VALUES (?)", (trial["merged_trial_id"],))

    keb_scope = _scope_pred("keb", scope)
    cl_lb_inc, cl_ub_inc = _inc_cols(cur, "constraint_clause_atoms", alias="cl")

    demo_bool = """
      SELECT d.patient_id, ('patient_sex_is_'||d.sex) AS base_var, d.tf_token, 1.0 AS value,
             d.tf_lb_hours, d.tf_ub_hours, d.tf_lb_inclusive, d.tf_ub_inclusive
      FROM patient_demographic_constraints d WHERE d.sex IS NOT NULL
    """
    demo_num = """
      SELECT patient_id, 'patient_age_value_recorded_in_years'  AS base_var, tf_token, age_years  AS value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_demographic_constraints WHERE age_years  IS NOT NULL
      UNION ALL
      SELECT patient_id, 'patient_age_value_recorded_in_months' AS base_var, tf_token, age_months AS value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_demographic_constraints WHERE age_months IS NOT NULL
      UNION ALL
      SELECT patient_id, 'patient_age_value_recorded_in_days'   AS base_var, tf_token, age_days   AS value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_demographic_constraints WHERE age_days   IS NOT NULL
    """
    ex_bool = """
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_exclusion_constraints WHERE kind='bool'
      UNION ALL
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM tmp_demo_bool
    """
    ex_num = """
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_exclusion_constraints WHERE kind='num'
      UNION ALL
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM tmp_demo_num
    """

    _materialize(cur, "demo_bool", demo_bool, {})
    _materialize(cur, "demo_num", demo_num, {})
    _materialize(cur, "ex_knowledge_bool", ex_bool, {})
    _materialize(cur, "ex_knowledge_num", ex_num, {})

    incl_constraint_clauses = """
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN tmp_candidates c ON c.id = mt.id
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
      UNION ALL
      SELECT mt.id AS merged_trial_id, tc2.clause_id
      FROM trials mt
      JOIN tmp_candidates c ON c.id = mt.id
      JOIN trial_constraint_sides ta ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
      JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
    """
    excl_constraint_clauses = """
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN tmp_candidates c ON c.id = mt.id
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.exclusion_trial_side_id
    """
    _materialize(cur, "inclusion_trial_constraint_clauses", incl_constraint_clauses, {})
    _materialize(cur, "exclusion_trial_constraint_clauses", excl_constraint_clauses, {})

    ikb = f"""
      SELECT itc.merged_trial_id, cl.clause_id,
             COUNT(DISTINCT CASE WHEN keb.patient_id IS NOT NULL THEN cl.literal_index END) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
      LEFT JOIN tmp_ex_knowledge_bool keb
        ON keb.patient_id=:patient_id
       AND keb.base_var=cl.base_var
       AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      GROUP BY itc.merged_trial_id, cl.clause_id
    """
    isb = f"""
      SELECT itc.merged_trial_id, cl.clause_id, COUNT(DISTINCT cl.literal_index) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
      JOIN tmp_ex_knowledge_bool keb
        ON keb.patient_id=:patient_id
       AND keb.base_var=cl.base_var
       AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      WHERE (cl.is_neg=0 AND CAST(keb.value AS NUMERIC)=1.0)
         OR (cl.is_neg=1 AND CAST(keb.value AS NUMERIC)=0.0)
      GROUP BY itc.merged_trial_id, cl.clause_id
    """

    ikn = """
      SELECT itc.merged_trial_id, cnr.clause_id,
             COUNT(DISTINCT CASE WHEN ken.patient_id IS NOT NULL THEN cnr.member_index END) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = itc.clause_id
      LEFT JOIN tmp_ex_knowledge_num ken
        ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      GROUP BY itc.merged_trial_id, cnr.clause_id
    """
    esn_overlap_val_pred = """
      (cnr.lb IS NULL OR CAST(ken.value AS NUMERIC) > cnr.lb OR (CAST(ken.value AS NUMERIC) = cnr.lb AND cnr.lb_inc=1))
      AND (cnr.ub IS NULL OR CAST(ken.value AS NUMERIC) < cnr.ub OR (CAST(ken.value AS NUMERIC) = cnr.ub AND cnr.ub_inc=1))
    """
    isn = f"""
      SELECT itc.merged_trial_id, cnr.clause_id, COUNT(DISTINCT cnr.member_index) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = itc.clause_id
      JOIN tmp_ex_knowledge_num ken
        ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      WHERE {esn_overlap_val_pred}
      GROUP BY itc.merged_trial_id, cnr.clause_id
    """

    _materialize(cur, "ikb", ikb, {"patient_id": patient_id})
    _materialize(cur, "isb", isb, {"patient_id": patient_id})
    _materialize(cur, "ikn", ikn, {"patient_id": patient_id})
    _materialize(cur, "isn", isn, {"patient_id": patient_id})

    inc_kvss = """
      SELECT itc.merged_trial_id, c.id AS clause_id, c.number_of_clause_members AS number_of_clause_members,
             COALESCE(ikb.n,0)+COALESCE(ikn.n,0) AS known_count,
             COALESCE(isb.n,0)+COALESCE(isn.n,0) AS sat_count
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clauses c ON c.id = itc.clause_id
      LEFT JOIN tmp_ikb ikb ON ikb.merged_trial_id=itc.merged_trial_id AND ikb.clause_id=itc.clause_id
      LEFT JOIN tmp_ikn ikn ON ikn.merged_trial_id=itc.merged_trial_id AND ikn.clause_id=itc.clause_id
      LEFT JOIN tmp_isb isb ON isb.merged_trial_id=itc.merged_trial_id AND isb.clause_id=itc.clause_id
      LEFT JOIN tmp_isn isn ON isn.merged_trial_id=itc.merged_trial_id AND isn.clause_id=itc.clause_id
    """
    _materialize(cur, "inc_known_vs_sat", inc_kvss, {})

    ekb = f"""
      SELECT etc.merged_trial_id, cl.clause_id,
             COUNT(DISTINCT CASE WHEN keb.patient_id IS NOT NULL THEN cl.literal_index END) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_atoms cl ON cl.clause_id = etc.clause_id
      LEFT JOIN tmp_ex_knowledge_bool keb
        ON keb.patient_id=:patient_id
       AND keb.base_var=cl.base_var
       AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      GROUP BY etc.merged_trial_id, cl.clause_id
    """
    esb = f"""
      SELECT etc.merged_trial_id, cl.clause_id, COUNT(DISTINCT cl.literal_index) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_atoms cl ON cl.clause_id = etc.clause_id
      JOIN tmp_ex_knowledge_bool keb
        ON keb.patient_id=:patient_id
       AND keb.base_var=cl.base_var
       AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      WHERE (cl.is_neg=0 AND CAST(keb.value AS NUMERIC)=1.0)
         OR (cl.is_neg=1 AND CAST(keb.value AS NUMERIC)=0.0)
      GROUP BY etc.merged_trial_id, cl.clause_id
    """
    ekn = """
      SELECT etc.merged_trial_id, cnr.clause_id,
             COUNT(DISTINCT CASE WHEN ken.patient_id IS NOT NULL THEN cnr.member_index END) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = etc.clause_id
      LEFT JOIN tmp_ex_knowledge_num ken
        ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      GROUP BY etc.merged_trial_id, cnr.clause_id
    """
    esn = f"""
      SELECT etc.merged_trial_id, cnr.clause_id, COUNT(DISTINCT cnr.member_index) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = etc.clause_id
      JOIN tmp_ex_knowledge_num ken
        ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      WHERE {esn_overlap_val_pred}
      GROUP BY etc.merged_trial_id, cnr.clause_id
    """

    _materialize(cur, "ekb", ekb, {"patient_id": patient_id})
    _materialize(cur, "esb", esb, {"patient_id": patient_id})
    _materialize(cur, "ekn", ekn, {"patient_id": patient_id})
    _materialize(cur, "esn", esn, {"patient_id": patient_id})

    exc_kvss = """
      SELECT etc.merged_trial_id, c.id AS clause_id, c.number_of_clause_members AS number_of_clause_members,
             COALESCE(ekb.n,0)+COALESCE(ekn.n,0) AS known_count,
             COALESCE(esb.n,0)+COALESCE(esn.n,0) AS sat_count
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clauses c ON c.id = etc.clause_id
      LEFT JOIN tmp_ekb ekb ON ekb.merged_trial_id=etc.merged_trial_id AND ekb.clause_id=etc.clause_id
      LEFT JOIN tmp_ekn ekn ON ekn.merged_trial_id=etc.merged_trial_id AND ekn.clause_id=etc.clause_id
      LEFT JOIN tmp_esb esb ON esb.merged_trial_id=etc.merged_trial_id AND esb.clause_id=etc.clause_id
      LEFT JOIN tmp_esn esn ON esn.merged_trial_id=etc.merged_trial_id AND esn.clause_id=etc.clause_id
    """
    _materialize(cur, "exc_known_vs_sat", exc_kvss, {})

    out: List[Dict[str, Any]] = []

    for side, tmp_name in [("inclusion", "tmp_inc_known_vs_sat"), ("exclusion", "tmp_exc_known_vs_sat")]:
        rows = cur.execute(
            f"""
            SELECT clause_id, number_of_clause_members, known_count, sat_count
            FROM {tmp_name}
            WHERE merged_trial_id=?
              AND number_of_clause_members > 0
              AND known_count = number_of_clause_members
              AND sat_count = 0
            ORDER BY clause_id
            """,
            (trial["merged_trial_id"],)
        ).fetchall()
        for clause_id, n_members, known_count, sat_count in rows:
            rendered = render_clause(conn, int(clause_id))
            out.append({
                "subcohort_id": subcohort_id,
                "merged_trial_id": trial["merged_trial_id"],
                "label_path": "explicit_contradiction",
                "reason_type": f"eliminate_{side}",
                "side": side,
                "clause_id": int(clause_id),
                "number_of_clause_members": int(n_members or 0),
                "known_count": int(known_count or 0),
                "sat_count": int(sat_count or 0),
                "clause_text": rendered["clause_text"],
                "members": rendered["members"],
            })

    if not out:
        out.append({
            "subcohort_id": subcohort_id,
            "merged_trial_id": trial["merged_trial_id"],
            "label_path": "explicit_contradiction",
            "status": "no_exact_clause_found",
            "details": "No eliminate() contradiction clause was recovered for this subcohort.",
        })

    return out


def locate_unsatisfied_inclusion_constraint_clauses(
    conn: sqlite3.Connection,
    patient_id: str,
    subcohort_id: str,
    important_table: str,
    scope: str = "any",
) -> List[Dict[str, Any]]:
    trial = _find_trial_row_by_nct(conn, subcohort_id)
    if trial is None:
        return [{
            "subcohort_id": subcohort_id,
            "status": "missing_trial_row",
            "details": f"trials.nct_id not found for {subcohort_id}",
        }]

    important_table = _safe_ident(important_table)
    cur = conn.cursor()

    if not table_has(cur, important_table):
        return [{
            "subcohort_id": subcohort_id,
            "status": "missing_important_table",
            "details": f"Missing table {important_table}",
        }]

    cur.execute("DROP TABLE IF EXISTS tmp_candidates_gap")
    cur.execute("CREATE TEMP TABLE tmp_candidates_gap(id INTEGER PRIMARY KEY)")
    cur.execute("INSERT INTO tmp_candidates_gap(id) VALUES (?)", (trial["merged_trial_id"],))

    trial_filter_join = "JOIN tmp_candidates_gap c ON c.id = mt.id"

    cur.execute("DROP TABLE IF EXISTS tmp_trials_base_gap")
    cur.execute(
        f"""
        CREATE TEMP TABLE tmp_trials_base_gap AS
        SELECT mt.id AS merged_trial_id
        FROM trials mt
        {trial_filter_join}
        """
    )

    kib_scope = _scope_pred("kib", scope)

    has_demo = table_has(cur, "patient_demographic_constraints")
    has_fi = table_has(cur, "patient_inclusion_constraints")
    has_fii = table_has(cur, important_table)

    demo_bool = (
        """
        SELECT d.patient_id, ('patient_sex_is_'||d.sex) AS base_var,
               d.tf_token, 1.0 AS value,
               d.tf_lb_hours, d.tf_ub_hours,
               d.tf_lb_inclusive, d.tf_ub_inclusive
        FROM patient_demographic_constraints d
        WHERE d.sex IS NOT NULL
        """
        if has_demo else
        """
        SELECT NULL AS patient_id, NULL AS base_var, NULL AS tf_token, NULL AS value,
               NULL AS tf_lb_hours, NULL AS tf_ub_hours,
               NULL AS tf_lb_inclusive, NULL AS tf_ub_inclusive
        WHERE 0
        """
    )
    _materialize(cur, "demo_bool", demo_bool, {})

    if has_fi:
        in_bool_fi_demo = """
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM patient_inclusion_constraints
          WHERE kind='bool'
          UNION ALL
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM tmp_demo_bool
        """
    else:
        in_bool_fi_demo = """
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM tmp_demo_bool
        """
    _materialize(cur, "in_bool_fi_demo", in_bool_fi_demo, {})

    if has_fi:
        patient_root_bool = """
          SELECT patient_id, base_var
          FROM patient_inclusion_constraints
          WHERE kind='bool' AND COALESCE(is_root,0)=1
          GROUP BY patient_id, base_var
        """
    else:
        patient_root_bool = """
          SELECT NULL AS patient_id, NULL AS base_var
          WHERE 0
        """
    _materialize(cur, "patient_root_bool", patient_root_bool, {})

    if has_fii:
        in_bool_fii_root = f"""
          SELECT fii.patient_id, fii.base_var, fii.tf_token, fii.value,
                 fii.tf_lb_hours, fii.tf_ub_hours,
                 fii.tf_lb_inclusive, fii.tf_ub_inclusive
          FROM {important_table} fii
          JOIN tmp_patient_root_bool pr
            ON pr.patient_id = fii.patient_id
           AND pr.base_var   = fii.base_var
          WHERE fii.kind='bool'
        """
        _materialize(cur, "in_bool_fii_root", in_bool_fii_root, {})
    else:
        cur.execute("DROP TABLE IF EXISTS tmp_in_bool_fii_root")
        cur.execute(
            """
            CREATE TEMP TABLE tmp_in_bool_fii_root AS
            SELECT NULL AS patient_id, NULL AS base_var, NULL AS tf_token, NULL AS value,
                   NULL AS tf_lb_hours, NULL AS tf_ub_hours,
                   NULL AS tf_lb_inclusive, NULL AS tf_ub_inclusive
            WHERE 0
            """
        )

    incl_constraint_clauses = f"""
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      {trial_filter_join}
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
      UNION ALL
      SELECT mt.id AS merged_trial_id, tc2.clause_id
      FROM trials mt
      {trial_filter_join}
      JOIN trial_constraint_sides ta
           ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
      JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
    """
    _materialize(cur, "inclusion_trial_constraint_clauses_gap", incl_constraint_clauses, {})

    cur.execute("DROP TABLE IF EXISTS tmp_inclusion_trial_constraint_clauses_gap_nonum")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_inclusion_trial_constraint_clauses_gap_nonum AS
        SELECT itc.merged_trial_id, itc.clause_id
        FROM tmp_inclusion_trial_constraint_clauses_gap itc
        WHERE NOT EXISTS (
          SELECT 1
          FROM constraint_clause_numeric_range cnr
          WHERE cnr.clause_id = itc.clause_id
        )
        """
    )

    cur.execute("DROP TABLE IF EXISTS tmp_clause_bool_size")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_clause_bool_size AS
        SELECT cl.clause_id,
               COUNT(DISTINCT cl.literal_index) AS n_bool_members
        FROM constraint_clause_atoms cl
        GROUP BY cl.clause_id
        """
    )

    has_laa = table_has(cur, "constraint_literal_alternatives")
    has_llm = table_has(cur, "constraint_lifted_atoms")
    lifted_source = "constraint_literal_alternatives" if has_laa else ("constraint_lifted_atoms" if has_llm else "")

    if lifted_source:
        hop_col = pick_col(cur, lifted_source, ["hop"])
        reason_col = pick_col(cur, lifted_source, ["reason"])

        hop_expr = f"x.{hop_col}" if hop_col else "NULL"
        reason_expr = f"x.{reason_col}" if reason_col else "NULL"

        lifted_sql = f"""
          SELECT
            x.trial_id,
            x.clause_id,
            x.literal_index,
            x.lifted_var,
            x.lifted_var_stem,
            x.base_var_stem,
            COALESCE(x.lifted_var_stem, x.base_var_stem) AS base_var_stem_key,
            COALESCE({hop_expr}, 0) AS hop,
            COALESCE({reason_expr}, 'self') AS reason
          FROM {_safe_ident(lifted_source)} x
        """

        semantic_sql = f"""
          WITH la AS (
            SELECT
              mt.id AS merged_trial_id,
              sub.trial_id,
              sub.clause_id,
              sub.literal_index,
              sub.base_var_stem,
              sub.lifted_var_stem,
              sub.base_var_stem_key,
              sub.hop,
              sub.reason
            FROM trials mt
            JOIN ({lifted_sql}) AS sub
              ON sub.trial_id IN (
                   mt.inclusion_trial_side_id,
                   mt.assumed_trial_side_id
                 )
          )
          SELECT
            itc.merged_trial_id,
            cl.clause_id,
            cl.literal_index,
            COALESCE(la.base_var_stem_key, cl.base_var) AS sem_base_var,
            CASE
              WHEN la.hop IS NOT NULL
                   AND la.hop > 0
                   AND COALESCE(la.reason, 'self') <> 'self'
              THEN 1 ELSE 0
            END AS sem_is_lifted
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
          LEFT JOIN la
            ON la.merged_trial_id = itc.merged_trial_id
           AND la.clause_id      = cl.clause_id
           AND la.literal_index  = cl.literal_index
        """
    else:
        semantic_sql = """
          SELECT
            itc.merged_trial_id,
            cl.clause_id,
            cl.literal_index,
            cl.base_var AS sem_base_var,
            0 AS sem_is_lifted
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
        """

    _materialize(cur, "literal_semantic_gap", semantic_sql, {})

    cl_lb_inc, cl_ub_inc = _inc_cols(cur, "constraint_clause_atoms", alias="cl")
    has_default = table_has(cur, "default_predicate_values")

    if has_default:
        canon_trial_nct = (
            "CASE WHEN t.nct_id LIKE 'NCT________%' "
            "THEN SUBSTR(t.nct_id, 1, 11) ELSE t.nct_id END"
        )
        canon_dv_nct = (
            "CASE WHEN dv.nct_id LIKE 'NCT________%' "
            "THEN SUBSTR(dv.nct_id, 1, 11) ELSE dv.nct_id END"
        )

    if has_default:
        ikb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          LEFT JOIN trials t
            ON t.id = itc.merged_trial_id
          LEFT JOIN default_predicate_values dv
            ON {canon_dv_nct} = {canon_trial_nct}
           AND dv.base_var     = sem.sem_base_var
           AND dv.kind         = 'inclusion'
          WHERE
              (
                sem.sem_is_lifted = 0
                AND (
                     kib_fd.patient_id IS NOT NULL
                  OR kib_root.patient_id IS NOT NULL
                  OR dv.base_var IS NOT NULL
                )
              )
           OR (
                sem.sem_is_lifted = 1
                AND (
                     kib_root.patient_id IS NOT NULL
                  OR dv.base_var IS NOT NULL
                )
              )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """
    else:
        ikb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          WHERE
              (
                sem.sem_is_lifted = 0
                AND (
                     kib_fd.patient_id IS NOT NULL
                  OR kib_root.patient_id IS NOT NULL
                )
              )
           OR (
                sem.sem_is_lifted = 1
                AND kib_root.patient_id IS NOT NULL
              )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """

    _materialize(cur, "gap_ikb", ikb, {"patient_id": patient_id})

    truthy_expr = lambda alias: (
        f"(CAST({alias}.value AS NUMERIC) = 1 "
        f"OR UPPER(CAST({alias}.value AS TEXT)) IN ('1','TRUE','T','Y','YES'))"
    )
    falsy_expr = lambda alias: (
        f"(CAST({alias}.value AS NUMERIC) = 0 "
        f"OR UPPER(CAST({alias}.value AS TEXT)) IN ('0','FALSE','F','N','NO'))"
    )

    if has_default:
        isb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          LEFT JOIN trials t
            ON t.id = itc.merged_trial_id
          LEFT JOIN default_predicate_values dv
            ON {canon_dv_nct} = {canon_trial_nct}
           AND dv.base_var     = sem.sem_base_var
           AND dv.kind         = 'inclusion'
          WHERE
            (
              sem.sem_is_lifted = 0 AND (
                (
                  kib_fd.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_fd')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_fd')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_root')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NULL
                  AND dv.base_var IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('dv')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('dv')})
                  )
                )
              )
            )
            OR
            (
              sem.sem_is_lifted = 1 AND (
                (
                  kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_root')})
                  )
                )
                OR
                (
                  kib_root.patient_id IS NULL
                  AND dv.base_var IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('dv')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('dv')})
                  )
                )
              )
            )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """
    else:
        isb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
              )} {kib_scope}
          WHERE
            (
              sem.sem_is_lifted = 0 AND (
                (
                  kib_fd.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_fd')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_fd')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_root')})
                  )
                )
              )
            )
            OR
            (
              sem.sem_is_lifted = 1 AND
              kib_root.patient_id IS NOT NULL AND
              (
                (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                OR
                (cl.is_neg = 1 AND {falsy_expr('kib_root')})
              )
            )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """

    _materialize(cur, "gap_isb", isb, {"patient_id": patient_id})

    inc_kvss = """
      SELECT
        itc.merged_trial_id,
        c.id AS clause_id,
        COALESCE(cbs.n_bool_members, 0) AS number_of_clause_members,
        COALESCE(ikb.n, 0) AS known_count,
        COALESCE(isb.n, 0) AS sat_count
      FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
      JOIN constraint_clauses c
        ON c.id = itc.clause_id
      LEFT JOIN tmp_clause_bool_size cbs
        ON cbs.clause_id = itc.clause_id
      LEFT JOIN tmp_gap_ikb ikb
        ON ikb.merged_trial_id = itc.merged_trial_id
       AND ikb.clause_id       = itc.clause_id
      LEFT JOIN tmp_gap_isb isb
        ON isb.merged_trial_id = itc.merged_trial_id
       AND isb.clause_id       = itc.clause_id
    """
    _materialize(cur, "inc_known_vs_sat_gap", inc_kvss, {})

    rows = cur.execute(
        """
        SELECT clause_id, number_of_clause_members, known_count, sat_count
        FROM tmp_inc_known_vs_sat_gap
        WHERE merged_trial_id=?
          AND number_of_clause_members > 0
          AND known_count = number_of_clause_members
          AND sat_count = 0
        ORDER BY clause_id
        """,
        (trial["merged_trial_id"],)
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for clause_id, n_members, known_count, sat_count in rows:
        rendered = render_clause(conn, int(clause_id))
        out.append({
            "subcohort_id": subcohort_id,
            "merged_trial_id": trial["merged_trial_id"],
            "label_path": "unsatisfied_inclusion",
            "reason_type": "inclusion_gap",
            "side": "inclusion",
            "clause_id": int(clause_id),
            "number_of_clause_members": int(n_members or 0),
            "known_count": int(known_count or 0),
            "sat_count": int(sat_count or 0),
            "clause_text": rendered["clause_text"],
            "members": rendered["members"],
        })

    if not out:
        out.append({
            "subcohort_id": subcohort_id,
            "merged_trial_id": trial["merged_trial_id"],
            "label_path": "unsatisfied_inclusion",
            "status": "no_exact_clause_found",
            "details": "No violated evaluable inclusion BOOL clause was recovered for this subcohort.",
        })

    return out


def locate_clause_evidence_for_subcohort(
    conn: sqlite3.Connection,
    *,
    patient_id: str,
    wrapper_mode: str,
    conflict_label: str,
    subcohort_id: str,
    scope: str = "any",
) -> List[Dict[str, Any]]:
    if conflict_label == "no_hit":
        return [{
            "subcohort_id": subcohort_id,
            "status": "not_applicable",
            "details": "No SMT negative label matched this canonical trial.",
        }]

    if conflict_label == "explicit_contradiction":
        return locate_explicit_contradiction_constraint_clauses(
            conn,
            patient_id=patient_id,
            subcohort_id=subcohort_id,
            scope=scope,
        )

    if conflict_label == "unsatisfied_inclusion":
        important_table = important_table_for_wrapper_mode(wrapper_mode)
        return locate_unsatisfied_inclusion_constraint_clauses(
            conn,
            patient_id=patient_id,
            subcohort_id=subcohort_id,
            important_table=important_table,
            scope=scope,
        )

    return [{
        "subcohort_id": subcohort_id,
        "status": "unsupported_conflict_label",
        "details": conflict_label,
    }]


# ----------------------------------------------------------------------
# Explanation text
# ----------------------------------------------------------------------

def build_conflict_explanation(
    *,
    patient_id: str,
    mode: str,
    trial_id: str,
    tg_rank: int,
    tg_k: int,
    conflict: SMTConflictMatch,
    clause_evidence: List[Dict[str, Any]],
) -> str:
    lines = [
        "TrialGPT top-K win with SMT-side conflict annotation",
        f"patient_id: {patient_id}",
        f"mode: {mode}",
        f"trial_id: {trial_id}",
        f"trialgpt_rank: {tg_rank}",
        f"trialgpt_fixed_k_used: {tg_k}",
        "",
        "Why this is still a win:",
        "- TrialGPT marked this trial as relevant=True and eligible=True.",
        "- This trial was within the fixed top-K cutoff for TrialGPT.",
        "- SMT did NOT have label == all_satisfied for this trial.",
        "",
        f"SMT conflict category: {conflict.conflict_label}",
    ]

    if conflict.conflict_label == "no_hit":
        lines += [
            "- Explanation: no SMT negative-label match was found for this canonical trial.",
            "- So the trial is categorized as no_hit.",
        ]
    else:
        lines += [
            f"- SMT negative label: {conflict.smt_label}",
            f"- SMT rank: {conflict.smt_rank}",
            f"- SMT act_tag: {conflict.act_tag}",
            f"- SMT source_file: {conflict.source_file}",
            f"- SMT matched/inferred subcohorts: {', '.join(conflict.subcohorts) if conflict.subcohorts else '(none listed)'}",
            "",
            "Exact SMT clause evidence:",
        ]
        if not clause_evidence:
            lines.append("- No clause evidence recovered.")
        else:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for ev in clause_evidence:
                grouped.setdefault(str(ev.get("subcohort_id", "(unknown)")), []).append(ev)

            for sub in sorted(grouped):
                lines.append("")
                lines.append(f"Subcohort: {sub}")
                for ev in grouped[sub]:
                    if ev.get("status"):
                        lines.append(f"  - status: {ev['status']}")
                        if ev.get("details"):
                            lines.append(f"    details: {ev['details']}")
                        continue

                    lines.append(
                        f"  - clause_id={ev.get('clause_id')} "
                        f"side={ev.get('side')} "
                        f"reason_type={ev.get('reason_type')} "
                        f"known={ev.get('known_count')}/{ev.get('number_of_clause_members')} "
                        f"sat={ev.get('sat_count')}"
                    )
                    lines.append(f"    clause: {ev.get('clause_text')}")
                    members = ev.get("members")
                    if isinstance(members, list):
                        for m in members:
                            if isinstance(m, dict) and m.get("text"):
                                lines.append(f"      * {m['text']}")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Per-K runner
# ----------------------------------------------------------------------

def run_for_k(
    *,
    k: int,
    args: argparse.Namespace,
    tg_runs: Dict[Tuple[str, str], TrialGPTRun],
    smt_runs: Dict[Tuple[str, str, str], SMTRun],
    corpus_index: Dict[str, Dict[str, Any]],
    query_index: Dict[str, Dict[str, Any]],
) -> None:
    out_dir = args.out / f"top{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    smt_pair_index: Dict[Tuple[str, str], List[SMTRun]] = {}
    for _, run in smt_runs.items():
        smt_pair_index.setdefault((run.patient_id, run.mode), []).append(run)

    keys_tg = set(tg_runs.keys())
    keys_smt = set(smt_pair_index.keys())
    keys_all = sorted(keys_tg & keys_smt) if args.require_overlap else sorted(keys_tg | keys_smt)

    rows: List[Dict[str, Any]] = []
    summary_by_mode: Dict[str, int] = {}
    summary_by_mode_and_conflict: Dict[Tuple[str, str], int] = {}
    copy_targets: List[Dict[str, Any]] = []

    pairs_eval = 0

    for (patient_id, mode) in keys_all:
        tg = tg_runs.get((patient_id, mode))
        smt_runs_for_pair = smt_pair_index.get((patient_id, mode), [])

        if tg is None:
            continue
        if args.require_overlap and not smt_runs_for_pair:
            continue

        pairs_eval += 1

        tg_top = tg.items[:k]

        smt_all_satisfied_keys: Set[str] = set()
        for smt_run in smt_runs_for_pair:
            for x in smt_run.items:
                if x.label == "all_satisfied":
                    smt_all_satisfied_keys.add(x.parent_trial_id)

        smt_all_satisfied_count = len(smt_all_satisfied_keys)

        for it in tg_top:
            if not (it.relevant is True and it.eligible is True):
                continue
            if it.key in smt_all_satisfied_keys:
                continue

            conflict = choose_best_smt_conflict_for_trial(it.key, smt_runs_for_pair)

            rows.append({
                "patient_id": patient_id,
                "mode": mode,
                "trial_id": it.key,
                "trialgpt_raw_trial_id": it.raw_trial_id,
                "trialgpt_rank": it.rank,
                "trialgpt_relevant": it.relevant,
                "trialgpt_eligible": it.eligible,
                "trialgpt_fixed_k_used": k,
                "smt_conflict_category": conflict.conflict_label,
                "smt_rank": conflict.smt_rank if conflict.smt_rank is not None else "",
                "smt_label": conflict.smt_label or "",
                "smt_act_tag": conflict.act_tag or "",
                "smt_num_subcohorts": len(conflict.subcohorts),
                "smt_subcohorts": "|".join(conflict.subcohorts),
                "smt_all_satisfied_count": smt_all_satisfied_count,
                "reason": "smt_missing_all_satisfied",
                "trialgpt_file": str(tg.source_file),
                "smt_file": str(conflict.source_file) if conflict.source_file else "",
            })

            summary_by_mode[mode] = summary_by_mode.get(mode, 0) + 1
            summary_by_mode_and_conflict[(mode, conflict.conflict_label)] = (
                summary_by_mode_and_conflict.get((mode, conflict.conflict_label), 0) + 1
            )

            if args.copy_mbench:
                copy_targets.append({
                    "patient_id": patient_id,
                    "mode": mode,
                    "trial_id": it.key,
                    "trialgpt_raw_trial_id": it.raw_trial_id,
                    "trialgpt_rank": it.rank,
                    "trialgpt_k": k,
                    "conflict_label": conflict.conflict_label,
                    "smt_label": conflict.smt_label,
                    "smt_rank": conflict.smt_rank,
                    "smt_act_tag": conflict.act_tag,
                    "smt_file": str(conflict.source_file) if conflict.source_file else "",
                    "smt_subcohorts": list(conflict.subcohorts),
                })

    header = [
        "patient_id", "mode",
        "trial_id", "trialgpt_raw_trial_id",
        "trialgpt_rank", "trialgpt_relevant", "trialgpt_eligible",
        "trialgpt_fixed_k_used",
        "smt_conflict_category", "smt_rank", "smt_label", "smt_act_tag",
        "smt_num_subcohorts", "smt_subcohorts",
        "smt_all_satisfied_count",
        "reason", "trialgpt_file", "smt_file",
    ]
    write_csv(out_dir / f"trialgpt_top{k}_re_true_smt_not_all_satisfied.csv", header, rows)
    write_csv(
        out_dir / "summary.csv",
        ["mode", "count"],
        [{"mode": m, "count": cnt} for (m, cnt) in sorted(summary_by_mode.items())]
    )
    write_csv(
        out_dir / "summary_by_conflict.csv",
        ["mode", "smt_conflict_category", "count"],
        [{"mode": m, "smt_conflict_category": c, "count": cnt}
         for ((m, c), cnt) in sorted(summary_by_mode_and_conflict.items())]
    )

    print(f"[ok][top{k}] wrote {out_dir / f'trialgpt_top{k}_re_true_smt_not_all_satisfied.csv'}")
    print(f"[ok][top{k}] wrote {out_dir / 'summary.csv'}")
    print(f"[ok][top{k}] wrote {out_dir / 'summary_by_conflict.csv'}")
    print(f"[info][top{k}] pairs_evaluated={pairs_eval} wins={len(rows)}")

    if not args.copy_mbench:
        return

    copied_dirs = 0
    skipped_dirs = 0
    missing_dirs = 0
    copied_files = 0
    skipped_files = 0
    missing_files = 0

    uniq: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    priority = {"explicit_contradiction": 0, "unsatisfied_inclusion": 1, "no_hit": 2}

    for x in copy_targets:
        key = (x["mode"], x["patient_id"], x["trial_id"])
        prev = uniq.get(key)
        if prev is None:
            uniq[key] = x
        else:
            merged_subs = set(prev.get("smt_subcohorts", []))
            merged_subs.update(x.get("smt_subcohorts", []))
            prev["smt_subcohorts"] = sorted(merged_subs)

            if priority.get(x.get("conflict_label", "no_hit"), 99) < priority.get(prev.get("conflict_label", "no_hit"), 99):
                prev["conflict_label"] = x.get("conflict_label")
                prev["smt_label"] = x.get("smt_label")
                prev["smt_rank"] = x.get("smt_rank")
                prev["smt_act_tag"] = x.get("smt_act_tag")
                prev["smt_file"] = x.get("smt_file")

    print(f"[info][top{k}] unique trial dirs to copy={len(uniq)}")

    conn = sqlite3.connect(str(args.db))

    for _, meta in sorted(uniq.items(), key=lambda kv: kv[0]):
        mode = meta["mode"]
        patient_id = meta["patient_id"]
        trial_id = meta["trial_id"]
        raw_trial_id = meta["trialgpt_raw_trial_id"]
        trialgpt_rank = meta["trialgpt_rank"]
        trialgpt_k = meta["trialgpt_k"]
        conflict_label = meta.get("conflict_label", "no_hit")
        smt_label = meta.get("smt_label")
        smt_rank = meta.get("smt_rank")
        smt_act_tag = meta.get("smt_act_tag")
        smt_file = meta.get("smt_file", "")
        smt_subs = list(meta.get("smt_subcohorts", []))

        base_dst = out_dir / "copied_trialgpt_mbench" / mode / patient_id / trial_id

        src_tg = args.trialgpt_mbench_root / mode / patient_id / trial_id
        ok, msg = copy_dir(src_tg, base_dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
        print(f"[trialgpt_mbench:{msg}][top{k}] {src_tg} -> {base_dst}")

        if not ok and msg == "missing_src":
            missing_dirs += 1
            continue
        elif ok and msg == "exists_skip":
            skipped_dirs += 1
        elif ok and msg in {"copied", "would_copy"}:
            copied_dirs += 1

        if args.add_listings:
            patient_obj = query_index.get(patient_id)
            trial_obj = corpus_index.get(trial_id) or corpus_index.get(raw_trial_id)

            if args.copy_dry_run:
                print(f"[would_write][top{k}] {base_dst / 'patient_query.json'}")
                print(f"[would_write][top{k}] {base_dst / 'trial_listing.json'}")
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
            smt_mbench_root=args.smt_mbench_root,
            disease_root=args.disease_root,
            poslit_root=args.poslit_root,
            default_vars_root=args.default_vars_root,
            projected_smt_root=args.projected_smt_root,
        )

        if not subs:
            subs = [trial_id]

        print(f"[info][top{k}] artifact subcohorts for {mode}/{patient_id}/{trial_id}: {subs}")

        if args.add_listings and not args.copy_dry_run:
            for sub in subs:
                sub_obj = corpus_index.get(sub)
                if sub_obj is not None:
                    write_json(base_dst / f"trial_listing__{sub}.json", sub_obj)

        for sub in subs:
            src = resolve_smt_mbench_src(
                smt_mbench_root=args.smt_mbench_root,
                mode=mode,
                patient_id=patient_id,
                trial_id=trial_id,
                sub=sub,
            )
            dst = base_dst / "smt_mbench" / sub
            if src is None:
                missing_dirs += 1
                print(f"[missing_smt_mbench][top{k}] no source found for mode={mode} patient={patient_id} trial={trial_id} sub={sub}")
            else:
                ok2, msg2 = copy_dir(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                if not ok2 and msg2 == "missing_src":
                    missing_dirs += 1
                    print(f"[missing_smt_mbench][top{k}] {src}")
                elif ok2 and msg2 == "exists_skip":
                    skipped_dirs += 1
                elif ok2 and msg2 in {"copied", "would_copy"}:
                    copied_dirs += 1
                print(f"[smt_mbench:{msg2}][top{k}] {src} -> {dst}")

        if args.copy_disease:
            for sub in subs:
                src = args.disease_root / f"{sub}_disease_link_filter_summary.json"
                dst = base_dst / "disease" / src.name
                ok3, msg3 = copy_file(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                if not ok3 and msg3 == "missing_src":
                    missing_files += 1
                    print(f"[missing_disease][top{k}] {src}")
                elif ok3 and msg3 == "exists_skip":
                    skipped_files += 1
                elif ok3 and msg3 in {"copied", "would_copy"}:
                    copied_files += 1
                print(f"[disease:{msg3}][top{k}] {src} -> {dst}")

        if args.copy_poslits:
            per_file = args.poslit_root / "per_file"
            for sub in subs:
                matches = sorted(per_file.glob(f"{sub}_*program*.smt2.json"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_poslits][top{k}] {per_file} glob={sub}_*program*.smt2.json")
                    continue
                for src in matches:
                    dst = base_dst / "positive_constraint_literals" / src.name
                    ok4, msg4 = copy_file(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                    if not ok4 and msg4 == "missing_src":
                        missing_files += 1
                        print(f"[missing_poslit][top{k}] {src}")
                    elif ok4 and msg4 == "exists_skip":
                        skipped_files += 1
                    elif ok4 and msg4 in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[poslit:{msg4}][top{k}] {src} -> {dst}")

        if args.copy_default_vars:
            for sub in subs:
                matches = sorted(args.default_vars_root.glob(f"{sub}_*.json"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_default_vars][top{k}] {args.default_vars_root} glob={sub}_*.json")
                    continue
                for src in matches:
                    dst = base_dst / "default_vars" / src.name
                    ok5, msg5 = copy_file(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                    if not ok5 and msg5 == "missing_src":
                        missing_files += 1
                        print(f"[missing_default_var][top{k}] {src}")
                    elif ok5 and msg5 == "exists_skip":
                        skipped_files += 1
                    elif ok5 and msg5 in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[default_vars:{msg5}][top{k}] {src} -> {dst}")

        if args.copy_projected_smt:
            for sub in subs:
                matches = sorted(args.projected_smt_root.glob(f"{sub}_*.smt2"))
                if not matches:
                    missing_files += 1
                    print(f"[missing_projected_smt][top{k}] {args.projected_smt_root} glob={sub}_*.smt2")
                    continue
                for src in matches:
                    dst = base_dst / "projected_smt" / src.name
                    ok6, msg6 = copy_file(src, dst, overwrite=args.copy_overwrite, dry_run=args.copy_dry_run)
                    if not ok6 and msg6 == "missing_src":
                        missing_files += 1
                        print(f"[missing_projected_smt_file][top{k}] {src}")
                    elif ok6 and msg6 == "exists_skip":
                        skipped_files += 1
                    elif ok6 and msg6 in {"copied", "would_copy"}:
                        copied_files += 1
                    print(f"[projected_smt:{msg6}][top{k}] {src} -> {dst}")

        clause_evidence: List[Dict[str, Any]] = []
        if conflict_label in {"explicit_contradiction", "unsatisfied_inclusion"}:
            for sub in subs:
                try:
                    ev = locate_clause_evidence_for_subcohort(
                        conn,
                        patient_id=patient_id,
                        wrapper_mode=mode,
                        conflict_label=conflict_label,
                        subcohort_id=sub,
                        scope=args.scope,
                    )
                    clause_evidence.extend(ev)
                except Exception as e:
                    clause_evidence.append({
                        "subcohort_id": sub,
                        "status": "exception",
                        "details": f"{type(e).__name__}: {e}",
                    })

        conflict_obj = {
            "patient_id": patient_id,
            "mode": mode,
            "trial_id": trial_id,
            "trialgpt_raw_trial_id": raw_trial_id,
            "trialgpt_rank": trialgpt_rank,
            "trialgpt_fixed_k_used": trialgpt_k,
            "smt_conflict_category": conflict_label,
            "smt_label": smt_label,
            "smt_rank": smt_rank,
            "smt_act_tag": smt_act_tag,
            "smt_file": smt_file,
            "smt_subcohorts": smt_subs,
            "artifact_subcohorts": subs,
            "scope_for_clause_recovery": args.scope,
            "clause_evidence": clause_evidence,
            "explanation": (
                "no_hit"
                if conflict_label == "no_hit"
                else "matched SMT negative label for this canonical trial and recovered clause-level evidence when possible"
            ),
        }

        explanation = build_conflict_explanation(
            patient_id=patient_id,
            mode=mode,
            trial_id=trial_id,
            tg_rank=trialgpt_rank,
            tg_k=trialgpt_k,
            conflict=SMTConflictMatch(
                conflict_label=conflict_label,
                smt_label=smt_label,
                smt_rank=smt_rank,
                act_tag=smt_act_tag,
                source_file=Path(smt_file) if smt_file else None,
                subcohorts=subs,
            ),
            clause_evidence=clause_evidence,
        )

        if args.copy_dry_run:
            print(f"[would_write][top{k}] {base_dst / 'smt_conflict_summary.json'}")
            print(f"[would_write][top{k}] {base_dst / 'smt_conflict_explanation.txt'}")
        else:
            write_json(base_dst / "smt_conflict_summary.json", conflict_obj)
            write_text(base_dst / "smt_conflict_explanation.txt", explanation)

    conn.close()

    print(
        f"[done][top{k}] dirs copied={copied_dirs} skipped={skipped_dirs} missing={missing_dirs} | "
        f"files copied={copied_files} skipped={skipped_files} missing={missing_files}"
    )
    if args.copy_dry_run:
        print(f"[info][top{k}] copy-dry-run enabled: no files were copied / written")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep TrialGPT fixed-topK wins against SMT all_satisfied, add SMT conflict explanations, "
                    "copy artifacts, and recover exact offending SMT constraint_clauses."
    )
    ap.add_argument("--trialgpt-root", type=Path, default=DEFAULT_TRIALGPT_ROOT)
    ap.add_argument("--smt-root", type=Path, default=DEFAULT_SMT_ROOT)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=Path("./trialgpt_topk_wins_out"))
    ap.add_argument("--trialgpt-ks", type=str, default="300,400,500",
                    help="Comma-separated fixed K values, e.g. 300,400,500")

    ap.add_argument("--dedup-canonical", action="store_true")
    ap.add_argument("--keep-subcohort-suffix", action="store_true")
    ap.add_argument("--require-overlap", action="store_true")
    ap.add_argument("--modes", type=str, default="")
    ap.add_argument("--scope", choices=["now", "any"], default="any",
                    help="Scope used when reconstructing exact SMT clause evidence.")

    ap.add_argument("--copy-mbench", dest="copy_mbench", action="store_true")
    ap.add_argument("--no-copy-mbench", dest="copy_mbench", action="store_false")
    ap.set_defaults(copy_mbench=True)

    ap.add_argument("--trialgpt-mbench-root", type=Path, default=DEFAULT_TRIALGPT_MBENCH_ROOT)
    ap.add_argument("--smt-mbench-root", type=Path, default=DEFAULT_SMT_MBENCH_ROOT)

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

    ks = _parse_ks_arg(args.trialgpt_ks)

    if not args.trialgpt_root.exists():
        print(f"[error] trialgpt root not found: {args.trialgpt_root}", file=sys.stderr)
        sys.exit(2)
    if not args.smt_root.exists():
        print(f"[error] smt root not found: {args.smt_root}", file=sys.stderr)
        sys.exit(2)
    if not args.db.exists():
        print(f"[error] db not found: {args.db}", file=sys.stderr)
        sys.exit(2)

    if args.copy_mbench:
        if not args.trialgpt_mbench_root.exists():
            print(f"[error] trialgpt mbench root not found: {args.trialgpt_mbench_root}", file=sys.stderr)
            sys.exit(2)
        if not args.smt_mbench_root.exists():
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

    if args.copy_default_vars and not args.default_vars_root.exists():
        print(f"[error] default vars root not found: {args.default_vars_root}", file=sys.stderr)
        sys.exit(2)

    if args.copy_projected_smt and not args.projected_smt_root.exists():
        print(f"[error] projected smt root not found: {args.projected_smt_root}", file=sys.stderr)
        sys.exit(2)

    collapse = not args.keep_subcohort_suffix
    mode_filter = _parse_modes_arg(args.modes)

    tg_runs = collect_trialgpt_runs(args.trialgpt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)
    smt_runs = collect_smt_runs(args.smt_root, collapse_subsuffix=collapse, dedup=args.dedup_canonical)

    if mode_filter is not None:
        tg_runs = {k: v for k, v in tg_runs.items() if k[1] in mode_filter}
        smt_runs = {k: v for k, v in smt_runs.items() if k[1] in mode_filter}

    corpus_index: Dict[str, Dict[str, Any]] = {}
    query_index: Dict[str, Dict[str, Any]] = {}
    if args.add_listings:
        corpus_index = load_jsonl_index(args.corpus_jsonl, id_key="_id")
        query_index = load_jsonl_index(args.queries_jsonl, id_key="_id")

    print(f"[info] K sweep = {ks}")
    for k in ks:
        run_for_k(
            k=k,
            args=args,
            tg_runs=tg_runs,
            smt_runs=smt_runs,
            corpus_index=corpus_index,
            query_index=query_index,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)