#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_smt_retrieval_eval.py

Run (LLM-judge) relevance -> eligibility for trials listed in patient_labels/*.json.

Features:
- Sample N patients from a patient_labels directory
- Expand parent trial_id to subcohort ids (if present in corpus.jsonl as _id prefix matches)
- Run relevance then eligibility for each (patient_id, subcohort_trial_id)
- Aggregate per parent trial_id:
    any_subcohort_relevant
    any_subcohort_eligible
    any_subcohort_relevant_and_eligible
- Write augmented patient label json to output dir
- Keep fine-grained intermediate outputs + cache under mbench_root

Compatibility fixes:
- Some repos define prompt-path constants as `str` but judge code does `.open()`.
  We normalize/monkeypatch those constants to `pathlib.Path` early.
- PairDiskCache.compute_meta() signature differs across repo versions.
  We adapt by inspecting the signature and passing only supported fields.
- In THIS repo, cache_utils.sha256_file expects Path; so we pass Path objects into compute_meta
  for relevance_prompt_path / eligibility_prompt_path, not str.

Debuggability fixes:
- Print full traceback on errors (so we can see where `.open()` on str is happening).
- Optional --debug-prompts prints the prompt constant types and resolved paths.

Progress bar:
- Uses tqdm if installed; otherwise falls back to a simple text progress counter.

UPDATED (2026-02-25):
- Fix eligibility parsing for NEW prompt output that returns a JSON LIST of dicts, e.g.
  <subcohort_eligibility_decisions>
  [
    {"eligibility_decision":"ineligible", ...}
  ]
  </subcohort_eligibility_decisions>
  Previously, naive list parsing treated any non-empty list as True; now it is True iff ANY dict says eligible.

UPDATED (2026-02-25, relevance tag fix):
- Fix relevance parsing for outputs that contain:
    <relevant_subcohorts>
      [ { "subcohort_name": "...", ... }, ... ]
    </relevant_subcohorts>
  Semantics for this tag are authoritative:
    - [] => False
    - any dict entry with a non-empty "subcohort_name" => True
  (fallback to legacy list semantics if schema differs)

UPDATED (2026-02-25, skip eligibility when not relevant):
- If relevance is confidently False, skip eligibility call entirely and emit a deterministic
  eligibility output that parses to False (via narrative-only rules).

UPDATED (2026-02-25, prompt bundling):
- Optional --dump-prompts-per-pair saves prompt templates + manifest.json into each
  per-pair mbench directory: <pair_dir>/_prompt_bundle/

UPDATED (2026-02-27, Azure param constraint fix):
- Some Azure deployments reject temperature != default(1). We set engine default_temperature=1.0
  so the SDK request is accepted. (This is the minimal fix for your observed error.)

UPDATED:
- --run-modes supports:
    {both, ccr, all, all-explore}

UPDATED (2026-02-28, specific instructions prompt factoring):
- Base relevance prompt now contains a #SPECIFIC_INSTRUCTIONS# placeholder.
  We supply per-mode instruction prompt files:
    - ./prompts/relevance_instructions_all.prompt
    - ./prompts/relevance_instructions_ccr.prompt
- We monkeypatch jr.RELEVANCE_INSTRUCTIONS_PROMPT_PATHS into judge_relevance module
  (if it doesn't exist), normalize paths to Path, and include this in prompt bundle + cache meta.

UPDATED (2026-03-03, SHARED CACHE):
- Add --cache-root (shared PairDiskCache dir) so this script and TrialGPT script can reuse results.
  Correctness is ensured by cache_key + expected_meta prompt_sha/model params.
  If prompt templates change, expected_meta changes and cache entries are treated as stale (no need to wipe).

NEW (2026-03-08, prompt-update rerun policy):
- --prompt-update-rerun {all,relevant}
  When prompt templates change and shared-cache entries become stale:
    * all      -> rerun all stale pairs (default; old behavior)
    * relevant -> rerun only pairs previously judged relevant, and reuse old mbench outputs
                  for previously-irrelevant pairs when available
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from smt_core.inference_engine_5 import AzureInferenceEngine

# IMPORTANT: import judge modules so we can monkeypatch their globals reliably
import judge_relevance as jr
import judge_eligibility as je

from judge_relevance import judge_relevance, RelevancePromptKey
from judge_eligibility import judge_eligibility

from prompt_trace import PromptTrace
from cache_utils import PairDiskCache, CacheKey

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# -------------------------
# Small CLI helpers
# -------------------------

def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return bool(s)


# -------------------------
# Prompt path normalization / monkeypatch
# -------------------------

PromptPathLike = Union[str, Path]


def _to_path(p: PromptPathLike) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _resolve_prompt_path(p: PromptPathLike) -> Path:
    """
    Ensure we return a Path object. If it exists somewhere reasonable, return the existing file.
    Otherwise return a Path anyway (so errors become FileNotFoundError, not attribute errors).
    """
    pp = _to_path(p)

    if pp.is_absolute():
        return pp

    script_dir = Path(__file__).resolve().parent
    bases = [
        Path.cwd(),
        script_dir,
        script_dir.parent,
        script_dir.parent / "eval",
        script_dir.parent / "meval",
        script_dir.parent / "meval" / "prompts",
        script_dir / "prompts",
        script_dir.parent / "prompts",
    ]

    for b in bases:
        cand = (b / pp).resolve()
        if cand.exists() and cand.is_file():
            return cand

    return (Path.cwd() / pp).resolve()


def _normalize_prompt_constants_in_judges() -> None:
    if hasattr(jr, "RELEVANCE_BASE_PROMPT_PATH"):
        jr.RELEVANCE_BASE_PROMPT_PATH = _resolve_prompt_path(jr.RELEVANCE_BASE_PROMPT_PATH)

    # ------------------------------------------------------------------
    # Definition prompts
    # ------------------------------------------------------------------
    def_map = getattr(jr, "RELEVANCE_DEFINITION_PROMPT_PATHS", None)
    if not isinstance(def_map, dict):
        def_map = {}

    try:
        for k, v in list(def_map.items()):
            def_map[k] = _resolve_prompt_path(v)
    except Exception:
        pass

    # Force canonical per-mode entries
    def_map.setdefault("ccr", _resolve_prompt_path("./prompts/relevance_def_ccr.prompt"))
    def_map.setdefault("all", _resolve_prompt_path("./prompts/relevance_def_all.prompt"))
    def_map.setdefault("all-explore", _resolve_prompt_path("./prompts/relevance_def_all-explore.prompt"))

    jr.RELEVANCE_DEFINITION_PROMPT_PATHS = def_map

    # ------------------------------------------------------------------
    # Instruction prompts
    # ------------------------------------------------------------------
    instr_attr_candidates = [
        "RELEVANCE_INSTRUCTIONS_PROMPT_PATHS",
        "RELEVANCE_SPECIFIC_INSTRUCTIONS_PROMPT_PATHS",
        "RELEVANCE_INSTRUCTION_PROMPT_PATHS",
    ]

    instr_dict = None
    for attr in instr_attr_candidates:
        if hasattr(jr, attr):
            instr_dict = getattr(jr, attr)
            break

    if not isinstance(instr_dict, dict):
        instr_dict = {}

    try:
        for k, v in list(instr_dict.items()):
            instr_dict[k] = _resolve_prompt_path(v)
    except Exception:
        pass

    instr_dict.setdefault("ccr", _resolve_prompt_path("./prompts/relevance_instructions_ccr.prompt"))
    instr_dict.setdefault("all", _resolve_prompt_path("./prompts/relevance_instructions_all.prompt"))
    instr_dict.setdefault("all-explore", _resolve_prompt_path("./prompts/relevance_instructions_all-explore.prompt"))

    jr.RELEVANCE_INSTRUCTIONS_PROMPT_PATHS = instr_dict

    if hasattr(je, "ELIGIBILITY_PROMPT_PATH"):
        je.ELIGIBILITY_PROMPT_PATH = _resolve_prompt_path(je.ELIGIBILITY_PROMPT_PATH)
        

_normalize_prompt_constants_in_judges()


def _debug_print_prompt_constants() -> None:
    def _fmt(x: object) -> str:
        try:
            return f"{x} (type={type(x).__name__})"
        except Exception:
            return f"<unprintable> (type={type(x).__name__})"

    print("[DEBUG] jr.RELEVANCE_BASE_PROMPT_PATH:", _fmt(getattr(jr, "RELEVANCE_BASE_PROMPT_PATH", None)))
    try:
        d = getattr(jr, "RELEVANCE_DEFINITION_PROMPT_PATHS", None)
        print("[DEBUG] jr.RELEVANCE_DEFINITION_PROMPT_PATHS keys:", list(d.keys()) if isinstance(d, dict) else d)
        if isinstance(d, dict):
            for k in ("ccr", "all", "all-explore", "cc"):
                if k in d:
                    print(f"[DEBUG] jr.RELEVANCE_DEFINITION_PROMPT_PATHS[{k!r}]:", _fmt(d[k]))
    except Exception as e:
        print("[DEBUG] could not inspect RELEVANCE_DEFINITION_PROMPT_PATHS:", e)

    try:
        d2 = getattr(jr, "RELEVANCE_INSTRUCTIONS_PROMPT_PATHS", None)
        print("[DEBUG] jr.RELEVANCE_INSTRUCTIONS_PROMPT_PATHS keys:", list(d2.keys()) if isinstance(d2, dict) else d2)
        if isinstance(d2, dict):
            for k in ("ccr", "all", "all-explore"):
                if k in d2:
                    print(f"[DEBUG] jr.RELEVANCE_INSTRUCTIONS_PROMPT_PATHS[{k!r}]:", _fmt(d2[k]))
    except Exception as e:
        print("[DEBUG] could not inspect RELEVANCE_INSTRUCTIONS_PROMPT_PATHS:", e)

    print("[DEBUG] je.ELIGIBILITY_PROMPT_PATH:", _fmt(getattr(je, "ELIGIBILITY_PROMPT_PATH", None)))


# -------------------------
# IO utils
# -------------------------

def load_jsonl_as_dict(path: Path, id_key: str = "_id") -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_key not in obj:
                raise ValueError(f"Missing key '{id_key}' in record: {line[:200]}")
            mapping[obj[id_key]] = obj
    return mapping


def get_patient_text(patient_id: str, patient_corpus: Dict[str, dict]) -> str:
    if patient_id not in patient_corpus:
        raise KeyError(f"Patient ID '{patient_id}' not found in patient corpus.")
    if "text" not in patient_corpus[patient_id]:
        raise KeyError(f"No 'text' field for patient_id={patient_id}")
    return patient_corpus[patient_id]["text"]


def get_trial_text(trial_id: str, trial_corpus: Dict[str, dict]) -> str:
    if trial_id not in trial_corpus:
        raise KeyError(f"Trial ID '{trial_id}' not found in trial corpus.")
    if "text" not in trial_corpus[trial_id]:
        raise KeyError(f"No 'text' field for trial_id={trial_id}")
    return trial_corpus[trial_id]["text"]


def build_engine(model_name: str) -> AzureInferenceEngine:
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError("OPENAI_ENDPOINT env var is not set.")
    return AzureInferenceEngine(
        endpoint=endpoint,
        model_name=model_name,
        default_temperature=1.0,
        default_max_tokens=None,
        default_top_p=None,
    )


# -------------------------
# Prompt hashing helpers
# -------------------------

def _resolve_existing_file(p: PromptPathLike) -> Optional[Path]:
    pp = _to_path(p)
    if pp.is_absolute():
        return pp if pp.exists() and pp.is_file() else None
    cand = (Path.cwd() / pp).resolve()
    return cand if cand.exists() and cand.is_file() else None


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file_or_literal(p: PromptPathLike) -> Tuple[str, str]:
    f = _resolve_existing_file(p)
    if f is not None:
        h = hashlib.sha256()
        with f.open("rb") as fin:
            for chunk in iter(lambda: fin.read(1024 * 1024), b""):
                h.update(chunk)
        return ("file", h.hexdigest())
    return ("literal", _sha256_bytes(str(p).encode("utf-8")))


# -------------------------
# Per-pair prompt bundle dumping
# -------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_text_if_file(p: PromptPathLike) -> Optional[str]:
    try:
        pp = _to_path(p)
        if pp.exists() and pp.is_file():
            return pp.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def dump_prompt_bundle(
    out_dir: Path,
    *,
    mode: str,
    patient_id: str,
    trial_id: str,
    rel_base_prompt: PromptPathLike,
    rel_def_prompt: PromptPathLike,
    rel_instr_prompt: Optional[PromptPathLike],
    elig_prompt: PromptPathLike,
    expected_meta: Optional[dict] = None,
) -> None:
    bundle_dir = out_dir / "_prompt_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rb_path = _to_path(rel_base_prompt)
    rd_path = _to_path(rel_def_prompt)
    ri_path = _to_path(rel_instr_prompt) if rel_instr_prompt is not None else None
    el_path = _to_path(elig_prompt)

    rb_txt = _read_text_if_file(rb_path)
    rd_txt = _read_text_if_file(rd_path)
    ri_txt = _read_text_if_file(ri_path) if ri_path is not None else None
    el_txt = _read_text_if_file(el_path)

    if rb_txt is not None:
        (bundle_dir / "relevance_base.prompt.txt").write_text(rb_txt, encoding="utf-8")
    if rd_txt is not None:
        (bundle_dir / f"relevance_definition.{mode}.prompt.txt").write_text(rd_txt, encoding="utf-8")
    if ri_txt is not None:
        (bundle_dir / f"relevance_instructions.{mode}.prompt.txt").write_text(ri_txt, encoding="utf-8")
    if el_txt is not None:
        (bundle_dir / "eligibility.prompt.txt").write_text(el_txt, encoding="utf-8")

    manifest = {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "mode": mode,
        "relevance_base_prompt_path": str(rb_path),
        "relevance_definition_prompt_path": str(rd_path),
        "relevance_instructions_prompt_path": str(ri_path) if ri_path is not None else None,
        "eligibility_prompt_path": str(el_path),
        "expected_meta": expected_meta,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# -------------------------
# Cache meta compatibility
# -------------------------

def compute_expected_meta_compat(
    cache: PairDiskCache,
    *,
    patient_text: str,
    trial_text: str,
    rel_base_prompt: PromptPathLike,
    rel_def_prompt: PromptPathLike,
    rel_instr_prompt: Optional[PromptPathLike],
    elig_prompt: PromptPathLike,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
) -> dict:
    sig = inspect.signature(cache.compute_meta)
    param_names = [p for p in sig.parameters.keys() if p != "self"]

    base = {
        "patient_text": patient_text,
        "trial_text": trial_text,
        "model_name": str(model_name),
        "temperature": float(temperature),
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "top_p": None if top_p is None else float(top_p),
    }

    rb_kind, rb_sha = _sha256_file_or_literal(rel_base_prompt)
    rd_kind, rd_sha = _sha256_file_or_literal(rel_def_prompt)

    if rel_instr_prompt is None:
        ri_kind, ri_sha = ("literal", _sha256_bytes(b"__NO_INSTRUCTIONS__"))
    else:
        ri_kind, ri_sha = _sha256_file_or_literal(rel_instr_prompt)

    el_kind, el_sha = _sha256_file_or_literal(elig_prompt)

    prompt_sha = hashlib.sha256(
        ("\n".join([
            rb_kind + ":" + rb_sha,
            rd_kind + ":" + rd_sha,
            ri_kind + ":" + ri_sha,
            el_kind + ":" + el_sha,
        ])).encode("utf-8")
    ).hexdigest()

    rb_path = _to_path(rel_base_prompt)
    rd_path = _to_path(rel_def_prompt)
    ri_path = _to_path(rel_instr_prompt) if rel_instr_prompt is not None else Path("__NO_INSTRUCTIONS__")
    el_path = _to_path(elig_prompt)

    prompt_paths_path = [rb_path, rd_path, ri_path, el_path]
    prompt_paths_str = [str(rb_path), str(rd_path), str(ri_path), str(el_path)]
    prompt_path_joined = "|".join(prompt_paths_str)

    if len(param_names) == 1:
        payload = dict(base)
        payload.update({
            "prompt_sha": prompt_sha,
            "prompt_path": prompt_path_joined,
            "prompt_paths": prompt_paths_str,
            "relevance_base_prompt_path_str": str(rb_path),
            "relevance_definition_prompt_path_str": str(rd_path),
            "relevance_instructions_prompt_path_str": str(ri_path),
            "eligibility_prompt_path_str": str(el_path),
            "relevance_base_prompt_path": rb_path,
            "relevance_prompt_path": rd_path,
            "relevance_instructions_prompt_path": ri_path,
            "eligibility_prompt_path": el_path,
        })
        return cache.compute_meta(payload)  # type: ignore[arg-type]

    kwargs: dict = {}
    for k, v in base.items():
        if k in sig.parameters:
            kwargs[k] = v

    if "prompt_paths" in sig.parameters:
        kwargs["prompt_paths"] = prompt_paths_path
    if "prompt_path" in sig.parameters:
        kwargs["prompt_path"] = prompt_path_joined
    if "prompt_sha" in sig.parameters:
        kwargs["prompt_sha"] = prompt_sha
    if "prompts_sha" in sig.parameters:
        kwargs["prompts_sha"] = prompt_sha

    if "relevance_prompt_path" in sig.parameters:
        kwargs["relevance_prompt_path"] = rd_path
    if "relevance_instructions_prompt_path" in sig.parameters:
        kwargs["relevance_instructions_prompt_path"] = ri_path
    if "eligibility_prompt_path" in sig.parameters:
        kwargs["eligibility_prompt_path"] = el_path

    if "relevance_prompt_sha" in sig.parameters:
        kwargs["relevance_prompt_sha"] = rb_sha + ":" + rd_sha + ":" + ri_sha
    if "eligibility_prompt_sha" in sig.parameters:
        kwargs["eligibility_prompt_sha"] = el_sha

    return cache.compute_meta(**kwargs)  # type: ignore[arg-type]


# -------------------------
# Subcohort expansion
# -------------------------

def expand_to_subcohorts(parent_trial_id: str, trial_corpus: Dict[str, dict]) -> List[str]:
    sub_ids: List[str] = []
    prefix = parent_trial_id + "__"
    if parent_trial_id in trial_corpus:
        sub_ids.append(parent_trial_id)
    for tid in trial_corpus.keys():
        if tid.startswith(prefix):
            sub_ids.append(tid)
    parent_first = [t for t in sub_ids if t == parent_trial_id]
    rest = sorted([t for t in sub_ids if t != parent_trial_id])
    return parent_first + rest


# -------------------------
# Parsing helpers
# -------------------------

def _coerce_boolish_str(s: str) -> Optional[bool]:
    sl = s.strip().lower()
    if sl in ("true", "yes", "y", "relevant", "eligible"):
        return True
    if sl in ("false", "no", "n", "irrelevant", "not relevant", "ineligible", "not eligible"):
        return False
    return None


def _extract_tagged_payload(text: str, tag: str) -> Optional[str]:
    s = text.strip()
    m = re.search(
        rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*(?:>|$)",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()


def _json_parse_best_effort(payload: str) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return ast.literal_eval(payload)


def _json_leading_value(text: str) -> Tuple[Optional[Any], Optional[int]]:
    s = text.lstrip()
    if not s:
        return None, None
    dec = json.JSONDecoder()
    try:
        obj, idx = dec.raw_decode(s)
        return obj, idx
    except Exception:
        return None, None


def _json_value_after_marker(text: str, marker_regex: str) -> Optional[Any]:
    m = re.search(marker_regex, text, flags=re.IGNORECASE)
    if not m:
        return None
    tail = text[m.end():].lstrip()
    if not tail:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _idx = dec.raw_decode(tail)
        return obj
    except Exception:
        return None


def _json_last_list_anywhere(text: str) -> Optional[list]:
    positions = [m.start() for m in re.finditer(r"\[", text)]
    dec = json.JSONDecoder()
    for pos in reversed(positions[-200:]):
        cand = text[pos:].strip()
        if not cand:
            continue
        if cand.startswith("[]"):
            return []
        try:
            obj, _idx = dec.raw_decode(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            return obj
    return None


def _extract_bool_from_obj(obj: Any, keys: List[str]) -> Optional[bool]:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return _coerce_boolish_str(obj)

    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                v = obj[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    b = _coerce_boolish_str(v)
                    if b is not None:
                        return b
                if isinstance(v, list):
                    return len(v) > 0
        for v in obj.values():
            b = _extract_bool_from_obj(v, keys)
            if b is not None:
                return b
        return None

    return None


def _list_any_true_semantics(
    obj_list: list,
    *,
    decision_key_candidates: List[str],
    true_tokens: List[str],
) -> Optional[bool]:
    if not isinstance(obj_list, list):
        return None
    if len(obj_list) == 0:
        return False

    if all(isinstance(x, str) for x in obj_list):
        return True

    if all(isinstance(x, dict) for x in obj_list):
        saw_any = False
        saw_true = False
        for d in obj_list:
            for k in decision_key_candidates:
                if k not in d:
                    continue
                v = d[k]
                saw_any = True
                if isinstance(v, bool):
                    if v:
                        saw_true = True
                elif isinstance(v, str):
                    if v.strip().lower() in true_tokens:
                        saw_true = True
        if saw_any:
            return True if saw_true else False
        return None

    return None


def parse_relevance_bool(relevance_output: str) -> Optional[bool]:
    payload = _extract_tagged_payload(relevance_output, "relevant_subcohorts")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                if len(obj) == 0:
                    return False

                for entry in obj:
                    if isinstance(entry, dict):
                        name = entry.get("subcohort_name")
                        if isinstance(name, str) and name.strip():
                            return True

                for entry in obj:
                    if isinstance(entry, dict):
                        name2 = entry.get("subcohortName") or entry.get("name")
                        if isinstance(name2, str) and name2.strip():
                            return True

                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
                    true_tokens=["relevant", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    obj0, _ = _json_leading_value(relevance_output)
    if isinstance(obj0, list):
        b = _list_any_true_semantics(
            obj0,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj1 = _json_value_after_marker(relevance_output, r"\boutput\s*:\s*")
    if isinstance(obj1, list):
        b = _list_any_true_semantics(
            obj1,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj2 = _json_last_list_anywhere(relevance_output)
    if isinstance(obj2, list):
        b = _list_any_true_semantics(
            obj2,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    try:
        obj = json.loads(relevance_output.strip())
        b = _extract_bool_from_obj(obj, ["relevant", "is_relevant", "relevance", "decision", "relevance_decision"])
        if b is not None:
            return b
    except Exception:
        pass

    txt = relevance_output.lower()
    if "not relevant" in txt or "irrelevant" in txt:
        return False
    if re.search(r"(^|\W)relevant(\W|$)", txt) and not re.search(r"\bnot\s+relevant\b", txt):
        return True

    return None


def parse_eligibility_bool(elig_output: str) -> Optional[bool]:
    s = elig_output.strip()

    payload = _extract_tagged_payload(s, "subcohort_eligibility_decisions")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=[
                        "eligibility_decision",
                        "eligibilityDecision",
                        "decision",
                        "eligible",
                        "is_eligible",
                    ],
                    true_tokens=["eligible", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    payload = _extract_tagged_payload(s, "eligible_subcohorts")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["eligible", "is_eligible", "decision", "eligibility_decision"],
                    true_tokens=["eligible", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    obj0, _ = _json_leading_value(s)
    if isinstance(obj0, list):
        b = _list_any_true_semantics(
            obj0,
            decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
            true_tokens=["eligible", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj1 = _json_value_after_marker(s, r"\boutput\s*:\s*")
    if isinstance(obj1, list):
        b = _list_any_true_semantics(
            obj1,
            decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
            true_tokens=["eligible", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj2 = _json_last_list_anywhere(s)
    if isinstance(obj2, list):
        b = _list_any_true_semantics(
            obj2,
            decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
            true_tokens=["eligible", "true", "yes", "y"],
        )
        if b is not None:
            return b

    try:
        obj = json.loads(s)
        b = _extract_bool_from_obj(
            obj,
            ["eligible", "is_eligible", "eligibility", "decision", "eligibility_decision", "eligibilityDecision"],
        )
        if b is not None:
            return b
    except Exception:
        pass

    tl = s.lower()
    if re.search(r"\bno\s+relevant\s+subcohorts?\b", tl):
        return False
    if re.search(r"\bno\s+eligibility\s+decisions?\s+are\s+required\b", tl):
        return False
    if re.search(r"\bno\s+eligibility\s+assessment\s+is\s+required\b", tl):
        return False
    if re.search(r"\bthere\s+are\s+no\s+subcohorts?\s+to\s+evaluate\b", tl):
        return False

    if "ineligible" in tl or "not eligible" in tl:
        return False
    if re.search(r"(^|\W)eligible(\W|$)", tl) and "ineligible" not in tl and "not eligible" not in tl:
        return True

    return None


# -------------------------
# Selective rerun helpers
# -------------------------

def load_previous_pair_outputs_if_any(out_dir: Path) -> Optional[Tuple[str, str]]:
    rel_path = out_dir / "relevance.txt"
    elig_path = out_dir / "eligibility.txt"
    if not rel_path.exists() or not elig_path.exists():
        return None
    return _read_text(rel_path), _read_text(elig_path)


def should_force_rerun_pair(
    *,
    prompt_update_rerun: str,
    out_dir: Path,
) -> bool:
    """
    Decide whether to rerun this pair when cache is stale due to prompt/meta change.

    Returns:
      True  -> call judge again
      False -> reuse existing mbench outputs if present
    """
    if prompt_update_rerun == "all":
        return True

    prev = load_previous_pair_outputs_if_any(out_dir)
    if prev is None:
        return True

    prev_rel_txt, _prev_elig_txt = prev
    prev_rel = parse_relevance_bool(prev_rel_txt)

    # Conservative:
    # - previously relevant  => rerun
    # - previously irrelevant => reuse
    # - unknown/unparsable => rerun
    return prev_rel is True or prev_rel is None


# -------------------------
# Core per-subcohort run
# -------------------------

@dataclass(frozen=True)
class RunConfig:
    patient_id: str
    trial_id: str
    patient_note_path: Path
    trial_description_path: Path
    mode: RelevancePromptKey


def run_one_pair(
    rc: RunConfig,
    engine: AzureInferenceEngine,
    trace: PromptTrace,
    patient_corpus_cache: Dict[Path, Dict[str, dict]],
    trial_corpus_cache: Dict[Path, Dict[str, dict]],
    cache: PairDiskCache,
    key_lock: threading.Lock,
    out_dir: Path,
    dump_prompts_per_pair: bool,
    prompt_update_rerun: str,
) -> Tuple[str, str]:

    if rc.patient_note_path not in patient_corpus_cache:
        patient_corpus_cache[rc.patient_note_path] = load_jsonl_as_dict(rc.patient_note_path)
    if rc.trial_description_path not in trial_corpus_cache:
        trial_corpus_cache[rc.trial_description_path] = load_jsonl_as_dict(rc.trial_description_path)

    patient_corpus = patient_corpus_cache[rc.patient_note_path]
    trial_corpus = trial_corpus_cache[rc.trial_description_path]

    patient_text = get_patient_text(rc.patient_id, patient_corpus)
    trial_text = get_trial_text(rc.trial_id, trial_corpus)

    cache_key = CacheKey(mode=rc.mode, patient_id=rc.patient_id, trial_id=rc.trial_id)

    rel_base_prompt = getattr(jr, "RELEVANCE_BASE_PROMPT_PATH", Path("__MISSING__"))
    rel_def_prompt = getattr(jr, "RELEVANCE_DEFINITION_PROMPT_PATHS", {}).get(rc.mode, Path("__MISSING__"))
    rel_instr_prompt = getattr(jr, "RELEVANCE_INSTRUCTIONS_PROMPT_PATHS", {}).get(rc.mode, None)
    elig_prompt = getattr(je, "ELIGIBILITY_PROMPT_PATH", Path("__MISSING__"))

    model_name = getattr(engine, "model_name", "gpt-4.1")
    temperature = float(getattr(engine, "default_temperature", 1.0))
    max_tokens = getattr(engine, "default_max_tokens", None)
    max_tokens_int = None if max_tokens is None else int(max_tokens)
    top_p = getattr(engine, "default_top_p", None)
    top_p_f = None if top_p is None else float(top_p)

    expected_meta = compute_expected_meta_compat(
        cache,
        patient_text=patient_text,
        trial_text=trial_text,
        rel_base_prompt=rel_base_prompt,
        rel_def_prompt=rel_def_prompt,
        rel_instr_prompt=rel_instr_prompt,
        elig_prompt=elig_prompt,
        model_name=str(model_name),
        temperature=temperature,
        max_tokens=max_tokens_int,
        top_p=top_p_f,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    if dump_prompts_per_pair:
        dump_prompt_bundle(
            out_dir,
            mode=str(rc.mode),
            patient_id=rc.patient_id,
            trial_id=rc.trial_id,
            rel_base_prompt=rel_base_prompt,
            rel_def_prompt=rel_def_prompt,
            rel_instr_prompt=rel_instr_prompt,
            elig_prompt=elig_prompt,
            expected_meta=expected_meta,
        )

    with key_lock:
        cached = cache.load_if_fresh(cache_key, expected_meta)

        if cached is not None:
            (out_dir / "relevance.txt").write_text(cached["relevance"], encoding="utf-8")
            (out_dir / "eligibility.txt").write_text(cached["eligibility"], encoding="utf-8")
            return cached["relevance"], cached["eligibility"]

        # Cache is stale or missing under current prompt/meta.
        # Optionally reuse old mbench outputs for previously-irrelevant pairs.
        if prompt_update_rerun != "all":
            prev = load_previous_pair_outputs_if_any(out_dir)
            if prev is not None and not should_force_rerun_pair(
                prompt_update_rerun=prompt_update_rerun,
                out_dir=out_dir,
            ):
                prev_rel_txt, prev_elig_txt = prev
                print(
                    f"[SKIP-RERUN] mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id} "
                    f"(previous relevance=False; reusing old outputs)"
                )
                return prev_rel_txt, prev_elig_txt

    relevance_output = judge_relevance(
        trial_text,
        patient_text,
        engine=engine,
        prompt_key=rc.mode,
        trace=trace,
        patient_id=rc.patient_id,
        trial_id=rc.trial_id,
    )

    rel_b = parse_relevance_bool(relevance_output)
    if rel_b is False:
        eligibility_output = (
            "No relevant subcohorts; no eligibility decisions are required.\n"
            "Eligibility was skipped because relevance was False."
        )

        with key_lock:
            cache.save(cache_key, expected_meta, relevance_output, eligibility_output)

        (out_dir / "relevance.txt").write_text(relevance_output, encoding="utf-8")
        (out_dir / "eligibility.txt").write_text(eligibility_output, encoding="utf-8")
        return relevance_output, eligibility_output

    eligibility_output = judge_eligibility(
        trial_text,
        patient_text,
        relevance_result=relevance_output,
        engine=engine,
        trace=trace,
        patient_id=rc.patient_id,
        trial_id=rc.trial_id,
    )

    with key_lock:
        cache.save(cache_key, expected_meta, relevance_output, eligibility_output)

    (out_dir / "relevance.txt").write_text(relevance_output, encoding="utf-8")
    (out_dir / "eligibility.txt").write_text(eligibility_output, encoding="utf-8")
    return relevance_output, eligibility_output


# -------------------------
# Patient-label processing
# -------------------------

def load_patient_label_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def list_patient_label_files(patient_labels_dir: Path) -> List[Path]:
    return sorted([p for p in patient_labels_dir.glob("*.json") if p.is_file()])


def pick_patient_files_single(dir_path: Path, sample_n: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    files = list_patient_label_files(dir_path)
    if not files:
        raise RuntimeError(f"No .json files found in {dir_path}")
    return files if len(files) <= sample_n else rng.sample(files, sample_n)


def pick_patients_shared_across_modes(
    mode_to_dir: Dict[str, Path],
    sample_n: int,
    seed: int,
) -> Dict[str, List[Path]]:
    rng = random.Random(seed)

    mode_to_id_to_path: Dict[str, Dict[str, Path]] = {}
    for mode, d in mode_to_dir.items():
        files = list_patient_label_files(d)
        if not files:
            raise RuntimeError(f"No .json files found in {d}")
        mode_to_id_to_path[mode] = {p.stem.split("__", 1)[0]: p for p in files}

    common_ids = None
    for id_to_path in mode_to_id_to_path.values():
        ids = set(id_to_path.keys())
        common_ids = ids if common_ids is None else (common_ids & ids)

    inter = sorted(common_ids or set())
    if not inter:
        raise RuntimeError("No intersection patients across requested patient_labels dirs.")

    chosen = inter if len(inter) <= sample_n else rng.sample(inter, sample_n)
    return {
        mode: [mode_to_id_to_path[mode][pid] for pid in chosen]
        for mode in mode_to_dir
    }


# -------------------------
# Progress bar helpers
# -------------------------

class _SimpleProgress:
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = max(1, int(total))
        self.i = 0
        self.desc = desc

    def update(self, n: int = 1) -> None:
        self.i += n
        if self.i > self.total:
            self.i = self.total
        if self.i == 1 or self.i == self.total or (self.i % 10 == 0):
            print(f"[{self.desc}] {self.i}/{self.total}")

    def close(self) -> None:
        if self.i < self.total:
            print(f"[{self.desc}] {self.i}/{self.total} (stopped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ccr-patient-labels-dir",
        type=str,
        default="../ops/out_compose/clean_eval__ccr__prevent/patient_labels",
    )
    ap.add_argument(
        "--all-patient-labels-dir",
        type=str,
        default="../ops/out_compose/clean_eval__all__prevent/patient_labels",
    )
    ap.add_argument(
        "--all-explore-patient-labels-dir",
        type=str,
        default="../ops/out_compose/clean_eval__all__prevent__nonact/patient_labels",
    )

    ap.add_argument(
        "--run-modes",
        type=str,
        default="both",
        choices=["both", "ccr", "all", "all-explore", "triple"],
        help="Which patient-label modes to run: both, ccr, all, all-explore, or triple.",
    )

    ap.add_argument("--sample-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shared-sample", action="store_true")

    ap.add_argument("--patient-corpus", type=str, default="../../dataset/clinical_trial/sigir/queries.jsonl")
    ap.add_argument("--trial-corpus", type=str, default="../../dataset/clinical_trial/sigir/corpus.jsonl")

    ap.add_argument("--model-name", type=str, default="gpt-4.1")

    ap.add_argument("--output-root", type=str, default="./smt_retrieval_eval_out")
    ap.add_argument("--mbench-root", type=str, default="./smt_retrieval_eval_mbench")

    ap.add_argument(
        "--cache-root",
        type=str,
        default="./_shared_pair_cache/pair_cache",
        help="Shared PairDiskCache directory. Use same value in both SMT and TrialGPT runners to reuse results.",
    )
    ap.add_argument(
        "--prompt-update-rerun",
        type=str,
        default="all",
        choices=["all", "relevant"],
        help=(
            "When prompt templates change and shared-cache entries become stale: "
            "'all' reruns all stale pairs; "
            "'relevant' reruns only pairs previously judged relevant and reuses old "
            "mbench outputs for previously-irrelevant pairs when available."
        ),
    )

    ap.add_argument("--num-workers", type=int, default=32)

    ap.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print resolved prompt constants and their types at startup.",
    )

    ap.add_argument(
        "--dump-prompts-per-pair",
        type=str,
        default="true",
        help="Save prompt template files + manifest.json into each mbench pair directory. (true/false)",
    )

    args = ap.parse_args()

    dump_prompts_per_pair = _str2bool(args.dump_prompts_per_pair)

    if args.debug_prompts:
        _debug_print_prompt_constants()

    mode_order_map = {
        "both": ["ccr", "all"],
        "triple": ["ccr", "all", "all-explore"],
        "ccr": ["ccr"],
        "all": ["all"],
        "all-explore": ["all-explore"],
    }
    selected_modes = mode_order_map[args.run_modes]

    mode_to_dir: Dict[str, Path] = {}
    if "ccr" in selected_modes:
        mode_to_dir["ccr"] = Path(args.ccr_patient_labels_dir).resolve()
    if "all" in selected_modes:
        mode_to_dir["all"] = Path(args.all_patient_labels_dir).resolve()
    if "all-explore" in selected_modes:
        mode_to_dir["all-explore"] = Path(args.all_explore_patient_labels_dir).resolve()

    for mode, d in mode_to_dir.items():
        if not d.exists():
            raise FileNotFoundError(f"Missing patient labels dir for mode={mode}: {d}")

    patient_corpus_path = Path(args.patient_corpus).resolve()
    trial_corpus_path = Path(args.trial_corpus).resolve()

    output_root = Path(args.output_root).resolve()
    mbench_root = Path(args.mbench_root).resolve()
    mbench_root.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root).resolve()
    cache_root.parent.mkdir(parents=True, exist_ok=True)

    engine = build_engine(args.model_name)

    trace = PromptTrace(mbench_root / "prompts_and_outputs_dump.txt", tokenizer_model=args.model_name)
    cache = PairDiskCache(cache_root)

    patient_corpus_cache: Dict[Path, Dict[str, dict]] = {}
    trial_corpus_cache: Dict[Path, Dict[str, dict]] = {}

    trial_corpus_cache[trial_corpus_path] = load_jsonl_as_dict(trial_corpus_path)
    trial_corpus = trial_corpus_cache[trial_corpus_path]

    locks: Dict[CacheKey, threading.Lock] = {}
    locks_guard = threading.Lock()

    def get_lock(k: CacheKey) -> threading.Lock:
        with locks_guard:
            if k not in locks:
                locks[k] = threading.Lock()
            return locks[k]

    if args.shared_sample and len(mode_to_dir) > 1:
        mode_to_files = pick_patients_shared_across_modes(mode_to_dir, args.sample_n, args.seed)
    else:
        mode_to_files = {
            mode: pick_patient_files_single(d, args.sample_n, args.seed)
            for mode, d in mode_to_dir.items()
        }

    def process_one_patient_label_file(label_path: Path, mode: RelevancePromptKey, pbar=None) -> None:
        label_obj = load_patient_label_file(label_path)
        patient_id = label_obj["patient_id"]

        trials = label_obj.get("trials", [])
        if not isinstance(trials, list):
            raise ValueError(f"Bad trials field in {label_path}")

        for t in trials:
            parent_trial_id = t["trial_id"]
            sub_ids = expand_to_subcohorts(parent_trial_id, trial_corpus)

            any_rel = False
            any_elig = False
            any_rel_and_elig = False
            sub_summaries: List[dict] = []

            for sub_id in sub_ids:
                rc = RunConfig(
                    patient_id=patient_id,
                    trial_id=sub_id,
                    patient_note_path=patient_corpus_path,
                    trial_description_path=trial_corpus_path,
                    mode=mode,
                )
                ck = CacheKey(mode=mode, patient_id=patient_id, trial_id=sub_id)
                out_dir = mbench_root / str(mode) / patient_id / parent_trial_id / sub_id

                rel_out, elig_out = run_one_pair(
                    rc=rc,
                    engine=engine,
                    trace=trace,
                    patient_corpus_cache=patient_corpus_cache,
                    trial_corpus_cache=trial_corpus_cache,
                    cache=cache,
                    key_lock=get_lock(ck),
                    out_dir=out_dir,
                    dump_prompts_per_pair=dump_prompts_per_pair,
                    prompt_update_rerun=args.prompt_update_rerun,
                )

                rel_b = parse_relevance_bool(rel_out)
                elig_b = parse_eligibility_bool(elig_out)

                rel_true = (rel_b is True)
                elig_true = (elig_b is True)

                any_rel = any_rel or rel_true
                any_elig = any_elig or elig_true
                any_rel_and_elig = any_rel_and_elig or (rel_true and elig_true)

                sub_summaries.append({
                    "subcohort_id": sub_id,
                    "relevant": rel_b,
                    "eligible": elig_b,
                })

                if pbar is not None:
                    pbar.update(1)

            t["any_subcohort_relevant"] = any_rel
            t["any_subcohort_eligible"] = any_elig
            t["any_subcohort_relevant_and_eligible"] = any_rel_and_elig
            t["subcohort_judge_summary"] = sub_summaries

        out_path = output_root / str(mode) / "patient_labels" / label_path.name
        write_json(out_path, label_obj)

    jobs: List[Tuple[Path, RelevancePromptKey]] = []
    for mode in selected_modes:
        jobs += [(p, mode) for p in mode_to_files.get(mode, [])]  # type: ignore[list-item]

    total_pairs = 0
    for label_path, mode in jobs:
        try:
            label_obj = load_patient_label_file(label_path)
            trials = label_obj.get("trials", [])
            if isinstance(trials, list):
                for t in trials:
                    parent_trial_id = t.get("trial_id")
                    if not parent_trial_id:
                        continue
                    total_pairs += len(expand_to_subcohorts(parent_trial_id, trial_corpus))
        except Exception:
            pass

    if tqdm is not None:
        pbar = tqdm(total=total_pairs, desc="Judge pairs", unit="pair", dynamic_ncols=True)
    else:
        pbar = _SimpleProgress(total=total_pairs, desc="Judge pairs")

    try:
        if args.num_workers <= 1:
            for p, m in jobs:
                print(f"[PATIENT] mode={m} file={p}")
                process_one_patient_label_file(p, m, pbar=pbar)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                futs = {ex.submit(process_one_patient_label_file, p, m, pbar): (p, m) for p, m in jobs}
                for fut in as_completed(futs):
                    p, m = futs[fut]
                    try:
                        fut.result()
                    except Exception:
                        import traceback
                        print(f"[ERROR] mode={m} file={p}")
                        print(traceback.format_exc())
    finally:
        try:
            pbar.close()
        except Exception:
            pass

    print(
        f"[TOKENS] prompt={trace.totals.prompt_tokens} "
        f"completion={trace.totals.completion_tokens} "
        f"total={trace.totals.total}"
    )
    print(f"[DONE] augmented labels in: {output_root}")
    print(f"[DONE] mbench outputs in:   {mbench_root}")
    print(f"[DONE] shared cache in:     {cache_root}")


if __name__ == "__main__":
    main()