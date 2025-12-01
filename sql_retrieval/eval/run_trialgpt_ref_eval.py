#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trialgpt_ref_eval.py

Robust version with two-layer rerun policy:

1) Prompt-staleness rerun policy
   --prompt-update-rerun {all,relevant}

2) Broken-output rerun policy
   --final-parse-rerun {never,missing,parse_error}

Behavior
--------
- retries judge calls if outputs are truncated / unparsable
- atomically writes relevance.txt and eligibility.txt
- validates cached outputs before reusing them
- during final aggregation, if an existing pair output is missing/broken,
  it can delete + regenerate that pair instead of aborting
- on repair rerun, bypasses cache entirely
- if a rerun still leaves empty outputs, empties are conservatively treated as False
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import random
import re
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from smt_core.inference_engine_5 import AzureInferenceEngine

import judge_relevance as jr
import judge_eligibility as je
from judge_relevance import judge_relevance, RelevancePromptKey
from judge_eligibility import judge_eligibility
from prompt_trace import PromptTrace
from cache_utils import PairDiskCache, CacheKey

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# =============================================================================
# Prompt path normalization / monkeypatch
# =============================================================================

PromptPathLike = Union[str, Path]


def _to_path(p: PromptPathLike) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _resolve_prompt_path(p: PromptPathLike) -> Path:
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

    def_map = getattr(jr, "RELEVANCE_DEFINITION_PROMPT_PATHS", None)
    if not isinstance(def_map, dict):
        def_map = {}

    try:
        for k, v in list(def_map.items()):
            def_map[k] = _resolve_prompt_path(v)
    except Exception:
        pass

    def_map.setdefault("ccr", _resolve_prompt_path("./prompts/relevance_def_ccr.prompt"))
    def_map.setdefault("all", _resolve_prompt_path("./prompts/relevance_def_all.prompt"))
    def_map.setdefault("all-explore", _resolve_prompt_path("./prompts/relevance_def_all-explore.prompt"))
    jr.RELEVANCE_DEFINITION_PROMPT_PATHS = def_map

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
        return f"{x} (type={type(x).__name__})"

    print("[DEBUG] jr.RELEVANCE_BASE_PROMPT_PATH:", _fmt(getattr(jr, "RELEVANCE_BASE_PROMPT_PATH", None)))
    try:
        print("[DEBUG] jr.RELEVANCE_DEFINITION_PROMPT_PATHS keys:", list(jr.RELEVANCE_DEFINITION_PROMPT_PATHS.keys()))
        for k in ("ccr", "all", "all-explore", "cc"):
            if k in jr.RELEVANCE_DEFINITION_PROMPT_PATHS:
                print(
                    f"[DEBUG] jr.RELEVANCE_DEFINITION_PROMPT_PATHS[{k!r}]:",
                    _fmt(jr.RELEVANCE_DEFINITION_PROMPT_PATHS[k]),
                )
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


# =============================================================================
# IO
# =============================================================================

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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def list_json_files(d: Path) -> List[Path]:
    return sorted([p for p in d.glob("*.json") if p.is_file()])


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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


# =============================================================================
# Cache meta compatibility
# =============================================================================

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

    params = [p for p in sig.parameters.keys() if p != "self"]
    if len(params) == 1:
        payload = dict(base)
        payload.update(
            {
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
            }
        )
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


# =============================================================================
# Subcohort expansion
# =============================================================================

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


# =============================================================================
# Parsing helpers
# =============================================================================

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


def parse_relevance_bool(text: str) -> Optional[bool]:
    payload = _extract_tagged_payload(text, "relevant_subcohorts")
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

    obj0, _ = _json_leading_value(text)
    if isinstance(obj0, list):
        b = _list_any_true_semantics(
            obj0,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj1 = _json_value_after_marker(text, r"\boutput\s*:\s*")
    if isinstance(obj1, list):
        b = _list_any_true_semantics(
            obj1,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj2 = _json_last_list_anywhere(text)
    if isinstance(obj2, list):
        b = _list_any_true_semantics(
            obj2,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    try:
        obj = json.loads(text.strip())
        b = _extract_bool_from_obj(obj, ["relevant", "is_relevant", "relevance", "decision", "relevance_decision"])
        if b is not None:
            return b
    except Exception:
        pass

    tl = text.lower()
    if "not relevant" in tl or "irrelevant" in tl:
        return False
    if re.search(r"(^|\W)relevant(\W|$)", tl) and not re.search(r"\bnot\s+relevant\b", tl):
        return True

    return None


def parse_eligibility_bool(text: str) -> Optional[bool]:
    s = text.strip()

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


def parse_relevance_bool_strict(text: str) -> bool:
    b = parse_relevance_bool(text)
    if b is None:
        raise ValueError("Could not strictly parse relevance boolean.")
    return bool(b)


def parse_eligibility_bool_strict(text: str) -> bool:
    b = parse_eligibility_bool(text)
    if b is None:
        raise ValueError("Could not strictly parse eligibility boolean.")
    return bool(b)


def parse_relevance_bool_or_empty_false(text: str) -> bool:
    if not (text or "").strip():
        return False
    return parse_relevance_bool_strict(text)


def parse_eligibility_bool_or_empty_false(text: str) -> bool:
    if not (text or "").strip():
        return False
    return parse_eligibility_bool_strict(text)


# =============================================================================
# Canonical fallback outputs for empty final files
# =============================================================================

def canonical_relevance_false_output() -> str:
    return "<relevant_subcohorts>[]</relevant_subcohorts>\n"


def canonical_eligibility_false_output() -> str:
    return (
        "<subcohort_eligibility_decisions>[]</subcohort_eligibility_decisions>\n"
        "No relevant subcohorts; no eligibility decisions are required.\n"
    )


def write_canonical_false_outputs_for_empty_pair(out_dir: Path) -> Tuple[bool, bool]:
    rel_path = out_dir / "relevance.txt"
    elig_path = out_dir / "eligibility.txt"

    rel_txt = _read_text(rel_path) if rel_path.exists() else ""
    elig_txt = _read_text(elig_path) if elig_path.exists() else ""

    if not rel_txt.strip():
        rel_txt = canonical_relevance_false_output()
        atomic_write_text(rel_path, rel_txt)

    if not elig_txt.strip():
        elig_txt = canonical_eligibility_false_output()
        atomic_write_text(elig_path, elig_txt)

    rel_b = parse_relevance_bool_or_empty_false(rel_txt)
    elig_b = parse_eligibility_bool_or_empty_false(elig_txt)
    return rel_b, elig_b


# =============================================================================
# Debug dump helpers
# =============================================================================

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_excerpt(text: str, n: int = 4000) -> str:
    return (text or "")[:n]


def _dump_parse_debug(
    *,
    debug_dir: Path,
    kind: str,
    mode: str,
    patient_id: str,
    trial_id: str,
    attempt: int,
    text: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{kind}__mode={mode}__patient={patient_id}__trial={trial_id}__attempt={attempt}.txt"
    (debug_dir / fname).write_text(text or "", encoding="utf-8", errors="replace")


def debug_parse_relevance(text: str) -> Tuple[Optional[bool], Dict[str, Any]]:
    info: Dict[str, Any] = {}

    payload = _extract_tagged_payload(text, "relevant_subcohorts")
    info["tag_payload_found"] = payload is not None
    if payload is not None:
        info["tag_payload_excerpt"] = _safe_excerpt(payload, 800)
        try:
            obj = _json_parse_best_effort(payload)
            info["tag_payload_type"] = type(obj).__name__
        except Exception as e:
            info["tag_payload_parse_error"] = repr(e)

    obj0, _idx0 = _json_leading_value(text)
    info["leading_json_type"] = None if obj0 is None else type(obj0).__name__

    obj1 = _json_value_after_marker(text, r"\boutput\s*:\s*")
    info["after_output_marker_type"] = None if obj1 is None else type(obj1).__name__

    obj2 = _json_last_list_anywhere(text)
    info["last_list_anywhere_found"] = obj2 is not None

    b = parse_relevance_bool(text)
    info["final_parse_result"] = b
    return b, info


def debug_parse_eligibility(text: str) -> Tuple[Optional[bool], Dict[str, Any]]:
    info: Dict[str, Any] = {}

    payload = _extract_tagged_payload(text, "subcohort_eligibility_decisions")
    info["tag_decisions_found"] = payload is not None
    if payload is not None:
        info["tag_decisions_excerpt"] = _safe_excerpt(payload, 800)
        try:
            obj = _json_parse_best_effort(payload)
            info["tag_decisions_type"] = type(obj).__name__
        except Exception as e:
            info["tag_decisions_parse_error"] = repr(e)

    payload2 = _extract_tagged_payload(text, "eligible_subcohorts")
    info["tag_eligible_subcohorts_found"] = payload2 is not None

    obj0, _idx0 = _json_leading_value(text)
    info["leading_json_type"] = None if obj0 is None else type(obj0).__name__

    obj1 = _json_value_after_marker(text, r"\boutput\s*:\s*")
    info["after_output_marker_type"] = None if obj1 is None else type(obj1).__name__

    obj2 = _json_last_list_anywhere(text)
    info["last_list_anywhere_found"] = obj2 is not None

    b = parse_eligibility_bool(text)
    info["final_parse_result"] = b
    return b, info


# =============================================================================
# STRICT rederive-from-mbench parser (and helpers)
# =============================================================================

def _excerpt(s: str, n: int = 800) -> str:
    return s[:n].replace("\n", "\\n")


def _looks_truncated_tag_output(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    sl = s.lower()
    if sl.startswith("<relevant_sub") and ("</relevant_subcohorts" not in sl):
        return True
    if sl.startswith("<eligible_sub") and ("</eligible_subcohorts" not in sl):
        return True
    if sl.startswith("<subcohort_eligibility") and ("</subcohort_eligibility_decisions" not in sl):
        return True
    if sl.startswith("<") and (">" not in sl[:200]):
        return True
    return False


def _ensure_parseable_or_raise(relevance_output: str, eligibility_output: str) -> None:
    _ = parse_relevance_bool_strict(relevance_output)
    _ = parse_eligibility_bool_strict(eligibility_output)


def _is_missing_or_empty(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        return path.stat().st_size == 0
    except Exception:
        return True


def validate_pair_outputs_texts(
    relevance_output: str,
    eligibility_output: str,
) -> Tuple[bool, str]:
    if not (relevance_output or "").strip():
        return False, "empty_relevance"
    if not (eligibility_output or "").strip():
        return False, "empty_eligibility"

    if _looks_truncated_tag_output(relevance_output):
        return False, "truncated_relevance"
    if _looks_truncated_tag_output(eligibility_output):
        return False, "truncated_eligibility"

    try:
        _ensure_parseable_or_raise(relevance_output, eligibility_output)
    except Exception as e:
        return False, f"strict_parse_failed: {e}"

    return True, "ok"


# =============================================================================
# Selective rerun helpers
# =============================================================================

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
    if prompt_update_rerun == "all":
        return True

    prev = load_previous_pair_outputs_if_any(out_dir)
    if prev is None:
        return True

    prev_rel_txt, _prev_elig_txt = prev
    prev_rel = parse_relevance_bool(prev_rel_txt)
    return prev_rel is True or prev_rel is None


# =============================================================================
# Pair state / repair helpers
# =============================================================================

@dataclass(frozen=True)
class PairKey:
    mode: str
    patient_id: str
    parent_trial_id: str
    subcohort_id: str


@dataclass
class PairOutputStatus:
    ok: bool
    reason: str
    rel_exists: bool
    elig_exists: bool
    rel_empty: bool
    elig_empty: bool
    rel_truncated: bool
    elig_truncated: bool
    rel_parse_ok: bool
    elig_parse_ok: bool
    rel_bool: Optional[bool] = None
    elig_bool: Optional[bool] = None


def pair_sub_dir(mbench_root: Path, key: PairKey) -> Path:
    return mbench_root / key.mode / key.patient_id / key.parent_trial_id / key.subcohort_id


def pair_output_status(mbench_root: Path, key: PairKey) -> PairOutputStatus:
    sub_dir = pair_sub_dir(mbench_root, key)
    rel_path = sub_dir / "relevance.txt"
    elig_path = sub_dir / "eligibility.txt"

    rel_exists = rel_path.exists()
    elig_exists = elig_path.exists()

    if not rel_exists or not elig_exists:
        return PairOutputStatus(
            ok=False,
            reason="missing_file",
            rel_exists=rel_exists,
            elig_exists=elig_exists,
            rel_empty=not rel_exists,
            elig_empty=not elig_exists,
            rel_truncated=False,
            elig_truncated=False,
            rel_parse_ok=False,
            elig_parse_ok=False,
        )

    rel_txt = _read_text(rel_path)
    elig_txt = _read_text(elig_path)

    rel_empty = not rel_txt.strip()
    elig_empty = not elig_txt.strip()

    rel_truncated = False if rel_empty else _looks_truncated_tag_output(rel_txt)
    elig_truncated = False if elig_empty else _looks_truncated_tag_output(elig_txt)

    rel_parse_ok = False
    elig_parse_ok = False
    rel_bool: Optional[bool] = None
    elig_bool: Optional[bool] = None

    if rel_truncated or elig_truncated:
        return PairOutputStatus(
            ok=False,
            reason="truncated_output",
            rel_exists=rel_exists,
            elig_exists=elig_exists,
            rel_empty=rel_empty,
            elig_empty=elig_empty,
            rel_truncated=rel_truncated,
            elig_truncated=elig_truncated,
            rel_parse_ok=False,
            elig_parse_ok=False,
        )

    if rel_empty:
        rel_parse_ok = True
        rel_bool = False
    else:
        try:
            rel_bool = parse_relevance_bool_strict(rel_txt)
            rel_parse_ok = True
        except Exception:
            rel_parse_ok = False

    if elig_empty:
        elig_parse_ok = True
        elig_bool = False
    else:
        try:
            elig_bool = parse_eligibility_bool_strict(elig_txt)
            elig_parse_ok = True
        except Exception:
            elig_parse_ok = False

    if rel_parse_ok and elig_parse_ok:
        if rel_empty or elig_empty:
            return PairOutputStatus(
                ok=False,
                reason="empty_file",
                rel_exists=rel_exists,
                elig_exists=elig_exists,
                rel_empty=rel_empty,
                elig_empty=elig_empty,
                rel_truncated=False,
                elig_truncated=False,
                rel_parse_ok=True,
                elig_parse_ok=True,
                rel_bool=rel_bool,
                elig_bool=elig_bool,
            )

        return PairOutputStatus(
            ok=True,
            reason="ok",
            rel_exists=rel_exists,
            elig_exists=elig_exists,
            rel_empty=False,
            elig_empty=False,
            rel_truncated=False,
            elig_truncated=False,
            rel_parse_ok=True,
            elig_parse_ok=True,
            rel_bool=rel_bool,
            elig_bool=elig_bool,
        )

    return PairOutputStatus(
        ok=False,
        reason="strict_parse_failed",
        rel_exists=rel_exists,
        elig_exists=elig_exists,
        rel_empty=rel_empty,
        elig_empty=elig_empty,
        rel_truncated=False,
        elig_truncated=False,
        rel_parse_ok=rel_parse_ok,
        elig_parse_ok=elig_parse_ok,
        rel_bool=rel_bool,
        elig_bool=elig_bool,
    )


def delete_pair_outputs(out_dir: Path) -> None:
    for name in ("relevance.txt", "eligibility.txt"):
        try:
            (out_dir / name).unlink(missing_ok=True)
        except Exception:
            pass


def load_and_parse_pair(mbench_root: Path, key: PairKey) -> Tuple[bool, bool]:
    sub_dir = pair_sub_dir(mbench_root, key)
    rel_path = sub_dir / "relevance.txt"
    elig_path = sub_dir / "eligibility.txt"

    if not rel_path.exists():
        raise FileNotFoundError(f"Missing {rel_path} for {key}")
    if not elig_path.exists():
        raise FileNotFoundError(f"Missing {elig_path} for {key}")

    rel_txt = _read_text(rel_path)
    elig_txt = _read_text(elig_path)

    try:
        rel_b = parse_relevance_bool_or_empty_false(rel_txt)
    except Exception as e:
        raise RuntimeError(
            f"[PARSE ERROR] relevance for {key}\n"
            f"PATH: {rel_path}\n"
            f"ERR:  {e}\n"
            f"EXCERPT: {_excerpt(rel_txt)}"
        )

    try:
        elig_b = parse_eligibility_bool_or_empty_false(elig_txt)
    except Exception as e:
        raise RuntimeError(
            f"[PARSE ERROR] eligibility for {key}\n"
            f"PATH: {elig_path}\n"
            f"ERR:  {e}\n"
            f"EXCERPT: {_excerpt(elig_txt)}"
        )

    return rel_b, elig_b


def repair_mbench_pairs(mbench_root: Path, modes: List[str]) -> int:
    deleted = 0
    for mode in modes:
        mode_dir = mbench_root / mode
        if not mode_dir.exists():
            continue

        for patient_dir in mode_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            for parent_dir in patient_dir.iterdir():
                if not parent_dir.is_dir():
                    continue
                for sub_dir in parent_dir.iterdir():
                    if not sub_dir.is_dir():
                        continue

                    key = PairKey(
                        mode=mode,
                        patient_id=patient_dir.name,
                        parent_trial_id=parent_dir.name,
                        subcohort_id=sub_dir.name,
                    )
                    status = pair_output_status(mbench_root, key)
                    if not status.ok:
                        delete_pair_outputs(sub_dir)
                        deleted += 1
    return deleted


# =============================================================================
# Judge run
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    patient_id: str
    trial_id: str
    patient_note_path: Path
    trial_description_path: Path
    mode: RelevancePromptKey

    debug_parse: bool = False
    debug_parse_dir: Path = Path("./_debug_parse_dump")
    print_judge_outputs: bool = False
    prompt_update_rerun: str = "all"
    max_judge_attempts: int = 3
    force_no_cache: bool = False


def run_one_pair(
    rc: RunConfig,
    engine: AzureInferenceEngine,
    trace: PromptTrace,
    patient_corpus_cache: Dict[Path, Dict[str, dict]],
    trial_corpus_cache: Dict[Path, Dict[str, dict]],
    cache: PairDiskCache,
    key_lock: threading.Lock,
    out_dir: Path,
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

    with key_lock:
        cached = None if rc.force_no_cache else cache.load_if_fresh(cache_key, expected_meta)
        if cached is not None:
            cached_rel = str(cached.get("relevance", ""))
            cached_elig = str(cached.get("eligibility", ""))

            ok_cached, cached_reason = validate_pair_outputs_texts(cached_rel, cached_elig)
            if ok_cached:
                atomic_write_text(out_dir / "relevance.txt", cached_rel)
                atomic_write_text(out_dir / "eligibility.txt", cached_elig)
                return cached_rel, cached_elig
            else:
                print(
                    f"[BAD CACHE IGNORED] mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id} "
                    f"reason={cached_reason}"
                )

        if rc.prompt_update_rerun != "all":
            prev = load_previous_pair_outputs_if_any(out_dir)
            if prev is not None and not should_force_rerun_pair(
                prompt_update_rerun=rc.prompt_update_rerun,
                out_dir=out_dir,
            ):
                prev_rel_txt, prev_elig_txt = prev
                ok_prev, prev_reason = validate_pair_outputs_texts(prev_rel_txt, prev_elig_txt)
                if ok_prev:
                    print(
                        f"[SKIP-RERUN] mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id} "
                        f"(previous relevance=False; reusing old outputs)"
                    )
                    return prev_rel_txt, prev_elig_txt
                else:
                    print(
                        f"[BAD PREV OUTPUTS IGNORED] mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id} "
                        f"reason={prev_reason}"
                    )

        last_err: Optional[Exception] = None
        for attempt in range(rc.max_judge_attempts):
            relevance_output = judge_relevance(
                trial_text,
                patient_text,
                engine=engine,
                prompt_key=rc.mode,
                trace=trace,
                patient_id=rc.patient_id,
                trial_id=rc.trial_id,
            )

            if rc.print_judge_outputs:
                print(
                    f"\n[RAW RELEVANCE] mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id} attempt={attempt}\n"
                    f"{relevance_output}\n"
                )

            if not (relevance_output or "").strip():
                last_err = RuntimeError("Relevance output is empty.")
                if rc.debug_parse:
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir,
                        kind="relevance",
                        mode=str(rc.mode),
                        patient_id=rc.patient_id,
                        trial_id=rc.trial_id,
                        attempt=attempt,
                        text=relevance_output,
                    )
                continue

            if _looks_truncated_tag_output(relevance_output):
                last_err = RuntimeError("Relevance output looks truncated (tag not closed / empty).")
                if rc.debug_parse:
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir,
                        kind="relevance",
                        mode=str(rc.mode),
                        patient_id=rc.patient_id,
                        trial_id=rc.trial_id,
                        attempt=attempt,
                        text=relevance_output,
                    )
                continue

            rel_b = parse_relevance_bool(relevance_output)
            if rel_b is False:
                eligibility_output = canonical_eligibility_false_output()
                try:
                    _ensure_parseable_or_raise(relevance_output, eligibility_output)
                except Exception as e:
                    last_err = e
                    if rc.debug_parse:
                        _dump_parse_debug(
                            debug_dir=rc.debug_parse_dir, kind="relevance",
                            mode=str(rc.mode), patient_id=rc.patient_id, trial_id=rc.trial_id,
                            attempt=attempt, text=relevance_output,
                        )
                        _dump_parse_debug(
                            debug_dir=rc.debug_parse_dir, kind="eligibility",
                            mode=str(rc.mode), patient_id=rc.patient_id, trial_id=rc.trial_id,
                            attempt=attempt, text=eligibility_output,
                        )
                    continue

                ok_pair, pair_reason = validate_pair_outputs_texts(relevance_output, eligibility_output)
                if not ok_pair:
                    last_err = RuntimeError(f"Generated pair invalid before save: {pair_reason}")
                    continue

                cache.save(cache_key, expected_meta, relevance_output, eligibility_output)
                atomic_write_text(out_dir / "relevance.txt", relevance_output)
                atomic_write_text(out_dir / "eligibility.txt", eligibility_output)
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

            if rc.print_judge_outputs:
                print(
                    f"\n[RAW ELIGIBILITY] mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id} attempt={attempt}\n"
                    f"{eligibility_output}\n"
                )

            if not (eligibility_output or "").strip():
                last_err = RuntimeError("Eligibility output is empty.")
                if rc.debug_parse:
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir,
                        kind="relevance",
                        mode=str(rc.mode),
                        patient_id=rc.patient_id,
                        trial_id=rc.trial_id,
                        attempt=attempt,
                        text=relevance_output,
                    )
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir,
                        kind="eligibility",
                        mode=str(rc.mode),
                        patient_id=rc.patient_id,
                        trial_id=rc.trial_id,
                        attempt=attempt,
                        text=eligibility_output,
                    )
                continue

            if _looks_truncated_tag_output(eligibility_output):
                last_err = RuntimeError("Eligibility output looks truncated (tag not closed / empty).")
                if rc.debug_parse:
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir,
                        kind="relevance",
                        mode=str(rc.mode),
                        patient_id=rc.patient_id,
                        trial_id=rc.trial_id,
                        attempt=attempt,
                        text=relevance_output,
                    )
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir,
                        kind="eligibility",
                        mode=str(rc.mode),
                        patient_id=rc.patient_id,
                        trial_id=rc.trial_id,
                        attempt=attempt,
                        text=eligibility_output,
                    )
                continue

            try:
                _ensure_parseable_or_raise(relevance_output, eligibility_output)
            except Exception as e:
                last_err = e
                if rc.debug_parse:
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir, kind="relevance",
                        mode=str(rc.mode), patient_id=rc.patient_id, trial_id=rc.trial_id,
                        attempt=attempt, text=relevance_output,
                    )
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir, kind="eligibility",
                        mode=str(rc.mode), patient_id=rc.patient_id, trial_id=rc.trial_id,
                        attempt=attempt, text=eligibility_output,
                    )
                continue

            ok_pair, pair_reason = validate_pair_outputs_texts(relevance_output, eligibility_output)
            if not ok_pair:
                last_err = RuntimeError(f"Generated pair invalid before save: {pair_reason}")
                if rc.debug_parse:
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir, kind="relevance",
                        mode=str(rc.mode), patient_id=rc.patient_id, trial_id=rc.trial_id,
                        attempt=attempt, text=relevance_output,
                    )
                    _dump_parse_debug(
                        debug_dir=rc.debug_parse_dir, kind="eligibility",
                        mode=str(rc.mode), patient_id=rc.patient_id, trial_id=rc.trial_id,
                        attempt=attempt, text=eligibility_output,
                    )
                continue

            cache.save(cache_key, expected_meta, relevance_output, eligibility_output)
            atomic_write_text(out_dir / "relevance.txt", relevance_output)
            atomic_write_text(out_dir / "eligibility.txt", eligibility_output)
            return relevance_output, eligibility_output

        raise RuntimeError(
            f"Judge outputs were not strictly parseable after retry for "
            f"mode={rc.mode} patient={rc.patient_id} trial={rc.trial_id}. "
            f"Last error: {last_err}"
        )


# =============================================================================
# TrialGPT ref loading + normalization
# =============================================================================

def _as_trial_list(x) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, list):
        out: List[str] = []
        for it in x:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict):
                for k in ("trial_id", "_id", "id", "nct_id", "NCT"):
                    v = it.get(k)
                    if isinstance(v, str) and v:
                        out.append(v)
                        break
        return out

    if isinstance(x, dict):
        for k in ("trials", "ranking", "ranked_trials", "trial_ids", "results"):
            if k in x:
                return _as_trial_list(x.get(k))
    return None


def load_trialgpt_ref(path: Path) -> Dict[str, List[str]]:
    obj = load_json(path)
    mapping: Dict[str, List[str]] = {}

    if isinstance(obj, dict):
        all_keys_are_patients = True
        for k, v in obj.items():
            if not isinstance(k, str):
                all_keys_are_patients = False
                break
            lst = _as_trial_list(v)
            if lst is None:
                all_keys_are_patients = False
                break
            mapping[k] = lst
        if all_keys_are_patients and mapping:
            return mapping

        if "patients" in obj and isinstance(obj["patients"], list):
            for rec in obj["patients"]:
                if not isinstance(rec, dict):
                    continue
                pid = rec.get("patient_id") or rec.get("_id") or rec.get("id")
                if not isinstance(pid, str) or not pid:
                    continue
                lst = _as_trial_list(rec) or _as_trial_list(rec.get("trials"))
                if lst is None:
                    continue
                mapping[pid] = lst
            if mapping:
                return mapping

    if isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            pid = rec.get("patient_id") or rec.get("_id") or rec.get("id")
            if not isinstance(pid, str) or not pid:
                continue
            lst = _as_trial_list(rec) or _as_trial_list(rec.get("trials"))
            if lst is None:
                continue
            mapping[pid] = lst
        if mapping:
            return mapping

    raise ValueError(f"Unrecognized TrialGPT ref JSON format: {path}")


# =============================================================================
# Compute m/n/o from SMT-augmented label JSONs
# =============================================================================

LABEL_ALL_SAT = "all_satisfied"
LABEL_UNSAT_INCL = "unsatisfied_inclusion"
LABEL_EXPL_CONTRA = "explicit_contradiction"


@dataclass
class MNOStats:
    patient_id: str
    m_i: int
    n_i: int
    o_i: int


def compute_mno_from_label_obj(label_obj: dict) -> MNOStats:
    pid = label_obj.get("patient_id")
    if not isinstance(pid, str) or not pid:
        raise ValueError("Label JSON missing patient_id")

    trials = label_obj.get("trials", [])
    if not isinstance(trials, list):
        raise ValueError(f"Bad trials field for patient={pid}")

    m_i = 0
    n_i = 0
    o_i = 0
    for t in trials:
        if not isinstance(t, dict):
            continue
        lab = t.get("label")
        if lab == LABEL_ALL_SAT:
            m_i += 1
            n_i += 1
            o_i += 1
        elif lab == LABEL_UNSAT_INCL:
            n_i += 1
            o_i += 1
        elif lab == LABEL_EXPL_CONTRA:
            o_i += 1

    return MNOStats(patient_id=pid, m_i=m_i, n_i=n_i, o_i=o_i)


def mean_int(xs: List[int]) -> float:
    return 0.0 if not xs else float(sum(xs)) / float(len(xs))


def round_nonneg(x: float) -> int:
    if x <= 0:
        return 0
    return int(round(x))


def floor_nonneg(x: float) -> int:
    if x <= 0:
        return 0
    return int(math.floor(x))


def ceil_nonneg(x: float) -> int:
    if x <= 0:
        return 0
    return int(math.ceil(x))


def interval_for_rank(rank: int, M: int, N: int, K: int) -> str:
    if rank <= M:
        return "m"
    if rank <= N:
        return "n"
    if rank <= K:
        return "o"
    return "tail"


# =============================================================================
# Build TrialGPT input label JSONs for top-K
# =============================================================================

def make_trialgpt_input_label_json(patient_id: str, trial_ids: List[str], tag: str) -> dict:
    trials = []
    for i, tid in enumerate(trial_ids, start=1):
        trials.append({"trial_id": tid, "rank": i, "label": tag})
    return {"patient_id": patient_id, "trials": trials}


# =============================================================================
# Progress bar fallback
# =============================================================================

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


# =============================================================================
# Regeneration-on-final-parse helper
# =============================================================================

def ensure_pair_outputs_and_parse(
    *,
    key: PairKey,
    mbench_root: Path,
    final_parse_rerun: str,
    rc: RunConfig,
    engine: AzureInferenceEngine,
    trace: PromptTrace,
    patient_corpus_cache: Dict[Path, Dict[str, dict]],
    trial_corpus_cache: Dict[Path, Dict[str, dict]],
    cache: PairDiskCache,
    get_lock_fn,
) -> Tuple[bool, bool]:
    status = pair_output_status(mbench_root, key)
    out_dir = pair_sub_dir(mbench_root, key)

    if status.ok:
        assert status.rel_bool is not None and status.elig_bool is not None
        return status.rel_bool, status.elig_bool

    should_rerun = False
    if final_parse_rerun == "never":
        should_rerun = False
    elif final_parse_rerun == "missing":
        should_rerun = status.reason in ("missing_file", "empty_file")
    elif final_parse_rerun == "parse_error":
        should_rerun = status.reason in ("missing_file", "empty_file", "truncated_output", "strict_parse_failed")
    else:
        raise ValueError(f"Unknown final_parse_rerun={final_parse_rerun}")

    if not should_rerun:
        if status.reason == "empty_file":
            print(
                f"[EMPTY->FALSE WITHOUT RERUN] mode={key.mode} patient={key.patient_id} "
                f"parent={key.parent_trial_id} sub={key.subcohort_id}"
            )
            return write_canonical_false_outputs_for_empty_pair(out_dir)
        raise RuntimeError(
            f"[FINAL PARSE FAILURE] key={key} reason={status.reason} "
            f"(policy final_parse_rerun={final_parse_rerun})"
        )

    print(
        f"[RERUN-PAIR] mode={key.mode} patient={key.patient_id} parent={key.parent_trial_id} sub={key.subcohort_id} "
        f"reason={status.reason}"
    )
    delete_pair_outputs(out_dir)

    rc2 = RunConfig(
        patient_id=key.patient_id,
        trial_id=key.subcohort_id,
        patient_note_path=rc.patient_note_path,
        trial_description_path=rc.trial_description_path,
        mode=rc.mode,
        debug_parse=rc.debug_parse,
        debug_parse_dir=rc.debug_parse_dir,
        print_judge_outputs=rc.print_judge_outputs,
        prompt_update_rerun="all",
        max_judge_attempts=max(rc.max_judge_attempts, 5),
        force_no_cache=True,
    )

    ck = CacheKey(mode=key.mode, patient_id=key.patient_id, trial_id=key.subcohort_id)
    try:
        run_one_pair(
            rc=rc2,
            engine=engine,
            trace=trace,
            patient_corpus_cache=patient_corpus_cache,
            trial_corpus_cache=trial_corpus_cache,
            cache=cache,
            key_lock=get_lock_fn(ck),
            out_dir=out_dir,
        )
    except Exception as e:
        status_after_exception = pair_output_status(mbench_root, key)
        if status_after_exception.reason == "empty_file":
            print(
                f"[RERUN FAILED BUT EMPTY->FALSE] mode={key.mode} patient={key.patient_id} "
                f"parent={key.parent_trial_id} sub={key.subcohort_id} err={e}"
            )
            return write_canonical_false_outputs_for_empty_pair(out_dir)
        raise

    status2 = pair_output_status(mbench_root, key)
    if status2.ok and status2.rel_bool is not None and status2.elig_bool is not None:
        return status2.rel_bool, status2.elig_bool

    if status2.reason == "empty_file":
        print(
            f"[FINAL EMPTY->FALSE] mode={key.mode} patient={key.patient_id} "
            f"parent={key.parent_trial_id} sub={key.subcohort_id}"
        )
        return write_canonical_false_outputs_for_empty_pair(out_dir)

    raise RuntimeError(
        f"[FINAL PARSE FAILURE AFTER RERUN] key={key} "
        f"reason_before={status.reason} reason_after={status2.reason}"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--run-modes",
        type=str,
        default="both",
        choices=["both", "ccr", "all", "all-explore", "triple"],
        help="Which modes to run if --modes is unset.",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="(Optional) Comma-separated modes to run (subset of ccr,all,all-explore). If set, overrides --run-modes.",
    )

    ap.add_argument("--smt-output-root", type=str, default="./smt_retrieval_eval_out")
    ap.add_argument("--num-patients", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--trialgpt-ref", type=str, default="../ops/trialgptref/trialgpt_retrieve.json")
    ap.add_argument("--default-patient-note-path", type=str, default="../../dataset/clinical_trial/sigir/queries.jsonl")
    ap.add_argument("--default-trial-description-path", type=str, default="../../dataset/clinical_trial/sigir/corpus.jsonl")

    ap.add_argument("--model-name", type=str, default="gpt-4.1")
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--debug-prompts", action="store_true")

    ap.add_argument("--trialgpt-output-root", type=str, default="./trialgpt_retrieval_eval_out")
    ap.add_argument("--trialgpt-mbench-root", type=str, default="./trialgpt_retrieval_eval_mbench")

    ap.add_argument("--cache-root", type=str, default="./_shared_pair_cache/pair_cache")
    ap.add_argument(
        "--prompt-update-rerun",
        type=str,
        default="all",
        choices=["all", "relevant"],
    )
    ap.add_argument(
        "--final-parse-rerun",
        type=str,
        default="parse_error",
        choices=["never", "missing", "parse_error"],
        help="When final strict re-parse sees missing/broken mbench outputs, whether to regenerate that pair.",
    )
    ap.add_argument(
        "--max-judge-attempts",
        type=int,
        default=3,
        help="Max attempts inside run_one_pair before giving up on that pair.",
    )

    ap.add_argument("--repair-mbench", action="store_true")
    ap.add_argument("--eval-k-from", type=str, default="o_round", choices=["o_round", "o_floor", "o_ceil", "fixed"])
    ap.add_argument("--eval-k", type=int, default=0)
    ap.add_argument("--min-k", type=int, default=1)

    ap.add_argument("--debug-parse", action="store_true")
    ap.add_argument("--debug-parse-dir", type=str, default="./_debug_parse_dump")
    ap.add_argument("--print-judge-outputs", action="store_true")

    args = ap.parse_args()

    if args.debug_prompts:
        _debug_print_prompt_constants()

    if args.modes.strip():
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        if args.run_modes == "both":
            modes = ["ccr", "all"]
        elif args.run_modes == "triple":
            modes = ["ccr", "all", "all-explore"]
        elif args.run_modes == "ccr":
            modes = ["ccr"]
        elif args.run_modes == "all":
            modes = ["all"]
        else:
            modes = ["all-explore"]

    for m in modes:
        if m not in ("ccr", "all", "all-explore"):
            raise ValueError(f"Unknown mode: {m}")

    smt_output_root = Path(args.smt_output_root).resolve()
    trialgpt_ref_path = Path(args.trialgpt_ref).resolve()
    ref = load_trialgpt_ref(trialgpt_ref_path)

    patient_corpus_path = Path(args.default_patient_note_path).resolve()
    trial_corpus_path = Path(args.default_trial_description_path).resolve()
    trial_corpus = load_jsonl_as_dict(trial_corpus_path)

    trialgpt_output_root = Path(args.trialgpt_output_root).resolve()
    trialgpt_mbench_root = Path(args.trialgpt_mbench_root).resolve()
    trialgpt_mbench_root.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root).resolve()
    cache_root.parent.mkdir(parents=True, exist_ok=True)

    if args.repair_mbench:
        n_deleted = repair_mbench_pairs(trialgpt_mbench_root, modes)
        print(f"[REPAIR] deleted outputs for {n_deleted} broken subcohort dirs; will regenerate as needed.")

    engine = build_engine(args.model_name)
    trace = PromptTrace(trialgpt_mbench_root / "prompts_and_outputs_dump.txt", tokenizer_model=args.model_name)
    cache = PairDiskCache(cache_root)

    patient_corpus_cache: Dict[Path, Dict[str, dict]] = {}
    trial_corpus_cache: Dict[Path, Dict[str, dict]] = {trial_corpus_path: trial_corpus}

    locks: Dict[CacheKey, threading.Lock] = {}
    locks_guard = threading.Lock()

    def get_lock(k: CacheKey) -> threading.Lock:
        with locks_guard:
            if k not in locks:
                locks[k] = threading.Lock()
            return locks[k]

    rng = random.Random(args.seed)
    debug_parse_dir = Path(args.debug_parse_dir).resolve()

    for mode in modes:
        smt_labels_dir = smt_output_root / mode / "patient_labels"
        if not smt_labels_dir.exists():
            raise FileNotFoundError(f"Missing SMT labels dir: {smt_labels_dir}")

        label_files = list_json_files(smt_labels_dir)
        if not label_files:
            raise RuntimeError(f"No label JSON files under {smt_labels_dir}")

        chosen_files = label_files if len(label_files) <= args.num_patients else rng.sample(label_files, args.num_patients)
        chosen_files = sorted(chosen_files)

        stats: List[MNOStats] = []
        for p in chosen_files:
            obj = load_json(p)
            stats.append(compute_mno_from_label_obj(obj))

        m_avg = mean_int([s.m_i for s in stats])
        n_avg = mean_int([s.n_i for s in stats])
        o_avg = mean_int([s.o_i for s in stats])

        if args.eval_k_from == "fixed":
            K = int(args.eval_k)
        elif args.eval_k_from == "o_floor":
            K = floor_nonneg(o_avg)
        elif args.eval_k_from == "o_ceil":
            K = ceil_nonneg(o_avg)
        else:
            K = round_nonneg(o_avg)

        if K < args.min_k:
            K = int(args.min_k)

        M = max(0, round_nonneg(m_avg))
        N = max(M, round_nonneg(n_avg))

        print(f"\n=== [MODE {mode}] ===")
        print("[MNO] per-patient:")
        for s in stats:
            print(f"  patient={s.patient_id} m_i={s.m_i} n_i={s.n_i} o_i={s.o_i}")
        print(f"[MNO] averages over {len(stats)} patients: m={m_avg:.3f} n={n_avg:.3f} o={o_avg:.3f}")
        print(f"[MNO] rounded thresholds: M={M} N={N} K(o-based)={K}")

        patient_ids = [s.patient_id for s in stats]

        input_labels_dir = trialgpt_output_root / "_input_labels" / mode / "patient_labels"
        input_labels_dir.mkdir(parents=True, exist_ok=True)

        kept_patient_ids: List[str] = []
        skipped_missing_ref: List[str] = []
        skipped_empty_ref: List[str] = []

        for pid in patient_ids:
            if pid not in ref:
                skipped_missing_ref.append(pid)
                print(f"[WARN] Skipping patient not present in TrialGPT ref: {pid}")
                continue

            trial_ids = ref[pid][:K]
            if not trial_ids:
                skipped_empty_ref.append(pid)
                print(f"[WARN] Skipping patient with empty TrialGPT ref list: {pid}")
                continue

            kept_patient_ids.append(pid)
            obj = make_trialgpt_input_label_json(pid, trial_ids, tag=f"trialgpt_ref_top{K}")
            write_json(input_labels_dir / f"{pid}__{mode}__trialgpt_ref_top{K}.json", obj)

        if skipped_missing_ref:
            print(
                f"[WARN] Skipped {len(skipped_missing_ref)} patients missing from TrialGPT ref file: "
                f"{trialgpt_ref_path}"
            )

        if skipped_empty_ref:
            print(f"[WARN] Skipped {len(skipped_empty_ref)} patients with empty TrialGPT ref lists.")

        if not kept_patient_ids:
            raise RuntimeError(
                f"After skipping missing/empty TrialGPT ref patients, no patients remain for mode={mode}."
            )

        label_paths = list_json_files(input_labels_dir)
        if not label_paths:
            raise RuntimeError(f"No input labels found under {input_labels_dir}")

        total_pairs = 0
        for lp in label_paths:
            try:
                obj = load_json(lp)
                trials = obj.get("trials", [])
                if isinstance(trials, list):
                    for t in trials:
                        parent_trial_id = t.get("trial_id")
                        if isinstance(parent_trial_id, str) and parent_trial_id:
                            total_pairs += len(expand_to_subcohorts(parent_trial_id, trial_corpus))
            except Exception:
                pass

        if tqdm is not None:
            pbar = tqdm(total=total_pairs, desc=f"TrialGPT judge pairs ({mode})", unit="pair", dynamic_ncols=True)
        else:
            pbar = _SimpleProgress(total=total_pairs, desc=f"TrialGPT judge pairs ({mode})")

        def process_one_label_file(lp: Path) -> None:
            label_obj = load_json(lp)
            pid = label_obj["patient_id"]
            trials = label_obj.get("trials", [])
            if not isinstance(trials, list):
                raise ValueError(f"Bad trials field in {lp}")

            for t in trials:
                parent_trial_id = t["trial_id"]
                sub_ids = expand_to_subcohorts(parent_trial_id, trial_corpus)

                for sub_id in sub_ids:
                    rc = RunConfig(
                        patient_id=pid,
                        trial_id=sub_id,
                        patient_note_path=patient_corpus_path,
                        trial_description_path=trial_corpus_path,
                        mode=mode,  # type: ignore[arg-type]
                        debug_parse=bool(args.debug_parse),
                        debug_parse_dir=debug_parse_dir,
                        print_judge_outputs=bool(args.print_judge_outputs),
                        prompt_update_rerun=str(args.prompt_update_rerun),
                        max_judge_attempts=int(args.max_judge_attempts),
                        force_no_cache=False,
                    )
                    ck = CacheKey(mode=mode, patient_id=pid, trial_id=sub_id)  # type: ignore[arg-type]
                    out_dir = trialgpt_mbench_root / mode / pid / parent_trial_id / sub_id
                    run_one_pair(
                        rc=rc,
                        engine=engine,
                        trace=trace,
                        patient_corpus_cache=patient_corpus_cache,
                        trial_corpus_cache=trial_corpus_cache,
                        cache=cache,
                        key_lock=get_lock(ck),
                        out_dir=out_dir,
                    )
                    pbar.update(1)

        try:
            if args.num_workers <= 1:
                for lp in label_paths:
                    print(f"[RUN] mode={mode} file={lp.name}")
                    process_one_label_file(lp)
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                    futs = {ex.submit(process_one_label_file, lp): lp for lp in label_paths}
                    for fut in as_completed(futs):
                        lp = futs[fut]
                        try:
                            fut.result()
                        except Exception:
                            import traceback
                            print(f"[ERROR] mode={mode} file={lp}")
                            print(traceback.format_exc())
                            raise
        finally:
            try:
                pbar.close()
            except Exception:
                pass

        final_out_dir = trialgpt_output_root / mode / "patient_labels"
        final_out_dir.mkdir(parents=True, exist_ok=True)

        for lp in label_paths:
            label_obj = load_json(lp)
            pid = label_obj["patient_id"]
            trials = label_obj.get("trials", [])
            if not isinstance(trials, list):
                raise ValueError(f"Bad trials field in {lp}")

            for t in trials:
                parent_trial_id = t["trial_id"]
                sub_ids = expand_to_subcohorts(parent_trial_id, trial_corpus)
                if not sub_ids:
                    raise FileNotFoundError(f"No subcohorts found for parent trial: {parent_trial_id}")

                any_rel = False
                any_elig = False
                any_rel_and_elig = False
                summaries: List[dict] = []

                for sub_id in sub_ids:
                    key = PairKey(
                        mode=mode,
                        patient_id=pid,
                        parent_trial_id=parent_trial_id,
                        subcohort_id=sub_id,
                    )
                    rc = RunConfig(
                        patient_id=pid,
                        trial_id=sub_id,
                        patient_note_path=patient_corpus_path,
                        trial_description_path=trial_corpus_path,
                        mode=mode,  # type: ignore[arg-type]
                        debug_parse=bool(args.debug_parse),
                        debug_parse_dir=debug_parse_dir,
                        print_judge_outputs=bool(args.print_judge_outputs),
                        prompt_update_rerun=str(args.prompt_update_rerun),
                        max_judge_attempts=int(args.max_judge_attempts),
                        force_no_cache=False,
                    )

                    rel_b, elig_b = ensure_pair_outputs_and_parse(
                        key=key,
                        mbench_root=trialgpt_mbench_root,
                        final_parse_rerun=str(args.final_parse_rerun),
                        rc=rc,
                        engine=engine,
                        trace=trace,
                        patient_corpus_cache=patient_corpus_cache,
                        trial_corpus_cache=trial_corpus_cache,
                        cache=cache,
                        get_lock_fn=get_lock,
                    )

                    any_rel = any_rel or rel_b
                    any_elig = any_elig or elig_b
                    any_rel_and_elig = any_rel_and_elig or (rel_b and elig_b)

                    summaries.append({"subcohort_id": sub_id, "relevant": rel_b, "eligible": elig_b})

                t["any_subcohort_relevant"] = any_rel
                t["any_subcohort_eligible"] = any_elig
                t["any_subcohort_relevant_and_eligible"] = any_rel_and_elig
                t["subcohort_judge_summary"] = summaries

                rank = t.get("rank")
                if isinstance(rank, int):
                    t["interval_mno"] = interval_for_rank(rank, M=M, N=N, K=K)

            write_json(final_out_dir / lp.name, label_obj)

        print(f"[DONE] mode={mode} augmented TrialGPT labels in: {final_out_dir}")

    print(
        f"\n[TOKENS] prompt={trace.totals.prompt_tokens} "
        f"completion={trace.totals.completion_tokens} "
        f"total={trace.totals.total}"
    )
    print(f"[DONE] TrialGPT mbench outputs in:  {Path(args.trialgpt_mbench_root).resolve()}")
    print(f"[DONE] TrialGPT outputs in:         {Path(args.trialgpt_output_root).resolve()}")
    print(f"[DONE] shared cache in:             {cache_root}")


if __name__ == "__main__":
    main()