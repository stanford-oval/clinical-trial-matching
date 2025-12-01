#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
- Fix eligibility parsing for new prompt output:
    <subcohort_eligibility_decisions> [ { "eligibility_decision": "eligible"/"ineligible", ... }, ... ] </...>
  Eligibility is True iff ANY dict says eligible (not merely list non-empty).
- Fix relevance parsing for your new relevance output:
    <relevant_subcohorts> [ { ... }, ... ] </...>
  Relevance is True iff the relevant_subcohorts list is non-empty (list[dict] supported).
- Optionally normalizes curly quotes inside tagged JSON payloads so json.loads works on older artifacts.
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
from typing import Any, Dict, List, Optional, Tuple, Union

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

    # Likely bases in this repo layout
    script_dir = Path(__file__).resolve().parent
    bases = [
        Path.cwd(),
        script_dir,                               # irsrc/eval/
        script_dir.parent,                        # irsrc/
        script_dir.parent / "eval",               # irsrc/eval/
        script_dir.parent / "meval",              # irsrc/meval/
        script_dir.parent / "meval" / "prompts",  # irsrc/meval/prompts/
        script_dir / "prompts",                   # irsrc/eval/prompts/
    ]

    for b in bases:
        cand = (b / pp).resolve()
        if cand.exists() and cand.is_file():
            return cand

    # fall back: cwd-relative Path object (still Path, may not exist)
    return (Path.cwd() / pp).resolve()


def _normalize_prompt_constants_in_judges() -> None:
    """
    Convert prompt-path constants in judge modules from str -> Path, and resolve if possible.
    """
    if hasattr(jr, "RELEVANCE_BASE_PROMPT_PATH"):
        jr.RELEVANCE_BASE_PROMPT_PATH = _resolve_prompt_path(jr.RELEVANCE_BASE_PROMPT_PATH)

    if hasattr(jr, "RELEVANCE_DEFINITION_PROMPT_PATHS"):
        d = jr.RELEVANCE_DEFINITION_PROMPT_PATHS
        try:
            for k, v in list(d.items()):
                d[k] = _resolve_prompt_path(v)
        except Exception:
            pass

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
        print("[DEBUG] jr.RELEVANCE_DEFINITION_PROMPT_PATHS keys:", list(jr.RELEVANCE_DEFINITION_PROMPT_PATHS.keys()))
        for k in ("ccr", "all", "cc"):
            if k in jr.RELEVANCE_DEFINITION_PROMPT_PATHS:
                print(f"[DEBUG] jr.RELEVANCE_DEFINITION_PROMPT_PATHS[{k!r}]:",
                      _fmt(jr.RELEVANCE_DEFINITION_PROMPT_PATHS[k]))
    except Exception as e:
        print("[DEBUG] could not inspect RELEVANCE_DEFINITION_PROMPT_PATHS:", e)

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
        default_temperature=0.0,
        default_max_tokens=None,
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
    """
    Returns (kind, sha):
      kind="file" if we hashed file contents
      kind="literal" if we hashed the string representation of p
    """
    f = _resolve_existing_file(p)
    if f is not None:
        h = hashlib.sha256()
        with f.open("rb") as fin:
            for chunk in iter(lambda: fin.read(1024 * 1024), b""):
                h.update(chunk)
        return ("file", h.hexdigest())
    return ("literal", _sha256_bytes(str(p).encode("utf-8")))


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
    elig_prompt: PromptPathLike,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int],
) -> dict:
    """
    Adapt to whatever PairDiskCache.compute_meta signature is in this repo.

    CRITICAL for this repo:
      - If compute_meta wants relevance_prompt_path / eligibility_prompt_path, pass Path objects,
        because cache_utils.sha256_file expects Path (uses path.open()).
    """
    sig = inspect.signature(cache.compute_meta)
    param_names = [p for p in sig.parameters.keys() if p != "self"]

    base = {
        "patient_text": patient_text,
        "trial_text": trial_text,
        "model_name": str(model_name),
        "temperature": float(temperature),
        "max_tokens": None if max_tokens is None else int(max_tokens),
    }

    rb_kind, rb_sha = _sha256_file_or_literal(rel_base_prompt)
    rd_kind, rd_sha = _sha256_file_or_literal(rel_def_prompt)
    el_kind, el_sha = _sha256_file_or_literal(elig_prompt)

    prompt_sha = hashlib.sha256(
        ("\n".join([rb_kind + ":" + rb_sha, rd_kind + ":" + rd_sha, el_kind + ":" + el_sha])).encode("utf-8")
    ).hexdigest()

    rel_def_path = _to_path(rel_def_prompt)
    elig_path = _to_path(elig_prompt)
    rb_path = _to_path(rel_base_prompt)

    prompt_paths_path = [rb_path, rel_def_path, elig_path]
    prompt_paths_str = [str(rb_path), str(rel_def_path), str(elig_path)]
    prompt_path_joined = "|".join(prompt_paths_str)

    # Variant A: compute_meta(payload_dict)
    if len(param_names) == 1:
        payload = dict(base)
        payload.update({
            "prompt_paths": prompt_paths_str,
            "prompt_path": prompt_path_joined,
            "prompt_sha": prompt_sha,
            "relevance_base_prompt_path": str(rb_path),
            "relevance_definition_prompt_path": str(rel_def_path),
            "eligibility_prompt_path": str(elig_path),
            # Path forms (this repo needs Path for sha256_file())
            "relevance_prompt_path": rel_def_path,
            "eligibility_prompt_path": elig_path,
        })
        return cache.compute_meta(payload)  # type: ignore[arg-type]

    # Variant B: compute_meta(**kwargs)
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
        kwargs["relevance_prompt_path"] = rel_def_path  # Path
    if "eligibility_prompt_path" in sig.parameters:
        kwargs["eligibility_prompt_path"] = elig_path   # Path

    if "relevance_prompt_sha" in sig.parameters:
        kwargs["relevance_prompt_sha"] = rb_sha + ":" + rd_sha
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
# Parsing helpers (FIXED)
# -------------------------

def _extract_tagged_payload(text: str, tag: str) -> Optional[str]:
    """
    Extract inner content between <tag> ... </tag>, tolerant to missing final '>' on closing tag.
    """
    s = text.strip()
    m = re.search(
        rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*(?:>|$)",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()


def _normalize_jsonish_payload(payload: str) -> str:
    """
    Normalize curly quotes for older artifacts so json.loads is less likely to fail.
    """
    return (
        payload
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )


def _json_parse_best_effort(payload: str) -> Any:
    payload = _normalize_jsonish_payload(payload)
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


def _coerce_boolish_str(s: str) -> Optional[bool]:
    sl = s.strip().lower()
    if sl in ("true", "yes", "y", "relevant", "eligible"):
        return True
    if sl in ("false", "no", "n", "irrelevant", "not relevant", "ineligible", "not eligible"):
        return False
    return None


def _extract_bool_from_obj(obj: Any, keys: List[str]) -> Optional[bool]:
    """
    For dict/object payloads only.
    IMPORTANT: does NOT treat list length as boolean; list semantics handled separately.
    """
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
                    # nested best-effort
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
    """
    Eligibility list semantics:
      - [] => False
      - list[str] => non-empty => True  (assumed positive list)
      - list[dict] => True iff ANY dict indicates true via decision keys
    """
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
    """
    Best-effort relevance parse (FIXED for list[dict] under <relevant_subcohorts>).

    Priority:
    1) <relevant_subcohorts> LIST </...> => True iff list non-empty (dicts allowed)
    2) JSON list at start / after Output: / last list => True iff list non-empty
    3) JSON object fallback
    4) token fallback
    """
    s = relevance_output.strip()

    # 1) tagged relevant_subcohorts: ANY NON-EMPTY LIST => True
    payload = _extract_tagged_payload(s, "relevant_subcohorts")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                return len(obj) > 0
        except Exception:
            pass

    # 2) JSON list fallbacks
    obj0, _ = _json_leading_value(s)
    if isinstance(obj0, list):
        return len(obj0) > 0

    obj1 = _json_value_after_marker(s, r"\boutput\s*:\s*")
    if isinstance(obj1, list):
        return len(obj1) > 0

    obj2 = _json_last_list_anywhere(s)
    if isinstance(obj2, list):
        return len(obj2) > 0

    # 3) JSON object fallback
    try:
        obj = json.loads(_normalize_jsonish_payload(s))
        b = _extract_bool_from_obj(obj, ["relevant", "is_relevant", "relevance", "decision", "relevance_decision"])
        if b is not None:
            return b
    except Exception:
        pass

    # 4) Token fallback
    tl = s.lower()
    if "not relevant" in tl or "irrelevant" in tl:
        return False
    if re.search(r"(^|\W)relevant(\W|$)", tl) and not re.search(r"\bnot\s+relevant\b", tl):
        return True

    return None


def parse_eligibility_bool(elig_output: str) -> Optional[bool]:
    """
    Best-effort eligibility parse (FIXED for list[dict] under <subcohort_eligibility_decisions>).

    Priority:
    1) <subcohort_eligibility_decisions> LIST </...> => True iff ANY dict says eligible
    2) <eligible_subcohorts> LIST </...> (legacy) => True iff list non-empty (or dicts include eligible)
    3) JSON list at start / after Output: / last list => apply eligibility list semantics
    4) JSON object fallback
    5) narrative-only => False
    6) token fallback
    """
    s = elig_output.strip()

    # 1) NEW tag: subcohort_eligibility_decisions
    payload = _extract_tagged_payload(s, "subcohort_eligibility_decisions")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["eligibility_decision", "eligibilityDecision", "decision", "eligible", "is_eligible"],
                    true_tokens=["eligible", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    # 2) legacy tag: eligible_subcohorts
    payload = _extract_tagged_payload(s, "eligible_subcohorts")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                # If list[str], non-empty => True; if list[dict], any eligible => True.
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
                    true_tokens=["eligible", "true", "yes", "y"],
                )
                if b is not None:
                    return b
                # if it's list[str] (already handled) or empty (handled); otherwise fall through.
        except Exception:
            pass

    # 3) JSON list fallbacks
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

    # 4) JSON object fallback
    try:
        obj = json.loads(_normalize_jsonish_payload(s))
        b = _extract_bool_from_obj(
            obj,
            ["eligible", "is_eligible", "eligibility", "decision", "eligibility_decision", "eligibilityDecision"],
        )
        if b is not None:
            return b
    except Exception:
        pass

    tl = s.lower()

    # 5) Narrative-only => False
    if re.search(r"\bno\s+relevant\s+subcohorts?\b", tl):
        return False
    if re.search(r"\bno\s+eligibility\s+decisions?\s+are\s+required\b", tl):
        return False
    if re.search(r"\bno\s+eligibility\s+assessment\s+is\s+required\b", tl):
        return False
    if re.search(r"\bthere\s+are\s+no\s+subcohorts?\s+to\s+evaluate\b", tl):
        return False

    # 6) Tokens
    if "ineligible" in tl or "not eligible" in tl:
        return False
    if re.search(r"(^|\W)eligible(\W|$)", tl) and "ineligible" not in tl and "not eligible" not in tl:
        return True

    return None


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

    rel_base_prompt = jr.RELEVANCE_BASE_PROMPT_PATH
    rel_def_prompt = jr.RELEVANCE_DEFINITION_PROMPT_PATHS[rc.mode]
    elig_prompt = je.ELIGIBILITY_PROMPT_PATH

    model_name = getattr(engine, "model_name", "gpt-4.1")
    temperature = float(getattr(engine, "default_temperature", 0.0))
    max_tokens = getattr(engine, "default_max_tokens", None)
    max_tokens_int = None if max_tokens is None else int(max_tokens)

    expected_meta = compute_expected_meta_compat(
        cache,
        patient_text=patient_text,
        trial_text=trial_text,
        rel_base_prompt=rel_base_prompt,
        rel_def_prompt=rel_def_prompt,
        elig_prompt=elig_prompt,
        model_name=str(model_name),
        temperature=temperature,
        max_tokens=max_tokens_int,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    with key_lock:
        cached = cache.load_if_fresh(cache_key, expected_meta)
        if cached is not None:
            (out_dir / "relevance.txt").write_text(cached["relevance"], encoding="utf-8")
            (out_dir / "eligibility.txt").write_text(cached["eligibility"], encoding="utf-8")
            return cached["relevance"], cached["eligibility"]

        relevance_output = judge_relevance(
            trial_text,
            patient_text,
            engine=engine,
            prompt_key=rc.mode,
            trace=trace,
            patient_id=rc.patient_id,
            trial_id=rc.trial_id,
        )

        eligibility_output = judge_eligibility(
            trial_text,
            patient_text,
            relevance_result=relevance_output,
            engine=engine,
            trace=trace,
            patient_id=rc.patient_id,
            trial_id=rc.trial_id,
        )

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


def pick_patients(
    ccr_dir: Optional[Path],
    all_dir: Optional[Path],
    sample_n: int,
    seed: int,
    shared: bool,
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)

    if ccr_dir and all_dir and shared:
        ccr_files = list_patient_label_files(ccr_dir)
        all_files = list_patient_label_files(all_dir)

        ccr_ids = {p.stem.split("__", 1)[0]: p for p in ccr_files}
        all_ids = {p.stem.split("__", 1)[0]: p for p in all_files}

        inter = sorted(set(ccr_ids.keys()) & set(all_ids.keys()))
        if not inter:
            raise RuntimeError("No intersection patients between CCR and ALL patient_labels dirs.")
        chosen = inter if len(inter) <= sample_n else rng.sample(inter, sample_n)
        return [ccr_ids[i] for i in chosen], [all_ids[i] for i in chosen]

    ccr_list: List[Path] = []
    all_list: List[Path] = []

    if ccr_dir:
        files = list_patient_label_files(ccr_dir)
        if not files:
            raise RuntimeError(f"No .json files found in {ccr_dir}")
        ccr_list = files if len(files) <= sample_n else rng.sample(files, sample_n)

    if all_dir:
        files = list_patient_label_files(all_dir)
        if not files:
            raise RuntimeError(f"No .json files found in {all_dir}")
        all_list = files if len(files) <= sample_n else rng.sample(files, sample_n)

    return ccr_list, all_list


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
    ap.add_argument("--ccr-patient-labels-dir", type=str,
                    default="../ops/out_compose/clean_eval__ccr__prevent/patient_labels")
    ap.add_argument("--all-patient-labels-dir", type=str,
                    default="../ops/out_compose/clean_eval__all__prevent/patient_labels")

    ap.add_argument("--sample-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shared-sample", action="store_true")

    ap.add_argument("--patient-corpus", type=str, default="../../dataset/clinical_trial/sigir/queries.jsonl")
    ap.add_argument("--trial-corpus", type=str, default="../../dataset/clinical_trial/sigir/corpus.jsonl")

    ap.add_argument("--model-name", type=str, default="gpt-4.1")

    ap.add_argument("--output-root", type=str, default="./smt_retrieval_eval_out")
    ap.add_argument("--mbench-root", type=str, default="./smt_retrieval_eval_mbench")
    ap.add_argument("--num-workers", type=int, default=16)

    ap.add_argument("--debug-prompts", action="store_true",
                    help="Print resolved prompt constants and their types at startup.")

    args = ap.parse_args()

    if args.debug_prompts:
        _debug_print_prompt_constants()

    ccr_dir = Path(args.ccr_patient_labels_dir).resolve() if args.ccr_patient_labels_dir else None
    all_dir = Path(args.all_patient_labels_dir).resolve() if args.all_patient_labels_dir else None
    if not ccr_dir and not all_dir:
        raise ValueError("Provide at least one of --ccr-patient-labels-dir or --all-patient-labels-dir")

    patient_corpus_path = Path(args.patient_corpus).resolve()
    trial_corpus_path = Path(args.trial_corpus).resolve()

    output_root = Path(args.output_root).resolve()
    mbench_root = Path(args.mbench_root).resolve()
    mbench_root.mkdir(parents=True, exist_ok=True)

    engine = build_engine(args.model_name)

    trace = PromptTrace(mbench_root / "prompts_and_outputs_dump.txt", tokenizer_model=args.model_name)
    cache = PairDiskCache(mbench_root / "pair_cache")

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

    ccr_files, all_files = pick_patients(ccr_dir, all_dir, args.sample_n, args.seed, args.shared_sample)

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
    jobs += [(p, "ccr") for p in ccr_files]
    jobs += [(p, "all") for p in all_files]

    # Precompute total work units (patient, parent_trial, subcohort) pairs
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


if __name__ == "__main__":
    main()