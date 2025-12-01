#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjudicate_smt_vs_nl.py  (extended)

NEW in this version:
- You can choose which TWO systems to compare:
    --compare {smt_vs_nl, trialgpt_vs_nl, smt_vs_trialgpt}

- If a side is SMT or TrialGPT, we can (optionally) verbalize it before adjudication:
    --run-verbalizer         (verbalize the "evaluated" side if it's SMT/TrialGPT)
    --run-adjudicator        (run adjudicator comparing the two systems' decisions+rationales)
    --overwrite              (overwrite existing outputs)

- TrialGPT rationale is derived from:
    <match_out>/<patient_id>/<PARENT_NCT>__trialgpt_judge.json
  which should have the structure produced by your matcher:
    { inclusion:{criteria,rows,...}, exclusion:{...}, aggregate:{eligible,...}, ... }

- Adjudicator output format supported:
    <scratchpad> ... </scratchpad>
    <your_decision> { "decision_over_decisions": ..., "confidence": ..., "brief_rationale": ... } </your_decision>

- Output artifacts:
  results/<patient_id>/<trial_parent_id>__{tag}_rationale.json
  results/<patient_id>/<trial_parent_id>__adjudication.json (+ meta)
  plus JSONL rollups (adjudications.jsonl, verbalizer_ratings.jsonl optional)

Notes:
- This script remains thread-based (ThreadPoolExecutor) for LLM calls.
- It uses your inference_engine / inference_engine_5 wrapper that supports engine(prompt, system_message=...).
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Defaults (templates)
# =============================================================================

# SMT verbalizer template (your newer "reference_assessment" + emphasis requirements)
DEFAULT_SMT_VERBALIZER_TEMPLATE = r"""
# === ROLE ===
You are a careful clinical trial eligibility explainer.

# === GOAL ===
You are given:
- a patient note
- a trial eligibility text
- decision artifacts produced by an anonymous system under evaluation
- extracted patient variable values used by that system
- a reference assessment (decision + reasoning) from an external reviewer for context that disagrees with our evaluated system

Your task: explain, in plain English, why the system concluded the provided eligibility label.

This is ONLY for evaluation. Do NOT suggest fixes. Do NOT propose code changes.
Do NOT reveal or speculate about the system’s implementation.

# === IMPORTANT ===
- Use ONLY evidence available in the provided inputs.
- Do NOT assume missing facts.
- The decision artifacts may include an "unsat_core" list when a side fails.
  Interpret "unsat_core" as a minimal set of named requirements/facts that could not all be satisfied together.
  Use it to explain what requirement(s) failed and which patient fact(s) were missing/contradictory.
- If the artifacts indicate OR semantics across multiple subcohorts/arms, explain the OR logic plainly.
- If unknown/missing patient facts affected the result, state that clearly.
- Keep the rationale concise and checkable. The length of rationale should be similar to the length of reference_assessment_reasoning.
- Even if you believe that the system has made a logical error (e.g., wrong decomposition, wrong requirement interpretation), you should always stick to the logic by which the system comes to the conclusion. That is, you just translate how the decision was made faithfully (that is, you ARGUE for the system's case), without correcting or criticizing how it came to conclusion.

# === EMPHASIS REQUIREMENT (TOP PRIORITY) ===
1) If the evaluated system decision is ineligible, start with the single most decision-driving eligibility requirement(s) and patient fact(s) for the system label.
2) In all cases (eligible or ineligible), prioritize eligibility requirement(s) that are known to be interpreted differently by the reference_assessment_system. These requirements should be emphasized internally because they are the most likely to alter the reference assessment if resolved differently.
3) You MUST NOT mention, quote, or characterize evaluated system or the external reviewer as a "system," and MUST NOT use phrases like "disagree," "another system," "in contrast to," or "compared to."
4) Instead, simply present the key requirement(s) and the exact evidence/unknowns that make the system label checkable. The final rationale must read as a standalone eligibility explanation, not a description of how a system computed the result.

# === OUTPUT REQUIREMENTS ===
Return a SINGLE JSON object (no markdown fences) with exactly these keys:
{
  "rationale": "one concise paragraph",
  "key_points": ["...", "..."],
  "evidence_quotes": {
    "patient_note": ["short quote 1", "short quote 2"],
    "trial_eligibility_text": ["short quote 1", "short quote 2"],
    "high_priority_requirements": ["requirement phrase(s) that drive the label, especially those likely to flip the reference assessment"],
    "system_artifacts": ["very short snippet(s) from SYSTEM_DECISION_ARTIFACTS if helpful"]
  }
}

# === INPUTS ===
<patient_note>
#PATIENT_NOTE#
</patient_note>

<trial_eligibility_text>
#TRIAL_ELIGIBILITY_TEXT#
</trial_eligibility_text>

<system_decision_label>
#SYSTEM_DECISION_LABEL#
</system_decision_label>

<system_decision_artifacts>
#SYSTEM_DECISION_ARTIFACTS#
</system_decision_artifacts>

<inclusion_extracted_variable_values>
#INCLUSION_EXTRACTED_VARIABLE_VALUES#
</inclusion_extracted_variable_values>

<exclusion_extracted_variable_values>
#EXCLUSION_EXTRACTED_VARIABLE_VALUES#
</exclusion_extracted_variable_values>

<reference_assessment_decision>
#NL_DECISION#
</reference_assessment_decision>

<reference_assessment_reasoning>
#NL_RATIONALE#
</reference_assessment_reasoning>
""".lstrip()


# TrialGPT verbalizer template (criterion-level)
DEFAULT_TRIALGPT_VERBALIZER_TEMPLATE = r"""
# === ROLE ===
You are a careful clinical trial eligibility explainer.

# === GOAL ===
You are given:
- a patient note
- a trial eligibility text
- criterion-level decision artifacts produced by an anonymous eligibility matcher
- a reference assessment (decision + reasoning) from an external reviewer for context

Your task: explain, in plain English, why the matcher concluded the provided eligibility label.

This is ONLY for evaluation. Do NOT suggest fixes. Do NOT propose code changes.
Do NOT reveal or speculate about the matcher's implementation.

# === IMPORTANT ===
- Use ONLY evidence available in the provided inputs.
- Do NOT assume missing facts beyond what the criterion-level artifacts imply.
- The artifacts include per-criterion labels for inclusion and exclusion criteria.
  Interpret them as follows:
  - Inclusion:
      "included" supports eligibility.
      "not included" blocks eligibility.
      "not enough information" means unknown.
      "not applicable" means irrelevant.
  - Exclusion:
      "excluded" blocks eligibility.
      "not excluded" supports eligibility.
      "not enough information" means unknown.
      "not applicable" means irrelevant.
- The final label is computed from criterion labels:
  - Inclusion side is satisfied if no criterion is "not included" and all are {included, not applicable}.
  - Exclusion side is satisfied if no criterion is "excluded" and all are {not excluded, not applicable}.
  - If any blocking label exists, final is ineligible.
  - If no blocking label exists but some are unknown, final may be unclear (or optimistic-eligible depending on the given label).
- Keep the rationale concise and checkable. The length of rationale should be similar to the length of reference_assessment_reasoning.
- Even if you believe the matcher made a logical error, stick to the logic implied by the artifacts and label.

# === EMPHASIS REQUIREMENT (TOP PRIORITY) ===
1) If the provided label is ineligible, start with the single most decision-driving criterion (or small set) and the patient evidence/unknowns associated with it.
2) In all cases, prioritize criteria that are likely to flip the reference assessment if resolved differently.
3) You MUST NOT mention, quote, or characterize the evaluated method or the external reviewer as a "system",
   and MUST NOT use phrases like "disagree," "another system," "in contrast to," or "compared to."
4) Present the key criteria and the evidence/unknowns that make the label checkable.

# === OUTPUT REQUIREMENTS ===
Return a SINGLE JSON object (no markdown fences) with exactly these keys:
{
  "rationale": "one concise paragraph",
  "key_points": ["...", "..."],
  "evidence_quotes": {
    "patient_note": ["short quote 1", "short quote 2"],
    "trial_eligibility_text": ["short quote 1", "short quote 2"],
    "high_priority_requirements": ["criterion phrase(s) that drive the label and are likely to flip the reference assessment"],
    "criterion_artifacts": ["very short snippets from CRITERION_DECISION_ARTIFACTS if helpful"]
  }
}

# === INPUTS ===
<patient_note>
#PATIENT_NOTE#
</patient_note>

<trial_eligibility_text>
#TRIAL_ELIGIBILITY_TEXT#
</trial_eligibility_text>

<provided_final_label>
#SYSTEM_DECISION_LABEL#
</provided_final_label>

<criterion_decision_artifacts>
#CRITERION_DECISION_ARTIFACTS#
</criterion_decision_artifacts>

<reference_assessment_decision>
#NL_DECISION#
</reference_assessment_decision>

<reference_assessment_reasoning>
#NL_RATIONALE#
</reference_assessment_reasoning>
""".lstrip()


DEFAULT_ADJUDICATOR_TEMPLATE = r"""
# === ROLE ===
You are an expert clinician who excels at logical reasoning.

# === GOAL ===
You are evaluating the correctness of the decisions concerning patient-trial matching eligibility.
You will be given the decision and rationale (both in natural language) from two candidates.
Your job is to determine which candidate's decision and reasoning is more correct or if both candidates are correct/incorrect.

From START: OVERARCHING MATCHING GUIDELINES to END: OVERARCHING MATCHING GUIDELINES below,
you will be given the high-level guidelines that the two candidates should follow in the matching process.

///////////// START: OVERARCHING MATCHING GUIDELINES /////////////

# === Use of Clinical Inferences ===
You should use clinical inferences for requirement checking (both inclusion and exclusion requirements).
The patient notes (prescreen vignettes) may not explicitly contain some patient facts, and you have to use clinical inference for deriving these facts.
You should be imposing the same inference strength for both inclusion and exclusion criteria.

# === On Inferring Into the Future ===
You should not derive facts that require change of patient states in the future.
However, assume any requirement that requires only logistical actions can be satisfied.
Procedure requirements:
- already undergone/undergoing: do not assume; refer/infer from note.
- able/willing to undergo: assume can be satisfied unless contraindicated.

# === Only Evaluate Projection onto Current Patient State ===
Only care about requirements projected onto current/past patient state.

# === Addressing Missing Information ===
If a significant patient fact is not supported anywhere, assign default values through commonsense reasoning:
- likely mentioned if satisfied -> default false
- likely mentioned if contradicted -> default true
- confidently ruled out -> false
- otherwise -> null

# === Addressing Partial Satisfaction ===
If partially satisfied:
- if potentially satisfiable pending more details -> do not fail
- if confidently not satisfiable -> fail

# === Don't Cares ===
1. Ignore geography
2. Ignore pure logistics/single-off actions
3. Ignore care settings and existing enrollments
4. Ignore requirements impossible to evaluate from prescreen vignettes
5. Ignore requirements that add no constraints

///////////// END: OVERARCHING MATCHING GUIDELINES /////////////

# === GUIDELINES FOR YOUR DECISION OVER THE TWO CANDIDATES ===
1. If A is clearly more correct: A_wins
2. If B is clearly more correct: B_wins
3. If both correct due to missing info policy: both_correct_missing_info_policy_difference
4. If both correct due to clinical interpretation: both_correct_clinical_interpretation_difference
5. If both incorrect: both_incorrect
6. If cannot determine: cannot_determine

# === OUTPUT REQUIREMENTS ===
In <scratchpad>, compare each candidate carefully.
Then in <your_decision>, return JSON with exactly keys below, no trailing commas.

<scratchpad>
...scratch...
</scratchpad>

<your_decision>
{
  "decision_over_decisions": "A_wins" | "B_wins" | "both_correct_missing_info_policy_difference" | "both_correct_clinical_interpretation_difference" | "both_incorrect" | "cannot_determine",
  "confidence": 0.0,
  "brief_rationale": "2-5 sentences"
}
</your_decision>

# === INPUTS ===
<patient_note>
#PATIENT_NOTE#
</patient_note>

<trial_eligibility_text>
#TRIAL_ELIGIBILITY_TEXT#
</trial_eligibility_text>

<candidate_A>
Decision: #CANDIDATE_A_DECISION#
Rationale:
#CANDIDATE_A_RATIONALE#
</candidate_A>

<candidate_B>
Decision: #CANDIDATE_B_DECISION#
Rationale:
#CANDIDATE_B_RATIONALE#
</candidate_B>
""".lstrip()

DEFAULT_VERBALIZER_SYSTEM_MESSAGE = "You are a careful, evidence-grounded clinical trial eligibility explainer."
DEFAULT_ADJUDICATOR_SYSTEM_MESSAGE = "You are an expert clinician who excels at logical reasoning."
DEFAULT_AOAI_API_VERSION = "2024-06-01"


# =============================================================================
# IO helpers
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(p: Path, obj: Any, lock: Optional[threading.Lock] = None) -> None:
    ensure_dir(p.parent)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    if lock is None:
        with p.open("a", encoding="utf-8") as f:
            f.write(line)
        return
    with lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(line)

def now_iso_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _truncate(s: Optional[str], n: int = 6000) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]..."


# =============================================================================
# Data loaders: patient notes, trial corpus
# =============================================================================

def load_patient_note_from_queries_jsonl(notes_jsonl: Path, patient_id: str) -> str:
    if not notes_jsonl or not notes_jsonl.exists():
        return f"[NOT FOUND: notes_jsonl missing] {notes_jsonl}"
    with notes_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("_id") == patient_id:
                txt = obj.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt
                return "[FOUND PATIENT BUT EMPTY TEXT]"
    return f"[PATIENT NOTE NOT FOUND IN {notes_jsonl}] patient_id={patient_id}"

def load_trial_entry_from_corpus_jsonl(corpus_jsonl: Path, trial_id: str) -> Optional[Dict[str, Any]]:
    if not corpus_jsonl or not corpus_jsonl.exists():
        return None
    with corpus_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("_id") == trial_id:
                return obj
    return None

def trial_entry_to_prompt_text(entry: Dict[str, Any], max_chars: int = 12000) -> str:
    try:
        s = json.dumps(entry, ensure_ascii=False, indent=2)
    except Exception:
        s = str(entry)
    return _truncate(s, max_chars) or s


# =============================================================================
# Trial id parent parsing (subcohorts)
# =============================================================================

NCT_PARENT_RE = re.compile(r"^(NCT\d{8})([a-z])?$")

def parse_parent_and_suffix(trial_id: str) -> Tuple[str, Optional[str]]:
    m = NCT_PARENT_RE.match(trial_id.strip())
    if not m:
        return trial_id.strip(), None
    return m.group(1), m.group(2)


# =============================================================================
# Robust JSON-ish extraction (LLM outputs)
# =============================================================================

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')

def _quote_bare_keys(s: str) -> str:
    return _BARE_KEY_RE.sub(r'\1"\2"\3', s)

def _jsonify_python_literals(s: str) -> str:
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s

def _pythonify_json_literals(s: str) -> str:
    s = re.sub(r"\btrue\b", "True", s)
    s = re.sub(r"\bfalse\b", "False", s)
    s = re.sub(r"\bnull\b", "None", s)
    return s

def _top_level_brace_blocks(raw: str) -> List[str]:
    blocks: List[str] = []
    depth = 0
    start: Optional[int] = None
    quote: Optional[str] = None
    esc = False
    for i, c in enumerate(raw):
        if quote is not None:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                quote = None
            continue
        if c in ("'", '"'):
            quote = c
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(raw[start : i + 1])
                    start = None
    return blocks

def extract_json_from_text(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw, str):
        return None, f"Response is not a string: {type(raw).__name__}"

    raw0 = _strip_code_fences(raw.strip())
    blocks = sorted(_top_level_brace_blocks(raw0), key=len, reverse=True)
    candidates: List[Tuple[str, str]] = [("full", raw0)] + [(f"block[{i}]", b) for i, b in enumerate(blocks)]
    errors: List[str] = []

    def _wrap_non_dict(obj: Any) -> Dict[str, Any]:
        return {"_non_dict_json": obj}

    for tag, cand in candidates:
        c = _strip_code_fences(cand.strip())
        if not c:
            continue

        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj, None
            return _wrap_non_dict(obj), None
        except Exception as e:
            errors.append(f"{tag}: json strict failed: {e!r}")

        try:
            c2 = _remove_trailing_commas(_quote_bare_keys(_jsonify_python_literals(c)))
            obj2 = json.loads(c2)
            if isinstance(obj2, dict):
                return obj2, None
            return _wrap_non_dict(obj2), None
        except Exception as e:
            errors.append(f"{tag}: json relaxed failed: {e!r}")

        for variant_name, c3 in [
            ("py literal (orig)", c),
            ("py literal (pythonified)", _pythonify_json_literals(_quote_bare_keys(c))),
        ]:
            try:
                obj3 = ast.literal_eval(c3)
                if isinstance(obj3, dict):
                    return obj3, None
                if isinstance(obj3, (list, tuple)):
                    return _wrap_non_dict(list(obj3)), None
                return _wrap_non_dict(obj3), None
            except Exception as e:
                errors.append(f"{tag}: {variant_name} failed: {e!r}")

    tail = "\n".join(errors[-8:])
    return None, "No parseable JSON-like object found. Recent errors:\n" + tail

def extract_json_from_text_prefer_key(raw: str, prefer_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw, str):
        return None, f"Response is not a string: {type(raw).__name__}"

    raw0 = _strip_code_fences(raw.strip())
    obj, err = extract_json_from_text(raw0)
    if isinstance(obj, dict) and prefer_key in obj:
        return obj, None
    for b in _top_level_brace_blocks(raw0):
        o, e = extract_json_from_text(b)
        if isinstance(o, dict) and prefer_key in o:
            return o, None
    return obj, err


# =============================================================================
# Tagged-block extraction for adjudicator
# =============================================================================

_TAG_BLOCK_RE_CACHE: Dict[str, re.Pattern] = {}

def extract_block_between_tags(raw: str, tag: str) -> Optional[str]:
    if not isinstance(raw, str) or not raw:
        return None
    key = tag.lower()
    cre = _TAG_BLOCK_RE_CACHE.get(key)
    if cre is None:
        cre = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.IGNORECASE | re.DOTALL)
        _TAG_BLOCK_RE_CACHE[key] = cre
    m = cre.search(raw)
    if not m:
        return None
    return m.group(1).strip()


# =============================================================================
# Engine selection: inference_engine / inference_engine_5
# =============================================================================

_DEPLOYMENT_IN_ENDPOINT_RE = re.compile(r"/openai/deployments/([^/?#]+)/?", re.IGNORECASE)

def _extract_deployment_from_endpoint(endpoint: str) -> Optional[str]:
    m = _DEPLOYMENT_IN_ENDPOINT_RE.search(endpoint or "")
    return m.group(1).strip() if m else None

def _looks_like_aoai_host(endpoint: str) -> bool:
    e = (endpoint or "").lower()
    return ".openai.azure.com" in e

def _normalize_endpoint_and_deployment_for_ai_inference(endpoint: str, deployment_or_hint: str) -> Tuple[str, str]:
    ep = (endpoint or "").strip()
    dep = (deployment_or_hint or "").strip()
    if not ep:
        return ep, dep
    ep = ep.rstrip("/")

    dep_in_ep = _extract_deployment_from_endpoint(ep)
    if dep_in_ep:
        if not dep:
            dep = dep_in_ep
        return ep, dep

    if _looks_like_aoai_host(ep):
        if "/openai/" not in ep.lower():
            if not dep:
                raise EnvironmentError(
                    "Azure OpenAI endpoint provided without deployment. "
                    "Set OPENAI_MODEL (deployment name) or pass --*-model-name, "
                    "or pass a full endpoint like "
                    "https://<resource>.openai.azure.com/openai/deployments/<deployment>."
                )
            ep2 = f"{ep}/openai/deployments/{dep}"
            return ep2, dep

        if "/openai/v1" in ep.lower():
            if not dep:
                raise EnvironmentError("Endpoint is /openai/v1 but no deployment provided.")
            return ep, dep

    return ep, dep

def _resolve_endpoint_model_and_api_version(
    *,
    endpoint_arg: str,
    model_arg: str,
    api_version_arg: str,
    endpoint_env_fallback: str,
    model_env_fallback: str,
    api_version_env_fallback: str,
    default_model: str = "gpt-5",
) -> Tuple[str, str, Optional[str]]:
    ep = (endpoint_arg or "").strip() or (os.environ.get(endpoint_env_fallback, "") or "").strip() or (os.environ.get("OPENAI_ENDPOINT", "") or "").strip()
    m = (model_arg or "").strip() or (os.environ.get(model_env_fallback, "") or "").strip() or (os.environ.get("OPENAI_MODEL", "") or "").strip()
    if not m:
        dep_in_ep = _extract_deployment_from_endpoint(ep)
        if dep_in_ep:
            m = dep_in_ep
    if not m:
        m = default_model

    ep2, m2 = _normalize_endpoint_and_deployment_for_ai_inference(ep, m)

    av = (api_version_arg or "").strip() or (os.environ.get(api_version_env_fallback, "") or "").strip() or (os.environ.get("OPENAI_API_VERSION", "") or "").strip() or (os.environ.get("AZURE_OPENAI_API_VERSION", "") or "").strip()
    if not av and _looks_like_aoai_host(ep2):
        av = DEFAULT_AOAI_API_VERSION

    return ep2, m2, (av or None)

from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
if ENGINE_VERSION == "gpt-5":
    from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
else:
    from smt_core.inference_engine import AzureInferenceEngine

_THREAD_LOCAL = threading.local()

def _make_engine(*, endpoint: str, api_key_env_var: str, model_hint: str, default_temperature: float = 0.0):
    if not endpoint:
        raise RuntimeError("Missing Azure endpoint. Pass --*-endpoint or set OPENAI_ENDPOINT.")
    if not model_hint:
        raise RuntimeError("Missing deployment/model hint. Pass --*-model-name or set OPENAI_MODEL.")

    return AzureInferenceEngine(
        endpoint=endpoint,
        api_key_env_var=api_key_env_var,
        model_name=model_hint,
        default_temperature=default_temperature,
    )

def _get_cached_engine(kind: str, *, endpoint: str, api_key_env_var: str, model_hint: str):
    cache = getattr(_THREAD_LOCAL, "engine_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.engine_cache = cache
    key = (kind, endpoint, api_key_env_var, model_hint)
    eng = cache.get(key)
    if eng is None:
        eng = _make_engine(endpoint=endpoint, api_key_env_var=api_key_env_var, model_hint=model_hint, default_temperature=0.0)
        cache[key] = eng
    return eng

def _normalize_engine_output(out: Any) -> str:
    if out is None:
        return ""
    try:
        first = out[0]
    except Exception:
        return str(out)
    if isinstance(first, (list, tuple)) and first:
        return str(first[0])
    return str(first)

def _call_engine_text(
    engine: Any,
    prompt: str,
    *,
    system_message: str,
    temperature: float,
    max_tokens: Optional[int],
    api_version: Optional[str],
) -> str:
    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if api_version:
        kwargs["api_version"] = api_version

    try:
        out = engine(prompt, system_message=system_message, **kwargs)
        txt = _normalize_engine_output(out)
        if txt.strip() == "":
            raise RuntimeError("Engine returned empty text.")
        return txt
    except TypeError:
        pass

    try:
        out = engine(prompt, **kwargs)
        txt = _normalize_engine_output(out)
        if txt.strip() == "":
            raise RuntimeError("Engine returned empty text.")
        return txt
    except TypeError:
        pass

    out = engine(prompt)
    txt = _normalize_engine_output(out)
    if txt.strip() == "":
        raise RuntimeError("Engine returned empty text.")
    return txt


# =============================================================================
# Decisions and rationale loading for each "system"
# =============================================================================

def _normalize_boollike(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"eligible", "yes", "true", "y"}:
            return True
        if s in {"ineligible", "no", "false", "n"}:
            return False
    return None

def bool_to_label(x: Optional[bool]) -> str:
    if x is True:
        return "eligible"
    if x is False:
        return "ineligible"
    return "unclear"


# ---------- NL (llm_judge) ----------
def load_llm_judge_decision(pair_dir: Path, trial_parent_id: str) -> Dict[str, Any]:
    llm_path = pair_dir / f"{trial_parent_id}__llm_judge.json"
    full_path = pair_dir / f"{trial_parent_id}__full.json"

    if llm_path.exists():
        payload = read_json(llm_path)
        if isinstance(payload, dict):
            res = payload.get("result")
            if isinstance(res, dict):
                return res

    if full_path.exists():
        full = read_json(full_path)
        if isinstance(full, dict):
            lj = full.get("llm_judge") or {}
            if isinstance(lj, dict):
                res = lj.get("result")
                if isinstance(res, dict):
                    return res
    return {}

def normalize_eligible_value(obj: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(obj, dict) or not obj:
        return None
    v = obj.get("eligible")
    if v is None:
        v = obj.get("decision")
    vb = _normalize_boollike(v)
    return vb

def extract_nl_rationale(llm_decision: Dict[str, Any]) -> str:
    for k in ["rationale", "reasoning", "explanation", "justification", "analysis", "notes"]:
        v = llm_decision.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = llm_decision.get("rationales")
    if isinstance(v, list) and v:
        ss = [str(x).strip() for x in v if str(x).strip()]
        if ss:
            return "\n".join(ss).strip()
    return _truncate(pretty(llm_decision), 6000) or "[NL rationale missing]"


# ---------- SMT (your match_out artifacts) ----------
def load_parent_or(pair_dir: Path, trial_parent_id: str) -> Optional[Dict[str, Any]]:
    p = pair_dir / f"{trial_parent_id}__parent_or.json"
    if not p.exists():
        return None
    obj = read_json(p)
    return obj if isinstance(obj, dict) else None

def subcohort_ids_from_parent_or(parent_or: Dict[str, Any]) -> List[str]:
    subs = parent_or.get("subcohorts") or []
    out: List[str] = []
    if isinstance(subs, list):
        for r in subs:
            if isinstance(r, dict):
                tid = r.get("trial_id") or r.get("nct_id")
                if isinstance(tid, str) and tid.strip():
                    out.append(tid.strip())
    seen = set()
    deduped: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped

def _system_bool_from_overall_obj(obj: Any) -> Tuple[Optional[bool], str]:
    if not isinstance(obj, dict):
        return None, "not_a_dict"
    for k in ["eligible", "eligible_parent", "eligible_strict", "eligible_relaxed"]:
        vb = _normalize_boollike(obj.get(k))
        if vb is not None:
            return vb, f"key:{k}"
    if isinstance(obj.get("result"), dict):
        vb = _normalize_boollike(obj["result"].get("eligible"))
        if vb is not None:
            return vb, "result.eligible"
    inc = _normalize_boollike(obj.get("inclusion_sat_like") or obj.get("inclusion_sat"))
    exc = _normalize_boollike(obj.get("exclusion_sat_like") or obj.get("exclusion_sat"))
    if inc is not None and exc is not None:
        return (inc and exc), "derived:inclusion_sat_like AND exclusion_sat_like"
    return None, "no_known_fields"

def _or_reduce(vals: List[Optional[bool]]) -> Optional[bool]:
    if any(v is True for v in vals):
        return True
    if vals and all(v is False for v in vals):
        return False
    return None

def compact_overall_for_prompt(overall: Any, max_items: int = 50) -> Dict[str, Any]:
    if not isinstance(overall, dict):
        return {"_error": f"overall_not_dict:{type(overall).__name__}"}
    out: Dict[str, Any] = {}
    for k in ["trial_id", "trial_id_parent", "patient_id", "timestamp", "eligible", "eligible_strict"]:
        if k in overall:
            out[k] = overall.get(k)
    # compact sides
    def _compact_side(side: Any) -> Dict[str, Any]:
        if not isinstance(side, dict):
            return {"status": None, "sat_like": None}
        sat_like = side.get("sat_like") if isinstance(side.get("sat_like"), bool) else None
        summary = side.get("summary") if isinstance(side.get("summary"), dict) else {}
        status = summary.get("status") if isinstance(summary.get("status"), str) else None
        unsat_core = summary.get("unsat_core") if isinstance(summary.get("unsat_core"), list) else None
        unsat_assertions = summary.get("unsat_assertions") if isinstance(summary.get("unsat_assertions"), list) else None
        return {
            "sat_like": sat_like,
            "status": status,
            "unsat_core": [str(x) for x in (unsat_core or [])][:max_items] if status == "unsat" else None,
            "unsat_assertions": [str(x) for x in (unsat_assertions or [])][:max_items] if status == "unsat" and not unsat_core else None,
        }
    out["inclusion"] = _compact_side(overall.get("inclusion"))
    out["exclusion"] = _compact_side(overall.get("exclusion"))
    return out

def compact_parent_or_for_prompt(parent_or: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parent_or, dict):
        return None
    subs = parent_or.get("subcohorts")
    sub_ids: List[str] = []
    if isinstance(subs, list):
        for r in subs:
            if isinstance(r, dict):
                tid = r.get("trial_id") or r.get("nct_id")
                if isinstance(tid, str) and tid.strip():
                    sub_ids.append(tid.strip())
    return {"eligible_parent": parent_or.get("eligible_parent"), "subcohort_ids": sorted(set(sub_ids))}

def derive_smt_parent_decision_and_artifacts(pair_dir: Path, trial_parent_id: str) -> Tuple[Optional[bool], Dict[str, Any]]:
    parent_or_path = pair_dir / f"{trial_parent_id}__parent_or.json"
    parent_overall_path = pair_dir / f"{trial_parent_id}__overall.json"

    parent_or_obj = read_json(parent_or_path) if parent_or_path.exists() else None
    if not isinstance(parent_or_obj, dict):
        parent_or_obj = None
    parent_overall_obj = read_json(parent_overall_path) if parent_overall_path.exists() else None
    if not isinstance(parent_overall_obj, dict):
        parent_overall_obj = None

    decision = _normalize_boollike(parent_or_obj.get("eligible_parent")) if parent_or_obj else None
    decision_source = "parent_or.eligible_parent" if decision is not None else "unknown"

    sub_ids: List[str] = []
    if decision is None:
        if parent_or_obj is not None:
            sub_ids = subcohort_ids_from_parent_or(parent_or_obj)
        if not sub_ids:
            sub_ids = [p.name.split("__overall.json")[0] for p in pair_dir.glob(f"{trial_parent_id}[a-z]__overall.json")]
            sub_ids = sorted(set(sub_ids))
        vals: List[Optional[bool]] = []
        for sid in sub_ids:
            ov_path = pair_dir / f"{sid}__overall.json"
            if not ov_path.exists():
                vals.append(None)
                continue
            try:
                ov = read_json(ov_path)
            except Exception:
                vals.append(None)
                continue
            v, _src = _system_bool_from_overall_obj(ov)
            vals.append(v)
        or_val = _or_reduce(vals) if sub_ids else None
        if or_val is not None:
            decision = or_val
            decision_source = "computed_or_over_subcohort_overalls"

    if decision is None and parent_overall_obj is not None:
        v, src = _system_bool_from_overall_obj(parent_overall_obj)
        if v is not None:
            decision = v
            decision_source = f"parent_overall.{src}"

    artifact: Dict[str, Any] = {
        "trial_parent_id": trial_parent_id,
        "decision_bool": decision,
        "decision_label": bool_to_label(decision),
        "decision_source": decision_source,
        "paths": {
            "parent_or": str(parent_or_path) if parent_or_path.exists() else None,
            "parent_overall": str(parent_overall_path) if parent_overall_path.exists() else None,
        },
        "parent_or": compact_parent_or_for_prompt(parent_or_obj),
        "parent_overall": compact_overall_for_prompt(parent_overall_obj) if parent_overall_obj is not None else None,
    }
    if parent_or_obj is not None:
        artifact["subcohort_ids"] = subcohort_ids_from_parent_or(parent_or_obj)
    return decision, artifact


# ---------- TrialGPT ----------
def load_trialgpt_judge(pair_dir: Path, trial_parent_id: str) -> Dict[str, Any]:
    p = pair_dir / f"{trial_parent_id}__trialgpt_judge.json"
    if p.exists():
        try:
            obj = read_json(p)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    full_path = pair_dir / f"{trial_parent_id}__full.json"
    if full_path.exists():
        try:
            full = read_json(full_path)
            if isinstance(full, dict):
                tg = full.get("trialgpt_judge")
                return tg if isinstance(tg, dict) else {}
        except Exception:
            pass
    return {}

def trialgpt_final_bool_and_source(tg: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    if not isinstance(tg, dict):
        return None, "not_a_dict"
    agg = tg.get("aggregate")
    if isinstance(agg, dict):
        vb = _normalize_boollike(agg.get("eligible"))
        if vb is not None:
            return vb, "aggregate.eligible"
    return None, "missing_aggregate.eligible"

def compact_trialgpt_for_prompt(tg: Dict[str, Any], max_criteria: int = 80, max_rows: int = 120) -> Dict[str, Any]:
    if not isinstance(tg, dict):
        return {"_error": "trialgpt_not_dict"}
    out: Dict[str, Any] = {
        "trial_id_parent": tg.get("trial_id_parent") or tg.get("trial_id") or tg.get("trial_id_original"),
        "patient_id": tg.get("patient_id"),
        "timestamp": tg.get("timestamp"),
        "aggregate": tg.get("aggregate") if isinstance(tg.get("aggregate"), dict) else {},
    }
    for side in ("inclusion", "exclusion"):
        s = tg.get(side)
        if not isinstance(s, dict):
            out[side] = {}
            continue
        crit = s.get("criteria") if isinstance(s.get("criteria"), list) else []
        rows = s.get("rows") if isinstance(s.get("rows"), list) else []
        crit2 = [str(x) for x in crit][:max_criteria]
        rows2: List[Dict[str, Any]] = []
        for r in rows[:max_rows]:
            if not isinstance(r, dict):
                continue
            rows2.append({
                "criterion_id": r.get("criterion_id"),
                "label": r.get("label"),
                "sentence_ids": r.get("sentence_ids"),
                "reasoning": _truncate(r.get("reasoning"), 320),
            })
        out[side] = {
            "criteria": crit2,
            "rows": rows2,
            "parse_error": s.get("parse_error"),
            "temperature_used": s.get("temperature_used"),
        }
    return out


# =============================================================================
# Extracted variable values loader (SMT only; best-effort)
# =============================================================================

def _get_patient_var_values_bundle(side_raw: Any) -> Dict[str, Any]:
    if not isinstance(side_raw, dict):
        return {"patient_var_values_rich": {}, "patient_var_values": {}}
    rich = side_raw.get("patient_var_values_rich") or {}
    plain = side_raw.get("patient_var_values") or {}
    if not isinstance(rich, dict):
        rich = {}
    if not isinstance(plain, dict):
        plain = {}
    return {"patient_var_values_rich": rich, "patient_var_values": plain}

def extract_vars_from_full(full: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inc_side = full.get("inclusion") or {}
    exc_side = full.get("exclusion") or {}
    inc_raw = (inc_side.get("raw") or {}) if isinstance(inc_side, dict) else {}
    exc_raw = (exc_side.get("raw") or {}) if isinstance(exc_side, dict) else {}
    return _get_patient_var_values_bundle(inc_raw), _get_patient_var_values_bundle(exc_raw)

def load_extracted_vars_for_parent_or_subcohorts(pair_dir: Path, trial_parent_id: str) -> Tuple[Any, Any]:
    parent_full_path = pair_dir / f"{trial_parent_id}__full.json"
    inc_vars_obj: Any = {"patient_var_values_rich": {}, "patient_var_values": {}}
    exc_vars_obj: Any = {"patient_var_values_rich": {}, "patient_var_values": {}}

    def _bundle_empty(b: Any) -> bool:
        if not isinstance(b, dict):
            return True
        rv = b.get("patient_var_values_rich") or {}
        pv = b.get("patient_var_values") or {}
        return (not isinstance(rv, dict) or not rv) and (not isinstance(pv, dict) or not pv)

    if parent_full_path.exists():
        try:
            parent_full = read_json(parent_full_path)
            if isinstance(parent_full, dict):
                inc_vars_obj, exc_vars_obj = extract_vars_from_full(parent_full)
        except Exception:
            pass

    parent_or = load_parent_or(pair_dir, trial_parent_id)
    if parent_or is not None:
        sub_ids = subcohort_ids_from_parent_or(parent_or)
    else:
        sub_ids = [p.name.split("__overall.json")[0] for p in pair_dir.glob(f"{trial_parent_id}[a-z]__overall.json")]
        sub_ids = sorted(set(sub_ids))

    if _bundle_empty(inc_vars_obj) and sub_ids:
        inc_map: Dict[str, Any] = {}
        for sid in sub_ids:
            sp = pair_dir / f"{sid}__full.json"
            if not sp.exists():
                continue
            try:
                sf = read_json(sp)
                if isinstance(sf, dict):
                    ib, _ = extract_vars_from_full(sf)
                    inc_map[sid] = ib
            except Exception:
                continue
        if inc_map:
            inc_vars_obj = inc_map

    if _bundle_empty(exc_vars_obj) and sub_ids:
        exc_map: Dict[str, Any] = {}
        for sid in sub_ids:
            sp = pair_dir / f"{sid}__full.json"
            if not sp.exists():
                continue
            try:
                sf = read_json(sp)
                if isinstance(sf, dict):
                    _, eb = extract_vars_from_full(sf)
                    exc_map[sid] = eb
            except Exception:
                continue
        if exc_map:
            exc_vars_obj = exc_map

    return inc_vars_obj, exc_vars_obj


# =============================================================================
# Prompt filling
# =============================================================================

def fill_template(tpl: str, mapping: Dict[str, str]) -> str:
    out = tpl
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


# =============================================================================
# Case discovery
# =============================================================================

def discover_trial_parents(pair_dir: Path) -> List[str]:
    prefixes: set[str] = set()
    for pat in ["*__overall.json", "*__llm_judge.json", "*__full.json", "*__parent_or.json", "*__trialgpt_judge.json"]:
        for p in pair_dir.glob(pat):
            name = p.name
            if "__" not in name:
                continue
            pref = name.split("__", 1)[0]
            if pref.startswith("NCT") and len(pref) >= 11:
                parent, _ = parse_parent_and_suffix(pref)
                prefixes.add(parent)
    return sorted(prefixes)

def _stable_rng(patient_id: str, trial_parent_id: str) -> random.Random:
    seed = int(sha256_text(f"{patient_id}::{trial_parent_id}")[:16], 16)
    return random.Random(seed)


# =============================================================================
# Compare mode: choose which systems are A/B
# =============================================================================

COMPARE_CHOICES = ("smt_vs_nl", "trialgpt_vs_nl", "smt_vs_trialgpt")

def system_tag_from_compare(compare: str) -> Tuple[str, str]:
    """
    Returns (left_tag, right_tag) where tags in {"smt","nl","trialgpt"}.
    """
    if compare == "smt_vs_nl":
        return "smt", "nl"
    if compare == "trialgpt_vs_nl":
        return "trialgpt", "nl"
    if compare == "smt_vs_trialgpt":
        return "smt", "trialgpt"
    raise ValueError(f"bad compare={compare}")

def load_decision_and_rationale(
    *,
    tag: str,
    pair_dir: Path,
    trial_parent_id: str,
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Returns:
      decision_bool, decision_label, supporting_artifact_dict
    Rationale text for NL comes from llm_judge; for SMT/TrialGPT we will verbalize separately.
    """
    if tag == "nl":
        d = load_llm_judge_decision(pair_dir, trial_parent_id)
        b = normalize_eligible_value(d)
        return b, bool_to_label(b), {"llm_judge_result": d}

    if tag == "smt":
        b, art = derive_smt_parent_decision_and_artifacts(pair_dir, trial_parent_id)
        return b, bool_to_label(b), art

    if tag == "trialgpt":
        tg = load_trialgpt_judge(pair_dir, trial_parent_id)
        b, src = trialgpt_final_bool_and_source(tg)
        art = {"trialgpt_src": src, "trialgpt_compact": compact_trialgpt_for_prompt(tg) if tg else {"_error": "missing_trialgpt"}}
        return b, bool_to_label(b), art

    raise ValueError(f"unknown tag={tag}")


# =============================================================================
# Adjudicator decision normalization
# =============================================================================

ALLOWED_DECISION_OVER_DECISIONS = {
    "A_wins",
    "B_wins",
    "both_correct_missing_info_policy_difference",
    "both_correct_clinical_interpretation_difference",
    "both_incorrect",
    "cannot_determine",
}

def _normalize_decision_over_decisions(v: Any) -> str:
    s = str(v or "").strip()
    if not s:
        return "cannot_determine"
    low = s.lower().strip().replace("-", "_").replace(" ", "_")
    low = re.sub(r"__+", "_", low)
    if low in {"a", "a_wins", "awins"}:
        return "A_wins"
    if low in {"b", "b_wins", "bwins"}:
        return "B_wins"
    if low in {"both_correct_missing_info_policy_difference", "missing_info_policy_difference", "tie_policy_difference", "both_correct_policy_difference"}:
        return "both_correct_missing_info_policy_difference"
    if low in {"both_correct_clinical_interpretation_difference", "clinical_interpretation_difference", "tie_clinical_interpretation", "both_correct_clinical_difference"}:
        return "both_correct_clinical_interpretation_difference"
    if low in {"both_incorrect"}:
        return "both_incorrect"
    if low in {"cannot_determine", "insufficient_evidence", "unclear"}:
        return "cannot_determine"
    if s in ALLOWED_DECISION_OVER_DECISIONS:
        return s
    return "cannot_determine"

def _decision_over_decisions_to_winner_bucket(dec: str) -> str:
    if dec == "A_wins":
        return "A"
    if dec == "B_wins":
        return "B"
    if dec == "both_correct_missing_info_policy_difference":
        return "tie_policy_difference"
    if dec == "both_correct_clinical_interpretation_difference":
        return "tie_clinical_interpretation"
    return "insufficient_evidence"

def _outcome_left_vs_right(dec_over: str, A_is_left: bool) -> str:
    """
    Stable outcome for (left_tag vs right_tag) regardless of A/B randomization.
    A_is_left indicates whether Candidate A corresponds to left_tag.
    """
    if dec_over == "A_wins":
        return "left_wins" if A_is_left else "right_wins"
    if dec_over == "B_wins":
        return "right_wins" if A_is_left else "left_wins"
    if dec_over == "both_correct_missing_info_policy_difference":
        return "tie_missing_info_policy_difference"
    if dec_over == "both_correct_clinical_interpretation_difference":
        return "tie_clinical_interpretation_difference"
    if dec_over == "both_incorrect":
        return "both_incorrect"
    return "cannot_determine"


# =============================================================================
# Engine config + case specs
# =============================================================================

@dataclass
class EngineConfig:
    endpoint: str
    api_key_env_var: str
    model_hint: str
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    system_message: str = ""

@dataclass
class CaseSpec:
    patient_id: str
    trial_parent_id: str
    pair_dir: Path

@dataclass
class CaseResult:
    patient_id: str
    trial_parent_id: str
    status: str
    reason: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Verbalizers (SMT / TrialGPT)
# =============================================================================

def _normalize_verbalizer_json(raw_resp: str, *, is_trialgpt: bool) -> Dict[str, Any]:
    parsed, perr = extract_json_from_text(raw_resp)
    if parsed is None or not isinstance(parsed, dict):
        # return a consistent structure
        base = {
            "rationale": "[PARSE ERROR]\n" + (perr or ""),
            "key_points": [],
            "evidence_quotes": {
                "patient_note": [],
                "trial_eligibility_text": [],
                "high_priority_requirements": [],
            },
        }
        if is_trialgpt:
            base["evidence_quotes"]["criterion_artifacts"] = []
        else:
            base["evidence_quotes"]["system_artifacts"] = []
        return base

    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = _truncate(pretty(parsed), 6000) or ""

    key_points = parsed.get("key_points")
    if not isinstance(key_points, list):
        key_points = []
    key_points = [str(x).strip() for x in key_points if str(x).strip()][:8]

    eq = parsed.get("evidence_quotes")
    if not isinstance(eq, dict):
        eq = {}

    def _norm_list(v: Any, n: int) -> List[str]:
        if not isinstance(v, list):
            return []
        out = [str(x).strip() for x in v if str(x).strip()]
        return out[:n]

    evidence_quotes: Dict[str, Any] = {
        "patient_note": _norm_list(eq.get("patient_note"), 6),
        "trial_eligibility_text": _norm_list(eq.get("trial_eligibility_text"), 6),
        "high_priority_requirements": _norm_list(eq.get("high_priority_requirements"), 8),
    }
    if is_trialgpt:
        evidence_quotes["criterion_artifacts"] = _norm_list(eq.get("criterion_artifacts"), 6)
    else:
        evidence_quotes["system_artifacts"] = _norm_list(eq.get("system_artifacts"), 6)

    return {"rationale": rationale.strip(), "key_points": key_points, "evidence_quotes": evidence_quotes}


def run_smt_verbalizer(
    *,
    patient_id: str,
    trial_parent_id: str,
    patient_note: str,
    trial_text: str,
    smt_label: str,
    smt_artifacts_pretty: str,
    inc_vars_obj: Any,
    exc_vars_obj: Any,
    nl_label: str,
    nl_rationale: str,
    verbalizer_template: str,
    verbalizer_engine_cfg: EngineConfig,
    results_root: Path,
    mbench_root: Path,
    overwrite: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (rationale_text, rationale_obj)
    """
    patient_results_dir = results_root / patient_id
    ensure_dir(patient_results_dir)
    out_path = patient_results_dir / f"{trial_parent_id}__smt_rationale.json"

    if out_path.exists() and not overwrite:
        try:
            obj = read_json(out_path)
            if isinstance(obj, dict):
                rat = str(obj.get("rationale") or "").strip()
                if rat:
                    return rat, obj
        except Exception:
            pass

    mapping = {
        "#PATIENT_NOTE#": patient_note,
        "#TRIAL_ELIGIBILITY_TEXT#": trial_text,
        "#SYSTEM_DECISION_LABEL#": smt_label,
        "#SYSTEM_DECISION_ARTIFACTS#": smt_artifacts_pretty,
        "#INCLUSION_EXTRACTED_VARIABLE_VALUES#": pretty(inc_vars_obj),
        "#EXCLUSION_EXTRACTED_VARIABLE_VALUES#": pretty(exc_vars_obj),
        "#NL_DECISION#": nl_label,
        "#NL_RATIONALE#": nl_rationale,
    }
    prompt = fill_template(verbalizer_template, mapping)

    mbench_dir = mbench_root / patient_id
    ensure_dir(mbench_dir)
    prompt_path = mbench_dir / f"{trial_parent_id}__smt_verbalizer_prompt.txt"
    resp_path = mbench_dir / f"{trial_parent_id}__smt_verbalizer_response.txt"
    write_text(prompt_path, prompt)

    eng = _get_cached_engine(
        "verbalizer_smt",
        endpoint=verbalizer_engine_cfg.endpoint,
        api_key_env_var=verbalizer_engine_cfg.api_key_env_var,
        model_hint=verbalizer_engine_cfg.model_hint,
    )
    raw_resp = _call_engine_text(
        eng,
        prompt,
        system_message=verbalizer_engine_cfg.system_message or DEFAULT_VERBALIZER_SYSTEM_MESSAGE,
        temperature=verbalizer_engine_cfg.temperature,
        max_tokens=verbalizer_engine_cfg.max_tokens,
        api_version=verbalizer_engine_cfg.api_version,
    )
    write_text(resp_path, raw_resp)

    norm = _normalize_verbalizer_json(raw_resp, is_trialgpt=False)
    obj = {
        "decision": smt_label,
        "timestamp_utc": now_iso_utc(),
        **norm,
    }
    write_json(out_path, obj)
    return obj["rationale"], obj


def run_trialgpt_verbalizer(
    *,
    patient_id: str,
    trial_parent_id: str,
    patient_note: str,
    trial_text: str,
    tg_label: str,
    tg_compact_artifacts_pretty: str,
    nl_label: str,
    nl_rationale: str,
    verbalizer_template: str,
    verbalizer_engine_cfg: EngineConfig,
    results_root: Path,
    mbench_root: Path,
    overwrite: bool,
) -> Tuple[str, Dict[str, Any]]:
    patient_results_dir = results_root / patient_id
    ensure_dir(patient_results_dir)
    out_path = patient_results_dir / f"{trial_parent_id}__trialgpt_rationale.json"

    if out_path.exists() and not overwrite:
        try:
            obj = read_json(out_path)
            if isinstance(obj, dict):
                rat = str(obj.get("rationale") or "").strip()
                if rat:
                    return rat, obj
        except Exception:
            pass

    mapping = {
        "#PATIENT_NOTE#": patient_note,
        "#TRIAL_ELIGIBILITY_TEXT#": trial_text,
        "#SYSTEM_DECISION_LABEL#": tg_label,
        "#CRITERION_DECISION_ARTIFACTS#": tg_compact_artifacts_pretty,
        "#NL_DECISION#": nl_label,
        "#NL_RATIONALE#": nl_rationale,
    }
    prompt = fill_template(verbalizer_template, mapping)

    mbench_dir = mbench_root / patient_id
    ensure_dir(mbench_dir)
    prompt_path = mbench_dir / f"{trial_parent_id}__trialgpt_verbalizer_prompt.txt"
    resp_path = mbench_dir / f"{trial_parent_id}__trialgpt_verbalizer_response.txt"
    write_text(prompt_path, prompt)

    eng = _get_cached_engine(
        "verbalizer_trialgpt",
        endpoint=verbalizer_engine_cfg.endpoint,
        api_key_env_var=verbalizer_engine_cfg.api_key_env_var,
        model_hint=verbalizer_engine_cfg.model_hint,
    )
    raw_resp = _call_engine_text(
        eng,
        prompt,
        system_message=verbalizer_engine_cfg.system_message or DEFAULT_VERBALIZER_SYSTEM_MESSAGE,
        temperature=verbalizer_engine_cfg.temperature,
        max_tokens=verbalizer_engine_cfg.max_tokens,
        api_version=verbalizer_engine_cfg.api_version,
    )
    write_text(resp_path, raw_resp)

    norm = _normalize_verbalizer_json(raw_resp, is_trialgpt=True)
    obj = {
        "decision": tg_label,
        "timestamp_utc": now_iso_utc(),
        **norm,
    }
    write_json(out_path, obj)
    return obj["rationale"], obj


# =============================================================================
# Core: process one case
# =============================================================================

def _read_template_or_default(path: Optional[Path], default: str) -> str:
    if path and path.exists():
        print(f"Using template at {path}")
        return read_text(path)
    return default


def process_case(
    case: CaseSpec,
    *,
    compare: str,
    notes_jsonl: Path,
    trial_corpus_jsonl: Path,
    results_root: Path,
    mbench_root: Path,
    overwrite: bool,
    randomize: bool,
    run_verbalizer: bool,
    run_adjudicator: bool,
    verbalizer_engine_cfg: EngineConfig,
    adjudicator_engine_cfg: EngineConfig,
    smt_verbalizer_template: str,
    trialgpt_verbalizer_template: str,
    adjudicator_template: str,
    adjudications_jsonl: Path,
    adjudications_lock: threading.Lock,
    max_patient_note_chars: int,
    max_trial_text_chars: int,
) -> CaseResult:
    pid = case.patient_id
    tid = case.trial_parent_id
    pair_dir = case.pair_dir

    left_tag, right_tag = system_tag_from_compare(compare)

    # Load trial text (prefer parent; fall back to any subcohort in corpus if needed)
    patient_note = load_patient_note_from_queries_jsonl(notes_jsonl, pid)
    patient_note = _truncate(patient_note, max_patient_note_chars) or patient_note

    trial_entry = load_trial_entry_from_corpus_jsonl(trial_corpus_jsonl, tid)
    if trial_entry is None:
        # try subcohorts if parent missing
        por = load_parent_or(pair_dir, tid)
        sub_ids: List[str] = subcohort_ids_from_parent_or(por) if isinstance(por, dict) else []
        for sid in sub_ids[:12]:
            te = load_trial_entry_from_corpus_jsonl(trial_corpus_jsonl, sid)
            if te is not None:
                trial_entry = te
                break

    if trial_entry is None:
        trial_text = _truncate(pretty({"_error": "TRIAL ENTRY NOT FOUND IN CORPUS", "trial_parent_id": tid}), max_trial_text_chars) or ""
    else:
        trial_text = trial_entry_to_prompt_text(trial_entry, max_chars=max_trial_text_chars)

    # Load decisions
    left_bool, left_label, left_art = load_decision_and_rationale(tag=left_tag, pair_dir=pair_dir, trial_parent_id=tid)
    right_bool, right_label, right_art = load_decision_and_rationale(tag=right_tag, pair_dir=pair_dir, trial_parent_id=tid)

    # If either missing comparability, skip
    if left_bool is None or right_bool is None:
        return CaseResult(pid, tid, status="skipped", reason="missing_comparable_decision")

    if left_bool == right_bool:
        return CaseResult(pid, tid, status="skipped", reason="no_disagreement")

    # Build rationales:
    # - NL: from llm_judge
    # - SMT / TrialGPT: either use existing rationale if present (from this script outputs), or verbalize
    def _nl_rationale() -> str:
        d = left_art.get("llm_judge_result") if left_tag == "nl" else right_art.get("llm_judge_result") if right_tag == "nl" else {}
        if not isinstance(d, dict):
            d = {}
        return _truncate(extract_nl_rationale(d), 6000) or "[NL rationale missing]"

    nl_rationale = _nl_rationale()

    left_rationale = nl_rationale if left_tag == "nl" else ""
    right_rationale = nl_rationale if right_tag == "nl" else ""

    # Verbalize SMT/TrialGPT as needed
    if run_verbalizer:
        # We treat "evaluated" side as whichever is not NL when comparing vs NL.
        # For smt_vs_trialgpt, we verbalize BOTH sides (since neither has NL rationale).
        inc_vars_obj, exc_vars_obj = load_extracted_vars_for_parent_or_subcohorts(pair_dir, tid)

        if left_tag == "smt":
            smt_artifacts_pretty = _truncate(pretty(left_art), 12000) or pretty(left_art)
            left_rationale, _ = run_smt_verbalizer(
                patient_id=pid,
                trial_parent_id=tid,
                patient_note=patient_note,
                trial_text=trial_text,
                smt_label=left_label,
                smt_artifacts_pretty=smt_artifacts_pretty,
                inc_vars_obj=inc_vars_obj,
                exc_vars_obj=exc_vars_obj,
                nl_label=right_label if right_tag == "nl" else bool_to_label(right_bool),
                nl_rationale=nl_rationale if right_tag == "nl" else "[no reference assessment provided]",
                verbalizer_template=smt_verbalizer_template,
                verbalizer_engine_cfg=verbalizer_engine_cfg,
                results_root=results_root,
                mbench_root=mbench_root,
                overwrite=overwrite,
            )

        if right_tag == "smt":
            smt_artifacts_pretty = _truncate(pretty(right_art), 12000) or pretty(right_art)
            right_rationale, _ = run_smt_verbalizer(
                patient_id=pid,
                trial_parent_id=tid,
                patient_note=patient_note,
                trial_text=trial_text,
                smt_label=right_label,
                smt_artifacts_pretty=smt_artifacts_pretty,
                inc_vars_obj=inc_vars_obj,
                exc_vars_obj=exc_vars_obj,
                nl_label=left_label if left_tag == "nl" else bool_to_label(left_bool),
                nl_rationale=nl_rationale if left_tag == "nl" else "[no reference assessment provided]",
                verbalizer_template=smt_verbalizer_template,
                verbalizer_engine_cfg=verbalizer_engine_cfg,
                results_root=results_root,
                mbench_root=mbench_root,
                overwrite=overwrite,
            )

        if left_tag == "trialgpt":
            tg_compact = left_art.get("trialgpt_compact") if isinstance(left_art, dict) else None
            tg_compact_pretty = _truncate(pretty(tg_compact), 12000) if tg_compact is not None else pretty({"_error": "missing_trialgpt_compact"})
            left_rationale, _ = run_trialgpt_verbalizer(
                patient_id=pid,
                trial_parent_id=tid,
                patient_note=patient_note,
                trial_text=trial_text,
                tg_label=left_label,
                tg_compact_artifacts_pretty=tg_compact_pretty or "",
                nl_label=right_label if right_tag == "nl" else bool_to_label(right_bool),
                nl_rationale=nl_rationale if right_tag == "nl" else "[no reference assessment provided]",
                verbalizer_template=trialgpt_verbalizer_template,
                verbalizer_engine_cfg=verbalizer_engine_cfg,
                results_root=results_root,
                mbench_root=mbench_root,
                overwrite=overwrite,
            )

        if right_tag == "trialgpt":
            tg_compact = right_art.get("trialgpt_compact") if isinstance(right_art, dict) else None
            tg_compact_pretty = _truncate(pretty(tg_compact), 12000) if tg_compact is not None else pretty({"_error": "missing_trialgpt_compact"})
            right_rationale, _ = run_trialgpt_verbalizer(
                patient_id=pid,
                trial_parent_id=tid,
                patient_note=patient_note,
                trial_text=trial_text,
                tg_label=right_label,
                tg_compact_artifacts_pretty=tg_compact_pretty or "",
                nl_label=left_label if left_tag == "nl" else bool_to_label(left_bool),
                nl_rationale=nl_rationale if left_tag == "nl" else "[no reference assessment provided]",
                verbalizer_template=trialgpt_verbalizer_template,
                verbalizer_engine_cfg=verbalizer_engine_cfg,
                results_root=results_root,
                mbench_root=mbench_root,
                overwrite=overwrite,
            )

    # If we didn't verbalize and a side isn't NL, try to load an existing rationale file
    def _try_load_existing_rationale(tag: str) -> Optional[str]:
        patient_results_dir = results_root / pid
        if tag == "smt":
            p = patient_results_dir / f"{tid}__smt_rationale.json"
        elif tag == "trialgpt":
            p = patient_results_dir / f"{tid}__trialgpt_rationale.json"
        else:
            return None
        if p.exists():
            try:
                obj = read_json(p)
                if isinstance(obj, dict):
                    t = str(obj.get("rationale") or "").strip()
                    return t if t else None
            except Exception:
                return None
        return None

    if left_tag != "nl" and not left_rationale:
        left_rationale = _try_load_existing_rationale(left_tag) or "[missing verbalized rationale]"
    if right_tag != "nl" and not right_rationale:
        right_rationale = _try_load_existing_rationale(right_tag) or "[missing verbalized rationale]"

    if not run_adjudicator:
        return CaseResult(pid, tid, status="written", reason="verbalized_only")

    # Adjudicator stage
    if randomize:
        rng = _stable_rng(pid, tid)
        flip = (rng.random() < 0.5)
    else:
        flip = False

    if not flip:
        A_dec, A_rat = left_label, left_rationale
        B_dec, B_rat = right_label, right_rationale
        A_is_left = True
    else:
        A_dec, A_rat = right_label, right_rationale
        B_dec, B_rat = left_label, left_rationale
        A_is_left = False

    mapping = {
        "#PATIENT_NOTE#": patient_note,
        "#TRIAL_ELIGIBILITY_TEXT#": trial_text,
        "#CANDIDATE_A_DECISION#": A_dec,
        "#CANDIDATE_A_RATIONALE#": A_rat,
        "#CANDIDATE_B_DECISION#": B_dec,
        "#CANDIDATE_B_RATIONALE#": B_rat,
    }
    adj_prompt = fill_template(adjudicator_template, mapping)

    mbench_dir = mbench_root / pid
    ensure_dir(mbench_dir)
    adj_prompt_path = mbench_dir / f"{tid}__adjudication_prompt.txt"
    adj_resp_path = mbench_dir / f"{tid}__adjudication_response.txt"
    write_text(adj_prompt_path, adj_prompt)

    try:
        eng = _get_cached_engine(
            "adjudicator",
            endpoint=adjudicator_engine_cfg.endpoint,
            api_key_env_var=adjudicator_engine_cfg.api_key_env_var,
            model_hint=adjudicator_engine_cfg.model_hint,
        )
        raw_resp = _call_engine_text(
            eng,
            adj_prompt,
            system_message=adjudicator_engine_cfg.system_message or DEFAULT_ADJUDICATOR_SYSTEM_MESSAGE,
            temperature=adjudicator_engine_cfg.temperature,
            max_tokens=adjudicator_engine_cfg.max_tokens,
            api_version=adjudicator_engine_cfg.api_version,
        )
        write_text(adj_resp_path, raw_resp)

        scratchpad_txt = extract_block_between_tags(raw_resp, "scratchpad")
        decision_block = extract_block_between_tags(raw_resp, "your_decision")

        if decision_block:
            parsed, perr = extract_json_from_text_prefer_key(decision_block, "decision_over_decisions")
        else:
            parsed, perr = None, "Missing <your_decision> block."
            p2, e2 = extract_json_from_text_prefer_key(raw_resp, "decision_over_decisions")
            if isinstance(p2, dict) and "decision_over_decisions" in p2:
                parsed, perr = p2, None
            else:
                perr = (perr or "") + ("\n" + (e2 or ""))

        if parsed is None or not isinstance(parsed, dict):
            adjud_obj: Dict[str, Any] = {
                "decision_over_decisions": "cannot_determine",
                "confidence": 0.0,
                "brief_rationale": "[PARSE ERROR]\n" + (perr or ""),
            }
        else:
            dec_over = _normalize_decision_over_decisions(parsed.get("decision_over_decisions"))
            brief = str(parsed.get("brief_rationale", "") or "").strip()
            try:
                conf = float(parsed.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            adjud_obj = {"decision_over_decisions": dec_over, "confidence": conf, "brief_rationale": brief}

        if isinstance(scratchpad_txt, str) and scratchpad_txt.strip():
            adjud_obj["scratchpad"] = _truncate(scratchpad_txt.strip(), 12000)

        adjud_obj["winner_bucket"] = _decision_over_decisions_to_winner_bucket(adjud_obj["decision_over_decisions"])
        adjud_obj["outcome_left_vs_right"] = _outcome_left_vs_right(adjud_obj["decision_over_decisions"], A_is_left)
        adjud_obj["left_tag"] = left_tag
        adjud_obj["right_tag"] = right_tag
        adjud_obj["compare"] = compare
        adjud_obj["left_decision"] = left_label
        adjud_obj["right_decision"] = right_label

        patient_results_dir = results_root / pid
        ensure_dir(patient_results_dir)
        adjud_path = patient_results_dir / f"{tid}__adjudication.json"
        adjud_meta_path = patient_results_dir / f"{tid}__adjudication_meta.json"
        write_json(adjud_path, adjud_obj)
        write_json(
            adjud_meta_path,
            {
                "timestamp_utc": now_iso_utc(),
                "patient_id": pid,
                "trial_parent_id": tid,
                "compare": compare,
                "left_tag": left_tag,
                "right_tag": right_tag,
                "left_decision": left_label,
                "right_decision": right_label,
                "candidate_A_is_left": A_is_left,
                "randomize": bool(randomize),
                "prompt_sha256": sha256_text(adj_prompt),
                "raw_response_sha256": sha256_text(raw_resp),
                "decision_over_decisions": adjud_obj.get("decision_over_decisions"),
                "winner_bucket": adjud_obj.get("winner_bucket"),
                "outcome_left_vs_right": adjud_obj.get("outcome_left_vs_right"),
            },
        )

        append_jsonl(
            adjudications_jsonl,
            {
                "timestamp_utc": now_iso_utc(),
                "patient_id": pid,
                "trial_parent_id": tid,
                "compare": compare,
                "left_tag": left_tag,
                "right_tag": right_tag,
                "decision_over_decisions": adjud_obj.get("decision_over_decisions"),
                "winner_bucket": adjud_obj.get("winner_bucket"),
                "outcome_left_vs_right": adjud_obj.get("outcome_left_vs_right"),
                "confidence": adjud_obj.get("confidence"),
                "randomize": bool(randomize),
            },
            lock=adjudications_lock,
        )

        return CaseResult(pid, tid, status="written", reason="adjudicated")

    except Exception as e:
        err_payload = {
            "error": repr(e),
            "stage": "adjudicator",
            "patient_id": pid,
            "trial_parent_id": tid,
            "endpoint": adjudicator_engine_cfg.endpoint,
            "deployment_or_model_hint": adjudicator_engine_cfg.model_hint,
            "api_version": adjudicator_engine_cfg.api_version,
            "api_key_env_var": adjudicator_engine_cfg.api_key_env_var,
            "temperature": adjudicator_engine_cfg.temperature,
            "max_tokens": adjudicator_engine_cfg.max_tokens,
            "system_message": adjudicator_engine_cfg.system_message,
            "prompt_sha256": sha256_text(adj_prompt),
            "prompt_chars": len(adj_prompt),
            "ts_utc": now_iso_utc(),
        }
        write_text(adj_resp_path, json.dumps(err_payload, ensure_ascii=False, indent=2) + "\n")
        return CaseResult(pid, tid, status="error", reason="adjudicator_error", error=repr(e))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--compare", choices=COMPARE_CHOICES, default="smt_vs_nl",
                    help="Choose which two systems to compare.")
    ap.add_argument("--match-out-root", type=str, default="../../cmsrc/match_out",
                    help="Root like <match_out>/<patient_id>/... (scan mode)")
    ap.add_argument("--pair-dir", type=str, default=None,
                    help="Single-case mode: path to <match_out>/<patient_id>/")
    ap.add_argument("--patient-id", type=str, default=None,
                    help="Patient ID (required for single-case; optional filter for scan)")
    ap.add_argument("--trial-id", type=str, default=None,
                    help="Trial ID or parent trial id (optional filter)")

    ap.add_argument("--notes-jsonl", type=str, default="../../dataset/clinical_trial/sigir/queries.jsonl",
                    help="queries.jsonl path (patient notes)")
    ap.add_argument("--trial-corpus-jsonl", type=str, default="../../dataset/clinical_trial/sigir/corpus.jsonl",
                    help="corpus.jsonl path (trial text)")

    ap.add_argument("--results-root", type=str, default="./results", help="Where to write results")
    ap.add_argument("--mbench-root", type=str, default="./mbench", help="Where to write prompt/response artifacts")
    ap.add_argument("--adjudications-jsonl", type=str, default=None, help="Optional override for results_root/adjudications.jsonl")

    ap.add_argument("--smt-verbalizer-template", type=str, default=None,
                    help="Path to SMT verbalizer template (optional)")
    ap.add_argument("--trialgpt-verbalizer-template", type=str, default=None,
                    help="Path to TrialGPT verbalizer template (optional)")
    ap.add_argument("--adjudicator-template", type=str, default=None,
                    help="Path to adjudicator template (optional)")

    ap.add_argument("--run-verbalizer", action="store_true", help="Verbalize SMT/TrialGPT side(s) as needed")
    ap.add_argument("--run-adjudicator", action="store_true", help="Run adjudicator stage")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--randomize", action="store_true", help="Randomize candidate A/B assignment (stable per patient/trial).")

    ap.add_argument("--api-key-env-var", type=str, default=os.environ.get("OPENAI_API_KEY_ENV_VAR", "OPENAI_API_KEY"))

    ap.add_argument("--verbalizer-endpoint", type=str, default="")
    ap.add_argument("--verbalizer-model-name", type=str, default="")
    ap.add_argument("--verbalizer-api-version", type=str, default="")
    ap.add_argument("--verbalizer-temperature", type=float, default=0.0)
    ap.add_argument("--verbalizer-max-tokens", type=int, default=1200)

    ap.add_argument("--adjudicator-endpoint", type=str, default="")
    ap.add_argument("--adjudicator-model-name", type=str, default="")
    ap.add_argument("--adjudicator-api-version", type=str, default="")
    ap.add_argument("--adjudicator-temperature", type=float, default=0.0)
    ap.add_argument("--adjudicator-max-tokens", type=int, default=1200)

    ap.add_argument("--max-workers", type=int, default=8, help="ThreadPool workers for cases")
    ap.add_argument("--max-patient-note-chars", type=int, default=12000)
    ap.add_argument("--max-trial-text-chars", type=int, default=12000)

    args = ap.parse_args()

    verbalizer_endpoint, verbalizer_deployment, verbalizer_api_version = _resolve_endpoint_model_and_api_version(
        endpoint_arg=args.verbalizer_endpoint,
        model_arg=args.verbalizer_model_name,
        api_version_arg=args.verbalizer_api_version,
        endpoint_env_fallback="OPENAI_VERBALIZER_ENDPOINT",
        model_env_fallback="OPENAI_VERBALIZER_MODEL",
        api_version_env_fallback="OPENAI_VERBALIZER_API_VERSION",
        default_model="gpt-5",
    )
    adjudicator_endpoint, adjudicator_deployment, adjudicator_api_version = _resolve_endpoint_model_and_api_version(
        endpoint_arg=args.adjudicator_endpoint,
        model_arg=args.adjudicator_model_name,
        api_version_arg=args.adjudicator_api_version,
        endpoint_env_fallback="OPENAI_ADJUDICATOR_ENDPOINT",
        model_env_fallback="OPENAI_ADJUDICATOR_MODEL",
        api_version_env_fallback="OPENAI_ADJUDICATOR_API_VERSION",
        default_model="gpt-5",
    )

    run_verbalizer = bool(args.run_verbalizer)
    run_adjudicator = bool(args.run_adjudicator)
    if not run_verbalizer and not run_adjudicator:
        run_verbalizer = True
        run_adjudicator = True

    results_root = Path(args.results_root)
    mbench_root = Path(args.mbench_root)
    ensure_dir(results_root)
    ensure_dir(mbench_root)

    adjudications_jsonl = Path(args.adjudications_jsonl) if args.adjudications_jsonl else (results_root / "adjudications.jsonl")
    adjudications_lock = threading.Lock()

    notes_jsonl = Path(args.notes_jsonl)
    trial_corpus_jsonl = Path(args.trial_corpus_jsonl)

    smt_tpl = _read_template_or_default(Path(args.smt_verbalizer_template) if args.smt_verbalizer_template else None,
                                       DEFAULT_SMT_VERBALIZER_TEMPLATE)
    tg_tpl = _read_template_or_default(Path(args.trialgpt_verbalizer_template) if args.trialgpt_verbalizer_template else None,
                                      DEFAULT_TRIALGPT_VERBALIZER_TEMPLATE)
    adj_tpl = _read_template_or_default(Path(args.adjudicator_template) if args.adjudicator_template else None,
                                       DEFAULT_ADJUDICATOR_TEMPLATE)

    verbalizer_engine_cfg = EngineConfig(
        endpoint=verbalizer_endpoint,
        api_key_env_var=args.api_key_env_var,
        model_hint=verbalizer_deployment,
        api_version=verbalizer_api_version,
        temperature=float(args.verbalizer_temperature),
        max_tokens=int(args.verbalizer_max_tokens) if args.verbalizer_max_tokens else None,
        system_message=DEFAULT_VERBALIZER_SYSTEM_MESSAGE,
    )
    adjudicator_engine_cfg = EngineConfig(
        endpoint=adjudicator_endpoint,
        api_key_env_var=args.api_key_env_var,
        model_hint=adjudicator_deployment,
        api_version=adjudicator_api_version,
        temperature=float(args.adjudicator_temperature),
        max_tokens=int(args.adjudicator_max_tokens) if args.adjudicator_max_tokens else None,
        system_message=DEFAULT_ADJUDICATOR_SYSTEM_MESSAGE,
    )

    # Build cases
    cases: List[CaseSpec] = []
    compare = str(args.compare)

    if args.pair_dir:
        if not args.patient_id or not args.trial_id:
            raise SystemExit("--pair-dir requires --patient-id and --trial-id")
        pair_dir = Path(args.pair_dir)
        trial_parent_id, _ = parse_parent_and_suffix(args.trial_id)
        cases = [CaseSpec(patient_id=args.patient_id, trial_parent_id=trial_parent_id, pair_dir=pair_dir)]
    else:
        root = Path(args.match_out_root)
        if not root.exists():
            raise SystemExit(f"match_out_root not found: {root}")

        patient_filter = args.patient_id
        trial_filter_parent = None
        if args.trial_id:
            trial_filter_parent, _ = parse_parent_and_suffix(args.trial_id)

        for pdir in sorted([p for p in root.iterdir() if p.is_dir()]):
            pid = pdir.name
            if patient_filter and pid != patient_filter:
                continue
            for tid in discover_trial_parents(pdir):
                if trial_filter_parent and tid != trial_filter_parent:
                    continue

                # include only if both sides exist + disagree
                left_tag, right_tag = system_tag_from_compare(compare)
                lb, _, _ = load_decision_and_rationale(tag=left_tag, pair_dir=pdir, trial_parent_id=tid)
                rb, _, _ = load_decision_and_rationale(tag=right_tag, pair_dir=pdir, trial_parent_id=tid)
                if lb is None or rb is None:
                    continue
                if lb == rb:
                    continue
                cases.append(CaseSpec(patient_id=pid, trial_parent_id=tid, pair_dir=pdir))

    if not cases:
        print("[DONE] no disagreement cases found for compare:", compare)
        return

    print(
        "[CFG] compare={!r} cases={} run_verbalizer={} run_adjudicator={} max_workers={} randomize={}".format(
            compare, len(cases), run_verbalizer, run_adjudicator, int(args.max_workers), bool(args.randomize)
        )
    )

    summary_counts: Dict[str, int] = {}
    errors: List[Dict[str, Any]] = []

    def _bump(k: str) -> None:
        summary_counts[k] = summary_counts.get(k, 0) + 1

    with cf.ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        futs = [
            ex.submit(
                process_case,
                c,
                compare=compare,
                notes_jsonl=notes_jsonl,
                trial_corpus_jsonl=trial_corpus_jsonl,
                results_root=results_root,
                mbench_root=mbench_root,
                overwrite=bool(args.overwrite),
                randomize=bool(args.randomize),
                run_verbalizer=run_verbalizer,
                run_adjudicator=run_adjudicator,
                verbalizer_engine_cfg=verbalizer_engine_cfg,
                adjudicator_engine_cfg=adjudicator_engine_cfg,
                smt_verbalizer_template=smt_tpl,
                trialgpt_verbalizer_template=tg_tpl,
                adjudicator_template=adj_tpl,
                adjudications_jsonl=adjudications_jsonl,
                adjudications_lock=adjudications_lock,
                max_patient_note_chars=int(args.max_patient_note_chars),
                max_trial_text_chars=int(args.max_trial_text_chars),
            )
            for c in cases
        ]

        for fut in cf.as_completed(futs):
            try:
                r: CaseResult = fut.result()
            except Exception as e:
                _bump("error_exception")
                errors.append({"error": repr(e)})
                continue

            _bump(f"status_{r.status}")
            if r.reason:
                _bump(f"reason_{r.reason}")
            if r.status == "error":
                errors.append(
                    {
                        "patient_id": r.patient_id,
                        "trial_parent_id": r.trial_parent_id,
                        "reason": r.reason,
                        "error": r.error,
                    }
                )

    run_summary = {
        "timestamp_utc": now_iso_utc(),
        "compare": compare,
        "cases_planned": len(cases),
        "counts": summary_counts,
        "errors": errors[:50],
        "run_verbalizer": run_verbalizer,
        "run_adjudicator": run_adjudicator,
        "randomize": bool(args.randomize),
        "verbalizer_endpoint": verbalizer_engine_cfg.endpoint,
        "verbalizer_deployment": verbalizer_engine_cfg.model_hint,
        "verbalizer_api_version": verbalizer_engine_cfg.api_version,
        "adjudicator_endpoint": adjudicator_engine_cfg.endpoint,
        "adjudicator_deployment": adjudicator_engine_cfg.model_hint,
        "adjudicator_api_version": adjudicator_engine_cfg.api_version,
        "api_key_env_var": args.api_key_env_var,
    }
    if len(cases) > 1:
        write_json(results_root / f"run_summary__{compare}.json", run_summary)

    print("[DONE] " + pretty(run_summary))


if __name__ == "__main__":
    main()