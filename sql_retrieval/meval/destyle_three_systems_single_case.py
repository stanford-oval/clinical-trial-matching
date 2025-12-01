#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
destyle_three_systems_single_case.py

Goal (single (patient_id, trial_id)):
1) Load System 1 = LLM judge (match_out artifacts): prompt + output
2) Run TWO verbalizers (produce "verbalized rationales"):
   - System 2 = SMT verbalizer
   - System 3 = TrialGPT verbalizer
3) Call destyle prompt:
   <SATIR_ROOT>/irsrc/meval/prompts/destyle_verbalization.prompt
   with placeholders:

# === INPUTS (Part 0) ===
<patient_note>
#PATIENT_NOTE#
</patient_note>

<trial_eligibility_text>
#TRIAL_ELIGIBILITY_TEXT#
</trial_eligibility_text>

# === INPUTS (Part 1) ===
<prompt_used_by_system_1>
#PROMPT_USED_BY_SYSTEM_1#
</prompt_used_by_system_1>

<output_system_1>
#OUTPUT_SYSTEM_1#
</output_system_1>

# === INPUTS (Part 2) ===
<prompt_used_by_system_2>
#PROMPT_USED_BY_SYSTEM_2#
</prompt_used_by_system_2>

<output_system_2>
#OUTPUT_SYSTEM_2#
</output_system_2>

# === INPUTS (Part 3) ===
<prompt_used_by_system_3>
#PROMPT_USED_BY_SYSTEM_3#
</prompt_used_by_system_3>

<output_system_3>
#OUTPUT_SYSTEM_3#
</output_system_3>

Outputs (default):
  <out_root>/<patient_id>/<trial_parent_id>/
    system1_prompt_used.txt
    system1_output.json
    system2_prompt_used.txt
    system2_output.json
    system3_prompt_used.txt
    system3_output.json
    destyle_prompt.txt
    destyle_response.txt
    destyled.json
    bundle.json

Notes:
- System 1 prompt is reconstructed from llm_judge.payload["prompt_path"] (template) + our
  locally loaded patient note + trial description derived from corpus.jsonl.
- System 2 / 3 prompts are the actual verbalizer prompts we send to the engine.
- System 2 / 3 outputs are the parsed verbalizer JSON objects (with "decision" attached).
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Templates (verbalizers)
# =============================================================================

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
- Even if you believe that the system has made a logical error (e.g., wrong decomposition, wrong requirement interpretation),
  you should always stick to the logic by which the system comes to the conclusion. Argue for the system's case.

# === EMPHASIS REQUIREMENT (TOP PRIORITY) ===
1) If ineligible, start with the single most decision-driving eligibility requirement(s) and patient fact(s).
2) Emphasize requirements likely to flip the reference assessment if interpreted differently.
3) Do NOT mention "system" / "disagree" / "compared to" etc.
4) Present checkable requirements + evidence/unknowns.

# === OUTPUT REQUIREMENTS ===
Return a SINGLE JSON object (no markdown fences) with exactly these keys:
{
  "rationale": "one concise paragraph",
  "key_points": ["...", "..."],
  "evidence_quotes": {
    "patient_note": ["short quote 1", "short quote 2"],
    "trial_eligibility_text": ["short quote 1", "short quote 2"],
    "high_priority_requirements": ["requirement phrase(s) that drive the label"],
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
- The artifacts include per-criterion labels:
  - Inclusion: included / not included / not enough information / not applicable
  - Exclusion: excluded / not excluded / not enough information / not applicable
- Final label logic:
  - Any blocking label => ineligible.
  - No blocking but unknowns => unclear (or optimistic-eligible depending on provided label).
- Keep concise and checkable.
- Even if you believe it is wrong, stick to logic implied by artifacts.

# === EMPHASIS REQUIREMENT (TOP PRIORITY) ===
1) If ineligible, start with the single most decision-driving criterion (or small set).
2) Prioritize criteria likely to flip the reference assessment.
3) Do NOT mention "system" / "disagree" / "compared to" etc.
4) Present checkable criteria + evidence/unknowns.

# === OUTPUT REQUIREMENTS ===
Return a SINGLE JSON object (no markdown fences) with exactly these keys:
{
  "rationale": "one concise paragraph",
  "key_points": ["...", "..."],
  "evidence_quotes": {
    "patient_note": ["short quote 1", "short quote 2"],
    "trial_eligibility_text": ["short quote 1", "short quote 2"],
    "high_priority_requirements": ["criterion phrase(s) that drive the label"],
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


DEFAULT_VERBALIZER_SYSTEM_MESSAGE = "You are a careful, evidence-grounded clinical trial eligibility explainer."
DEFAULT_DESTYLE_SYSTEM_MESSAGE = "You are a careful editor who rewrites rationales in neutral plain English."

DEFAULT_DESTYLE_TEMPLATE_PATH = "<SATIR_ROOT>/irsrc/meval/prompts/destyle_verbalization.prompt"
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

def now_iso_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _truncate(s: Optional[str], n: int = 12000) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]..."


# =============================================================================
# Prompt filling
# =============================================================================

def fill_template(tpl: str, mapping: Dict[str, str]) -> str:
    out = tpl
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


# =============================================================================
# Trial id parent parsing
# =============================================================================

_NCT_PARENT_RE = re.compile(r"^(NCT\d{8})([a-z])?$", re.IGNORECASE)

def parse_parent_and_suffix(trial_id: str) -> Tuple[str, Optional[str]]:
    m = _NCT_PARENT_RE.match((trial_id or "").strip())
    if not m:
        return (trial_id or "").strip(), None
    return m.group(1).upper(), (m.group(2).lower() if m.group(2) else None)


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

def format_trial_description_from_corpus_entry(entry: Dict[str, Any]) -> str:
    """
    Approximate match_patient_to_trial.format_trial_description()
    using corpus.jsonl shape.
    """
    md = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    title = md.get("brief_title") or entry.get("title") or md.get("official_title") or entry.get("brief_title") or ""
    summary = md.get("brief_summary") or entry.get("brief_summary") or md.get("summary") or entry.get("text") or ""
    inc = md.get("inclusion_criteria") or entry.get("inclusion_criteria") or ""
    exc = md.get("exclusion_criteria") or entry.get("exclusion_criteria") or ""

    if isinstance(inc, list):
        inc = "\n".join(str(x) for x in inc)
    if isinstance(exc, list):
        exc = "\n".join(str(x) for x in exc)

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary:\n{summary}")
    if inc:
        parts.append(f"Inclusion Criteria:\n{inc}")
    if exc:
        parts.append(f"Exclusion Criteria:\n{exc}")
    if not parts:
        parts.append(json.dumps(entry, ensure_ascii=False, indent=2))
    return "\n\n".join(parts).strip()


# =============================================================================
# Robust JSON-ish extraction (LLM outputs)
# =============================================================================

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
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

        # strict json
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj, None
            return _wrap_non_dict(obj), None
        except Exception as e:
            errors.append(f"{tag}: json strict failed: {e!r}")

        # relaxed json
        try:
            c2 = _remove_trailing_commas(_quote_bare_keys(_jsonify_python_literals(c)))
            obj2 = json.loads(c2)
            if isinstance(obj2, dict):
                return obj2, None
            return _wrap_non_dict(obj2), None
        except Exception as e:
            errors.append(f"{tag}: json relaxed failed: {e!r}")

        # python literal eval
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
                    "Set OPENAI_MODEL (deployment name) or pass --*-model-name."
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

    # Preferred: engine(prompt, system_message=...)
    try:
        out = engine(prompt, system_message=system_message, **kwargs)
        txt = _normalize_engine_output(out)
        if txt.strip() == "":
            raise RuntimeError("Engine returned empty text.")
        return txt
    except TypeError:
        pass

    # Fallback: engine(prompt, **kwargs)
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
# Decisions / artifacts loading
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

# ---------- System 1 (LLM judge) ----------

def load_llm_judge_payload(pair_dir: Path, trial_parent_id: str) -> Dict[str, Any]:
    """
    Prefer: <pair_dir>/<PARENT>__llm_judge.json
    Fallback: <pair_dir>/<PARENT>__full.json -> ["llm_judge"]
    """
    llm_path = pair_dir / f"{trial_parent_id}__llm_judge.json"
    if llm_path.exists():
        obj = read_json(llm_path)
        return obj if isinstance(obj, dict) else {}

    full_path = pair_dir / f"{trial_parent_id}__full.json"
    if full_path.exists():
        full = read_json(full_path)
        if isinstance(full, dict) and isinstance(full.get("llm_judge"), dict):
            return full["llm_judge"]
    return {}

def load_llm_judge_result(pair_dir: Path, trial_parent_id: str) -> Dict[str, Any]:
    payload = load_llm_judge_payload(pair_dir, trial_parent_id)
    res = payload.get("result") if isinstance(payload, dict) else None
    return res if isinstance(res, dict) else {}

def extract_nl_rationale(llm_result: Dict[str, Any]) -> str:
    for k in ["rationale", "reasoning", "explanation", "justification", "analysis", "notes"]:
        v = llm_result.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = llm_result.get("rationales")
    if isinstance(v, list) and v:
        ss = [str(x).strip() for x in v if str(x).strip()]
        if ss:
            return "\n".join(ss).strip()
    return _truncate(pretty(llm_result), 6000) or "[NL rationale missing]"

def normalize_eligible_value(llm_result: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(llm_result, dict) or not llm_result:
        return None
    v = llm_result.get("eligible")
    if v is None:
        v = llm_result.get("decision")
    return _normalize_boollike(v)

# ---------- System 2 (SMT parent OR artifacts) ----------

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
    dedup: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup

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

# ---------- System 3 (TrialGPT judge) ----------

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
# Extracted variable values (SMT only; best-effort)
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
# Verbalizer normalization
# =============================================================================

def normalize_verbalizer_output(raw_resp: str, *, is_trialgpt: bool) -> Dict[str, Any]:
    parsed, perr = extract_json_from_text(raw_resp)
    if parsed is None or not isinstance(parsed, dict):
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

    return {"rationale": str(rationale).strip(), "key_points": key_points, "evidence_quotes": evidence_quotes}


# =============================================================================
# Engine config
# =============================================================================

@dataclass
class EngineConfig:
    endpoint: str
    api_key_env_var: str
    model_hint: str
    api_version: Optional[str]
    temperature: float
    max_tokens: Optional[int]
    system_message: str


# =============================================================================
# System 1 prompt reconstruction
# =============================================================================

def reconstruct_llm_judge_prompt_used(
    llm_payload: Dict[str, Any],
    *,
    patient_note: str,
    trial_desc: str,
    max_chars: int = 24000,
) -> str:
    """
    Best-effort:
      - read template from payload["prompt_path"]
      - replace #CLINICAL_TRIAL_DESCRIPTION# and #PATIENT_NOTE#
    """
    if not isinstance(llm_payload, dict) or not llm_payload:
        return "[missing llm_judge payload]"

    prompt_path = llm_payload.get("prompt_path")
    template = None
    if isinstance(prompt_path, str) and prompt_path.strip():
        p = Path(prompt_path).expanduser()
        if p.exists():
            try:
                template = read_text(p)
            except Exception:
                template = None

    if not template:
        # still provide something informative
        return _truncate(f"[prompt_path missing/unreadable]\n{pretty(llm_payload)}", max_chars) or ""

    rendered = template.replace("#CLINICAL_TRIAL_DESCRIPTION#", trial_desc).replace("#PATIENT_NOTE#", patient_note)

    out = (
        f"[prompt_path]\n{prompt_path}\n\n"
        f"[rendered_prompt]\n{rendered}"
    )
    return _truncate(out, max_chars) or out


# =============================================================================
# Main pipeline (single case)
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Single-case: verbalize SMT+TrialGPT, then destyle 3-system rationales.")

    ap.add_argument("--match-out-root", type=str, default="../../cmsrc/match_out",
                    help="Root like <match_out>/<patient_id>/... (default: ../../cmsrc/match_out)")
    ap.add_argument("--pair-dir", type=str, default=None,
                    help="Optional explicit path to <match_out>/<patient_id>/ (overrides --match-out-root)")
    ap.add_argument("--patient-id", type=str, required=True)
    ap.add_argument("--trial-id", type=str, required=True, help="Trial id or subcohort id; will normalize to parent NCT########")

    ap.add_argument("--notes-jsonl", type=str, default="../../dataset/clinical_trial/sigir/queries.jsonl")
    ap.add_argument("--trial-corpus-jsonl", type=str, default="../../dataset/clinical_trial/sigir/corpus.jsonl")

    ap.add_argument("--out-root", type=str, default="./destyle_results")
    ap.add_argument("--mbench-root", type=str, default="./destyle_mbench")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--smt-verbalizer-template", type=str, default=None)
    ap.add_argument("--trialgpt-verbalizer-template", type=str, default=None)

    ap.add_argument("--destyle-template", type=str, default=DEFAULT_DESTYLE_TEMPLATE_PATH)

    ap.add_argument("--api-key-env-var", type=str, default=os.environ.get("OPENAI_API_KEY_ENV_VAR", "OPENAI_API_KEY"))

    # Verbalizer engine
    ap.add_argument("--verbalizer-endpoint", type=str, default="")
    ap.add_argument("--verbalizer-model-name", type=str, default="")
    ap.add_argument("--verbalizer-api-version", type=str, default="")
    ap.add_argument("--verbalizer-temperature", type=float, default=0.0)
    ap.add_argument("--verbalizer-max-tokens", type=int, default=1200)

    # Destyle engine (can be same or different)
    ap.add_argument("--destyle-endpoint", type=str, default="")
    ap.add_argument("--destyle-model-name", type=str, default="")
    ap.add_argument("--destyle-api-version", type=str, default="")
    ap.add_argument("--destyle-temperature", type=float, default=0.0)
    ap.add_argument("--destyle-max-tokens", type=int, default=2000)

    ap.add_argument("--max-patient-note-chars", type=int, default=12000)
    ap.add_argument("--max-trial-text-chars", type=int, default=12000)

    args = ap.parse_args()

    patient_id = str(args.patient_id)
    trial_parent_id, _ = parse_parent_and_suffix(str(args.trial_id))

    # Pair dir
    if args.pair_dir:
        pair_dir = Path(args.pair_dir)
    else:
        pair_dir = Path(args.match_out_root) / patient_id
    if not pair_dir.exists():
        raise SystemExit(f"pair_dir not found: {pair_dir}")

    notes_jsonl = Path(args.notes_jsonl)
    corpus_jsonl = Path(args.trial_corpus_jsonl)

    out_root = Path(args.out_root) / patient_id / trial_parent_id
    mbench_root = Path(args.mbench_root) / patient_id / trial_parent_id
    ensure_dir(out_root)
    ensure_dir(mbench_root)

    # Templates
    smt_tpl = read_text(Path(args.smt_verbalizer_template)) if args.smt_verbalizer_template else DEFAULT_SMT_VERBALIZER_TEMPLATE
    tg_tpl = read_text(Path(args.trialgpt_verbalizer_template)) if args.trialgpt_verbalizer_template else DEFAULT_TRIALGPT_VERBALIZER_TEMPLATE

    destyle_template_path = Path(args.destyle_template)
    if not destyle_template_path.exists():
        raise SystemExit(f"destyle template not found: {destyle_template_path}")
    destyle_tpl = read_text(destyle_template_path)

    # Engine configs
    v_ep, v_model, v_av = _resolve_endpoint_model_and_api_version(
        endpoint_arg=args.verbalizer_endpoint,
        model_arg=args.verbalizer_model_name,
        api_version_arg=args.verbalizer_api_version,
        endpoint_env_fallback="OPENAI_VERBALIZER_ENDPOINT",
        model_env_fallback="OPENAI_VERBALIZER_MODEL",
        api_version_env_fallback="OPENAI_VERBALIZER_API_VERSION",
        default_model="gpt-5",
    )
    d_ep, d_model, d_av = _resolve_endpoint_model_and_api_version(
        endpoint_arg=args.destyle_endpoint,
        model_arg=args.destyle_model_name,
        api_version_arg=args.destyle_api_version,
        endpoint_env_fallback="OPENAI_DESTYLE_ENDPOINT",
        model_env_fallback="OPENAI_DESTYLE_MODEL",
        api_version_env_fallback="OPENAI_DESTYLE_API_VERSION",
        default_model="gpt-5",
    )

    verbalizer_engine_cfg = EngineConfig(
        endpoint=v_ep,
        api_key_env_var=args.api_key_env_var,
        model_hint=v_model,
        api_version=v_av,
        temperature=float(args.verbalizer_temperature),
        max_tokens=int(args.verbalizer_max_tokens) if args.verbalizer_max_tokens else None,
        system_message=DEFAULT_VERBALIZER_SYSTEM_MESSAGE,
    )
    destyle_engine_cfg = EngineConfig(
        endpoint=d_ep,
        api_key_env_var=args.api_key_env_var,
        model_hint=d_model,
        api_version=d_av,
        temperature=float(args.destyle_temperature),
        max_tokens=int(args.destyle_max_tokens) if args.destyle_max_tokens else None,
        system_message=DEFAULT_DESTYLE_SYSTEM_MESSAGE,
    )

    # -------------------------------------------------------------------------
    # Load patient note + trial text
    # -------------------------------------------------------------------------
    patient_note = load_patient_note_from_queries_jsonl(notes_jsonl, patient_id)
    patient_note = _truncate(patient_note, int(args.max_patient_note_chars)) or patient_note

    trial_entry = load_trial_entry_from_corpus_jsonl(corpus_jsonl, trial_parent_id)
    if trial_entry is None:
        # try subcohorts if parent missing
        por = load_parent_or(pair_dir, trial_parent_id)
        sub_ids = subcohort_ids_from_parent_or(por) if isinstance(por, dict) else []
        for sid in sub_ids[:12]:
            te = load_trial_entry_from_corpus_jsonl(corpus_jsonl, sid)
            if te is not None:
                trial_entry = te
                break

    if trial_entry is None:
        trial_text = _truncate(pretty({"_error": "TRIAL ENTRY NOT FOUND IN CORPUS", "trial_parent_id": trial_parent_id}),
                               int(args.max_trial_text_chars)) or ""
        trial_desc = trial_text
    else:
        trial_text = trial_entry_to_prompt_text(trial_entry, max_chars=int(args.max_trial_text_chars))
        trial_desc = format_trial_description_from_corpus_entry(trial_entry)

    # -------------------------------------------------------------------------
    # System 1: LLM judge (prompt + output)
    # -------------------------------------------------------------------------
    llm_payload = load_llm_judge_payload(pair_dir, trial_parent_id)
    llm_result = load_llm_judge_result(pair_dir, trial_parent_id)

    nl_bool = normalize_eligible_value(llm_result)
    nl_label = bool_to_label(nl_bool)
    nl_rationale = _truncate(extract_nl_rationale(llm_result), 6000) or ""

    prompt_used_sys1 = reconstruct_llm_judge_prompt_used(
        llm_payload,
        patient_note=patient_note,
        trial_desc=trial_desc,
        max_chars=24000,
    )
    output_sys1_obj = {
        "decision": nl_label,
        "eligible": nl_bool,
        "rationale": nl_rationale,
        "llm_judge_result": llm_result,
        "llm_judge_payload_meta": {k: llm_payload.get(k) for k in ["prompt_path", "temperature", "trial_id_parent", "patient_id"] if isinstance(llm_payload, dict)},
    }
    output_sys1 = pretty(output_sys1_obj)

    write_text(out_root / "system1_prompt_used.txt", prompt_used_sys1)
    write_json(out_root / "system1_output.json", output_sys1_obj)

    # -------------------------------------------------------------------------
    # System 2: SMT verbalizer
    # -------------------------------------------------------------------------
    smt_bool, smt_art = derive_smt_parent_decision_and_artifacts(pair_dir, trial_parent_id)
    smt_label = bool_to_label(smt_bool)

    inc_vars_obj, exc_vars_obj = load_extracted_vars_for_parent_or_subcohorts(pair_dir, trial_parent_id)
    smt_artifacts_pretty = _truncate(pretty(smt_art), 12000) or pretty(smt_art)

    prompt_used_sys2 = fill_template(
        smt_tpl,
        {
            "#PATIENT_NOTE#": patient_note,
            "#TRIAL_ELIGIBILITY_TEXT#": trial_text,
            "#SYSTEM_DECISION_LABEL#": smt_label,
            "#SYSTEM_DECISION_ARTIFACTS#": smt_artifacts_pretty,
            "#INCLUSION_EXTRACTED_VARIABLE_VALUES#": _truncate(pretty(inc_vars_obj), 12000) or pretty(inc_vars_obj),
            "#EXCLUSION_EXTRACTED_VARIABLE_VALUES#": _truncate(pretty(exc_vars_obj), 12000) or pretty(exc_vars_obj),
            "#NL_DECISION#": nl_label,
            "#NL_RATIONALE#": nl_rationale or "[NL rationale missing]",
        },
    )

    write_text(out_root / "system2_prompt_used.txt", prompt_used_sys2)

    eng_v = _get_cached_engine(
        "verbalizer",
        endpoint=verbalizer_engine_cfg.endpoint,
        api_key_env_var=verbalizer_engine_cfg.api_key_env_var,
        model_hint=verbalizer_engine_cfg.model_hint,
    )
    raw_sys2 = _call_engine_text(
        eng_v,
        prompt_used_sys2,
        system_message=verbalizer_engine_cfg.system_message,
        temperature=verbalizer_engine_cfg.temperature,
        max_tokens=verbalizer_engine_cfg.max_tokens,
        api_version=verbalizer_engine_cfg.api_version,
    )
    write_text(out_root / "system2_smt_verbalizer_response.txt", raw_sys2)

    norm_sys2 = normalize_verbalizer_output(raw_sys2, is_trialgpt=False)
    output_sys2_obj = {"decision": smt_label, "eligible": smt_bool, **norm_sys2}
    output_sys2 = pretty(output_sys2_obj)
    write_json(out_root / "system2_output.json", output_sys2_obj)

    # -------------------------------------------------------------------------
    # System 3: TrialGPT verbalizer
    # -------------------------------------------------------------------------
    tg = load_trialgpt_judge(pair_dir, trial_parent_id)
    tg_bool, tg_src = trialgpt_final_bool_and_source(tg)
    tg_label = bool_to_label(tg_bool)

    tg_compact = compact_trialgpt_for_prompt(tg) if tg else {"_error": "missing_trialgpt"}
    tg_compact_pretty = _truncate(pretty(tg_compact), 12000) or pretty(tg_compact)

    prompt_used_sys3 = fill_template(
        tg_tpl,
        {
            "#PATIENT_NOTE#": patient_note,
            "#TRIAL_ELIGIBILITY_TEXT#": trial_text,
            "#SYSTEM_DECISION_LABEL#": tg_label,
            "#CRITERION_DECISION_ARTIFACTS#": tg_compact_pretty,
            "#NL_DECISION#": nl_label,
            "#NL_RATIONALE#": nl_rationale or "[NL rationale missing]",
        },
    )

    write_text(out_root / "system3_prompt_used.txt", prompt_used_sys3)

    raw_sys3 = _call_engine_text(
        eng_v,
        prompt_used_sys3,
        system_message=verbalizer_engine_cfg.system_message,
        temperature=verbalizer_engine_cfg.temperature,
        max_tokens=verbalizer_engine_cfg.max_tokens,
        api_version=verbalizer_engine_cfg.api_version,
    )
    write_text(out_root / "system3_trialgpt_verbalizer_response.txt", raw_sys3)

    norm_sys3 = normalize_verbalizer_output(raw_sys3, is_trialgpt=True)
    output_sys3_obj = {"decision": tg_label, "eligible": tg_bool, "trialgpt_src": tg_src, **norm_sys3}
    output_sys3 = pretty(output_sys3_obj)
    write_json(out_root / "system3_output.json", output_sys3_obj)

    # -------------------------------------------------------------------------
    # Destyle: call destyle_verbalization.prompt
    # -------------------------------------------------------------------------
    destyle_prompt = fill_template(
        destyle_tpl,
        {
            "#PATIENT_NOTE#": patient_note,
            "#TRIAL_ELIGIBILITY_TEXT#": trial_text,
            "#PROMPT_USED_BY_SYSTEM_1#": prompt_used_sys1,
            "#OUTPUT_SYSTEM_1#": output_sys1,
            "#PROMPT_USED_BY_SYSTEM_2#": prompt_used_sys2,
            "#OUTPUT_SYSTEM_2#": output_sys2,
            "#PROMPT_USED_BY_SYSTEM_3#": prompt_used_sys3,
            "#OUTPUT_SYSTEM_3#": output_sys3,
        },
    )
    write_text(out_root / "destyle_prompt.txt", destyle_prompt)

    eng_d = _get_cached_engine(
        "destyle",
        endpoint=destyle_engine_cfg.endpoint,
        api_key_env_var=destyle_engine_cfg.api_key_env_var,
        model_hint=destyle_engine_cfg.model_hint,
    )
    raw_destyle = _call_engine_text(
        eng_d,
        destyle_prompt,
        system_message=destyle_engine_cfg.system_message,
        temperature=destyle_engine_cfg.temperature,
        max_tokens=destyle_engine_cfg.max_tokens,
        api_version=destyle_engine_cfg.api_version,
    )
    write_text(out_root / "destyle_response.txt", raw_destyle)

    destyled_obj, destyled_err = extract_json_from_text(raw_destyle)
    if destyled_obj is None:
        destyled_obj = {"_parse_error": destyled_err, "raw_text": raw_destyle}
    write_json(out_root / "destyled.json", destyled_obj)

    # Bundle everything
    bundle = {
        "timestamp_utc": now_iso_utc(),
        "patient_id": patient_id,
        "trial_parent_id": trial_parent_id,
        "pair_dir": str(pair_dir),
        "notes_jsonl": str(notes_jsonl),
        "trial_corpus_jsonl": str(corpus_jsonl),
        "system_1": {
            "tag": "llm_judge",
            "prompt_used": prompt_used_sys1,
            "output": output_sys1_obj,
        },
        "system_2": {
            "tag": "smt_verbalized",
            "prompt_used": prompt_used_sys2,
            "output": output_sys2_obj,
        },
        "system_3": {
            "tag": "trialgpt_verbalized",
            "prompt_used": prompt_used_sys3,
            "output": output_sys3_obj,
        },
        "destyle": {
            "template_path": str(destyle_template_path),
            "prompt_sha256": sha256_text(destyle_prompt),
            "response_sha256": sha256_text(raw_destyle),
            "parsed": destyled_obj,
        },
        "engine_cfg": {
            "verbalizer": {
                "endpoint": verbalizer_engine_cfg.endpoint,
                "model_hint": verbalizer_engine_cfg.model_hint,
                "api_version": verbalizer_engine_cfg.api_version,
                "temperature": verbalizer_engine_cfg.temperature,
                "max_tokens": verbalizer_engine_cfg.max_tokens,
                "system_message": verbalizer_engine_cfg.system_message,
            },
            "destyle": {
                "endpoint": destyle_engine_cfg.endpoint,
                "model_hint": destyle_engine_cfg.model_hint,
                "api_version": destyle_engine_cfg.api_version,
                "temperature": destyle_engine_cfg.temperature,
                "max_tokens": destyle_engine_cfg.max_tokens,
                "system_message": destyle_engine_cfg.system_message,
            },
        },
    }
    write_json(out_root / "bundle.json", bundle)

    print(json.dumps({"ok": True, "out_dir": str(out_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()