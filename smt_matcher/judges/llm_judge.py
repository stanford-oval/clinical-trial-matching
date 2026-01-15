"""
LLM eligibility judge: GPT-4 NL decision on full trial text + patient note.

A parallel decision baseline for comparing against SMT matcher output.
Extracted from cmsrc/match_patient_to_trial.py.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
from typing import Any, Dict, Optional

from smt_matcher.cache import (
    CACHE_SCHEMA_VERSION,
    cache_file_for,
    cache_payload_matches,
    safe_read_json,
    safe_write_json_atomic,
    stable_hash,
)

_NCT_PARENT_RE = re.compile(r"^(NCT\d{8})")


def parent_nct_id(trial_id: str) -> str:
    if not isinstance(trial_id, str):
        return str(trial_id)
    m = _NCT_PARENT_RE.match(trial_id.strip())
    return (m.group(1).upper() if m else trial_id.strip())


# ── Trial + patient rendering ────────────────────────────────────────────

def format_trial_description(trial_obj: Dict[str, Any]) -> str:
    title = (trial_obj.get("brief_title") or trial_obj.get("title")
             or trial_obj.get("official_title") or "")
    summary = (trial_obj.get("brief_summary") or trial_obj.get("summary")
               or trial_obj.get("description") or "")
    inc = trial_obj.get("inclusion_criteria") or trial_obj.get("inclusion") or ""
    exc = trial_obj.get("exclusion_criteria") or trial_obj.get("exclusion") or ""

    if isinstance(inc, list):
        inc = "\n".join(str(x) for x in inc)
    if isinstance(exc, list):
        exc = "\n".join(str(x) for x in exc)

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary:\n{summary}")
    if inc:
        parts.append(f"Inclusion Criteria:\n{inc}")
    if exc:
        parts.append(f"Exclusion Criteria:\n{exc}")
    if not parts:
        parts.append(json.dumps(trial_obj, ensure_ascii=False, indent=2))
    return "\n\n".join(parts).strip()


def render_eligibility_prompt(template: str, trial_desc: str, patient_note: str) -> str:
    return (template.replace("#CLINICAL_TRIAL_DESCRIPTION#", trial_desc)
                    .replace("#PATIENT_NOTE#", patient_note))


def extract_patient_note_text(patient: Dict[str, Any]) -> str:
    for k in ("text", "note", "patient_note"):
        v = patient.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return json.dumps(patient, ensure_ascii=False, indent=2)


# ── Engine call adapter ──────────────────────────────────────────────────

def _call_engine_text(engine: Any, prompt: str, *, temperature: Optional[float] = 0.0) -> str:
    """Call an AzureInferenceEngine-like engine with a single prompt. Returns text."""
    messages = [{"role": "user", "content": prompt}]

    def _extract_text(out: Any) -> Optional[str]:
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            if isinstance(out.get("text"), str):
                return out["text"]
            if isinstance(out.get("content"), str):
                return out["content"]
            choices = out.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    msg = c0.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    if isinstance(c0.get("text"), str):
                        return c0["text"]
        return None

    last_err: Optional[Exception] = None

    for method_name in ("chat_complete", "complete", "invoke", "run", "generate", "completion"):
        m = getattr(engine, method_name, None)
        if not callable(m):
            continue

        for call_form in (
            lambda: m(prompt, temperature=temperature),
            lambda: m(prompt),
            lambda: m(messages, temperature=temperature),
            lambda: m(messages),
            lambda: m(messages=messages, temperature=temperature),
        ):
            try:
                out = call_form()
                text = _extract_text(out)
                if text is not None:
                    return text
            except TypeError:
                continue
            except Exception as e:
                last_err = e

    if callable(engine):
        for call in (
            lambda: engine(messages=messages, temperature=temperature),
            lambda: engine(messages),
            lambda: engine(prompt),
        ):
            try:
                out = call()
                text = _extract_text(out)
                if text is not None:
                    return text
            except Exception as e:
                last_err = e

    raise RuntimeError(
        "Could not call engine with prompt or chat messages. "
        f"Last error: {type(last_err).__name__ if last_err else 'none'}: {last_err}"
    )


# ── Output parsing ───────────────────────────────────────────────────────

def parse_eligibility_judge_output(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()

    m_reason = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reasoning = m_reason.group(1).strip() if m_reason else None

    m = re.search(
        r"<eligibility_decision>\s*(\{.*?\})\s*</eligibility_decision>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    json_str = m.group(1).strip() if m else None
    if json_str is None:
        m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        json_str = m2.group(1).strip() if m2 else None

    obj: Dict[str, Any] = {}
    parse_error = None
    if json_str:
        try:
            obj = json.loads(json_str)
        except Exception as e:
            parse_error = f"json_parse_error: {e}"
            obj = {}
    else:
        parse_error = "no_json_found"

    elig = obj.get("eligibility")
    eligible_bool: Optional[bool] = None
    if isinstance(elig, str):
        s = elig.strip().lower()
        if s == "eligible":
            eligible_bool = True
        elif s == "ineligible":
            eligible_bool = False

    return {
        "eligible": eligible_bool,
        "eligibility": elig,
        "reasoning": reasoning,
        "explanation": obj.get("explanation"),
        "parse_error": parse_error,
        "raw_text": raw_text,
    }


# ── Main judge entry point ───────────────────────────────────────────────

def _llm_judge_fingerprint(
    *,
    template: str,
    trial_desc: str,
    patient_note: str,
    model_name: str,
    temperature: float,
    prompt_path: pathlib.Path,
    trial_id: str,
    patient_id: str,
) -> str:
    return stable_hash({
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage": "llm_judge",
        "template": template,
        "trial_desc": trial_desc,
        "patient_note": patient_note,
        "model_name": model_name,
        "temperature": temperature,
        "prompt_path": str(prompt_path),
        "trial_id": trial_id,
        "patient_id": patient_id,
    })


def run_llm_eligibility_judge(
    trial_id: str,
    trial_obj: Dict[str, Any],
    patient: Dict[str, Any],
    engine: Any,
    *,
    prompt_path: pathlib.Path,
    out_root: pathlib.Path,
    model_name: str = "gpt-4.1",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Run GPT-4 NL eligibility judge on a (trial, patient) pair.

    Args:
        trial_id:    Trial identifier (NCT or variant)
        trial_obj:   Dict with brief_title/brief_summary/inclusion_criteria/exclusion_criteria
        patient:     Dict with text/note/patient_note
        engine:      AzureInferenceEngine instance
        prompt_path: Path to eligibility.*.prompt template
        out_root:    Directory for cache (will create <out_root>/_prompt_cache/llm_judge/)
        model_name:  Model identifier (for fingerprinting)
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        Dict with keys: trial_id_original, trial_id_parent, patient_id,
        prompt_path, temperature, result (eligible/eligibility/reasoning/explanation/raw_text).
    """
    template = prompt_path.read_text(encoding="utf-8")
    parent_id = parent_nct_id(trial_id)
    patient_id = (patient.get("patient_id") or patient.get("_id") or patient.get("id") or "unknown")
    trial_desc = format_trial_description(trial_obj)
    patient_note = extract_patient_note_text(patient)

    fingerprint = _llm_judge_fingerprint(
        template=template,
        trial_desc=trial_desc,
        patient_note=patient_note,
        model_name=model_name,
        temperature=temperature,
        prompt_path=prompt_path,
        trial_id=parent_id,
        patient_id=str(patient_id),
    )

    cache_path = cache_file_for(out_root, "llm_judge", fingerprint)
    cached = safe_read_json(cache_path)
    if cache_payload_matches(cached, fingerprint):
        payload = cached.get("payload")
        if isinstance(payload, dict):
            return payload

    prompt = render_eligibility_prompt(template, trial_desc, patient_note)
    raw = _call_engine_text(engine, prompt, temperature=temperature)
    parsed = parse_eligibility_judge_output(raw)

    payload = {
        "cache": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": "llm_judge",
            "fingerprint": fingerprint,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "temperature": temperature,
        },
        "trial_id_original": trial_id,
        "trial_id_parent": parent_id,
        "patient_id": patient_id,
        "prompt_path": str(prompt_path),
        "temperature": temperature,
        "result": parsed,
    }

    safe_write_json_atomic(cache_path, {"cache": payload["cache"], "payload": payload})
    return payload
