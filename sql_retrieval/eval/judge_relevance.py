#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_relevance.py
(base prompt + definition snippets + OPTIONAL per-mode instructions snippet)

Supported prompt keys:
  - cc
  - ccr
  - all
  - all-explore   (NEW)

Files (defaults):
  prompts/relevance_base.prompt
  prompts/relevance_def_cc.prompt
  prompts/relevance_def_ccr.prompt
  prompts/relevance_def_all.prompt
  prompts/relevance_def_all-explore.prompt                (NEW)

Optional instructions snippets (defaults; only used if base has #SPECIFIC_INSTRUCTIONS#):
  prompts/relevance_instructions_cc.prompt
  prompts/relevance_instructions_ccr.prompt
  prompts/relevance_instructions_all.prompt
  prompts/relevance_instructions_all-explore.prompt       (NEW)

Notes:
- If the instructions file for a key is missing, we inject "" (empty).
- If the base prompt does not contain #SPECIFIC_INSTRUCTIONS#, we just ignore instructions safely.
- RELEVANCE_DEFINITIONS is computed from RELEVANCE_DEFINITION_PROMPT_PATHS at import time.
  You may monkeypatch either dict before calling judge_relevance(), but if you patch paths after import,
  you should also patch RELEVANCE_DEFINITIONS accordingly (or just patch RELEVANCE_DEFINITIONS directly).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Literal

# Be robust to both engine module names used in your repo.
try:
    from smt_core.inference_engine_5 import AzureInferenceEngine
except ImportError:
    from smt_core.inference_engine import AzureInferenceEngine

from prompt_trace import PromptTrace

_PROMPT_DIR = Path(__file__).parent / "prompts"

# -------------------------
# Prompt paths (can be monkeypatched by callers)
# -------------------------

RELEVANCE_BASE_PROMPT_PATH: Path = _PROMPT_DIR / "relevance_base.prompt"

RELEVANCE_DEFINITION_PROMPT_PATHS: Dict[str, Path] = {
    "cc": _PROMPT_DIR / "relevance_def_cc.prompt",
    "ccr": _PROMPT_DIR / "relevance_def_ccr.prompt",
    "all": _PROMPT_DIR / "relevance_def_all.prompt",
    # NEW:
    "all-explore": _PROMPT_DIR / "relevance_def_all-explore.prompt",
}

# Optional per-mode instructions (only used if base contains #SPECIFIC_INSTRUCTIONS#)
RELEVANCE_INSTRUCTIONS_PROMPT_PATHS: Dict[str, Path] = {
    "cc": _PROMPT_DIR / "relevance_instructions_cc.prompt",
    "ccr": _PROMPT_DIR / "relevance_instructions_ccr.prompt",
    "all": _PROMPT_DIR / "relevance_instructions_all.prompt",
    # NEW:
    "all-explore": _PROMPT_DIR / "relevance_instructions_all-explore.prompt",
}

# Keep a narrow Literal for default usage, but allow str keys at runtime.
RelevancePromptKey = Literal["cc", "ccr", "all", "all-explore"]


def _load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _maybe_load_text(path: Path) -> str:
    try:
        if path.exists() and path.is_file():
            return _load_text(path)
    except Exception:
        pass
    return ""


# -------------------------
# Load base template + definitions at import time
# -------------------------

if not RELEVANCE_BASE_PROMPT_PATH.exists():
    raise FileNotFoundError(f"Missing relevance base prompt file: {RELEVANCE_BASE_PROMPT_PATH}")

_BASE_TEMPLATE = _load_text(RELEVANCE_BASE_PROMPT_PATH)

RELEVANCE_DEFINITIONS: Dict[str, str] = {}
for k, p in RELEVANCE_DEFINITION_PROMPT_PATHS.items():
    if not p.exists():
        raise FileNotFoundError(f"Missing relevance definition prompt file for key='{k}': {p}")
    RELEVANCE_DEFINITIONS[k] = _load_text(p)


def get_default_engine() -> AzureInferenceEngine:
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "OPENAI_ENDPOINT env var is not set. "
            "Set it, or pass an AzureInferenceEngine instance into judge_relevance()."
        )

    # NOTE: your retrieval runner sometimes needs temp=1.0 for Azure.
    # Keep module default at 0.0 (unchanged); caller can pass engine with its defaults.
    return AzureInferenceEngine(
        endpoint=endpoint,
        model_name="gpt-4.1",
        default_temperature=0.0,
        default_max_tokens=None,
    )


def _render_prompt(
    *,
    base_template: str,
    definition: str,
    specific_instructions: str,
    patient_note: str,
    clinical_trial_description: str,
) -> str:
    p = (
        base_template
        .replace("#RELEVANCE_DEFINITION#", definition.strip())
        .replace("#PATIENT_NOTE#", patient_note)
        .replace("#TRIAL_DESCRIPTION#", clinical_trial_description)
    )

    # Optional placeholder: only replace if present (safe for older base templates)
    if "#SPECIFIC_INSTRUCTIONS#" in p:
        p = p.replace("#SPECIFIC_INSTRUCTIONS#", (specific_instructions or "").strip())

    return p


def judge_relevance(
    clinical_trial_description: str,
    patient_note: str,
    engine: Optional[AzureInferenceEngine] = None,
    prompt_key: str = "cc",
    trace: Optional[PromptTrace] = None,
    patient_id: Optional[str] = None,
    trial_id: Optional[str] = None,
) -> str:
    if engine is None:
        engine = get_default_engine()

    if prompt_key not in RELEVANCE_DEFINITIONS:
        raise ValueError(
            f"Unknown prompt_key='{prompt_key}'. Expected one of: {sorted(RELEVANCE_DEFINITIONS.keys())}"
        )

    definition = RELEVANCE_DEFINITIONS[prompt_key]

    instr_path = RELEVANCE_INSTRUCTIONS_PROMPT_PATHS.get(prompt_key)
    specific_instructions = _maybe_load_text(instr_path) if instr_path is not None else ""

    prompt = _render_prompt(
        base_template=_BASE_TEMPLATE,
        definition=definition,
        specific_instructions=specific_instructions,
        patient_note=patient_note,
        clinical_trial_description=clinical_trial_description,
    )

    outputs: Sequence[str] = engine(prompt)
    out0 = outputs[0]

    if trace is not None and patient_id is not None and trial_id is not None:
        trace.log(
            stage="relevance",
            patient_id=patient_id,
            trial_id=trial_id,
            prompt=prompt,
            output=out0,
        )

    return out0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run relevance judging between a clinical trial and a patient note.")
    parser.add_argument("--trial-file", type=str, default="test_files/test_trial.txt")
    parser.add_argument("--note-file", type=str, default="test_files/test_patient.txt")
    parser.add_argument("--prompt-key", type=str, choices=sorted(RELEVANCE_DEFINITIONS.keys()), default="cc")
    args = parser.parse_args()

    trial_text = Path(args.trial_file).read_text(encoding="utf-8")
    note_text = Path(args.note_file).read_text(encoding="utf-8")
    print(judge_relevance(trial_text, note_text, prompt_key=args.prompt_key))