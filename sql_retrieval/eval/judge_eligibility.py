#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_eligibility.py

Uses prompts/eligibility.prompt template and fills:
  #PATIENT_NOTE#
  #TRIAL_DESCRIPTION#
  #SUBCOHORTS_OF_INTEREST#   (the relevance output)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

# Be robust to both engine module names used in your repo.
try:
    from smt_core.inference_engine_5 import AzureInferenceEngine
except ImportError:
    from smt_core.inference_engine import AzureInferenceEngine

from prompt_trace import PromptTrace

_PROMPT_DIR = Path(__file__).parent / "prompts"
ELIGIBILITY_PROMPT_PATH: Path = _PROMPT_DIR / "eligibility.prompt"


def _load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


if not ELIGIBILITY_PROMPT_PATH.exists():
    raise FileNotFoundError(f"Missing eligibility prompt file: {ELIGIBILITY_PROMPT_PATH}")

_ELIGIBILITY_TEMPLATE = _load_text(ELIGIBILITY_PROMPT_PATH)


def get_default_engine() -> AzureInferenceEngine:
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "OPENAI_ENDPOINT env var is not set. "
            "Set it, or pass an AzureInferenceEngine instance into judge_eligibility()."
        )

    return AzureInferenceEngine(
        endpoint=endpoint,
        model_name="gpt-4.1",
        default_temperature=0.0,
        default_max_tokens=None,
    )


def judge_eligibility(
    clinical_trial_description: str,
    patient_note: str,
    relevance_result: str,
    engine: Optional[AzureInferenceEngine] = None,
    trace: Optional[PromptTrace] = None,
    patient_id: Optional[str] = None,
    trial_id: Optional[str] = None,
) -> str:
    """
    Run the eligibility prompt using:
      - patient note
      - trial description
      - subcohorts_of_interest (derived from relevance output)
    """
    if engine is None:
        engine = get_default_engine()

    prompt = (
        _ELIGIBILITY_TEMPLATE
        .replace("#PATIENT_NOTE#", patient_note)
        .replace("#TRIAL_DESCRIPTION#", clinical_trial_description)
        .replace("#SUBCOHORTS_OF_INTEREST#", relevance_result.strip())
    )

    outputs: Sequence[str] = engine(prompt)
    out0 = outputs[0]

    if trace is not None and patient_id is not None and trial_id is not None:
        trace.log(
            stage="eligibility",
            patient_id=patient_id,
            trial_id=trial_id,
            prompt=prompt,
            output=out0,
        )

    return out0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run eligibility judging for subcohorts of interest (from relevance output)."
    )
    parser.add_argument("--trial-file", type=str, default="test_files/test_trial.txt")
    parser.add_argument("--note-file", type=str, default="test_files/test_patient.txt")
    parser.add_argument("--relevance-file", type=str, default="test_files/test_relevance.txt")
    args = parser.parse_args()

    trial_text = Path(args.trial_file).read_text(encoding="utf-8")
    note_text = Path(args.note_file).read_text(encoding="utf-8")
    relevance_text = Path(args.relevance_file).read_text(encoding="utf-8")

    print(judge_eligibility(trial_text, note_text, relevance_text))