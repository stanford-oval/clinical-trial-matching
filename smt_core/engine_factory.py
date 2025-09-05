"""
Centralized engine selection based on OPENAI_ENDPOINT / OPENAI_MODEL env vars.
Replaces the duplicated if/elif blocks in compile_trial_to_smt.py,
compile_patient_to_variables.py, match_patient_to_trial.py, etc.
"""
from __future__ import annotations

import os
from typing import Any, Optional


def _pick_engine_and_model(hint: str) -> tuple[str, str]:
    """
    Decide (ENGINE_VERSION, MODEL_NAME) from a lowercase hint string.
    ENGINE_VERSION selects which engine wrapper module to import.
    MODEL_NAME is passed to the Azure engine.
    """
    if not hint:
        raise EnvironmentError(
            "Cannot determine model: OPENAI_MODEL and OPENAI_ENDPOINT are both empty/unset."
        )
    if "gpt-5" in hint:
        return ("gpt-5", "gpt-5")
    if "gpt-4.1" in hint:
        return ("gpt-4.1", "gpt-4.1")
    if "gpt-4o" in hint:
        return ("gpt-4o", "gpt-4o")
    if "o3" in hint:
        return ("o3", "o3")
    raise EnvironmentError(
        f"Unrecognized model hint: {hint!r}. "
        "Expected one of: gpt-4o, gpt-4.1, gpt-5, o3."
    )


def detect_engine_and_model() -> tuple[str, str]:
    """
    Auto-detect (engine_version, model_name) from environment variables.
    Priority: OPENAI_MODEL > OPENAI_ENDPOINT suffix.
    """
    model_hint = os.environ.get("OPENAI_MODEL", "").strip().lower()
    endpoint_hint = os.environ.get("OPENAI_ENDPOINT", "").strip().lower()
    hint = model_hint or endpoint_hint
    return _pick_engine_and_model(hint)


def create_engine(
    endpoint: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
):
    """
    Factory: create the appropriate AzureInferenceEngine based on env vars.

    Args:
        endpoint: Override OPENAI_ENDPOINT env var.
        model_name: Override detected model name.
        **kwargs: Passed through to the engine constructor.

    Returns:
        An AzureInferenceEngine instance.
    """
    engine_version, detected_model = detect_engine_and_model()
    endpoint = endpoint or os.environ.get("OPENAI_ENDPOINT", "")
    model_name = model_name or detected_model

    if engine_version == "gpt-5":
        from smt_core.inference_engine_5 import AzureInferenceEngine
    else:
        from smt_core.inference_engine import AzureInferenceEngine

    return AzureInferenceEngine(
        endpoint=endpoint,
        model_name=model_name,
        **kwargs,
    )
