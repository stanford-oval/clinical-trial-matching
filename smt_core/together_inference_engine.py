# together_inference_engine.py
"""Thin Together.ai chat-completions wrapper that mimics AzureInferenceEngine's
public surface (callable + .run + .chat_complete) so that downstream code
(`run_llm_eligibility_judge`, `edit_core.llm_decide_with_rationale`,
`d_paraphrase_reliability`) can use it as a drop-in.

Together exposes an OpenAI-compatible REST API at https://api.together.xyz/v1.
We use the official `openai` SDK with `base_url` overridden.
"""
from __future__ import annotations

import logging
import os
import random
import time
from threading import Lock
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "openai package not installed. `pip install openai`"
    ) from e


TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 598, 599}


# Module-level token counters (thread-safe) for cost-guard.
_USAGE_LOCK = Lock()
_USAGE: Dict[str, Dict[str, int]] = {}


def get_usage_snapshot() -> Dict[str, Dict[str, int]]:
    with _USAGE_LOCK:
        return {k: dict(v) for k, v in _USAGE.items()}


def reset_usage(model: Optional[str] = None) -> None:
    with _USAGE_LOCK:
        if model is None:
            _USAGE.clear()
        else:
            _USAGE.pop(model, None)


def _bump_usage(model: str, p: int, c: int) -> None:
    with _USAGE_LOCK:
        slot = _USAGE.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})
        slot["prompt_tokens"] += p
        slot["completion_tokens"] += c
        slot["total_tokens"] += (p + c)
        slot["calls"] += 1


class TogetherInferenceEngine:
    """OpenAI-compat client pointed at https://api.together.xyz/v1.

    Methods exposed (match what `_call_engine_text` in `llm_judge.py` probes):
        engine(prompt, temperature=...)          -> [content_str]
        engine.run(messages, temperature=...)    -> content_str
        engine.chat_complete(prompt_or_messages) -> content_str
    """

    def __init__(
        self,
        model_name: str,
        api_key_env_var: str = "TOGETHER_API_KEY",
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
        default_top_p: float = 1.0,
        default_seed: int = 42,
        request_timeout: float = 120.0,
        max_retries: int = 5,
        base_backoff: float = 1.5,
        backoff_cap: float = 30.0,
        verbose: bool = False,
    ):
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(f"Env var '{api_key_env_var}' not set or empty.")
        self._client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            timeout=request_timeout,
            max_retries=0,  # we handle retries ourselves
        )
        self.model_name = model_name
        self.kwargs = {
            "temperature": default_temperature,
            "max_tokens": default_max_tokens,
            "top_p": default_top_p,
            "seed": default_seed,
        }
        self._max_retries = max(1, int(max_retries))
        self._base_backoff = float(base_backoff)
        self._backoff_cap = float(backoff_cap)
        self._log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self._log.setLevel(logging.INFO)

    # --------- helpers ---------
    def _backoff(self, attempt: int) -> float:
        base = min(self._backoff_cap, self._base_backoff * (2 ** (attempt - 1)))
        return random.uniform(0, base) + random.uniform(0, 1.0)

    @staticmethod
    def _normalize_messages(messages_or_prompt: Any) -> List[Dict[str, str]]:
        if isinstance(messages_or_prompt, str):
            return [{"role": "user", "content": messages_or_prompt}]
        out: List[Dict[str, str]] = []
        for m in messages_or_prompt:
            if isinstance(m, dict):
                out.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})
            else:
                # Azure SDK SystemMessage/UserMessage objects
                role = getattr(m, "role", None) or m.__class__.__name__.lower().replace("message", "") or "user"
                content = getattr(m, "content", None) or ""
                if role not in ("system", "user", "assistant"):
                    role = "user"
                out.append({"role": role, "content": str(content)})
        return out

    # --------- core ---------
    def run(
        self,
        messages: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        if temperature is None: temperature = self.kwargs["temperature"]
        if max_tokens is None:  max_tokens  = self.kwargs["max_tokens"]
        if top_p is None:       top_p       = self.kwargs["top_p"]
        if seed is None:        seed        = self.kwargs["seed"]
        # strip profile hints used by Azure path
        for k in ("_profile_stage", "_profile_trial_id", "_profile_side", "_profile_cohort_id"):
            kwargs.pop(k, None)

        msgs = self._normalize_messages(messages)
        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    # Together accepts seed but it's best-effort.
                    seed=seed,
                )
                u = getattr(resp, "usage", None)
                if u is not None:
                    _bump_usage(self.model_name, getattr(u, "prompt_tokens", 0) or 0,
                                getattr(u, "completion_tokens", 0) or 0)
                if not resp.choices:
                    raise RuntimeError("Empty Together response (no choices)")
                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError("Together response missing content")
                return content
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_err = e
                if attempt < self._max_retries:
                    delay = self._backoff(attempt)
                    self._log.info("Together transient on attempt %d (%s); retrying in %.2fs",
                                   attempt, type(e).__name__, delay)
                    time.sleep(delay)
                    continue
                raise
            except APIError as e:
                last_err = e
                status = getattr(e, "status_code", None)
                if status in TRANSIENT_STATUS and attempt < self._max_retries:
                    delay = self._backoff(attempt)
                    time.sleep(delay)
                    continue
                raise
            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    delay = self._backoff(attempt)
                    time.sleep(delay)
                    continue
                raise
        if last_err: raise last_err
        raise RuntimeError("TogetherInferenceEngine.run unknown failure")

    # --------- public API ---------
    def __call__(self, prompt: Any, **kwargs) -> List[str]:
        if isinstance(prompt, str):
            content = self.run([{"role": "user", "content": prompt}], **kwargs)
        else:
            content = self.run(prompt, **kwargs)
        return [content]

    def chat_complete(self, prompt: Any, **kwargs) -> str:
        if isinstance(prompt, str):
            return self.run([{"role": "user", "content": prompt}], **kwargs)
        return self.run(prompt, **kwargs)
