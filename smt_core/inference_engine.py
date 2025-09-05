# inference_engine.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import random
import time
import uuid
from datetime import datetime
from typing import Any, List, Optional, Dict

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    AzureError,
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)

# Some network-layer errors surface via requests underneath azure-core
try:
    from requests.exceptions import (
        ReadTimeout as RequestsReadTimeout,
        ConnectionError as RequestsConnectionError,
        ChunkedEncodingError,
    )
except Exception:  # pragma: no cover
    RequestsReadTimeout = tuple()  # type: ignore
    RequestsConnectionError = tuple()  # type: ignore
    ChunkedEncodingError = tuple()  # type: ignore


# HTTP statuses that are safe/likely to retry
TRANSIENT_STATUS = {
    408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 598, 599
}


class AzureInferenceEngine:
    """
    Robust Azure AI Inference chat completions wrapper with:
      - Exponential full-jitter backoff + Retry-After honoring
      - Timeout ramp-up per attempt
      - Optional fallback endpoint
      - Defensive response validation
      - OPTIONAL: profiling of call durations (if profiler is provided)

    Usage (unchanged):
        engine = AzureInferenceEngine(endpoint=os.environ["AZURE_OPENAI_ENDPOINT"])
        out = engine("prompt text")[0]
    """

    def __init__(
        self,
        endpoint: str,
        api_key_env_var: str = "AZURE_OPENAI_API_KEY",
        model_name: str = "gpt-4o",
        default_temperature: float = 0.5,
        default_max_tokens: int = 8192 * 2,
        default_top_p: float = 0.001,
        default_seed: int = 42,
        # retry / timeout controls
        per_attempt_read_timeout: float = 55.0,
        per_attempt_connect_timeout: float = 10.0,
        read_timeout_cap: float = 120.0,       # max read timeout as we ramp up
        max_retries: int = 5,                  # bump default
        base_backoff: float = 1.0,             # seconds
        backoff_cap: float = 30.0,             # maximum backoff
        backoff_jitter: float = 1.0,           # upper bound for full-jitter
        # optional second endpoint to fail over on late attempts
        fallback_endpoint: Optional[str] = None,
        verbose: bool = False,
        # profiling
        profiler: Optional[Any] = None,
        profiler_run_id: Optional[str] = None,
        # reproducibility logging (Phase 3 hook). When set, every successful
        # LLM call is persisted to <log_to_dir>/llm_calls/<ts>_<hash>.json
        # with the full prompt, full response, model/temperature/seed, and a
        # call_id for correlating across pipeline stages. Backwards-compatible:
        # default None means no logging is performed.
        log_to_dir: Optional[Any] = None,
    ):
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(f"Env var '{api_key_env_var}' not set or empty.")

        self._primary_client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        self._fallback_client = (
            ChatCompletionsClient(
                endpoint=fallback_endpoint,
                credential=AzureKeyCredential(api_key),
            )
            if fallback_endpoint
            else None
        )

        self.model_name = model_name
        self.kwargs = {
            "temperature": default_temperature,
            "max_tokens": default_max_tokens,
            "top_p": default_top_p,
            "seed": default_seed,
        }

        self._read_timeout0 = per_attempt_read_timeout
        self._conn_timeout0 = per_attempt_connect_timeout
        self._read_timeout_cap = read_timeout_cap

        self._max_retries = max(1, int(max_retries))
        self._base_backoff = max(0.0, float(base_backoff))
        self._backoff_cap = max(backoff_cap, self._base_backoff)
        self._backoff_jitter = max(0.0, float(backoff_jitter))

        self._log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self._log.setLevel(logging.INFO)

        # profiling
        self._profiler = profiler
        self._profiler_run_id = profiler_run_id or "run"
        # Defaults used if caller doesn't pass profile hints via kwargs
        self._default_profile_trial_id = "?"
        self._default_profile_side = "?"

        # reproducibility logging
        self._log_to_dir: Optional[pathlib.Path] = (
            pathlib.Path(log_to_dir) if log_to_dir is not None else None
        )
        if self._log_to_dir is not None:
            (self._log_to_dir / "llm_calls").mkdir(parents=True, exist_ok=True)

    # --------------------------- helpers --------------------------- #
    def _client_for_attempt(self, attempt: int):
        """
        Use primary client for early attempts; switch to fallback (if provided)
        on the last attempt to dodge sticky DC/router issues.
        """
        if self._fallback_client and attempt == self._max_retries:
            return self._fallback_client
        return self._primary_client

    def _ramp_timeouts(self, attempt: int) -> tuple[float, float]:
        """
        Increase read timeout each attempt up to a cap; keep connect timeout fixed.
        """
        read = min(self._read_timeout0 * (1.5 ** (attempt - 1)), self._read_timeout_cap)
        return float(self._conn_timeout0), float(read)

    def _full_jitter_backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        """
        Respect Retry-After when present; else exponential backoff with full jitter.
        """
        if retry_after is not None:
            return max(0.0, float(retry_after))
        base = min(self._backoff_cap, self._base_backoff * (2 ** (attempt - 1)))
        # Full jitter: random in [0, base] + tiny extra jitter
        return random.uniform(0, base) + random.uniform(0, self._backoff_jitter)

    @staticmethod
    def _parse_retry_after(exc: HttpResponseError) -> Optional[float]:
        try:
            headers = getattr(exc.response, "headers", None) or {}
            for key in ("retry-after", "Retry-After"):
                if key in headers:
                    return float(headers[key])
        except Exception:
            pass
        return None

    @staticmethod
    def _should_retry_http(exc: HttpResponseError) -> bool:
        try:
            status = getattr(exc, "status_code", None) or getattr(exc.response, "status_code", None)
            if status is None:
                return False
            if status in TRANSIENT_STATUS:
                return True
            # Retry all 5xx by default
            return 500 <= int(status) <= 599
        except Exception:
            return False

    @staticmethod
    def _is_networkish_error(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                ServiceRequestError,
                ServiceResponseError,
                RequestsReadTimeout,
                RequestsConnectionError,
                ChunkedEncodingError,
                TimeoutError,
            ),
        )

    @staticmethod
    def _validate_and_extract_content(response: Any) -> str:
        """
        Defensive extraction; raises RuntimeError if shape unexpected.
        """
        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Empty/invalid response: no choices present.")
        choice0 = response.choices[0]
        msg = getattr(choice0, "message", None)
        if not msg:
            raise RuntimeError("Invalid response: missing 'message'.")
        content = getattr(msg, "content", None)
        if content is None:
            # Some SDKs put text directly on choice/message; try best-effort fallbacks.
            for attr in ("content", "text", "delta"):
                v = getattr(choice0, attr, None) or getattr(msg, attr, None)
                if isinstance(v, str) and v.strip():
                    return v
            raise RuntimeError("Invalid response: missing 'content'.")
        return str(content)

    # --------------------------- core run --------------------------- #
    def run(
        self,
        messages: List[SystemMessage | UserMessage | AssistantMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Returns the assistant message content string.
        Retries on transient network/HTTP errors with timeout ramp-up and backoff.
        """
        # Resolve defaults
        if temperature is None:
            temperature = self.kwargs["temperature"]
        if max_tokens is None:
            max_tokens = self.kwargs["max_tokens"]
        if top_p is None:
            top_p = self.kwargs.get("top_p", 1.0)
        if seed is None:
            seed = self.kwargs.get("seed", 42)

        # Pull optional profiling hints (won't leak to client.complete)
        profile_stage = kwargs.pop("_profile_stage", "LLM")
        profile_trial_id = kwargs.pop("_profile_trial_id", self._default_profile_trial_id)
        profile_side = kwargs.pop("_profile_side", self._default_profile_side)
        profile_cohort_id = kwargs.pop("_profile_cohort_id", None)

        # Ensure we don't pass duplicates to SDK
        kwargs.pop("top_p", None)
        kwargs.pop("temperature", None)
        kwargs.pop("max_tokens", None)
        kwargs.pop("seed", None)

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            client = self._client_for_attempt(attempt)
            conn_timeout, read_timeout = self._ramp_timeouts(attempt)

            try:
                t0 = time.perf_counter()
                response = client.complete(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    connection_timeout=conn_timeout,
                    read_timeout=read_timeout,
                    **kwargs,
                )
                t1 = time.perf_counter()

                # Collect token usage if available
                usage: Dict[str, Any] = {}
                try:
                    u = getattr(response, "usage", None)
                    if u:
                        usage = {
                            "input_tokens": getattr(u, "prompt_tokens", None),
                            "output_tokens": getattr(u, "completion_tokens", None),
                            "total_tokens": getattr(u, "total_tokens", None),
                        }
                except Exception:
                    pass

                # Emit profiling for the LLM call
                if self._profiler:
                    self._profiler.log_llm_call(
                        run_id=self._profiler_run_id,
                        trial_id=profile_trial_id,
                        side=profile_side,
                        stage=profile_stage,
                        cohort_id=profile_cohort_id,
                        model=self.model_name,
                        duration_s=(t1 - t0),
                        **({"usage": usage} if usage else {}),
                    )

                content = self._validate_and_extract_content(response)

                # Reproducibility hook: persist the full call if enabled.
                if self._log_to_dir is not None:
                    try:
                        self._persist_call(
                            messages=messages,
                            content=content,
                            model=self.model_name,
                            temperature=temperature,
                            seed=seed,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            duration_s=(t1 - t0),
                            usage=usage,
                            stage=profile_stage,
                            trial_id=profile_trial_id,
                            side=profile_side,
                            cohort_id=profile_cohort_id,
                        )
                    except Exception as _e:  # never break inference on logging failure
                        self._log.warning("log_to_dir persist failed: %s", _e)

                return content

            except HttpResponseError as e:
                last_error = e
                if self._should_retry_http(e) and attempt < self._max_retries:
                    ra = self._parse_retry_after(e)
                    delay = self._full_jitter_backoff(attempt, retry_after=ra)
                    self._log.info(
                        "HTTP %s on attempt %d; retrying in %.2fs",
                        getattr(e, "status_code", "?"),
                        attempt,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise

            except AzureError as e:
                # Other Azure SDK errors can be transient (transport/connection)
                last_error = e
                if self._is_networkish_error(e) and attempt < self._max_retries:
                    delay = self._full_jitter_backoff(attempt, retry_after=None)
                    self._log.info("AzureError (network-ish) on attempt %d; retrying in %.2fs", attempt, delay)
                    time.sleep(delay)
                    continue
                raise

            except Exception as e:
                # requests.* timeouts/connection issues or unknowns
                last_error = e
                if self._is_networkish_error(e) and attempt < self._max_retries:
                    delay = self._full_jitter_backoff(attempt, retry_after=None)
                    self._log.info("Network error on attempt %d; retrying in %.2fs", attempt, delay)
                    time.sleep(delay)
                    continue
                raise

        # Shouldn't reach here; surface the last seen error
        if last_error:
            raise last_error
        raise RuntimeError("Unknown failure in AzureInferenceEngine.run")

    # ----------------- reproducibility logging helper ----------------- #
    @staticmethod
    def _serialize_messages(messages: List[Any]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages:
            role = (
                "system" if isinstance(m, SystemMessage)
                else "assistant" if isinstance(m, AssistantMessage)
                else "user"
            )
            out.append({"role": role, "content": str(getattr(m, "content", ""))})
        return out

    def _persist_call(
        self,
        *,
        messages: List[Any],
        content: str,
        model: str,
        temperature: float,
        seed: int,
        top_p: float,
        max_tokens: int,
        duration_s: float,
        usage: Dict[str, Any],
        stage: str,
        trial_id: str,
        side: str,
        cohort_id: Optional[str],
    ) -> None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        msgs_serialized = self._serialize_messages(messages)
        prompt_text = "\n\n".join(m["content"] for m in msgs_serialized)
        h = hashlib.sha256(prompt_text.encode("utf-8", errors="replace")).hexdigest()[:16]
        call_id = f"{ts}_{h}_{uuid.uuid4().hex[:6]}"
        record = {
            "schema_version": "reproducibility-v1",
            "call_id": call_id,
            "timestamp_utc": ts,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "duration_s": duration_s,
            "usage": usage,
            "stage": stage,
            "trial_id": trial_id,
            "side": side,
            "cohort_id": cohort_id,
            "messages": msgs_serialized,
            "response": content,
            "prompt_sha256_16": h,
        }
        path = self._log_to_dir / "llm_calls" / f"{call_id}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    # --------------------------- public API --------------------------- #
    def __call__(self, prompt: str, **kwargs) -> List[str]:
        """
        Simple string-in, string-out (list) interface to match existing callers.
        If you want richer profiling labels, pass:
          _profile_stage="CANON", _profile_trial_id="NCT123", _profile_side="inclusion", _profile_cohort_id="NCT123_C1"
        """
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=prompt),
        ]
        content = self.run(messages, **kwargs)
        return [content]