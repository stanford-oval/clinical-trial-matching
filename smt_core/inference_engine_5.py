# inference_engine_5.py
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, List, Optional, Tuple, Dict

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    AzureError,
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)

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


TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 598, 599}

SAMPLING_ALLOWED_PREFIXES = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-4",
    "gpt-35-turbo",
)

REQUIRES_MAX_COMPLETION_PREFIXES = ("gpt-5", "o1", "o2", "o3", "o4")


class AzureInferenceEngine:
    """
    Azure AI Inference chat wrapper (gpt-5 default) with robust retries and optional profiling.
    See inference_engine.py for detailed docstring — implementation matches 1:1.
    """

    def __init__(
        self,
        endpoint: str,
        api_key_env_var: str = "AZURE_OPENAI_API_KEY",
        model_name: str = "gpt-5",
        default_temperature: float = 0.0,
        default_max_tokens: Optional[int] = None,
        default_top_p: float = 0.001,
        per_attempt_read_timeout: float = 55.0,
        per_attempt_connect_timeout: float = 10.0,
        read_timeout_cap: float = 120.0,
        max_retries: int = 5,
        base_backoff: float = 1.0,
        backoff_cap: float = 30.0,
        backoff_jitter: float = 1.0,
        fallback_endpoint: Optional[str] = None,
        verbose: bool = False,
        profiler: Any | None = None,
        profiler_run_id: Optional[str] = None,
        **_: Any,
    ):
        api_key = (
            os.environ.get(api_key_env_var)
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("AZUREAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                f"Azure API key not found. Tried '{api_key_env_var}', 'AZURE_OPENAI_API_KEY', 'AZUREAI_API_KEY'."
            )

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
        self._profiler = profiler
        self._profiler_run_id = profiler_run_id

        self.kwargs = {
            "temperature": default_temperature,
            "max_tokens": default_max_tokens,
            "top_p": default_top_p,
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

    def _supports_sampling(self) -> bool:
        n = (self.model_name or "").lower()
        return any(n.startswith(pfx) for pfx in SAMPLING_ALLOWED_PREFIXES)

    def _uses_max_completion_tokens(self) -> bool:
        n = (self.model_name or "").lower()
        return any(n.startswith(pfx) for pfx in REQUIRES_MAX_COMPLETION_PREFIXES)

    def _client_for_attempt(self, attempt: int):
        if self._fallback_client and attempt == self._max_retries:
            return self._fallback_client
        return self._primary_client

    def _ramp_timeouts(self, attempt: int) -> Tuple[float, float]:
        read = min(self._read_timeout0 * (1.5 ** (attempt - 1)), self._read_timeout_cap)
        return float(self._conn_timeout0), float(read)

    def _full_jitter_backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return max(0.0, float(retry_after))
        base = min(self._backoff_cap, self._base_backoff * (2 ** (attempt - 1)))
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
        except Exception:
            return False
        if status is None:
            return False
        if status in TRANSIENT_STATUS:
            return True
        try:
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
        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Empty/invalid response: no choices present.")
        choice0 = response.choices[0]
        msg = getattr(choice0, "message", None)
        if not msg:
            raise RuntimeError("Invalid response: missing 'message'.")
        content = getattr(msg, "content", None)
        if content is None:
            for attr in ("content", "text", "delta"):
                v = getattr(choice0, attr, None) or getattr(msg, attr, None)
                if isinstance(v, str) and v.strip():
                    return v
            raise RuntimeError("Invalid response: missing 'content'.")
        return str(content)

    def run(
        self,
        messages: List[SystemMessage | UserMessage | AssistantMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        supports_sampling = self._supports_sampling()
        uses_max_completion = self._uses_max_completion_tokens()

        if temperature is None:
            temperature = self.kwargs["temperature"] if supports_sampling else None
        if top_p is None:
            top_p = self.kwargs["top_p"] if supports_sampling else None
        if max_tokens is None:
            max_tokens = self.kwargs["max_tokens"]

        prof_trial_id = kwargs.pop("__trial_id", "?")
        prof_side = kwargs.pop("__side", "?")
        prof_stage = kwargs.pop("__stage", "engine")
        prof_cohort_id = kwargs.pop("__cohort_id", None)

        req: Dict[str, Any] = dict(
            messages=messages,
            model=self.model_name,
            seed=42,
        )

        user_model_extras = kwargs.pop("model_extras", None)
        if user_model_extras is None:
            model_extras: Dict[str, Any] = {}
        elif isinstance(user_model_extras, dict):
            model_extras = dict(user_model_extras)
        else:
            raise TypeError("model_extras must be a dict if provided.")

        if uses_max_completion:
            if "max_completion_tokens" not in model_extras:
                model_extras["max_completion_tokens"] = int(max_tokens if max_tokens is not None else 16384)
        else:
            if max_tokens is not None:
                req["max_tokens"] = int(max_tokens)

        if supports_sampling:
            if temperature is not None:
                req["temperature"] = float(temperature)
            if top_p is not None:
                req["top_p"] = float(top_p)
        else:
            req.pop("temperature", None)
            req.pop("top_p", None)

        for k in ("top_p", "temperature", "max_tokens", "max_completion_tokens"):
            kwargs.pop(k, None)

        if model_extras:
            req["model_extras"] = model_extras

        last_error: Optional[Exception] = None
        t_start = time.perf_counter()

        for attempt in range(1, self._max_retries + 1):
            client = self._client_for_attempt(attempt)
            conn_timeout, read_timeout = self._ramp_timeouts(attempt)
            try:
                response = client.complete(
                    connection_timeout=conn_timeout,
                    read_timeout=read_timeout,
                    **req,
                    **kwargs,
                )
                content = self._validate_and_extract_content(response)

                if self._profiler:
                    duration_s = time.perf_counter() - t_start
                    try:
                        self._profiler.log_llm_call(
                            run_id=self._profiler_run_id or "run",
                            trial_id=prof_trial_id,
                            side=prof_side,
                            stage=prof_stage,
                            cohort_id=prof_cohort_id,
                            model=self.model_name,
                            duration_s=duration_s,
                        )
                    except Exception:
                        pass

                return content

            except HttpResponseError as e:
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
                last_error = e
                raise

            except AzureError as e:
                last_error = e
                if self._is_networkish_error(e) and attempt < self._max_retries:
                    delay = self._full_jitter_backoff(attempt, retry_after=None)
                    self._log.info("AzureError (network-ish) on attempt %d; retrying in %.2fs", attempt, delay)
                    time.sleep(delay)
                    continue
                raise

            except Exception as e:
                last_error = e
                if self._is_networkish_error(e) and attempt < self._max_retries:
                    delay = self._full_jitter_backoff(attempt, retry_after=None)
                    self._log.info("Network error on attempt %d; retrying in %.2fs", attempt, delay)
                    time.sleep(delay)
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("Unknown failure in AzureInferenceEngine.run")

    def __call__(self, prompt: str, system_message: str = "You are a helpful assistant.", **kwargs) -> List[str]:
        messages = [
            SystemMessage(content=system_message),
            UserMessage(content=prompt),
        ]
        content = self.run(messages, **kwargs)
        return [content]