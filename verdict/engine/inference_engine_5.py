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
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 598, 599}

# --------------------------------------------------------------------
# Model family rules
# --------------------------------------------------------------------
# NOTE: Many Azure deployments for "chat" models still disallow sampling params (temperature/top_p).
# We'll be conservative: we *can* try sending them, but we will auto-strip if unsupported.

# Chat-family models where sampling controls *may* be supported.
SAMPLING_ALLOWED_PREFIXES = (
    "gpt-4o",         # includes subvariants like gpt-4o-mini
    "gpt-4.1",
    "gpt-4",
    "gpt-35-turbo",
)

# Families that use max_completion_tokens (and typically forbid sampling knobs)
REQUIRES_MAX_COMPLETION_PREFIXES = ("gpt-5", "o1", "o2", "o3", "o4")


class AzureInferenceEngine:
    """
    Azure AI Inference chat wrapper with:
      - Exponential full-jitter backoff + Retry-After honoring
      - Timeout ramp-up per attempt
      - Optional fallback endpoint
      - Defensive response validation
      - Model-specific params:
          • classic chat: use `max_tokens` (if provided)
          • reasoning families: send `max_completion_tokens` via `model_extras`
      - IMPORTANT robustness:
          • Some Azure deployments reject temperature != default(1) and/or reject top_p entirely.
            We (a) omit temperature unless it is exactly 1.0, and
            (b) omit top_p by default unless caller explicitly passes it, AND
            (c) if the service still errors with unsupported_parameter, we strip and retry once.
    """

    def __init__(
        self,
        endpoint: str,
        api_key_env_var: str = "OPENAI_API_KEY",
        model_name: str = "gpt-5",
        # Keep your original defaults for classic chat models:
        default_temperature: float = 1.0,          # IMPORTANT: safest default for constrained deployments
        default_max_tokens: Optional[int] = None,  # mapped at call time
        default_top_p: Optional[float] = None,     # IMPORTANT: many deployments reject top_p; default None -> omit
        # retry / timeout controls
        per_attempt_read_timeout: float = 55.0,
        per_attempt_connect_timeout: float = 10.0,
        read_timeout_cap: float = 120.0,
        max_retries: int = 5,
        base_backoff: float = 1.0,
        backoff_cap: float = 30.0,
        backoff_jitter: float = 1.0,
        # optional second endpoint
        fallback_endpoint: Optional[str] = None,
        verbose: bool = False,
    ):
        # Accept a few common env var names
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

        # Store defaults; we will only send supported ones.
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

    # --------------------------- model helpers --------------------------- #
    def _supports_sampling_family(self) -> bool:
        n = (self.model_name or "").lower()
        return any(n.startswith(pfx) for pfx in SAMPLING_ALLOWED_PREFIXES)

    def _uses_max_completion_tokens(self) -> bool:
        n = (self.model_name or "").lower()
        return any(n.startswith(pfx) for pfx in REQUIRES_MAX_COMPLETION_PREFIXES)

    # --------------------------- retry/backoff helpers --------------------------- #
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

    @staticmethod
    def _is_unsupported_param_error(e: HttpResponseError) -> bool:
        # Azure uses codes like (unsupported_parameter) or (unsupported_value)
        code = getattr(e, "error", None)
        # fallback: parse message string
        msg = str(e)
        return ("unsupported_parameter" in msg) or ("unsupported_value" in msg)

    @staticmethod
    def _unsupported_param_name(e: HttpResponseError) -> Optional[str]:
        # Try to extract "Unsupported parameter: 'top_p'"
        msg = str(e)
        m = None
        try:
            import re
            m = re.search(r"Unsupported parameter:\s*'([^']+)'", msg)
            if m:
                return m.group(1)
            m = re.search(r"Unsupported value:\s*'([^']+)'", msg)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    # --------------------------- core run --------------------------- #
    def run(
        self,
        messages: List[SystemMessage | UserMessage | AssistantMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Robust request builder:
          - If reasoning family: uses model_extras["max_completion_tokens"], and omits sampling knobs.
          - Else: uses max_tokens (if provided).
          - Sampling knobs:
              • temperature is ONLY sent if exactly 1.0 (many deployments reject any other value)
              • top_p is omitted by default unless explicitly provided and accepted
          - If Azure returns unsupported_parameter/value, strip offending param and retry once immediately.
        """
        supports_sampling_family = self._supports_sampling_family()
        uses_max_completion = self._uses_max_completion_tokens()

        # Resolve defaults
        if temperature is None:
            temperature = self.kwargs["temperature"] if supports_sampling_family else None
        if top_p is None:
            top_p = self.kwargs["top_p"] if supports_sampling_family else None
        if max_tokens is None:
            max_tokens = self.kwargs["max_tokens"]

        # Build base request
        req: Dict[str, Any] = dict(
            messages=messages,
            model=self.model_name,
        )

        # NOTE: seed can also be unsupported; we include it, but will strip if rejected.
        req["seed"] = 42

        # Merge model_extras
        user_model_extras = kwargs.pop("model_extras", None)
        if user_model_extras is None:
            model_extras: Dict[str, Any] = {}
        elif isinstance(user_model_extras, dict):
            model_extras = dict(user_model_extras)
        else:
            raise TypeError("model_extras must be a dict if provided.")

        # Map token controls per family
        if uses_max_completion:
            if "max_completion_tokens" not in model_extras:
                # honor env override; default bumped from 16384 -> 32768
                env_mct = os.environ.get('CMSRC_MAX_COMPLETION_TOKENS')
                model_extras["max_completion_tokens"] = int(env_mct) if env_mct else int(max_tokens if max_tokens is not None else 32768)
            # honor env-driven reasoning_effort for gpt-5/o-series
            env_re = os.environ.get('CMSRC_REASONING_EFFORT')
            if env_re and 'reasoning_effort' not in model_extras:
                model_extras['reasoning_effort'] = env_re
        else:
            if max_tokens is not None:
                req["max_tokens"] = int(max_tokens)

        # Sampling knobs ONLY if (family suggests) and we choose to send them
        # IMPORTANT: Many Azure deployments reject top_p entirely and reject temperature != 1.
        if supports_sampling_family and not uses_max_completion:
            # temperature: only send if exactly 1.0
            if temperature is not None:
                t = float(temperature)
                if abs(t - 1.0) < 1e-9:
                    req["temperature"] = t
                else:
                    # omit temperature to use service default
                    pass

            # top_p: only send if not None; many deployments reject, will auto-strip on error
            if top_p is not None:
                req["top_p"] = float(top_p)
        else:
            # ensure absent
            req.pop("temperature", None)
            req.pop("top_p", None)

        # Strip keys that must NOT leak into **kwargs**
        for k in ("top_p", "temperature", "max_tokens", "max_completion_tokens"):
            kwargs.pop(k, None)

        # Attach model_extras
        if model_extras:
            req["model_extras"] = model_extras

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            client = self._client_for_attempt(attempt)
            conn_timeout, read_timeout = self._ramp_timeouts(attempt)

            # We allow ONE immediate "strip unsupported param and retry" within the same attempt.
            stripped_once = False

            while True:
                try:
                    response = client.complete(
                        connection_timeout=conn_timeout,
                        read_timeout=read_timeout,
                        **req,
                        **kwargs,
                    )
                    return self._validate_and_extract_content(response)

                except HttpResponseError as e:
                    # If Azure rejects a param (top_p/temperature/seed/etc.), strip and retry once immediately
                    if (not stripped_once) and self._is_unsupported_param_error(e):
                        pname = self._unsupported_param_name(e)

                        # If we can identify the param, drop it; else drop known common offenders defensively.
                        if pname:
                            req.pop(pname, None)
                            # Also handle nested: sometimes max_completion_tokens is in model_extras
                            if pname == "max_completion_tokens":
                                if "model_extras" in req and isinstance(req["model_extras"], dict):
                                    req["model_extras"].pop("max_completion_tokens", None)
                        else:
                            # Defensive: drop the usual suspects
                            req.pop("top_p", None)
                            req.pop("temperature", None)
                            req.pop("seed", None)

                        stripped_once = True
                        continue  # retry immediately with stripped params

                    # Normal transient retry logic
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
                        break  # go to next attempt

                    last_error = e
                    raise

                except AzureError as e:
                    last_error = e
                    if self._is_networkish_error(e) and attempt < self._max_retries:
                        delay = self._full_jitter_backoff(attempt, retry_after=None)
                        self._log.info("AzureError (network-ish) on attempt %d; retrying in %.2fs", attempt, delay)
                        time.sleep(delay)
                        break
                    raise

                except Exception as e:
                    last_error = e
                    if self._is_networkish_error(e) and attempt < self._max_retries:
                        delay = self._full_jitter_backoff(attempt, retry_after=None)
                        self._log.info("Network error on attempt %d; retrying in %.2fs", attempt, delay)
                        time.sleep(delay)
                        break
                    raise

        if last_error:
            raise last_error
        raise RuntimeError("Unknown failure in AzureInferenceEngine.run")

    # --------------------------- public API --------------------------- #
    def __call__(self, prompt: str, system_message: str = "You are a helpful assistant.", **kwargs) -> List[str]:
        messages = [
            SystemMessage(content=system_message),
            UserMessage(content=prompt),
        ]
        content = self.run(messages, **kwargs)
        return [content]