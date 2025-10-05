# modules/DiseaseListPreprocessor/stages/DiseaseListExtractor.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import dspy  # type: ignore
from pathlib import Path
import os

_LOG = logging.getLogger(__name__)


def _dict_to_multiline_string(d: Dict[str, Any]) -> str:
    """
    Convert dictionary into readable multiline string.

    - Each top-level key on new line.
    - If value is dict → each sub-key on new line.
    - If value is list → join by comma.
    """
    lines = []

    for key, value in d.items():
        # Case 1: nested dict (e.g., metadata)
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, list):
                    sub_val_str = ", ".join(str(v) for v in sub_val)
                else:
                    sub_val_str = str(sub_val)
                lines.append(f"{sub_key}: {sub_val_str}")

        # Case 2: list
        elif isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
            lines.append(f"{key}: {value_str}")

        # Case 3: normal value
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)

class DiseaseListExtractor(dspy.Module):
    def __init__(
        self,
        engine: Any,
        *,
        verbose: bool = False,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
        log_dir: str | Path = "mbench/DiseaseListExtractor",
    ) -> None:
        super().__init__()
        self.engine = engine
        self.verbose = verbose
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Logging directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ───────────────────── public API ─────────────────────
    


    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if self.engine is None:
            _LOG.warning("[DiseaseListExtractor] engine is None; skipping LLM extraction.")
            ctx.setdefault("disease_list", [])
            return ctx

        parsed_context = ctx.get("contextual")

        if isinstance(parsed_context, dict):
            # Remove the raw "text" field, repeated
            filtered = {k: v for k, v in parsed_context.items() if k != "text"}

            contextual_text = _dict_to_multiline_string(filtered)
        else:
            contextual_text = ctx.get("contextual_text", "") or ""

        print("***", contextual_text)
        cohort_contextual_text: str = ctx.get("cohort_contextual_text", "") or ""

        has_cohort_ctx = bool(cohort_contextual_text.strip())

        # Pick prompt: cohort-specific if present, else default
        if has_cohort_ctx:
            prompt_template: Optional[str] = ctx.get("DiseaseListExtractor_cohort_prompt")  # NEW
        else:
            prompt_template = ctx.get("DiseaseListExtractor_prompt")

        prompt = self._build_prompt(
            tmpl=prompt_template,
            context=contextual_text,
            cohort_context=cohort_contextual_text if has_cohort_ctx else None,
            use_cohort_prompt=has_cohort_ctx,
        )

        if self.verbose:
            _LOG.info("[DiseaseListExtractor] Calling LLM with prompt length=%d", len(prompt))

        raw_output = self._call_llm(prompt)

        if self.verbose:
            _LOG.info("[DiseaseListExtractor] Raw LLM output: %r", raw_output[:500])

        disease_list = self._parse_disease_list(raw_output)
        ctx["processed_disease_list"] = disease_list

        trial_id = ctx.get("trial_id", "UNKNOWN")
        self._log_extraction(trial_id, prompt, disease_list)

        if self.verbose:
            _LOG.info("[DiseaseListExtractor] Parsed %d disease items.", len(disease_list))

        return ctx

    # ───────────────────── internals ─────────────────────

    def _build_prompt(
        self,
        tmpl: Optional[str],
        context: str,
        cohort_context: Optional[str] = None,
        use_cohort_prompt: bool = False,
    ) -> str:
        """
        Build the final prompt.

        Supports placeholders:
        - #CONTEXTUAL_TEXT (trial-level)
        - #COHORT_CONTEXTUAL_TEXT (cohort-specific, optional)

        If tmpl is not provided:
        - load default extractor prompt (existing behavior)
        - if use_cohort_prompt=True, try load a cohort-specific default file (optional)
        """
        default_prompt_path = (
            Path(__file__).resolve().parents[3]
            / "prompts" / "clinical_trial" / "RequirementExtractor"
            / "RequirementDiseaseListExtractor.prompt"
        )
        
        # prompt for subcohort-specific context (if applicable)
        default_cohort_prompt_path = (
            Path(__file__).resolve().parents[3]
            / "prompts" / "clinical_trial" / "RequirementExtractor"
            / "RequirementDiseaseListExtractor_Subcohort.prompt"
        )

        # If tmpl not supplied from ctx -> load from file
        if not tmpl:
            try:
                if use_cohort_prompt and default_cohort_prompt_path.exists():
                    tmpl = default_cohort_prompt_path.read_text(encoding="utf-8")
                else:
                    tmpl = default_prompt_path.read_text(encoding="utf-8")
                print("prompt_template:", tmpl)
            except Exception as e:
                _LOG.warning(
                    "[DiseaseListExtractor] Prompt file missing, using fallback prompt. Error: %s",
                    e,
                )
                # Fallback prompt includes both contexts if available
                prompt = (
                    "You are a medical expert. From the following clinical trial text, "
                    "extract all diseases, conditions, and diagnoses that define the target disease.\n"
                    "Return ONLY a JSON object: {\"disease_list\": [..]}.\n\n"
                    f"#CONTEXTUAL_TEXT\n{context}\n"
                )
                if cohort_context and cohort_context.strip():
                    prompt += f"\n#COHORT_CONTEXTUAL_TEXT\n{cohort_context}\n"
                return prompt

        prompt = tmpl

        # Replace trial placeholder if present
        if "#TRIAL_CONTEXTUAL_TEXT#" in prompt:
            prompt = prompt.replace("#TRIAL_CONTEXTUAL_TEXT#", context)
        else:
            prompt = prompt.rstrip() + f"\n\n### Trial Context\n{context}\n"

        # Replace/append cohort placeholder if cohort context is provided
        if cohort_context and cohort_context.strip():
            if "#COHORT_CONTEXTUAL_TEXT" in prompt:
                prompt = prompt.replace("#COHORT_CONTEXTUAL_TEXT#", cohort_context)
            else:
                prompt = prompt.rstrip() + f"\n\n### Cohort-specific Context\n{cohort_context}\n"

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the underlying inference engine.

        Expects either:
        - engine.complete(prompt, temperature=..., max_tokens=...) -> str
        - OR a callable engine(prompt: str) -> str
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if hasattr(self.engine, "complete"):
                    return self.engine.complete(
                        prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                elif callable(self.engine):
                    # Some engines are directly callable
                    return self.engine(prompt)
                else:
                    raise TypeError(
                        "Engine does not support `.complete` or direct call."
                    )
            except Exception as e:  # pragma: no cover - defensive
                last_err = e
                _LOG.warning(
                    "[DiseaseListExtractor] LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )

        raise RuntimeError(
            "[DiseaseListExtractor] LLM call failed after "
            f"{self.max_retries} attempts."
        ) from last_err

    # ───────────────────── parsing helpers ─────────────────────

    def _parse_disease_list(self, raw: Any) -> List[str]:
        """
        Parse raw LLM output into a List[str] of diseases.

        Expected canonical format (as JSON):

            {
            "disease_list": ["<disease1>", "<disease2>", ...]
            }

        Handles:
        - dict with "disease_list"
        - list containing that dict or its JSON string
        - bare list of disease strings
        - string containing that JSON (possibly with code fences or extra text)
        """

        # 1) None -> empty
        if raw is None:
            _LOG.warning("[DiseaseListExtractor] LLM output is None; returning empty list.")
            return []

        # 2) Dict → look for "disease_list"
        if isinstance(raw, dict):
            dl = raw.get("disease_list")
            if isinstance(dl, list):
                return self._normalize_list(dl)
            _LOG.warning(
                "[DiseaseListExtractor] Dict output missing 'disease_list' key or not a list; raw keys=%s",
                list(raw.keys()),
            )
            return []

        # 3) List → several possibilities
        if isinstance(raw, list):
            # 3a) Single-element list whose element is a string that looks like JSON
            if len(raw) == 1 and isinstance(raw[0], str):
                _LOG.info(
                    "[DiseaseListExtractor] Single-element list of string detected; "
                    "treating inner string as JSON candidate."
                )
                # Recurse on the inner string (will go through the string path below)
                return self._parse_disease_list(raw[0])

            # 3b) List containing a dict with "disease_list"
            for elem in raw:
                if isinstance(elem, dict) and isinstance(elem.get("disease_list"), list):
                    _LOG.info(
                        "[DiseaseListExtractor] Found dict with 'disease_list' inside list; using that."
                    )
                    return self._normalize_list(elem["disease_list"])

            # 3c) Otherwise: treat as bare list of disease strings
            _LOG.warning(
                "[DiseaseListExtractor] LLM returned a list without a clear 'disease_list' dict; "
                "using it directly as disease list."
            )
            return self._normalize_list(raw)

        # 4) Treat everything else as string and try JSON
        if not isinstance(raw, str):
            raw = str(raw)

        text = self._strip_code_fences(raw).strip()

        # Try to parse as JSON object or list
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            # Try to extract the JSON object substring if there is surrounding text
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                candidate = text[start:end]
                obj = json.loads(candidate)
            except Exception as e:
                _LOG.warning(
                    "[DiseaseListExtractor] Failed to parse JSON from LLM output: %s. Returning empty list.",
                    e,
                )
                return []

        # 5) JSON dict → look for "disease_list"
        if isinstance(obj, dict):
            dl = obj.get("disease_list")
            if isinstance(dl, list):
                return self._normalize_list(dl)
            _LOG.warning(
                "[DiseaseListExtractor] Parsed JSON dict missing 'disease_list' or not a list; keys=%s",
                list(obj.keys()),
            )
            return []

        # 6) JSON root is a bare list → accept as fallback
        if isinstance(obj, list):
            _LOG.warning(
                "[DiseaseListExtractor] Parsed JSON is a bare list instead of object with 'disease_list'; "
                "using it directly."
            )
            return self._normalize_list(obj)

        _LOG.warning(
            "[DiseaseListExtractor] Parsed JSON is neither dict nor list (type=%s); returning empty list.",
            type(obj).__name__,
        )
        return []




    def _normalize_list(self, items: List[Any]) -> List[str]:
        """Strip, drop empties, and dedupe (case-insensitive, order-preserving)."""
        seen = set()
        out: List[str] = []
        for it in items:
            s = str(it).strip()
            if not s:
                continue
            key = s.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out


    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        Remove ```...``` markdown fences if the model responded with a code block.
        """
        s = text.strip()
        if s.startswith("```"):
            # Remove first line (``` or ```json)
            lines = s.splitlines()
            # Drop the opening fence
            if lines:
                lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines)
        return text

    @staticmethod
    def _try_json_list(text: str) -> Optional[List[str]]:
        """
        Try to interpret the text as JSON and normalize to List[str].
        """
        try:
            data = json.loads(text)
        except Exception:
            # Sometimes model wraps JSON in extra text; attempt a crude extraction.
            # This is intentionally simple; you can make it stricter later.
            try:
                start = text.index("[")
                end = text.rindex("]") + 1
                data = json.loads(text[start:end])
            except Exception:
                return None

        # Direct list
        if isinstance(data, list):
            return [str(x) for x in data]

        # Sometimes model returns {"diseases": [...]}
        if isinstance(data, dict):
            for key in ("diseases", "disease_list", "conditions", "items"):
                if key in data and isinstance(data[key], list):
                    return [str(x) for x in data[key]]

        return None

    @staticmethod
    def _dedupe_strip(items: List[str]) -> List[str]:
        """
        Strip whitespace, drop empties, and deduplicate (case-insensitive, preserve order).
        """
        seen = set()
        out: List[str] = []
        for raw in items:
            s = str(raw).strip()
            if not s:
                continue
            key = s.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _log_extraction(self, trial_id: str, prompt: str, output: Any):
        """
        Write the exact prompt + raw LLM output to:
        mbench/DiseaseListExtractor/<trial_id>_diseaselist_extractor.txt
        """
        path = self.log_dir / f"{trial_id}_diseaselist_extractor.txt"

        try:
            with path.open("w", encoding="utf-8") as f:
                f.write("=== LLM Disease List Extraction ===\n\n")
                f.write("### PROMPT ###\n")
                f.write(prompt)
                f.write("\n\n### OUTPUT ###\n")
                if isinstance(output, str):
                    f.write(output)
                else:
                    f.write(json.dumps(output, ensure_ascii=False, indent=2))
                f.write("\n")
        except Exception as e:
            _LOG.warning("[DiseaseListExtractor] Failed writing log file %s: %s", path, e)
