from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from trial_compiler.disease_compiler.modules.DiseaseListPreprocessor.stages.DiseaseListExtractor import _dict_to_multiline_string


import dspy  # type: ignore

_LOG = logging.getLogger(__name__)


class DiseaseListEliminator(dspy.Module):
    """
    Second-stage LLM module that refines (eliminates) diseases.

    Inputs in ctx:
        - "contextual_text": str
        - "disease_list": List[str]    # preprocessed disease list from earlier stage

    Output in ctx:
        - "original_disease_list": List[str]
        - "processed_disease_list": List[str]
    """

    def __init__(
        self,
        engine: Any,
        *,
        verbose: bool = False,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
        log_dir: str | Path = "mbench/DiseaseListEliminator",
    ) -> None:
        super().__init__()
        self.engine = engine
        self.verbose = verbose
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────── public API ───────────────

    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call LLM, parse processed_disease_list, and store back into ctx.
        """
        if self.engine is None:
            _LOG.warning(
                "[DiseaseListEliminator] engine is None; skipping LLM elimination."
            )
            dl = ctx.get("disease_list", []) or []
            ctx.setdefault("original_disease_list", list(dl))
            ctx.setdefault("processed_disease_list", list(dl))
            return ctx

        # ---- build trial-level context from parsed ctx["contextual"] ----
        parsed_context = ctx.get("contextual")
        if isinstance(parsed_context, dict):
            filtered = {k: v for k, v in parsed_context.items() if k != "text"}
            contextual_text = _dict_to_multiline_string(filtered)
        else:
            contextual_text = ctx.get("contextual_text", "") or ""

        # ---- cohort context ----
        cohort_contextual_text: str = ctx.get("cohort_contextual_text", "") or ""
        has_cohort_ctx = bool(cohort_contextual_text.strip())

        preprocessed_list: List[str] = ctx.get("processed_disease_list", []) or []

        if not preprocessed_list:
            _LOG.info("[DiseaseListEliminator] preprocessed disease_list is empty; skipping LLM.")
            ctx.setdefault("processed_disease_list", [])
            return ctx

        # Pick prompt: cohort-specific if present, else default
        if has_cohort_ctx:
            prompt_template: Optional[str] = ctx.get("DiseaseListEliminator_cohort_prompt")  # NEW
        else:
            prompt_template = ctx.get("DiseaseListEliminator_prompt")

        prompt = self._build_prompt(
            tmpl=prompt_template,
            context=contextual_text,
            disease_list=preprocessed_list,
            cohort_context=cohort_contextual_text if has_cohort_ctx else None,
            use_cohort_prompt=has_cohort_ctx,
        )


        if self.verbose:
            _LOG.info(
                "[DiseaseListEliminator] Calling LLM with prompt length=%d; num_diseases=%d",
                len(prompt),
                len(preprocessed_list),
            )

        raw_output = self._call_llm(prompt)

        if self.verbose:
            preview = raw_output if isinstance(raw_output, str) else str(raw_output)
            _LOG.info(
                "[DiseaseListEliminator] Raw LLM output (preview): %r", preview[:500]
            )

        processed_list = self._parse_processed_disease_list(raw_output)

        #ctx["original_disease_list"] = list(preprocessed_list)
        ctx["processed_disease_list"] = processed_list

        trial_id = ctx.get("trial_id", "UNKNOWN")
        self._log_extraction(trial_id, prompt, raw_output)

        if self.verbose:
            _LOG.info(
                "[DiseaseListEliminator] Parsed %d processed disease items.",
                len(processed_list),
            )

        return ctx

    # ─────────────── internals ───────────────

    def _build_prompt(
        self,
        tmpl: Optional[str],
        context: str,
        disease_list: List[str],
        cohort_context: Optional[str] = None,
        use_cohort_prompt: bool = False,
    ) -> str:
        """
        Build eliminator prompt.

        - If ctx["DiseaseListEliminator_prompt"] exists, use it.
        - Otherwise load:
            prompts/clinical_trial/RequirementExtractor/RequirementDiseaseEliminator.prompt
        - Replace:
            #CONTEXTUAL_TEXT#           -> context
            #PREPROCESSED_DISEASE_LIST# -> JSON list of diseases
        """
        default_prompt_path = (
            Path(__file__)
            .resolve()
            .parents[3]
            / "prompts"
            / "clinical_trial"
            / "RequirementExtractor"
            / "RequirementDiseaseEliminator.prompt"
        )
        default_cohort_prompt_path = (
            Path(__file__).resolve().parents[3]
            / "prompts" / "clinical_trial" / "RequirementExtractor"
            / "RequirementDiseaseEliminator_Subcohort.prompt"
        )

        if not tmpl:
            try:
                if use_cohort_prompt and default_cohort_prompt_path.exists():
                    tmpl = default_cohort_prompt_path.read_text(encoding="utf-8")
                else:
                    tmpl = default_prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                _LOG.warning(
                    "[DiseaseListEliminator] Prompt file missing, using fallback. Error: %s",
                    e,
                )
                base = (
                    "You are a medical expert. Given the trial text and a list of diseases, "
                    "remove any items that are not truly part of the target disease definition.\n\n"
                    f"Trial text:\n{context}\n\n"
                )
                if cohort_context and cohort_context.strip():
                    base += f"Cohort-specific context:\n{cohort_context}\n\n"
                base += (
                    f"Preprocessed disease list:\n{json.dumps(disease_list, ensure_ascii=False)}\n\n"
                    "Return JSON in the format:\n"
                    '{ "original_disease_list": [...], "processed_disease_list": [...] }'
                )
                return base

        prompt = tmpl

        # Trial context
        if "#TRIAL_CONTEXTUAL_TEXT#" in prompt:
            prompt = prompt.replace("#TRIAL_CONTEXTUAL_TEXT#", context)
        else:
            prompt = prompt.rstrip() + f"\n\n### Trial Context\n{context}\n"

        # Cohort context (NEW)
        if cohort_context and cohort_context.strip():
            if "#COHORT_CONTEXTUAL_TEXT#" in prompt:
                prompt = prompt.replace("#COHORT_CONTEXTUAL_TEXT#", cohort_context)
            else:
                prompt = prompt.rstrip() + f"\n\n### Cohort-specific Context\n{cohort_context}\n"

        # Disease list
        dl_json = json.dumps(disease_list, ensure_ascii=False)
        if "#PREPROCESSED_DISEASE_LIST#" in prompt:
            prompt = prompt.replace("#PREPROCESSED_DISEASE_LIST#", dl_json)
        else:
            prompt = prompt.rstrip() + f"\n\n### Preprocessed disease list\n{dl_json}\n"

        return prompt

    def _call_llm(self, prompt: str) -> Any:
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
                    return self.engine(prompt)
                else:
                    raise TypeError(
                        "Engine does not support `.complete` or direct call."
                    )
            except Exception as e:
                last_err = e
                _LOG.warning(
                    "[DiseaseListEliminator] LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )

        raise RuntimeError(
            "[DiseaseListEliminator] LLM call failed after "
            f"{self.max_retries} attempts."
        ) from last_err

    def _parse_processed_disease_list(self, raw: Any) -> List[str]:
        """
        Parse:

            {
                "original_disease_list": [],
                "processed_disease_list": []
            }
        """
        if raw is None:
            _LOG.warning(
                "[DiseaseListEliminator] LLM output is None; returning empty processed_disease_list."
            )
            return []

        # Dict case
        if isinstance(raw, dict):
            dl = raw.get("processed_disease_list")
            if isinstance(dl, list):
                return self._normalize_list(dl)

            alt = raw.get("disease_list")
            if isinstance(alt, list):
                _LOG.warning(
                    "[DiseaseListEliminator] Dict missing 'processed_disease_list'; using 'disease_list'."
                )
                return self._normalize_list(alt)

            _LOG.warning(
                "[DiseaseListEliminator] Dict has no usable processed list; keys=%s",
                list(raw.keys()),
            )
            return []

        # List case
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], str):
                _LOG.info(
                    "[DiseaseListEliminator] Single-element list of string; re-parsing inner JSON."
                )
                return self._parse_processed_disease_list(raw[0])

            for elem in raw:
                if isinstance(elem, dict) and isinstance(
                    elem.get("processed_disease_list"), list
                ):
                    _LOG.info(
                        "[DiseaseListEliminator] Found dict with 'processed_disease_list' inside list."
                    )
                    return self._normalize_list(elem["processed_disease_list"])

            _LOG.warning(
                "[DiseaseListEliminator] Bare list returned; treating it as processed_disease_list."
            )
            return self._normalize_list(raw)

        # String → JSON
        if not isinstance(raw, str):
            raw = str(raw)

        text = self._strip_code_fences(raw).strip()

        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                candidate = text[start:end]
                obj = json.loads(candidate)
            except Exception as e:
                _LOG.warning(
                    "[DiseaseListEliminator] Failed to parse JSON from LLM output: %s. Returning empty list.",
                    e,
                )
                return []

        if isinstance(obj, dict):
            dl = obj.get("processed_disease_list")
            if isinstance(dl, list):
                return self._normalize_list(dl)

            alt = obj.get("disease_list")
            if isinstance(alt, list):
                _LOG.warning(
                    "[DiseaseListEliminator] JSON dict lacks 'processed_disease_list'; using 'disease_list'."
                )
                return self._normalize_list(alt)

            _LOG.warning(
                "[DiseaseListEliminator] Parsed JSON dict has no usable processed list; keys=%s",
                list(obj.keys()),
            )
            return []

        if isinstance(obj, list):
            _LOG.warning(
                "[DiseaseListEliminator] Parsed JSON is bare list; treating as processed_disease_list."
            )
            return self._normalize_list(obj)

        _LOG.warning(
            "[DiseaseListEliminator] Parsed JSON is neither dict nor list (type=%s); returning empty list.",
            type(obj).__name__,
        )
        return []

    def _strip_code_fences(self, text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines)
        return text

    def _normalize_list(self, items: List[Any]) -> List[str]:
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

    def _log_extraction(self, trial_id: str, prompt: str, output: Any) -> None:
        path = self.log_dir / f"{trial_id}_disease_eliminator.txt"
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write("=== LLM Disease List Eliminator ===\n\n")
                f.write("### PROMPT ###\n")
                f.write(prompt)
                f.write("\n\n### OUTPUT ###\n")
                if isinstance(output, str):
                    f.write(output)
                else:
                    f.write(json.dumps(output, ensure_ascii=False, indent=2))
                f.write("\n")
        except Exception as e:
            _LOG.warning(
                "[DiseaseListEliminator] Failed writing log file %s: %s", path, e
            )
