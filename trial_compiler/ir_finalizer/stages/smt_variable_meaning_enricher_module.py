#!/usr/bin/env python3
"""
smt_variable_meaning_enricher_module.py

Core module to enrich/repair variable definitions in SMT-LIB v2 programs by ensuring
every (declare-const ...) has an inline JSON definition comment on the same line.

This stage is VARIABLE-DEFINITION-ONLY:
- Prompt should enforce "do not modify asserts or other code".

Outputs meta with best-effort token/cost estimation via costing.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from trial_compiler.ir_finalizer.utils.costing import OPENAI_PRICING, count_tokens, estimate_cost_usd


@dataclass
class MeaningEnricherInputs:
    trial_id: str
    effective_trial_id: str
    side: str  # "inclusion" or "exclusion"
    cohort_id: str
    cohort_label: str

    inclusion_criteria: str
    exclusion_criteria: str

    context: str
    contextual_text: str
    shared_context: str = ""


class SMTVariableMeaningEnricher:
    """
    LLM wrapper that returns an updated SMT program (full text),
    or a sentinel indicating no modification is required.
    """

    NO_MODIFICATION_SENTINELS = {
        "NO_MODIFICATION_REQUIRED",
        '"NO_MODIFICATION_REQUIRED"',
        "'NO_MODIFICATION_REQUIRED'",
    }

    def __init__(
        self,
        call_llm: Optional[Callable[[str], str]] = None,
        prompt_template: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.call_llm = call_llm
        self.prompt_template = prompt_template
        self.model_name = model_name

        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

    # Public API
    def enrich(
        self,
        subcohort_ctx: Dict[str, Any],
        smt_text: str,
        *,
        side_hint: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self.last_prompt = None
        self.last_raw = None

        try:
            pi = self._extract_inputs(subcohort_ctx, side_hint=side_hint)
        except Exception as e:
            logging.error("[MEANING] Failed to extract inputs: %s", e)
            return smt_text, {"executed": False, "error": f"failed_to_extract_inputs: {e}"}

        model = self.model_name or "unknown"

        # Placeholder mode (no LLM call)
        if self.call_llm is None:
            meta: Dict[str, Any] = {
                "executed": True,
                "reason": "placeholder meaning enricher; SMT copied unchanged",
                "trial_id": pi.trial_id,
                "effective_trial_id": pi.effective_trial_id,
                "side": pi.side,
                "cohort_id": pi.cohort_id,
                "cohort_label": pi.cohort_label,
                "changed_smt": False,
                "no_change_required": True,
                "model": model,
            }
            pr = OPENAI_PRICING.get(model)
            if pr:
                meta["pricing_used"] = {
                    "input_per_1m": pr.input_per_1m,
                    "output_per_1m": pr.output_per_1m,
                    "cached_input_per_1m": pr.cached_input_per_1m,
                    "source": "https://platform.openai.com/docs/pricing",
                }
            return smt_text, meta

        if not self.prompt_template:
            return smt_text, {"executed": False, "error": "missing_prompt_template", "model": model}

        # Real mode: call LLM first, then interpret sentinel or JSON
        try:
            prompt = self._build_prompt(pi, smt_text)
            self.last_prompt = prompt
            raw = self.call_llm(prompt)
            self.last_raw = raw
        except Exception as e:
            logging.error("[MEANING] LLM call failed: %s", e)
            return smt_text, {"executed": False, "error": f"llm_call_error: {e}", "model": model}

        # Best-effort token + cost estimation
        prompt_toks, prompt_note = count_tokens(self.last_prompt or "", model)
        comp_toks, comp_note = count_tokens(self.last_raw or "", model)
        est_cost = None
        if prompt_toks is not None and comp_toks is not None:
            est_cost = estimate_cost_usd(model, prompt_toks, comp_toks)

        pr = OPENAI_PRICING.get(model)
        pricing_used = None
        if pr is not None:
            pricing_used = {
                "input_per_1m": pr.input_per_1m,
                "output_per_1m": pr.output_per_1m,
                "cached_input_per_1m": pr.cached_input_per_1m,
                "source": "https://platform.openai.com/docs/pricing",
            }

        raw_text = (raw or "").strip()
        if raw_text in self.NO_MODIFICATION_SENTINELS:
            return smt_text, {
                "executed": True,
                "reason": "llm_declared_no_modification_required",
                "trial_id": pi.trial_id,
                "effective_trial_id": pi.effective_trial_id,
                "side": pi.side,
                "cohort_id": pi.cohort_id,
                "cohort_label": pi.cohort_label,
                "no_change_required": True,
                "changed_smt": False,
                "model": model,
                "prompt_tokens_est": prompt_toks,
                "completion_tokens_est": comp_toks,
                "tokenizer_note_prompt": prompt_note,
                "tokenizer_note_completion": comp_note,
                "estimated_cost_usd": est_cost,
                "pricing_used": pricing_used,
            }

        try:
            obj = self._parse_json(raw_text)
        except Exception as e:
            logging.error("[MEANING] JSON parsing failed: %s", e)
            return smt_text, {
                "executed": False,
                "error": f"llm_or_parse_error: {e}",
                "model": model,
                "prompt_tokens_est": prompt_toks,
                "completion_tokens_est": comp_toks,
                "tokenizer_note_prompt": prompt_note,
                "tokenizer_note_completion": comp_note,
                "estimated_cost_usd": est_cost,
                "pricing_used": pricing_used,
            }

        repaired = obj.get("repaired_smt")
        notes = obj.get("notes", "")

        if not isinstance(repaired, str):
            return smt_text, {
                "executed": False,
                "error": "non_string_repaired_smt",
                "notes": notes,
                "model": model,
                "prompt_tokens_est": prompt_toks,
                "completion_tokens_est": comp_toks,
                "tokenizer_note_prompt": prompt_note,
                "tokenizer_note_completion": comp_note,
                "estimated_cost_usd": est_cost,
                "pricing_used": pricing_used,
            }

        repaired_stripped = repaired.strip()
        if not repaired_stripped:
            return smt_text, {
                "executed": False,
                "error": "empty_repaired_smt",
                "notes": notes,
                "model": model,
                "prompt_tokens_est": prompt_toks,
                "completion_tokens_est": comp_toks,
                "tokenizer_note_prompt": prompt_note,
                "tokenizer_note_completion": comp_note,
                "estimated_cost_usd": est_cost,
                "pricing_used": pricing_used,
            }

        meta = {
            "executed": True,
            "notes": notes,
            "trial_id": pi.trial_id,
            "effective_trial_id": pi.effective_trial_id,
            "side": pi.side,
            "cohort_id": pi.cohort_id,
            "cohort_label": pi.cohort_label,
            "model": model,
            "prompt_tokens_est": prompt_toks,
            "completion_tokens_est": comp_toks,
            "tokenizer_note_prompt": prompt_note,
            "tokenizer_note_completion": comp_note,
            "estimated_cost_usd": est_cost,
            "pricing_used": pricing_used,
        }
        meta["changed_smt"] = (repaired_stripped != (smt_text or "").strip())
        meta["no_change_required"] = not meta["changed_smt"]
        return repaired, meta

    # ───────────────────── ctx → inputs ─────────────────────

    def _extract_inputs(self, ctx: Dict[str, Any], *, side_hint: Optional[str] = None) -> MeaningEnricherInputs:
        trial_id = ctx.get("trial_id_parent") or ctx.get("parent_trial_id") or ctx.get("trial_id")
        eff_tid = ctx.get("trial_id_effective") or ctx.get("trial_id") or trial_id

        # Prefer explicit side in subctx; else side_hint; else unknown
        side = (ctx.get("inc_exc") or side_hint or "").strip().lower()
        if side not in ("inclusion", "exclusion"):
            side = (side_hint or "unknown").strip().lower()

        cohort_id = ctx.get("cohort_id") or ctx.get("substudy_id") or "C?"
        cohort_label = ctx.get("cohort_label") or ctx.get("substudy_label") or "Unknown cohort"

        inc = ctx.get("inclusion_criteria", "") or ""
        exc = ctx.get("exclusion_criteria", "") or ""

        ctx_str = ctx.get("context", "") or ctx.get("cohort_context_raw", "") or ""
        contextual_text = ctx.get("contextual_text", "") or ""
        shared_context = ctx.get("shared_context", "") or ""

        pn = ctx.get("preprocessor_normalized") or {}
        if not shared_context:
            shared_context = pn.get("shared_context", "") or ""

        if not contextual_text:
            contextual_text = shared_context or ""

        return MeaningEnricherInputs(
            trial_id=str(trial_id),
            effective_trial_id=str(eff_tid),
            side=str(side),
            cohort_id=str(cohort_id),
            cohort_label=str(cohort_label),
            inclusion_criteria=str(inc) if inc else "(none provided)",
            exclusion_criteria=str(exc) if exc else "(none provided)",
            context=str(ctx_str),
            contextual_text=str(contextual_text),
            shared_context=str(shared_context),
        )

    # ───────────────────── prompt & JSON ─────────────────────

    def _build_prompt(self, pi: MeaningEnricherInputs, smt_text: str) -> str:
        t = self.prompt_template or ""

        # Strict placeholder contract (uppercase). If you also have {{side}} somewhere, support it too.
        t = t.replace("{{TRIAL_ID}}", pi.trial_id)
        t = t.replace("{{EFFECTIVE_TRIAL_ID}}", pi.effective_trial_id)
        t = t.replace("{{SIDE}}", pi.side)
        t = t.replace("{{side}}", pi.side)

        t = t.replace("{{COHORT_ID}}", pi.cohort_id)
        t = t.replace("{{COHORT_LABEL}}", pi.cohort_label)

        t = t.replace("{{SHARED_CONTEXT}}", pi.shared_context)
        t = t.replace("{{CONTEXT}}", pi.context)
        t = t.replace("{{CONTEXTUAL_TEXT}}", pi.contextual_text)

        t = t.replace("{{INCLUSION_CRITERIA}}", pi.inclusion_criteria)
        t = t.replace("{{EXCLUSION_CRITERIA}}", pi.exclusion_criteria)

        t = t.replace("{{SMT_PROGRAM}}", smt_text)
        return t

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Robust-ish JSON extractor:
        - Accepts fenced ```json blocks
        - Otherwise attempts to parse the full string
        - Otherwise finds the first {...} span and parses that
        """
        text = (raw or "").strip()
        if not text:
            raise ValueError("empty LLM output")

        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            if lines and lines[0].strip().lower() == "json":
                lines = lines[1:]
            text = "\n".join(lines).strip()

        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("parsed JSON is not an object")
            return obj
        except Exception:
            pass

        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            snippet = text[l : r + 1]
            obj = json.loads(snippet)
            if not isinstance(obj, dict):
                raise ValueError("parsed JSON is not an object")
            return obj

        raise ValueError("could not parse JSON from LLM output")