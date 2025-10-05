#!/usr/bin/env python3
"""
smt_criteria_gate_module.py

LLM-only gate. Uses a TWO-BLOCK context layout:

BLOCK 1 (authoritative, subcohort-scoped):
  - shared_context
  - subcohort_context
  - side_criteria (ONLY the requested side: inclusion OR exclusion)

BLOCK 2 (raw, unprocessed):
  - corpus_item_json (the whole JSONL corpus item for the parent trial; may include all subcohorts)

Goal:
Decide whether the TARGET SIDE criteria for THIS SUBCOHORT is substantively present.

Expected LLM JSON (binary):
  {"criteria_present": true|false, "reason": "short"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from trial_compiler.ir_finalizer.utils.costing import OPENAI_PRICING, count_tokens, estimate_cost_usd  # type: ignore
except Exception:
    OPENAI_PRICING = {}  # type: ignore

    def count_tokens(text: str, model: str):  # type: ignore
        return None, "costing_py_missing"

    def estimate_cost_usd(  # type: ignore
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,
    ):
        return None


@dataclass
class CriteriaGateInputs:
    trial_id: str
    effective_trial_id: str
    side: str  # "inclusion" or "exclusion"
    cohort_id: str
    cohort_label: str

    # Block 1 (authoritative)
    shared_context: str
    subcohort_context: str
    side_criteria: str

    # Block 2 (raw)
    corpus_item_json: str

    # Debug/trace
    shared_context_truncated: bool = False
    subcohort_context_truncated: bool = False
    side_criteria_truncated: bool = False
    corpus_item_truncated: bool = False
    corpus_item_len_original: int = 0


def _as_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _cap(s: str, max_chars: int) -> Tuple[str, bool]:
    s = s or ""
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars], True
    return s, False


def _jsonish(x: Any) -> Tuple[str, int]:
    """
    Convert x to a JSON string if possible; otherwise fallback to str(x).
    Returns (string, original_len).
    """
    if x is None:
        return "", 0
    if isinstance(x, str):
        return x, len(x)
    try:
        s = json.dumps(x, ensure_ascii=False)
        return s, len(s)
    except Exception:
        s = str(x)
        return s, len(s)


class SMTCriteriaGate:
    """
    Gatekeeper that decides whether the corresponding NL criteria for THIS SUBCOHORT & SIDE is present.

    If NOT present, returns meta["delete_file"]=True so the orchestrator can delete/skip writing the SMT file.
    """

    def __init__(
        self,
        call_llm: Optional[Callable[[str], str]] = None,
        prompt_template: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        max_shared_chars: int = 8000,
        max_subcohort_chars: int = 8000,
        # Backward-compat: previously used for contextual_text; now used as side_criteria cap if max_side_criteria_chars is None.
        max_contextual_chars: int = 8000,
        max_side_criteria_chars: Optional[int] = None,
        max_corpus_item_chars: int = 24000,
    ) -> None:
        self.call_llm = call_llm
        self.prompt_template = prompt_template
        self.model_name = model_name or "unknown"

        self.max_shared_chars = int(max_shared_chars)
        self.max_subcohort_chars = int(max_subcohort_chars)
        self.max_contextual_chars = int(max_contextual_chars)

        self.max_side_criteria_chars = (
            int(max_side_criteria_chars) if max_side_criteria_chars is not None else int(max_contextual_chars)
        )
        self.max_corpus_item_chars = int(max_corpus_item_chars)

        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

    def gate(self, subcohort_ctx: Dict[str, Any], side: str, smt_text: str) -> Tuple[str, Dict[str, Any]]:
        self.last_prompt = None
        self.last_raw = None

        if side not in ("inclusion", "exclusion"):
            return smt_text, {"executed": False, "error": f"invalid_side:{side}", "model": self.model_name}

        gi = self._extract_inputs(subcohort_ctx, side)
        model = self.model_name

        def _pricing_used() -> Optional[Dict[str, Any]]:
            pr = OPENAI_PRICING.get(model)
            if pr is None:
                return None
            return {
                "input_per_1m": pr.input_per_1m,
                "output_per_1m": pr.output_per_1m,
                "cached_input_per_1m": pr.cached_input_per_1m,
                "source": "https://platform.openai.com/docs/pricing",
            }

        if self.call_llm is None:
            return smt_text, {
                "executed": False,
                "error": "call_llm_required_for_gate",
                "model": model,
                "pricing_used": _pricing_used(),
            }

        # Safety guard: if we have literally no context at all, never delete.
        if (
            (not gi.shared_context.strip())
            and (not gi.subcohort_context.strip())
            and (not gi.side_criteria.strip())
            and (not gi.corpus_item_json.strip())
        ):
            return smt_text, {
                "executed": True,
                "gate_action": "KEEP",
                "delete_file": False,
                "criteria_present": True,
                "criteria_present_model": None,
                "reason": "forced_keep:all_context_fields_empty",
                "changed_smt": False,
                "no_change_required": True,
                "model": model,
                "estimated_cost_usd": None,
                "pricing_used": _pricing_used(),
                "trial_id": gi.trial_id,
                "effective_trial_id": gi.effective_trial_id,
                "side": gi.side,
                "cohort_id": gi.cohort_id,
                "cohort_label": gi.cohort_label,
                "gate_fast_path": True,
                # audit
                "cohort_match_method": subcohort_ctx.get("cohort_match_method"),
                "cohort_match_index": subcohort_ctx.get("cohort_match_index"),
            }

        prompt = self._build_prompt(gi)
        self.last_prompt = prompt

        try:
            raw = self.call_llm(prompt)
        except Exception as e:
            return smt_text, {
                "executed": False,
                "error": f"llm_call_error:{e}",
                "model": model,
                "pricing_used": _pricing_used(),
            }

        self.last_raw = raw

        try:
            obj = self._parse_json(raw or "")
        except Exception as e:
            return smt_text, {
                "executed": False,
                "error": f"json_parse_error:{e}",
                "model": model,
                "pricing_used": _pricing_used(),
            }

        criteria_present = obj.get("criteria_present")
        reason = obj.get("reason", "")

        if not isinstance(criteria_present, bool):
            return smt_text, {
                "executed": False,
                "error": "parsed_json_missing_or_nonbool:criteria_present",
                "model": model,
                "pricing_used": _pricing_used(),
            }

        # Binary deletion rule:
        delete_file = (criteria_present is False)

        pt, pnote = count_tokens(self.last_prompt or "", model)
        ct, cnote = count_tokens(self.last_raw or "", model)
        est_cost = None
        if pt is not None and ct is not None:
            est_cost = estimate_cost_usd(model, int(pt), int(ct))

        meta_out: Dict[str, Any] = {
            "executed": True,
            "gate_action": "DELETE" if delete_file else "KEEP",
            "delete_file": bool(delete_file),
            "criteria_present": bool(criteria_present),
            "criteria_present_model": bool(criteria_present),
            "reason": str(reason or ""),
            "changed_smt": False,
            "no_change_required": True,
            "model": model,
            "prompt_tokens_est": pt,
            "completion_tokens_est": ct,
            "tokenizer_note_prompt": str(pnote),
            "tokenizer_note_completion": str(cnote),
            "estimated_cost_usd": est_cost,
            "pricing_used": _pricing_used(),
            "trial_id": gi.trial_id,
            "effective_trial_id": gi.effective_trial_id,
            "side": gi.side,
            "cohort_id": gi.cohort_id,
            "cohort_label": gi.cohort_label,
            # lengths + truncation
            "shared_context_len": len(gi.shared_context or ""),
            "subcohort_context_len": len(gi.subcohort_context or ""),
            "side_criteria_len": len(gi.side_criteria or ""),
            "corpus_item_json_len": len(gi.corpus_item_json or ""),
            "corpus_item_len_original": int(gi.corpus_item_len_original),
            "shared_context_truncated": bool(gi.shared_context_truncated),
            "subcohort_context_truncated": bool(gi.subcohort_context_truncated),
            "side_criteria_truncated": bool(gi.side_criteria_truncated),
            "corpus_item_truncated": bool(gi.corpus_item_truncated),
            # audit: how we matched eff_tid -> cohort
            "cohort_match_method": subcohort_ctx.get("cohort_match_method"),
            "cohort_match_index": subcohort_ctx.get("cohort_match_index"),
        }
        return smt_text, meta_out

    def _extract_inputs(self, ctx: Dict[str, Any], side: str) -> CriteriaGateInputs:
        trial_id = ctx.get("trial_id_parent") or ctx.get("parent_trial_id") or ctx.get("trial_id") or ""
        eff_tid = ctx.get("trial_id_effective") or ctx.get("trial_id") or trial_id or ""
        cohort_id = ctx.get("cohort_id") or ctx.get("substudy_id") or "C?"
        cohort_label = ctx.get("cohort_label") or ctx.get("substudy_label") or "Unknown cohort"

        # Block 1
        shared_context = _as_str(ctx.get("shared_context"))

        # Prefer explicit subcohort_context; fallback to legacy key "context"
        subcohort_context = _as_str(ctx.get("subcohort_context")) or _as_str(ctx.get("context"))

        # Side-specific criteria (ONLY requested side)
        side_criteria = _as_str(ctx.get("side_criteria"))
        if not side_criteria.strip():
            # Compatibility fallbacks if upstream still passes both fields
            if side == "inclusion":
                side_criteria = _as_str(ctx.get("inclusion_criteria"))
            else:
                side_criteria = _as_str(ctx.get("exclusion_criteria"))

        # Block 2 (raw corpus JSONL item)
        corpus_item = (
            ctx.get("corpus_item")
            or ctx.get("corpus_item_json")
            or ctx.get("corpus_jsonl_item")
            or ctx.get("raw_corpus_item")
        )
        corpus_item_json, corpus_len_orig = _jsonish(corpus_item)

        # Cap sizes to keep prompts bounded
        shared_context, shared_trunc = _cap(shared_context, self.max_shared_chars)
        subcohort_context, subcohort_trunc = _cap(subcohort_context, self.max_subcohort_chars)
        side_criteria, side_trunc = _cap(side_criteria, self.max_side_criteria_chars)

        corpus_trunc = False
        if self.max_corpus_item_chars > 0 and len(corpus_item_json) > self.max_corpus_item_chars:
            corpus_item_json = corpus_item_json[: self.max_corpus_item_chars]
            corpus_trunc = True

        return CriteriaGateInputs(
            trial_id=str(trial_id),
            effective_trial_id=str(eff_tid),
            side=str(side),
            cohort_id=str(cohort_id),
            cohort_label=str(cohort_label),
            shared_context=shared_context,
            subcohort_context=subcohort_context,
            side_criteria=side_criteria,
            corpus_item_json=corpus_item_json,
            shared_context_truncated=shared_trunc,
            subcohort_context_truncated=subcohort_trunc,
            side_criteria_truncated=side_trunc,
            corpus_item_truncated=corpus_trunc,
            corpus_item_len_original=int(corpus_len_orig),
        )

    def _build_prompt(self, gi: CriteriaGateInputs) -> str:
        if self.prompt_template:
            t = self.prompt_template

            repl = {
                "{{TRIAL_ID}}": gi.trial_id,
                "{{EFFECTIVE_TRIAL_ID}}": gi.effective_trial_id,
                "{{SIDE}}": gi.side,
                "{{COHORT_ID}}": gi.cohort_id,
                "{{COHORT_LABEL}}": gi.cohort_label,
                "{{SHARED_CONTEXT}}": gi.shared_context or "",
                "{{CONTEXT}}": gi.subcohort_context or "",
                "{{SUBCOHORT_CONTEXT}}": gi.subcohort_context or "",
                "{{SIDE_CRITERIA}}": gi.side_criteria or "",
                "{{CORPUS_ITEM_JSON}}": gi.corpus_item_json or "",
                "{{CORPUS_ITEM}}": gi.corpus_item_json or "",
            }
            for k, v in repl.items():
                t = t.replace(k, v)

            # Enforce "two-block" if old templates reference legacy placeholders
            t = t.replace("{{CONTEXTUAL_TEXT}}", "")
            t = t.replace("{{CONFIDENCE}}", "")
            return t

        # fallback built-in prompt (TWO blocks, binary output)
        return f"""
You are a conservative gatekeeper for a clinical-trial eligibility pipeline.

Conservative policy:
- Mark criteria_present=false ONLY when you are very sure the TARGET SIDE criteria for THIS SUBCOHORT is not present.
- If uncertain, choose criteria_present=true (KEEP). Deletion is destructive.

TARGET SIDE: {gi.side.upper()}

CRITICAL: subcohort scope
- This trial may have multiple subcohorts/arms.
- You MUST judge ONLY the subcohort identified by COHORT_ID/COHORT_LABEL and described in <subcohort_context>.
- Do NOT count criteria that clearly belong to other subcohorts.
- If you cannot confidently attribute a criterion to THIS subcohort, assume it is NOT safe to delete; choose KEEP.

Two-block input:
- Block 1 (authoritative) is the subcohort-scoped bundle, including ONLY the requested side's criteria.
- Block 2 is the raw corpus JSONL item for the whole trial (may include all subcohorts). Use it only as a safety check
  (e.g., if Block 1 seems empty due to preprocessing misses). Still respect subcohort scope.

Definition (criteria_present):
- true  => ANY real eligibility content for the TARGET SIDE for THIS SUBCOHORT exists (even one condition like "Age >= 18").
- false => the TARGET SIDE criteria for THIS SUBCOHORT is effectively missing/placeholder-only/boilerplate.

Return ONLY JSON (no markdown, no extra text):
{{"criteria_present": true|false, "reason": "short"}}

TRIAL_ID={gi.trial_id}
EFFECTIVE_TRIAL_ID={gi.effective_trial_id}
COHORT_ID={gi.cohort_id}
COHORT_LABEL={gi.cohort_label}

# === BLOCK 1: SUBCOHORT SIDE BUNDLE (authoritative) ===
<shared_context>
{gi.shared_context}
</shared_context>

<subcohort_context>
{gi.subcohort_context}
</subcohort_context>

<side_criteria>
{gi.side_criteria}
</side_criteria>

# === BLOCK 2: CORPUS ITEM (raw, unprocessed; all subcohorts) ===
<corpus_item_json>
{gi.corpus_item_json}
</corpus_item_json>
""".strip()

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("empty LLM output")

        if text.startswith("```"):
            lines = text.splitlines()[1:]
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
