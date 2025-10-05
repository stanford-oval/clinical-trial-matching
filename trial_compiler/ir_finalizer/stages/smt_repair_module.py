#!/usr/bin/env python3
"""
smt_repair_module.py

Core SMT repair module.

Public API:
    from trial_compiler.ir_finalizer.stages.smt_repair_module import SMTRepairer
    repaired_smt, meta = SMTRepairer().repair(subcohort_ctx, smt_text)

Default behavior (no call_llm):
    - Extracts IDs and key text fields from `subcohort_ctx`
      (inclusion_criteria, exclusion_criteria, context, contextual_text).
    - Does NOT modify the SMT program.
    - Returns the original SMT and metadata with executed=True so the
      driver can copy SMTs into ir_repaired and write mbench logs.

If you pass call_llm(prompt: str) -> str:
    - Builds a prompt (from ./prompt/smt_repair.prompt if present, otherwise
      a built-in default).
    - Calls the LLM.
    - Expects JSON like:
        {"repaired_smt": "<FULL SMT PROGRAM or 'NO MODIFICATION REQUIRED'>",
         "notes": "<short explanation>"}
    - If "repaired_smt" is the sentinel string "NO MODIFICATION REQUIRED",
      the original SMT program is returned unchanged (BUT see strict format enforcement below).
    - Otherwise, the SMT program from "repaired_smt" is returned.

Strict format enforcement (optional; default ON):
    - Enforces variable declaration format constraints:
        * No (declare-fun ...) or (define-fun ...)
        * Only (declare-const <sym> Bool|Int|Real)
        * Each (declare-const ...) line must include a same-line JSON object comment
          with required keys (Bool vs numeric).
    - If the LLM output violates strict format, the module retries (up to max_retries).

Retry mechanism:
    - Retries up to max_retries times (default 1) when:
        * LLM call errors
        * JSON parsing errors / malformed output
        * "NO MODIFICATION REQUIRED" returned but the ORIGINAL SMT fails strict format (when enforced)
        * Repaired SMT fails strict format (when enforced)
        * Empty repaired_smt

NEW POLICY (per request):
    - If retries are exhausted due to strict-format failures, LET THE LAST ATTEMPT PASS THROUGH:
        * If last attempt returns NO MOD but original fails strict format -> return original anyway.
        * If last attempt returns repaired SMT but it fails strict format -> return repaired anyway.
      In these pass-through cases, meta will include:
        passed_through_last_attempt=True, format_ok=False, format_issues=[...],
        error="retry_exhausted_pass_through_last_attempt"

For mbench/debugging, last_prompt/last_raw include ALL attempts separated by headers.

EXPORTING (batched) (NEW):
    If you pass export_dir (str or pathlib.Path) to SMTRepairer:
      - For each repair() call where metadata["executed"] is True, the module queues two SMT-LIB files:
            <effective_trial_id>__<side>__<cohort_id>.before.smt2
            <effective_trial_id>__<side>__<cohort_id>.after.smt2
      - Files are written in BATCHES to reduce overhead and avoid “only appears on Ctrl+C”.
      - Control batching via:
            export_flush_every (default 100)  # queue length threshold for flushing
            export_atomic (default True)      # write temp + atomic rename
      - Exports are also flushed automatically at process exit (atexit).
      - You can call repairer.flush_exports() / repairer.close_exports() manually.

    Note:
      - The queue stores file contents in memory until flushed; reduce export_flush_every if SMTs are large.

Cost tracking:
    - Adds best-effort token + USD cost estimation fields to meta:
        model, prompt_tokens_est, completion_tokens_est,
        tokenizer_note_prompt, tokenizer_note_completion,
        estimated_cost_usd, pricing_used
    - For retries, these fields reflect the TOTAL across all attempts (best-effort).
    - Requires costing.py + tiktoken to produce token counts; otherwise
      estimated_cost_usd will be None and tokenizer_note_* will explain.

For mbench logging, the last prompt and raw response are exposed as:
    repairer.last_prompt, repairer.last_raw
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# costing.py is expected (used by polarity/logic/meaning modules too)
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


# ───────────────────────────── batched exporter ─────────────────────────────


class _BatchedExporter:
    """
    Queue (path, text) writes and flush them in batches.

    - Uses atomic temp + os.replace if atomic=True (default).
    - Flushes at process exit via atexit.register(self.flush).
    """

    def __init__(self, export_dir: Path, *, flush_every: int = 100, atomic: bool = True) -> None:
        self.export_dir = export_dir
        self.flush_every = max(1, int(flush_every))
        self.atomic = bool(atomic)
        self._q: List[Tuple[Path, str]] = []
        self._closed = False
        atexit.register(self.flush)

    def enqueue_write(self, path: Path, text: str) -> None:
        if self._closed:
            return
        self._q.append((path, text))
        if len(self._q) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._closed or not self._q:
            return

        # Snapshot & clear queue first so we don't lose newly enqueued items
        batch = self._q
        self._q = []

        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error("[REPAIR][EXPORT] Failed to create export_dir=%s: %s", self.export_dir, e)
            # Put batch back (best effort) so a future flush might succeed
            self._q = batch + self._q
            return

        for path, text in batch:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if self.atomic:
                    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
                    tmp_path = Path(tmp_name)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(text)
                            f.flush()
                        os.replace(str(tmp_path), str(path))  # atomic rename on POSIX
                    finally:
                        try:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        except Exception:
                            pass
                else:
                    path.write_text(text, encoding="utf-8")
            except Exception as e:
                logging.error("[REPAIR][EXPORT] Failed to write %s: %s", path, e)
                # If one file fails, continue; don't poison whole batch

    def close(self) -> None:
        self.flush()
        self._closed = True


# ───────────────────────────── core repairer ─────────────────────────────


@dataclass
class RepairInputs:
    trial_id: str
    effective_trial_id: str
    side: str
    cohort_id: str
    cohort_label: str
    inclusion_criteria: str
    exclusion_criteria: str
    context: str
    contextual_text: str
    shared_context: str = ""


class SMTRepairer:
    """
    SMTRepairer encapsulates the logic for "repairing" a single SMT program
    given a subcohort context.
    """

    NO_MOD_SENTINELS = {
        "NO MODIFICATION REQUIRED",
        "NO_MODIFICATION_REQUIRED",
    }

    # Strict declaration checking: require single-line declare-const with optional comments.
    _DECL_CONST_LINE_RE = re.compile(
        r"^\s*\(declare-const\s+([^\s\)]+)\s+([^\s\)]+)\)\s*(?:;.*)?$"
    )
    _FORBIDDEN_DECL_TOKENS = ("(declare-fun", "(define-fun")

    def __init__(
        self,
        call_llm: Optional[Callable[[str], str]] = None,
        prompt_template: Optional[str] = None,
        export_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        *,
        enforce_strict_format: bool = True,
        max_retries: int = 1,
        # NEW: batched export controls
        export_flush_every: int = 100,
        export_atomic: bool = True,
    ) -> None:
        """
        Args:
            call_llm:
                Optional function prompt -> string, e.g. Azure/OpenAI wrapper.

            prompt_template:
                Optional text template loaded from a file (e.g. ./prompt/smt_repair.prompt).

            export_dir:
                Optional directory (str or Path). If set, repair() will export
                "before" and "after" SMT programs for mbench under this folder.

            model_name:
                Optional model name for token/cost estimation metadata.

            enforce_strict_format:
                If True, validate repaired SMT (and "NO MOD" originals) for strict variable
                declaration + JSON annotation rules. If validation fails, retry.

            max_retries:
                Number of retries after the first attempt.
                Total attempts = 1 + max_retries.

            export_flush_every:
                Write exports in batches; flush when queued writes reach this size.
                Each repair() enqueues 2 writes (before+after).

            export_atomic:
                If True, write tmp + os.replace for each file (avoids partial files).
        """
        self.call_llm = call_llm
        self.prompt_template = prompt_template
        self.export_dir: Optional[Path] = (
            Path(export_dir).expanduser().resolve() if export_dir is not None else None
        )
        self.model_name = model_name
        self.enforce_strict_format = bool(enforce_strict_format)
        self.max_retries = max(0, int(max_retries))

        # For mbench logging: last prompt and raw output per repair() call
        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

        # NEW: batched exporter
        self._exporter: Optional[_BatchedExporter] = None
        if self.export_dir is not None:
            self._exporter = _BatchedExporter(
                self.export_dir,
                flush_every=int(export_flush_every),
                atomic=bool(export_atomic),
            )

    # Optional: let drivers flush at step boundaries
    def flush_exports(self) -> None:
        if self._exporter is not None:
            self._exporter.flush()

    def close_exports(self) -> None:
        if self._exporter is not None:
            self._exporter.close()

    # ───────────────────── public entrypoint ─────────────────────

    def repair(self, subcohort_ctx: Dict[str, Any], smt_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main entry point.

        Returns:
            (repaired_smt_text, metadata_dict)
        """
        # Reset last prompt/raw for this call
        self.last_prompt = None
        self.last_raw = None

        model = self.model_name or "unknown"

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

        def _cost_for(prompt: str, raw: str) -> Tuple[Optional[int], Optional[int], Optional[float], str, str]:
            pt, pnote = count_tokens(prompt or "", model)
            ct, cnote = count_tokens(raw or "", model)
            est = None
            if pt is not None and ct is not None:
                est = estimate_cost_usd(model, pt, ct)
            return pt, ct, est, str(pnote), str(cnote)

        # Aggregate attempt costs (best-effort)
        total_prompt_toks: Optional[int] = 0
        total_comp_toks: Optional[int] = 0
        total_cost: Optional[float] = 0.0
        tokenizer_note_prompt_last: str = ""
        tokenizer_note_completion_last: str = ""

        # Extract minimal repair inputs used in metadata and prompts
        try:
            ri = self._extract_repair_inputs(subcohort_ctx)
        except Exception as e:
            logging.error("[REPAIR] Failed to extract repair inputs: %s", e)
            meta: Dict[str, Any] = {
                "executed": False,
                "error": f"failed_to_extract_repair_inputs: {e}",
                "model": model,
                "pricing_used": _pricing_used(),
            }

            # NEW: still export a before/after pair (identical) if export_dir is set,
            # using a fallback naming scheme from whatever is available in ctx.
            if self.export_dir is not None:
                eff = str(
                    subcohort_ctx.get("trial_id_effective")
                    or subcohort_ctx.get("trial_id")
                    or subcohort_ctx.get("trial_id_parent")
                    or "UNKNOWN_TRIAL"
                )
                side = str(subcohort_ctx.get("inc_exc") or "unknown_side")
                cohort_id = str(subcohort_ctx.get("cohort_id") or subcohort_ctx.get("substudy_id") or "C?")
                ri_fallback = RepairInputs(
                    trial_id=eff,
                    effective_trial_id=eff,
                    side=side,
                    cohort_id=cohort_id,
                    cohort_label=str(subcohort_ctx.get("cohort_label") or "Unknown cohort"),
                    inclusion_criteria="",
                    exclusion_criteria="",
                    context="",
                    contextual_text="",
                    shared_context="",
                )
                self._export_smt_pair(ri_fallback, smt_text, smt_text, meta)

            return smt_text, meta

        # ───── Placeholder branch: no LLM, no modification ─────
        if self.call_llm is None:
            logging.info(
                "[REPAIR] Placeholder mode (no call_llm): SMT for eff_tid=%s side=%s cohort=%s left unchanged.",
                ri.effective_trial_id,
                ri.side,
                ri.cohort_id,
            )
            meta = {
                "executed": True,
                "reason": "placeholder repairer; SMT copied unchanged to ir_repaired and mbench",
                "trial_id": ri.trial_id,
                "effective_trial_id": ri.effective_trial_id,
                "side": ri.side,
                "cohort_id": ri.cohort_id,
                "cohort_label": ri.cohort_label,
                "changed_smt": False,
                "model": model,
                "pricing_used": _pricing_used(),
            }
            if self.enforce_strict_format:
                ok, issues = self._strict_format_check(smt_text)
                meta["format_ok"] = ok
                if not ok:
                    meta["format_issues"] = issues

            self._export_smt_pair(ri, smt_text, smt_text, meta)
            return smt_text, meta

        # ───── Real repair branch with retries ─────
        base_prompt = self._build_prompt(ri, smt_text)

        attempt_prompts: List[str] = []
        attempt_raws: List[str] = []
        attempt_errors: List[str] = []
        attempt_format_issues: List[List[str]] = []

        max_attempts = 1 + self.max_retries
        chosen_repaired: Optional[str] = None
        chosen_notes: str = ""
        chosen_no_mod: bool = False
        chosen_format_ok: Optional[bool] = None
        chosen_format_issues: List[str] = []

        def _append_attempt(prompt: str, raw: str) -> None:
            attempt_prompts.append(prompt)
            attempt_raws.append(raw)
            pt, ct, est, pnote, cnote = _cost_for(prompt, raw)

            nonlocal total_prompt_toks, total_comp_toks, total_cost, tokenizer_note_prompt_last, tokenizer_note_completion_last
            tokenizer_note_prompt_last = pnote
            tokenizer_note_completion_last = cnote

            if pt is None:
                total_prompt_toks = None
            elif total_prompt_toks is not None:
                total_prompt_toks += int(pt)

            if ct is None:
                total_comp_toks = None
            elif total_comp_toks is not None:
                total_comp_toks += int(ct)

            if est is None:
                total_cost = None
            elif total_cost is not None:
                total_cost += float(est)

        for attempt_idx in range(max_attempts):
            is_last_attempt = (attempt_idx == max_attempts - 1)

            prompt = base_prompt
            if attempt_idx > 0:
                last_err = attempt_errors[-1] if attempt_errors else "unknown_error"
                last_fmt = attempt_format_issues[-1] if attempt_format_issues else []
                prompt = self._make_retry_prompt(
                    base_prompt=base_prompt,
                    attempt_idx=attempt_idx,
                    last_error=last_err,
                    format_issues=last_fmt,
                )

            # Call LLM
            try:
                raw = self.call_llm(prompt)
            except Exception as e:
                err = f"llm_call_error: {e}"
                logging.error("[REPAIR] LLM call failed (attempt %d/%d): %s", attempt_idx + 1, max_attempts, e)
                attempt_errors.append(err)
                attempt_format_issues.append([])
                _append_attempt(prompt, f"<<EXCEPTION: {e}>>")
                continue

            _append_attempt(prompt, raw)

            # Parse JSON
            try:
                obj = self._parse_json(raw or "")
            except Exception as e:
                err = f"json_parse_error: {e}"
                logging.error("[REPAIR] JSON parsing failed (attempt %d/%d): %s", attempt_idx + 1, max_attempts, e)
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

            repaired_smt_raw = obj.get("repaired_smt")
            notes = obj.get("notes", "") or ""

            if not isinstance(repaired_smt_raw, str):
                err = "non_string_repaired_smt"
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

            sentinel_value = repaired_smt_raw.strip()
            sentinel_upper = sentinel_value.upper()

            # If model says NO MODIFICATION REQUIRED:
            if sentinel_upper in self.NO_MOD_SENTINELS:
                if self.enforce_strict_format:
                    ok, issues = self._strict_format_check(smt_text)
                    if not ok:
                        err = "no_mod_but_original_strict_format_invalid"
                        attempt_errors.append(err)
                        attempt_format_issues.append(list(issues))

                        if not is_last_attempt:
                            continue

                        # PASS THROUGH last attempt: return original even though strict-invalid
                        chosen_format_ok = False
                        chosen_format_issues = list(issues)
                        chosen_repaired = smt_text
                        chosen_notes = notes or (
                            "LLM returned NO MODIFICATION REQUIRED, but original failed strict format; "
                            "passing through last attempt."
                        )
                        chosen_no_mod = True
                        break

                    chosen_format_ok = True
                    chosen_format_issues = []

                chosen_repaired = smt_text
                chosen_notes = notes or "LLM indicated that no semantic repairs were required; original SMT retained."
                chosen_no_mod = True
                break

            # Else: model returned a full SMT
            repaired_smt = repaired_smt_raw
            if not repaired_smt.strip():
                err = "empty_or_invalid_repaired_smt"
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

            if self.enforce_strict_format:
                ok, issues = self._strict_format_check(repaired_smt)
                if not ok:
                    err = "strict_format_violation_in_repaired_smt"
                    attempt_errors.append(err)
                    attempt_format_issues.append(list(issues))

                    if not is_last_attempt:
                        continue

                    # PASS THROUGH last attempt: return repaired SMT even though strict-invalid
                    chosen_format_ok = False
                    chosen_format_issues = list(issues)
                    chosen_repaired = repaired_smt
                    chosen_notes = notes or (
                        "Repaired SMT failed strict format on final attempt; passing through last attempt."
                    )
                    chosen_no_mod = False
                    break

                chosen_format_ok = True
                chosen_format_issues = []

            chosen_repaired = repaired_smt
            chosen_notes = notes
            chosen_no_mod = False
            break

        # Join attempts for mbench visibility
        self.last_prompt = self._join_attempts("PROMPT", attempt_prompts)
        self.last_raw = self._join_attempts("RAW", attempt_raws)

        # If never got an acceptable output, fallback to original
        if chosen_repaired is None:
            meta = {
                "executed": True,
                "error": "all_attempts_failed_fallback_to_original",
                "attempts": max_attempts,
                "max_retries": self.max_retries,
                "attempt_errors": attempt_errors,
                "attempt_format_issues": attempt_format_issues,
                "trial_id": ri.trial_id,
                "effective_trial_id": ri.effective_trial_id,
                "side": ri.side,
                "cohort_id": ri.cohort_id,
                "cohort_label": ri.cohort_label,
                "changed_smt": False,
                "model": model,
                "prompt_tokens_est": total_prompt_toks,
                "completion_tokens_est": total_comp_toks,
                "estimated_cost_usd": total_cost,
                "tokenizer_note_prompt": tokenizer_note_prompt_last,
                "tokenizer_note_completion": tokenizer_note_completion_last,
                "pricing_used": _pricing_used(),
                "passed_through_last_attempt": False,
            }
            if self.enforce_strict_format:
                ok0, issues0 = self._strict_format_check(smt_text)
                meta["format_ok"] = ok0
                if not ok0:
                    meta["format_issues"] = issues0

            self._export_smt_pair(ri, smt_text, smt_text, meta)
            return smt_text, meta

        # Success path (NO MOD accepted, repaired accepted, or pass-through)
        repaired_out = chosen_repaired
        changed = (repaired_out.strip() != (smt_text or "").strip())
        pass_through = bool(self.enforce_strict_format and (chosen_format_ok is False))

        meta = {
            "executed": True,
            "notes": chosen_notes,
            "no_change_required": bool(chosen_no_mod),
            "attempts": len(attempt_prompts),
            "max_retries": self.max_retries,
            "attempt_errors": attempt_errors,
            "attempt_format_issues": attempt_format_issues,
            "trial_id": ri.trial_id,
            "effective_trial_id": ri.effective_trial_id,
            "side": ri.side,
            "cohort_id": ri.cohort_id,
            "cohort_label": ri.cohort_label,
            "changed_smt": bool(changed) and (not chosen_no_mod),
            "model": model,
            "prompt_tokens_est": total_prompt_toks,
            "completion_tokens_est": total_comp_toks,
            "estimated_cost_usd": total_cost,
            "tokenizer_note_prompt": tokenizer_note_prompt_last,
            "tokenizer_note_completion": tokenizer_note_completion_last,
            "pricing_used": _pricing_used(),
            "passed_through_last_attempt": pass_through,
        }

        if pass_through:
            meta["error"] = "retry_exhausted_pass_through_last_attempt"

        if self.enforce_strict_format:
            if chosen_format_ok is None:
                meta["format_ok"] = True
            else:
                meta["format_ok"] = bool(chosen_format_ok)
            if chosen_format_issues:
                meta["format_issues"] = chosen_format_issues

        self._export_smt_pair(ri, smt_text, repaired_out, meta)
        return repaired_out, meta

    # ───────────────────── context → RepairInputs ─────────────────────

    def _extract_repair_inputs(self, ctx: Dict[str, Any]) -> RepairInputs:
        trial_id = ctx.get("trial_id_parent") or ctx.get("parent_trial_id") or ctx.get("trial_id")
        eff_tid = ctx.get("trial_id_effective") or ctx.get("trial_id") or trial_id
        side = ctx.get("inc_exc", "inclusion")

        cohort_id = ctx.get("cohort_id") or ctx.get("substudy_id") or "C?"
        cohort_label = ctx.get("cohort_label") or ctx.get("substudy_label") or "Unknown cohort"

        inc = ctx.get("inclusion_criteria", "") or ""
        exc = ctx.get("exclusion_criteria", "") or ""
        ctx_str = ctx.get("context", "") or ctx.get("cohort_context_raw", "") or ""
        contextual_text = ctx.get("contextual_text", "") or ""
        shared_context = ctx.get("shared_context", "") or ""

        # Fallback to preprocessor_normalized if some pieces are missing
        pn = ctx.get("preprocessor_normalized") or {}
        if not shared_context:
            shared_context = pn.get("shared_context", "")

        if (not inc) or (not contextual_text) or (not ctx_str):
            enroll = pn.get("enrollment_cohorts") or []
            for co in enroll:
                if co.get("trial_id_effective") == eff_tid or co.get("id") == cohort_id:
                    inc = inc or (co.get("inclusion_criteria", "") or "")
                    exc = exc or (co.get("exclusion_criteria", "") or "")
                    ctx_str = ctx_str or (co.get("context", "") or "")
                    contextual_text = contextual_text or (co.get("contextual_text", "") or "")
                    break

        if not (inc or exc):
            raise ValueError(
                "No inclusion/exclusion criteria found for "
                f"trial_id={trial_id}, effective_trial_id={eff_tid}, cohort_id={cohort_id}"
            )

        return RepairInputs(
            trial_id=str(trial_id),
            effective_trial_id=str(eff_tid),
            side=str(side),
            cohort_id=str(cohort_id),
            cohort_label=str(cohort_label),
            inclusion_criteria=str(inc),
            exclusion_criteria=str(exc),
            context=str(ctx_str),
            contextual_text=str(contextual_text or shared_context),
            shared_context=str(shared_context),
        )

    # ───────────────────── prompt helpers ─────────────────────

    def _make_retry_prompt(
        self,
        *,
        base_prompt: str,
        attempt_idx: int,
        last_error: str,
        format_issues: List[str],
    ) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append(f"# === RETRY ATTEMPT {attempt_idx + 1} ===")
        lines.append("Your previous output was rejected by an automated validator.")
        lines.append(f"Rejection reason: {last_error}")
        if format_issues:
            lines.append("Validator issues (fix ALL):")
            for it in format_issues[:200]:
                lines.append(f"- {it}")
        lines.append("")
        lines.append("You MUST follow the strict output JSON schema exactly.")
        lines.append("If you previously returned NO MODIFICATION REQUIRED but formatting was invalid,")
        lines.append('you MUST return the FULL corrected SMT (not "NO MODIFICATION REQUIRED").')
        return base_prompt + "\n".join(lines)

    def _join_attempts(self, kind: str, items: List[str]) -> Optional[str]:
        if not items:
            return None
        parts: List[str] = []
        for i, txt in enumerate(items, start=1):
            parts.append(f"==== ATTEMPT {i} {kind} ====")
            parts.append(txt if txt is not None else "")
        return "\n\n".join(parts)

    def _build_prompt(self, ri: RepairInputs, smt_text: str) -> str:
        if self.prompt_template is not None:
            t = self.prompt_template
            t = t.replace("{{TRIAL_ID}}", ri.trial_id)
            t = t.replace("{{EFFECTIVE_TRIAL_ID}}", ri.effective_trial_id)
            t = t.replace("{{SIDE}}", ri.side)
            t = t.replace("{{COHORT_ID}}", ri.cohort_id)
            t = t.replace("{{COHORT_LABEL}}", ri.cohort_label)
            t = t.replace("{{CONTEXT}}", ri.context)
            t = t.replace("{{SHARED_CONTEXT}}", ri.shared_context)
            t = t.replace("{{CONTEXTUAL_TEXT}}", ri.contextual_text)
            t = t.replace("{{INCLUSION_CRITERIA}}", ri.inclusion_criteria or "(none provided)")
            t = t.replace("{{EXCLUSION_CRITERIA}}", ri.exclusion_criteria or "(none provided)")
            t = t.replace("{{SMT_PROGRAM}}", smt_text)
            return t

        return f"""
You are an expert in clinical trial eligibility criteria and SMT-LIB v2.

You are repairing the SMT encoding for a single subcohort of a trial.

Identify:
  - Trial ID: {ri.trial_id}
  - Effective subcohort trial ID: {ri.effective_trial_id}
  - Side: {ri.side}
  - Cohort: {ri.cohort_id} — {ri.cohort_label}

SHORT CONTEXT:
{ri.context}

SHARED CONTEXT (trial-level):
{ri.shared_context}

FULL CONTEXTUAL TEXT:
{ri.contextual_text}

Eligibility criteria for this side:

INCLUSION CRITERIA:
{ri.inclusion_criteria or "(none provided)"}

EXCLUSION CRITERIA:
{ri.exclusion_criteria or "(none provided)"}

CURRENT SMT PROGRAM (SMT-LIB v2):
{smt_text}

Your task:

  1. Check whether the SMT program is consistent with the above eligibility
     description for this side.
  2. If there are inconsistencies, missing constraints, or syntax problems,
     repair the SMT program with minimal edits.
  3. Be careful not to make the SMT encoding stricter than the natural-language
     intent in a way that would wrongly exclude eligible patients.

Output format (IMPORTANT):

  Return ONLY a single JSON object, no extra commentary, with keys:

    {{
      "repaired_smt": "<FULL SMT-LIB v2 program for this subcohort and side> "
                      "OR the exact string \\"NO MODIFICATION REQUIRED\\" if no changes are needed",
      "notes": "<short explanation of what you changed and why, or why no changes are needed>"
    }}

  - Normally, "repaired_smt" must contain the full SMT-LIB v2 program.
  - Exception: if after careful review you conclude that the current SMT program
    already matches the criteria and needs no semantic repair, set
    "repaired_smt": "NO MODIFICATION REQUIRED" and do NOT include the SMT code.
  - Do NOT wrap the SMT code or JSON in backticks.
  - Do NOT include any other keys or text.
""".strip()

    # ───────────────────── strict format enforcement ─────────────────────

    def _strict_format_check(self, smt_text: str) -> Tuple[bool, List[str]]:
        """
        Enforces *structural* constraints (variable definitions):
          - forbid declare-fun/define-fun
          - allow only declare-const with sort Bool/Int/Real
          - each declare-const line must include a same-line JSON comment with required keys

        Returns: (ok, issues)
        """
        issues: List[str] = []
        s = smt_text or ""

        lowered = s.lower()
        for tok in self._FORBIDDEN_DECL_TOKENS:
            if tok in lowered:
                issues.append(f"forbidden_token_present:{tok}")

        for ln_no, line in enumerate(s.splitlines(), start=1):
            if "(declare-const" not in line:
                continue

            m = self._DECL_CONST_LINE_RE.match(line)
            if not m:
                issues.append(f"bad_declare_const_format:line{ln_no}")
                continue

            sort = m.group(2)
            if sort not in ("Bool", "Int", "Real"):
                issues.append(f"forbidden_sort:{sort}:line{ln_no}")
                continue

            obj, err = self._extract_decl_json(line)
            if err is not None:
                issues.append(f"{err}:line{ln_no}")
                continue

            if sort == "Bool":
                required = {"when_to_set_to_true", "when_to_set_to_false", "when_to_set_to_null", "meaning"}
            else:
                required = {"when_to_set_to_value", "when_to_set_to_null", "meaning"}

            missing = [k for k in sorted(required) if k not in obj]
            if missing:
                issues.append(f"missing_json_keys:{','.join(missing)}:line{ln_no}")
                continue

            for k in required:
                if not isinstance(obj.get(k), str):
                    issues.append(f"json_value_not_string:{k}:line{ln_no}")
                    break

        return (len(issues) == 0), issues

    def _extract_decl_json(self, line: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Extract first JSON object from the comment portion of a declare-const line.

        Expected pattern (flexible about extra comments):
            (declare-const ...) ;; { ... } ...
        """
        cpos = line.find(";;")
        if cpos == -1:
            return {}, "missing_comment_for_declare_const"
        comment = line[cpos + 2 :]

        l = comment.find("{")
        r = comment.rfind("}")
        if l == -1 or r == -1 or r < l:
            return {}, "missing_json_object_on_declare_const_line"

        jtxt = comment[l : r + 1].strip()
        try:
            obj = json.loads(jtxt)
        except Exception as e:
            return {}, f"invalid_json_object:{e}"

        if not isinstance(obj, dict):
            return {}, "json_comment_not_object"
        return obj, None

    # ───────────────────── exporting ─────────────────────

    def _export_smt_pair(
        self,
        ri: RepairInputs,
        original_smt: str,
        repaired_smt: str,
        meta: Dict[str, Any],
    ) -> None:
        if self.export_dir is None:
            return

        def _safe_component(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(s))

        prefix = "__".join(_safe_component(x) for x in (ri.effective_trial_id, ri.side, ri.cohort_id))
        before_path = self.export_dir / f"{prefix}.before.smt2"
        after_path = self.export_dir / f"{prefix}.after.smt2"

        # Queue writes (batched)
        if self._exporter is not None:
            self._exporter.enqueue_write(before_path, original_smt)
            self._exporter.enqueue_write(after_path, repaired_smt)
        else:
            # fallback: best-effort direct write
            try:
                self.export_dir.mkdir(parents=True, exist_ok=True)
                before_path.write_text(original_smt, encoding="utf-8")
                after_path.write_text(repaired_smt, encoding="utf-8")
            except Exception as e:
                logging.error("[REPAIR] Failed to write SMT export files: %s", e)
                meta.setdefault("export_error", f"failed_to_write_smt_exports: {e}")
                return

        meta["export_dir"] = str(self.export_dir)
        meta["mbench_smt_before_path"] = str(before_path)
        meta["mbench_smt_after_path"] = str(after_path)
        meta["mbench_smt_before_len"] = len(original_smt)
        meta["mbench_smt_after_len"] = len(repaired_smt)

    # ───────────────────── JSON helpers ─────────────────────

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
