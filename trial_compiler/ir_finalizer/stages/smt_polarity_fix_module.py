#!/usr/bin/env python3
"""
smt_polarity_fix_module.py

Core module to fix polarity flips in *exclusion* SMT programs.

Key behaviors:
- Mirrors SMTRepairer-style extraction logic and DOES NOT hard-fail if
  exclusion_criteria is missing. Instead:
    - sets exclusion_criteria to "(none provided)"
    - returns meta["missing_exclusion_criteria"]=True

Cost tracking:
- Best-effort token + USD cost estimation using costing.py + tiktoken (if available).
- Adds:
    model, prompt_tokens_est, completion_tokens_est,
    tokenizer_note_prompt, tokenizer_note_completion,
    estimated_cost_usd, pricing_used

Strict format enforcement (optional; default ON):
- Enforces variable declaration format constraints (same spirit as repair module):
    * Forbid (declare-fun ...) and (define-fun ...)
    * Allow only (declare-const <sym> Bool|Int|Real)
    * Each (declare-const ...) line must include a same-line JSON object comment
      with required keys (Bool vs numeric).
- If LLM output violates strict format, the module retries (up to 2 retries by default).

Retry mechanism:
- Retries up to max_retries times (default 2) when:
    * LLM call error
    * JSON parse error
    * malformed "repaired_smt"
    * empty repaired_smt
    * "NO MODIFICATION REQUIRED" returned but ORIGINAL SMT fails strict format (when enforced)
    * repaired SMT fails strict format (when enforced)
- last_prompt / last_raw include ALL attempts separated by headers.

NEW POLICY (per request):
- If strict-format failures persist until retries are exhausted, LET THE LAST ATTEMPT PASS THROUGH
  (upstream caller will take care of it):
    * If last attempt returns NO MOD but original fails strict format -> return original anyway.
    * If last attempt returns repaired SMT but it fails strict format -> return repaired anyway.
  In these pass-through cases, meta will include:
    passed_through_last_attempt=True, format_ok=False, format_issues=[...],
    error="retry_exhausted_pass_through_last_attempt"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# costing.py is expected (used by other stage modules too); keep a safe fallback.
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
class PolarityFixInputs:
    trial_id: str
    effective_trial_id: str
    side: str  # always "exclusion"
    cohort_id: str
    cohort_label: str
    exclusion_criteria: str
    context: str
    contextual_text: str
    shared_context: str = ""
    missing_exclusion_criteria: bool = False


class SMTPolarityFixer:
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
        model_name: Optional[str] = None,
        *,
        enforce_strict_format: bool = True,
        max_retries: int = 1,
    ) -> None:
        self.call_llm = call_llm
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.enforce_strict_format = bool(enforce_strict_format)
        self.max_retries = max(0, int(max_retries))

        # For mbench logging: last prompt and raw output per fix() call
        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

    # ───────────────────── public entrypoint ─────────────────────

    def fix(self, subcohort_ctx: Dict[str, Any], smt_text: str) -> Tuple[str, Dict[str, Any]]:
        self.last_prompt = None
        self.last_raw = None

        try:
            pi = self._extract_inputs(subcohort_ctx)
        except Exception as e:
            logging.error("[POLARITY] Failed to extract inputs: %s", e)
            return smt_text, {"executed": False, "error": f"failed_to_extract_inputs: {e}"}

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

        # Placeholder mode
        if self.call_llm is None:
            meta: Dict[str, Any] = {
                "executed": True,
                "reason": "placeholder polarity fixer; SMT copied unchanged",
                "trial_id": pi.trial_id,
                "effective_trial_id": pi.effective_trial_id,
                "side": pi.side,
                "cohort_id": pi.cohort_id,
                "cohort_label": pi.cohort_label,
                "changed_smt": False,
                "missing_exclusion_criteria": pi.missing_exclusion_criteria,
                "model": model,
                "pricing_used": _pricing_used(),
                "passed_through_last_attempt": False,
            }
            if self.enforce_strict_format:
                ok0, issues0 = self._strict_format_check(smt_text)
                meta["format_ok"] = ok0
                if not ok0:
                    meta["format_issues"] = issues0
            return smt_text, meta

        # Real mode with retries
        base_prompt = self._build_prompt(pi, smt_text)

        attempt_prompts: List[str] = []
        attempt_raws: List[str] = []
        attempt_errors: List[str] = []
        attempt_format_issues: List[List[str]] = []

        # Aggregate attempt costs (best-effort)
        total_prompt_toks: Optional[int] = 0
        total_comp_toks: Optional[int] = 0
        total_cost: Optional[float] = 0.0
        tokenizer_note_prompt_last: str = ""
        tokenizer_note_completion_last: str = ""

        def _append_attempt(prompt: str, raw: str) -> None:
            attempt_prompts.append(prompt)
            attempt_raws.append(raw)

            nonlocal total_prompt_toks, total_comp_toks, total_cost, tokenizer_note_prompt_last, tokenizer_note_completion_last
            pt, ct, est, pnote, cnote = _cost_for(prompt, raw)
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

        max_attempts = 1 + self.max_retries
        chosen_smt: Optional[str] = None
        chosen_notes: str = ""
        chosen_no_mod: bool = False
        chosen_format_ok: Optional[bool] = None
        chosen_format_issues: List[str] = []

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
                logging.error("[POLARITY] LLM call failed (attempt %d/%d): %s", attempt_idx + 1, max_attempts, e)
                attempt_errors.append(err)
                attempt_format_issues.append([])
                _append_attempt(prompt, f"<<EXCEPTION: {e}>>")
                continue

            _append_attempt(prompt, raw)

            # Parse JSON
            try:
                obj = self._parse_json(raw)
            except Exception as e:
                err = f"json_parse_error: {e}"
                logging.error("[POLARITY] JSON parsing failed (attempt %d/%d): %s", attempt_idx + 1, max_attempts, e)
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

            repaired = obj.get("repaired_smt")
            notes = obj.get("notes", "") or ""

            if not isinstance(repaired, str):
                err = "non_string_repaired_smt"
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

            repaired_stripped = repaired.strip()
            if repaired_stripped.upper() in self.NO_MOD_SENTINELS:
                # Accept only if original passes strict format when enforcement is ON,
                # unless we are on the last attempt (pass-through policy).
                if self.enforce_strict_format:
                    ok0, issues0 = self._strict_format_check(smt_text)
                    if not ok0:
                        err = "no_mod_but_original_strict_format_invalid"
                        attempt_errors.append(err)
                        attempt_format_issues.append(list(issues0))

                        if not is_last_attempt:
                            continue

                        # PASS THROUGH last attempt: return original even though strict-invalid
                        chosen_format_ok = False
                        chosen_format_issues = list(issues0)
                        chosen_smt = smt_text
                        chosen_notes = notes or (
                            "LLM returned NO MODIFICATION REQUIRED, but original failed strict format; "
                            "passing through last attempt."
                        )
                        chosen_no_mod = True
                        break

                    chosen_format_ok = True
                    chosen_format_issues = []

                chosen_smt = smt_text
                chosen_notes = notes or "No polarity flips detected; original SMT retained."
                chosen_no_mod = True
                break

            if not repaired_stripped:
                err = "empty_repaired_smt"
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

            # Strict format gate for the repaired SMT (with last-attempt pass-through policy)
            if self.enforce_strict_format:
                ok, issues = self._strict_format_check(repaired)
                if not ok:
                    err = "strict_format_violation_in_repaired_smt"
                    attempt_errors.append(err)
                    attempt_format_issues.append(list(issues))

                    if not is_last_attempt:
                        continue

                    # PASS THROUGH last attempt: return repaired even though strict-invalid
                    chosen_format_ok = False
                    chosen_format_issues = list(issues)
                    chosen_smt = repaired
                    chosen_notes = notes or (
                        "Repaired SMT failed strict format on final attempt; passing through last attempt."
                    )
                    chosen_no_mod = False
                    break

                chosen_format_ok = True
                chosen_format_issues = []

            chosen_smt = repaired
            chosen_notes = notes
            chosen_no_mod = False
            break

        # For mbench/debugging: include all attempts
        self.last_prompt = self._join_attempts("PROMPT", attempt_prompts)
        self.last_raw = self._join_attempts("RAW", attempt_raws)

        # Fallback if all attempts failed (call/json errors, etc.)
        if chosen_smt is None:
            meta: Dict[str, Any] = {
                "executed": True,  # keep pipeline moving; output original SMT
                "error": "all_attempts_failed_fallback_to_original",
                "attempts": max_attempts,
                "max_retries": self.max_retries,
                "attempt_errors": attempt_errors,
                "attempt_format_issues": attempt_format_issues,
                "trial_id": pi.trial_id,
                "effective_trial_id": pi.effective_trial_id,
                "side": pi.side,
                "cohort_id": pi.cohort_id,
                "cohort_label": pi.cohort_label,
                "changed_smt": False,
                "missing_exclusion_criteria": pi.missing_exclusion_criteria,
                "model": model,
                "prompt_tokens_est": total_prompt_toks,
                "completion_tokens_est": total_comp_toks,
                "tokenizer_note_prompt": tokenizer_note_prompt_last,
                "tokenizer_note_completion": tokenizer_note_completion_last,
                "estimated_cost_usd": total_cost,
                "pricing_used": _pricing_used(),
                "passed_through_last_attempt": False,
            }
            if self.enforce_strict_format:
                ok0, issues0 = self._strict_format_check(smt_text)
                meta["format_ok"] = ok0
                if not ok0:
                    meta["format_issues"] = issues0
            return smt_text, meta

        pass_through = bool(self.enforce_strict_format and (chosen_format_ok is False))

        # Success
        changed = (chosen_smt.strip() != (smt_text or "").strip()) and (not chosen_no_mod)
        meta: Dict[str, Any] = {
            "executed": True,
            "notes": chosen_notes,
            "no_change_required": bool(chosen_no_mod),
            "attempts": len(attempt_prompts),
            "max_retries": self.max_retries,
            "attempt_errors": attempt_errors,
            "attempt_format_issues": attempt_format_issues,
            "trial_id": pi.trial_id,
            "effective_trial_id": pi.effective_trial_id,
            "side": pi.side,
            "cohort_id": pi.cohort_id,
            "cohort_label": pi.cohort_label,
            "changed_smt": bool(changed),
            "missing_exclusion_criteria": pi.missing_exclusion_criteria,
            "model": model,
            "prompt_tokens_est": total_prompt_toks,
            "completion_tokens_est": total_comp_toks,
            "tokenizer_note_prompt": tokenizer_note_prompt_last,
            "tokenizer_note_completion": tokenizer_note_completion_last,
            "estimated_cost_usd": total_cost,
            "pricing_used": _pricing_used(),
            "passed_through_last_attempt": pass_through,
        }

        if self.enforce_strict_format:
            if chosen_format_ok is None:
                meta["format_ok"] = True
            else:
                meta["format_ok"] = bool(chosen_format_ok)
            if chosen_format_issues:
                meta["format_issues"] = chosen_format_issues

        if pass_through:
            meta["error"] = "retry_exhausted_pass_through_last_attempt"

        return chosen_smt, meta

    # ───────────────────── ctx → inputs (mirrors SMTRepairer style) ─────────────────────

    def _extract_inputs(self, ctx: Dict[str, Any]) -> PolarityFixInputs:
        trial_id = ctx.get("trial_id_parent") or ctx.get("parent_trial_id") or ctx.get("trial_id")
        eff_tid = ctx.get("trial_id_effective") or ctx.get("trial_id") or trial_id
        side = "exclusion"

        cohort_id = ctx.get("cohort_id") or ctx.get("substudy_id") or "C?"
        cohort_label = ctx.get("cohort_label") or ctx.get("substudy_label") or "Unknown cohort"

        exc = ctx.get("exclusion_criteria", "") or ""
        ctx_str = ctx.get("context", "") or ctx.get("cohort_context_raw", "") or ""
        contextual_text = ctx.get("contextual_text", "") or ""
        shared_context = ctx.get("shared_context", "") or ""

        pn = ctx.get("preprocessor_normalized") or {}
        if not shared_context:
            shared_context = pn.get("shared_context", "") or ""

        # Fallback to enrollment_cohorts (same spirit as SMTRepairer)
        if (not exc) or (not contextual_text) or (not ctx_str):
            enroll = pn.get("enrollment_cohorts") or []
            for co in enroll:
                if co.get("trial_id_effective") == eff_tid or co.get("id") == cohort_id:
                    exc = exc or (co.get("exclusion_criteria", "") or "")
                    ctx_str = ctx_str or (co.get("context", "") or "")
                    contextual_text = contextual_text or (co.get("contextual_text", "") or "")
                    break

        missing_exc = False
        if not exc:
            # Do NOT fail; polarity can still be fixed from SMT alone.
            missing_exc = True
            exc = "(none provided)"

        # If contextual_text still missing, fall back to shared_context (same as SMTRepairer)
        if not contextual_text:
            contextual_text = shared_context or ""

        return PolarityFixInputs(
            trial_id=str(trial_id),
            effective_trial_id=str(eff_tid),
            side=side,
            cohort_id=str(cohort_id),
            cohort_label=str(cohort_label),
            exclusion_criteria=str(exc),
            context=str(ctx_str),
            contextual_text=str(contextual_text),
            shared_context=str(shared_context),
            missing_exclusion_criteria=missing_exc,
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

            # Required keys (do NOT require absence of extra keys)
            if sort == "Bool":
                required = {"when_to_set_to_true", "when_to_set_to_false", "when_to_set_to_null", "meaning"}
            else:
                required = {"when_to_set_to_value", "when_to_set_to_null", "meaning"}

            missing = [k for k in sorted(required) if k not in obj]
            if missing:
                issues.append(f"missing_json_keys:{','.join(missing)}:line{ln_no}")
                continue

            # Optional sanity: required keys must be strings
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

    # ───────────────────── prompt & JSON ─────────────────────

    def _build_prompt(self, pi: PolarityFixInputs, smt_text: str) -> str:
        if self.prompt_template is not None:
            t = self.prompt_template
            t = t.replace("{{TRIAL_ID}}", pi.trial_id)
            t = t.replace("{{EFFECTIVE_TRIAL_ID}}", pi.effective_trial_id)
            t = t.replace("{{SIDE}}", pi.side)
            t = t.replace("{{COHORT_ID}}", pi.cohort_id)
            t = t.replace("{{COHORT_LABEL}}", pi.cohort_label)
            t = t.replace("{{CONTEXT}}", pi.context)
            t = t.replace("{{SHARED_CONTEXT}}", pi.shared_context)
            t = t.replace("{{CONTEXTUAL_TEXT}}", pi.contextual_text)
            t = t.replace("{{EXCLUSION_CRITERIA}}", pi.exclusion_criteria or "(none provided)")
            t = t.replace("{{SMT_PROGRAM}}", smt_text)
            return t

        return f"""
# === ROLE ===
You are an expert bioinformatician and logic expert. Your task is to fix the polarities of assertions in the SMT version of exclusion criteria.

# === CONTEXT ===
<shared_context>
{pi.shared_context}
</shared_context>

<subcohort_context>
{pi.context}
</subcohort_context>

<contextual_text>
{pi.contextual_text}
</contextual_text>

# === ELIGIBILITY CRITERIA FOR THIS SUBCOHORT & SIDE ===
<exclusion_criteria>
{pi.exclusion_criteria}
</exclusion_criteria>

# === CURRENT SMT-LIB PROGRAM ===
<current_smt_program>
{smt_text}
</current_smt_program>

# === POLARITY SPECIFICATION ===
The patient is ELIGIBLE iff SMT programs of both inclusion constraints and exclusion constraints are SAT.
Exclusion constraints are SAT under the "NOT excluded" encoding used here.
Some criteria may be encoded backwards such that SAT corresponds to the patient being excluded.
Fix only the criteria that are flipped; others may be correct.

# === GUIDELINES ===
1. Do minimal fixes. Smallest edits to correct polarity flips.
2. Preserve structure: keep :named labels and comment style.
3. Reuse existing symbols; prefer qualifiers <stem>@@<qualifier> if needed.
4. If no polarity flips exist, return "NO MODIFICATION REQUIRED" and do not echo SMT.

# === OUTPUT FORMAT (STRICT) ===
Return ONLY a JSON object:

{{
  "repaired_smt": "<FULL SMT-LIB v2 program> OR the exact string \\"NO MODIFICATION REQUIRED\\"",
  "notes": "<short explanation>"
}}
""".strip()

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

        # Handle fenced blocks
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            if lines and lines[0].strip().lower() == "json":
                lines = lines[1:]
            text = "\n".join(lines).strip()

        # Try direct parse
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("parsed JSON is not an object")
            return obj
        except Exception:
            pass

        # Try extracting the first {...} region
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            snippet = text[l : r + 1]
            obj = json.loads(snippet)
            if not isinstance(obj, dict):
                raise ValueError("parsed JSON is not an object")
            return obj

        raise ValueError("could not parse JSON from LLM output")
