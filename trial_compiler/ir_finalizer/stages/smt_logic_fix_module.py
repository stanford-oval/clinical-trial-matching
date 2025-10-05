#!/usr/bin/env python3
"""
smt_logic_fix_module.py

Core module to fix logical/semantic encoding issues in SMT programs (inclusion OR exclusion),
using context + BOTH NL inclusion/exclusion criteria.

Key points:
- Accepts `side` per call and plugs it into {{SIDE}}.
- Includes BOTH inclusion_criteria and exclusion_criteria in the prompt for context.
- Best-effort token + cost estimation via costing.py (tiktoken).
- Does NOT hard-fail if inclusion/exclusion criteria are missing; uses "(none provided)" and sets meta flags.

Strict format enforcement (default ON) + retries (default 2):
    * Forbid (declare-fun ...) and (define-fun ...)
    * Allow only (declare-const <sym> Bool|Int|Real)
    * Each (declare-const ...) must have a SAME-LINE JSON comment with required keys
Retry up to max_retries times (total attempts = 1 + max_retries) when:
    * LLM call error
    * JSON parse error / malformed output
    * non-string or empty repaired_smt
    * "NO MODIFICATION REQUIRED" returned but ORIGINAL SMT fails strict format (when enforced)
    * repaired SMT fails strict format (when enforced)
last_prompt/last_raw contain ALL attempts separated by headers for mbench inspection.

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

# costing.py is expected; keep a safe fallback so this module never hard-crashes.
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
class LogicFixInputs:
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
    missing_inclusion_criteria: bool = False
    missing_exclusion_criteria: bool = False


class SMTLogicFixer:
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

        # For mbench logging
        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

    # ───────────────────── public entrypoint ─────────────────────

    def fix(self, subcohort_ctx: Dict[str, Any], side: str, smt_text: str) -> Tuple[str, Dict[str, Any]]:
        self.last_prompt = None
        self.last_raw = None

        if side not in ("inclusion", "exclusion"):
            return smt_text, {"executed": False, "error": f"invalid_side:{side}"}

        try:
            li = self._extract_inputs(subcohort_ctx, side)
        except Exception as e:
            logging.error("[LOGIC] Failed to extract inputs: %s", e)
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
                "reason": "placeholder logic fixer; SMT copied unchanged",
                "trial_id": li.trial_id,
                "effective_trial_id": li.effective_trial_id,
                "side": li.side,
                "cohort_id": li.cohort_id,
                "cohort_label": li.cohort_label,
                "changed_smt": False,
                "missing_inclusion_criteria": li.missing_inclusion_criteria,
                "missing_exclusion_criteria": li.missing_exclusion_criteria,
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
        base_prompt = self._build_prompt(li, smt_text)

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
                logging.error("[LOGIC] LLM call failed (attempt %d/%d): %s", attempt_idx + 1, max_attempts, e)
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
                logging.error("[LOGIC] JSON parsing failed (attempt %d/%d): %s", attempt_idx + 1, max_attempts, e)
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

            # NO MOD path
            if repaired_stripped.upper() in self.NO_MOD_SENTINELS:
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
                chosen_notes = notes or "No semantic drift found; SMT retained."
                chosen_no_mod = True
                break

            # Full SMT path
            if not repaired_stripped:
                err = "empty_repaired_smt"
                attempt_errors.append(err)
                attempt_format_issues.append([])
                continue

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
                "executed": True,
                "error": "all_attempts_failed_fallback_to_original",
                "attempts": max_attempts,
                "max_retries": self.max_retries,
                "attempt_errors": attempt_errors,
                "attempt_format_issues": attempt_format_issues,
                "trial_id": li.trial_id,
                "effective_trial_id": li.effective_trial_id,
                "side": li.side,
                "cohort_id": li.cohort_id,
                "cohort_label": li.cohort_label,
                "changed_smt": False,
                "missing_inclusion_criteria": li.missing_inclusion_criteria,
                "missing_exclusion_criteria": li.missing_exclusion_criteria,
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
            "trial_id": li.trial_id,
            "effective_trial_id": li.effective_trial_id,
            "side": li.side,
            "cohort_id": li.cohort_id,
            "cohort_label": li.cohort_label,
            "changed_smt": bool(changed),
            "missing_inclusion_criteria": li.missing_inclusion_criteria,
            "missing_exclusion_criteria": li.missing_exclusion_criteria,
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

    # ───────────────────── ctx → inputs ─────────────────────

    def _extract_inputs(self, ctx: Dict[str, Any], side: str) -> LogicFixInputs:
        trial_id = ctx.get("trial_id_parent") or ctx.get("parent_trial_id") or ctx.get("trial_id")
        eff_tid = ctx.get("trial_id_effective") or ctx.get("trial_id") or trial_id

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

        # Fallback to enrollment_cohorts (same spirit as SMTRepairer)
        if (not inc) or (not exc) or (not contextual_text) or (not ctx_str):
            enroll = pn.get("enrollment_cohorts") or []
            for co in enroll:
                if co.get("trial_id_effective") == eff_tid or co.get("id") == cohort_id:
                    inc = inc or (co.get("inclusion_criteria", "") or "")
                    exc = exc or (co.get("exclusion_criteria", "") or "")
                    ctx_str = ctx_str or (co.get("context", "") or "")
                    contextual_text = contextual_text or (co.get("contextual_text", "") or "")
                    break

        missing_inc = False
        missing_exc = False
        if not inc:
            missing_inc = True
            inc = "(none provided)"
        if not exc:
            missing_exc = True
            exc = "(none provided)"

        if not contextual_text:
            contextual_text = shared_context or ""

        return LogicFixInputs(
            trial_id=str(trial_id),
            effective_trial_id=str(eff_tid),
            side=str(side),
            cohort_id=str(cohort_id),
            cohort_label=str(cohort_label),
            inclusion_criteria=str(inc),
            exclusion_criteria=str(exc),
            context=str(ctx_str),
            contextual_text=str(contextual_text),
            shared_context=str(shared_context),
            missing_inclusion_criteria=missing_inc,
            missing_exclusion_criteria=missing_exc,
        )

    # ───────────────────── retry prompt helper ─────────────────────

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

    def _build_prompt(self, li: LogicFixInputs, smt_text: str) -> str:
        if self.prompt_template is not None:
            t = self.prompt_template
            t = t.replace("{{TRIAL_ID}}", li.trial_id)
            t = t.replace("{{EFFECTIVE_TRIAL_ID}}", li.effective_trial_id)
            t = t.replace("{{SIDE}}", li.side)
            t = t.replace("{{COHORT_ID}}", li.cohort_id)
            t = t.replace("{{COHORT_LABEL}}", li.cohort_label)
            t = t.replace("{{CONTEXT}}", li.context)
            t = t.replace("{{SHARED_CONTEXT}}", li.shared_context)
            t = t.replace("{{CONTEXTUAL_TEXT}}", li.contextual_text)
            t = t.replace("{{INCLUSION_CRITERIA}}", li.inclusion_criteria or "(none provided)")
            t = t.replace("{{EXCLUSION_CRITERIA}}", li.exclusion_criteria or "(none provided)")
            t = t.replace("{{SMT_PROGRAM}}", smt_text)
            return t

        # Built-in fallback prompt (includes strict formatting gate)
        return f"""
# === ROLE ===
You are an expert bioinformatician and logic expert. Your task is to fix logical encoding issues in the SMT version of natural language eligibility criteria, using context to avoid incorrect scoping/qualification.

You are given the program representing {{SIDE}} criteria in this case. Focus ONLY on {{SIDE}} constraints; the other side is handled elsewhere.

# === CONTEXT ===
<shared_context>
{li.shared_context}
</shared_context>

<subcohort_context>
{li.context}
</subcohort_context>

<contextual_text>
{li.contextual_text}
</contextual_text>

# === ELIGIBILITY CRITERIA FOR THIS SUBCOHORT ===
<inclusion_criteria>
{li.inclusion_criteria}
</inclusion_criteria>

<exclusion_criteria>
{li.exclusion_criteria}
</exclusion_criteria>

# === CURRENT SMT-LIB PROGRAM ===
<current_smt_program>
{smt_text}
</current_smt_program>

# === GUIDELINES ===
1. Do minimal fixes. Prefer the smallest set of edits that makes the SMT program faithful to the clinical intent.
2. Reuse existing symbols as much as possible. Treat existing symbols as stems and prefer adding qualifiers of the form <stem>@@<qualifier>.
   If both <stem> and <stem>@@<qualifier> are Bool, add:
     (assert (=> <stem>@@<qualifier> <stem>))
3. Preserve structure: keep existing assertion tags, :named labels, and comment style (e.g., ;; comments, REQ* labels).
4. If no semantic drift/logic bug exists AND the SMT already satisfies STRICT FORMAT & ANNOTATION REQUIREMENTS below,
   return "NO MODIFICATION REQUIRED" and do not echo SMT.
5. When you DO modify, return the entire updated SMT-LIB v2 program in "repaired_smt" (not a diff), and ensure it is a valid SMT-LIB v2 script.

# === STRICT FORMAT & ANNOTATION REQUIREMENTS (MUST MATCH EXACTLY) ===
If ANY of these are violated, you MUST output a full corrected SMT (do NOT return "NO MODIFICATION REQUIRED").

A) Allowed declaration forms / types (HARD)
- You MUST only use:
    (declare-const <sym> Bool)
    (declare-const <sym> Int)
    (declare-const <sym> Real)
- Do NOT introduce declare-fun, define-fun, Strings, enums, Datatypes, Arrays, or uninterpreted sorts.

B) Variable declaration annotation (HARD)
- EVERY (declare-const ...) line MUST have a single JSON object comment on the SAME LINE.
- If an existing declaration already has a JSON object comment, keep it verbatim (do NOT rewrite it).
- JSON must be valid (double quotes) and based ONLY on variable meaning.

Required JSON keys (exact):
- For Bool:
  {{"when_to_set_to_true":"...","when_to_set_to_false":"...","when_to_set_to_null":"...","meaning":"..."}}
- For Int/Real:
  {{"when_to_set_to_value":"...","when_to_set_to_null":"...","meaning":"..."}}

# === OUTPUT FORMAT (STRICT) ===
Return ONLY a single JSON object, with no extra text, no Markdown fences, no comments:

{{
  "repaired_smt": "<FULL SMT-LIB v2 program> OR the exact string \\"NO MODIFICATION REQUIRED\\"",
  "notes": "<short explanation (1–3 sentences or bullet points) of what you changed and why, or why no changes are needed>"
}}
""".replace("{{SIDE}}", li.side).strip()

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
