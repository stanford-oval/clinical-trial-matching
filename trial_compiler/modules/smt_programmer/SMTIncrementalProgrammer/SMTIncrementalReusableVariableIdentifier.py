# modules/SMTIncrementalReusableVariableIdentifier.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json

import dspy
from .namer_checks import (
    _REUSABLE_RE,
    _parse_json_array_relaxed,
    _validate_reusable_vars,
    _extract_declared_symbols,
    _schema_log,
)

try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}


class SMTIncrementalReusableVariableIdentifier(dspy.Module):
    """
    Stage-0: Identify variables that ALREADY exist in the SMT program and are
    semantically reusable for ONE eligibility requirement.

    Inputs from `context`:
      - smt_program_lines: List[str]
      - requirements: List[str|dict]
      - current_requirement_index: int
      - trial_id, inc_exc: optional (for logging)
      - SMTIncrementalReusableVariableIdentifier_prompt: str

    Outputs into `context`:
      - reusable_variables: List[dict]  (validated, deduped, sorted)
      - reuse_errors: List[dict]
      - reuse_stage: "reusables_only"
    """

    MAX_ATTEMPTS = 2

    def __init__(
        self,
        engine,
        *,
        log_dir: Optional[str] = None,
        allow_mixed_case: bool = False,
        skip_name_checks: bool = True,
    ):
        super().__init__()
        # allow_mixed_case can be overridden by env at runtime
        self._allow_mixed_case_default = allow_mixed_case
        # skip_name_checks can be overridden by env at runtime
        self._skip_name_checks_default = skip_name_checks
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _allow_mixed_case(self) -> bool:
        # env takes precedence; default is constructor arg
        return _env_truthy("SMT_REUSE_ALLOW_MIXED_CASE", self._allow_mixed_case_default)

    def _skip_name_checks(self) -> bool:
        # env takes precedence; default is constructor arg
        return _env_truthy("SMT_REUSE_SKIP_NAME_CHECKS", self._skip_name_checks_default)

    def _fallback_all(self) -> bool:
        # opt-in safety-off fallback: mark all existing SMT symbols as reusable if LLM yields none
        return _env_truthy("SMT_REUSE_FALLBACK_ALL", False)

    def _build_prompt(self, context: Dict[str, Any], idx: int) -> str:
        tpl = context.get("SMTIncrementalReusableVariableIdentifier_prompt", "")
        if not tpl:
            return ""
        smt_so_far = "\n".join(context.get("smt_program_lines", []))
        req = context["requirements"][idx]
        requirement_txt = req.get("requirement") if isinstance(req, dict) else str(req)
        return tpl.replace("#SMT_PROGRAM_BY_FAR#", smt_so_far).replace(
            "#REQUIREMENT#", requirement_txt
        )

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        idx: int = int(context["current_requirement_index"])
        trial_id = context.get("trial_id", "unknown_trial")
        side = context.get("inc_exc", "unknown")

        prompt = self._build_prompt(context, idx)
        if not prompt:
            _log(
                "reuse ✗",
                idx,
                "prompt template missing (SMTIncrementalReusableVariableIdentifier_prompt)",
            )
            context["reusable_variables"] = []
            context["reuse_errors"] = [{"code": "PROMPT_MISSING"}]
            context["reuse_stage"] = "reusables_only"
            return context

        # mbench layout: <log_root>/<trial_id>/<inclusion|exclusion>/reqNNN/0reuse/
        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc", "inclusion", "include", "in"}:
                return "inclusion"
            if s in {"exc", "exclusion", "exclude", "ex"}:
                return "exclusion"
            return s or "unknown"

        stage_dir = os.path.join(self.log_dir, str(trial_id), _bucket(side), f"req{idx:03d}")
        os.makedirs(stage_dir, exist_ok=True)
        p_log = os.path.join(stage_dir, "0reuse_prompt.txt")
        r_log = os.path.join(stage_dir, "0reuse_raw.txt")
        plan_log = os.path.join(stage_dir, "0reuse_plan.json")

        with open(p_log, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        existing_vars = _extract_declared_symbols(context.get("smt_program_lines", []))
        # Normalize existing vars to both a set (fast membership) and map (case-insensitive)
        if isinstance(existing_vars, (set, list, tuple)):
            existing_list = list(existing_vars)
        else:
            # defensive: if some impl returns an iterator
            existing_list = list(existing_vars)
        existing_set = set(existing_list)
        existing_ci = {v.lower(): v for v in existing_list}

        errors: List[Dict[str, Any]] = []
        reusable_variables: List[Dict[str, Any]] = []

        last_error: Optional[Exception] = None
        _log("reuse", idx, f"existing_symbols={len(existing_set)}")

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            out: str = self.engine(prompt)[0]

            with open(r_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- attempt {attempt} ---\n{out}\n")

            m_reuse = _REUSABLE_RE.search(out)
            if not m_reuse:
                _schema_log(
                    errors,
                    idx=idx,
                    block="top_level",
                    item_index=None,
                    code="MISSING_REUSE_BLOCK",
                    message="Could not find <reusable_variables> JSON block.",
                )
                last_error = RuntimeError("Missing <reusable_variables> block")
                continue

            try:
                reuse_raw = _parse_json_array_relaxed(m_reuse.group(1))
                candidates_len = len(reuse_raw) if isinstance(reuse_raw, list) else 0

                if self._skip_name_checks():
                    # Lenient path: only require existence among current SMT symbols.
                    reusable_variables = []
                    for j, item in enumerate(reuse_raw or []):
                        try:
                            name = str(item.get("variable_name", "")).strip()
                            if not name:
                                _schema_log(
                                    errors,
                                    idx=idx,
                                    block="reusable_variables",
                                    item_index=j,
                                    code="MISSING_NAME",
                                    message="Empty variable_name in reusable candidate.",
                                )
                                continue
                            # Existence check (case-sensitive first, then insensitive)
                            if name in existing_set:
                                canonical = name
                            else:
                                name_ci = name.lower()
                                if name_ci in existing_ci:
                                    canonical = existing_ci[name_ci]
                                else:
                                    _schema_log(
                                        errors,
                                        idx=idx,
                                        block="reusable_variables",
                                        item_index=j,
                                        code="UNKNOWN_SYMBOL",
                                        message="Reusable variable not found among declared SMT symbols.",
                                        detail=name,
                                    )
                                    continue

                            # Normalize case if mixed case is NOT allowed
                            if not self._allow_mixed_case():
                                name_out = canonical
                            else:
                                name_out = name

                            why = str(item.get("why", "") or "reused existing SMT symbol")
                            reusable_variables.append(
                                {"variable_name": name_out, "why": why}
                            )
                        except Exception as _exc:
                            _schema_log(
                                errors,
                                idx=idx,
                                block="reusable_variables",
                                item_index=j,
                                code="ITEM_ERROR",
                                message="Error processing reusable candidate.",
                                detail=str(_exc),
                            )
                            continue
                    kept_len = len(reusable_variables)
                else:
                    # Strict/legacy path: run the full validator (name regex, etc.)
                    reusable_variables = _validate_reusable_vars(
                        reuse_raw,
                        existing_list,
                        allow_mixed_case=self._allow_mixed_case(),
                        diag=errors,
                        context_idx=idx,
                        block_label="reusable_variables",
                    )
                    kept_len = len(reusable_variables)

                # Success payload + file
                plan = {
                    "existing_symbols_count": len(existing_set),
                    "candidates_from_llm_count": candidates_len,
                    "kept_reusable_count": kept_len,
                    "reusable_variables": reusable_variables,
                    "stage": "reusables_only",
                    "errors": errors or [],
                }
                with open(plan_log, "w", encoding="utf-8") as fh:
                    json.dump(plan, fh, indent=2, ensure_ascii=False)

                _log("reuse ✓", idx, f"candidates={candidates_len}; kept={kept_len}")
                context["reusable_variables"] = reusable_variables
                context["reuse_errors"] = errors or []
                context["reuse_stage"] = "reusables_only"
                return context

            except Exception as exc:
                last_error = exc
                _schema_log(
                    errors,
                    idx=idx,
                    block="reusable_variables",
                    item_index=None,
                    code="JSON_OR_VALIDATION",
                    message="Failed to parse/validate reusable variables.",
                    detail=str(exc),
                )
                continue

        # Fallback: empty reuse list (or optional 'all' fallback)
        if self._fallback_all() and existing_set:
            # Extremely permissive; only when explicitly enabled
            reusable_variables = [
                {
                    "variable_name": v,
                    "why": "fallback: all existing symbols treated as reusable",
                }
                for v in sorted(existing_set)
            ]
            plan = {
                "existing_symbols_count": len(existing_set),
                "candidates_from_llm_count": 0,
                "kept_reusable_count": len(reusable_variables),
                "reusable_variables": reusable_variables,
                "stage": "reusables_only",
                "errors": errors + [{"code": "FALLBACK_ALL"}],
            }
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan, fh, indent=2, ensure_ascii=False)
            _log("reuse ⚠", idx, f"fallback_all enabled → kept={len(reusable_variables)}")
            context["reusable_variables"] = reusable_variables
            context["reuse_errors"] = plan["errors"]
            context["reuse_stage"] = "reusables_only"
            return context

        _log("reuse ⚠", idx, f"fallback to empty list due to: {last_error!r}")
        context["reusable_variables"] = []
        context["reuse_errors"] = errors or [
            {"code": "FALLBACK_EMPTY", "detail": str(last_error) if last_error else ""}
        ]
        context["reuse_stage"] = "reusables_only"
        # Write a minimal plan for traceability
        plan = {
            "existing_symbols_count": len(existing_set),
            "candidates_from_llm_count": 0,
            "kept_reusable_count": 0,
            "reusable_variables": [],
            "stage": "reusables_only",
            "errors": context["reuse_errors"],
        }
        with open(plan_log, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2, ensure_ascii=False)
        return context
