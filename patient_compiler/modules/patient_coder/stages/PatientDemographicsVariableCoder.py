# modules/PatientDemographicsVariableCoder.py
from __future__ import annotations

import json, os, warnings
from typing import Dict, List

import dspy
from .namer_common import (
    _log, _minify_bundle, _minify_from_valid_entities,
    _NEWASPS_RE, _NEWOTHER_RE, _NEWNONCANON_RE, _ERRORS_RE,
    _extract_declared_symbols, _parse_json_array_relaxed,
    _validate_decl_list, _coerce_bounds_for_json,
)

class PatientDemographicsVariableCoder(dspy.Module):
    """
    Stage-1 (ASPS only): parses, validates, and writes
    <new_age_sex_pregnancystatus_declarations>.
    Leaves canonical untouched if present in context.
    """

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        engine,
        *,
        log_dir: str | None = None,
        enforce_suffix_in_stem: bool = True,
        allow_mixed_case: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self._enforce_suffix_in_stem = enforce_suffix_in_stem
        self._allow_mixed_case = allow_mixed_case

    def forward(self, context: Dict) -> Dict:  # type: ignore[override]
        requirements: List = context.get("requirements", [])
        idx: int = context["current_requirement_index"]
        if not requirements:
            return context

        req_entry = requirements[idx]
        requirement_txt = (
            req_entry.get("requirement") if isinstance(req_entry, dict) else str(req_entry)
        )
        namer_prompt_tpl = context.get("PatientDemographicsVariableCoder_prompt", "")
        if not namer_prompt_tpl:
            _log("namer ✗", idx, "prompt template missing (ASPS)")
            return context

        # Build a lightweight canonical bundle for the prompt (even though ASPS doesn't gate on it)
        bundles: List[dict] = context.get("requirement_bundles", []) or []
        bundles_by_index = {b.get("req_index"): b for b in bundles if isinstance(b, dict)}
        b = bundles_by_index.get(idx)
        canon_bundle = ""
        ver_all: dict = context.get("valid_entities_by_req", {}) or {}
        ver_for_req = ver_all.get(str(idx)) or {}
        if ver_for_req:
            canon_bundle = _minify_from_valid_entities(ver_for_req)
        elif b:
            canon_bundle = _minify_bundle(b, include_attributes=False)

        # Build prompt
        prompt = (
            namer_prompt_tpl
            .replace("#SMT_PROGRAM_BY_FAR#", "\n".join(context.get("smt_program_lines", [])))
            .replace("#CANONICAL_FORMS#", canon_bundle)
            .replace("#PATIENT_FACT#", requirement_txt)
            .replace("#PATIENT_NOTE#", context.get("requirement_text", ""))
        )

        # Logging paths
        note_id = context.get("note_id", "unknown_patient_note")
        base_dir = os.path.join(self.log_dir, note_id)
        os.makedirs(base_dir, exist_ok=True)
        base_name = f"req{idx:03d}_namer_asps"
        p_log = os.path.join(base_dir, f"{base_name}_prompt.txt")
        r_log = os.path.join(base_dir, f"{base_name}_raw.txt")
        plan_log = os.path.join(base_dir, f"{base_name}_plan.json")
        with open(p_log, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        existing_vars = _extract_declared_symbols(context.get("smt_program_lines", []))
        succeeded = False
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            llm_out: str = self.engine(prompt)[0]
            with open(r_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")

            # Parse blocks relevant to ASPS
            m_asps = _NEWASPS_RE.search(llm_out)
            illegal_other = _NEWOTHER_RE.search(llm_out) is not None
            illegal_noncanon_legacy = _NEWNONCANON_RE.search(llm_out) is not None
            try:
                if not m_asps:
                    raise RuntimeError("ASPS block <new_age_sex_pregnancystatus_declarations> not found")
                asps_payload = _coerce_bounds_for_json(m_asps.group(1))
                asps_list_raw = _parse_json_array_relaxed(asps_payload)
                m_err = _ERRORS_RE.search(llm_out)
                errors_list = _parse_json_array_relaxed(m_err.group(1)) if m_err else []
            except Exception as exc:
                last_error = exc
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"ASPS parse error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue

            if illegal_other or illegal_noncanon_legacy:
                errors_list = errors_list or []
                errors_list.append({
                    "invariant": "S1",
                    "problem": "Non-canonical/non-ASPS blocks emitted in ASPS stage",
                    "fix": "Ignored in ASPS stage.",
                })

            try:
                sanitize_notes: list[dict] = []
                asps_list = _validate_decl_list(
                    asps_list_raw,
                    kind="age_sex_preg",
                    allow_backcompat=False,
                    #enforce_suffix_in_stem=self._enforce_suffix_in_stem,
                    allow_mixed_case=self._allow_mixed_case,
                    collect_sanitize_corrections=sanitize_notes,
                    enable_template_checks=False,
                )
                #print("[debug demographic] validated list: ", asps_list)

                # Drop re-declarations against SMT program
                names_asps = [o["entity_variable_name"] for o in asps_list]
                dups_existing = sorted({n for n in names_asps if n in existing_vars})
                if dups_existing:
                    asps_list = [o for o in asps_list if o["entity_variable_name"] not in dups_existing]
                    _log("namer ⚠", idx, "ASPS dedup against SMT program: " + ", ".join(dups_existing))
                    errors_list = (errors_list or [])
                    errors_list.append({
                        "invariant": "I2-soft",
                        "problem": "ASPS attempted redeclare existing variables; dropped",
                        "dropped_variables": dups_existing,
                    })

                # Soft cross-dedup vs any canonical already in context
                prev_canon = context.get("new_canonical_variable_declarations", []) or []
                prev_canon_names = {
                    o.get("entity_variable_name") for o in prev_canon if isinstance(o, dict)
                }
                dups_cross = sorted({o["entity_variable_name"] for o in asps_list if o["entity_variable_name"] in prev_canon_names})
                if dups_cross:
                    asps_list = [o for o in asps_list if o["entity_variable_name"] not in dups_cross]
                    _log("namer ⚠", idx, "ASPS cross-dedup vs canonical: " + ", ".join(dups_cross))
                    errors_list.append({
                        "invariant": "I2-soft-cross",
                        "problem": "ASPS duplicate names vs canonical; ASPS entries dropped",
                        "dropped_variables": dups_cross,
                    })

                # Uniqueness within ASPS new entries
                names_asps = [o["entity_variable_name"] for o in asps_list]
                dup_self = {n for n in names_asps if names_asps.count(n) > 1}
                if dup_self:
                    raise ValueError("duplicate ASPS variable names in new declarations: " + ", ".join(sorted(dup_self)))

                if sanitize_notes:
                    errors_list = (errors_list or [])
                    errors_list.append({
                        "invariant": "I4-sanitize",
                        "problem": "auto-corrected names to canonical snake_case/position",
                        "corrections": sanitize_notes,
                    })

            except Exception as ve:
                last_error = ve
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"ASPS schema error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue

            # Success
            for _o in asps_list:
                _o.pop("_original_entity_variable_name", None)

            canon_existing = context.get("new_canonical_variable_declarations", []) or []
            plan_obj = {
                "new_age_sex_pregnancystatus_declarations": asps_list,
                "new_canonical_variable_declarations": canon_existing,
                "all_other_variable_declarations": [],
                "errors": errors_list or [],
                "stage": "demographics_canonical",   # keep string unchanged for downstreams
            }
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan_obj, fh, indent=2, ensure_ascii=False)

            # Update context
            context["new_age_sex_pregnancystatus_declarations"] = asps_list
            context["all_other_variable_declarations"] = []
            context["new_noncanonical_variable_declarations"] = []
            context["new_variable_declarations"] = asps_list + (canon_existing or [])
            context["errors"] = (context.get("errors") or []) + (errors_list or [])
            context["namer_mode"] = "demographics_only"
            context["namer_stage"] = "demographics"
            _log("namer ✓", idx, "ASPS accepted")
            succeeded = True
            break

        if not succeeded:
            warnings.warn(f"ASPS namer fallback (req#{idx}): {last_error}", RuntimeWarning)
            _log("namer ⚠ fallback", idx, f"ASPS: {last_error!r}")
            errors_list = [{
                "invariant": "FALLBACK-ASPS",
                "problem": "ASPS could not be parsed/validated after retries; leaving empty",
                "detail": str(last_error) if last_error else "unknown",
            }]

            canon_existing = context.get("new_canonical_variable_declarations", []) or []
            plan_obj = {
                "new_age_sex_pregnancystatus_declarations": [],
                "new_canonical_variable_declarations": canon_existing,
                "all_other_variable_declarations": [],
                "errors": errors_list,
                "stage": "demographics_canonical",
                "fallback": True,
            }
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan_obj, fh, indent=2, ensure_ascii=False)

            context["new_age_sex_pregnancystatus_declarations"] = []
            context["all_other_variable_declarations"] = []
            context["new_noncanonical_variable_declarations"] = []
            context["new_variable_declarations"] = (canon_existing or [])
            context["errors"] = (context.get("errors") or []) + errors_list
            context["namer_stage"] = "demographics"
            context["namer_mode"] = "demographics_only"
            context["namer_fallback"] = True

        return context
