# modules/PatientCanonicalVariableCoder.py
from __future__ import annotations

"""
PatientCanonicalVariableCoder (CANON-ONLY)
────────────────────────────────────────────────────────────────────────────
Parses, validates, and writes <new_canonical_variable_declarations>.

Canon-only policy:
- Gating/coverage use ONLY `entity_canonical_form` (never preferred_term).
- Prompt is expected to provide a list whose items contain:
    { "span", "entity_canonical_form", "type", ["start","end"]? }
- Model outputs must use `entity_canonical_form_used` exactly as provided.

Assumes `_minify_from_valid_entities(...)` and `_minify_bundle(...)` emit
objects with an `entity_canonical_form` key (no PT fallback).

Writes to context:
- new_canonical_variable_declarations (list[dict])
- new_variable_declarations          (ASPS + canonical)
- errors                             (appended)
- namer_mode                         ("entities_only"|"full")
- namer_stage                        ("canonical")
"""

import json, os, warnings
from typing import Dict, List, Any

import dspy
from .namer_common import (
    _log, _minify_bundle, _minify_from_valid_entities,
    _NEWCANON_RE, _NEWDECL_RE_OLD, _NEWOTHER_RE, _NEWNONCANON_RE, _ERRORS_RE,
    _parse_json_array_relaxed,
    _validate_decl_list, _sort_key,
    _default_timeframe_for_type, _template_for_type, _synthesize_stem,
    _coerce_bounds_for_json,
)


class PatientCanonicalVariableCoder(dspy.Module):
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        engine,
        *,
        log_dir: str | None = None,
        entities_only: bool | None = True,
        enforce_suffix_in_stem: bool = True,
        allow_mixed_case: bool = False,
        require_canonical_coverage: bool = True,
    ):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self._entities_only = entities_only
        self._enforce_suffix_in_stem = enforce_suffix_in_stem
        self._allow_mixed_case = allow_mixed_case
        self._require_canonical_coverage = require_canonical_coverage

    @staticmethod
    def _env_truthy(name: str, default: bool = False) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "on"}

    def forward(self, context: Dict) -> Dict:  # type: ignore[override]
        requirements: List = context.get("requirements", [])
        idx: int = context["current_requirement_index"]
        if not requirements:
            return context

        # Entities-only gating (ctor arg > context > env)
        entities_only = (
            self._entities_only
            if self._entities_only is not None
            else (bool(context.get("namer_entities_only"))
                  or self._env_truthy("SMT_NAMER_ENTITIES_ONLY", default=False))
        )

        # Coverage policy (ctor arg > context > env)
        cov_ctx = context.get("namer_require_canonical_coverage", None)
        if cov_ctx is None:
            coverage_required = self._env_truthy(
                "SMT_NAMER_REQUIRE_CANONICAL_COVERAGE",
                default=self._require_canonical_coverage,
            )
        else:
            coverage_required = bool(cov_ctx)

        # Partial coverage allowed? (default TRUE)
        partial_ok_ctx = context.get("namer_allow_partial_canonical_coverage", None)
        if partial_ok_ctx is None:
            partial_coverage_ok = self._env_truthy("SMT_NAMER_ALLOW_PARTIAL_COVERAGE", True)
        else:
            partial_coverage_ok = bool(partial_ok_ctx)

        req_entry = requirements[idx]
        requirement_txt = (
            req_entry.get("requirement") if isinstance(req_entry, dict) else str(req_entry)
        )

        namer_prompt_tpl = context.get("PatientCanonicalVariableCoder_prompt", "")
        if not namer_prompt_tpl:
            _log("namer ✗", idx, "prompt template missing (canonical)")
            return context

        # —— Build canonical-form bundle & allowed set (CANON-ONLY) —— #
        bundles: List[dict] = context.get("requirement_bundles", []) or []
        bundles_by_index = {b.get("req_index"): b for b in bundles if isinstance(b, dict)}

        allowed_canon_forms: set[str] = set()
        allowed_info: Dict[str, Dict[str, str]] = {}

        if entities_only:
            ver_all: dict = context.get("valid_entities_by_req", {}) or {}
            # Tolerate both string and int keys
            ver_for_req = ver_all.get(str(idx)) or ver_all.get(idx) or {}
            if ver_for_req:
                for e in ver_for_req.values():
                    s = e.get("entity_canonical_form")  # ← canon only
                    if s:
                        allowed_canon_forms.add(s)
                        allowed_info[s] = {
                            "type": e.get("type", "") or "",
                            "span": e.get("extracted_span", s) or s,
                        }
            canon_bundle = _minify_from_valid_entities(ver_for_req) if ver_for_req else ""
            if not canon_bundle:
                b = bundles_by_index.get(idx)
                canon_bundle = _minify_bundle(b, include_attributes=False)
                if b:
                    for ent in (b.get("entities") or []):
                        s = ent.get("entity_canonical_form")  # ← canon only
                        if s:
                            allowed_canon_forms.add(s)
                            allowed_info[s] = {
                                "type": ent.get("type", "") or "",
                                "span": ent.get("span", s) or s,
                            }
                _log("namer", idx, "entities_only: empty valid_entities_by_req → fell back to requirement_bundles")
        else:
            b = bundles_by_index.get(idx)
            canon_bundle = _minify_bundle(b, include_attributes=True)
            if b:
                for ent in (b.get("entities") or []):
                    s = ent.get("entity_canonical_form")  # ← canon only
                    if s:
                        allowed_canon_forms.add(s)
                        allowed_info[s] = {"type": ent.get("type", "") or "", "span": ent.get("span", s) or s}

        # Build prompt
        prompt = (
            namer_prompt_tpl
            .replace("#SMT_PROGRAM_BY_FAR#", "\n".join(context.get("smt_program_lines", [])))
            .replace("#CANONICAL_FORMS#", canon_bundle)
            .replace("#PATIENT_FACT#", requirement_txt)
            .replace("#PATIENT_NOTE#", context.get("requirement_text", ""))
        )

        # Logs
        note_id = context.get("note_id", "unknown_patient_note")
        base_dir = os.path.join(self.log_dir, note_id)
        os.makedirs(base_dir, exist_ok=True)
        base_name = f"req{idx:03d}_namer_canon"
        p_log = os.path.join(base_dir, f"{base_name}_prompt.txt")
        r_log = os.path.join(base_dir, f"{base_name}_raw.txt")
        plan_log = os.path.join(base_dir, f"{base_name}_plan.json")
        with open(p_log, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        _log(
            "namer",
            idx,
            f"mode={'ENTITIES_ONLY' if entities_only else 'FULL'}; "
            f"coverage_required={coverage_required}; partial_ok={partial_coverage_ok}"
        )

        # Helper: gather concept meta by canonical form for this requirement
        def _canon_meta_map(ctx: Dict[str, Any], req_idx: int) -> Dict[str, Dict[str, Any]]:
            meta: Dict[str, Dict[str, Any]] = {}

            # 1) requirement_bundles (preferred)
            b = bundles_by_index.get(req_idx)
            if b:
                for ent in (b.get("entities") or []):
                    cf = ent.get("entity_canonical_form")
                    if not cf:
                        continue
                    ent_flat = ent.get("entity") if isinstance(ent.get("entity"), dict) else ent
                    slot = meta.setdefault(cf, {})
                    if ent_flat.get("conceptId"):
                        slot["conceptId"] = str(ent_flat.get("conceptId"))
                    if ent_flat.get("preferred_term"):
                        slot["preferred_term"] = ent_flat.get("preferred_term")
                    if ent_flat.get("fully_specified_name"):
                        slot["fully_specified_name"] = ent_flat.get("fully_specified_name")
                    if ent_flat.get("type"):
                        slot["type"] = ent_flat.get("type")
                    if ent.get("span") or ent_flat.get("span"):
                        slot["span"] = ent.get("span") or ent_flat.get("span")

            # 2) valid_entities_by_req (fallback)
            ver_all2 = ctx.get("valid_entities_by_req") or {}
            ver_req = ver_all2.get(str(req_idx)) or ver_all2.get(req_idx) or {}
            for node in (ver_req or {}).values():
                cf = node.get("entity_canonical_form")
                if not cf:
                    continue
                slot = meta.setdefault(cf, {})
                slot.setdefault("conceptId", str(node.get("conceptId")) if node.get("conceptId") else None)
                slot.setdefault("preferred_term", node.get("preferred_term"))
                slot.setdefault("fully_specified_name", node.get("fully_specified_name"))
                slot.setdefault("type", node.get("type"))
                slot.setdefault("span", node.get("extracted_span"))

            # prune Nones
            for cf in list(meta.keys()):
                meta[cf] = {k: v for k, v in meta[cf].items() if v not in (None, "", [])}
                if not meta[cf]:
                    del meta[cf]
            return meta

        succeeded = False
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            llm_out: str = self.engine(prompt)[0]
            with open(r_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")

            # Parse canonical block (accept legacy single-block)
            m_canon = _NEWCANON_RE.search(llm_out)
            if not m_canon:
                m_old = _NEWDECL_RE_OLD.search(llm_out)
                if m_old:
                    m_canon = m_old

            illegal_other = _NEWOTHER_RE.search(llm_out) is not None
            illegal_noncanon_legacy = _NEWNONCANON_RE.search(llm_out) is not None

            try:
                if not m_canon:
                    raise RuntimeError("Canonical block <new_canonical_variable_declarations> not found")
                #canon_list_raw = _parse_json_array_relaxed(m_canon.group(1))
                #print("[debug canon] raw canon block: ", m_canon.group(1))
                canon_payload = _coerce_bounds_for_json(m_canon.group(1))
                canon_list_raw = _parse_json_array_relaxed(canon_payload)
                m_err = _ERRORS_RE.search(llm_out)
                errors_list = _parse_json_array_relaxed(m_err.group(1)) if m_err else []
            except Exception as exc:
                last_error = exc
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"canonical parse error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue
            #print("[debug canon] parsed raw canon list: ", canon_list_raw)
            if illegal_other or illegal_noncanon_legacy:
                errors_list = errors_list or []
                errors_list.append({
                    "invariant": "S1",
                    "problem": "Non-canonical/non-ASPS blocks emitted in canonical stage",
                    "fix": "Ignored in canonical stage.",
                })

            try:
                sanitize_notes: list[dict] = []
                canon_list = _validate_decl_list(
                    canon_list_raw,
                    kind="canonical",
                    allow_backcompat=True,
                    # enforce_suffix_in_stem=self._enforce_suffix_in_stem,
                    allow_mixed_case=self._allow_mixed_case,
                    collect_sanitize_corrections=sanitize_notes,
                    enable_template_checks=False,
                )

                # Hard guard: forbid PT fields; enforce canon-only outputs
                for _i, _o in enumerate(canon_list):
                    if "preferred_term" in _o or "preferred_term_used" in _o:
                        raise ValueError("preferred_term fields are not allowed; use entity_canonical_form_used only")

                # I3: gating by allowed canonical forms (canon-only)
                if allowed_canon_forms:
                    bad = [c for c in canon_list if c.get("entity_canonical_form_used") not in allowed_canon_forms]
                    if bad:
                        raise ValueError(
                            "canonical declarations include entities not in #CANONICAL_FORMS#: "
                            + ", ".join(sorted({(b.get("entity_canonical_form_used") or "") for b in bad}))
                        )

                # I1: timeframe enforcement (will canonicalize stem token and validate)
                #_enforce_timeframe_in_stem(canon_list, allow_mixed_case=self._allow_mixed_case)

                if sanitize_notes:
                    errors_list = (errors_list or [])
                    errors_list.append({
                        "invariant": "I4-sanitize",
                        "problem": "auto-corrected names to canonical snake_case/position",
                        "corrections": sanitize_notes,
                    })

                # Coverage check remains (based on what the model returned)
                if allowed_canon_forms:
                    present = {c.get("entity_canonical_form_used") for c in canon_list}
                    missing = sorted({a for a in allowed_canon_forms if a not in present})
                    if missing:
                        if coverage_required and not partial_coverage_ok:
                            raise ValueError("missing canonical declarations for: " + ", ".join(missing))
                        _log("namer ⚠", idx, "partial canonical coverage; missing: " + ", ".join(missing))
                        errors_list.append({
                            "invariant": "I3-partial-coverage",
                            "problem": "not all canonical entities were declared; proceeding with subset",
                            "missing_canonical_forms": missing,
                        })

            except Exception as ve:
                last_error = ve
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"canonical schema error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue

            # Success — enrich canonical rows with concept metadata so export keeps them
            meta_map = _canon_meta_map(context, idx)
            for _o in canon_list:
                _o.pop("_original_entity_variable_name", None)
                cf = _o.get("entity_canonical_form_used")
                m = meta_map.get(cf, {})
                if m.get("conceptId"):
                    _o["conceptId"] = m["conceptId"]
                if m.get("preferred_term"):
                    _o["preferred_term"] = m["preferred_term"]
                if m.get("fully_specified_name"):
                    _o["fully_specified_name"] = m["fully_specified_name"]
                if m.get("type"):
                    _o["entity_type"] = m["type"]
                if (not _o.get("span")) and m.get("span"):
                    _o["span"] = m["span"]

            asps_existing = context.get("new_age_sex_pregnancystatus_declarations", []) or []
            plan_obj = {
                "new_age_sex_pregnancystatus_declarations": asps_existing,
                "new_canonical_variable_declarations": canon_list,
                "all_other_variable_declarations": [],
                "errors": errors_list or [],
                "stage": "demographics_canonical",
            }
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan_obj, fh, indent=2, ensure_ascii=False)

            context["new_canonical_variable_declarations"] = canon_list
            context["all_other_variable_declarations"] = []
            context["new_noncanonical_variable_declarations"] = []
            context["new_variable_declarations"] = (asps_existing or []) + canon_list
            context["errors"] = (context.get("errors") or []) + (errors_list or [])
            context["namer_mode"] = "entities_only" if entities_only else "full"
            context["namer_stage"] = "canonical"
            _log("namer ✓", idx, "canonical accepted")
            succeeded = True
            break

        if not succeeded:
            warnings.warn(f"canonical namer fallback (req#{idx}): {last_error}", RuntimeWarning)
            _log("namer ⚠ fallback", idx, f"canonical: {last_error!r}")

            errors_list = [{
                "invariant": "FALLBACK",
                "problem": "LLM outputs could not be parsed/validated after retries; synthesized declarations",
                "detail": str(last_error) if last_error else "unknown",
            }]

            # Synthesize canonical declarations from allowed forms (canon-only)
            canon_list: list[dict] = []
            for canon in sorted(allowed_canon_forms):
                info = allowed_info.get(canon, {"type": "", "span": canon})
                typ = info.get("type", "")
                timeframe = _default_timeframe_for_type(typ)
                stem = _synthesize_stem(canon, typ, timeframe, allow_mixed_case=self._allow_mixed_case)
                template = _template_for_type(typ)
                item = {
                    "span": info.get("span", canon),
                    "entity_variable_name": stem,
                    "template": template,
                    "timeframe": timeframe,
                    "entity_canonical_form_used": canon,  # ← canon only
                    "usage": "fallback synthesized declaration to preserve canonical coverage",
                    "qualifier_predicates": [],
                }
                # Try to enrich fallback too
                m = (_canon_meta_map(context, idx)).get(canon, {})
                if m.get("conceptId"):            item["conceptId"] = m["conceptId"]
                if m.get("preferred_term"):       item["preferred_term"] = m["preferred_term"]
                if m.get("fully_specified_name"): item["fully_specified_name"] = m["fully_specified_name"]
                if m.get("type"):                 item["entity_type"] = m["type"]
                if (not item.get("span")) and m.get("span"):
                    item["span"] = m["span"]

                canon_list.append(item)

            canon_list.sort(key=_sort_key)

            # try:
            #     _enforce_timeframe_in_stem(canon_list, allow_mixed_case=self._allow_mixed_case)
            # except Exception as e:
            #     errors_list.append({
            #         "invariant": "I1-relaxed-fallback",
            #         "problem": "timeframe token check failed in synthesized stem; proceeding",
            #         "detail": str(e),
            #     })

            missing = sorted({a for a in allowed_canon_forms if a not in {c["entity_canonical_form_used"] for c in canon_list}})
            if missing:
                errors_list.append({
                    "invariant": "I3-coverage-relaxed-fallback",
                    "problem": "coverage relaxed in fallback; some canonical forms not synthesized",
                    "missing_canonical_forms": missing,
                })

            asps_existing = context.get("new_age_sex_pregnancystatus_declarations", []) or []
            plan_obj = {
                "new_age_sex_pregnancystatus_declarations": asps_existing,
                "new_canonical_variable_declarations": canon_list,
                "all_other_variable_declarations": [],
                "errors": errors_list,
                "stage": "demographics_canonical",
                "fallback": True,
            }
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan_obj, fh, indent=2, ensure_ascii=False)

            context["new_canonical_variable_declarations"] = canon_list
            context["all_other_variable_declarations"] = []
            context["new_noncanonical_variable_declarations"] = []
            context["new_variable_declarations"] = (asps_existing or []) + canon_list
            context["errors"] = (context.get("errors") or []) + errors_list
            context["namer_stage"] = "canonical"
            context["namer_mode"] = "entities_only" if entities_only else "full"
            context["namer_fallback"] = True

        return context
