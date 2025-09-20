# PatientSMTVariableCoderChecker.py
# ===============================================================
# Verifies that each declared SMT variable faithfully encodes the
# corresponding patient fact using an LLM prompt + strict parsers.
#
# CHANGELOG (2025-09-12):
# - Prompt source is now EXCLUSIVELY context["PatientSMTVariableCoderChecker_prompt"].
# - If absent/empty, the verifier skips LLM invocation and uses heuristic fallback.
# - __init__(..., prompt_tmpl=...) retained for backward compatibility but ignored.
# ===============================================================

from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import dspy  # type: ignore

# ---------------------------------------------------------------------------
# Fallback logger (mirrors other modules)
# ---------------------------------------------------------------------------
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [verifier] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)


# ---------------------------------------------------------------------------
# Relaxed JSON helpers (strip fences/comments/trailing commas)
# ---------------------------------------------------------------------------
_CODEFENCE_RE = re.compile(r"^\s*```(?:json|JSON|text)?\s*|\s*```$", re.M)

def _strip_code_fence(s: str) -> str:
    return re.sub(_CODEFENCE_RE, "", s or "")

def _strip_json_comments_and_trailing_commas(s: str) -> str:
    s = re.sub(r"(?m)^\s*//.*$", "", s or "")
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def _parse_json_relaxed(text: str) -> Any:
    s = _strip_code_fence(text)
    s = _strip_json_comments_and_trailing_commas(s).strip()
    if "{" in s and "}" in s:
        first = s.find("{")
        last = s.rfind("}")
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return json.loads(s)


# ---------------------------------------------------------------------------
# Tag regex (if the model outputs angle-tagged arrays)
# ---------------------------------------------------------------------------
_VER_CANON_RE = re.compile(
    r"<new_canonical_variable_declarations>\s*(\[.*?])\s*</new_canonical_variable_declarations>",
    re.I | re.S,
)
_VER_ASPS_RE = re.compile(
    r"<new_age_sex_pregnancystatus_declarations>\s*(\[.*?])\s*</new_age_sex_pregnancystatus_declarations>",
    re.I | re.S,
)


# ---------------------------------------------------------------------------
# Canonical-forms mapping builders
# ---------------------------------------------------------------------------
def _canon_map_from_valid_entities(ver_for_req: dict | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in (ver_for_req or {}).values():
        span = e.get("extracted_span") or e.get("span")
        canon = e.get("preferred_term") or e.get("entity_canonical_form")
        if span and canon:
            out[str(span)] = str(canon)
    return out

def _canon_map_from_bundle(bundle: dict | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not bundle:
        return out
    for ent in bundle.get("entities", []) or []:
        span = ent.get("span")
        canon = ent.get("entity_canonical_form")
        if span and canon:
            out[str(span)] = str(canon)
    return out

def _build_canonical_forms_map(context: Dict, idx: int) -> Dict[str, str]:
    ver_all: dict = context.get("valid_entities_by_req", {}) or {}
    ver_for_req = ver_all.get(str(idx)) or {}
    if ver_for_req:
        return _canon_map_from_valid_entities(ver_for_req)
    bundles: List[dict] = context.get("requirement_bundles", []) or []
    by_idx = {b.get("req_index"): b for b in bundles if isinstance(b, dict)}
    return _canon_map_from_bundle(by_idx.get(idx))


# ---------------------------------------------------------------------------
# Declared variables collector
# ---------------------------------------------------------------------------
def _collect_declared_variables(context: Dict) -> Dict[str, List[dict]]:
    if "declared_smt_variables" in context and isinstance(context["declared_smt_variables"], dict):
        dv = context["declared_smt_variables"]
        return {
            "new_age_sex_pregnancystatus_declarations": list(dv.get("new_age_sex_pregnancystatus_declarations", []) or []),
            "new_canonical_variable_declarations": list(dv.get("new_canonical_variable_declarations", []) or []),
        }
    return {
        "new_age_sex_pregnancystatus_declarations": list(context.get("new_age_sex_pregnancystatus_declarations", []) or []),
        "new_canonical_variable_declarations": list(context.get("new_canonical_variable_declarations", []) or []),
    }


# ---------------------------------------------------------------------------
# Small normalizers & validators for verifier outputs
# ---------------------------------------------------------------------------
def _normalize_yes_no(val: Any, *, allow_unknown: bool = False) -> str:
    s = str(val or "").strip().upper()
    opts = {"YES", "NO"}
    if allow_unknown:
        opts.add("UNKNOWN")
    return s if s in opts else ("UNKNOWN" if allow_unknown else "NO")

def _compute_all_good(span_equal: str, entail: str) -> str:
    return "YES" if (span_equal == "YES" and entail == "YES") else "NO"

def _ensure_keys_and_fix_typos(item: dict, *, is_canonical: bool) -> dict:
    o = dict(item or {})
    if "PATEINT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING" in o and "PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING" not in o:
        o["PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING"] = o.pop("PATEINT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING")
    for k in ["span", "template", "timeframe", "entity_variable_name", "type", "extracted_value"]:
        o.setdefault(k, item.get(k))
    if is_canonical:
        o.setdefault("entity_canonical_form_used", item.get("entity_canonical_form_used"))
        o.setdefault("qualifier_predicates", item.get("qualifier_predicates", []))
    o["SPAN_EQUAL_TO_ENTITY_CANONICAL_FORM_USED"] = _normalize_yes_no(o.get("SPAN_EQUAL_TO_ENTITY_CANONICAL_FORM_USED"), allow_unknown=False)
    o["PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING"] = _normalize_yes_no(o.get("PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING"), allow_unknown=True)
    if not o.get("entity_variable_meaning"):
        o["entity_variable_meaning"] = f"This variable asserts '{o.get('entity_variable_name', '')}' is set to {o.get('extracted_value', None)} for the patient."
    if not o.get("ALL_GOOD"):
        o["ALL_GOOD"] = _compute_all_good(
            o["SPAN_EQUAL_TO_ENTITY_CANONICAL_FORM_USED"],
            o["PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING"],
        )
    return o


# ---------------------------------------------------------------------------
# Heuristic fallback computation
# ---------------------------------------------------------------------------
_UNIT_FROM_STEM_RE = re.compile(r"_withunit_([a-z0-9_]+)$", re.I)
_STATUS_FROM_STEM_RE = re.compile(r"_(is_positive|is_negative|is_adequate|is_inadequate)_", re.I)

def _tf_phrase(tf: str) -> str:
    s = (tf or "").strip().lower()
    if s == "now":
        return "now"
    if s == "inthehistory":
        return "in the history"
    if s == "inthefuture":
        return "in the future"
    m = re.match(r"inthe(past|future)(\d+)([a-z]+)", s)
    if m:
        pf, n, u = m.groups()
        return f"in the {pf} {n} {u}"
    return s or "now"

def _guess_meaning_sentence(obj: dict, *, is_canonical: bool) -> str:
    stem = str(obj.get("entity_variable_name", ""))
    tfp = _tf_phrase(str(obj.get("timeframe", "")))
    extracted = obj.get("extracted_value", None)
    if obj.get("template") == "observable_entities_numeric":
        unit = ""
        m = _UNIT_FROM_STEM_RE.search(stem)
        if m:
            unit = " " + m.group(1).replace("_", " ")
        ent = stem.split("_value_recorded_", 1)[0].replace("_", " ")
        return f"This variable asserts that the patient's {ent} value recorded {tfp} is {extracted}{unit}."
    if obj.get("template") == "observable_entities_status":
        ent = stem.split("_is_", 1)[0].replace("_", " ")
        sm = _STATUS_FROM_STEM_RE.search(stem)
        status = sm.group(1).replace("_", " ") if sm else "in a specific status"
        return f"This variable asserts that the patient's {ent} is {status} {tfp}."
    if obj.get("template") == "procedures":
        parts = stem.split("_")
        if parts and parts[0] in {"has", "is", "will", "can"}:
            verb = " ".join(parts[:2]).replace("_", " ")
            ent = "_".join(parts[2:-1]).replace("_", " ")
        else:
            verb = "has undergone"
            ent = stem.replace("_", " ")
        return f"This variable asserts that the patient {verb} {ent} {tfp} (value={extracted})."
    if obj.get("template") == "product":
        ent = stem.split("_", 2)[-1].rsplit("_", 1)[0].replace("_", " ")
        return f"This variable asserts that the patient uses medication '{ent}' {tfp} (value={extracted})."
    if obj.get("template") == "substance":
        ent = stem.replace("is_exposed_to_", "").rsplit("_", 1)[0].replace("_", " ")
        return f"This variable asserts that the patient is exposed to '{ent}' {tfp} (value={extracted})."
    ent = obj.get("entity_canonical_form_used") or stem.replace("has_finding_of_", "").rsplit("_", 1)[0].replace("_", " ")
    return f"This variable asserts that the patient has a finding of {ent} {tfp} (value={extracted})."

def _fallback_verify_lists(
    asps: List[dict], canon: List[dict], canon_map: Dict[str, str]
) -> Tuple[List[dict], List[dict], List[dict]]:
    errors: List[dict] = []

    out_asps: List[dict] = []
    for i, o in enumerate(asps or []):
        item = dict(o)
        item["SPAN_EQUAL_TO_ENTITY_CANONICAL_FORM_USED"] = "YES"
        item["entity_variable_meaning"] = _guess_meaning_sentence(item, is_canonical=False)
        item["PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING"] = "UNKNOWN"
        item["ALL_GOOD"] = _compute_all_good(item["SPAN_EQUAL_TO_ENTITY_CANONICAL_FORM_USED"], "UNKNOWN")
        out_asps.append(_ensure_keys_and_fix_typos(item, is_canonical=False))

    out_canon: List[dict] = []
    for i, o in enumerate(canon or []):
        item = dict(o)
        span = str(item.get("span", ""))
        canon_used = str(item.get("entity_canonical_form_used", ""))
        canon_from_span = canon_map.get(span, "")
        span_equal = "YES" if canon_from_span and canon_from_span.strip().lower() == canon_used.strip().lower() else "NO"
        item["SPAN_EQUAL_TO_ENTITY_CANONICAL_FORM_USED"] = span_equal
        item["entity_variable_meaning"] = _guess_meaning_sentence(item, is_canonical=True)
        item["PATIENT_FACT_ENTAIL_ENTITY_VARIABLE_MEANING"] = "UNKNOWN"
        item["ALL_GOOD"] = _compute_all_good(span_equal, "UNKNOWN")
        out_canon.append(_ensure_keys_and_fix_typos(item, is_canonical=True))

        if not canon_from_span:
            errors.append({
                "invariant": "SPAN→CANON-MAP",
                "problem": f"Span {span!r} not found in canonical forms map; could not verify equality rigorously.",
                "decl_index": i,
            })

    return out_asps, out_canon, errors


# ---------------------------------------------------------------------------
# Parser for model outputs (accepts angle tags or a JSON object)
# ---------------------------------------------------------------------------
def _parse_model_output(text: str) -> Dict[str, List[dict]]:
    m_asps = _VER_ASPS_RE.search(text or "")
    m_canon = _VER_CANON_RE.search(text or "")
    if m_asps or m_canon:
        asps = json.loads(_strip_json_comments_and_trailing_commas(m_asps.group(1))) if m_asps else []
        canon = json.loads(_strip_json_comments_and_trailing_commas(m_canon.group(1))) if m_canon else []
        return {
            "new_age_sex_pregnancystatus_declarations": asps,
            "new_canonical_variable_declarations": canon,
        }
    try:
        obj = _parse_json_relaxed(text or "{}")
    except Exception as e:
        raise ValueError(f"Could not parse verifier output as JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Verifier output must be a JSON object with the two arrays.")
    return {
        "new_age_sex_pregnancystatus_declarations": obj.get("new_age_sex_pregnancystatus_declarations", []) or [],
        "new_canonical_variable_declarations": obj.get("new_canonical_variable_declarations", []) or [],
    }


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class PatientSMTVariableCoderChecker(dspy.Module):
    """
    Calls an LLM with the strict verification prompt; parses/normalizes outputs;
    fills fallbacks when parsing fails; updates context with 'coder_verifier_results'.

    IMPORTANT: The LLM prompt template MUST be supplied via
    context["PatientSMTVariableCoderChecker_prompt"]. If missing/empty,
    the module will not call the engine and will use heuristic fallback.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, engine, *, log_dir: Optional[str] = None, prompt_tmpl: Optional[str] = None):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./verifier_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        if prompt_tmpl is not None:
            warnings.warn(
                "PatientSMTVariableCoderChecker: 'prompt_tmpl' is ignored. "
                "Provide the prompt via context['PatientSMTVariableCoderChecker_prompt'] instead.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "").strip() or default)
        except Exception:
            return default

    def _get_requirement_text(self, context: Dict, idx: int) -> str:
        reqs = context.get("requirements", []) or []
        if not reqs:
            return str(context.get("requirement_text", "")) or ""
        entry = reqs[idx]
        if isinstance(entry, dict):
            return str(entry.get("requirement") or entry.get("text") or "")
        return str(entry)

    @staticmethod
    def _build_prompt(prompt_tmpl: str, patient_fact: str, canon_map: Dict[str, str], declared_vars: Dict[str, List[dict]]) -> str:
        return (
            (prompt_tmpl or "")
            .replace("#PATIENT_FACT#", patient_fact)
            .replace("#CANONICAL_FORMS#", json.dumps(canon_map, ensure_ascii=False, indent=2))
            .replace("#Declared_SMT_VARIABLES#", json.dumps(declared_vars, ensure_ascii=False, indent=2))
        )

    def forward(self, context: Dict) -> Dict:  # type: ignore[override]
        idx: int = int(context.get("current_requirement_index", 0))
        note_id: str = str(context.get("note_id", "unknown_patient_note") or "unknown_patient_note")
        base_dir = os.path.join(self.log_dir, note_id)
        os.makedirs(base_dir, exist_ok=True)

        base_name = f"req{idx:03d}_verifier"
        p_log = os.path.join(base_dir, f"{base_name}_prompt.txt")
        r_log = os.path.join(base_dir, f"{base_name}_raw.txt")
        plan_log = os.path.join(base_dir, f"{base_name}_plan.json")

        # Gather inputs
        patient_fact = self._get_requirement_text(context, idx)
        canon_map = _build_canonical_forms_map(context, idx)
        declared_vars = _collect_declared_variables(context)

        if not (declared_vars["new_age_sex_pregnancystatus_declarations"] or declared_vars["new_canonical_variable_declarations"]):
            _log("verifier ⚠", idx, "no declared variables found; nothing to verify")
            context["coder_verifier_results"] = {
                "new_age_sex_pregnancystatus_declarations": [],
                "new_canonical_variable_declarations": [],
                "errors": [{"problem": "No declared variables to verify."}],
            }
            return context

        # Prompt MUST come from context
        prompt_tmpl: str = str(context.get("PatientSMTVariableCoderChecker_prompt") or "").strip()
        use_fallback_due_to_missing_prompt = not prompt_tmpl

        if use_fallback_due_to_missing_prompt:
            _log("verifier ⚠", idx, "no prompt template in context['PatientSMTVariableCoderChecker_prompt']; using heuristic fallback")
            with open(p_log, "w", encoding="utf-8") as fh:
                fh.write("/* NO PROMPT PROVIDED IN CONTEXT. Skipping LLM verification and using heuristic fallback. */\n")
        else:
            prompt = self._build_prompt(prompt_tmpl, patient_fact, canon_map, declared_vars)
            with open(p_log, "w", encoding="utf-8") as fh:
                fh.write(prompt)

        _log("verifier", idx, f"declared vars: ASPS={len(declared_vars['new_age_sex_pregnancystatus_declarations'])}, CANON={len(declared_vars['new_canonical_variable_declarations'])}")

        # Prepare outputs
        max_attempts = self._env_int("SMT_VERIFIER_MAX_ATTEMPTS", self.MAX_ATTEMPTS)
        last_error: Optional[Exception] = None
        succeeded = False

        results_asps: List[dict] = []
        results_canon: List[dict] = []
        errors: List[dict] = []

        if use_fallback_due_to_missing_prompt:
            # Directly take heuristic path
            results_asps, results_canon, fb_errs = _fallback_verify_lists(
                declared_vars["new_age_sex_pregnancystatus_declarations"],
                declared_vars["new_canonical_variable_declarations"],
                canon_map,
            )
            errors.extend([{
                "invariant": "FALLBACK",
                "problem": "Missing context['PatientSMTVariableCoderChecker_prompt']; used heuristic verifier.",
            }] + fb_errs)
        else:
            for attempt in range(1, max_attempts + 1):
                try:
                    llm_out: str = self.engine(prompt)[0]
                except Exception as e:
                    last_error = e
                    with open(r_log, "a", encoding="utf-8") as fh:
                        fh.write(f"\n--- attempt {attempt} ENGINE ERROR ---\n{repr(e)}\n")
                    _log("verifier", idx, f"engine error; retry ({attempt}/{max_attempts})")
                    continue

                with open(r_log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")

                try:
                    parsed = _parse_model_output(llm_out)
                    asps = parsed.get("new_age_sex_pregnancystatus_declarations", []) or []
                    canon = parsed.get("new_canonical_variable_declarations", []) or []

                    results_asps = [_ensure_keys_and_fix_typos(x, is_canonical=False) for x in asps]
                    results_canon = [_ensure_keys_and_fix_typos(x, is_canonical=True) for x in canon]
                    succeeded = True
                    break
                except Exception as e:
                    last_error = e
                    _log("verifier", idx, f"parse error; retry ({attempt}/{max_attempts})")

            if not succeeded:
                _log("verifier ⚠ fallback", idx, f"entering fallback due to: {last_error!r}")
                warnings.warn(f"SMT verifier fallback engaged (req#{idx}): {last_error}", RuntimeWarning)
                results_asps, results_canon, fb_errs = _fallback_verify_lists(
                    declared_vars["new_age_sex_pregnancystatus_declarations"],
                    declared_vars["new_canonical_variable_declarations"],
                    canon_map,
                )
                errors.extend([{
                    "invariant": "FALLBACK",
                    "problem": "LLM outputs could not be parsed/validated after retries; used heuristic verifier.",
                    "detail": str(last_error) if last_error else "unknown",
                }] + fb_errs)

        # Persist plan
        plan_obj = {
            "patient_fact": patient_fact,
            "canonical_forms_map": canon_map,
            "new_age_sex_pregnancystatus_declarations": results_asps,
            "new_canonical_variable_declarations": results_canon,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }
        with open(plan_log, "w", encoding="utf-8") as fh:
            json.dump(plan_obj, fh, indent=2, ensure_ascii=False)

        # Update context
        context["coder_verifier_results"] = plan_obj
        _log("verifier ✓", idx, f"verified {len(results_asps)} demographics + {len(results_canon)} canonical declarations")
        return context


# ---------------------------------------------------------------------------
# Convenience: direct function use (optional)
# ---------------------------------------------------------------------------
def run_verifier_once(
    engine,
    patient_fact: str,
    canonical_forms_map: Dict[str, str],
    declared_smt_variables: Dict[str, List[dict]],
    *,
    log_dir: str = "./verifier_logs",
    prompt_tmpl: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Utility for ad-hoc verification outside the pipeline.

    NOTE: The prompt is ONLY honored via context['PatientSMTVariableCoderChecker_prompt'].
    Provide it with 'prompt_tmpl' if desired; otherwise heuristic fallback will be used.
    """
    module = PatientSMTVariableCoderChecker(engine, log_dir=log_dir)
    context: Dict[str, Any] = {
        "requirements": [{"requirement": patient_fact}],
        "current_requirement_index": 0,
        "declared_smt_variables": declared_smt_variables,
        "valid_entities_by_req": {"0": {}},
        # Inject canonical mapping via bundles to bypass valid_entities_by_req
        "requirement_bundles": [{
            "req_index": 0,
            "entities": [{"span": k, "entity_canonical_form": v} for k, v in (canonical_forms_map or {}).items()],
        }],
    }
    if prompt_tmpl:
        context["PatientSMTVariableCoderChecker_prompt"] = prompt_tmpl
    return module(context)["coder_verifier_results"]  # type: ignore
