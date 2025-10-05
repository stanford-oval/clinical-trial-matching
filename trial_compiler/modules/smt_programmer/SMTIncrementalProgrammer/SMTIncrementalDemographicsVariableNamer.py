from __future__ import annotations
"""
SMTIncrementalDemographicsVariableNamer
──────────────────────────────────────
Declares **demographic** variables only (age / sex / pregnancy / CBP/lactation/menopause/transition/infertility/
inpatient/outpatient/age-groups/obstetric-status/care-setting) for ONE requirement.
Does NOT touch disease/procedure/etc. entities.

LLM must output exactly:

<new_age_sex_pregnancystatus_declarations>
[
  {
    "span": "...",
    "variable_meaning": "...",
    "usage_description": "...",
    "template": "<concrete template name>",     # MUST be concrete (e.g., patient_is_pregnant_now)
    "timeframe": "<now|inthehistory|inthefuture|inthepast{n}{units}|inthefuture{n}{units}|foradurationof{n}{units}>",
    "entity_variable_name": "<stem>"            # MUST include the concrete timeframe token
  }
]
</new_age_sex_pregnancystatus_declarations>

Writes to context:
  - new_age_sex_pregnancystatus_declarations: List[dict]
  - age_sex_preg_stage: "demographics_only"
  - age_sex_preg_errors: List[dict]

Allowed stems/templates (MUST be **concrete**, new style only):
  - patient_age_value_recorded_<timeframe>_in_years
  - patient_age_value_recorded_<timeframe>_in_months
  - patient_age_value_recorded_<timeframe>_in_days
  - patient_sex_is_<male|female|other>_<timeframe>
  - patient_is_pregnant_<timeframe>
  - patient_is_able_to_be_pregnant_<timeframe>
  - patient_has_childbearing_potential_<timeframe>
  - patient_is_breastfeeding_<timeframe>
  - patient_is_lactating_<timeframe>
  - patient_is_postmenopausal_<timeframe>
  - patient_is_in_transition_to_<timeframe>
  - patient_is_infertile_<timeframe>
  - patient_is_inpatient_<timeframe>
  - patient_is_outpatient_<timeframe>
  - patient_has_been_inpatient_<timeframe>
  - patient_has_been_outpatient_<timeframe>
  - patient_is_child_<timeframe>
  - patient_is_adolescent_<timeframe>
  - patient_is_adult_<timeframe>
  - patient_is_middle_aged_<timeframe>
  - patient_is_older_adult_<timeframe>
  - patient_is_neonate_<timeframe>
  - patient_is_toddler_<timeframe>
  - patient_is_preschooler_<timeframe>
  - patient_is_school_aged_<timeframe>
  - patient_is_premenopausal_<timeframe>
  - patient_is_perimenopausal_<timeframe>
  - patient_is_postpartum_<timeframe>
  - patient_is_postabortion_<timeframe>
  - patient_is_emergency_department_patient_<timeframe>
  - patient_is_long_term_care_resident_<timeframe>
  - patient_is_nursing_home_resident_<timeframe>
  - patient_is_assisted_living_resident_<timeframe>
  - patient_was_born_preterm_<timeframe>
  - patient_was_born_at_term_<timeframe>
  - patient_was_born_postterm_<timeframe>
  - patient_has_history_of_preterm_delivery_<timeframe>
  - patient_has_history_of_term_delivery_<timeframe>
  - patient_has_history_of_postterm_delivery_<timeframe>
  - patient_has_history_of_recurrent_preterm_delivery_<timeframe>
  - patient_has_history_of_preterm_premature_rupture_of_membranes_<timeframe>
  - patient_has_history_of_stillbirth_<timeframe>
  - patient_has_history_of_spontaneous_abortion_<timeframe>

Auto-remedy:
  - If name misses timeframe → append _<timeframe>.
  - If AGE uses old order ..._in_{unit}_{tf} → rewrite to ..._{tf}_in_{unit}.
  - Rename legacy 'patient_has_potential_to_be_pregnant_{tf}' → 'patient_has_childbearing_potential_{tf}'.
  - Sync template to match the category/unit inferred from the name (CONCRETE).
  - If timeframe field disagrees with name token → set field = token in name.

Zero-output policy:
  - If model returns [] or all items invalid, write an empty plan with no errors.
"""

from typing import Any, Dict, List, Optional, Tuple
import json, os, re, logging, sys, warnings
import dspy

try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [demog-namer] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)

# Helpers from namer_checks
from .namer_checks import (
    _extract_declared_symbols,
    _schema_log,
)

# ───────────────────────── parsing helpers ─────────────────────────
_NEWASPS_RE = re.compile(
    r"<new_age_sex_pregnancystatus_declarations>\s*(\[[\s\S]*?\])\s*</new_age_sex_pregnancystatus_declarations>",
    re.IGNORECASE,
)

# timeframe field validation (closed vocabulary)
_TF_TOKEN = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years))"
)
_TIMEFRAME_RE = re.compile(r"^" + _TF_TOKEN + r"$")   # 字段校验用（整段锚定）

def _to_snake(s: Any) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# ───────────────────────── naming regexes (new style + remedy hooks) ─────────────────────────
# AGE
_AGE_NAME_RE = re.compile(
    r"^patient_age_value_recorded_(?P<tf>" + _TF_TOKEN + r")_in_(?P<u>years|months|days)$"
)
_AGE_OLD_ORDER_RE = re.compile(
    r"^patient_age_value_recorded_in_(?P<u>years|months|days)_(?P<tf>" + _TF_TOKEN + r")$"
)
_AGE_WITHOUT_TF_RE = re.compile(r"^patient_age_value_recorded_in_(?P<u>years|months|days)$")

# SEX
_SEX_NAME_RE = re.compile(r"^patient_sex_is_(?P<sex>male|female|other)_(?P<tf>" + _TF_TOKEN + r")$")
_SEX_WITHOUT_TF_RE = re.compile(r"^patient_sex_is_(male|female|other)$")

# PREG / ABLE / CBP / BF / LAC
_PREG_NAME_RE = re.compile(r"^patient_is_pregnant_(?P<tf>" + _TF_TOKEN + r")$")
_PREG_WITHOUT_TF_RE = re.compile(r"^patient_is_pregnant$")

_ABLE_PREG_NAME_RE = re.compile(r"^patient_is_able_to_be_pregnant_(?P<tf>" + _TF_TOKEN + r")$")
_ABLE_PREG_WITHOUT_TF_RE = re.compile(r"^patient_is_able_to_be_pregnant$")

_CBP_NAME_RE = re.compile(r"^patient_has_childbearing_potential_(?P<tf>" + _TF_TOKEN + r")$")
_CBP_WITHOUT_TF_RE = re.compile(r"^patient_has_childbearing_potential$")

_CBP_LEGACY_RE = re.compile(r"^patient_has_potential_to_be_pregnant_(?P<tf>" + _TF_TOKEN + r")$")
_CBP_LEGACY_WITHOUT_TF_RE = re.compile(r"^patient_has_potential_to_be_pregnant$")

_BF_NAME_RE = re.compile(r"^patient_is_breastfeeding_(?P<tf>" + _TF_TOKEN + r")$")
_BF_WITHOUT_TF_RE = re.compile(r"^patient_is_breastfeeding$")

_LAC_NAME_RE = re.compile(r"^patient_is_lactating_(?P<tf>" + _TF_TOKEN + r")$")
_LAC_WITHOUT_TF_RE = re.compile(r"^patient_is_lactating$")

# Menopause & transition & infertility
_POSTMENO_NAME_RE = re.compile(r"^patient_is_postmenopausal_(?P<tf>" + _TF_TOKEN + r")$")
_POSTMENO_WITHOUT_TF_RE = re.compile(r"^patient_is_postmenopausal$")

_TRANSITION_NAME_RE = re.compile(r"^patient_is_in_transition_to_(?P<tf>" + _TF_TOKEN + r")$")
_TRANSITION_WITHOUT_TF_RE = re.compile(r"^patient_is_in_transition_to$")

_INFERTILE_NAME_RE = re.compile(r"^patient_is_infertile_(?P<tf>" + _TF_TOKEN + r")$")
_INFERTILE_WITHOUT_TF_RE = re.compile(r"^patient_is_infertile$")

# Patient setting: inpatient/outpatient + history
_INPATIENT_NAME_RE = re.compile(r"^patient_is_inpatient_(?P<tf>" + _TF_TOKEN + r")$")
_INPATIENT_WITHOUT_TF_RE = re.compile(r"^patient_is_inpatient$")

_OUTPATIENT_NAME_RE = re.compile(r"^patient_is_outpatient_(?P<tf>" + _TF_TOKEN + r")$")
_OUTPATIENT_WITHOUT_TF_RE = re.compile(r"^patient_is_outpatient$")

_HAS_BEEN_INPATIENT_NAME_RE = re.compile(r"^patient_has_been_inpatient_(?P<tf>" + _TF_TOKEN + r")$")
_HAS_BEEN_INPATIENT_WITHOUT_TF_RE = re.compile(r"^patient_has_been_inpatient$")

_HAS_BEEN_OUTPATIENT_NAME_RE = re.compile(r"^patient_has_been_outpatient_(?P<tf>" + _TF_TOKEN + r")$")
_HAS_BEEN_OUTPATIENT_WITHOUT_TF_RE = re.compile(r"^patient_has_been_outpatient$")

# Age-category groups
_CHILD_NAME_RE = re.compile(r"^patient_is_child_(?P<tf>" + _TF_TOKEN + r")$")
_CHILD_WITHOUT_TF_RE = re.compile(r"^patient_is_child$")

_ADOLESCENT_NAME_RE = re.compile(r"^patient_is_adolescent_(?P<tf>" + _TF_TOKEN + r")$")
_ADOLESCENT_WITHOUT_TF_RE = re.compile(r"^patient_is_adolescent$")

_ADULT_NAME_RE = re.compile(r"^patient_is_adult_(?P<tf>" + _TF_TOKEN + r")$")
_ADULT_WITHOUT_TF_RE = re.compile(r"^patient_is_adult$")

_MIDDLE_AGED_NAME_RE = re.compile(r"^patient_is_middle_aged_(?P<tf>" + _TF_TOKEN + r")$")
_MIDDLE_AGED_WITHOUT_TF_RE = re.compile(r"^patient_is_middle_aged$")

_OLDER_ADULT_NAME_RE = re.compile(r"^patient_is_older_adult_(?P<tf>" + _TF_TOKEN + r")$")
_OLDER_ADULT_WITHOUT_TF_RE = re.compile(r"^patient_is_older_adult$")

_NEONATE_NAME_RE = re.compile(r"^patient_is_neonate_(?P<tf>" + _TF_TOKEN + r")$")
_NEONATE_WITHOUT_TF_RE = re.compile(r"^patient_is_neonate$")

_TODDLER_NAME_RE = re.compile(r"^patient_is_toddler_(?P<tf>" + _TF_TOKEN + r")$")
_TODDLER_WITHOUT_TF_RE = re.compile(r"^patient_is_toddler$")

_PRESCHOOLER_NAME_RE = re.compile(r"^patient_is_preschooler_(?P<tf>" + _TF_TOKEN + r")$")
_PRESCHOOLER_WITHOUT_TF_RE = re.compile(r"^patient_is_preschooler$")

_SCHOOL_AGED_NAME_RE = re.compile(r"^patient_is_school_aged_(?P<tf>" + _TF_TOKEN + r")$")
_SCHOOL_AGED_WITHOUT_TF_RE = re.compile(r"^patient_is_school_aged$")

# Reproductive life stages
_PREMENOPAUSAL_NAME_RE = re.compile(r"^patient_is_premenopausal_(?P<tf>" + _TF_TOKEN + r")$")
_PREMENOPAUSAL_WITHOUT_TF_RE = re.compile(r"^patient_is_premenopausal$")

_PERIMENOPAUSAL_NAME_RE = re.compile(r"^patient_is_perimenopausal_(?P<tf>" + _TF_TOKEN + r")$")
_PERIMENOPAUSAL_WITHOUT_TF_RE = re.compile(r"^patient_is_perimenopausal$")

_POSTPARTUM_NAME_RE = re.compile(r"^patient_is_postpartum_(?P<tf>" + _TF_TOKEN + r")$")
_POSTPARTUM_WITHOUT_TF_RE = re.compile(r"^patient_is_postpartum$")

_POSTABORTION_NAME_RE = re.compile(r"^patient_is_postabortion_(?P<tf>" + _TF_TOKEN + r")$")
_POSTABORTION_WITHOUT_TF_RE = re.compile(r"^patient_is_postabortion$")

# Additional care settings / residence
_ED_PATIENT_NAME_RE = re.compile(r"^patient_is_emergency_department_patient_(?P<tf>" + _TF_TOKEN + r")$")
_ED_PATIENT_WITHOUT_TF_RE = re.compile(r"^patient_is_emergency_department_patient$")

_LT_CARE_RESIDENT_NAME_RE = re.compile(r"^patient_is_long_term_care_resident_(?P<tf>" + _TF_TOKEN + r")$")
_LT_CARE_RESIDENT_WITHOUT_TF_RE = re.compile(r"^patient_is_long_term_care_resident$")

_NH_RESIDENT_NAME_RE = re.compile(r"^patient_is_nursing_home_resident_(?P<tf>" + _TF_TOKEN + r")$")
_NH_RESIDENT_WITHOUT_TF_RE = re.compile(r"^patient_is_nursing_home_resident$")

_AL_RESIDENT_NAME_RE = re.compile(r"^patient_is_assisted_living_resident_(?P<tf>" + _TF_TOKEN + r")$")
_AL_RESIDENT_WITHOUT_TF_RE = re.compile(r"^patient_is_assisted_living_resident$")

# Obstetric history / birth history
_PRETERM_BORN_NAME_RE = re.compile(
    r"^patient_was_born_preterm_(?P<tf>" + _TF_TOKEN + r")$"
)
_PRETERM_BORN_WITHOUT_TF_RE = re.compile(r"^patient_was_born_preterm$")

_TERM_BORN_NAME_RE = re.compile(
    r"^patient_was_born_at_term_(?P<tf>" + _TF_TOKEN + r")$"
)
_TERM_BORN_WITHOUT_TF_RE = re.compile(r"^patient_was_born_at_term$")

_POSTTERM_BORN_NAME_RE = re.compile(
    r"^patient_was_born_postterm_(?P<tf>" + _TF_TOKEN + r")$"
)
_POSTTERM_BORN_WITHOUT_TF_RE = re.compile(r"^patient_was_born_postterm$")

_PRETERM_DELIV_NAME_RE = re.compile(
    r"^patient_has_history_of_preterm_delivery_(?P<tf>" + _TF_TOKEN + r")$"
)
_PRETERM_DELIV_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_preterm_delivery$"
)

_TERM_DELIV_NAME_RE = re.compile(
    r"^patient_has_history_of_term_delivery_(?P<tf>" + _TF_TOKEN + r")$"
)
_TERM_DELIV_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_term_delivery$"
)

_POSTTERM_DELIV_NAME_RE = re.compile(
    r"^patient_has_history_of_postterm_delivery_(?P<tf>" + _TF_TOKEN + r")$"
)
_POSTTERM_DELIV_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_postterm_delivery$"
)

_REC_PRETERM_DELIV_NAME_RE = re.compile(
    r"^patient_has_history_of_recurrent_preterm_delivery_(?P<tf>" + _TF_TOKEN + r")$"
)
_REC_PRETERM_DELIV_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_recurrent_preterm_delivery$"
)

_PPROM_HX_NAME_RE = re.compile(
    r"^patient_has_history_of_preterm_premature_rupture_of_membranes_(?P<tf>" + _TF_TOKEN + r")$"
)
_PPROM_HX_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_preterm_premature_rupture_of_membranes$"
)

_STILLBIRTH_HX_NAME_RE = re.compile(
    r"^patient_has_history_of_stillbirth_(?P<tf>" + _TF_TOKEN + r")$"
)
_STILLBIRTH_HX_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_stillbirth$"
)

_SAB_HX_NAME_RE = re.compile(
    r"^patient_has_history_of_spontaneous_abortion_(?P<tf>" + _TF_TOKEN + r")$"
)
_SAB_HX_WITHOUT_TF_RE = re.compile(
    r"^patient_has_history_of_spontaneous_abortion$"
)

# ───────────────────────── concrete template patterns ─────────────────────────
_TEMPLATE_PATTERNS = [
    # Age numeric
    re.compile(r"^patient_age_value_recorded_" + _TF_TOKEN + r"_in_years$"),
    re.compile(r"^patient_age_value_recorded_" + _TF_TOKEN + r"_in_months$"),
    re.compile(r"^patient_age_value_recorded_" + _TF_TOKEN + r"_in_days$"),

    # Sex
    re.compile(r"^patient_sex_is_(?:male|female|other)_" + _TF_TOKEN + r"$"),

    # Core families
    re.compile(r"^patient_is_pregnant_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_able_to_be_pregnant_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_childbearing_potential_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_breastfeeding_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_lactating_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_postmenopausal_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_in_transition_to_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_infertile_" + _TF_TOKEN + r"$"),

    # Patient setting: current + history
    re.compile(r"^patient_is_inpatient_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_outpatient_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_been_inpatient_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_been_outpatient_" + _TF_TOKEN + r"$"),

    # Age categories
    re.compile(r"^patient_is_child_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_adolescent_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_adult_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_middle_aged_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_older_adult_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_neonate_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_toddler_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_preschooler_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_school_aged_" + _TF_TOKEN + r"$"),

    # Reproductive stages / obstetric status
    re.compile(r"^patient_is_premenopausal_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_perimenopausal_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_postpartum_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_postabortion_" + _TF_TOKEN + r"$"),

    # Care setting / residence
    re.compile(r"^patient_is_emergency_department_patient_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_long_term_care_resident_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_nursing_home_resident_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_is_assisted_living_resident_" + _TF_TOKEN + r"$"),

    # Birth history
    re.compile(r"^patient_was_born_preterm_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_was_born_at_term_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_was_born_postterm_" + _TF_TOKEN + r"$"),

    # Obstetric history
    re.compile(r"^patient_has_history_of_preterm_delivery_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_history_of_term_delivery_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_history_of_postterm_delivery_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_history_of_recurrent_preterm_delivery_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_history_of_preterm_premature_rupture_of_membranes_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_history_of_stillbirth_" + _TF_TOKEN + r"$"),
    re.compile(r"^patient_has_history_of_spontaneous_abortion_" + _TF_TOKEN + r"$"),
]

def _template_ok(t: str) -> bool:
    t = str(t or "")
    return any(p.match(t) for p in _TEMPLATE_PATTERNS)

def _age_expected_template(unit: str, tf: str) -> str:
    return f"patient_age_value_recorded_{tf}_in_{unit}"

# ───────────────────────── all_linked_variales helper ─────────────────────────
def _extend_all_linked_variales(context: Dict[str, Any], items: List[Dict[str, Any]], *, idx: int) -> None:
    arr = context.get("all_linked_variales")
    if not isinstance(arr, list):
        arr = []
    existing: set = set()
    for e in arr:
        if isinstance(e, dict):
            n = e.get("entity_variable_name")
            if isinstance(n, str):
                existing.add(n)
        elif isinstance(e, str):
            existing.add(e)
    added = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = it.get("entity_variable_name")
        if isinstance(name, str) and name not in existing:
            arr.append(it); existing.add(name); added += 1
    context["all_linked_variales"] = arr
    if added:
        _log("demog→link ✓", idx, f"added {added} demog vars (total linked: {len(arr)})")

# ───────────────────────── normalization & auto-remedy ─────────────────────────
def _normalize_and_autofix_item(
    item: Dict[str, Any], *, diag: List[Dict[str, Any]], ctx_idx: int
) -> Dict[str, Any] | None:
    """Normalize fields and apply *minimal but effective* auto-remedies aligned with the concrete templates."""
    if not isinstance(item, dict):
        _schema_log(diag, idx=ctx_idx, block="new_age_sex_pregnancystatus_declarations", item_index=None,
                    code="TYPE", message="Item must be an object/dict.")
        return None

    it = dict(item)  # copy

    # Basic fields
    it.setdefault("span", "")
    it.setdefault("variable_meaning", "")
    it.setdefault("usage_description", "")

    template = str(it.get("template") or "").strip()
    timeframe = str(it.get("timeframe") or "").strip()
    name_raw = it.get("entity_variable_name")

    # NEW: keep variable_declaration from raw if present (string)
    var_decl_raw = it.get("variable_declaration")
    if isinstance(var_decl_raw, str) and var_decl_raw.strip():
        it["variable_declaration"] = var_decl_raw.strip()

    # Required fields check
    missing = []
    if not isinstance(name_raw, str) or not name_raw.strip(): missing.append("entity_variable_name")
    if not template: missing.append("template")
    if not timeframe: missing.append("timeframe")
    if missing:
        _schema_log(diag, idx=ctx_idx, block="new_age_sex_pregnancystatus_declarations", item_index=None,
                    code="MISSING", message=f"Missing fields: {', '.join(missing)}")
        return None

    # timeframe field grammar
    if not _TIMEFRAME_RE.match(timeframe):
        _schema_log(diag, idx=ctx_idx, block="timeframe", item_index=None,
                    code="TIMEFRAME", message="Invalid timeframe value.", value=timeframe)
        return None

    # --- Resolve literal timeframe placeholders in NAME before normalization ---
    if isinstance(name_raw, str) and name_raw.strip():
        if "{timeframe}" in name_raw:
            name_fixed = name_raw.replace("{timeframe}", timeframe)
            _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                        code="SANITIZE_TF_PLACEHOLDER_RESOLVED",
                        message="Resolved '{timeframe}' placeholder in name using timeframe field.",
                        from_name=name_raw, to_name=name_fixed, timeframe=timeframe)
            name_raw = name_fixed
            it["entity_variable_name"] = name_raw

        # Handle lingering '_timeframe' token pre-normalization (common after brace loss)
        name_fixed2 = re.sub(r"(?<=_)timeframe\b", timeframe, name_raw)
        if name_fixed2 != name_raw:
            _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                        code="SANITIZE_TF_TOKEN_RESOLVED",
                        message="Resolved '_timeframe' token in name using timeframe field.",
                        from_name=name_raw, to_name=name_fixed2, timeframe=timeframe)
            name_raw = name_fixed2
            it["entity_variable_name"] = name_raw

    # Normalize entity name to snake_case
    name_norm = _to_snake(name_raw)
    if name_norm != name_raw:
        _schema_log(diag, idx=ctx_idx, block="normalize", item_index=None,
                    code="NAME_NORMALIZED",
                    message="entity_variable_name normalized to snake_case",
                    original=name_raw, normalized=name_norm)
    name = name_norm

    # ── Auto-remedy A: AGE old-order → new-order
    m_old_age = _AGE_OLD_ORDER_RE.match(name)
    if m_old_age:
        u  = m_old_age.group("u")
        tf = m_old_age.group("tf")
        new_name = f"patient_age_value_recorded_{tf}_in_{u}"
        new_tmpl = _age_expected_template(u, tf)
        _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                    code="SANITIZE_AGE_ORDER",
                    message="Rewrote age old-order name to new-order.",
                    from_name=name, to_name=new_name)
        name = new_name
        it["entity_variable_name"] = name
        it["timeframe"] = tf
        it["template"] = new_tmpl
        timeframe = tf
        template = new_tmpl

    # ── Auto-remedy B: AGE missing timeframe → append
    if _AGE_WITHOUT_TF_RE.match(name):
        u = _AGE_WITHOUT_TF_RE.match(name).group("u")  # type: ignore
        new_name = f"patient_age_value_recorded_{timeframe}_in_{u}"
        new_tmpl = _age_expected_template(u, timeframe)
        _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                    code="SANITIZE_AGE_APPEND_TF",
                    message="Appended timeframe to age name.",
                    from_name=name, to_name=new_name, timeframe=timeframe)
        name = new_name
        it["entity_variable_name"] = name
        it["template"] = new_tmpl
        template = new_tmpl

    # ── Auto-remedy C: legacy CBP → new CBP
    m_legacy_cbp = _CBP_LEGACY_RE.match(name)
    if m_legacy_cbp:
        tf = m_legacy_cbp.group("tf")
        new_name = f"patient_has_childbearing_potential_{tf}"
        new_tmpl = f"patient_has_childbearing_potential_{tf}"
        _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                    code="SANITIZE_CBP_RENAME",
                    message="Renamed legacy CBP to childbearing_potential.",
                    from_name=name, to_name=new_name)
        name = new_name
        it["entity_variable_name"] = name
        it["timeframe"] = tf
        it["template"] = new_tmpl
        timeframe = tf
        template = new_tmpl

    if _CBP_LEGACY_WITHOUT_TF_RE.match(name):
        new_name = f"patient_has_childbearing_potential_{timeframe}"
        new_tmpl = f"patient_has_childbearing_potential_{timeframe}"
        _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                    code="SANITIZE_CBP_LEGACY_APPEND_TF",
                    message="Legacy CBP without timeframe → appended and renamed.",
                    from_name=name, to_name=new_name, timeframe=timeframe)
        name = new_name
        it["entity_variable_name"] = name
        it["template"] = new_tmpl
        template = new_tmpl

    # ── Auto-remedy D: missing timeframe on other families → append
    def _append_tf_if(pattern_without_tf, fam_label: str):
        nonlocal name
        if pattern_without_tf.match(name):
            new_name = f"{name}_{timeframe}"
            _schema_log(diag, idx=ctx_idx, block="sanitize", item_index=None,
                        code=f"SANITIZE_{fam_label}_APPEND_TF",
                        message=f"Appended timeframe to {fam_label.lower()} name.",
                        from_name=name, to_name=new_name)
            name = new_name
            it["entity_variable_name"] = name

    _append_tf_if(_SEX_WITHOUT_TF_RE, "SEX")
    _append_tf_if(_PREG_WITHOUT_TF_RE, "PREG")
    _append_tf_if(_ABLE_PREG_WITHOUT_TF_RE, "ABLE")
    _append_tf_if(_CBP_WITHOUT_TF_RE, "CBP")
    _append_tf_if(_BF_WITHOUT_TF_RE, "BF")
    _append_tf_if(_LAC_WITHOUT_TF_RE, "LAC")
    _append_tf_if(_POSTMENO_WITHOUT_TF_RE, "POSTMENO")
    _append_tf_if(_TRANSITION_WITHOUT_TF_RE, "TRANSITION")
    _append_tf_if(_INFERTILE_WITHOUT_TF_RE, "INFERTILE")
    _append_tf_if(_INPATIENT_WITHOUT_TF_RE, "INPATIENT")
    _append_tf_if(_OUTPATIENT_WITHOUT_TF_RE, "OUTPATIENT")
    _append_tf_if(_HAS_BEEN_INPATIENT_WITHOUT_TF_RE, "HAS_BEEN_INPATIENT")
    _append_tf_if(_HAS_BEEN_OUTPATIENT_WITHOUT_TF_RE, "HAS_BEEN_OUTPATIENT")

    # Age groups
    _append_tf_if(_CHILD_WITHOUT_TF_RE, "CHILD")
    _append_tf_if(_ADOLESCENT_WITHOUT_TF_RE, "ADOLESCENT")
    _append_tf_if(_ADULT_WITHOUT_TF_RE, "ADULT")
    _append_tf_if(_MIDDLE_AGED_WITHOUT_TF_RE, "MIDDLE_AGED")
    _append_tf_if(_OLDER_ADULT_WITHOUT_TF_RE, "OLDER_ADULT")
    _append_tf_if(_NEONATE_WITHOUT_TF_RE, "NEONATE")
    _append_tf_if(_TODDLER_WITHOUT_TF_RE, "TODDLER")
    _append_tf_if(_PRESCHOOLER_WITHOUT_TF_RE, "PRESCHOOLER")
    _append_tf_if(_SCHOOL_AGED_WITHOUT_TF_RE, "SCHOOL_AGED")

    # Reproductive stages extra
    _append_tf_if(_PREMENOPAUSAL_WITHOUT_TF_RE, "PREMENOPAUSAL")
    _append_tf_if(_PERIMENOPAUSAL_WITHOUT_TF_RE, "PERIMENOPAUSAL")
    _append_tf_if(_POSTPARTUM_WITHOUT_TF_RE, "POSTPARTUM")
    _append_tf_if(_POSTABORTION_WITHOUT_TF_RE, "POSTABORTION")

    # Care/residence extras
    _append_tf_if(_ED_PATIENT_WITHOUT_TF_RE, "ED_PATIENT")
    _append_tf_if(_LT_CARE_RESIDENT_WITHOUT_TF_RE, "LT_CARE_RESIDENT")
    _append_tf_if(_NH_RESIDENT_WITHOUT_TF_RE, "NH_RESIDENT")
    _append_tf_if(_AL_RESIDENT_WITHOUT_TF_RE, "AL_RESIDENT")

    # Obstetric history / birth history extras
    _append_tf_if(_PRETERM_BORN_WITHOUT_TF_RE, "PRETERM_BORN")
    _append_tf_if(_TERM_BORN_WITHOUT_TF_RE, "TERM_BORN")
    _append_tf_if(_POSTTERM_BORN_WITHOUT_TF_RE, "POSTTERM_BORN")
    _append_tf_if(_PRETERM_DELIV_WITHOUT_TF_RE, "PRETERM_DELIVERY")
    _append_tf_if(_TERM_DELIV_WITHOUT_TF_RE, "TERM_DELIVERY")
    _append_tf_if(_POSTTERM_DELIV_WITHOUT_TF_RE, "POSTTERM_DELIVERY")
    _append_tf_if(_REC_PRETERM_DELIV_WITHOUT_TF_RE, "REC_PRETERM_DELIVERY")
    _append_tf_if(_PPROM_HX_WITHOUT_TF_RE, "PPROM_HISTORY")
    _append_tf_if(_STILLBIRTH_HX_WITHOUT_TF_RE, "STILLBIRTH_HISTORY")
    _append_tf_if(_SAB_HX_WITHOUT_TF_RE, "SAB_HISTORY")

    # ── Sync timeframe field to the token in name (final authority: NAME)
    def _sync_simple(match: re.Match, expected_prefix: str, fam: str):
        nonlocal timeframe, template
        tf_in_name = match.group("tf")
        if timeframe != tf_in_name:
            _schema_log(diag, idx=ctx_idx, block="timeframe", item_index=None,
                        code="TIMEFRAME_FIELD_CORRECTED",
                        message=f"timeframe field corrected to match name token ({fam})",
                        from_field=timeframe, to_field=tf_in_name)
        timeframe = tf_in_name
        it["timeframe"] = timeframe
        expected = f"{expected_prefix}_{timeframe}"
        if template != expected:
            _schema_log(diag, idx=ctx_idx, block="template", item_index=None,
                        code="TEMPLATE_FIXED",
                        message=f"Template corrected for {fam} (concrete).",
                        from_template=template, to_template=expected)
            template = expected
            it["template"] = expected

    # AGE (special: has unit)
    m_age = _AGE_NAME_RE.match(name)
    if m_age:
        tf_in_name = m_age.group("tf")
        u = m_age.group("u")
        if timeframe != tf_in_name:
            _schema_log(diag, idx=ctx_idx, block="timeframe", item_index=None,
                        code="TIMEFRAME_FIELD_CORRECTED",
                        message="timeframe field corrected to match name token (AGE)",
                        from_field=timeframe, to_field=tf_in_name)
        timeframe = tf_in_name
        it["timeframe"] = timeframe
        expected = _age_expected_template(u, timeframe)
        if template != expected:
            _schema_log(diag, idx=ctx_idx, block="template", item_index=None,
                        code="TEMPLATE_FIXED",
                        message="Template corrected to match age unit (concrete).",
                        from_template=template, to_template=expected)
            template = expected
            it["template"] = expected

    # SEX (special: includes sex)
    m_sex = _SEX_NAME_RE.match(name)
    if m_sex:
        tf_in_name = m_sex.group("tf")
        sex = m_sex.group("sex")
        if timeframe != tf_in_name:
            _schema_log(diag, idx=ctx_idx, block="timeframe", item_index=None,
                        code="TIMEFRAME_FIELD_CORRECTED",
                        message="timeframe field corrected to match name token (SEX)",
                        from_field=timeframe, to_field=tf_in_name)
        timeframe = tf_in_name
        it["timeframe"] = timeframe
        expected = f"patient_sex_is_{sex}_{timeframe}"
        if template != expected:
            _schema_log(diag, idx=ctx_idx, block="template", item_index=None,
                        code="TEMPLATE_FIXED",
                        message="Template corrected for SEX (concrete).",
                        from_template=template, to_template=expected)
            template = expected
            it["template"] = expected

    # Simple families (prefix, regex match, code)
    _simple_syncs = [
        ("patient_is_pregnant",          _PREG_NAME_RE,           "PREG"),
        ("patient_is_able_to_be_pregnant", _ABLE_PREG_NAME_RE,    "ABLE_PREG"),
        ("patient_has_childbearing_potential", _CBP_NAME_RE,      "CBP"),
        ("patient_is_breastfeeding",     _BF_NAME_RE,             "BF"),
        ("patient_is_lactating",         _LAC_NAME_RE,            "LAC"),
        ("patient_is_postmenopausal",    _POSTMENO_NAME_RE,       "POSTMENO"),
        ("patient_is_in_transition_to",  _TRANSITION_NAME_RE,     "TRANSITION"),
        ("patient_is_infertile",         _INFERTILE_NAME_RE,      "INFERTILE"),
        ("patient_is_inpatient",         _INPATIENT_NAME_RE,      "INPATIENT"),
        ("patient_is_outpatient",        _OUTPATIENT_NAME_RE,     "OUTPATIENT"),
        ("patient_has_been_inpatient",   _HAS_BEEN_INPATIENT_NAME_RE, "HAS_BEEN_INPATIENT"),
        ("patient_has_been_outpatient",  _HAS_BEEN_OUTPATIENT_NAME_RE, "HAS_BEEN_OUTPATIENT"),

        # Age categories
        ("patient_is_child",             _CHILD_NAME_RE,          "CHILD"),
        ("patient_is_adolescent",        _ADOLESCENT_NAME_RE,     "ADOLESCENT"),
        ("patient_is_adult",             _ADULT_NAME_RE,          "ADULT"),
        ("patient_is_middle_aged",       _MIDDLE_AGED_NAME_RE,    "MIDDLE_AGED"),
        ("patient_is_older_adult",       _OLDER_ADULT_NAME_RE,    "OLDER_ADULT"),
        ("patient_is_neonate",           _NEONATE_NAME_RE,        "NEONATE"),
        ("patient_is_toddler",           _TODDLER_NAME_RE,        "TODDLER"),
        ("patient_is_preschooler",       _PRESCHOOLER_NAME_RE,    "PRESCHOOLER"),
        ("patient_is_school_aged",       _SCHOOL_AGED_NAME_RE,    "SCHOOL_AGED"),

        # Reproductive/obstetric
        ("patient_is_premenopausal",     _PREMENOPAUSAL_NAME_RE,  "PREMENOPAUSAL"),
        ("patient_is_perimenopausal",    _PERIMENOPAUSAL_NAME_RE, "PERIMENOPAUSAL"),
        ("patient_is_postpartum",        _POSTPARTUM_NAME_RE,     "POSTPARTUM"),
        ("patient_is_postabortion",      _POSTABORTION_NAME_RE,   "POSTABORTION"),

        # Care/residence
        ("patient_is_emergency_department_patient", _ED_PATIENT_NAME_RE, "ED_PATIENT"),
        ("patient_is_long_term_care_resident", _LT_CARE_RESIDENT_NAME_RE, "LT_CARE_RESIDENT"),
        ("patient_is_nursing_home_resident", _NH_RESIDENT_NAME_RE, "NH_RESIDENT"),
        ("patient_is_assisted_living_resident", _AL_RESIDENT_NAME_RE, "AL_RESIDENT"),

        # Obstetric history / birth history
        ("patient_was_born_preterm", _PRETERM_BORN_NAME_RE, "PRETERM_BORN"),
        ("patient_was_born_at_term", _TERM_BORN_NAME_RE, "TERM_BORN"),
        ("patient_was_born_postterm", _POSTTERM_BORN_NAME_RE, "POSTTERM_BORN"),
        ("patient_has_history_of_preterm_delivery", _PRETERM_DELIV_NAME_RE, "PRETERM_DELIVERY"),
        ("patient_has_history_of_term_delivery", _TERM_DELIV_NAME_RE, "TERM_DELIVERY"),
        ("patient_has_history_of_postterm_delivery", _POSTTERM_DELIV_NAME_RE, "POSTTERM_DELIVERY"),
        ("patient_has_history_of_recurrent_preterm_delivery", _REC_PRETERM_DELIV_NAME_RE, "REC_PRETERM_DELIVERY"),
        ("patient_has_history_of_preterm_premature_rupture_of_membranes", _PPROM_HX_NAME_RE, "PPROM_HISTORY"),
        ("patient_has_history_of_stillbirth", _STILLBIRTH_HX_NAME_RE, "STILLBIRTH_HISTORY"),
        ("patient_has_history_of_spontaneous_abortion", _SAB_HX_NAME_RE, "SAB_HISTORY"),
    ]
    for prefix, regex, fam in _simple_syncs:
        m = regex.match(name)
        if m:
            _sync_simple(m, prefix, fam)

    # Finally, enforce that template matches the concrete pattern set
    if not _template_ok(template):
        _schema_log(diag, idx=ctx_idx, block="template", item_index=None,
                    code="TEMPLATE_DISALLOWED",
                    message="Template not in allowed pattern set after remedy.", template=template)
        return None

    it["entity_variable_name"] = name
    return it


def _validate_demog_list(
    raw: List[Any], *, declared_names: set, reusable_names: set,
    diag: List[Dict[str, Any]], ctx_idx: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (validated_items, rejected_items_with_reasons)
    """
    out: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen: set = set()

    for i, item in enumerate(raw or []):
        fixed = _normalize_and_autofix_item(item, diag=diag, ctx_idx=ctx_idx)
        if fixed is None:
            rejected.append({"index": i, "reason": "MALFORMED"})
            continue

        span = str(fixed.get("span") or "").strip()
        meaning = str(fixed.get("variable_meaning") or "").strip()
        usage = str(fixed.get("usage_description") or "").strip()
        template = str(fixed.get("template") or "").strip()
        timeframe = str(fixed.get("timeframe") or "").strip()
        name = str(fixed.get("entity_variable_name") or "").strip()

        # Final strict check against patterns (after remedies)
        name_ok = any([
            _AGE_NAME_RE.match(name),
            _SEX_NAME_RE.match(name),
            _PREG_NAME_RE.match(name),
            _ABLE_PREG_NAME_RE.match(name),
            _CBP_NAME_RE.match(name),
            _BF_NAME_RE.match(name),
            _LAC_NAME_RE.match(name),
            _POSTMENO_NAME_RE.match(name),
            _TRANSITION_NAME_RE.match(name),
            _INFERTILE_NAME_RE.match(name),
            _INPATIENT_NAME_RE.match(name),
            _OUTPATIENT_NAME_RE.match(name),
            _HAS_BEEN_INPATIENT_NAME_RE.match(name),
            _HAS_BEEN_OUTPATIENT_NAME_RE.match(name),
            _CHILD_NAME_RE.match(name),
            _ADOLESCENT_NAME_RE.match(name),
            _ADULT_NAME_RE.match(name),
            _MIDDLE_AGED_NAME_RE.match(name),
            _OLDER_ADULT_NAME_RE.match(name),
            _NEONATE_NAME_RE.match(name),
            _TODDLER_NAME_RE.match(name),
            _PRESCHOOLER_NAME_RE.match(name),
            _SCHOOL_AGED_NAME_RE.match(name),
            _PREMENOPAUSAL_NAME_RE.match(name),
            _PERIMENOPAUSAL_NAME_RE.match(name),
            _POSTPARTUM_NAME_RE.match(name),
            _POSTABORTION_NAME_RE.match(name),
            _ED_PATIENT_NAME_RE.match(name),
            _LT_CARE_RESIDENT_NAME_RE.match(name),
            _NH_RESIDENT_NAME_RE.match(name),
            _AL_RESIDENT_NAME_RE.match(name),
            _PRETERM_BORN_NAME_RE.match(name),
            _TERM_BORN_NAME_RE.match(name),
            _POSTTERM_BORN_NAME_RE.match(name),
            _PRETERM_DELIV_NAME_RE.match(name),
            _TERM_DELIV_NAME_RE.match(name),
            _POSTTERM_DELIV_NAME_RE.match(name),
            _REC_PRETERM_DELIV_NAME_RE.match(name),
            _PPROM_HX_NAME_RE.match(name),
            _STILLBIRTH_HX_NAME_RE.match(name),
            _SAB_HX_NAME_RE.match(name),
        ])
        tmpl_ok = _template_ok(template)

        if not (name_ok and tmpl_ok):
            _schema_log(
                diag, idx=ctx_idx, block="name_consistency", item_index=i,
                code="NAME_MISMATCH",
                message="entity_variable_name/template still fail after remedies.",
                entity_variable_name=name, timeframe=timeframe, template=template
            )
            rejected.append({"index": i, "name": name, "reason": "NAME_MISMATCH"})
            continue

        # Redeclaration guards
        if name in declared_names or name in reusable_names:
            _schema_log(diag, idx=ctx_idx, block="dedup", item_index=i,
                        code="REDECLARE_COLLAPSED",
                        message="Dropping re-declaration of existing/reusable symbol.",
                        symbol=name)
            rejected.append({"index": i, "name": name, "reason": "REDECLARE_COLLAPSED"})
            continue

        # Per-batch duplicates
        if name in seen:
            _schema_log(diag, idx=ctx_idx, block="dedup", item_index=i,
                        code="DUP_COLLAPSED", message="Duplicate item collapsed.", symbol=name)
            rejected.append({"index": i, "name": name, "reason": "DUP_COLLAPSED"})
            continue
        seen.add(name)

        rec = {
            "span": span,
            "variable_meaning": meaning,
            "usage_description": usage,
            "template": template,
            "timeframe": timeframe,
            "entity_variable_name": name,
        }
        var_decl = fixed.get("variable_declaration")
        if isinstance(var_decl, str) and var_decl.strip():
            rec["variable_declaration"] = var_decl.strip()
        out.append(rec)

    return out, rejected


class SMTIncrementalDemographicsVariableNamer(dspy.Module):
    MAX_ATTEMPTS = 3

    def __init__(self, engine, *, log_dir: Optional[str] = None):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _build_prompt(self, context: Dict[str, Any], idx: int) -> str:
        tpl = context.get("SMTIncrementalDemographicsVariableNamer_prompt", "")
        if not tpl:
            return ""
        req = context["requirements"][idx]
        requirement_txt = req.get("requirement") if isinstance(req, dict) else str(req)
        reusable_json = json.dumps(context.get("reusable_variables", []), ensure_ascii=False, indent=2)
        return (tpl
                .replace("#REQUIREMENT#", requirement_txt)
                .replace("#REUSABLE_VARIABLES#", reusable_json))

    def _augment_with_feedback(self, base_prompt: str, rejected: List[Dict[str, Any]]) -> str:
        """Append concise validation feedback to steer the LLM on retries."""
        hints = [
            "- Use ONLY concrete stems; examples:",
            "  patient_age_value_recorded_now_in_years | patient_sex_is_other_inthefuture30days |",
            "  patient_is_pregnant_now | patient_has_childbearing_potential_foradurationof12months |",
            "  patient_is_inpatient_now | patient_has_been_outpatient_inthepast2years |",
            "  patient_is_child_now | patient_is_middle_aged_now | patient_is_neonate_now |",
            "  patient_is_premenopausal_now | patient_is_postpartum_now |",
            "  patient_is_emergency_department_patient_now | patient_is_assisted_living_resident_now",
            "- Templates must be CONCRETE (include the actual timeframe), matching the name/timeframe field.",
            "- All names MUST include a single timeframe token that exactly matches the 'timeframe' field.",
            "- It's OK to output an empty array [] if no demographics are needed for this requirement.",
            "- Do not redeclare symbols already present or marked reusable.",
            "- Output ONLY the <new_age_sex_pregnancystatus_declarations> JSON array; no commentary.",
        ]
        if rejected:
            bad_names = [r.get("name") for r in rejected if r.get("name")]
            if bad_names:
                hints.append("- Problematic entries (fix or drop): " + ", ".join(sorted(set(bad_names))))
        fb = ("\n\n# VALIDATION FEEDBACK (fix and re-emit ONLY the corrected block)\n" + "\n".join(hints) + "\n")
        return base_prompt + fb

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        if not context.get("requirements"):
            return context
        idx: int = int(context["current_requirement_index"])

        prompt = self._build_prompt(context, idx)
        trial_id = context.get("trial_id", "unknown_trial")
        side = context.get("inc_exc", "unknown")

        # layout: <log_root>/<trial_id>/<inclusion|exclusion>/reqNNN/1demographics/
        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc", "inclusion", "include", "in"}:  return "inclusion"
            if s in {"exc", "exclusion", "exclude", "ex"}:  return "exclusion"
            return s or "unknown"
        stage_dir = os.path.join(self.log_dir, str(trial_id), _bucket(side), f"req{idx:03d}")
        os.makedirs(stage_dir, exist_ok=True)
        p_log    = os.path.join(stage_dir, "1demographics_prompt.txt")
        r_log    = os.path.join(stage_dir, "1demographics_raw.txt")
        plan_log = os.path.join(stage_dir, "1demographics_plan.json")

        if not prompt:
            _log("demog ✗", idx, "prompt template missing (SMTIncrementalDemographicsVariableNamer_prompt)")
            context["new_age_sex_pregnancystatus_declarations"] = []
            context["age_sex_preg_errors"] = [{"code": "PROMPT_MISSING"}]
            context["age_sex_preg_stage"] = "demographics_only"
            return context

        with open(p_log, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        declared_names = set(_extract_declared_symbols(context.get("smt_program_lines", [])))
        reusable_names = {
            (rv or {}).get("variable_name") for rv in (context.get("reusable_variables") or []) if isinstance(rv, dict)
        }
        errors: List[Dict[str, Any]] = []
        accepted: List[Dict[str, Any]] = []
        accepted_names: set = set()
        last_error: Optional[Exception] = None

        current_prompt = prompt

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            out: str = self.engine(current_prompt)[0]
            with open(r_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- attempt {attempt} ---\n{out}\n")

            m = _NEWASPS_RE.search(out)
            if not m:
                _schema_log(errors, idx=idx, block="top_level", item_index=None,
                            code="MISSING_BLOCK", message="Could not find <new_age_sex_pregnancystatus_declarations> JSON block.")
                last_error = RuntimeError("Missing <new_age_sex_pregnancystatus_declarations> block")
                current_prompt = self._augment_with_feedback(prompt, [])
                continue

            try:
                payload = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", m.group(1), flags=re.S)
                raw_list = json.loads(payload)

                # Zero-output is valid
                if isinstance(raw_list, list) and len(raw_list) == 0:
                    plan = {
                        "new_age_sex_pregnancystatus_declarations": accepted,
                        "stage": "demographics_only",
                        "errors": errors or [],
                    }
                    with open(plan_log, "w", encoding="utf-8") as fh:
                        json.dump(plan, fh, indent=2, ensure_ascii=False)
                    context["new_age_sex_pregnancystatus_declarations"] = accepted
                    context["age_sex_preg_errors"] = []
                    context["age_sex_preg_stage"] = "demographics_only"
                    _log("demog ✓", idx, "0 demographic variables (none required)")
                    return context

                validated, rejected = _validate_demog_list(
                    raw_list,
                    declared_names=declared_names,
                    reusable_names={n for n in reusable_names if n} | accepted_names,
                    diag=errors,
                    ctx_idx=idx,
                )

                for v in validated:
                    name = v.get("entity_variable_name")
                    if name and name not in accepted_names:
                        accepted.append(v); accepted_names.add(name)

                if raw_list and len(validated) == len(raw_list):
                    plan = {
                        "new_age_sex_pregnancystatus_declarations": accepted,
                        "stage": "demographics_only",
                        "errors": errors or [],
                    }
                    with open(plan_log, "w", encoding="utf-8") as fh:
                        json.dump(plan, fh, indent=2, ensure_ascii=False)
                    context["new_age_sex_pregnancystatus_declarations"] = accepted
                    context["age_sex_preg_errors"] = errors or []
                    context["age_sex_preg_stage"] = "demographics_only"
                    _extend_all_linked_variales(context, accepted, idx=idx)
                    _log("demog ✓", idx, f"{len(accepted)} demographic variables declared (all valid)")
                    return context

                if attempt < self.MAX_ATTEMPTS:
                    current_prompt = self._augment_with_feedback(prompt, rejected)
                    continue

                # Exhausted attempts
                if len(accepted) == 0:
                    plan = {
                        "new_age_sex_pregnancystatus_declarations": [],
                        "stage": "demographics_only",
                        "errors": errors or [],
                    }
                    with open(plan_log, "w", encoding="utf-8") as fh:
                        json.dump(plan, fh, indent=2, ensure_ascii=False)
                    context["new_age_sex_pregnancystatus_declarations"] = []
                    context["age_sex_preg_errors"] = []
                    context["age_sex_preg_stage"] = "demographics_only"
                    _log("demog ✓", idx, "0 demographic variables (none required)")
                    return context

                plan = {
                    "new_age_sex_pregnancystatus_declarations": accepted,
                    "stage": "demographics_only",
                    "errors": errors or [],
                }
                with open(plan_log, "w", encoding="utf-8") as fh:
                    json.dump(plan, fh, indent=2, ensure_ascii=False)
                context["new_age_sex_pregnancystatus_declarations"] = accepted
                context["age_sex_preg_errors"] = errors or []
                context["age_sex_preg_stage"] = "demographics_only"
                _extend_all_linked_variales(context, accepted, idx=idx)
                _log("demog ⚠", idx, f"partial pass-through with {len(accepted)} valid items (invalid stripped)")
                return context

            except Exception as exc:
                last_error = exc
                _schema_log(errors, idx=idx, block="new_age_sex_pregnancystatus_declarations", item_index=None,
                            code="JSON_OR_VALIDATION", message="Failed to parse/validate demographics block.", detail=str(exc))
                current_prompt = self._augment_with_feedback(prompt, [])
                continue

        # Fallback: empty (clean)
        if len(accepted) == 0:
            plan = {
                "new_age_sex_pregnancystatus_declarations": [],
                "stage": "demographics_only",
                "errors": [],
            }
            with open(plan_log, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(plan, indent=2, ensure_ascii=False))
            context["new_age_sex_pregnancystatus_declarations"] = []
            context["age_sex_preg_errors"] = []
            context["age_sex_preg_stage"] = "demographics_only"
            _log("demog ✓", idx, "0 demographic variables (none required)")
            return context

        # Shouldn't reach here often; keep accepted subset
        warnings.warn(f"SMT demographics pass fallback engaged (req#{idx}): {last_error}", RuntimeWarning)
        plan = {
            "new_age_sex_pregnancystatus_declarations": accepted,
            "stage": "demographics_only",
            "errors": errors or ([{"code": "FALLBACK_EMPTY", "detail": str(last_error) if last_error else ""}]),
        }
        with open(plan_log, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(plan, indent=2, ensure_ascii=False))
        context["new_age_sex_pregnancystatus_declarations"] = accepted
        context["age_sex_preg_errors"] = plan["errors"]
        context["age_sex_preg_stage"] = "demographics_only"
        _extend_all_linked_variales(context, accepted, idx=idx)
        _log("demog ⚠", idx, "fallback to accepted subset (non-empty)")
        return context
