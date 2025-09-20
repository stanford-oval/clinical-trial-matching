# modules/namer_common.py
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Dict, List, Any
import math

# ---------------------------------------------------------------------------
# Fallback logger (mirrors smt_incremental_translator.py)
# ---------------------------------------------------------------------------
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)


# ---------------------------------------------------------------------------
# Canonical-form helpers (ensure snake_case, never PT leakage)
# ---------------------------------------------------------------------------

_TAG_RE_PT = re.compile(r"\s*\([^)]*\)\s*$")  # strip SNOMED semantic tag e.g., " (disorder)"

def _strip_tag(term: str | None) -> str:
    return _TAG_RE_PT.sub("", (term or "").strip())

def _to_var(s: str | None) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def _canon_from(ent: Dict[str, Any]) -> str:
    """
    Prefer already-minted snake_case 'entity_canonical_form'.
    Otherwise, derive snake_case from PT/FSN/span. NEVER return title-cased PT.
    Supports both flat dicts and nested {'entity': {...}}.
    """
    # nested first
    if isinstance(ent.get("entity"), dict):
        nested = ent["entity"]
        if nested.get("entity_canonical_form"):
            return str(nested["entity_canonical_form"])
    # flat
    if ent.get("entity_canonical_form"):
        return str(ent["entity_canonical_form"])
    # derive
    pt = _strip_tag(
        ent.get("preferred_term")
        or ent.get("fully_specified_name")
        or ent.get("span")
        or ent.get("extracted_span")
        or ""
    )
    return _to_var(pt)


# ---------------------------------------------------------------------------
# Helper: pretty-print canonical-form bundles
# ---------------------------------------------------------------------------

def _minify_bundle(bundle: dict | None, *, include_attributes: bool = False) -> str:
    """
    Return a slim, readable JSON representation of a canonical-form bundle
    for prompt consumption. Guarantees snake_case 'entity_canonical_form'.
    """
    if not bundle:
        return ""  # nothing to print

    keep_attr = (
        "attribute_class_canonical_form",
        "attribute_value_canonical_form",
    )

    slim: list[dict] = []
    for ent in bundle.get("entities", []):
        ent_core = ent.get("entity") if isinstance(ent.get("entity"), dict) else ent
        ent_slim = {
            "entity_canonical_form": _canon_from(ent_core),
            "type": ent_core.get("type"),
            "span": ent_core.get("span") or ent_core.get("extracted_span"),
            "start": ent_core.get("start"),
            "end": ent_core.get("end"),
        }

        if include_attributes:
            attrs = ent_core.get("attributes", [])
            if attrs:
                ent_slim["attributes"] = [
                    {k: a[k] for k in keep_attr if k in a} for a in attrs if isinstance(a, dict)
                ]

        slim.append(ent_slim)

    return json.dumps(slim, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helper: pretty-print from valid_entities_by_req (entities-only path)
# ---------------------------------------------------------------------------

def _minify_from_valid_entities(ents_for_req: dict | None) -> str:
    """
    Build a slim, readable JSON list from context['valid_entities_by_req'][str(idx)].
    Guarantees snake_case 'entity_canonical_form'.
    """
    if not ents_for_req:
        return ""

    rows = sorted(
        ents_for_req.values(),
        key=lambda e: (
            e.get("start", 10**9),
            e.get("end", 10**9),
            e.get("extracted_span", "") or e.get("span", ""),
        ),
    )

    out: list[dict] = []
    for e in rows:
        out.append(
            {
                "entity_canonical_form": _canon_from(e),
                "type": e.get("type"),
                "entity": e.get("entity_name"),
                "span": e.get("extracted_span") or e.get("span"),
                "start": e.get("start"),
                "end": e.get("end"),
                "select_reason": e.get("select_reason"),
            }
        )
    return json.dumps(out, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Regex helpers (block extractors)
# ---------------------------------------------------------------------------

_NEWCANON_RE = re.compile(
    r"<new_canonical_variable_declarations>\s*(\[.*?])\s*</new_canonical_variable_declarations>",
    re.I | re.S,
)

# Age/Sex/PregnancyStatus block
_NEWASPS_RE = re.compile(
    r"<new_age_sex_pregnancystatus_declarations>\s*(\[.*?])\s*</new_age_sex_pregnancystatus_declarations>",
    re.I | re.S,
)

# (Stage-1 ban) Detect but do not parse/accept these:
_NEWOTHER_RE = re.compile(
    r"<all_other_variable_declarations>\s*(\[.*?])\s*</all_other_variable_declarations>",
    re.I | re.S,
)
_NEWNONCANON_RE = re.compile(
    r"<new_noncanonical_variable_declarations>\s*(\[.*?])\s*</new_noncanonical_variable_declarations>",
    re.I | re.S,
)

_ERRORS_RE = re.compile(
    r"<errors>\s*(\[.*?])\s*</errors>",
    re.I | re.S,
)

# BACKCOMPAT: old single block, if present we’ll treat as canonical by default
_NEWDECL_RE_OLD = re.compile(
    r"<new_variable_declarations>\s*(\[.*?])\s*</new_variable_declarations>",
    re.I | re.S,
)

_TAG_RE = re.compile(r":named\s+([Rr]\d+_A\d+_[A-Z0-9_]+)")


# ---------------------------------------------------------------------------
# Schema / validation helpers
# ---------------------------------------------------------------------------
# Canonical block (new spec)
_REQUIRED_FIELDS_CANON = (
    "span",
    "entity_canonical_form_used",
    "template",
    # "timeframe",
    "entity_variable_name",
    "type",
    "extracted_value",
    #"reason",
    "usage",
    "qualifier_predicates",
    "timewindow_this_patient_fact_certainly_holds",
    "largest_timewindow_this_patient_fact_may_hold",
)

# Age/Sex/PregnancyStatus block (no canonical form field; qualifiers optional)
_REQUIRED_FIELDS_ASPS = (
    "span",
    "template",
    # "timeframe",
    "entity_variable_name",
    "type",
    "extracted_value",
    #"reason",
    "usage",
    "timewindow_this_patient_fact_certainly_holds",
    "largest_timewindow_this_patient_fact_may_hold",
)

# Acceptable timeframe tokens
# NOTE: last entry uses "...later" (BC: we still recognize "...after" while normalizing to later)
_TIMEFRAME_RE = re.compile(
    r"^("
    r"now|inthehistory|inthefuture|"
    r"inthepast\d+(minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(minutes|hours|days|weeks|months|years)|"
    r"\d+(minutes|hours|days|weeks|months|years)(ago|later)"
    r")$"
)

# Qualifier pattern (case allowed if enabled; characters limited)
_STEM_QUAL_RE = re.compile(r"^[A-Za-z0-9_]+@@[A-Za-z0-9_]+$")

# --- naming-system: allowed templates and shape checkers --------------------
_ALLOWED_TEMPLATES = {
    "findings",
    "procedures",
    "observable_entities_numeric",
    "observable_entities_status",
    "product",
    "substance",
}

def _template_shape_regexes(timeframe: str) -> dict[str, re.Pattern]:
    tf = re.escape(timeframe)
    return {
        # Findings (patient_* prefixes per spec)
        # patient_has_{diagnosis_of|finding_of|symptoms_of|clinical_signs_of|suspicion_of}_{entity}_{tf}
        "findings": re.compile(
            rf"^patient_has_(?:diagnosis_of|finding_of|symptoms_of|clinical_signs_of|suspicion_of)_[a-z0-9_]+_{tf}$"
        ),
        # Procedures (patient_*; outcome suffix allowed ONLY for patient_has_undergone_*_{tf})
        "procedures": re.compile(
            rf"^(?:"
            rf"patient_has_undergone_[a-z0-9_]+_{tf}(?:_outcome_is_(?:positive|negative|normal|abnormal))?"
            rf"|patient_is_undergoing_[a-z0-9_]+_{tf}"
            rf"|patient_needs_to_undergo_[a-z0-9_]+_{tf}"
            rf"|patient_will_undergo_[a-z0-9_]+_{tf}"
            rf"|patient_can_undergo_[a-z0-9_]+_{tf}"
            rf")$"
        ),
        # Observable numeric: patient_{entity}_value_recorded_{tf}_withunit_{unit}
        "observable_entities_numeric": re.compile(
            rf"^patient_[a-z0-9_]+_value_recorded_{tf}_withunit_[a-z0-9_]+$"
        ),
        # Observable status: patients_{entity}_is_{positive|negative|normal|abnormal}_{tf}
        # (accepts patient_ or patients_ for back-compat)
        "observable_entities_status": re.compile(
            rf"^patients?_[a-z0-9_]+_(?:is_positive|is_negative|is_normal|is_abnormal)_{tf}$"
        ),
        # Product: patient_{is_taking|has_taken|has_hypersensitivity_to|has_intolerance_to|has_allergy_to|has_nonimmune_hypersensitivity_to}_{entity}_{tf}
        "product": re.compile(
            rf"^patient_(?:is_taking|has_taken|has_hypersensitivity_to|has_intolerance_to|has_allergy_to|has_nonimmune_hypersensitivity_to)_[a-z0-9_]+_{tf}$"
        ),
        # Substance: patient_{is_exposed_to|was_exposed_to|has_hypersensitivity_to|has_intolerance_to|has_allergy_to|has_nonimmune_hypersensitivity_to}_{entity}_{tf}
        "substance": re.compile(
            rf"^patient_(?:is_exposed_to|was_exposed_to|has_hypersensitivity_to|has_intolerance_to|has_allergy_to|has_nonimmune_hypersensitivity_to)_[a-z0-9_]+_{tf}$"
        ),
    }


def _assert_stem_matches_template(stem: str, template: str, timeframe: str) -> None:
    t = (template or "").strip().lower()
    if t not in _ALLOWED_TEMPLATES:
        raise ValueError(f"template {template!r} must be one of {_ALLOWED_TEMPLATES}")
    rx = _template_shape_regexes(timeframe)[t]
    if not rx.fullmatch(stem):
        raise ValueError(
            f"stem {stem!r} does not match naming template={t!r} for timeframe={timeframe!r}"
        )


def _assert_asps_stem_shape(stem: str, timeframe: str) -> None:
    """Demographics stems must match one of the allowed forms (with correct timeframe position)."""
    tf = re.escape(timeframe)
    patterns = [
        re.compile(rf"^patient_age_value_recorded_{tf}_in_(?:years|months|days)$"),
        re.compile(rf"^patient_sex_is_(?:male|female|other)_{tf}$"),
        re.compile(rf"^patient_is_pregnant_{tf}$"),
        re.compile(rf"^patient_is_able_to_be_pregnant_{tf}$"),
        re.compile(rf"^patient_has_childbearing_potential_{tf}$"),
        re.compile(rf"^patient_is_breastfeeding_{tf}$"),
        re.compile(rf"^patient_is_lactating_{tf}$"),
    ]
    if not any(p.fullmatch(stem) for p in patterns):
        raise ValueError(
            f"demographics stem {stem!r} must match one of the allowed forms for timeframe={timeframe!r}"
        )


# Disallow timeframe/value/unit semantics inside qualifiers (per spec)
_QUAL_TIMEFRAME_OR_UNIT_INFIX = re.compile(
    r"(?:^|_)(?:"
    r"now|inthehistory|inthefuture|"
    r"inthe(?:past|future)\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years)|"
    r"\d+(?:minutes|hours|days|weeks|months|years)(?:ago|later|after)|"  # include 'after' for BC
    r"withunit|in_(?:years|months|days)"
    r")(?:_|$)"
)


def _ensure_qualifiers_do_not_repeat_time_or_unit(qp_list: list[str]) -> None:
    """Qualifiers must NOT repeat timeframe/value/unit semantics per naming system."""
    for qp in qp_list or []:
        try:
            _, qual = qp.split("@@", 1)
        except ValueError:
            qual = qp
        if _QUAL_TIMEFRAME_OR_UNIT_INFIX.search(qual):
            raise ValueError(
                f"qualifier {qp!r} must not contain timeframe/value/unit tokens "
                "(e.g., now/inthepast…/foradurationof…/…ago/…later/withunit/in_years|months|days)"
            )


# --- timeframe canonicalizer -----------------------------------------------
_TIME_UNIT_CANON = {
    "y": "years", "yr": "years", "yrs": "years", "year": "years", "years": "years",
    "m": "months", "mo": "months", "mos": "months", "mon": "months", "month": "months", "months": "months",
    "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks",
    "d": "days", "day": "days", "days": "days",
}


def _canon_timeframe(tf: str) -> str:
    """
    Normalize a variety of timeframe spellings to:
      now | inthehistory | inthefuture |
      inthepast{n}{unit} | inthefuture{n}{unit} |
      foradurationof{n}{unit} | {n}{unit}ago | {n}{unit}later
    Also maps:
      - 'within{n}{unit}' → 'inthepast{n}{unit}'
      - 'for|during{n}{unit}' → 'foradurationof{n}{unit}'
      - (BC) '{n}{unit}after' → '{n}{unit}later'
    """
    raw = str(tf or "").strip().lower()
    s = re.sub(r"[\s_\-]+", "", raw)

    # simple tokens
    if s in {"now", "today", "present", "current"}:
        return "now"
    if s in {"history", "inthehistory", "past", "previous"}:
        return "inthehistory"
    if s in {"future", "inthefuture", "upcoming"}:
        return "inthefuture"

    # withinNunit → inthepastNunit
    m = re.match(r"^within(\d+)([a-z]+)$", s)
    if m:
        n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"inthepast{n}{unit}"

    # foradurationofNunit (already canonical)
    m = re.match(r"^foradurationof(\d+)([a-z]+)$", s)
    if m:
        n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"foradurationof{n}{unit}"

    # forNunit / duringNunit → foradurationofNunit
    m = re.match(r"^(?:for|during)(\d+)([a-z]+)$", s)
    if m:
        n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"foradurationof{n}{unit}"

    # inthepastNunit / inthefutureNunit (already canonical)
    m = re.match(r"^inthe(past|future)(\d+)([a-z]+)$", s)
    if m:
        past_future, n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"inthe{past_future}{n}{unit}"

    # last/next N units
    m = re.match(r"^(last|past)(\d+)([a-z]+)$", s)
    if m:
        _, n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"inthepast{n}{unit}"
    m = re.match(r"^(next)(\d+)([a-z]+)$", s)
    if m:
        _, n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"inthefuture{n}{unit}"

    # Nunitago / Nunitlater / (BC) Nunitafter → canonicalize to 'ago' or 'later'
    m = re.match(r"^(\d+)([a-z]+)(ago|later|after)$", s)
    if m:
        n, unit, suffix = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        out_suffix = "later" if suffix in {"later", "after"} else "ago"
        return f"{n}{unit}{out_suffix}"

    # fall through: keep raw if it already matches allowed pattern
    if _TIMEFRAME_RE.match(raw):
        return raw
    return s  # let validator reject if still not matching


# --- symbol sanitizers (punctuation semantics, mixed case toggle) -----------
def _symbol_sanitize(s: str, *, lower: bool) -> str:
    """
    Safe SMT symbol segment:
    - optional lowercasing
    - '+' → 'plus', '/' → '_or_'
    - numeric range '2-4a' → '2to4a'
    - other hyphens/space → '_', other punct → '_'
    - collapse underscores
    """
    if not isinstance(s, str):
        s = str(s or "")
    s = s.strip()

    # --- NEW: remove any qualifier part like '@@progressively_worse' ---
    # gpt-4o is putting qualifier into stem sometimes; strip it here
    if "@@" in s:
        s = s.split("@@", 1)[0].strip()

    if lower:
        s = s.lower()

    s = s.replace("+", "plus")
    s = s.replace("/", "_or_")
    s = re.sub(r'(?<=\d)-(?=\d[a-zA-Z]?)', 'to', s)
    s = s.replace("-", "_")
    s = re.sub(r'\s+', "_", s)
    s = re.sub(r'[^A-Za-z0-9_]', "_", s)
    s = re.sub(r'_+', "_", s).strip("_")
    return s


def _normalize_old_keys(obj: dict) -> dict:
    """
    Back-compat shim: map old keys to the new schema when possible.
    - Move `qualifiers` -> `qualifier_predicates` by prefixing 'stem@@'
    - Convert `is_canonical` -> `entity_canonical_form_used` (best-effort)
    """
    o = dict(obj)

    # qualifiers -> qualifier_predicates (prefix with stem@@)
    if "qualifier_predicates" not in o and "qualifiers" in o and isinstance(o["qualifiers"], list):
        stem = o.get("entity_variable_name", "")
        qp: list[str] = []
        for q in o["qualifiers"]:
            if isinstance(q, str) and q:
                qp.append(f"{stem}@@{q}")
        o["qualifier_predicates"] = qp
        o.pop("qualifiers", None)

    # is_canonical -> entity_canonical_form_used
    if "entity_canonical_form_used" not in o:
        flag = str(o.get("is_canonical", "")).strip().upper()
        if flag == "NO":
            o["entity_canonical_form_used"] = ""
        else:
            o.setdefault("entity_canonical_form_used", "")
        o.pop("is_canonical", None)

    return o


def _sort_key(obj: dict) -> tuple:
    return (
        obj.get("entity_canonical_form_used", "") or "",
        obj.get("timeframe", "") or "",
        obj.get("entity_variable_name", "") or "",
    )


# ---------------------------------------------------------------------------
# Relaxed JSON parsing helpers (handle // comments, /* */ comments, trailing commas)
# ---------------------------------------------------------------------------
_CODEFENCE_RE = re.compile(r"^\s*```(?:json|JSON|smt|SMT|text)?\s*|\s*```$", re.M)

def _strip_code_fence(s: str) -> str:
    return re.sub(_CODEFENCE_RE, "", s)


def _strip_json_comments_and_trailing_commas(s: str) -> str:
    # remove // line comments
    s = re.sub(r"(?m)^\s*//.*$", "", s)
    # remove /* ... */ comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    # collapse trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _parse_json_array_relaxed(text: str) -> list:
    s = _strip_code_fence(text or "")
    s = _strip_json_comments_and_trailing_commas(s)
    s = s.strip()
    if not s:
        return []
    arr = json.loads(s)
    if not isinstance(arr, list):
        raise ValueError("expected JSON array")
    return arr


def _validate_decl_list(
    items: Any,
    *,
    kind: str,  # "canonical" | "age_sex_preg"
    allow_backcompat: bool = True,
    enforce_suffix_in_stem: bool = False,
    allow_mixed_case: bool = True,
    collect_sanitize_corrections: list | None = None,
    enable_template_checks: bool = False,
) -> list[dict]:
    """Ensure items is a list of objects with required keys and correct formats."""
    if not isinstance(items, list):
        raise ValueError("declarations must be a JSON array")

    if kind == "canonical":
        required = _REQUIRED_FIELDS_CANON
    elif kind == "age_sex_preg":
        required = _REQUIRED_FIELDS_ASPS
    else:
        raise ValueError(f"unknown declaration kind: {kind}")

    # dynamic patterns & sanitizer mode
    name_pat = r"[A-Za-z0-9_]+" if allow_mixed_case else r"[a-z0-9_]+"
    stem_rx = re.compile(rf"^{name_pat}$")
    stemqual_rx = re.compile(rf"^{name_pat}@@{name_pat}$")
    lower_for_sanitize = not allow_mixed_case

    out: list[dict] = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"{kind} declaration #{i} must be an object")

        # Backcompat only applies to historical canonical schemas
        obj = _normalize_old_keys(raw) if (allow_backcompat and kind == "canonical") else raw

        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(f"{kind} declaration #{i} missing keys: {missing}")

        # timeframe: canonicalize then validate
        # tf_raw = obj.get("timeframe", "")
        # tf_norm = _canon_timeframe(tf_raw)
        # obj["timeframe"] = tf_norm
        # if not _TIMEFRAME_RE.match(tf_norm):
        #     raise ValueError(
        #         f"{kind} declaration #{i} has invalid timeframe={tf_raw!r} → {tf_norm!r}; "
        #         "use now|inthehistory|inthefuture|inthepast{n}{minutes|hours|days|weeks|months|years}|"
        #         "inthefuture{n}{…}|foradurationof{n}{…}|{n}{…}ago|{n}{…}later"
        #     )

        # entity_variable_name: sanitize (keep or force lower case based on toggle)
        stem_in = str(obj["entity_variable_name"])
        stem = _symbol_sanitize(stem_in, lower=lower_for_sanitize)
        obj["entity_variable_name"] = stem
        # surface I4 auto-corrections (soft note)
        if collect_sanitize_corrections is not None and stem_in != stem:
            collect_sanitize_corrections.append(
                {"kind": kind, "original": stem_in, "sanitized": stem}
            )

        # Optional suffix enforcement (Stage-1 still checks via _enforce_timeframe_in_stem)
        # if enforce_suffix_in_stem:
        #     tf_seg_pat = re.compile(rf"(?:^|_){re.escape(tf_norm)}(?:_|$)")
        #     if not tf_seg_pat.search(stem):
        #         raise ValueError(
        #             f"{kind} declaration #{i} entity_variable_name {stem!r} must include timeframe segment {tf_norm!r}"
        #         )

        # qualifiers:
        # - canonical: required by schema (may be [])
        # - age_sex_preg: optional; validate if present
        has_qp_field = "qualifier_predicates" in obj
        if kind == "canonical" or has_qp_field:
            qps = obj.get("qualifier_predicates", [])
            if not isinstance(qps, list):
                raise ValueError(f"{kind} declaration #{i} 'qualifier_predicates' must be an array")

            qps_norm: list[str] = []
            for qp in qps:
                if not isinstance(qp, str) or "@@" not in qp:
                    raise ValueError(f"{kind} declaration #{i} qualifier must be 'stem@@qualifier'")
                s_part_raw, qual_raw = qp.split("@@", 1)
                s_part = _symbol_sanitize(s_part_raw, lower=lower_for_sanitize)
                qual = _symbol_sanitize(qual_raw, lower=lower_for_sanitize)
                qps_norm.append(f"{s_part}@@{qual}")
            obj["qualifier_predicates"] = qps_norm
            # naming-system: qualifiers must not repeat timeframe/value/unit tokens
            _ensure_qualifiers_do_not_repeat_time_or_unit(obj["qualifier_predicates"])

            # validations
            for qp in obj["qualifier_predicates"]:
                if not stemqual_rx.fullmatch(qp):
                    raise ValueError(
                        f"{kind} declaration #{i} qualifier {qp!r} must match 'stem@@qualifier' with allowed chars"
                    )
                stem_part, _ = qp.split("@@", 1)
                if (stem_part.lower() if allow_mixed_case else stem_part) != (
                    stem.lower() if allow_mixed_case else stem
                ):
                    raise ValueError(
                        f"{kind} declaration #{i} qualifier {qp!r} stem {stem_part!r} "
                        f"must equal entity_variable_name {stem!r}"
                    )

        # stem name check
        if not stem_rx.fullmatch(stem):
            raise ValueError(f"{kind} declaration #{i} invalid entity_variable_name={stem!r}")

        # canonical-specific: entity_canonical_form_used must be non-empty and stem must match its template
        if kind == "canonical":
            ecfu = obj.get("entity_canonical_form_used", "")
            if not isinstance(ecfu, str) or not ecfu.strip():
                raise ValueError(f"{kind} declaration #{i} must set non-empty 'entity_canonical_form_used'")
        #     if enable_template_checks:
        #         _assert_stem_matches_template(stem, obj.get("template", ""), tf_norm)
        # else:
        #     # demographics block shape check
        #     if enable_template_checks:
        #         _assert_asps_stem_shape(stem, tf_norm)
                
                # Normalize / stash all four endpoints (no parse_bound_field needed)
        # ── normalize + keep raw -------------------------------------------------------
        # ── normalize + keep raw -------------------------------------------------------
        for _key in (
            "timewindow_this_patient_fact_certainly_holds",
            "largest_timewindow_this_patient_fact_may_hold",
        ):
            if _key in obj:
                rawv = obj.get(_key)
                obj.setdefault(f"_{_key}_raw", rawv)  # keep original structure
                obj[_key] = rawv if isinstance(rawv, dict) else None

        # prefer canonical key; fall back to alias if present
        SMALLEST_KEY = "timewindow_this_patient_fact_certainly_holds"
        LARGEST_KEY  = "largest_timewindow_this_patient_fact_may_hold"
        if obj.get(LARGEST_KEY) is None and isinstance(obj.get("largest_timewindow_this_patient_may_hold"), dict):
            obj[LARGEST_KEY] = obj["largest_timewindow_this_patient_may_hold"]

        # ── helpers -------------------------------------------------------------------
        def _to_hours_or_none(bound_dict):
            """{temporal_direction, temporal_magnitude, units, inclusive} -> signed hours or None."""
            if isinstance(bound_dict, dict):
                try:
                    return bound_to_hours(bound_dict)
                except Exception:
                    return None
            return None

        def _same_side_of_now(a_h, b_h):
            return (
                (a_h < 0 and b_h < 0) or
                (a_h > 0 and b_h > 0) or
                (a_h == 0 and b_h == 0)
            )

        def _check_window_pair(window_key, pair_label):
            win = obj.get(window_key)
            if not isinstance(win, dict):
                return
            s_h = _to_hours_or_none(win.get("start_time"))
            e_h = _to_hours_or_none(win.get("end_time"))
            if s_h is None or e_h is None:
                return
            if _same_side_of_now(s_h, e_h) and math.isfinite(s_h) and math.isfinite(e_h):
                if e_h < s_h:
                    raise ValueError(
                        f"{kind} declaration #{i}: {pair_label} has end < start "
                        f"(end={e_h} h < start={s_h} h) while both are on the same side of now"
                    )

        def _assert_leq_bounds(left_win_key, left_bound_name, right_win_key, right_bound_name, label):
            lw = obj.get(left_win_key)  if isinstance(obj.get(left_win_key), dict)  else None
            rw = obj.get(right_win_key) if isinstance(obj.get(right_win_key), dict) else None
            if lw is None or rw is None:
                return
            l_h = _to_hours_or_none(lw.get(left_bound_name))
            r_h = _to_hours_or_none(rw.get(right_bound_name))
            if l_h is None or r_h is None:
                return
            if l_h > r_h:
                raise ValueError(
                    f"{kind} declaration #{i} violates {label}: "
                    f"{left_win_key}.{left_bound_name} ({l_h} h) > {right_win_key}.{right_bound_name} ({r_h} h)"
                )

        def _emit_parsed_window(window_key, prefix):
            """Write parsed keys for start/end hours + inclusive flags. Missing → None."""
            win = obj.get(window_key) if isinstance(obj.get(window_key), dict) else None
            if not win:
                obj[f"{prefix}_timewindow_start_time_in_hours"]      = None
                obj[f"{prefix}_timewindow_end_time_in_hours"]        = None
                obj[f"{prefix}_timewindow_start_time_inclusive"]     = None
                obj[f"{prefix}_timewindow_end_time_inclusive"]       = None
                return

            s_bound = win.get("start_time")
            e_bound = win.get("end_time")

            obj[f"{prefix}_timewindow_start_time_in_hours"]  = _to_hours_or_none(s_bound)
            obj[f"{prefix}_timewindow_end_time_in_hours"]    = _to_hours_or_none(e_bound)
            obj[f"{prefix}_timewindow_start_time_inclusive"] = (
                bool(s_bound.get("inclusive")) if isinstance(s_bound, dict) and "inclusive" in s_bound else None
            )
            obj[f"{prefix}_timewindow_end_time_inclusive"]   = (
                bool(e_bound.get("inclusive")) if isinstance(e_bound, dict) and "inclusive" in e_bound else None
            )
        try:
            # ── pairwise ordering inside each window --------------------------------------
            _check_window_pair(SMALLEST_KEY, "conservative timeframe")
            _check_window_pair(LARGEST_KEY,  "possible timeframe")

            # ── subset constraints (conservative ⊆ possible) ------------------------------
            _assert_leq_bounds(LARGEST_KEY,  "start_time", SMALLEST_KEY, "start_time",
                            label="subset (largest.start_time ≤ smallest.start_time)")
            _assert_leq_bounds(SMALLEST_KEY, "end_time",   LARGEST_KEY,  "end_time",
                            label="subset (smallest.end_time ≤ largest.end_time)")

            # ── parsed outputs -------------------------------------------------------------
            _emit_parsed_window(SMALLEST_KEY, prefix="smallest")
            _emit_parsed_window(LARGEST_KEY,  prefix="largest")

            out.append(obj)
        except Exception as e:
            # skip this obj on any validation error
            print(f"[warn] Skipping {kind} declaration #{i} due to timewindow error: {e}")
            # (optional) you can log more details if needed:
            # import traceback; traceback.print_exc()
            pass
        
    # Deterministic sort (I4)
    out.sort(key=_sort_key)
    return out

# ---------------------------------------------------------------------------
# === Guards enforcing timeframe token presence/uniqueness (I1) ==============
# ---------------------------------------------------------------------------

# Broaden capture to allow singular/abbrev units in stems; we'll canonicalize later.
_TF_TOKEN_RE = re.compile(
    r"(?:^|_)(?:"
    r"now|inthehistory|inthefuture|"
    r"inthepast\d+[a-z]+|"          # e.g., inthepast1year / inthepast1yrs / inthepast3months
    r"inthefuture\d+[a-z]+|"        # e.g., inthefuture2wk
    r"foradurationof\d+[a-z]+|"     # e.g., foradurationof1month
    r"\d+[a-z]+(?:ago|later|after)" # e.g., 1yrago / 2yearsafter
    r")(?:_|$)",
    re.I,
)

# Capture Bool/Int/Real variable declarations from existing SMT program
_DECL_ANY_RE = re.compile(r"\(declare-const\s+([^\s\)]+)\s+(?:Bool|Int|Real)\)")

def _extract_declared_symbols(smt_lines: List[str]) -> set[str]:
    names: set[str] = set()
    for ln in smt_lines or []:
        m = _DECL_ANY_RE.search(ln)
        if m:
            names.add(m.group(1))
    return names



# ---- fallback helpers ------------------------------------------------------

def _default_timeframe_for_type(typ: str) -> str:
    t = (typ or "").lower()
    if "procedure" in t:
        return "inthehistory"
    if "observable" in t or "measurement" in t or "lab" in t:
        return "inthehistory"
    # default for findings/diagnoses/conditions
    return "now"


def _template_for_type(typ: str) -> str:
    t = (typ or "").lower()
    if "procedure" in t:
        return "procedures"
    # keep it simple/robust for fallback
    return "findings"


def _synthesize_stem(canon: str, typ: str, timeframe: str, *, allow_mixed_case: bool) -> str:
    base = _symbol_sanitize(canon, lower=not allow_mixed_case)
    t = (typ or "").lower()
    if "procedure" in t:
        prefix = "patient_has_undergone_"
    else:
        prefix = "patient_has_finding_of_"
    return f"{prefix}{base}_{timeframe}"

# ---- New helpers for bounds parsing ----
def _to_float_or_none(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    xs = str(x).strip()
    if xs.lower() in {"none", ""}: return None
    try:
        return float(xs)
    except Exception:
        return None

def _to_bool_inclusive(x):
    if x is None: return None
    xs = str(x).strip().lower()
    if xs in {"inclusive", "true", "t", "yes", "y", "1"}: return True
    if xs in {"exclusive", "false", "f", "no", "n", "0"}: return False
    return None  # unknown

_TUPLE_RE = re.compile(
    r"""^\s*\(                     # opening paren
        \s*(['"]?)(past|now|future)\1\s*,      # anchor
        \s*(['"]?)([^,'")]+|None)\3\s*,        # value (or None)
        \s*(['"]?)([^,'")]+|None)\5\s*,        # unit (or None)
        \s*(['"]?)(inclusive|exclusive)\7      # inclusive flag
        \s*\)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

def _parse_bound_field(raw):
    """
    Accepts one of:
      - None / "None" -> None
      - string like ("past","5","days","inclusive")
      - ["past","5","days","inclusive"]
      - {"anchor":..., "value":..., "unit":..., "inclusive":...}
    Returns normalized dict or None.
    """
    if raw is None:
        return None

    # dict form
    if isinstance(raw, dict):
        anchor = (raw.get("anchor") or raw.get("time") or raw.get("when") or "").strip().lower()
        value  = _to_float_or_none(raw.get("value"))
        unit   = raw.get("unit")
        if isinstance(unit, str):
            unit = unit.strip()
            if unit.lower() == "none" or unit == "":
                unit = None
        inclusive = _to_bool_inclusive(raw.get("inclusive"))
        if anchor in {"past", "now", "future"}:
            return {"anchor": anchor, "value": value, "unit": unit, "inclusive": inclusive}
        return None

    # list/tuple form
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        anchor = str(raw[0]).strip().lower()
        value  = _to_float_or_none(raw[1])
        unit   = raw[2]
        if isinstance(unit, str):
            unit = unit.strip()
            if unit.lower() == "none" or unit == "":
                unit = None
        inclusive = _to_bool_inclusive(raw[3])
        if anchor in {"past", "now", "future"}:
            return {"anchor": anchor, "value": value, "unit": unit, "inclusive": inclusive}
        return None

    # string form
    s = str(raw).strip()
    if s.lower() == "none":
        return None

    m = _TUPLE_RE.match(s)
    if m:
        anchor = m.group(2).lower()
        value  = _to_float_or_none(m.group(4))
        unit_s = m.group(6)
        unit   = None if (unit_s is None or unit_s.lower() == "none") else unit_s
        inclusive = _to_bool_inclusive(m.group(8))
        return {"anchor": anchor, "value": value, "unit": unit, "inclusive": inclusive}

    # fallback: try comma-split without parens
    if "," in s:
        parts = [p.strip().strip('"\'') for p in s.split(",")]
        if len(parts) == 4:
            anchor = parts[0].lower()
            value  = _to_float_or_none(parts[1])
            unit   = None if parts[2].lower() == "none" or parts[2] == "" else parts[2]
            inclusive = _to_bool_inclusive(parts[3])
            if anchor in {"past", "now", "future"}:
                return {"anchor": anchor, "value": value, "unit": unit, "inclusive": inclusive}

    return None

# --- Coerce bounds tuple-literals to JSON before json.loads ---


def _coerce_bounds_for_json(s: str) -> str:
    """
    Normalize ONLY the confirmable time endpoint dicts into strict JSON:

      "confirmable_latest_start_time":  { ... }
      "confirmable_earliest_end_time": { ... }

    Fixes inside those dicts:
      - inclusive: True/False -> true/false
      - temporal_magnitude: Inf/inf -> "Inf"
      - temporal_direction: quote bare past|now|future
      - units: quote bare tokens + enforce plural
      - strip trailing commas, None -> null
    Leaves everything else untouched.
    """
    import re

    # Match field head and a (flat) dict body: field: { ... }
    DICT_RE = re.compile(
        r'("confirmable_(?:latest_start_time|earliest_end_time)"\s*:\s*)\{\s*(?P<body>[^{}]*?)\s*\}',
        flags=re.S
    )

    UNITS_PLURALS = {
        "minute": "minutes", "hour": "hours", "day": "days",
        "week": "weeks", "month": "months", "year": "years",
        "minutes": "minutes", "hours": "hours", "days": "days",
        "weeks": "weeks", "months": "months", "years": "years",
    }

    def _pluralize_unit(u: str) -> str:
        u = (u or "").strip().lower()
        return UNITS_PLURALS.get(u, (u + "s") if (u and not u.endswith("s")) else u)

    def _fix_dict(m: re.Match) -> str:
        head = m.group(1)
        body = m.group("body") or ""

        # Normalize Python-ish tokens
        body2 = re.sub(r'("inclusive"\s*:\s*)True\b',  r'\1true',  body)
        body2 = re.sub(r'("inclusive"\s*:\s*)False\b', r'\1false', body2)

        # temporal_magnitude: Inf/inf -> "Inf"
        body2 = re.sub(r'("temporal_magnitude"\s*:\s*)(Inf|inf)\b', r'\1"Inf"', body2)

        # 🔧 NEW: remove quotes around numeric magnitudes (e.g. "0.0" → 0.0)
        body2 = re.sub(r'("temporal_magnitude"\s*:\s*)"([+-]?\d+(?:\.\d+)?)"', r'\1\2', body2)

        # temporal_direction: quote bare enums
        body2 = re.sub(r'("temporal_direction"\s*:\s*)(past|now|future)\b', r'\1"\2"', body2, flags=re.I)

        # units: quote bare words
        body2 = re.sub(r'("units"\s*:\s*)([A-Za-z]+)\b', r'\1"\2"', body2)

        # None -> null
        body2 = re.sub(r'\bNone\b', 'null', body2, flags=re.I)

        # Trailing commas before } or ]
        body2 = re.sub(r",\s*([}\]])", r"\1", body2)

        # Enforce plural units if present (rewrite the value)
        def _plural_units_cb(mm: re.Match) -> str:
            prefix = mm.group(1)
            val = mm.group(2)
            up = _pluralize_unit(val.strip('"').strip("'"))
            return f'{prefix}"{up}"'

        body2 = re.sub(r'("units"\s*:\s*)"(.*?)"', _plural_units_cb, body2)

        return f"{head}{{{body2}}}"

    # Clean dict-like bodies for the two fields
    s2 = DICT_RE.sub(_fix_dict, s)

    # Final guard: stray bare None elsewhere (safe for JSON)
    s2 = re.sub(r'\bNone\b', 'null', s2, flags=re.IGNORECASE)
    return s2


def bound_to_hours(bound: dict | None) -> float:
    """
    Convert a time bound dict into signed hours (float).

    Expected keys:
      - temporal_direction: "past" | "now" | "future"
      - temporal_magnitude: <float | "Inf" | None>
      - units: plural or singular ("hours"|"hour"|..., case-insensitive)
      - inclusive: bool  (ignored for numeric conversion)

    Conventions:
      - past   → negative hours
      - now    → 0.0 (magnitude/units ignored)
      - future → positive hours
      - magnitude ∈ {None, "Inf", ±Inf} → unbounded:
          past→ -inf, now→ 0.0, future→ +inf
    """

    INF_SENTINEL = 1000000000.0
    if not bound:
        return 0.0

    direction = (bound.get("temporal_direction") or "").strip().lower()
    value = bound.get("temporal_magnitude")
    unit = _norm_unit(bound.get("units"))

    # "now" always maps to 0 regardless of magnitude/units
    if direction == "now":
        return 0.0

    # treat None / "Inf" / ±Inf as unbounded in the given direction
    if _is_infinite_token(value) or value is None:
        if direction == "past":
            return -INF_SENTINEL
        if direction == "future":
            return INF_SENTINEL
        # unknown/empty direction → safest is 0.0
        return 0.0

    # parse finite magnitude
    v = _to_float_or_none(value)
    if v is None:
        # unparseable → treat as unbounded in direction
        if direction == "past":
            return -INF_SENTINEL
        if direction == "future":
            return INF_SENTINEL
        return 0.0

    # convert units → hours (unknown unit falls back to 1:1)
    if unit in _HOURS_PER:
        hours = float(abs(v)) * _HOURS_PER[unit]
    else:
        hours = float(abs(v))

    # apply sign by direction
    if direction == "past":
        return -hours
    if direction == "future":
        return hours
    return 0.0


_HOURS_PER = {
    "minute": 1.0 / 60.0,
    "hour":   1.0,
    "day":    24.0,
    "week":   7.0 * 24.0,
    "month":  30.0 * 24.0,       # or 365.2425/12*24 for astronomical avg
    "year":   365.0 * 24.0,      # or 365.2425*24 ≈ 8765.82
}

def _is_infinite_token(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (int, float)):
        return math.isinf(float(x))
    s = str(x).strip().lower()
    return s in {"inf", "+inf", "-inf", "infinite", "infinity"}

def _to_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.lower() in {"none", ""}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _norm_unit(u: str | None) -> str | None:
    if not u:
        return None
    u = str(u).strip().lower()
    if u.endswith("s"):
        u = u[:-1]  # normalize plurals
    alias = {
        "min": "minute", "mins": "minute",
        "hr": "hour", "h": "hour",
        "wk": "week", "w": "week",
        "mo": "month",
        "yr": "year", "y": "year",
    }
    return alias.get(u, u)
