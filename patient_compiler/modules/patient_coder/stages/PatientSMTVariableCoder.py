from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from typing import Dict, List, Any

import dspy

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
# Helper: pretty-print canonical-form bundles
# ---------------------------------------------------------------------------

def _minify_bundle(bundle: dict | None, *, include_attributes: bool = False) -> str:
    """
    Return a slim, readable JSON representation of a canonical-form bundle
    for prompt consumption.
    """
    if not bundle:
        return ""  # nothing to print

    keep_ent = (
        "entity_canonical_form",
        "type",
        "span",
        "start",
        "end",
    )
    keep_attr = (
        "attribute_class_canonical_form",
        "attribute_value_canonical_form",
    )

    slim: list[dict] = []
    for ent in bundle.get("entities", []):
        ent_slim = {k: ent[k] for k in keep_ent if k in ent}

        if include_attributes:
            attrs = ent.get("attributes", [])
            if attrs:
                ent_slim["attributes"] = [
                    {k: a[k] for k in keep_attr if k in a} for a in attrs
                ]

        slim.append(ent_slim)

    return json.dumps(slim, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helper: pretty-print from valid_entities_by_req (entities-only path)
# ---------------------------------------------------------------------------

def _minify_from_valid_entities(ents_for_req: dict | None) -> str:
    """
    Build a slim, readable JSON list from context['valid_entities_by_req'][str(idx)].
    """
    if not ents_for_req:
        return ""

    rows = sorted(
        ents_for_req.values(),
        key=lambda e: (
            e.get("start", 10**9),
            e.get("end", 10**9),
            e.get("extracted_span", ""),
        ),
    )

    out: list[dict] = []
    for e in rows:
        out.append(
            {
                "entity_canonical_form": e.get("preferred_term"),
                "type": e.get("type"),
                "span": e.get("extracted_span"),
                "start": e.get("start"),
                "end": e.get("end"),
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
    "timeframe",
    "entity_variable_name",
    "type",
    "extracted_value",
    "reason",
    "usage",
    "qualifier_predicates",
)

# Age/Sex/PregnancyStatus block (no canonical form field; qualifiers optional)
_REQUIRED_FIELDS_ASPS = (
    "span",
    "template",
    "timeframe",
    "entity_variable_name",
    "type",
    "extracted_value",
    "reason",
    "usage",
)

# Acceptable timeframe tokens (new closed vocab)
_TIMEFRAME_RE = re.compile(
    r"^("
    r"now|inthehistory|inthefuture|"
    r"inthepast\d+(minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(minutes|hours|days|weeks|months|years)"
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
    """Return regexes that a stem must match for a given template + timeframe."""
    tf = re.escape(timeframe)
    return {
        # Findings: has_finding_of_{entity}_{tf}
        "findings": re.compile(rf"^has_finding_of_[a-z0-9_]+_{tf}$"),
        # Procedures: {has_undergone|is_undergoing|will_undergo|can_undergo}_{entity}_{tf}
        "procedures": re.compile(
            rf"^(?:has_undergone|is_undergoing|will_undergo|can_undergo)_[a-z0-9_]+_{tf}$"
        ),
        # Observable numeric: {entity}_value_recorded_{tf}_withunit_{unit}
        "observable_entities_numeric": re.compile(
            rf"^[a-z0-9_]+_value_recorded_{tf}_withunit_[a-z0-9_]+$"
        ),
        # Observable status: {entity}_{is_positive|is_negative|is_adequate|is_inadequate}_{tf}
        "observable_entities_status": re.compile(
            rf"^[a-z0-9_]+_(?:is_positive|is_negative|is_adequate|is_inadequate)_{tf}$"
        ),
        # Product: {is_taking|has_taken}_{entity}_{tf}
        "product": re.compile(rf"^(?:is_taking|has_taken)_[a-z0-9_]+_{tf}$"),
        # Substance: is_exposed_to_{entity}_{tf}
        "substance": re.compile(rf"^is_exposed_to_[a-z0-9_]+_{tf}$"),
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
        re.compile(rf"^age_value_recorded_{tf}_in_(?:years|months|days)$"),
        re.compile(rf"^patient_sex_is_[a-z0-9_]+_{tf}$"),
        re.compile(rf"^patient_is_pregnant_{tf}$"),
        re.compile(rf"^patient_is_able_to_be_pregnant_{tf}$"),
    ]
    if not any(p.fullmatch(stem) for p in patterns):
        raise ValueError(
            f"demographics stem {stem!r} must match one of the allowed forms for timeframe={timeframe!r}"
        )

_QUAL_TIMEFRAME_OR_UNIT_INFIX = re.compile(
    r"(?:^|_)(?:now|inthehistory|inthefuture|inthe(?:past|future)\d+(?:days|weeks|months|years)|"
    r"withunit|in_(?:years|months|days))(?:_|$)"
)

def _ensure_qualifiers_do_not_repeat_time_or_unit(qp_list: list[str]) -> None:
    """Qualifiers must NOT repeat timeframe/value/unit semantics per naming system."""
    for qp in qp_list or []:
        # qp is "stem@@qual"
        try:
            _, qual = qp.split("@@", 1)
        except ValueError:
            qual = qp
        if _QUAL_TIMEFRAME_OR_UNIT_INFIX.search(qual):
            raise ValueError(
                f"qualifier {qp!r} must not contain timeframe/value/unit tokens "
                "(e.g., now/inthepast…/withunit/in_years|months|days)"
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
      inthepast{n}{unit} | inthefuture{n}{unit}
    Also maps legacy 'within{n}{unit}' → 'inthepast{n}{unit}'.
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
    enable_template_checks: bool = True,
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
        tf_raw = obj.get("timeframe", "")
        tf_norm = _canon_timeframe(tf_raw)
        obj["timeframe"] = tf_norm
        if not _TIMEFRAME_RE.match(tf_norm):
            raise ValueError(
                f"{kind} declaration #{i} has invalid timeframe={tf_raw!r} → {tf_norm!r}; "
                "use now|inthehistory|inthefuture|inthepast{n}{days|weeks|months|years}|inthefuture{n}{…}"
            )

        # entity_variable_name: sanitize (keep or force lower case based on toggle)
        stem_in = str(obj["entity_variable_name"])
        stem = _symbol_sanitize(stem_in, lower=lower_for_sanitize)
        obj["entity_variable_name"] = stem
        # surface I4 auto-corrections (soft note)
        if collect_sanitize_corrections is not None and stem_in != stem:
            collect_sanitize_corrections.append(
                {"kind": kind, "original": stem_in, "sanitized": stem}
            )

        # Optional suffix enforcement (not required; Stage-1 checks presence via _enforce_timeframe_in_stem)
        if enforce_suffix_in_stem:
            tf_seg_pat = re.compile(rf"(?:^|_){re.escape(tf_norm)}(?:_|$)")
            if not tf_seg_pat.search(stem):
                raise ValueError(
                    f"{kind} declaration #{i} entity_variable_name {stem!r} must include timeframe segment {tf_norm!r}"
                )

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

        # canonical-specific: entity_canonical_form_used must be a non-empty string
        if kind == "canonical":
            ecfu = obj.get("entity_canonical_form_used", "")
            if not isinstance(ecfu, str) or not ecfu.strip():
                raise ValueError(f"{kind} declaration #{i} must set non-empty 'entity_canonical_form_used'")
            # naming-system: template must be valid and stem must match its shape
            if enable_template_checks:
                _assert_stem_matches_template(stem, obj.get("template", ""), tf_norm)
        else:
            # demographics block shape check
            if enable_template_checks:
                _assert_asps_stem_shape(stem, tf_norm)

        out.append(obj)

    # Deterministic sort (I4)
    out.sort(key=_sort_key)
    return out


# ---------------------------------------------------------------------------
# === NEW: additional guards and helpers enforcing I1–I3 =====================
# ---------------------------------------------------------------------------
_TF_TOKEN_RE = re.compile(
    r"(?:^|_)(now|inthehistory|inthefuture|"
    r"inthepast\d+(?:days|weeks|months|years)|"
    r"inthefuture\d+(?:days|weeks|months|years))(?:_|$)"
)

# Capture Bool/Int/Real variable declarations
_DECL_ANY_RE = re.compile(r"\(declare-const\s+([^\s\)]+)\s+(?:Bool|Int|Real)\)")

def _extract_declared_symbols(smt_lines: List[str]) -> set[str]:
    names: set[str] = set()
    for ln in smt_lines or []:
        m = _DECL_ANY_RE.search(ln)
        if m:
            names.add(m.group(1))
    return names

def _enforce_timeframe_in_stem(lst: list[dict], *, allow_mixed_case: bool) -> None:
    for j, obj in enumerate(lst):
        stem = str(obj["entity_variable_name"])
        tf   = str(obj["timeframe"])
        found = [m.group(1) for m in _TF_TOKEN_RE.finditer(stem)]
        norm  = (lambda s: s.lower()) if allow_mixed_case else (lambda s: s)
        uniq  = sorted(set(norm(x) for x in found))
        if len(uniq) != 1 or norm(uniq[0]) != norm(tf):
            raise ValueError(
                f"stem/timeframe mismatch for '{stem}': found {found} but timeframe field is {tf!r}"
            )


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
        prefix = "has_undergone_"
    else:
        prefix = "has_finding_of_"
    return f"{prefix}{base}_{timeframe}"


# ---------------------------------------------------------------------------
# Variable-naming module (Stage-1: Demographics + Canonical only)
# ---------------------------------------------------------------------------
class PatientSMTVariableCoder(dspy.Module):
    """Stage-1 namer with soft de-dup, partial coverage, and robust fallback."""

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

    # ------------------------------------------------------------------
    def forward(self, context: Dict) -> Dict:  # type: ignore[override]
        requirements: List = context.get("requirements", [])
        idx: int = context["current_requirement_index"]
        if not requirements:
            return context

        # Entities-only mode (ctor arg > context > env)
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

        # Partial coverage allowed? (default TRUE per request)
        partial_ok_ctx = context.get("namer_allow_partial_canonical_coverage", None)
        if partial_ok_ctx is None:
            partial_coverage_ok = self._env_truthy("SMT_NAMER_ALLOW_PARTIAL_COVERAGE", True)
        else:
            partial_coverage_ok = bool(partial_ok_ctx)

        req_entry = requirements[idx]
        requirement_txt = (
            req_entry.get("requirement") if isinstance(req_entry, dict) else str(req_entry)
        )

        namer_prompt_tpl = context.get("PatientSMTVariableCoder_prompt", "")
        if not namer_prompt_tpl:
            _log("namer ✗", idx, "prompt template missing")
            return context

        # --- build canonical-form block --------------------------------
        bundles: List[dict] = context.get("requirement_bundles", [])
        bundles_by_index = {b.get("req_index"): b for b in bundles if isinstance(b, dict)}

        # Collect allowed canonical forms + lightweight type/span for fallback & gating
        allowed_canon_forms: set[str] = set()
        allowed_info: Dict[str, Dict[str, str]] = {}

        if entities_only:
            ver_all: dict = context.get("valid_entities_by_req", {}) or {}
            ver_for_req = ver_all.get(str(idx)) or {}
            if ver_for_req:
                for e in ver_for_req.values():
                    s = e.get("preferred_term") or e.get("entity_canonical_form")
                    if s:
                        allowed_canon_forms.add(s)
                        allowed_info[s] = {
                            "type": e.get("type", ""),
                            "span": e.get("extracted_span", s),
                        }
            canon_bundle = _minify_from_valid_entities(ver_for_req) if ver_for_req else ""
            if not canon_bundle:
                b = bundles_by_index.get(idx)
                canon_bundle = _minify_bundle(b, include_attributes=False)
                if b:
                    for ent in (b.get("entities") or []):
                        s = ent.get("entity_canonical_form")
                        if s:
                            allowed_canon_forms.add(s)
                            allowed_info[s] = {"type": ent.get("type", ""), "span": ent.get("span", s)}
                _log("namer", idx, "entities_only: empty valid_entities_by_req → fell back to requirement_bundles")
        else:
            b = bundles_by_index.get(idx)
            canon_bundle = _minify_bundle(b, include_attributes=True)
            if b:
                for ent in (b.get("entities") or []):
                    s = ent.get("entity_canonical_form")
                    if s:
                        allowed_canon_forms.add(s)
                        allowed_info[s] = {"type": ent.get("type", ""), "span": ent.get("span", s)}

        # --- build prompt ----------------------------------------------
        prompt = (
            namer_prompt_tpl
            .replace("#SMT_PROGRAM_BY_FAR#", "\n".join(context.get("smt_program_lines", [])))
            .replace("#CANONICAL_FORMS#", canon_bundle)
            .replace("#PATIENT_FACT#", requirement_txt)
            .replace("#PATIENT_NOTE#", context.get("requirement_text", ""))
        )

        # --- logging paths ---------------------------------------------
        note_id = context.get("note_id", "unknown_patient_note")

        base_dir = os.path.join(self.log_dir, note_id)
        os.makedirs(base_dir, exist_ok=True)

        base_name = f"req{idx:03d}_namer"
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

        # --- derive enforcement contexts --------------------------------
        existing_vars = _extract_declared_symbols(context.get("smt_program_lines", []))

        # --- model interaction loop ------------------------------------
        succeeded = False
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            llm_out: str = self.engine(prompt)[0]

            with open(r_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")

            m_canon = _NEWCANON_RE.search(llm_out)
            m_asps  = _NEWASPS_RE.search(llm_out)

            # Stage-1 policy: detect illegal blocks (ignore, flag)
            illegal_other = _NEWOTHER_RE.search(llm_out) is not None
            illegal_noncanon_legacy = _NEWNONCANON_RE.search(llm_out) is not None

            # Backcompat (very old single block) — treat as canonical by default
            if not m_canon:
                m_old = _NEWDECL_RE_OLD.search(llm_out)
                if m_old:
                    m_canon = m_old  # parsed below as canonical-shaped

            if not m_canon:
                last_error = RuntimeError(
                    f"Couldn’t find expected JSON blocks for requirement #{idx} on attempt {attempt}."
                )
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"missing blocks; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue

            try:
                canon_list_raw = _parse_json_array_relaxed(m_canon.group(1))
                asps_list_raw  = _parse_json_array_relaxed(m_asps.group(1)) if m_asps else []
                # optional <errors> collection
                m_err = _ERRORS_RE.search(llm_out)
                errors_list = _parse_json_array_relaxed(m_err.group(1)) if m_err else []
            except Exception as exc:
                last_error = exc
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"JSON error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue

            # Stage-1: record policy error if illegal blocks were emitted
            if illegal_other or illegal_noncanon_legacy:
                errors_list = errors_list or []
                errors_list.append({
                    "invariant": "S1",
                    "problem": "Non-canonical block emitted in Stage-1 "
                               "(all_other_variable_declarations/new_noncanonical_variable_declarations)",
                    "fix": "Ignored in Stage-1; handle in Stage-2 module."
                })

            # Validate + sort (enforcing timeframe & @@ qualifiers)
            try:
                # 1) declaration lists (Stage-1: only ASPS + CANON)
                sanitize_notes: list[dict] = []
                canon_list = _validate_decl_list(
                    canon_list_raw,
                    kind="canonical",
                    allow_backcompat=True,
                    enforce_suffix_in_stem=self._enforce_suffix_in_stem,
                    allow_mixed_case=self._allow_mixed_case,
                    collect_sanitize_corrections=sanitize_notes,
                    enable_template_checks=True,
                )
                canon_list_pre_dedup = list(canon_list)  # snapshot for coverage via existing vars
                asps_list = _validate_decl_list(
                    asps_list_raw,
                    kind="age_sex_preg",
                    allow_backcompat=False,  # explicit new schema
                    enforce_suffix_in_stem=self._enforce_suffix_in_stem,
                    allow_mixed_case=self._allow_mixed_case,
                    collect_sanitize_corrections=sanitize_notes,
                    enable_template_checks=True,
                )

                # 2) I3: canonical gating – entity must be from #CANONICAL_FORMS#
                if allowed_canon_forms:
                    bad = [c for c in canon_list if c["entity_canonical_form_used"] not in allowed_canon_forms]
                    if bad:
                        raise ValueError(
                            "canonical declarations include entities not in #CANONICAL_FORMS#: "
                            + ", ".join(sorted({b["entity_canonical_form_used"] for b in bad}))
                        )

                # 3) I1: timeframe token must appear exactly once in stem and match field
                _enforce_timeframe_in_stem(canon_list, allow_mixed_case=self._allow_mixed_case)
                _enforce_timeframe_in_stem(asps_list,  allow_mixed_case=self._allow_mixed_case)

                # 4) de-dup — drop any new decl whose name already exists in the SMT program
                stage_new = asps_list + canon_list
                new_names = [o["entity_variable_name"] for o in stage_new]
                dups_existing = sorted({n for n in new_names if n in existing_vars})
                if dups_existing:
                    # filter out colliding entries
                    asps_list  = [o for o in asps_list  if o["entity_variable_name"] not in dups_existing]
                    canon_list = [o for o in canon_list if o["entity_variable_name"] not in dups_existing]
                    # soft warning (log + warnings + plan/errors)
                    msg = "dedup: dropped re-declarations → " + ", ".join(dups_existing)
                    _log("namer ⚠", idx, msg)
                    warnings.warn(f"SMT namer soft-dedup (req#{idx}): {msg}", RuntimeWarning)
                    errors_list = (errors_list or [])
                    errors_list.append({
                        "invariant": "I2-soft",
                        "problem": "attempted to redeclare existing variables; dropped from new declarations",
                        "dropped_variables": dups_existing,
                    })

                # 5) Re-check coverage after considering existing vars
                if allowed_canon_forms:
                    # present in remaining new canon_list
                    present = {c["entity_canonical_form_used"] for c in canon_list}
                    # also count those whose stems already exist (dedupbed above)
                    present |= {
                        c["entity_canonical_form_used"]
                        for c in canon_list_pre_dedup
                        if c.get("entity_variable_name") in dups_existing
                    }
                    missing = sorted(allowed_canon_forms - present)
                    if missing:
                        if coverage_required and not partial_coverage_ok:
                            raise ValueError("missing canonical declarations for: " + ", ".join(missing))
                        # soft path: log & annotate, proceed
                        warn_msg = "partial canonical coverage; missing: " + ", ".join(missing)
                        _log("namer ⚠", idx, warn_msg)
                        warnings.warn(f"SMT namer partial coverage (req#{idx}): {warn_msg}", RuntimeWarning)
                        errors_list.append({
                            "invariant": "I3-partial-coverage",
                            "problem": "not all canonical entities were declared; proceeding with subset",
                            "missing_canonical_forms": missing,
                        })

                # 6) uniqueness across Stage-1 new blocks
                stage_new = asps_list + canon_list
                new_names = [o["entity_variable_name"] for o in stage_new]
                dups_cross = {n for n in new_names if new_names.count(n) > 1}
                if dups_cross:
                    raise ValueError("duplicate variable names across new declarations: " + ", ".join(sorted(dups_cross)))

                # surface I4 sanitization notes (soft)
                if sanitize_notes:
                    errors_list = (errors_list or [])
                    errors_list.append({
                        "invariant": "I4-sanitize",
                        "problem": "auto-corrected names to lowercase snake_case; ensure model emits canonical form",
                        "corrections": sanitize_notes,
                    })

            except ValueError as ve:
                last_error = ve
                if attempt == self.MAX_ATTEMPTS:
                    break
                _log("namer", idx, f"schema error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                continue

            # --- success ------------------------------------------------
            # strip internal debug keys before persisting
            for _o in canon_list:
                _o.pop("_original_entity_variable_name", None)
            for _o in asps_list:
                _o.pop("_original_entity_variable_name", None)
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "new_age_sex_pregnancystatus_declarations": asps_list,
                        "new_canonical_variable_declarations": canon_list,
                        "all_other_variable_declarations": [],  # Stage-1: always empty
                        "errors": errors_list or [],
                        "stage": "demographics_canonical",
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            # Update context (Stage-1 outputs only)
            context["new_age_sex_pregnancystatus_declarations"] = asps_list
            context["new_canonical_variable_declarations"] = canon_list

            # Stage-1: explicitly empty non-canonical outputs for downstream safety
            context["all_other_variable_declarations"] = []
            context["new_noncanonical_variable_declarations"] = []

            # Convenience merged list for downstreams — Stage-1 only
            context["new_variable_declarations"] = asps_list + canon_list

            context["errors"] = errors_list or []
            context["namer_mode"] = "entities_only" if entities_only else "full"
            context["namer_stage"] = "demographics_canonical"

            _log("namer ✓", idx, f"accepted on attempt {attempt}")
            succeeded = True
            break

        # --------------------- FALLBACK if all attempts failed ---------------------
        if not succeeded:
            _log("namer ⚠ fallback", idx, f"entering fallback due to: {last_error!r}")
            warnings.warn(
                f"SMT namer fallback engaged (req#{idx}): {last_error}", RuntimeWarning
            )

            errors_list = [{
                "invariant": "FALLBACK",
                "problem": "LLM outputs could not be parsed/validated after retries; synthesized declarations",
                "detail": str(last_error) if last_error else "unknown",
            }]

            # Synthesize canonical declarations from allowed forms (best-effort)
            canon_list: list[dict] = []
            for canon in sorted(allowed_canon_forms):
                info = allowed_info.get(canon, {"type": "", "span": canon})
                typ = info.get("type", "")
                timeframe = _default_timeframe_for_type(typ)
                stem = _synthesize_stem(canon, typ, timeframe, allow_mixed_case=self._allow_mixed_case)
                template = _template_for_type(typ)

                if stem in existing_vars:
                    # Do not redeclare; count towards coverage below
                    continue
                else:
                    canon_list.append({
                        "span": info.get("span", canon),
                        "entity_variable_name": stem,
                        "template": template,
                        "timeframe": timeframe,
                        "entity_canonical_form_used": canon,
                        "usage": "fallback synthesized declaration to preserve canonical coverage",
                        "qualifier_predicates": [],
                    })

            # Sort outputs
            canon_list.sort(key=_sort_key)

            # Enforce timeframe-in-stem invariant (defensive; soft on failure)
            try:
                _enforce_timeframe_in_stem(canon_list, allow_mixed_case=self._allow_mixed_case)
            except Exception as e:
                errors_list.append({
                    "invariant": "I1-relaxed-fallback",
                    "problem": "timeframe token check failed in synthesized stem; proceeding",
                    "detail": str(e),
                })

            # Coverage in fallback (count existing stems as covered)
            present_forms = {c["entity_canonical_form_used"] for c in canon_list}
            for canon in allowed_canon_forms:
                info = allowed_info.get(canon, {"type": "", "span": canon})
                typ = info.get("type", "")
                timeframe = _default_timeframe_for_type(typ)
                synth_stem = _synthesize_stem(canon, typ, timeframe, allow_mixed_case=self._allow_mixed_case)
                if synth_stem in existing_vars:
                    present_forms.add(canon)

            missing = sorted(allowed_canon_forms - present_forms)
            if missing:
                errors_list.append({
                    "invariant": "I3-coverage-relaxed-fallback",
                    "problem": "coverage relaxed in fallback; some canonical forms not synthesized (or already existed)",
                    "missing_canonical_forms": missing,
                })

            # Finalize plan
            asps_list: list[dict] = []  # demographics unchanged in fallback
            for _o in canon_list:
                _o.pop("_original_entity_variable_name", None)
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "new_age_sex_pregnancystatus_declarations": asps_list,
                        "new_canonical_variable_declarations": canon_list,
                        "all_other_variable_declarations": [],
                        "errors": errors_list,
                        "stage": "demographics_canonical",
                        "fallback": True,
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            # Update context with fallback outputs
            context["new_age_sex_pregnancystatus_declarations"] = asps_list
            context["new_canonical_variable_declarations"] = canon_list
            context["all_other_variable_declarations"] = []
            context["new_noncanonical_variable_declarations"] = []
            context["new_variable_declarations"] = asps_list + canon_list
            context["errors"] = errors_list
            context["namer_mode"] = "entities_only" if entities_only else "full"
            context["namer_stage"] = "demographics_canonical"
            context["namer_fallback"] = True

            _log("namer ✓ fallback", idx, f"synthesized {len(canon_list)} canonical decls")

        return context
