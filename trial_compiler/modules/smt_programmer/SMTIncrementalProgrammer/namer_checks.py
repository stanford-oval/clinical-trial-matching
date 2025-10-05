# namer_checks.py
from __future__ import annotations
import json, logging, re, sys
from typing import Any, List

# ---------------------------------------------------------------------------
# Fallback logger (same behavior as caller modules)
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
__all__ = [
    "_schema_log", "_get_canon",
    "_REUSABLE_RE", "_NEWCANON_RE", "_NEWASPS_RE", "_NEWOTHER_RE", "_NEWNONCANON_RE", "_ERRORS_RE", "_NEWDECL_RE_OLD",
    "_strip_code_fence", "_parse_json_array_relaxed",
    "_canon_timeframe", "_ensure_qualifiers_do_not_repeat_time_or_unit",
    "_symbol_sanitize", "_normalize_template_field", "_normalize_template_field_public",
    "_extract_declared_symbols", "_validate_reusable_vars", "_validate_decl_list",
    "_enforce_timeframe_in_stem", "_sort_key",
    "_default_timeframe_for_type", "_template_for_type", "_synthesize_stem",
]
# ---------------------------------------------------------------------------
# Public: schema logger
# ---------------------------------------------------------------------------
def _schema_log(
    diag: list | None,
    *,
    idx: int | None,
    block: str,
    item_index: int | None,
    code: str,
    message: str,
    **extra: Any,
) -> None:
    entry = {"block": block, "index": item_index, "code": code, "message": message}
    if extra:
        entry["extra"] = extra
    if diag is not None:
        diag.append(entry)
    try:
        label = f"[{block}{'' if item_index is None else f'[{item_index}]'}] {code}: {message}"
        if extra:
            label += f" | extra={extra}"
        _log("namer ✗ schema", idx or -1, label)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Canonical key helpers
# ---------------------------------------------------------------------------
_CANON_KEY_NEW = "entity_canonical_form_from_entity_canonical_forms_block"
_CANON_KEY_OLD_1 = "entity_canonical_form_used"
_CANON_KEY_OLD_2 = "entity_canonical_form"

def _get_canon(ent_like: dict) -> str:
    if not isinstance(ent_like, dict):
        return ""
    return (
        ent_like.get(_CANON_KEY_NEW)
        or ent_like.get(_CANON_KEY_OLD_1)
        or ent_like.get(_CANON_KEY_OLD_2)
        or ent_like.get("preferred_term")
        or ""
    )

# ---------------------------------------------------------------------------
# Regex extractors for blocks
# ---------------------------------------------------------------------------
_REUSABLE_RE   = re.compile(r"<reusable_variables>\s*(\[.*?])\s*</reusable_variables>", re.I | re.S)
_NEWCANON_RE   = re.compile(r"<new_canonical_variable_declarations>\s*(\[.*?])\s*</new_canonical_variable_declarations>", re.I | re.S)
_NEWASPS_RE    = re.compile(r"<new_age_sex_pregnancystatus_declarations>\s*(\[.*?])\s*</new_age_sex_pregnancystatus_declarations>", re.I | re.S)
_NEWOTHER_RE   = re.compile(r"<all_other_variable_declarations>\s*(\[.*?])\s*</all_other_variable_declarations>", re.I | re.S)
_NEWNONCANON_RE= re.compile(r"<new_noncanonical_variable_declarations>\s*(\[.*?])\s*</new_noncanonical_variable_declarations>", re.I | re.S)
_ERRORS_RE     = re.compile(r"<errors>\s*(\[.*?])\s*</errors>", re.I | re.S)
_NEWDECL_RE_OLD= re.compile(r"<new_variable_declarations>\s*(\[.*?])\s*</new_variable_declarations>", re.I | re.S)
_TAG_RE        = re.compile(r":named\s+([Rr]\d+_A\d+_[A-Z0-9_]+)")

# ---------------------------------------------------------------------------
# Relaxed JSON parsing helpers
# ---------------------------------------------------------------------------
_CODEFENCE_RE = re.compile(r"^\s*```(?:json|JSON|smt|SMT|text)?\s*|\s*```\s*$", re.M)

def _strip_code_fence(s: str) -> str:
    return re.sub(_CODEFENCE_RE, "", s)

def _strip_json_comments_and_trailing_commas(s: str) -> str:
    s = re.sub(r"(?m)^\s*//.*$", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
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

# ---------------------------------------------------------------------------
# Timeframe + qualifier token policies
# ---------------------------------------------------------------------------
_TIMEFRAME_RE = re.compile(
    r"^(now|inthehistory|inthefuture|"
    r"inthepast\d+(minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(minutes|hours|days|weeks|months|years))$"
)


_QUAL_TIMEFRAME_OR_UNIT_INFIX = re.compile(
    r"(?:^|_)(?:now|inthehistory|inthefuture|"
    r"inthe(?:past|future)\d+(?:minutes|hours|days|weeks|months|years)|"
    r"foradurationof\d+(?:minutes|hours|days|weeks|months|years)|"
    r"withunit|in_(?:years|months|days))(?:_|$)"
)


def _ensure_qualifiers_do_not_repeat_time_or_unit(qp_list: list[str]) -> None:
    for qp in qp_list or []:
        try:
            _, qual = qp.split("@@", 1)
        except ValueError:
            qual = qp
        if _QUAL_TIMEFRAME_OR_UNIT_INFIX.search(qual):
            raise ValueError(
                f"qualifier {qp!r} must not contain timeframe/value/unit tokens "
                "(e.g., now/inthepast…/withunit/in_years|months|days)"
            )

# --- timeframe canonicalizer ---
_TIME_UNIT_CANON = {
    "y": "years", "yr": "years", "yrs": "years", "year": "years", "years": "years",
    "m": "months", "mo": "months", "mos": "months", "mon": "months", "month": "months", "months": "months",
    "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks",
    "d": "days", "day": "days", "days": "days",
    "h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours",
    "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
}

def _canon_timeframe(tf: str) -> str:
    raw = str(tf or "").strip().lower()
    s = re.sub(r"[\s_\-]+", "", raw)
    if s in {"now","today","present","current"}: return "now"
    if s in {"history","inthehistory","past","previous"}: return "inthehistory"
    if s in {"future","inthefuture","upcoming"}: return "inthefuture"

    m = re.match(r"^foradurationof(\d+)([a-z]+)$", s)
    if m:
        n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"foradurationof{n}{unit}"

    m = re.match(r"^within(\d+)([a-z]+)$", s)
    if m:
        n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"inthepast{n}{unit}"

    m = re.match(r"^inthe(past|future)(\d+)([a-z]+)$", s)
    if m:
        pf, n, unit = m.groups()
        unit = _TIME_UNIT_CANON.get(unit, unit)
        return f"inthe{pf}{n}{unit}"

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

    if _TIMEFRAME_RE.match(raw): return raw
    return s

# ---------------------------------------------------------------------------
# Sanitizers
# ---------------------------------------------------------------------------
def _symbol_sanitize(s: str, *, lower: bool) -> str:
    """
    Sanitize a variable stem while preserving the composed qualifier separator '@@'.
    - Lowercases if `lower=True`
    - Replaces disallowed chars with underscores
    - Keeps a single '@@' segment intact (anywhere in the stem)
    """
    if not isinstance(s, str):
        s = str(s or "")
    s = s.strip()
    if lower:
        s = s.lower()

    # 1) Protect composed separator(s)
    # Use a sentinel that will survive the regex below.
    SENT = "§§"
    s = s.replace("@@", SENT)

    # 2) Gentle normalizations that shouldn't break tokens
    s = s.replace("+", "plus").replace("/", "_or_")
    # turn digit-digit hyphens into 'to' (e.g., 'ct2-ct4a' -> 'ct2toct4a')
    s = re.sub(r'(?<=\d)-(?=\d[a-zA-Z]?)', 'to', s)
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)

    # 3) Strip anything not allowed in our symbols (allow sentinel + underscore)
    s = re.sub(r"[^A-Za-z0-9_§]", "_", s)

    # 4) Collapse underscores and trim
    s = re.sub(r"_+", "_", s).strip("_")

    # 5) Restore composed separator
    s = s.replace(SENT, "@@")
    return s


# ---------------------------------------------------------------------------
# Template normalization (no stem-shape enforcement for canonical)
# ---------------------------------------------------------------------------
_ALLOWED_TEMPLATES = {
    "findings","procedures","observable_entities_numeric",
    "observable_entities_status","product","substance"
}
# _TEMPLATE_SYNONYM_MAP = [
#     (re.compile(r"^has_finding_of_"), "findings"),
#     (re.compile(r"^(?:has_undergone|is_undergoing|will_undergo|can_undergo)_"), "procedures"),
#     (re.compile(r"_value_recorded_.*_withunit_"), "observable_entities_numeric"),
#     (re.compile(r"_(?:is_positive|is_negative|is_adequate|is_inadequate)_"), "observable_entities_status"),
#     (re.compile(r"^(?:is_taking|has_taken)_"), "product"),
#     (re.compile(r"^is_exposed_to_"), "substance"),
# ]

_TEMPLATE_SYNONYM_MAP = [
    # ---- 新增：patient*/patients* 变体 ----
    (re.compile(r"^patient_has_(?:diagnosis|finding|symptoms|clinical_signs|suspicion)_of_"), "findings"),
    (re.compile(
        r"^patient_(?:has_undergone|is_undergoing|will_undergo|can_undergo|needs_to_undergo)_"
    ), "procedures"),
    (re.compile(r"^patient_has_undergone_.*_outcome_is_(?:positive|negative|normal|abnormal)"), "procedures"),
    (re.compile(r"^patient_.*_value_recorded_.*_withunit_"), "observable_entities_numeric"),
    (re.compile(r"^patients_.*_is_(?:positive|negative|normal|abnormal)_"), "observable_entities_status"),
    (re.compile(r"_(?:is_positive|is_negative|is_adequate|is_inadequate|is_normal|is_abnormal)_"), "observable_entities_status"),
    (re.compile(r"^patient_(?:is_taking|has_taken)_"), "product"),
    (re.compile(r"^patient_is_exposed_to_"), "substance"),
]


def _normalize_template_field(tpl: str) -> str:
    t = (tpl or "").strip().lower()
    if t in _ALLOWED_TEMPLATES: return t
    for rx, label in _TEMPLATE_SYNONYM_MAP:
        if rx.search(t): return label
    return t

# ---------------------------------------------------------------------------
# Validators & helpers
# ---------------------------------------------------------------------------
def _sort_key(obj: dict) -> tuple:
    return (
        obj.get(_CANON_KEY_NEW, "") or "",
        obj.get("timeframe", "") or "",
        obj.get("entity_variable_name", "") or "",
    )

# any declare-const symbol
_DECL_ANY_RE = re.compile(r"\(declare-const\s+([^\s\)]+)\s+(?:Bool|Int|Real)\)")

def _extract_declared_symbols(smt_lines: List[str]) -> set[str]:
    names: set[str] = set()
    for ln in smt_lines or []:
        m = _DECL_ANY_RE.search(ln)
        if m: names.add(m.group(1))
    return names

def _assert_asps_stem_shape(stem: str, timeframe: str) -> None:
    tf = re.escape(timeframe)
    patterns = [
        # age（新旧兼容）
        re.compile(rf"^patient_age_value_recorded_{tf}_in_(?:years|months|days)$"),
        re.compile(rf"^age_value_recorded_{tf}_in_(?:years|months|days)$"),
        # sex / pregnancy（既有）
        re.compile(rf"^patient_sex_is_[a-z0-9_]+_{tf}$"),
        re.compile(rf"^patient_is_pregnant_{tf}$"),
        re.compile(rf"^patient_is_able_to_be_pregnant_{tf}$"),
        # 新增：
        re.compile(rf"^patient_has_childbearing_potential_{tf}$"),
        re.compile(rf"^patient_is_breastfeeding_{tf}$"),
        re.compile(rf"^patient_is_lactating_{tf}$"),
    ]
    if not any(p.fullmatch(stem) for p in patterns):
        raise ValueError(
            f"demographics stem {stem!r} must match one of the allowed forms for timeframe={timeframe!r}"
        )

def _enforce_timeframe_in_stem(
    lst: list[dict],
    *,
    allow_mixed_case: bool,
    diag: list | None = None,
    context_idx: int | None = None,
    block_label: str = "canonical_or_demo"
) -> None:
    """
    Enforce that the STEM (substring before '@@') contains exactly one timeframe
    token and that it matches the 'timeframe' field. Treat '@@' as a boundary.
    """
    TF_TOKEN_RE = re.compile(
        r"(?:^|_)(now|inthehistory|inthefuture|"
        r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
        r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
        r"foradurationof\d+(?:minutes|hours|days|weeks|months|years))(?:_|$)",
        re.I if allow_mixed_case else 0,
    )


    def _norm(s: str) -> str:
        return s.lower() if allow_mixed_case else s

    for j, obj in enumerate(lst):
        stem_full = str(obj["entity_variable_name"])
        tf_field  = str(obj["timeframe"])
        # check ONLY the stem (before '@@')
        stem = stem_full.split("@@", 1)[0]

        found = [m.group(1) for m in TF_TOKEN_RE.finditer(stem)]
        uniq  = sorted(set(_norm(x) for x in found))
        if len(uniq) != 1 or _norm(uniq[0]) != _norm(tf_field):
            _schema_log(diag, idx=context_idx, block=block_label, item_index=j,
                        code="TIMEFRAME_MISMATCH",
                        message="Timeframe token(s) in stem do not match the 'timeframe' field.",
                        entity_variable_name=stem_full,  # keep original for visibility
                        timeframe=tf_field, timeframe_tokens_found=found)
            raise ValueError(
                f"stem/timeframe mismatch for '{stem_full}': found {found} in stem '{stem}' "
                f"but timeframe field is {tf_field!r}"
            )


def _validate_reusable_vars(
    rv: Any,
    existing: set[str],
    *,
    allow_mixed_case: bool,
    diag: list | None = None,
    context_idx: int | None = None,
    block_label: str = "reusable_variables",
) -> list[dict]:
    if not isinstance(rv, list):
        _schema_log(diag, idx=context_idx, block=block_label, item_index=None,
                    code="NOT_ARRAY", message="Expected JSON array for <reusable_variables>.",
                    got_type=type(rv).__name__)
        raise ValueError("<reusable_variables> must be a JSON array")

    out: list[dict] = []
    name_pat = r"[A-Za-z0-9_]+" if allow_mixed_case else r"[a-z0-9_]+"
    rx = re.compile(rf"^{name_pat}$")
    for i, it in enumerate(rv):
        if not isinstance(it, dict) or "variable_name" not in it or "why" not in it:
            _schema_log(diag, idx=context_idx, block=block_label, item_index=i,
                        code="SHAPE", message="Missing 'variable_name' or 'why' in item.", item=it)
            raise ValueError(f"<reusable_variables>[{i}] must have 'variable_name' and 'why'")
        name = str(it["variable_name"])
        if not rx.fullmatch(name):
            _schema_log(diag, idx=context_idx, block=block_label, item_index=i,
                        code="BAD_NAME", message="Invalid variable_name format.",
                        variable_name=name, allow_mixed_case=allow_mixed_case)
            raise ValueError(f"<reusable_variables>[{i}] invalid variable_name={name!r}")
        if name not in existing:
            _schema_log(diag, idx=context_idx, block=block_label, item_index=i,
                        code="NOT_DECLARED", message="Variable not found in SMT program for reuse.",
                        variable_name=name)
            raise ValueError(f"<reusable_variables>[{i}] {name!r} not found in SMT program")
        out.append({"variable_name": name, "why": str(it["why"])})

    out.sort(key=lambda d: d["variable_name"])
    return out

def _normalize_template_field_public(tpl: str) -> str:
    # thin wrapper to export consistent name if you prefer not to re-export the private one
    return _normalize_template_field(tpl)

def _validate_decl_list(
    items: Any,
    *,
    kind: str,  # "canonical" | "age_sex_preg"
    allow_backcompat: bool = True,
    enforce_suffix_in_stem: bool = False,
    allow_mixed_case: bool = True,
    collect_sanitize_corrections: list | None = None,
    enable_template_checks: bool = True,
    diag: list | None = None,
    context_idx: int | None = None,
    block_label: str | None = None,
) -> list[dict]:
    block = block_label or kind
    if not isinstance(items, list):
        _schema_log(diag, idx=context_idx, block=block, item_index=None,
                    code="NOT_ARRAY", message="Expected JSON array for declarations.",
                    got_type=type(items).__name__)
        raise ValueError("declarations must be a JSON array")

    REQUIRED_CANON = (
        "span","entity_variable_name","template","timeframe",
        _CANON_KEY_NEW,"usage_description","qualifier_predicates_for_semantics_not_already_captured_with_stem"
    )
    REQUIRED_ASPS = ("span","entity_variable_name","template","timeframe","usage_description")
    required = REQUIRED_CANON if kind == "canonical" else REQUIRED_ASPS

    if allow_mixed_case:
        base_pat = r"[A-Za-z0-9_]+"
        composed_pat = r"[A-Za-z0-9_]+(?:@@[A-Za-z0-9_]+)?"
    else:
        base_pat = r"[a-z0-9_]+"
        composed_pat = r"[a-z0-9_]+(?:@@[a-z0-9_]+)?"
    stem_rx = re.compile(rf"^{composed_pat}$")
    out: list[dict] = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            _schema_log(diag, idx=context_idx, block=block, item_index=i,
                        code="NOT_OBJECT", message="Each declaration must be a JSON object.",
                        got_type=type(raw).__name__)
            raise ValueError(f"{kind} declaration #{i} must be an object")

        obj = dict(raw)
        # Back-compat canonical fields
        if allow_backcompat and kind == "canonical":
            if _CANON_KEY_NEW not in obj:
                for legacy in (_CANON_KEY_OLD_1, _CANON_KEY_OLD_2):
                    if legacy in obj and isinstance(obj[legacy], str) and obj[legacy].strip():
                        obj[_CANON_KEY_NEW] = obj.pop(legacy); break
            obj.pop("is_canonical", None)  # ignore legacy noise

        missing = [k for k in required if k not in obj]
        if missing:
            _schema_log(diag, idx=context_idx, block=block, item_index=i,
                        code="MISSING_KEYS", message="Declaration missing required keys.",
                        missing=missing, present=list(obj.keys()))
        # timeframe normalize/validate
        tf_raw = obj.get("timeframe","")
        tf_norm = _canon_timeframe(tf_raw)
        obj["timeframe"] = tf_norm
        if not _TIMEFRAME_RE.match(tf_norm):
            _schema_log(diag, idx=context_idx, block=block, item_index=i,
                        code="BAD_TIMEFRAME", message="Invalid timeframe token.",
                        raw=tf_raw, normalized=tf_norm,
                        allowed="now|inthehistory|inthefuture|inthepastN<unit>|inthefutureN<unit>")
            raise ValueError(f"{kind} declaration #{i} invalid timeframe={tf_raw!r} → {tf_norm!r}")

        # sanitize stem
        stem_in = str(obj.get("entity_variable_name",""))
        stem = _symbol_sanitize(stem_in, lower=not allow_mixed_case)
        obj["entity_variable_name"] = stem
        if collect_sanitize_corrections is not None and stem_in != stem:
            collect_sanitize_corrections.append({"kind": kind, "original": stem_in, "sanitized": stem})

        if enforce_suffix_in_stem:
            flags = re.I if allow_mixed_case else 0
            # Accept '@@' as a legitimate boundary after the timeframe token
            tf_seg_pat = re.compile(rf"(?:^|_|@@){re.escape(tf_norm)}(?:_|$|@@)", flags)
            if not tf_seg_pat.search(stem):
                _schema_log(diag, idx=context_idx, block=block, item_index=i,
                            code="TIMEFRAME_SUFFIX",
                            message="Stem missing timeframe segment.",
                            entity_variable_name=stem, timeframe=tf_norm)
                raise ValueError(f"{kind} declaration #{i} entity_variable_name {stem!r} must include timeframe {tf_norm!r}")

        # qualifiers (array) basic check
        if kind == "canonical" or ("qualifier_predicates" in obj):
            qps = obj.get("qualifier_predicates", [])
            if not isinstance(qps, list):
                _schema_log(diag, idx=context_idx, block=block, item_index=i,
                            code="QP_NOT_ARRAY", message="'qualifier_predicates' must be an array.",
                            got_type=type(qps).__name__)
                raise ValueError(f"{kind} declaration #{i} 'qualifier_predicates' must be an array")
            _ensure_qualifiers_do_not_repeat_time_or_unit(qps)

        if not stem_rx.fullmatch(stem):
            _schema_log(diag, idx=context_idx, block=block, item_index=i,
                        code="STEM_CHARS", message="Invalid characters in entity_variable_name.",
                        entity_variable_name=stem, allow_mixed_case=allow_mixed_case)
            raise ValueError(f"{kind} declaration #{i} invalid entity_variable_name={stem!r}")

        if kind == "canonical":
            ecf = obj.get(_CANON_KEY_NEW,"")
            if not isinstance(ecf,str) or not ecf.strip():
                _schema_log(diag, idx=context_idx, block=block, item_index=i,
                            code="EMPTY_CANON", message=f"Non-empty '{_CANON_KEY_NEW}' required.")
                raise ValueError(f"{kind} declaration #{i} must set non-empty '{_CANON_KEY_NEW}'")
            obj["template"] = _normalize_template_field(obj.get("template",""))
        else:
            # demographic stem shape checks
            try:
                _assert_asps_stem_shape(stem, tf_norm)
            except ValueError as ve:
                _schema_log(diag, idx=context_idx, block=block, item_index=i,
                            code="DEMOGRAPHIC_STEM_SHAPE", message=str(ve),
                            entity_variable_name=stem, timeframe=tf_norm)
                raise

        out.append(obj)

    out.sort(key=_sort_key)
    return out


# --- fallback helpers ------------------------------------------------------

def _default_timeframe_for_type(typ: str) -> str:
    """
    Choose a coarse timeframe token based on a SNOMED-like type string.
    Procedures and observables default to 'inthehistory'; everything else → 'now'.
    """
    t = (typ or "").lower()
    if "procedure" in t:
        return "inthehistory"
    if "observable" in t or "measurement" in t or "lab" in t:
        return "inthehistory"
    return "now"


def _template_for_type(typ: str) -> str:
    """
    Pick a naming template family based on the entity type.
    """
    t = (typ or "").lower()
    if "procedure" in t:
        return "procedures"
    if "observable" in t or "measurement" in t or "lab" in t:
        return "observable_entities_status"
    if "product" in t or "drug" in t or "medication" in t or "pharmaceutical" in t:
        return "product"
    if "substance" in t or "exposure" in t:
        return "substance"
    return "findings"


def _synthesize_stem(canon: str, typ: str, timeframe: str, *, allow_mixed_case: bool) -> str:
    base = _symbol_sanitize(canon, lower=not allow_mixed_case)
    t = (typ or "").lower()

    if "procedure" in t:
        # Use 'needs_to_undergo' for now/future timeframes, keep 'has_undergone' for history.
        tf = timeframe or ""
        is_futurey = tf.startswith("inthefuture") or tf == "inthefuture" or tf == "now"
        if is_futurey:
            return f"patient_needs_to_undergo_{base}_{timeframe}"
        return f"patient_has_undergone_{base}_{timeframe}"

    if "observable" in t or "measurement" in t or "lab" in t:
        return f"patients_{base}_is_positive_{timeframe}"  # 状态类使用 patients_ 前缀
    if "product" in t or "drug" in t or "medication" in t or "pharmaceutical" in t:
        return f"patient_is_taking_{base}_{timeframe}"
    if "substance" in t or "exposure" in t:
        return f"patient_is_exposed_to_{base}_{timeframe}"
    return f"patient_has_finding_of_{base}_{timeframe}"


