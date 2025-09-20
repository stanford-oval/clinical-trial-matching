# modules/PatientStateExplicitDiagnoseExtractor.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import re
import math
import dspy

_OPEN = "<diagnosis_list>"
_CLOSE = "</diagnosis_list>"


# ---------- extraction helpers ----------
def _extract_block(raw: str) -> str | None:
    """Extract the <diagnosis_list>...</diagnosis_list> block."""
    m = re.search(re.escape(_OPEN) + r"(.*?)" + re.escape(_CLOSE), raw, flags=re.S | re.I)
    return m.group(1).strip() if m else None


def _parse_json_block(raw: str) -> List[Dict[str, Any]] | None:
    """
    Extract the <diagnosis_list>...</diagnosis_list> block and parse into a list[dict].
    Repairs common JSON-ish issues for the temporal bound dictionaries.
    """
    body = _extract_block(raw)
    if not body:
        return None

    # 1) Normalize quotes/whitespace
    body = (
        body.replace("\u201c", '"').replace("\u201d", '"')  # curly double
            .replace("\u2018", "'").replace("\u2019", "'")  # curly single
    )
    body = re.sub(r"[ \t]+\n", "\n", body)

    # 2) Field-targeted repairs for bound dictionaries:
    #    temporal_magnitude: allow Inf/inf -> "Inf"
    body = re.sub(r'("temporal_magnitude"\s*:\s*)(Inf|inf)\b', r'\1"Inf"', body)

    #    temporal_direction: quote bare enums (past|now)
    body = re.sub(
        r'("temporal_direction"\s*:\s*)(past|now)\b',
        r'\1"\2"', body, flags=re.I
    )

    #    units: quote bare words
    body = re.sub(
        r'("units"\s*:\s*)([A-Za-z]+)\b',
        r'\1"\2"', body
    )

    #    inclusive: Python booleans -> JSON booleans
    body = re.sub(r'("inclusive"\s*:\s*)True\b', r'\1true', body)
    body = re.sub(r'("inclusive"\s*:\s*)False\b', r'\1false', body)

    # 3) Be tolerant to trailing commas
    body = re.sub(r",\s*([}\]])", r"\1", body)

    stripped = body.strip()
    # 4) Ensure it's an array at top-level; wrap if the model returns a lone object
    if not (stripped.startswith("[") and stripped.endswith("]")):
        if stripped.startswith("{") or re.search(r'^\s*\{', stripped, flags=re.M):
            body = "[\n" + body + "\n]"
            stripped = body.strip()

    # 5) Parse JSON
    try:
        arr = json.loads(stripped)
        return arr if isinstance(arr, list) else None
    except Exception:
        # Last resort: try parsing a single object and wrap
        try:
            obj = json.loads(stripped)
            return [obj] if isinstance(obj, dict) else None
        except Exception:
            return None


def _norm_diag_name(x: str) -> str:
    return (x or "").strip().lower()


# ---------- timeframe canonicalization helpers ----------

_UNITS = {
    "minute", "minutes",
    "hour", "hours",
    "day", "days",
    "week", "weeks",
    "month", "months",
    "year", "years",
}


def _unit_canon(u: str) -> Optional[str]:
    """Return canonical singular unit or None if invalid."""
    u = (u or "").strip().lower()
    if u not in _UNITS:
        return None
    return u[:-1] if u.endswith("s") else u


def _norm_unit(u: str | None) -> str | None:
    if not u:
        return None
    u = str(u).strip().lower()
    if u.endswith("s"):
        u = u[:-1]
    alias = {
        "min": "minute", "mins": "minute",
        "hr": "hour", "h": "hour",
        "day": "day",
        "wk": "week", "w": "week",
        "mo": "month",
        "yr": "year", "y": "year",
    }
    return alias.get(u, u)


_HOURS_PER = {
    "minute": 1.0 / 60.0,
    "hour": 1.0,
    "day": 24.0,
    "week": 7.0 * 24.0,
    "month": 30.0 * 24.0,
    "year": 365.0 * 24.0,
}


def _is_infinite_token(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (int, float)) and math.isinf(float(x)):
        return True
    s = str(x).strip().lower()
    return s in {"inf", "+inf", "-inf", "infinite", "infinity", "none"}


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


def _canon_window(v: Any) -> Optional[Dict[str, Any]]:
    """
    Canonicalize a window endpoint dictionary:

      {
        "temporal_direction": "past" | "now",
        "temporal_magnitude": <real> | "Inf",
        "units": "<temporal_unit_in_plural>",
        "inclusive": true | false
      }
    """
    if not isinstance(v, dict):
        return None

    td = str(v.get("temporal_direction", "")).strip().lower()
    if td not in {"past", "now"}:
        return None

    mag_raw = v.get("temporal_magnitude", None)
    if isinstance(mag_raw, (int, float)):
        mag_val = float(mag_raw)
        mag_canon = mag_val
        mag_is_inf = False
    elif isinstance(mag_raw, str):
        s = mag_raw.strip()
        if s.lower() == "inf":
            mag_val = float("inf")
            mag_canon = "Inf"
            mag_is_inf = True
        else:
            try:
                mag_val = float(s)
                mag_canon = mag_val
                mag_is_inf = False
            except Exception:
                return None
    else:
        return None

    units_raw = str(v.get("units", "")).strip().lower()
    u_singular = _unit_canon(units_raw)
    if not u_singular:
        return None
    u_plural = u_singular + "s"

    inc_raw = v.get("inclusive", None)
    if isinstance(inc_raw, bool):
        inc = inc_raw
    elif isinstance(inc_raw, str):
        inc = inc_raw.strip().lower() == "true"
    else:
        return None

    return {
        "temporal_direction": td,
        "temporal_magnitude": "Inf" if mag_is_inf else mag_canon,
        "units": u_plural,
        "inclusive": inc,
    }


def is_inclusive(bound) -> bool:
    """
    Safely return the 'inclusive' flag from a bound dict.

    Returns True if:
      - bound is None,
      - bound is not a dict,
      - or bound lacks the 'inclusive' key.
    """
    try:
        if isinstance(bound, dict):
            return bool(bound.get("inclusive", True))
        return True
    except Exception:
        return True


def bound_to_hours(bound: dict | None) -> float:
    """
    Convert a confirmable-time bound dict into hours (float).

    Expected keys:
      - temporal_direction: "past" | "now"
      - temporal_magnitude: <float | "Inf" | None>
      - units: "hours" | "days" | "weeks" | "months" | "years"
      - inclusive: bool

    Rules:
      - "past"   → negative hours
      - "now"    → 0.0
      - "Inf" or None → -∞ (past) or 0.0 (now)
      - NEVER return a positive (future) value
    """
    INF_SENTINEL = 1000000000.0

    if not bound:
        return 0.0

    direction = (bound.get("temporal_direction") or "").strip().lower()
    value = bound.get("temporal_magnitude")
    unit = _norm_unit(bound.get("units"))

    if _is_infinite_token(value) or value is None:
        if direction == "past":
            return -INF_SENTINEL
        # "now" or anything else → treat as 0.0
        return 0.0

    v = _to_float_or_none(value)
    if v is None:
        if direction == "past":
            return -INF_SENTINEL
        return 0.0

    if unit in _HOURS_PER:
        hours = v * _HOURS_PER[unit]
    else:
        hours = v

    if direction == "past":
        return -abs(hours)
    # "now" → 0.0 regardless of magnitude
    return 0.0


# ---------- verifier for explicit diagnoses ----------
class PatientStateExplicitDiagnoseVerifier(dspy.Module):
    """
    Validates/normalizes the explicit diagnosis extractor output list.

    Expected LLM output (per item):
      diagnosis: str
      supporting_evidence: List[str]
      rationale: str
      confirmable_latest_start_time: {...}
      confirmable_earliest_end_time: {...}
      timeframe_rationale: str

    Normalized item adds:
      - start_time_in_hours
      - end_time_in_hours
      - start_time_inclusive
      - end_time_inclusive
    """

    def __init__(self, *, min_items: int = 0, max_items: int = 50, dedupe: bool = True):
        super().__init__()
        self.min_items = min_items
        self.max_items = max_items
        self.dedupe = dedupe

    def forward(self, arr: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        issues: List[str] = []
        if not isinstance(arr, list):
            return False, [], ["Output is not a list"]

        if len(arr) > self.max_items:
            issues.append(f"Too many diagnoses: {len(arr)} > {self.max_items}; truncating")
            arr = arr[: self.max_items]

        normed: List[Dict[str, Any]] = []
        seen = set()

        for i, it in enumerate(arr):
            if not isinstance(it, dict):
                issues.append(f"Entry {i} is not an object; skipped")
                continue

            name = str(it.get("diagnosis") or "").strip()
            if not name:
                issues.append(f"Entry {i} missing 'diagnosis'; skipped")
                continue

            key = _norm_diag_name(name)
            if self.dedupe and key in seen:
                issues.append(f"Duplicate diagnosis '{name}' at entry {i}; skipped")
                continue
            seen.add(key)

            start_canon = _canon_window(it.get("confirmable_latest_start_time"))
            end_canon = _canon_window(it.get("confirmable_earliest_end_time"))

            if it.get("confirmable_latest_start_time", None) is not None and start_canon is None:
                issues.append(
                    f"confirmable_latest_start_time not canonical for '{name}': "
                    f"{it.get('confirmable_latest_start_time')!r}"
                )
            if it.get("confirmable_earliest_end_time", None) is not None and end_canon is None:
                issues.append(
                    f"confirmable_earliest_end_time not canonical for '{name}': "
                    f"{it.get('confirmable_earliest_end_time')!r}"
                )

            start_time_in_hours = bound_to_hours(start_canon)
            end_time_in_hours = bound_to_hours(end_canon)

            # validity checks — drop diagnosis if window is future or inverted
            def _is_positive(x):
                try:
                    return x is not None and float(x) > 0.0
                except Exception:
                    return False

            invalid = False
            if _is_positive(start_time_in_hours) or _is_positive(end_time_in_hours):
                which = []
                if _is_positive(start_time_in_hours):
                    which.append(f"start={start_time_in_hours}h")
                if _is_positive(end_time_in_hours):
                    which.append(f"end={end_time_in_hours}h")
                issues.append(
                    f"Dropped '{name}': time bound(s) > 0 (future) — {', '.join(which)}."
                )
                invalid = True

            if not invalid and start_time_in_hours is not None and end_time_in_hours is not None:
                if start_time_in_hours > end_time_in_hours:
                    issues.append(
                        f"Dropped '{name}': start ({start_time_in_hours}h) > end ({end_time_in_hours}h)."
                    )
                    invalid = True

            if invalid:
                continue

            normed.append(
                {
                    "diagnosis": name,
                    "supporting_evidence": list(
                        map(str, it.get("supporting_evidence", []) or [])
                    ),
                    "rationale": str(it.get("rationale", "")).strip(),
                    "confirmable_latest_start_time": start_canon,
                    "confirmable_earliest_end_time": end_canon,
                    "start_time_in_hours": start_time_in_hours,
                    "end_time_in_hours": end_time_in_hours,
                    "start_time_inclusive": is_inclusive(start_canon),
                    "end_time_inclusive": is_inclusive(end_canon),
                    "timeframe_rationale": str(it.get("timeframe_rationale", "")).strip(),
                }
            )

        ok = len(normed) >= self.min_items
        if not ok and not issues:
            issues.append("No valid explicit diagnoses after normalization")
        return ok, normed, issues


# ---------- main explicit-diagnosis extractor ----------
class PatientStateExplicitDiagnoseExtractor(dspy.Module):
    """
    Extract explicit diagnoses from the FULL patient note.

    Expects the prompt template under key:
      'PatientStateExplicitDiagnoseExtractor_prompt'
    with placeholder:
      #PATIENT_NOTE#
    And output in the form:
      <diagnosis_list> [ ... JSON array ... ] </diagnosis_list>

    Normalized results are stored in:
      context["explicit_diagnose"]  (list[dict])
      context["explicit_diagnose_summary"] (counts)
    """

    def __init__(self, engine, *, log_dir: str | Path | None = None):
        super().__init__()
        self.engine = engine
        self.log_dir = Path(log_dir) if log_dir else None
        self.verifier = PatientStateExplicitDiagnoseVerifier()

    def _complete(self, prompt: str) -> str:
        """Single LLM call with deterministic decoding."""
        try:
            return self.engine(prompt, temperature=0.0, top_p=1.0)[0]
        except TypeError:
            # Fallback if engine doesn't accept kwargs
            resp = self.engine(prompt)
            return resp[0] if isinstance(resp, (list, tuple)) else str(resp)

    def forward(self, context: Dict[str, Any], use_full_context: bool = True) -> Dict[str, Any]:
        # Full note
        note_txt = context.get("patient_note") or context.get("requirement_text", "")
        if not isinstance(note_txt, str):
            note_txt = str(note_txt or "")

        tmpl = context.get("PatientStateExplicitDiagnoseExtractor_prompt")
        if not tmpl:
            raise KeyError("Missing 'PatientStateExplicitDiagnoseExtractor_prompt' in context")

        prompt = tmpl.replace("#PATIENT_NOTE#", note_txt)

        # Call LLM once
        raw = self._complete(prompt)
        arr = _parse_json_block(raw)

        ok, normed, issues = self.verifier(arr or [])

        # Store in context
        context["explicit_diagnose"] = normed
        context["explicit_diagnose_summary"] = {
            "n": len(normed),
            "ok": ok,
            "issues": issues,
        }

        # Optional log artifacts
        if self.log_dir:
            try:
                self.log_dir.parent.mkdir(parents=True, exist_ok=True)
                stem = Path(self.log_dir).stem
                out_dir = self.log_dir.parent
                (out_dir / f"{stem}.explicit_diagnose_raw.json").write_text(
                    json.dumps({"prompt": prompt, "raw": raw}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (out_dir / f"{stem}.explicit_diagnose.json").write_text(
                    json.dumps(normed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (out_dir / f"{stem}.explicit_diagnose_issues.json").write_text(
                    json.dumps(issues, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                # best-effort debug logs
                pass

        return context
