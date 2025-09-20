# PatientStateDifferentialDiagnoser.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json, re, random
import dspy
import math

_OPEN = "<diagnosis_list>"
_CLOSE = "</diagnosis_list>"

# ===== Calibration thresholds (tweak if desired) =====
ALMOST_CERTAIN_T = 0.85   # confidence >= this → almost_certain
LIKELY_T         = 0.60   # confidence >= this (and < ALMOST_CERTAIN_T) → likely
# else → possible

# ===== Decoding controls (hard-coded here) =====
DIAGNOSER_TEMPERATURE = 0.20
DIAGNOSER_TOP_P       = 0.40

# Multi-sample controls
N_SAMPLES   = 5            # how many independent samples to draw
BASE_SEED   = 1337         # change if you want a different sequence

# Small jitter to decorrelate samples even if engine's seed is fixed
TEMP_JITTER_STD  = 0.05    # stdev of Gaussian jitter on temperature
TOPP_JITTER_STD  = 0.05    # stdev of Gaussian jitter on top_p
TEMP_MIN, TEMP_MAX = 0.0, 1.0
TOPP_MIN, TOPP_MAX = 1e-3, 1.0

# ---------- extraction helpers ----------
def _extract_block(raw: str) -> str | None:
    m = re.search(re.escape(_OPEN) + r"(.*?)" + re.escape(_CLOSE), raw, flags=re.S | re.I)
    return m.group(1).strip() if m else None

def _parse_json_block(raw: str) -> List[Dict[str, Any]] | None:
    """
    Extract the <diagnosis_list>...</diagnosis_list> block and parse into a list.
    Repairs common JSON-ish issues for the new window fields:

      confirmable_latest_start_time: {
        "temporal_direction": "past|now|future",
        "temporal_magnitude": "<real|'Inf'>",
        "units": "<plural unit>",
        "inclusive": true|false
      }
      confirmable_earliest_end_time: { ...same keys... }

    Returns:
      list[dict] on success, else None.
    """
    body = _extract_block(raw)
    if not body:
        return None

    # 1) Normalize quotes/whitespace
    body = (body
            .replace("\u201c", '"').replace("\u201d", '"')  # curly double
            .replace("\u2018", "'").replace("\u2019", "'")) # curly single
    body = re.sub(r"[ \t]+\n", "\n", body)

    # 2) Field-targeted repairs for new bound dictionaries
    #    temporal_magnitude: allow Inf/inf -> "Inf" (string per your grammar)
    body = re.sub(r'("temporal_magnitude"\s*:\s*)(Inf|inf)\b', r'\1"Inf"', body)

    #    temporal_direction: quote bare enums (past|now|future)
    body = re.sub(
        r'("temporal_direction"\s*:\s*)(past|now|future)\b',
        r'\1"\2"', body, flags=re.I
    )

    #    units: quote bare words (hours, days, weeks, months, years, etc.)
    body = re.sub(
        r'("units"\s*:\s*)([A-Za-z]+)\b',
        r'\1"\2"', body
    )

    #    inclusive: Python booleans -> JSON booleans
    body = re.sub(r'("inclusive"\s*:\s*)True\b',  r'\1true',  body)
    body = re.sub(r'("inclusive"\s*:\s*)False\b', r'\1false', body)

    # 3) Be tolerant to trailing commas
    body = re.sub(r",\s*([}\]])", r"\1", body)

    # 4) Ensure it's an array at top-level; wrap if the model returns a lone object
    stripped = body.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        if stripped.startswith("{") or re.search(r'^\s*\{', stripped, flags=re.M):
            body = "[\n" + body + "\n]"

    # 5) Parse JSON
    try:
        arr = json.loads(body)
        return arr if isinstance(arr, list) else None
    except Exception:
        # Last-resort attempt: try to parse a single object and wrap
        try:
            obj = json.loads(stripped)
            return [obj] if isinstance(obj, dict) else None
        except Exception:
            return None

def _norm_diag_name(x: str) -> str:
    return (x or "").strip().lower()

def _canon_window(v: Any) -> Optional[Dict[str, Any]]:
    """
    Canonicalize a window endpoint dictionary:

      {
        "temporal_direction": "past" | "now" | "future",
        "temporal_magnitude": <real> | "Inf",
        "units": "<temporal_unit_in_plural>",
        "inclusive": true | false
      }

    Returns:
      (canon_dict, norm_dict) or (None, None) if invalid.

    canon_dict:
      - temporal_direction: lowercased ("past"/"now"/"future")
      - temporal_magnitude: number or "Inf" (string)
      - units: canonical plural ("hours", "days", "weeks", "months", "years")
      - inclusive: bool

    norm_dict:
      - orientation: "past" | "now" | "future"
      - value: float('inf') for Inf, else float magnitude
      - unit: canonical plural (same as canon)
      - inclusive: bool
    """
    if not isinstance(v, dict):
        return None

    # --- temporal_direction ---
    td = str(v.get("temporal_direction", "")).strip().lower()
    if td not in {"past", "now", "future"}:
        return None

    # --- temporal_magnitude ---
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

    # --- units ---
    units_raw = str(v.get("units", "")).strip().lower()
    u_singular = _unit_canon(units_raw)
    if not u_singular:
        return None
    # always enforce plural form in both canon and norm
    u_plural = u_singular + "s"

    # --- inclusive ---
    inc_raw = v.get("inclusive", None)
    if isinstance(inc_raw, bool):
        inc = inc_raw
    elif isinstance(inc_raw, str):
        inc = inc_raw.strip().lower() == "true"
    else:
        return None

    # --- build canonical + normalized forms ---
    canon = {
        "temporal_direction": td,
        "temporal_magnitude": "Inf" if mag_is_inf else mag_canon,
        "units": u_plural,
        "inclusive": inc,
    }
    # norm = {
    #     "orientation": td,
    #     "value": mag_val,
    #     "unit": u_plural,  # keep plural for consistency
    #     "inclusive": inc,
    # }
    return canon       #, norm

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
      - temporal_direction: "past" | "now" | "future"
      - temporal_magnitude: <float | "Inf" | None>
      - units: "hours" | "days" | "weeks" | "months" | "years"
      - inclusive: bool

    Rules:
      - "past"   → negative hours
      - "now"    → 0.0
      - "future" → positive hours
      - "Inf" or None → +/-∞ (past/future) or 0.0 (now)
    """
    INF_SENTINEL = 1000000000.0

    # Default: unspecified bound → 0 hours
    if not bound:
        return 0.0

    direction = (bound.get("temporal_direction") or "").strip().lower()
    value = bound.get("temporal_magnitude")
    unit = _norm_unit(bound.get("units"))

    # Infinite or missing magnitude → unbounded in that direction
    if _is_infinite_token(value) or value is None:
        if direction == "past":
            return -INF_SENTINEL
        if direction == "future":
            return INF_SENTINEL
        return 0.0

    # Convert to numeric
    v = _to_float_or_none(value)
    if v is None:
        # Unparseable magnitude → treat as unbounded
        if direction == "past":
            return -INF_SENTINEL
        # if direction == "future":
        #     return INF_SENTINEL
        return 0.0

    # Convert to hours (default 1:1 if unknown)
    if unit in _HOURS_PER:
        hours = v * _HOURS_PER[unit]
    else:
        hours = v

    # Sign by direction
    if direction == "past":
        return -abs(hours)
    if direction == "future":
        return abs(hours)
    return 0.0

# hours per unit (choose the convention you want)
_HOURS_PER = {
    'minute': 1.0/60.0,
    'hour':   1.0,
    'day':    24.0,
    'week':   7.0 * 24.0,
    'month':  30.0 * 24.0,     # simple month (30 days). If needed: 365.2425/12*24 ≈ 730.485
    'year':   365.0 * 24.0,    # simple year (365 days). If needed: 365.2425*24 ≈ 8765.82
}

def _is_infinite_token(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (int, float)) and math.isinf(float(x)):
        return True
    s = str(x).strip().lower()
    return s in {'inf', '+inf', '-inf', 'infinite', 'infinity', "None", "none"}

def _to_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.lower() in {'none', ''}:
        return None
    try:
        return float(s)
    except Exception:
        return None
    
def _norm_unit(u: str | None) -> str | None:
    if not u:
        return None
    u = str(u).strip().lower()
    # strip trailing 's' for plural, but keep 'hours' → 'hour'
    if u.endswith('s'):
        u = u[:-1]
    # basic aliases (optional to extend)
    alias = {
        'min': 'minute', 'mins': 'minute',
        'hr': 'hour', 'h': 'hour',
        'day': 'day',
        'wk': 'week', 'w': 'week',
        'mo': 'month',
        'yr': 'year', 'y': 'year',
    }
    return alias.get(u, u)

# ---------- timeframe & duration canonicalization ----------
_UNITS = {"minute","minutes","hour","hours","day","days","week","weeks","month","months","year","years"}

def _unit_canon(u: str) -> Optional[str]:
    """Return canonical singular unit or None if invalid."""
    u = (u or "").strip().lower()
    if u not in _UNITS:
        return None
    return u[:-1] if u.endswith("s") else u

# returns (canon_str, norm_dict) or ("", None) if invalid/empty
def _canon_timeframe(s: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not s or not isinstance(s, str):
        return "", None
    t = s.strip().lower().replace(" ", "")
    if t == "now":
        return "now", {"kind": "now"}
    if t == "inthehistory":
        return "inthehistory", {"kind": "history"}
    if t == "inthefuture":
        return "inthefuture", {"kind": "future_unspecified"}

    m = re.match(r"^inthepast(\d+)([a-z]+)$", t)
    if m:
        n, u = int(m.group(1)), _unit_canon(m.group(2))
        if u and n > 0:
            return f"inthepast{n}{u}s", {"kind": "past", "n": n, "unit": u}

    m = re.match(r"^inthefuture(\d+)([a-z]+)$", t)
    if m:
        n, u = int(m.group(1)), _unit_canon(m.group(2))
        if u and n > 0:
            return f"inthefuture{n}{u}s", {"kind": "future", "n": n, "unit": u}

    m = re.match(r"^(\d+)([a-z]+)ago$", t)
    if m:
        n, u = int(m.group(1)), _unit_canon(m.group(2))
        if u and n > 0:
            return f"{n}{u}sago", {"kind": "past", "n": n, "unit": u}

    m = re.match(r"^(\d+)([a-z]+)later$", t)
    if m:
        n, u = int(m.group(1)), _unit_canon(m.group(2))
        if u and n > 0:
            return f"{n}{u}slater", {"kind": "future", "n": n, "unit": u}

    return "", None

# duration CAN be null; return (None, None) when absent/invalid
def _canon_duration(s: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if s is None:
        return None, None
    if not isinstance(s, str):
        return None, None
    t = s.strip().lower().replace(" ", "")
    if not t:
        return None, None
    m = re.match(r"^for(\d+)([a-z]+)$", t)
    if m:
        n, u = int(m.group(1)), _unit_canon(m.group(2))
        if u and n > 0:
            return f"for{n}{u}s", {"n": n, "unit": u}
    return None, None

# ---------- verifier ----------
class PatientStateDifferentialDiagnoserVerifier(dspy.Module):
    """
    Validates/normalizes the diagnoser output list and returns (ok, normalized_list, issues[])

    LLM output contract (per item):
      diagnosis: str
      confidence: float in [0,1]
      supporting_evidence: List[str]
      rationale: str
      timeframe: str  (per strict grammar)
      duration: null | str ("for{n}{units}")

    Normalized item adds an internal status bucket:
      status: "almost_certain" | "likely" | "possible"
    And canonical forms for temporal fields:
      timeframe_norm: dict | None
      duration_norm: dict | None
    """
    def __init__(self, *, min_items=1, max_items=25, dedupe=True):
        super().__init__()
        self.min_items, self.max_items, self.dedupe = min_items, max_items, dedupe

    def _status_from_conf(self, conf: float) -> str:
        if conf >= ALMOST_CERTAIN_T:
            return "almost_certain"
        if conf >= LIKELY_T:
            return "likely"
        return "possible"

    def forward(self, arr: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        issues: List[str] = []
        if not isinstance(arr, list):
            return False, [], ["Output is not a list"]
        if len(arr) < self.min_items:
            issues.append(f"Too few diagnoses: {len(arr)} < {self.min_items}")
        if len(arr) > self.max_items:
            issues.append(f"Too many diagnoses: {len(arr)} > {self.max_items}; truncating")
            arr = arr[: self.max_items]

        normed: List[Dict[str, Any]] = []
        seen = set()
        for i, it in enumerate(arr):
            if not isinstance(it, dict):
                issues.append(f"Entry {i} is not an object; skipped")
                continue

            name = str(it.get("diagnosis") or it.get("name") or "").strip()
            if not name:
                issues.append(f"Entry {i} missing 'diagnosis'; skipped")
                continue

            key = _norm_diag_name(name)
            if self.dedupe and key in seen:
                # merge lightweight: keep higher confidence and union evidence
                for j in range(len(normed)):
                    if _norm_diag_name(normed[j]["diagnosis"]) == key:
                        old = normed[j]
                        try:
                            new_conf = float(it.get("confidence", 0.0) or 0.0)
                        except Exception:
                            new_conf = 0.0
                        old["confidence"] = max(float(old.get("confidence", 0.0)), new_conf)
                        old_ev = set(map(str, old.get("supporting_evidence", [])))
                        new_ev = set(map(str, it.get("supporting_evidence", []) or []))
                        old["supporting_evidence"] = list(old_ev | new_ev)
                        break
                continue
            seen.add(key)

            # base fields
            try:
                conf = float(it.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            status = self._status_from_conf(conf)

            # timeframe + duration normalization
            # tf_raw = it.get("timeframe", "")
            # tf_raw = "" if tf_raw is None else str(tf_raw).strip()
            # tf_canon, tf_norm = _canon_timeframe(tf_raw)
            # if tf_raw and not tf_canon:
            #     issues.append(f"Timeframe not canonical for '{name}': {tf_raw!r}")

            # dur_canon, dur_norm = _canon_duration(it.get("duration", None))
            # if it.get("duration", None) is not None and dur_canon is None:
            #     s = str(it.get("duration"))
            #     if s.strip() != "":
            #         issues.append(f"Duration not canonical for '{name}': {s!r}")

            start_canon = _canon_window(it.get("confirmable_latest_start_time", None))
            end_canon = _canon_window(it.get("confirmable_earliest_end_time", None))
            if it.get("confirmable_latest_start_time", None) is not None and start_canon is None:
                issues.append(f"confirmable_latest_start_time not canonical for '{name}': {it.get('confirmable_latest_start_time')!r}")
            if it.get("confirmable_earliest_end_time", None) is not None and end_canon is None:
                issues.append(f"confirmable_earliest_end_time not canonical for '{name}': {it.get('confirmable_earliest_end_time')!r}")
            start_time_in_hours = bound_to_hours(start_canon)
            end_time_in_hours = bound_to_hours(end_canon)
            print("[debug] start time canon",start_canon)
            print("[debug] end time canon",end_canon)

            # ─────────────────────────────────────────────────────────
            # NEW: validity checks — drop diagnosis if future or inverted
            # Conditions:
            #   1) If both numbers exist and start > end → drop
            #   2) If either start or end is > 0 (i.e., in the future) → drop
            def _is_positive(x):
                try:
                    return x is not None and float(x) > 0.0
                except Exception:
                    return False

            invalid = False
            # case 1: start or end is in the future (> 0)
            if _is_positive(start_time_in_hours) or _is_positive(end_time_in_hours):
                which = []
                if _is_positive(start_time_in_hours):
                    which.append(f"start={start_time_in_hours}h")
                if _is_positive(end_time_in_hours):
                    which.append(f"end={end_time_in_hours}h")
                issues.append(f"Dropped '{name}': time bound(s) > 0 (future) — {', '.join(which)}.")
                invalid = True

            # case 2: start is later than end (if both exist)
            if not invalid and start_time_in_hours is not None and end_time_in_hours is not None:
                if start_time_in_hours > end_time_in_hours:
                    issues.append(
                        f"Dropped '{name}': start ({start_time_in_hours}h) > end ({end_time_in_hours}h)."
                    )
                    invalid = True

            if invalid:
                continue
            # ─────────────────────────────────────────────────────────

            normed.append({
                "diagnosis": name,
                "confidence": conf,
                "supporting_evidence": list(map(str, it.get("supporting_evidence", []) or [])),
                "rationale": str(it.get("rationale", "")).strip(),
                "status": status,                 # internal bucket
                # "timeframe": tf_canon,            # "" if missing/invalid
                # "timeframe_norm": tf_norm,        # dict or None
                # "duration": dur_canon,            # None or "for{n}{units}"
                # "duration_norm": dur_norm,        # dict or None
                # "confirmable_latest_start_time": start_canon,
                # "confirmable_earliest_end_time": end_canon,
                "start_time_in_hours": start_time_in_hours,
                "end_time_in_hours": end_time_in_hours,
                "start_time_inclusive": is_inclusive(start_canon),
                "end_time_inclusive": is_inclusive(end_canon),
                "timeframe_rationale": str(it.get("timeframe_rationale", "")).strip(),
            })

        ok = len(normed) >= self.min_items
        if not ok and not issues:
            issues.append("No valid diagnoses after normalization")
        return ok, normed, issues

# ---------- main diagnoser (NOTE-DRIVEN) ----------
class PatientStateDifferentialDiagnoser(dspy.Module):
    """
    Differential diagnosis from the FULL patient note.
    Expects the prompt template under key 'PatientStateDifferentialDiagnoser_prompt'
    with placeholder #PATIENT_NOTE#.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, engine, *, log_dir: str | Path | None = None):
        super().__init__()
        self.engine = engine
        self.log_dir = Path(log_dir) if log_dir else None
        self.verifier = PatientStateDifferentialDiagnoserVerifier()

    # ---- engine call w/ hard-coded decoding params + seed override ----
    def _complete_with_decoding(self, prompt: str, *, seed: int, temp: float, top_p: float) -> str:
        """
        Tries to pass per-call temperature/top_p and a seed.
        If the engine ignores 'seed' (e.g., hard-coded internally), jitter still provides diversity.
        """
        try:
            return self.engine(
                prompt,
                temperature=temp,
                top_p=top_p,
                seed=seed,   # honored only if engine forwards it
            )[0]
        except TypeError:
            # Fallback: set defaults on the engine then call without kwargs
            try:
                if hasattr(self.engine, "kwargs") and isinstance(self.engine.kwargs, dict):
                    self.engine.kwargs["temperature"] = temp
                    self.engine.kwargs["top_p"] = top_p
                return self.engine(prompt)[0]
            except Exception:
                return self.engine(prompt)[0]

    # ---- merge N lists of normalized diagnoses into a single list ----
    @staticmethod
    def _merge_diagnosis_lists(lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        by_key: Dict[str, Dict[str, Any]] = {}

        def _status_rank(s: str) -> int:
            # higher is better
            return {"possible": 0, "likely": 1, "almost_certain": 2}.get(s or "", 0)

        for arr in lists:
            for it in arr:
                key = _norm_diag_name(it.get("diagnosis", ""))
                if not key:
                    continue
                if key not in by_key:
                    # shallow copy; ensure sets for evidence union during merge
                    rec = dict(it)
                    rec["supporting_evidence"] = list(set(map(str, rec.get("supporting_evidence", []) or [])))
                    by_key[key] = rec
                else:
                    dst = by_key[key]
                    # confidence: take max
                    dst["confidence"] = max(float(dst.get("confidence", 0.0)), float(it.get("confidence", 0.0)))
                    # status: take stronger one
                    if _status_rank(it.get("status")) > _status_rank(dst.get("status")):
                        dst["status"] = it.get("status")
                    # evidence: union
                    ev = set(map(str, dst.get("supporting_evidence", []) or [])) | set(map(str, it.get("supporting_evidence", []) or []))
                    dst["supporting_evidence"] = list(ev)
                    # rationale: keep the longer non-empty (heuristic)
                    r0, r1 = (dst.get("rationale", "") or "").strip(), (it.get("rationale", "") or "").strip()
                    if len(r1) > len(r0):
                        dst["rationale"] = r1
                    # # timeframe/duration: prefer specific non-empty if dst empty
                    # if not (dst.get("timeframe") or "") and (it.get("timeframe") or ""):
                    #     dst["timeframe"] = it.get("timeframe")
                    #     dst["timeframe_norm"] = it.get("timeframe_norm")
                    # if dst.get("duration") is None and it.get("duration") is not None:
                    #     dst["duration"] = it.get("duration")
                    #     dst["duration_norm"] = it.get("duration_norm")

        # sort by status then confidence
        merged = list(by_key.values())
        merged.sort(key=lambda x: (
            {"almost_certain": 2, "likely": 1, "possible": 0}.get(x.get("status", ""), 0),
            float(x.get("confidence", 0.0))
        ), reverse=True)
        return merged

    @staticmethod
    def _jittered(val: float, std: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, random.gauss(val, std)))

    def forward(self, context: Dict[str, Any], use_full_context: bool = True) -> Dict[str, Any]:
        # Full note
        note_txt = context.get("patient_note") or context.get("requirement_text", "")
        if not isinstance(note_txt, str):
            note_txt = str(note_txt or "")

        tmpl = context.get("PatientStateDifferentialDiagnoser_prompt")
        if not tmpl:
            raise KeyError("Missing 'PatientStateDifferentialDiagnoser_prompt' in context")

        prompt = tmpl.replace("#PATIENT_NOTE#", note_txt)

        all_attempt_logs: List[Dict[str, Any]] = []
        per_sample_best: List[List[Dict[str, Any]]] = []

        # N independent samples with distinct seeds (if supported) and slight decoding jitter
        for s in range(N_SAMPLES):
            sample_attempts: List[Dict[str, Any]] = []
            sample_best: List[Dict[str, Any]] | None = None
            seed_base = BASE_SEED + s * 9973  # large stride to decorrelate

            # per-sample jitter (helps even if engine ignores 'seed')
            temp_s  = self._jittered(DIAGNOSER_TEMPERATURE, TEMP_JITTER_STD, TEMP_MIN, TEMP_MAX)
            topp_s  = self._jittered(DIAGNOSER_TOP_P,       TOPP_JITTER_STD, TOPP_MIN, TOPP_MAX)

            for k in range(1, self.MAX_ATTEMPTS + 1):
                seed = seed_base + k
                raw = self._complete_with_decoding(prompt, seed=seed, temp=temp_s, top_p=topp_s)
                arr = _parse_json_block(raw)
                ok, normed, issues = self.verifier(arr or [])
                sample_attempts.append({
                    "sample": s + 1,
                    "attempt": k,
                    "ok": ok,
                    "n": len(normed),
                    "issues": issues,
                    "raw": raw
                })
                if ok:
                    sample_best = normed
                    break
                if sample_best is None or len(normed) > len(sample_best):
                    sample_best = normed  # keep most informative partial

            per_sample_best.append(sample_best or [])
            all_attempt_logs.extend(sample_attempts)

        # Union across samples
        diagnoses = self._merge_diagnosis_lists(per_sample_best)

        # Summaries in context
        context["diagnosis_candidates"] = diagnoses
        context["diagnosis_summary"] = {
            "n": len(diagnoses),
            "by_status": {s: sum(1 for d in diagnoses if d.get("status") == s)
                          for s in ["almost_certain", "likely", "possible"]}
        }

        # optional log artifacts
        if self.log_dir:
            self.log_dir.parent.mkdir(parents=True, exist_ok=True)
            stem = Path(self.log_dir).stem
            out_dir = self.log_dir.parent
            (out_dir / f"{stem}.diagnosis_attempts.json").write_text(
                json.dumps(all_attempt_logs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / f"{stem}.diagnosis_candidates.json").write_text(
                json.dumps(diagnoses, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / f"{stem}.diagnosis_samples.json").write_text(
                json.dumps(per_sample_best, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return context
