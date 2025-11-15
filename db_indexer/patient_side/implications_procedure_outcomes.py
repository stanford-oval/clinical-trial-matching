#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implications_procedure_outcomes.py

Outcome- and timeframe-preserving entailments for procedure variables.

Fixes legacy behavior that produced alias vars like:
  patient_has_undergone_imaging_abnormal
by emitting canonical, fully specified forms like:
  patient_has_undergone_imaging_{TF}_outcome_is_abnormal

Exposes:
  - outcome_hypernym_entailments(source_varname) -> dict|None
  - normalize_legacy_alias(varname, fallback_tf=None) -> str|None
  - is_canonical_outcome_var(varname) -> bool
  - upgrade_or_identity(varname, known_tf=None) -> str   # best-effort upgrade
"""

from __future__ import annotations
import json, os, re
from typing import Dict, List, Optional

# ───────────────────────────────────────────────────────────
# Timeframe + templates
# ───────────────────────────────────────────────────────────

_TF_RX = re.compile(r"(?:now|inthehistory|inthepast\d+days)$", re.I)

CANON_TEMPLATES: Dict[str, str] = {
    "positive": "patient_has_undergone_{entity}_{tf}_outcome_is_positive",
    "negative": "patient_has_undergone_{entity}_{tf}_outcome_is_negative",
    "normal":   "patient_has_undergone_{entity}_{tf}_outcome_is_normal",
    "abnormal": "patient_has_undergone_{entity}_{tf}_outcome_is_abnormal",
}

# ───────────────────────────────────────────────────────────
# Hypernym map (extend/override via JSON file if present)
# ───────────────────────────────────────────────────────────
# Base defaults (safe, conservative). Add more as needed.
_DEFAULT_HYPERNYM_MAP: Dict[str, List[str]] = {
    # Example: ultrasonography of abdomen → a ladder of hypernyms
    "ultrasonography_of_abdomen": [
        "imaging_of_abdomen",
        "imaging_by_body_site",
        "imaging",
        "procedure_by_site",
        "procedure_on_abdomen",
        "procedure_on_trunk",
        "procedure_on_body_region",
        "procedure_by_method",
        "procedure",
        "evaluation_procedure",
        "ultrasonography",
        "ultrasound_studies_by_site",
        "ultrasound_procedure_on_topographic_region",
        # "snomed_ct_concept",  # extremely broad; enable only if you truly want it
    ],
}

def _load_hypernyms_from_file() -> Dict[str, List[str]]:
    """
    Optional override/extension:
      env IMP_PROC_OUTCOME_HYPERNYMS = path/to/hypernyms.json
    File format: { "entity_canonical": ["hyper1", "hyper2", ...], ... }
    """
    path = os.getenv("IMP_PROC_OUTCOME_HYPERNYMS", "").strip()
    if not path:
        return _DEFAULT_HYPERNYM_MAP
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # shallow-merge defaults with file, file takes precedence
        merged = dict(_DEFAULT_HYPERNYM_MAP)
        for k, v in (data or {}).items():
            if isinstance(v, list):
                merged[k.strip().lower()] = [str(x).strip().lower().replace(" ", "_") for x in v]
        return merged
    except Exception:
        return _DEFAULT_HYPERNYM_MAP

_HYPERNYM_MAP = _load_hypernyms_from_file()

# ───────────────────────────────────────────────────────────
# Parsing / composing helpers
# ───────────────────────────────────────────────────────────

def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

# Canonical: patient_has_undergone_{entity}_{tf}_outcome_is_{status}
_VAR_RX = re.compile(
    r"^(?P<prefix>patient_has_undergone)_(?P<entity>[a-z0-9_]+)_(?P<tf>now|inthehistory|inthepast\d+days)_outcome_is_(?P<status>positive|negative|normal|abnormal)$",
    re.I
)

# Legacy alias: patient_has_undergone_{entity}_{status}   (NO timeframe)
_ALIAS_RX = re.compile(
    r"^(?P<prefix>patient_has_undergone)_(?P<entity>[a-z0-9_]+)_(?P<status>positive|negative|normal|abnormal)$",
    re.I
)

def parse_canonical_outcome_var(varname: str) -> Optional[Dict[str, str]]:
    m = _VAR_RX.match((varname or "").strip())
    if not m:
        return None
    g = m.groupdict()
    return {
        "prefix": g["prefix"].lower(),
        "entity": _canon(g["entity"]),
        "tf":     g["tf"].lower(),
        "status": g["status"].lower(),
    }

def parse_legacy_alias_var(varname: str) -> Optional[Dict[str, str]]:
    m = _ALIAS_RX.match((varname or "").strip())
    if not m:
        return None
    g = m.groupdict()
    return {
        "prefix": g["prefix"].lower(),
        "entity": _canon(g["entity"]),
        "status": g["status"].lower(),
    }

def is_canonical_outcome_var(varname: str) -> bool:
    return parse_canonical_outcome_var(varname) is not None

def compose_outcome_var(entity: str, tf: str, status: str) -> str:
    tpl = CANON_TEMPLATES[status]
    return tpl.format(entity=_canon(entity), tf=tf)

def normalize_legacy_alias(varname: str, *, fallback_tf: Optional[str]) -> Optional[str]:
    """
    Upgrade a legacy alias (no timeframe) into canonical outcome var using fallback_tf.
    Returns upgraded var or None if not an alias or TF is unavailable/invalid.
    """
    parsed = parse_legacy_alias_var(varname)
    if not parsed:
        return None
    tf = (fallback_tf or "").strip().lower()
    if not tf or not _TF_RX.match(tf):
        return None
    return compose_outcome_var(parsed["entity"], tf, parsed["status"])

def upgrade_or_identity(varname: str, known_tf: Optional[str] = None) -> str:
    """
    If canonical → return as-is.
    If legacy alias → try upgrading using known_tf or a TF parsed from varname.
    Otherwise → return original.
    """
    if is_canonical_outcome_var(varname):
        return varname
    # try known_tf first
    out = normalize_legacy_alias(varname, fallback_tf=known_tf)
    if out:
        return out
    # try to recover tf directly from the string (rarely helps for aliases)
    m = re.search(r"(now|inthehistory|inthepast\d+days)", varname or "", flags=re.I)
    if m:
        out = normalize_legacy_alias(varname, fallback_tf=m.group(1).lower())
        if out:
            return out
    return varname

# ───────────────────────────────────────────────────────────
# Entailment logic
# ───────────────────────────────────────────────────────────

def outcome_hypernym_entailments(source_varname: str) -> Optional[Dict[str, List[str]]]:
    """
    Given a canonical outcome-bearing source, return:
      {
        "canonical": <normalized source name>,
        "entailed": [hypernym vars preserving timeframe + outcome]
      }
    Return None if source is not canonical outcome-bearing.
    """
    p = parse_canonical_outcome_var(source_varname)
    if not p:
        return None
    entity, tf, status = p["entity"], p["tf"], p["status"]
    hypers = _HYPERNYM_MAP.get(entity, [])
    canonical_self = compose_outcome_var(entity, tf, status)
    entailed = [compose_outcome_var(h, tf, status) for h in hypers]
    return {"canonical": canonical_self, "entailed": entailed}
