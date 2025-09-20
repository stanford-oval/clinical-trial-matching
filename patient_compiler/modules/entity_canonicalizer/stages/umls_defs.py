"""
umls_defs.py  –  minimal utilities for pulling textual definitions from the
UMLS Metathesaurus REST API.

Docs consulted:
  • Source-asserted identifiers           :contentReference[oaicite:0]{index=0}
  • Source-asserted attributes (DEF)      :contentReference[oaicite:1]{index=1}
  • Concept-level /definitions endpoint   :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations
import os, requests
from functools import lru_cache

BASE = "https://uts-ws.nlm.nih.gov/rest"
API_KEY = os.getenv("UMLS_API_KEY")          # <-- put your key in the env

class UMLSClientError(RuntimeError):
    ...

def _get_json(path: str, *, params: dict | None = None) -> dict:
    if not API_KEY:
        raise UMLSClientError("Set UMLS_API_KEY in your environment.")
    p = {"apiKey": API_KEY, **(params or {})}
    r = requests.get(f"{BASE}{path}", params=p, timeout=20)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise UMLSClientError(f"{e}: {r.text[:160]}") from None
    return r.json()["result"]

# ---------------------------------------------------------------------
# 1)  Source-asserted helpers  (code + SAB → data)
# ---------------------------------------------------------------------
@lru_cache(maxsize=4096)
def get_source_concept(code: str, sab: str = "SNOMEDCT") -> dict:
    """Return the full JSON for a source-asserted identifier."""
    return _get_json(f"/content/current/source/{sab}/{code}")

@lru_cache(maxsize=8192)
def get_source_definitions(code: str, sab: str = "SNOMEDCT") -> list[str]:
    """List textual definitions attached directly to the source concept."""
    res = _get_json(f"/content/current/source/{sab}/{code}/attributes",
                    params={"type": "DEF", "pageSize": 100})
    return [row["value"] for row in res]

@lru_cache(maxsize=4096)
def get_cui(code: str, sab: str = "SNOMEDCT") -> str | None:
    """Map a source identifier to its UMLS CUI (or None if absent)."""
    try:
        return get_source_concept(code, sab)["ui"]
    except UMLSClientError:
        return None

# ---------------------------------------------------------------------
# 2)  Concept-level helpers  (CUI → data)
# ---------------------------------------------------------------------
@lru_cache(maxsize=8192)
def get_cui_definitions(cui: str, *, sabs: str | None = None) -> list[str]:
    """List definitions for a CUI (optionally restrict to certain vocabularies)."""
    res = _get_json(f"/content/current/CUI/{cui}/definitions",
                    params={"pageSize": 100, **({"sabs": sabs} if sabs else {})})
    return [row["value"] for row in res]

# ---------------------------------------------------------------------
# 3)  Convenience wrapper (code → single ‘best’ definition)
# ---------------------------------------------------------------------
def get_best_definition(code: str, *, sab: str = "SNOMEDCT") -> str | None:
    """
    Return the first definition found using a two-step fallback:
        1. DEF attribute on the SNOMED concept itself (authoritative, sparse)
        2. Any definition attached to the shared CUI (MeSH, NCIt, MedlinePlus…)
    """
    defs = get_source_definitions(code, sab)
    if defs:
        return defs[0]

    if (cui := get_cui(code, sab)):
        defs = get_cui_definitions(cui)
        if defs:
            return defs[0]

    return None
