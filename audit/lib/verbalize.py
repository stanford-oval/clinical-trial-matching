"""Translate SMT-LIB REQ bodies into clinician-friendly plain English.

The trial compiler already stores:
  - per-variable: meaning / when_to_set_true/false/null
  - per-REQ: a trailing `;; "criterion text"` comment

This module walks a parsed s-expression and produces a single-sentence
rendering suitable for a clinician.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Union

Expr = Union[str, List["Expr"]]


def _extract_meaning(raw: str) -> str:
    """Given a variable's raw description (may contain '; {JSON} ;; "short"'),
    return the clean 'meaning' field or short text, else empty."""
    if not raw: return ""
    s = raw.strip().lstrip(";").strip()
    # Trailing `;; "short text"` after the JSON
    short = ""
    import re as _re
    m = _re.search(r';;\s*"([^"]*)"', raw)
    if m: short = m.group(1)
    # Find the JSON blob
    if s.startswith("{"):
        try:
            end = _re.search(r'\}(\s|;;|$)', s)
            j = s[: (end.start()+1 if end else len(s))]
            parsed = json.loads(j)
            return parsed.get("meaning") or short or ""
        except Exception:
            pass
    return short or s


def _atom_display(atom: str, vindex: Dict[str, dict]) -> str:
    """Render an atom for display: variable name → friendly phrase; literals as-is."""
    if atom in ("true", "True"): return "true"
    if atom in ("false", "False"): return "false"
    try:
        float(atom); return atom
    except ValueError: pass
    info = vindex.get(atom, {})
    meaning = _extract_meaning(info.get("meaning") or info.get("description") or "")
    if meaning: return meaning
    return atom.replace("_", " ").replace("@@", " — ")


def verbalize(expr: Expr, vindex: Dict[str, dict]) -> str:
    """Render an SMT expression as a plain-English sentence."""
    if isinstance(expr, str):
        return _atom_display(expr, vindex)
    if not isinstance(expr, list) or not expr:
        return str(expr)
    op = expr[0] if isinstance(expr[0], str) else None

    if op == "not" and len(expr) == 2:
        sub = expr[1]
        if isinstance(sub, str):
            return f"patient does NOT have: {_atom_display(sub, vindex)}"
        return f"NOT ({verbalize(sub, vindex)})"

    if op == "=" and len(expr) == 3:
        a, b = expr[1], expr[2]
        if isinstance(a, str) and not isinstance(b, list):
            lit = _atom_display(b, vindex)
            if lit == "true": return _atom_display(a, vindex)
            if lit == "false": return f"patient does NOT have: {_atom_display(a, vindex)}"
            return f"{_atom_display(a, vindex)} = {lit}"

    if op in (">=", "<=", ">", "<") and len(expr) == 3:
        a, b = expr[1], expr[2]
        sym = {">=": "≥", "<=": "≤", ">": ">", "<": "<"}[op]
        if isinstance(a, str) and not isinstance(b, list):
            return f"{_atom_display(a, vindex)} {sym} {b}"
        return f"{verbalize(a, vindex)} {sym} {verbalize(b, vindex)}"

    if op == "and":
        parts = [verbalize(s, vindex) for s in expr[1:]]
        return " AND ".join(f"({p})" for p in parts)

    if op == "or":
        parts = [verbalize(s, vindex) for s in expr[1:]]
        return " OR ".join(f"({p})" for p in parts)

    if op == "=>" and len(expr) == 3:
        return f"if [{verbalize(expr[1], vindex)}] then [{verbalize(expr[2], vindex)}]"

    return " ".join(verbalize(s, vindex) if isinstance(s, list) else str(s) for s in expr)


def variable_meaning(var: str, vindex: Dict[str, dict]) -> str:
    """Pull the best human-readable description for a variable."""
    info = vindex.get(var, {})
    for field in ("meaning", "description"):
        v = _extract_meaning(info.get(field) or "")
        if v: return v
    return var.replace("_", " ").replace("@@", " — ")


def value_display(val, sort: str = "") -> str:
    if val is None: return "not documented"
    if isinstance(val, bool): return "yes" if val else "no"
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "yes", "1", "t"): return "yes"
        if s in ("false", "no", "0", "f"): return "no"
        if s in ("null", "none", ""): return "not documented"
        return val
    return str(val)
