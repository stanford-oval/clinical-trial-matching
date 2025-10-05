from __future__ import annotations
"""
Shared helpers for validators/namer modules.

Exports:
  - _schema_log(diag, *, idx, block, item_index, code, message, **extra)
  - _extract_declared_symbols(smt_lines)

Notes:
  • Logs auto-remedies/dedups as warnings (⚠), hard errors as ✗.
  • Provides a tiny fallback logger if utils._log is unavailable.
"""
import logging, re, sys
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

__all__ = ["_schema_log", "_extract_declared_symbols"]

# any declare-const symbol
_DECL_ANY_RE = re.compile(r"\(declare-const\s+([^\s\)]+)\s+(?:Bool|Int|Real)\)")

def _extract_declared_symbols(smt_lines: List[str]) -> set[str]:
    names: set[str] = set()
    for ln in smt_lines or []:
        m = _DECL_ANY_RE.search(ln)
        if m:
            names.add(m.group(1))
    return names

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
    """
    Record a structured validation event and mirror it to logs.
    - Auto-remediations / dedups are warnings (⚠)
    - Hard errors remain ✗
    """
    entry = {"block": block, "index": item_index, "code": code, "message": message}
    if extra:
        entry["extra"] = extra
    if diag is not None:
        diag.append(entry)
    try:
        label = f"[{block}{'' if item_index is None else f'[{item_index}]'}] {code}: {message}"
        if extra:
            label += f" | extra={extra}"
        fixed_like = (
            code.endswith("_FIXED")
            or code.startswith("SANITIZE_")
            or code in {"TIMEFRAME_FIELD_CORRECTED", "NAME_NORMALIZED", "REDECLARE_COLLAPSED", "DUP_COLLAPSED"}
        )
        sev = "✗"
        if fixed_like:
            sev = "⚠"
        _log(f"namer {sev} schema", idx or -1, label)
    except Exception:
        pass
