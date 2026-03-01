"""Highlight chart sentences that provided evidence for mined variables.

For each mined variable with evidence (a chart excerpt), locate the excerpt
within the chart and add an inline span tag with color coding. Colors:
  green: mined as true / numeric value present
  red:   mined as false
  gray:  null / insufficient
"""
from __future__ import annotations
import html, re
from typing import Dict, List, Tuple


def _classify_value(val) -> str:
    """green / red / gray."""
    if val is True: return "true"
    if val is False: return "false"
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "t", "yes"): return "true"
        if s in ("false", "0", "f", "no"): return "false"
        if s in ("", "null", "none"): return "null"
    if val is None: return "null"
    return "numeric"


_COLOR = {
    "true":    ("#bbf7d0", "#065f46"),  # light green bg, dark green text
    "false":   ("#fecaca", "#7f1d1d"),  # light red
    "numeric": ("#bfdbfe", "#1e3a8a"),  # light blue
    "null":    ("#e5e7eb", "#374151"),  # gray
}


def highlight_chart(chart: str, pvv: Dict[str, dict]) -> str:
    """Return HTML with matched chart sentences highlighted.

    Each mined variable with evidence gets a highlight. Overlapping matches
    are merged (last-write-wins). The output is html-escaped with `<mark>`
    spans inserted.
    """
    if not chart:
        return ""
    # Collect (start, end, color_class, tooltip) by searching for each evidence string
    matches: List[Tuple[int, int, str, str]] = []
    for var, info in pvv.items():
        if not isinstance(info, dict): continue
        evidence = info.get("evidence") or ""
        val = info.get("value")
        cls = _classify_value(val)
        if cls == "null": continue  # don't highlight null
        # Evidence can be a list or string; normalize
        ev_strs = evidence if isinstance(evidence, list) else [evidence]
        for ev in ev_strs:
            if not isinstance(ev, str) or len(ev.strip()) < 8: continue
            # Find the evidence in the chart (case-insensitive, forgiving)
            # Try exact first; fall back to first 50 chars
            needle = ev.strip()[:100]
            idx = chart.lower().find(needle.lower())
            if idx < 0 and len(needle) > 30:
                idx = chart.lower().find(needle.lower()[:30])
            if idx < 0: continue
            end = idx + min(len(needle), len(chart) - idx)
            tooltip = f"{var.replace('_', ' ')}: {val}"
            matches.append((idx, end, cls, tooltip))
    # Sort + merge overlapping regions (keep the first for color)
    matches.sort()
    merged: List[Tuple[int, int, str, str]] = []
    for m in matches:
        if merged and m[0] < merged[-1][1]:
            # overlap; extend end
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], m[1]), prev[2], prev[3])
        else:
            merged.append(m)
    # Build HTML
    out = []
    cursor = 0
    for start, end, cls, tip in merged:
        if cursor < start:
            out.append(html.escape(chart[cursor:start]))
        bg, fg = _COLOR.get(cls, _COLOR["null"])
        out.append(
            f'<mark style="background:{bg};color:{fg};padding:1px 2px;border-radius:2px" '
            f'title="{html.escape(tip)}">{html.escape(chart[start:end])}</mark>'
        )
        cursor = end
    if cursor < len(chart):
        out.append(html.escape(chart[cursor:]))
    return "".join(out)
