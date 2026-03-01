"""Side-by-side rationale comparison across all three systems.

For each system (AEGIS, GPT-4.1 Direct, TrialGPT), extract its natural-language
rationale and render them side by side for clinician comparison.
"""
from __future__ import annotations
import html, json, pathlib


def _aegis_nl_rationale(smt: dict) -> str:
    """Build an AEGIS natural-language rationale from the structured output."""
    eligible = smt.get("eligible")
    inc_raw = (smt.get("inclusion") or {}).get("raw") or {}
    exc_raw = (smt.get("exclusion") or {}).get("raw") or {}
    # Look for a precomputed rationale in the pair dir too
    parts = []
    inc_er = inc_raw.get("eval_result") or {}
    exc_er = exc_raw.get("eval_result") or {}
    if eligible:
        parts.append("<b>Decision:</b> eligible for prescreen. No contra-evidence from chart.")
    elif eligible is False:
        parts.append("<b>Decision:</b> ineligible.")
        # Include unsat REQ counts
        inc_unsat = [l for l in (inc_er.get("label_status") or {}).get("unsat") or [] if "_AUXILIARY" not in l]
        exc_unsat = [l for l in (exc_er.get("label_status") or {}).get("unsat") or [] if "_AUXILIARY" not in l]
        if inc_unsat:
            parts.append(f"<b>Inclusion violations:</b> {len(inc_unsat)} requirement{'s' if len(inc_unsat) != 1 else ''} not satisfied.")
        if exc_unsat:
            parts.append(f"<b>Exclusion violations:</b> {len(exc_unsat)} requirement{'s' if len(exc_unsat) != 1 else ''} not satisfied.")
    else:
        parts.append("<b>Decision:</b> defer — insufficient evidence to decide.")
    return "<br>".join(parts)


def _llmd_rationale(llmd: dict) -> str:
    res = llmd.get("result") or {}
    expl = res.get("explanation") or ""
    eligible = res.get("eligible")
    decision = "eligible" if eligible is True else ("ineligible" if eligible is False else "unknown")
    return f"<b>Decision:</b> {decision}<br><br>{html.escape(expl)}"


def _tg_rationale(tg: dict) -> str:
    agg = tg.get("aggregate") or {}
    eligible = agg.get("eligible")
    inc_detail = agg.get("inclusion_detail") or {}
    exc_detail = agg.get("exclusion_detail") or {}
    decision = "eligible" if eligible is True else ("ineligible" if eligible is False else "unknown")
    parts = [f"<b>Decision:</b> {decision}"]
    # TG output stores per-criterion labels; surface counts
    for side, detail in (("inclusion", inc_detail), ("exclusion", exc_detail)):
        violated = len(detail.get("violated") or [])
        unknown = len(detail.get("unknown") or [])
        if violated or unknown:
            note = []
            if violated: note.append(f"{violated} {side} criteria violated")
            if unknown: note.append(f"{unknown} unknown")
            parts.append("; ".join(note))
    # Top-level blockers: scan rows
    blockers = []
    for side in ("inclusion", "exclusion"):
        side_data = tg.get(side, {}) or {}
        rows = side_data.get("rows", []) or []
        criteria = side_data.get("criteria", []) or []
        for row in rows[:5]:
            label = (row.get("label") or "").lower()
            cid = row.get("criterion_id")
            is_blocker = ((side == "inclusion" and label == "not included") or
                          (side == "exclusion" and label == "excluded"))
            if is_blocker and cid is not None and cid < len(criteria):
                blockers.append(criteria[cid])
    if blockers:
        parts.append("<br><b>Cited blockers:</b><br>" + "<br>".join(f"• {html.escape(b[:180])}" for b in blockers[:5]))
    return "<br>".join(parts)


def render_rationale_comparison(smt: dict, llmd: dict, tg: dict) -> str:
    aegis_html = _aegis_nl_rationale(smt)
    llmd_html = _llmd_rationale(llmd)
    tg_html = _tg_rationale(tg)
    return f"""
    <section style="margin:24px 0">
      <h2 style="margin:0 0 8px 0">Side-by-side rationales</h2>
      <p class="meta">Each system's stated reasoning, shown blind-style so you can compare them directly.</p>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px">
        <div style="border:2px solid #3b82f6;padding:12px;border-radius:6px;background:#eff6ff">
          <div style="font-weight:600;color:#1e40af;margin-bottom:6px">AEGIS (neuro-symbolic)</div>
          <div style="font-size:0.9em;color:#1e3a8a">{aegis_html}</div>
        </div>
        <div style="border:1px solid #e5e7eb;padding:12px;border-radius:6px">
          <div style="font-weight:600;color:#6b7280;margin-bottom:6px">GPT-4.1 Direct</div>
          <div style="font-size:0.9em;color:#374151">{llmd_html}</div>
        </div>
        <div style="border:1px solid #e5e7eb;padding:12px;border-radius:6px">
          <div style="font-weight:600;color:#6b7280;margin-bottom:6px">TrialGPT</div>
          <div style="font-size:0.9em;color:#374151">{tg_html}</div>
        </div>
      </div>
    </section>"""
