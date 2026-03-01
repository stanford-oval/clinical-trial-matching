"""Generate a clinician-friendly HTML audit report for a single (patient, trial) pair.

Inputs: the pair directory under evaluation/results/verbalize_judge_235_v3/shard_*/
Output: a standalone HTML file the clinician can open, print, or share.

Usage:
    python audit/generate_report.py sigir-20141__NCT00000520 [--out audit/sample_reports/]
"""
from __future__ import annotations
import argparse, html, json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))

from lib.verbalize import verbalize, variable_meaning, value_display
sys.path.insert(0, str(ROOT))
from evaluation.explainability.smtlib_parser import extract_named_assertions
from audit.lib.cf_explorer import find_cf_targets, render_cf_section_html
from audit.lib.trial_meta import load_trial_meta
from audit.lib.chart_highlight import highlight_chart
from audit.lib.rationale_compare import render_rationale_comparison
from audit.lib.formula_view import render_formula_view
from audit.lib.interactive import render_interactive_block


REQ_COMMENT_RE = re.compile(r':named\s+(\S+?)\s*\)\)\s*;;\s*"([^"]*)"')


def _parse_req_comments(smt_source: str) -> dict:
    """Return {REQ_label: criterion_text} from `:named X)) ;; "text"` patterns."""
    return {m.group(1): m.group(2) for m in REQ_COMMENT_RE.finditer(smt_source)}


def _status_icon(status: str) -> str:
    return {"sat": "✓", "unsat": "✗", "unknown": "?"}.get((status or "").lower(), "·")


def _status_color(status: str) -> str:
    return {"sat": "#16a34a", "unsat": "#dc2626", "unknown": "#f59e0b"}.get(
        (status or "").lower(), "#6b7280"
    )


def find_pair_dir(pair: str, root: pathlib.Path) -> pathlib.Path:
    for shard in root.glob("shard_*"):
        cand = shard / pair
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Pair {pair} not found under {root}")


def load_pair(pair: str) -> dict:
    root = ROOT / "evaluation" / "results" / "verbalize_judge_235_v3"
    pd = find_pair_dir(pair, root)
    smt = json.load(open(pd / "smt_decision.json"))
    llmd = json.load(open(pd / "llm_direct_decision.json"))
    tg = json.load(open(pd / "trialgpt_decision.json"))
    return {"pair": pair, "path": pd, "smt": smt, "llm_d": llmd, "tg": tg}


def render_criterion_row(req_label: str, body_expr, criterion_text: str,
                         unsat_core: set, sat_labels: set, vindex: dict) -> str:
    if req_label in unsat_core:
        status = "unsat"; status_label = "NOT SATISFIED"
    elif req_label in sat_labels:
        status = "sat"; status_label = "SATISFIED"
    else:
        status = "unknown"; status_label = "INSUFFICIENT EVIDENCE"
    color = _status_color(status); icon = _status_icon(status)
    plain = verbalize(body_expr, vindex) if body_expr else "(no body)"
    is_aux = "_AUXILIARY" in req_label
    aux_tag = ' <span style="color:#6b7280;font-size:0.85em">(auxiliary)</span>' if is_aux else ''
    return f"""
    <tr data-status="{status}">
      <td style="color:{color};font-weight:bold;text-align:center;vertical-align:top;padding:8px">{icon}</td>
      <td style="vertical-align:top;padding:8px">
        <div style="font-weight:600">{html.escape(criterion_text or '(no criterion text)')}{aux_tag}</div>
        <details style="color:#6b7280;font-size:0.85em;margin-top:4px">
          <summary style="cursor:pointer;color:#9ca3af">show compiled rule</summary>
          <div style="margin-top:6px;padding:6px 8px;background:#f9fafb;border-radius:4px;font-family:monospace">{html.escape(plain)}</div>
        </details>
      </td>
      <td style="color:{color};font-weight:600;vertical-align:top;padding:8px;white-space:nowrap">{status_label}</td>
    </tr>
    """


def _evidence_confidence(evidence) -> tuple:
    """Classify evidence strength based on presence/length. Returns (label, color)."""
    if not evidence: return ("inferred / no chart excerpt", "#dc2626")
    if isinstance(evidence, list):
        evidence = " ".join(str(e) for e in evidence)
    s = str(evidence).strip()
    if not s: return ("inferred / no chart excerpt", "#dc2626")
    if len(s) < 25: return ("weak evidence", "#f59e0b")
    return ("explicit", "#16a34a")


def render_mined_facts_table(pvv: dict, vindex: dict) -> str:
    # Separate "with value" vs "null" rows for clarity
    yes_no_rows, null_rows = [], []
    for var, vinfo in sorted(pvv.items()):
        val = vinfo.get("value") if isinstance(vinfo, dict) else vinfo
        meaning = variable_meaning(var, vindex)
        vshow = value_display(val)
        evidence = (vinfo.get("evidence") or "") if isinstance(vinfo, dict) else ""
        conf_label, conf_color = _evidence_confidence(evidence)
        color = "#16a34a" if vshow == "yes" else "#dc2626" if vshow == "no" else "#6b7280"
        row_html = f"""
        <tr>
          <td style="padding:6px 8px;vertical-align:top">{html.escape(meaning[:200])}</td>
          <td style="padding:6px 8px;vertical-align:top;color:{color};font-weight:600;white-space:nowrap">{html.escape(vshow)}</td>
          <td style="padding:6px 8px;vertical-align:top;color:{conf_color};font-size:0.85em;white-space:nowrap">{conf_label}</td>
          <td style="padding:6px 8px;vertical-align:top;color:#6b7280;font-size:0.9em">{html.escape(str(evidence)[:300])}</td>
        </tr>
        """
        if vshow == "not documented":
            null_rows.append(row_html)
        else:
            yes_no_rows.append(row_html)

    if not yes_no_rows and not null_rows:
        return '<p style="color:#6b7280;font-style:italic">No mined facts for this side.</p>'

    header = (
        '<thead><tr style="background:#f3f4f6">'
        '<th style="text-align:left;padding:8px">Fact</th>'
        '<th style="text-align:left;padding:8px">Value</th>'
        '<th style="text-align:left;padding:8px">Evidence strength</th>'
        '<th style="text-align:left;padding:8px">Chart excerpt</th>'
        '</tr></thead>'
    )
    out = '<table style="width:100%;border-collapse:collapse;border:1px solid #e5e7eb">' + header + '<tbody>' + "".join(yes_no_rows) + '</tbody></table>'
    if null_rows:
        out += f"""
        <details style="margin-top:8px">
          <summary style="cursor:pointer;color:#6b7280;font-size:0.9em">{len(null_rows)} additional variables not documented in chart</summary>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e5e7eb;margin-top:8px">
            {header}<tbody>{"".join(null_rows)}</tbody>
          </table>
        </details>"""
    return out


def render_summary_stats(inc_raw: dict, exc_raw: dict) -> str:
    """Show at-a-glance counts: X satisfied / Y violated / Z insufficient, per side."""
    counts = {"inclusion": {"sat": 0, "unsat": 0, "unknown": 0},
              "exclusion": {"sat": 0, "unsat": 0, "unknown": 0}}
    for side, raw in [("inclusion", inc_raw), ("exclusion", exc_raw)]:
        er = raw.get("eval_result") or {}
        labels = er.get("label_status") or {}
        # Exclude AUXILIARY labels from user-facing counts
        for kind in ("sat", "unsat", "unknown"):
            for lbl in (labels.get(kind) or []):
                if "_AUXILIARY" not in lbl:
                    counts[side][kind] += 1
    def bar(side, c):
        total = c["sat"] + c["unsat"] + c["unknown"]
        if total == 0: return f"<div style='color:#6b7280'>No {side} requirements</div>"
        pct = lambda n: (n / total) * 100 if total else 0
        return f"""
        <div style="margin:4px 0">
          <span style="font-weight:600">{side.title()}</span>
          <span style="color:#6b7280;margin-left:8px">{total} requirement{'s' if total != 1 else ''}</span>
          <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin-top:4px">
            <div style="background:#16a34a;width:{pct(c['sat'])}%" title="{c['sat']} satisfied"></div>
            <div style="background:#dc2626;width:{pct(c['unsat'])}%" title="{c['unsat']} violated"></div>
            <div style="background:#f59e0b;width:{pct(c['unknown'])}%" title="{c['unknown']} insufficient evidence"></div>
          </div>
          <div style="font-size:0.9em;margin-top:2px">
            <span style="color:#16a34a">✓ {c['sat']} satisfied</span> &nbsp;
            <span style="color:#dc2626">✗ {c['unsat']} violated</span> &nbsp;
            <span style="color:#f59e0b">? {c['unknown']} insufficient</span>
          </div>
        </div>"""
    return f"""
    <section style="margin:12px 0;padding:12px 16px;background:#fafafa;border:1px solid #e5e7eb;border-radius:6px">
      <div style="font-weight:600;margin-bottom:8px">Requirements summary</div>
      {bar('inclusion', counts['inclusion'])}
      {bar('exclusion', counts['exclusion'])}
    </section>"""


def render_side(side_name: str, side_raw: dict) -> str:
    er = side_raw.get("eval_result") or {}
    status = er.get("status", "unknown")
    labels = er.get("label_status") or {}
    unsat_labels = set(labels.get("unsat") or [])
    sat_labels = set(labels.get("sat") or [])
    smt_source = "\n".join(side_raw.get("smt_program_lines") or [])
    named = extract_named_assertions(smt_source)
    req_comments = _parse_req_comments(smt_source)
    vindex = side_raw.get("variable_index") or {}
    pvv = side_raw.get("patient_var_values") or {}

    color = _status_color(status); icon = _status_icon(status)
    status_explain = {
        "sat": "Patient satisfies all requirements on this side.",
        "unsat": "Patient fails at least one requirement on this side (shown in red below).",
        "unknown": "Solver could not decide; some requirements need more chart evidence.",
    }.get(status, status)

    # Split rows into violated (unsat) vs satisfied/unknown
    req_rows = []
    for lbl in named:
        if "_AUXILIARY" in lbl: continue  # hide auxiliaries by default
        criterion_text = req_comments.get(lbl, "")
        req_rows.append((lbl, criterion_text, lbl in unsat_labels))
    # Sort: violated first
    req_rows.sort(key=lambda x: (not x[2], x[0]))

    rows_html = "".join(
        render_criterion_row(lbl, named.get(lbl), crit, unsat_labels, sat_labels, vindex)
        for lbl, crit, _ in req_rows
    )

    return f"""
    <section style="margin:24px 0">
      <h2 style="color:{color};margin:0 0 4px 0">
        {icon} {side_name.title()}: {status.upper()}
      </h2>
      <p style="color:#4b5563;margin:0 0 12px 0">{status_explain}</p>

      <h3 style="margin:16px 0 8px 0">Trial requirements on this side</h3>
      <table style="width:100%;border-collapse:collapse;border:1px solid #e5e7eb">
        <thead><tr style="background:#f3f4f6">
          <th style="width:40px;padding:8px"></th>
          <th style="text-align:left;padding:8px">Requirement</th>
          <th style="text-align:left;padding:8px;width:160px">Status</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>

      <h3 style="margin:16px 0 8px 0">What the chart says (mined facts)</h3>
      {render_mined_facts_table(pvv, vindex)}
    </section>
    """


def render_report(pair_data: dict) -> str:
    pair = pair_data["pair"]
    smt = pair_data["smt"]
    pid, tid = pair.split("__")
    inc_raw = (smt.get("inclusion") or {}).get("raw") or {}
    exc_raw = (smt.get("exclusion") or {}).get("raw") or {}
    eligible = smt.get("eligible")

    chart = inc_raw.get("patient_notes") or exc_raw.get("patient_notes") or ""
    if isinstance(chart, list): chart = "\n".join(chart)

    decision_label = "ELIGIBLE" if eligible is True else ("INELIGIBLE" if eligible is False else "DEFER")
    decision_color = "#16a34a" if eligible is True else ("#dc2626" if eligible is False else "#f59e0b")

    # Other system decisions for comparison
    llmd_e = (pair_data["llm_d"].get("result") or {}).get("eligible")
    tg_e = (pair_data["tg"].get("aggregate") or {}).get("eligible")
    def lbl(e): return "Eligible" if e is True else ("Ineligible" if e is False else "Defer/Unknown")

    trial_meta = load_trial_meta(tid)
    trial_title = trial_meta.get("brief_title") or trial_meta.get("official_title") or ""
    trial_summary = trial_meta.get("brief_summary", "")
    trial_phase = trial_meta.get("phase", "")
    trial_study_type = trial_meta.get("study_type", "")
    trial_inclusion = trial_meta.get("inclusion_criteria", "")
    trial_exclusion = trial_meta.get("exclusion_criteria", "")

    trial_header = ""
    if trial_title:
        meta_tags = []
        if trial_phase: meta_tags.append(html.escape(trial_phase))
        if trial_study_type: meta_tags.append(html.escape(trial_study_type))
        meta_line = " · ".join(meta_tags) if meta_tags else ""
        official_criteria = ""
        if trial_summary or trial_inclusion or trial_exclusion:
            parts = []
            if trial_summary:
                parts.append(f'<div style="margin-bottom:8px"><b style="color:#374151">Summary.</b> {html.escape(trial_summary[:1500])}</div>')
            if trial_inclusion:
                parts.append(f'<div style="margin-bottom:8px"><b style="color:#374151">Inclusion criteria.</b><pre style="margin:4px 0;padding:8px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;white-space:pre-wrap;font-size:0.85em">{html.escape(trial_inclusion[:3000])}</pre></div>')
            if trial_exclusion:
                parts.append(f'<div><b style="color:#374151">Exclusion criteria.</b><pre style="margin:4px 0;padding:8px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;white-space:pre-wrap;font-size:0.85em">{html.escape(trial_exclusion[:3000])}</pre></div>')
            official_criteria = (
                f'<details style="margin-top:8px"><summary style="cursor:pointer;'
                f'color:#6b7280;font-size:0.9em">Show trial summary & official criteria</summary>'
                f'<div style="padding:10px 0;color:#374151;font-size:0.9em">{"".join(parts)}</div>'
                f'</details>'
            )
        trial_header = f"""
  <div style="padding:6px 0;margin:8px 0 14px 0;border-bottom:1px solid #e5e7eb">
    <div style="font-size:1.05em;font-weight:600;color:#111827">{html.escape(trial_title)}</div>
    <div style="color:#6b7280;font-size:0.85em;margin-top:2px">
      <a href="https://clinicaltrials.gov/study/{html.escape(tid)}" style="color:#3730a3">{html.escape(tid)}</a>
      {' · ' + meta_line if meta_line else ''}
    </div>
    {official_criteria}
  </div>"""

    html_out = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>AEGIS Audit — {html.escape(pair)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
             max-width: 900px; margin: 20px auto; padding: 0 20px; color: #111827; line-height: 1.5; }}
    h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-bottom: 4px; }}
    table {{ font-size: 0.95em; }}
    .decision-banner {{ padding: 16px; border-radius: 8px; color: white;
                        font-weight: 700; font-size: 1.4em; margin: 16px 0; }}
    .meta {{ color: #6b7280; font-size: 0.9em; }}
    .chart {{ background:#f9fafb; border-left: 4px solid #9ca3af;
              padding: 12px 16px; margin: 12px 0; font-family: Georgia, serif; }}
    .other-systems {{ display: flex; gap: 16px; margin: 12px 0; }}
    .other-systems .card {{ flex: 1; border: 1px solid #e5e7eb; padding: 12px; border-radius: 6px; }}
    @media print {{
      body {{ max-width: none; margin: 0; padding: 0; font-size: 10pt; }}
      .decision-banner {{ page-break-after: avoid; }}
      section {{ page-break-inside: avoid; }}
      .filter-controls {{ display: none; }}
      details {{ display: block; }}
      details[open] summary {{ display: none; }}
      summary {{ display: none; }}
      details > div:not([style*="display"]) {{ display: block; }}
    }}
    .filter-controls {{ background:#f3f4f6;padding:10px 14px;border-radius:6px;margin:16px 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap }}
    .filter-controls label {{ cursor:pointer;user-select:none;font-size:0.9em }}
    .filter-controls input[type=text] {{ padding:4px 8px;border:1px solid #d1d5db;border-radius:4px;font-size:0.9em;width:200px }}
    tr[data-status].hidden-by-filter {{ display:none }}
    tr[data-status].hidden-by-search {{ display:none }}
    @media (max-width: 700px) {{
      body {{ padding: 0 12px; font-size: 0.95em; }}
      .other-systems {{ flex-direction: column; }}
      table td {{ padding: 6px 4px !important; font-size: 0.9em; }}
      [style*="grid-template-columns:1fr 1fr 1fr"] {{ display:block !important; }}
      [style*="grid-template-columns:1fr 1fr 1fr"] > div {{ margin-bottom: 8px !important; }}
    }}
  </style>
  <script>
    function applyFilter() {{
      const showOnlyViolations = document.getElementById('only-violations').checked;
      const search = (document.getElementById('search-box')?.value || '').toLowerCase();
      document.querySelectorAll('tr[data-status]').forEach(row => {{
        // Status filter
        if (showOnlyViolations && row.dataset.status !== 'unsat') {{
          row.classList.add('hidden-by-filter');
        }} else {{
          row.classList.remove('hidden-by-filter');
        }}
        // Search filter
        if (search && !row.textContent.toLowerCase().includes(search)) {{
          row.classList.add('hidden-by-search');
        }} else {{
          row.classList.remove('hidden-by-search');
        }}
      }});
    }}
    function expandAll(open) {{
      document.querySelectorAll('details').forEach(d => d.open = open);
    }}
    // Keyboard shortcuts
    document.addEventListener('keydown', e => {{
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'v') {{
        const cb = document.getElementById('only-violations');
        cb.checked = !cb.checked; applyFilter();
      }} else if (e.key === 'e') {{
        expandAll(true);
      }} else if (e.key === 'c') {{
        expandAll(false);
      }} else if (e.key === 'p') {{
        window.print();
      }} else if (e.key === '/') {{
        e.preventDefault();
        document.getElementById('search-box')?.focus();
      }}
    }});
  </script>
</head>
<body>
  <h1>AEGIS Audit Report</h1>
  <div class="meta">Patient: {html.escape(pid)} &nbsp;•&nbsp; Trial: {html.escape(tid)}</div>

  {trial_header}

  {render_interactive_block(smt)}

  <div style="color:#6b7280;font-size:0.85em;margin:-6px 0 14px 0">
    Peer systems — GPT-4.1 Direct: <b style="color:#374151">{lbl(llmd_e)}</b>
    · TrialGPT: <b style="color:#374151">{lbl(tg_e)}</b>
    {(" · <span style='color:#16a34a'>all three agree</span>"
      if (eligible == llmd_e == tg_e)
      else " · <span style='color:#f59e0b'>⚠ systems disagree</span>")}
  </div>

  {render_formula_view(smt)}

  <details style="margin:20px 0">
    <summary style="cursor:pointer;font-weight:600;color:#374151;padding:8px 0;
                    border-top:1px solid #e5e7eb">
      Patient chart <span style="color:#9ca3af;font-weight:400">— highlighted spans are evidence for each fact</span>
    </summary>
    <div style="padding-top:8px">
      <div class="chart">{highlight_chart(chart, {**(inc_raw.get("patient_var_values") or {}), **(exc_raw.get("patient_var_values") or {})})}</div>
      <div class="meta" style="margin:6px 0 16px 0">
        <span style="background:#bbf7d0;color:#065f46;padding:1px 6px;border-radius:3px">yes</span>
        <span style="background:#fecaca;color:#7f1d1d;padding:1px 6px;border-radius:3px;margin-left:4px">no</span>
        <span style="background:#bfdbfe;color:#1e3a8a;padding:1px 6px;border-radius:3px;margin-left:4px">numeric/other</span>
      </div>
    </div>
  </details>

  <details style="margin:20px 0">
    <summary style="cursor:pointer;font-weight:600;color:#374151;padding:8px 0;
                    border-top:1px solid #e5e7eb">
      Compare rationales from all three systems
    </summary>
    <div style="padding-top:8px">{render_rationale_comparison(smt, pair_data["llm_d"], pair_data["tg"])}</div>
  </details>

  <details style="margin:20px 0">
    <summary style="cursor:pointer;font-weight:600;color:#374151;padding:8px 0;
                    border-top:1px solid #e5e7eb">
      Developer view — verbose per-criterion list + CF targets
    </summary>
    <div style="padding-top:8px">
      {render_cf_section_html(find_cf_targets(smt))}
      {render_side("inclusion", inc_raw)}
      {render_side("exclusion", exc_raw)}
    </div>
  </details>

  <hr style="margin:32px 0;border:none;border-top:1px solid #e5e7eb">
  <p class="meta">
    This report was generated from AEGIS's structured rationale.
    Every requirement status is derived from the SMT solver's evaluation of the compiled trial
    program against patient facts extracted by the LLM miner. Review each "NOT SATISFIED" row
    and its associated mined facts to understand why the trial was matched this way.
  </p>
</body>
</html>
"""
    return html_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", help="Pair ID like sigir-20141__NCT00000520")
    ap.add_argument("--out", default=str(ROOT/"audit"/"sample_reports"))
    args = ap.parse_args()
    data = load_pair(args.pair)
    html_out = render_report(data)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.pair}.html"
    out_file.write_text(html_out)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
