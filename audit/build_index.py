"""Build an index.html listing all reports under sample_reports/.
Shows each pair's AEGIS decision + 3-system comparison at a glance.
Also links to multi-trial patient views."""
from __future__ import annotations
import html, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = pathlib.Path(__file__).resolve().parent / "sample_reports"
V3 = ROOT / "evaluation" / "results" / "verbalize_judge_235_v3"

def _load_pair_meta(pair):
    for shard in V3.glob("shard_*"):
        pd = shard / pair
        if not pd.exists(): continue
        try:
            smt = json.load(open(pd/"smt_decision.json"))
            llmd = json.load(open(pd/"llm_direct_decision.json"))
            tg = json.load(open(pd/"trialgpt_decision.json"))
            return {
                "smt_eligible": smt.get("eligible"),
                "llmd_eligible": (llmd.get("result") or {}).get("eligible"),
                "tg_eligible": (tg.get("aggregate") or {}).get("eligible"),
            }
        except Exception:
            return None
    return None

def lbl(e):
    if e is True: return ("eligible", "#16a34a")
    if e is False: return ("ineligible", "#dc2626")
    return ("unknown", "#f59e0b")

# Split reports into pair reports vs patient views vs trial views
all_reports = sorted([r for r in REPORTS.glob("*.html") if r.name != "index.html"])
patient_views = [r for r in all_reports if r.name.startswith("patient_")]
trial_views = [r for r in all_reports if r.name.startswith("trial_")]
pair_reports = [r for r in all_reports
                if not r.name.startswith("patient_") and not r.name.startswith("trial_")]

# Pair rows
pair_rows = []
for r in pair_reports:
    pair = r.stem
    meta = _load_pair_meta(pair) or {}
    s = lbl(meta.get("smt_eligible")); l = lbl(meta.get("llmd_eligible")); t = lbl(meta.get("tg_eligible"))
    pair_rows.append(f"""
    <tr>
      <td style="padding:8px 12px"><a href="{html.escape(r.name)}" style="color:#2563eb;text-decoration:none">{html.escape(pair)}</a></td>
      <td style="padding:8px 12px;color:{s[1]};font-weight:600">{s[0]}</td>
      <td style="padding:8px 12px;color:{l[1]}">{l[0]}</td>
      <td style="padding:8px 12px;color:{t[1]}">{t[0]}</td>
    </tr>""")

# Patient-view rows
patient_rows = []
for r in patient_views:
    patient_id = r.stem.replace("patient_", "")
    patient_rows.append(f"""
    <li style="margin:6px 0">
      <a href="{html.escape(r.name)}" style="color:#2563eb;text-decoration:none;font-weight:500">{html.escape(patient_id)}</a>
      <span style="color:#6b7280;font-size:0.9em">— all trials for this patient</span>
    </li>""")

# Trial-view rows
trial_rows = []
for r in trial_views:
    trial_id = r.stem.replace("trial_", "")
    trial_rows.append(f"""
    <li style="margin:6px 0">
      <a href="{html.escape(r.name)}" style="color:#2563eb;text-decoration:none;font-weight:500">{html.escape(trial_id)}</a>
      <span style="color:#6b7280;font-size:0.9em">— all patients for this trial</span>
    </li>""")

index = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>AEGIS audit reports</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:1000px;margin:20px auto;padding:0 20px;color:#111827}}
  h1{{border-bottom:2px solid #e5e7eb;padding-bottom:8px}}
  h2{{margin:24px 0 8px 0}}
  table{{width:100%;border-collapse:collapse;border:1px solid #e5e7eb;margin-top:8px}}
  .tip{{background:#fef3c7;border-left:4px solid #f59e0b;padding:10px 14px;margin:12px 0;border-radius:4px;color:#78350f}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
</style></head>
<body>
<h1>AEGIS Audit Reports</h1>

<p>Clinician-friendly audit reports for AEGIS's trial-matching decisions. Each report shows:</p>
<ul>
  <li>The <b>decision</b> from AEGIS, and what GPT-4.1 Direct and TrialGPT concluded for comparison</li>
  <li>Per-criterion evaluation with plain-English rule text</li>
  <li>The patient chart with highlighted evidence spans</li>
  <li>What chart facts the miner extracted (with chart excerpts)</li>
  <li><b>What would need to change</b> for this patient to potentially flip to eligible (when ineligible)</li>
</ul>

<div class="tip">
  <b>How to read a report:</b> open any pair link below. Use the <em>"Show only violations"</em>
  toggle to focus on what's blocking eligibility. The <em>"What would change this decision?"</em>
  section tells you which chart facts to re-verify during enrollment screening.
</div>

<h2>Per-patient overview</h2>
<p style="color:#6b7280;font-size:0.9em">For clinicians: see which trials a given patient could match.</p>
<ul style="list-style:none;padding-left:0">
{"".join(patient_rows)}
</ul>

<h2>Per-trial overview</h2>
<p style="color:#6b7280;font-size:0.9em">For trial coordinators: see which patients could enroll in a given trial.</p>
<ul style="list-style:none;padding-left:0">
{"".join(trial_rows)}
</ul>

<h2>Per-pair audit reports ({len(pair_rows)})</h2>
<table>
<thead style="background:#f3f4f6"><tr>
  <th style="text-align:left;padding:8px 12px">Pair</th>
  <th style="text-align:left;padding:8px 12px">AEGIS</th>
  <th style="text-align:left;padding:8px 12px">GPT-4.1 Direct</th>
  <th style="text-align:left;padding:8px 12px">TrialGPT</th>
</tr></thead>
<tbody>{"".join(pair_rows)}</tbody>
</table>

<h2>Generate more reports</h2>
<pre style="background:#f3f4f6;padding:10px;border-radius:4px;overflow-x:auto">
# Single pair
python audit/generate_report.py sigir-20141__NCT00000520

# All trials for one patient
python audit/patient_view.py sigir-20141

# All patients for one trial
python audit/trial_view.py NCT00000402

# Update index
python audit/build_index.py
</pre>
</body></html>
"""
(REPORTS / "index.html").write_text(index)
print(f"Wrote index: {len(pair_rows)} pair reports + {len(patient_rows)} patient views + {len(trial_rows)} trial views")
