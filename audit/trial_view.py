"""Per-trial view: given a trial, show all patients ranked by match quality.
Useful for trial coordinators seeking candidates."""
from __future__ import annotations
import argparse, html, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit.lib.trial_meta import load_trial_meta

V3 = ROOT / "evaluation" / "results" / "verbalize_judge_235_v3"
REPORTS_DIR = ROOT / "audit" / "sample_reports"


def _collect_trial_patients(trial_id: str) -> list[dict]:
    rows = []
    for shard in V3.glob("shard_*"):
        for pd in shard.iterdir():
            if not pd.is_dir(): continue
            if not pd.name.endswith(f"__{trial_id}"): continue
            try:
                smt = json.load(open(pd/"smt_decision.json"))
                llmd = json.load(open(pd/"llm_direct_decision.json"))
                tg = json.load(open(pd/"trialgpt_decision.json"))
            except Exception: continue
            pid = pd.name.split("__")[0]
            inc_raw = (smt.get("inclusion") or {}).get("raw") or {}
            exc_raw = (smt.get("exclusion") or {}).get("raw") or {}

            def cnt(raw):
                er = raw.get("eval_result") or {}
                lbls = er.get("label_status") or {}
                s = sum(1 for l in (lbls.get("sat") or []) if "_AUXILIARY" not in l)
                u = sum(1 for l in (lbls.get("unsat") or []) if "_AUXILIARY" not in l)
                k = sum(1 for l in (lbls.get("unknown") or []) if "_AUXILIARY" not in l)
                return s, u, k
            inc = cnt(inc_raw); exc = cnt(exc_raw)
            tot = sum(inc) + sum(exc)
            score = (inc[0] + exc[0]) / tot if tot else 0.0

            rows.append({
                "pair": pd.name, "pid": pid,
                "smt_eligible": smt.get("eligible"),
                "llmd_eligible": (llmd.get("result") or {}).get("eligible"),
                "tg_eligible": (tg.get("aggregate") or {}).get("eligible"),
                "inc": inc, "exc": exc, "score": score,
            })
    return rows


def render(trial_id: str, rows: list[dict]) -> str:
    meta = load_trial_meta(trial_id) or {}
    title = meta.get("brief_title", "")
    phase = meta.get("phase", "")
    def lbl(e):
        if e is True: return ("eligible", "#16a34a")
        if e is False: return ("ineligible", "#dc2626")
        return ("?", "#f59e0b")

    rows_sorted = sorted(rows, key=lambda r: (
        0 if r["smt_eligible"] is True else (1 if r["smt_eligible"] is None else 2),
        -r["score"]))

    tbl = []
    for r in rows_sorted:
        s = lbl(r["smt_eligible"]); l = lbl(r["llmd_eligible"]); t = lbl(r["tg_eligible"])
        report_link = f"{r['pair']}.html"
        report_exists = (REPORTS_DIR / report_link).exists()
        pid_link = (f'<a href="{report_link}" style="color:#2563eb;text-decoration:none">{html.escape(r["pid"])}</a>'
                    if report_exists else html.escape(r["pid"]))
        patient_overview = f"patient_{r['pid']}.html"
        overview_link = (f' &nbsp;<a href="{patient_overview}" style="color:#6b7280;font-size:0.85em">[all trials]</a>'
                         if (REPORTS_DIR / patient_overview).exists() else '')
        tbl.append(f"""
        <tr>
          <td style="padding:8px 12px">{pid_link}{overview_link}</td>
          <td style="padding:8px 12px;color:{s[1]};font-weight:600">{s[0]}</td>
          <td style="padding:8px 12px;color:{l[1]};font-size:0.9em">{l[0]}</td>
          <td style="padding:8px 12px;color:{t[1]};font-size:0.9em">{t[0]}</td>
          <td style="padding:8px 12px;font-size:0.85em">{r['inc'][0]}/{sum(r['inc'])} ✓</td>
          <td style="padding:8px 12px;font-size:0.85em">{r['exc'][0]}/{sum(r['exc'])} ✓</td>
          <td style="padding:8px 12px;font-size:0.85em">{r['score']:.0%}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Candidates for {html.escape(trial_id)}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:1100px;margin:20px auto;padding:0 20px;color:#111827}}
  h1{{border-bottom:2px solid #e5e7eb;padding-bottom:8px}}
  table{{width:100%;border-collapse:collapse;border:1px solid #e5e7eb}}
  th{{text-align:left;padding:8px 12px;background:#f3f4f6}}
  tr:not(:first-child){{border-top:1px solid #e5e7eb}}
</style></head>
<body>
  <h1>Candidates for {html.escape(trial_id)}</h1>
  <div style="color:#4b5563;margin-bottom:12px">
    {html.escape(title)}{f' · <span style="color:#6b7280">{html.escape(phase)}</span>' if phase else ''}<br>
    <a href="https://clinicaltrials.gov/study/{html.escape(trial_id)}" style="color:#3730a3;font-size:0.9em">View on clinicaltrials.gov</a>
  </div>

  <p class="meta" style="color:#6b7280">Patients evaluated against this trial; ranked by AEGIS decision, then match-score (fraction of requirements satisfied).</p>

  <table>
    <thead><tr>
      <th>Patient</th><th>AEGIS</th><th>LLM-d</th><th>TG</th>
      <th>Inc satisfied</th><th>Exc clear</th><th>Match score</th>
    </tr></thead>
    <tbody>{"".join(tbl)}</tbody>
  </table>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial", help="Trial NCT ID like NCT00000520")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    args = ap.parse_args()
    rows = _collect_trial_patients(args.trial)
    if not rows:
        print(f"No patients evaluated for {args.trial}"); return
    html_out = render(args.trial, rows)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"trial_{args.trial}.html"
    out_file.write_text(html_out)
    print(f"Wrote {out_file}  ({len(rows)} patients)")


if __name__ == "__main__":
    main()
