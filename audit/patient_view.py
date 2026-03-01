"""Multi-trial view: given a patient, show all trials ranked by match quality."""
from __future__ import annotations
import argparse, html, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit.lib.trial_meta import load_trial_meta

V3 = ROOT / "evaluation" / "results" / "verbalize_judge_235_v3"
REPORTS_DIR = ROOT / "audit" / "sample_reports"


def _collect_patient_trials(patient_id: str) -> list[dict]:
    rows = []
    for shard in V3.glob("shard_*"):
        for pd in shard.iterdir():
            if not pd.is_dir(): continue
            if not pd.name.startswith(patient_id + "__"): continue
            try:
                smt = json.load(open(pd/"smt_decision.json"))
                llmd = json.load(open(pd/"llm_direct_decision.json"))
                tg = json.load(open(pd/"trialgpt_decision.json"))
            except Exception: continue
            pair = pd.name
            _, tid = pair.split("__")
            trial_meta = load_trial_meta(tid) or {}
            inc_raw = (smt.get("inclusion") or {}).get("raw") or {}
            exc_raw = (smt.get("exclusion") or {}).get("raw") or {}

            def count_side(raw):
                er = raw.get("eval_result") or {}
                labels = er.get("label_status") or {}
                s = sum(1 for l in (labels.get("sat") or []) if "_AUXILIARY" not in l)
                u = sum(1 for l in (labels.get("unsat") or []) if "_AUXILIARY" not in l)
                k = sum(1 for l in (labels.get("unknown") or []) if "_AUXILIARY" not in l)
                return s, u, k

            inc_s, inc_u, inc_k = count_side(inc_raw)
            exc_s, exc_u, exc_k = count_side(exc_raw)
            total = inc_s + inc_u + inc_k + exc_s + exc_u + exc_k
            match_score = (inc_s + exc_s) / total if total else 0.0

            rows.append({
                "pair": pair,
                "tid": tid,
                "trial_title": trial_meta.get("brief_title", ""),
                "smt_eligible": smt.get("eligible"),
                "llmd_eligible": (llmd.get("result") or {}).get("eligible"),
                "tg_eligible": (tg.get("aggregate") or {}).get("eligible"),
                "inc": (inc_s, inc_u, inc_k),
                "exc": (exc_s, exc_u, exc_k),
                "match_score": match_score,
            })
    return rows


def render(patient_id: str, rows: list[dict]) -> str:
    def lbl(e):
        if e is True: return ("eligible", "#16a34a")
        if e is False: return ("ineligible", "#dc2626")
        return ("?", "#f59e0b")

    rows_sorted = sorted(rows, key=lambda r: (
        0 if r["smt_eligible"] is True else (1 if r["smt_eligible"] is None else 2),
        -r["match_score"]))
    tbl = []
    for r in rows_sorted:
        s = lbl(r["smt_eligible"]); l = lbl(r["llmd_eligible"]); t = lbl(r["tg_eligible"])
        inc_s, inc_u, inc_k = r["inc"]; exc_s, exc_u, exc_k = r["exc"]
        report_link = f"{r['pair']}.html"
        report_exists = (REPORTS_DIR / report_link).exists()
        link_html = (f'<a href="{report_link}" style="color:#2563eb;text-decoration:none">{html.escape(r["tid"])}</a>'
                     if report_exists else html.escape(r["tid"]))
        trial_overview = f"trial_{r['tid']}.html"
        overview_html = (f' &nbsp;<a href="{trial_overview}" style="color:#6b7280;font-size:0.85em">[all patients]</a>'
                         if (REPORTS_DIR / trial_overview).exists() else '')
        tbl.append(f"""
        <tr>
          <td style="padding:8px 12px;vertical-align:top">{link_html}{overview_html}<br>
            <div style="color:#6b7280;font-size:0.85em">{html.escape(r['trial_title'][:100])}</div>
          </td>
          <td style="padding:8px 12px;color:{s[1]};font-weight:600;white-space:nowrap">{s[0]}</td>
          <td style="padding:8px 12px;color:{l[1]};font-size:0.9em">{l[0]}</td>
          <td style="padding:8px 12px;color:{t[1]};font-size:0.9em">{t[0]}</td>
          <td style="padding:8px 12px;font-size:0.85em">{inc_s}/{inc_s+inc_u+inc_k} ✓</td>
          <td style="padding:8px 12px;font-size:0.85em">{exc_s}/{exc_s+exc_u+exc_k} ✓</td>
          <td style="padding:8px 12px;font-size:0.85em">{r['match_score']:.0%}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Trials for {html.escape(patient_id)}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:1100px;margin:20px auto;padding:0 20px;color:#111827}}
  h1{{border-bottom:2px solid #e5e7eb;padding-bottom:8px}}
  table{{width:100%;border-collapse:collapse;border:1px solid #e5e7eb}}
  th{{text-align:left;padding:8px 12px;background:#f3f4f6}}
  tr:not(:first-child){{border-top:1px solid #e5e7eb}}
</style></head>
<body>
  <h1>Trial matches for patient {html.escape(patient_id)}</h1>
  <p class="meta" style="color:#6b7280">Ranked by AEGIS decision (eligible first), then by match-score (fraction of trial requirements AEGIS classified as satisfied).</p>
  <table>
    <thead><tr>
      <th>Trial</th><th>AEGIS</th><th>LLM-d</th><th>TG</th>
      <th>Inc satisfied</th><th>Exc clear</th><th>Match score</th>
    </tr></thead>
    <tbody>{"".join(tbl)}</tbody>
  </table>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patient", help="Patient ID like sigir-20141")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    args = ap.parse_args()
    rows = _collect_patient_trials(args.patient)
    if not rows:
        print(f"No trials found for {args.patient}")
        return
    html_out = render(args.patient, rows)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"patient_{args.patient}.html"
    out_file.write_text(html_out)
    print(f"Wrote {out_file}  ({len(rows)} trials)")


if __name__ == "__main__":
    main()
