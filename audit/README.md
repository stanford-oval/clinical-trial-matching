# AEGIS Clinician Audit Interface

A static HTML report generator that renders AEGIS's structured rationale in a clinician-friendly form.

## What it produces

For each (patient, trial) pair, a self-contained `<pair>.html` file showing:

1. **Decision banner** — eligible / ineligible / defer — with color coding.
2. **Comparison** — AEGIS vs GPT-4.1 Direct vs TrialGPT at a glance.
3. **Patient chart** — the full text.
4. **Per-side requirements table** (inclusion + exclusion):
   - Each requirement in natural language (from the trial compiler's criterion text).
   - Its status: SATISFIED / NOT SATISFIED / INSUFFICIENT EVIDENCE.
   - A collapsible "show compiled rule" for auditors who want the machine-readable form.
5. **Mined facts table** — what the LLM miner extracted from the chart, including supporting chart snippet.

Each row is intended to map to one clinical auditor question:
- "Is this requirement satisfied?"
- "If not, why not?"
- "What does the chart actually say?"

## Usage

Generate one report:
```bash
python audit/generate_report.py sigir-20141__NCT00000520
```

Generate several + build an index:
```bash
for p in sigir-20141__NCT00000520 sigir-20145__NCT01830517 sigir-20142__NCT02272920; do
    python audit/generate_report.py "$p"
done
python audit/build_index.py
```

Open `audit/sample_reports/index.html` in a browser.

## Design choices

- **No JavaScript** — pure HTML/CSS. Works in any browser, shareable via email.
- **Print-friendly** — rendered cleanly on US Letter.
- **Criterion text over SMT** — clinicians see plain English by default. The compiled rule is available on click for formal audit.
- **Status semantics match prescreen policy**:
  - `SATISFIED` = solver returned SAT on this requirement given mined values
  - `NOT SATISFIED` = requirement is in the unsat core
  - `INSUFFICIENT EVIDENCE` = marked unknown (chart silent on required facts)

## Architecture

```
audit/
├── lib/verbalize.py       # SMT → plain English
├── generate_report.py     # single-pair runner
├── build_index.py         # index page builder
├── sample_reports/        # generated .html files
└── README.md
```

`generate_report.py` reads from `evaluation/results/verbalize_judge_235_v3/shard_*/<pair>/smt_decision.json` — no re-running needed.

## Extending

- **Interactive version**: port to a Streamlit app for clinician "what-if" overrides.
- **EHR integration**: point `load_pair` at live EHR data instead of saved JSON.
- **Multi-trial view**: loop over trials for a single patient, rank by number of criteria satisfied.
