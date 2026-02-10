# Verbalizer — symbolic-rationale → natural-language

Standalone module that turns AEGIS's symbolic decision (verdict + UNSAT cores + atom values + chart + criteria) into a clinician-facing NL rationale.

## Layout

```
verbalizer/
├── prompts/
│   ├── verbalize_smt_rationale.prompt   ← canonical AEGIS verbalize prompt (used by run_aegis_verbalize.py)
│   ├── verbalize_smt_rationale_v2.prompt    — earlier iteration (kept for ablation)
│   ├── verbalize_smt_rationale_new.prompt
│   ├── verbalize_smt_rationale_prescreen.prompt
│   ├── verbalize_neutral.prompt              — neutral framing variant
│   ├── verbalize_prescreen_unified.prompt    — single prompt for SMT + V5 + TG (paraphraser)
│   ├── verbalize_prescreen_unified_v2.prompt … v5
│   ├── verbalize_trialgpt_rationale.prompt   — TG-specific rationale paraphraser
│   └── destyle_verbalization.prompt          — strip stylistic features (length/voice normalizer)
└── run_aegis_verbalize.py                — main runner
```

## Canonical prompt

`prompts/verbalize_smt_rationale.prompt` (also symlinked from `matchers/systems/aegis/prompts/verbalize.prompt`).

Key design choices baked into the prompt:
1. **UNSAT-core direction**: inclusion-side core → patient FAILS inclusion → ineligible; exclusion-side core → exclusion FIRES → ineligible. Both directions guarded against the common LLM misreading "core = requirement not met by trial".
2. **Evidence-quote requirement**: rationale must cite both chart text (verbatim short quote) and criterion text (verbatim short quote). Aim for ≥2 chart quotes + ≥2 criterion quotes per rationale.
3. **Authoritative label**: the system_decision_label is treated as ground truth; rationale must justify it, not contradict it.

## Running

```bash
# Re-verbalize all 539 AEGIS verdicts using canonical prompt
python verbalizer/run_aegis_verbalize.py --workers 16

# Output: overnight/aegis_verbalized.jsonl with {pair, aegis_label, verbalized_rationale, key_points, evidence_quotes}
```

## Tuning

To experiment with a different verbalize style:

```bash
# Edit canonical prompt, then re-run
$EDITOR verbalizer/prompts/verbalize_smt_rationale.prompt
python verbalizer/run_aegis_verbalize.py --workers 16

# Or A/B test against the v2 prompt
python verbalizer/run_aegis_verbalize.py --prompt verbalizer/prompts/verbalize_smt_rationale_v2.prompt
```

## Why standalone

The verbalizer is independent of AEGIS's solve stage — it consumes AEGIS's outputs (verdict + UNSAT cores + atoms) and produces NL. Different downstream uses (judge ensemble, clinician browser, paper figures) all consume the same verbalized output, so we keep this concern separate from the matcher.

## History

- Earlier versions of these prompts lived under `sql_retrieval/meval/prompts/`. They've been extracted here to make the verbalizer a standalone module independent of the retrieval pipeline.
- The polarity bug fix and evidence-quote integration documented in `experiments/accuracy/CHANGELOG.md` were applied to `verbalize_smt_rationale.prompt`.
