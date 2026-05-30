# Reproduce From Scratch — Full Pipeline

End-to-end reproduction of the accuracy and counterfactual self-faithfulness
results, starting from raw inputs (SIGIR dataset + prompts). No reliance on
pre-computed experimental artifacts.

If you have already-computed v9 SMT mine + arbiter cache, you can skip
Stages 1–3 and start at Stage 4. The "From Scratch" path runs all stages.

---

## 0. Prerequisites

```bash
# Python 3.11.9 with z3-solver
pyenv install 3.11.9 && pyenv shell 3.11.9
pip install z3-solver requests

# Azure OpenAI deployments for gpt-4.1 and gpt-5
cat > .env <<EOF
OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/deployments/gpt-4.1
OPENAI_ENDPOINT_GPT5=https://<resource>.openai.azure.com/openai/deployments/gpt-5
OPENAI_API_KEY=<key>
EOF

# The cmsrc external repo (Stage 1 SMT atom mining driver)
# Expected at $CMSRC_DIR  (a sibling TrialGPT-SMT/cmsrc checkout)
# Override with: export CMSRC_DIR=<path>
```

Raw inputs (already in repo):
- `dataset/clinical_trial/sigir/{corpus.jsonl, queries.jsonl}` — 3,621 trials + 60 patient charts (sigir-2014x, sigir-2015x)
- `matchers/systems/<system>/prompts/*.prompt` — per-system prompts
- `verbalizer/prompts/*.prompt` — verbalize templates
- `experiments/03_judge_rubric/prompts_5sys/*.prompt` — 5-judge ensemble framings

---

## Stage 1 — SMT atom mining (the heavy step)

Produces `experiments/53_v2_full/cmsrc_out_REMINE_v9_full/<pid>/<full_tid>__full.json`
for every (patient, trial-cohort-variant) pair via the cmsrc external driver.
Each pair × variant requires ~2-4 gpt-4.1 calls. ~600 pairs × ~1.5 variants = ~1000
mines × ~3 calls = ~3000 LLM calls.

```bash
# From the cmsrc repo
cd $CMSRC_DIR
python batch_match_from_eval_union.py \
    --out-root /path/to/TrialGPT-SMT-Refactored/experiments/53_v2_full/cmsrc_out_REMINE_v9_full \
    --prompt-root /path/to/TrialGPT-SMT-Refactored/experiments/53_v2_full/inputs/prompt_root \
    --data-root  /path/to/TrialGPT-SMT-Refactored/dataset/clinical_trial \
    --workers 16
```

Output structure:
```
cmsrc_out_REMINE_v9_full/
  sigir-201410/
    NCT00471289__full.json
    NCT00471289b__full.json   (cohort variant)
    ...
```

Cost: ~$50 in Azure gpt-4.1 calls; runtime ~4-8 hours at 16 parallel workers.

---

## Stage 2 — Arbiter (chart-grounded atom overrides)

The atom miner produces "silent-False" atoms that should sometimes be revised
to True when the chart actually contains evidence. The arbiter pass re-checks
each silent-False atom against the chart with a separate prompt.

```bash
python matchers/systems/aegis/aegis_arbiter.py \
    --mine-root experiments/53_v2_full/cmsrc_out_REMINE_v9_full \
    --out-cache experiments/53_v2_full/arbiter_cache \
    --workers 16
```

Output: `experiments/53_v2_full/arbiter_cache/<pair>__<full_tid>__<side>.json`
(one file per pair × variant × side; each contains override decisions).

---

## Stage 3 — Per-system verdict files

Run each of the 4 matchers. Each produces `verdicts.jsonl` and
`rationales.jsonl`.

```bash
# VERDICT (uses cmsrc mine + arbiter cache + Z3 solve)
python matchers/systems/aegis/run.py

# Single-shot LLM (V5_TWO_STEP)
python matchers/systems/single_shot_llm/run.py --variant V5_TWO_STEP --workers 10

# TrialGPT (per-criterion Jin/Yang 2024)
python matchers/systems/trialgpt/run.py --workers 10

# Shahlab Koopman
python matchers/systems/shahlab/run.py --workers 10
```

Output (per system):
```
matchers/systems/<system>/
  verdicts.jsonl     — {pair, eligibility, ...}
  rationales.jsonl   — {pair, rationale, ...}
```

---

## Stage 4 — Verbalize VERDICT

VERDICT produces SMT atoms + Z3 model; verbalizer converts to evidence-rich NL.

```bash
python verbalizer/run_aegis_v9_arbiter_verbalize.py --workers 16
# → matchers/systems/aegis/rationales_v9_arbiter.jsonl
```

---

## Stage 5 — 5-judge gold + bootstrap CI

5 LLM judges (different prompt framings) vote on each verbalized rationale to
produce gold labels with bootstrap CIs.

```bash
python experiments/accuracy/scripts/run_5judges_verbalized.py --workers 16
# → experiments/accuracy/data/judges5_verbalized.jsonl

python experiments/accuracy/scripts/build_refined_gold_from_5judges.py
# → experiments/accuracy/data/gold_refined_5judges_verbalized.json

python experiments/accuracy/scripts/bootstrap_ci.py
# → experiments/accuracy/bootstrap_ci.json
```

---

## Stage 6 — Meta-fusion

```bash
python experiments/accuracy/scripts/run_meta_fusion.py --mode balanced  --workers 12
python experiments/accuracy/scripts/run_meta_fusion.py --mode precision --workers 12
python experiments/accuracy/scripts/run_meta_fusion.py --mode recall    --workers 12
```

---

## Stage 7 — Accuracy mbench (inspection)

Per-system FP/FN/agreement drilldown for clinician inspection.

```bash
# VERDICT FP/FN balanced
python experiments/accuracy/scripts/build_aegis_fpfn_mbench_balanced.py
# → experiments/accuracy/inspection/aegis_fpfn_mbench_balanced/

# VERDICT-vs-LLM-only disagreement
python experiments/accuracy/scripts/build_aegis_v5_mbench.py
# → experiments/accuracy/inspection/aegis_v5_mbench/

# Arbiter-effect inspection
python experiments/accuracy/scripts/build_arbiter_v2_mbench.py
# → experiments/accuracy/inspection/arbiter_v2_mbench/
```

---

## Stage 8 — Counterfactual self-faithfulness (3-cell ablation)

For each system, generate counterfactual charts that should flip its verdict,
re-run the matcher, measure flip rate.

```bash
cd experiments/counterfactual/05_self_faithfulness/ablation_full_corpus
bash launch_all.sh           # 3 cells × 16 slices, ~50-90 min
python summarize_3cell.py    # → summary_3cell.csv
```

3 cells:
- `cell1_4.1m_4.1v_filt` — gpt-4.1 modifier + gpt-4.1 itemized validator
- `cell2_4.1m_v3v_filt`  — gpt-4.1 modifier + gpt-5 v3 simclin validator
- `cell3_5m_v3v_filt`   — gpt-5 modifier + gpt-5 v3 simclin validator

7 systems each: aegis (random-cohort, joint-MaxSat available behind flag), v5,
v5_cot, v5_cot_gpt5, v5_blockers, tg, shah (binary-forced).

Optional: random-cohort robustness sweep:
```bash
AEGIS_COHORT=random AEGIS_COHORT_SEED=2 bash launch_all.sh
```

See `experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/REPRODUCE.md`
for full env-var reference, expected numbers, and empirical reproduction proof.

---

## Stage 9 — Counterfactual mbench (per-cell × per-system inspection)

```bash
cd experiments/counterfactual/05_self_faithfulness/ablation_full_corpus
python build_mbench_3cell.py
# → experiments/counterfactual/05_self_faithfulness/mbench_3cell/
```

Each pair × system gets a directory tree:
```
mbench_3cell/cell1_.../<system>/{flipped,not_flipped,invalid_cf}/<pair>/
  00_summary.md       — verdict before/after, validator status, cohort flags
  01_first_round/     — original matcher prompt + response (pointer for VERDICT)
  02_blockers/        — targets/rationale handed to the modifier
  03_cf_generation/   — generated CF chart (vs original — see where_to_find_original.md)
  04_validator/       — validator audit response
  05_rejudged/        — final matcher verdict (+ cmsrc full.json for VERDICT)
```

---

## Stage 10 — Clinician validation sample (K=7 audit)

Build a clinician audit instrument from cell-3 CF data.

```bash
python experiments/clinician_validation/build_cf_audit_K7_v2.py
# → experiments/clinician_validation/cf_audit_K7_cell3.json
# → also publishes to clinical-trial-annotation-frontend/private/clinician_review.json

# Inject claude pre-judgments (admin-visible) into rewrites:
python experiments/clinician_validation/inject_simclin_into_topics.py

# (Optional) seed the simclin_demo user with pre-filled answers:
python experiments/clinician_validation/claude_simclin_judgments.py
```

---

## Master driver

`scripts/reproduce_all.sh` runs Stages 1-10 in order. Skip stages by env var
(useful when you have artifacts from a prior stage):
```bash
bash scripts/reproduce_all.sh --from stage4   # skip mine + arbiter + matchers
bash scripts/reproduce_all.sh --through stage7  # stop after accuracy mbench
```

---

## Expected results

### Accuracy (5-system gold, F1 on 539 pairs)

| System | F1 |
|---|---|
| VERDICT (gpt-4.1 mine + Z3) | 0.873 |
| single_shot_llm (V5_TWO_STEP) | 0.904 |
| trialgpt | 0.804 |
| shahlab | 0.836 |
| VERDICT+LLM-only+TrialGPT MAJ | **0.906** |

See `experiments/accuracy/results.md` for full tables.

### Counterfactual self-faithfulness (cell 3, gpt-5 mod + gpt-5 val, no truncation, VERDICT random-cohort seed=2)

| System | flipped/valid | rate |
|---|---|---|
| **aegis** | 161/189 | **85.2%** |
| v5_cot | 274/417 | 65.7% |
| tg | 185/294 | 62.9% |
| v5_cot_gpt5 | 258/447 | 57.7% |
| v5_blockers | 220/379 | 58.0% |
| v5 | 111/225 | 49.3% |
| shah (binary) | 56/203 | 27.6% |

See `experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/REPRODUCE.md`
for cell 1/2 tables and ±2 pp stochasticity envelope.

### Clinician validation (K=7 CF audit, cell 3, claude pre-judgments)

| System | bucket | coherent | flipped_cited |
|---|---|---|---|
| aegis | flipped (n=8) | 62% | 100% |
| aegis | not_flipped (n=7) | **14%** | **43%** |
| LLM baselines | both buckets | 86-100% | 86-100% |

Shows VERDICT's failure mode is **modifier-side** (atom-level edits don't always
compose into clinically coherent prose) — not atom-selection or matcher.

---

## Per-stage timing & cost estimates

| Stage | LLM calls | Wall time (16 workers) | Azure cost (est) |
|---|---|---|---|
| 1 SMT mine | ~3000 | 4-8 hr | ~$50 |
| 2 Arbiter | ~600 | 30 min | ~$10 |
| 3 Per-system | ~2000 | 1 hr | ~$30 |
| 4 Verbalize | ~530 | 15 min | ~$8 |
| 5 5-judge gold | ~2650 | 45 min | ~$40 |
| 6 Meta-fusion | ~1600 | 30 min | ~$25 |
| 7 Accuracy mbench | 0 | 5 min | $0 |
| 8 CF ablation (3 cells × 6 systems × 285 pairs) | ~25,000 | 90 min | ~$200 |
| 9 CF mbench | 0 | 5 min | $0 |
| 10 Clinician audit | ~120 | 5 min | ~$3 |
| **Total from scratch** | **~36,000** | **~8 hr wall** | **~$370** |

---

## Reproducibility notes

- **Z3 MaxSat (VERDICT atom selection)** is bit-exact deterministic given the
  same input program + LLM-asserted atoms.
- **LLM calls** are temperature=0 for gpt-4.1; gpt-5 ignores temperature so
  bit-exact reproduction isn't achievable, but aggregate metrics stable ±2 pp.
- **Random-cohort selection** uses `md5(pair) + seed` (deterministic Python).
- See `experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/REPRODUCE.md`
  for an empirical 13-pair reproduction test (12/12 deterministic-core match,
  10/12 LLM-rejudge match, 7/11 v5-full-pipeline match).
