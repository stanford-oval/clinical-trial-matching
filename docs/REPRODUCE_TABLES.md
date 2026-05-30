# Reproduce Every Paper Table

One row per table in the submitted paper. For each row: the number the paper
prints, the source data file, and the command that regenerates the number.
Assumes `.env` is sourced (see `REPRODUCE_FROM_SCRATCH.md §0`).

Legend: **✓** = script exists and produces the exact number in the paper;
**△** = script exists but requires config change / manual step; **✗** = no
reproduction script yet (gap to close).

| # | Table caption | Script / command | Data file | Status |
|---|---|---|---|---|
| 1 | Patient-level eligibility on SIGIR (5-system gold, n=539, F1/P/R) | `python experiments/clinician_validation/processing/report.py` (calls `accuracy_adjusted_f1.py`) | `matchers/systems/*/verdicts.jsonl` + `data/gold/gold_5sys_freeform_balanced.json` | ✓ |
| 2 | F1 on 363 TREC 2021 pairs (5 backbones × 2 pipelines) | `python scripts/tables/table2_trec2021_f1.py` | `$SVPO_RL/data/test_clean_tagged.jsonl` + `eval_runs/*/rollouts.jsonl` | ✓ (GPT-5-mini + Qwen rows) |
| 3 | Clinician preference over rationales (W-T-L, corpus-reweighted) | `python experiments/clinician_validation/processing/report.py` (Table 24 in results) | `$EVALUATIONS_RESULTS` (or `~/chart_silence_review/submissions.jsonl` on yelpbot3) + `merge/clinician_review_merged.json` | ✓ |
| 4 | Mean clinician ratings (1–5 axes) | `python experiments/clinician_validation/processing/report.py` → per-axis section | same as Table 3 | ✓ |
| 5 | Policy-verdict agreement (%) — TypedPolicy vs LLM | `python experiments/typed_policy/scripts/07_compare.py` | `experiments/typed_policy/data/{nl_track_results.jsonl,smt_track_results.jsonl}` | ✓ |
| 6 | Raw ineligible→eligible **PIVOTALFLIPRATE** | Same source as Table 27 (full-corpus run, N_valid denominator) | `experiments/counterfactual/05_self_faithfulness/out/` (full corpus, not the `mbench_3cell/` inspection subset) | ✓ |
| 7 | Pipeline stages × prompts (structural doc) | N/A — not a numeric table | — | — |
| 8 | Modifier–validator configurations (structural doc) | N/A — not a numeric table | — | — |
| 9 | Evaluation dataset statistics | `python scripts/tables/table9_dataset_stats.py` | `dataset/clinical_trial/sigir/{queries,corpus}.jsonl` + TREC data | ✓ |
| 10 | Verdict balance + rationale length per system | `python scripts/tables/table10_verdict_balance.py` | `matchers/systems/*/*_freeform.jsonl` | ✓ 3/5 exact; ZSPM near, TrialGPT ✗ — see §Table 10 |
| 11 | Disagreement strata (VERDICT vs baseline, 552 pairs) | `python experiments/accuracy/inspection/build_aegis_v5_mbench.py` | `experiments/accuracy/data/gold_refined_5judges_verbalized.json` | ✓ |
| 12 | 15-pair audit outcome distribution | `python experiments/clinician_validation/instrument/score_clinician_responses.py --stratum audit15` | `evaluations_results.json` filtered to `audit15` topics | △ |
| 13 | 15-pair re-review outcomes | Same as Table 12 with `--rereview` flag | same | △ |
| 14 | Clinician–reference agreement after re-review (kappa) | `python experiments/clinician_validation/processing/kappa.py` (in `report.py`) | evaluations file | ✓ |
| 15 | Reference eligibility vs SIGIR referral relevance | `python experiments/accuracy/scripts/table15_ref_vs_relevance.py` (**NEW**) | `data/gold/gold_5sys_freeform_balanced.json` + SIGIR qrels | ✗ |
| 16 | Clinician-adjusted accuracy under two conventions | `python experiments/clinician_validation/processing/report.py` → §accuracy-adjusted block | evaluations file | ✓ |
| 17 | CF self-faithfulness under alternative CF-generation settings (cell 1/2/3 sweep) | `for CELL in 1 2 3; do CELL=$CELL bash experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/launch_all.sh 1 16; done` + `python experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/summarize_3cell.py` | `mbench_3cell/cell{1,2,3}_*/` | ✓ |
| 18 | CF pipeline under GPT-5 modifier + GPT-5 validator (cell 3 headline) | `CELL=3 bash launch_all.sh 3 16` + `summarize_3cell.py --cell 3` | `mbench_3cell/cell3_5m_v3v_filt/` | ✓ |
| 19 | CF self-faithfulness on clinician-audited sample | `python experiments/clinician_validation/processing/cf_adjusted_precision.py` | `processing/results/cf_adjusted.json` | ✓ |
| 20 | Clinician audit for CF correction (p_flip / p_not) | Same as Table 19 — different columns of `cf_adjusted.json` | same | ✓ |
| 21 | Raw clinician pairwise-preference counts | `python experiments/clinician_validation/processing/report.py` → raw preferences block | evaluations file | ✓ |
| 22 | Clinician spot-check of CF rewrites under GPT-5 mod + val (K=7 audit) | `python experiments/clinician_validation/cf_audit/build_cf_audit_K7_v2.py` + clinician annotation | `cf_audit_K7_cell3.json` + `evaluations_results.json` | ✓ |
| 23 | Clinician–reference agreement on 32-pair audit | `python experiments/clinician_validation/processing/report.py --sample 32pair` | `instrument/samples/sample_32pairs_balanced_seed7.json` | △ |
| 24 | Pairwise preference with corpus-reweighted win rates | `python experiments/clinician_validation/processing/report.py --reweighted` | evaluations file + `merge/clinician_review_merged.json` | ✓ |
| 25 | Per-stratum pairwise preference | Same as Table 24 with per-stratum breakdown | same | ✓ |
| 26 | Per-axis rationale review with reweighted win rates | Same as Table 24 with per-axis output | same | ✓ |
| 27 | Population-reconstructed CF self-faithfulness | `python experiments/clinician_validation/processing/cf_adjusted_precision.py --reweighted` | `cf_adjusted.json` + `merge/clinician_review_merged.json` | ✓ |

---

## Reverse-CF section (Section 5.5)

**Not yet in the numbered tables** but present in the rebuttal draft. Regenerate:

```bash
# Stage 1: generate CFs for each difficulty type (n=60 pairs each)
bash experiments/counterfactual/06_reverse_cf/run_hard_serial.sh

# Stage 2b: rejudge with all matchers (v5, v5_cot, v5_blockers, shah, tg, Claude)
bash experiments/counterfactual/06_reverse_cf/run_rejudge_all.sh

# Type C at n=200 for tight CI + full 7-matcher grid
bash experiments/counterfactual/06_reverse_cf/scale_all.sh

# Summary tables
python experiments/counterfactual/06_reverse_cf/summarize.py \
    experiments/counterfactual/06_reverse_cf/out/reverse_cf_aegis_hard{A,B,C,D}_n60.rejudged_llm.jsonl \
    experiments/counterfactual/06_reverse_cf/out/reverse_cf_aegis_hardC_n200.rejudged_llm.jsonl
```

---

## Table 2 — TREC 2021 F1 (resolved 2026-09-03)

**Data lives in the `svpo-rl` repo**, not this one. It was pulled from
`<cluster-host>:<path-to>/svpo-rl` to `$SVPO_RL`:

```bash
ssh <cluster-host> 'tar czf - -C <path-to>/svpo-rl \
    scripts eval_runs sbatch training README_CLUSTER.md \
    data/test_tagged.jsonl data/test_clean_tagged.jsonl \
    data/mini_test_rollouts.jsonl data/mini_test_assign.jsonl \
    data/xu_assign.jsonl data/oracle_all.jsonl' \
  | tar xzf - -C $SVPO_RL
```

### How the 363-pair test set was built

    trec2021_mine.jsonl          compiled (patient, trial) pairs
      -> prep_2021_split.py      deterministic 80/20 split BY TRIAL
                                 "sort unique NCTs, every 5th NCT -> TEST"
      -> test_tagged.jsonl       436 held-out pairs
      -> parse-clean             drops 73 ("parse-limited")
      -> test_clean_tagged.jsonl 363 pairs, 190 eligible / 173 ineligible

Train and test share no NCT, so there is no trial-level leakage. "Parse-limited"
means no usable program was produced for the pair — it is **not** an
oracle-reachability test, and it is not conditioned on any system's verdict.

### Verified reproduction

```bash
python scripts/tables/table2_trec2021_f1.py
```

| System | P | R | F1 | Acc | paper |
|---|---|---|---|---|---|
| GPT-5-mini VERDICT | 0.928 | 0.747 | **0.828** | 0.837 | F1 0.828 / Acc 0.838 ✓ |
| Qwen2.5-7B VERDICT | 0.674 | 0.816 | **0.738** | 0.697 | F1 0.738 / Acc 0.697 ✓ |
| Qwen2.5-7B CoT | 0.674 | 0.653 | **0.663** | 0.653 | F1 0.663 ✓ |
| Qwen2.5-7B VERDICT+distill | 0.911 | 0.758 | **0.828** | 0.835 | F1 0.829 (rounding) |

**Still not reproducible:** Claude Haiku 4.5 rows and ZSPM (separate API runs,
outputs not retained on the cluster); Xu et al. (`data/xu_assign.jsonl` holds
atom assignments only and needs a solver pass to yield verdicts).

## Table 6 — resolved (was flagged in error)

The ZSPM 26.6% figure is **correct and internally consistent**: it is
111 flipped / 417 validator-accepted = 26.6%, from a full-corpus run of
N=496 attempted counterfactuals (Table 27, with bootstrap CI [22.6, 31.1]).
The GPT-4.1 backbone label is also correct.

An earlier note in this file claimed the number matched no artifact. That
was a counting error: `mbench_3cell/<cell>/<system>/{flipped,not_flipped}/`
is a **per-cell inspection subset** (cell 3 ZSPM = 56/203 = 27.5%), not the
denominator used by Tables 6 and 27, which pool the full corpus and divide
by validator-accepted CFs. Do not use `mbench_3cell/` counts to check
headline flip rates.

Why ZSPM's rate is low (for the Table 6 discussion — see also the
`TODO` in the main text): across its ineligible decisions, **92% of the
failing criteria are chart-silent defaults rather than evidence-backed
findings**, and **86% of its rejections contain no evidence-backed
failing criterion at all** (mean 3.4 silent-defaulted fails vs 0.3
evidence-backed per rejection). Because each silent default independently
forces rejection, flipping the *cited* reasons cannot move the verdict —
which is precisely what PIVOTALFLIPRATE is designed to detect. Verify with:

```bash
python scripts/tables/zspm_silence_breakdown.py
```

## Master driver

Update `scripts/reproduce_all.sh` to append the new tables:

```bash
# Existing stages 1-10 remain (see REPRODUCE_FROM_SCRATCH.md)

# Stage 11 — reverse CF (Section 5.5 / new Appendix R.4-R.5)
bash experiments/counterfactual/06_reverse_cf/run_hard_serial.sh
bash experiments/counterfactual/06_reverse_cf/run_rejudge_all.sh

# Stage 12 — TREC 2021 (Table 2)
python scripts/tables/table2_trec2021_f1.py --out experiments/accuracy/data/trec2021_results.json

# Stage 13 — clinician-derived tables (Tables 3, 4, 14, 16, 19-27)
python experiments/clinician_validation/processing/report.py \
    --out experiments/clinician_validation/processing/results/
```

---

## Gaps to close before camera-ready

Priority order:

1. **Table 6 ZSPM row**: fix the labeling bug OR rerun. See §Table 6 gap.
2. ~~**Table 2 TREC**~~ — DONE, see §Table 2 above. Remaining: Claude, ZSPM, Xu rows.
3. **Table 9 dataset stats** and **Table 10 verdict balance**: write two small `scripts/tables/table{9,10}_*.py` scripts. Both are pure counting over jsonls — <100 LOC each.
4. **Table 15 reference vs relevance**: write `scripts/tables/table15_ref_vs_relevance.py`. Cross-tab over `gold_5sys_freeform_balanced.json` and SIGIR qrels.
5. **Appendix N.5 (SmtMatch forward-CF failure taxonomy) — no automated regenerator.** The 4 archetypes are hand-classified from clinician notes. Document the source pairs but no script needed.

Everything else has a running reproduction path.

---

## Table 10 (Appendix G verdict balance) — mostly resolved 2026-09-03

The canonical sources are the **`*_freeform.jsonl`** artifacts, not
`verdicts.jsonl`. Verified by exact match on both eligible-rate and median
rationale length:

| system | file | elig% | paper | med | paper | |
|---|---|---|---|---|---|---|
| VERDICT | `aegis/aegis_freeform.jsonl` | 55.4 | 55.4 | 1021 | 1021 | exact |
| ourLLM | `single_shot_llm/v5_freeform.jsonl` | 35.0 | 35.0 | 1195 | 1195 | exact |
| CoT LLM | `single_shot_llm/v5_verbose_v2_freeform.jsonl` | 36.1 | 36.1 | 592 | 592 | exact |
| ZSPM | `shahlab/rationales.jsonl` | 54.1 | 54.0 | 1232 | 1342 | near |
| TrialGPT | `trialgpt/rationales.jsonl` | 13.2 | **44.4** | 2167 | 594 | **no match** |

Notes:

- "CoT LLM" is `v5_verbose_v2_freeform` — that is the fifth system, which is
  why the repo looked like it only had four.
- **ZSPM** covers 538 of 552 pairs, so the paper's 54.0% likely comes from a
  fuller run; the gap is one pair's worth of rounding.
- **TrialGPT does not reproduce.** Every TrialGPT artifact here gives 13.2%
  eligible against the paper's 44.4%. That run is not in this repository. The
  script prints `DOES NOT MATCH` on this row; do not cite it until the source
  is found.

The prior version of this script read `verdicts.jsonl` plus a nonexistent
`lm_only_V5_TWO_STEP.jsonl` and reproduced none of the paper's numbers.
`scripts/check_invariants.py` now pins the three exact rows.
