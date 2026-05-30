# Which matcher is which

Three different things in this repository decide patient--trial eligibility.
They are not interchangeable, and only one produces the paper's numbers.

## 1. `verdict/headline.py` — the paper's system

**This is VERDICT as published.** Vendored from `overnight/repro_headline.py`;
only the I/O paths were changed.

```bash
verdict headline          # or: python verdict/headline.py
```

It is not a bare SMT matcher. It is v6 mined atoms plus three doctrine layers:

- **silence-null, both sides** — drop FALSE-on-silence on inclusion *and*
  exclusion
- **58 compiled patches** — `NULL -> FALSE` coercion on patched atoms only,
  with `(assert atom=true)` injected
- **population gate** — 65 pairs the LM gate declines to forward

Gold is the majority of 5 LM judges (>=3/5). Inputs ship under
`data/headline/` (21 MB); the v6 artifact is reduced to the four fields the
script reads and gives results identical to the full 270 MB originals.

Output, on the shipped artifacts:

| row | F1 | P | R |
|---|---|---|---|
| VERDICT+silence-null | 0.835 | 0.769 | 0.915 |
| **VERDICT default (compiled)** | **0.873** | 0.884 | 0.862 |
| VERDICT opt-in (∪LLM-only) | 0.868 | 0.784 | 0.972 |
| VERDICT+verdict-gate | 0.873 | 0.901 | 0.846 |
| LLM-only | 0.875 | 0.803 | 0.960 |

**Known discrepancy.** The paper reports **0.863 / 0.837 / 0.891** for the
default row; this run gives **0.873 / 0.884 / 0.862**. Two of the five rows
(silence-null, verdict-gate) reproduce bit-exactly, three shift by 1--2 F1
points. Script and inputs all predate the paper's stored result, so the cause
is environmental -- most likely the solver version, since which satisfying
assignment Z3 returns can change with it. Every qualitative claim is
unaffected: VERDICT ties LLM-only on F1, VERDICT has higher precision, LLM-only higher recall.
Unresolved; do not present the numbers as bit-reproducible.

## 2. `matchers/variants.py` — a reimplementation, not the paper

Six variants (`verdict`, `smt-only`, `atoms`, `lm-only`, `hybrid`,
`trialgpt`) behind `verdict match` and `verdict.match()`. Useful for
inspecting one pair and its audit trail, and for comparing approaches through
the registry.

**It does not reproduce the paper.** Measured against the paper's own
`verdicts.jsonl`, the best variant agrees on **88.7%** of pairs. It reads a
different mining snapshot (`53_v2_full`, April) and has none of the three
doctrine layers above -- so, structurally, it can never overturn an
SMT-accept, which is where a third of the disagreements come from.

## 3. `smt_matcher/` — SatIR's matcher

The retrieval-side matcher, 630 lines. Distinct from `cmsrc`'s 1,812-line
matcher, which produced the mined atoms. Used by the SatIR pipeline, not by
the paper's headline.

## The mined snapshots

`experiments/53_v2_full/cmsrc_out*` are mining runs for the distillation and
counterfactual work. The headline uses `53h_v6_full`, a different run. Reading
the wrong one costs ~5 points of agreement, and nothing warns you.

## What is reproducible today

| | |
|---|---|
| Headline table, structure and counts (856 / 553 / 539 / 58 / 65) | yes, `verdict headline` |
| Headline numbers, exactly as printed in the paper | **no** -- see the discrepancy above |
| The paper's per-pair verdicts | yes, they ship as `matchers/systems/aegis/verdicts.jsonl` (F1 0.861 vs the paper's 0.863) |
| Re-running the SMT matcher from scratch | needs `cmsrc` and Azure credentials |

## Names

The paper's names are **VERDICT** (the system) and **LLM-only** (the
single-prompt baseline). Earlier drafts called them **AEGIS** and **V5**; those
are historical names and should not appear in anything a reader sees. An
invariant enforces that.

They are deliberately still present in two places:

- **On-disk paths** — `matchers/systems/aegis/`, `aegis_freeform.jsonl`.
  Renaming these would break every stored reference and the provenance chain
  back to the runs that produced the paper.
- **Inside data artifacts** — records carry `"system": "aegis_freeform"`.
  Editing a stored artifact to match new terminology would be falsifying it.

| paper | legacy name | where the legacy name survives |
|---|---|---|
| VERDICT | AEGIS | `matchers/systems/aegis/`, artifact `system` fields |
| LLM-only | V5 | `matchers/systems/single_shot_llm/`, `lm_only_V5_TWO_STEP.jsonl` |

## The vendored matcher (`verdict/engine/`)

`verdict run` executes the real matcher, vendored from the `cmsrc` checkout
that produced the paper's results. This is the system, not a reimplementation
— the distinction that `matchers/variants.py` failed to satisfy.

**What was changed on vendoring.** Only module resolution and I/O paths:

- flat imports (`from utils import ...`) became package-relative;
- `data_root`, `build_root`, `project_root` were resolved against the *current
  working directory* upstream (`../build`, `../dataset/clinical_trial`), so the
  matcher only ran from inside its own directory. They are now anchored to the
  package and overridable via `$TRIAL_DATA` and `$VERDICT_BUILD`.
- `prompt_root` likewise resolved against `cwd()`.

No prompt text, solver logic, or decision rule was modified.

**Which prompts shipped, and why it matters.** The prompts in the `cmsrc`
working tree are *older* (17 Mar) than the snapshot the paper ran against
(27 Apr), and are missing 84 lines across `smt.prompt` and
`SMTVariableProjectionRewriter.prompt` implementing the chart-typical-detail
calibration — the "would this be mentioned if true" test that the silence-null
results depend on.

Vendoring the working copy would therefore have shipped a **pre-silence**
configuration under the paper's name. `verdict/engine/prompts/prompt_out/` is
the 27 Apr snapshot from `53_v2_full/inputs/prompt_root`, which is what the
published runs used.

**What is still not shipped.** Compiled trial programs (`$VERDICT_BUILD`:
IR, symtab, linkmap, canon) and the patient corpora. `verdict run` needs a
build tree and an LLM endpoint; it reports which is missing rather than
failing obscurely.

## Why the TrialGPT baseline is not shipped

TrialGPT (Jin et al., NCBI/NLM) was used during development as an external
comparison. Its criterion-matching prompt and the code around it were carried
into this codebase — verbatim prompt text in the matcher, plus a
`matchers/systems/trialgpt/` runner. Both have been removed.

**Why.** TrialGPT is a United States Government Work under the NCBI Public
Domain Notice. That is maximally permissive — no restriction on use or
reproduction — so this was never a licensing *bar*. It was removed because
carrying another group's system inside ours costs more than it returns:

1. **It made the provenance story harder to state.** This repository ships one
   matcher. A second, third-party judge sitting dormant in the same file
   invited exactly the confusion this project has already paid for once, when
   a reimplementation was mistaken for the published system.
2. **It could not be cited honestly from here.** Our stored TrialGPT artifacts
   score 13.2% where the paper reports 44.4%. The row was already marked
   "do not cite". Shipping a baseline we cannot reproduce is worse than
   shipping no baseline.
3. **It was dormant.** The judge was opt-in (`--trialgpt-judge`) and no
   shipped entry point ever enabled it, so nothing in the tool lost a feature.

**What this is not.** Not a claim about TrialGPT's quality, and not a
licensing dispute. Anyone wanting the comparison should run TrialGPT from its
own repository, against its own maintained prompts, and cite it directly:

    Qiao Jin, Zifeng Wang, Charalampos S. Floudas, Fangyuan Chen,
    Changlin Gong, Dara Bracken-Clarke, Elisabetta Xue, Yifan Yang,
    Jimeng Sun, Zhiyong Lu.
    Matching Patients to Clinical Trials with Large Language Models.

Citing TrialGPT as prior work in prose and reporting its published numbers is
unaffected; only the vendored code and artifacts are gone. The removed files
are preserved outside the repository, under `_backups/`.
