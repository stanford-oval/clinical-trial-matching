# AEGIS prompts — paper-canonical set

This folder holds the **exact prompts that produced the AEGIS results in the
paper** (the `rationales_v9_arbiter.jsonl` GPT-4.1 pipeline). Each file was
recovered and hash-verified against the May-5 cmsrc per-pair cache
(`experiments/53_v2_full/cmsrc_out_REMINE_v9_full/_prompt_cache/`), which
embeds the prompt template inside every one of its 1,711 cached calls. All
1,711 calls recorded a single, identical version of each prompt.

Do **not** edit these files. They are a frozen reproducibility snapshot.

## Pipeline stage → prompt

| Stage | Module | File |
|-------|--------|------|
| 2 | Trial-side scope miner (`SP_T`)        | `trial_scope_miner.prompt` |
| 3 | Trial-side projection rewriter (`SP_T`) | `trial_projection_rewriter.prompt` |
| 4 | Patient-side value miner — inclusion (`SP_P`) | `patient_value_miner_inclusion.prompt` |
| 4 | Patient-side value miner — exclusion (`SP_P`) | `patient_value_miner_exclusion.prompt` |
| — | Solver resolution module / arbiter (`S`)      | `arbiter.prompt` |
| — | Verbalizer (`V`)                              | `verbalizer.prompt` |

Stages 1 (leaf collection), 5 (alias remapping), and 6 (Z3 program
evaluation) are deterministic and use no prompt. The criterion-to-SMT
compilation that produces the trial formula builds on the SatIR semantic
parser (Zhou et al., 2026) and is not reproduced here.

## Provenance and the stale on-disk copies

The live working copies on disk are **not** a faithful record of the paper run:

- `experiments/53_v2_full/inputs/prompt_root/prompt_out/smt_inclusion.prompt`
  and `smt_exclusion.prompt` were edited on **May 7**, two days after the
  V9 mine (the canonical results) was built on **May 5**. The May-7 edits
  added silence-handling/qualifier guidance that never influenced the paper
  numbers.
- `matchers/systems/aegis/prompts/` contains a third, separately-drifted
  version of the exclusion miner.

The scope-miner and projection-rewriter prompts were unchanged between the
working copies and the May-5 cache; the value-miner prompts were not. This
folder is the authoritative set.

## Source mapping

| File here | Recovered from |
|-----------|----------------|
| `trial_scope_miner.prompt`            | May-5 cache (`SMTVariableScopeMiner_prompt`) |
| `trial_projection_rewriter.prompt`    | May-5 cache (`SMTVariableProjectionRewriter_prompt`) |
| `patient_value_miner_inclusion.prompt`| May-5 cache (`SMTVariableValueMinerInclusion_prompt`) |
| `patient_value_miner_exclusion.prompt`| May-5 cache (`SMTVariableValueMinerExclusion_prompt`) |
| `arbiter.prompt`                      | `matchers/systems/aegis/aegis_arbiter.py` (`ARBITER_PROMPT`; file unchanged since the May-5 23:54 arbiter run) |
| `verbalizer.prompt`                   | `verbalizer/prompts/verbalize_aegis_v9_arbiter.prompt` (May-7; stable before the May-17 verbalization run) |

The same six files are mirrored under `paper/prompts/system/` for the
appendix's `\VerbatimInput` directives.
