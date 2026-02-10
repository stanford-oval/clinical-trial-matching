# Clinical Trial Matchers — System Bundles

Each subfolder is one of the four shared systems compared in the paper. All systems read the canonical SIGIR `corpus.jsonl` and produce eligibility verdicts on the 539-pair eval set.

## Systems

| System | Folder | Description |
|---|---|---|
| **AEGIS** | `aegis/` | SMT-backed atom-mining + Z3-solve + verbalize |
| **TrialGPT** | `trialgpt/` | Canonical Jin/Yang 2024 per-criterion matching prompt |
| **single_shot_llm** | `single_shot_llm/` | LLM-only baseline (formerly V5 TWO_STEP) |
| **shahlab** | `shahlab/` | Stanford som-shahlab Koopman-style prompt |

## Per-system layout (consistent)

```
<system>/
├── prompts/         — *.prompt files (easily editable to tune behavior)
├── run.py           — generator script (reads chart+criteria, writes verdicts)
├── verdicts.jsonl   — {pair, eligibility, ...} per pair
├── rationales.jsonl — verbalize/reasoning text per pair (used by judges)
├── (run_verbalize.py / build_judge_input.py — system-specific helpers)
└── README.md        — system-specific notes
```

## Tuning prompts

Each prompt is a plain-text file. Edit and re-run:

```bash
# Edit the AEGIS atom-mining prompt
$EDITOR matchers/systems/aegis/prompts/atom_mining.prompt

# Re-mine + re-solve for the canonical eval set
python matchers/systems/aegis/run.py
```

## Common interface (Python)

The pre-existing `matchers/variants.py` exposes a common `decide(pair_id)` interface for programmatic comparison:

```python
from matchers import variants
d = variants.smt_lm_evidence_arbiter("sigir-201414__NCT02192320")
print(d.decision, d.reasoning)
```

See `matchers/README.md` for the full API.

## Generator scripts (paths)

- `aegis/run.py` — `compute_aegis_strict_inc.py` (Z3 solve over cached atoms)
- `aegis/run_verbalize.py` — produces `rationales.jsonl` from verdicts + atoms
- `trialgpt/run.py` — `run_trialgpt_matching.py` (canonical matching prompt)
- `trialgpt/build_judge_input.py` — produces `rationales.jsonl` per-criterion summaries
- `single_shot_llm/run.py` — `lm_prompt_sweep.py --variant V5_TWO_STEP`
- `shahlab/run.py` — `run_stanford_baseline.py`
- `shahlab/build_judge_input.py` — produces TG-style per-criterion summary
