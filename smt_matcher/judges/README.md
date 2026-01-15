# smt_matcher/judges/ — Parallel Decision Baselines

Each judge produces an eligibility decision on the same (patient, trial) pair
using a different approach. Run alongside the SMT matcher for head-to-head
comparison in research evaluation.

## Judges

### `llm_judge.run_llm_eligibility_judge`

GPT-4 end-to-end NL eligibility on the full trial text + patient note.

```python
from smt_matcher.judges import run_llm_eligibility_judge

payload = run_llm_eligibility_judge(
    trial_id="NCT00000402",
    trial_obj=trial_record,          # dict with brief_title/summary/criteria
    patient=patient_record,           # dict with text/note
    engine=azure_engine,
    prompt_path=Path("prompts/.../eligibility.explicit.prompt"),
    out_root=Path("./out"),           # creates out/_prompt_cache/llm_judge/
    temperature=0.0,
)
# payload["result"] = {"eligible": True/False/None, "eligibility": "eligible"/..., "reasoning": "...", ...}
```

### `trialgpt_judge.run_trialgpt_judge`

Sentence-level criterion-by-criterion judge (TrialGPT methodology). Each
inclusion/exclusion criterion is evaluated individually and linked to specific
patient-note sentence IDs.

```python
from smt_matcher.judges import run_trialgpt_judge

payload = run_trialgpt_judge(
    trial_id="NCT00000402",
    trial_obj=trial_record,
    patient=patient_record,
    engine=azure_engine,
    out_root=Path("./out"),           # creates out/_prompt_cache/trialgpt_*/
    temperature=0.0,
)
# payload["aggregate"] = {"eligible": bool, "inclusion_sat_like": ..., "exclusion_sat_like": ..., ...}
# payload["inclusion"]["rows"] = per-criterion labels
```

## Caching

Both judges use the fingerprinting scheme in `smt_matcher.cache`. Identical
inputs (prompt text, patient note, trial, model, temperature) hit the cache
instead of re-calling the LLM. Bump `CACHE_SCHEMA_VERSION` in `cache.py`
to invalidate all caches.

## Research use

See `evaluation/` sibling package for:
- `evaluation/fliprate/` — measures run-to-run consistency
- `evaluation/agreement/` — three-way (SMT / LLM / TrialGPT) comparison
- `evaluation/taxonomy/` — LLM failure categorization
