"""
Parallel decision baselines for SMT matcher evaluation.

Each judge produces an eligibility decision on the same (patient, trial)
pair using a different approach:

    - llm_judge:      GPT-4 NL eligibility on full trial text + patient note
    - trialgpt_judge: Sentence-level TrialGPT criterion-by-criterion matcher

These judges run in parallel with the SMT matcher and let you compare
SMT (formal) vs LLM (end-to-end) vs TrialGPT (sentence-level) decisions
for research/evaluation purposes.

See smt_matcher/judges/README.md for usage.
"""
from smt_matcher.judges.llm_judge import run_llm_eligibility_judge
from smt_matcher.judges.trialgpt_judge import run_trialgpt_judge

__all__ = ["run_llm_eligibility_judge", "run_trialgpt_judge"]
