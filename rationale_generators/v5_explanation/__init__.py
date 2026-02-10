"""V5 / V5-gpt-5 rationale generator: pass through the matcher's `explanation` field.

V5 prompts produce a short (1-3 sentence) blocker-focused explanation as part of
their structured output. No filtering needed — the explanation is already a
narrative summary of why the patient is ineligible.

Used for both:
  - single_shot_llm.v5 (V5_PROMPT, gpt-4.1)
  - single_shot_llm.v5_gpt5 (V5_PROMPT, gpt-5)
"""
from __future__ import annotations


def generate(matcher_output: dict, pair_meta: dict) -> dict:
    expl = (matcher_output.get('explanation')
            or matcher_output.get('rationale')
            or matcher_output.get('cited_rationale')
            or '').strip()
    return {
        'kind':      'rationale_text',
        'rationale': expl or '(no V5 explanation extracted)',
        'supports':  '',  # V5 has no structured supports list
        'source':    'single_shot_llm.v5_explanation',
    }
