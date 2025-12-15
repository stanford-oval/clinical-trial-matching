"""Stage 1 — strict-extractive chart value mining.

For each (chart, predicate_list) pair, emit a per-predicate observed
value (true / false / number / null) backed by direct chart evidence.
The prompt forbids silence-derived defaults; a post-hoc regex scrubber
catches any policy-leakage that slips through.
"""
import json
import pathlib
import re
from typing import Any, Optional

from . import _llm

HERE = pathlib.Path(__file__).parent
PROMPT_PATH = HERE / 'prompts' / 'value_miner.prompt'
PROMPT = PROMPT_PATH.read_text()

# Heuristic detector for policy leakage. Any evidence string matching one
# of these forces the predicate value back to null.
POLICY_LEAK_PATTERNS = [
    r'\bsilent\b', r'defer to visit', r'\bpolicy\b', r'\bdefault\b',
    r'\bnot mentioned\b', r'\bno mention\b', r'\bnot addressed\b',
    r'\bno evidence\b(?! of \w+)',
    r'\bnot documented\b', r'\bsilent on\b',
]
_LEAK_RE = re.compile('|'.join(POLICY_LEAK_PATTERNS), re.IGNORECASE)


def mine(chart: str, predicates: list[dict], model: str = 'gpt-4.1') -> list[dict]:
    """Run the strict value miner on one (chart, predicate-list) pair.

    Args:
        chart: patient chart text.
        predicates: list of {name, type, polarity} dicts.
        model: LLM model name (default gpt-4.1).

    Returns:
        list of {name, value, evidence} dicts. value is one of:
          true, false, a number, or None (chart silent / scrubbed).
    """
    filled = (PROMPT
              .replace('{{CHART}}', chart[:8000])
              .replace('{{PREDICATES}}', json.dumps(predicates, indent=2)[:12000]))
    raw = _llm.call(filled, model=model, max_tokens=6000)
    obj = _llm.parse_json(raw)
    out = []
    for r in obj.get('predicates', []):
        name = r.get('name')
        val  = r.get('value')
        evid = r.get('evidence', '') or ''
        # Scrubber: any leak pattern in evidence forces null
        if val is False and _LEAK_RE.search(evid):
            val = None
            evid = f'(scrubbed policy-leak) {evid}'
        out.append({'name': name, 'value': val, 'evidence': evid})
    return out
