"""Stage 2 — typed predicate -> (Level 1, Level 2) category.

For each unique (name, type, polarity) triple, the LLM picks one of
the v5 taxonomy's 29 (Level 1, Level 2) pairs. Output includes the
classifier's one-line rationale. Invalid pairs fall back to Other/All.
"""
import json
import pathlib

from . import _llm

HERE = pathlib.Path(__file__).parent
PROMPT_PATH = HERE / 'prompts' / 'classifier.prompt'
PROMPT = PROMPT_PATH.read_text()

VALID = [
    ('Disease/Disorder', 'Common Chronic Disease'),
    ('Disease/Disorder', 'Uncommon Chronic Disease'),
    ('Disease/Disorder', 'Common Acute Disease'),
    ('Disease/Disorder', 'Uncommon Acute Disease'),
    ('Disease/Disorder', 'Other Disease'),
    ('Symptom/Sign', 'Past Symptom'),
    ('Symptom/Sign', 'Current Symptom'),
    ('Symptom/Sign', 'Exam Finding (Current Visit)'),
    ('Symptom/Sign', 'Exam Finding (Past Visits)'),
    ('Symptom/Sign', 'Other Symptom/Sign'),
    ('Numerical', 'All'),
    ('Numerical Threshold', 'Encoding an Uncommon Disease/Disorder/Symptom'),
    ('Numerical Threshold', 'Encoding a Common Disease/Disorder/Symptom'),
    ('Numerical Threshold', 'Other'),
    ('Procedure', 'Planned / Scheduled Procedure'),
    ('Procedure', 'Past or Ongoing Imaging Study'),
    ('Procedure', 'Past or Ongoing Diagnostic Procedure'),
    ('Procedure', 'Past or Ongoing Surgical Procedure'),
    ('Procedure', 'Past or Ongoing Therapeutic Non-Surgical Procedure'),
    ('Procedure', 'Past or Ongoing Vaccination / Immunization'),
    ('Procedure', 'Past or Ongoing Routine Procedure'),
    ('Procedure', 'Other'),
    ('Demographic / Personal Info', 'Age/Sex'),
    ('Demographic / Personal Info', 'Pregnancy / Lactation'),
    ('Demographic / Personal Info', 'Race / Ethnicity / Nationality'),
    ('Demographic / Personal Info', 'Other'),
    ('Consent / Setting', 'Consent / Logistic'),
    ('Consent / Setting', 'Other'),
    ('Other', 'All'),
]
VALID_SET = set(VALID)


def classify(triples: list[tuple], model: str = 'gpt-4.1',
             batch_size: int = 30) -> list[dict]:
    """Classify a list of (name, type, polarity) triples.

    Args:
        triples: list of (name, type, polarity) tuples.
        model: LLM model name.
        batch_size: predicates per LLM call.

    Returns:
        list of {name, type, polarity, level1, level2, rationale} dicts.
    """
    out = []
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i+batch_size]
        atom_list = json.dumps(
            [{'name': n, 'type': t, 'polarity': p} for (n, t, p) in batch],
            indent=2,
        )
        filled = PROMPT.replace('{atom_list}', atom_list)
        raw = _llm.call(filled, model=model, max_tokens=8000)
        obj = _llm.parse_json(raw)
        emitted = {}
        for r in obj.get('classifications', []):
            nm = r.get('name')
            l1 = (r.get('level1') or '').strip()
            l2 = (r.get('level2') or '').strip()
            rationale = (r.get('rationale') or '')[:300]
            if (l1, l2) not in VALID_SET:
                l1, l2 = 'Other', 'All'
            emitted[nm] = (l1, l2, rationale)
        for (n, t, p) in batch:
            if n in emitted:
                l1, l2, rat = emitted[n]
                out.append({'name': n, 'type': t, 'polarity': p,
                            'level1': l1, 'level2': l2, 'rationale': rat})
            else:
                out.append({'name': n, 'type': t, 'polarity': p,
                            'level1': 'Other', 'level2': 'All',
                            'rationale': 'omitted by classifier'})
    return out
