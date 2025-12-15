"""Typed missingness policy — silence-rule lookup.

The policy is a CSV with columns:
    Level 1 Type, Level 2 Type,
    Polarity_positive_silence_default_value,
    Polarity_negative_silence_default_value

For each predicate, the policy maps the (Level1, Level2) category and
the predicate's polarity_role to an evidence state when the chart is
silent (no observed value).

States:
    explicit_true   chart documents the predicate holds
    explicit_false  chart documents the predicate does NOT hold
    assumed_true    chart silent; policy treats as true (silence ≈ presence)
    assumed_false   chart silent; policy treats as false (silence ≈ absence)
    unknown_defer   chart silent; policy defers to the visit

Inversion constraint: for a predicate-name pair (has_X, has_no_X), the
silence defaults must be inverses (assumed_false ↔ assumed_true;
unknown_defer is its own inverse).
"""
import csv
import hashlib
import pathlib

HERE = pathlib.Path(__file__).parent
POLICY_CSV = HERE / 'policies' / 'typed_missingness.csv'
POLICY_MD  = HERE / 'policies' / 'typed_missingness.md'

EVIDENCE_STATES = {
    'explicit_true', 'explicit_false',
    'assumed_true',  'assumed_false',
    'unknown_defer',
}

# Map AST-derived polarity_role -> demand direction
ROLE_TO_DEMAND = {
    'inc_demand_true':       'positive',
    'exc_forbid_false':      'positive',
    'inc_demand_false':      'negative',
    'exc_forbid_true':       'negative',
    'inc_alternatives':      'alternatives',
    'exc_alternatives':      'alternatives',
    'inc_value_constrained': 'value_constrained',
    'exc_value_constrained': 'value_constrained',
}
POLARITY_ROLES = set(ROLE_TO_DEMAND)


def _load_table():
    table = {}
    with POLICY_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (row['Level 1 Type'].strip(), row['Level 2 Type'].strip())
            table[key] = (
                row['Polarity_positive_silence_default_value'].strip(),
                row['Polarity_negative_silence_default_value'].strip(),
            )
    return table


TABLE = _load_table()
POLICY_TEXT = POLICY_MD.read_text() if POLICY_MD.exists() else ''
POLICY_HASH = hashlib.sha256(POLICY_TEXT.encode()).hexdigest()[:16]
POLICY_VERSION = 'v5'


def resolve_silence(level1: str, level2: str, polarity_role: str) -> str:
    """Look up the silence-default evidence state for a (category, polarity_role) cell."""
    demand = ROLE_TO_DEMAND.get(polarity_role)
    if demand in ('alternatives', 'value_constrained'):
        return 'unknown_defer'
    key = (level1, level2)
    if key not in TABLE:
        for fallback in [(level1, 'Other'), ('Other', 'All')]:
            if fallback in TABLE:
                key = fallback
                break
        else:
            return 'unknown_defer'
    pos, neg = TABLE[key]
    cell = pos if demand == 'positive' else neg
    return cell if cell in EVIDENCE_STATES else 'unknown_defer'


def is_load_bearing_failing(polarity_role: str, state: str) -> bool:
    """True iff this (role, state) combination forces an ineligible verdict."""
    demand = ROLE_TO_DEMAND.get(polarity_role)
    if demand == 'positive':
        return state in ('explicit_false', 'assumed_false')
    if demand == 'negative':
        return state in ('explicit_true', 'assumed_true')
    return False


def evidence_state(level1: str, level2: str, observed_value,
                   polarity_role: str) -> str:
    """Resolve a predicate's evidence state from chart observation + policy."""
    if observed_value is not None:
        if observed_value is True:  return 'explicit_true'
        if observed_value is False: return 'explicit_false'
        return 'explicit_true'   # non-null non-bool -> treat as present
    return resolve_silence(level1, level2, polarity_role)
