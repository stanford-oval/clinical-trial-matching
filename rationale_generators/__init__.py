"""Rationale generators.

Each generator turns a matcher's raw output for an INELIGIBLE pair into a
**blocker-only** rationale (the inputs the counterfactual modifier consumes).

Uniform interface:

    generate(matcher_output: dict, pair_meta: dict) -> RationaleSpec

where RationaleSpec is:

    {
        'kind':         'atom_targets' | 'rationale_text',
        'atom_targets': [...]   # only when kind == 'atom_targets'
        'rationale':    '...'   # only when kind == 'rationale_text'
        'supports':     '...'   # optional preserve hints
        'source':       '<system>.<config>'   # for traceability
    }

Two kinds:
  * `atom_targets`: structured atom flip list (aegis MaxSat).
  * `rationale_text`: free-text blocker list (everything else).

The counterfactual_modifier package routes by `kind`.
"""

from .aegis_maxsat import generate as aegis_maxsat_generate
from .shahlab_blockers import generate as shahlab_blockers_generate
from .trialgpt_blockers import generate as trialgpt_blockers_generate
from .v5_explanation import generate as v5_explanation_generate
from .v5_blockers_list import generate as v5_blockers_list_generate

GENERATORS = {
    'aegis':       aegis_maxsat_generate,
    'shahlab':     shahlab_blockers_generate,
    'trialgpt':    trialgpt_blockers_generate,
    'v5':          v5_explanation_generate,
    'v5_gpt5':     v5_explanation_generate,
    'v5_blockers': v5_blockers_list_generate,
}


def generate(system: str, matcher_output: dict, pair_meta: dict) -> dict:
    fn = GENERATORS.get(system)
    if fn is None:
        raise KeyError(f"no rationale generator for system '{system}'")
    return fn(matcher_output, pair_meta)
