"""Bridge from stored pair data to the MaxSMT accountability artifacts.

The matchers decide a pair; this module answers the questions the decision
alone does not: *what did it assume, and what would change the answer.*

Input contract. `smt_core.maxsmt.solve` needs two things, and they carry
different meanings, so they are read from different places:

  phi_t        the trial's requirements, as hard constraints.
               Read from `raw["smt_program_lines"]`. These are the trial's
               asserts only -- the patient never appears in them.

  conditions   every c_i with the status q_i that decides how it is treated:
                 OBSERVED    the chart settles it -> becomes part of S, a soft
                             constraint the solver may have to give up
                 UNRESOLVED  the chart is silent -> excluded from S, and the
                             solver assigns it a value, which is the assumption
               Read from `raw["patient_var_values"]`: a value means OBSERVED,
               `None` means the miner found nothing, i.e. UNRESOLVED.

That split is the whole point. Passing a chart-silent condition as though it
were observed would turn an assumption into a finding, and the artifacts would
claim the chart said something it never said.

IMPUTED (a value supplied by policy rather than by the chart) is not
distinguishable in this stored format -- the miner records the value, not its
provenance. Everything non-null is therefore reported as OBSERVED. Runs that
carry policy provenance should pass it through explicitly.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from smt_core.maxsmt import (MAXSMT, OBSERVED, UNRESOLVED, Artifacts,
                             Condition, solve)
from verdict.data import pair_root

#: bookkeeping entries the miner writes alongside real atoms
_DERIVED_PREFIX = "__THRESH__::"

SIDES = ("inclusion", "exclusion")


def _load_raw(pair_id: str, side: str) -> Optional[Dict[str, Any]]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    try:
        patient, nct = pair_id.split("__", 1)
    except ValueError:
        raise ValueError(f"pair id must be '<patient>__<NCT>', got {pair_id!r}")
    pdir = pair_root() / "cmsrc_out" / patient
    files = sorted(pdir.glob(f"{nct}*__full.json")) if pdir.exists() else []
    if not files:
        return None
    obj = json.loads(files[0].read_text())
    return (obj.get(side) or {}).get("raw") or None


def conditions_from_pair(pair_id: str, side: str = "inclusion"
                         ) -> Tuple[List[str], List[Condition]]:
    """Read (phi_t, conditions) for one side of a pair.

    Returns ([], []) when the pair has no stored program for that side.
    """
    raw = _load_raw(pair_id, side)
    if not raw:
        return [], []
    phi = list(raw.get("smt_program_lines") or [])
    values = raw.get("patient_var_values") or {}
    rich = raw.get("patient_var_values_rich") or {}

    conditions: List[Condition] = []
    seen = set()
    for name, value in values.items():
        if name.startswith(_DERIVED_PREFIX):
            continue                      # miner bookkeeping, not a condition
        note = None
        entry = rich.get(name)
        if isinstance(entry, dict):
            note = entry.get("assessment")
        conditions.append(Condition(
            name=name,
            value=value,
            # a value means the chart settled it; None means it stayed silent
            status=OBSERVED if value is not None else UNRESOLVED,
            policy=note,
        ))
        seen.add(name)

    # phi_t = phi(c_1..c_n): EVERY variable the program declares is a condition.
    # One the miner never emitted a value for is not "absent", it is UNRESOLVED
    # -- and it matters. A declared variable left out of the condition list is
    # free, so the solver can satisfy -phi through it while keeping every
    # patient constraint, which silently empties delta_I and breaks the paper's
    # invariant that delta is never empty.
    for name in declared_variables(phi):
        if name not in seen:
            conditions.append(Condition(name=name, value=None,
                                        status=UNRESOLVED))
    return phi, conditions


_DECL = re.compile(r"\(declare-(?:const|fun)\s+\|?([^\s|)]+)\|?")


def declared_variables(phi_lines: Sequence[str]) -> List[str]:
    """Every variable phi_t declares, in program order."""
    out, seen = [], set()
    for line in phi_lines:
        m = _DECL.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def artifacts_for(pair_id: str, side: str = "inclusion",
                  version: str = MAXSMT) -> Optional[Artifacts]:
    """Accountability artifacts for one side of a pair.

    Returns None when the pair has no stored program, or when no working
    solver is installed -- callers fall back to the plain audit trail rather
    than failing.
    """
    phi, conditions = conditions_from_pair(pair_id, side)
    if not phi or not conditions:
        return None
    try:
        return solve(phi, conditions, version=version)
    except ImportError:
        return None                       # no usable z3; caller degrades


def summarize(pair_id: str, side: str = "inclusion",
              version: str = MAXSMT) -> str:
    """Human-readable assumptions and pivotal conditions, or '' if unavailable."""
    a = artifacts_for(pair_id, side, version)
    if a is None:
        return ""
    phi, _ = conditions_from_pair(pair_id, side)
    view = a.for_verbalizer(phi)
    out = [f"accountability artifacts ({side}, {a.version})",
           f"  decision by solver : {a.decision}"]

    if view["assumptions"]:
        out.append(f"  assumed ({len(view['assumptions'])} conditions the chart "
                   f"did not settle):")
        for name, info in list(view["assumptions"].items())[:8]:
            req = info.get("requirement")
            # report the requirement, never the solver's placeholder value
            out.append(f"    {name}: " +
                       (f"assumed to meet {req}" if req
                        else f"assumed {info.get('value')}"))
        if len(view["assumptions"]) > 8:
            out.append(f"    ... and {len(view['assumptions']) - 8} more")
    else:
        out.append("  assumed: nothing -- the chart settled every condition")

    out.append("  would change the answer:")
    for c in a.pivotal[:8] or ["(none found)"]:
        out.append(f"    {c}")
    return "\n".join(out)
