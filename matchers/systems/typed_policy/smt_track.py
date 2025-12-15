"""Stage 3 — SMT-based deterministic aggregation track.

Given a trial's typed predicates, observed values, classifications, and
polarity_roles, applies the policy mechanically and uses Z3 to check
satisfiability of inclusion + exclusion programs. Verdict is ground
truth by construction.
"""
import z3

from . import policy as _policy


def _make_const(name, ty):
    if ty == 'Bool': return z3.Bool(name)
    if ty == 'Real': return z3.Real(name)
    if ty == 'Int':  return z3.Int(name)
    return z3.Bool(name)


def _coerce(val, ty):
    if ty == 'Bool': return z3.BoolVal(val)
    if ty == 'Real': return z3.RealVal(1 if val else 0)
    return z3.BoolVal(val)


def solve_side(smt_text: str, predicates: list[dict],
               observed: dict, categories: dict,
               polarity_roles: dict, side: str):
    """Solve one polarity side (inclusion or exclusion).

    Args:
        smt_text: SMT-LIB program text for this side.
        predicates: list of {name, type, polarity} dicts for this side.
        observed: {predicate_name: observed_value or None}.
        categories: {(name, type, polarity): (level1, level2)}.
        polarity_roles: {predicate_name: polarity_role_str}.
        side: 'inclusion' or 'exclusion'.

    Returns:
        (sat_like: bool|None, per_predicate_states: list[dict]).
        sat_like = True iff Z3 found a satisfying model.
    """
    ctx = z3.Solver()
    try:
        ctx.add(z3.parse_smt2_string(smt_text))
    except z3.Z3Exception as e:
        return None, [{'error': f'parse: {str(e)[:200]}'}]

    per_pred = []
    for p in predicates:
        name, ty = p['name'], p['type']
        l1, l2 = categories.get((name, ty, p['polarity']), ('Other', 'All'))
        role = polarity_roles.get(name) or (
            'inc_demand_true' if side == 'inclusion' else 'exc_forbid_true'
        )
        ov = observed.get(name)
        state = _policy.evidence_state(l1, l2, ov, role)
        if state in ('explicit_true', 'assumed_true'):
            const = _make_const(name, ty)
            ctx.add(const == _coerce(True, ty))
        elif state in ('explicit_false', 'assumed_false'):
            const = _make_const(name, ty)
            ctx.add(const == _coerce(False, ty))
        # unknown_defer: leave unconstrained (Z3 finds a model)
        per_pred.append({
            'name': name, 'polarity': p['polarity'],
            'level1': l1, 'level2': l2,
            'polarity_role': role,
            'observed': ov, 'state': state,
        })

    result = ctx.check()
    sat_like = (result == z3.sat) if result != z3.unknown else None
    return sat_like, per_pred


def solve_pair(inc_smt: str, exc_smt: str, predicates: list[dict],
               observed: dict, categories: dict,
               polarity_roles: dict) -> dict:
    """Solve a full (patient, trial) pair.

    Args:
        inc_smt: SMT-LIB inclusion program text.
        exc_smt: SMT-LIB exclusion program text.
        predicates: all predicates (both polarities) from the inventory.
        observed: {predicate_name: observed_value or None}.
        categories: {(name, type, polarity): (level1, level2)}.
        polarity_roles: {predicate_name: polarity_role_str}.

    Returns:
        {verdict, inc_sat_like, exc_sat_like, per_predicate,
         policy_version, policy_hash}
    """
    inc_preds = [p for p in predicates if p['polarity'] == 'inclusion']
    exc_preds = [p for p in predicates if p['polarity'] == 'exclusion']
    inc_sat, inc_states = solve_side(inc_smt, inc_preds, observed, categories,
                                     polarity_roles, 'inclusion')
    exc_sat, exc_states = solve_side(exc_smt, exc_preds, observed, categories,
                                     polarity_roles, 'exclusion')
    if inc_sat is None or exc_sat is None:
        verdict = 'unknown'
    else:
        eligible = bool(inc_sat) and not bool(exc_sat)
        verdict = 'eligible' if eligible else 'ineligible'
    return {
        'verdict': verdict,
        'inc_sat_like': inc_sat, 'exc_sat_like': exc_sat,
        'per_predicate': inc_states + exc_states,
        'policy_version': _policy.POLICY_VERSION,
        'policy_hash': _policy.POLICY_HASH,
    }
