"""The accountability artifacts, both published formulations."""
from __future__ import annotations

import pytest

from smt_core.maxsmt import (ASSUMED, ELIGIBLE, IMPUTED, INELIGIBLE, MAXSMT,
                             OBSERVED, PAPER_OF, RESIDUAL, SYMBOLS, UNRESOLVED,
                             Condition, requirement_for, solve)

from _support import requires_z3


# ---------------------------------------------------------------- pure logic
def test_requirement_recovers_bound(phi):
    assert requirement_for("crcl", phi) == ">= 60"


def test_requirement_is_none_for_boolean(phi):
    """A boolean has no numeric bound; the assigned value IS the fact."""
    assert requirement_for("egfr", phi) is None


def test_requirement_handles_reversed_operands():
    assert requirement_for("age", ["(assert (<= 18 |age|))"]) == ">= 18"


def test_assumed_is_accepted_as_alias_for_imputed():
    """The submitted paper says ASSUMED, the update says IMPUTED."""
    assert Condition("x", True, ASSUMED).resolved
    assert Condition("x", True, IMPUTED).resolved
    assert not Condition("x", None, UNRESOLVED).resolved


def test_unknown_version_rejected(phi):
    with pytest.raises(ValueError, match="version must be one of"):
        solve(phi, [], version="nope")


# ---------------------------------------------------------------- the solver
@requires_z3
@pytest.mark.parametrize("conds,expected", [
    ([Condition("egfr", False, OBSERVED), Condition("crcl", None, UNRESOLVED)], INELIGIBLE),
    ([Condition("egfr", True, OBSERVED),  Condition("crcl", None, UNRESOLVED)], ELIGIBLE),
    ([Condition("egfr", True, OBSERVED),  Condition("crcl", 80, OBSERVED)],     ELIGIBLE),
    ([Condition("egfr", True, OBSERVED),  Condition("crcl", 40, OBSERVED)],     INELIGIBLE),
])
def test_decision(phi, conds, expected):
    assert solve(phi, conds).decision == expected


@requires_z3
@pytest.mark.parametrize("conds", [
    [Condition("egfr", False, OBSERVED), Condition("crcl", None, UNRESOLVED)],
    [Condition("egfr", True, OBSERVED),  Condition("crcl", None, UNRESOLVED)],
    [Condition("egfr", True, OBSERVED),  Condition("crcl", 80, OBSERVED)],
    [Condition("egfr", True, OBSERVED),  Condition("crcl", 40, OBSERVED)],
    [Condition("egfr", True, IMPUTED, "assume-normal"),
     Condition("crcl", None, UNRESOLVED)],
])
def test_paper_invariants(phi, conds):
    """delta_E = {} iff ELIGIBLE; delta_I = {} iff INELIGIBLE; delta != {}."""
    a = solve(phi, conds)
    assert (a.delta_e == []) == (a.decision == ELIGIBLE)
    assert (a.delta_i == []) == (a.decision == INELIGIBLE)
    assert a.pivotal != []


@requires_z3
def test_worked_example_matches_paper(phi):
    """Paper: d=INELIGIBLE, rho records crcl, delta = {egfr}."""
    a = solve(phi, [Condition("egfr", False, OBSERVED),
                    Condition("crcl", None, UNRESOLVED)])
    assert a.decision == INELIGIBLE
    assert a.pivotal == ["egfr"]
    assert "crcl" in a.assumptions


# ------------------------------------------------- the two formulations
@requires_z3
def test_rho_differs_in_kind_between_versions(phi):
    """Submitted rho is a requirement; updated rho is a witness."""
    conds = [Condition("egfr", False, OBSERVED), Condition("crcl", None, UNRESOLVED)]
    r = solve(phi, conds, version=RESIDUAL)
    u = solve(phi, conds, version=MAXSMT)
    assert r.assumptions["crcl"] == ">= 60"          # a constraint
    assert isinstance(u.assumptions["crcl"], float)  # a value satisfying it
    assert r.decision == u.decision


@requires_z3
def test_delta_e_i_only_exist_in_updated_version(phi):
    conds = [Condition("egfr", False, OBSERVED), Condition("crcl", None, UNRESOLVED)]
    assert solve(phi, conds, version=RESIDUAL).delta_e == []
    assert solve(phi, conds, version=MAXSMT).delta_e == ["egfr"]


@requires_z3
def test_both_versions_agree_on_the_decision(phi):
    """Only the artifacts differ between papers; d must not."""
    for conds in ([Condition("egfr", False, OBSERVED)],
                  [Condition("egfr", True, OBSERVED), Condition("crcl", 80, OBSERVED)]):
        assert (solve(phi, conds, version=RESIDUAL).decision
                == solve(phi, conds, version=MAXSMT).decision)


# ---------------------------------------------------------------- export
@requires_z3
def test_export_is_keyed_by_paper_symbol(phi):
    a = solve(phi, [Condition("egfr", False, OBSERVED),
                    Condition("crcl", None, UNRESOLVED)])
    e = a.to_paper()
    assert e["paper"] == PAPER_OF[MAXSMT] == "update"
    for sym in ("d", "gamma", "rho", "delta", "delta_E", "delta_I"):
        assert sym in e and "meaning" in e[sym] and "step" in e[sym]
    assert e["rho"]["step"] == "Step 4"


@requires_z3
def test_export_omits_symbols_absent_from_a_version(phi):
    """Omitted, not empty: 'not computed' must not read as 'none found'."""
    e = solve(phi, [Condition("egfr", False, OBSERVED)],
              version=RESIDUAL).to_paper()
    assert "delta_E" not in e and "delta_I" not in e
    assert e["rho"]["step"] == "Step 3"


@requires_z3
def test_verbalizer_reports_requirement_not_witness(phi):
    """A witness is arbitrary within the satisfying region."""
    a = solve(phi, [Condition("egfr", False, OBSERVED),
                    Condition("crcl", None, UNRESOLVED)])
    v = a.for_verbalizer(phi)["assumptions"]["crcl"]
    assert v["requirement"] == ">= 60"
    assert "witness" in v


@requires_z3
def test_describe_is_renderable(phi):
    out = solve(phi, [Condition("egfr", False, OBSERVED)]).describe()
    assert "update paper" in out and "Step 2" in out


def test_symbols_table_covers_every_artifact_field():
    """SYMBOLS is the code<->paper map; it must not fall behind Artifacts."""
    from dataclasses import fields
    from smt_core.maxsmt import Artifacts
    mapped = {s["field"] for s in SYMBOLS.values()}
    unmapped = {f.name for f in fields(Artifacts)} - mapped - {
        "status", "version"}
    assert not unmapped, f"artifact fields with no paper symbol: {unmapped}"


# ------------------------------------------- the bridge from stored pairs
def _pair_data() -> bool:
    from verdict.data import pair_root
    return (pair_root() / "cmsrc_out").exists()


needs_pairs = pytest.mark.skipif(not _pair_data(), reason="needs stage-1 pair data")


@requires_z3
@needs_pairs
def test_every_declared_variable_becomes_a_condition():
    """phi_t = phi(c_1..c_n): a declared variable left out would be free.

    A free variable lets the solver satisfy -phi while keeping every patient
    constraint, which silently empties delta_I and breaks the invariant that
    delta is never empty. This caught exactly that bug on real data.
    """
    from verdict.artifacts import conditions_from_pair, declared_variables
    import verdict
    for pair in verdict.pairs()[:3]:
        phi, conds = conditions_from_pair(pair)
        if not phi:
            continue
        missing = set(declared_variables(phi)) - {c.name for c in conds}
        assert not missing, f"{pair}: declared but not a condition: {missing}"


@requires_z3
@needs_pairs
def test_paper_invariants_hold_on_real_pairs():
    """The three invariants must survive real programs, not just toy ones."""
    from smt_core.maxsmt import ELIGIBLE, INELIGIBLE
    from verdict.artifacts import artifacts_for
    import verdict
    checked = 0
    for pair in verdict.pairs()[:5]:
        a = artifacts_for(pair)
        if a is None:
            continue
        checked += 1
        assert (a.delta_e == []) == (a.decision == ELIGIBLE), (pair, a)
        assert (a.delta_i == []) == (a.decision == INELIGIBLE), (pair, a)
        assert a.pivotal != [], f"{pair}: delta must never be empty"
    assert checked, "no pair yielded artifacts"


@requires_z3
@needs_pairs
def test_assumptions_are_chart_silent_conditions_only():
    """Never report an OBSERVED condition as an assumption."""
    from smt_core.maxsmt import OBSERVED
    from verdict.artifacts import artifacts_for, conditions_from_pair
    import verdict
    pair = verdict.pairs()[0]
    a = artifacts_for(pair)
    if a is None:
        pytest.skip("no artifacts for this pair")
    _phi, conds = conditions_from_pair(pair)
    observed = {c.name for c in conds if c.status == OBSERVED}
    assert not (set(a.assumptions) & observed), \
        "an observed condition was reported as an assumption"


@needs_pairs
def test_explain_still_works_without_a_solver():
    """The audit trail must never depend on the artifacts being available."""
    import verdict
    pair = verdict.pairs()[0]
    plain = verdict.explain(pair, artifacts=False)
    assert "audit trail" in plain
    assert "accountability artifacts" not in plain
