"""rho must reach a reader as something they can act on.

The raw artifact is an SMT fact -- `crcl = 60.0` under MAXSMT, `crcl >= 60`
under RESIDUAL. Neither says what was taken for granted, that the chart did
not supply it, or what to check. These tests pin the rendering, and pin the
two ways it could state something false.
"""
import pytest

from smt_core.maxsmt import (Condition, solve, humanize, OBSERVED, UNRESOLVED,
                             RESIDUAL, MAXSMT)

PHI = ["(declare-fun crcl () Real)", "(declare-fun age () Int)",
       "(declare-fun on_warfarin () Bool)",
       "(assert (>= crcl 60))", "(assert (>= age 18))",
       "(assert (not on_warfarin))"]
CONDS = [Condition("age", 58, OBSERVED),
         Condition("crcl", status=UNRESOLVED),
         Condition("on_warfarin", status=UNRESOLVED)]
LABELS = {"crcl": "creatinine clearance", "on_warfarin": "warfarin use"}

z3 = pytest.importorskip("z3", reason="solver unavailable")


@pytest.mark.parametrize("version", [RESIDUAL, MAXSMT])
def test_report_never_states_a_witness_as_a_finding(version):
    """'creatinine clearance 60' would invent a lab result."""
    a = solve(PHI, CONDS, version=version)
    for rec in a.assumptions_report(PHI, labels=LABELS):
        if rec["condition"] != "crcl":
            continue
        assert rec["requirement"] == ">= 60"
        assert "at least 60" in rec["statement"]
        # the bare number must never be the claim
        assert "clearance 60" not in rec["statement"]
        assert "= 60" not in rec["statement"]


def test_boolean_polarity_is_not_inverted():
    """`(not on_warfarin)` means the patient was assumed NOT on warfarin."""
    a = solve(PHI, CONDS, version=MAXSMT)
    rec = next(r for r in a.assumptions_report(PHI, labels=LABELS)
               if r["condition"] == "on_warfarin")
    assert "absent" in rec["statement"]
    assert "is present" not in rec["statement"]


def test_residual_does_not_guess_polarity():
    """RESIDUAL stores no witness, so the polarity is genuinely unknown."""
    a = solve(PHI, CONDS, version=RESIDUAL)
    rec = next(r for r in a.assumptions_report(PHI, labels=LABELS)
               if r["condition"] == "on_warfarin")
    assert "not established" in rec["statement"]
    assert "absent" not in rec["statement"] and "present" not in rec["statement"]


@pytest.mark.parametrize("version", [RESIDUAL, MAXSMT])
def test_every_assumption_says_it_was_not_observed(version):
    a = solve(PHI, CONDS, version=version)
    recs = a.assumptions_report(PHI, labels=LABELS)
    assert recs
    for rec in recs:
        assert rec["basis"] == "Not found in the chart."
        assert rec["action"]


def test_pivotal_assumptions_are_marked_and_sorted_first():
    a = solve(PHI, CONDS, version=MAXSMT)
    text = a.render_assumptions(PHI, labels=LABELS)
    lines = [l for l in text.splitlines() if l[:1] in "!-"]
    if any(l.startswith("!") for l in lines):
        assert lines[0].startswith("!"), text
        assert "decided the outcome" in text


def test_humanize_strips_encoding_noise():
    assert humanize("patient_age_value_recorded_now_in_years") == "age"
    assert humanize("__THRESH__::patient_crcl::gt::60") == "crcl"
    assert "finding of" not in humanize(
        "patient_has_finding_of_angina_at_rest_inthehistory")


def test_screenresult_carries_records_not_witnesses():
    import pipeline
    r = pipeline.ScreenResult(nct_id="NCT1", rank=1, retrieval_label="x")
    assert isinstance(r.assumptions, list)
    assert "No assumptions" in r.assumptions_text()
    a = solve(PHI, CONDS, version=MAXSMT)
    r.assumptions = a.assumptions_report(PHI, labels=LABELS)
    txt = r.assumptions_text()
    assert "Not found in the chart." in txt
