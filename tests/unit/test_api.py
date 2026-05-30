"""The two public APIs: satir and verdict."""
from __future__ import annotations

import subprocess
import sys

import pytest

import pathlib

import satir
import verdict

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- satir
def test_satir_exposes_the_pipeline():
    for name in ("config", "compile_trial", "compile_patient", "index",
                 "retrieve", "match"):
        assert name in satir.__all__ and callable(getattr(satir, name))


def test_satir_config_resolves():
    assert type(satir.config()).__name__ == "SatIRConfig"


def test_satir_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        satir.definitely_not_a_function


def test_satir_does_not_eagerly_import_heavy_backends():
    """`import satir` must stay cheap; backends load only when used."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, satir;"
         "print([m for m in ('torch','elasticsearch','matplotlib','dspy','azure')"
         " if m in sys.modules])"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"eager imports: {out.stdout}"


# ---------------------------------------------------------------- verdict
def test_verdict_lists_its_variants():
    s = verdict.systems()
    assert "verdict" in s and "lm-only" in s
    assert all(isinstance(v, str) and v for v in s.values())


def test_verdict_rejects_unknown_system():
    with pytest.raises(KeyError, match="unknown system"):
        verdict.match("a__b", system="not-a-system")


def test_verdict_defaults_to_strict(pair_data_available):
    """A pair that cannot be loaded must raise, not return 'ineligible'.

    This is the harmful direction for a trial matcher: a missing file must
    never read as a decision that the patient does not qualify.
    """
    with pytest.raises(verdict.MissingPairData):
        verdict.match("definitely__NCT0")


def test_verdict_lenient_mode_keeps_the_paper_sentinel(pair_data_available):
    if not pair_data_available:
        pytest.skip("no pair data")
    d = verdict.match("definitely__NCT0", strict=False)
    assert d.decision == "ineligible"
    assert d.is_missing_data and d.reasoning == verdict.NO_DATA


def test_verdict_match_and_explain_agree(pair_data_available):
    if not pair_data_available:
        pytest.skip("no pair data")
    pair = verdict.pairs()[0]
    d = verdict.match(pair)
    assert d.decision in ("eligible", "ineligible")
    assert d.decision.upper() in verdict.explain(pair)


# ---------------------------------------------------------------- registry
def test_the_papers_matchers_are_registered_in_order():
    """Adding a registry must not change which systems ship, or their order."""
    assert list(verdict.systems()) == [
        "verdict", "smt-only", "atoms", "lm-only", "hybrid"]


def test_third_party_matcher_can_be_registered():
    from matchers.schema import Decision

    @verdict.register("t-test-matcher", description="registered in a test")
    def _m(pair_id):
        return Decision(pair_id, "t-test-matcher", "eligible", "because", [])
    try:
        assert "t-test-matcher" in verdict.systems()
        d = verdict.match("any__pair", system="t-test-matcher")
        assert d.decision == "eligible" and d.reasoning == "because"
    finally:
        verdict.unregister("t-test-matcher")
    assert "t-test-matcher" not in verdict.systems()


def test_registering_over_a_builtin_is_refused_by_default():
    """Silently shadowing 'verdict' would make two people's results differ."""
    with pytest.raises(ValueError, match="already registered"):
        verdict.register("verdict", lambda p: None)


def test_override_is_allowed_when_explicit():
    from matchers.schema import Decision
    original = verdict.systems()["hybrid"]
    try:
        verdict.register("hybrid", lambda p: Decision(p, "x", "eligible", "", []),
                         description="replaced", override=True)
        assert verdict.systems()["hybrid"] == "replaced"
    finally:
        verdict.unregister("hybrid")
        verdict.systems()                      # re-registers the built-in
        assert verdict.systems()["hybrid"] == original


def test_cli_and_api_share_one_system_list():
    """The list used to be duplicated in verdict_cli.py; it must not drift."""
    import verdict_cli
    assert set(verdict_cli._systems()) == set(verdict.systems())


# ---------------------------------------------------------------- pipeline
def test_pipeline_is_the_only_module_touching_both_systems():
    """SatIR and VERDICT stay independent; pipeline joins them."""
    import ast
    src = ast.parse((ROOT_DIR / "pipeline.py").read_text())
    names = {n.names[0].name.split(".")[0] for n in ast.walk(src)
             if isinstance(n, ast.Import)}
    names |= {n.module.split(".")[0] for n in ast.walk(src)
              if isinstance(n, ast.ImportFrom) and n.module}
    assert {"satir", "verdict"} <= names


def test_screen_result_never_confuses_undecided_with_ineligible():
    """decision=None must not read as 'not eligible'."""
    from pipeline import ScreenResult
    undecided = ScreenResult(nct_id="NCT0", rank=1, retrieval_label="x")
    assert undecided.decision is None
    assert undecided.eligible is None          # not False
    assert not undecided.evaluated

    no = ScreenResult(nct_id="NCT1", rank=2, retrieval_label="x",
                      decision="ineligible")
    assert no.eligible is False and no.evaluated
