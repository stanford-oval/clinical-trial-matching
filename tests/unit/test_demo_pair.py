"""The bundled demo must keep working: it is the first thing anyone runs.

It needs no corpus, no endpoint and no build tree, so unlike most of the
tool it can be exercised in CI end to end.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DEMO = REPO / "data" / "demo" / "pairs"
PAIR = "demo-patient-01__NCT00000000"

pytest.importorskip("z3", reason="solver unavailable")


def _env():
    return dict(os.environ, PYTHONPATH=str(REPO), VERDICT_PAIR_DATA=str(DEMO))


def test_demo_pair_is_present_and_declares_itself_synthetic():
    f = DEMO / "cmsrc_out" / "demo-patient-01" / "NCT00000000__full.json"
    assert f.exists(), "the demo pair is what makes the tool runnable from a clone"
    obj = json.loads(f.read_text())
    assert "Synthetic" in obj.get("_demo", ""), \
        "a fabricated patient must say so inside the artifact, not only in a README"


def test_cli_lists_the_demo_pair():
    out = subprocess.run([sys.executable, "verdict_cli.py", "list"], cwd=REPO,
                         capture_output=True, text=True, env=_env())
    assert out.returncode == 0, out.stderr
    assert PAIR in out.stdout


def test_the_two_systems_disagree():
    """The demo's whole point. If they ever agree, it teaches nothing."""
    got = {}
    for system in ("smt-only", "lm-only"):
        out = subprocess.run(
            [sys.executable, "verdict_cli.py", "match", PAIR, "--system", system],
            cwd=REPO, capture_output=True, text=True, env=_env())
        assert out.returncode == 0, out.stderr
        line = next(l for l in out.stdout.splitlines() if l.startswith("decision:"))
        got[system] = line.split(":", 1)[1].strip()
    assert got["smt-only"] == "INELIGIBLE", got
    assert got["lm-only"] == "ELIGIBLE", got


def test_the_assumption_names_the_requirement_not_a_value():
    from verdict.artifacts import artifacts_for, conditions_from_pair
    os.environ["VERDICT_PAIR_DATA"] = str(DEMO)
    phi, _ = conditions_from_pair(PAIR, "inclusion")
    text = artifacts_for(PAIR, side="inclusion").render_assumptions(
        phi, labels={"egfr_ml_min": "renal function (eGFR)"})
    assert "at least 60" in text
    assert "Not found in the chart." in text


def test_example_script_runs():
    out = subprocess.run([sys.executable, "examples/05_the_demo_pair.py"],
                         cwd=REPO, capture_output=True, text=True, env=_env())
    assert out.returncode == 0, out.stderr
    assert "The disagreement, located" in out.stdout
