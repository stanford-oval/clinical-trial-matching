"""The compile chain must agree with the matcher on one build tree.

Before this was wired, the compiler honoured $SATIR_BUILD (cwd-relative),
the matcher $VERDICT_BUILD, and two chain stages carried a hard-coded path
to a directory outside the repository. Each stage wrote where the next one
did not look, so no end-to-end run was possible from a clone.
"""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "db_indexer" / "trial_side" / "scripts"


def test_no_hardcoded_external_root():
    """No chain stage may point outside the repository."""
    offenders = []
    for f in SCRIPTS.glob("*.py"):
        for line in f.read_text(errors="ignore").splitlines():
            if line.lstrip().startswith("#"):
                continue          # comments documenting the old line are fine
            if 'Path("../../../' in line or "Path('../../../" in line:
                offenders.append(f"{f.name}:{line.strip()[:50]}")
    assert not offenders, f"hard-coded external root in: {offenders}"


def test_compiler_and_matcher_share_one_build_root():
    from smt_core.buildroot import build_root
    from verdict.engine.match_patient_to_trial import Config as MatcherCfg
    assert MatcherCfg().build_root == build_root()


def test_env_override_is_honoured_by_both(tmp_path):
    code = ("from smt_core.buildroot import build_root;"
            "from verdict.engine.match_patient_to_trial import Config;"
            "print(build_root() == Config().build_root, build_root())")
    env = dict(os.environ, PYTHONPATH=str(REPO), VERDICT_BUILD=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("True"), out.stdout
    assert str(tmp_path) in out.stdout


def test_legacy_env_still_works(tmp_path):
    from smt_core.buildroot import build_root
    old = {k: os.environ.get(k) for k in ("VERDICT_BUILD", "SATIR_BUILD")}
    try:
        os.environ.pop("VERDICT_BUILD", None)
        os.environ["SATIR_BUILD"] = str(tmp_path)
        assert build_root() == tmp_path.resolve()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_stage_modules_import_from_any_cwd(tmp_path):
    """They import sibling modules; that only worked from inside scripts/."""
    code = (
        "import importlib.util, pathlib, sys\n"
        f"for n in ['batch_slice_ir','batch_link_qualifiers']:\n"
        f"    p = pathlib.Path(r'{SCRIPTS}') / (n + '.py')\n"
        "    spec = importlib.util.spec_from_file_location(n, p)\n"
        "    m = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(m)\n"
        "print('OK')\n")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_compile_without_endpoint_explains_itself():
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_ENDPOINT", "OPENAI_MODEL")}
    out = subprocess.run([sys.executable, "verdict_cli.py", "compile", "NCT1"],
                         cwd=REPO, capture_output=True, text=True, env=env)
    assert out.returncode != 0
    assert "OPENAI_ENDPOINT" in (out.stdout + out.stderr)
    assert "Traceback" not in out.stderr


def test_empty_build_tree_is_not_reported_as_success(tmp_path):
    """A chain that produced nothing must fail, not print a phantom path."""
    env = dict(os.environ, PYTHONPATH=str(REPO), VERDICT_BUILD=str(tmp_path),
               OPENAI_ENDPOINT="https://example.invalid")
    out = subprocess.run(
        [sys.executable, "verdict_cli.py", "compile", "NCT1", "--stages", "normalize"],
        cwd=REPO, capture_output=True, text=True, env=env)
    assert out.returncode != 0, out.stdout
    # the point is that nothing is reported as produced -- whether the chain
    # stopped for a missing extra or ran and yielded nothing
    assert "done ->" not in out.stdout, out.stdout
    blob = out.stdout + out.stderr
    assert ("produced no program" in blob) or ("compile extra" in blob), blob
