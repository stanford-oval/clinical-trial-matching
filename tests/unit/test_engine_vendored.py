"""The vendored matcher must be runnable from anywhere.

Upstream resolved its roots against the current working directory, so it only
worked when invoked from inside the matcher directory. These tests pin the
fixes that made it importable and locatable as a package.
"""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_engine_imports():
    import verdict.engine.match_patient_to_trial as m
    assert hasattr(m, "Config") and hasattr(m, "main")


def test_prompts_resolve_from_unrelated_cwd(tmp_path):
    """The bug that would hit a real user: prompts resolved against cwd."""
    code = (
        "from verdict.engine.match_patient_to_trial import Config;"
        "ps = Config().prompt_sources();"
        "print(sum(1 for v in ps.values() if not v.exists()))"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("0"), "prompts missing: " + out.stdout


def test_config_roots_are_absolute():
    from verdict.engine.match_patient_to_trial import Config
    c = Config()
    for name in ("data_root", "build_root", "project_root", "prompt_root"):
        p = getattr(c, name)
        assert p.is_absolute(), f"{name} is cwd-relative: {p}"


def test_run_without_endpoint_explains_itself():
    """A missing endpoint must produce guidance, not a traceback."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_ENDPOINT", "OPENAI_MODEL")}
    out = subprocess.run([sys.executable, "verdict_cli.py", "run", "NCT1", "P1"],
                         cwd=REPO, capture_output=True, text=True, env=env)
    assert out.returncode != 0
    assert "OPENAI_ENDPOINT" in (out.stdout + out.stderr)
    assert "Traceback" not in out.stderr


def test_engine_imports_without_optional_extras(tmp_path):
    """A base install has no azure/dspy; importing the engine must still work.

    This is what CI caught: the heavy clients were imported at module load, so
    `verdict run` raised ModuleNotFoundError instead of reporting the missing
    endpoint. Kept as a test because a local dev venv has the extras and will
    not notice the regression.
    """
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "BLOCK = {'azure', 'dspy', 'nltk'}\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('blocked for test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]))
    # Prove the blocker engaged before trusting the result -- otherwise this
    # test passes vacuously in any venv that simply has the extras installed.
    code = (
        "try:\n"
        "    import azure\n"
        "    raise SystemExit('BLOCKER-INACTIVE')\n"
        "except ImportError:\n"
        "    pass\n"
        "from verdict.engine.match_patient_to_trial import Config\n"
        "print('OK', sum(1 for v in Config().prompt_sources().values() if not v.exists()))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, env=env)
    assert "BLOCKER-INACTIVE" not in (out.stdout + out.stderr), \
        "the import blocker did not engage; test would be vacuous"
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("OK 0"), out.stdout
