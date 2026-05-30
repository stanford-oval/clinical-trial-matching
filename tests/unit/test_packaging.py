"""Packaging claims must be true: pip verifies none of them."""
from __future__ import annotations

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_declared_packages_exist():
    for pat in CFG["tool"]["setuptools"]["packages"]["find"]["include"]:
        name = pat.rstrip("*")
        if name:
            assert (ROOT / name).is_dir(), f"declared but absent: {pat}"


def test_declared_py_modules_exist():
    for mod in CFG["tool"]["setuptools"].get("py-modules", []):
        assert (ROOT / f"{mod}.py").exists(), f"py-module missing: {mod}"


def test_console_scripts_point_at_real_modules():
    for script, target in CFG["project"]["scripts"].items():
        mod = target.split(":")[0]
        assert ((ROOT / (mod.replace(".", "/") + ".py")).exists()
                or (ROOT / mod.replace(".", "/") / "__init__.py").exists()), \
            f"console script {script} -> {target} resolves to nothing"


def test_env_example_has_no_secrets_or_internal_hosts():
    import re
    txt = (ROOT / ".env.example").read_text()
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", txt)
    for host in re.findall(r"https?://([^/\"\s]+)", txt):
        assert host.startswith(("YOUR-", "localhost", "127.0.0.1")), host


def test_env_is_gitignored():
    gi = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in gi and "!.env.example" in gi


def test_license_present_and_matches_pyproject():
    """No licence file means all rights reserved, even in a public repo."""
    lic = ROOT / "LICENSE"
    assert lic.exists()
    txt = lic.read_text()
    assert "Apache License" in txt
    assert "Version 2.0, January 2004" in txt
    assert "Grant of Patent License" in txt          # why Apache over MIT/BSD
    assert (ROOT / "NOTICE").exists()
    assert "Apache-2.0" in str(CFG["project"].get("license", ""))
