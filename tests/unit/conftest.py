"""Shared fixtures for the fast unit suite.

These tests need no services, no API keys and no corpus. Anything that does
belongs in the service-level scripts at tests/*.py, which are run explicitly.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def phi():
    """The paper's worked example: egfr must hold and crcl must be >= 60."""
    return ["(declare-const |egfr| Bool)",
            "(declare-const |crcl| Real)",
            "(assert |egfr|)",
            "(assert (>= |crcl| 60))"]


@pytest.fixture
def pair_data_available() -> bool:
    from verdict_cli import pair_root
    return (pair_root() / "cmsrc_out").exists()
