"""Shared skip conditions for the unit suite (imported, not a conftest)."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def z3_usable() -> bool:
    """z3-solver importable AND functional.

    A bare `import z3` can bind to an unrelated namespace package, so probe
    for the API rather than trusting the import.
    """
    try:
        import z3
        return all(hasattr(z3, a) for a in ("parse_smt2_string", "Optimize"))
    except Exception:
        return False


requires_z3 = pytest.mark.skipif(not z3_usable(),
                                 reason="needs a working z3-solver")
