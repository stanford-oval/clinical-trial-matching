"""Where compiled trial programs live.

Every stage of the compile chain and the matcher that consumes it must agree
on one directory, or each silently writes to a tree the next stage never
reads. Before this module they did not: the compiler honoured $SATIR_BUILD
(defaulting to a *cwd-relative* "build"), the matcher honoured $VERDICT_BUILD
(defaulting to <repo>/build), and two chain stages carried a hard-coded
`Path("../../../TrialGPT-SMT")` pointing outside the repository entirely.
"""
from __future__ import annotations

import os
import pathlib

#: Preferred variable. $SATIR_BUILD is accepted as a deprecated alias so
#: existing local setups keep working.
ENV = "VERDICT_BUILD"
LEGACY_ENV = "SATIR_BUILD"

REPO = pathlib.Path(__file__).resolve().parents[1]


def build_root() -> pathlib.Path:
    """The build tree, as an absolute path.

    $VERDICT_BUILD, else $SATIR_BUILD, else <repo>/build.
    """
    for var in (ENV, LEGACY_ENV):
        v = os.environ.get(var)
        if v:
            return pathlib.Path(v).expanduser().resolve()
    return REPO / "build"


#: Stage outputs, in pipeline order. Each stage reads the previous one.
STAGES = ("ir", "ir_normalized", "noslice_ir", "noslice_ir_linked")


def stage_dir(name: str) -> pathlib.Path:
    if name not in STAGES and name not in ("symtab", "linkmap", "canon",
                                           "slice_ir", "slice_ir_linked"):
        raise ValueError(f"unknown build stage: {name}")
    return build_root() / name


def describe() -> str:
    """Human-readable resolution, for error messages."""
    src = next((v for v in (ENV, LEGACY_ENV) if os.environ.get(v)), None)
    origin = f"${src}" if src else "default <repo>/build"
    return f"{build_root()}  (from {origin})"
