"""Where pair data lives.

Shared by the `verdict` package and the `verdict` command so the two cannot
disagree about which pairs exist.
"""
from __future__ import annotations

import os
import pathlib
import warnings
from typing import Iterator


def pair_root() -> pathlib.Path:
    """Directory of per-pair stage-1 artifacts.

    Resolution order:
      1. $VERDICT_PAIR_DATA
      2. <repo>/data/pairs           the tool's own location
      3. <repo>/experiments/53_v2_full   historical, kept so existing
                                          checkouts keep working

    The tool does not otherwise read anything under experiments/; that tree is
    self-contained research code with its own entry points.
    """
    if os.environ.get("VERDICT_PAIR_DATA"):
        return pathlib.Path(os.environ["VERDICT_PAIR_DATA"])
    root = pathlib.Path(os.environ.get(
        "VERDICT_ROOT", pathlib.Path(__file__).resolve().parents[1]))
    preferred = root / "data" / "pairs"
    if (preferred / "cmsrc_out").exists():
        return preferred
    return root / "experiments" / "53_v2_full"


def pair_data_available() -> bool:
    """Whether pair artifacts are actually present.

    Callers that would otherwise report an empty result should use this to
    tell "no pairs matched" apart from "the data was never installed".
    """
    return (pair_root() / "cmsrc_out").exists()


def iter_pairs() -> Iterator[str]:
    """Yield every available pair id, as '<patient>__<NCT>'."""
    base = pair_root() / "cmsrc_out"
    if not base.exists():
        # Silence here reads as "no pairs exist" when the truth is "the data
        # is not installed". Say which, once, instead of returning empty.
        warnings.warn(
            "no pair data found at " + str(base) + " -- returning no pairs. "
            "Set $VERDICT_PAIR_DATA, or install pair artifacts under "
            "data/pairs/cmsrc_out (see docs/DATA.md).",
            stacklevel=2)
        return
    for pdir in sorted(base.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith("_"):
            continue
        for f in sorted(pdir.glob("*__full.json")):
            yield f'{pdir.name}__{f.name.split("__")[0]}'
