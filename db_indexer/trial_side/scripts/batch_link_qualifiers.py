#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys

# Build paths come from smt_core.buildroot so every stage of the chain and the
# matcher agree on one tree. This line previously read
#     ROOT = Path("../../../TrialGPT-SMT")
# which pointed outside the repository at one machine's layout.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling stage modules
from smt_core.buildroot import build_root as _build_root

ROOT = _build_root().parent


# Branch 1: link qualifiers AFTER slicing
IN_DIR_SLICE = ROOT / "build" / "slice_ir"
OUT_DIR_SLICE = ROOT / "build" / "slice_ir_linked"

# Branch 2: link qualifiers on "noslice_pruned" programs
IN_DIR_NOSLICE = ROOT / "build" / "noslice_ir"
OUT_DIR_NOSLICE = ROOT / "build" / "noslice_ir_linked"

# Import the library
sys.path.insert(0, str(ROOT / "irsrc"))
from smt_qualifier_linker import link_qualifiers_dir


def main() -> None:
    # ── Branch 1: sliced -> linked
    if IN_DIR_SLICE.exists():
        OUT_DIR_SLICE.mkdir(parents=True, exist_ok=True)
        link_qualifiers_dir(
            in_dir=str(IN_DIR_SLICE),
            out_dir=str(OUT_DIR_SLICE),
            glob="*.smt2",
            write_manifests=True,
            skip_if_fresh=True,
        )
        print(f"\n[done] wrote linked programs to {OUT_DIR_SLICE}")
    else:
        print(f"[warn] missing {IN_DIR_SLICE}; skip slice_ir linking")

    # ── Branch 2: noslice_pruned -> linked
    if IN_DIR_NOSLICE.exists():
        OUT_DIR_NOSLICE.mkdir(parents=True, exist_ok=True)
        link_qualifiers_dir(
            in_dir=str(IN_DIR_NOSLICE),
            out_dir=str(OUT_DIR_NOSLICE),
            glob="*.smt2",
            write_manifests=True,
            skip_if_fresh=True,
        )
        print(f"\n[done] wrote linked programs to {OUT_DIR_NOSLICE}")
    else:
        print(f"[warn] missing {IN_DIR_NOSLICE}; run the slice step first to create noslice_ir")


if __name__ == "__main__":
    main()
