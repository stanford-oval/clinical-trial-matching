#!/usr/bin/env python3
# Batch IR-slice all SMT2 files in build/ir_normalized -> build/slice_ir using smt_ir_slicer.py
# Outputs:
#   A) main slice (build/slice_ir):
#       - Inclusion files: KEEP STRICT-tagged constraints
#       - Exclusion files: KEEP everything EXCEPT 'always-satisfiable' tag(s)
#   B) assumed slice (build/slice_ir):
#       - KEEP ONLY constraints tagged as "OTHER_REQUIREMENTS[...]"
#   C) noslice-pruned slice (build/noslice_ir)  [NEW]:
#       - KEEP EVERYTHING (including untagged) EXCEPT 'always-satisfiable' tag(s)
#       - Still uses slicer, so decl/define closure is computed (not a raw filter)
#
# Manifests + _index.json are written for BOTH outputs.

from __future__ import annotations
from pathlib import Path
import sys
import json

# ── Config (adjust if you move things)
# Build paths come from smt_core.buildroot so every stage of the chain and the
# matcher agree on one tree. This line previously read
#     ROOT = Path("../../../TrialGPT-SMT")
# which pointed outside the repository at one machine's layout.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling stage modules
from smt_core.buildroot import build_root as _build_root

ROOT = _build_root().parent


IN_DIR = ROOT / "build" / "ir_normalized"

# Existing outputs
OUT_DIR = ROOT / "build" / "slice_ir"
MANIFESTS = True
MANIFEST_DIR = OUT_DIR / "_manifests"

# NEW output: "noslice but pruned"
NOSLICE_OUT_DIR = ROOT / "build" / "noslice_ir"
NOSLICE_MANIFEST_DIR = NOSLICE_OUT_DIR / "_manifests"

INCLUDE_AUX = True
SKIP_IF_FRESH = True

# Optional renames/suffixes for outputs
RENAME_SUFFIX_MAIN = ""               # e.g., "_ir"
RENAME_SUFFIX_ASSUMED = ".assumed"    # produces foo.assumed.smt2

# Tag policies
STRICT_TOKEN = "PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE"

# Anything that means "always satisfiable if you can take an action"; exclude
EXCLUDE_ALWAYS = [
    "NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION",  # your existing token
    # add synonyms if they exist in your IR
]

# Assumed-satisfiable constraints that we keep in a SEPARATE file for elimination-by-contradiction logic
ASSUMED_SUBSTRING = "OTHER_REQUIREMENTS"
ASSUMED_FOR_EXCLUSION = False  # do NOT emit assumed slices for exclusion files

# ── Import the slicer (param-only API) from your irsrc folder
SLICER_DIR = ROOT / "irsrc"
sys.path.insert(0, str(SLICER_DIR))
from smt_ir_slicer import slice_ir_paths  # supports include/exclude substrings


def is_exclusion_file(p: Path) -> bool:
    n = p.name.lower()
    return ("exclusion" in n) or ("_exclusion_" in n) or n.endswith("_exclusion_program.smt2")


def needs_run(in_path: Path, out_path: Path) -> bool:
    if not SKIP_IF_FRESH:
        return True
    if not out_path.exists():
        return True
    return out_path.stat().st_mtime < in_path.stat().st_mtime


def _read_manifest_or_blank(man_path: Path | None) -> dict:
    if not man_path:
        return {
            "kept_assert_count": None, "kept_aux_assert_count": None,
            "kept_decl_count": None, "kept_define_count": None,
            "missing_declarations": [], "used_symbols": []
        }
    try:
        if man_path.exists():
            return json.loads(man_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "kept_assert_count": None, "kept_aux_assert_count": None,
        "kept_decl_count": None, "kept_define_count": None,
        "missing_declarations": [], "used_symbols": []
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOSLICE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFESTS:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        NOSLICE_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    smt_files = sorted(IN_DIR.glob("*.smt2"))
    if not smt_files:
        print(f"[info] No .smt2 files under {IN_DIR}")
        return

    for smt in smt_files:
        # -----------------------
        # A) MAIN SLICE (existing)
        # -----------------------
        out_main_name = smt.stem + RENAME_SUFFIX_MAIN + smt.suffix
        out_main_path = OUT_DIR / out_main_name
        manifest_main = (MANIFEST_DIR / (smt.stem + RENAME_SUFFIX_MAIN + ".json")) if MANIFESTS else None

        run_main = needs_run(smt, out_main_path)
        if run_main:
            if is_exclusion_file(smt):
                # Exclusion: KEEP everything except the always-satisfiable tag(s)
                _, man_main = slice_ir_paths(
                    in_path=str(smt),
                    out_path=str(out_main_path),
                    manifest_path=str(manifest_main) if manifest_main else None,
                    include_auxiliary=INCLUDE_AUX,
                    include_tag_substrings=None,
                    exclude_tag_substrings=EXCLUDE_ALWAYS,
                )
                tag_policy_main = {"mode": "exclusion", "include": None, "exclude": EXCLUDE_ALWAYS}
            else:
                # Inclusion: KEEP STRICT only
                _, man_main = slice_ir_paths(
                    in_path=str(smt),
                    out_path=str(out_main_path),
                    manifest_path=str(manifest_main) if manifest_main else None,
                    include_auxiliary=INCLUDE_AUX,
                    include_tag_substrings=[STRICT_TOKEN],
                )
                tag_policy_main = {"mode": "inclusion", "include": [STRICT_TOKEN], "exclude": []}

            print(f"[ok][main] {smt.name} -> {out_main_path.name} "
                  f"(kept={man_main.get('kept_assert_count')}, aux={man_main.get('kept_aux_assert_count')}, "
                  f"decls={man_main.get('kept_decl_count')}, defines={man_main.get('kept_define_count')}, "
                  f"missing_decls={len(man_main.get('missing_declarations', []))})")
        else:
            man_main = _read_manifest_or_blank(manifest_main)
            tag_policy_main = {"mode": "skipped", "include": None, "exclude": None}
            print(f"[skip][main] up-to-date: {smt.name}")

        # --------------------------
        # B) ASSUMED SLICE (existing)
        # --------------------------
        out_assumed_name = smt.stem + RENAME_SUFFIX_ASSUMED + smt.suffix
        out_assumed_path = OUT_DIR / out_assumed_name
        manifest_assumed = (MANIFEST_DIR / (smt.stem + RENAME_SUFFIX_ASSUMED + ".json")) if MANIFESTS else None

        run_assumed = (not is_exclusion_file(smt) or ASSUMED_FOR_EXCLUSION) and needs_run(smt, out_assumed_path)
        if not is_exclusion_file(smt) or ASSUMED_FOR_EXCLUSION:
            if run_assumed:
                _, man_assumed = slice_ir_paths(
                    in_path=str(smt),
                    out_path=str(out_assumed_path),
                    manifest_path=str(manifest_assumed) if manifest_assumed else None,
                    include_auxiliary=INCLUDE_AUX,
                    include_tag_substrings=[ASSUMED_SUBSTRING],
                )
                print(f"[ok][assumed] {smt.name} -> {out_assumed_path.name} "
                      f"(kept={man_assumed.get('kept_assert_count')}, aux={man_assumed.get('kept_aux_assert_count')}, "
                      f"decls={man_assumed.get('kept_decl_count')}, defines={man_assumed.get('kept_define_count')}, "
                      f"missing_decls={len(man_assumed.get('missing_declarations', []))})")
            else:
                man_assumed = _read_manifest_or_blank(manifest_assumed)
                print(f"[skip][assumed] up-to-date: {smt.name}")
        else:
            man_assumed = {
                "kept_assert_count": 0, "kept_aux_assert_count": 0,
                "kept_decl_count": 0, "kept_define_count": 0,
                "missing_declarations": [], "used_symbols": []
            }
            print(f"[note][assumed] skipped for exclusion file: {smt.name}")

        # ----------------------------------------------
        # C) NOSLICE-PRUNED SLICE (NEW): build/noslice_ir
        # ----------------------------------------------
        noslice_out_path = NOSLICE_OUT_DIR / smt.name
        noslice_manifest = (NOSLICE_MANIFEST_DIR / (smt.stem + ".json")) if MANIFESTS else None

        run_noslice = needs_run(smt, noslice_out_path)
        if run_noslice:
            # Keep everything EXCEPT EXCLUDE_ALWAYS, and KEEP UNTAGGED too:
            # require_named=False => untagged asserts pass the policy
            _, man_noslice = slice_ir_paths(
                in_path=str(smt),
                out_path=str(noslice_out_path),
                manifest_path=str(noslice_manifest) if noslice_manifest else None,
                include_auxiliary=INCLUDE_AUX,
                include_tag_substrings=None,
                exclude_tag_substrings=EXCLUDE_ALWAYS,
                require_named=False,
            )
            print(f"[ok][noslice_pruned] {smt.name} -> {noslice_out_path.name} "
                  f"(kept={man_noslice.get('kept_assert_count')}, aux={man_noslice.get('kept_aux_assert_count')}, "
                  f"decls={man_noslice.get('kept_decl_count')}, defines={man_noslice.get('kept_define_count')}, "
                  f"missing_decls={len(man_noslice.get('missing_declarations', []))})")
        else:
            man_noslice = _read_manifest_or_blank(noslice_manifest)
            print(f"[skip][noslice_pruned] up-to-date: {smt.name}")

        # Collect summary info
        results.append({
            "file": smt.name,
            "main": {
                "out": out_main_path.name,
                "kept": man_main.get("kept_assert_count"),
                "aux": man_main.get("kept_aux_assert_count"),
                "decls": man_main.get("kept_decl_count"),
                "defines": man_main.get("kept_define_count"),
                "used_symbols": man_main.get("used_symbols", []),
                "missing_declarations": man_main.get("missing_declarations", []),
                "tag_policy": tag_policy_main,
            },
            "assumed": {
                "out": out_assumed_path.name,
                "kept": man_assumed.get("kept_assert_count"),
                "aux": man_assumed.get("kept_aux_assert_count"),
                "decls": man_assumed.get("kept_decl_count"),
                "defines": man_assumed.get("kept_define_count"),
                "used_symbols": man_assumed.get("used_symbols", []),
                "missing_declarations": man_assumed.get("missing_declarations", []),
                "tag_policy": {"mode": "assumed_only", "include": [ASSUMED_SUBSTRING], "exclude": []},
            },
            "noslice_pruned": {
                "out": noslice_out_path.name,
                "kept": man_noslice.get("kept_assert_count"),
                "aux": man_noslice.get("kept_aux_assert_count"),
                "decls": man_noslice.get("kept_decl_count"),
                "defines": man_noslice.get("kept_define_count"),
                "used_symbols": man_noslice.get("used_symbols", []),
                "missing_declarations": man_noslice.get("missing_declarations", []),
                "tag_policy": {"mode": "noslice_pruned", "include": None, "exclude": EXCLUDE_ALWAYS, "require_named": False},
            },
        })

    # Optional summary indexes
    if MANIFESTS:
        (OUT_DIR / "_index.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (NOSLICE_OUT_DIR / "_index.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(
        f"\nWrote:\n"
        f"  - main slices   -> {OUT_DIR}\n"
        f"  - assumed slices-> {OUT_DIR}\n"
        f"  - noslice_pruned-> {NOSLICE_OUT_DIR}\n"
    )


if __name__ == "__main__":
    main()
