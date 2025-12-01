#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

# Root of all patient kits.
# Default: "out_sigir_kits_threeway" under the current working directory.
# You can override via:  SIGIR_ROOT=/full/path/to/out_sigir_kits_threeway
ROOT = Path(os.environ.get("SIGIR_ROOT", "out_sigir_kits_threeway")).resolve()

BASELINE_BUNDLE = "A_baseline_ranked"
ORIG_SURV_BUNDLE = "O_original_survivors"
DEST_BUNDLE = "F_baseline_not_ours_survivor"

NCT_RE = re.compile(r"(NCT\d{8})")


def extract_trial_id_from_name(name: str) -> Optional[str]:
    """
    Extract NCT######## from a directory name like 'rank1_NCT01682096'.
    Returns None if nothing is found.
    """
    m = NCT_RE.search(name)
    return m.group(1) if m else None


def find_rank_dirs(bundle_dir: Path) -> Dict[str, Path]:
    """
    Recursively find all directories under bundle_dir whose name looks like rankX_NCT...
    Returns: { trial_id (NCT########): path_to_rank_dir }
    """
    trials: Dict[str, Path] = {}
    if not bundle_dir.exists():
        return trials

    for d in bundle_dir.rglob("*"):
        if not d.is_dir():
            continue
        if not d.name.startswith("rank"):
            continue
        trial_id = extract_trial_id_from_name(d.name)
        if not trial_id:
            continue
        trials.setdefault(trial_id, d)
    return trials


def safe_copy_dir_tree(src: Path, dst: Path) -> None:
    """
    Copy a directory tree from src to dst, but be forgiving:
    - create parent directories as needed
    - skip files that are missing or broken symlinks
    """
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)

        # Copy files
        for fname in files:
            s = root_path / fname
            d = target_root / fname
            try:
                if s.is_symlink():
                    # replicate the symlink (even if dangling)
                    link_target = os.readlink(s)
                    if d.exists():
                        d.unlink()
                    d.symlink_to(link_target)
                else:
                    shutil.copy2(s, d)
            except FileNotFoundError:
                # Source disappeared or is a broken symlink – just skip
                print(f"      ! skipping missing file: {s}")
            except OSError as e:
                # Any other weird file issue – log and continue
                print(f"      ! skipping {s}: {e}")


def main() -> None:
    print(f"Using ROOT = {ROOT}")
    if not ROOT.exists():
        print(f"ROOT does not exist: {ROOT}")
        return

    # Each immediate subdirectory under ROOT is treated as a patient directory
    for patient_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        patient_id = patient_dir.name

        baseline_dir = patient_dir / BASELINE_BUNDLE
        orig_surv_dir = patient_dir / ORIG_SURV_BUNDLE
        dest_dir = patient_dir / DEST_BUNDLE

        if not baseline_dir.exists():
            print(f"[{patient_id}] Skipping: {BASELINE_BUNDLE} not found")
            continue

        baseline_trials = find_rank_dirs(baseline_dir)
        orig_surv_trials = find_rank_dirs(orig_surv_dir)

        baseline_ids = set(baseline_trials.keys())
        orig_ids = set(orig_surv_trials.keys())

        to_copy_ids = sorted(baseline_ids - orig_ids)

        if not to_copy_ids:
            print(
                f"[{patient_id}] Nothing to copy "
                f"(baseline={len(baseline_ids)}, original_survivors={len(orig_ids)})"
            )
            continue

        print(
            f"[{patient_id}] baseline={len(baseline_ids)}, "
            f"original_survivors={len(orig_ids)}, to_copy={len(to_copy_ids)}"
        )

        for tid in to_copy_ids:
            src_rank_dir = baseline_trials[tid]
            rel = src_rank_dir.relative_to(baseline_dir)
            dst_rank_dir = dest_dir / rel

            print(f"  -> copying {tid}:")
            print(f"       {src_rank_dir}")
            print(f"       -> {dst_rank_dir}")

            safe_copy_dir_tree(src_rank_dir, dst_rank_dir)


if __name__ == "__main__":
    main()
