#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge ALL procedure facts (for every sigir-20xxx bundle) into the
final inclusion/exclusion fact files for sigir-20141.

Sources (per-procedure bundles):
  patient_procedure_build/patient_facts_export/<pid>/inclusion/canonical.jsonl
  patient_procedure_build/patient_facts_export/<pid>/exclusion/canonical.jsonl

Destinations (single real patient sigir-20141):
  patient_build_inclusion/patient_facts_export_inclusion/sigir-20141/inclusion/canonical.final.jsonl
  patient_build_exclusion/patient_facts_export_exclusion/sigir-20141/exclusion/canonical.final.jsonl

Deduplication:
  by (entity_variable_name,
      start_time_in_hours, end_time_in_hours,
      start_time_inclusive, end_time_inclusive)

So you can re-run the script without duplicating rows.
"""

from pathlib import Path
import json

# --- paths ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

PROC_EXPORT_ROOT = ROOT / "patient_procedure_build" / "patient_facts_export"

INCL_DEST = (
    ROOT
    / "patient_build_inclusion"
    / "patient_facts_export_inclusion"
    / "sigir-20141"
    / "inclusion"
    / "canonical.final.jsonl"
)

EXCL_DEST = (
    ROOT
    / "patient_build_exclusion"
    / "patient_facts_export_exclusion"
    / "sigir-20141"
    / "exclusion"
    / "canonical.final.jsonl"
)


# --- helpers -------------------------------------------------------------------

def dedupe_key(obj):
    return (
        obj.get("entity_variable_name"),
        obj.get("start_time_in_hours"),
        obj.get("end_time_in_hours"),
        bool(obj.get("start_time_inclusive")),
        bool(obj.get("end_time_inclusive")),
    )


def load_seen(path: Path) -> set:
    """Load existing dest file and collect dedupe keys."""
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seen.add(dedupe_key(obj))
    return seen


def append_new(src: Path, dest: Path, seen: set, label: str) -> int:
    """Append rows from src into dest if they are not already in `seen`."""
    if not src.exists():
        print(f"[{label}] no source file, skip: {src}")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    new_rows = []

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = dedupe_key(obj)
            if k in seen:
                continue
            seen.add(k)
            new_rows.append(obj)

    if not new_rows:
        print(f"[{label}] nothing new to append")
        return 0

    # Append, making sure there is exactly one newline at the end
    with dest.open("ab+") as f:
        f.seek(0, 2)
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        for obj in new_rows:
            f.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")

    print(f"[{label}] appended {len(new_rows)} rows → {dest}")
    return len(new_rows)


# --- main ----------------------------------------------------------------------

def main():
    print(f"[info] repo root: {ROOT}")
    print(f"[info] procedure export root: {PROC_EXPORT_ROOT}")
    print(f"[info] dest inclusion: {INCL_DEST}")
    print(f"[info] dest exclusion: {EXCL_DEST}")

    if not PROC_EXPORT_ROOT.is_dir():
        raise SystemExit(f"Missing source root: {PROC_EXPORT_ROOT}")

    # existing rows in the final files (if they already exist)
    seen_incl = load_seen(INCL_DEST)
    seen_excl = load_seen(EXCL_DEST)

    total_incl = 0
    total_excl = 0

    # loop over ALL procedure patient bundles (sigir-20142, sigir-20143, ...)
    for pdir in sorted(d for d in PROC_EXPORT_ROOT.iterdir() if d.is_dir()):
        pid = pdir.name
        src_incl = pdir / "inclusion" / "canonical.jsonl"
        src_excl = pdir / "exclusion" / "canonical.jsonl"

        print(f"\n===== procedure bundle: {pid} =====")
        total_incl += append_new(src_incl, INCL_DEST, seen_incl, f"{pid} inclusion")
        total_excl += append_new(src_excl, EXCL_DEST, seen_excl, f"{pid} exclusion")

    print("\n[done]")
    print(f"  total new inclusion rows: {total_incl}")
    print(f"  total new exclusion rows: {total_excl}")
    print(f"  final inclusion file: {INCL_DEST}")
    print(f"  final exclusion file: {EXCL_DEST}")


if __name__ == "__main__":
    main()
