#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patch_uncertain_can_exclusion_from_certain.py

Traverse UNCERTAIN jsonl.
If a diagnosis variable appears in CERTAIN jsonl (match on entity_variable_name),
then force UNCERTAIN row's can_be_used_for_exclusion = True.

Match is diagnosis-only:
  entity_variable_name startswith "patient_has_diagnosis_of_"

Normalization for matching (simplified per request):
  - lowercase only (no qualifier stripping, no timeframe token stripping)

When upgraded to True:
  - append exclusion_reason with "Upgraded due to certainty." (or set it if missing)

CLI:
  patient_id (positional)
  --uncertain-jsonl (required)
  --certain-jsonl   (required)
  --out (optional; default: <uncertain>.patched.jsonl)
  --inplace (optional)
  --backup  (optional; only with --inplace)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

CAN_EXC_KEY = "can_be_used_for_exclusion"
REASON_KEY = "exclusion_reason"
UPGRADE_SENTENCE = "Upgraded due to certainty."


def _boolish(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def _is_diag(name: str) -> bool:
    return (name or "").strip().lower().startswith("patient_has_diagnosis_of_")


def _normalize_name(name: str) -> str:
    # Per request: only lowercase/strip; entity_variable_name already clean.
    return (name or "").strip().lower()


def _get_entity_name(obj: Dict[str, Any]) -> Optional[str]:
    n = obj.get("entity_variable_name")
    if isinstance(n, str) and n.strip():
        return n.strip()
    return None


def _append_upgrade_reason(obj: Dict[str, Any]) -> None:
    """
    Ensure exclusion_reason includes the upgrade sentence.
    - If missing/empty: set to the sentence.
    - If present: append once (avoid duplicates).
    """
    cur = obj.get(REASON_KEY)
    if not isinstance(cur, str) or not cur.strip():
        obj[REASON_KEY] = UPGRADE_SENTENCE
        return

    # avoid double append
    if UPGRADE_SENTENCE.lower() in cur.lower():
        return

    sep = " " if cur.rstrip().endswith((".", "!", "?")) else ". "
    obj[REASON_KEY] = cur.rstrip() + sep + UPGRADE_SENTENCE


def load_certain_diag_set(certain_path: Path) -> Set[str]:
    """
    Collect normalized diagnosis entity_variable_name set from CERTAIN jsonl.
    """
    s: Set[str] = set()
    with certain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            n = _get_entity_name(obj)
            if not n:
                continue
            if _is_diag(n):
                s.add(_normalize_name(n))
    return s


def patch_uncertain(
    patient_id: str,
    uncertain_path: Path,
    certain_path: Path,
    *,
    out_path: Optional[Path] = None,
    inplace: bool = False,
    backup: bool = False,
) -> Tuple[int, int, int, Path]:
    """
    Returns: (total_lines, diag_lines_seen, diag_lines_upgraded, written_path)
    """
    if not uncertain_path.is_file():
        raise SystemExit(f"[err] uncertain jsonl not found: {uncertain_path}")
    if not certain_path.is_file():
        raise SystemExit(f"[err] certain jsonl not found: {certain_path}")

    certain_diag = load_certain_diag_set(certain_path)

    # decide output
    if inplace:
        tmp_out = uncertain_path.with_name(uncertain_path.name + ".tmp_patch")
        final_out = uncertain_path
    else:
        if out_path is None:
            final_out = uncertain_path.with_name(uncertain_path.stem + ".patched" + uncertain_path.suffix)
        else:
            final_out = out_path
        tmp_out = final_out

    # backup
    if inplace and backup:
        bak = uncertain_path.with_suffix(uncertain_path.suffix + ".bak")
        bak.parent.mkdir(parents=True, exist_ok=True)
        bak.write_bytes(uncertain_path.read_bytes())
        print(f"[{patient_id}] ✓ backup saved: {bak}")

    total = 0
    diag_seen = 0
    upgraded = 0

    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    with uncertain_path.open("r", encoding="utf-8") as fin, tmp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            n = _get_entity_name(obj)
            if n and _is_diag(n):
                diag_seen += 1
                if _normalize_name(n) in certain_diag:
                    # force True (even if missing/null/false)
                    if _boolish(obj.get(CAN_EXC_KEY)) is not True:
                        obj[CAN_EXC_KEY] = True
                        _append_upgrade_reason(obj)
                        upgraded += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if inplace:
        os.replace(tmp_out, final_out)

    print(
        f"[{patient_id}] total_lines={total}, diag_seen={diag_seen}, upgraded_to_true={upgraded}\n"
        f"[{patient_id}] uncertain={uncertain_path}\n"
        f"[{patient_id}] certain={certain_path}\n"
        f"[{patient_id}] wrote={final_out}"
    )
    return total, diag_seen, upgraded, final_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patch UNCERTAIN diagnosis can_be_used_for_exclusion using CERTAIN presence (diagnosis-only)."
    )
    ap.add_argument("patient_id", help="Patient id for logging (e.g., sigir-20141)")
    ap.add_argument("--uncertain-jsonl", required=True, help="Path to uncertain inferred jsonl (to be patched)")
    ap.add_argument("--certain-jsonl", required=True, help="Path to deterministic/certain jsonl (reference)")
    ap.add_argument("--out", default="", help="Output path (default: <uncertain>.patched.jsonl). Ignored if --inplace.")
    ap.add_argument("--inplace", action="store_true", help="Overwrite --uncertain-jsonl in place (safe replace).")
    ap.add_argument("--backup", action="store_true", help="When using --inplace, create .bak before overwrite.")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve() if args.out else None

    patch_uncertain(
        patient_id=args.patient_id,
        uncertain_path=Path(args.uncertain_jsonl).expanduser().resolve(),
        certain_path=Path(args.certain_jsonl).expanduser().resolve(),
        out_path=out_path,
        inplace=bool(args.inplace),
        backup=bool(args.backup),
    )


if __name__ == "__main__":
    main()
