#!/usr/bin/env python3
# scripts/export_solver_status_from_list_to_match.py
"""
Aggregate final SMT solver status per patient, using the list_to_match output
from compose_trial_eval.py to define the ordering and subcohorts.

Inputs (per patient):
  - list_to_match/{patient_id}.json         ← from compose_trial_eval.py
      {
        "patient_id": "...",
        "canonical_trials": [
          {
            "canonical_nct_id": "NCT00000369",
            "status": "survivor" | "mixed" | "eliminated",
            "label": "all_satisfied" | "unsatisfied_inclusion" | "explicit_contradiction",
            "sub_nct_ids": ["NCT00000369a", "NCT00000369b", ...],
            "subcohorts": [
              {
                "trial_id": 123,
                "nct_id": "NCT00000369a",
                "status": "survivor" | "eliminated",
                "label": "all_satisfied" | "unsatisfied_inclusion" | "explicit_contradiction",
                "rank": 1
              },
              ...
            ]
          },
          ...
        ]
      }

  - match_out/{patient_id}/{canonical_nct_id}__full.json
      ← from match_patient_to_trial.py

Output (per patient):
  - solver_status/{patient_id}__solver_status.json

      {
        "patient_id": "...",
        "canonical_trials": [
          {
            "canonical_nct_id": "NCT00000369",
            "status": "survivor",
            "label": "all_satisfied",
            "sub_nct_ids": [...],
            "subcohorts": [...],
            "rank": 1,      # canonical rank (order from list_to_match)
            "solver": {
              "trial_id": "NCT00000369",
              "eligible": true | false | null,
              "inclusion_status": "sat" | "unsat" | "unknown" | "error" | null,
              "exclusion_status": "sat" | "unsat" | "unknown" | "error" | null,
              "inclusion_unsat_assertions": [...],
              "exclusion_unsat_assertions": [...],
              "inclusion_unsat_core": [...],
              "exclusion_unsat_core": [...],
              "error": "...optional, if __full.json missing or unreadable..."
            }
          },
          ...
        ]
      }

We preserve:
  • the exact canonical trial order from list_to_match
  • all subcohorts for each trial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_default(o):
    import pathlib as _p, datetime as _d, re as _r
    if isinstance(o, set):
        try:
            return sorted(o)
        except TypeError:
            return list(o)
    if isinstance(o, (_p.Path,)):
        return str(o)
    if isinstance(o, (_d.date, _d.datetime)):
        return o.isoformat()
    if isinstance(o, _r.Pattern):
        return o.pattern
    return str(o)


# ────────────────────────────────────────────────────────────────────────
# Loading helpers
# ────────────────────────────────────────────────────────────────────────

def load_list_to_match(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"list_to_match file not found: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from {path}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object at top-level in {path}, got {type(obj)}")
    if "patient_id" not in obj or "canonical_trials" not in obj:
        raise ValueError(
            f"{path} is missing required keys ('patient_id', 'canonical_trials')"
        )
    if not isinstance(obj["canonical_trials"], list):
        raise ValueError(
            f"'canonical_trials' must be a list in {path}"
        )
    return obj


def load_full_solver_result(
    match_out_root: Path,
    patient_id: str,
    canonical_nct_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Load match_out/{patient_id}/{canonical_nct_id}__full.json
    (as written by match_patient_to_trial.py).

    Returns None if the file is missing or unreadable.
    """
    patient_dir = match_out_root / patient_id
    full_path = patient_dir / f"{canonical_nct_id}__full.json"
    if not full_path.exists():
        return None
    try:
        return json.loads(full_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────
# Solver summary extraction
# ────────────────────────────────────────────────────────────────────────

def build_solver_summary(
    canonical_nct_id: str,
    full: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract a compact view of the solver result for this patient/trial pair.

    We assume 'full' has the structure written by match_patient_to_trial.py:

      {
        "trial_id": "NCT...",
        "patient_id": "...",
        "eligible": true/false/null,
        "inclusion": {
          "side": "inclusion",
          "sat_like": ...,
          "summary": {
            "status": "sat" | "unsat" | "unknown" | "error" | null,
            "unsat_assertions": [...],
            "unsat_core": [...]
          },
          ...
        },
        "exclusion": { ... same shape ... }
      }
    """
    solver: Dict[str, Any] = {
        "trial_id": canonical_nct_id,
        "eligible": None,
        "inclusion_status": None,
        "exclusion_status": None,
        "inclusion_unsat_assertions": [],
        "exclusion_unsat_assertions": [],
        "inclusion_unsat_core": [],
        "exclusion_unsat_core": [],
    }

    if full is None:
        solver["error"] = "no __full.json found for this (patient, canonical_nct_id) pair"
        return solver

    # Top-level eligibility
    solver["eligible"] = full.get("eligible")

    # Inclusion side
    inc = full.get("inclusion") or {}
    inc_summary = inc.get("summary") or {}
    solver["inclusion_status"] = inc_summary.get("status")
    solver["inclusion_unsat_assertions"] = inc_summary.get("unsat_assertions", []) or []
    solver["inclusion_unsat_core"] = inc_summary.get("unsat_core", []) or []

    # Exclusion side
    exc = full.get("exclusion") or {}
    exc_summary = exc.get("summary") or {}
    solver["exclusion_status"] = exc_summary.get("status")
    solver["exclusion_unsat_assertions"] = exc_summary.get("unsat_assertions", []) or []
    solver["exclusion_unsat_core"] = exc_summary.get("unsat_core", []) or []

    return solver


# ────────────────────────────────────────────────────────────────────────
# Per-patient aggregation
# ────────────────────────────────────────────────────────────────────────

def aggregate_for_patient(
    list_root: Path,
    match_out_root: Path,
    patient_id: str,
) -> Dict[str, Any]:
    """
    For one patient:
      - read list_to_match/{patient}.json
      - for each canonical trial (in order), attach solver summary
      - keep all subcohorts
      - assign canonical rank = position in canonical_trials list (1-based)
    """
    ltm_path = list_root / f"{patient_id}.json"
    ltm = load_list_to_match(ltm_path)

    canonical_trials = ltm.get("canonical_trials") or []
    out_trials: List[Dict[str, Any]] = []

    for idx, ct in enumerate(canonical_trials, start=1):
        # Copy input canonical trial entry so we don't mutate the original object
        canonical_nct_id = str(ct.get("canonical_nct_id", "")).strip()
        new_ct = dict(ct)  # shallow copy is enough; subcohorts list is kept as-is

        # canonical-level rank derived from ordering in list_to_match
        new_ct["rank"] = idx

        # Load solver result for (patient, canonical_nct_id)
        full = load_full_solver_result(match_out_root, patient_id, canonical_nct_id)
        solver_summary = build_solver_summary(canonical_nct_id, full)
        new_ct["solver"] = solver_summary

        out_trials.append(new_ct)

    return {
        "patient_id": ltm.get("patient_id", patient_id),
        "canonical_trials": out_trials,
    }


def write_patient_summary(out_root: Path, patient_summary: Dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    patient_id = str(patient_summary.get("patient_id", "UNKNOWN"))
    out_path = out_root / f"{patient_id}__solver_status.json"
    out_path.write_text(
        json.dumps(patient_summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export final SMT solver status as JSON per patient, "
            "using list_to_match outputs to define trial order and subcohorts."
        )
    )
    ap.add_argument(
        "--list-root",
        default="../irsrc/ops/list_to_match",
        help="Directory containing {patient_id}.json from compose_trial_eval.py (default: list_to_match).",
    )
    ap.add_argument(
        "--match-out-root",
        default="match_out",
        help="Root directory where match_patient_to_trial.py wrote per-pair __full.json files (default: match_out).",
    )
    ap.add_argument(
        "--out-root",
        default="solver_status",
        help="Output directory for per-patient solver status JSON (default: solver_status).",
    )
    ap.add_argument(
        "--patient",
        default=None,
        help="Run for a single patient_id. If omitted, process all *.json under --list-root.",
    )

    args = ap.parse_args(argv)

    list_root = Path(args.list_root).resolve()
    match_out_root = Path(args.match_out_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not list_root.exists():
        print(f"[error] list-root directory does not exist: {list_root}", file=sys.stderr)
        sys.exit(2)

    patient_ids: List[str]
    if args.patient:
        patient_ids = [args.patient]
    else:
        # Discover all patient files under list_root/*.json
        patient_ids = []
        for p in sorted(list_root.glob("*.json")):
            patient_ids.append(p.stem)

    if not patient_ids:
        print(f"[warn] no patient JSONs found under {list_root}", file=sys.stderr)
        return

    for pid in patient_ids:
        try:
            summary = aggregate_for_patient(list_root, match_out_root, pid)
        except Exception as e:
            print(f"[error] failed for patient {pid}: {e}", file=sys.stderr)
            continue
        write_patient_summary(out_root, summary)


if __name__ == "__main__":
    main()
