#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_strict_fallback_ir_v2.py

Export "best available" SMT IR programs from an orchestrator work dir, using a strict-compliance
fallback chain, with optional per-stage meaning-fix attempts for declare-const JSON issues.

Fallback order per (effective_trial_id, side):
  stage3_logic/merged_ir
  -> stage2_repair/merged_ir
  -> stage1_polarity/merged_ir
  -> original --ir-dir

Meaning-fix behavior (optional, enabled via --attempt-meaning-fix):
  For EACH candidate in the fallback chain (stage3 -> stage2 -> stage1 -> input):
    - if candidate fails strict compliance AND the errors are *only* declare_const_*,
      then run meaning_enrich_irs.py ON THAT candidate once, validate strict again.
    - if strict-ok after meaning fix, choose it.
    - otherwise fall back to the next stage candidate.

Final fallback behavior (always):
  If NO strict-ok candidate exists anywhere in the chain (even after meaning-fix attempts),
  we STILL export the original input IR file (noncompliant), so the output dir contains all pairs.

Outputs:
  --out-dir/*.smt2                            (canonical filenames)
  --out-dir/export_report.jsonl               (per-pair decision trace)
  --out-dir/export_summary.json               (aggregate counts)
  --out-dir/_meaning_fix/*                    (meaning-fix workspaces, if enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Reuse the exact strict compliance logic + helpers from your orchestrator
from trial_compiler.ir_finalizer.orchestrators.orchestrate_ir_fixes import (  # type: ignore
    STRICT_GUARDRAILS_DEFAULT,
    canonical_index,
    ensure_dir,
    find_latest_work_dir,
    is_nonempty_smt,
    validate_smt_file,
)

Pair = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export IRs with strict-compliance fallback (stage3->stage2->stage1->input), "
            "optional per-stage meaning-fix for declare-const JSON issues, and always export input "
            "as final fallback even if noncompliant."
        )
    )
    p.add_argument("--ir-dir", default="../build/ir", help="Original canonical IR dir (NCT*_program.smt2).")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Orchestrator work dir (ir_orchestrated_...). If omitted, auto-picks newest under parent of --ir-dir.",
    )
    p.add_argument("--out-dir", required=True, help="Where to copy outputs (canonical filenames).")
    p.add_argument("--skip-assert-checks", action="store_true", help="Match orchestrator mode: disable assert/:named checks.")
    p.add_argument("--clean-out-dir", action="store_true", help="Remove existing exports in --out-dir before running.")

    # Meaning-fix (LLM) optional pass
    p.add_argument(
        "--attempt-meaning-fix",
        default=True,
        action="store_true",
        help=(
            "If set, run meaning_enrich_irs.py once per stage candidate whose strict failure is "
            "ONLY declare_const_* errors (attempt-before-fallback semantics)."
        ),
    )
    p.add_argument(
        "--snapshot-dir",
        default="../subcohort_results",
        help="Snapshot dir needed by meaning_enrich_irs.py (only used if --attempt-meaning-fix).",
    )
    p.add_argument(
        "--meaning-max-workers",
        type=int,
        default=16,
        help="Max workers for meaning_enrich_irs.py (only if enabled).",
    )
    return p.parse_args()


def strict_ok_with_errs(p: Path, *, strict_cfg) -> Tuple[bool, List[str]]:
    if not p.exists():
        return False, ["missing"]
    if not is_nonempty_smt(p):
        return False, ["empty_or_whitespace"]
    ok, errs = validate_smt_file(p, cfg=strict_cfg)
    return ok, [str(e) for e in errs]


def declare_const_only_failure(errs: List[str]) -> bool:
    """
    True iff errors exist and ALL errors are declare_const_*.
    This matches "sheerly fails for meaning fix issues".
    """
    if not errs:
        return False
    return all(str(e).startswith("declare_const_") for e in errs)


def write_allowlist(path: Path, trials: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(set(trials)):
            f.write(t + "\n")


def run_cmd(cmd: List[str], *, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def try_meaning_fix_once(
    *,
    eff: str,
    side: str,
    source_label: str,
    cand_path: Path,
    snapshot_dir: Path,
    meaning_py: Path,
    scripts_dir: Path,
    strict_cfg,
    meaning_root: Path,
    meaning_max_workers: int,
) -> Tuple[Path, bool, List[str]]:
    """
    Runs meaning_enrich_irs.py once on cand_path (copied into a todo dir).
    Returns: (out_file_path, strict_ok, errors)
    """
    run_dir = meaning_root / f"{eff}_{side}__from_{source_label}"
    todo = run_dir / "ir_todo"
    raw_out = run_dir / "raw_out"
    logs = run_dir / "logs"
    mbench = run_dir / "mbench"
    ensure_dir(todo)
    ensure_dir(raw_out)
    ensure_dir(logs)
    ensure_dir(mbench)

    # Clear todo + raw_out so we only consider current run outputs
    for p in todo.glob("NCT*_program.smt2"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in raw_out.glob("NCT*_program_meaning_enriched.smt2"):
        try:
            p.unlink()
        except Exception:
            pass

    # Copy candidate into todo (must be canonical name)
    dst_in = todo / cand_path.name
    shutil.copy2(cand_path, dst_in)

    allow = run_dir / "allowlist.txt"
    write_allowlist(allow, [eff])

    # Require endpoint env (meaning script uses it)
    if not os.environ.get("OPENAI_ENDPOINT", "").strip():
        raise EnvironmentError("OPENAI_ENDPOINT is required for --attempt-meaning-fix (meaning_enrich_irs.py).")

    summary = run_dir / "summary_run.jsonl"
    cmd = [
        sys.executable,
        str(meaning_py),
        "--ir-dir",
        str(todo),
        "--snapshot-dir",
        str(snapshot_dir),
        "--out-ir-dir",
        str(raw_out),
        "--summary-jsonl",
        str(summary),
        "--log-dir",
        str(logs),
        "--mbench-dir",
        str(mbench),
        "--max-workers",
        str(meaning_max_workers),
        "--trial-allowlist",
        str(allow),
        "--side",
        "both",
    ]
    run_cmd(cmd, cwd=scripts_dir)

    out_file = raw_out / f"{eff}_{side}_program_meaning_enriched.smt2"
    ok, errs = strict_ok_with_errs(out_file, strict_cfg=strict_cfg)
    return out_file, ok, errs


def main() -> None:
    args = parse_args()

    strict_cfg = STRICT_GUARDRAILS_DEFAULT
    if args.skip_assert_checks:
        strict_cfg = replace(strict_cfg, check_asserts_and_named=False)

    in_ir_dir = Path(args.ir_dir).resolve()
    if not in_ir_dir.exists():
        raise FileNotFoundError(f"--ir-dir does not exist: {in_ir_dir}")

    # Resolve work_dir
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
    else:
        latest = find_latest_work_dir(in_ir_dir.parent)
        if latest is None:
            raise FileNotFoundError(f"Could not auto-find ir_orchestrated_* under: {in_ir_dir.parent}")
        work_dir = latest.resolve()

    # Stage merged dirs
    st3 = work_dir / "stage3_logic" / "merged_ir"
    st2 = work_dir / "stage2_repair" / "merged_ir"
    st1 = work_dir / "stage1_polarity" / "merged_ir"

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    report_path = out_dir / "export_report.jsonl"
    summary_path = out_dir / "export_summary.json"

    # Clean outputs if requested
    if args.clean_out_dir:
        for p in out_dir.glob("NCT*_program.smt2"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in (report_path, summary_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        # Remove meaning_fix workspace if present
        mf = out_dir / "_meaning_fix"
        if mf.exists():
            shutil.rmtree(mf, ignore_errors=True)

    if report_path.exists():
        report_path.unlink()

    # Meaning-fix setup
    scripts_dir = THIS_DIR
    meaning_py = scripts_dir / "meaning_enrich_irs.py"
    snapshot_dir = Path(args.snapshot_dir).resolve()
    meaning_root = out_dir / "_meaning_fix"

    if args.attempt_meaning_fix:
        if not meaning_py.exists():
            raise FileNotFoundError(f"Missing meaning script next to exporter: {meaning_py}")
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"--snapshot-dir does not exist: {snapshot_dir}")
        ensure_dir(meaning_root)

    # Pairs from original input
    input_idx: Dict[Pair, Path] = canonical_index(in_ir_dir)

    sources: List[Tuple[str, Path]] = [
        ("stage3", st3),
        ("stage2", st2),
        ("stage1", st1),
        ("input", in_ir_dir),
    ]

    chosen_counts: Counter[str] = Counter()

    exported_any = 0
    exported_strict_ok = 0
    exported_noncompliant_fallback = 0
    meaning_fixed_count = 0

    no_strict_ok_pairs: List[Pair] = []

    for (eff, side), input_path in sorted(input_idx.items()):
        filename = f"{eff}_{side}_program.smt2"

        tried: List[dict] = []
        chosen_label: Optional[str] = None
        chosen_path: Optional[Path] = None
        chosen_is_strict_ok: bool = False

        # Attempt meaning fix once per source label (stage3/stage2/stage1/input)
        meaning_fix_attempted_sources: Set[str] = set()

        for label, base_dir in sources:
            cand = input_path if label == "input" else (base_dir / filename)

            ok, errs = strict_ok_with_errs(cand, strict_cfg=strict_cfg)
            tried.append(
                {
                    "label": label,
                    "path": str(cand),
                    "strict_ok": bool(ok),
                    "errors": errs[:8],
                }
            )

            if ok:
                chosen_label = label
                chosen_path = cand
                chosen_is_strict_ok = True
                break

            # Attempt meaning-fix for THIS candidate before falling back further
            if (
                args.attempt_meaning_fix
                and label not in meaning_fix_attempted_sources
                and cand.exists()
                and is_nonempty_smt(cand)
                and declare_const_only_failure(errs)
            ):
                meaning_fix_attempted_sources.add(label)
                print(f"[MEANING-FIX] attempting for {eff}_{side} using candidate from {label}", flush=True)

                out_file, ok2, errs2 = try_meaning_fix_once(
                    eff=eff,
                    side=side,
                    source_label=label,
                    cand_path=cand,
                    snapshot_dir=snapshot_dir,
                    meaning_py=meaning_py,
                    scripts_dir=scripts_dir,
                    strict_cfg=strict_cfg,
                    meaning_root=meaning_root,
                    meaning_max_workers=args.meaning_max_workers,
                )

                tried.append(
                    {
                        "label": f"meaning_fix({label})",
                        "path": str(out_file),
                        "strict_ok": bool(ok2),
                        "errors": errs2[:8],
                    }
                )

                if ok2:
                    chosen_label = f"meaning_fix({label})"
                    chosen_path = out_file
                    chosen_is_strict_ok = True
                    meaning_fixed_count += 1
                    break
                else:
                    print(
                        f"[MEANING-FIX] still strict-invalid for {eff}_{side} from {label}. First errors={errs2[:5]}",
                        flush=True,
                    )
                    # continue falling back

        # Final fallback: export input even if noncompliant
        if chosen_path is None:
            chosen_label = "input_noncompliant_fallback"
            chosen_path = input_path
            chosen_is_strict_ok = False
            no_strict_ok_pairs.append((eff, side))

        # Copy chosen into output using canonical filename
        dst = out_dir / filename
        shutil.copy2(chosen_path, dst)
        exported_any += 1
        chosen_counts[chosen_label] += 1

        if chosen_is_strict_ok:
            exported_strict_ok += 1
        else:
            exported_noncompliant_fallback += 1

        row = {
            "effective_trial_id": eff,
            "side": side,
            "chosen_from": chosen_label,
            "chosen_path": str(chosen_path),
            "exported": True,
            "exported_strict_ok": bool(chosen_is_strict_ok),
            "out_path": str(dst),
            "tried": tried,
        }
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "work_dir": str(work_dir),
        "ir_dir": str(in_ir_dir),
        "out_dir": str(out_dir),
        "skip_assert_checks": bool(args.skip_assert_checks),
        "attempt_meaning_fix": bool(args.attempt_meaning_fix),
        "snapshot_dir": str(snapshot_dir) if args.attempt_meaning_fix else None,
        "total_pairs_in_input": len(input_idx),
        "exported_any": exported_any,
        "exported_strict_ok": exported_strict_ok,
        "exported_noncompliant_fallback": exported_noncompliant_fallback,
        "meaning_fixed_count": meaning_fixed_count,
        "no_strict_ok_pairs": len(no_strict_ok_pairs),
        "chosen_from_counts": dict(chosen_counts),
        "no_strict_ok_pair_list": [{"effective_trial_id": eff, "side": side} for (eff, side) in no_strict_ok_pairs],
        "report_jsonl": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[EXPORT DONE]")
    print(f"  work_dir: {work_dir}")
    print(f"  out_dir:  {out_dir}")
    print(f"  exported_any: {exported_any} / {len(input_idx)}")
    print(f"  exported_strict_ok: {exported_strict_ok} / {len(input_idx)}")
    print(f"  exported_noncompliant_fallback: {exported_noncompliant_fallback}")
    print(f"  meaning_fixed_count: {meaning_fixed_count}")
    print(f"  no_strict_ok_pairs: {len(no_strict_ok_pairs)}")
    print(f"  chosen_from_counts: {dict(chosen_counts)}")
    print(f"  report:  {report_path}")
    print(f"  summary: {summary_path}")
    if args.attempt_meaning_fix:
        print(f"  meaning_fix_workspace: {meaning_root}")
    print("", flush=True)


if __name__ == "__main__":
    main()
