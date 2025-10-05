#!/usr/bin/env python3
"""
# DEPRECATED: Use orchestrate.py instead (the simplified parallel orchestrator).
# This file is retained for reference but is not used in the standard pipeline.

fix_meaning_ir_strict_final.py

Meaning-only fixer for canonical IR directories (Stage0-like), modeled after orchestrate_ir_fixes.py:

- Detect pairs that fail strict compliance due to declare_const_* errors
- Run meaning_enrich_irs.py only for those pairs (materialized TODO ir-dir)
- Retry meaning fixing up to 2 times for remaining failing pairs
- Merge results as: base=input + overlay=strict-ok meaning outputs only
- Strict-invalid/empty outputs are quarantined and NEVER propagate
- Final output is written to a NEW directory (default: build/ir_all_final)
  with an additional backfill pass:
    if a declare-const inline JSON is missing/empty meaning, infer meaning from variable name.

Defaults:
  --ir-dir        <SATIR_ROOT>/build/ir_strict_final
  --snapshot-dir  <SATIR_ROOT>/subcohort_results
  --final-ir-dir  <SATIR_ROOT>/build/ir_all_final

Outputs:
  <work-dir>/stage0_meaning/{raw_out, merged_ir, quarantine_invalid, logs, mbench, summary.jsonl}
  and writes final IR -> --final-ir-dir (clearing old NCT*_program.smt2 first)

Assumes colocated:
  - orchestrate_ir_fixes.py  (for strict validation helpers)
  - meaning_enrich_irs.py    (the stage0 meaning script)

Usage:
  export OPENAI_ENDPOINT="..."
  python3 fix_meaning_ir_strict_final.py --yes
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

# ---- Import strict checks + helpers from orchestrator (must be colocated) ----
try:
    from trial_compiler.ir_finalizer.orchestrators.orchestrate_ir_fixes import (  # type: ignore
        STRICT_GUARDRAILS_DEFAULT,
        GuardrailConfig,
        validate_smt_file,
        is_nonempty_smt,
        strict_ok_file,
        ensure_dir,
        canonical_index,
        materialize_view_from_index,
        materialize_todo_ir_from_index,
        overlays_from_meaning_summary,
        overlays_from_meaning_raw_out,
        apply_overlays,
        sanitize_existing_raw_out,
        write_allowlist,
        append_jsonl,
        make_stage_paths,
        replace,
        # needed for meaning backfill
        RE_DECLARE_CONST_HDR,
        code_part_before_comment,
        declare_const_in_code,
        extract_first_json_object_from_line,
    )
except Exception as e:
    raise SystemExit(
        f"[FATAL] Failed to import from orchestrate_ir_fixes.py. "
        f"Place this script next to orchestrate_ir_fixes.py. Error: {e}"
    )


DEFAULT_IR_DIR = os.getenv("SATIR_BUILD", "build") + "/ir_under_repair/canonical"
DEFAULT_SNAPSHOT_DIR = os.getenv("SATIR_BUILD", "build") + "/../subcohort_results"
DEFAULT_FINAL_IR_DIR = os.getenv("SATIR_BUILD", "build") + "/ir_all_final"

MAX_MEANING_RETRIES = 2  # requested: retry 2 times (=> up to 3 total attempts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meaning-only strict fixer (Stage0-like) for canonical IR directories.")
    p.add_argument(
        "--ir-dir",
        default=DEFAULT_IR_DIR,
        help=f"Canonical IR directory containing NCT*_program.smt2. Default: {DEFAULT_IR_DIR}",
    )
    p.add_argument(
        "--snapshot-dir",
        default=DEFAULT_SNAPSHOT_DIR,
        help=f"Snapshot directory required by meaning_enrich_irs.py. Default: {DEFAULT_SNAPSHOT_DIR}",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Work directory for outputs. Default: <ir-dir>/../meaning_fixed_<timestamp>",
    )
    p.add_argument(
        "--final-ir-dir",
        default=DEFAULT_FINAL_IR_DIR,
        help=f"Write merged strict-ok IR files to this directory (default: {DEFAULT_FINAL_IR_DIR}).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    p.add_argument("--yes", action="store_true", help="Proceed without interactive prompt.")
    p.add_argument("--max-workers", type=int, default=16, help="Max workers for meaning_enrich_irs.py.")
    p.add_argument(
        "--decl-only",
        action="store_true",
        help="Only run meaning fixer when the file fails STRICTLY due to declare_const_* errors (no other errors).",
    )
    p.add_argument(
        "--skip-assert-checks",
        action="store_true",
        help="Disable checks requiring asserts and :named tags (same semantics as orchestrator).",
    )
    return p.parse_args()


def confirm_or_exit(prompt: str, *, yes: bool) -> None:
    if yes:
        print("[CONFIRM] --yes set; proceeding.", flush=True)
        return
    if not sys.stdin.isatty():
        raise SystemExit("[ABORT] Non-interactive stdin and --yes not set.")
    ans = input(prompt).strip().lower()
    if ans not in ("y", "yes"):
        raise SystemExit("[ABORT] User declined.")


def should_fix_meaning(
    smt_path: Path,
    *,
    strict_cfg: GuardrailConfig,
    decl_only: bool,
) -> Tuple[bool, List[str]]:
    """
    Returns (needs_meaning_fix, errs)

    Policy:
      - if strict-ok => no
      - else if any error startswith declare_const_ => yes
        (unless decl_only and there are non-decl errors)
    """
    if not is_nonempty_smt(smt_path):
        return (False, ["empty_or_missing"])

    ok, errs = validate_smt_file(smt_path, cfg=strict_cfg)
    if ok:
        return (False, [])
    has_decl = any(str(e).startswith("declare_const_") for e in errs)
    if not has_decl:
        return (False, errs)

    if decl_only:
        non_decl = [e for e in errs if not str(e).startswith("declare_const_")]
        if non_decl:
            return (False, errs)

    return (True, errs)


def run_cmd(cmd: List[str], *, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def clear_canonical_outputs(dir_path: Path) -> int:
    """Delete NCT*_program.smt2 files in dir_path, so output dir represents THIS run cleanly."""
    n = 0
    if not dir_path.exists():
        return 0
    for p in dir_path.glob("NCT*_program.smt2"):
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


# -------------------------
# Meaning backfill helpers
# -------------------------
QUAL_SPLIT_RE = re.compile(r"@@+")


def infer_meaning_from_var(var_name: str) -> str:
    """
    Heuristic: turn snake_case into a readable phrase, drop some boilerplate prefixes,
    and include qualifier suffix if present (foo@@documented -> "foo (documented)").
    """
    s = (var_name or "").strip()
    if not s:
        return ""

    parts = QUAL_SPLIT_RE.split(s)
    stem = parts[0]
    quals = parts[1:] if len(parts) > 1 else []

    for pref in ("patient_", "trial_", "subject_", "participant_"):
        if stem.startswith(pref):
            stem = stem[len(pref):]

    stem = stem.replace("__", "_").strip("_")
    words = [w for w in stem.split("_") if w]
    phrase = " ".join(words)

    if quals:
        q_words: List[str] = []
        for q in quals:
            q = (q or "").strip("_")
            q_words.extend([w for w in q.split("_") if w])
        if q_words:
            phrase = f"{phrase} ({' '.join(q_words)})" if phrase else f"({' '.join(q_words)})"

    return phrase if phrase else s


def backfill_missing_meaning_in_smt_text(text: str) -> Tuple[str, int]:
    """
    Return (new_text, num_filled).
    For each declare-const line with inline JSON:
      - if meaning missing/empty/non-string => set to inferred value
    """
    out_lines: List[str] = []
    filled = 0

    lines = text.splitlines()
    for line in lines:
        if not declare_const_in_code(line):
            out_lines.append(line)
            continue

        code = code_part_before_comment(line).rstrip()
        m = RE_DECLARE_CONST_HDR.match(code)
        if not m:
            out_lines.append(line)
            continue

        var_name = m.group(1)

        jb = extract_first_json_object_from_line(line)
        if not jb:
            out_lines.append(line)
            continue

        try:
            obj = json.loads(jb)
        except Exception:
            out_lines.append(line)
            continue

        if isinstance(obj, dict):
            meaning = obj.get("meaning", None)
            if (not isinstance(meaning, str)) or (not meaning.strip()):
                obj["meaning"] = infer_meaning_from_var(var_name)
                new_jb = json.dumps(obj, ensure_ascii=False)
                new_line = line.replace(jb, new_jb, 1)
                out_lines.append(new_line)
                filled += 1
                continue

        out_lines.append(line)

    new_text = "\n".join(out_lines)
    # preserve trailing newline behavior
    if text.endswith("\n"):
        new_text += "\n"
    return new_text, filled


def copy_ir_dir_with_meaning_backfill(src_dir: Path, dst_dir: Path) -> Tuple[int, int]:
    """
    Copy NCT*_program.smt2 from src_dir -> dst_dir, but rewrite files to backfill missing meaning.
    Returns (files_written, meanings_filled).
    """
    ensure_dir(dst_dir)
    files_written = 0
    meanings_filled = 0

    for src in src_dir.glob("NCT*_program.smt2"):
        txt = src.read_text(encoding="utf-8", errors="ignore")
        new_txt, filled = backfill_missing_meaning_in_smt_text(txt)
        (dst_dir / src.name).write_text(new_txt, encoding="utf-8")
        files_written += 1
        meanings_filled += filled

    return files_written, meanings_filled


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()

    strict_cfg = STRICT_GUARDRAILS_DEFAULT
    if args.skip_assert_checks:
        strict_cfg = replace(strict_cfg, check_asserts_and_named=False)

    if not os.environ.get("OPENAI_ENDPOINT", "").strip():
        raise EnvironmentError("OPENAI_ENDPOINT is required (used by meaning_enrich_irs.py).")

    scripts_dir = Path(__file__).resolve().parent
    meaning_py = scripts_dir / "meaning_enrich_irs.py"
    if not meaning_py.exists():
        raise FileNotFoundError(f"Missing meaning_enrich_irs.py next to this script: {meaning_py}")

    in_ir_dir = Path(args.ir_dir).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not in_ir_dir.exists():
        raise FileNotFoundError(f"--ir-dir does not exist: {in_ir_dir}")

    now_ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (in_ir_dir.parent / f"meaning_fixed_{now_ts}").resolve()
    ensure_dir(work_dir)

    st0 = make_stage_paths(work_dir, "stage0_meaning")

    print("[INFO] input_ir_dir:", in_ir_dir, flush=True)
    print("[INFO] snapshot_dir :", snapshot_dir, flush=True)
    print("[INFO] work_dir     :", work_dir, flush=True)
    print("[INFO] final_ir_dir :", args.final_ir_dir, flush=True)
    print("[INFO] strict_compliance=ON, assert_checks=" + ("OFF" if args.skip_assert_checks else "ON"), flush=True)
    print("[INFO] decl_only=" + ("ON" if args.decl_only else "OFF"), flush=True)
    print(f"[INFO] meaning_retry_budget=MAX_MEANING_RETRIES={MAX_MEANING_RETRIES} (total_attempts={1+MAX_MEANING_RETRIES})", flush=True)

    sanitize_existing_raw_out(
        stage_name="stage0_meaning",
        raw_out_dir=st0.raw_out_dir,
        kind="meaning",
        quarantine_dir=st0.quarantine_dir,
        strict_cfg=strict_cfg,
    )

    input_idx = canonical_index(in_ir_dir)

    # initial selection based on INPUT (not the merged)
    need_pairs: Set[Tuple[str, str]] = set()
    decl_err_examples: List[Tuple[str, str, str]] = []
    for pair, p in input_idx.items():
        need, errs = should_fix_meaning(p, strict_cfg=strict_cfg, decl_only=args.decl_only)
        if need:
            need_pairs.add(pair)
            decl_err_examples.append((pair[0], pair[1], str(errs[0]) if errs else "unknown"))

    trials = sorted({eff for (eff, _side) in need_pairs})

    print("\n[PLAN]")
    print(f"  total_pairs={len(input_idx)}")
    print(f"  need_meaning_fix_pairs={len(need_pairs)} (trials={len(trials)})")
    if decl_err_examples:
        print("  sample reasons:")
        for (eff, side, reason) in decl_err_examples[:10]:
            print(f"    - {eff} {side}: {reason}")

    if args.dry_run:
        print("\n[DRY RUN] exiting.", flush=True)
        return

    if not need_pairs:
        print("\n[NOOP] No declare-const strict failures detected; building merged_ir = input and exiting.", flush=True)
        materialize_view_from_index(input_idx, st0.merged_ir_dir)
    else:
        confirm_or_exit("Proceed with meaning_enrich_irs.py on these pairs? [y/N] ", yes=args.yes)

        remaining_pairs: Set[Tuple[str, str]] = set(need_pairs)

        for attempt in range(0, MAX_MEANING_RETRIES + 1):
            trials_attempt = sorted({eff for (eff, _side) in remaining_pairs})
            tag = "ATTEMPT0" if attempt == 0 else f"RETRY{attempt}"

            print(f"\n[STAGE0 {tag}] remaining_pairs={len(remaining_pairs)} trials={len(trials_attempt)}", flush=True)
            if not remaining_pairs:
                break

            # Materialize TODO from the ORIGINAL input idx (never from possibly bad outputs)
            todo = st0.todo_dir / f"todo_{now_ts}_{tag}"
            copied = materialize_todo_ir_from_index(idx=input_idx, pairs=remaining_pairs, todo_dir=todo)
            print(f"[STAGE0 {tag}] todo_ir={todo} copied={copied}", flush=True)

            allow = st0.stage_dir / f"allowlist_{now_ts}_{tag}.txt"
            write_allowlist(allow, trials_attempt)

            stage0_run_summary = st0.stage_dir / f"summary_run_{now_ts}_{tag}.jsonl"

            cmd = [
                sys.executable, str(meaning_py),
                "--ir-dir", str(todo),
                "--snapshot-dir", str(snapshot_dir),
                "--out-ir-dir", str(st0.raw_out_dir),
                "--summary-jsonl", str(stage0_run_summary),
                "--log-dir", str(st0.log_dir),
                "--mbench-dir", str(st0.mbench_dir),
                "--max-workers", str(args.max_workers),
                "--trial-allowlist", str(allow),
                "--side", "both",
            ]
            run_cmd(cmd, cwd=scripts_dir)

            append_jsonl(st0.summary_master_jsonl, stage0_run_summary)
            sanitize_existing_raw_out(
                stage_name="stage0_meaning",
                raw_out_dir=st0.raw_out_dir,
                kind="meaning",
                quarantine_dir=st0.quarantine_dir,
                strict_cfg=strict_cfg,
            )

            # Rebuild merged_ir each attempt: base=input, overlay=strict-ok meaning outputs
            materialize_view_from_index(input_idx, st0.merged_ir_dir)
            ov0 = overlays_from_meaning_summary(st0.summary_master_jsonl)
            ov0 += overlays_from_meaning_raw_out(st0.raw_out_dir)
            applied, missing_src, skipped_empty, skipped_invalid = apply_overlays(
                st0.merged_ir_dir,
                ov0,
                use_hardlinks=False,
                quarantine_dir=st0.quarantine_dir,
                strict_cfg=strict_cfg,
            )
            print(
                f"[MERGE {tag}] overlays={len(ov0)} applied={applied} missing_src={missing_src} "
                f"skipped_empty={skipped_empty} skipped_invalid={skipped_invalid}",
                flush=True,
            )

            # Recompute remaining based on merged result: if merged is strict-ok, we're done for that pair
            merged_idx = canonical_index(st0.merged_ir_dir)
            next_remaining: Set[Tuple[str, str]] = set()
            for pair in remaining_pairs:
                p = merged_idx.get(pair)
                if not p or not p.exists():
                    next_remaining.add(pair)
                    continue
                if strict_ok_file(p, strict_cfg=strict_cfg):
                    continue
                next_remaining.add(pair)

            newly_fixed = len(remaining_pairs) - len(next_remaining)
            remaining_pairs = next_remaining
            print(f"[STAGE0 {tag}] newly_fixed={newly_fixed} still_remaining={len(remaining_pairs)}", flush=True)

    # Final strict audit summary (on merged_ir)
    merged_idx = canonical_index(st0.merged_ir_dir)
    bad: List[Tuple[str, str, str]] = []
    for (eff, side), p in merged_idx.items():
        ok, errs = validate_smt_file(p, cfg=strict_cfg)
        if not ok:
            bad.append((eff, side, str(errs[0]) if errs else "unknown"))

    print("\n[AUDIT merged_ir]")
    print(f"  strict_fail_pairs={len(bad)}")
    if bad:
        for eff, side, reason in bad[:50]:
            print(f"    - {eff} {side}: {reason}")

    # Write to final dir with meaning backfill
    final_dir = Path(args.final_ir_dir).resolve()
    ensure_dir(final_dir)
    removed = clear_canonical_outputs(final_dir)
    files_written, meanings_filled = copy_ir_dir_with_meaning_backfill(st0.merged_ir_dir, final_dir)

    print(
        f"\n[DONE] Wrote final IR to: {final_dir} "
        f"(files={files_written}, removed_old={removed}, meanings_filled={meanings_filled})",
        flush=True,
    )

    print(f"\n[DONE] stage0_meaning outputs in: {st0.stage_dir}", flush=True)
    print(f"[DONE] merged_ir directory: {st0.merged_ir_dir}", flush=True)


if __name__ == "__main__":
    main()
