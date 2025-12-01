#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trialgpt_top200.py

Runs run_trialgpt_ref_eval.py once, with:
    --eval-k-from fixed
    --eval-k 200

Design goals
------------
- writes outputs to one fixed folder:
      trialgpt_retrieval_eval_out_top200
- writes mbench to one fixed folder:
      trialgpt_retrieval_eval_mbench_top200
- reuses the same shared cache across runs
- resolves child script path relative to THIS wrapper file, not shell cwd
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


VALID_MODES = {"ccr", "all", "all-explore"}


def add_flag(cmd: List[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def add_bool_flag(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def run_cmd(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print("\n" + "=" * 100)
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    if cwd is not None:
        print(f"[CWD] {cwd}")
    print("=" * 100)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_path_relative_to_wrapper(p: str, wrapper_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    return (wrapper_dir / pp).resolve()


def expand_modes(run_modes: str) -> List[str]:
    if run_modes == "both":
        return ["ccr", "all"]
    if run_modes == "triple":
        return ["ccr", "all", "all-explore"]
    return [run_modes]


def main() -> None:
    ap = argparse.ArgumentParser()

    # Script locations / Python executable
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    ap.add_argument("--trialgpt-script", default="run_trialgpt_ref_eval.py")
    ap.add_argument(
        "--cwd",
        default=None,
        help=(
            "Optional working directory for child processes. "
            "By default they run in this wrapper's directory."
        ),
    )

    # Shared mode selection
    ap.add_argument(
        "--run-modes",
        default="triple",
        choices=["both", "triple", "ccr", "all", "all-explore"],
        help="Shared mode selection. both=ccr+all, triple=ccr+all+all-explore.",
    )

    # Shared args passed through
    ap.add_argument("--smt-output-root", default="./smt_retrieval_eval_out")
    ap.add_argument("--num-patients", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--trialgpt-ref",
        default="../ops/trialgptref/trialgpt_retrieve5.json",
        help="TrialGPT reference JSON to evaluate.",
    )
    ap.add_argument(
        "--default-patient-note-path",
        default="../../dataset/clinical_trial/sigir/queries.jsonl",
    )
    ap.add_argument(
        "--default-trial-description-path",
        default="../../dataset/clinical_trial/sigir/corpus.jsonl",
    )

    # Judge model
    ap.add_argument("--model-name", default="gpt-5.1")
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--debug-prompts", action="store_true")

    # Fixed output roots
    ap.add_argument(
        "--trialgpt-output-root",
        default="./trialgpt_retrieval_eval_out_top200",
    )
    ap.add_argument(
        "--trialgpt-mbench-root",
        default="./trialgpt_retrieval_eval_mbench_top200",
    )

    # Shared cache root (reused)
    ap.add_argument(
        "--cache-root",
        default="./_shared_pair_cache/pair_cache",
    )
    ap.add_argument(
        "--prompt-update-rerun",
        default="all",
        choices=["all", "relevant"],
    )
    ap.add_argument(
        "--final-parse-rerun",
        default="parse_error",
        choices=["never", "missing", "parse_error"],
    )
    ap.add_argument("--max-judge-attempts", type=int, default=3)

    ap.add_argument("--repair-mbench", action="store_true")
    ap.add_argument("--min-k", type=int, default=1)

    ap.add_argument("--debug-parse", action="store_true")
    ap.add_argument("--debug-parse-dir", default="./_debug_parse_dump_top200")
    ap.add_argument("--print-judge-outputs", action="store_true")

    # Optional raw passthrough extras
    ap.add_argument(
        "--trialgpt-extra",
        action="append",
        default=[],
        help="Extra raw argument token to append to run_trialgpt_ref_eval.py. Repeatable.",
    )

    args = ap.parse_args()

    wrapper_dir = Path(__file__).resolve().parent
    child_cwd = Path(args.cwd).resolve() if args.cwd else wrapper_dir

    trialgpt_script = resolve_path_relative_to_wrapper(args.trialgpt_script, wrapper_dir)
    if not trialgpt_script.exists():
        raise FileNotFoundError(
            f"TrialGPT eval script not found: {trialgpt_script}\n"
            f"Pass --trialgpt-script with the correct path relative to {wrapper_dir} "
            f"or as an absolute path."
        )

    trialgpt_ref = resolve_path_relative_to_wrapper(args.trialgpt_ref, wrapper_dir)
    if not trialgpt_ref.exists():
        raise FileNotFoundError(f"TrialGPT ref not found: {trialgpt_ref}")

    modes = expand_modes(args.run_modes)
    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Unknown mode: {m}")

    output_root = Path(args.trialgpt_output_root)
    mbench_root = Path(args.trialgpt_mbench_root)
    debug_parse_dir = Path(args.debug_parse_dir)

    print(f"[INFO] wrapper_dir={wrapper_dir}")
    print(f"[INFO] child_cwd={child_cwd}")
    print(f"[INFO] trialgpt_script={trialgpt_script}")
    print(f"[INFO] trialgpt_ref={trialgpt_ref}")
    print(f"[INFO] modes={modes}")
    print(f"[INFO] trialgpt_output_root={output_root}")
    print(f"[INFO] trialgpt_mbench_root={mbench_root}")
    print(f"[INFO] cache_root={args.cache_root}")
    print("[INFO] eval_k_from=fixed")
    print("[INFO] eval_k=200")

    cmd: List[str] = [args.python, str(trialgpt_script)]

    add_flag(cmd, "--modes", ",".join(modes))
    add_flag(cmd, "--smt-output-root", args.smt_output_root)
    add_flag(cmd, "--num-patients", args.num_patients)
    add_flag(cmd, "--seed", args.seed)

    add_flag(cmd, "--trialgpt-ref", str(trialgpt_ref))
    add_flag(cmd, "--default-patient-note-path", args.default_patient_note_path)
    add_flag(cmd, "--default-trial-description-path", args.default_trial_description_path)

    add_flag(cmd, "--model-name", args.model_name)
    add_flag(cmd, "--num-workers", args.num_workers)
    add_bool_flag(cmd, "--debug-prompts", args.debug_prompts)

    add_flag(cmd, "--trialgpt-output-root", str(output_root))
    add_flag(cmd, "--trialgpt-mbench-root", str(mbench_root))

    add_flag(cmd, "--cache-root", args.cache_root)
    add_flag(cmd, "--prompt-update-rerun", args.prompt_update_rerun)
    add_flag(cmd, "--final-parse-rerun", args.final_parse_rerun)
    add_flag(cmd, "--max-judge-attempts", args.max_judge_attempts)

    add_flag(cmd, "--eval-k-from", "fixed")
    add_flag(cmd, "--eval-k", 200)
    add_flag(cmd, "--min-k", args.min_k)

    add_bool_flag(cmd, "--repair-mbench", args.repair_mbench)

    add_bool_flag(cmd, "--debug-parse", args.debug_parse)
    add_flag(cmd, "--debug-parse-dir", str(debug_parse_dir))
    add_bool_flag(cmd, "--print-judge-outputs", args.print_judge_outputs)

    cmd.extend(args.trialgpt_extra)

    run_cmd(cmd, cwd=child_cwd)

    print("\n[DONE] TrialGPT top-200 run finished successfully.")
    print(f"[DONE] modes:                  {','.join(modes)}")
    print(f"[DONE] trialgpt ref:           {trialgpt_ref}")
    print(f"[DONE] trialgpt output root:   {output_root}")
    print(f"[DONE] trialgpt mbench root:   {mbench_root}")
    print(f"[DONE] shared cache root:      {args.cache_root}")
    print(f"[DONE] final-parse-rerun:      {args.final_parse_rerun}")
    print(f"[DONE] max-judge-attempts:     {args.max_judge_attempts}")
    print(f"[DONE] prompt-update-rerun:    {args.prompt_update_rerun}")
    print("[DONE] eval-k-from:            fixed")
    print("[DONE] eval-k:                 200")


if __name__ == "__main__":
    main()