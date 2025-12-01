#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trialgpt_topk_sweep.py

Runs run_trialgpt_ref_eval.py sequentially with:
    --eval-k-from fixed
    --eval-k 300
    --eval-k 400
    --eval-k 500

Behavior
--------
- Creates separate output folders for each K:
      trialgpt_retrieval_eval_out_top300
      trialgpt_retrieval_eval_out_top400
      trialgpt_retrieval_eval_out_top500

- Creates separate mbench folders for each K:
      trialgpt_retrieval_eval_mbench_top300
      trialgpt_retrieval_eval_mbench_top400
      trialgpt_retrieval_eval_mbench_top500

- Reuses the same shared cache across runs
- Resolves child script path relative to THIS wrapper file, not shell cwd
- Runs sequentially and stops on first failure:
      300 must succeed before 400 starts
      400 must succeed before 500 starts
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


VALID_MODES = {"ccr", "all", "all-explore"}
DEFAULT_TOPK_LIST = [300, 400, 500]


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


def parse_topk_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"All eval-k values must be positive, got: {k}")
        vals.append(k)
    if not vals:
        raise ValueError("No valid top-k values provided.")
    return vals


def make_output_root(base_dir: str, k: int) -> Path:
    return Path(f"{base_dir}_top{k}")


def build_child_cmd(
    args: argparse.Namespace,
    trialgpt_script: Path,
    trialgpt_ref: Path,
    modes: List[str],
    output_root: Path,
    mbench_root: Path,
    debug_parse_dir: Path,
    eval_k: int,
) -> List[str]:
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
    add_flag(cmd, "--eval-k", eval_k)
    add_flag(cmd, "--min-k", args.min_k)

    add_bool_flag(cmd, "--repair-mbench", args.repair_mbench)

    add_bool_flag(cmd, "--debug-parse", args.debug_parse)
    add_flag(cmd, "--debug-parse-dir", str(debug_parse_dir))
    add_bool_flag(cmd, "--print-judge-outputs", args.print_judge_outputs)

    cmd.extend(args.trialgpt_extra)
    return cmd


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

    # Base output roots; actual folders become <base>_top300, <base>_top400, ...
    ap.add_argument(
        "--trialgpt-output-root-base",
        default="./trialgpt_retrieval_eval_out",
    )
    ap.add_argument(
        "--trialgpt-mbench-root-base",
        default="./trialgpt_retrieval_eval_mbench",
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
    ap.add_argument("--debug-parse-dir-base", default="./_debug_parse_dump")
    ap.add_argument("--print-judge-outputs", action="store_true")

    ap.add_argument(
        "--topk-list",
        default="300,400,500",
        help="Comma-separated fixed eval-k values to run sequentially.",
    )

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

    topk_list = parse_topk_list(args.topk_list)

    print(f"[INFO] wrapper_dir={wrapper_dir}")
    print(f"[INFO] child_cwd={child_cwd}")
    print(f"[INFO] trialgpt_script={trialgpt_script}")
    print(f"[INFO] trialgpt_ref={trialgpt_ref}")
    print(f"[INFO] modes={modes}")
    print(f"[INFO] cache_root={args.cache_root}")
    print(f"[INFO] topk_list={topk_list}")
    print(f"[INFO] trialgpt_output_root_base={args.trialgpt_output_root_base}")
    print(f"[INFO] trialgpt_mbench_root_base={args.trialgpt_mbench_root_base}")
    print("[INFO] eval_k_from=fixed")

    completed: List[int] = []

    for eval_k in topk_list:
        output_root = make_output_root(args.trialgpt_output_root_base, eval_k)
        mbench_root = make_output_root(args.trialgpt_mbench_root_base, eval_k)
        debug_parse_dir = make_output_root(args.debug_parse_dir_base, eval_k)

        print("\n" + "#" * 100)
        print(f"[STAGE] Starting eval-k={eval_k}")
        print(f"[STAGE] output_root={output_root}")
        print(f"[STAGE] mbench_root={mbench_root}")
        print(f"[STAGE] debug_parse_dir={debug_parse_dir}")
        print("#" * 100)

        cmd = build_child_cmd(
            args=args,
            trialgpt_script=trialgpt_script,
            trialgpt_ref=trialgpt_ref,
            modes=modes,
            output_root=output_root,
            mbench_root=mbench_root,
            debug_parse_dir=debug_parse_dir,
            eval_k=eval_k,
        )

        try:
            run_cmd(cmd, cwd=child_cwd)
        except subprocess.CalledProcessError as e:
            print("\n[FAILED] Stopping sweep because this stage failed.")
            print(f"[FAILED] eval-k: {eval_k}")
            print(f"[FAILED] completed earlier stages: {completed}")
            print(f"[FAILED] return code: {e.returncode}")
            raise

        completed.append(eval_k)

        print(f"\n[DONE] eval-k={eval_k} finished successfully.")
        print(f"[DONE] output root: {output_root}")
        print(f"[DONE] mbench root: {mbench_root}")

    print("\n" + "=" * 100)
    print("[DONE] Entire TrialGPT top-k sweep finished successfully.")
    print(f"[DONE] completed ks: {completed}")
    print(f"[DONE] modes: {','.join(modes)}")
    print(f"[DONE] trialgpt ref: {trialgpt_ref}")
    print(f"[DONE] shared cache root: {args.cache_root}")
    print(f"[DONE] final-parse-rerun: {args.final_parse_rerun}")
    print(f"[DONE] max-judge-attempts: {args.max_judge_attempts}")
    print(f"[DONE] prompt-update-rerun: {args.prompt_update_rerun}")
    print("[DONE] eval-k-from: fixed")


if __name__ == "__main__":
    main()