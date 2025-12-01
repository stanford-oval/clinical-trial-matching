#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_both_retrieval_evals.py

Runs, in order:
  1) run_smt_retrieval_eval.py
  2) run_trialgpt_ref_eval.py for:
      - global baselines (same ref file used across all selected modes)
      - mode-specific baselines (different ref file per mode)

Key features
------------
- Shared arguments are specified once.
- Supports wrapper modes:
    * both   -> ccr, all
    * triple -> ccr, all, all-explore
    * ccr
    * all
    * all-explore
- SMT runner is invoked once per mode.
- Global baseline runner is invoked once per baseline, over all selected modes.
- Mode-specific baseline runner is invoked once per (baseline, mode), using that mode's file only.
- Child script paths are resolved relative to THIS wrapper file, not the shell cwd.

Autodiscovery behavior
----------------------
- legacy files directly under trialgptref/ are kept if present
- newhybrid/<encoder>/*_N500.json
    -> objectiveless baseline, used globally across all selected modes
- newhybrid/<encoder>/*_{ccr,all,all_explore}.json
    -> objective baseline, used mode-specifically
- text/<encoder>/*_{ccr,all,all_explore}.json
    -> mode-specific baselines

Robustness passthrough
----------------------
- --prompt-update-rerun {all,relevant}
- --final-parse-rerun {never,missing,parse_error}
- --max-judge-attempts
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple


VALID_MODES = {"ccr", "all", "all-explore"}

FILE_SUFFIX_TO_MODE = {
    "_ccr.json": "ccr",
    "_all.json": "all",
    "_all_explore.json": "all-explore",
}


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


def expand_modes(run_modes: str) -> List[str]:
    if run_modes == "both":
        return ["ccr", "all"]
    if run_modes == "triple":
        return ["ccr", "all", "all-explore"]
    return [run_modes]


def resolve_path_relative_to_wrapper(p: str, wrapper_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    return (wrapper_dir / pp).resolve()


def parse_baseline_specs(raw_specs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()

    for s in raw_specs:
        if "=" not in s:
            raise ValueError(
                f"Bad --baseline-ref spec: {s!r}. Expected format name=path/to/file.json"
            )
        name, path = s.split("=", 1)
        name = name.strip()
        path = path.strip()

        if not name:
            raise ValueError(f"Bad --baseline-ref spec: {s!r}. Empty baseline name.")
        if not path:
            raise ValueError(f"Bad --baseline-ref spec: {s!r}. Empty baseline path.")
        if name in seen:
            raise ValueError(f"Duplicate baseline name: {name!r}")
        seen.add(name)
        out.append((name, path))

    return out


def parse_mode_baseline_specs(
    raw_specs: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    Parse:
      baseline_name:mode=path/to/file.json
    into:
      {
        baseline_name: {
          mode: path,
          ...
        },
        ...
      }
    """
    out: DefaultDict[str, Dict[str, str]] = defaultdict(dict)

    for s in raw_specs:
        if "=" not in s:
            raise ValueError(
                f"Bad --mode-baseline-ref spec: {s!r}. "
                f"Expected format baseline_name:mode=path/to/file.json"
            )

        lhs, path = s.split("=", 1)
        lhs = lhs.strip()
        path = path.strip()

        if ":" not in lhs:
            raise ValueError(
                f"Bad --mode-baseline-ref spec: {s!r}. "
                f"Expected format baseline_name:mode=path/to/file.json"
            )

        baseline_name, mode = lhs.rsplit(":", 1)
        baseline_name = baseline_name.strip()
        mode = mode.strip()

        if not baseline_name:
            raise ValueError(f"Bad --mode-baseline-ref spec: {s!r}. Empty baseline name.")
        if mode not in VALID_MODES:
            raise ValueError(
                f"Bad --mode-baseline-ref spec: {s!r}. "
                f"Mode must be one of {sorted(VALID_MODES)}."
            )
        if not path:
            raise ValueError(f"Bad --mode-baseline-ref spec: {s!r}. Empty path.")

        if mode in out[baseline_name]:
            raise ValueError(
                f"Duplicate mode-specific baseline entry for baseline={baseline_name!r}, mode={mode!r}"
            )

        out[baseline_name][mode] = path

    return dict(out)


def _safe_baseline_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def short_model_tag_from_filename(fname: str) -> str:
    lower = fname.lower()
    if "gpt-4.1" in lower:
        return "gpt41"
    if "gpt-5" in lower:
        return "gpt5"
    if "raw" in lower:
        return "raw"
    return "unk"


def autodiscover_trialgptref_baselines(
    trialgptref_root: Path,
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, str]]]:
    """
    Returns:
      global_baselines: [(baseline_name, path_str), ...]
      mode_specific_baselines: { baseline_name: {mode: path_str, ...}, ... }

    Conventions:
    - newhybrid/<encoder>/*_N500.json                    -> global (objectiveless)
    - newhybrid/<encoder>/*_{ccr,all,all_explore}.json  -> mode-specific (with objective)
    - text/<encoder>/*_{ccr,all,all_explore}.json       -> mode-specific
    """

    global_baselines: List[Tuple[str, str]] = []
    mode_specific: Dict[str, Dict[str, str]] = {}

    seen_global_names = set()

    def add_global(name: str, path: Path) -> None:
        if name in seen_global_names:
            raise ValueError(f"Duplicate auto-discovered global baseline name: {name}")
        seen_global_names.add(name)
        global_baselines.append((name, str(path)))

    # ------------------------------------------------------------------
    # Keep legacy / directly-rooted baselines if present
    # ------------------------------------------------------------------
    legacy_global_candidates = [
        ("trialgpt41", "trialgpt_retrieve41.json"),
        ("trialgpt5", "trialgpt_retrieve5.json"),
        ("clinicianA", "qid2nctids_results_Clinician_A_sigir_k20_bm25wt1_medcptwt1_N500.json"),
        ("clinicianB", "qid2nctids_results_Clinician_B_sigir_k20_bm25wt1_medcptwt1_N500.json"),
        ("clinicianC", "qid2nctids_results_Clinician_C_sigir_k20_bm25wt1_medcptwt1_N500.json"),
        ("clinicianD", "qid2nctids_results_Clinician_D_sigir_k20_bm25wt1_medcptwt1_N500.json"),
    ]
    for name, rel in legacy_global_candidates:
        p = trialgptref_root / rel
        if p.exists() and p.is_file():
            add_global(name, p)

    # ------------------------------------------------------------------
    # newhybrid: objectiveless globals + objective mode-specific
    # ------------------------------------------------------------------
    newhybrid_root = trialgptref_root / "newhybrid"
    if newhybrid_root.exists():
        for encoder_dir in sorted([p for p in newhybrid_root.iterdir() if p.is_dir()]):
            encoder = encoder_dir.name

            for jf in sorted(encoder_dir.glob("*.json")):
                fname = jf.name

                matched_mode = None
                for suffix, mode in FILE_SUFFIX_TO_MODE.items():
                    if fname.endswith(suffix):
                        matched_mode = mode
                        break

                if matched_mode is not None:
                    tag = short_model_tag_from_filename(fname)
                    baseline_name = _safe_baseline_name(f"newhybrid-{encoder}-obj-{tag}")
                    mode_specific.setdefault(baseline_name, {})[matched_mode] = str(jf)
                    continue

                if fname.endswith("_N500.json"):
                    tag = short_model_tag_from_filename(fname)
                    baseline_name = _safe_baseline_name(f"newhybrid-{encoder}-noobj-{tag}")
                    add_global(baseline_name, jf)

    # ------------------------------------------------------------------
    # text: all are mode-specific
    # ------------------------------------------------------------------
    text_root = trialgptref_root / "text"
    if text_root.exists():
        for encoder_dir in sorted([p for p in text_root.iterdir() if p.is_dir()]):
            encoder = encoder_dir.name
            baseline_name = _safe_baseline_name(f"text-{encoder}")
            mode_map: Dict[str, str] = {}

            for jf in sorted(encoder_dir.glob("*.json")):
                fname = jf.name
                for suffix, mode in FILE_SUFFIX_TO_MODE.items():
                    if fname.endswith(suffix):
                        if mode in mode_map:
                            raise ValueError(
                                f"Duplicate text baseline file for encoder={encoder!r}, mode={mode!r}: "
                                f"{mode_map[mode]} and {jf}"
                            )
                        mode_map[mode] = str(jf)
                        break

            if mode_map:
                mode_specific[baseline_name] = mode_map

    return global_baselines, mode_specific


def main() -> None:
    ap = argparse.ArgumentParser()

    # Script locations / Python executable
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    ap.add_argument("--smt-script", default="run_smt_retrieval_eval.py")
    ap.add_argument("--trialgpt-script", default="run_trialgpt_ref_eval.py")
    ap.add_argument(
        "--cwd",
        default=None,
        help="Optional working directory for child processes. "
             "By default they run in this wrapper's directory.",
    )

    ap.add_argument(
        "--trialgptref-root",
        default="../ops/trialgptref",
        help="Root dir containing trialgptref baselines (newhybrid/, text/, legacy files, etc.).",
    )

    # Shared args
    ap.add_argument(
        "--run-modes",
        default="triple",
        choices=["both", "triple", "ccr", "all", "all-explore"],
        help="Shared mode selection. both=ccr+all, triple=ccr+all+all-explore.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="gpt-4.1")
    ap.add_argument("--num-workers", type=int, default=64)
    ap.add_argument("--debug-prompts", action="store_true")

    ap.add_argument("--patient-corpus", default="../../dataset/clinical_trial/sigir/queries.jsonl")
    ap.add_argument("--trial-corpus", default="../../dataset/clinical_trial/sigir/corpus.jsonl")
    ap.add_argument(
        "--cache-root",
        default="./_shared_pair_cache/pair_cache",
        help="Shared PairDiskCache dir used by SMT and all baseline runs.",
    )
    ap.add_argument(
        "--prompt-update-rerun",
        default="relevant",
        choices=["all", "relevant"],
        help=(
            "When prompt templates change and shared-cache entries become stale: "
            "'all' reruns all stale pairs; "
            "'relevant' reruns only pairs previously judged relevant and reuses old "
            "mbench outputs for previously-irrelevant pairs when available."
        ),
    )

    # SMT args
    ap.add_argument(
        "--ccr-patient-labels-dir",
        default="../ops/out_compose/clean_eval__ccr__prevent__act/patient_labels",
    )
    ap.add_argument(
        "--all-patient-labels-dir",
        default="../ops/out_compose/clean_eval__all__prevent__act/patient_labels",
    )
    ap.add_argument(
        "--all-explore-patient-labels-dir",
        default="../ops/out_compose/clean_eval__all__prevent__nonact/patient_labels",
    )
    ap.add_argument("--sample-n", type=int, default=60)
    ap.add_argument("--shared-sample", action="store_true")
    ap.add_argument("--smt-output-root", default="./smt_retrieval_eval_out")
    ap.add_argument("--smt-mbench-root", default="./smt_retrieval_eval_mbench")
    ap.add_argument("--dump-prompts-per-pair", default="true")

    # Global baseline refs
    ap.add_argument(
        "--baseline-ref",
        action="append",
        default=[],
        help=(
            "Global baseline spec in the form name=path. Repeatable. "
            "This file is used across all selected modes."
        ),
    )

    # Mode-specific baseline refs
    ap.add_argument(
        "--mode-baseline-ref",
        action="append",
        default=[],
        help=(
            "Mode-specific baseline spec in the form baseline_name:mode=path. Repeatable. "
            "Example: "
            "--mode-baseline-ref mybaseline:ccr=../ops/trialgptref/foo_ccr.json"
        ),
    )

    # TrialGPT/baseline-eval args
    ap.add_argument(
        "--num-patients",
        type=int,
        default=None,
        help="If unset, defaults to --sample-n.",
    )
    ap.add_argument("--trialgpt-output-root", default="./trialgpt_retrieval_eval_out")
    ap.add_argument("--trialgpt-mbench-root", default="./trialgpt_retrieval_eval_mbench")
    ap.add_argument("--repair-mbench", action="store_true")
    ap.add_argument(
        "--final-parse-rerun",
        default="parse_error",
        choices=["never", "missing", "parse_error"],
        help=(
            "When final strict parsing sees missing/broken mbench outputs: "
            "'never' fails immediately; "
            "'missing' reruns only missing/empty pairs; "
            "'parse_error' reruns missing, empty, truncated, or strict-parse-failed pairs."
        ),
    )
    ap.add_argument(
        "--max-judge-attempts",
        type=int,
        default=3,
        help="Max attempts per pair before giving up.",
    )
    ap.add_argument(
        "--eval-k-from",
        default="o_round",
        choices=["o_round", "o_floor", "o_ceil", "fixed"],
    )
    ap.add_argument("--eval-k", type=int, default=0)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--debug-parse", action="store_true")
    ap.add_argument("--debug-parse-dir", default="./_debug_parse_dump")
    ap.add_argument("--print-judge-outputs", action="store_true")

    # Optional passthrough extras
    ap.add_argument(
        "--smt-extra",
        action="append",
        default=[],
        help="Extra raw argument token to append to SMT script. Repeatable.",
    )
    ap.add_argument(
        "--trialgpt-extra",
        action="append",
        default=[],
        help="Extra raw argument token to append to baseline-eval script. Repeatable.",
    )

    args = ap.parse_args()

    wrapper_dir = Path(__file__).resolve().parent
    child_cwd = Path(args.cwd).resolve() if args.cwd else wrapper_dir

    smt_script = resolve_path_relative_to_wrapper(args.smt_script, wrapper_dir)
    trialgpt_script = resolve_path_relative_to_wrapper(args.trialgpt_script, wrapper_dir)

    if not smt_script.exists():
        raise FileNotFoundError(
            f"SMT script not found: {smt_script}\n"
            f"Pass --smt-script with the correct path relative to {wrapper_dir} or as an absolute path."
        )
    if not trialgpt_script.exists():
        raise FileNotFoundError(
            f"Baseline eval script not found: {trialgpt_script}\n"
            f"Pass --trialgpt-script with the correct path relative to {wrapper_dir} or as an absolute path."
        )

    trialgptref_root = resolve_path_relative_to_wrapper(args.trialgptref_root, wrapper_dir)
    if not trialgptref_root.exists():
        raise FileNotFoundError(f"trialgptref root not found: {trialgptref_root}")

    auto_global_baselines, auto_mode_specific_baselines = autodiscover_trialgptref_baselines(
        trialgptref_root
    )

    if args.baseline_ref:
        global_baseline_specs = parse_baseline_specs(args.baseline_ref)
    else:
        global_baseline_specs = auto_global_baselines

    if args.mode_baseline_ref:
        mode_specific_specs = parse_mode_baseline_specs(args.mode_baseline_ref)
    else:
        mode_specific_specs = auto_mode_specific_baselines

    resolved_global_baselines: List[Tuple[str, Path]] = []
    seen_global_names = set()
    for name, rel_or_abs_path in global_baseline_specs:
        if name in seen_global_names:
            raise ValueError(f"Duplicate baseline name: {name}")
        seen_global_names.add(name)

        baseline_path = resolve_path_relative_to_wrapper(rel_or_abs_path, wrapper_dir)
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline ref file not found for {name}: {baseline_path}"
            )
        resolved_global_baselines.append((name, baseline_path))

    resolved_mode_specific_baselines: Dict[str, Dict[str, Path]] = {}
    for baseline_name, mode_to_path in mode_specific_specs.items():
        resolved_mode_specific_baselines[baseline_name] = {}
        for mode, rel_or_abs_path in mode_to_path.items():
            baseline_path = resolve_path_relative_to_wrapper(rel_or_abs_path, wrapper_dir)
            if not baseline_path.exists():
                raise FileNotFoundError(
                    f"Mode-specific baseline ref file not found for "
                    f"baseline={baseline_name}, mode={mode}: {baseline_path}"
                )
            resolved_mode_specific_baselines[baseline_name][mode] = baseline_path

    num_patients = args.num_patients if args.num_patients is not None else args.sample_n
    modes = expand_modes(args.run_modes)

    print(f"[INFO] wrapper_dir={wrapper_dir}")
    print(f"[INFO] child_cwd={child_cwd}")
    print(f"[INFO] smt_script={smt_script}")
    print(f"[INFO] baseline_eval_script={trialgpt_script}")
    print(f"[INFO] trialgptref_root={trialgptref_root}")
    print(f"[INFO] modes={modes}")

    print("[INFO] global baselines=")
    for baseline_name, baseline_path in resolved_global_baselines:
        print(f"  - {baseline_name}: {baseline_path}")

    print("[INFO] mode-specific baselines=")
    for baseline_name, mode_map in resolved_mode_specific_baselines.items():
        print(f"  - {baseline_name}")
        for mode in sorted(mode_map.keys()):
            print(f"      {mode}: {mode_map[mode]}")

    if not resolved_global_baselines and not resolved_mode_specific_baselines:
        raise RuntimeError(
            f"No baselines discovered under {trialgptref_root}. "
            f"Check the directory structure or pass --baseline-ref / --mode-baseline-ref explicitly."
        )

    # ------------------------------------------------------------------
    # 1) Run SMT once per mode
    # ------------------------------------------------------------------
    for mode in modes:
        smt_cmd: List[str] = [args.python, str(smt_script)]

        add_flag(smt_cmd, "--run-modes", mode)
        add_flag(smt_cmd, "--seed", args.seed)
        add_flag(smt_cmd, "--model-name", args.model_name)
        add_flag(smt_cmd, "--num-workers", args.num_workers)
        add_bool_flag(smt_cmd, "--debug-prompts", args.debug_prompts)

        add_flag(smt_cmd, "--patient-corpus", args.patient_corpus)
        add_flag(smt_cmd, "--trial-corpus", args.trial_corpus)
        add_flag(smt_cmd, "--cache-root", args.cache_root)
        add_flag(smt_cmd, "--prompt-update-rerun", args.prompt_update_rerun)

        add_flag(smt_cmd, "--ccr-patient-labels-dir", args.ccr_patient_labels_dir)
        add_flag(smt_cmd, "--all-patient-labels-dir", args.all_patient_labels_dir)
        add_flag(smt_cmd, "--all-explore-patient-labels-dir", args.all_explore_patient_labels_dir)

        add_flag(smt_cmd, "--sample-n", args.sample_n)
        add_bool_flag(smt_cmd, "--shared-sample", args.shared_sample)
        add_flag(smt_cmd, "--output-root", args.smt_output_root)
        add_flag(smt_cmd, "--mbench-root", args.smt_mbench_root)
        add_flag(smt_cmd, "--dump-prompts-per-pair", args.dump_prompts_per_pair)

        smt_cmd.extend(args.smt_extra)

        print(f"[INFO] SMT step for mode={mode}")
        run_cmd(smt_cmd, cwd=child_cwd)

    # ------------------------------------------------------------------
    # 2A) Run global baselines once per baseline across all selected modes
    # ------------------------------------------------------------------
    for baseline_name, baseline_path in resolved_global_baselines:
        baseline_output_root = Path(args.trialgpt_output_root) / baseline_name
        baseline_mbench_root = Path(args.trialgpt_mbench_root) / baseline_name
        baseline_debug_parse_dir = Path(args.debug_parse_dir) / baseline_name

        trialgpt_cmd: List[str] = [args.python, str(trialgpt_script)]

        add_flag(trialgpt_cmd, "--modes", ",".join(modes))
        add_flag(trialgpt_cmd, "--seed", args.seed)
        add_flag(trialgpt_cmd, "--model-name", args.model_name)
        add_flag(trialgpt_cmd, "--num-workers", args.num_workers)
        add_bool_flag(trialgpt_cmd, "--debug-prompts", args.debug_prompts)

        add_flag(trialgpt_cmd, "--default-patient-note-path", args.patient_corpus)
        add_flag(trialgpt_cmd, "--default-trial-description-path", args.trial_corpus)
        add_flag(trialgpt_cmd, "--cache-root", args.cache_root)
        add_flag(trialgpt_cmd, "--prompt-update-rerun", args.prompt_update_rerun)

        add_flag(trialgpt_cmd, "--smt-output-root", args.smt_output_root)
        add_flag(trialgpt_cmd, "--num-patients", num_patients)
        add_flag(trialgpt_cmd, "--trialgpt-ref", str(baseline_path))
        add_flag(trialgpt_cmd, "--trialgpt-output-root", str(baseline_output_root))
        add_flag(trialgpt_cmd, "--trialgpt-mbench-root", str(baseline_mbench_root))

        add_bool_flag(trialgpt_cmd, "--repair-mbench", args.repair_mbench)
        add_flag(trialgpt_cmd, "--final-parse-rerun", args.final_parse_rerun)
        add_flag(trialgpt_cmd, "--max-judge-attempts", args.max_judge_attempts)
        add_flag(trialgpt_cmd, "--eval-k-from", args.eval_k_from)
        add_flag(trialgpt_cmd, "--eval-k", args.eval_k)
        add_flag(trialgpt_cmd, "--min-k", args.min_k)

        add_bool_flag(trialgpt_cmd, "--debug-parse", args.debug_parse)
        add_flag(trialgpt_cmd, "--debug-parse-dir", str(baseline_debug_parse_dir))
        add_bool_flag(trialgpt_cmd, "--print-judge-outputs", args.print_judge_outputs)

        trialgpt_cmd.extend(args.trialgpt_extra)

        print(f"[INFO] Global baseline step name={baseline_name} modes={','.join(modes)}")
        run_cmd(trialgpt_cmd, cwd=child_cwd)

    # ------------------------------------------------------------------
    # 2B) Run mode-specific baselines once per (baseline, mode)
    # ------------------------------------------------------------------
    for baseline_name, mode_map in resolved_mode_specific_baselines.items():
        for mode in modes:
            if mode not in mode_map:
                print(
                    f"[WARN] Skipping mode-specific baseline={baseline_name} for mode={mode} "
                    f"because no file was provided."
                )
                continue

            baseline_path = mode_map[mode]
            baseline_output_root = Path(args.trialgpt_output_root) / baseline_name
            baseline_mbench_root = Path(args.trialgpt_mbench_root) / baseline_name
            baseline_debug_parse_dir = Path(args.debug_parse_dir) / baseline_name / mode

            trialgpt_cmd: List[str] = [args.python, str(trialgpt_script)]

            add_flag(trialgpt_cmd, "--modes", mode)
            add_flag(trialgpt_cmd, "--seed", args.seed)
            add_flag(trialgpt_cmd, "--model-name", args.model_name)
            add_flag(trialgpt_cmd, "--num-workers", args.num_workers)
            add_bool_flag(trialgpt_cmd, "--debug-prompts", args.debug_prompts)

            add_flag(trialgpt_cmd, "--default-patient-note-path", args.patient_corpus)
            add_flag(trialgpt_cmd, "--default-trial-description-path", args.trial_corpus)
            add_flag(trialgpt_cmd, "--cache-root", args.cache_root)
            add_flag(trialgpt_cmd, "--prompt-update-rerun", args.prompt_update_rerun)

            add_flag(trialgpt_cmd, "--smt-output-root", args.smt_output_root)
            add_flag(trialgpt_cmd, "--num-patients", num_patients)
            add_flag(trialgpt_cmd, "--trialgpt-ref", str(baseline_path))
            add_flag(trialgpt_cmd, "--trialgpt-output-root", str(baseline_output_root))
            add_flag(trialgpt_cmd, "--trialgpt-mbench-root", str(baseline_mbench_root))

            add_bool_flag(trialgpt_cmd, "--repair-mbench", args.repair_mbench)
            add_flag(trialgpt_cmd, "--final-parse-rerun", args.final_parse_rerun)
            add_flag(trialgpt_cmd, "--max-judge-attempts", args.max_judge_attempts)
            add_flag(trialgpt_cmd, "--eval-k-from", args.eval_k_from)
            add_flag(trialgpt_cmd, "--eval-k", args.eval_k)
            add_flag(trialgpt_cmd, "--min-k", args.min_k)

            add_bool_flag(trialgpt_cmd, "--debug-parse", args.debug_parse)
            add_flag(trialgpt_cmd, "--debug-parse-dir", str(baseline_debug_parse_dir))
            add_bool_flag(trialgpt_cmd, "--print-judge-outputs", args.print_judge_outputs)

            trialgpt_cmd.extend(args.trialgpt_extra)

            print(
                f"[INFO] Mode-specific baseline step "
                f"name={baseline_name} mode={mode} ref={baseline_path.name}"
            )
            run_cmd(trialgpt_cmd, cwd=child_cwd)

    print("\n[DONE] All runs finished successfully.")
    print(f"[DONE] modes:                  {','.join(modes)}")
    print(f"[DONE] SMT output root:        {args.smt_output_root}")
    print(f"[DONE] SMT mbench root:        {args.smt_mbench_root}")
    print(f"[DONE] Baseline output root:   {args.trialgpt_output_root}")
    print(f"[DONE] Baseline mbench root:   {args.trialgpt_mbench_root}")
    print(f"[DONE] Shared cache root:      {args.cache_root}")
    print(f"[DONE] final-parse-rerun:      {args.final_parse_rerun}")
    print(f"[DONE] max-judge-attempts:     {args.max_judge_attempts}")
    print(f"[DONE] prompt-update-rerun:    {args.prompt_update_rerun}")

    print("[DONE] global baselines:")
    for baseline_name, baseline_path in resolved_global_baselines:
        print(f"  - {baseline_name}: {baseline_path}")

    print("[DONE] mode-specific baselines:")
    for baseline_name, mode_map in resolved_mode_specific_baselines.items():
        print(f"  - {baseline_name}")
        for mode in sorted(mode_map.keys()):
            print(f"      {mode}: {mode_map[mode]}")


if __name__ == "__main__":
    main()