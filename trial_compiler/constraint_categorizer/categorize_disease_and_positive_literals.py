#!/usr/bin/env python3
# categorize_disease_and_positive_constraint_literals.py
# -*- coding: utf-8 -*-

import os
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from trial_compiler.constraint_categorizer.modules.disease_categorizer import DiseaseCategorizer
from trial_compiler.constraint_categorizer.modules.positive_literal_categorizer import PositiveLiteralCategorizer
from trial_compiler.constraint_categorizer.modules.joint_specificity_filter import JointSpecificityFilter


# ─────────────────────────────────────────
# Engine selection + creation
# ─────────────────────────────────────────

def create_engine():
    endpoint_env = os.environ.get("OPENAI_ENDPOINT", "").strip().lower()

    if endpoint_env.endswith("gpt-5"):
        from smt_core.inference_engine_5 import AzureInferenceEngine
        engine_version = "gpt-5"
        model_name = "gpt-5"
    elif endpoint_env.endswith("gpt-4o"):
        from smt_core.inference_engine import AzureInferenceEngine
        engine_version = "gpt-4o"
        model_name = "gpt-4o"
    elif endpoint_env.endswith("gpt-4.1"):
        from smt_core.inference_engine import AzureInferenceEngine
        engine_version = "gpt-4.1"
        model_name = "gpt-4.1"
    elif endpoint_env.endswith("gpt-4.1-mini"):
        from smt_core.inference_engine import AzureInferenceEngine
        engine_version = "gpt-4.1-mini"
        model_name = "gpt-4.1-mini"
    else:
        raise EnvironmentError(
            f"Unrecognized or missing OPENAI_ENDPOINT: {endpoint_env!r}. "
            "Expected it to end with 'gpt-4o' or 'gpt-5' or 'gpt-4.1' or 'gpt-4.1-mini'."
        )

    azure_endpoint = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("OPENAI_ENDPOINT")
    )
    if not azure_endpoint:
        raise EnvironmentError(
            "Missing Azure endpoint (AZURE_OPENAI_ENDPOINT or OPENAI_ENDPOINT)."
        )

    print(f"[INFO] Using AzureInferenceEngine for {engine_version} ({model_name})")
    return AzureInferenceEngine(
        endpoint=azure_endpoint,
        api_key_env_var="OPENAI_API_KEY",
        model_name=model_name,
    )


# ─────────────────────────────────────────
# Retry + failure recording helpers
# ─────────────────────────────────────────

def _result_failed(res: Dict[str, Any]) -> bool:
    return bool(res) and bool(res.get("error"))

def _safe_retry_delay(seconds: float):
    if seconds and seconds > 0:
        time.sleep(seconds)

def _result_key(res: Dict[str, Any]) -> str:
    return (
        str(res.get("json_path"))
        or str(res.get("trial_id"))
        or json.dumps(res, sort_keys=True, ensure_ascii=False)
    )

def _write_eventual_failures(
    *,
    build_root: Path,
    stage_name: str,
    failures: List[Dict[str, Any]],
) -> None:
    if not failures:
        return

    out_dir = Path("mbench") / "eventual_failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stage_name}_eventual_failures.json"
    csv_path = out_dir / f"{stage_name}_eventual_failures.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    fieldnames = sorted({k for row in failures for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failures)

    print(f"[FAILURES] Wrote {len(failures)} eventual failures:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

def retry_failed_results(
    *,
    stage_name: str,
    build_root: Path,
    initial_results: List[Dict[str, Any]],
    retry_fn: Callable[[Dict[str, Any], int], Dict[str, Any]],
    outer_retries: int,
    retry_delay: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Retry only results that still have non-empty result['error'].
    """
    results_by_key: Dict[str, Dict[str, Any]] = {
        _result_key(r): r for r in initial_results
    }

    for retry_round in range(1, outer_retries + 1):
        failed_now = [r for r in results_by_key.values() if _result_failed(r)]
        if not failed_now:
            break

        print(
            f"[RETRY] stage={stage_name} "
            f"round={retry_round}/{outer_retries} "
            f"failed_items={len(failed_now)}"
        )

        updated_results: List[Dict[str, Any]] = []
        for failed_result in failed_now:
            try:
                updated = retry_fn(failed_result, retry_round)
            except Exception as e:
                updated = dict(failed_result)
                updated["error"] = f"OuterRetryException: {repr(e)}"
            updated_results.append(updated)

        for r in updated_results:
            results_by_key[_result_key(r)] = r

        if retry_round < outer_retries:
            _safe_retry_delay(retry_delay)

    final_results = list(results_by_key.values())
    eventual_failures = [r for r in final_results if _result_failed(r)]
    _write_eventual_failures(
        build_root=build_root,
        stage_name=stage_name,
        failures=eventual_failures,
    )
    return final_results


# ─────────────────────────────────────────
# Helper: run a categorizer in parallel over its files
# ─────────────────────────────────────────

def run_disease_parallel(disease_categorizer: DiseaseCategorizer, seed: int, jobs: int):
    """
    Parallel wrapper around DiseaseCategorizer.categorize_file.
    """
    disease_dir = disease_categorizer.disease_dir
    if not disease_dir.exists():
        raise FileNotFoundError(f"Disease dir not found: {disease_dir}")

    json_files = sorted(disease_dir.glob("*.json"))
    if not json_files:
        print(f"[INFO] No json files found in {disease_dir}")
        return []

    if jobs <= 1:
        return disease_categorizer.run_all(seed=seed)

    print(f"[PIPELINE] Disease categorization in parallel with {jobs} threads, {len(json_files)} files.")

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        future_to_path = {}
        for idx, path in enumerate(json_files):
            future = ex.submit(disease_categorizer.categorize_file, path, seed=seed + idx)
            future_to_path[future] = path

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                res = future.result()
            except Exception as e:
                res = {
                    "trial_id": path.stem,
                    "json_path": str(path),
                    "output_path": None,
                    "error": f"WorkerException: {repr(e)}",
                    "raw_output": None,
                }
            results.append(res)

    return results


def run_poslit_parallel(poslit_categorizer: PositiveLiteralCategorizer, seed: int, jobs: int):
    """
    Parallel wrapper around PositiveLiteralCategorizer.categorize_file.
    """
    positive_dir = poslit_categorizer.positive_dir
    if not positive_dir.exists():
        raise FileNotFoundError(f"Positive literal dir not found: {positive_dir}")

    json_files = sorted(positive_dir.glob("*.json"))
    if not json_files:
        print(f"[INFO] No json files found in {positive_dir}")
        return []

    if jobs <= 1:
        return poslit_categorizer.run_all(seed=seed)

    print(f"[PIPELINE] Positive literal categorization in parallel with {jobs} threads, {len(json_files)} files.")

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        future_to_path = {}
        for idx, path in enumerate(json_files):
            future = ex.submit(poslit_categorizer.categorize_file, path, seed=seed + idx)
            future_to_path[future] = path

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                res = future.result()
            except Exception as e:
                res = {
                    "trial_id": path.stem,
                    "json_path": str(path),
                    "output_path": None,
                    "error": f"WorkerException: {repr(e)}",
                    "raw_output": None,
                }
            results.append(res)

    return results


# ─────────────────────────────────────────
# Per-item outer retry helpers
# ─────────────────────────────────────────

def retry_one_disease_result(
    failed_result: Dict[str, Any],
    retry_round: int,
    *,
    disease_categorizer: DiseaseCategorizer,
    seed: int,
) -> Dict[str, Any]:
    json_path_str = failed_result.get("json_path")
    if not json_path_str:
        return {
            **failed_result,
            "error": "Missing json_path for disease retry",
        }

    json_path = Path(json_path_str)
    if not json_path.exists():
        return {
            **failed_result,
            "error": f"Missing file for disease retry: {json_path}",
        }

    retry_seed = seed + 100000 * retry_round
    print(f"[RETRY][disease] round={retry_round} file={json_path.name}")
    return disease_categorizer.categorize_file(json_path, seed=retry_seed)


def retry_one_poslit_result(
    failed_result: Dict[str, Any],
    retry_round: int,
    *,
    poslit_categorizer: PositiveLiteralCategorizer,
    seed: int,
) -> Dict[str, Any]:
    json_path_str = failed_result.get("json_path")
    if not json_path_str:
        return {
            **failed_result,
            "error": "Missing json_path for positive literal retry",
        }

    json_path = Path(json_path_str)
    if not json_path.exists():
        return {
            **failed_result,
            "error": f"Missing file for positive literal retry: {json_path}",
        }

    retry_seed = seed + 100000 * retry_round
    print(f"[RETRY][positive_literal] round={retry_round} file={json_path.name}")
    return poslit_categorizer.categorize_file(json_path, seed=retry_seed)


def retry_one_joint_filter_result(
    failed_result: Dict[str, Any],
    retry_round: int,
    *,
    jf: JointSpecificityFilter,
    seed: int,
) -> Dict[str, Any]:
    trial_id = failed_result.get("trial_id")
    disease_in = failed_result.get("disease_in")
    poslit_in = failed_result.get("poslit_in")

    if not trial_id:
        return {
            **failed_result,
            "error": "Missing trial_id for joint filter retry",
        }

    disease_path = Path(disease_in) if disease_in else None
    poslit_path = Path(poslit_in) if poslit_in else None

    retry_seed = seed + 100000 * retry_round
    print(f"[RETRY][joint_filter] round={retry_round} trial={trial_id}")

    res = jf.filter_one_trial(
        trial_id=trial_id,
        disease_path=disease_path,
        poslit_path=poslit_path,
        seed=retry_seed,
    )
    return {
        "trial_id": res.trial_id,
        "disease_in": res.disease_in,
        "poslit_in": res.poslit_in,
        "disease_out": res.disease_out,
        "poslit_out": res.poslit_out,
        "error": res.error,
        "raw_output": res.raw_output,
    }


# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Categorize diseases and/or positive literals using LLM."
    )
    parser.add_argument(
        "--build-root",
        type=str,
        default="../../build",
        help="Root build directory (default: ../../build)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for LLM calls.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["disease", "positive_literal", "all"],
        default="all",
        help="Which task to run: disease | positive_literal | all",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=24,
        help="Number of parallel threads to use per task (default: 24).",
    )

    parser.add_argument(
        "--outer-retries",
        type=int,
        default=2,
        help="Number of pipeline-level retries for items that still fail after inner retries.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between outer retry rounds.",
    )

    # prefilter default ON, disable with --no-prefilter
    parser.add_argument(
        "--no-prefilter",
        dest="prefilter",
        action="store_false",
        help="Disable joint specificity filter before categorization.",
    )
    parser.set_defaults(prefilter=True)

    parser.add_argument(
        "--filter-jobs",
        type=int,
        default=24,
        help="Number of parallel threads to use for the prefilter stage (default: 24).",
    )

    args = parser.parse_args()

    build_root = Path(args.build_root).resolve()
    engine = create_engine()

    print(
        f"[PIPELINE] Running task={args.task} jobs={args.jobs} "
        f"prefilter={args.prefilter} outer_retries={args.outer_retries}"
    )

    disease_input_folder = "disease"
    poslit_input_subdir = "positive_constraint_literals/per_file"

    # ─────────────────────────────────────────
    # 0. Joint specificity prefilter
    # ─────────────────────────────────────────
    if args.prefilter:
        print("[PIPELINE] Starting joint specificity prefilter...")

        jf = JointSpecificityFilter(
            engine=engine,
            build_root=build_root,
            disease_subdir="disease",
            positive_literal_subdir="positive_constraint_literals/per_file",
            canon_expanded_subdir="canon",
            prompt_path=Path("prompts/JointSpecificityFilter.prompt"),
            default_temp=0.0,
            default_top_p=1.0,
            max_retries=3,
            log_dir=Path("mbench/joint_filter"),
        )

        jf_results_raw = jf.run_all(seed=args.seed, jobs=args.filter_jobs)

        jf_results = [
            {
                "trial_id": r.trial_id,
                "disease_in": r.disease_in,
                "poslit_in": r.poslit_in,
                "disease_out": r.disease_out,
                "poslit_out": r.poslit_out,
                "error": r.error,
                "raw_output": r.raw_output,
            }
            for r in jf_results_raw
        ]

        jf_results = retry_failed_results(
            stage_name="joint_filter",
            build_root=build_root,
            initial_results=jf_results,
            retry_fn=lambda failed_result, retry_round: retry_one_joint_filter_result(
                failed_result,
                retry_round,
                jf=jf,
                seed=args.seed,
            ),
            outer_retries=args.outer_retries,
            retry_delay=args.retry_delay,
        )

        failures = [r for r in jf_results if r.get("error")]
        print(f"[PIPELINE] Prefilter completed. trials={len(jf_results)} failures={len(failures)}")

        disease_input_folder = "disease_filtered"
        poslit_input_subdir = "positive_constraint_literals_filtered/per_file"

    # ─────────────────────────────────────────
    # 1. Disease categorization
    # ─────────────────────────────────────────
    if args.task in ("disease", "all"):
        print(f"[PIPELINE] Starting disease categorization (input={disease_input_folder})...")

        disease_categorizer = DiseaseCategorizer(
            engine=engine,
            build_root=build_root,
            input_folder_name=disease_input_folder,
            prompt_path=Path("prompts/DiseaseCategorization.prompt"),
            default_temp=0.0,
            default_top_p=1.0,
            max_retries=3,
            log_dir=Path("mbench/disease_categorized"),
        )

        disease_results = run_disease_parallel(
            disease_categorizer,
            seed=args.seed,
            jobs=args.jobs,
        )

        disease_results = retry_failed_results(
            stage_name="disease_categorization",
            build_root=build_root,
            initial_results=disease_results,
            retry_fn=lambda failed_result, retry_round: retry_one_disease_result(
                failed_result,
                retry_round,
                disease_categorizer=disease_categorizer,
                seed=args.seed,
            ),
            outer_retries=args.outer_retries,
            retry_delay=args.retry_delay,
        )

        disease_failures = [r for r in disease_results if r.get("error")]
        print(
            f"[PIPELINE] Disease categorization completed. "
            f"{len(disease_results)} files processed; failures={len(disease_failures)}"
        )

    # ─────────────────────────────────────────
    # 2. Positive literal categorization
    # ─────────────────────────────────────────
    if args.task in ("positive_literal", "all"):
        print(f"[PIPELINE] Starting positive literal categorization (input={poslit_input_subdir})...")

        poslit_categorizer = PositiveLiteralCategorizer(
            engine=engine,
            build_root=build_root,
            positive_literal_subdir=poslit_input_subdir,
            canon_expanded_subdir="canon",
            default_temp=0.0,
            default_top_p=1.0,
            max_retries=3,
            log_dir=Path("mbench"),
        )

        poslit_results = run_poslit_parallel(
            poslit_categorizer,
            seed=args.seed,
            jobs=args.jobs,
        )

        poslit_results = retry_failed_results(
            stage_name="positive_literal_categorization",
            build_root=build_root,
            initial_results=poslit_results,
            retry_fn=lambda failed_result, retry_round: retry_one_poslit_result(
                failed_result,
                retry_round,
                poslit_categorizer=poslit_categorizer,
                seed=args.seed,
            ),
            outer_retries=args.outer_retries,
            retry_delay=args.retry_delay,
        )

        poslit_failures = [r for r in poslit_results if r.get("error")]
        print(
            f"[PIPELINE] Positive literal categorization completed. "
            f"{len(poslit_results)} files processed; failures={len(poslit_failures)}"
        )

    print("[PIPELINE] All tasks completed.")


if __name__ == "__main__":
    main()