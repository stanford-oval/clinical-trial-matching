# =========================
# scripts/run_relevance_then_eligibility_batch.py
# (adjusted: factor out relevance definition blocks + robust cache invalidation)
# =========================

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal

from concurrent.futures import ThreadPoolExecutor, as_completed

from smt_core.inference_engine_5 import AzureInferenceEngine
from judge_eligibility import judge_eligibility, ELIGIBILITY_PROMPT_PATH
from judge_relevance import (
    judge_relevance,
    RelevancePromptKey,
    RELEVANCE_BASE_PROMPT_PATH,
    RELEVANCE_DEFINITION_PROMPT_PATHS,
)
from prompt_trace import PromptTrace

from cache_utils import PairDiskCache, CacheKey


@dataclass
class PairConfig:
    patient_id: str
    trial_id: str
    patient_note_path: Path
    trial_description_path: Path


def load_jsonl_as_dict(path: Path, id_key: str = "_id") -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_key not in obj:
                raise ValueError(f"Missing key '{id_key}' in record: {line[:200]}")
            mapping[obj[id_key]] = obj
    return mapping


def get_patient_text(patient_id: str, patient_corpus: Dict[str, dict]) -> str:
    if patient_id not in patient_corpus:
        raise KeyError(f"Patient ID '{patient_id}' not found in patient corpus.")
    if "text" not in patient_corpus[patient_id]:
        raise KeyError(f"No 'text' field for patient_id={patient_id}")
    return patient_corpus[patient_id]["text"]


def get_trial_text(trial_id: str, trial_corpus: Dict[str, dict]) -> str:
    if trial_id not in trial_corpus:
        raise KeyError(f"Trial ID '{trial_id}' not found in trial corpus.")
    if "text" not in trial_corpus[trial_id]:
        raise KeyError(f"No 'text' field for trial_id={trial_id}")
    return trial_corpus[trial_id]["text"]


def build_engine() -> AzureInferenceEngine:
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "OPENAI_ENDPOINT env var is not set. "
            "Please set it before running this script."
        )

    return AzureInferenceEngine(
        endpoint=endpoint,
        model_name="gpt-4.1",
        default_temperature=0.0,
        default_max_tokens=None,
    )


def run_judges_for_pair(
    pair: PairConfig,
    engine: AzureInferenceEngine,
    output_root: Path,
    relevance_prompt: RelevancePromptKey,
    trace: PromptTrace,
    patient_corpus_cache: Dict[Path, Dict[str, dict]],
    trial_corpus_cache: Dict[Path, Dict[str, dict]],
    cache: PairDiskCache,
    key_lock: threading.Lock,
) -> None:
    # --- load corpora once per path (cached) ---
    if pair.patient_note_path not in patient_corpus_cache:
        patient_corpus_cache[pair.patient_note_path] = load_jsonl_as_dict(pair.patient_note_path)
    if pair.trial_description_path not in trial_corpus_cache:
        trial_corpus_cache[pair.trial_description_path] = load_jsonl_as_dict(pair.trial_description_path)

    patient_corpus = patient_corpus_cache[pair.patient_note_path]
    trial_corpus = trial_corpus_cache[pair.trial_description_path]

    patient_text = get_patient_text(pair.patient_id, patient_corpus)
    trial_text = get_trial_text(pair.trial_id, trial_corpus)

    cache_key = CacheKey(mode=relevance_prompt, patient_id=pair.patient_id, trial_id=pair.trial_id)

    # Prompt paths for this mode (base + mode-specific definition)
    rel_base_prompt_path = RELEVANCE_BASE_PROMPT_PATH
    rel_def_prompt_path = RELEVANCE_DEFINITION_PROMPT_PATHS[relevance_prompt]
    elig_prompt_path = ELIGIBILITY_PROMPT_PATH

    # Capture engine config so changing model/temp/max_tokens invalidates cache.
    model_name = getattr(engine, "model_name", "gpt-4.1")
    temperature = float(getattr(engine, "default_temperature", 0.0))
    max_tokens = getattr(engine, "default_max_tokens", None)
    max_tokens_int = None if max_tokens is None else int(max_tokens)

    # IMPORTANT: include BOTH base + definition prompt paths in meta so edits invalidate cache
    expected_meta = cache.compute_meta(
        patient_text=patient_text,
        trial_text=trial_text,
        relevance_base_prompt_path=rel_base_prompt_path,
        relevance_definition_prompt_path=rel_def_prompt_path,
        eligibility_prompt_path=elig_prompt_path,
        model_name=str(model_name),
        temperature=temperature,
        max_tokens=max_tokens_int,
    )

    # Outputs are mode-separated to avoid overwriting
    out_dir = output_root / relevance_prompt / pair.patient_id / pair.trial_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lock per key so two threads don't duplicate work
    with key_lock:
        cached = cache.load_if_fresh(cache_key, expected_meta)
        if cached is not None:
            (out_dir / "relevance.txt").write_text(cached["relevance"], encoding="utf-8")
            (out_dir / "eligibility.txt").write_text(cached["eligibility"], encoding="utf-8")
            print(f"[CACHE HIT] patient_id={pair.patient_id}, trial_id={pair.trial_id}, mode={relevance_prompt}")
            return

        # --- run relevance first ---
        relevance_result = judge_relevance(
            trial_text,
            patient_text,
            engine=engine,
            prompt_key=relevance_prompt,
            trace=trace,
            patient_id=pair.patient_id,
            trial_id=pair.trial_id,
        )

        # --- run eligibility using relevance output ---
        eligibility_result = judge_eligibility(
            trial_text,
            patient_text,
            relevance_result=relevance_result,
            engine=engine,
            trace=trace,
            patient_id=pair.patient_id,
            trial_id=pair.trial_id,
        )

        # Save cache (atomic) + write outputs
        cache.save(cache_key, expected_meta, relevance_result, eligibility_result)
        (out_dir / "relevance.txt").write_text(relevance_result, encoding="utf-8")
        (out_dir / "eligibility.txt").write_text(eligibility_result, encoding="utf-8")


def run_batch(
    pairs: List[PairConfig],
    output_root: Path,
    relevance_prompt: RelevancePromptKey,
    max_workers: Optional[int] = None,
) -> None:
    engine = build_engine()
    trace = PromptTrace(output_root / "prompts_and_outputs_dump.txt", tokenizer_model="gpt-4.1")

    cache = PairDiskCache(output_root)

    patient_corpus_cache: Dict[Path, Dict[str, dict]] = {}
    trial_corpus_cache: Dict[Path, Dict[str, dict]] = {}

    # Per-key locks to avoid duplicate work
    _locks: Dict[CacheKey, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def _get_lock(k: CacheKey) -> threading.Lock:
        with _locks_guard:
            if k not in _locks:
                _locks[k] = threading.Lock()
            return _locks[k]

    def _task(pair: PairConfig) -> None:
        print(f"[START] patient_id={pair.patient_id}, trial_id={pair.trial_id}, mode={relevance_prompt}")
        ck = CacheKey(mode=relevance_prompt, patient_id=pair.patient_id, trial_id=pair.trial_id)
        run_judges_for_pair(
            pair=pair,
            engine=engine,
            output_root=output_root,
            relevance_prompt=relevance_prompt,
            trace=trace,
            patient_corpus_cache=patient_corpus_cache,
            trial_corpus_cache=trial_corpus_cache,
            cache=cache,
            key_lock=_get_lock(ck),
        )
        print(f"[DONE]  patient_id={pair.patient_id}, trial_id={pair.trial_id}, mode={relevance_prompt}")

    if max_workers is None or max_workers <= 1:
        for p in pairs:
            _task(p)
        print(
            f"[TOKENS] prompt={trace.totals.prompt_tokens} "
            f"completion={trace.totals.completion_tokens} "
            f"total={trace.totals.total}"
        )
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, p): p for p in pairs}
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] patient_id={pair.patient_id}, trial_id={pair.trial_id}: {e}")

    print(
        f"[TOKENS] prompt={trace.totals.prompt_tokens} "
        f"completion={trace.totals.completion_tokens} "
        f"total={trace.totals.total}"
    )


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(
        description=(
            "Run relevance then eligibility for a batch of patient/trial pairs. "
            "Eligibility uses relevance output as additional context. "
            "Includes disk caching per (patient_id, trial_id, mode) with prompt SHA invalidation "
            "(base + mode-specific definition)."
        )
    )

    parser.add_argument(
        "--pairs-file",
        type=str,
        required=True,
        help=(
            "Path to a JSONL or JSON file with a list of objects each containing: "
            "patient_id, trial_id, patient_note_path (optional), trial_description_path (optional). "
            "If paths are omitted, defaults are used."
        ),
    )
    parser.add_argument(
        "--default-patient-note-path",
        type=str,
        default="../../dataset/clinical_trial/sigir/queries.jsonl",
        help="Default path to the patient note JSONL corpus.",
    )
    parser.add_argument(
        "--default-trial-description-path",
        type=str,
        default="../../dataset/clinical_trial/sigir/corpus.jsonl",
        help="Default path to the trial description JSONL corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory where outputs will be written.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count() * 2,
        help="Number of parallel workers (threads) to use.",
    )
    parser.add_argument(
        "--relevance-prompt",
        type=str,
        choices=["cc", "ccr", "all"],
        default="cc",
        help="Which relevance prompt to use: cc / ccr / all.",
    )

    args = parser.parse_args()

    default_patient_path = Path(args.default_patient_note_path).resolve()
    default_trial_path = Path(args.default_trial_description_path).resolve()
    output_root = Path(args.output_dir).resolve()
    relevance_prompt: RelevancePromptKey = args.relevance_prompt  # type: ignore

    # Load pairs file (JSON or JSONL)
    pairs_raw: List[dict] = []
    pairs_path = Path(args.pairs_file)

    with pairs_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line.strip():
            raise ValueError("Pairs file is empty.")
        if first_line.lstrip().startswith("["):
            f.seek(0)
            pairs_raw = json.load(f)
        else:
            pairs_raw.append(json.loads(first_line))
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs_raw.append(json.loads(line))

    pairs: List[PairConfig] = []
    for obj in pairs_raw:
        patient_id = obj["patient_id"]
        trial_id = obj["trial_id"]

        patient_note_path_str = obj.get("patient_note_path")
        trial_description_path_str = obj.get("trial_description_path")

        patient_note_path = Path(patient_note_path_str).resolve() if patient_note_path_str else default_patient_path
        trial_description_path = Path(trial_description_path_str).resolve() if trial_description_path_str else default_trial_path

        pairs.append(
            PairConfig(
                patient_id=patient_id,
                trial_id=trial_id,
                patient_note_path=patient_note_path,
                trial_description_path=trial_description_path,
            )
        )

    run_batch(
        pairs=pairs,
        output_root=output_root,
        relevance_prompt=relevance_prompt,
        max_workers=args.num_workers,
    )