#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast

# -----------------------------
# Provided: create_engine()
# -----------------------------
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
            "Expected it to end with 'gpt-4o' or 'gpt-5' or 'gpt-4.1'."
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

class LLMCaller:
    def __init__(self, engine):
        self.engine = engine

    def _call_llm(self, prompt: str) -> str:
        out = self.engine(prompt)
        # most of your engines return a list-like, where [0] is the text
        if isinstance(out, (list, tuple)) and out:
            return out[0]
        # fallback
        return str(out)


# -----------------------------
# Parsing / Verification
# -----------------------------

class ParsingError(RuntimeError):
    pass

class VerificationError(RuntimeError):
    pass

_JSON_BLOCK_RE = re.compile(r"\[[\s\S]*\]", re.MULTILINE)

def _extract_json_array(text: str) -> str:
    """
    Extracts the first JSON array-like block from the LLM output.
    """
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ParsingError("Could not find a JSON array in LLM output.")
    return m.group(0)

def _unwrap_pythonish_wrappers(raw: Any) -> str:
    """
    Normalize raw engine output into a string that contains the JSON array.

    Handles:
      - str containing JSON array
      - str containing python repr: "['...']" or "'[...]'"
      - list[str] where first element contains JSON array
      - dict/other => str(...)
    """
    # If engine returned already-parsed list/dict etc.
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], str):
            return raw[0]
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)

    # Otherwise treat as string
    s = raw if isinstance(raw, str) else str(raw)
    s = s.strip()

    # Fast path: already looks like JSON array
    if s.startswith("[") and s.endswith("]"):
        return s

    # Try JSON decode: could be a JSON string containing the array
    # e.g. "\"[ {..} ]\""
    try:
        j = json.loads(s)
        if isinstance(j, str):
            return j
        if isinstance(j, list) and len(j) == 1 and isinstance(j[0], str):
            return j[0]
        if isinstance(j, list):
            return json.dumps(j, ensure_ascii=False)
    except Exception:
        pass

    # Try Python literal eval: handles "['...']" and "'[...]'"
    try:
        v = ast.literal_eval(s)
        if isinstance(v, str):
            return v
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
            return v[0]
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
    except Exception:
        pass

    # If nothing worked, return original (we'll still attempt bracket extraction)
    return s

def parse_llm_output(raw_out: Any) -> List[Dict[str, Any]]:
    """
    Robust parser for LLM output.

    Expected core structure after normalization:
    [
      {"id":"C1","subcohort_is_control_group":true/false,"rationale":"..."},
      ...
    ]
    """
    normalized = _unwrap_pythonish_wrappers(raw_out)
    block = _extract_json_array(normalized)
    print(block)

    try:
        data = json.loads(block)
    except Exception as e:
        raise ParsingError(f"Failed to json.loads extracted array: {e}") from e

    if not isinstance(data, list):
        raise ParsingError("Parsed JSON is not a list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ParsingError(f"Item {i} is not a dict.")
        if "id" not in item or "subcohort_is_control_group" not in item or "rationale" not in item:
            raise ParsingError(f"Item {i} missing required keys (id, subcohort_is_control_group, rationale).")
        if not isinstance(item["id"], str):
            raise ParsingError(f"Item {i} 'id' must be a string.")
        if not isinstance(item["subcohort_is_control_group"], bool):
            raise ParsingError(f"Item {i} 'subcohort_is_control_group' must be a boolean.")
        if not isinstance(item["rationale"], str):
            raise ParsingError(f"Item {i} 'rationale' must be a string.")

    return data

def verify_llm_output(
    parsed: List[Dict[str, Any]],
    enrollment_cohorts: List[Dict[str, Any]],
) -> None:
    """
    Verify:
    (1) length matches
    (2) order matches expected C1, C2, ..., Cn
    """
    n_expected = len(enrollment_cohorts)
    if len(parsed) != n_expected:
        raise VerificationError(f"LLM output length {len(parsed)} != expected {n_expected}.")

    for idx, item in enumerate(parsed, start=1):
        expected_id = f"C{idx}"
        got_id = item.get("id")
        if got_id != expected_id:
            raise VerificationError(f"Order/id mismatch at position {idx}: expected {expected_id}, got {got_id!r}.")

# -----------------------------
# Engine call wrapper
# -----------------------------

def call_llm(engine, prompt_text: str) -> str:
    """
    Calls your AzureInferenceEngine. Adjust the method name here if your engine differs.
    Common patterns:
      - engine(prompt_text)
      - engine.complete(prompt_text)
      - engine.infer(prompt_text)
    """
    # Try a few common method names to be robust.
    if callable(engine):
        return engine(prompt_text)
    for meth in ("complete", "infer", "run", "generate"):
        fn = getattr(engine, meth, None)
        if callable(fn):
            return fn(prompt_text)
    raise AttributeError("Engine is not callable and has no known completion method (complete/infer/run/generate).")

# -----------------------------
# Main processing
# -----------------------------

def load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def label_single_cohort(trial_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    If only one cohort, set is_control=False in both locations.
    """
    cohorts = trial_obj.get("enrollment_cohorts")
    if isinstance(cohorts, list) and len(cohorts) == 1:
        cohorts[0]["is_control"] = False

    pn = trial_obj.get("preprocessor_normalized")
    if isinstance(pn, dict):
        pn_cohorts = pn.get("enrollment_cohorts")
        if isinstance(pn_cohorts, list) and len(pn_cohorts) == 1:
            pn_cohorts[0]["is_control"] = False

    return trial_obj


def apply_labels_by_id(
    trial_obj: Dict[str, Any],
    llm_parsed: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Add 'is_control' into:
      - trial_obj["enrollment_cohorts"]
      - trial_obj["preprocessor_normalized"]["enrollment_cohorts"] (if present)

    Mapping is positional via C1, C2, ...
    """
    id_to_bool = {x["id"]: x["subcohort_is_control_group"] for x in llm_parsed}

    def _label_cohorts(cohorts: List[Dict[str, Any]]):
        for idx, c in enumerate(cohorts, start=1):
            if not isinstance(c, dict):
                continue
            cid = f"C{idx}"
            c["is_control"] = bool(id_to_bool.get(cid, False))

    # Top-level enrollment_cohorts
    cohorts = trial_obj.get("enrollment_cohorts")
    if isinstance(cohorts, list):
        _label_cohorts(cohorts)

    # preprocessor_normalized.enrollment_cohorts
    pn = trial_obj.get("preprocessor_normalized")
    if isinstance(pn, dict):
        pn_cohorts = pn.get("enrollment_cohorts")
        if isinstance(pn_cohorts, list):
            _label_cohorts(pn_cohorts)

    return trial_obj

def process_trial_file(
    llm,
    prompt_template: str,
    in_path: Path,
    out_path: Path,
    log_dir: Path,
    max_retries: int = 3,
    sleep_s: float = 0.5,
) -> Tuple[bool, Optional[str]]:
    """
    Returns (success, error_message).

    Output contract (ALWAYS):
      - Write ONLY the labeled enrollment_cohorts list to out_path.
      - Write ONLY enrollment_cohorts to parsed log as well.
    Single-cohort special case:
      - Assign is_control=False deterministically (no LLM call).
    """
    trial_id = in_path.stem
    try:
        raw_in = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(raw_in, dict):
            return (False, "Input JSON is not an object/dict.")

        cohorts = raw_in.get("enrollment_cohorts")
        if not isinstance(cohorts, list):
            return (False, "Missing or invalid 'enrollment_cohorts' (must be a list).")

        ensure_dir(log_dir)
        ensure_dir(out_path.parent)

        # --------
        # Single cohort: deterministic label
        # --------
        if len(cohorts) == 1:
            labeled_obj = label_single_cohort(raw_in)
            cohorts_out = labeled_obj.get("enrollment_cohorts", [])

            write_json(out_path, cohorts_out)
            write_json(
                log_dir / f"{trial_id}.parsed_enrollment_subcohort.json",
                cohorts_out,
            )
            return (True, None)

        # --------
        # Multi cohort: LLM labeling + parse/verify
        # --------
        enrollment_str = json.dumps(cohorts, indent=2, ensure_ascii=False)
        prompt = prompt_template.replace("#ENROLLMENT_COHORTS#", enrollment_str)

        write_text(log_dir / f"{trial_id}.prompt.txt", prompt)

        last_err = None
        for attempt in range(1, max_retries + 1):
            raw_out = llm._call_llm(prompt)
            if not isinstance(raw_out, str):
                raw_out = str(raw_out)

            write_text(
                log_dir / f"{trial_id}.raw_llm_output.attempt{attempt}.txt",
                raw_out,
            )

            try:
                parsed = parse_llm_output(raw_out)
                verify_llm_output(parsed, cohorts)

                labeled_obj = apply_labels_by_id(raw_in, parsed)
                cohorts_out = labeled_obj.get("enrollment_cohorts", [])

                write_json(out_path, cohorts_out)
                write_json(
                    log_dir / f"{trial_id}.parsed_enrollment_subcohort.json",
                    cohorts_out,
                )
                return (True, None)

            except (ParsingError, VerificationError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    time.sleep(sleep_s)
                continue

        return (False, last_err or "Unknown error in LLM parsing/verifying.")

    except Exception as e:
        return (False, f"Unhandled exception: {type(e).__name__}: {e}")


def iter_trial_json_files(in_dir: Path) -> List[Path]:
    return sorted([p for p in in_dir.rglob("*.json") if p.is_file()])


def _process_one_file(args):
    """
    Worker function for multiprocessing.
    """
    (
        in_path,
        in_dir,
        out_dir,
        log_dir,
        prompt_template,
        max_retries,
    ) = args

    # IMPORTANT: create engine INSIDE the process
    engine = create_engine()
    llm = LLMCaller(engine)

    rel = in_path.relative_to(in_dir)
    out_path = out_dir / rel

    ok, err = process_trial_file(
        llm=llm,
        prompt_template=prompt_template,
        in_path=in_path,
        out_path=out_path,
        log_dir=log_dir,
        max_retries=max_retries,
    )
    return (in_path, ok, err)


def main():
    ap = argparse.ArgumentParser(description="Label enrollment subcohorts as control/non-control.")
    ap.add_argument(
        "--in-dir",
        default="../../subcohort_results",
        help="Directory containing <trial_id>.json files (recursively searched).",
    )
    ap.add_argument(
        "--out-dir",
        default="../../build/subcohort_results_control_labeled",
        help="Output directory for labeled json files.",
    )
    ap.add_argument(
        "--log-dir",
        default="mbench/subcohort_control",
        help="Directory to write logs (prompt/raw output/parsed cohorts).",
    )
    ap.add_argument(
        "--prompt-path",
        default="prompts/control_group_identifier.prompt",
        help="Path to prompt template containing #ENROLLMENT_COHORTS# placeholder.",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for LLM call when parse/verify fails.",
    )

    ap.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs (processes). Use 1 for sequential execution.",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    prompt_path = Path(args.prompt_path).expanduser().resolve()

    if not in_dir.exists():
        print(f"[ERROR] in-dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(2)

    prompt_template = load_prompt_template(prompt_path)
    engine = create_engine()
    llm = LLMCaller(engine)

    files = iter_trial_json_files(in_dir)
    if not files:
        print(f"[WARN] No .json files found under {in_dir}")
        return

    n_ok = 0
    n_fail = 0

    # for in_path in files:
    #     # Preserve relative structure under out_dir
    #     rel = in_path.relative_to(in_dir)
    #     out_path = out_dir / rel

    #     ok, err = process_trial_file(
    #         llm=llm,
    #         prompt_template=prompt_template,
    #         in_path=in_path,
    #         out_path=out_path,
    #         log_dir=log_dir,
    #         max_retries=args.max_retries,
    #     )

    #     if ok:
    #         n_ok += 1
    #         print(f"[OK] {in_path} -> {out_path}")
    #     else:
    #         n_fail += 1
    #         print(f"[FAIL] {in_path}: {err}", file=sys.stderr)
    if args.n_jobs == 1:
        # Sequential (original behavior)
        engine = create_engine()
        llm = LLMCaller(engine)

        for in_path in files:
            rel = in_path.relative_to(in_dir)
            out_path = out_dir / rel

            ok, err = process_trial_file(
                llm=llm,
                prompt_template=prompt_template,
                in_path=in_path,
                out_path=out_path,
                log_dir=log_dir,
                max_retries=args.max_retries,
            )

            if ok:
                n_ok += 1
                print(f"[OK] {in_path} -> {out_path}")
            else:
                n_fail += 1
                print(f"[FAIL] {in_path}: {err}", file=sys.stderr)

    else:
        # Parallel execution
        from multiprocessing import Pool

        work_items = [
            (
                in_path,
                in_dir,
                out_dir,
                log_dir,
                prompt_template,
                args.max_retries,
            )
            for in_path in files
        ]

        with Pool(processes=args.n_jobs) as pool:
            for in_path, ok, err in pool.imap_unordered(_process_one_file, work_items):
                if ok:
                    n_ok += 1
                    print(f"[OK] {in_path}")
                else:
                    n_fail += 1
                    print(f"[FAIL] {in_path}: {err}", file=sys.stderr)


    print(f"[DONE] ok={n_ok} fail={n_fail} total={n_ok+n_fail}")

if __name__ == "__main__":
    main()
