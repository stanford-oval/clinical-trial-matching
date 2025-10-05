#!/usr/bin/env python3
"""
Select 'must-suffice' canonical variables using an LLM (Azure AI Inference),
with robust filename parsing, multiprocessing, cache sharding, skip-existing/uptodate
fast-paths, AND optional restriction to positive literals discovered from SMT2.

Integration notes:
- Pass --positive-dir to point at the outputs from scripts/find_positive_canon_literals.py
- Use --positives-required to strictly filter to positive stems only; otherwise we'll
  filter when available and fall back gracefully if missing.

The prompt follows the clinical-logic assistant spec and expects the model to output
<scratchpad>...</scratchpad> and a <chosen_set> that contains a JSON array of selected variables.
We ignore the scratchpad and only parse the array inside <chosen_set>.

As of SCRIPT_VERSION 1.9.1, the prompt is fed only these fields for each canonical variable:
- entity_variable_name
- variable_meaning
- template
- timeframe
- entity_type

And the mbench payload logs:
- variable_candidates (all variables shown to the LLM)
- variable_must_suffice (chosen)
- variable_not_chosen (complement)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)

# Your local utils
from trial_compiler.constraint_augmenter.utils import preprocess_data

# Import your robust Azure wrapper
try:
    from smt_core.inference_engine import AzureInferenceEngine
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Could not import AzureInferenceEngine from inference_engine.py.\n"
        "Make sure inference_engine.py is on your PYTHONPATH or in the same directory.\n"
        f"Import error: {e}"
    )

# ------------------------------ logging helpers ------------------------------ #
class _StreamToLogger:
    """Redirect a stream (stdout/stderr) into logging."""
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, msg):
        if not isinstance(msg, str):
            try:
                msg = msg.decode("utf-8", "replace")
            except Exception:
                msg = str(msg)
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buf:
            self.logger.log(self.level, self._buf.rstrip())
            self._buf = ""


def _init_worker_logging(log_path_str: str) -> None:
    """Initializer for worker processes so they append to the same run log file."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(processName)s %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path_str, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Also capture worker stdout/stderr into the file
    sys.stdout = _StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)


# ------------------------------ datamodels ------------------------------ #
@dataclass
class DefaultDecision:
    entity_variable_name: str
    default_value: str  # one of: "true" | "false" | "null"
    confidence: float
    rationale: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "entity_variable_name": self.entity_variable_name,
            "default_value": self.default_value,
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale.strip(),
        }


# ------------------------------ utils ------------------------------ #
JSON_PATTERN = re.compile(r"\{[\s\S]*\}")
DECISION_LIST_PATTERN = re.compile(r"\[[\s\S]*\]")

# Strict block for chosen_set
CHOSEN_SET_BLOCK = re.compile(
    r"<chosen_set>\s*(?P<body>[\s\S]*?)\s*</chosen_set>",
    re.IGNORECASE
)


def _extract_array(text: str) -> Optional[List[str]]:
    """Try to find a JSON array within text and return list[str]."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if isinstance(x, str)]
    except Exception:
        pass
    m = DECISION_LIST_PATTERN.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, str)]
        except Exception:
            pass
    return None


def parse_llm_selection(text: str) -> List[str]:
    """
    Parse LLM output that uses:
      <scratchpad> ... </scratchpad>
      <chosen_set>
      [
        "var1",
        "var2"
      ]
      </chosen_set>

    We ignore <scratchpad> and only extract the array inside <chosen_set>.
    If <chosen_set> is missing, we fall back to scanning for any top-level JSON array.
    """
    raw = text.strip()

    # Prefer the strict wrapper first
    m = CHOSEN_SET_BLOCK.search(raw)
    if m:
        body = m.group("body").strip()
        arr = _extract_array(body)
        if arr is not None:
            return arr

    # Fallbacks: direct array or first array anywhere
    arr = _extract_array(raw)
    return arr or []


def parse_filename_parts(path: Path) -> Tuple[str, str]:
    """
    From 'NCT02065596a_exclusion_canonical_variables.json' -> ('NCT02065596a', 'exclusion').
    Accepts optional trailing letters after the digits in the NCT id.
    Tolerates optional '_canonical_variables' suffix.
    """
    stem = path.stem  # e.g., NCT02065596a_exclusion_canonical_variables
    m = re.match(r"^(NCT\d+[A-Za-z]*)_([A-Za-z]+)(?:_canonical_variables)?$", stem)
    if m:
        trial_id = m.group(1)
        inc_exc = m.group(2).lower()
        return trial_id, inc_exc

    # Fallback: split + validate
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unrecognized canonical filename format: {path.name}")
    trial_id = parts[0]
    inc_exc = parts[1].lower()
    if not re.match(r"^NCT\d+[A-Za-z]*$", trial_id):
        raise ValueError(f"Unrecognized canonical filename format: {path.name}")
    if inc_exc not in {"inclusion", "exclusion"}:
        raise ValueError(f"Unrecognized include/exclude segment in: {path.name}")
    return trial_id, inc_exc


# ------------------------------ positive literal integration ------------------------------ #
def _stem_of(var: str) -> str:
    """Match the finder: split at '@@' and take the left part."""
    return var.split('@@', 1)[0] if isinstance(var, str) else var


def _normalize_trial_id_for_positives(trial_id: str) -> str:
    """
    Positive-literal finder emits NCT######## (digits only).
    Canon files may be like NCT########a. Normalize to 8 digits.
    """
    m = re.match(r"^(NCT\d{8})", trial_id)
    return m.group(1) if m else trial_id


def _load_positive_stems(positive_dir: Path) -> Dict[Tuple[str, str], Set[str]]:
    """
    Read <positive_dir>/summary.csv from find_positive_canon_literals.py
    and build {(trial_id8, arm): {stem,...}}
    CSV columns: trial_id, arm, smt2_file, num_hits, hits, direction
    where 'hits' is semicolon-separated (stems).
    """
    from csv import DictReader
    summary = positive_dir / "summary.csv"
    out: Dict[Tuple[str, str], Set[str]] = {}
    if not summary.exists():
        return out
    with summary.open("r", encoding="utf-8", newline="") as f:
        r = DictReader(f)
        for row in r:
            tid = (row.get("trial_id") or "").strip()
            arm = (row.get("arm") or "").strip().lower()
            hits = (row.get("hits") or "").strip()
            if not tid or not arm:
                continue
            key = (tid, arm)
            stems = {x for x in hits.split(";") if x}
            if key not in out:
                out[key] = set()
            out[key].update(stems)
    return out


# ------------------------------ prompt loading ------------------------------ #
PROMPT_SYSTEM = (
    "# === ROLE ===\n"
    "You are a meticulous clinical-logic assistant. Your job is to choose a set of variables "
    "representing patient facts. Select only those entity_variables that, if true for a patient, "
    "would definitely be reported in and canonicalized from the patient note.\n"
)

PROMPT_INSTRUCTIONS = (
    "# === GUIDELINES  ===\n"
    "1. A variable may be chosen only if it represents a fact about a finding, procedure, or substance.\n"
    "2. A variable may be chosen only when if a patient has the fact represented by that variable, then the patient note (vignette) of that patient would definitely explicitly mention that fact.\n"
    "3. Please reference the patient notes (vignettes) in the \"EXAMPLE PATIENT NOTE\" section to understand the level of details in the patient notes (vignettes).\n"
    "4. Do not choose any variable that represents the patient's willingness to participate. This is because those won't be mentioned in the patient notes, and we always assume that those facts are true.\n"
    "5. Do not choose facts that are general, vague, or trivial.\n"
    "6. Even if a variable represents a fact that is significant, do not select it if it is possible that it won't be explicitly mentioned in the patient notes (vignettes) of patients who have this fact.\n"
    "7. Think in <scratchpad> first before you output the final set of chosen variables.\n"
    "\n"
    "# === EXAMPLE PATIENT NOTE (VIGNETTES) ===\n"
    '{"_id": "sigir-20141", "text": "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. The latest episode of pain ended half an hour prior to her arrival. She is known to have hypertension and obesity. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. She currently takes no medications. Physical examination is normal. The EKG shows nonspecific changes."}\n'
    '{"_id": "sigir-20142", "text": "An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. He has just returned from a 5 day vacation in Colorado. Parents report that prior to the onset of fever and cough, he had loose stools. He denies upper respiratory tract symptoms. On examination he is in respiratory distress and has bronchial respiratory sounds on the left. A chest x-ray shows bilateral lung infiltrates."}\n'
    "\n"
    "# === Output format (STRICT) ===\n"
    "Be sure to wrap these two sections in <scratchpad> ... </scratchpad> and <chosen_set>\n"
    "<scratchpad>\n"
    "... Think about \n"
    "</scratchpad>\n"
    "\n"
    "<chosen_set>\n"
    "# Output a list of selected variables STRICTLY following the following format\n"
    "[\n"
    '  \"<selected_variable_name1_verbatim>\",\n'
    '  \"<selected_variable_name2_verbatim>\",\n'
    "]\n"
    "</chosen_set>\n"
    "\n"
    "# === INPUT ===\n"
    "\n"
    "#CANON_VARIABLES#\n"
    "\n"
    "Note: Only choose variables whose facts would definitely be explicitly reported in notes of patients who have them. "
    "Avoid willingness/consent, vague or trivial facts. Output must include <chosen_set> as shown. "
    "We will ignore the <scratchpad> section in downstream parsing.\n"
)

PROMPT_FALLBACK = f"{PROMPT_SYSTEM}\n\n{PROMPT_INSTRUCTIONS}"


def load_prompt_template(path: Optional[Path]) -> str:
    """Load the prompt template from disk. Fallback to PROMPT_FALLBACK if missing."""
    if not path:
        return PROMPT_FALLBACK
    try:
        if path.exists():
            txt = path.read_text(encoding="utf-8")
            return txt.strip() if txt.strip() else PROMPT_FALLBACK
    except Exception:
        pass
    return PROMPT_FALLBACK


# ------------------------------ core runner ------------------------------ #
class DefaultFinder:
    def __init__(
        self,
        endpoint: str,
        model: str = "gpt-4.1",
        fallback_endpoint: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.001,
        max_tokens: int = 4096,
        cache_path: Optional[Path] = None,
        prompt_text: Optional[str] = None,
    ) -> None:
        self.engine = AzureInferenceEngine(
            endpoint=endpoint,
            api_key_env_var="OPENAI_API_KEY",
            fallback_endpoint=fallback_endpoint,
            model_name=model,
            default_temperature=temperature,
            default_top_p=top_p,
            default_max_tokens=max_tokens,
            verbose=False,
        )
        self.cache: Dict[str, DefaultDecision] = {}
        self.cache_path = cache_path
        self.prompt_text = prompt_text or PROMPT_FALLBACK
        if cache_path and cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            dd = DefaultDecision(
                                entity_variable_name=rec["entity_variable_name"],
                                default_value=rec["default_value"],
                                confidence=float(rec.get("confidence", 0.5)),
                                rationale=rec.get("rationale", ""),
                            )
                            self.cache[rec["cache_key"]] = dd
                        except Exception:
                            continue
            except Exception:
                pass


# ------------------------------ IO helpers ------------------------------ #
def load_canon_file(path: Path) -> Tuple[str, str, Dict]:
    trial_id, inc_exc = parse_filename_parts(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return trial_id, inc_exc, data


def write_defaults_file(
    output_dir: Path,
    trial_id: str,
    inc_exc: str,
    decisions: List[DefaultDecision],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{trial_id}_{inc_exc}_defaults.json"
    payload = {
        "trial_id": trial_id,
        "inc_exc": inc_exc,
        "defaults": [d.to_json() for d in decisions],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


# ------------------------------ mbench logging ------------------------------ #
SCRIPT_VERSION = "1.9.1"  # now logs not-chosen candidates into mbench


def _sha256(text: str) -> str:
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except Exception:
        return "NA"


def _is_uptodate(
    *,
    cpath: Path,
    out_path: Path,
    mbench_dir: Path,
    trial_id: str,
    model: str,
    prompt_text: str,
) -> bool:
    """
    True if (a) output exists and is newer than input AND
            (b) mbench meta exists and matches model + prompt hash.
    """
    if not out_path.exists():
        return False
    try:
        if out_path.stat().st_mtime < cpath.stat().st_mtime:
            return False
    except Exception:
        return False

    meta_path = mbench_dir / trial_id / "mbench_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if str(meta.get("model", "")) != str(model):
        return False
    if meta.get("prompt_sha256") != _sha256(prompt_text):
        return False
    return True


# ---- post process helpers: validate llm output ----
def _validate_llm_selections(
    selections: List[str],
    canonical_subset: List[Dict],
) -> Tuple[bool, List[str]]:
    """
    Ensure every LLM-selected string is exactly one of the
    entity_variable_name values in canonical_subset.

    Returns:
        (is_valid, invalid_items)
    """
    allowed = {
        cv.get("entity_variable_name")
        for cv in canonical_subset
        if "entity_variable_name" in cv
    }
    allowed.discard(None)

    invalid = [s for s in selections if s not in allowed]
    return (len(invalid) == 0), invalid


# ------------------------------ worker ------------------------------ #
def _worker_process_file(
    cpath_str: str,
    *,
    resolved_endpoint: str,
    fallback_endpoint: Optional[str],
    model: str,
    no_cache: bool,
    cache_path: Path,
    prompt_text: str,
    prompt_path: Optional[Path],
    output_dir: Path,
    mbench_dir: Path,
    no_mbench: bool,
    skip_existing: bool,
    skip_uptodate: bool,
    # NEW: positive literal integration
    positive_map: Dict[Tuple[str, str], Set[str]],
    positives_required: bool,
) -> Tuple[str, Optional[str]]:

    cpath = Path(cpath_str)

    try:
        # --- Load canonical file ---
        trial_id, inc_exc, data = load_canon_file(cpath)
        out_path = output_dir / f"{trial_id}_{inc_exc}.json"

        # --- Skip logic ---
        if skip_existing and out_path.exists():
            return (str(out_path), "SKIPPED (exists)")

        if skip_uptodate and _is_uptodate(
            cpath=cpath,
            out_path=out_path,
            mbench_dir=mbench_dir,
            trial_id=trial_id,
            model=model,
            prompt_text=prompt_text,
        ):
            return (str(out_path), "SKIPPED (up-to-date)")

        # --- Setup engine / finder ---
        finder = DefaultFinder(
            endpoint=resolved_endpoint,
            fallback_endpoint=fallback_endpoint,
            model=model,
            cache_path=None if no_cache else cache_path,
            prompt_text=prompt_text,
        )

        # Source canonical variables (may contain many fields)
        canonical_variables_full = (data.get("canonical_variables", []) or [])
        # preprocess data, remove demographic and numeric variables
        canonical_variables_full = preprocess_data(canonical_variables_full)

        # === NEW: Restrict to POSITIVE LITERALS ONLY ===
        base_tid = _normalize_trial_id_for_positives(trial_id)  # convert NCT########a -> NCT########
        pos_key = (base_tid, inc_exc.lower())                   # arm is 'inclusion'/'exclusion'
        positive_stems = positive_map.get(pos_key, set())

        if positive_stems:
            before = len(canonical_variables_full)
            canonical_variables_full = [
                cv for cv in canonical_variables_full
                if _stem_of(cv.get("entity_variable_name", "")) in positive_stems
            ]
            logging.info(
                "Positive-filter %s/%s: %d -> %d",
                trial_id, inc_exc, before, len(canonical_variables_full)
            )
        elif positives_required:
            logging.warning(
                "No positive stems found for %s/%s; producing empty selection due to --positives-required.",
                trial_id, inc_exc
            )
            canonical_variables_full = []
        else:
            logging.info(
                "No positive stems for %s/%s; leaving canonical set unfiltered.",
                trial_id, inc_exc
            )

        # ---- Project to prompt fields (unchanged) ----
        def project_for_prompt(item: Dict) -> Dict:
            return {
                "entity_variable_name": item.get("entity_variable_name"),
                "variable_meaning": item.get("variable_meaning"),
                "template": item.get("template"),
                "timeframe": item.get("timeframe"),
                "entity_type": item.get("entity_type"),
            }

        canonical_variables = [project_for_prompt(x) for x in canonical_variables_full]

        # Helper: chunk canonical_variables
        def chunk_vars(seq, size):
            for i in range(0, len(seq), size):
                yield i // size, seq[i: i + size]

        # Helper: robust LLM call with retry/backoff
        def call_engine_with_retry(prompt: str,
                                   profile_stage: str,
                                   profile_trial_id: str,
                                   profile_side: str,
                                   max_attempts: int = 5):
            attempt = 1
            while True:
                try:
                    return finder.engine(
                        prompt,
                        _profile_stage=profile_stage,
                        _profile_trial_id=profile_trial_id,
                        _profile_side=profile_side,
                    )[0]
                except (HttpResponseError, ServiceRequestError, ServiceResponseError) as e:
                    # Decide if transient / rate-limit
                    msg = str(e) or ""
                    status = getattr(e, "status_code", None)
                    is_rl = (
                        isinstance(e, HttpResponseError)
                        and ("RateLimitReached" in msg or status in (429, 503))
                    )
                    is_net = isinstance(e, (ServiceRequestError, ServiceResponseError))
                    if not (is_rl or is_net) or attempt >= max_attempts:
                        # Non-transient or exceeded retries
                        raise

                    # Compute delay: Retry-After if present, else exponential backoff
                    delay = None
                    if isinstance(e, HttpResponseError):
                        resp = getattr(e, "response", None)
                        if resp is not None:
                            headers = getattr(resp, "headers", {}) or {}
                            ra = headers.get("Retry-After") or headers.get("retry-after")
                            if ra is not None:
                                try:
                                    delay = float(ra)
                                except (TypeError, ValueError):
                                    delay = None
                    if delay is None:
                        delay = min(2 ** (attempt - 1), 30.0)

                    logging.warning(
                        f"Transient error for {trial_id}/{inc_exc} "
                        f"(attempt {attempt}/{max_attempts}): {e} -- retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                    attempt += 1

        # Helper: dedupe while preserving order
        def unique_preserve_order(seq: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        # Helper: validate output. retry llm call if output variables are not in input set
        # Returns (selections, raw_output_used)
        def _call_llm_with_validation(
            prompt: str,
            canonical_subset: List[Dict],
            *,
            profile_stage: str,
            trial_id: str,
            inc_exc: str,
            max_semantic_retries: int = 3,
        ) -> Tuple[List[str], str]:
            allowed = {
                cv.get("entity_variable_name")
                for cv in canonical_subset
                if "entity_variable_name" in cv
            }
            allowed.discard(None)

            last_selections: List[str] = []
            last_raw: str = ""

            for attempt in range(1, max_semantic_retries + 1):
                raw = call_engine_with_retry(
                    prompt,
                    profile_stage=profile_stage,
                    profile_trial_id=trial_id,
                    profile_side=inc_exc,
                )
                last_raw = raw
                selections = parse_llm_selection(raw)
                last_selections = selections

                invalid = [s for s in selections if s not in allowed]
                if not invalid:
                    return selections, last_raw

                logging.warning(
                    f"Invalid LLM selections (attempt {attempt}/{max_semantic_retries}) "
                    f"for {trial_id} {inc_exc} {profile_stage}. Invalid: {invalid}"
                )

            filtered = [s for s in last_selections if s in allowed]
            if filtered:
                logging.warning(
                    f"Using filtered valid subset after failed retries for "
                    f"{trial_id} {inc_exc} {profile_stage}: {filtered}"
                )
                return filtered, last_raw

            logging.error(
                f"No valid LLM selections after {max_semantic_retries} attempts for "
                f"{trial_id} {inc_exc} {profile_stage}. Returning empty list."
            )
            return [], last_raw

        # --- Build prompts (batched if needed) and query LLM ---
        selections_all: List[str] = []

        # ensure trial mbench dir exists for logging
        def _ensure_trial_dir():
            if not no_mbench:
                td = mbench_dir / trial_id
                td.mkdir(parents=True, exist_ok=True)
                return td
            return None

        if len(canonical_variables) == 0:
            selections_all = []

        elif len(canonical_variables) <= 5:
            canon_json = json.dumps(canonical_variables, indent=2, ensure_ascii=False)
            template_text = finder.prompt_text

            if "#CANON_VARIABLES#" in template_text:
                prompt = template_text.replace("#CANON_VARIABLES#", canon_json)
            else:
                prompt = f"{template_text.rstrip()}\n\n#CANON_VARIABLES#\n{canon_json}"

            # Save prompt and later output (mbench)
            trial_prompt_dir = _ensure_trial_dir()
            if trial_prompt_dir is not None:
                try:
                    (trial_prompt_dir / f"{inc_exc}_prompt.txt").write_text(
                        prompt, encoding="utf-8"
                    )
                except Exception:
                    pass

            single_sel, raw_used = _call_llm_with_validation(
                prompt,
                canonical_subset=canonical_variables,
                profile_stage="VARIABLE_MUST_SUFFICE",
                trial_id=trial_id,
                inc_exc=inc_exc,
            )
            selections_all.extend(single_sel)

            # Save output (mbench)
            if trial_prompt_dir is not None:
                try:
                    (trial_prompt_dir / f"{inc_exc}_output.txt").write_text(
                        raw_used, encoding="utf-8"
                    )
                except Exception:
                    pass

        else:
            logging.info("Large trial; batched processing: %s", trial_id)
            template_text = finder.prompt_text
            batch_size = 5
            num_batches = (len(canonical_variables) + batch_size - 1) // batch_size

            for batch_idx, batch_vars in chunk_vars(canonical_variables, batch_size):
                canon_json = json.dumps(batch_vars, indent=2, ensure_ascii=False)

                if "#CANON_VARIABLES#" in template_text:
                    prompt = template_text.replace("#CANON_VARIABLES#", canon_json)
                else:
                    prompt = (
                        f"{template_text.rstrip()}\n\n"
                        f"You are now given batch {batch_idx + 1} of {num_batches} "
                        f"canonical variables for this trial. "
                        f"Only consider variables shown in this batch.\n\n"
                        f"#CANON_VARIABLES#\n{canon_json}"
                    )

                # Save per-batch prompt (mbench)
                trial_prompt_dir = _ensure_trial_dir()
                if trial_prompt_dir is not None:
                    try:
                        (trial_prompt_dir / f"{inc_exc}_prompt_batch{batch_idx + 1}.txt").write_text(
                            prompt, encoding="utf-8"
                        )
                    except Exception:
                        pass

                batch_sel, raw_used = _call_llm_with_validation(
                    prompt,
                    canonical_subset=batch_vars,
                    profile_stage=f"VARIABLE_MUST_SUFFICE_BATCH_{batch_idx + 1}",
                    trial_id=trial_id,
                    inc_exc=inc_exc,
                )
                selections_all.extend(batch_sel)

                # Save per-batch output (mbench)
                if trial_prompt_dir is not None:
                    try:
                        (trial_prompt_dir / f"{inc_exc}_output_batch{batch_idx + 1}.txt").write_text(
                            raw_used, encoding="utf-8"
                        )
                    except Exception:
                        pass

        # --- Merge & dedupe across batches ---
        variable_list = unique_preserve_order(selections_all)

        # Candidate variables that were actually shown to the LLM
        candidate_names: List[str] = [
            cv.get("entity_variable_name")
            for cv in canonical_variables
            if cv.get("entity_variable_name") is not None
        ]

        # Complement = “not chosen” variables
        not_chosen_names: List[str] = [
            name for name in candidate_names if name not in variable_list
        ]

        # --- Build output payload (main pipeline) ---
        # Keep this minimal to avoid breaking downstream consumers.
        payload = {
            "trial_id": trial_id,
            "inc_exc": inc_exc,
            "variable_must_suffice": variable_list,
        }

        # --- Write to main output ---
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # --- Optional mbench copy & meta ---
        if not no_mbench:
            try:
                trial_dir = mbench_dir / trial_id
                trial_dir.mkdir(parents=True, exist_ok=True)

                # mbench gets *extra* info: all candidates and complement.
                mbench_payload = {
                    "trial_id": trial_id,
                    "inc_exc": inc_exc,
                    "variable_candidates": candidate_names,
                    "variable_must_suffice": variable_list,
                    "variable_not_chosen": not_chosen_names,
                }

                with (trial_dir / f"{inc_exc}_must_suffice.json").open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(mbench_payload, f, ensure_ascii=False, indent=2)

                # Write/refresh meta with model+prompt hash
                meta_path = trial_dir / "mbench_meta.json"
                meta = {
                    "script_version": SCRIPT_VERSION,
                    "model": model,
                    "prompt_sha256": _sha256(prompt_text),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return (str(out_path), None)

    except Exception as e:
        # Bubble a concise message back to main
        return (cpath_str, f"{type(e).__name__}: {e}")


# ------------------------------ main ------------------------------ #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="LLM-backed selector for canonical variables (parallelized, conservative queueing) with optional positive-literal restriction"
    )
    p.add_argument("--canon-dir", default="../build/canon", type=Path,
                   help="Directory with *_canonical_variables.json files")
    p.add_argument("--output-dir", default="../build/default_vars", type=Path,
                   help="Where to write output JSON files")
    p.add_argument("--endpoint", default=os.environ.get("OPENAI_ENDPOINT"),
                   help="Inference endpoint URL (default: $OPENAI_ENDPOINT)")
    p.add_argument("--fallback-endpoint", default=None,
                   help="Optional secondary endpoint")
    p.add_argument("--model", default="gpt-4.1",
                   help="Model name (default: gpt-4.1)")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable on-disk JSONL cache")
    p.add_argument("--cache-path", type=Path,
                   default=Path("./.default_cache.jsonl"),
                   help="Cache file path")
    p.add_argument("--max-files", type=int, default=None,
                   help="Process only the first N canonical files")
    p.add_argument("--prompt-path", type=Path,
                   default=Path("./prompts/variable_must_suffice.prompt"),
                   help="Path to prompt template file")
    p.add_argument("--mbench-dir", type=Path, default=Path("./mbench"),
                   help="Directory for per-trial benchmark logs (default: ./mbench)")
    p.add_argument("--no-mbench", action="store_true",
                   help="Disable writing mbench logs")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of worker processes & in-flight jobs (default: 2x CPU)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip files whose output already exists")
    p.add_argument("--skip-uptodate", action="store_true",
                   help="Skip files whose output is newer than input AND matches model+prompt")

    # NEW: positive-literal integration flags
    p.add_argument(
        "--positive-dir",
        type=Path,
        default=Path("../build/positive_constraint_literals"),
        help="Directory containing positive literal outputs (summary.csv, per_file/...).",
    )
    p.add_argument(
        "--positives-required",
        default=True,
        action="store_true",
        help="If set, ONLY keep canonical variables whose stems are in the positive-literal set for that trial/arm.",
    )

    args = p.parse_args(argv)

    # --- per-run log file under ./run_logs ---
    run_log_dir = Path("./run_logs")
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_log_dir / f"defaults_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}.log"

    # Configure root logger: file (detailed) + console (concise)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(processName)s %(message)s",
                                 "%Y-%m-%d %H:%M:%S")
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")

    fh = logging.FileHandler(run_log_path, encoding="utf-8")
    fh.setFormatter(file_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(console_fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    # Redirect stdout/stderr to logging so stray prints are captured
    sys.stdout = _StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    logging.info("Run log file: %s", run_log_path)
    logging.info("Script version: %s", SCRIPT_VERSION)

    resolved_endpoint = args.endpoint or os.environ.get("OPENAI_ENDPOINT")
    if not resolved_endpoint:
        logging.error("No endpoint provided: set $OPENAI_ENDPOINT or pass --endpoint")
        return 2

    # Load prompt once here just for logging / sanity; workers will use this text
    prompt_text = load_prompt_template(args.prompt_path)
    logging.info(
        "Using prompt from: %s%s",
        args.prompt_path,
        " (fallback)" if prompt_text == PROMPT_FALLBACK else "",
    )

    # Collect ONLY inclusion files
    canon_files: List[Path] = []
    try:
        entries = sorted(args.canon_dir.iterdir())
    except FileNotFoundError:
        logging.error("Canon dir not found: %s", args.canon_dir)
        return 2

    for pth in entries:
        if not (pth.is_file() and pth.name.endswith("_canonical_variables.json")):
            continue
        try:
            _, inc_exc = parse_filename_parts(pth)
        except ValueError:
            continue
        if inc_exc == "inclusion":
            canon_files.append(pth)

    if args.max_files is not None:
        canon_files = canon_files[: args.max_files]

    if not canon_files:
        logging.error("No inclusion canonical files found in %s", args.canon_dir)
        return 2

    # Determine workers & in-flight cap
    if args.workers is None:
        cpu = os.cpu_count() or 1
        workers = max(1, cpu * 2)
    else:
        workers = max(1, args.workers)
    logging.info("Using %d worker process(es), %d max in-flight job(s)", workers, workers)

    # Load positives once and share to workers
    positive_map = _load_positive_stems(args.positive_dir)
    logging.info(
        "Loaded positive stems for %d trial/arm pairs from %s",
        len(positive_map), args.positive_dir
    )

    worker_kwargs = dict(
        resolved_endpoint=resolved_endpoint,
        fallback_endpoint=args.fallback_endpoint,
        model=args.model,
        no_cache=args.no_cache,
        cache_path=args.cache_path,
        prompt_text=prompt_text,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        mbench_dir=args.mbench_dir,
        no_mbench=args.no_mbench,
        skip_existing=args.skip_existing,
        skip_uptodate=args.skip_uptodate,
        # NEW:
        positive_map=positive_map,
        positives_required=args.positives_required,
    )

    written: List[str] = []
    errors: List[Tuple[str, str]] = []
    files_iter = iter(canon_files)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker_logging,
        initargs=(str(run_log_path),),
    ) as ex:
        pending: dict = {}

        # Helper to submit next file if available
        def submit_next() -> bool:
            try:
                cpath = next(files_iter)
            except StopIteration:
                return False
            fut = ex.submit(_worker_process_file, str(cpath), **worker_kwargs)
            pending[fut] = cpath
            return True

        # Prime up to `workers` tasks
        for _ in range(min(workers, len(canon_files))):
            if not submit_next():
                break

        # Process as tasks complete; keep at most `workers` in flight
        while pending:
            # Wait for at least one future to finish
            for fut in as_completed(list(pending.keys()), timeout=None):
                cpath = pending.pop(fut)
                cpath_str = str(cpath)
                try:
                    written_path, err = fut.result()
                    if err and str(err).startswith("SKIPPED"):
                        logging.info("%s %s", Path(written_path).name, err)
                    elif err:
                        errors.append((cpath_str, err))
                        logging.error("Failed: %s -> %s", Path(cpath_str).name, err)
                    else:
                        logging.info("Wrote %s", written_path)
                        written.append(written_path)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    errors.append((cpath_str, msg))
                    logging.error("Failed: %s -> %s", Path(cpath_str).name, msg)
                # After handling ONE completed future, break to allow refill
                break

            # Top up the queue to maintain at most `workers` in-flight
            while len(pending) < workers and submit_next():
                pass

    # Merge per-PID cache shards back into main cache file (if using cache)
    if not args.no_cache:
        base = args.cache_path
        shard_glob = f"{base.stem}.*{base.suffix}"
        shard_paths = sorted(base.parent.glob(shard_glob))
        if shard_paths:
            try:
                base.parent.mkdir(parents=True, exist_ok=True)
                with base.open("a", encoding="utf-8") as out_f:
                    for sp in shard_paths:
                        with sp.open("r", encoding="utf-8") as in_f:
                            for line in in_f:
                                out_f.write(line)
                        try:
                            sp.unlink(missing_ok=True)
                        except TypeError:
                            try:
                                sp.unlink()
                            except FileNotFoundError:
                                pass
                logging.info("Merged %d cache shard(s) into %s", len(shard_paths), base)
            except Exception as e:
                logging.warning("Failed to merge cache shards: %s", e)

    logging.info("Done. %d files written. %d error(s).", len(written), len(errors))
    logging.info("Full run log at: %s", run_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
