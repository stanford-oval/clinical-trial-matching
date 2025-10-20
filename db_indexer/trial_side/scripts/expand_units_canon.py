#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expand_units_canon.py
────────────────────────────────────────────────────────────────────────────
Reads all *_canonical_variables.json files in a given input directory,
identifies variables whose "usage_description" contains numeric references,
and sends them to an LLM prompt (defined in prompts/unit_expander.prompt)
to expand or normalize unit-related variable names.

The script:
  1. Recursively scans the input directory for *_canonical_variables.json files.
  2. Loads each file into memory while preserving its full structure.
  3. Deterministically detects numeric variables based on digits in their
     "usage_description" fields.
  4. Builds an LLM prompt with the detected variables and calls the LLM
     using the `_llm_call` helper.
  5. Parses the LLM output into (new_name, index) pairs and applies updates
     in-place — only replacing the "entity_variable_name" values, leaving all
     formatting, indentation, and line counts unchanged.
  6. Writes per-file logs (containing both the extracted numeric variables
     and the parsed LLM pairs) under a separate log directory.
  7. Always produces a mirrored JSON file in the output directory, even if
     no changes were made, to keep the directory structure synchronized.

This design guarantees:
  • The original JSON files remain structurally identical except for the
    modified variable names.
  • Each processing step (input load, LLM expansion, and rewrite) is fully
    traceable through a corresponding log file.
  • Safe recovery is possible via optional .bak backups for any in-place edits.
"""

import json
import re
import os
import concurrent.futures
import threading
import shutil
import sqlite3
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

# engine init
azure_endpoint: str = os.getenv("OPENAI_ENDPOINT", "")

from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
print(f"[info] Using ENGINE_VERSION={ENGINE_VERSION}, MODEL_NAME={MODEL_NAME}")
model_name = ENGINE_VERSION
run_id = "run"

if ENGINE_VERSION == "gpt-5":
    from smt_core.inference_engine_5 import AzureInferenceEngine
else:
    from smt_core.inference_engine import AzureInferenceEngine

_LLM_ENGINE = AzureInferenceEngine(
    endpoint=azure_endpoint,
    api_key_env_var="OPENAI_API_KEY",
    model_name=model_name,
)


def _llm_call(prompt: str) -> str:
    """Normalize engine output into a plain string."""
    if _LLM_ENGINE is None:
        raise RuntimeError("LLM engine not set. Call set_llm_engine(...) once.")
    out = _LLM_ENGINE(prompt)
    try:
        first = out[0]
        if isinstance(first, (list, tuple)):
            return str(first[0])
        return str(first)
    except Exception:
        return str(out)


# --------------------------- SQLite cache helpers ---------------------------

def _init_cache(db_path: Path) -> None:
    """
    Initialize the unit expansion cache.

    The cache key is (original_variable, model_name, prompt_version) so that
    changing model or prompt won't re-use stale expansions.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unit_expansion_cache (
                original_variable TEXT NOT NULL,
                model_name        TEXT NOT NULL,
                prompt_version    TEXT NOT NULL,
                expanded_variable TEXT NOT NULL,
                PRIMARY KEY (original_variable, model_name, prompt_version)
            )
            """
        )
        conn.commit()


def _cache_lookup(
    db_path: Path,
    original_variable: str,
    model_name: str,
    prompt_version: str,
) -> str | None:
    """
    Return cached expanded_variable if present, else None.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT expanded_variable
            FROM unit_expansion_cache
            WHERE original_variable = ?
              AND model_name        = ?
              AND prompt_version    = ?
            """,
            (original_variable, model_name, prompt_version),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _cache_upsert(
    db_path: Path,
    original_variable: str,
    expanded_variable: str,
    model_name: str,
    prompt_version: str,
) -> None:
    """
    Insert or update a cached expansion. Only call this on *successful*,
    validated expansions.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO unit_expansion_cache (
                original_variable,
                model_name,
                prompt_version,
                expanded_variable
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(original_variable, model_name, prompt_version)
            DO UPDATE SET expanded_variable = excluded.expanded_variable
            """,
            (original_variable, model_name, prompt_version, expanded_variable),
        )
        conn.commit()


# --------------------------- File discovery ---------------------------

def _iter_canon_files(in_dir: Path) -> Iterable[Path]:
    """Yield every file in `in_dir` whose filename ends with `_canonical_variables.json`."""
    yield from in_dir.rglob("*_canonical_variables.json")


# --------------------------- Parsing helpers ---------------------------

def _load_canonical_variables_file(p: Path) -> Dict[str, Any]:
    """
    Read and parse a single *_canonical_variables.json file into a dict:
      {
        "path": <Path>,
        "canonical_variables": [ { ...original keys... }, ... ]
      }
    """
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("canonical_variables", [])
    if not isinstance(items, list):
        raise ValueError(f"{p}: 'canonical_variables' must be a list.")

    parsed_items: List[Dict[str, Any]] = []
    for entry in items:
        if not isinstance(entry, dict):
            # skip non-dict entries to be safe
            continue
        e = dict(entry)  # shallow copy, preserve all original keys
        parsed_items.append(e)

    return {"path": p, "canonical_variables": parsed_items}


# --------------------------- Numeric-variable detector ---------------------------

_digit_re = re.compile(r"\d")


def find_withunit_variables(cvf: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Find variables whose entity_variable_name contains 'withunit'.
    Output format: [(entity_variable_name, canonical_variable_id), ...]
    """
    out: List[Tuple[str, int]] = []
    vars_list = cvf.get("canonical_variables", [])
    for idx, item in enumerate(vars_list):
        name = item.get("entity_variable_name", "")
        if isinstance(name, str) and "withunit" in name.lower():
            out.append((name, idx))

    return out


# --------------------------- LLM prompt I/O helpers ---------------------------

def _load_prompt(path: Path | str) -> str:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return f.read()


def _unwrap_code_fence(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    return m.group(1).strip() if m else s


def _parse_llm_unit_expansion(
    raw: str,
    expected_original: str | None = None,
) -> Tuple[str, str]:
    """
    Parse LLM output for a *single* variable.

    Expected LLM output (possibly wrapped in ```json fences):

        {
            "original_variable": "<original_variable_verbatim>",
            "unit_expanded_variable": "<unit_expanded_variable>"
        }

    Returns: (original_variable, unit_expanded_variable).

    If expected_original is provided, we verify it matches (after stripping
    whitespace); if not, a ValueError is raised.
    """

    def _unwrap_code_fence_local(s: str) -> str:
        s = s.strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I)
        return m.group(1).strip() if m else s

    txt = _unwrap_code_fence_local(raw)

    try:
        data = json.loads(txt)
    except Exception:
        # Try to grab the first JSON-looking block
        m = re.search(r"(\{.*\})", txt, flags=re.S)
        if not m:
            raise ValueError(f"Could not parse LLM output as JSON: {txt!r}")
        data = json.loads(m.group(1))

    # Accept either a bare dict or a list with one dict
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected LLM JSON type (expected dict): {type(data).__name__}")

    if "original_variable" not in data or "unit_expanded_variable" not in data:
        raise ValueError(
            "LLM JSON missing required keys 'original_variable' / "
            f"'unit_expanded_variable': {data}"
        )

    orig = str(data["original_variable"]).strip()
    expanded = str(data["unit_expanded_variable"]).strip()

    if expected_original is not None:
        if orig.strip() != expected_original.strip():
            raise ValueError(
                "original_variable mismatch.\n"
                f"  expected: {expected_original!r}\n"
                f"  got:      {orig!r}"
            )

    return orig, expanded


def _apply_expansion_inplace(cvf: Dict[str, Any], pairs: List[Tuple[str, str]]) -> None:
    """
    Apply LLM-produced renames in-place using (new_name, index) pairs.

    Each pair is:
        (new_entity_variable_name: str, canonical_variable_id: str|int)

    Behavior:
      - Converts index to int, validates range.
      - If the new name differs from the current one, sets
        `_original_entity_variable_name` once and updates `entity_variable_name`.
      - Later pairs for the same index override earlier ones (last-wins).
    """
    if not pairs:
        return

    vars_list = cvf.get("canonical_variables", [])
    n = len(vars_list)
    if not isinstance(vars_list, list) or n == 0:
        return

    for new_name, idx_raw in pairs:
        # Basic validation of new_name
        if not isinstance(new_name, str) or not new_name.strip():
            # Skip invalid names silently or log if you prefer:
            # print(f"[warn] Skipping invalid new_name={new_name!r}")
            continue

        # Coerce index → int
        try:
            idx = int(idx_raw)
        except Exception:
            # print(f"[warn] Bad index {idx_raw!r} (not int); skipping")
            continue

        # Range check
        if idx < 0 or idx >= n:
            # print(f"[warn] Index out of range: {idx} (0..{n-1}); skipping")
            continue

        item = vars_list[idx]
        old_name = item.get("entity_variable_name")

        # Only update if it actually changes
        if old_name != new_name:
            if "_original_entity_variable_name" not in item:
                item["_original_entity_variable_name"] = old_name
            item["entity_variable_name"] = new_name


def _write_canonical_variables_file(out_dir: Path, in_dir: Path, cvf: Dict[str, Any]) -> Path:
    """
    Write JSON with the same outer structure and filename under out_dir,
    preserving any subdirectory structure relative to in_dir.
    """
    src_path: Path = cvf["path"]
    rel = src_path.relative_to(in_dir)
    dst_path = (out_dir / rel).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Recreate the original envelope shape: {"canonical_variables": [...]}
    payload = {
        "canonical_variables": cvf.get("canonical_variables", [])
    }
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return dst_path


# --------------------------- logging ---------------------------

def _log_path_for(
    out_log_dir: Path,
    in_dir: Path,
    cvf: Dict[str, Any],
    suffix: str = "_raw.txt",
) -> Path:
    """
    Compute the log path under out_log_dir preserving the relative structure
    from in_dir, and appending `suffix` to the base name.
    """
    src_path: Path = cvf["path"]
    rel = src_path.relative_to(in_dir)
    fn = rel.name
    stem = fn.rsplit(".", 1)[0]  # drop only last extension
    log_rel = rel.with_name(f"{stem}{suffix}")  # e.g., foo.json -> foo_raw.txt
    return (out_log_dir / log_rel).resolve()


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _validate_single_expansion(orig_name: str, expanded_name: str) -> None:
    """
    Validate that the expanded variable name is consistent with the original.

    Rule:
      - Split both names on '_withunit_' and compare the prefix (part[0]).
      - If prefixes differ, raise ValueError so the caller can retry.

    Example:
      orig:  patient_weight_value_recorded_now_withunit_kg
      new:   patient_weight_value_recorded_now_withunit_kilograms

      -> prefixes are both 'patient_weight_value_recorded_now'
         => valid
    """
    orig_prefix = orig_name.split("_withunit_", 1)[0]
    new_prefix = expanded_name.split("_withunit_", 1)[0]

    if orig_prefix != new_prefix:
        raise ValueError(
            "prefix mismatch between original and expanded variable\n"
            f"  original: {orig_name}\n"
            f"  expanded: {expanded_name}\n"
            f"  orig_prefix: {orig_prefix}\n"
            f"  new_prefix : {new_prefix}"
        )


def _validate_llm_pairs_sequential(
    out_list: List[Tuple[str, int]],
    pairs: List[Tuple[str, str]],
) -> Tuple[bool, str]:
    """
    Validate LLM output `pairs` against `out_list`, item by item.

    Checks:
      1) Same number of items
      2) For each i:
         - indices match
         - orig_name == new_name split by '_withunit_' prefix
    Returns (ok, message).
    """
    # 1) same length
    if len(out_list) != len(pairs):
        return False, f"length mismatch: out_list={len(out_list)} pairs={len(pairs)}"

    if not out_list and not pairs:
        return True, "both empty"

    # 2) item-by-item checking
    for i, ((orig_name, orig_idx), (new_name, idx_str)) in enumerate(zip(out_list, pairs)):
        # Index match
        try:
            out_idx = int(idx_str)
        except ValueError:
            return False, f"item {i}: index '{idx_str}' is not an integer"

        if out_idx != orig_idx:
            return False, (
                f"item {i}: index mismatch:\n"
                f"  expected index: {orig_idx}\n"
                f"  got index:      {out_idx}"
            )

        # Prefix comparison using split("_withunit_")
        orig_prefix = orig_name.split("_withunit_", 1)[0]
        new_prefix = new_name.split("_withunit_", 1)[0]

        if orig_prefix != new_prefix:
            return False, (
                f"item {i}: name prefix mismatch:\n"
                f"  orig prefix: {orig_prefix}\n"
                f"  new  prefix: {new_prefix}"
            )

    return True, "ok"


# Optional lock if your LLM engine isn't thread-safe
_LLM_LOCK = threading.Lock()


def _process_one(
    cvf: Dict[str, Any],
    *,
    in_dir: Path,
    out_dir: Path,
    out_log_dir: Path,
    prompt_tmpl: str,
    dry_run: bool,
    llm_serial: bool = False,   # set True if your LLM client isn't thread-safe
    ir_in_dir: Path | None = None,
    ir_out_dir: Path | None = None,
    cache_db_path: Path | None = None,
    prompt_version: str | None = None,
) -> Dict[str, Any]:
    """
    Worker: process one canonical file dict.
    Returns a small result dict for progress/summary.
    Also, if ir_in_dir/out_dir are provided, renames variables in the
    corresponding SMT IR file and writes to ir_out_dir.
    """
    res: Dict[str, Any] = {
        "file": str(cvf["path"]),
        "ok": False,
        "num_vars": len(cvf.get("canonical_variables", [])),
        "num_pairs": 0,
        "error": None,
    }
    try:
        # 1) deterministic extraction of numeric vars
        out_list = find_withunit_variables(cvf)  # [(entity_variable_name, idx), ...]
        # Build index -> old_name mapping BEFORE we mutate cvf
        idx_to_old: Dict[int, str] = {idx: name for (name, idx) in out_list}

        pairs: List[Tuple[str, str]] = []       # LLM parsed pairs (new_name, idx)
        rename_map_smt: Dict[str, str] = {}
        unchanged_unit: List[str] = []

        # 2) LLM call only if we have something to expand
        if out_list:
            # For each (orig_name, idx), call cache first, then LLM (if needed)
            for orig_name, orig_idx in out_list:
                max_attempts = 3
                attempt = 0
                expanded_name: str | None = None

                # 0) Try cache
                if cache_db_path is not None and prompt_version is not None:
                    cached = _cache_lookup(
                        cache_db_path,
                        original_variable=orig_name,
                        model_name=MODEL_NAME,
                        prompt_version=prompt_version,
                    )
                    if cached is not None:
                        try:
                            _validate_single_expansion(orig_name, cached)
                            expanded_name = cached
                            # print(f"[cache-hit] {orig_name!r} -> {cached!r}")
                        except Exception as e:
                            print(
                                f"[warn] Invalid cache entry for {orig_name!r}; "
                                f"ignoring and falling back to LLM: {e}"
                            )
                            expanded_name = None

                # 1) If no valid cache hit, fall back to LLM
                while expanded_name is None and attempt < max_attempts:
                    attempt += 1

                    # You can customize this; here we just drop the single variable in
                    full_prompt = prompt_tmpl.replace(
                        "##UNIT_TEXT##",
                        json.dumps(
                            {"original_variable": orig_name},
                            ensure_ascii=False,
                        ),
                    )

                    try:
                        if llm_serial:
                            with _LLM_LOCK:
                                raw = _llm_call(full_prompt)
                        else:
                            raw = _llm_call(full_prompt)

                        # Parse single-variable JSON
                        # also check if original == output original
                        _orig_check, expanded = _parse_llm_unit_expansion(
                            raw,
                            expected_original=orig_name,
                        )

                        # check prefix (variable before _withunit_) are the same. if not, raise error
                        _validate_single_expansion(orig_name, expanded)
                        expanded_name = expanded

                        # 2) On success, upsert into cache
                        if cache_db_path is not None and prompt_version is not None:
                            _cache_upsert(
                                cache_db_path,
                                original_variable=orig_name,
                                expanded_variable=expanded_name,
                                model_name=MODEL_NAME,
                                prompt_version=prompt_version,
                            )

                    except Exception as e:
                        print(
                            f"[warn] LLM call/parse failed for {Path(res['file']).name} "
                            f"var={orig_name!r} (attempt {attempt}/{max_attempts}): {e}"
                        )
                        expanded_name = None

                if expanded_name is None:
                    print(
                        f"[error] Giving up on variable {orig_name!r} in "
                        f"{Path(res['file']).name} after {max_attempts} attempts."
                    )
                    # Skip this variable; do not append to pairs
                    continue

                # Success: add to pairs with the original index
                pairs.append((expanded_name, str(orig_idx)))

            # After building all pairs, run existing sequential validator
            ok_pairs, msg = _validate_llm_pairs_sequential(out_list, pairs)
            if not ok_pairs:
                print(
                    f"[warn] LLM unit-expansion validation failed for "
                    f"{Path(res['file']).name} after per-variable calls:\n  {msg}"
                )
                pairs = []
                rename_map_smt = {}
            else:
                # Build rename_map_smt using your unit-suffix logic
                rename_map_smt = {}
                for (orig_name, orig_idx), (new_name, idx_str) in zip(out_list, pairs):
                    if new_name != orig_name:
                        unit1 = "_withunit_" + orig_name.split("_withunit_", 1)[1]
                        unit2 = "_withunit_" + new_name.split("_withunit_", 1)[1]
                        rename_map_smt[unit1] = unit2

        else:
            print(f"[info] {Path(res['file']).name}: no numeric variables detected (out_list empty)")

        # 3) Build rename map and apply expansions (even if pairs == [])
        rename_map: Dict[str, str] = {}
        if pairs:
            # new_name, idx -> old_name from idx_to_old
            for new_name, idx in pairs:
                old_name = idx_to_old.get(int(idx))
                if isinstance(old_name, str) and old_name != new_name:
                    rename_map[old_name] = new_name

            # Apply to canonical variables in-place
            _apply_expansion_inplace(cvf, pairs)

        res["num_pairs"] = len(pairs)
        res["rename_map_smt"] = rename_map_smt
        res["unchanged_unit"] = unchanged_unit

        # 3b) If IR dirs are provided, also rename in SMT IR and write to ir_out_dir
        _rewrite_smt_for_cvf(
            cvf=cvf,
            rename_map_smt=rename_map_smt,
            ir_in_dir=ir_in_dir,
            ir_out_dir=ir_out_dir,
        )

        # 4) Always write:
        #    - the log (out + pairs) under out_dir/normalize_log
        #    - the JSON under out_dir (mirrored filename), even if unchanged
        log_payload = {
            "file": str(cvf["path"]),
            "numerical_variables_extracted": out_list,
            "expanded": pairs,
            "rename_map_smt": rename_map_smt,
            "unchanged_unit": unchanged_unit,
        }
        log_path = _log_path_for(out_log_dir, in_dir, cvf, suffix="_raw.txt")
        _write_text(log_path, json.dumps(log_payload, ensure_ascii=False, indent=2))

        dst = _write_canonical_variables_file(out_dir, in_dir, cvf)

        print(f"[wrote]{'-dry' if dry_run else ''} log={log_path}")
        print(f"[wrote]{'-dry' if dry_run else ''} json={dst}")

        res["ok"] = True
        res["log_path"] = str(log_path)
        res["out_path"] = str(dst)
        return res

    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        print(f"[fail] {Path(res['file']).name}: {res['error']}")
        return res


# --------------------------- SMT writer helper ---------------------------

def _rewrite_smt_for_cvf(
    cvf: Dict[str, Any],
    rename_map_smt: Dict[str, str],
    ir_in_dir: Path | None,
    ir_out_dir: Path | None,
) -> None:
    """
    Given a canonical-variables file dict `cvf` and a rename_map_smt {old->new},
    find the corresponding SMT file:

        ir_in_dir / "<trial_id>_<side>_program.smt2"

    where <side> is "inclusion" or "exclusion" inferred from the cvf path.

    If rename_map_smt is empty: copy SMT from ir_in_dir to ir_out_dir.
    Otherwise: replace unit suffixes ONLY when they appear at the end of an
    SMT identifier (optionally followed by @@decorator), and write result to
    ir_out_dir.
    """
    if ir_in_dir is None or ir_out_dir is None:
        return

    cvf_path = Path(cvf["path"])
    fname = cvf_path.name  # e.g. "NCT00000000_inclusion_canonical_variables.json"

    # Strip extension
    if fname.endswith(".json"):
        fname_core = fname[:-5]
    else:
        fname_core = fname

    # Strip "_canonical_variables" suffix
    suffix = "_canonical_variables"
    if fname_core.endswith(suffix):
        base = fname_core[: -len(suffix)]
    else:
        base = fname_core  # fallback

    # Expect "<trial_id>_<side>"
    parts = base.split("_", 2)  # allow extra underscores after side if ever needed
    if len(parts) < 2:
        print(f"[ir] WARNING: cannot infer trial_id/side from {fname}")
        return

    trial_id = parts[0]
    side = parts[1]  # expected "inclusion" or "exclusion"

    smt_name = f"{trial_id}_{side}_program.smt2"
    smt_src = (ir_in_dir / smt_name).resolve()
    smt_dst = (ir_out_dir / smt_name).resolve()

    if not smt_src.exists():
        print(f"[ir] WARNING: SMT file not found for {fname}: {smt_src}")
        return

    smt_dst.parent.mkdir(parents=True, exist_ok=True)

    # If no renames, just copy (and avoid SameFileError if dirs are same)
    if not rename_map_smt:
        if smt_src != smt_dst:
            shutil.copy2(smt_src, smt_dst)
            print(f"[ir] copied (no renames): {smt_src.name} -> {smt_dst}")
        else:
            print(f"[ir] no renames, leaving SMT as-is: {smt_src}")
        return

    # Otherwise, read / replace / write
    text = smt_src.read_text(encoding="utf-8")

    # To reduce surprises, apply longer suffixes first
    # e.g. "_withunit_percent_predicted" before "_withunit_percent"
    for old in sorted(rename_map_smt.keys(), key=len, reverse=True):
        new = rename_map_smt[old]

        # Match:
        #   <prefix>[A-Za-z0-9_]*  +  old  +  optional @@decorator  +  word-boundary
        #
        # Examples matched:
        #   patient_..._withunit_percent
        #   patient_..._withunit_percent@@measured_by_multigated_acquisition_scan
        #
        # Not matched for "_withunit_percent":
        #   patient_..._withunit_percent_predicted
        #
        pattern = re.compile(
            r"(?P<prefix>[A-Za-z0-9_]+)"
            + re.escape(old) +
            r"(?P<decorator>(?:@@[A-Za-z0-9_]+)?)\b"
        )

        def _repl(m: re.Match) -> str:
            return m.group("prefix") + new + m.group("decorator")

        text = pattern.sub(_repl, text)

    smt_dst.write_text(text, encoding="utf-8")
    print(
        f"[ir] rewrote {smt_src.name} -> {smt_dst} "
        f"(renamed {len(rename_map_smt)} unit suffixes)"
    )


# --------------------------- run() ---------------------------

def run(
    in_dir: Path | str = "../../build/canon",
    out_dir: Path | str = "../../build/canon_expanded",
    out_log_dir: Path | str = "mbench/canon_expanded_log",
    dry_run: bool = False,
    prompt_path: Path | str = "scripts/prompts/unit_expander.prompt",
    max_workers: int | None = None,
    llm_serial: bool = False,   # set True if your LLM client isn't thread-safe
    ir_in_dir: Path | str = "../../build/ir_merged",
    ir_out_dir: Path | str = "../../build/ir_expanded",
    cache_db_path: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """
    - Load prompt from `prompt_path`
    - Load all *_canonical_variables.json
    - Find unit-bearing variables (entity_variable_name contains 'withunit')
    - LLM expand unit-bearing variable names
    - Parse LLM output, apply renames in-place (same data structure)
    - Log 'out' and 'pairs' under out_log_dir
    - Write files with same names under out_dir (even if unchanged)
    - Process files concurrently with a thread pool
    """
    in_dir = Path(in_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_log_dir = Path(out_log_dir).expanduser().resolve()
    ir_in_dir = Path(ir_in_dir).expanduser().resolve()
    ir_out_dir = Path(ir_out_dir).expanduser().resolve()

    # Make sure directories exist
    out_dir.mkdir(parents=True, exist_ok=True)
    out_log_dir.mkdir(parents=True, exist_ok=True)
    ir_out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir does not exist: {in_dir}")

    prompt_tmpl = _load_prompt(prompt_path)

    # Stable identifier for this prompt version (short SHA-256)
    prompt_version = hashlib.sha256(prompt_tmpl.encode("utf-8")).hexdigest()[:16]

    # Cache DB location (default under out_log_dir)
    if cache_db_path is None:
        cache_db_path = Path("../../ontology_curate/unit_expansion.db")
    else:
        cache_db_path = Path(cache_db_path)

    cache_db_path = cache_db_path.expanduser().resolve()
    _init_cache(cache_db_path)

    files = sorted(_iter_canon_files(in_dir))
    print(f"[info] Found {len(files)} canonical files under {in_dir}")

    # Load all files (I/O bound; fast)
    loaded: List[Dict[str, Any]] = []
    for i, p in enumerate(files, 1):
        try:
            cvf = _load_canonical_variables_file(p)
            loaded.append(cvf)
            print(f"[ok] ({i}/{len(files)}) {p.name}: {len(cvf['canonical_variables'])} variables")
        except Exception as e:
            print(f"[warn] Failed to parse {p}: {e}")

    if not loaded:
        print("[info] No files to process.")
        return []

    # Choose worker count
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu)

    results: List[Dict[str, Any]] = []
    print(f"[info] Processing with max_workers={max_workers} llm_serial={llm_serial}")

    # Run per-file in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                _process_one,
                cvf,
                in_dir=in_dir,
                out_dir=out_dir,
                out_log_dir=out_log_dir,
                prompt_tmpl=prompt_tmpl,
                dry_run=dry_run,
                llm_serial=llm_serial,
                ir_in_dir=ir_in_dir,
                ir_out_dir=ir_out_dir,
                cache_db_path=cache_db_path,
                prompt_version=prompt_version,
            )
            for cvf in loaded
        ]
        for fut in concurrent.futures.as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                res = {"file": "<unknown>", "ok": False, "error": f"{type(e).__name__}: {e}"}
            results.append(res)

    # Summary
    ok = sum(1 for r in results if r.get("ok"))
    total_pairs = sum(r.get("num_pairs", 0) for r in results)
    total = len(results)
    failures = total - ok

    print(f"[summary] processed={total} ok={ok} failed={failures} total_pairs={total_pairs}")
    print(f"[info] Complete. out_dir: {out_dir}")
    print(f"[info] Logs under: {out_log_dir}")

    # ----------------- unit_expansion_summary.jsonl -----------------
    summary_path = out_log_dir / "unit_expansion_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for r in results:
            file_path = r.get("file") or ""
            base = Path(file_path).name

            # Infer trial_id and side from canonical filename:
            #   NCT00000000_inclusion_canonical_variables.json
            #   NCT00000000_exclusion_canonical_variables.json
            if base.endswith(".json"):
                stem = base[:-5]
            else:
                stem = base

            core = stem
            suffix = "_canonical_variables"
            if core.endswith(suffix):
                core = core[: -len(suffix)]

            trial_id = None
            side = None
            parts = core.split("_", 2)
            if len(parts) >= 2:
                trial_id, side = parts[0], parts[1]

            rec = {
                "trial_id": trial_id,
                "side": side,                      # "inclusion" / "exclusion" (if parseable)
                "ok": bool(r.get("ok")),
                "num_pairs": r.get("num_pairs", 0),
                "error": r.get("error"),          # None if success
                "rename_map_smt": r.get("rename_map_smt", {}),
                "unchanged_unit": r.get("unchanged_unit", []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Final summary record
        summary_rec = {
            "summary": True,
            "total_files": total,
            "ok": ok,
            "failed": failures,
            "total_pairs": total_pairs,
        }
        f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")

    print(f"[info] Wrote unit_expansion_summary.jsonl -> {summary_path}")
    # ---------------------------------------------------------------
    return results


if __name__ == "__main__":
    run()
