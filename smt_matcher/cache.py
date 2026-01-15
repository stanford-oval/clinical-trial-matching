"""
Prompt / stage fingerprinting and caching for SMT matcher.

Extracted from cmsrc/match_patient_to_trial.py. Lets stages skip LLM
calls when the inputs (prompt text, patient text, trial context, miner
mode) are identical to a prior cached run.

Usage:
    from smt_matcher.cache import (
        CACHE_SCHEMA_VERSION, stable_hash, cache_file_for,
        safe_read_json, safe_write_json_atomic, cache_payload_matches,
    )

    fp = stable_hash({"prompt": prompt_text, "patient": patient_text, "trial_id": tid})
    path = cache_file_for(out_root, stage="llm_judge", fingerprint=fp)
    cached = safe_read_json(path)
    if cache_payload_matches(cached, fp):
        return cached["result"]
    result = call_llm(...)
    safe_write_json_atomic(path, {"cache": {"schema_version": CACHE_SCHEMA_VERSION,
                                              "fingerprint": fp},
                                   "result": result})
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import re as _re
from typing import Any, Dict, Optional

# Bumping this invalidates all existing caches.
CACHE_SCHEMA_VERSION = "2026-04-15-refactored-v1"


# ── JSON helpers ─────────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, set):
        try:
            return sorted(o)
        except TypeError:
            return list(o)
    if isinstance(o, pathlib.Path):
        return str(o)
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if isinstance(o, _re.Pattern):
        return o.pattern
    return str(o)


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_json_default)


def stable_hash(obj: Any) -> str:
    """SHA-256 hex digest of a stable JSON serialization."""
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def safe_read_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_write_json_atomic(path: pathlib.Path, obj: Any) -> bool:
    """Write JSON atomically via a tmp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        tmp.replace(path)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


# ── Cache path helpers ────────────────────────────────────────────────────

def cache_root_for(out_root: pathlib.Path) -> pathlib.Path:
    return out_root / "_prompt_cache"


def cache_file_for(out_root: pathlib.Path, stage: str, fingerprint: str) -> pathlib.Path:
    return cache_root_for(out_root) / stage / f"{fingerprint}.json"


def cache_payload_matches(payload: Optional[Dict[str, Any]], fingerprint: str) -> bool:
    """True iff `payload` is a cache envelope whose fingerprint matches."""
    if not isinstance(payload, dict):
        return False
    cache = payload.get("cache")
    if not isinstance(cache, dict):
        return False
    return (
        cache.get("schema_version") == CACHE_SCHEMA_VERSION
        and cache.get("fingerprint") == fingerprint
    )


def make_cache_envelope(result: Any, fingerprint: str) -> Dict[str, Any]:
    """Wrap a result for caching."""
    return {
        "cache": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
        },
        "result": result,
    }


def fingerprint_ctx(ctx: Dict[str, Any], *, fields: Optional[list] = None) -> str:
    """
    Compute a fingerprint for the relevant parts of a match context.
    By default hashes: trial_id, side, patient_note_text, miner_mode, prompt_text.
    """
    if fields is None:
        fields = [
            "trial_id", "side", "patient_note_text", "miner_mode",
            "SMTVariableValueMinerInclusion_prompt",
            "SMTVariableValueMinerExclusion_prompt",
        ]
    subset = {k: ctx.get(k) for k in fields if k in ctx}
    return stable_hash(subset)
