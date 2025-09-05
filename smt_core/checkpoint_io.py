# checkpoint_io.py
# unified, JSON-safe checkpoint helpers with deep sanitization

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import os

"""
Key behaviors
-------------
• Deeply sanitize contexts before saving:
  – remove any 'engine' fields at ANY depth
  – drop callables
  – Path → str, set/tuple → list (deterministic)
  – unknown objects → str(obj)
• Inject a `_saved_at` ISO-8601 timestamp for provenance
• Keep loader API that re-inserts the runtime `engine` at the root
"""

# -------------------------------------------------------------------
#  default locations (used only by CLI/tests – pipeline passes paths)
# -------------------------------------------------------------------
PREPROC_FILE   = Path("checkpoints/preproc/preproc_context.chkpt.json")
CANON_FILE     = Path("checkpoints/canon/canon_context.chkpt.json")
NON_CANON_FILE = Path("checkpoints/noncanon/noncanon_context.chkpt.json")
ATTR_FILE      = Path("checkpoints/attr/attr_context.chkpt.json")
PROGRAM_FILE   = Path("checkpoints/program/program_context.chkpt.json")
FINAL_FILE     = Path("checkpoints/final/final_context.chkpt.json")

# -------------------------------------------------------------------
#  deep sanitization
# -------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
    """Return a JSON-serializable version of obj.

    Rules:
      - primitives pass through
      - Path -> str
      - set/tuple -> list (sorted for sets)
      - dict -> dict (recursively sanitized), drops 'engine' key and callables
      - list -> list (recursively sanitized)
      - fallback -> str(obj)
    """
    # primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # set / tuple
    if isinstance(obj, set):
        return [_jsonify(v) for v in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]

    # list
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]

    # dict
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            # strip engines at ANY depth
            if k == "engine":
                continue
            # drop callables (LLM call hooks, lambdas, etc.)
            if callable(v):
                continue
            out[str(k)] = _jsonify(v)
        return out

    # fallback: stable string representation
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable:{type(obj).__name__}>"

# Public alias so other modules can reuse the sanitizer
json_sanitize = _jsonify

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
#  generic save / load templates
# -------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically to avoid torn checkpoint files."""
    _ensure_parent(path)
    dir_ = str(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)  # atomic on POSIX

def _save_ckpt(ctx: Dict[str, Any], file: Path | str) -> None:
    file = Path(file)
    payload = _jsonify(ctx)
    # provenance timestamp at the root
    if isinstance(payload, dict):
        payload["_saved_at"] = dt.datetime.now().isoformat(timespec="seconds")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_text(file, text)
    print(f"[✓] Saved checkpoint → {file}")

def _load_ckpt(engine, file: Path | str) -> Optional[Dict[str, Any]]:
    file = Path(file)
    if not file.exists():
        return None
    try:
        ctx = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(ctx, dict):
            ctx = {"_root": ctx}
        # Reattach the runtime engine only at the ROOT (never in subtrees)
        ctx["engine"] = engine
        print(f"[↻] Loaded checkpoint ← {file}")
        return ctx
    except Exception as exc:
        print(f"[!] Failed to load checkpoint {file}: {exc}")
        return None

# -------------------------------------------------------------------
#  PUBLIC APIS – thin wrappers for clarity / future hooks
# -------------------------------------------------------------------

def save_preproc_ckpt(ctx: Dict[str, Any], file: Path | str = PREPROC_FILE):
    _save_ckpt(ctx, file)

def load_preproc_ckpt(engine, file: Path | str = PREPROC_FILE):
    return _load_ckpt(engine, file)

def save_canon_ckpt(ctx: Dict[str, Any], file: Path | str = CANON_FILE):
    _save_ckpt(ctx, file)

def load_canon_ckpt(engine, file: Path | str = CANON_FILE):
    return _load_ckpt(engine, file)

def save_noncanon_ckpt(ctx: Dict[str, Any], file: Path | str = NON_CANON_FILE):
    _save_ckpt(ctx, file)

def load_noncanon_ckpt(engine, file: Path | str = NON_CANON_FILE):
    return _load_ckpt(engine, file)

def save_attr_ckpt(ctx: Dict[str, Any], file: Path | str = ATTR_FILE):
    _save_ckpt(ctx, file)

def load_attr_ckpt(engine, file: Path | str = ATTR_FILE):
    return _load_ckpt(engine, file)

def save_program_ckpt(ctx: Dict[str, Any], file: Path | str = PROGRAM_FILE):
    """Persist *post-programmer* context (includes partial SMT & stats)."""
    _save_ckpt(ctx, file)

def load_program_ckpt(engine, file: Path | str = PROGRAM_FILE):
    return _load_ckpt(engine, file)

def save_final_ckpt(ctx: Dict[str, Any], file: Path | str = FINAL_FILE):
    _save_ckpt(ctx, file)

def load_final_ckpt(engine, file: Path | str = FINAL_FILE):
    return _load_ckpt(engine, file)
