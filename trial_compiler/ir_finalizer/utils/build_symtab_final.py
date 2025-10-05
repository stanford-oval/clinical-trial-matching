#!/usr/bin/env python3
"""
build_symtab_final_from_ir_dir.py

Batch-build symtab_final (variable_index.json) from finalized canonical SMT2 programs.

Defaults:
  --ir-dir   <SATIR_ROOT>/build/ir_all_final
  --out-dir  <SATIR_ROOT>/build/symtab_final

This script:
- Reads each NCT*_program.smt2
- Extracts every (declare-const <name> <sort>) line + its inline JSON object
- If decl_json.meaning is missing/empty/non-string, infer it from variable name (same heuristic as fixer)
- Writes <trial>_<side>_variable_index.json into symtab_final
- Writes symtab_final/_summary.json

Place this script next to orchestrate_ir_fixes.py so we can reuse its regex/helpers.

Usage:
  python3 build_symtab_final_from_ir_dir.py
  # or
  python3 build_symtab_final_from_ir_dir.py --ir-dir .../build/ir_all_final --out-dir .../build/symtab_final
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from trial_compiler.ir_finalizer.orchestrators.orchestrate_ir_fixes import (  # type: ignore
        CANON_RE,
        RE_DECLARE_CONST_HDR,
        code_part_before_comment,
        declare_const_in_code,
        extract_first_json_object_from_line,
        ensure_dir,
        iter_canonical_ir_files,
    )
except Exception as e:
    raise SystemExit(
        f"[FATAL] Failed to import from orchestrate_ir_fixes.py. "
        f"Place this script next to orchestrate_ir_fixes.py. Error: {e}"
    )


DEFAULT_IR_DIR = os.getenv("SATIR_BUILD", "build") + "/ir_all_final"
DEFAULT_OUT_DIR = os.getenv("SATIR_BUILD", "build") + "/symtab_final"

QUAL_SPLIT_RE = re.compile(r"@@+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build symtab_final from canonical SMT2 files.")
    p.add_argument(
        "--ir-dir",
        default=DEFAULT_IR_DIR,
        help=f"Directory containing canonical NCT*_program.smt2. Default: {DEFAULT_IR_DIR}",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for *_variable_index.json. Default: {DEFAULT_OUT_DIR}",
    )
    p.add_argument(
        "--fail-on-any-error",
        action="store_true",
        help="If set, abort if any file has a parse error; otherwise record errors and continue.",
    )
    return p.parse_args()


def infer_meaning_from_var(var_name: str) -> str:
    s = (var_name or "").strip()
    if not s:
        return ""

    parts = QUAL_SPLIT_RE.split(s)
    stem = parts[0]
    quals = parts[1:] if len(parts) > 1 else []

    for pref in ("patient_", "trial_", "subject_", "participant_"):
        if stem.startswith(pref):
            stem = stem[len(pref):]

    stem = stem.replace("__", "_").strip("_")
    words = [w for w in stem.split("_") if w]
    phrase = " ".join(words)

    if quals:
        q_words: List[str] = []
        for q in quals:
            q = (q or "").strip("_")
            q_words.extend([w for w in q.split("_") if w])
        if q_words:
            phrase = f"{phrase} ({' '.join(q_words)})" if phrase else f"({' '.join(q_words)})"

    return phrase if phrase else s


def _canonical_pair_from_filename(name: str) -> Optional[Tuple[str, str]]:
    m = CANON_RE.match(name)
    if not m:
        return None
    return (m.group(1), m.group(2))


def _parse_smt_to_variable_index(smt_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns:
      variable_index dict mapping var_name -> entry
      errors list (strings)
    """
    errors: List[str] = []
    var_index: Dict[str, Any] = {}

    try:
        text = smt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {}, [f"read_error:{e}"]

    for ln_no, line in enumerate(text.splitlines(), start=1):
        if not declare_const_in_code(line):
            continue

        code = code_part_before_comment(line).rstrip()
        m = RE_DECLARE_CONST_HDR.match(code)
        if not m:
            errors.append(f"line{ln_no}:declare_const_bad_header")
            continue
        var_name = m.group(1)
        sort = m.group(2).strip()

        jb = extract_first_json_object_from_line(line)
        if not jb:
            errors.append(f"line{ln_no}:{var_name}:missing_inline_json")
            continue
        try:
            obj = json.loads(jb)
        except Exception as e:
            errors.append(f"line{ln_no}:{var_name}:json_parse_failed:{e}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line{ln_no}:{var_name}:json_not_object")
            continue

        # backfill meaning if missing
        meaning = obj.get("meaning", None)
        if (not isinstance(meaning, str)) or (not meaning.strip()):
            obj["meaning"] = infer_meaning_from_var(var_name)

        var_index[var_name] = {
            "sort": sort,
            "decl_json": obj,
            "declare_const_line": code,
            "line_no": ln_no,
        }

    return var_index, errors


def main() -> None:
    args = parse_args()
    in_dir = Path(args.ir_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    summary: Dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(),
        "input_ir_dir": str(in_dir),
        "output_symtab_dir": str(out_dir),
        "files_total": 0,
        "files_ok": 0,
        "files_with_errors": 0,
        "total_decls": 0,
        "errors": [],
    }

    if not in_dir.exists():
        raise FileNotFoundError(f"--ir-dir does not exist: {in_dir}")

    for smt_path in iter_canonical_ir_files(in_dir):
        pair = _canonical_pair_from_filename(smt_path.name)
        if not pair:
            continue
        trial_id, side = pair
        summary["files_total"] += 1

        var_index, errs = _parse_smt_to_variable_index(smt_path)
        summary["total_decls"] += len(var_index)

        payload = {
            "trial_id": trial_id,
            "side": side,
            "generated_at": dt.datetime.now().isoformat(),
            "source_smt_path": str(smt_path),
            "num_decls": len(var_index),
            "variable_index": {k: var_index[k] for k in sorted(var_index.keys())},
        }

        out_path = out_dir / f"{trial_id}_{side}_variable_index.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if errs:
            summary["files_with_errors"] += 1
            summary["errors"].append({"file": str(smt_path), "trial_id": trial_id, "side": side, "errors": errs})
            if args.fail_on_any_error:
                raise SystemExit(f"[ABORT] Errors in {smt_path}:\n" + "\n".join(errs))
        else:
            summary["files_ok"] += 1

    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Wrote symtab_final to:", out_dir)
    print("[DONE] Summary:", out_dir / "_summary.json")
    print(
        f"[DONE] files_total={summary['files_total']} files_ok={summary['files_ok']} "
        f"files_with_errors={summary['files_with_errors']} total_decls={summary['total_decls']}"
    )


if __name__ == "__main__":
    main()
