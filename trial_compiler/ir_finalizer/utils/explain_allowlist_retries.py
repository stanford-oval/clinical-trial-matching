#!/usr/bin/env python3
"""
explain_allowlist_retries.py

Usage:
  python explain_allowlist_retries.py \
    --allowlist /path/to/stage1_polarity/allowlist_remaining_..._attempt1.txt \
    --stage-dir /path/to/stage1_polarity

It will:
- load stage summary.jsonl
- for each trial in allowlist, find the last meta.executed==true row
- choose a validation path (using your current priority order)
- run the same strict checks and print the reason
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

# --- copy-paste from your orchestrator (minimal pieces) ---

CANON_NAME = lambda eff, side: f"{eff}_{side}_program.smt2"

NAMED_OK_RE = re.compile(r"^REQ\d+_(AUXILIARY\d+|COMPONENT\d+_[A-Z0-9_]+)$")
ASSERT_BLOCK_RE = re.compile(r"\(assert\b", re.IGNORECASE)
NAMED_TAG_RE = re.compile(r":named\s+([^\s\)]+)")

FORBIDDEN_PATTERNS = [
    re.compile(r"\(declare-datatype\b", re.IGNORECASE),
    re.compile(r"\(declare-datatypes\b", re.IGNORECASE),
    re.compile(r"\bDatatype\b", re.IGNORECASE),
    re.compile(r"\(declare-sort\b", re.IGNORECASE),
]

COMMENT_RE_SINGLE = re.compile(r";.*?$", re.MULTILINE)
RE_DECLARE_CONST_HDR = re.compile(r"^\s*\(declare-const\s+(\S+)\s+(.+)\)\s*$")
ARRAY_SORT_RE = re.compile(r"^\(Array\s+Int\s+(Bool|Int|Real|String)\)$")
BOOL_REQUIRED_KEYS = {"when_to_set_to_true","when_to_set_to_false","when_to_set_to_null","meaning"}
VALUE_REQUIRED_KEYS = {"when_to_set_to_value","when_to_set_to_null","meaning"}

def read_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def strip_comments(s: str) -> str:
    return re.sub(COMMENT_RE_SINGLE, "", s)

def paren_balance_ok(s: str) -> bool:
    t = strip_comments(s)
    bal = 0
    for ch in t:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return False
    return bal == 0

def is_nonempty_file(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return False
        return bool(p.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        return False

def code_part_before_comment(line: str) -> str:
    i = line.find(";")
    return line if i == -1 else line[:i]

def declare_const_in_code(line: str) -> bool:
    return "(declare-const" in code_part_before_comment(line)

def extract_first_json_object_from_line(line: str) -> Optional[str]:
    start = line.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(line)):
        ch = line[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return line[start:i+1]
    return None

def validate_declare_const_json(line: str) -> Optional[str]:
    code = code_part_before_comment(line).rstrip()
    m = RE_DECLARE_CONST_HDR.match(code)
    if not m:
        return "declare_const_bad_header"
    _var, sort = m.group(1), m.group(2).strip()

    allowed_sorts: Set[str] = {"Bool","Int","Real","String"}
    if (sort not in allowed_sorts) and (not ARRAY_SORT_RE.match(sort)):
        return f"declare_const_disallowed_sort:{sort}"

    blob = extract_first_json_object_from_line(line)
    if not blob:
        return "declare_const_bad_format_or_missing_json_comment"
    try:
        obj = json.loads(blob)
    except Exception:
        return "declare_const_json_parse_failed"
    if not isinstance(obj, dict):
        return "declare_const_json_not_object"

    required = BOOL_REQUIRED_KEYS if sort == "Bool" else VALUE_REQUIRED_KEYS
    missing = required - set(obj.keys())
    if missing:
        return f"declare_const_json_missing_keys:{sorted(missing)}"
    extra = set(obj.keys()) - required
    if extra:
        return f"declare_const_json_extra_keys:{sorted(extra)}"
    if not isinstance(obj.get("meaning"), str):
        return "declare_const_json_meaning_not_string"
    for k in required:
        if k == "meaning":
            continue
        if not isinstance(obj.get(k), str):
            return f"declare_const_json_field_not_string:{k}"
    return None

def validate_strict(p: Path) -> Tuple[bool, List[str]]:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, [f"read_error:{e}"]
    errs: List[str] = []
    if not text.strip():
        return False, ["empty_or_whitespace"]
    if not paren_balance_ok(text):
        errs.append("paren_balance_failed")
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            errs.append(f"forbidden_pattern:{pat.pattern}")
    assert_count = len(list(ASSERT_BLOCK_RE.finditer(text)))
    named_count = len(list(NAMED_TAG_RE.finditer(text)))
    if assert_count == 0:
        errs.append("no_assert_found")
    elif named_count < assert_count:
        errs.append(f"missing_named_tags:asserts={assert_count},named={named_count}")
    for m in NAMED_TAG_RE.finditer(text):
        if not NAMED_OK_RE.match(m.group(1)):
            errs.append(f"bad_named_tag:{m.group(1)}")
    decl_seen = 0
    for line in text.splitlines():
        if not declare_const_in_code(line):
            continue
        issue = validate_declare_const_json(line)
        if issue:
            errs.append(issue)
            decl_seen += 1
            if decl_seen >= 50:
                break
    return (len(errs) == 0), errs

def meta_executed(row: Dict) -> bool:
    meta = row.get("meta")
    return isinstance(meta, dict) and meta.get("executed") is True

def last_executed_row(summary_rows: List[Dict], eff: str, side: str) -> Optional[Dict]:
    best = None
    for r in summary_rows:
        if str(r.get("effective_trial_id")) == eff and str(r.get("side")) == side and meta_executed(r):
            best = r
    return best

def best_validation_path(stage_merged_ir: Path, eff: str, side: str, row: Optional[Dict]) -> Optional[Path]:
    # YOUR CURRENT PRIORITY ORDER (merged_ir first)
    canon = stage_merged_ir / CANON_NAME(eff, side)
    if canon.exists() and canon.is_file():
        return canon
    if row:
        for k in ("fixed_smt_path","polarity_fixed_smt_path","smt_path"):
            v = row.get(k)
            if v:
                p = Path(v)
                if p.exists() and p.is_file():
                    return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--stage-dir", required=True)
    args = ap.parse_args()

    allow = Path(args.allowlist)
    stage_dir = Path(args.stage_dir)
    merged_ir = stage_dir / "merged_ir"
    summary = stage_dir / "summary.jsonl"

    trials = [t.strip() for t in allow.read_text().splitlines() if t.strip()]
    rows = read_jsonl(summary)

    for eff in trials:
        side = "exclusion"
        row = last_executed_row(rows, eff, side)
        if row is None:
            print(f"{eff}: NEW (no meta.executed==true row)")
            continue

        p = best_validation_path(merged_ir, eff, side, row)
        if p is None:
            print(f"{eff}: MISSING_ARTIFACT (no validation path)")
            continue

        if not is_nonempty_file(p):
            print(f"{eff}: RERUN_STRICT (empty)  path={p}")
            continue

        ok, errs = validate_strict(p)
        if ok:
            print(f"{eff}: OK (strict)  path={p}")
        else:
            head = ", ".join(errs[:4])
            print(f"{eff}: RERUN_STRICT ({len(errs)} errs)  path={p}  errs={head}")

if __name__ == "__main__":
    main()
