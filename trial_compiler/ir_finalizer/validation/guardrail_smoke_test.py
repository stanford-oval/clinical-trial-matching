#!/usr/bin/env python3
"""
guardrail_smoke_test.py

Hard-fails on:
  - declare_const_bad_header
  - declare_const_bad_format_or_missing_json_comment
    (unless skipped by one of the skip modes below)

Also enforces:
  - non-empty, paren balance, no forbidden constructs
  - at least one assert
  - named_count >= assert_count
  - :named tags match NAMED_OK_RE

Important:
  - ignores commented-out "(declare-const ...)" lines.
  - supports SKIP modes to keep moving and surface other failure classes.

Skip modes (WARN only, not ERROR):
  --skip-auto-synth-missing-json
  --skip-nearby-json-comment-block
  --skip-group-block-missing-json
  --skip-direct-json-comment-above (NEW): if a JSON blob appears in the N lines directly above

Artifacts under --out-dir:
  - report.jsonl / failures.jsonl / summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CANON_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program\.smt2$")

FORBIDDEN_PATTERNS = [
    re.compile(r"\(declare-datatype\b", re.IGNORECASE),
    re.compile(r"\(declare-datatypes\b", re.IGNORECASE),
    re.compile(r"\bDatatype\b", re.IGNORECASE),
    re.compile(r"\(declare-sort\b", re.IGNORECASE),
]

NAMED_OK_RE = re.compile(r"^REQ\d+_(AUXILIARY\d+|COMPONENT\d+_[A-Z0-9_]+)$")
ASSERT_BLOCK_RE = re.compile(r"\(assert\b", re.IGNORECASE)
NAMED_TAG_RE = re.compile(r":named\s+([^\s\)]+)")

COMMENT_RE_DOUBLE = re.compile(r";;.*?$", re.MULTILINE)
COMMENT_RE_SINGLE = re.compile(r";.*?$", re.MULTILINE)

# NOTE:
# - We validate the declare-const "header" on the CODE PART ONLY (strip comments first),
#   because JSON blobs often live in comments.
# - Sort may be a single token (Bool, Int, Real, String) OR a parenthesized S-expression,
#   e.g. (Array Int Real).
RE_DECLARE_CONST_HDR = re.compile(r"^\s*\(declare-const\s+(\S+)\s+(.+)\)\s*$")

BOOL_REQUIRED_KEYS = {
    "when_to_set_to_true",
    "when_to_set_to_false",
    "when_to_set_to_null",
    "meaning",
}
VALUE_REQUIRED_KEYS = {
    "when_to_set_to_value",
    "when_to_set_to_null",
    "meaning",
}

HARD_FAIL_DECL_ISSUES = {
    "declare_const_bad_header",
    "declare_const_bad_format_or_missing_json_comment",
}

AUTO_SYNTH_MARKER_RE = re.compile(r"\bauto-synthesized variable\b", re.IGNORECASE)

GROUP_BLOCK_MARKER_RE = re.compile(
    r"(group\s+fit\s+variables|boolean,\s*one\s+per\s+group|one\s+per\s+group)",
    re.IGNORECASE,
)

# Allow Array sorts without requiring --decl-allow-any-sort.
# Default: (Array Int <base>) where base is one of Bool/Int/Real/String.
ARRAY_SORT_RE = re.compile(r"^\(Array\s+Int\s+(Bool|Int|Real|String)\)$")


# -------------------------
# FS helpers
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_nonempty_smt(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_file():
            return False
        return bool(p.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        return False


def iter_canonical_ir_files(ir_dir: Path) -> Iterable[Path]:
    for p in ir_dir.glob("NCT*_program.smt2"):
        if CANON_RE.match(p.name):
            yield p


# -------------------------
# Comment-aware line helpers
# -------------------------
def code_part_before_comment(line: str) -> str:
    i = line.find(";")
    return line if i == -1 else line[:i]


def declare_const_in_code(line: str) -> bool:
    return "(declare-const" in code_part_before_comment(line)


# -------------------------
# Guardrail helpers
# -------------------------
def strip_comments(s: str, *, strict_double_semicolon: bool) -> str:
    return re.sub(COMMENT_RE_DOUBLE if strict_double_semicolon else COMMENT_RE_SINGLE, "", s)


def paren_balance_diagnostics(s: str, *, strict_double_semicolon: bool) -> Dict:
    t = strip_comments(s, strict_double_semicolon=strict_double_semicolon)
    bal = 0
    first_neg: Optional[int] = None
    for i, ch in enumerate(t):
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0 and first_neg is None:
                first_neg = i
                break
    return {"ok": (bal == 0 and first_neg is None), "final_balance": bal, "first_negative_index": first_neg}


def _is_str(x) -> bool:
    return isinstance(x, str)


def extract_first_json_object_from_line(line: str) -> Optional[str]:
    start = line.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
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
                    return line[start : i + 1]
    return None


def extract_json_from_comment_lookback(
    lines: List[str],
    decl_lineno_1based: int,
    var_name: str,
    lookback: int,
) -> Optional[str]:
    start = max(1, decl_lineno_1based - lookback)
    window = lines[start - 1 : decl_lineno_1based - 1]
    if not window:
        return None

    if var_name not in "".join(window):
        return None

    for ln in reversed(window):
        jb = extract_first_json_object_from_line(ln)
        if jb:
            return jb
    return None


def has_group_block_marker_and_json(
    lines: List[str],
    decl_lineno_1based: int,
    lookback: int,
) -> bool:
    start = max(1, decl_lineno_1based - lookback)
    window = lines[start - 1 : decl_lineno_1based - 1]
    if not window:
        return False

    marker = any(GROUP_BLOCK_MARKER_RE.search(ln) for ln in window)
    if not marker:
        return False

    has_json = any(extract_first_json_object_from_line(ln) for ln in window)
    return bool(has_json)


def has_json_in_direct_lookback(
    lines: List[str],
    decl_lineno_1based: int,
    lookback: int,
) -> bool:
    """
    Aggressive skip: if any JSON blob appears within the last `lookback` lines above
    the declare-const line, consider it a narrative metadata block and skip.
    """
    start = max(1, decl_lineno_1based - lookback)
    window = lines[start - 1 : decl_lineno_1based - 1]
    return any(extract_first_json_object_from_line(ln) for ln in window)


def validate_declare_const_json_comment(
    line: str,
    *,
    allow_extra_keys: bool,
    allowed_sorts: set,
    allow_any_sort: bool,
    skip_auto_synth_missing_json: bool,
    all_lines: Optional[List[str]],
    lineno_1based: Optional[int],
    skip_nearby_json_comment_block: bool,
    decl_json_lookback: int,
    skip_group_block_missing_json: bool,
    group_block_lookback: int,
    skip_direct_json_comment_above: bool,
    direct_json_lookback: int,
) -> Optional[str]:
    # IMPORTANT: parse header from code-only, not including comments/JSON.
    code = code_part_before_comment(line).rstrip()
    m = RE_DECLARE_CONST_HDR.match(code)
    if not m:
        return "declare_const_bad_header"

    var_name, sort = m.group(1), m.group(2).strip()

    # sort allowlist
    if not allow_any_sort:
        if (sort not in allowed_sorts) and (not ARRAY_SORT_RE.match(sort)):
            return f"declare_const_disallowed_sort:{sort}"

    # JSON blob may be in the original line (comment), not the code-only part.
    json_blob = extract_first_json_object_from_line(line)
    if not json_blob:
        if skip_auto_synth_missing_json and AUTO_SYNTH_MARKER_RE.search(line):
            return "declare_const_missing_json_auto_synth_skipped"

        if (
            skip_nearby_json_comment_block
            and all_lines is not None
            and lineno_1based is not None
            and lineno_1based >= 2
        ):
            nearby = extract_json_from_comment_lookback(
                all_lines,
                decl_lineno_1based=lineno_1based,
                var_name=var_name,
                lookback=decl_json_lookback,
            )
            if nearby:
                return "declare_const_missing_json_nearby_comment_block_skipped"

        if (
            skip_group_block_missing_json
            and all_lines is not None
            and lineno_1based is not None
            and lineno_1based >= 2
        ):
            if has_group_block_marker_and_json(all_lines, lineno_1based, group_block_lookback):
                return "declare_const_missing_json_group_block_skipped"

        if (
            skip_direct_json_comment_above
            and all_lines is not None
            and lineno_1based is not None
            and lineno_1based >= 2
        ):
            if has_json_in_direct_lookback(all_lines, lineno_1based, direct_json_lookback):
                return "declare_const_missing_json_direct_json_above_skipped"

        return "declare_const_bad_format_or_missing_json_comment"

    try:
        obj = json.loads(json_blob)
    except Exception:
        return "declare_const_json_parse_failed"

    if not isinstance(obj, dict):
        return "declare_const_json_not_object"

    required = BOOL_REQUIRED_KEYS if sort == "Bool" else VALUE_REQUIRED_KEYS

    missing = required - set(obj.keys())
    if missing:
        return f"declare_const_json_missing_keys:{sorted(missing)}"

    if not allow_extra_keys:
        extra = set(obj.keys()) - required
        if extra:
            return f"declare_const_json_extra_keys:{sorted(extra)}"

    if not _is_str(obj.get("meaning")):
        return "declare_const_json_meaning_not_string"

    for k in required:
        if k == "meaning":
            continue
        if not _is_str(obj.get(k)):
            return f"declare_const_json_field_not_string:{k}"

    return None


def validate_smt_content(
    text: str,
    *,
    allow_extra_decl_json_keys: bool,
    strict_double_semicolon: bool,
    decl_json_strict: bool,
    allow_any_sort: bool,
    skip_auto_synth_missing_json: bool,
    skip_nearby_json_comment_block: bool,
    decl_json_lookback: int,
    skip_group_block_missing_json: bool,
    group_block_lookback: int,
    skip_direct_json_comment_above: bool,
    direct_json_lookback: int,
) -> Tuple[bool, List[str], List[str], Dict]:
    errors: List[str] = []
    warnings: List[str] = []
    diag: Dict = {}

    if not text.strip():
        errors.append("empty_or_whitespace")
        return False, errors, warnings, diag

    pb = paren_balance_diagnostics(text, strict_double_semicolon=strict_double_semicolon)
    diag["paren_balance"] = pb
    if not pb["ok"]:
        errors.append("paren_balance_failed")

    forbidden_hits: List[str] = []
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            forbidden_hits.append(pat.pattern)
            errors.append(f"forbidden_pattern:{pat.pattern}")
    diag["forbidden_hits"] = forbidden_hits

    assert_count = len(list(ASSERT_BLOCK_RE.finditer(text)))
    named_matches = list(NAMED_TAG_RE.finditer(text))
    named_count = len(named_matches)

    diag["assert_count"] = assert_count
    diag["named_count"] = named_count

    if assert_count == 0:
        errors.append("no_assert_found")
    else:
        if named_count < assert_count:
            errors.append(f"missing_named_tags:asserts={assert_count},named={named_count}")

    bad_named: List[str] = []
    for m in named_matches:
        name = m.group(1)
        if not NAMED_OK_RE.match(name):
            bad_named.append(name)
            errors.append(f"bad_named_tag:{name}")
    diag["bad_named_tags"] = bad_named[:50]

    decl_issues: List[Dict] = []
    allowed_sorts = {"Bool", "Int", "Real", "String"}
    all_lines = text.splitlines()

    for i, line in enumerate(all_lines, start=1):
        if not declare_const_in_code(line):
            continue

        issue = validate_declare_const_json_comment(
            line,
            allow_extra_keys=allow_extra_decl_json_keys,
            allowed_sorts=allowed_sorts,
            allow_any_sort=allow_any_sort,
            skip_auto_synth_missing_json=skip_auto_synth_missing_json,
            all_lines=all_lines,
            lineno_1based=i,
            skip_nearby_json_comment_block=skip_nearby_json_comment_block,
            decl_json_lookback=decl_json_lookback,
            skip_group_block_missing_json=skip_group_block_missing_json,
            group_block_lookback=group_block_lookback,
            skip_direct_json_comment_above=skip_direct_json_comment_above,
            direct_json_lookback=direct_json_lookback,
        )
        if issue:
            decl_issues.append({"line_no": i, "issue": issue, "line": line[:900]})

            if issue in (
                "declare_const_missing_json_auto_synth_skipped",
                "declare_const_missing_json_nearby_comment_block_skipped",
                "declare_const_missing_json_group_block_skipped",
                "declare_const_missing_json_direct_json_above_skipped",
            ):
                warnings.append(issue)
            elif issue in HARD_FAIL_DECL_ISSUES:
                errors.append(issue)
            else:
                if decl_json_strict:
                    errors.append(issue)
                else:
                    warnings.append(issue)

            if len(decl_issues) >= 20:
                break

    diag["declare_const_issues"] = decl_issues
    ok = (len(errors) == 0)
    return ok, errors, warnings, diag


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-dir", default="../build/ir")
    ap.add_argument("--sample-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="../build/guardrails_smoke_test")

    ap.add_argument("--allow-extra-decl-json-keys", action="store_true")
    ap.add_argument("--strict-double-semicolon-comments", action="store_true")

    ap.add_argument("--decl-json-strict", action="store_true")
    ap.add_argument("--decl-allow-any-sort", action="store_true")

    ap.add_argument("--skip-auto-synth-missing-json", action="store_true")
    ap.add_argument("--skip-nearby-json-comment-block", action="store_true")
    ap.add_argument("--decl-json-lookback", type=int, default=8)

    ap.add_argument("--skip-group-block-missing-json", action="store_true")
    ap.add_argument("--group-block-lookback", type=int, default=20)

    # NEW
    ap.add_argument(
        "--skip-direct-json-comment-above",
        action="store_true",
        help="Skip missing-inline-JSON declare-const lines if a JSON blob appears in the N lines immediately above (warn only).",
    )
    ap.add_argument("--direct-json-lookback", type=int, default=3)

    ap.add_argument("--copy-failing-files", action="store_true")
    ap.add_argument("--head-chars", type=int, default=2000)
    ap.add_argument("--only", default=None)

    ap.add_argument("--print-first-failure", action="store_true")
    ap.add_argument("--max-print-chars", type=int, default=20000)
    ap.add_argument("--print-no-truncate", action="store_true")

    return ap.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> int:
    args = parse_args()

    ir_dir = Path(args.ir_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    report_path = out_dir / "report.jsonl"
    failures_path = out_dir / "failures.jsonl"
    summary_path = out_dir / "summary.json"
    failing_files_dir = out_dir / "failing_files"

    if args.copy_failing_files:
        ensure_dir(failing_files_dir)

    all_files = list(iter_canonical_ir_files(ir_dir))
    if args.only:
        all_files = [p for p in all_files if p.name == args.only]

    nonempty = [p for p in all_files if is_nonempty_smt(p)]
    skipped_empty = len(all_files) - len(nonempty)

    rnd = random.Random(args.seed)
    chosen = rnd.sample(nonempty, args.sample_size) if len(nonempty) > args.sample_size else nonempty

    counters = Counter()
    err_counts = Counter()
    warn_counts = Counter()
    printed_first = False

    report_path.write_text("", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    for p in chosen:
        txt = p.read_text(encoding="utf-8", errors="ignore")

        ok, errors, warnings, diag = validate_smt_content(
            txt,
            allow_extra_decl_json_keys=args.allow_extra_decl_json_keys,
            strict_double_semicolon=args.strict_double_semicolon_comments,
            decl_json_strict=args.decl_json_strict,
            allow_any_sort=args.decl_allow_any_sort,
            skip_auto_synth_missing_json=args.skip_auto_synth_missing_json,
            skip_nearby_json_comment_block=args.skip_nearby_json_comment_block,
            decl_json_lookback=int(args.decl_json_lookback),
            skip_group_block_missing_json=args.skip_group_block_missing_json,
            group_block_lookback=int(args.group_block_lookback),
            skip_direct_json_comment_above=args.skip_direct_json_comment_above,
            direct_json_lookback=int(args.direct_json_lookback),
        )

        row = {
            "path": str(p),
            "filename": p.name,
            "bytes": p.stat().st_size if p.exists() else None,
            "ok": ok,
            "errors": errors,
            "warnings": warnings,
            "diagnostics": diag,
        }

        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        counters["tested"] += 1
        counters["passed" if ok else "failed"] += 1

        for e in errors:
            err_counts[e] += 1
        for w in warnings:
            warn_counts[w] += 1

        if not ok:
            fail_row = dict(row)
            fail_row["head"] = txt[: int(args.head_chars)]
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fail_row, ensure_ascii=False) + "\n")

            copied_to: Optional[str] = None
            if args.copy_failing_files:
                try:
                    dst = failing_files_dir / p.name
                    shutil.copy2(p, dst)
                    copied_to = str(dst)
                except Exception:
                    copied_to = None

            if args.print_first_failure and (not printed_first):
                printed_first = True
                print("\n" + "=" * 100)
                print("[FIRST FAILURE]")
                print("file:", p)
                if copied_to:
                    print("copied_to:", copied_to)
                print("errors:", errors[:80])
                if warnings:
                    print("warnings (first):", warnings[:40])

                issues = diag.get("declare_const_issues") or []
                if issues:
                    print("\n[declare-const issues (first up to 20)]")
                    for it in issues[:20]:
                        print(f"  line {it['line_no']}: {it['issue']}")
                        print(f"    {it['line']}")

                pb = diag.get("paren_balance") or {}
                if pb:
                    print("\n[paren balance]")
                    print(json.dumps(pb, indent=2))

                print("\n[SMT]")
                if args.print_no_truncate:
                    print(txt)
                else:
                    n = int(args.max_print_chars)
                    print(txt[:n])
                    if len(txt) > n:
                        print(f"\n... [truncated: printed {n} / {len(txt)} chars] ...")
                print("=" * 100 + "\n")

    summary = {
        "ir_dir": str(ir_dir),
        "out_dir": str(out_dir),
        "sample_size_requested": args.sample_size,
        "sample_size_tested": len(chosen),
        "seed": args.seed,
        "skipped_empty_canonical_files": skipped_empty,
        "counts": dict(counters),
        "top_errors": err_counts.most_common(50),
        "top_warnings": warn_counts.most_common(50),
        "comment_stripping_mode": "STRICT_;;_ONLY" if args.strict_double_semicolon_comments else "SAFE_;_ANY",
        "decl_json_strict": bool(args.decl_json_strict),
        "decl_allow_any_sort": bool(args.decl_allow_any_sort),
        "allow_extra_decl_json_keys": bool(args.allow_extra_decl_json_keys),
        "skip_auto_synth_missing_json": bool(args.skip_auto_synth_missing_json),
        "skip_nearby_json_comment_block": bool(args.skip_nearby_json_comment_block),
        "decl_json_lookback": int(args.decl_json_lookback),
        "skip_group_block_missing_json": bool(args.skip_group_block_missing_json),
        "group_block_lookback": int(args.group_block_lookback),
        "skip_direct_json_comment_above": bool(args.skip_direct_json_comment_above),
        "direct_json_lookback": int(args.direct_json_lookback),
        "artifacts": {
            "report_jsonl": str(report_path),
            "failures_jsonl": str(failures_path),
            "failing_files_dir": str(failing_files_dir) if args.copy_failing_files else None,
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    return 0 if counters["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
