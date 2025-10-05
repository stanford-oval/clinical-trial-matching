#!/usr/bin/env python3
"""
audit_quarantine_guardrails.py

Audit (and optionally apply) the quarantine guardrails against an existing work_dir.

It checks stage raw_out outputs and reports what WOULD be quarantined (default),
or actually quarantines them (--apply).

Guardrails audited:
  - empty output (whitespace-only)
  - synthesized-from-empty-input: canonical input is empty, but stage output is non-empty
  - format invalid: lightweight SMT checks + strict declaration JSON key requirements

Outputs:
  - prints summary to stdout
  - optional JSON report (--report-json path)

Usage:
  python3 audit_quarantine_guardrails.py \
    --ir-dir ../build/ir \
    --work-dir ../build/ir_orchestrated_20260107T123456 \
    --report-json audit.json

  # Actually quarantine (move files) like orchestrator would:
  python3 audit_quarantine_guardrails.py ... --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# -------------------------
# Patterns matching your orchestrator
# -------------------------
CANON_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program\.smt2$")
POLARITY_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_exclusion_program_polarity_fixed\.smt2$")
REPAIR_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_repaired\.smt2$")
LOGIC_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_logic_fixed\.smt2$")

FORBIDDEN_PATTERNS = [
    re.compile(r"\(declare-datatype\b", re.IGNORECASE),
    re.compile(r"\(declare-datatypes\b", re.IGNORECASE),
    re.compile(r"\bDatatype\b", re.IGNORECASE),
    re.compile(r"\(declare-sort\b", re.IGNORECASE),
]
NAMED_OK_RE = re.compile(r"^REQ\d+_(AUXILIARY\d+|COMPONENT\d+_[A-Z0-9_]+)$")
ASSERT_BLOCK_RE = re.compile(r"\(assert\b", re.IGNORECASE)
NAMED_TAG_RE = re.compile(r":named\s+([^\s\)]+)")
COMMENT_RE = re.compile(r";;.*?$", re.MULTILINE)

RE_DECLARE_CONST = re.compile(
    r"^\s*\(declare-const\s+(\S+)\s+(Bool|Int|Real)\)\s*;;\s*(\{.*\})\s*$"
)
BOOL_REQUIRED_KEYS = {
    "when_to_set_to_true",
    "when_to_set_to_false",
    "when_to_set_to_null",
    "meaning",
}
NUM_REQUIRED_KEYS = {
    "when_to_set_to_value",
    "when_to_set_to_null",
    "meaning",
}

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def strip_comments(s: str) -> str:
    return re.sub(COMMENT_RE, "", s)

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

def is_nonempty_smt(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return False
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return bool(txt.strip())
    except Exception:
        return False

def validate_declare_const_json_comment(line: str, *, allow_extra_keys: bool) -> Optional[str]:
    m = RE_DECLARE_CONST.match(line)
    if not m:
        if "(declare-const" in line:
            return "declare_const_bad_format_or_missing_json_comment"
        return None

    _var, ty, json_blob = m.group(1), m.group(2), m.group(3)
    try:
        obj = json.loads(json_blob)
    except Exception:
        return "declare_const_json_parse_failed"
    if not isinstance(obj, dict):
        return "declare_const_json_not_object"

    required = BOOL_REQUIRED_KEYS if ty == "Bool" else NUM_REQUIRED_KEYS
    missing = required - set(obj.keys())
    if missing:
        return f"declare_const_json_missing_keys:{sorted(missing)}"
    if not allow_extra_keys:
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

def validate_smt_file(p: Path, *, allow_extra_decl_json_keys: bool) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, [f"read_error:{e}"]

    if not txt.strip():
        return False, ["empty_or_whitespace"]

    if not paren_balance_ok(txt):
        errs.append("paren_balance_failed")

    for pat in FORBIDDEN_PATTERNS:
        if pat.search(txt):
            errs.append(f"forbidden_pattern:{pat.pattern}")

    if not ASSERT_BLOCK_RE.search(txt):
        errs.append("no_assert_found")

    assert_count = len(list(ASSERT_BLOCK_RE.finditer(txt)))
    named_count = len(list(NAMED_TAG_RE.finditer(txt)))
    if assert_count > 0 and named_count < assert_count:
        errs.append(f"missing_named_tags:asserts={assert_count},named={named_count}")

    for m in NAMED_TAG_RE.finditer(txt):
        name = m.group(1)
        if not NAMED_OK_RE.match(name):
            errs.append(f"bad_named_tag:{name}")

    for line in txt.splitlines():
        if "(declare-const" in line:
            e = validate_declare_const_json_comment(line, allow_extra_keys=allow_extra_decl_json_keys)
            if e:
                errs.append(e)
                break

    return (len(errs) == 0), errs

def canonical_index(ir_dir: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for p in ir_dir.glob("NCT*_program.smt2"):
        m = CANON_RE.match(p.name)
        if not m:
            continue
        out[(m.group(1), m.group(2))] = p
    return out

def stage_outputs(raw_out: Path, kind: str) -> List[Tuple[Path, Tuple[str, str]]]:
    out: List[Tuple[Path, Tuple[str, str]]] = []
    if not raw_out.exists():
        return out
    for p in raw_out.glob("NCT*.smt2"):
        pair: Optional[Tuple[str, str]] = None
        if kind == "polarity":
            m = POLARITY_OUT_RE.match(p.name)
            if m:
                pair = (m.group(1), "exclusion")
        elif kind == "repair":
            m = REPAIR_OUT_RE.match(p.name)
            if m:
                pair = (m.group(1), m.group(2))
        elif kind == "logic":
            m = LOGIC_OUT_RE.match(p.name)
            if m:
                pair = (m.group(1), m.group(2))
        if pair is not None:
            out.append((p, pair))
    return out

def quarantine_move(src: Path, quarantine_dir: Path, reason: str) -> Path:
    ensure_dir(quarantine_dir)
    dst = quarantine_dir / f"{reason}__{src.name}"
    i = 1
    while dst.exists():
        dst = quarantine_dir / f"{reason}__{i}__{src.name}"
        i += 1
    shutil.move(str(src), str(dst))
    return dst

# -------------------------
# Audit data model
# -------------------------
@dataclass
class Finding:
    stage: str
    kind: str
    path: str
    eff: str
    side: str
    reason: str
    errors: List[str]

# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--apply", action="store_true", help="Actually move quarantined files into stage/quarantine_invalid/")
    ap.add_argument("--report-json", default=None, help="Write a JSON report here.")
    ap.add_argument("--decl-json-allow-extra-keys", action="store_true")
    ap.add_argument("--no-format-checks", action="store_true")
    ap.add_argument("--show-first", type=int, default=30, help="Print first N findings.")
    args = ap.parse_args()

    ir_dir = Path(args.ir_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    canon = canonical_index(ir_dir)
    allow_extra = bool(args.decl_json_allow_extra_keys)
    do_format = not args.no_format_checks

    stages = [
        ("stage1_polarity", "polarity"),
        ("stage2_repair", "repair"),
        ("stage3_logic", "logic"),
    ]

    findings: List[Finding] = []
    total_files = 0

    for stage_name, kind in stages:
        raw_out = work_dir / stage_name / "raw_out"
        quarantine_dir = work_dir / stage_name / "quarantine_invalid"

        for p, (eff, side) in stage_outputs(raw_out, kind):
            total_files += 1

            # 1) empty output
            if not is_nonempty_smt(p):
                findings.append(Finding(stage_name, kind, str(p), eff, side, "empty_output", []))
                if args.apply:
                    quarantine_move(p, quarantine_dir, "empty_output")
                continue

            # 2) synthesized from empty input
            base = canon.get((eff, side))
            if base is not None and not is_nonempty_smt(base):
                findings.append(Finding(stage_name, kind, str(p), eff, side, "synthesized_from_empty_input", []))
                if args.apply:
                    quarantine_move(p, quarantine_dir, "synthesized_from_empty_input")
                continue

            # 3) format invalid
            if do_format:
                ok, errs = validate_smt_file(p, allow_extra_decl_json_keys=allow_extra)
                if not ok:
                    findings.append(Finding(stage_name, kind, str(p), eff, side, "format_invalid", errs))
                    if args.apply:
                        quarantine_move(p, quarantine_dir, "format_invalid")
                    continue

    # Summary
    by_reason: Dict[str, int] = {}
    by_stage_reason: Dict[str, Dict[str, int]] = {}
    for f in findings:
        by_reason[f.reason] = by_reason.get(f.reason, 0) + 1
        by_stage_reason.setdefault(f.stage, {})
        by_stage_reason[f.stage][f.reason] = by_stage_reason[f.stage].get(f.reason, 0) + 1

    print("\n[AUDIT SUMMARY]")
    print(f"  work_dir={work_dir}")
    print(f"  total_stage_outputs_scanned={total_files}")
    print(f"  would_quarantine={len(findings)} ({(len(findings)/total_files*100.0 if total_files else 0):.2f}%)")
    print(f"  format_checks={'ON' if do_format else 'OFF'}  decl_json_allow_extra_keys={allow_extra}")

    print("\n[BY REASON]")
    for k in sorted(by_reason.keys()):
        print(f"  {k}: {by_reason[k]}")

    print("\n[BY STAGE + REASON]")
    for stage in ["stage1_polarity", "stage2_repair", "stage3_logic"]:
        d = by_stage_reason.get(stage, {})
        if not d:
            print(f"  {stage}: (none)")
            continue
        parts = ", ".join([f"{r}={d[r]}" for r in sorted(d.keys())])
        print(f"  {stage}: {parts}")

    if findings and args.show_first > 0:
        print(f"\n[FIRST {min(args.show_first, len(findings))} FINDINGS]")
        for f in findings[: args.show_first]:
            extra = f" errors={f.errors}" if f.errors else ""
            print(f"  - {f.stage} {f.eff}_{f.side}: {f.reason} :: {f.path}{extra}")

    if args.report_json:
        report = {
            "work_dir": str(work_dir),
            "ir_dir": str(ir_dir),
            "total_scanned": total_files,
            "would_quarantine": len(findings),
            "by_reason": by_reason,
            "by_stage_reason": by_stage_reason,
            "findings": [f.__dict__ for f in findings],
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[WROTE REPORT] {args.report_json}")

    if args.apply:
        print("\n[APPLY] Quarantine moves completed.")


if __name__ == "__main__":
    main()
