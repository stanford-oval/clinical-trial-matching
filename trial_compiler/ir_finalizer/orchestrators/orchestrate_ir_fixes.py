#!/usr/bin/env python3
"""
# DEPRECATED: Use orchestrate.py instead (the simplified parallel orchestrator).
# This file is retained for reference but is not used in the standard pipeline.

orchestrate_ir_fixes.py  (FIXED WORKFLOW, minimal flags + PLAN + AUDIT + CONFIRM)
(FIXED: strict-invalid artifacts never propagate downstream; fallback to previous stage; bounded retries; optional skip assert checks)

Pipeline (fixed):
  Stage0: meaning_enrich_irs.py (ONLY for pairs that fail strict declare-const JSON requirements)
  Stage1: polarity_fix_irs.py
  Stage2: repair_all_ir.py
  Stage3: logic_fix_irs.py

STRICT COMPLIANCE (default; --skip-assert-checks disables assert/:named checks):
- Non-empty
- Paren balance
- No forbidden patterns: declare-datatype(s), Datatype token, declare-sort
- (Optional) Assert checks:
    - At least one assert
    - Every assert must have a :named tag, and tag must match NAMED_OK_RE
- Every declare-const must have inline JSON on the same line; JSON must parse; required keys must exist;
  extra keys are forbidden; required fields must be strings; sort allowlist applies.

RESUME / FALLBACK POLICY (requested behavior):
- For each stage K, the "current thing to resume from" is stageK/merged_ir/<eff>_<side>_program.smt2.
- If that file is strict-invalid for a pair, we DO NOT use it as input to downstream stages.
  Instead, we fall back to the previous stage's version (K-1), and rerun stage K.
- If after the allowed retry budget the stage still fails, we stop rerunning and simply propagate the old version
  (i.e., downstream sees the strict-valid fallback if it exists; otherwise it sees whatever earlier stage/input provides).

How we implement this:
- Each stage's merged_ir is built as:
    base = the previous stage "best view" for all pairs
    overlay = stage outputs that are strict-ok (only)
  => Any strict-invalid stage output is ignored/quarantined and DOES NOT enter merged_ir.
- For running stage scripts, we materialize a TODO ir-dir (per attempt) from the previous stage "best view"
  so the stage never reads strict-invalid current artifacts.

Attempt budgeting:
- MAX_TOTAL_ATTEMPTS_PER_PAIR_PER_STAGE = 1 + MAX_RETRIES_PER_STAGE
- Attempt counting is persistent across --resume runs via summary.jsonl:
  we count rows with meta dict for that pair.
- If a pair has already consumed its total attempts, we DO NOT rerun it again even if strict-invalid.
  (This is the "eventually fails -> propagate old version" behavior.)

Key behaviors retained:
- attempted set for Stage1/2/3 is ANY summary.jsonl row with meta dict
- "output empty is fine": we do NOT rerun pairs whose last row has meta.executed != true (attempted-failed)
  and we also cap retries via attempted counts.

Stage0 skip:
- --skip-stage0 will NOT execute Stage0.
- It WILL:
    * materialize stage0/merged_ir from existing stage0 artifacts (offline, no LLM),
      even if stage0 summary is missing/empty by scanning stage0/raw_out.
    * compute Stage0 "best view" per pair: use stage0/merged_ir if strict-ok else input file.
    * force downstream reruns for all (eff,side) pairs that Stage0 touched previously (history).

Declare-const loopback (optional):
- If after Stage1-3 pass(es), final merged IR still has strict failures solely due to declare_const_*,
  we can loop those back through meaning fixer and rerun Stage1-3. (bounded)

Minimal CLI:
  --resume
  --work-dir PATH
  --ir-dir PATH
  --snapshot-dir PATH
  --final-ir-dir PATH
  --dry-run
  --yes
  --rerun-if-input-noncompliant
  --skip-stage0
  --skip-assert-checks
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


# -------------------------
# Fixed knobs (no CLI)
# -------------------------
MAX_RETRIES_PER_STAGE = 1
MAX_TOTAL_ATTEMPTS_PER_PAIR_PER_STAGE = 1 + MAX_RETRIES_PER_STAGE

MAX_DECL_LOOPBACK_PASSES = 2   # bounded loopback passes for declare-const-only strict failures

DEFAULT_MAX_WORKERS_MEANING = 16
DEFAULT_MAX_WORKERS_POLARITY = 16
DEFAULT_MAX_WORKERS_REPAIR = 16
DEFAULT_MAX_WORKERS_LOGIC = 16

LOGIC_SIDE = "both"
USE_HARDLINKS = False

# Audit output knobs (fixed)
AUDIT_MAX_FILES = 2000          # cap how many rerun candidates to inspect per stage
AUDIT_TOPK_REASONS = 20         # show top-N error reasons
AUDIT_MAX_SAMPLES_PER_REASON = 3


# -------------------------
# Filename patterns
# -------------------------
CANON_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program\.smt2$")

MEANING_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_meaning_enriched\.smt2$")
POLARITY_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_exclusion_program_polarity_fixed\.smt2$")
REPAIR_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_repaired\.smt2$")
LOGIC_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_logic_fixed\.smt2$")

WORKDIR_RE = re.compile(r"^ir_orchestrated_(\d{8}T\d{6})$")


# -------------------------
# Strict checks
# -------------------------
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

ARRAY_SORT_RE = re.compile(r"^\(Array\s+Int\s+(Bool|Int|Real|String)\)$")


@dataclass(frozen=True)
class GuardrailConfig:
    allow_extra_decl_json_keys: bool
    strict_double_semicolon: bool
    decl_json_strict: bool
    allow_any_sort: bool

    # Optional assertion checks
    check_asserts_and_named: bool


STRICT_GUARDRAILS_DEFAULT = GuardrailConfig(
    allow_extra_decl_json_keys=False,
    strict_double_semicolon=False,
    decl_json_strict=True,
    allow_any_sort=False,
    check_asserts_and_named=True,
)


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


def code_part_before_comment(line: str) -> str:
    i = line.find(";")
    return line if i == -1 else line[:i]


def declare_const_in_code(line: str) -> bool:
    return "(declare-const" in code_part_before_comment(line)


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


def validate_declare_const_json_comment(
    line: str,
    *,
    cfg: GuardrailConfig,
    allowed_sorts: Set[str],
) -> Optional[str]:
    code = code_part_before_comment(line).rstrip()
    m = RE_DECLARE_CONST_HDR.match(code)
    if not m:
        return "declare_const_bad_header"

    _var_name, sort = m.group(1), m.group(2).strip()

    if not cfg.allow_any_sort:
        if (sort not in allowed_sorts) and (not ARRAY_SORT_RE.match(sort)):
            return f"declare_const_disallowed_sort:{sort}"

    json_blob = extract_first_json_object_from_line(line)
    if not json_blob:
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


def validate_smt_content(text: str, *, cfg: GuardrailConfig) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not text.strip():
        errors.append("empty_or_whitespace")
        return False, errors

    pb = paren_balance_diagnostics(text, strict_double_semicolon=cfg.strict_double_semicolon)
    if not pb["ok"]:
        errors.append("paren_balance_failed")

    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            errors.append(f"forbidden_pattern:{pat.pattern}")

    if cfg.check_asserts_and_named:
        assert_count = len(list(ASSERT_BLOCK_RE.finditer(text)))
        named_matches = list(NAMED_TAG_RE.finditer(text))
        named_count = len(named_matches)

        if assert_count == 0:
            errors.append("no_assert_found")
        else:
            if named_count < assert_count:
                errors.append(f"missing_named_tags:asserts={assert_count},named={named_count}")

        for m in named_matches:
            name = m.group(1)
            if not NAMED_OK_RE.match(name):
                errors.append(f"bad_named_tag:{name}")

    allowed_sorts: Set[str] = {"Bool", "Int", "Real", "String"}
    decl_issues_seen = 0
    for line in text.splitlines():
        if not declare_const_in_code(line):
            continue
        issue = validate_declare_const_json_comment(line, cfg=cfg, allowed_sorts=allowed_sorts)
        if issue:
            errors.append(issue)
            decl_issues_seen += 1
            if decl_issues_seen >= 50:
                break

    return (len(errors) == 0), errors


def validate_smt_file(p: Path, *, cfg: GuardrailConfig) -> Tuple[bool, List[str]]:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, [f"read_error:{e}"]
    return validate_smt_content(txt, cfg=cfg)


# -------------------------
# FS helpers
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def is_nonempty_smt(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_file():
            return False
        if p.stat().st_size == 0:
            return False
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(4096)
        if chunk.strip():
            return True
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            rest = f.read()
        return bool(rest.strip())
    except Exception:
        return False


def strict_ok_file(p: Path, *, strict_cfg: GuardrailConfig) -> bool:
    if not is_nonempty_smt(p):
        return False
    ok, _ = validate_smt_file(p, cfg=strict_cfg)
    return ok


def copy_or_link_file(src: Path, dst: Path, *, use_hardlinks: bool) -> None:
    ensure_dir(dst.parent)
    _safe_unlink(dst)
    if use_hardlinks:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def iter_canonical_ir_files(ir_dir: Path) -> Iterable[Path]:
    for p in ir_dir.glob("NCT*_program.smt2"):
        if CANON_RE.match(p.name):
            yield p


def has_any_canonical_files(ir_dir: Path) -> bool:
    for _p in iter_canonical_ir_files(ir_dir):
        return True
    return False


def canonical_index(ir_dir: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for p in iter_canonical_ir_files(ir_dir):
        m = CANON_RE.match(p.name)
        assert m
        eff, side = m.group(1), m.group(2)
        out[(eff, side)] = p
    return out


def materialize_view_from_index(idx: Dict[Tuple[str, str], Path], dst_dir: Path) -> int:
    """
    Write a canonical IR directory representing idx. Overwrites per-file.
    """
    ensure_dir(dst_dir)
    n = 0
    for (eff, side), src in idx.items():
        if not src.exists():
            continue
        dst = dst_dir / f"{eff}_{side}_program.smt2"
        copy_or_link_file(src, dst, use_hardlinks=USE_HARDLINKS)
        n += 1
    return n


def materialize_todo_ir_from_index(
    *,
    idx: Dict[Tuple[str, str], Path],
    pairs: Iterable[Tuple[str, str]],
    todo_dir: Path,
) -> int:
    ensure_dir(todo_dir)
    for p in todo_dir.glob("NCT*_program.smt2"):
        _safe_unlink(p)
    copied = 0
    for pair in pairs:
        src = idx.get(pair)
        if not src:
            continue
        if not src.exists():
            continue
        dst = todo_dir / f"{pair[0]}_{pair[1]}_program.smt2"
        copy_or_link_file(src, dst, use_hardlinks=USE_HARDLINKS)
        copied += 1
    return copied


# -------------------------
# JSONL helpers
# -------------------------
def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(dst: Path, src: Path) -> None:
    ensure_dir(dst.parent)
    if not src.exists():
        return
    data = src.read_text(encoding="utf-8", errors="ignore")
    if not data.strip():
        return
    with dst.open("a", encoding="utf-8") as fout:
        fout.write(data)
        if not data.endswith("\n"):
            fout.write("\n")


# -------------------------
# Quarantine helpers
# -------------------------
def quarantine_file(src: Path, quarantine_dir: Path, reason: str) -> None:
    try:
        ensure_dir(quarantine_dir)
        base = src.name
        dst = quarantine_dir / f"{reason}__{base}"
        i = 1
        while dst.exists():
            dst = quarantine_dir / f"{reason}__{i}__{base}"
            i += 1
        shutil.move(str(src), str(dst))
    except Exception:
        _safe_unlink(src)


# -------------------------
# Stage overlays
# -------------------------
@dataclass(frozen=True)
class OverlayItem:
    src_path: Path
    dst_name: str


def overlays_from_meaning_summary(summary_jsonl: Path) -> List[OverlayItem]:
    items: List[OverlayItem] = []
    for row in read_jsonl(summary_jsonl):
        outp = row.get("meaning_enriched_smt_path")
        eff = row.get("effective_trial_id")
        side = row.get("side")
        if not outp or not eff or side not in ("inclusion", "exclusion"):
            continue
        items.append(OverlayItem(src_path=Path(outp), dst_name=f"{eff}_{side}_program.smt2"))
    return items


def overlays_from_meaning_raw_out(raw_out_dir: Path) -> List[OverlayItem]:
    items: List[OverlayItem] = []
    if not raw_out_dir.exists():
        return items
    for p in raw_out_dir.glob("NCT*_program_meaning_enriched.smt2"):
        m = MEANING_OUT_RE.match(p.name)
        if not m:
            continue
        eff, side = m.group(1), m.group(2)
        items.append(OverlayItem(src_path=p, dst_name=f"{eff}_{side}_program.smt2"))
    return items


def overlays_from_polarity_summary(summary_jsonl: Path) -> List[OverlayItem]:
    items: List[OverlayItem] = []
    for row in read_jsonl(summary_jsonl):
        fixed = row.get("fixed_smt_path")
        eff = row.get("effective_trial_id")
        side = row.get("side")
        if not fixed or not eff or side != "exclusion":
            continue
        items.append(OverlayItem(src_path=Path(fixed), dst_name=f"{eff}_exclusion_program.smt2"))
    return items


def overlays_from_repair_summary(summary_jsonl: Path) -> List[OverlayItem]:
    items: List[OverlayItem] = []
    for row in read_jsonl(summary_jsonl):
        rep = row.get("repaired_smt_path")
        eff = row.get("effective_trial_id")
        side = row.get("side")
        if not rep or not eff or side not in ("inclusion", "exclusion"):
            continue
        items.append(OverlayItem(src_path=Path(rep), dst_name=f"{eff}_{side}_program.smt2"))
    return items


def overlays_from_logic_summary(summary_jsonl: Path) -> List[OverlayItem]:
    items: List[OverlayItem] = []
    for row in read_jsonl(summary_jsonl):
        fixed = row.get("fixed_smt_path")
        eff = row.get("effective_trial_id")
        side = row.get("side")
        if not fixed or not eff or side not in ("inclusion", "exclusion"):
            continue
        items.append(OverlayItem(src_path=Path(fixed), dst_name=f"{eff}_{side}_program.smt2"))
    return items


def apply_overlays(
    merged_dir: Path,
    overlays: List[OverlayItem],
    *,
    use_hardlinks: bool,
    quarantine_dir: Path,
    strict_cfg: GuardrailConfig,
) -> Tuple[int, int, int, int]:
    """
    Apply overlays into merged_dir. Only accepts overlay files that are strict-ok; otherwise quarantines.
    Returns: (applied, missing_src, skipped_empty, skipped_invalid)
    """
    applied = 0
    missing = 0
    skipped_empty = 0
    skipped_invalid = 0

    for it in overlays:
        if not it.src_path.exists():
            missing += 1
            continue

        if not is_nonempty_smt(it.src_path):
            skipped_empty += 1
            quarantine_file(it.src_path, quarantine_dir, "empty_output")
            continue

        ok, _errs = validate_smt_file(it.src_path, cfg=strict_cfg)
        if not ok:
            skipped_invalid += 1
            quarantine_file(it.src_path, quarantine_dir, "format_invalid_strict")
            continue

        copy_or_link_file(it.src_path, merged_dir / it.dst_name, use_hardlinks=use_hardlinks)
        applied += 1

    return applied, missing, skipped_empty, skipped_invalid


# -------------------------
# Sanitize existing raw_out
# -------------------------
def sanitize_existing_raw_out(
    *,
    stage_name: str,
    raw_out_dir: Path,
    kind: str,
    quarantine_dir: Path,
    strict_cfg: GuardrailConfig,
) -> Tuple[int, int]:
    """
    Quarantine files in raw_out that are empty or strict-invalid.
    Returns: (quarantined, strict_invalid)
    """
    quarantined = 0
    invalid = 0

    for p in raw_out_dir.glob("NCT*.smt2"):
        if not is_nonempty_smt(p):
            quarantine_file(p, quarantine_dir, "empty_output")
            quarantined += 1
            continue

        ok, _errs = validate_smt_file(p, cfg=strict_cfg)
        if not ok:
            quarantine_file(p, quarantine_dir, "format_invalid_strict")
            quarantined += 1
            invalid += 1
            continue

    if quarantined:
        print(f"[SANITIZE {stage_name}] quarantined={quarantined} strict_invalid={invalid}", flush=True)
    return quarantined, invalid


def purge_stage_outputs_forced(
    stage_name: str,
    raw_out_dir: Path,
    kind: str,
    forced_pairs: Set[Tuple[str, str]],
    quarantine_dir: Path,
) -> int:
    n = 0
    if not raw_out_dir.exists():
        return 0

    def pair_from_name(name: str) -> Optional[Tuple[str, str]]:
        if kind == "meaning":
            m = MEANING_OUT_RE.match(name)
            return (m.group(1), m.group(2)) if m else None
        if kind == "polarity":
            m = POLARITY_OUT_RE.match(name)
            return (m.group(1), "exclusion") if m else None
        if kind == "repair":
            m = REPAIR_OUT_RE.match(name)
            return (m.group(1), m.group(2)) if m else None
        if kind == "logic":
            m = LOGIC_OUT_RE.match(name)
            return (m.group(1), m.group(2)) if m else None
        return None

    for p in raw_out_dir.glob("NCT*.smt2"):
        pair = pair_from_name(p.name)
        if pair is None:
            continue
        if pair in forced_pairs:
            quarantine_file(p, quarantine_dir, "forced_rerun")
            n += 1

    if n:
        print(f"[FORCE-RERUN] {stage_name}: quarantined {n} outputs from raw_out", flush=True)
    return n


# -------------------------
# Stage0 touched history
# -------------------------
def stage0_touched_pairs_from_summary(stage0_summary: Path) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if not stage0_summary.exists():
        return out
    for r in read_jsonl(stage0_summary):
        eff = r.get("effective_trial_id")
        side = r.get("side")
        outp = r.get("meaning_enriched_smt_path")
        if eff and side in ("inclusion", "exclusion") and outp:
            out.add((str(eff), str(side)))
    return out


# -------------------------
# Attempted indexing + budgeting
# -------------------------
def _meta_attempted(row: Dict) -> bool:
    meta = row.get("meta")
    return isinstance(meta, dict)


def _meta_executed_true(row: Dict) -> bool:
    meta = row.get("meta")
    return isinstance(meta, dict) and meta.get("executed") is True


def build_attempted_last_and_counts(stage_summary: Path, kind: str) -> Tuple[Dict[Tuple[str, str], Dict], Dict[Tuple[str, str], int]]:
    """
    Returns:
      - last attempted row per pair (attempted = any row with meta dict)
      - attempted count per pair
    """
    last: Dict[Tuple[str, str], Dict] = {}
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    if not stage_summary.exists():
        return last, counts

    for r in read_jsonl(stage_summary):
        eff = r.get("effective_trial_id")
        side = r.get("side")
        if not eff or side not in ("inclusion", "exclusion"):
            continue
        if kind == "polarity" and side != "exclusion":
            continue
        if not _meta_attempted(r):
            continue
        pair = (str(eff), str(side))
        last[pair] = r
        counts[pair] += 1

    return last, counts


def stage_canonical_path(stage_merged_ir: Path, pair: Tuple[str, str]) -> Path:
    eff, side = pair
    return stage_merged_ir / f"{eff}_{side}_program.smt2"


@dataclass(frozen=True)
class StagePlan:
    kind: str
    eligible_pairs: Set[Tuple[str, str]]

    attempted_pairs: Set[Tuple[str, str]]
    succeeded_pairs: Set[Tuple[str, str]]

    new_pairs: Set[Tuple[str, str]]
    rerun_strict_pairs: Set[Tuple[str, str]]
    missing_artifact_pairs: Set[Tuple[str, str]]
    forced_pairs: Set[Tuple[str, str]]

    attempted_failed_pairs: Set[Tuple[str, str]]
    budget_exhausted_pairs: Set[Tuple[str, str]]

    remaining_pairs: Set[Tuple[str, str]]


def compute_stage_plan(
    *,
    kind: str,
    eligible_pairs: Set[Tuple[str, str]],
    stage_summary: Path,
    stage_merged_ir: Path,
    strict_cfg: GuardrailConfig,
    forced_pairs: Set[Tuple[str, str]],
    missing_artifact_mode: str = "warn",  # "warn"|"ignore"|"rerun"
) -> StagePlan:
    """
    Rerun policy with fallback+budget:
      - attempted is any row with meta dict
      - new = eligible - attempted
      - rerun_strict: only for pairs whose stage merged canonical is strict-invalid
                     AND attempt_count < MAX_TOTAL_ATTEMPTS_PER_PAIR_PER_STAGE (unless forced)
      - attempted_failed: attempted but last meta.executed != true => never rerun by default
      - budget_exhausted: strict-invalid but attempts exhausted => do not rerun (propagate old)
    """
    last_idx, counts = build_attempted_last_and_counts(stage_summary, kind)

    attempted_pairs = set(last_idx.keys()) & eligible_pairs
    succeeded_pairs: Set[Tuple[str, str]] = set()
    attempted_failed_pairs: Set[Tuple[str, str]] = set()

    for pair in attempted_pairs:
        row = last_idx.get(pair)
        if row is not None and _meta_executed_true(row):
            succeeded_pairs.add(pair)
        else:
            attempted_failed_pairs.add(pair)

    new_pairs = eligible_pairs - attempted_pairs

    rerun_strict: Set[Tuple[str, str]] = set()
    missing_artifact: Set[Tuple[str, str]] = set()
    budget_exhausted: Set[Tuple[str, str]] = set()

    # Only succeeded pairs are candidates for strict-invalid rerun.
    for pair in sorted(succeeded_pairs):
        # if last attempt says executed true, we check the stage merged canonical as the resume artifact
        p = stage_canonical_path(stage_merged_ir, pair)
        if not p.exists():
            missing_artifact.add(pair)
            continue

        ok = strict_ok_file(p, strict_cfg=strict_cfg)
        if ok:
            continue

        # strict-invalid resume artifact -> rerun IF budget available, else stop
        attempts = counts.get(pair, 0)
        if attempts >= MAX_TOTAL_ATTEMPTS_PER_PAIR_PER_STAGE and pair not in forced_pairs:
            budget_exhausted.add(pair)
        else:
            rerun_strict.add(pair)

    remaining = set(new_pairs) | set(rerun_strict) | (forced_pairs & eligible_pairs)
    if missing_artifact_mode == "rerun":
        remaining |= missing_artifact

    # never rerun attempted_failed unless forced
    remaining -= (attempted_failed_pairs - (forced_pairs & eligible_pairs))

    # never rerun budget_exhausted unless forced
    remaining -= (budget_exhausted - (forced_pairs & eligible_pairs))

    return StagePlan(
        kind=kind,
        eligible_pairs=set(eligible_pairs),
        attempted_pairs=set(attempted_pairs),
        succeeded_pairs=set(succeeded_pairs),
        new_pairs=set(new_pairs),
        rerun_strict_pairs=set(rerun_strict),
        missing_artifact_pairs=set(missing_artifact),
        forced_pairs=set(forced_pairs & eligible_pairs),
        attempted_failed_pairs=set(attempted_failed_pairs),
        budget_exhausted_pairs=set(budget_exhausted),
        remaining_pairs=set(remaining),
    )


def audit_strict_failure_reasons_for_pairs(
    *,
    kind: str,
    stage_merged_ir: Path,
    pairs: Set[Tuple[str, str]],
    strict_cfg: GuardrailConfig,
    max_files: int = AUDIT_MAX_FILES,
    topk: int = AUDIT_TOPK_REASONS,
    max_samples_per_reason: int = AUDIT_MAX_SAMPLES_PER_REASON,
) -> None:
    """
    Audit reasons based on the stage merged canonical files (resume artifacts).
    """
    if not pairs:
        print(f"\n[AUDIT {kind}] rerun_strict_pairs=0")
        return

    counts: Counter[str] = Counter()
    samples: DefaultDict[str, List[str]] = defaultdict(list)

    checked = 0
    for pair in sorted(pairs):
        if checked >= max_files:
            counts["__truncated__"] += 1
            break
        checked += 1

        p = stage_canonical_path(stage_merged_ir, pair)
        if not p.exists():
            counts["missing_artifact"] += 1
            continue

        if not is_nonempty_smt(p):
            counts["empty_output"] += 1
            if len(samples["empty_output"]) < max_samples_per_reason:
                samples["empty_output"].append(str(p))
            continue

        ok, errs = validate_smt_file(p, cfg=strict_cfg)
        if ok:
            counts["actually_ok"] += 1
            continue

        for e in errs:
            counts[str(e)] += 1
        primary = str(errs[0]) if errs else "unknown_error"
        if len(samples[primary]) < max_samples_per_reason:
            samples[primary].append(str(p))

    print(f"\n[AUDIT {kind}] pairs={len(pairs)} checked={checked} (cap={max_files})")
    print("  Top reasons (counts are per-error-occurrence; one file may contribute multiple errors):")
    for reason, c in counts.most_common(topk):
        print(f"    {reason}: {c}")
        for sp in samples.get(reason, []):
            print(f"      sample: {sp}")


# -------------------------
# Misc helpers
# -------------------------
def confirm_or_exit(prompt: str, *, yes: bool) -> None:
    if yes:
        print("[CONFIRM] --yes set; proceeding.", flush=True)
        return
    if not sys.stdin.isatty():
        raise SystemExit("[ABORT] Non-interactive stdin and --yes not set.")
    ans = input(prompt).strip().lower()
    if ans not in ("y", "yes"):
        raise SystemExit("[ABORT] User declined.")


def write_allowlist(path: Path, trials: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(set(trials)):
            f.write(t + "\n")


def run_cmd(cmd: List[str], *, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def find_latest_work_dir(parent: Path) -> Optional[Path]:
    cand: List[Path] = []
    for p in parent.iterdir():
        if p.is_dir() and WORKDIR_RE.match(p.name):
            cand.append(p)
    if not cand:
        return None
    cand.sort(key=lambda x: x.name)
    return cand[-1]


@dataclass
class StagePaths:
    stage_dir: Path
    raw_out_dir: Path
    merged_ir_dir: Path
    summary_master_jsonl: Path
    log_dir: Path
    mbench_dir: Path
    quarantine_dir: Path
    todo_dir: Path


def make_stage_paths(work_dir: Path, name: str) -> StagePaths:
    stage_dir = work_dir / name
    raw_out_dir = stage_dir / "raw_out"
    merged_ir_dir = stage_dir / "merged_ir"
    summary_master_jsonl = stage_dir / "summary.jsonl"
    log_dir = stage_dir / "logs"
    mbench_dir = stage_dir / "mbench"
    quarantine_dir = stage_dir / "quarantine_invalid"
    todo_dir = stage_dir / "ir_todo"
    for p in (stage_dir, raw_out_dir, merged_ir_dir, log_dir, mbench_dir, quarantine_dir, todo_dir):
        ensure_dir(p)
    return StagePaths(stage_dir, raw_out_dir, merged_ir_dir, summary_master_jsonl, log_dir, mbench_dir, quarantine_dir, todo_dir)


def materialize_stage0_merged_from_base(
    *,
    base_ir_dir: Path,
    st0: StagePaths,
    strict_cfg: GuardrailConfig,
) -> None:
    """
    Build stage0/merged_ir by copying canonical from base_ir_dir then applying meaning overlays (strict-ok only).
    This is ONLY for materializing the stage0 directory; the stage0 "best view" is computed separately per pair.
    """
    if not has_any_canonical_files(base_ir_dir):
        raise FileNotFoundError(f"Base IR dir has no canonical files: {base_ir_dir}")

    # copy base dir to merged_ir (overwrite)
    base_idx = canonical_index(base_ir_dir)
    materialize_view_from_index(base_idx, st0.merged_ir_dir)

    ov0 = overlays_from_meaning_summary(st0.summary_master_jsonl)
    ov0 += overlays_from_meaning_raw_out(st0.raw_out_dir)
    applied0, missing0, skipped0, invalid0 = apply_overlays(
        st0.merged_ir_dir,
        ov0,
        use_hardlinks=USE_HARDLINKS,
        quarantine_dir=st0.quarantine_dir,
        strict_cfg=strict_cfg,
    )
    print(
        f"[MERGE stage0] overlays={len(ov0)} applied={applied0} missing_src={missing0} skipped_empty={skipped0} skipped_invalid={invalid0}",
        flush=True,
    )


def compute_stage0_best_index(
    *,
    input_idx: Dict[Tuple[str, str], Path],
    stage0_merged_ir: Path,
    strict_cfg: GuardrailConfig,
) -> Dict[Tuple[str, str], Path]:
    """
    Per-pair best view after Stage0:
      use stage0/merged_ir/<pair> if strict-ok, else fall back to input.
    """
    out: Dict[Tuple[str, str], Path] = {}
    for pair, inp in input_idx.items():
        cand = stage0_merged_ir / f"{pair[0]}_{pair[1]}_program.smt2"
        if cand.exists() and strict_ok_file(cand, strict_cfg=strict_cfg):
            out[pair] = cand
        else:
            out[pair] = inp
    return out


# -------------------------
# Declare-const loopback helpers (optional)
# -------------------------
def strict_fail_decl_only_pairs(ir_dir: Path, *, strict_cfg: GuardrailConfig) -> Set[Tuple[str, str]]:
    """
    Return (eff,side) pairs whose canonical program fails strict compliance and whose errors are ONLY declare_const_*.
    """
    out: Set[Tuple[str, str]] = set()
    for p in iter_canonical_ir_files(ir_dir):
        m = CANON_RE.match(p.name)
        if not m:
            continue
        eff, side = m.group(1), m.group(2)
        if not is_nonempty_smt(p):
            continue
        ok, errs = validate_smt_file(p, cfg=strict_cfg)
        if ok:
            continue
        non_decl = [e for e in errs if not str(e).startswith("declare_const_")]
        if non_decl:
            continue
        out.add((eff, side))
    return out


def run_meaning_loopback(
    *,
    scripts_dir: Path,
    meaning_py: Path,
    snapshot_dir: Path,
    st0: StagePaths,
    strict_cfg: GuardrailConfig,
    input_ir_dir: Path,
    loop_pairs: Set[Tuple[str, str]],
    run_tag: str,
) -> None:
    if not loop_pairs:
        return

    purge_stage_outputs_forced("stage0_meaning", st0.raw_out_dir, "meaning", loop_pairs, st0.quarantine_dir)

    todo = st0.stage_dir / "ir_todo_loopback"
    ensure_dir(todo)
    for p in todo.glob("NCT*_program.smt2"):
        _safe_unlink(p)

    copied = 0
    for (eff, side) in sorted(loop_pairs):
        src = input_ir_dir / f"{eff}_{side}_program.smt2"
        if not src.exists() or not is_nonempty_smt(src):
            continue
        dst = todo / src.name
        copy_or_link_file(src, dst, use_hardlinks=USE_HARDLINKS)
        copied += 1

    trials = sorted({eff for (eff, _side) in loop_pairs})
    allow = st0.stage_dir / f"allowlist_loopback_{run_tag}.txt"
    write_allowlist(allow, trials)

    print(
        f"[LOOPBACK->STAGE0] decl-only failing pairs={len(loop_pairs)} trials={len(trials)} todo_ir={todo} copied={copied}",
        flush=True,
    )

    stage0_run_summary = st0.stage_dir / f"summary_run_loopback_{run_tag}.jsonl"
    cmd = [
        sys.executable, str(meaning_py),
        "--ir-dir", str(todo),
        "--snapshot-dir", str(snapshot_dir),
        "--out-ir-dir", str(st0.raw_out_dir),
        "--summary-jsonl", str(stage0_run_summary),
        "--log-dir", str(st0.log_dir),
        "--mbench-dir", str(st0.mbench_dir),
        "--max-workers", str(DEFAULT_MAX_WORKERS_MEANING),
        "--trial-allowlist", str(allow),
        "--side", "both",
    ]
    run_cmd(cmd, cwd=scripts_dir)
    append_jsonl(st0.summary_master_jsonl, stage0_run_summary)

    sanitize_existing_raw_out(
        stage_name="stage0_meaning",
        raw_out_dir=st0.raw_out_dir,
        kind="meaning",
        quarantine_dir=st0.quarantine_dir,
        strict_cfg=strict_cfg,
    )


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-workflow orchestrator: meaning -> polarity -> repair -> logic (strict compliance, resume-safe) with fallback to previous stage and bounded retries."
    )
    p.add_argument("--ir-dir", default="../build/ir", help="Canonical IR dir (NCT*_program.smt2).")
    p.add_argument("--snapshot-dir", default="../subcohort_results", help="Snapshot directory used by stage scripts.")
    p.add_argument("--work-dir", default=None, help="Existing work dir to resume/continue.")
    p.add_argument("--resume", action="store_true", help="Resume mode; if --work-dir omitted picks newest ir_orchestrated_*.")
    p.add_argument("--final-ir-dir", default=None, help="Optional output directory to copy final merged IR.")
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    p.add_argument("--yes", action="store_true", help="Proceed without interactive confirmation prompts.")
    p.add_argument(
        "--rerun-if-input-noncompliant",
        action="store_true",
        help="If ORIGINAL input SMT fails strict compliance, force end-to-end rerun for that (trial,side).",
    )
    p.add_argument(
        "--skip-stage0",
        action="store_true",
        help="Skip executing Stage0 meaning step. Still uses existing stage0 outputs/summary to build stage0/merged_ir and forces downstream reruns for Stage0-touched history.",
    )
    p.add_argument(
        "--skip-assert-checks",
        action="store_true",
        help="Disable checks requiring asserts and :named tags (no_assert_found / missing_named_tags / bad_named_tag).",
    )
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()

    strict_cfg = STRICT_GUARDRAILS_DEFAULT
    if args.skip_assert_checks:
        strict_cfg = replace(strict_cfg, check_asserts_and_named=False)

    if not os.environ.get("OPENAI_ENDPOINT", "").strip():
        raise EnvironmentError("OPENAI_ENDPOINT is required (used by stage scripts).")

    scripts_dir = Path(__file__).resolve().parent
    meaning_py = scripts_dir / "meaning_enrich_irs.py"
    polarity_py = scripts_dir / "polarity_fix_irs.py"
    repair_py = scripts_dir / "repair_all_ir.py"
    logic_py = scripts_dir / "logic_fix_irs.py"
    for sp in (meaning_py, polarity_py, repair_py, logic_py):
        if not sp.exists():
            raise FileNotFoundError(f"Missing stage script: {sp}")

    in_ir_dir = Path(args.ir_dir).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    now_ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    if not in_ir_dir.exists():
        raise FileNotFoundError(f"--ir-dir does not exist: {in_ir_dir}")
    if not has_any_canonical_files(in_ir_dir):
        raise FileNotFoundError(f"No canonical IR files found under: {in_ir_dir}")

    # Work dir resolution
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        if args.resume and not work_dir.exists():
            raise FileNotFoundError(f"--resume set but --work-dir does not exist: {work_dir}")
        ensure_dir(work_dir)
    else:
        if args.resume:
            latest = find_latest_work_dir(in_ir_dir.parent)
            if latest is None:
                raise FileNotFoundError(f"--resume set but no ir_orchestrated_* found under: {in_ir_dir.parent}")
            work_dir = latest.resolve()
        else:
            work_dir = (in_ir_dir.parent / f"ir_orchestrated_{now_ts}").resolve()
            ensure_dir(work_dir)

    run_ts = now_ts

    st0 = make_stage_paths(work_dir, "stage0_meaning")
    st1 = make_stage_paths(work_dir, "stage1_polarity")
    st2 = make_stage_paths(work_dir, "stage2_repair")
    st3 = make_stage_paths(work_dir, "stage3_logic")

    print(f"[INFO] work_dir={work_dir}", flush=True)
    print(f"[INFO] strict_compliance=ON, assert_checks={'OFF' if args.skip_assert_checks else 'ON'}", flush=True)
    print(f"[INFO] retry_budget_per_pair_per_stage={MAX_TOTAL_ATTEMPTS_PER_PAIR_PER_STAGE} (1+MAX_RETRIES_PER_STAGE)", flush=True)

    print("\n[PATHS]")
    print("  Stage0 meaning  raw_out:", st0.raw_out_dir)
    print("  Stage0 meaning merged_ir:", st0.merged_ir_dir)
    print("  Stage1 polarity raw_out:", st1.raw_out_dir)
    print("  Stage1 polarity merged_ir:", st1.merged_ir_dir)
    print("  Stage2 repair   raw_out:", st2.raw_out_dir)
    print("  Stage2 repair  merged_ir:", st2.merged_ir_dir)
    print("  Stage3 logic    raw_out:", st3.raw_out_dir)
    print("  Stage3 logic   merged_ir:", st3.merged_ir_dir)
    print("", flush=True)

    if args.dry_run:
        print("[DRY RUN] exiting.", flush=True)
        return

    # Input canonical index
    input_idx = canonical_index(in_ir_dir)

    # Resume sanitation (safe)
    if args.resume:
        sanitize_existing_raw_out(stage_name="stage0_meaning", raw_out_dir=st0.raw_out_dir, kind="meaning", quarantine_dir=st0.quarantine_dir, strict_cfg=strict_cfg)
        sanitize_existing_raw_out(stage_name="stage1_polarity", raw_out_dir=st1.raw_out_dir, kind="polarity", quarantine_dir=st1.quarantine_dir, strict_cfg=strict_cfg)
        sanitize_existing_raw_out(stage_name="stage2_repair", raw_out_dir=st2.raw_out_dir, kind="repair", quarantine_dir=st2.quarantine_dir, strict_cfg=strict_cfg)
        sanitize_existing_raw_out(stage_name="stage3_logic", raw_out_dir=st3.raw_out_dir, kind="logic", quarantine_dir=st3.quarantine_dir, strict_cfg=strict_cfg)

    # -------------------------
    # Stage0: either run or skip, but ensure stage0/merged_ir exists
    # -------------------------
    # Choose base input for materializing stage0 merged.
    # In resume mode, if stage0 merged exists we keep it; otherwise base is input.
    base_for_stage0_merge = st0.merged_ir_dir if (args.resume and has_any_canonical_files(st0.merged_ir_dir)) else in_ir_dir

    if args.skip_stage0:
        print("[STAGE0] --skip-stage0 set; will not execute Stage0.", flush=True)
        if not has_any_canonical_files(st0.merged_ir_dir):
            print("[STAGE0] materializing stage0/merged_ir from existing stage0 artifacts (no LLM).", flush=True)
            materialize_stage0_merged_from_base(base_ir_dir=base_for_stage0_merge, st0=st0, strict_cfg=strict_cfg)
    else:
        # Minimal Stage0 behavior in this orchestrator: we assume stage0 was already run or handled elsewhere.
        # If you still want the earlier "need_meaning_pairs" planner, add it back; this file focuses on fallback semantics.
        if not has_any_canonical_files(st0.merged_ir_dir):
            print("[STAGE0] no stage0/merged_ir present; initializing from input.", flush=True)
            materialize_stage0_merged_from_base(base_ir_dir=in_ir_dir, st0=st0, strict_cfg=strict_cfg)

    if not has_any_canonical_files(st0.merged_ir_dir):
        raise FileNotFoundError("stage0_meaning/merged_ir has no canonical files; cannot proceed.")

    # Stage0 history forcing (for downstream reruns)
    forced_pairs_stage0_history: Set[Tuple[str, str]] = set()
    if args.skip_stage0:
        forced_pairs_stage0_history = stage0_touched_pairs_from_summary(st0.summary_master_jsonl)
        if forced_pairs_stage0_history:
            print(f"[STAGE0-HISTORY] forcing downstream reruns for {len(forced_pairs_stage0_history)} pairs touched by Stage0 in previous runs.", flush=True)
            purge_stage_outputs_forced("stage1_polarity", st1.raw_out_dir, "polarity", forced_pairs_stage0_history, st1.quarantine_dir)
            purge_stage_outputs_forced("stage2_repair", st2.raw_out_dir, "repair", forced_pairs_stage0_history, st2.quarantine_dir)
            purge_stage_outputs_forced("stage3_logic", st3.raw_out_dir, "logic", forced_pairs_stage0_history, st3.quarantine_dir)

    # Compute Stage0 best view (per pair): stage0 strict-ok else input
    best0_idx = compute_stage0_best_index(input_idx=input_idx, stage0_merged_ir=st0.merged_ir_dir, strict_cfg=strict_cfg)

    # Optional E2E forced set: if input noncompliant, force reruns (still bounded by budget unless you also purge)
    forced_pairs_e2e: Set[Tuple[str, str]] = set()
    if args.rerun_if_input_noncompliant:
        for pair, p in input_idx.items():
            if not strict_ok_file(p, strict_cfg=strict_cfg):
                forced_pairs_e2e.add(pair)
        print(f"[E2E-RERUN] enabled. input_noncompliant_pairs={len(forced_pairs_e2e)}", flush=True)
        purge_stage_outputs_forced("stage0_meaning", st0.raw_out_dir, "meaning", forced_pairs_e2e, st0.quarantine_dir)
        purge_stage_outputs_forced("stage1_polarity", st1.raw_out_dir, "polarity", forced_pairs_e2e, st1.quarantine_dir)
        purge_stage_outputs_forced("stage2_repair", st2.raw_out_dir, "repair", forced_pairs_e2e, st2.quarantine_dir)
        purge_stage_outputs_forced("stage3_logic", st3.raw_out_dir, "logic", forced_pairs_e2e, st3.quarantine_dir)

    # -------------------------
    # Pipeline passes (bounded) with optional decl loopback
    # -------------------------
    forced_pairs_loopback: Set[Tuple[str, str]] = set()
    final_merged_dir: Path = st3.merged_ir_dir

    for pass_i in range(MAX_DECL_LOOPBACK_PASSES + 1):
        pass_tag = f"{run_ts}_pass{pass_i}"
        print(f"\n========== [PIPELINE PASS {pass_i}] tag={pass_tag} ==========", flush=True)

        # Total forced pairs (downstream)
        forced_pairs_total: Set[Tuple[str, str]] = set(forced_pairs_stage0_history) | set(forced_pairs_e2e) | set(forced_pairs_loopback)

        # -------------------------
        # Stage1 planning (eligible from best0)
        # -------------------------
        eligible1: Set[Tuple[str, str]] = set()
        for (eff, side), p in best0_idx.items():
            if side != "exclusion":
                continue
            if is_nonempty_smt(p):
                eligible1.add((eff, "exclusion"))

        # Ensure stage1 merged exists as a resume artifact view:
        # base = best0 view, then overlay prior strict-ok stage1 outputs (if any)
        materialize_view_from_index(best0_idx, st1.merged_ir_dir)
        ov1_init = overlays_from_polarity_summary(st1.summary_master_jsonl)
        apply_overlays(st1.merged_ir_dir, ov1_init, use_hardlinks=USE_HARDLINKS, quarantine_dir=st1.quarantine_dir, strict_cfg=strict_cfg)

        p1 = compute_stage_plan(
            kind="polarity",
            eligible_pairs=eligible1,
            stage_summary=st1.summary_master_jsonl,
            stage_merged_ir=st1.merged_ir_dir,
            strict_cfg=strict_cfg,
            forced_pairs={(eff, "exclusion") for (eff, _s) in forced_pairs_total},
            missing_artifact_mode="warn",
        )

        # -------------------------
        # Stage2 planning (eligible from stage1 merged)
        # -------------------------
        canon_after1 = canonical_index(st1.merged_ir_dir)
        eligible2 = {(eff, side) for (eff, side), p in canon_after1.items() if is_nonempty_smt(p)}

        # Ensure stage2 merged exists as resume artifact view:
        materialize_view_from_index(canon_after1, st2.merged_ir_dir)
        ov2_init = overlays_from_repair_summary(st2.summary_master_jsonl)
        apply_overlays(st2.merged_ir_dir, ov2_init, use_hardlinks=USE_HARDLINKS, quarantine_dir=st2.quarantine_dir, strict_cfg=strict_cfg)

        p2 = compute_stage_plan(
            kind="repair",
            eligible_pairs=eligible2,
            stage_summary=st2.summary_master_jsonl,
            stage_merged_ir=st2.merged_ir_dir,
            strict_cfg=strict_cfg,
            forced_pairs=forced_pairs_total,
            missing_artifact_mode="warn",
        )

        # -------------------------
        # Stage3 planning (eligible from stage2 merged)
        # -------------------------
        canon_after2 = canonical_index(st2.merged_ir_dir)
        eligible3: Set[Tuple[str, str]] = set()
        for (eff, side), p in canon_after2.items():
            if not is_nonempty_smt(p):
                continue
            if LOGIC_SIDE == "both" or side == LOGIC_SIDE:
                eligible3.add((eff, side))

        materialize_view_from_index(canon_after2, st3.merged_ir_dir)
        ov3_init = overlays_from_logic_summary(st3.summary_master_jsonl)
        apply_overlays(st3.merged_ir_dir, ov3_init, use_hardlinks=USE_HARDLINKS, quarantine_dir=st3.quarantine_dir, strict_cfg=strict_cfg)

        p3 = compute_stage_plan(
            kind="logic",
            eligible_pairs=eligible3,
            stage_summary=st3.summary_master_jsonl,
            stage_merged_ir=st3.merged_ir_dir,
            strict_cfg=strict_cfg,
            forced_pairs=forced_pairs_total,
            missing_artifact_mode="warn",
        )

        rem1_trials = sorted({eff for (eff, side) in p1.remaining_pairs if side == "exclusion"})
        rem2_pairs = sorted(p2.remaining_pairs)
        rem3_pairs = sorted(p3.remaining_pairs)
        rem3_trials = sorted({eff for (eff, _side) in rem3_pairs})

        print("\n[PLAN STAGE1-3] (fallback-to-prev + bounded retries)")
        print(
            f"  Stage1 polarity: eligible={len(p1.eligible_pairs)} attempted={len(p1.attempted_pairs)} succeeded={len(p1.succeeded_pairs)} "
            f"remaining_trials={len(rem1_trials)} (new={len(p1.new_pairs)}, rerun_strict={len(p1.rerun_strict_pairs)}, "
            f"attempted_failed={len(p1.attempted_failed_pairs)}, budget_exhausted={len(p1.budget_exhausted_pairs)})"
        )
        print(
            f"  Stage2 repair:   eligible={len(p2.eligible_pairs)} attempted={len(p2.attempted_pairs)} succeeded={len(p2.succeeded_pairs)} "
            f"remaining_pairs={len(rem2_pairs)} (new={len(p2.new_pairs)}, rerun_strict={len(p2.rerun_strict_pairs)}, "
            f"attempted_failed={len(p2.attempted_failed_pairs)}, budget_exhausted={len(p2.budget_exhausted_pairs)})"
        )
        print(
            f"  Stage3 logic:    eligible={len(p3.eligible_pairs)} attempted={len(p3.attempted_pairs)} succeeded={len(p3.succeeded_pairs)} "
            f"remaining_pairs={len(rem3_pairs)} (new={len(p3.new_pairs)}, rerun_strict={len(p3.rerun_strict_pairs)}, "
            f"attempted_failed={len(p3.attempted_failed_pairs)}, budget_exhausted={len(p3.budget_exhausted_pairs)}) (trials={len(rem3_trials)})"
        )
        print(f"  forced_pairs_total={len(forced_pairs_total)}", flush=True)

        audit_strict_failure_reasons_for_pairs(kind="polarity", stage_merged_ir=st1.merged_ir_dir, pairs=p1.rerun_strict_pairs, strict_cfg=strict_cfg)
        audit_strict_failure_reasons_for_pairs(kind="repair", stage_merged_ir=st2.merged_ir_dir, pairs=p2.rerun_strict_pairs, strict_cfg=strict_cfg)
        audit_strict_failure_reasons_for_pairs(kind="logic", stage_merged_ir=st3.merged_ir_dir, pairs=p3.rerun_strict_pairs, strict_cfg=strict_cfg)

        if pass_i == 0:
            confirm_or_exit("Proceed with Stage1-3 runs? [y/N] ", yes=args.yes)

        # -------------------------
        # Stage1 execution (polarity): TODO dir from best0_idx
        # -------------------------
        for attempt in range(MAX_RETRIES_PER_STAGE + 1):
            p1 = compute_stage_plan(
                kind="polarity",
                eligible_pairs=eligible1,
                stage_summary=st1.summary_master_jsonl,
                stage_merged_ir=st1.merged_ir_dir,
                strict_cfg=strict_cfg,
                forced_pairs={(eff, "exclusion") for (eff, _s) in forced_pairs_total},
                missing_artifact_mode="warn",
            )
            rem1_trials = sorted({eff for (eff, side) in p1.remaining_pairs if side == "exclusion"})
            tag = "STAGE1" if attempt == 0 else f"STAGE1 RETRY {attempt}"
            print(
                f"[{tag}] remaining_trials={len(rem1_trials)} "
                f"(new={len(p1.new_pairs)} rerun_strict={len(p1.rerun_strict_pairs)} forced={len(p1.forced_pairs)} "
                f"attempted_failed={len(p1.attempted_failed_pairs)} budget_exhausted={len(p1.budget_exhausted_pairs)})",
                flush=True,
            )
            if not rem1_trials:
                print("[STAGE1] nothing remaining; skipping execution.", flush=True)
                break

            todo1 = st1.todo_dir / f"{pass_tag}_attempt{attempt}"
            pairs1 = [(eff, "exclusion") for eff in rem1_trials]
            copied1 = materialize_todo_ir_from_index(idx=best0_idx, pairs=pairs1, todo_dir=todo1)
            print(f"[STAGE1] todo_ir={todo1} copied={copied1}", flush=True)

            stage1_run_summary = st1.stage_dir / f"summary_run_{pass_tag}_attempt{attempt}.jsonl"
            allow1 = st1.stage_dir / f"allowlist_remaining_{pass_tag}_attempt{attempt}.txt"
            write_allowlist(allow1, rem1_trials)

            run_cmd(
                [
                    sys.executable, str(polarity_py),
                    "--ir-dir", str(todo1),
                    "--snapshot-dir", str(snapshot_dir),
                    "--out-ir-dir", str(st1.raw_out_dir),
                    "--summary-jsonl", str(stage1_run_summary),
                    "--log-dir", str(st1.log_dir),
                    "--mbench-dir", str(st1.mbench_dir),
                    "--max-workers", str(DEFAULT_MAX_WORKERS_POLARITY),
                    "--trial-allowlist", str(allow1),
                ],
                cwd=scripts_dir,
            )
            append_jsonl(st1.summary_master_jsonl, stage1_run_summary)
            sanitize_existing_raw_out(stage_name="stage1_polarity", raw_out_dir=st1.raw_out_dir, kind="polarity", quarantine_dir=st1.quarantine_dir, strict_cfg=strict_cfg)

            # Rebuild stage1 merged: base=best0 view, overlay=strict-ok polarity outputs
            materialize_view_from_index(best0_idx, st1.merged_ir_dir)
            ov1 = overlays_from_polarity_summary(st1.summary_master_jsonl)
            applied1, missing1, skipped1, invalid1 = apply_overlays(
                st1.merged_ir_dir,
                ov1,
                use_hardlinks=USE_HARDLINKS,
                quarantine_dir=st1.quarantine_dir,
                strict_cfg=strict_cfg,
            )
            print(
                f"[MERGE stage1] overlays={len(ov1)} applied={applied1} missing_src={missing1} skipped_empty={skipped1} skipped_invalid={invalid1}",
                flush=True,
            )

        # -------------------------
        # Stage2 execution (repair): TODO dir from stage1 merged (which already propagates old versions)
        # -------------------------
        canon_after1 = canonical_index(st1.merged_ir_dir)

        for attempt in range(MAX_RETRIES_PER_STAGE + 1):
            p2 = compute_stage_plan(
                kind="repair",
                eligible_pairs=eligible2,
                stage_summary=st2.summary_master_jsonl,
                stage_merged_ir=st2.merged_ir_dir,
                strict_cfg=strict_cfg,
                forced_pairs=forced_pairs_total,
                missing_artifact_mode="warn",
            )
            rem2_pairs = sorted(p2.remaining_pairs)
            tag = "STAGE2" if attempt == 0 else f"STAGE2 RETRY {attempt}"
            print(
                f"[{tag}] remaining_pairs={len(rem2_pairs)} "
                f"(new={len(p2.new_pairs)} rerun_strict={len(p2.rerun_strict_pairs)} forced={len(p2.forced_pairs)} "
                f"attempted_failed={len(p2.attempted_failed_pairs)} budget_exhausted={len(p2.budget_exhausted_pairs)})",
                flush=True,
            )
            if not rem2_pairs:
                print("[STAGE2] nothing remaining; skipping execution.", flush=True)
                break

            todo2 = st2.todo_dir / f"{pass_tag}_attempt{attempt}"
            copied2 = materialize_todo_ir_from_index(idx=canon_after1, pairs=rem2_pairs, todo_dir=todo2)
            print(f"[STAGE2] todo_ir={todo2} copied={copied2}", flush=True)

            stage2_run_summary = st2.stage_dir / f"summary_run_{pass_tag}_attempt{attempt}.jsonl"
            run_cmd(
                [
                    sys.executable, str(repair_py),
                    "--ir-dir", str(todo2),
                    "--snapshot-dir", str(snapshot_dir),
                    "--repaired-ir-dir", str(st2.raw_out_dir),
                    "--summary-jsonl", str(stage2_run_summary),
                    "--log-dir", str(st2.log_dir),
                    "--mbench-dir", str(st2.mbench_dir),
                    "--max-workers", str(DEFAULT_MAX_WORKERS_REPAIR),
                ],
                cwd=scripts_dir,
            )
            append_jsonl(st2.summary_master_jsonl, stage2_run_summary)
            sanitize_existing_raw_out(stage_name="stage2_repair", raw_out_dir=st2.raw_out_dir, kind="repair", quarantine_dir=st2.quarantine_dir, strict_cfg=strict_cfg)

            # Rebuild stage2 merged: base=stage1 merged, overlay=strict-ok repair outputs
            materialize_view_from_index(canon_after1, st2.merged_ir_dir)
            ov2 = overlays_from_repair_summary(st2.summary_master_jsonl)
            applied2, missing2, skipped2, invalid2 = apply_overlays(
                st2.merged_ir_dir,
                ov2,
                use_hardlinks=USE_HARDLINKS,
                quarantine_dir=st2.quarantine_dir,
                strict_cfg=strict_cfg,
            )
            print(
                f"[MERGE stage2] overlays={len(ov2)} applied={applied2} missing_src={missing2} skipped_empty={skipped2} skipped_invalid={invalid2}",
                flush=True,
            )

        # -------------------------
        # Stage3 execution (logic): reads from stage2 merged; TODO via allowlist only
        # -------------------------
        canon_after2 = canonical_index(st2.merged_ir_dir)
        eligible3 = {(eff, side) for (eff, side), p in canon_after2.items() if is_nonempty_smt(p)}

        for attempt in range(MAX_RETRIES_PER_STAGE + 1):
            p3 = compute_stage_plan(
                kind="logic",
                eligible_pairs=eligible3,
                stage_summary=st3.summary_master_jsonl,
                stage_merged_ir=st3.merged_ir_dir,
                strict_cfg=strict_cfg,
                forced_pairs=forced_pairs_total,
                missing_artifact_mode="warn",
            )
            rem3_pairs = sorted(p3.remaining_pairs)
            rem3_trials = sorted({eff for (eff, _side) in rem3_pairs})
            tag = "STAGE3" if attempt == 0 else f"STAGE3 RETRY {attempt}"
            print(
                f"[{tag}] remaining_pairs={len(rem3_pairs)} remaining_trials={len(rem3_trials)} "
                f"(new={len(p3.new_pairs)} rerun_strict={len(p3.rerun_strict_pairs)} forced={len(p3.forced_pairs)} "
                f"attempted_failed={len(p3.attempted_failed_pairs)} budget_exhausted={len(p3.budget_exhausted_pairs)})",
                flush=True,
            )
            if not rem3_pairs:
                print("[STAGE3] nothing remaining; skipping execution.", flush=True)
                break

            stage3_run_summary = st3.stage_dir / f"summary_run_{pass_tag}_attempt{attempt}.jsonl"
            allow3 = st3.stage_dir / f"allowlist_remaining_{pass_tag}_attempt{attempt}.txt"
            write_allowlist(allow3, rem3_trials)

            run_cmd(
                [
                    sys.executable, str(logic_py),
                    "--ir-dir", str(st2.merged_ir_dir),
                    "--snapshot-dir", str(snapshot_dir),
                    "--out-ir-dir", str(st3.raw_out_dir),
                    "--summary-jsonl", str(stage3_run_summary),
                    "--log-dir", str(st3.log_dir),
                    "--mbench-dir", str(st3.mbench_dir),
                    "--max-workers", str(DEFAULT_MAX_WORKERS_LOGIC),
                    "--trial-allowlist", str(allow3),
                    "--side", LOGIC_SIDE,
                ],
                cwd=scripts_dir,
            )
            append_jsonl(st3.summary_master_jsonl, stage3_run_summary)
            sanitize_existing_raw_out(stage_name="stage3_logic", raw_out_dir=st3.raw_out_dir, kind="logic", quarantine_dir=st3.quarantine_dir, strict_cfg=strict_cfg)

            # Rebuild stage3 merged: base=stage2 merged, overlay=strict-ok logic outputs
            materialize_view_from_index(canon_after2, st3.merged_ir_dir)
            ov3 = overlays_from_logic_summary(st3.summary_master_jsonl)
            applied3, missing3, skipped3, invalid3 = apply_overlays(
                st3.merged_ir_dir,
                ov3,
                use_hardlinks=USE_HARDLINKS,
                quarantine_dir=st3.quarantine_dir,
                strict_cfg=strict_cfg,
            )
            print(
                f"[MERGE stage3] overlays={len(ov3)} applied={applied3} missing_src={missing3} skipped_empty={skipped3} skipped_invalid={invalid3}",
                flush=True,
            )

        final_merged_dir = st3.merged_ir_dir
        print(f"\n[PASS {pass_i} DONE] merged IR directory: {final_merged_dir}", flush=True)

        # -------------------------
        # Optional declare-const loopback
        # -------------------------
        decl_pairs = strict_fail_decl_only_pairs(final_merged_dir, strict_cfg=strict_cfg)
        if not decl_pairs:
            print("[DECL-LOOPBACK] none needed; pipeline converged.", flush=True)
            break
        if pass_i >= MAX_DECL_LOOPBACK_PASSES:
            print(
                f"[DECL-LOOPBACK] still have decl-only strict failures after {pass_i} pass(es): {len(decl_pairs)} pairs. "
                f"Reached MAX_DECL_LOOPBACK_PASSES={MAX_DECL_LOOPBACK_PASSES}; stopping loopback.",
                flush=True,
            )
            break

        print(
            f"[DECL-LOOPBACK] detected {len(decl_pairs)} decl-only strict-failing pairs in final merged IR; "
            f"looping back through meaning fixer then re-running Stage1-3.",
            flush=True,
        )

        run_meaning_loopback(
            scripts_dir=scripts_dir,
            meaning_py=meaning_py,
            snapshot_dir=snapshot_dir,
            st0=st0,
            strict_cfg=strict_cfg,
            input_ir_dir=final_merged_dir,
            loop_pairs=decl_pairs,
            run_tag=f"{pass_tag}_declloop",
        )

        # Re-materialize stage0 merged on top of current best IR (final merged)
        materialize_stage0_merged_from_base(base_ir_dir=final_merged_dir, st0=st0, strict_cfg=strict_cfg)
        best0_idx = compute_stage0_best_index(input_idx=input_idx, stage0_merged_ir=st0.merged_ir_dir, strict_cfg=strict_cfg)

        forced_pairs_loopback |= set(decl_pairs)
        purge_stage_outputs_forced("stage1_polarity", st1.raw_out_dir, "polarity", {(eff, "exclusion") for (eff, _s) in decl_pairs}, st1.quarantine_dir)
        purge_stage_outputs_forced("stage2_repair", st2.raw_out_dir, "repair", decl_pairs, st2.quarantine_dir)
        purge_stage_outputs_forced("stage3_logic", st3.raw_out_dir, "logic", decl_pairs, st3.quarantine_dir)

    print(f"\n[DONE] Final merged IR directory: {final_merged_dir}", flush=True)

    if args.final_ir_dir:
        final_dir = Path(args.final_ir_dir).resolve()
        ensure_dir(final_dir)
        # copy canonical files from final_merged_dir
        final_idx = canonical_index(final_merged_dir)
        n = materialize_view_from_index(final_idx, final_dir)
        print(f"[DONE] Copied final IR to: {final_dir} (files={n})", flush=True)


if __name__ == "__main__":
    main()
