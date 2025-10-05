#!/usr/bin/env python3
"""
recover_to_new_workdir.py  (MODIFIED: multi-stage modified => rerun full chain + preview + confirm)

Non-destructive recovery into a new *_repaired work dir, plus an optional paid rerun
policy that matches your requested logic:

Policy:
  - Detect which stages ACTUALLY modified each (trial, side) using content hashes.
  - Aggregate to per-trial modified stages among:
        S1 = polarity (exclusion only)
        S2 = repair    (both)
        S3 = logic     (both)
  - If a trial has >=2 distinct modified stages => FULL RERUN that trial through:
        Stage1 (exclusion only) -> Stage2 (both sides) -> Stage3 (both sides)
  - Otherwise, do NOT rerun unless the trial still fails strict checks in final merged IR;
    for those strict-failing trials (and not full rerun), do MINIMAL FIX (Stage3-only).

Preview & confirmation:
  - Before executing any paid reruns, prints example trial lists showing which stages modified them.
  - If --execute is set and --yes is NOT set, prompts once for confirmation.

Writes:
  - Everything goes under --out-work-dir (default: work_dir + "_repaired")
  - Original --work-dir is never modified.

Recovery steps:
1) Copy strict-ok outputs from original stage*/raw_out -> repaired stage*/raw_out
2) Copy strict-ok outputs from original stage*/quarantine_invalid -> repaired stage*/raw_out
3) Best-effort copy strict-ok *.smt2 from original stage*/mbench -> repaired stage*/raw_out
4) Offline rebuild repaired stage1/2/3 merged_ir using recovered outputs + any strict-ok summary-path outputs still existing
5) Compute modified stages and rerun plan
6) Optionally execute reruns into repaired folder and offline re-merge again

Notes:
- Strict checks must match your intended regime (use --skip-assert-checks if needed).
- Hashing is whitespace-insensitive; comments are NOT stripped for hashing (can be added if needed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict, Counter


# -------------------------
# Regex patterns (match your orchestrator)
# -------------------------
CANON_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program\.smt2$")

MEANING_OUT_RE  = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_meaning_enriched\.smt2$")
POLARITY_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_exclusion_program_polarity_fixed\.smt2$")
REPAIR_OUT_RE   = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_repaired\.smt2$")
LOGIC_OUT_RE    = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_logic_fixed\.smt2$")

# -------------------------
# Strict checks (match orchestrate_ir_fixes.py)
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
    check_asserts_and_named: bool


STRICT_GUARDRAILS_DEFAULT = GuardrailConfig(
    allow_extra_decl_json_keys=False,
    strict_double_semicolon=False,
    decl_json_strict=True,
    allow_any_sort=False,
    check_asserts_and_named=True,
)

USE_HARDLINKS = False  # keep it safe; use copies


# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    if extra and (not cfg.allow_extra_decl_json_keys):
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
    if dst.exists():
        dst.unlink()
    if use_hardlinks:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


# -------------------------
# Content hashing for "did modify?"
# -------------------------
_WHITESPACE_RE = re.compile(r"\s+", re.MULTILINE)


def smt_fingerprint(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if not txt.strip():
        return None
    norm = _WHITESPACE_RE.sub(" ", txt).strip()
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()


def did_modify(base_p: Path, out_p: Path) -> bool:
    hb = smt_fingerprint(base_p)
    ho = smt_fingerprint(out_p)
    if hb is None or ho is None:
        return False
    return hb != ho


# -------------------------
# Canonical indexing helpers
# -------------------------
def iter_canonical_ir_files(ir_dir: Path) -> Iterable[Path]:
    for p in ir_dir.glob("NCT*_program.smt2"):
        if CANON_RE.match(p.name):
            yield p


def canonical_index(ir_dir: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for p in iter_canonical_ir_files(ir_dir):
        m = CANON_RE.match(p.name)
        assert m
        eff, side = m.group(1), m.group(2)
        out[(eff, side)] = p
    return out


def materialize_view_from_index(idx: Dict[Tuple[str, str], Path], dst_dir: Path) -> int:
    ensure_dir(dst_dir)
    n = 0
    for (eff, side), src in idx.items():
        if not src.exists():
            continue
        dst = dst_dir / f"{eff}_{side}_program.smt2"
        copy_or_link_file(src, dst, use_hardlinks=USE_HARDLINKS)
        n += 1
    return n


def expected_canon_name(pair: Tuple[str, str]) -> str:
    eff, side = pair
    return f"{eff}_{side}_program.smt2"


# -------------------------
# Summary helpers (still used to locate strict-ok outputs referenced by summary)
# -------------------------
def read_jsonl_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def pair_from_row(row: Dict, *, kind: str) -> Optional[Tuple[str, str]]:
    eff = row.get("effective_trial_id")
    side = row.get("side")
    if not eff or side not in ("inclusion", "exclusion"):
        return None
    if kind == "polarity" and side != "exclusion":
        return None
    return (str(eff), str(side))


# -------------------------
# Stage paths (src vs dst)
# -------------------------
@dataclass(frozen=True)
class StageSrc:
    stage_dir: Path
    raw_out_dir: Path
    merged_ir_dir: Path
    summary_master_jsonl: Path
    quarantine_dir: Path
    mbench_dir: Path


@dataclass(frozen=True)
class StageDst:
    stage_dir: Path
    raw_out_dir: Path
    merged_ir_dir: Path
    summary_master_jsonl: Path
    quarantine_dir: Path
    mbench_dir: Path
    logs_dir: Path
    todo_dir: Path


def make_stage_src(work_dir: Path, name: str) -> StageSrc:
    sd = work_dir / name
    return StageSrc(
        stage_dir=sd,
        raw_out_dir=sd / "raw_out",
        merged_ir_dir=sd / "merged_ir",
        summary_master_jsonl=sd / "summary.jsonl",
        quarantine_dir=sd / "quarantine_invalid",
        mbench_dir=sd / "mbench",
    )


def make_stage_dst(out_work_dir: Path, name: str) -> StageDst:
    sd = out_work_dir / name
    raw = sd / "raw_out"
    merged = sd / "merged_ir"
    summ = sd / "summary.jsonl"
    quar = sd / "quarantine_invalid"
    mb = sd / "mbench"
    logs = sd / "logs"
    todo = sd / "ir_todo"
    for p in (sd, raw, merged, quar, mb, logs, todo):
        ensure_dir(p)
    return StageDst(
        stage_dir=sd,
        raw_out_dir=raw,
        merged_ir_dir=merged,
        summary_master_jsonl=summ,
        quarantine_dir=quar,
        mbench_dir=mb,
        logs_dir=logs,
        todo_dir=todo,
    )


# -------------------------
# Recovery into dst/raw_out
# -------------------------
def match_kind_by_filename(fname: str) -> Optional[str]:
    if MEANING_OUT_RE.match(fname):
        return "meaning"
    if POLARITY_OUT_RE.match(fname):
        return "polarity"
    if REPAIR_OUT_RE.match(fname):
        return "repair"
    if LOGIC_OUT_RE.match(fname):
        return "logic"
    return None


def extract_base_filename_from_quarantine(name: str) -> Optional[str]:
    parts = name.split("__")
    if len(parts) < 2:
        return None
    base = parts[-1]
    if not base.endswith(".smt2"):
        return None
    return base


def copy_strict_ok_from_src_raw_out(src: StageSrc, dst: StageDst, *, strict_cfg: GuardrailConfig) -> int:
    n = 0
    if not src.raw_out_dir.exists():
        return 0
    for p in src.raw_out_dir.glob("*.smt2"):
        if match_kind_by_filename(p.name) is None:
            continue
        if not strict_ok_file(p, strict_cfg=strict_cfg):
            continue
        outp = dst.raw_out_dir / p.name
        if outp.exists():
            continue
        shutil.copy2(p, outp)
        n += 1
    return n


def restore_strict_ok_from_src_quarantine(src: StageSrc, dst: StageDst, *, strict_cfg: GuardrailConfig) -> int:
    n = 0
    if not src.quarantine_dir.exists():
        return 0
    for q in src.quarantine_dir.glob("*.smt2"):
        base = extract_base_filename_from_quarantine(q.name)
        if not base:
            continue
        if match_kind_by_filename(base) is None:
            continue
        if not strict_ok_file(q, strict_cfg=strict_cfg):
            continue
        outp = dst.raw_out_dir / base
        if outp.exists():
            continue
        shutil.copy2(q, outp)
        n += 1
    return n


def restore_strict_ok_from_src_mbench(src: StageSrc, dst: StageDst, *, strict_cfg: GuardrailConfig) -> int:
    n = 0
    if not src.mbench_dir.exists():
        return 0
    for p in src.mbench_dir.rglob("*.smt2"):
        if match_kind_by_filename(p.name) is None:
            continue
        if not strict_ok_file(p, strict_cfg=strict_cfg):
            continue
        outp = dst.raw_out_dir / p.name
        if outp.exists():
            continue
        shutil.copy2(p, outp)
        n += 1
    return n


# -------------------------
# Offline merges into dst/merged_ir, plus record "chosen overlay" per pair
# -------------------------
def pair_from_output_filename(kind: str, fname: str) -> Optional[Tuple[str, str]]:
    if kind == "polarity":
        m = POLARITY_OUT_RE.match(fname)
        return (m.group(1), "exclusion") if m else None
    if kind == "repair":
        m = REPAIR_OUT_RE.match(fname)
        return (m.group(1), m.group(2)) if m else None
    if kind == "logic":
        m = LOGIC_OUT_RE.match(fname)
        return (m.group(1), m.group(2)) if m else None
    if kind == "meaning":
        m = MEANING_OUT_RE.match(fname)
        return (m.group(1), m.group(2)) if m else None
    return None


def collect_best_from_dst_raw_out(dst: StageDst, *, kind: str, strict_cfg: GuardrailConfig) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    best_mtime: Dict[Tuple[str, str], float] = {}
    for p in dst.raw_out_dir.glob("*.smt2"):
        pair = pair_from_output_filename(kind, p.name)
        if pair is None:
            continue
        if not strict_ok_file(p, strict_cfg=strict_cfg):
            continue
        mt = p.stat().st_mtime
        if pair not in best_mtime or mt > best_mtime[pair]:
            best_mtime[pair] = mt
            out[pair] = p
    return out


def collect_best_from_src_summary_paths(src: StageSrc, *, kind: str, strict_cfg: GuardrailConfig) -> Dict[Tuple[str, str], Path]:
    key = {
        "meaning": "meaning_enriched_smt_path",
        "polarity": "fixed_smt_path",
        "repair": "repaired_smt_path",
        "logic": "fixed_smt_path",
    }[kind]
    rows = read_jsonl_rows(src.summary_master_jsonl)
    out: Dict[Tuple[str, str], Path] = {}
    for r in rows:
        pair = pair_from_row(r, kind=kind)
        if pair is None:
            continue
        pth = r.get(key)
        if not pth:
            continue
        p = Path(str(pth))
        if not p.exists():
            continue
        if not strict_ok_file(p, strict_cfg=strict_cfg):
            continue
        out[pair] = p  # last wins
    return out


@dataclass(frozen=True)
class MergeResult:
    idx: Dict[Tuple[str, str], Path]                 # canonical_index of merged dir
    chosen_overlay: Dict[Tuple[str, str], Path]      # pair -> chosen overlay source path (if any)


def merge_stage_offline_record(
    *,
    base_idx: Dict[Tuple[str, str], Path],
    src: StageSrc,
    dst: StageDst,
    kind: str,
    strict_cfg: GuardrailConfig,
) -> MergeResult:
    materialize_view_from_index(base_idx, dst.merged_ir_dir)

    raw_best = collect_best_from_dst_raw_out(dst, kind=kind, strict_cfg=strict_cfg)
    sum_best = collect_best_from_src_summary_paths(src, kind=kind, strict_cfg=strict_cfg)

    chosen: Dict[Tuple[str, str], Path] = {}
    applied = 0
    for pair in set(raw_best.keys()) | set(sum_best.keys()):
        srcp = raw_best.get(pair) or sum_best.get(pair)
        if not srcp:
            continue
        dstp = dst.merged_ir_dir / expected_canon_name(pair)
        copy_or_link_file(srcp, dstp, use_hardlinks=USE_HARDLINKS)
        chosen[pair] = srcp
        applied += 1

    print(f"[OFFLINE MERGE -> repaired] {dst.stage_dir.name}: kind={kind} base={len(base_idx)} overlays={applied}")
    return MergeResult(idx=canonical_index(dst.merged_ir_dir), chosen_overlay=chosen)


def compute_stage0_best_index(
    *,
    input_idx: Dict[Tuple[str, str], Path],
    stage0_merged_ir_dir_src: Path,
    strict_cfg: GuardrailConfig,
) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for pair, inp in input_idx.items():
        cand = stage0_merged_ir_dir_src / expected_canon_name(pair)
        if cand.exists() and strict_ok_file(cand, strict_cfg=strict_cfg):
            out[pair] = cand
        else:
            out[pair] = inp
    return out


def strict_fail_pairs_in_dir(ir_dir: Path, *, strict_cfg: GuardrailConfig) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for p in iter_canonical_ir_files(ir_dir):
        m = CANON_RE.match(p.name)
        if not m:
            continue
        pair = (m.group(1), m.group(2))
        if not strict_ok_file(p, strict_cfg=strict_cfg):
            out.add(pair)
    return out


# -------------------------
# Multi-stage modification detection + plan logic + preview
# -------------------------
@dataclass(frozen=True)
class FullRerunPlan:
    full_rerun_trials: Set[str]   # >=2 stages modified
    minimal_fix_trials: Set[str]  # strict failures only; not full

    def total_trials(self) -> int:
        return len(self.full_rerun_trials | self.minimal_fix_trials)


def compute_modified_stages_per_trial(
    *,
    best0_idx: Dict[Tuple[str, str], Path],
    st1_chosen: Dict[Tuple[str, str], Path],
    st2_chosen: Dict[Tuple[str, str], Path],
    st3_chosen: Dict[Tuple[str, str], Path],
    dst1: StageDst,
    dst2: StageDst,
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    trial2stages: Dict[str, Set[str]] = defaultdict(set)
    pair2stages: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    # S1: polarity (exclusion only): compare output vs best0 base
    for (eff, side), outp in st1_chosen.items():
        if side != "exclusion":
            continue
        basep = best0_idx.get((eff, "exclusion"))
        if basep and did_modify(basep, outp):
            trial2stages[eff].add("S1")
            pair2stages[(eff, "exclusion")].add("S1")

    # S2: repair: compare output vs base after S1 merged (dst1 merged canonical)
    for (eff, side), outp in st2_chosen.items():
        basep = dst1.merged_ir_dir / expected_canon_name((eff, side))
        if did_modify(basep, outp):
            trial2stages[eff].add("S2")
            pair2stages[(eff, side)].add("S2")

    # S3: logic: compare output vs base after S2 merged (dst2 merged canonical)
    for (eff, side), outp in st3_chosen.items():
        basep = dst2.merged_ir_dir / expected_canon_name((eff, side))
        if did_modify(basep, outp):
            trial2stages[eff].add("S3")
            pair2stages[(eff, side)].add("S3")

    return dict(trial2stages), dict(pair2stages)


def compute_full_rerun_policy_plan(
    *,
    trial2stages: Dict[str, Set[str]],
    final_fail_pairs: Set[Tuple[str, str]],
) -> FullRerunPlan:
    full: Set[str] = set()
    minimal: Set[str] = set()

    for trial, stages in trial2stages.items():
        if len(stages) >= 2:
            full.add(trial)

    for (eff, _side) in final_fail_pairs:
        if eff not in full:
            minimal.add(eff)

    return FullRerunPlan(full_rerun_trials=full, minimal_fix_trials=minimal)


def _combo_key(stages: Set[str]) -> str:
    if not stages:
        return "NONE"
    return "+".join(sorted(stages))


def print_rerun_preview(
    *,
    trial2stages: Dict[str, Set[str]],
    pair2stages: Dict[Tuple[str, str], Set[str]],
    full_trials: Set[str],
    minimal_trials: Set[str],
    sample_n: int,
) -> None:
    print("\n[PREVIEW] Trials that would be rerun (and why)")

    if full_trials:
        combos_full = Counter(_combo_key(trial2stages.get(t, set())) for t in full_trials)
        print(f"\n  FULL RERUN trials = {len(full_trials)} (>=2 stages modified)")
        print("  Full-rerun stage-combo breakdown:")
        for k, c in combos_full.most_common():
            print(f"    {k}: {c}")

        print(f"  Example FULL RERUN trials (up to {sample_n}):")
        for t in sorted(full_trials)[:sample_n]:
            stages = sorted(trial2stages.get(t, set()))
            # side detail
            incl = sorted(pair2stages.get((t, "inclusion"), set()))
            excl = sorted(pair2stages.get((t, "exclusion"), set()))
            print(f"    {t}: trial_stages={stages}  inclusion={incl}  exclusion={excl}")
    else:
        print("\n  FULL RERUN trials = 0")

    if minimal_trials:
        combos_min = Counter(_combo_key(trial2stages.get(t, set())) for t in minimal_trials)
        print(f"\n  MINIMAL FIX trials = {len(minimal_trials)} (strict failures only; not full rerun)")
        print("  Minimal-fix stage-combo breakdown (context):")
        for k, c in combos_min.most_common():
            print(f"    {k}: {c}")

        print(f"  Example MINIMAL FIX trials (up to {sample_n}):")
        for t in sorted(minimal_trials)[:sample_n]:
            stages = sorted(trial2stages.get(t, set()))
            incl = sorted(pair2stages.get((t, "inclusion"), set()))
            excl = sorted(pair2stages.get((t, "exclusion"), set()))
            print(f"    {t}: trial_stages={stages}  inclusion={incl}  exclusion={excl}")
    else:
        print("\n  MINIMAL FIX trials = 0")


# -------------------------
# Execution (writes into repaired folder)
# -------------------------
def write_allowlist(path: Path, trials: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(set(trials)):
            f.write(t + "\n")


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


def run_cmd(cmd: List[str], *, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def confirm_or_exit(prompt: str, *, yes: bool) -> None:
    if yes:
        print("[CONFIRM] --yes set; proceeding.", flush=True)
        return
    if not sys.stdin.isatty():
        raise SystemExit("[ABORT] Non-interactive stdin and --yes not set.")
    ans = input(prompt).strip().lower()
    if ans not in ("y", "yes"):
        raise SystemExit("[ABORT] User declined.")


def execute_full_rerun_trials(
    *,
    trials: Set[str],
    scripts_dir: Path,
    snapshot_dir: Path,
    dst1: StageDst,
    dst2: StageDst,
    dst3: StageDst,
    max_workers: int,
) -> None:
    if not trials:
        return

    polarity_py = scripts_dir / "polarity_fix_irs.py"
    repair_py   = scripts_dir / "repair_all_ir.py"
    logic_py    = scripts_dir / "logic_fix_irs.py"
    for sp in (polarity_py, repair_py, logic_py):
        if not sp.exists():
            raise FileNotFoundError(f"Missing stage script: {sp}")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Stage1 allowlist
    allow1 = dst1.stage_dir / f"allowlist_fullrerun_stage1_{ts}.txt"
    write_allowlist(allow1, trials)

    # Stage1 TODO
    todo1 = dst1.todo_dir / f"fullrerun_stage1_{ts}"
    ensure_dir(todo1)
    for p in todo1.glob("NCT*_program.smt2"):
        p.unlink()

    copied = 0
    for eff in sorted(trials):
        src = dst1.merged_ir_dir / f"{eff}_exclusion_program.smt2"
        if not src.exists():
            continue
        shutil.copy2(src, todo1 / src.name)
        copied += 1
    print(f"[FULL-RERUN] Stage1 TODO={todo1} copied={copied}")

    run_sum1 = dst1.stage_dir / f"summary_run_fullrerun_stage1_{ts}.jsonl"
    run_cmd(
        [
            sys.executable, str(polarity_py),
            "--ir-dir", str(todo1),
            "--snapshot-dir", str(snapshot_dir),
            "--out-ir-dir", str(dst1.raw_out_dir),
            "--summary-jsonl", str(run_sum1),
            "--log-dir", str(dst1.logs_dir),
            "--mbench-dir", str(dst1.mbench_dir),
            "--max-workers", str(max_workers),
            "--trial-allowlist", str(allow1),
        ],
        cwd=scripts_dir,
    )
    append_jsonl(dst1.summary_master_jsonl, run_sum1)

    # Stage2 TODO (both sides)
    todo2 = dst2.todo_dir / f"fullrerun_stage2_{ts}"
    ensure_dir(todo2)
    for p in todo2.glob("NCT*_program.smt2"):
        p.unlink()

    copied2 = 0
    for eff in sorted(trials):
        for side in ("inclusion", "exclusion"):
            src = dst2.merged_ir_dir / f"{eff}_{side}_program.smt2"
            if not src.exists():
                src = dst1.merged_ir_dir / f"{eff}_{side}_program.smt2"
            if not src.exists():
                continue
            shutil.copy2(src, todo2 / src.name)
            copied2 += 1
    print(f"[FULL-RERUN] Stage2 TODO={todo2} copied={copied2}")

    run_sum2 = dst2.stage_dir / f"summary_run_fullrerun_stage2_{ts}.jsonl"
    run_cmd(
        [
            sys.executable, str(repair_py),
            "--ir-dir", str(todo2),
            "--snapshot-dir", str(snapshot_dir),
            "--repaired-ir-dir", str(dst2.raw_out_dir),
            "--summary-jsonl", str(run_sum2),
            "--log-dir", str(dst2.logs_dir),
            "--mbench-dir", str(dst2.mbench_dir),
            "--max-workers", str(max_workers),
        ],
        cwd=scripts_dir,
    )
    append_jsonl(dst2.summary_master_jsonl, run_sum2)

    # Stage3 allowlist (both sides)
    allow3 = dst3.stage_dir / f"allowlist_fullrerun_stage3_{ts}.txt"
    write_allowlist(allow3, trials)

    run_sum3 = dst3.stage_dir / f"summary_run_fullrerun_stage3_{ts}.jsonl"
    run_cmd(
        [
            sys.executable, str(logic_py),
            "--ir-dir", str(dst2.merged_ir_dir),
            "--snapshot-dir", str(snapshot_dir),
            "--out-ir-dir", str(dst3.raw_out_dir),
            "--summary-jsonl", str(run_sum3),
            "--log-dir", str(dst3.logs_dir),
            "--mbench-dir", str(dst3.mbench_dir),
            "--max-workers", str(max_workers),
            "--trial-allowlist", str(allow3),
            "--side", "both",
        ],
        cwd=scripts_dir,
    )
    append_jsonl(dst3.summary_master_jsonl, run_sum3)


def execute_minimal_fix_trials_stage3_only(
    *,
    trials: Set[str],
    scripts_dir: Path,
    snapshot_dir: Path,
    dst2: StageDst,
    dst3: StageDst,
    max_workers: int,
) -> None:
    if not trials:
        return

    logic_py = scripts_dir / "logic_fix_irs.py"
    if not logic_py.exists():
        raise FileNotFoundError(f"Missing stage script: {logic_py}")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    allow3 = dst3.stage_dir / f"allowlist_minfix_stage3_{ts}.txt"
    write_allowlist(allow3, trials)

    run_sum3 = dst3.stage_dir / f"summary_run_minfix_stage3_{ts}.jsonl"
    run_cmd(
        [
            sys.executable, str(logic_py),
            "--ir-dir", str(dst2.merged_ir_dir),
            "--snapshot-dir", str(snapshot_dir),
            "--out-ir-dir", str(dst3.raw_out_dir),
            "--summary-jsonl", str(run_sum3),
            "--log-dir", str(dst3.logs_dir),
            "--mbench-dir", str(dst3.mbench_dir),
            "--max-workers", str(max_workers),
            "--trial-allowlist", str(allow3),
            "--side", "both",
        ],
        cwd=scripts_dir,
    )
    append_jsonl(dst3.summary_master_jsonl, run_sum3)


# -------------------------
# CLI / main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover stage outputs into a new *_repaired work dir (non-destructive).")
    p.add_argument("--work-dir", required=True, help="Original ir_orchestrated_* directory (read-only).")
    p.add_argument("--out-work-dir", default=None, help="Output repaired work dir. Default: <work-dir>_repaired")
    p.add_argument("--ir-dir", required=True, help="Canonical input IR directory (NCT*_program.smt2).")
    p.add_argument("--snapshot-dir", required=True, help="Snapshot directory for stage scripts (if --execute).")
    p.add_argument("--scripts-dir", required=True, help="Directory containing polarity_fix_irs.py, repair_all_ir.py, logic_fix_irs.py.")
    p.add_argument("--skip-assert-checks", action="store_true", help="Disable assert/:named strict checks (must match your intended regime).")
    p.add_argument("--no-copy-src-raw-out", action="store_true", help="Do not copy original raw_out strict-ok files into repaired raw_out.")
    p.add_argument("--no-restore-from-quarantine", action="store_true", help="Do not restore strict-ok files from original quarantine_invalid.")
    p.add_argument("--no-restore-from-mbench", action="store_true", help="Do not restore strict-ok *.smt2 from original mbench.")
    p.add_argument("--execute", action="store_true", help="Actually run reruns (spends money) writing into repaired folder.")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--sample-n", type=int, default=20, help="How many example trials to print per bucket before confirmation.")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    strict_cfg = STRICT_GUARDRAILS_DEFAULT
    if args.skip_assert_checks:
        strict_cfg = replace(strict_cfg, check_asserts_and_named=False)

    work_dir = Path(args.work_dir).resolve()
    out_work_dir = Path(args.out_work_dir).resolve() if args.out_work_dir else Path(str(work_dir) + "_repaired").resolve()

    in_ir_dir = Path(args.ir_dir).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    scripts_dir = Path(args.scripts_dir).resolve()

    if not work_dir.exists():
        raise FileNotFoundError(f"--work-dir not found: {work_dir}")
    if not in_ir_dir.exists():
        raise FileNotFoundError(f"--ir-dir not found: {in_ir_dir}")
    if not scripts_dir.exists():
        raise FileNotFoundError(f"--scripts-dir not found: {scripts_dir}")

    ensure_dir(out_work_dir)

    # Source stage dirs
    src0 = make_stage_src(work_dir, "stage0_meaning")
    src1 = make_stage_src(work_dir, "stage1_polarity")
    src2 = make_stage_src(work_dir, "stage2_repair")
    src3 = make_stage_src(work_dir, "stage3_logic")

    # Destination stage dirs (repaired)
    dst0 = make_stage_dst(out_work_dir, "stage0_meaning")  # structure only
    dst1 = make_stage_dst(out_work_dir, "stage1_polarity")
    dst2 = make_stage_dst(out_work_dir, "stage2_repair")
    dst3 = make_stage_dst(out_work_dir, "stage3_logic")

    # Copy original summaries into repaired folder for traceability
    for (s, d) in [
        (src1.summary_master_jsonl, dst1.summary_master_jsonl),
        (src2.summary_master_jsonl, dst2.summary_master_jsonl),
        (src3.summary_master_jsonl, dst3.summary_master_jsonl),
    ]:
        if s.exists() and not d.exists():
            shutil.copy2(s, d)

    print(f"[INFO] original work_dir={work_dir}")
    print(f"[INFO] repaired  out_work_dir={out_work_dir}")
    print(f"[INFO] strict assert checks={'OFF' if args.skip_assert_checks else 'ON'}")

    # Step A: seed repaired raw_out from original raw_out (strict-ok only)
    if not args.no_copy_src_raw_out:
        c1 = copy_strict_ok_from_src_raw_out(src1, dst1, strict_cfg=strict_cfg)
        c2 = copy_strict_ok_from_src_raw_out(src2, dst2, strict_cfg=strict_cfg)
        c3 = copy_strict_ok_from_src_raw_out(src3, dst3, strict_cfg=strict_cfg)
        print(f"[SEED] copied strict-ok from original raw_out -> repaired raw_out: stage1={c1} stage2={c2} stage3={c3}")

    # Step B: restore from original quarantine_invalid
    if not args.no_restore_from_quarantine:
        r1 = restore_strict_ok_from_src_quarantine(src1, dst1, strict_cfg=strict_cfg)
        r2 = restore_strict_ok_from_src_quarantine(src2, dst2, strict_cfg=strict_cfg)
        r3 = restore_strict_ok_from_src_quarantine(src3, dst3, strict_cfg=strict_cfg)
        print(f"[RESTORE] from original quarantine -> repaired raw_out: stage1={r1} stage2={r2} stage3={r3}")

    # Step C: restore from original mbench
    if not args.no_restore_from_mbench:
        m1 = restore_strict_ok_from_src_mbench(src1, dst1, strict_cfg=strict_cfg)
        m2 = restore_strict_ok_from_src_mbench(src2, dst2, strict_cfg=strict_cfg)
        m3 = restore_strict_ok_from_src_mbench(src3, dst3, strict_cfg=strict_cfg)
        print(f"[RESTORE] from original mbench -> repaired raw_out: stage1={m1} stage2={m2} stage3={m3}")

    # Step D: offline re-merge into repaired merged_ir (record overlays)
    input_idx = canonical_index(in_ir_dir)
    if not input_idx:
        raise FileNotFoundError(f"No canonical IR files under: {in_ir_dir}")

    best0_idx = compute_stage0_best_index(input_idx=input_idx, stage0_merged_ir_dir_src=src0.merged_ir_dir, strict_cfg=strict_cfg)

    st1_merge = merge_stage_offline_record(base_idx=best0_idx, src=src1, dst=dst1, kind="polarity", strict_cfg=strict_cfg)
    st2_merge = merge_stage_offline_record(base_idx=st1_merge.idx, src=src2, dst=dst2, kind="repair", strict_cfg=strict_cfg)
    st3_merge = merge_stage_offline_record(base_idx=st2_merge.idx, src=src3, dst=dst3, kind="logic", strict_cfg=strict_cfg)

    final_fail_pairs = strict_fail_pairs_in_dir(dst3.merged_ir_dir, strict_cfg=strict_cfg)
    print(f"[FINAL repaired] strict failures in repaired stage3 merged: {len(final_fail_pairs)}")

    # Step E: compute actual modifications (content-based)
    trial2stages, pair2stages = compute_modified_stages_per_trial(
        best0_idx=best0_idx,
        st1_chosen=st1_merge.chosen_overlay,
        st2_chosen=st2_merge.chosen_overlay,
        st3_chosen=st3_merge.chosen_overlay,
        dst1=dst1,
        dst2=dst2,
    )

    num_trials_any = sum(1 for ss in trial2stages.values() if ss)
    num_trials_2p = sum(1 for ss in trial2stages.values() if len(ss) >= 2)
    print(f"[MODIFIED] trials_with_any_stage_modification={num_trials_any}")
    print(f"[MODIFIED] trials_with_>=2_stage_modification={num_trials_2p}")

    # Step F: apply policy
    plan = compute_full_rerun_policy_plan(trial2stages=trial2stages, final_fail_pairs=final_fail_pairs)

    print("\n[RERUN POLICY PLAN]")
    print(f"  full_rerun_trials (>=2 stages modified): {len(plan.full_rerun_trials)}")
    print(f"  minimal_fix_trials (strict failures only, not full): {len(plan.minimal_fix_trials)}")
    print(f"  total_trials_to_touch: {plan.total_trials()}")

    # Write plan files under repaired folder
    plan_dir = out_work_dir / "_rerun_policy_plan"
    ensure_dir(plan_dir)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    (plan_dir / f"trial2stages_{ts}.json").write_text(
        json.dumps({k: sorted(list(v)) for k, v in sorted(trial2stages.items())}, indent=2),
        encoding="utf-8",
    )
    (plan_dir / f"pair2stages_{ts}.json").write_text(
        json.dumps({f"{k[0]}::{k[1]}": sorted(list(v)) for k, v in sorted(pair2stages.items())}, indent=2),
        encoding="utf-8",
    )
    (plan_dir / f"full_rerun_trials_{ts}.txt").write_text(
        "\n".join(sorted(plan.full_rerun_trials)) + ("\n" if plan.full_rerun_trials else ""),
        encoding="utf-8",
    )
    (plan_dir / f"minimal_fix_trials_{ts}.txt").write_text(
        "\n".join(sorted(plan.minimal_fix_trials)) + ("\n" if plan.minimal_fix_trials else ""),
        encoding="utf-8",
    )

    allow_full = plan_dir / f"allowlist_full_rerun_trials_{ts}.txt"
    allow_min  = plan_dir / f"allowlist_minimal_fix_trials_{ts}.txt"
    write_allowlist(allow_full, plan.full_rerun_trials)
    write_allowlist(allow_min, plan.minimal_fix_trials)

    print("\n[WROTE plan files to repaired folder]")
    print(f"  {plan_dir}/trial2stages_{ts}.json")
    print(f"  {plan_dir}/pair2stages_{ts}.json")
    print(f"  {allow_full} (n={len(plan.full_rerun_trials)})")
    print(f"  {allow_min}  (n={len(plan.minimal_fix_trials)})")

    # Preview
    print_rerun_preview(
        trial2stages=trial2stages,
        pair2stages=pair2stages,
        full_trials=plan.full_rerun_trials,
        minimal_trials=plan.minimal_fix_trials,
        sample_n=args.sample_n,
    )

    if not args.execute:
        print("\n[DONE] No reruns executed. Repaired merged IR is here:")
        print(f"  {dst3.merged_ir_dir}")
        return

    # Single confirmation gate before any paid calls
    confirm_or_exit(
        f"\nProceed to EXECUTE reruns?\n"
        f"  FULL RERUN trials:   {len(plan.full_rerun_trials)}\n"
        f"  MINIMAL FIX trials:  {len(plan.minimal_fix_trials)}\n"
        f"  Total trials touched:{plan.total_trials()}\n"
        f"[y/N] ",
        yes=args.yes,
    )

    # Step G: execute reruns
    execute_full_rerun_trials(
        trials=plan.full_rerun_trials,
        scripts_dir=scripts_dir,
        snapshot_dir=snapshot_dir,
        dst1=dst1, dst2=dst2, dst3=dst3,
        max_workers=args.max_workers,
    )

    # Offline re-merge after full rerun
    print("\n[POST FULL-RERUN] Offline re-merge repaired merged_ir again...")
    st1_merge = merge_stage_offline_record(base_idx=best0_idx, src=src1, dst=dst1, kind="polarity", strict_cfg=strict_cfg)
    st2_merge = merge_stage_offline_record(base_idx=st1_merge.idx, src=src2, dst=dst2, kind="repair", strict_cfg=strict_cfg)
    st3_merge = merge_stage_offline_record(base_idx=st2_merge.idx, src=src3, dst=dst3, kind="logic", strict_cfg=strict_cfg)

    # Minimal fix trials (stage3-only) not in full rerun
    remaining_min = set(plan.minimal_fix_trials) - set(plan.full_rerun_trials)
    execute_minimal_fix_trials_stage3_only(
        trials=remaining_min,
        scripts_dir=scripts_dir,
        snapshot_dir=snapshot_dir,
        dst2=dst2,
        dst3=dst3,
        max_workers=args.max_workers,
    )

    # Final offline re-merge
    print("\n[POST-EXEC] Offline re-merge repaired merged_ir again (to incorporate new raw_out)...")
    st1_merge = merge_stage_offline_record(base_idx=best0_idx, src=src1, dst=dst1, kind="polarity", strict_cfg=strict_cfg)
    st2_merge = merge_stage_offline_record(base_idx=st1_merge.idx, src=src2, dst=dst2, kind="repair", strict_cfg=strict_cfg)
    st3_merge = merge_stage_offline_record(base_idx=st2_merge.idx, src=src3, dst=dst3, kind="logic", strict_cfg=strict_cfg)

    final_fail2 = strict_fail_pairs_in_dir(dst3.merged_ir_dir, strict_cfg=strict_cfg)
    print(f"[POST-EXEC FINAL] strict failures in repaired stage3 merged: {len(final_fail2)}")
    print(f"[POST-EXEC FINAL] repaired merged IR directory: {dst3.merged_ir_dir}")


if __name__ == "__main__":
    main()
