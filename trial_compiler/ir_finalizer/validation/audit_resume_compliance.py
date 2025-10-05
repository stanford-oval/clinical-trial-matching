#!/usr/bin/env python3
"""
audit_resume_compliance.py (OFFLINE) — summary-executed + strict failures => rerun

Definitions (per stage, on eligible set):
- eligible: pairs that should be processed (from base IR dirs)
- executed: pairs with meta.executed == true in summary.jsonl
- new: eligible - executed
- done_ok: executed pairs that are strict-ok (based on best on-disk path)
- rerun_strict: executed pairs that are strict-failing (based on best on-disk path)
- missing_artifact: executed pairs with no on-disk file found to validate

Rerun policy:
- strict failures MUST be rerun => rerun = new ∪ rerun_strict (plus optionally missing_artifact)

Validation path priority:
1) stage merged_ir canonical path: <merged_ir>/<eff>_<side>_program.smt2 (if exists)
2) summary stage-produced path (fixed/repaired) if exists
3) summary smt_path if exists

This avoids treating "missing raw_out" as failure while still rerunning genuine strict failures.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


# -------------------------
# Canonical IR filename pattern (inputs + merged_ir outputs)
# -------------------------
CANON_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program\.smt2$")

# -------------------------
# Stage raw_out filename patterns (emitted artifacts only)
# -------------------------
POLARITY_OUT_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_exclusion_program_polarity_fixed\.smt2$")
REPAIR_OUT_RE   = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_repaired\.smt2$")
LOGIC_OUT_RE    = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program_logic_fixed\.smt2$")


# -------------------------
# Strict checks (no-skip)
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

    skip_auto_synth_missing_json: bool
    skip_nearby_json_comment_block: bool
    decl_json_lookback: int

    skip_group_block_missing_json: bool
    group_block_lookback: int

    skip_direct_json_comment_above: bool
    direct_json_lookback: int


STRICT_GUARDRAILS = GuardrailConfig(
    allow_extra_decl_json_keys=False,
    strict_double_semicolon=False,
    decl_json_strict=True,
    allow_any_sort=False,
    skip_auto_synth_missing_json=False,
    skip_nearby_json_comment_block=False,
    decl_json_lookback=8,
    skip_group_block_missing_json=False,
    group_block_lookback=20,
    skip_direct_json_comment_above=False,
    direct_json_lookback=3,
)


# -------------------------
# Basic file helpers
# -------------------------
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


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_jsonl(path: Path) -> List[Dict]:
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
                rows.append({"__jsonl_parse_failed__": True, "__raw__": line})
    return rows


def iter_canonical_ir_files(ir_dir: Path) -> Iterable[Path]:
    if not ir_dir.exists():
        return
    for p in ir_dir.glob("NCT*_program.smt2"):
        if CANON_RE.match(p.name):
            yield p


def has_any_canonical_files(ir_dir: Path) -> bool:
    for _ in iter_canonical_ir_files(ir_dir):
        return True
    return False


def eligible_pairs_from_ir_dir(ir_dir: Path) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for p in iter_canonical_ir_files(ir_dir):
        m = CANON_RE.match(p.name)
        assert m
        eff, side = m.group(1), m.group(2)
        if is_nonempty_smt(p):
            out.add((eff, side))
    return out


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

    if not cfg.allow_extra_decl_json_keys:
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


def validate_smt_content(text: str, *, cfg: GuardrailConfig) -> Tuple[bool, List[str], Dict]:
    errors: List[str] = []
    extra: Dict = {}

    if not text.strip():
        errors.append("empty_or_whitespace")
        return False, errors, extra

    pb = paren_balance_diagnostics(text, strict_double_semicolon=cfg.strict_double_semicolon)
    extra["paren_balance"] = pb
    if not pb["ok"]:
        errors.append("paren_balance_failed")

    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            errors.append(f"forbidden_pattern:{pat.pattern}")

    assert_count = len(list(ASSERT_BLOCK_RE.finditer(text)))
    named_matches = list(NAMED_TAG_RE.finditer(text))
    named_count = len(named_matches)
    extra["assert_count"] = assert_count
    extra["named_count"] = named_count

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
    if bad_named:
        errors.append(f"bad_named_tag:{bad_named[0]}")
        extra["bad_named_tags"] = bad_named[:50]

    allowed_sorts: Set[str] = {"Bool", "Int", "Real", "String"}
    decl_issues_seen = 0
    decl_issue_examples: List[Tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not declare_const_in_code(line):
            continue
        issue = validate_declare_const_json_comment(line, cfg=cfg, allowed_sorts=allowed_sorts)
        if issue:
            errors.append(issue)
            decl_issues_seen += 1
            if len(decl_issue_examples) < 5:
                decl_issue_examples.append((lineno, issue, line.rstrip("\n")))
            if decl_issues_seen >= 50:
                break
    if decl_issue_examples:
        extra["declare_const_issue_examples"] = decl_issue_examples

    return (len(errors) == 0), errors, extra


def validate_smt_file(p: Path, *, cfg: GuardrailConfig) -> Tuple[bool, List[str], Dict]:
    try:
        txt = read_text(p)
    except Exception as e:
        return False, [f"read_error:{e}"], {}
    return validate_smt_content(txt, cfg=cfg)


# -------------------------
# Summary helpers
# -------------------------
def _meta_executed(r: Dict) -> bool:
    meta = r.get("meta")
    return isinstance(meta, dict) and meta.get("executed") is True


def executed_pairs_from_summary(summary_path: Path, kind: str) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if not summary_path.exists():
        return out
    for r in read_jsonl(summary_path):
        eff = r.get("effective_trial_id")
        side = r.get("side")
        if not eff or side not in ("inclusion", "exclusion"):
            continue
        if kind == "polarity" and side != "exclusion":
            continue
        if _meta_executed(r):
            out.add((str(eff), str(side)))
    return out


def build_executed_row_index(summary_path: Path, kind: str) -> Dict[Tuple[str, str], Dict]:
    """
    Last executed row wins.
    """
    idx: Dict[Tuple[str, str], Dict] = {}
    if not summary_path.exists():
        return idx
    for r in read_jsonl(summary_path):
        eff = r.get("effective_trial_id")
        side = r.get("side")
        if not eff or side not in ("inclusion", "exclusion"):
            continue
        if kind == "polarity" and side != "exclusion":
            continue
        if not _meta_executed(r):
            continue
        idx[(str(eff), str(side))] = r
    return idx


def best_validation_path(spec, kind: str, pair: Tuple[str, str], row: Optional[Dict]) -> Optional[Path]:
    """
    Find the best on-disk file to validate strict compliance for this pair.
    Priority:
      1) canonical merged_ir file for this stage
      2) stage-produced path in summary (fixed/repaired) if exists
      3) smt_path in summary if exists
    """
    eff, side = pair

    # 1) merged_ir canonical
    canon = spec.merged_ir_dir / f"{eff}_{side}_program.smt2"
    if canon.exists() and canon.is_file():
        return canon

    if row is None:
        return None

    # 2) stage-produced paths
    cand_list: List[Optional[str]] = []
    if kind == "polarity":
        cand_list = [row.get("fixed_smt_path"), row.get("polarity_fixed_smt_path")]
    elif kind == "repair":
        cand_list = [row.get("repaired_smt_path")]
    elif kind == "logic":
        cand_list = [row.get("fixed_smt_path"), row.get("logic_fixed_smt_path")]

    for s in cand_list:
        if not s:
            continue
        p = Path(s)
        if p.exists() and p.is_file():
            return p

    # 3) fallback smt_path
    s2 = row.get("smt_path")
    if s2:
        p2 = Path(s2)
        if p2.exists() and p2.is_file():
            return p2

    return None


# -------------------------
# Stage abstraction
# -------------------------
@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str
    raw_out_dir: Path
    merged_ir_dir: Path
    summary_path: Path


def stage_specs(work_dir: Path) -> List[StageSpec]:
    return [
        StageSpec("stage1_polarity", "polarity",
                  work_dir / "stage1_polarity" / "raw_out",
                  work_dir / "stage1_polarity" / "merged_ir",
                  work_dir / "stage1_polarity" / "summary.jsonl"),
        StageSpec("stage2_repair", "repair",
                  work_dir / "stage2_repair" / "raw_out",
                  work_dir / "stage2_repair" / "merged_ir",
                  work_dir / "stage2_repair" / "summary.jsonl"),
        StageSpec("stage3_logic", "logic",
                  work_dir / "stage3_logic" / "raw_out",
                  work_dir / "stage3_logic" / "merged_ir",
                  work_dir / "stage3_logic" / "summary.jsonl"),
    ]


def _pair_from_output_filename(kind: str, fname: str) -> Optional[Tuple[str, str]]:
    if kind == "polarity":
        m = POLARITY_OUT_RE.match(fname)
        if m:
            return (m.group(1), "exclusion")
    if kind == "repair":
        m = REPAIR_OUT_RE.match(fname)
        if m:
            return (m.group(1), m.group(2))
    if kind == "logic":
        m = LOGIC_OUT_RE.match(fname)
        if m:
            return (m.group(1), m.group(2))
    return None


def strict_failures_in_canon_dir(
    canon_dir: Path, *, cfg: GuardrailConfig, cap: int, topk_samples: int,
) -> Tuple[Counter[str], DefaultDict[str, List[str]], int, int, int]:
    counts: Counter[str] = Counter()
    samples: DefaultDict[str, List[str]] = defaultdict(list)
    scanned = 0
    okc = 0
    failc = 0

    if not canon_dir.exists():
        return counts, samples, 0, 0, 0

    for p in iter_canonical_ir_files(canon_dir):
        scanned += 1
        if scanned > cap:
            counts["__truncated__"] += 1
            break
        if not is_nonempty_smt(p):
            counts["empty_output"] += 1
            failc += 1
            if len(samples["empty_output"]) < topk_samples:
                samples["empty_output"].append(str(p))
            continue
        ok, errs, _extra = validate_smt_file(p, cfg=cfg)
        if ok:
            okc += 1
            continue
        failc += 1
        primary = str(errs[0]) if errs else "unknown_error"
        counts[primary] += 1
        if len(samples[primary]) < topk_samples:
            samples[primary].append(str(p))
    return counts, samples, scanned, okc, failc


def print_stage_header(spec: StageSpec) -> None:
    print("\n" + "=" * 100)
    print(f"[{spec.name}] kind={spec.kind}")
    print(f"  raw_out_dir:  {spec.raw_out_dir}")
    print(f"  merged_ir:    {spec.merged_ir_dir}")
    print(f"  summary:      {spec.summary_path}")
    print("=" * 100, flush=True)


def audit_stage(
    spec: StageSpec,
    *,
    cfg: GuardrailConfig,
    eligible_pairs: Set[Tuple[str, str]],
    missing_artifact_mode: str,
    topk: int,
    max_samples_per_reason: int,
    merged_scan_cap: int,
) -> None:
    print_stage_header(spec)

    # RAW_OUT inventory
    all_raw = list(spec.raw_out_dir.glob("*.smt2")) if spec.raw_out_dir.exists() else []
    raw_match = sum(1 for p in all_raw if _pair_from_output_filename(spec.kind, p.name))
    raw_nonempty = sum(1 for p in all_raw if is_nonempty_smt(p))
    raw_strict_ok = 0
    for p in all_raw:
        if not is_nonempty_smt(p):
            continue
        ok, _errs, _extra = validate_smt_file(p, cfg=cfg)
        if ok:
            raw_strict_ok += 1

    print("[RAW_OUT INVENTORY] (stage-emitted only)")
    print(f"  files_total={len(all_raw)}")
    print(f"  files_matching_expected_pattern={raw_match}")
    print(f"  nonempty={raw_nonempty}")
    print(f"  strict_ok_files={raw_strict_ok}")
    print("", flush=True)

    # MERGED_IR inventory
    merged_all = list(iter_canonical_ir_files(spec.merged_ir_dir)) if spec.merged_ir_dir.exists() else []
    merged_nonempty = sum(1 for p in merged_all if is_nonempty_smt(p))
    merged_strict_ok = 0
    for p in merged_all:
        if not is_nonempty_smt(p):
            continue
        ok, _errs, _extra = validate_smt_file(p, cfg=cfg)
        if ok:
            merged_strict_ok += 1

    print("[MERGED_IR INVENTORY] (canonical programs)")
    print(f"  canonical_files_total={len(merged_all)}")
    print(f"  nonempty={merged_nonempty}")
    print(f"  strict_ok_files={merged_strict_ok}")
    print("", flush=True)

    # Summary executed index
    exec_idx = build_executed_row_index(spec.summary_path, spec.kind)
    executed = set(exec_idx.keys()) & eligible_pairs

    new = eligible_pairs - executed

    done_ok: Set[Tuple[str, str]] = set()
    rerun_strict: Set[Tuple[str, str]] = set()
    missing_artifact: Set[Tuple[str, str]] = set()

    # For histogram: primary reasons for strict failures among executed
    fail_reasons: Counter[str] = Counter()
    fail_samples: DefaultDict[str, List[str]] = defaultdict(list)

    for pair in sorted(executed):
        row = exec_idx.get(pair)
        p = best_validation_path(spec, spec.kind, pair, row)
        if p is None:
            missing_artifact.add(pair)
            continue

        if not is_nonempty_smt(p):
            rerun_strict.add(pair)
            reason = "empty_output"
            fail_reasons[reason] += 1
            if len(fail_samples[reason]) < max_samples_per_reason:
                fail_samples[reason].append(str(p))
            continue

        ok, errs, _extra = validate_smt_file(p, cfg=cfg)
        if ok:
            done_ok.add(pair)
        else:
            rerun_strict.add(pair)
            primary = str(errs[0]) if errs else "unknown_error"
            fail_reasons[primary] += 1
            if len(fail_samples[primary]) < max_samples_per_reason:
                fail_samples[primary].append(str(p))

    rerun = set(new) | set(rerun_strict)
    if missing_artifact_mode == "rerun":
        rerun |= missing_artifact

    print("[PLAN] (strict failures rerun; summary meta.executed defines executed)")
    print(f"  eligible={len(eligible_pairs)}")
    print(f"  executed={len(executed)}")
    print(f"  new={len(new)}")
    print(f"  done_ok={len(done_ok)}")
    print(f"  rerun_strict={len(rerun_strict)}")
    print(f"  missing_artifact={len(missing_artifact)} (mode={missing_artifact_mode})")
    print(f"  RERUN_TOTAL={len(rerun)} (new + strict_fail{' + missing_artifact' if missing_artifact_mode=='rerun' else ''})")
    print("", flush=True)

    if missing_artifact and missing_artifact_mode in ("warn", "ignore"):
        # keep it brief; user can increase samples-per-reason if desired
        ex = list(sorted(missing_artifact))[:3]
        msg = "WARN" if missing_artifact_mode == "warn" else "INFO"
        print(f"[{msg}] {spec.name}: {len(missing_artifact)} executed pairs had no on-disk file to validate.")
        print(f"       sample pairs: {ex}")
        print("", flush=True)

    if rerun_strict:
        print("[STRICT FAILURE REASONS] (among executed; based on best on-disk validation path)")
        for reason, c in fail_reasons.most_common(topk):
            print(f"  {reason}: {c}")
            for sp in fail_samples.get(reason, [])[:max_samples_per_reason]:
                print(f"    sample: {sp}")
        print("", flush=True)

    # Snapshot health (merged_ir)
    if spec.merged_ir_dir.exists() and has_any_canonical_files(spec.merged_ir_dir):
        counts_m, samples_m, scanned_m, ok_m, fail_m = strict_failures_in_canon_dir(
            spec.merged_ir_dir, cfg=cfg, cap=merged_scan_cap, topk_samples=max_samples_per_reason
        )
        print("[MERGED_IR STRICT FAILURE HISTOGRAM] (snapshot-wide)")
        print(f"  scanned={scanned_m} ok={ok_m} fail={fail_m} (cap={merged_scan_cap})")
        for reason, c in counts_m.most_common(topk):
            print(f"  {reason}: {c}")
            for sp in samples_m.get(reason, [])[:max_samples_per_reason]:
                print(f"    sample: {sp}")
        print("", flush=True)


# -------------------------
# Base IR inference
# -------------------------
def infer_base_ir_dir(work_dir: Path, user_base: Optional[Path]) -> Path:
    if user_base is not None:
        return user_base.resolve()
    cand = work_dir / "stage0_meaning" / "merged_ir"
    if cand.exists() and has_any_canonical_files(cand):
        return cand.resolve()
    fallback = work_dir.parent / "ir"
    return fallback.resolve()


def infer_stage3_ir_dir(work_dir: Path) -> Optional[Path]:
    cand = work_dir / "stage2_repair" / "merged_ir"
    if cand.exists() and has_any_canonical_files(cand):
        return cand.resolve()
    return None


# -------------------------
# Args / main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline auditor: strict failures rerun; executed from summary meta.executed."
    )
    p.add_argument("--work-dir", required=True, help="Path to ir_orchestrated_* directory.")
    p.add_argument("--base-ir-dir", default=None, help="Base IR dir for Stage1/2 eligibility. If omitted, inferred.")
    p.add_argument("--missing-artifact", choices=["warn", "ignore", "rerun"], default="warn",
                   help="What to do if an executed pair has no on-disk file to validate.")
    p.add_argument("--topk", type=int, default=20, help="Top-K reasons to print.")
    p.add_argument("--samples-per-reason", type=int, default=3, help="Max sample paths per reason.")
    p.add_argument("--merged-scan-cap", type=int, default=20000, help="Cap for scanning merged_ir canonical files per stage.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"--work-dir does not exist: {work_dir}")

    user_base = Path(args.base_ir_dir).resolve() if args.base_ir_dir else None
    base_ir = infer_base_ir_dir(work_dir, user_base)
    stage3_ir = infer_stage3_ir_dir(work_dir)

    if not base_ir.exists() or not has_any_canonical_files(base_ir):
        print(f"[WARN] base_ir_dir for Stage1/2 has no canonical files: {base_ir}")
        print("       Provide --base-ir-dir explicitly if this is unexpected.\n")

    print("[AUDIT CONFIG]")
    print(f"  work_dir={work_dir}")
    print(f"  base_ir_dir(stage1/2 eligible)={base_ir}")
    print(f"  stage3_input_dir(stage3 eligible)={stage3_ir if stage3_ir else '(missing -> eligible=0)'}")
    print(f"  missing_artifact_mode={args.missing_artifact}")
    print(f"  strict_double_semicolon={STRICT_GUARDRAILS.strict_double_semicolon}")
    print(f"  allow_extra_decl_json_keys={STRICT_GUARDRAILS.allow_extra_decl_json_keys}")
    print(f"  allow_any_sort={STRICT_GUARDRAILS.allow_any_sort}")
    print(f"  NAMED_OK_RE={NAMED_OK_RE.pattern}")
    print("", flush=True)

    eligible_all = eligible_pairs_from_ir_dir(base_ir) if base_ir.exists() else set()
    eligible_excl = {(eff, side) for (eff, side) in eligible_all if side == "exclusion"}
    eligible_stage3 = eligible_pairs_from_ir_dir(stage3_ir) if stage3_ir else set()

    for spec in stage_specs(work_dir):
        if spec.kind == "polarity":
            eligible = eligible_excl
        elif spec.kind == "repair":
            eligible = eligible_all
        elif spec.kind == "logic":
            eligible = eligible_stage3
        else:
            eligible = set()

        audit_stage(
            spec,
            cfg=STRICT_GUARDRAILS,
            eligible_pairs=eligible,
            missing_artifact_mode=args.missing_artifact,
            topk=args.topk,
            max_samples_per_reason=args.samples_per_reason,
            merged_scan_cap=args.merged_scan_cap,
        )

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
