#!/usr/bin/env python3
"""
orchestrate_ir_fixes_simplified.py  (PARALLEL VERSION, THREADPOOL)

Order:
  step0_repair -> step1_polarity -> step2_logic -> step3_gate -> step4_underconstraint -> step5_meaning

Key semantics:
- canonical/ is the single source of truth (in-place updates)
- per-step bundles in step_runs/<step>/<run_id>/{before,after,inspect,parse_inspect,mbench,summary.json}
- strict validation decides whether repaired SMT overwrites canonical (except step3_gate which can delete files)
- real-time after/ mirroring per file + final "after snapshot" at end of each step

NEW (mbench detail):
- mbench now stores:
    mbench/<eff_tid>_<side>_cohort-<cid>/
      call_001_prompt.txt
      call_001_raw.txt
      call_002_prompt.txt ...
      events.jsonl
      final_meta.json
- prompt+raw is saved for EVERY LLM call attempt (even empty raw, even failures).
- final_meta.json is written even on early exits (read_error, missing_snapshot, executed_false, exceptions, etc.)

NEW (gate LLM-only):
- step3_gate is completely LLM-based (no static heuristics).
- step3_gate runs EVEN if SMT file is empty, because deletion decision is about NL criteria presence.

NEW (underconstraint stage):
- step4_underconstraint runs a dedicated prompt (./prompt/smt_fix_underconstraint.prompt) to tighten
  under-encoded / undercontextualized constraints (reduce false positives).

Meaning stage:
- step5_meaning runs AFTER underconstraint; it only runs on files that fail declare-const JSON checks
  (after deterministic normalization), same as the old step4 behavior.

PATCH (Jan 2026, corpus block for gate):
- Load corpus.jsonl into corpus_lookup keyed by BASE trial id
- Attach raw corpus jsonl item into subctx["corpus_item"] so gate can include Block 2
- Attach side-specific criteria into subctx["side_criteria"] so gate Block 1 is side-scoped

PATCH (Jan 2026, explicit eff_tid → cohort mapping + gate subcohort_context):
- Prefer cohort_record.trial_id_effective/trial_id matching when present.
- Else use positional mapping effective_trial_ids[i] ↔ enrollment_cohorts[i] when lengths match and no duplicates.
- Populate subctx["subcohort_context"] for gate Block 1, preferring cohort.contextual_text, then cohort.context, then ctx_str.
- Attach audit fields: subctx["cohort_match_method"], subctx["cohort_match_index"].
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -------------------------
# Cost tracking (stage + pipeline)
# -------------------------
@dataclass
class CostTotals:
    done: int = 0
    with_cost: int = 0
    total_cost_usd: float = 0.0

    def add(self, cost_usd: Optional[float]) -> None:
        self.done += 1
        if isinstance(cost_usd, (int, float)):
            self.with_cost += 1
            self.total_cost_usd += float(cost_usd)

    def avg_cost(self) -> float:
        return (self.total_cost_usd / self.with_cost) if self.with_cost else 0.0

    def predicted_final_cost(self, total_planned: int) -> Optional[float]:
        if self.with_cost == 0:
            return None
        return self.avg_cost() * float(total_planned)

    def predicted_remaining_cost(self, total_planned: int) -> Optional[float]:
        pf = self.predicted_final_cost(total_planned)
        if pf is None:
            return None
        return max(0.0, pf - self.total_cost_usd)


def _extract_cost_usd(meta: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    c = meta.get("estimated_cost_usd")
    if isinstance(c, (int, float)):
        return float(c)
    return None


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "(n/a)"
    return f"${x:.6f}"


# -------------------------
# Canonical filename patterns (for processing)
# -------------------------
CANON_LAX_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program.*\.smt2$", re.IGNORECASE)
BASE_NCT_RE = re.compile(r"^(NCT[0-9]+)", re.IGNORECASE)


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


# -------------------------
# DEFINITE JSON/parse failures only
# -------------------------
_DEFINITE_PARSE_MESSAGES = (
    "empty llm output",
    "could not parse json from llm output",
    "parsed json is not an object",
)


def is_definite_parse_failure(
    *,
    exc: Optional[BaseException] = None,
    msg: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Returns True ONLY when we have hard evidence this is a JSON parse failure.
    """
    if exc is not None and isinstance(exc, json.JSONDecodeError):
        return True

    if meta and isinstance(meta, dict):
        ae = meta.get("attempt_errors")
        if isinstance(ae, list):
            for it in ae:
                s = str(it).strip().lower()
                if s.startswith("json_parse_error:"):
                    return True

        err = meta.get("error")
        if isinstance(err, str) and err.strip().lower().startswith("json_parse_error:"):
            return True

    m = (msg or "").strip().lower()
    if m.startswith("json_parse_error:"):
        return True
    for canonical in _DEFINITE_PARSE_MESSAGES:
        if canonical in m:
            return True

    return False


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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def strip_comments(s: str, *, strict_double_semicolon: bool) -> str:
    return re.sub(COMMENT_RE_DOUBLE if strict_double_semicolon else COMMENT_RE_SINGLE, "", s)


def paren_balance_ok(s: str, *, strict_double_semicolon: bool) -> bool:
    t = strip_comments(s, strict_double_semicolon=strict_double_semicolon)
    bal = 0
    for ch in t:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return False
    return bal == 0


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


# -------------------------
# normalize detached JSON blocks -> inline on declare-const line
# -------------------------
BLOCK_VAR_LINE_RE = re.compile(r"^\s*;;\s*([A-Za-z0-9_@]+)\s*$")


def normalize_decl_json_blocks_to_inline(
    text: str,
    *,
    max_lookahead_lines: int = 6,
) -> Tuple[str, int]:
    """
    Convert detached comment-block JSON annotations into inline JSON on the matching declare-const line.

    Pattern supported:
      ;; var_name
      ;; { ...json... }
      (declare-const var_name Sort) ;; "free text"

    If declare-const line already has an inline JSON object, do nothing.
    """
    lines = (text or "").splitlines()
    out: List[str] = []
    attached = 0

    pending_var: Optional[str] = None
    pending_json: Optional[str] = None
    pending_deadline = -1  # inclusive line index where we stop trying to attach

    for idx, line in enumerate(lines):
        mvar = BLOCK_VAR_LINE_RE.match(line)
        if mvar:
            pending_var = mvar.group(1)
            pending_json = None
            pending_deadline = idx + max_lookahead_lines
            out.append(line)
            continue

        if pending_var is not None and idx <= pending_deadline:
            if line.lstrip().startswith(";;") and "{" in line:
                jb = extract_first_json_object_from_line(line)
                if jb:
                    pending_json = jb

        code = code_part_before_comment(line).rstrip()
        mdecl = RE_DECLARE_CONST_HDR.match(code)
        if mdecl:
            var = mdecl.group(1)
            has_inline_json = (extract_first_json_object_from_line(line) is not None)

            if (
                (not has_inline_json)
                and pending_var is not None
                and pending_json is not None
                and idx <= pending_deadline
                and var == pending_var
            ):
                line = line.rstrip() + " ;; " + pending_json
                attached += 1
                pending_var = None
                pending_json = None
                pending_deadline = -1

        if pending_var is not None and idx > pending_deadline:
            pending_var = None
            pending_json = None
            pending_deadline = -1

        out.append(line)

    normalized = "\n".join(out)
    if text.endswith("\n"):
        normalized += "\n"
    return normalized, attached


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

    if not isinstance(obj.get("meaning"), str):
        return "declare_const_json_meaning_not_string"

    for k in required:
        if k == "meaning":
            continue
        if not isinstance(obj.get(k), str):
            return f"declare_const_json_field_not_string:{k}"

    return None


def declare_const_issue_list(text: str, *, cfg: GuardrailConfig) -> List[str]:
    """
    Return ONLY declare-const JSON annotation issues.
    If this returns [], meaning enrichment is not needed.
    """
    allowed_sorts: Set[str] = {"Bool", "Int", "Real", "String"}
    issues: List[str] = []
    for line in (text or "").splitlines():
        if not declare_const_in_code(line):
            continue
        issue = validate_declare_const_json_comment(line, cfg=cfg, allowed_sorts=allowed_sorts)
        if issue:
            issues.append(issue)
            if len(issues) >= 50:
                break
    return issues


def validate_smt_content(text: str, *, cfg: GuardrailConfig) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not text.strip():
        errors.append("empty_or_whitespace")
        return False, errors

    if not paren_balance_ok(text, strict_double_semicolon=cfg.strict_double_semicolon):
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


def is_empty_file(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_file():
            return True
        if p.stat().st_size == 0:
            return True
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return not bool(txt.strip())
    except Exception:
        return True


def base_trial_id(eff_tid: str) -> str:
    m = BASE_NCT_RE.match(eff_tid or "")
    return m.group(1) if m else (eff_tid or "")


# -------------------------
# Shared criteria parsing
# -------------------------
SHARED_INC_RE = re.compile(
    r"Shared\s+inclusion\s+criteria.*?:\s*(.*?)\s*Shared\s+exclusion\s+criteria.*?:",
    re.IGNORECASE | re.DOTALL,
)
SHARED_EXC_RE = re.compile(
    r"Shared\s+exclusion\s+criteria.*?:\s*(.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_shared_inc_exc(shared_context: str) -> Tuple[str, str]:
    inc = ""
    exc = ""
    s = shared_context or ""
    m1 = SHARED_INC_RE.search(s)
    if m1:
        inc = (m1.group(1) or "").strip()
    m2 = SHARED_EXC_RE.search(s)
    if m2:
        exc = (m2.group(1) or "").strip()
    return inc, exc


# -------------------------
# Snapshot + subcohort mapping
# -------------------------
def _candidate_parent_ids(eff_tid: str) -> List[str]:
    base = base_trial_id(eff_tid)
    if eff_tid == base:
        return [base]
    return [eff_tid, base]


def load_snapshot_for_program(snapshot_dir: Path, eff_tid: str, side: str) -> Optional[Dict[str, Any]]:
    for tid in _candidate_parent_ids(eff_tid):
        for suffix in (f"_{side}", ""):
            snap_path = snapshot_dir / f"{tid}{suffix}.json"
            if not snap_path.exists():
                continue
            try:
                return json.loads(snap_path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def _select_enrollment_cohort_for_eff_tid(
    parent_ctx: Dict[str, Any],
    eff_tid: str,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[int]]:
    """
    Return (cohort_record, match_method, match_index).

    match_method:
      - "trial_id_effective"  : matched by cohort_record.trial_id_effective / trial_id
      - "index_effective_ids" : matched by position in parent_ctx.effective_trial_ids
      - "none"                : no match
    """
    pn = parent_ctx.get("preprocessor_normalized") or {}
    enroll = pn.get("enrollment_cohorts") or []
    if not isinstance(enroll, list) or not enroll:
        return None, "none", None

    # 1) Prefer explicit per-cohort mapping when present
    for i, co in enumerate(enroll):
        if not isinstance(co, dict):
            continue
        if co.get("trial_id_effective") == eff_tid or co.get("trial_id") == eff_tid:
            return co, "trial_id_effective", i

    # 2) Fallback: positional mapping effective_trial_ids[i] -> enrollment_cohorts[i]
    eff_ids = parent_ctx.get("effective_trial_ids") or []
    if isinstance(eff_ids, list) and str(eff_tid) in set(map(str, eff_ids)) and len(eff_ids) == len(enroll):
        # extra safety: avoid ambiguous duplicates
        if len(set(map(str, eff_ids))) != len(eff_ids):
            return None, "none", None
        idx = list(map(str, eff_ids)).index(str(eff_tid))
        if 0 <= idx < len(enroll) and isinstance(enroll[idx], dict):
            return enroll[idx], "index_effective_ids", idx

    return None, "none", None


# -------------------------
# IR discovery (LAX + de-dupe)
# -------------------------
@dataclass(frozen=True)
class IRProgram:
    eff_tid: str
    side: str
    path: Path


def discover_ir_programs(ir_dir: Path) -> List[IRProgram]:
    """
    Find candidate programs in canonical/:
      NCT..._(inclusion|exclusion)_program*.smt2
    De-dupe by (eff_tid, side), preferring the shortest filename (usually the canonical one).
    """
    candidates: List[IRProgram] = []
    for p in ir_dir.glob("*.smt2"):
        m = CANON_LAX_RE.match(p.name)
        if not m:
            continue
        eff, side = m.group(1), m.group(2).lower()
        candidates.append(IRProgram(eff_tid=eff, side=side, path=p))

    best: Dict[Tuple[str, str], IRProgram] = {}
    for prog in candidates:
        k = (prog.eff_tid, prog.side)
        if k not in best:
            best[k] = prog
        else:
            if len(prog.path.name) < len(best[k].path.name):
                best[k] = prog

    out = list(best.values())
    out.sort(key=lambda x: (x.eff_tid, x.side))
    return out


def copy_canonical_files(src_ir_dir: Path, dst_ir_dir: Path) -> int:
    """
    Initialize canonical/ by copying ALL *.smt2 from src.
    """
    ensure_dir(dst_ir_dir)
    n = 0
    for p in sorted(src_ir_dir.glob("*.smt2")):
        dst = dst_ir_dir / p.name
        shutil.copy2(p, dst)
        n += 1
    return n


def load_allowlist_prefixes(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Allowlist not found: {path}")
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def program_selected(
    prog: IRProgram,
    *,
    trial_prefix: Optional[str],
    allow_prefixes: Set[str],
    side_filter: Optional[str],
) -> bool:
    if side_filter and prog.side != side_filter:
        return False
    if trial_prefix and not prog.eff_tid.startswith(trial_prefix):
        return False
    if allow_prefixes and (not any(prog.eff_tid.startswith(pref) for pref in allow_prefixes)):
        return False
    return True


# -------------------------
# Step specs (gate before underconstraint before meaning)
# -------------------------
PIPELINE_ORDER = [0, 1, 2, 3, 4, 5]
STEP_NAME = {
    0: "step0_repair",
    1: "step1_polarity",
    2: "step2_logic",
    3: "step3_gate",
    4: "step4_underconstraint",
    5: "step5_meaning",
}
PROMPT_FILE = {
    0: "smt_repair.prompt",
    1: "smt_polarity_fix.prompt",
    2: "smt_logic_fix.prompt",
    3: "smt_criteria_gate.prompt",
    4: "smt_fix_underconstraint.prompt",
    5: "smt_variable_meaning_enricher.prompt",
}


def _format_stage_breakdown(prior: Dict[str, CostTotals], current_step_name: str, current_stage: CostTotals) -> str:
    parts: List[str] = []
    for sid in PIPELINE_ORDER:
        nm = STEP_NAME[sid]
        if nm == current_step_name:
            ct = current_stage
        else:
            ct = prior.get(nm)
        if ct is None:
            continue
        if ct.done == 0 and ct.total_cost_usd == 0.0:
            continue
        parts.append(f"{sid}:{_fmt_money(ct.total_cost_usd)}({ct.with_cost}/{ct.done})")
    return " ".join(parts) if parts else "(none)"


# -------------------------
# Corpus.jsonl loader (KEYED BY BASE ID)
# -------------------------
def build_corpus_lookup(corpus_jsonl: Optional[Path], wanted_base_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load rows from corpus JSONL into a dict keyed by BASE NCT id.
    This matches how the pipeline uses parent/base ids, and allows early stop.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if corpus_jsonl is None or (not corpus_jsonl.exists()) or (not wanted_base_ids):
        return out

    try:
        with corpus_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tid = obj.get("_id")
                if not isinstance(tid, str) or not tid:
                    continue
                base = base_trial_id(tid)
                if base in wanted_base_ids and base not in out:
                    out[base] = obj
                    if len(out) == len(wanted_base_ids):
                        break
    except Exception:
        return out

    return out


def _get_corpus_inc_exc(corpus_lookup: Dict[str, Dict[str, Any]], parent_tid: str) -> Tuple[str, str, str]:
    row = corpus_lookup.get(parent_tid) or corpus_lookup.get(base_trial_id(parent_tid))
    if not row:
        return "", "", ""
    md = row.get("metadata") or {}
    inc = (md.get("inclusion_criteria") or "").strip()
    exc = (md.get("exclusion_criteria") or "").strip()
    ctx = (row.get("text") or md.get("brief_summary") or "").strip()
    return inc, exc, ctx


def _get_corpus_shared_context(corpus_lookup: Dict[str, Dict[str, Any]], parent_tid: str) -> str:
    row = corpus_lookup.get(parent_tid) or corpus_lookup.get(base_trial_id(parent_tid))
    if not row:
        return ""
    md = row.get("metadata") or {}
    txt = (row.get("text") or "").strip()
    if txt:
        return txt
    for k in ("brief_summary", "detailed_description", "description"):
        v = (md.get(k) or "").strip()
        if v:
            return v
    return (row.get("title") or "").strip()


def _build_minimal_subctx(
    parent_ctx: Dict[str, Any],
    eff_tid: str,
    side: str,
    cohort_record: Optional[Dict[str, Any]],
    corpus_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    parent_tid = (
        parent_ctx.get("parent_trial_id")
        or parent_ctx.get("trial_id_parent")
        or parent_ctx.get("trial_id")
        or base_trial_id(eff_tid)
    )

    parent_tid_s = str(parent_tid)
    corpus_item = corpus_lookup.get(parent_tid_s) or corpus_lookup.get(base_trial_id(parent_tid_s))

    pn = parent_ctx.get("preprocessor_normalized") or {}

    snapshot_shared_context = parent_ctx.get("shared_context") or pn.get("shared_context", "") or ""
    corpus_shared_context = _get_corpus_shared_context(corpus_lookup, parent_tid_s)
    shared_context = corpus_shared_context.strip() or (snapshot_shared_context or "")

    if cohort_record is not None:
        cid = (
            cohort_record.get("id")
            or cohort_record.get("cohort_id")
            or cohort_record.get("substudy_id")
            or "C?"
        )
        label = (
            cohort_record.get("label")
            or cohort_record.get("cohort_label")
            or cohort_record.get("cohort_name")
            or cohort_record.get("substudy_label")
            or str(cid)
        )
        label = cohort_record.get("label") or cohort_record.get("cohort_name") or str(cid)
        inc = cohort_record.get("inclusion_criteria", "") or ""
        exc = cohort_record.get("exclusion_criteria", "") or ""
        ctx_str = cohort_record.get("context", "") or ""
        ctxt = cohort_record.get("contextual_text") or ""
        if not ctxt:
            pieces = [p for p in (shared_context, ctx_str) if p]
            ctxt = "\n\n".join(pieces)
    else:
        cid = "default"
        label = "Default cohort"

        inc = parent_ctx.get("inclusion_criteria", "") or ""
        exc = parent_ctx.get("exclusion_criteria", "") or ""

        ctx_str = parent_ctx.get("cohort_context_raw", "") or parent_ctx.get("context", "") or ""
        ctxt = parent_ctx.get("contextual_text") or shared_context or ""

        if (not inc.strip()) and (not exc.strip()) and snapshot_shared_context:
            inc2, exc2 = extract_shared_inc_exc(snapshot_shared_context)
            if inc2.strip() or exc2.strip():
                inc = inc2
                exc = exc2
                label = "Default cohort (from snapshot shared_context)"

        enroll = pn.get("enrollment_cohorts") or []
        if (not inc.strip()) and (not exc.strip()) and enroll:
            c0 = enroll[0] or {}
            inc = (c0.get("inclusion_criteria") or "").strip()
            exc = (c0.get("exclusion_criteria") or "").strip()
            cid = c0.get("id") or "C1"
            label = f"Default cohort (fallback to {c0.get('label') or c0.get('cohort_name') or cid})"
            if not ctxt.strip():
                ctxt = (c0.get("contextual_text") or shared_context or "").strip()

        if (not inc.strip()) and (not exc.strip()):
            inc_c, exc_c, ctx_c = _get_corpus_inc_exc(corpus_lookup, parent_tid_s)
            if inc_c or exc_c:
                inc = inc_c
                exc = exc_c
                label = "Default cohort (from corpus.jsonl)"
                if not ctxt.strip():
                    ctxt = (ctx_c or shared_context or "").strip()

    # Side-specific criteria ONLY (used by gate Block 1).
    side_criteria = inc if side == "inclusion" else exc

    # Gate subcohort context (Block 1): prefer rich cohort contextual_text, then cohort context, then ctx_str.
    subcohort_context = ""
    if cohort_record is not None:
        subcohort_context = (cohort_record.get("contextual_text") or "").strip()
        if not subcohort_context:
            subcohort_context = (cohort_record.get("context") or "").strip()
    if not subcohort_context:
        subcohort_context = (ctx_str or "").strip()

    return {
        "trial_id_parent": parent_tid,
        "trial_id": eff_tid,
        "trial_id_effective": eff_tid,
        "cohort_id": cid,
        "cohort_label": label,
        "inclusion_criteria": inc,
        "exclusion_criteria": exc,
        "side_criteria": side_criteria,

        # NEW: explicit field used by gate for <subcohort_context>
        "subcohort_context": subcohort_context,

        # legacy/debug
        "context": ctx_str,
        "contextual_text": ctxt,
        "shared_context": shared_context,
        "inc_exc": side,
        "preprocessor_normalized": pn,

        # Raw corpus JSON object for gate Block 2.
        "corpus_item": corpus_item,
    }


def extract_subcohort_for_eff_tid(
    parent_ctx: Dict[str, Any],
    eff_tid: str,
    side: str,
    corpus_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    # 1) Normal path FIRST: use preprocessor_normalized.enrollment_cohorts mapping
    cohort_record, method, idx = _select_enrollment_cohort_for_eff_tid(parent_ctx, eff_tid)
    if cohort_record is not None:
        out = _build_minimal_subctx(parent_ctx, eff_tid, side, cohort_record, corpus_lookup)
        out["cohort_match_method"] = method
        out["cohort_match_index"] = idx
        return out

    # 0) Fallback ONLY: upstream explicit cohort contexts (often context-only, may omit criteria)
    subctxs = parent_ctx.get("__cohort_contexts__") or parent_ctx.get("__substudy_contexts__", [])
    if subctxs:
        for sc in subctxs:
            if sc.get("trial_id") == eff_tid or sc.get("trial_id_effective") == eff_tid:
                out = _build_minimal_subctx(parent_ctx, eff_tid, side, sc, corpus_lookup)
                out["cohort_match_method"] = "explicit___cohort_contexts__"
                out["cohort_match_index"] = None
                return out

        eff_ids = parent_ctx.get("effective_trial_ids") or []
        if isinstance(eff_ids, list) and str(eff_tid) in set(map(str, eff_ids)) and len(eff_ids) == len(subctxs):
            idx2 = list(map(str, eff_ids)).index(str(eff_tid))
            out = _build_minimal_subctx(parent_ctx, eff_tid, side, subctxs[idx2], corpus_lookup)
            out["cohort_match_method"] = "explicit___cohort_contexts___index"
            out["cohort_match_index"] = idx2
            return out

    # 2) Final fallback: build default cohort context
    out = _build_minimal_subctx(parent_ctx, eff_tid, side, None, corpus_lookup)
    out["cohort_match_method"] = "none_default"
    out["cohort_match_index"] = None
    return out


# -------------------------
# Artifact helpers
# -------------------------
def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suf = base.suffix
    parent = base.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}__{i}{suf}"
        if not cand.exists():
            return cand
    return parent / f"{stem}__{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}{suf}"


def _write_inspect_artifacts(
    *,
    inspect_dir: Path,
    filename: str,
    repaired_smt: Optional[str],
    strict_errors: Optional[List[str]],
    meta: Optional[Dict[str, Any]],
) -> None:
    ensure_dir(inspect_dir)

    if repaired_smt is not None:
        out_smt = _unique_path(inspect_dir / filename)
        out_smt.write_text(repaired_smt, encoding="utf-8")

    if strict_errors is not None:
        err_path = _unique_path(inspect_dir / f"{filename}.strict_errors.txt")
        err_path.write_text("\n".join(strict_errors) + "\n", encoding="utf-8")

    if meta is not None:
        meta_path = _unique_path(inspect_dir / f"{filename}.meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_parse_inspect(
    *,
    parse_inspect_dir: Path,
    filename: str,
    eff_tid: str,
    side: str,
    cohort_id: str,
    step_name: str,
    prompt: Optional[str],
    raw: Optional[str],
    meta: Optional[Dict[str, Any]],
    note: str,
) -> None:
    ensure_dir(parse_inspect_dir)
    base = f"{eff_tid}_{side}_cohort-{cohort_id}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    prefix = f"{base}__{stamp}"

    if prompt is not None:
        (parse_inspect_dir / f"{prefix}_prompt.txt").write_text(prompt, encoding="utf-8")
    if raw is not None:
        (parse_inspect_dir / f"{prefix}_raw.txt").write_text(raw, encoding="utf-8")

    meta_out: Dict[str, Any] = dict(meta or {})
    meta_out.update(
        {
            "note": note,
            "step_name": step_name,
            "eff_tid": eff_tid,
            "side": side,
            "cohort_id": cohort_id,
            "filename": filename,
            "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    (parse_inspect_dir / f"{prefix}_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mirror_to_after(path: Path, after_dir: Path) -> None:
    """Best-effort mirror of a canonical SMT file into after/."""
    try:
        if not path.exists() or not path.is_file():
            return
        shutil.copy2(path, after_dir / path.name)
    except Exception as e:
        print(f"[WARN] failed to mirror file into after/: {path.name}: {e}", file=sys.stderr, flush=True)


# -------------------------
# Snapshot canonical dir into before/after (COPY ALL *.smt2)
# -------------------------
def _snapshot_dir_all_smt2(*, src_canonical_dir: Path, dst_dir: Path) -> int:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    ensure_dir(dst_dir)

    n = 0
    for p in sorted(src_canonical_dir.glob("*.smt2")):
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n


# -------------------------
# Detailed MBENCH helpers (per program dir, per attempt prompt/raw)
# -------------------------
def _mbench_prog_key(eff_tid: str, side: str, cohort_id: str) -> str:
    return f"{eff_tid}_{side}_cohort-{cohort_id}"


def _mbench_prog_dir(mbench_dir: Path, eff_tid: str, side: str, cohort_id: str) -> Path:
    d = mbench_dir / _mbench_prog_key(eff_tid, side, cohort_id)
    ensure_dir(d)
    return d


def _append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_mbench_call(
    *,
    mbench_dir: Path,
    step_name: str,
    eff_tid: str,
    side: str,
    cohort_id: str,
    filename: str,
    attempt: int,
    prompt: str,
    raw: str,
    model_name: str,
    azure_endpoint: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    d = _mbench_prog_dir(mbench_dir, eff_tid, side, cohort_id)

    a = f"{attempt:03d}"
    (d / f"call_{a}_prompt.txt").write_text(prompt or "", encoding="utf-8")
    (d / f"call_{a}_raw.txt").write_text(raw or "", encoding="utf-8")

    evt: Dict[str, Any] = {
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "llm_call",
        "step_name": step_name,
        "eff_tid": eff_tid,
        "side": side,
        "cohort_id": cohort_id,
        "filename": filename,
        "attempt": attempt,
        "prompt_len": len(prompt or ""),
        "raw_len": len(raw or ""),
        "model_name": model_name,
        "openai_endpoint": azure_endpoint,
        "raw_empty": (not bool((raw or "").strip())),
    }
    if extra and isinstance(extra, dict):
        evt.update(extra)

    _append_jsonl(d / "events.jsonl", evt)


def _write_mbench_final(
    *,
    mbench_dir: Path,
    step_name: str,
    eff_tid: str,
    side: str,
    cohort_id: str,
    filename: str,
    meta: Optional[Dict[str, Any]],
    strict_errors: Optional[List[str]] = None,
    note: str = "",
) -> None:
    d = _mbench_prog_dir(mbench_dir, eff_tid, side, cohort_id)

    out: Dict[str, Any] = dict(meta or {})
    out.update(
        {
            "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kind": "final_meta",
            "step_name": step_name,
            "eff_tid": eff_tid,
            "side": side,
            "cohort_id": cohort_id,
            "filename": filename,
            "note": note,
        }
    )
    if strict_errors is not None:
        out["strict_errors"] = list(strict_errors)

    (d / "final_meta.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -------------------------
# Per-program worker
# -------------------------
def process_program(
    *,
    prog: IRProgram,
    step_id: int,
    step_name: str,
    snapshot_dir: Path,
    strict_cfg: GuardrailConfig,
    corpus_lookup: Dict[str, Dict[str, Any]],
    engine_version: str,
    azure_endpoint: str,
    model_name: str,
    prompt_template: Optional[str],
    parse_inspect_dir: Path,
    inspect_dir: Path,
    mbench_dir: Path,
    after_dir: Path,
) -> Tuple[Optional[float], Dict[str, int], Dict[str, int]]:
    """
    Returns (cost_usd, counts_delta, reasons_delta).
    """
    from trial_compiler.ir_finalizer.stages.smt_repair_module import SMTRepairer
    from trial_compiler.ir_finalizer.stages.smt_polarity_fix_module import SMTPolarityFixer
    from trial_compiler.ir_finalizer.stages.smt_logic_fix_module import SMTLogicFixer
    from trial_compiler.ir_finalizer.stages.smt_criteria_gate_module import SMTCriteriaGate
    from trial_compiler.ir_finalizer.stages.smt_fix_underconstraint_module import SMTUnderconstraintFixer
    from trial_compiler.ir_finalizer.stages.smt_variable_meaning_enricher_module import SMTVariableMeaningEnricher

    if engine_version == "gpt-5":
        from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
    else:
        from smt_core.inference_engine import AzureInferenceEngine  # type: ignore

    counts = Counter()
    reasons = Counter()
    cost_usd: Optional[float] = None

    p = prog.path

    # 1) Empty / whitespace:
    # - For gate step: DO NOT early-exit; gate decision is about NL criteria presence.
    # - For other steps: keep old behavior (no LLM call).
    if is_empty_file(p):
        if step_id != 3:
            counts["empty"] += 1
            reasons["empty"] += 1
            try:
                _write_mbench_final(
                    mbench_dir=mbench_dir,
                    step_name=step_name,
                    eff_tid=prog.eff_tid,
                    side=prog.side,
                    cohort_id="unknown",
                    filename=p.name,
                    meta={"executed": False, "error": "empty_file_skip_non_gate"},
                    note="empty_file_skip_non_gate",
                )
            except Exception:
                pass
            _mirror_to_after(p, after_dir)
            return cost_usd, dict(counts), dict(reasons)

    # 2) Read original SMT (if empty, read_text is fine; we still run gate)
    try:
        original = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    except Exception as e:
        _write_inspect_artifacts(
            inspect_dir=inspect_dir,
            filename=p.name,
            repaired_smt=None,
            strict_errors=None,
            meta={"error": f"read_error:{e}"},
        )
        counts["error"] += 1
        reasons["read_error"] += 1
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id="unknown",
                filename=p.name,
                meta={"executed": False, "error": f"read_error:{e}"},
                note="read_error",
            )
        except Exception:
            pass
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 3) Load subcohort snapshot
    parent_ctx = load_snapshot_for_program(snapshot_dir, prog.eff_tid, prog.side)
    if parent_ctx is None:
        counts["missing_snapshot"] += 1
        reasons["missing_snapshot"] += 1
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id="unknown",
                filename=p.name,
                meta={"executed": False, "error": "missing_snapshot"},
                note="missing_snapshot",
            )
        except Exception:
            pass
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    subctx = extract_subcohort_for_eff_tid(parent_ctx, prog.eff_tid, prog.side, corpus_lookup)
    cohort_id = str(subctx.get("cohort_id") or subctx.get("substudy_id") or "default")

    # 4) Step applicability (polarity step only for exclusion)
    if step_id == 1 and prog.side != "exclusion":
        counts["no_change"] += 1
        reasons["not_applicable_side"] += 1
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta={"executed": False, "error": "not_applicable_side"},
                note="not_applicable_side",
            )
        except Exception:
            pass
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 4b) Logic step applicability (step2_logic ONLY for exclusion)
    if step_id == 2 and prog.side != "exclusion":
        counts["no_change"] += 1
        reasons["not_applicable_side"] += 1
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta={"executed": False, "error": "not_applicable_side"},
                note="not_applicable_side",
            )
        except Exception:
            pass
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 5) Meaning step preflight (step_id == 5):
    #    deterministic normalization ONLY; DO NOT skip meaning anymore.
    if step_id == 5:
        orig_norm, n_attached = normalize_decl_json_blocks_to_inline(original)
        if n_attached and (orig_norm.strip() != original.strip()):
            try:
                p.write_text(orig_norm, encoding="utf-8")
                original = orig_norm
                reasons["normalized_decl_json_blocks_pre_meaning"] += 1
            except Exception as e:
                _write_inspect_artifacts(
                    inspect_dir=inspect_dir,
                    filename=p.name,
                    repaired_smt=orig_norm,
                    strict_errors=None,
                    meta={"error": f"write_error_pre_meaning_normalize:{e}"},
                )
                counts["error"] += 1
                reasons["write_error"] += 1
                try:
                    _write_mbench_final(
                        mbench_dir=mbench_dir,
                        step_name=step_name,
                        eff_tid=prog.eff_tid,
                        side=prog.side,
                        cohort_id=cohort_id,
                        filename=p.name,
                        meta={"executed": False, "error": f"write_error_pre_meaning_normalize:{e}"},
                        note="write_error_pre_meaning_normalize",
                    )
                except Exception:
                    pass
                _mirror_to_after(p, after_dir)
                return cost_usd, dict(counts), dict(reasons)

    # 6) Engine + call_llm + fixer
    current: Dict[str, str] = {
        "eff_tid": prog.eff_tid,
        "side": prog.side,
        "cohort_id": cohort_id,
        "filename": p.name,
    }
    call_idx = 0

    engine = AzureInferenceEngine(
        endpoint=azure_endpoint,
        api_key_env_var="OPENAI_API_KEY",
        model_name=model_name,
    )

    def call_llm(prompt: str) -> str:
        nonlocal call_idx
        call_idx += 1

        try:
            raw0 = engine(prompt)[0]
        except Exception as e:
            # Log prompt even when engine fails
            try:
                _write_mbench_call(
                    mbench_dir=mbench_dir,
                    step_name=step_name,
                    eff_tid=current["eff_tid"],
                    side=current["side"],
                    cohort_id=current["cohort_id"],
                    filename=current["filename"],
                    attempt=call_idx,
                    prompt=prompt,
                    raw="",
                    model_name=model_name,
                    azure_endpoint=azure_endpoint,
                    extra={"error": f"engine_call_error:{e}", "exception_type": type(e).__name__},
                )
            except Exception:
                pass
            raise

        raw_s = "" if raw0 is None else str(raw0)

        # ALWAYS store per-attempt prompt+raw (even empty raw)
        try:
            _write_mbench_call(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=current["eff_tid"],
                side=current["side"],
                cohort_id=current["cohort_id"],
                filename=current["filename"],
                attempt=call_idx,
                prompt=prompt,
                raw=raw_s,
                model_name=model_name,
                azure_endpoint=azure_endpoint,
            )
        except Exception:
            pass

        return raw_s

    if step_id == 0:
        fixer = SMTRepairer(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)
    elif step_id == 1:
        fixer = SMTPolarityFixer(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)
    elif step_id == 2:
        fixer = SMTLogicFixer(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)
    elif step_id == 3:
        fixer = SMTCriteriaGate(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)
    elif step_id == 4:
        fixer = SMTUnderconstraintFixer(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)
    elif step_id == 5:
        fixer = SMTVariableMeaningEnricher(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)
    else:
        raise ValueError(f"unknown step_id={step_id}")

    # 7) Run fixer
    try:
        if step_id == 0:
            repaired, meta = fixer.repair(subctx, original)  # type: ignore[attr-defined]
        elif step_id == 1:
            repaired, meta = fixer.fix(subctx, original)  # type: ignore[attr-defined]
        elif step_id == 2:
            repaired, meta = fixer.fix(subctx, prog.side, original)  # type: ignore[attr-defined]
        elif step_id == 3:
            repaired, meta = fixer.gate(subctx, prog.side, original)  # type: ignore[attr-defined]
        elif step_id == 4:
            repaired, meta = fixer.repair(subctx, original)  # type: ignore[attr-defined]
        elif step_id == 5:
            repaired, meta = fixer.enrich(subctx, original, side_hint=prog.side)  # type: ignore[attr-defined]
        else:
            raise ValueError("unknown step")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        msg = str(e)
        prompt_last = getattr(fixer, "last_prompt", None)
        raw_last = getattr(fixer, "last_raw", None)

        if is_definite_parse_failure(exc=e, msg=msg, meta=None):
            try:
                _write_parse_inspect(
                    parse_inspect_dir=parse_inspect_dir,
                    filename=p.name,
                    eff_tid=prog.eff_tid,
                    side=prog.side,
                    cohort_id=cohort_id,
                    step_name=step_name,
                    prompt=prompt_last,
                    raw=raw_last,
                    meta={"error": msg, "exception_type": type(e).__name__},
                    note="definite_parse_failure: fixer_exception",
                )
            except Exception:
                pass

        _write_inspect_artifacts(
            inspect_dir=inspect_dir,
            filename=p.name,
            repaired_smt=None,
            strict_errors=None,
            meta={"error": f"fixer_exception:{e}"},
        )
        counts["error"] += 1
        reasons["fixer_exception"] += 1

        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta={"executed": False, "error": f"fixer_exception:{e}", "exception_type": type(e).__name__},
                note="fixer_exception",
            )
        except Exception:
            pass

        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 8) Meta-indicated JSON parse failure → parse_inspect
    try:
        if is_definite_parse_failure(meta=meta or {}, msg=""):
            prompt_last = getattr(fixer, "last_prompt", None)
            raw_last = getattr(fixer, "last_raw", None)
            _write_parse_inspect(
                parse_inspect_dir=parse_inspect_dir,
                filename=p.name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                step_name=step_name,
                prompt=prompt_last,
                raw=raw_last,
                meta=meta or {},
                note="definite_parse_failure: meta_indicates_json_parse_error",
            )
    except Exception:
        pass

    # 9) Cost extraction
    cost_usd = _extract_cost_usd(meta or {})

    # 10) Executed flag
    executed = (meta or {}).get("executed") is True
    if not executed:
        _write_inspect_artifacts(
            inspect_dir=inspect_dir,
            filename=p.name,
            repaired_smt=None,
            strict_errors=None,
            meta=meta or {"error": "executed_false"},
        )
        counts["error"] += 1
        reasons["executed_false"] += 1

        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta=meta or {"executed": False, "error": "executed_false"},
                note="executed_false",
            )
        except Exception:
            pass

        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 11) Gate step handling (DELETE file or KEEP file; no strict validation)
    if step_id == 3:
        # Write final meta BEFORE delete/keep return
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta=meta or {},
                note=("gate_delete" if (meta or {}).get("delete_file") is True else "gate_keep"),
            )
        except Exception:
            pass

        if (meta or {}).get("delete_file") is True:
            try:
                if p.exists():
                    p.unlink()
                counts["deleted"] += 1
                reasons["gate_delete"] += 1
            except Exception as e:
                _write_inspect_artifacts(
                    inspect_dir=inspect_dir,
                    filename=p.name,
                    repaired_smt=None,
                    strict_errors=None,
                    meta={"error": f"delete_error:{e}", **(meta or {})},
                )
                counts["error"] += 1
                reasons["delete_error"] += 1
                _mirror_to_after(p, after_dir)
                return cost_usd, dict(counts), dict(reasons)

            # Best-effort: remove any already-mirrored copy in after_dir (real-time mirror)
            try:
                ap = after_dir / p.name
                if ap.exists():
                    ap.unlink()
            except Exception:
                pass

            return cost_usd, dict(counts), dict(reasons)

        counts["no_change"] += 1
        reasons["gate_keep"] += 1
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 12) No-change policies (with normalization attempt)
    if (meta or {}).get("no_change_required") is True or (meta or {}).get("changed_smt") is False:
        orig_norm, n_attached = normalize_decl_json_blocks_to_inline(original)
        if n_attached:
            ok0, _errs0 = validate_smt_content(orig_norm, cfg=strict_cfg)
            if ok0:
                try:
                    p.write_text(orig_norm, encoding="utf-8")
                    counts["updated"] += 1
                    reasons["normalized_decl_json_blocks"] += 1
                    try:
                        _write_mbench_final(
                            mbench_dir=mbench_dir,
                            step_name=step_name,
                            eff_tid=prog.eff_tid,
                            side=prog.side,
                            cohort_id=cohort_id,
                            filename=p.name,
                            meta=meta or {},
                            note="no_change_but_normalized_decl_json_blocks",
                        )
                    except Exception:
                        pass
                    _mirror_to_after(p, after_dir)
                    return cost_usd, dict(counts), dict(reasons)
                except Exception as e:
                    _write_inspect_artifacts(
                        inspect_dir=inspect_dir,
                        filename=p.name,
                        repaired_smt=orig_norm,
                        strict_errors=None,
                        meta={"error": f"write_error:{e}", **(meta or {})},
                    )
                    counts["error"] += 1
                    reasons["write_error"] += 1
                    try:
                        _write_mbench_final(
                            mbench_dir=mbench_dir,
                            step_name=step_name,
                            eff_tid=prog.eff_tid,
                            side=prog.side,
                            cohort_id=cohort_id,
                            filename=p.name,
                            meta={"executed": False, "error": f"write_error:{e}", **(meta or {})},
                            note="write_error_after_normalize",
                        )
                    except Exception:
                        pass
                    _mirror_to_after(p, after_dir)
                    return cost_usd, dict(counts), dict(reasons)

        counts["no_change"] += 1
        reasons["no_change"] += 1
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta=meta or {},
                note="no_change",
            )
        except Exception:
            pass
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    if (repaired or "").strip() == (original or "").strip():
        counts["no_change"] += 1
        reasons["same_text"] += 1
        try:
            _write_mbench_final(
                mbench_dir=mbench_dir,
                step_name=step_name,
                eff_tid=prog.eff_tid,
                side=prog.side,
                cohort_id=cohort_id,
                filename=p.name,
                meta=meta or {},
                note="same_text",
            )
        except Exception:
            pass
        _mirror_to_after(p, after_dir)
        return cost_usd, dict(counts), dict(reasons)

    # 13) Strict SMT validation (normalize detached blocks first)
    repaired_norm, n_attached_rep = normalize_decl_json_blocks_to_inline(repaired)
    if n_attached_rep:
        meta = dict(meta or {})
        meta["normalized_decl_json_blocks_attached"] = int(n_attached_rep)
        repaired = repaired_norm

    ok, errs = validate_smt_content(repaired, cfg=strict_cfg)
    if ok:
        try:
            p.write_text(repaired, encoding="utf-8")
            counts["updated"] += 1
            reasons["updated"] += 1
            note = "updated"
        except Exception as e:
            _write_inspect_artifacts(
                inspect_dir=inspect_dir,
                filename=p.name,
                repaired_smt=repaired,
                strict_errors=None,
                meta={"error": f"write_error:{e}", **(meta or {})},
            )
            counts["error"] += 1
            reasons["write_error"] += 1
            note = f"write_error:{e}"
    else:
        _write_inspect_artifacts(
            inspect_dir=inspect_dir,
            filename=p.name,
            repaired_smt=repaired,
            strict_errors=errs,
            meta=meta,
        )
        counts["invalid_strict"] += 1
        reasons[(errs[0] if errs else "invalid_strict")] += 1
        note = "invalid_strict"

    try:
        _write_mbench_final(
            mbench_dir=mbench_dir,
            step_name=step_name,
            eff_tid=prog.eff_tid,
            side=prog.side,
            cohort_id=cohort_id,
            filename=p.name,
            meta=meta or {},
            strict_errors=(errs if not ok else None),
            note=note,
        )
    except Exception:
        pass

    _mirror_to_after(p, after_dir)
    return cost_usd, dict(counts), dict(reasons)


# -------------------------
# Run a step
# -------------------------
def run_step(
    *,
    step_id: int,
    canonical_dir: Path,
    out_root_dir: Path,
    snapshot_dir: Path,
    strict_cfg: GuardrailConfig,
    trial_prefix: Optional[str],
    allow_prefixes: Set[str],
    yes: bool,
    corpus_jsonl: Optional[Path],
    pipeline_costs_so_far: Dict[str, CostTotals],
    pipeline_total_cost_usd_so_far: float,
    progress_every: int,
    max_workers: int,
) -> CostTotals:
    step_name = STEP_NAME[step_id]
    print(f"\n===== RUN {step_name} =====", flush=True)

    # step1 and step2 should run ONLY on exclusion files
    side_filter = "exclusion" if step_id in (1, 2) else None

    progs = [
        p
        for p in discover_ir_programs(canonical_dir)
        if program_selected(p, trial_prefix=trial_prefix, allow_prefixes=allow_prefixes, side_filter=side_filter)
    ]

    # Step5 (meaning) now runs on ALL files (no pre-filtering).
    if step_id == 5:
        print(f"[MEANING_FILTER] disabled; will_process={len(progs)}", flush=True)

    total = len(progs)
    print(f"[PLAN] files={total} (canonical_dir={canonical_dir})", flush=True)

    if total == 0:
        print("[SKIP] nothing to do.", flush=True)
        return CostTotals()

    if not yes and sys.stdin.isatty():
        ans = input(f"Proceed with {step_name} on {total} file(s)? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            raise SystemExit("[ABORT] User declined.")

    scripts_dir = Path(__file__).resolve().parent
    prompt_path = scripts_dir / "prompt" / PROMPT_FILE[step_id]
    prompt_template: Optional[str] = None
    if prompt_path.exists():
        prompt_template = prompt_path.read_text(encoding="utf-8")
        print(f"[PROMPT] loaded {prompt_path}", flush=True)
    else:
        # Meaning step (step5) requires template; others can use built-ins
        if step_id == 5:
            raise FileNotFoundError(f"[PROMPT] meaning step requires template missing at {prompt_path}")
        print(f"[PROMPT] not found at {prompt_path}; fixer will use built-in prompt if supported.", flush=True)

    azure_endpoint = os.environ.get("OPENAI_ENDPOINT", "").strip()
    if not azure_endpoint:
        raise EnvironmentError("OPENAI_ENDPOINT is required.")

    from smt_core.engine_factory import detect_engine_and_model
    engine_version, model_name = detect_engine_and_model()

    print(f"[ENGINE] {engine_version} (model={model_name})", flush=True)
    print(f"[EXECUTOR] max_workers={max_workers if max_workers > 0 else 1} (threads)", flush=True)

    wanted_base = {base_trial_id(p.eff_tid) for p in progs}
    corpus_lookup = build_corpus_lookup(corpus_jsonl, wanted_base)
    if corpus_jsonl is not None:
        print(f"[CORPUS] corpus_jsonl={corpus_jsonl} wanted={len(wanted_base)} loaded={len(corpus_lookup)}", flush=True)

    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    step_run_dir = out_root_dir / "step_runs" / step_name / run_id
    before_dir = step_run_dir / "before"
    after_dir = step_run_dir / "after"
    inspect_dir = step_run_dir / "inspect"
    parse_inspect_dir = step_run_dir / "parse_inspect"
    mbench_dir = step_run_dir / "mbench"

    ensure_dir(step_run_dir)
    ensure_dir(inspect_dir)
    ensure_dir(parse_inspect_dir)
    ensure_dir(mbench_dir)
    ensure_dir(after_dir)

    before_n = _snapshot_dir_all_smt2(src_canonical_dir=canonical_dir, dst_dir=before_dir)
    print(f"[STEPDIR] {step_run_dir}", flush=True)
    print(f"[SNAPSHOT] before: {before_dir} (files={before_n})", flush=True)

    interrupted = False
    counts = Counter()
    reasons = Counter()
    stage_cost = CostTotals()

    def _print_running_cost_line() -> None:
        done = stage_cost.done

        pred_final_stage = stage_cost.predicted_final_cost(total)
        pred_rem_stage = stage_cost.predicted_remaining_cost(total)
        pred_stage_part = (
            f"pred_stage_total={_fmt_money(pred_final_stage)} "
            f"pred_stage_remaining={_fmt_money(pred_rem_stage)}"
        )

        pipeline_now = pipeline_total_cost_usd_so_far + stage_cost.total_cost_usd

        if pred_final_stage is not None:
            pipeline_pred_total = pipeline_total_cost_usd_so_far + pred_final_stage
            pipeline_pred_rem = max(0.0, pipeline_pred_total - pipeline_now)
            pred_pipeline_part = (
                f"pred_pipeline_total={_fmt_money(pipeline_pred_total)} "
                f"pred_pipeline_remaining={_fmt_money(pipeline_pred_rem)}"
            )
        else:
            pred_pipeline_part = "pred_pipeline_total=(n/a) pred_pipeline_remaining=(n/a)"

        breakdown = _format_stage_breakdown(pipeline_costs_so_far, step_name, stage_cost)

        msg = (
            f"\r[COST] step={step_name} done={done}/{total} "
            f"stage_cost={_fmt_money(stage_cost.total_cost_usd)} "
            f"pipeline_cost={_fmt_money(pipeline_now)} "
            f"stage_cost_lines={stage_cost.with_cost} "
            f"unknown_cost_lines={stage_cost.done - stage_cost.with_cost} "
            f"{pred_stage_part} {pred_pipeline_part} "
            f"by_stage={breakdown}"
        )
        print(msg, end="", file=sys.stdout, flush=True)

    try:
        if max_workers is None or max_workers <= 1:
            for prog in progs:
                cost_usd, c_counts, c_reasons = process_program(
                    prog=prog,
                    step_id=step_id,
                    step_name=step_name,
                    snapshot_dir=snapshot_dir,
                    strict_cfg=strict_cfg,
                    corpus_lookup=corpus_lookup,
                    engine_version=engine_version,
                    azure_endpoint=azure_endpoint,
                    model_name=model_name,
                    prompt_template=prompt_template,
                    parse_inspect_dir=parse_inspect_dir,
                    inspect_dir=inspect_dir,
                    mbench_dir=mbench_dir,
                    after_dir=after_dir,
                )
                stage_cost.add(cost_usd)
                counts.update(c_counts)
                reasons.update(c_reasons)
                if stage_cost.done % max(1, int(progress_every)) == 0 or stage_cost.done == total:
                    _print_running_cost_line()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        process_program,
                        prog=prog,
                        step_id=step_id,
                        step_name=step_name,
                        snapshot_dir=snapshot_dir,
                        strict_cfg=strict_cfg,
                        corpus_lookup=corpus_lookup,
                        engine_version=engine_version,
                        azure_endpoint=azure_endpoint,
                        model_name=model_name,
                        prompt_template=prompt_template,
                        parse_inspect_dir=parse_inspect_dir,
                        inspect_dir=inspect_dir,
                        mbench_dir=mbench_dir,
                        after_dir=after_dir,
                    )
                    for prog in progs
                ]
                for fut in as_completed(futures):
                    cost_usd, c_counts, c_reasons = fut.result()
                    stage_cost.add(cost_usd)
                    counts.update(c_counts)
                    reasons.update(c_reasons)
                    if stage_cost.done % max(1, int(progress_every)) == 0 or stage_cost.done == total:
                        _print_running_cost_line()

    except KeyboardInterrupt:
        interrupted = True
        print("\n[CTRL-C] interrupt received; finalizing step snapshots...", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            after_n = _snapshot_dir_all_smt2(src_canonical_dir=canonical_dir, dst_dir=after_dir)
            print(f"\n[SNAPSHOT] after:  {after_dir} (files={after_n})", flush=True)
        except Exception as e:
            print(f"[WARN] failed to write after snapshot: {e}", flush=True)

    if stage_cost.done > 0:
        _print_running_cost_line()
        print("", flush=True)

    print(f"[SUMMARY {step_name}] {dict(counts)}", flush=True)

    if counts.get("invalid_strict", 0) or counts.get("error", 0):
        print(f"[TOP REASONS {step_name}]")
        for k, v in reasons.most_common(15):
            print(f"  {k}: {v}")
        print(f"[INSPECT] artifacts under: {inspect_dir}", flush=True)

    print(f"[PARSE_INSPECT] artifacts under: {parse_inspect_dir}", flush=True)
    print(f"[MBENCH] artifacts under: {mbench_dir}", flush=True)

    t1 = dt.datetime.now(dt.timezone.utc).isoformat()

    stage_pred_final = stage_cost.predicted_final_cost(total)
    stage_pred_remaining = stage_cost.predicted_remaining_cost(total)
    pipeline_now = pipeline_total_cost_usd_so_far + stage_cost.total_cost_usd

    pipeline_pred_total = None
    pipeline_pred_remaining = None
    if stage_pred_final is not None:
        pipeline_pred_total = pipeline_total_cost_usd_so_far + stage_pred_final
        pipeline_pred_remaining = max(0.0, pipeline_pred_total - pipeline_now)

    summary = {
        "step_id": step_id,
        "step_name": step_name,
        "run_id": run_id,
        "time_utc_start": None,
        "time_utc_end": t1,
        "interrupted": interrupted,
        "canonical_dir": str(canonical_dir),
        "snapshot_before_dir": str(before_dir),
        "snapshot_after_dir": str(after_dir),
        "inspect_dir": str(inspect_dir),
        "parse_inspect_dir": str(parse_inspect_dir),
        "mbench_dir": str(mbench_dir),
        "filters": {
            "trial_prefix": trial_prefix,
            "allowlist_prefixes_count": len(allow_prefixes),
            "side_filter": side_filter,
        },
        "corpus_fallback": {
            "corpus_jsonl": str(corpus_jsonl) if corpus_jsonl else None,
            "wanted_base_ids": len(wanted_base),
            "loaded_rows": len(corpus_lookup),
        },
        "engine": {
            "engine_version": engine_version,
            "model_name": model_name,
            "openai_endpoint": os.environ.get("OPENAI_ENDPOINT", ""),
            "openai_model_env": os.environ.get("OPENAI_MODEL", ""),
        },
        "strict": {
            "check_asserts_and_named": strict_cfg.check_asserts_and_named,
            "strict_double_semicolon": strict_cfg.strict_double_semicolon,
            "allow_any_sort": strict_cfg.allow_any_sort,
            "decl_json_strict": strict_cfg.decl_json_strict,
            "allow_extra_decl_json_keys": strict_cfg.allow_extra_decl_json_keys,
        },
        "counts": dict(counts),
        "top_reasons": [{"reason": k, "count": v} for k, v in reasons.most_common(30)],
        "cost": {
            "stage_total_cost_usd": stage_cost.total_cost_usd,
            "stage_with_cost": stage_cost.with_cost,
            "stage_unknown_cost_lines": stage_cost.done - stage_cost.with_cost,
            "stage_predicted_final_cost_usd": stage_pred_final,
            "stage_predicted_remaining_cost_usd": stage_pred_remaining,
            "pipeline_total_cost_usd_so_far": pipeline_now,
            "pipeline_predicted_final_cost_usd_known_steps": pipeline_pred_total,
            "pipeline_predicted_remaining_cost_usd_known_steps": pipeline_pred_remaining,
            "by_stage_total_cost_usd_so_far": {
                **{k: v.total_cost_usd for k, v in pipeline_costs_so_far.items()},
                step_name: stage_cost.total_cost_usd,
            },
        },
    }
    (step_run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SUMMARY] wrote: {step_run_dir / 'summary.json'}", flush=True)
    print(
        f"[COST {step_name}] stage_total={_fmt_money(stage_cost.total_cost_usd)} "
        f"pipeline_total={_fmt_money(pipeline_total_cost_usd_so_far + stage_cost.total_cost_usd)}",
        flush=True,
    )

    return stage_cost


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simplified orchestrator (single-folder canonical, per-step step_runs). PARALLEL THREADPOOL.")
    p.add_argument("--ir-dir", default="../build/ir", help="Input IR dir (source). Used only to initialize canonical if needed.")
    p.add_argument("--snapshot-dir", default="../subcohort_results", help="Directory with subcohort snapshots.")
    p.add_argument("--out-root-dir", default="../build/ir_under_repair", help="Output root. Writes canonical/ and step_runs/ here.")
    p.add_argument("--corpus-jsonl", default="../dataset/clinical_trial/sigir/corpus.jsonl", help="Optional SIGIR corpus.jsonl path for criteria fallback + raw corpus block for gate.")

    p.add_argument("--reset-out-dir", action="store_true", help="Delete and re-initialize out-root-dir before running.")
    p.add_argument("--start-step", type=int, choices=[0, 1, 2, 3, 4, 5], default=5)
    p.add_argument("--stop-step", type=int, choices=[0, 1, 2, 3, 4, 5], default=5)

    p.add_argument("--trial-prefix", default=None, help="Restrict to effective trial IDs starting with this prefix.")
    p.add_argument("--trial-allowlist", default=None, help="Path to allowlist (one NCT prefix per line).")

    p.add_argument("--max-workers", type=int, default=48, help="Max concurrent workers per step (ThreadPool). 1 => sequential.")
    p.add_argument("--max-inflight", type=int, default=0, help="Reserved; currently unused.")

    p.add_argument("--progress-every", type=int, default=50, help="Update running cost line every N completed files per step.")

    p.add_argument(
        "--skip-assert-checks",
        dest="skip_assert_checks",
        action="store_true",
        default=True,
        help="Disable assert/:named checks in strict validation (default: ON, i.e. checks are skipped).",
    )
    p.add_argument(
        "--no-skip-assert-checks",
        dest="skip_assert_checks",
        action="store_false",
        help="Re-enable assert/:named checks (turn strict assert checks back ON).",
    )

    p.add_argument("--yes", action="store_true", help="No interactive confirmations.")
    return p.parse_args()


def load_allowlist_prefixes_from_arg(arg: Optional[str]) -> Set[str]:
    if not arg:
        return set()
    path = Path(arg).resolve()
    return load_allowlist_prefixes(path)


def main() -> None:
    args = parse_args()

    in_ir_dir = Path(args.ir_dir).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out_root_dir = Path(args.out_root_dir).resolve()
    corpus_jsonl = Path(args.corpus_jsonl).resolve() if args.corpus_jsonl else None

    if not in_ir_dir.exists():
        raise FileNotFoundError(f"--ir-dir not found: {in_ir_dir}")
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"--snapshot-dir not found: {snapshot_dir}")
    if corpus_jsonl is not None and (not corpus_jsonl.exists()):
        print(f"[WARN] corpus-jsonl not found: {corpus_jsonl} (will be ignored)", flush=True)
        corpus_jsonl = None

    strict_cfg = STRICT_GUARDRAILS_DEFAULT
    if args.skip_assert_checks:
        strict_cfg = replace(strict_cfg, check_asserts_and_named=False)

    idx = {s: i for i, s in enumerate(PIPELINE_ORDER)}
    start_i = idx[args.start_step]
    stop_i = idx[args.stop_step]
    if start_i > stop_i:
        raise SystemExit(f"Invalid range: start-step={args.start_step} AFTER stop-step={args.stop_step} in order {PIPELINE_ORDER}.")
    steps_to_run = PIPELINE_ORDER[start_i : stop_i + 1]

    if args.reset_out_dir and out_root_dir.exists():
        shutil.rmtree(out_root_dir)

    canonical_dir = out_root_dir / "canonical"
    ensure_dir(canonical_dir)
    ensure_dir(out_root_dir / "step_runs")

    if len(list(canonical_dir.glob("*.smt2"))) == 0:
        n = copy_canonical_files(in_ir_dir, canonical_dir)
        print(f"[INIT] initialized canonical from input: {canonical_dir} (files={n})", flush=True)
    else:
        print(f"[INIT] using existing canonical: {canonical_dir} (files={len(list(canonical_dir.glob('*.smt2')))})", flush=True)

    allow_prefixes = load_allowlist_prefixes_from_arg(args.trial_allowlist)
    if allow_prefixes:
        print(f"[FILTER] allowlist prefixes={len(allow_prefixes)}", flush=True)
    if args.trial_prefix:
        print(f"[FILTER] trial_prefix={args.trial_prefix}", flush=True)

    print(f"[ORDER] steps_to_run={steps_to_run} (order={PIPELINE_ORDER})", flush=True)
    print(f"[STRICT] assert_checks={'OFF' if args.skip_assert_checks else 'ON'}", flush=True)
    print(f"[COST] progress_every={int(args.progress_every)}", flush=True)

    pipeline_costs: Dict[str, CostTotals] = {}
    pipeline_total_cost: float = 0.0
    max_workers = int(args.max_workers) if int(args.max_workers) > 0 else 1

    for s in steps_to_run:
        step_name = STEP_NAME[s]
        stage_cost = run_step(
            step_id=s,
            canonical_dir=canonical_dir,
            out_root_dir=out_root_dir,
            snapshot_dir=snapshot_dir,
            strict_cfg=strict_cfg,
            trial_prefix=args.trial_prefix,
            allow_prefixes=allow_prefixes,
            yes=bool(args.yes),
            corpus_jsonl=corpus_jsonl,
            pipeline_costs_so_far=dict(pipeline_costs),
            pipeline_total_cost_usd_so_far=pipeline_total_cost,
            progress_every=int(args.progress_every),
            max_workers=max_workers,
        )
        pipeline_costs[step_name] = stage_cost
        pipeline_total_cost += stage_cost.total_cost_usd

    if pipeline_costs:
        parts = []
        for sid in PIPELINE_ORDER:
            nm = STEP_NAME[sid]
            ct = pipeline_costs.get(nm)
            if ct is None:
                continue
            parts.append(f"{nm}={_fmt_money(ct.total_cost_usd)}({ct.with_cost}/{ct.done})")
        print("\n[PIPELINE_COST_SUMMARY]", " ".join(parts), f"TOTAL={_fmt_money(pipeline_total_cost)}", flush=True)

    print(f"\n[DONE] out_root_dir: {out_root_dir}", flush=True)
    print(f"[DONE] canonical:    {canonical_dir}", flush=True)
    print(f"[DONE] step_runs:    {out_root_dir / 'step_runs'}", flush=True)


if __name__ == "__main__":
    main()
