#!/usr/bin/env python3
"""
smt_presence_and_coverage_check.py

Two audits, two rerun lists:

(A) PARENT-LEVEL PRESENCE (missing SMT):
  - SMT is OR over subcohorts; SMT for (parent_id, side) is "present" if ANY canonical/*.smt2
    has base id == parent_id for that side.
  - For each missing (parent_id, side): ask LLM if that side's criteria is substantively present
    anywhere in the parent trial eligibility content (concat all subcohorts).
  - If criteria_present=true => SHOULD HAVE SMT => add to rerun_missing_parent_sides.

(B) FILE-LEVEL COVERAGE (existing SMT):
  - For each existing canonical SMT file (eff_tid, side): ask LLM if SMT plausibly encodes the
    TARGET SIDE criteria (simple conservative check).
  - If coverage_ok=false => add to rerun_bad_existing_eff_sides.

Input filling adopts the same "two-block" spirit as smt_criteria_gate_module.py:
  Block 1 (structured bundle): shared_context, subcohort_context (if available), side_criteria (side-only)
  Block 2 (raw): corpus_item_json (parent trial raw corpus JSON)
Plus Block 3 for coverage: SMT program text.

Optional snapshot-dir:
  If provided and snapshots exist, we use them to build subcohort-scoped Block 1 for coverage checks
  (matching how orchestrator maps eff_tid -> enrollment cohort). For missing parent-level presence,
  we aggregate across all cohorts in the snapshot when possible.

Env:
  Requires OPENAI_ENDPOINT and OPENAI_API_KEY
  Optional OPENAI_MODEL

Outputs under: out_root_dir/presence_coverage_check_runs/<run_id>/
  - rerun_missing_parent_sides.txt                         (final)
  - rerun_missing_parent_items.json                        (final, array)
  - rerun_bad_existing_eff_sides.txt                        (final)
  - rerun_bad_existing_eff_items.json                       (final, array)
  - needs_review_items.json                                 (final, array)
  - summary.json                                            (final run summary)
  - summary_running.json                                    (incremental checkpoint; overwritten periodically)
  - rerun_missing_parent_items.jsonl                         (incremental append)
  - rerun_bad_existing_eff_items.jsonl                       (incremental append)
  - needs_review_items.jsonl                                 (incremental append)
  - rerun_missing_parent_sides_incremental.txt               (incremental append)
  - rerun_bad_existing_eff_sides_incremental.txt             (incremental append)
  - mbench/<key>/call_001_prompt.txt, call_001_raw.txt, events.jsonl, final_meta.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# -------------------------
# Optional costing
# -------------------------
try:
    from trial_compiler.ir_finalizer.utils.costing import OPENAI_PRICING, count_tokens, estimate_cost_usd  # type: ignore
except Exception:
    OPENAI_PRICING = {}  # type: ignore

    def count_tokens(text: str, model: str):  # type: ignore
        return None, "costing_py_missing"

    def estimate_cost_usd(  # type: ignore
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,
    ):
        return None


# -------------------------
# Canonical filename patterns
# -------------------------
CANON_LAX_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program.*\.smt2$", re.IGNORECASE)
BASE_NCT_RE = re.compile(r"^(NCT[0-9]+)", re.IGNORECASE)


def base_trial_id(eff_tid: str) -> str:
    m = BASE_NCT_RE.match(eff_tid or "")
    return m.group(1).upper() if m else (eff_tid or "").upper()


# -------------------------
# IO helpers
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _append_line(p: Path, line: str) -> None:
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "(n/a)"
    return f"${x:.6f}"


def _truncate(s: str, max_chars: int) -> Tuple[str, bool]:
    s = s or ""
    if max_chars <= 0 or len(s) <= max_chars:
        return s, False
    return s[:max_chars] + "\n...[truncated]...", True


def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


# -------------------------
# Corpus reading (jsonl OR json)
# -------------------------
def _iter_corpus_rows_any(corpus_path: Path) -> Iterable[Dict[str, Any]]:
    suf = corpus_path.suffix.lower()
    if suf == ".jsonl":
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    # .json: list/dict containers
    try:
        obj = json.loads(corpus_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
        return

    if isinstance(obj, dict):
        for k in ("trials", "data", "rows", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it
                return
        if "_id" in obj or "trial_id" in obj or "nct_id" in obj:
            yield obj
        return


def _get_trial_id_from_row(row: Dict[str, Any]) -> Optional[str]:
    tid = row.get("_id") or row.get("trial_id") or row.get("nct_id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return None


def _jsonish(x: Any) -> Tuple[str, int]:
    if x is None:
        return "", 0
    if isinstance(x, str):
        return x, len(x)
    try:
        s = json.dumps(x, ensure_ascii=False)
        return s, len(s)
    except Exception:
        s = str(x)
        return s, len(s)


def _get_corpus_inc_exc_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    inc = _safe_str(md.get("inclusion_criteria")).strip() if isinstance(md, dict) else ""
    exc = _safe_str(md.get("exclusion_criteria")).strip() if isinstance(md, dict) else ""
    return inc, exc


def _get_corpus_shared_context_from_row(row: Dict[str, Any]) -> str:
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    txt = _safe_str(row.get("text")).strip()
    if txt:
        return txt
    for k in ("brief_summary", "detailed_description", "description"):
        v = _safe_str(md.get(k)).strip() if isinstance(md, dict) else ""
        if v:
            return v
    title = _safe_str(row.get("title")).strip()
    return title


def build_corpus_lookup(corpus_jsonl: Path, wanted_base_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if (not corpus_jsonl.exists()) or (not wanted_base_ids):
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
                if not isinstance(obj, dict):
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


# -------------------------
# Filters
# -------------------------
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


def trial_selected(trial_id: str, *, trial_prefix: Optional[str], allow_prefixes: Set[str]) -> bool:
    s = (trial_id or "").upper()
    if trial_prefix and (not s.startswith(trial_prefix.upper())):
        return False
    if allow_prefixes and (not any(s.startswith(p.upper()) for p in allow_prefixes)):
        return False
    return True


# -------------------------
# Discover SMT presence at parent-level
# -------------------------
def discover_parent_side_presence(canonical_dir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    present: Dict[str, Set[str]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for p in canonical_dir.glob("*.smt2"):
        m = CANON_LAX_RE.match(p.name)
        if not m:
            continue
        eff_tid = m.group(1)
        side = m.group(2).lower()
        base = base_trial_id(eff_tid)
        present.setdefault(base, set()).add(side)
        counts.setdefault(base, {}).setdefault(side, 0)
        counts[base][side] += 1

    return present, counts


@dataclass(frozen=True)
class IRProgram:
    eff_tid: str
    side: str
    path: Path


def discover_ir_programs(canonical_dir: Path) -> List[IRProgram]:
    candidates: List[IRProgram] = []
    for p in canonical_dir.glob("*.smt2"):
        m = CANON_LAX_RE.match(p.name)
        if not m:
            continue
        eff, side = m.group(1).upper(), m.group(2).lower()
        candidates.append(IRProgram(eff_tid=eff, side=side, path=p))

    # de-dupe by (eff_tid, side), prefer shortest filename
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


# -------------------------
# Snapshot loading + cohort mapping (subset of orchestrator logic)
# -------------------------
def _candidate_parent_ids(eff_tid: str) -> List[str]:
    base = base_trial_id(eff_tid)
    if eff_tid.upper() == base:
        return [base]
    return [eff_tid.upper(), base]


def load_snapshot_for_program(snapshot_dir: Optional[Path], eff_tid: str, side: str) -> Optional[Dict[str, Any]]:
    if snapshot_dir is None:
        return None
    if not snapshot_dir.exists():
        return None
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
    pn = parent_ctx.get("preprocessor_normalized") or {}
    enroll = pn.get("enrollment_cohorts") or []
    if not isinstance(enroll, list) or not enroll:
        return None, "none", None

    # prefer explicit mapping
    for i, co in enumerate(enroll):
        if not isinstance(co, dict):
            continue
        if co.get("trial_id_effective") == eff_tid or co.get("trial_id") == eff_tid:
            return co, "trial_id_effective", i

    # fallback positional mapping
    eff_ids = parent_ctx.get("effective_trial_ids") or []
    if isinstance(eff_ids, list) and str(eff_tid) in set(map(str, eff_ids)) and len(eff_ids) == len(enroll):
        if len(set(map(str, eff_ids))) != len(eff_ids):
            return None, "none", None
        idx = list(map(str, eff_ids)).index(str(eff_tid))
        if 0 <= idx < len(enroll) and isinstance(enroll[idx], dict):
            return enroll[idx], "index_effective_ids", idx

    return None, "none", None


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


def _get_parent_tid_from_snapshot(parent_ctx: Dict[str, Any], eff_tid: str) -> str:
    parent_tid = (
        parent_ctx.get("parent_trial_id")
        or parent_ctx.get("trial_id_parent")
        or parent_ctx.get("trial_id")
        or base_trial_id(eff_tid)
    )
    return str(parent_tid).upper()


def _aggregate_parent_level_from_snapshot(parent_ctx: Dict[str, Any], side: str) -> Tuple[str, str, str]:
    """
    For presence check (parent-level OR over cohorts), aggregate across enrollment_cohorts if present.
    Returns (shared_context, subcohort_context_agg, side_criteria_agg)
    """
    pn = parent_ctx.get("preprocessor_normalized") or {}
    shared_context = _safe_str(parent_ctx.get("shared_context")) or _safe_str(pn.get("shared_context")) or ""

    enroll = pn.get("enrollment_cohorts") or []
    side_parts: List[str] = []
    ctx_parts: List[str] = []

    if isinstance(enroll, list) and enroll:
        for co in enroll:
            if not isinstance(co, dict):
                continue
            cid = co.get("id") or co.get("cohort_id") or co.get("substudy_id") or "C?"
            label = co.get("label") or co.get("cohort_name") or co.get("cohort_label") or str(cid)

            inc = _safe_str(co.get("inclusion_criteria")).strip()
            exc = _safe_str(co.get("exclusion_criteria")).strip()
            crit = inc if side == "inclusion" else exc
            if crit:
                side_parts.append(f"[COHORT {cid} | {label}]\n{crit}")

            ctxt = _safe_str(co.get("contextual_text")).strip() or _safe_str(co.get("context")).strip()
            if ctxt:
                ctx_parts.append(f"[COHORT {cid} | {label}]\n{ctxt}")

    side_criteria = "\n\n".join(side_parts).strip()
    subcohort_context = "\n\n".join(ctx_parts).strip()

    # fallback: parse shared_context for shared inc/exc if cohort-level empty
    if (not side_criteria.strip()) and shared_context:
        inc2, exc2 = extract_shared_inc_exc(shared_context)
        side_criteria = inc2 if side == "inclusion" else exc2

    return shared_context.strip(), subcohort_context.strip(), side_criteria.strip()


def _build_subctx_like_gate(
    *,
    eff_tid: str,
    side: str,
    parent_ctx: Optional[Dict[str, Any]],
    corpus_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a minimal dict containing fields the two-block prompt expects.
    For coverage check we prefer cohort-scoped mapping from snapshot when available;
    otherwise fallback to corpus-only parent-level.
    """
    eff_tid_u = eff_tid.upper()
    parent_tid = base_trial_id(eff_tid_u)

    corpus_item = corpus_lookup.get(parent_tid)

    # corpus fallbacks
    corpus_shared = _get_corpus_shared_context_from_row(corpus_item) if corpus_item else ""
    inc_c, exc_c = _get_corpus_inc_exc_from_row(corpus_item) if corpus_item else ("", "")
    side_crit_c = inc_c if side == "inclusion" else exc_c

    if parent_ctx is None:
        return {
            "trial_id_parent": parent_tid,
            "trial_id_effective": eff_tid_u,
            "cohort_id": "unknown",
            "cohort_label": "Unknown cohort (no snapshot)",
            "shared_context": corpus_shared,
            "subcohort_context": "",
            "side_criteria": side_crit_c,
            "corpus_item": corpus_item,
            "cohort_match_method": "none_no_snapshot",
            "cohort_match_index": None,
        }

    # snapshot path: map eff_tid -> cohort record
    cohort_record, method, idx = _select_enrollment_cohort_for_eff_tid(parent_ctx, eff_tid_u)
    pn = parent_ctx.get("preprocessor_normalized") or {}

    # shared context: prefer corpus, else snapshot
    snapshot_shared = _safe_str(parent_ctx.get("shared_context")) or _safe_str(pn.get("shared_context")) or ""
    shared_context = (corpus_shared.strip() or snapshot_shared.strip())

    if cohort_record is not None:
        cid = cohort_record.get("id") or cohort_record.get("cohort_id") or cohort_record.get("substudy_id") or "C?"
        label = cohort_record.get("label") or cohort_record.get("cohort_name") or cohort_record.get("cohort_label") or str(cid)
        inc = _safe_str(cohort_record.get("inclusion_criteria")).strip()
        exc = _safe_str(cohort_record.get("exclusion_criteria")).strip()
        side_criteria = inc if side == "inclusion" else exc

        subcohort_context = _safe_str(cohort_record.get("contextual_text")).strip()
        if not subcohort_context:
            subcohort_context = _safe_str(cohort_record.get("context")).strip()

        return {
            "trial_id_parent": _get_parent_tid_from_snapshot(parent_ctx, eff_tid_u),
            "trial_id_effective": eff_tid_u,
            "cohort_id": str(cid),
            "cohort_label": str(label),
            "shared_context": shared_context,
            "subcohort_context": subcohort_context,
            "side_criteria": side_criteria,
            "corpus_item": corpus_item,
            "cohort_match_method": method,
            "cohort_match_index": idx,
        }

    # fallback to parent-level criteria (snapshot shared + corpus side)
    return {
        "trial_id_parent": _get_parent_tid_from_snapshot(parent_ctx, eff_tid_u),
        "trial_id_effective": eff_tid_u,
        "cohort_id": "default",
        "cohort_label": "Default cohort (no cohort match)",
        "shared_context": shared_context,
        "subcohort_context": "",
        "side_criteria": side_crit_c,
        "corpus_item": corpus_item,
        "cohort_match_method": "none_default",
        "cohort_match_index": None,
    }


# -------------------------
# Engine resolution
# -------------------------
def resolve_engine_from_env() -> Tuple[str, str, str]:
    azure_endpoint = os.environ.get("OPENAI_ENDPOINT", "").strip()
    if not azure_endpoint:
        raise EnvironmentError("OPENAI_ENDPOINT is required.")

    from smt_core.engine_factory import detect_engine_and_model
    engine_version, model_name = detect_engine_and_model()
    return engine_version, model_name, azure_endpoint


_THREAD_LOCAL = threading.local()


def _get_thread_engine(engine_version: str, azure_endpoint: str, model_name: str):
    eng = getattr(_THREAD_LOCAL, "engine", None)
    if eng is not None:
        return eng

    if engine_version == "gpt-5":
        from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
    else:
        from smt_core.inference_engine import AzureInferenceEngine  # type: ignore

    eng = AzureInferenceEngine(
        endpoint=azure_endpoint,
        api_key_env_var="OPENAI_API_KEY",
        model_name=model_name,
    )
    _THREAD_LOCAL.engine = eng
    return eng


# -------------------------
# MBENCH logging
# -------------------------
def _mbench_prog_dir(mbench_dir: Path, key: str) -> Path:
    d = mbench_dir / key
    ensure_dir(d)
    return d


def _write_mbench_call(
    *,
    mbench_dir: Path,
    key: str,
    step_name: str,
    attempt: int,
    prompt: str,
    raw: str,
    model_name: str,
    azure_endpoint: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    d = _mbench_prog_dir(mbench_dir, key)
    a = f"{attempt:03d}"
    (d / f"call_{a}_prompt.txt").write_text(prompt or "", encoding="utf-8")
    (d / f"call_{a}_raw.txt").write_text(raw or "", encoding="utf-8")

    evt: Dict[str, Any] = {
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "llm_call",
        "step_name": step_name,
        "key": key,
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
    key: str,
    step_name: str,
    meta: Dict[str, Any],
    note: str,
) -> None:
    d = _mbench_prog_dir(mbench_dir, key)
    out: Dict[str, Any] = dict(meta or {})
    out.update(
        {
            "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kind": "final_meta",
            "step_name": step_name,
            "key": key,
            "note": note,
        }
    )
    # NOTE: overwrite is intentional for per-task final_meta.json
    (d / "final_meta.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -------------------------
# Robust JSON parsing
# -------------------------
def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:] if lines else []
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        return "\n".join(lines).strip()
    return t


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = _strip_code_fences(raw or "")
    if not text:
        raise ValueError("empty LLM output")
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("parsed JSON is not an object")
        return obj
    except Exception:
        pass

    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        snippet = text[l : r + 1]
        obj2 = json.loads(snippet)
        if not isinstance(obj2, dict):
            raise ValueError("parsed JSON is not an object")
        return obj2

    raise ValueError("could not parse JSON from LLM output")


# -------------------------
# Prompts (default if files not provided)
# -------------------------
DEFAULT_PRESENCE_PROMPT = r"""
You are a conservative gatekeeper for a clinical-trial eligibility pipeline.

We are checking NL PRESENCE for a PARENT trial ID. This trial may have multiple subcohorts/arms.
Policy: treat this parent's eligibility text as the concatenation/aggregation across all subcohorts.
SMT is an OR over subcohorts; SMT for this SIDE is missing if there are NO SMT files at all for this parent+side.

TASK:
Decide whether the TARGET SIDE criteria is substantively present anywhere in the trial eligibility content.

Conservative policy:
- Mark criteria_present=false ONLY when you are very sure the TARGET SIDE criteria is missing/placeholder-only/boilerplate.
- If uncertain, choose criteria_present=true.

Definition (criteria_present):
- true  => ANY real eligibility content for the TARGET SIDE exists (even one condition like "Age >= 18").
- false => the TARGET SIDE criteria is effectively absent.

Return ONLY JSON (no markdown, no extra text):
{"criteria_present": true|false, "reason": "short"}

TRIAL_ID={{TRIAL_ID}}
TARGET_SIDE={{SIDE}}

# === BLOCK 1: STRUCTURED BUNDLE (best-effort) ===
<shared_context>
{{SHARED_CONTEXT}}
</shared_context>

<all_subcohorts_context>
{{SUBCOHORT_CONTEXT}}
</all_subcohorts_context>

<side_criteria>
{{SIDE_CRITERIA}}
</side_criteria>

# === BLOCK 2: CORPUS ITEM (raw, unprocessed; all subcohorts) ===
<corpus_item_json>
{{CORPUS_ITEM_JSON}}
</corpus_item_json>
""".strip()


DEFAULT_COVERAGE_PROMPT = r"""
You are a conservative auditor for a clinical-trial SMT eligibility pipeline.

We have an EXISTING SMT program for this (EFFECTIVE_TRIAL_ID, TARGET_SIDE).
Decide whether the SMT program SUBSTANTIVELY encodes the TARGET SIDE criteria for the same trial/cohort.

Conservative policy:
- Mark coverage_ok=false ONLY when you are very sure the SMT does NOT encode the TARGET SIDE criteria
  (e.g., wrong side, placeholder/tautology, empty/unrelated).
- If uncertain, choose coverage_ok=true.

Definition (coverage_ok):
- true  => SMT contains meaningful constraints plausibly corresponding to at least some TARGET SIDE criteria.
- false => SMT clearly fails to encode the TARGET SIDE criteria (not just incomplete).

Return ONLY JSON (no markdown, no extra text):
{"coverage_ok": true|false, "reason": "short"}

TRIAL_ID={{TRIAL_ID}}
EFFECTIVE_TRIAL_ID={{EFFECTIVE_TRIAL_ID}}
TARGET_SIDE={{SIDE}}

# === BLOCK 1: CONTEXT BUNDLE (best-effort) ===
<shared_context>
{{SHARED_CONTEXT}}
</shared_context>

<subcohort_context>
{{SUBCOHORT_CONTEXT}}
</subcohort_context>

<side_criteria>
{{SIDE_CRITERIA}}
</side_criteria>

# === BLOCK 2: CORPUS ITEM (raw, unprocessed; all subcohorts) ===
<corpus_item_json>
{{CORPUS_ITEM_JSON}}
</corpus_item_json>

# === BLOCK 3: SMT PROGRAM (canonical) ===
<smt_program>
{{SMT_TEXT}}
</smt_program>
""".strip()


# -------------------------
# Gates
# -------------------------
@dataclass(frozen=True)
class ParentPresenceTask:
    parent_tid: str
    side: str  # inclusion/exclusion
    shared_context: str
    subcohort_context: str  # aggregated if available
    side_criteria: str
    corpus_item_json: str


@dataclass(frozen=True)
class CoverageTask:
    eff_tid: str
    parent_tid: str
    side: str
    cohort_id: str
    cohort_label: str
    shared_context: str
    subcohort_context: str
    side_criteria: str
    corpus_item_json: str
    smt_text: str
    smt_path: str
    cohort_match_method: str
    cohort_match_index: Optional[int]


class _BaseGate:
    def __init__(self, *, call_llm, prompt_template: str, model_name: str):
        self.call_llm = call_llm
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

    def _pricing_used(self) -> Optional[Dict[str, Any]]:
        pr = OPENAI_PRICING.get(self.model_name)
        if pr is None:
            return None
        return {
            "input_per_1m": pr.input_per_1m,
            "output_per_1m": pr.output_per_1m,
            "cached_input_per_1m": pr.cached_input_per_1m,
            "source": "https://platform.openai.com/docs/pricing",
        }

    def _cost_meta(self) -> Dict[str, Any]:
        pt, pnote = count_tokens(self.last_prompt or "", self.model_name)
        ct, cnote = count_tokens(self.last_raw or "", self.model_name)
        est_cost = None
        if pt is not None and ct is not None:
            est_cost = estimate_cost_usd(self.model_name, int(pt), int(ct))
        return {
            "prompt_tokens_est": pt,
            "completion_tokens_est": ct,
            "tokenizer_note_prompt": str(pnote),
            "tokenizer_note_completion": str(cnote),
            "estimated_cost_usd": est_cost,
            "pricing_used": self._pricing_used(),
        }


class ParentPresenceGate(_BaseGate):
    def run(self, task: ParentPresenceTask) -> Dict[str, Any]:
        self.last_prompt = None
        self.last_raw = None

        t = self.prompt_template
        repl = {
            "{{TRIAL_ID}}": task.parent_tid,
            "{{EFFECTIVE_TRIAL_ID}}": task.parent_tid,
            "{{SIDE}}": task.side,
            "{{TARGET_SIDE}}": task.side,
            "{{SHARED_CONTEXT}}": task.shared_context or "",
            "{{SUBCOHORT_CONTEXT}}": task.subcohort_context or "",
            "{{SIDE_CRITERIA}}": task.side_criteria or "",
            "{{CORPUS_ITEM_JSON}}": task.corpus_item_json or "",
            # legacy placeholder support if someone kept old prompt shape
            "{{CONTEXTUAL_TEXT}}": "\n\n".join(
                [
                    f"TRIAL_ID={task.parent_tid}",
                    "\n=== SHARED_CONTEXT ===\n" + (task.shared_context or ""),
                    "\n=== ALL_SUBCOHORTS_CONTEXT ===\n" + (task.subcohort_context or ""),
                    "\n=== SIDE_CRITERIA ===\n" + (task.side_criteria or ""),
                    "\n=== CORPUS_ITEM_JSON ===\n" + (task.corpus_item_json or ""),
                ]
            ),
        }
        for k, v in repl.items():
            t = t.replace(k, v)

        self.last_prompt = t
        raw = self.call_llm(t)
        self.last_raw = raw

        obj = _parse_json_object(raw)
        criteria_present = obj.get("criteria_present")
        reason = obj.get("reason", "")

        if not isinstance(criteria_present, bool):
            return {
                "executed": False,
                "error": "parsed_json_missing_or_nonbool:criteria_present",
                "model": self.model_name,
                "pricing_used": self._pricing_used(),
            }

        if not isinstance(reason, str):
            reason = ""

        out = {
            "executed": True,
            "criteria_present": bool(criteria_present),
            "reason": reason.strip(),
            "model": self.model_name,
            "parent_tid": task.parent_tid,
            "side": task.side,
        }
        out.update(self._cost_meta())
        return out


class CoverageGate(_BaseGate):
    def run(self, task: CoverageTask) -> Dict[str, Any]:
        self.last_prompt = None
        self.last_raw = None

        t = self.prompt_template
        repl = {
            "{{TRIAL_ID}}": task.parent_tid,
            "{{EFFECTIVE_TRIAL_ID}}": task.eff_tid,
            "{{SIDE}}": task.side,
            "{{SHARED_CONTEXT}}": task.shared_context or "",
            "{{SUBCOHORT_CONTEXT}}": task.subcohort_context or "",
            "{{SIDE_CRITERIA}}": task.side_criteria or "",
            "{{CORPUS_ITEM_JSON}}": task.corpus_item_json or "",
            "{{SMT_TEXT}}": task.smt_text or "",
        }
        for k, v in repl.items():
            t = t.replace(k, v)

        self.last_prompt = t
        raw = self.call_llm(t)
        self.last_raw = raw

        obj = _parse_json_object(raw)
        coverage_ok = obj.get("coverage_ok")
        reason = obj.get("reason", "")

        if not isinstance(coverage_ok, bool):
            return {
                "executed": False,
                "error": "parsed_json_missing_or_nonbool:coverage_ok",
                "model": self.model_name,
                "pricing_used": self._pricing_used(),
            }

        if not isinstance(reason, str):
            reason = ""

        out = {
            "executed": True,
            "coverage_ok": bool(coverage_ok),
            "reason": reason.strip(),
            "model": self.model_name,
            "parent_tid": task.parent_tid,
            "eff_tid": task.eff_tid,
            "side": task.side,
            "smt_path": task.smt_path,
            "cohort_id": task.cohort_id,
            "cohort_label": task.cohort_label,
            "cohort_match_method": task.cohort_match_method,
            "cohort_match_index": task.cohort_match_index,
        }
        out.update(self._cost_meta())
        return out


# -------------------------
# Build inputs for tasks
# -------------------------
def build_parent_presence_task(
    *,
    parent_tid: str,
    side: str,
    corpus_row: Dict[str, Any],
    corpus_lookup: Dict[str, Dict[str, Any]],
    snapshot_dir: Optional[Path],
    max_shared_chars: int,
    max_subcohort_chars: int,
    max_side_criteria_chars: int,
    max_corpus_item_chars: int,
) -> ParentPresenceTask:
    parent_tid_u = parent_tid.upper()
    corpus_item = corpus_row or corpus_lookup.get(parent_tid_u) or corpus_lookup.get(base_trial_id(parent_tid_u))

    corpus_shared = _get_corpus_shared_context_from_row(corpus_item) if corpus_item else ""
    inc_c, exc_c = _get_corpus_inc_exc_from_row(corpus_item) if corpus_item else ("", "")
    side_crit_c = inc_c if side == "inclusion" else exc_c

    snap = load_snapshot_for_program(snapshot_dir, parent_tid_u, side)
    if snap is not None:
        snap_shared, snap_subctx, snap_sidecrit = _aggregate_parent_level_from_snapshot(snap, side)
    else:
        snap_shared, snap_subctx, snap_sidecrit = ("", "", "")

    # prefer corpus shared_context first (matches orchestrator philosophy), then snapshot
    shared_context = (corpus_shared.strip() or snap_shared.strip())
    subcohort_context = snap_subctx.strip()
    side_criteria = (snap_sidecrit.strip() or side_crit_c.strip())

    corpus_item_json, _ = _jsonish(corpus_item)

    shared_context, _ = _truncate(shared_context, max_shared_chars)
    subcohort_context, _ = _truncate(subcohort_context, max_subcohort_chars)
    side_criteria, _ = _truncate(side_criteria, max_side_criteria_chars)
    if max_corpus_item_chars > 0 and len(corpus_item_json) > max_corpus_item_chars:
        corpus_item_json = corpus_item_json[:max_corpus_item_chars]

    return ParentPresenceTask(
        parent_tid=parent_tid_u,
        side=side,
        shared_context=shared_context,
        subcohort_context=subcohort_context,
        side_criteria=side_criteria,
        corpus_item_json=corpus_item_json,
    )


def build_coverage_task(
    *,
    prog: IRProgram,
    corpus_lookup: Dict[str, Dict[str, Any]],
    snapshot_dir: Optional[Path],
    max_shared_chars: int,
    max_subcohort_chars: int,
    max_side_criteria_chars: int,
    max_corpus_item_chars: int,
    max_smt_chars: int,
) -> CoverageTask:
    eff = prog.eff_tid.upper()
    side = prog.side
    parent_tid = base_trial_id(eff)

    snap = load_snapshot_for_program(snapshot_dir, eff, side)
    subctx = _build_subctx_like_gate(eff_tid=eff, side=side, parent_ctx=snap, corpus_lookup=corpus_lookup)

    shared = _safe_str(subctx.get("shared_context")).strip()
    subcohort_ctx = _safe_str(subctx.get("subcohort_context")).strip()
    side_criteria = _safe_str(subctx.get("side_criteria")).strip()
    corpus_item = subctx.get("corpus_item")
    corpus_item_json, _ = _jsonish(corpus_item)

    shared, _ = _truncate(shared, max_shared_chars)
    subcohort_ctx, _ = _truncate(subcohort_ctx, max_subcohort_chars)
    side_criteria, _ = _truncate(side_criteria, max_side_criteria_chars)
    if max_corpus_item_chars > 0 and len(corpus_item_json) > max_corpus_item_chars:
        corpus_item_json = corpus_item_json[:max_corpus_item_chars]

    # SMT text (cap)
    try:
        smt_raw = prog.path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        smt_raw = ""
    smt_text, _ = _truncate(smt_raw, max_smt_chars)

    cohort_id = str(subctx.get("cohort_id") or "unknown")
    cohort_label = str(subctx.get("cohort_label") or "Unknown cohort")
    cohort_match_method = str(subctx.get("cohort_match_method") or "none")
    cohort_match_index = subctx.get("cohort_match_index")
    if not isinstance(cohort_match_index, int):
        cohort_match_index = None

    return CoverageTask(
        eff_tid=eff,
        parent_tid=str(subctx.get("trial_id_parent") or parent_tid).upper(),
        side=side,
        cohort_id=cohort_id,
        cohort_label=cohort_label,
        shared_context=shared,
        subcohort_context=subcohort_ctx,
        side_criteria=side_criteria,
        corpus_item_json=corpus_item_json,
        smt_text=smt_text,
        smt_path=str(prog.path),
        cohort_match_method=cohort_match_method,
        cohort_match_index=cohort_match_index,
    )


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Audit SMT presence (missing) and coverage (existing) with consistent two-block inputs."
    )
    ap.add_argument(
        "--canonical-dir",
        default=os.getenv("SATIR_BUILD", "build") + "/ir_under_repair/canonical",
        help="canonical/ directory containing *.smt2",
    )
    ap.add_argument(
        "--corpus",
        default=os.getenv("TRIAL_DATA", "dataset/clinical_trial") + "/sigir/corpus.jsonl",
        help="corpus.jsonl (preferred) or corpus.json; keyed by _id=NCT########",
    )
    ap.add_argument(
        "--snapshot-dir",
        default=None,
        help="Optional snapshot dir (subcohort_results). If provided, improves coverage check scoping.",
    )
    ap.add_argument(
        "--out-root-dir",
        default="./to_be_compiled/",
        help="Output root dir; writes presence_coverage_check_runs/<run_id>/ here.",
    )

    ap.add_argument(
        "--presence-prompt-file",
        default="./prompt/smt_presence_check.prompt",
        help="Presence prompt template file (optional).",
    )
    ap.add_argument(
        "--coverage-prompt-file",
        default="./prompt/smt_coverage_check.prompt",
        help="Coverage prompt template file (optional).",
    )

    ap.add_argument("--trial-prefix", default=None, help="Restrict to trial IDs starting with this prefix.")
    ap.add_argument("--trial-allowlist", default=None, help="Allowlist file: one NCT prefix per line.")

    ap.add_argument("--max-workers", type=int, default=16, help="ThreadPool workers.")
    ap.add_argument("--progress-every", type=int, default=50, help="Progress print frequency.")

    ap.add_argument("--max-shared-chars", type=int, default=8000)
    ap.add_argument("--max-subcohort-chars", type=int, default=8000)
    ap.add_argument("--max-side-criteria-chars", type=int, default=8000)
    ap.add_argument("--max-corpus-item-chars", type=int, default=24000)
    ap.add_argument("--max-smt-chars", type=int, default=24000)

    ap.add_argument("--skip-coverage", action="store_true", help="Only do missing presence audit; skip existing SMT coverage.")
    return ap.parse_args()


# -------------------------
# main
# -------------------------
def main() -> None:
    args = parse_args()

    canonical_dir = Path(args.canonical_dir).resolve()
    corpus_path = Path(args.corpus).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve() if args.snapshot_dir else None
    out_root_dir = Path(args.out_root_dir).resolve()

    if not canonical_dir.exists():
        raise FileNotFoundError(f"--canonical-dir not found: {canonical_dir}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"--corpus not found: {corpus_path}")
    if snapshot_dir is not None and (not snapshot_dir.exists()):
        print(f"[WARN] snapshot-dir not found: {snapshot_dir} (will ignore)", flush=True)
        snapshot_dir = None

    allow_prefixes = load_allowlist_prefixes(Path(args.trial_allowlist).resolve()) if args.trial_allowlist else set()
    trial_prefix = args.trial_prefix

    # Prompt files (optional; fallback to defaults)
    presence_prompt_path = Path(args.presence_prompt_file).resolve() if args.presence_prompt_file else None
    coverage_prompt_path = Path(args.coverage_prompt_file).resolve() if args.coverage_prompt_file else None

    presence_prompt = DEFAULT_PRESENCE_PROMPT
    if presence_prompt_path and presence_prompt_path.exists():
        presence_prompt = presence_prompt_path.read_text(encoding="utf-8")

    coverage_prompt = DEFAULT_COVERAGE_PROMPT
    if coverage_prompt_path and coverage_prompt_path.exists():
        coverage_prompt = coverage_prompt_path.read_text(encoding="utf-8")

    engine_version, model_name, azure_endpoint = resolve_engine_from_env()

    print(f"[ENGINE] {engine_version} (model={model_name})", flush=True)
    print(f"[CANONICAL] {canonical_dir}", flush=True)
    print(f"[CORPUS] {corpus_path}", flush=True)
    if snapshot_dir:
        print(f"[SNAPSHOT] {snapshot_dir}", flush=True)

    present_parent, present_counts = discover_parent_side_presence(canonical_dir)
    print(f"[DISCOVER] parent_ids_with_any_smt={len(present_parent)}", flush=True)

    progs_all = discover_ir_programs(canonical_dir)
    progs = [
        p for p in progs_all
        if trial_selected(p.eff_tid, trial_prefix=trial_prefix, allow_prefixes=allow_prefixes)
    ]
    print(f"[DISCOVER] programs_total={len(progs_all)} selected_for_coverage={len(progs)}", flush=True)

    # corpus lookup for coverage tasks (wanted base ids)
    wanted_base = {base_trial_id(p.eff_tid) for p in progs}
    corpus_lookup: Dict[str, Dict[str, Any]] = {}
    if corpus_path.suffix.lower() == ".jsonl" and wanted_base:
        corpus_lookup = build_corpus_lookup(corpus_path, wanted_base)
    print(f"[CORPUS_LOOKUP] wanted_base={len(wanted_base)} loaded={len(corpus_lookup)}", flush=True)

    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = out_root_dir / "presence_coverage_check_runs" / run_id
    ensure_dir(run_dir)
    mbench_dir = run_dir / "mbench"
    ensure_dir(mbench_dir)

    # -------------------------
    # Incremental aggregated outputs (NEW)
    # -------------------------
    rerun_missing_items_jsonl = run_dir / "rerun_missing_parent_items.jsonl"
    rerun_bad_existing_items_jsonl = run_dir / "rerun_bad_existing_eff_items.jsonl"
    needs_review_items_jsonl = run_dir / "needs_review_items.jsonl"

    rerun_missing_parent_sides_inc_txt = run_dir / "rerun_missing_parent_sides_incremental.txt"
    rerun_bad_existing_eff_sides_inc_txt = run_dir / "rerun_bad_existing_eff_sides_incremental.txt"

    summary_running_json = run_dir / "summary_running.json"

    max_shared = int(args.max_shared_chars)
    max_subctx = int(args.max_subcohort_chars)
    max_sidecrit = int(args.max_side_criteria_chars)
    max_corpus_item = int(args.max_corpus_item_chars)
    max_smt = int(args.max_smt_chars)

    max_workers = int(args.max_workers) if int(args.max_workers) > 0 else 1
    progress_every = max(1, int(args.progress_every))

    # -------------------------
    # Phase A: Missing parent-side presence audit
    # -------------------------
    presence_tasks: List[ParentPresenceTask] = []
    corpus_rows_seen = 0
    filtered_trials_seen = 0

    for row in _iter_corpus_rows_any(corpus_path):
        tid0 = _get_trial_id_from_row(row)
        if not tid0:
            continue
        parent_tid = base_trial_id(tid0)
        corpus_rows_seen += 1

        if not trial_selected(parent_tid, trial_prefix=trial_prefix, allow_prefixes=allow_prefixes):
            continue
        filtered_trials_seen += 1

        sides_present = present_parent.get(parent_tid, set())
        need_inc = ("inclusion" not in sides_present)
        need_exc = ("exclusion" not in sides_present)
        if not need_inc and not need_exc:
            continue

        if need_inc:
            presence_tasks.append(
                build_parent_presence_task(
                    parent_tid=parent_tid,
                    side="inclusion",
                    corpus_row=row,
                    corpus_lookup=corpus_lookup,
                    snapshot_dir=snapshot_dir,
                    max_shared_chars=max_shared,
                    max_subcohort_chars=max_subctx,
                    max_side_criteria_chars=max_sidecrit,
                    max_corpus_item_chars=max_corpus_item,
                )
            )
        if need_exc:
            presence_tasks.append(
                build_parent_presence_task(
                    parent_tid=parent_tid,
                    side="exclusion",
                    corpus_row=row,
                    corpus_lookup=corpus_lookup,
                    snapshot_dir=snapshot_dir,
                    max_shared_chars=max_shared,
                    max_subcohort_chars=max_subctx,
                    max_side_criteria_chars=max_sidecrit,
                    max_corpus_item_chars=max_corpus_item,
                )
            )

    print(f"[PLAN A] corpus_rows_seen={corpus_rows_seen} filtered_trials_seen={filtered_trials_seen}", flush=True)
    print(f"[PLAN A] missing_parent_side_candidates={len(presence_tasks)}", flush=True)

    rerun_missing_items: List[Dict[str, Any]] = []
    needs_review_items: List[Dict[str, Any]] = []

    # cost tracking
    cost_total = 0.0
    cost_with = 0
    cost_done = 0

    lock = threading.Lock()

    def _call_llm_factory(step_name: str, key: str):
        engine = _get_thread_engine(engine_version, azure_endpoint, model_name)
        call_idx = 0

        def call_llm(prompt: str) -> str:
            nonlocal call_idx
            call_idx += 1
            try:
                raw0 = engine(prompt)[0]
            except Exception as e:
                _write_mbench_call(
                    mbench_dir=mbench_dir,
                    key=key,
                    step_name=step_name,
                    attempt=call_idx,
                    prompt=prompt,
                    raw="",
                    model_name=model_name,
                    azure_endpoint=azure_endpoint,
                    extra={"error": f"engine_call_error:{e}", "exception_type": type(e).__name__},
                )
                raise
            raw_s = "" if raw0 is None else str(raw0)
            _write_mbench_call(
                mbench_dir=mbench_dir,
                key=key,
                step_name=step_name,
                attempt=call_idx,
                prompt=prompt,
                raw=raw_s,
                model_name=model_name,
                azure_endpoint=azure_endpoint,
            )
            return raw_s

        return call_llm

    def _write_running_summary(phase: str, countsA: Counter, countsB: Counter, doneA: int, totalA: int, doneB: int, totalB: int) -> None:
        snap = {
            "run_id": run_id,
            "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "phase": phase,
            "progress": {
                "presence_done": doneA,
                "presence_total": totalA,
                "coverage_done": doneB,
                "coverage_total": totalB,
            },
            "counts": {
                "presence": dict(countsA),
                "coverage": dict(countsB),
            },
            "cost": {"total_cost_usd": cost_total, "with_cost": cost_with, "done": cost_done},
            "outputs_incremental": {
                "rerun_missing_parent_items_jsonl": str(rerun_missing_items_jsonl),
                "rerun_bad_existing_eff_items_jsonl": str(rerun_bad_existing_items_jsonl),
                "needs_review_items_jsonl": str(needs_review_items_jsonl),
                "rerun_missing_parent_sides_incremental_txt": str(rerun_missing_parent_sides_inc_txt),
                "rerun_bad_existing_eff_sides_incremental_txt": str(rerun_bad_existing_eff_sides_inc_txt),
            },
        }
        summary_running_json.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _print_progress(tag: str, done: int, total: int, counts: Counter) -> None:
        print(
            f"\r[{tag}] done={done}/{total} rerun={counts.get('rerun', 0)} needs_review={counts.get('needs_review', 0)} "
            f"cost_total={_fmt_money(cost_total)} cost_lines={cost_with}/{cost_done}",
            end="",
            file=sys.stdout,
            flush=True,
        )

    totalB_planned = 0  # will fill later for running summary

    # Run presence tasks
    countsA = Counter()
    doneA = 0
    totalA = len(presence_tasks)
    countsB = Counter()
    doneB = 0
    totalB = 0

    def _worker_presence(t: ParentPresenceTask) -> Tuple[ParentPresenceTask, Dict[str, Any]]:
        key = f"{t.parent_tid}_{t.side}_cohort-parent_or_all"
        call_llm = _call_llm_factory("presence_check", key)
        gate = ParentPresenceGate(call_llm=call_llm, prompt_template=presence_prompt, model_name=model_name)

        # conservative: if we literally have no usable input, mark needs_review
        if not (t.shared_context.strip() or t.subcohort_context.strip() or t.side_criteria.strip() or t.corpus_item_json.strip()):
            meta = {"executed": False, "error": "all_input_fields_empty", "model": model_name}
            _write_mbench_final(mbench_dir=mbench_dir, key=key, step_name="presence_check", meta=meta, note="all_input_fields_empty")
            return t, meta

        try:
            meta = gate.run(t)
        except Exception as e:
            meta = {"executed": False, "error": f"gate_exception:{e}", "exception_type": type(e).__name__, "model": model_name}

        note = "ok" if meta.get("executed") is True else "executed_false_or_error"
        _write_mbench_final(mbench_dir=mbench_dir, key=key, step_name="presence_check", meta=meta, note=note)
        return t, meta

    if totalA > 0:
        if max_workers <= 1:
            for t in presence_tasks:
                task, meta = _worker_presence(t)
                doneA += 1

                with lock:
                    # cost update
                    cost_done += 1
                    c = meta.get("estimated_cost_usd")
                    if isinstance(c, (int, float)):
                        cost_total += float(c)
                        cost_with += 1

                    executed = (meta.get("executed") is True)
                    if executed and isinstance(meta.get("criteria_present"), bool):
                        if meta["criteria_present"] is True:
                            countsA["rerun"] += 1
                            item = {
                                "parent_trial_id": task.parent_tid,
                                "side": task.side,
                                "criteria_present": True,
                                "reason": str(meta.get("reason", "")),
                                "canonical_present_counts": present_counts.get(task.parent_tid, {}),
                                "meta": meta,
                            }
                            rerun_missing_items.append(item)
                            # incremental writes
                            _append_jsonl(rerun_missing_items_jsonl, item)
                            _append_line(rerun_missing_parent_sides_inc_txt, f"{task.parent_tid}\t{task.side}")
                        else:
                            countsA["ok_missing"] += 1
                    else:
                        countsA["needs_review"] += 1
                        item = {
                            "kind": "presence_check",
                            "parent_trial_id": task.parent_tid,
                            "side": task.side,
                            "canonical_present_counts": present_counts.get(task.parent_tid, {}),
                            "meta": meta,
                        }
                        needs_review_items.append(item)
                        # incremental writes
                        _append_jsonl(needs_review_items_jsonl, item)

                    if doneA % progress_every == 0 or doneA == totalA:
                        _print_progress("PRESENCE", doneA, totalA, countsA)
                        _write_running_summary("presence_check", countsA, countsB, doneA, totalA, doneB, totalB)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_worker_presence, t) for t in presence_tasks]
                for fut in as_completed(futs):
                    task, meta = fut.result()
                    with lock:
                        doneA += 1
                        # cost update
                        cost_done += 1
                        c = meta.get("estimated_cost_usd")
                        if isinstance(c, (int, float)):
                            cost_total += float(c)
                            cost_with += 1

                        executed = (meta.get("executed") is True)
                        if executed and isinstance(meta.get("criteria_present"), bool):
                            if meta["criteria_present"] is True:
                                countsA["rerun"] += 1
                                item = {
                                    "parent_trial_id": task.parent_tid,
                                    "side": task.side,
                                    "criteria_present": True,
                                    "reason": str(meta.get("reason", "")),
                                    "canonical_present_counts": present_counts.get(task.parent_tid, {}),
                                    "meta": meta,
                                }
                                rerun_missing_items.append(item)
                                # incremental writes
                                _append_jsonl(rerun_missing_items_jsonl, item)
                                _append_line(rerun_missing_parent_sides_inc_txt, f"{task.parent_tid}\t{task.side}")
                            else:
                                countsA["ok_missing"] += 1
                        else:
                            countsA["needs_review"] += 1
                            item = {
                                "kind": "presence_check",
                                "parent_trial_id": task.parent_tid,
                                "side": task.side,
                                "canonical_present_counts": present_counts.get(task.parent_tid, {}),
                                "meta": meta,
                            }
                            needs_review_items.append(item)
                            # incremental writes
                            _append_jsonl(needs_review_items_jsonl, item)

                        if doneA % progress_every == 0 or doneA == totalA:
                            _print_progress("PRESENCE", doneA, totalA, countsA)
                            _write_running_summary("presence_check", countsA, countsB, doneA, totalA, doneB, totalB)
        print("", flush=True)

    # -------------------------
    # Phase B: Existing SMT coverage audit
    # -------------------------
    rerun_bad_existing_items: List[Dict[str, Any]] = []

    if not args.skip_coverage:
        coverage_tasks: List[CoverageTask] = []
        for prog in progs:
            # quick deterministic skip: empty file => definitely rerun
            try:
                txt = prog.path.read_text(encoding="utf-8", errors="ignore")
                if not txt.strip():
                    coverage_tasks.append(
                        CoverageTask(
                            eff_tid=prog.eff_tid,
                            parent_tid=base_trial_id(prog.eff_tid),
                            side=prog.side,
                            cohort_id="unknown",
                            cohort_label="Unknown cohort (empty SMT)",
                            shared_context="",
                            subcohort_context="",
                            side_criteria="",
                            corpus_item_json="",
                            smt_text="",
                            smt_path=str(prog.path),
                            cohort_match_method="empty_smt",
                            cohort_match_index=None,
                        )
                    )
                else:
                    coverage_tasks.append(
                        build_coverage_task(
                            prog=prog,
                            corpus_lookup=corpus_lookup,
                            snapshot_dir=snapshot_dir,
                            max_shared_chars=max_shared,
                            max_subcohort_chars=max_subctx,
                            max_side_criteria_chars=max_sidecrit,
                            max_corpus_item_chars=max_corpus_item,
                            max_smt_chars=max_smt,
                        )
                    )
            except Exception:
                coverage_tasks.append(
                    CoverageTask(
                        eff_tid=prog.eff_tid,
                        parent_tid=base_trial_id(prog.eff_tid),
                        side=prog.side,
                        cohort_id="unknown",
                        cohort_label="Unknown cohort (read error)",
                        shared_context="",
                        subcohort_context="",
                        side_criteria="",
                        corpus_item_json="",
                        smt_text="",
                        smt_path=str(prog.path),
                        cohort_match_method="read_error",
                        cohort_match_index=None,
                    )
                )

        totalB = len(coverage_tasks)
        print(f"[PLAN B] coverage_programs={totalB}", flush=True)

        def _worker_coverage(t: CoverageTask) -> Tuple[CoverageTask, Dict[str, Any]]:
            key = f"{t.eff_tid}_{t.side}_cohort-{t.cohort_id}"
            # deterministic empty SMT => no LLM call, directly fail coverage
            if t.cohort_match_method in ("empty_smt", "read_error") and not t.smt_text.strip():
                meta = {
                    "executed": True,
                    "coverage_ok": False,
                    "reason": t.cohort_match_method,
                    "model": model_name,
                    "parent_tid": t.parent_tid,
                    "eff_tid": t.eff_tid,
                    "side": t.side,
                    "smt_path": t.smt_path,
                    "cohort_id": t.cohort_id,
                    "cohort_label": t.cohort_label,
                    "cohort_match_method": t.cohort_match_method,
                    "cohort_match_index": t.cohort_match_index,
                    "estimated_cost_usd": None,
                    "pricing_used": None,
                }
                _write_mbench_final(mbench_dir=mbench_dir, key=key, step_name="coverage_check", meta=meta, note="deterministic_fail")
                return t, meta

            call_llm = _call_llm_factory("coverage_check", key)
            gate = CoverageGate(call_llm=call_llm, prompt_template=coverage_prompt, model_name=model_name)

            # if we have SMT but no side criteria/context at all, treat as needs_review (avoid false rerun)
            if (t.smt_text.strip()) and not (t.shared_context.strip() or t.subcohort_context.strip() or t.side_criteria.strip() or t.corpus_item_json.strip()):
                meta = {"executed": False, "error": "no_context_for_coverage_check", "model": model_name}
                _write_mbench_final(mbench_dir=mbench_dir, key=key, step_name="coverage_check", meta=meta, note="no_context_for_coverage_check")
                return t, meta

            try:
                meta = gate.run(t)
            except Exception as e:
                meta = {"executed": False, "error": f"gate_exception:{e}", "exception_type": type(e).__name__, "model": model_name}

            note = "ok" if meta.get("executed") is True else "executed_false_or_error"
            _write_mbench_final(mbench_dir=mbench_dir, key=key, step_name="coverage_check", meta=meta, note=note)
            return t, meta

        if totalB > 0:
            if max_workers <= 1:
                for t in coverage_tasks:
                    task, meta = _worker_coverage(t)
                    doneB += 1

                    with lock:
                        # cost update
                        cost_done += 1
                        c = meta.get("estimated_cost_usd")
                        if isinstance(c, (int, float)):
                            cost_total += float(c)
                            cost_with += 1

                        executed = (meta.get("executed") is True)
                        if executed and isinstance(meta.get("coverage_ok"), bool):
                            if meta["coverage_ok"] is False:
                                countsB["rerun"] += 1
                                item = {
                                    "parent_trial_id": task.parent_tid,
                                    "effective_trial_id": task.eff_tid,
                                    "side": task.side,
                                    "smt_path": task.smt_path,
                                    "reason": str(meta.get("reason", "")),
                                    "meta": meta,
                                }
                                rerun_bad_existing_items.append(item)
                                # incremental writes
                                _append_jsonl(rerun_bad_existing_items_jsonl, item)
                                _append_line(
                                    rerun_bad_existing_eff_sides_inc_txt,
                                    f"{task.eff_tid}\t{task.side}\t{task.smt_path}",
                                )
                            else:
                                countsB["ok"] += 1
                        else:
                            countsB["needs_review"] += 1
                            item = {
                                "kind": "coverage_check",
                                "parent_trial_id": task.parent_tid,
                                "effective_trial_id": task.eff_tid,
                                "side": task.side,
                                "smt_path": task.smt_path,
                                "meta": meta,
                            }
                            needs_review_items.append(item)
                            # incremental writes
                            _append_jsonl(needs_review_items_jsonl, item)

                        if doneB % progress_every == 0 or doneB == totalB:
                            _print_progress("COVERAGE", doneB, totalB, countsB)
                            _write_running_summary("coverage_check", countsA, countsB, doneA, totalA, doneB, totalB)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_worker_coverage, t) for t in coverage_tasks]
                    for fut in as_completed(futs):
                        task, meta = fut.result()
                        with lock:
                            doneB += 1
                            # cost update
                            cost_done += 1
                            c = meta.get("estimated_cost_usd")
                            if isinstance(c, (int, float)):
                                cost_total += float(c)
                                cost_with += 1

                            executed = (meta.get("executed") is True)
                            if executed and isinstance(meta.get("coverage_ok"), bool):
                                if meta["coverage_ok"] is False:
                                    countsB["rerun"] += 1
                                    item = {
                                        "parent_trial_id": task.parent_tid,
                                        "effective_trial_id": task.eff_tid,
                                        "side": task.side,
                                        "smt_path": task.smt_path,
                                        "reason": str(meta.get("reason", "")),
                                        "meta": meta,
                                    }
                                    rerun_bad_existing_items.append(item)
                                    # incremental writes
                                    _append_jsonl(rerun_bad_existing_items_jsonl, item)
                                    _append_line(
                                        rerun_bad_existing_eff_sides_inc_txt,
                                        f"{task.eff_tid}\t{task.side}\t{task.smt_path}",
                                    )
                                else:
                                    countsB["ok"] += 1
                            else:
                                countsB["needs_review"] += 1
                                item = {
                                    "kind": "coverage_check",
                                    "parent_trial_id": task.parent_tid,
                                    "effective_trial_id": task.eff_tid,
                                    "side": task.side,
                                    "smt_path": task.smt_path,
                                    "meta": meta,
                                }
                                needs_review_items.append(item)
                                # incremental writes
                                _append_jsonl(needs_review_items_jsonl, item)

                            if doneB % progress_every == 0 or doneB == totalB:
                                _print_progress("COVERAGE", doneB, totalB, countsB)
                                _write_running_summary("coverage_check", countsA, countsB, doneA, totalA, doneB, totalB)
            print("", flush=True)

    # -------------------------
    # Write final outputs (unchanged behavior, but now you also have JSONL + incremental TXT)
    # -------------------------
    rerun_missing_items.sort(key=lambda x: (x.get("parent_trial_id", ""), x.get("side", "")))
    rerun_bad_existing_items.sort(key=lambda x: (x.get("effective_trial_id", ""), x.get("side", "")))
    needs_review_items.sort(key=lambda x: (x.get("kind", ""), x.get("parent_trial_id", ""), x.get("side", "")))

    rerun_missing_lines = [f"{it['parent_trial_id']}\t{it['side']}" for it in rerun_missing_items]
    rerun_bad_existing_lines = [f"{it['effective_trial_id']}\t{it['side']}\t{it.get('smt_path','')}" for it in rerun_bad_existing_items]

    # Final (stable) list files
    (run_dir / "rerun_missing_parent_sides.txt").write_text(
        "\n".join(rerun_missing_lines) + ("\n" if rerun_missing_lines else ""), encoding="utf-8"
    )
    (run_dir / "rerun_missing_parent_items.json").write_text(
        json.dumps(rerun_missing_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (run_dir / "rerun_bad_existing_eff_sides.txt").write_text(
        "\n".join(rerun_bad_existing_lines) + ("\n" if rerun_bad_existing_lines else ""), encoding="utf-8"
    )
    (run_dir / "rerun_bad_existing_eff_items.json").write_text(
        json.dumps(rerun_bad_existing_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (run_dir / "needs_review_items.json").write_text(
        json.dumps(needs_review_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "run_id": run_id,
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "canonical_dir": str(canonical_dir),
        "corpus_path": str(corpus_path),
        "snapshot_dir": (str(snapshot_dir) if snapshot_dir else None),
        "filters": {"trial_prefix": trial_prefix, "allowlist_prefixes_count": len(allow_prefixes)},
        "engine": {"engine_version": engine_version, "model_name": model_name, "openai_endpoint": azure_endpoint},
        "counts": {
            "presence_candidates": len(presence_tasks),
            "presence_rerun_missing_parent_side": len(rerun_missing_items),
            "coverage_rerun_bad_existing_eff_side": len(rerun_bad_existing_items),
            "needs_review_items": len(needs_review_items),
        },
        "cost": {"total_cost_usd": cost_total, "with_cost": cost_with, "done": cost_done},
        "outputs": {
            "rerun_missing_parent_sides_txt": str(run_dir / "rerun_missing_parent_sides.txt"),
            "rerun_bad_existing_eff_sides_txt": str(run_dir / "rerun_bad_existing_eff_sides.txt"),
            "summary_json": str(run_dir / "summary.json"),
            "summary_running_json": str(summary_running_json),
            "mbench_dir": str(mbench_dir),
            # incremental outputs
            "rerun_missing_parent_items_jsonl": str(rerun_missing_items_jsonl),
            "rerun_bad_existing_eff_items_jsonl": str(rerun_bad_existing_items_jsonl),
            "needs_review_items_jsonl": str(needs_review_items_jsonl),
            "rerun_missing_parent_sides_incremental_txt": str(rerun_missing_parent_sides_inc_txt),
            "rerun_bad_existing_eff_sides_incremental_txt": str(rerun_bad_existing_eff_sides_inc_txt),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # write one last running summary snapshot (phase=final)
    _write_running_summary("final", countsA, countsB, doneA, totalA, doneB, totalB)

    print(f"[DONE] run_dir: {run_dir}", flush=True)
    print(f"[RERUN A] missing parent-sides: {len(rerun_missing_items)} => {run_dir / 'rerun_missing_parent_sides.txt'}", flush=True)
    print(f"[RERUN B] bad existing eff-sides: {len(rerun_bad_existing_items)} => {run_dir / 'rerun_bad_existing_eff_sides.txt'}", flush=True)
    print(f"[NEEDS_REVIEW] items: {len(needs_review_items)} => {run_dir / 'needs_review_items.json'}", flush=True)
    print(f"[COST] total={_fmt_money(cost_total)} cost_lines={cost_with}/{cost_done}", flush=True)

    print(f"[INCREMENTAL] rerun_missing_parent_items.jsonl => {rerun_missing_items_jsonl}", flush=True)
    print(f"[INCREMENTAL] rerun_bad_existing_eff_items.jsonl => {rerun_bad_existing_items_jsonl}", flush=True)
    print(f"[INCREMENTAL] needs_review_items.jsonl => {needs_review_items_jsonl}", flush=True)
    print(f"[INCREMENTAL] rerun_missing_parent_sides_incremental.txt => {rerun_missing_parent_sides_inc_txt}", flush=True)
    print(f"[INCREMENTAL] rerun_bad_existing_eff_sides_incremental.txt => {rerun_bad_existing_eff_sides_inc_txt}", flush=True)
    print(f"[INCREMENTAL] summary_running.json => {summary_running_json}", flush=True)


if __name__ == "__main__":
    main()
