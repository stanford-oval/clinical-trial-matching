#!/usr/bin/env python3
# scripts/match_patient_to_trial.py
"""
Matcher/miner–only runner that DOES NOT require PROGRAM checkpoints.
It builds a fresh context from persisted artifacts.

What it uses
------------
• build/symtab/<trial_id>_<side>_variable_index.json   (REQUIRED)
• build/ir/<trial_id>_<side>_program.smt2              (optional)
• build/linkmap/<trial_id>_<side>_entities.json        (optional)
• SMTVariableValueMiner prompt (required – via --prompt-root or --prompt-map)

Patient sourcing
---------------
• sigir-*       → <DATA_ROOT>/sigir/queries.jsonl
• trec-2021*    → <DATA_ROOT>/trec_2021/queries.jsonl
• trec-2022*    → <DATA_ROOT>/trec_2022/queries.jsonl
• else          → <DATA_ROOT>/patients/<patient_id>.json
• or override   → --patient-file

Ground-truth labels (optional)
------------------------------
If you pass --labels pointing to split JSONL files (or a directory that
contains trial_criteria_{train,val,test}.jsonl), the script will attach:
  - inclusion: inclusion_gpt4_eligibility, inclusion_expert_eligibility
  - exclusion: exclusion_gpt4_eligibility, exclusion_expert_eligibility
to the final JSON output under result['inclusion']['labels'] / ['exclusion']['labels'].

Additionally, this script now surfaces those labels (and criteria) into the matcher context
so SMTMatcher can compute sequential alignment (ours | GPT-4 | Expert) and show criterion text.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import datetime as dt
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import os
import re

# ── project imports ───────────────────────────────────────────────────────
from smt_core.inference_engine import AzureInferenceEngine
from smt_matcher.modules.smt_matcher.SMTMatcher import SMTMatcher
from smt_core.utils.text_utils import dict_to_readable_string

SIDES = ("inclusion", "exclusion")

# Only the miner prompt is needed for matching
REQUIRED_PROMPTS: Dict[str, str] = {
    "SMTVariableValueMinerInclusion_prompt": "SMTMatcher/SMTVariableValueMinerInclusion.prompt",
    "SMTVariableValueMinerExclusion_prompt": "SMTMatcher/SMTVariableValueMinerExclusion.prompt",
}

# Keys expected in split JSONLs
TRIAL_ID_COL = "trial_id"
PATIENT_ID_COL = "patient_id"

INCL_GPT4 = "inclusion_gpt4_eligibility"
INCL_EXPT = "inclusion_expert_eligibility"
EXCL_GPT4 = "exclusion_gpt4_eligibility"
EXCL_EXPT = "exclusion_expert_eligibility"

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data_root: pathlib.Path = pathlib.Path(os.getenv("TRIAL_DATA", "../dataset/clinical_trial"))
    build_root: pathlib.Path = pathlib.Path(os.getenv("SATIR_BUILD", "build"))
    prompt_root: pathlib.Path = pathlib.Path(__file__).resolve().parent / "prompts" / "clinical_trial"
    prompt_map_json: pathlib.Path | None = None

    # Azure client (used by matcher)
    azure_endpoint: str = os.getenv("OPENAI_ENDPOINT", "")
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # derived dirs
    ir_dir: pathlib.Path = field(init=False)
    symtab_dir: pathlib.Path = field(init=False)
    linkmap_dir: pathlib.Path = field(init=False)

    def __post_init__(self):
        self.ir_dir = (self.build_root / "ir")
        self.symtab_dir = (self.build_root / "symtab")
        self.linkmap_dir = (self.build_root / "linkmap")
        # Do not create dirs; we just read them.

    def prompt_sources(self) -> Dict[str, pathlib.Path]:
        """Map prompt keys → absolute file paths (JSON map overrides)."""
        base = self.prompt_root if self.prompt_root.is_absolute() else (pathlib.Path.cwd() / self.prompt_root)
        mapping = {k: (base / rel).resolve() for k, rel in REQUIRED_PROMPTS.items()}

        if self.prompt_map_json:
            raw = json.loads(self.prompt_map_json.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("--prompt-map must be a JSON object {key: path}")
            for k, v in raw.items():
                p = pathlib.Path(v)
                if not p.is_absolute():
                    p = (base / p)
                mapping[k] = p.resolve()
        return mapping

    def persisted_paths(self, trial_id: str, side: str) -> Dict[str, pathlib.Path]:
        return {
            "smt": self.ir_dir / f"{trial_id}_{side}_program.smt2",
            "var": self.symtab_dir / f"{trial_id}_{side}_variable_index.json",
            "ent": self.linkmap_dir / f"{trial_id}_{side}_entities.json",
        }

# ──────────────────────────────────────────────────────────────────────────
# Prompt loader (minimal)
# ──────────────────────────────────────────────────────────────────────────
def ensure_prompt_templates(ctx: Dict[str, Any], *, pathmap: Dict[str, pathlib.Path]) -> Dict[str, Any]:
    """Attach ONLY the matcher/miner prompts into ctx (overwrite if present)."""
    updates: Dict[str, str] = {}
    for key, path in pathmap.items():
        try:
            updates[key] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt not found for key '{key}': {path}")
    ctx.update(updates)
    return ctx

# ──────────────────────────────────────────────────────────────────────────
# Patient loader (router for SIGIR/TREC; else JSON file)
# ──────────────────────────────────────────────────────────────────────────
_SIGIR_RE = re.compile(r"^sigir-\d{5,}$", re.IGNORECASE)
_TREC21_RE = re.compile(r"^trec-2021\d+$", re.IGNORECASE)
_TREC22_RE = re.compile(r"^trec-2022\d+$", re.IGNORECASE)

def _infer_query_dataset_dir(patient_id: str) -> str | None:
    if _SIGIR_RE.match(patient_id): return "sigir"
    if _TREC21_RE.match(patient_id): return "trec_2021"
    if _TREC22_RE.match(patient_id): return "trec_2022"
    return None

def _load_from_queries_jsonl(jsonl_path: pathlib.Path, patient_id: str) -> Dict[str, Any]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Queries file not found: {jsonl_path}")
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("_id") == patient_id:
                return {
                    "patient_id": patient_id,
                    "text": obj.get("text", ""),
                    "metadata": obj.get("metadata", {}),
                    "_raw": obj,
                }
    raise FileNotFoundError(f"Patient id '{patient_id}' not found in {jsonl_path}")

def load_patient(patient_id: str, data_root: pathlib.Path, patient_file: str | None) -> Dict[str, Any]:
    """
    Precedence:
      1) --patient-file explicit JSON
      2) Known ID → <data_root>/<sigir|trec_2021|trec_2022>/queries.jsonl
      3) Fallback → <data_root>/patients/<patient_id>.json
    """
    if patient_file:
        p = pathlib.Path(patient_file)
        if not p.exists():
            raise FileNotFoundError(f"Patient file not found: {p}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj.setdefault("patient_id", patient_id)
        return obj

    subdir = _infer_query_dataset_dir(patient_id)
    if subdir:
        return _load_from_queries_jsonl(data_root / subdir / "queries.jsonl", patient_id)

    guess = data_root / "patients" / f"{patient_id}.json"
    if not guess.exists():
        raise FileNotFoundError(
            f"Patient not found. Tried: {guess}\n"
            "Provide --patient-file or use a known ID (sigir-*, trec-2021*, trec-2022*)."
        )
    obj = json.loads(guess.read_text(encoding="utf-8"))
    obj.setdefault("patient_id", patient_id)
    return obj

# ──────────────────────────────────────────────────────────────────────────
# Ground-truth labels loader (from split JSONLs)
# ──────────────────────────────────────────────────────────────────────────
def _iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def _find_pair_in_jsonl(path: pathlib.Path, trial_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
    for obj in _iter_jsonl(path):
        if obj.get(TRIAL_ID_COL) == trial_id and obj.get(PATIENT_ID_COL) == patient_id:
            return obj
    return None

def load_labels_for_pair(
    labels_inputs: List[pathlib.Path],
    trial_id: str,
    patient_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Search one or more paths for the (trial_id, patient_id) record.
    Each path may be a JSONL file or a directory that contains the three split files.
    Returns the full JSON object for the pair if found, else None.
    """
    CAND_FILENAMES = (
        "trial_criteria_train.jsonl",
        "trial_criteria_val.jsonl",
        "trial_criteria_test.jsonl",
    )

    for p in labels_inputs:
        if p.is_dir():
            for name in CAND_FILENAMES:
                f = p / name
                if f.exists():
                    rec = _find_pair_in_jsonl(f, trial_id, patient_id)
                    if rec is not None:
                        return rec
        elif p.is_file():
            rec = _find_pair_in_jsonl(p, trial_id, patient_id)
            if rec is not None:
                return rec
    return None

def extract_side_labels(labels_obj: Dict[str, Any], side: str) -> Dict[str, List[str]]:
    if side == "inclusion":
        return {
            "gpt4":   list(labels_obj.get(INCL_GPT4, []) or []),
            "expert": list(labels_obj.get(INCL_EXPT, []) or []),
        }
    else:
        return {
            "gpt4":   list(labels_obj.get(EXCL_GPT4, []) or []),
            "expert": list(labels_obj.get(EXCL_EXPT, []) or []),
        }

# ⬇️ NEW: criteria extraction
def extract_side_criteria(labels_obj: Dict[str, Any], side: str) -> List[str]:
    key = "inclusion_criteria" if side == "inclusion" else "exclusion_criteria"
    return list(labels_obj.get(key, []) or [])

# ──────────────────────────────────────────────────────────────────────────
# Build fresh context from persisted artifacts (no checkpoints)
# ──────────────────────────────────────────────────────────────────────────
def build_ctx_from_persisted(trial_id: str, side: str, cfg: Config) -> Dict[str, Any]:
    paths = cfg.persisted_paths(trial_id, side)

    if not paths["var"].exists():
        raise FileNotFoundError(
            f"Variable index missing: {paths['var']}\n"
            "Run the programmer so it persists artifacts, or point --build-root to the correct folder."
        )

    ctx: Dict[str, Any] = {
        "trial_id": trial_id,
        "inc_exc": side,
        "variable_index": json.loads(paths["var"].read_text(encoding="utf-8")),
    }

    if paths["smt"].exists():
        ctx["smt_program_lines"] = paths["smt"].read_text(encoding="utf-8").splitlines()

    if paths["ent"].exists():
        ctx["requirement_bundles"] = json.loads(paths["ent"].read_text(encoding="utf-8"))

    return ctx


def _json_default(o):
    # Local imports to avoid adding globals
    import pathlib as _p, datetime as _d, re as _r
    if isinstance(o, set):
        try:
            return sorted(o)
        except TypeError:
            return list(o)
    if isinstance(o, (_p.Path,)):
        return str(o)
    if isinstance(o, (_d.date, _d.datetime)):
        return o.isoformat()
    if isinstance(o, _r.Pattern):
        return o.pattern
    return str(o)

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _extract_note_text(patient: Dict[str, Any]) -> str | None:
    """Pick the first non-empty string among common note keys."""
    for k in ("text", "note", "content", "summary"):
        v = patient.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _sat_like_from_eval_result(result: Dict[str, Any]) -> bool | None:
    """
    Try to compute a boolean 'sat-like' from SMTMatcher's context result.
    Returns True if overall status is 'sat', False if 'unsat', else None.
    Handles single- and multi-patient shapes.
    """
    if not isinstance(result, dict):
        return None
    er = result.get("eval_result")
    if not isinstance(er, dict):
        return None

    # Single-patient shape
    if "per_requirement" in er:
        status = er.get("overall_status_grouped") or er.get("overall_status")
        if isinstance(status, str):
            if status.lower() == "sat":
                return True
            if status.lower() == "unsat":
                return False
        return None

    # Multi-patient shape
    decisions: List[bool] = []
    for _pid, res in er.items():
        if not isinstance(res, dict):
            continue
        status = res.get("overall_status_grouped") or res.get("overall_status")
        if isinstance(status, str):
            if status.lower() == "sat":
                decisions.append(True)
            elif status.lower() == "unsat":
                decisions.append(False)
    if not decisions:
        return None
    return all(decisions)

# ──────────────────────────────────────────────────────────────────────────
# Trial record loading (for optional judges)
# ──────────────────────────────────────────────────────────────────────────
def _best_effort_load_trial(trial_id: str, cfg: Config) -> Dict[str, Any]:
    """
    Load a trial record for judge rendering. Tries corpus JSONLs under
    cfg.data_root first (standard SIGIR/TREC layout), falls back to a
    minimal stub built from build artifacts.
    """
    # Strip subcohort suffix (e.g., NCT00000402a -> NCT00000402)
    parent = trial_id
    m = re.match(r"^(NCT\d{8})", trial_id or "")
    if m:
        parent = m.group(1)

    # Check common corpus locations
    for rel in ("sigir/corpus.jsonl", "trec_2021/corpus.jsonl", "trec_2022/corpus.jsonl"):
        corpus_path = cfg.data_root / rel
        if not corpus_path.exists():
            continue
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("_id") == parent or obj.get("nct_id") == parent:
                        return _normalize_trial_obj(obj)
        except Exception:
            continue

    # Fallback: minimal stub
    return {
        "nct_id": parent,
        "brief_title": parent,
        "brief_summary": "",
        "inclusion_criteria": "",
        "exclusion_criteria": "",
    }


def _normalize_trial_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a corpus JSONL object to the shape judges expect."""
    md = obj.get("metadata") or {}
    title = obj.get("title") or obj.get("brief_title") or md.get("brief_title") or ""
    summary = obj.get("text") or obj.get("description") or md.get("brief_summary") or ""

    inc = md.get("inclusion_criteria") or obj.get("inclusion_criteria") or ""
    exc = md.get("exclusion_criteria") or obj.get("exclusion_criteria") or ""

    # If no split criteria, try to split the full text
    if not inc and not exc and summary:
        mm = re.search(r"inclusion\s*criteria[:\s]*", summary, flags=re.IGNORECASE)
        if mm:
            split_at = mm.end()
            post = summary[split_at:]
            exc_m = re.search(r"exclusion\s*criteria[:\s]*", post, flags=re.IGNORECASE)
            if exc_m:
                inc = post[:exc_m.start()].strip()
                exc = post[exc_m.end():].strip()
            else:
                inc = post.strip()

    return {
        "nct_id": obj.get("_id") or obj.get("nct_id"),
        "brief_title": title,
        "brief_summary": summary,
        "inclusion_criteria": inc,
        "exclusion_criteria": exc,
        "diseases_list": md.get("diseases_list") or [],
        "drugs_list": md.get("drugs_list") or [],
        "metadata": md,
    }


# ──────────────────────────────────────────────────────────────────────────
# Core match per side
# ──────────────────────────────────────────────────────────────────────────
def run_match_for_side(
    side: str,
    trial_id: str,
    patient: Dict[str, Any],
    cfg: Config,
    engine: AzureInferenceEngine,
    labels_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build fresh ctx from persisted files, attach miner prompt + patient, run SMTMatcher."""
    ctx = build_ctx_from_persisted(trial_id, side, cfg)

    # Load only the miner prompt
    ctx = ensure_prompt_templates(ctx, pathmap=cfg.prompt_sources())

    # Patient payload
    ctx["patient_id"] = patient.get("patient_id") or patient.get("_id") or patient.get("id") or "UNKNOWN"
    ctx["patient"] = patient
    ctx["patient_contextual_text"] = dict_to_readable_string(patient)

    # Surface the note so SMTVariableValueMiner can see it
    note_text = _extract_note_text(patient)
    if note_text:
        ctx["patient_notes"] = [note_text]
        ctx["patient_notes_db"] = [{"_id": ctx["patient_id"], "text": note_text}]

    # Minimal microbench config
    ctx.setdefault("MBENCH_ENABLED", True)
    ctx.setdefault("MBENCH_ROOT", "mbench")
    ctx["MBENCH_TS"] = False
    ctx["MBENCH_RUN_LABEL"] = f"{trial_id}_{side}_{ctx['patient_id']}"

    # ⬇️ NEW: surface side-specific labels + criteria INTO the matcher context
    if labels_obj:
        ctx["labels"] = extract_side_labels(labels_obj, side)      # {"gpt4": [...], "expert": [...]}
        ctx["criteria"] = extract_side_criteria(labels_obj, side)  # ["criterion 1", "criterion 2", ...]

    # Run matcher
    result_ctx = SMTMatcher(engine=engine).forward(ctx)

    # Convenience boolean derived from eval_result
    sat_like = _sat_like_from_eval_result(result_ctx)

    out: Dict[str, Any] = {
        "side": side,
        "sat_like": sat_like,
        "raw": result_ctx,
    }

    # Attach labels/criteria to CLI output for visibility
    if labels_obj:
        out["labels"] = extract_side_labels(labels_obj, side)
        out["criteria"] = extract_side_criteria(labels_obj, side)

    # Bubble the alignment into CLI result for convenience
    if isinstance(result_ctx, dict) and "alignment_sequential" in result_ctx:
        out["alignment_sequential"] = result_ctx["alignment_sequential"]

    return out

# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Matcher/miner-only (no checkpoints): uses persisted artifacts.")
    ap.add_argument("trial_id", help="NCT id, e.g., NCT02509286")
    ap.add_argument("patient_id", help="sigir-20141 | trec-20211 | trec-20221 | P001 ...")
    ap.add_argument("--data-root", default="../dataset/clinical_trial",
                    help="Dataset root (used by the router and fallback JSON).")
    ap.add_argument("--patient-file", default=None,
                    help="Explicit path to a patient JSON (bypass router).")
    ap.add_argument("--build-root", default="../build",
                    help="Root containing ir/, symtab/, linkmap/ persisted artifacts.")
    ap.add_argument("--prompt-root", default="prompts/clinical_trial",
                    help="Base directory for matcher prompts.")
    ap.add_argument("--prompt-map", default=None,
                    help="JSON { 'SMTVariableValueMinerInclusion_prompt': '/path/to/file', ... }")
    ap.add_argument("--labels", action="append", default=[],
                    help="Path(s) to split JSONL file(s) or a directory containing trial_criteria_{train,val,test}.jsonl. "
                         "Can be passed multiple times.")
    ap.add_argument("--log-json", default=None,
                    help="Optional path to write the full JSON output.")
    # Optional parallel judge baselines (OFF by default — preserves invariant output)
    ap.add_argument("--enable-llm-judge", action="store_true",
                    help="Run GPT-4 NL eligibility judge alongside SMT match (adds 'llm_judge' to output).")
    ap.add_argument("--enable-trialgpt-judge", action="store_true",
                    help="Run TrialGPT sentence-level judge alongside SMT match (adds 'trialgpt_judge' to output).")
    ap.add_argument("--judge-out-root", default=None,
                    help="Output/cache root for judges (default: <build-root>/../judge_out).")
    ap.add_argument("--judge-prompt", default=None,
                    help="Path to eligibility prompt (default: <prompt-root>/SMTMatcher/eligibility.explicit.prompt).")
    ap.add_argument("--judge-temperature", type=float, default=0.0,
                    help="Temperature for judge LLM calls (default: 0.0).")
    args = ap.parse_args(argv)

    cfg = Config(
        data_root=pathlib.Path(args.data_root),
        build_root=pathlib.Path(args.build_root),
        prompt_root=pathlib.Path(args.prompt_root),
        prompt_map_json=pathlib.Path(args.prompt_map) if args.prompt_map else None,
    )
    engine = AzureInferenceEngine(
        endpoint=cfg.azure_endpoint,
        api_key_env_var="OPENAI_API_KEY",
        model_name=cfg.model_name,
    )

    # Patient
    patient = load_patient(args.patient_id, cfg.data_root, args.patient_file)

    # Optional labels
    labels_inputs: List[pathlib.Path] = [pathlib.Path(p) for p in args.labels] if args.labels else []
    labels_obj: Optional[Dict[str, Any]] = None
    if labels_inputs:
        labels_obj = load_labels_for_pair(labels_inputs, args.trial_id, args.patient_id)

    # Match both sides
    side_results: Dict[str, Any] = {}
    for side in SIDES:
        try:
            side_results[side] = run_match_for_side(side, args.trial_id, patient, cfg, engine, labels_obj)
        except FileNotFoundError as e:
            print(
                json.dumps(
                    {
                        "trial_id": args.trial_id,
                        "patient_id": args.patient_id,
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "error": str(e),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            sys.exit(2)

    inc_sat = side_results["inclusion"]["sat_like"]
    exc_sat = side_results["exclusion"]["sat_like"]
    eligible = None if (inc_sat is None and exc_sat is None) else bool(inc_sat) and not bool(exc_sat)

    out = {
        "trial_id": args.trial_id,
        "patient_id": args.patient_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "eligible": eligible,
        "inclusion": side_results["inclusion"],
        "exclusion": side_results["exclusion"],
    }

    # Optional parallel judges (research/paper evaluation use)
    if args.enable_llm_judge or args.enable_trialgpt_judge:
        judge_out_root = pathlib.Path(
            args.judge_out_root or (pathlib.Path(args.build_root).parent / "judge_out")
        )
        trial_obj = _best_effort_load_trial(args.trial_id, cfg)

        if args.enable_llm_judge:
            from smt_matcher.judges import run_llm_eligibility_judge
            prompt_path = pathlib.Path(args.judge_prompt) if args.judge_prompt else (
                cfg.prompt_root / "SMTMatcher" / "eligibility.explicit.prompt"
            )
            try:
                out["llm_judge"] = run_llm_eligibility_judge(
                    trial_id=args.trial_id,
                    trial_obj=trial_obj,
                    patient=patient,
                    engine=engine,
                    prompt_path=prompt_path,
                    out_root=judge_out_root,
                    model_name=cfg.model_name,
                    temperature=args.judge_temperature,
                )
            except Exception as e:
                out["llm_judge_error"] = f"{type(e).__name__}: {e}"

        if args.enable_trialgpt_judge:
            from smt_matcher.judges import run_trialgpt_judge
            try:
                out["trialgpt_judge"] = run_trialgpt_judge(
                    trial_id=args.trial_id,
                    trial_obj=trial_obj,
                    patient=patient,
                    engine=engine,
                    out_root=judge_out_root,
                    model_name=cfg.model_name,
                    temperature=args.judge_temperature,
                )
            except Exception as e:
                out["trialgpt_judge_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default))

    if args.log_json:
        p = pathlib.Path(args.log_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default),
                     encoding="utf-8")


if __name__ == "__main__":
    main()
