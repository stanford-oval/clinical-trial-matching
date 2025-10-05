#!/usr/bin/env python3
"""
polarity_fix_irs.py

Enumerate all exclusion SMT programs under an IR directory, find the matching
subcohort context for each effective trial id, and run the SMTPolarityFixer
on (subcohort_ctx, smt_text).

Key fixes vs earlier version:
- Reuses repair_all_ir-style snapshot loading & cohort mapping (side-aware).
- Adds explicit allowlist support: --trial-allowlist.
- Filters are intersected: allowlist ∩ trial-prefix ∩ explicit-contradiction (if provided).

Cost features:
- Per-program estimated token + cost recorded in summary JSONL (from SMTPolarityFixer meta).
- Long-running running-total cost counter printed to terminal (single live line).
- Live prediction of FINAL total cost:
    predicted_final = avg_cost_so_far * total_planned
    predicted_remaining = predicted_final - current_total_cost
  (works even when parallelized because main process syncs futures and sums).
- Optional *preflight* cost prediction before running:
    - builds prompts (no LLM calls), counts prompt tokens (tiktoken),
    - assumes a configurable average completion token count per call,
    - prints estimated total cost for gpt-5, gpt-4.1, gpt-4.1-mini.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from trial_compiler.ir_finalizer.stages.smt_polarity_fix_module import SMTPolarityFixer

# costing.py is expected (used by smt_polarity_fix_module too)
try:
    from trial_compiler.ir_finalizer.utils.costing import OPENAI_PRICING, count_tokens, estimate_cost_usd  # type: ignore
except Exception:
    OPENAI_PRICING = {}  # type: ignore

    def count_tokens(text: str, model: str):  # type: ignore
        return None, "costing_py_missing"

    def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0):  # type: ignore
        return None


# ───────────────────── engine selection ─────────────────────

from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
if ENGINE_VERSION == "gpt-5":
    from smt_core.inference_engine_5 import AzureInferenceEngine
else:
    from smt_core.inference_engine import AzureInferenceEngine

print(f"[INFO] polarity_fix_irs: using {ENGINE_VERSION} (model={MODEL_NAME})")

DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) * 2)

# ───────────────────── Config & IR discovery ─────────────────────


@dataclass
class Config:
    repo_root: Path
    ir_dir: Path
    snapshot_dir: Path
    out_ir_dir: Path
    summary_jsonl: Path
    log_dir: Path
    mbench_dir: Path

    def __post_init__(self) -> None:
        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() else (self.repo_root / p).resolve()

        self.ir_dir = _resolve(self.ir_dir)
        self.snapshot_dir = _resolve(self.snapshot_dir)
        self.out_ir_dir = _resolve(self.out_ir_dir)
        self.summary_jsonl = _resolve(self.summary_jsonl)
        self.log_dir = _resolve(self.log_dir)
        self.mbench_dir = _resolve(self.mbench_dir)

        self.ir_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.out_ir_dir.mkdir(parents=True, exist_ok=True)
        self.summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mbench_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class IRProgram:
    eff_tid: str
    side: str
    path: Path


IR_PATTERN = re.compile(r"^(NCT[0-9]+[A-Za-z]?)_(inclusion|exclusion)_program\.smt2$")


def discover_ir_programs(
    cfg: Config, trial_prefix: Optional[str] = None, side_filter: Optional[str] = None
) -> List[IRProgram]:
    programs: List[IRProgram] = []
    for p in cfg.ir_dir.glob("NCT*_program.smt2"):
        m = IR_PATTERN.match(p.name)
        if not m:
            continue
        eff_tid, side = m.groups()
        if side_filter and side != side_filter:
            continue
        if trial_prefix and not eff_tid.startswith(trial_prefix):
            continue
        programs.append(IRProgram(eff_tid=eff_tid, side=side, path=p))
    programs.sort(key=lambda x: (x.eff_tid, x.side))
    return programs


# ───────────────────── Snapshot loading & subcohort mapping (MATCH repair_all_ir) ─────────────────────


def _candidate_parent_ids(eff_tid: str) -> List[str]:
    base_match = re.match(r"^(NCT[0-9]+)", eff_tid)
    if base_match:
        base = base_match.group(1)
        if eff_tid == base:
            return [base]
        return [eff_tid, base]
    return [eff_tid]


def load_snapshot_for_program(snapshot_dir: Path, eff_tid: str, side: str) -> Optional[Dict[str, Any]]:
    """
    Try:
      <tid>_<side>.json then <tid>.json
    for tid in [eff_tid, parent_prefix].
    """
    for tid in _candidate_parent_ids(eff_tid):
        for suffix in (f"_{side}", ""):
            snap_path = snapshot_dir / f"{tid}{suffix}.json"
            if not snap_path.exists():
                continue
            try:
                with snap_path.open(encoding="utf-8") as fh:
                    ctx = json.load(fh)
                logging.info("[SNAP] Loaded snapshot %s for eff_tid=%s side=%s", snap_path, eff_tid, side)
                return ctx
            except Exception as e:
                logging.error("[SNAP] Failed to read %s: %s", snap_path, e)
                continue

    logging.warning("[SNAP] No snapshot found for eff_tid=%s side=%s under %s", eff_tid, side, snapshot_dir)
    return None


def _build_minimal_subctx(
    parent_ctx: Dict[str, Any], eff_tid: str, side: str, cohort_record: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    parent_tid = parent_ctx.get("parent_trial_id") or parent_ctx.get("trial_id_parent") or parent_ctx.get("trial_id")

    pn = parent_ctx.get("preprocessor_normalized") or {}
    shared_context = parent_ctx.get("shared_context") or pn.get("shared_context", "")

    if cohort_record is not None:
        cid = cohort_record.get("id") or cohort_record.get("cohort_id") or cohort_record.get("label") or "C?"
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

    return {
        "trial_id_parent": parent_tid,
        "trial_id": eff_tid,
        "trial_id_effective": eff_tid,
        "cohort_id": cid,
        "cohort_label": label,
        "inclusion_criteria": inc,
        "exclusion_criteria": exc,
        "context": ctx_str,
        "contextual_text": ctxt,
        "shared_context": shared_context,
        "inc_exc": side,
        "preprocessor_normalized": pn,
    }


def extract_subcohort_for_eff_tid(parent_ctx: Dict[str, Any], eff_tid: str, side: str) -> Dict[str, Any]:
    subctxs = parent_ctx.get("__cohort_contexts__") or parent_ctx.get("__substudy_contexts__", [])
    if subctxs:
        for sc in subctxs:
            if sc.get("trial_id") == eff_tid or sc.get("trial_id_effective") == eff_tid:
                logging.info("[MAP] eff_tid=%s matched via __cohort_contexts__", eff_tid)
                return _build_minimal_subctx(parent_ctx, eff_tid, side, sc)

        eff_ids = parent_ctx.get("effective_trial_ids") or []
        if eff_tid in eff_ids and len(eff_ids) == len(subctxs):
            idx = eff_ids.index(eff_tid)
            logging.info("[MAP] eff_tid=%s matched via index into __cohort_contexts__", eff_tid)
            return _build_minimal_subctx(parent_ctx, eff_tid, side, subctxs[idx])

    pn = parent_ctx.get("preprocessor_normalized") or {}
    enroll = pn.get("enrollment_cohorts") or []
    if enroll:
        for co in enroll:
            if co.get("trial_id_effective") == eff_tid:
                logging.info("[MAP] eff_tid=%s matched via enrollment_cohorts", eff_tid)
                return _build_minimal_subctx(parent_ctx, eff_tid, side, co)

        eff_ids = parent_ctx.get("effective_trial_ids") or []
        if eff_tid in eff_ids and len(eff_ids) == len(enroll):
            idx = eff_ids.index(eff_tid)
            logging.info("[MAP] eff_tid=%s matched via index into enrollment_cohorts", eff_tid)
            return _build_minimal_subctx(parent_ctx, eff_tid, side, enroll[idx])

    logging.warning("[MAP] Could not find explicit cohort for eff_tid=%s; using parent snapshot as single cohort.", eff_tid)
    return _build_minimal_subctx(parent_ctx, eff_tid, side, None)


# ───────────────────── filters: explicit contradiction + allowlist ─────────────────────


def load_trial_prefixes_from_filtered_dir(filtered_dir: Path) -> Set[str]:
    prefixes: Set[str] = set()
    for csv_path in filtered_dir.glob("*.csv"):
        if "explicit_contradiction" not in csv_path.name:
            continue
        try:
            with csv_path.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    nct = (row.get("nct_id") or row.get("trial_id") or "").strip()
                    if nct:
                        prefixes.add(nct)
        except Exception as e:
            logging.error("[FILTER] Failed to read %s: %s", csv_path, e)
    logging.info("[FILTER] Loaded %d trial prefixes from %s", len(prefixes), filtered_dir)
    return prefixes


def load_trial_prefixes_from_allowlist(path: Path) -> Set[str]:
    prefixes: Set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            prefixes.add(s)
    except Exception as e:
        logging.error("[ALLOWLIST] Failed to read %s: %s", path, e)
    logging.info("[ALLOWLIST] Loaded %d prefixes from %s", len(prefixes), path)
    return prefixes


def _filter_by_prefixes(progs: List[IRProgram], prefixes: Set[str], label: str) -> List[IRProgram]:
    if not prefixes:
        return progs
    before = len(progs)
    progs = [p for p in progs if any(p.eff_tid.startswith(pref) for pref in prefixes)]
    logging.info("[%s] Filtered programs: %d → %d", label, before, len(progs))
    return progs


# ───────────────────── mbench writer ─────────────────────


def write_mbench_record(
    cfg: Config, prog: IRProgram, subctx: Dict[str, Any], prompt: Optional[str], raw: Optional[str]
) -> None:
    cid = subctx.get("cohort_id") or subctx.get("substudy_id") or "default"
    base = f"{prog.eff_tid}_{prog.side}_cohort-{cid}"

    if prompt is not None:
        (cfg.mbench_dir / f"{base}_prompt.txt").write_text(prompt, encoding="utf-8")
    if raw is not None:
        (cfg.mbench_dir / f"{base}_raw.txt").write_text(raw, encoding="utf-8")


# ───────────────────── running totals + prediction (main-process only) ─────────────────────


@dataclass
class RunningTotals:
    done: int = 0
    with_cost: int = 0
    total_cost_usd: float = 0.0

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


def _extract_cost(summary: Dict[str, Any]) -> Optional[float]:
    meta = summary.get("meta") or {}
    c = meta.get("estimated_cost_usd")
    if isinstance(c, (int, float)):
        return float(c)
    return None


def _print_progress_line(totals: RunningTotals, total_planned: int) -> None:
    unknown = totals.done - totals.with_cost
    avg = totals.avg_cost()
    pred_final = totals.predicted_final_cost(total_planned)
    pred_rem = totals.predicted_remaining_cost(total_planned)

    if pred_final is None or pred_rem is None:
        pred_part = "pred_total=(n/a) pred_remaining=(n/a)"
    else:
        pred_part = f"pred_total=${pred_final:.6f} pred_remaining=${pred_rem:.6f}"

    msg = (
        f"\r[RUNNING_COST] done={totals.done}/{total_planned} "
        f"cost_lines={totals.with_cost} unknown_cost_lines={unknown} "
        f"total_cost=${totals.total_cost_usd:.6f} avg_cost=${avg:.6f} "
        f"{pred_part}"
    )
    print(msg, end="", file=sys.stdout, flush=True)


def _commit_summary_and_update(
    out_f,
    summary: Dict[str, Any],
    totals: RunningTotals,
    total_planned: int,
    *,
    progress_every: int,
) -> None:
    out_f.write(json.dumps(summary) + "\n")
    out_f.flush()

    totals.done += 1
    c = _extract_cost(summary)
    if c is not None:
        totals.with_cost += 1
        totals.total_cost_usd += c

    if totals.done % max(1, progress_every) == 0 or totals.done == total_planned:
        _print_progress_line(totals, total_planned)


# ───────────────────── preflight prediction (no LLM calls) ─────────────────────


def _preflight_estimate_total_costs(
    cfg: Config,
    progs: List[IRProgram],
    prompt_template: Optional[str],
    assumed_completion_tokens: int,
    *,
    side: str = "exclusion",
) -> None:
    """
    Computes:
      - total_prompt_tokens over all (eligible) programs by building prompts and counting tokens
      - predicts total cost for gpt-5 / gpt-4.1 / gpt-4.1-mini using assumed completion tokens per call
    This is only a prediction; real completion length varies.
    """
    if not OPENAI_PRICING:
        logging.info("[PREDICT] skipping preflight (costing.py missing or pricing table unavailable)")
        return

    # count tokens via tiktoken
    dummy = SMTPolarityFixer(call_llm=None, prompt_template=prompt_template, model_name=MODEL_NAME)

    total_prompt_tokens = 0
    counted = 0
    skipped = 0

    for prog in progs:
        if prog.side != side:
            continue
        parent_ctx = load_snapshot_for_program(cfg.snapshot_dir, prog.eff_tid, prog.side)
        if parent_ctx is None:
            skipped += 1
            continue
        subctx = extract_subcohort_for_eff_tid(parent_ctx, prog.eff_tid, prog.side)
        try:
            smt_text = prog.path.read_text(encoding="utf-8")
        except Exception:
            skipped += 1
            continue

        try:
            # Use the same internal prompt-building logic as the fixer (no LLM call)
            pi = dummy._extract_inputs(subctx)          # type: ignore[attr-defined]
            prompt = dummy._build_prompt(pi, smt_text)  # type: ignore[attr-defined]
        except Exception:
            skipped += 1
            continue

        toks, note = count_tokens(prompt, MODEL_NAME)
        if toks is None:
            logging.info("[PREDICT] skipping preflight (tiktoken unavailable or tokenizer error: %s)", note)
            return

        total_prompt_tokens += toks
        counted += 1

    if counted == 0:
        logging.info("[PREDICT] preflight: no programs counted (skipped=%d).", skipped)
        return

    # Total completion tokens = assumed per call * counted
    total_completion_tokens = int(assumed_completion_tokens) * counted

    logging.info(
        "[PREDICT] preflight counted=%d skipped=%d total_prompt_tokens=%d assumed_completion_tokens_per_call=%d",
        counted,
        skipped,
        total_prompt_tokens,
        assumed_completion_tokens,
    )

    # Predict for the 3 models requested (and print current model too if different)
    models = ["gpt-5", "gpt-4.1", "gpt-4.1-mini"]
    if MODEL_NAME not in models:
        models.append(MODEL_NAME)

    for m in models:
        if m not in OPENAI_PRICING:
            continue
        est = estimate_cost_usd(m, total_prompt_tokens, total_completion_tokens)
        if est is None:
            continue
        logging.info(
            "[PREDICT] estimated_total_cost_usd model=%s calls=%d => $%.6f (prompt_toks=%d, completion_toks=%d)",
            m,
            counted,
            est,
            total_prompt_tokens,
            total_completion_tokens,
        )


# ───────────────────── per-program processing ─────────────────────


def process_program(
    cfg: Config, prog: IRProgram, fixer: SMTPolarityFixer, run_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if prog.side != "exclusion":
        return None

    parent_ctx = load_snapshot_for_program(cfg.snapshot_dir, prog.eff_tid, prog.side)
    if parent_ctx is None:
        return None

    subctx = extract_subcohort_for_eff_tid(parent_ctx, prog.eff_tid, prog.side)

    try:
        smt_text = prog.path.read_text(encoding="utf-8")
    except Exception as e:
        logging.error("Could not read SMT file %s: %s", prog.path, e)
        return None

    fixed_smt, meta = fixer.fix(subctx, smt_text)

    write_mbench_record(cfg, prog, subctx, getattr(fixer, "last_prompt", None), getattr(fixer, "last_raw", None))

    merged_meta: Dict[str, Any] = dict(run_meta)
    merged_meta.update(meta or {})

    summary: Dict[str, Any] = {
        "effective_trial_id": prog.eff_tid,
        "side": prog.side,
        "smt_path": str(prog.path),
        "meta": merged_meta,
        "cohort_id": subctx.get("cohort_id"),
        "cohort_label": subctx.get("cohort_label"),
    }

    if (merged_meta.get("executed") is True) and (merged_meta.get("no_change_required") is not True):
        out_path = cfg.out_ir_dir / f"{prog.eff_tid}_exclusion_program_polarity_fixed.smt2"
        try:
            out_path.write_text(fixed_smt, encoding="utf-8")
            summary["fixed_smt_path"] = str(out_path)
            logging.info("[OK] Polarity-fixed eff_tid=%s cohort=%s → %s", prog.eff_tid, subctx.get("cohort_id"), out_path)
        except Exception as e:
            logging.error("Failed to write fixed SMT %s: %s", out_path, e)
            summary.setdefault("meta", {})["write_error"] = str(e)

    return summary


# ───────────────────── worker globals ─────────────────────

_worker_cfg: Optional[Config] = None
_worker_fixer: Optional[SMTPolarityFixer] = None
_worker_run_meta: Optional[Dict[str, Any]] = None


def _silence_worker_terminal_logging() -> None:
    """
    Keep the terminal live running-cost line readable:
    - workers should NOT write to terminal
    - main process prints the running total line
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.CRITICAL)


def _worker_init(
    cfg: Config,
    azure_endpoint: str,
    model_name: str,
    prompt_template: Optional[str],
    run_meta: Dict[str, Any],
) -> None:
    global _worker_cfg, _worker_fixer, _worker_run_meta
    _worker_cfg = cfg
    _worker_run_meta = run_meta

    _silence_worker_terminal_logging()

    engine = AzureInferenceEngine(
        endpoint=azure_endpoint,
        api_key_env_var="OPENAI_API_KEY",
        model_name=model_name,
    )

    def call_llm(prompt: str) -> str:
        return engine(prompt)[0]

    _worker_fixer = SMTPolarityFixer(call_llm=call_llm, prompt_template=prompt_template, model_name=model_name)


def _worker_run(prog: IRProgram) -> Optional[Dict[str, Any]]:
    assert _worker_cfg is not None, "Worker cfg not initialized"
    assert _worker_fixer is not None, "Worker fixer not initialized"
    assert _worker_run_meta is not None, "Worker run_meta not initialized"
    return process_program(_worker_cfg, prog, _worker_fixer, _worker_run_meta)


# ───────────────────── CLI & main ─────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix polarity flips in exclusion IR SMT programs.")
    p.add_argument("--ir-dir", default="../build/ir", help="Directory containing *_program.smt2 files.")
    p.add_argument("--snapshot-dir", default="../subcohort_results", help="Directory with subcohort snapshots.")
    p.add_argument("--out-ir-dir", default="../build/ir_polarity_fixed", help="Directory to write polarity-fixed SMT programs.")
    p.add_argument("--summary-jsonl", default="../build/ir_polarity_fix_summary.jsonl", help="Path to JSONL summary.")
    p.add_argument("--log-dir", default="./polarity_run_logs", help="Directory for log files.")
    p.add_argument("--mbench-dir", default="polarity_mbench", help="Directory for mbench prompt/raw logs.")

    p.add_argument("--trial-prefix", default=None, help="Only process eff_tid starting with this prefix.")
    p.add_argument("--trial-allowlist", default=None, help="Path to allowlist file (one NCT prefix per line).")
    p.add_argument("--filtered-explicit-contradiction-dir", default=None, help="Restrict to explicit_contradiction CSVs dir.")

    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers.")

    # Running-cost display cadence
    p.add_argument("--progress-every", type=int, default=1, help="Update running-cost line every N completed programs.")

    # Prediction knobs
    p.add_argument(
        "--preflight-predict",
        action="store_true",
        help="Run a preflight prompt-token count (no LLM calls) and print predicted total cost for gpt-5/gpt-4.1/gpt-4.1-mini.",
    )
    p.add_argument(
        "--assumed-completion-tokens",
        type=int,
        default=1200,
        help="For preflight prediction only: assume this many completion tokens per call.",
    )

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging (DEBUG).")
    return p.parse_args()


def setup_logging(cfg: Config, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = cfg.log_dir / f"polarity_fix_ir_{ts}.log"

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info("Logging to %s", log_path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    cfg = Config(
        repo_root=repo_root,
        ir_dir=Path(args.ir_dir),
        snapshot_dir=Path(args.snapshot_dir),
        out_ir_dir=Path(args.out_ir_dir),
        summary_jsonl=Path(args.summary_jsonl),
        log_dir=Path(args.log_dir),
        mbench_dir=Path(args.mbench_dir),
    )
    setup_logging(cfg, args.verbose)

    azure_endpoint = os.environ.get("OPENAI_ENDPOINT", "")
    if not azure_endpoint:
        raise EnvironmentError("OPENAI_ENDPOINT is required for polarity_fix_irs.py")

    prompt_template: Optional[str] = None
    prompt_path = repo_root / "prompt" / "smt_polarity_fix.prompt"
    if prompt_path.exists():
        prompt_template = prompt_path.read_text(encoding="utf-8")
        logging.info("Loaded polarity-fix prompt template from %s", prompt_path)
    else:
        logging.info("No prompt template found at %s; using built-in prompt.", prompt_path)

    progs = discover_ir_programs(cfg, trial_prefix=None, side_filter="exclusion")

    if args.trial_prefix:
        progs = [p for p in progs if p.eff_tid.startswith(args.trial_prefix)]
        logging.info("[PREFIX] Remaining programs after --trial-prefix: %d", len(progs))

    if args.trial_allowlist is not None:
        al = Path(args.trial_allowlist)
        if not al.is_absolute():
            al = (repo_root / al).resolve()
        allow_prefixes = load_trial_prefixes_from_allowlist(al)
        progs = _filter_by_prefixes(progs, allow_prefixes, "ALLOWLIST")

    if args.filtered_explicit_contradiction_dir is not None:
        filtered_dir = Path(args.filtered_explicit_contradiction_dir)
        if not filtered_dir.is_absolute():
            filtered_dir = (repo_root / filtered_dir).resolve()
        bad_prefixes = load_trial_prefixes_from_filtered_dir(filtered_dir)
        progs = _filter_by_prefixes(progs, bad_prefixes, "EXPL_CONTRAD")

    total_planned = len(progs)
    logging.info("Discovered %d exclusion IR programs under %s after filtering", total_planned, cfg.ir_dir)

    max_workers = max(1, int(args.max_workers))

    run_meta: Dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "model_name": MODEL_NAME,
        "max_workers": max_workers,
        "pricing_basis": "openai_api_per_1m_token_list_prices",
        "pricing_url": "https://platform.openai.com/docs/pricing",
        "cost_note": "Estimated via local token counting when tiktoken is available; API usage would be more authoritative.",
    }

    # Optional preflight prediction (no LLM calls)
    if args.preflight_predict:
        _preflight_estimate_total_costs(
            cfg,
            progs,
            prompt_template,
            int(args.assumed_completion_tokens),
        )

    totals = RunningTotals()
    _print_progress_line(totals, total_planned)

    with cfg.summary_jsonl.open("w", encoding="utf-8") as out_f:
        if max_workers == 1:
            engine = AzureInferenceEngine(
                endpoint=azure_endpoint,
                api_key_env_var="OPENAI_API_KEY",
                model_name=MODEL_NAME,
            )

            def call_llm(prompt: str) -> str:
                return engine(prompt)[0]

            fixer = SMTPolarityFixer(call_llm=call_llm, prompt_template=prompt_template, model_name=MODEL_NAME)

            for prog in progs:
                summary = process_program(cfg, prog, fixer, run_meta)
                if summary is None:
                    continue
                _commit_summary_and_update(
                    out_f,
                    summary,
                    totals,
                    total_planned,
                    progress_every=max(1, int(args.progress_every)),
                )
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_worker_init,
                initargs=(cfg, azure_endpoint, MODEL_NAME, prompt_template, run_meta),
            ) as executor:
                future_to_prog = {executor.submit(_worker_run, prog): prog for prog in progs}
                for fut in as_completed(future_to_prog):
                    prog = future_to_prog[fut]
                    try:
                        summary = fut.result()
                    except Exception as e:
                        logging.error("Worker failed for eff_tid=%s side=%s: %s", prog.eff_tid, prog.side, e)
                        continue
                    if summary is None:
                        continue
                    _commit_summary_and_update(
                        out_f,
                        summary,
                        totals,
                        total_planned,
                        progress_every=max(1, int(args.progress_every)),
                    )

    # finalize live line
    print("", file=sys.stdout, flush=True)

    unknown_cost = totals.done - totals.with_cost
    avg_cost = totals.avg_cost()
    pred_final = totals.predicted_final_cost(total_planned)

    logging.info("Finished. Processed %d/%d IR programs. Summary → %s", totals.done, total_planned, cfg.summary_jsonl)
    logging.info("[SUMMARY] engine=%s model=%s max_workers=%d", ENGINE_VERSION, MODEL_NAME, max_workers)
    logging.info("[SUMMARY] total_cost=$%.6f cost_lines=%d unknown_cost_lines=%d avg_cost=$%.6f",
                 totals.total_cost_usd, totals.with_cost, unknown_cost, avg_cost)

    if pred_final is not None:
        logging.info("[SUMMARY] predicted_final_total_cost_from_observed_avg=$%.6f", pred_final)
    else:
        logging.info("[SUMMARY] predicted_final_total_cost_from_observed_avg=(n/a; no cost lines yet)")

    if totals.with_cost == 0:
        logging.info("[SUMMARY] Tip: install tiktoken so costs can be estimated.")
        logging.info("          pip install tiktoken")


if __name__ == "__main__":
    main()
