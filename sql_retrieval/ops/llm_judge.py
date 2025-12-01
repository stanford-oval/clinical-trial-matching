#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_judge.py — Batch judge over kit-dirs.

Now supports:
  - Multi-patient mode (default): loops through all patient folders under --patients-root (defaults to ./out_sigir_kits_threeway)
  - Single-patient mode: process exactly one patient via --patient-root
  - Subfolder filtering: use --only-folder <name> (repeatable) to restrict to specific subfolders (e.g., O_original_survivors)

STRICT ascending processing by rank (within each patient):
  rank1_* -> rank2_* -> ... -> unranked (last)

Failure handling:
  - Retries each LLM call (relevance/eligibility) with exponential backoff.
  - On every failed attempt, writes debug JSON with prompts + raw response.

Prompts:
  - Relevance:   ./prompt/LLMJudgeRelevance.prompt
  - Eligibility: ./prompt/LLMJudgeEligibility.prompt
"""

from __future__ import annotations
import argparse, json, os, re, sys, time, logging, random, traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable, TypeVar
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Quiet noisy HTTP logs unless DEBUG
os.environ.setdefault("AZURE_HTTP_LOGGING_DISABLE", "1")

# ----------------------------
# Engine/model resolution
# ----------------------------

from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
if ENGINE_VERSION == "gpt-5":
    from smt_core.inference_engine_5 import AzureInferenceEngine  # type: ignore
else:
    from smt_core.inference_engine import AzureInferenceEngine

def _resolve_endpoint(cli_endpoint: Optional[str]) -> str:
    return (cli_endpoint or os.environ.get("OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()

# ----------------------------
# Small utils
# ----------------------------

def _canon8(nct: str) -> str:
    m = re.match(r"^(NCT\d{8})", nct or "")
    return m.group(1) if m else (nct or "")

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _find_kit_files(kit_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    pn = kit_dir / "0patient_note" / "patient_note.json"
    cp = kit_dir / "1trial" / "corpus" / "corpus.json"
    return (pn if pn.exists() else None, cp if cp.exists() else None)

def _is_kit_dir(p: Path) -> bool:
    pn, cp = _find_kit_files(p)
    return bool(pn and cp)

def _infer_ids_from_kit(kit_dir: Path, pn_obj: Dict[str, Any], cp_obj: Dict[str, Any]) -> Tuple[str, str]:
    patient_id = str(pn_obj.get("_id") or pn_obj.get("patient_id") or "").strip()
    nct_hint = str(cp_obj.get("_id") or "").strip()
    if not nct_hint:
        m = re.search(r"NCT\d{8}", str(kit_dir))
        if m:
            nct_hint = m.group(0)
    nct8 = _canon8(nct_hint)
    return patient_id, nct8

def _save_to_kit(kit_dir: Path, engine: str, model: str, payload: Dict[str, Any]) -> Path:
    out_dir = kit_dir / "1trial" / "llm_judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judgment_{engine}_{model}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def extract_trial_text(doc: Dict[str, Any], fallback_title: str = "") -> str:
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    title = str(doc.get("title") or meta.get("brief_title") or fallback_title or doc.get("_id") or "").strip()
    brief_summary = str(meta.get("brief_summary") or "").strip()
    inclusion = str(meta.get("inclusion_criteria") or "").strip()
    exclusion = str(meta.get("exclusion_criteria") or "").strip()
    full_text = str(doc.get("text") or "").strip()
    parts: List[str] = []
    if title: parts.append(f"# {title}")
    if brief_summary: parts.append("## Brief Summary\n" + brief_summary)
    if inclusion: parts.append("## Inclusion Criteria\n" + inclusion)
    if exclusion: parts.append("## Exclusion Criteria\n" + exclusion)
    if full_text and (not brief_summary or not ("Inclusion" in full_text[:80] or "Exclusion" in full_text[:80])):
        parts.append("## Full Text\n" + full_text)
    return "\n\n".join(parts).strip()

# ----------------------------
# Prompts
# ----------------------------

RELEVANCE_SYSTEM = (
    "You are a careful clinical NLP analyst. Compare patient context to clinical-trial text conservatively, cite exact snippets, and return strict JSON."
)
ELIGIBILITY_SYSTEM = (
    "You are a clinical trial screening assistant. Decide eligibility conservatively based only on provided text; do not assume missing facts. Return strict JSON."
)

RELEVANCE_PROMPT_PATH = Path("./prompt/LLMJudgeRelevance.prompt")
ELIGIBILITY_PROMPT_PATH = Path("./prompt/LLMJudgeEligibility.prompt")

def _read_prompt_or_die(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")

def _format_prompt(tmpl: str, *, trial_text: str, patient_note_text: str) -> str:
    s = tmpl
    s = s.replace("#TRIAL#", trial_text)
    s = s.replace("#PATIENTNOTE#", patient_note_text)
    return s

_ELIG_OUTCOME_CANON = {
    "immediately eligible": "immediately eligible",
    "almost eligible (high-probability completion)": "almost eligible (high-probability completion)",
    "potentially eligible (needs more info)": "potentially eligible (needs more info)",
    "eliminated": "eliminated",
    "not relevant": "not relevant",
}

def _normalize_elig_outcome(val: Any, eligibility_obj: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(val, str):
        val = str(val or "")
    v = val.strip().lower()

    if v in {"1", "immediately eligible", "immediately-eligible", "immediate eligible"}:
        return "immediately eligible"
    if v in {"2", "almost eligible", "almost eligible (high probability completion)", "almost eligible (high-probability completion)", "high-probability completion"}:
        return "almost eligible (high-probability completion)"
    if v in {"3", "potentially eligible", "potentially eligible (needs more info)", "needs more info", "potentially-eligible"}:
        return "potentially eligible (needs more info)"
    if v in {"4", "eliminated", "excluded", "ineligible due to exclusion"}:
        return "eliminated"
    if v in {"5", "not relevant", "irrelevant"}:
        return "not relevant"

    if v in {"fit"}:
        return "immediately eligible"
    if v in {"unclear"}:
        return "potentially eligible (needs more info)"
    if v in {"mismatch"}:
        bc = []
        if isinstance(eligibility_obj, dict):
            bc = eligibility_obj.get("blocking_criteria") or []
        has_explicit_exclusion = any((isinstance(x, dict) and str(x.get("type","")).lower() == "exclusion") for x in bc)
        return "eliminated" if has_explicit_exclusion else "not relevant"

    for k in _ELIG_OUTCOME_CANON:
        if v == k:
            return k
    return "potentially eligible (needs more info)"

# ----------------------------
# Logging
# ----------------------------

logger = logging.getLogger("llm_judge")
_JSONL_FP = None
_JSONL_LOCK = threading.Lock()

def log_event(event: str, **fields: Any) -> None:
    global _JSONL_FP
    lvl = fields.pop("_level", "INFO").upper()
    msg = fields.pop("_msg", "")
    if lvl == "DEBUG":
        logger.debug("%s%s", f"[{event}] " if event else "", msg)
    elif lvl == "WARNING":
        logger.warning("%s%s", f"[{event}] " if event else "", msg)
    elif lvl == "ERROR":
        logger.error("%s%s", f"[{event}] " if event else "", msg)
    else:
        logger.info("%s%s", f"[{event}] " if event else "", msg)
    if _JSONL_FP:
        payload = {"event": event, **fields}
        payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            with _JSONL_LOCK:
                _JSONL_FP.write(json.dumps(payload, ensure_ascii=False) + "\n")
                _JSONL_FP.flush()
        except Exception:
            pass

# ----------------------------
# LLM driver + retry + failure dumps
# ----------------------------

def _load_engine(cli_endpoint: Optional[str], fallback_endpoint: Optional[str], verbose: bool=False) -> AzureInferenceEngine:
    endpoint = _resolve_endpoint(cli_endpoint)
    if not endpoint:
        raise SystemExit("Azure endpoint required: pass --endpoint or set OPENAI_ENDPOINT / AZURE_OPENAI_ENDPOINT.")
    return AzureInferenceEngine(endpoint=endpoint, fallback_endpoint=fallback_endpoint, model_name=MODEL_NAME, verbose=verbose)

class LLMCallError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str = "", system_prompt: str = "", user_prompt: str = "", tb: str = ""):
        super().__init__(message)
        self.raw_response = raw_response
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.traceback = tb

def call_llm_json(engine_obj: AzureInferenceEngine, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
    """Return parsed JSON; on failure raise LLMCallError with prompts + raw content attached."""
    from azure.ai.inference.models import SystemMessage, UserMessage
    try:
        content = engine_obj.run(
            messages=[SystemMessage(content=system_prompt), UserMessage(content=user_prompt)],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.001,
        )
        raw = (content or "").strip()
        m = re.search(r"\{[\s\S]*\}\s*$", raw)
        text = m.group(0) if m else raw
        return json.loads(text)
    except Exception as e:
        raise LLMCallError(
            f"LLM parse/transport error: {e}",
            raw_response=(raw if 'raw' in locals() else ""),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tb=traceback.format_exc()
        )

T = TypeVar("T")

def write_failure_dump(debug_dir: Path, op_name: str, attempt_idx: int, err: LLMCallError) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    dump = {
        "op": op_name,
        "attempt": attempt_idx + 1,
        "error": str(err),
        "traceback": err.traceback,
        "system_prompt": err.system_prompt,
        "user_prompt": err.user_prompt,
        "raw_response": err.raw_response,
    }
    fp = debug_dir / f"failed_{op_name}_attempt{attempt_idx+1}.json"
    fp.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp

def with_retries(op_name: str, func: Callable[[], T], retries: int, wait: float,
                 debug_dir: Path) -> T:
    attempt = 0
    while True:
        try:
            return func()
        except LLMCallError as e:
            dump_path = write_failure_dump(debug_dir, op_name, attempt, e)
            raw_preview = (e.raw_response or "").replace("\n"," ")[:200]
            log_event(
                "retry",
                _level="WARNING",
                _msg=(f"{op_name} failed attempt {attempt+1}: {e}. "
                      f"raw[0:200]={raw_preview!r} -> dump={dump_path}"),
                op=op_name, attempt=attempt+1, dump=str(dump_path)
            )
            if attempt >= retries:
                raise
            backoff = wait * (2 ** attempt)
            jitter = backoff * random.uniform(-0.2, 0.2)
            time.sleep(max(0.5, backoff + jitter))
            attempt += 1

# ----------------------------
# STRICT rank-ordered walking (per patient)
# ----------------------------

_RANK_RE = re.compile(r"(?i)^rank(\d+)\b")

def _parse_rank(name: str) -> Optional[int]:
    m = _RANK_RE.match(name)
    return int(m.group(1)) if m else None

def _rank_roots_strict(root: Path) -> List[Tuple[int, Path]]:
    """
    Find all directories named rank\\d+ anywhere under `root`, not just as direct children.
    Supports layouts like:
      <patient> / rank1_*                       (old)
      <patient> / O_original_default / rank1_*  (new)
      <patient> / */*/rank2_*                   (arbitrary nesting)
    """
    pairs: List[Tuple[int, Path]] = []
    if not root.is_dir():
        return pairs
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        r = _parse_rank(d.name)
        if r is not None:
            pairs.append((r, d))
    pairs.sort(key=lambda x: x[0])
    return pairs

def _kits_under(dir_: Path) -> List[Path]:
    found: List[Path] = []
    for d in dir_.rglob("*"):
        if d.is_dir() and _is_kit_dir(d):
            found.append(d)
    found.sort(key=lambda p: str(p).lower())
    return found

def _ordered_kits(patient_root: Path) -> List[Path]:
    ordered: List[Path] = []
    rank_roots = _rank_roots_strict(patient_root)
    ranked_paths = set()
    for _, rank_root in rank_roots:
        kits = _kits_under(rank_root)
        ordered.extend(kits)
        for k in kits:
            ranked_paths.add(k)

    unranked: List[Path] = []
    for d in patient_root.rglob("*"):
        if d.is_dir() and _is_kit_dir(d) and d not in ranked_paths:
            unranked.append(d)
    unranked.sort(key=lambda p: str(p).lower())
    ordered.extend(unranked)
    return ordered

def _group_kits_by_rank(patient_root: Path) -> List[Tuple[Optional[int], List[Path]]]:
    groups: List[Tuple[Optional[int], List[Path]]] = []
    for r, rank_root in _rank_roots_strict(patient_root):
        kits = _kits_under(rank_root)
        groups.append((r, kits))
    ranked_set = {p for _, ks in groups for p in ks}
    unranked = []
    for d in patient_root.rglob("*"):
        if d.is_dir() and _is_kit_dir(d) and d not in ranked_set:
            unranked.append(d)
    unranked.sort(key=lambda p: str(p).lower())
    if unranked:
        groups.append((None, unranked))
    return groups

# ----------------------------
# Helpers for multi-patient mode
# ----------------------------

def _has_any_kits(dir_: Path) -> bool:
    if not dir_.is_dir():
        return False
    try:
        for d in dir_.rglob("*"):
            if d.is_dir() and _is_kit_dir(d):
                return True
    except Exception:
        pass
    return False

def _process_one_patient(args, patient_root: Path,
                         relevance_tmpl: str, eligibility_tmpl: str,
                         only_folders: Optional[List[str]] = None
                         ) -> Dict[str, Any]:
    # If restricted subfolder(s) are provided, build groups from those; else use rank grouping
    if only_folders:
        groups: List[Tuple[Optional[int], List[Path]]] = []
        for name in only_folders:
            sub = patient_root / name
            if not sub.is_dir():
                log_event("scan.missing_subdir",
                          _level="WARNING",
                          _msg=f"Subfolder '{name}' not found under {patient_root}",
                          patient_root=str(patient_root), subfolder=name)
                continue
            kits = _kits_under(sub)
            groups.append((None, kits))
        if not groups or all(not ks for _, ks in groups):
            log_event("scan.empty",
                      _level="ERROR",
                      _msg=f"No kit-dirs under requested subfolder(s) for {patient_root}",
                      patient_root=str(patient_root), only_folders=",".join(only_folders))
            raise SystemExit(f"No kit-dirs found under requested subfolder(s) of: {patient_root}")
    else:
        groups = _group_kits_by_rank(patient_root)

    if not groups or all(not ks for _, ks in groups):
        log_event("scan.empty",
                  _level="ERROR",
                  _msg=f"No kit-dirs under {patient_root}",
                  patient_root=str(patient_root))
        raise SystemExit(f"No kit-dirs found under: {patient_root}")

    total_summary: List[Dict[str, Any]] = []
    processed = skipped = errors = 0

    for rank, kits in groups:
        if not kits:
            continue
        label = f"rank{rank}" if rank is not None else "unranked"
        log_event("rank.start", _msg=f"Processing {label} with {len(kits)} kits (workers={args.workers})",
                  rank=rank if rank is not None else "unranked", count=len(kits), workers=args.workers)

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            future_map = {
                ex.submit(
                    _process_kit,
                    kd,
                    engine_version=ENGINE_VERSION,
                    model_name=MODEL_NAME,
                    cli_endpoint=args.endpoint,
                    fallback_endpoint=args.fallback_endpoint,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                    retry_wait=args.retry_wait,
                    overwrite=args.overwrite,
                    log_skips=args.log_skips,
                    verbose=args.verbose,
                    relevance_tmpl=relevance_tmpl,
                    eligibility_tmpl=eligibility_tmpl,
                ): kd
                for kd in kits
            }
            for fut in as_completed(future_map):
                res = fut.result()
                results.append(res)

        results.sort(key=lambda r: r.get("kit_dir","").lower())
        total_summary.extend(results)

        for r in results:
            st = r.get("status","")
            if st.startswith("ok"):
                processed += 1
            elif st.startswith("skipped"):
                skipped += 1
            else:
                errors += 1

        log_event("rank.finish",
                  _msg=(f"Finished {label}: ok={sum(1 for r in results if r.get('status','').startswith('ok'))} "
                        f"skipped={sum(1 for r in results if r.get('status','').startswith('skipped'))} "
                        f"errors={sum(1 for r in results if not r.get('status','').startswith(('ok','skipped')))}"),
                  rank=rank if rank is not None else "unranked")

    return {
        "engine": ENGINE_VERSION,
        "model": MODEL_NAME,
        "patient_root": str(patient_root),
        "count": len(total_summary),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "trials": total_summary,
    }

# ----------------------------
# Worker function (per kit)
# ----------------------------

def _process_kit(
    kd: Path,
    *,
    engine_version: str,
    model_name: str,
    cli_endpoint: Optional[str],
    fallback_endpoint: Optional[str],
    max_tokens: int,
    retries: int,
    retry_wait: float,
    overwrite: bool,
    log_skips: bool,
    verbose: bool,
    relevance_tmpl: str,
    eligibility_tmpl: str,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        pn_p, cp_p = _find_kit_files(kd)
        assert pn_p and cp_p
        pn_obj = _load_json(pn_p)
        cp_obj = _load_json(cp_p)
        patient_id, nct8 = _infer_ids_from_kit(kd, pn_obj, cp_obj)
        if not patient_id or not nct8:
            raise RuntimeError("missing ids")

        out_dir = kd / "1trial" / "llm_judge"
        out_path = out_dir / f"judgment_{engine_version}_{model_name}.json"
        if out_path.exists() and not overwrite:
            (log_event("trial.skip", _level="INFO" if log_skips else "DEBUG",
                       _msg=f"Skip {kd.name}: exists -> {out_path}"))
            return {
                "kit_dir": str(kd),
                "patient_id": patient_id,
                "nct": nct8,
                "status": "skipped (exists)",
                "saved_path": str(out_path),
            }

        log_event("trial.start", _msg=f"{kd.name} | patient={patient_id} nct={nct8}")

        patient_note_text = str(pn_obj.get("text") or "").strip()
        trial_text = extract_trial_text(cp_obj, fallback_title=nct8)
        debug_dir = out_dir / "debug"

        engine_obj = _load_engine(cli_endpoint, fallback_endpoint, verbose=verbose)

        # Relevance
        rel_user = _format_prompt(relevance_tmpl, trial_text=trial_text, patient_note_text=patient_note_text)
        relevance = with_retries(
            "relevance",
            lambda: call_llm_json(engine_obj, RELEVANCE_SYSTEM, rel_user, max_tokens=max_tokens),
            retries=retries,
            wait=retry_wait,
            debug_dir=debug_dir
        )
        rel_score = relevance.get("score")

        # Eligibility
        elg_user = _format_prompt(eligibility_tmpl, trial_text=trial_text, patient_note_text=patient_note_text)
        eligibility = with_retries(
            "eligibility",
            lambda: call_llm_json(engine_obj, ELIGIBILITY_SYSTEM, elg_user, max_tokens=max_tokens),
            retries=retries,
            wait=retry_wait,
            debug_dir=debug_dir
        )
        elg_outcome = _normalize_elig_outcome(eligibility.get("outcome"), eligibility)
        eligibility["outcome"] = elg_outcome

        out = {
            "engine": engine_version,
            "model": model_name,
            "patient_id": patient_id,
            "nct": nct8,
            "relevance": relevance,
            "eligibility": eligibility,
        }
        saved = _save_to_kit(kd, engine_version, model_name, out)
        elapsed = time.time() - t0

        log_event("trial.result",
                  _msg=f"Saved -> {saved} | relevance={rel_score} eligibility={elg_outcome} ({elapsed:.2f}s)",
                  kit_dir=str(kd), saved_path=str(saved),
                  relevance_score=rel_score, eligibility_outcome=elg_outcome, elapsed_s=round(elapsed,3))

        return {
            "kit_dir": str(kd),
            "patient_id": patient_id,
            "nct": nct8,
            "status": "ok",
            "saved_path": str(saved),
            "relevance_score": rel_score,
            "eligibility_outcome": elg_outcome,
            "elapsed_s": round(elapsed,3),
        }
    except Exception as e:
        elapsed = time.time() - t0
        log_event("trial.error", _level="ERROR", _msg=f"{kd.name} failed: {e} ({elapsed:.2f}s)")
        return {"kit_dir": str(kd), "status": f"error: {e}"}

# ----------------------------
# CLI main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch LLM judge across kit-dirs (supports multi-patient).")

    # Single-patient (optional) OR multi-patient (default path)
    ap.add_argument("--patient-root", type=Path, default=None,
                    help="Process exactly one patient directory (e.g., out_sigir_kits_threeway/sigir-20141). If omitted, multi-patient mode is used.")
    ap.add_argument("--patients-root", type=Path, default=Path("./out_sigir_kits_threeway"),
                    help="Directory containing many patient folders (default: ./out_sigir_kits_threeway)")

    # New: restrict to specific subfolder names under each patient (repeatable)
    ap.add_argument(
        "--only-folder",
        dest="only_folders",
        action="append",
        default=None,
        help="Only scan kits under subfolder(s) of each patient (exact name). "
             "Repeatable, e.g., --only-folder O_original_survivors"
    )

    ap.add_argument("--endpoint", default=None, help="Azure endpoint; else $OPENAI_ENDPOINT/$AZURE_OPENAI_ENDPOINT")
    ap.add_argument("--fallback-endpoint", default=None)
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if a judgment file already exists")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--verbose", action="store_true")
    # Logging knobs
    ap.add_argument("--log-file", type=Path, default=None, help="Write human-readable logs to this file")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Log verbosity")
    ap.add_argument("--quiet", action="store_true", help="Silence console logs (file/JSONL still recorded)")
    ap.add_argument("--jsonl-log", type=Path, default=None, help="Write structured events to this JSONL file")
    ap.add_argument("--log-skips", action="store_true", help="Emit a line for each skipped trial (default: aggregate only)")
    # Retry knobs
    ap.add_argument("--retries", type=int, default=2, help="Number of retries per LLM call on failure")
    ap.add_argument("--retry-wait", type=float, default=3.0, help="Base wait seconds for exponential backoff")
    # Parallelism
    ap.add_argument("--workers", type=int, default=16,
                    help="Max concurrent kits per rank (default: 16). Use 1 to run serially.")

    args = ap.parse_args()

    # Configure logging
    handlers: List[logging.Handler] = []
    if not args.quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handlers.append(console)
    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handlers.append(fh)
    logging.basicConfig(level=getattr(args, "log_level"), handlers=handlers or None)
    if args.log_level != "DEBUG":
        for name in ("azure", "azure.core", "azure.ai", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)

    # Open JSONL if requested
    global _JSONL_FP
    if args.jsonl_log:
        args.jsonl_log.parent.mkdir(parents=True, exist_ok=True)
        _JSONL_FP = args.jsonl_log.open("a", encoding="utf-8")

    log_event("run.start",
              _msg=f"Starting; engine={ENGINE_VERSION}, model={MODEL_NAME}",
              engine=ENGINE_VERSION, model=MODEL_NAME,
              log_level=args.log_level, max_tokens=args.max_tokens, overwrite=args.overwrite)

    # Load external prompts once
    relevance_tmpl = _read_prompt_or_die(RELEVANCE_PROMPT_PATH)
    eligibility_tmpl = _read_prompt_or_die(ELIGIBILITY_PROMPT_PATH)

    # If a specific patient is provided, run single-patient mode
    if args.patient_root:
        patient_root = args.patient_root
        if not patient_root.is_dir():
            if _JSONL_FP:
                try: _JSONL_FP.close()
                except Exception: pass
            raise SystemExit(f"Not a directory: {patient_root}")
        result = _process_one_patient(args, patient_root, relevance_tmpl, eligibility_tmpl,
                                      only_folders=args.only_folders)
        log_event("run.finish", _msg=f"Done. processed={result['processed']} skipped={result['skipped']} errors={result['errors']}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if _JSONL_FP:
            try: _JSONL_FP.close()
            except Exception: pass
        return

    # Default: multi-patient mode using --patients-root
    pr = args.patients_root
    if not pr.is_dir():
        if _JSONL_FP:
            try: _JSONL_FP.close()
            except Exception: pass
        raise SystemExit(f"Not a directory: {pr}")

    # Gather patient dirs that actually contain kits, optionally gated by requested subfolders
    patient_dirs: List[Path] = []
    for child in sorted(pr.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if args.only_folders:
            has = False
            for name in args.only_folders:
                sub = child / name
                if sub.is_dir() and _has_any_kits(sub):
                    has = True
                    break
            if has:
                patient_dirs.append(child)
        else:
            if _has_any_kits(child):
                patient_dirs.append(child)

    if not patient_dirs:
        log_event("scan.empty", _level="ERROR",
                  _msg=f"No patient folders with kits under {pr}", patients_root=str(pr))
        if _JSONL_FP:
            try: _JSONL_FP.close()
            except Exception: pass
        raise SystemExit(f"No kit-dirs found under any children of: {pr}")

    all_results: List[Dict[str, Any]] = []
    totals = {"processed": 0, "skipped": 0, "errors": 0, "patients": 0}

    for patient_root in patient_dirs:
        log_event("patient.start", _msg=f"Processing patient folder {patient_root.name}",
                  patient_root=str(patient_root))
        res = _process_one_patient(args, patient_root, relevance_tmpl, eligibility_tmpl,
                                   only_folders=args.only_folders)
        all_results.append(res)
        totals["processed"] += res.get("processed", 0)
        totals["skipped"] += res.get("skipped", 0)
        totals["errors"] += res.get("errors", 0)
        totals["patients"] += 1
        log_event("patient.finish", _msg=f"Done {patient_root.name}: ok={res.get('processed',0)} "
                                         f"skipped={res.get('skipped',0)} errors={res.get('errors',0)}")

    out = {
        "engine": ENGINE_VERSION,
        "model": MODEL_NAME,
        "patients_root": str(pr),
        "patients": all_results,
        "totals": totals,
    }
    log_event("run.finish", _msg=f"All patients done. {totals}")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if _JSONL_FP:
        try: _JSONL_FP.close()
        except Exception: pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_event("run.abort", _level="WARNING", _msg="Aborted by user (SIGINT)")
        sys.exit(130)
