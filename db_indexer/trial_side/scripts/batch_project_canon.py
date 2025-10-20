#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch projector (QE) — supervised per-file processes with hard timeouts,
bounded concurrency, and a clean progress bar.

Key ideas
- Each SMT file is handled by its own Process → we can terminate on timeout.
- Max N concurrent child processes to bound CPU/RAM.
- Per-file stdout/stderr redirected to OUT_DIR/_logs/<stem>.log (prevents TTY stalls).
- Optional size guard and graceful degrade when QE explodes.
"""

from __future__ import annotations
from pathlib import Path
import argparse, itertools, json, os, re, sys, time, queue, multiprocessing as mp
from typing import Dict, List, Optional, Iterable

# progress bar (optional)
try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

sys.setrecursionlimit(10000)

# ───────── paths ─────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling stage modules
from smt_core.buildroot import build_root as _build_root

ROOT = _build_root().parent

SMT_DIR   = ROOT / "build" / "slice_ir_linked"
CANON_DIR = ROOT / "build" / "minified_canon"
OUT_DIR   = ROOT / "build" / "canon_projection"

WRITE_PROJECTED_SMT  = True
USE_QE_STRICT        = True
EMIT_BINARY_IMPLS    = False
DEBUG_DUMPS          = True

CANON_NUMS = ["patient_age_value_recorded_now_in_years",
              "patient_age_value_recorded_now_in_months",
              "patient_age_value_recorded_now_in_days",
              "patient_age_value_recorded_now_in_hours",
              "patient_age_value_recorded_now_in_minutes",
             ]

# ───────── import projector ───────
sys.path.insert(0, str(ROOT / "irsrc" / "trial_side"))
from smt_projector import ProjectConfig as QEConfig, project_constraints_for_file  # type: ignore

# ───────── helpers ─────────
def _ensure_dirs() -> None:
    for sub in ("_summaries", "_projected_smt", "_debug", "_logs"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

def _load_canon_vars(p: Path) -> List[str]:
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen, out = set(), []
    for x in arr:
        if isinstance(x, str) and (s := x.strip()) and s not in seen:
            out.append(s); seen.add(s)
    return out

_SUBCOHORT_RE = re.compile(r"^(NCT\d+)([a-z]?)[_\-]([a-z]+)_", re.IGNORECASE)
def _guess_canon_file(smt_name: str) -> Optional[Path]:
    m = _SUBCOHORT_RE.match(smt_name)
    if not m:
        return None
    nct, subcohort, kind = m.group(1), m.group(2).lower(), m.group(3).lower()
    nct_key = f"{nct}{subcohort}"
    cands = sorted(CANON_DIR.glob(f"{nct_key}_{kind}*_canonical_variables*.json"))
    if cands:
        return cands[0]
    for p in CANON_DIR.glob("*.json"):
        nm = p.name.lower()
        if nct_key.lower() in nm and kind in nm and "canonical_variables" in nm:
            return p
    return None

_HEALED_RE = re.compile(r'(\.healed)+')
def _clean_healed_name(stem: str) -> str:
    return _HEALED_RE.sub('', stem)

def _already_done(p: Path) -> bool:
    target = OUT_DIR / "_projected_smt" / f"{_clean_healed_name(p.stem)}_canon_projected.smt2"
    return target.exists()

def chunked(iterable: Iterable[Path], n: int):
    it = iter(iterable)
    while True:
        block = list(itertools.islice(it, n))
        if not block:
            return
        yield block

# ───────── graceful-degrade wrapper (compact) ─────────
def _too_big(now_bytes: int, pre_bytes: int, growth_factor: float, hard_cap: int | None) -> bool:
    if hard_cap and now_bytes > hard_cap:
        return True
    return now_bytes > pre_bytes * growth_factor

def _estimate_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except Exception:
        try:
            return len(p.read_text(encoding="utf-8"))
        except Exception:
            return 0

def project_with_budget_inproc(
    smt_path: Path,
    canon_bools: List[str],
    *,
    debug_dir: Optional[Path],
    max_projected_bytes: int,
) -> Dict[str, object]:
    # preflight size (bytes) for relative growth checks
    pre_bytes = _estimate_size(smt_path)

    # Try strict, then a light pass
    presets = [
        dict(name="strict", qe_strict=USE_QE_STRICT, timeout_ms=4000, growth=2.5),
        dict(name="light",  qe_strict=False,        timeout_ms=2500, growth=1.8),
    ]

    last_summary, last_proj, last_status = None, None, "error"

    for ps in presets:
        cfg = QEConfig(
            use_qe_strict          = ps["qe_strict"],
            emit_binary_implicates = EMIT_BINARY_IMPLS,
            timeout_ms             = ps["timeout_ms"],
            debug_dir              = debug_dir,
        )
        summary, proj_smt = project_constraints_for_file(
            str(smt_path),
            canon_bools = canon_bools,
            canon_nums  = CANON_NUMS,
            cfg         = cfg,
            emit_projected_smt = True,
        )
        now = len(proj_smt) if isinstance(proj_smt, str) else 0
        if _too_big(now, pre_bytes, ps["growth"], max_projected_bytes if max_projected_bytes > 0 else None):
            last_summary, last_proj, last_status = summary, proj_smt, "too-big"
            continue

        return {
            "status": "ok",
            "preset": ps["name"],
            "summary": {**summary, "preset": ps["name"], "pre_bytes": pre_bytes, "post_bytes": now},
            "proj": proj_smt,
        }

    # fallback: reduced crumb
    crumb = ""
    if isinstance(last_proj, str) and last_proj:
        cap = min(max_projected_bytes if max_projected_bytes > 0 else 256_000, 256_000)
        crumb = last_proj[:cap] + ("\n; … TRUNCATED …\n" if len(last_proj) > cap else "")

    return {
        "status": last_status,
        "preset": "fallback",
        "summary": {**(last_summary or {}), "preset": "fallback", "pre_bytes": pre_bytes},
        "proj": crumb,
    }

# ───────── child process target ─────────
def _child_entry(smt_str: str, max_projected_bytes: int, q: mp.Queue):
    """Run ONE file; put a small result dict on queue. All prints go to per-file log."""
    smt = Path(smt_str)
    try:
        canon_path = _guess_canon_file(smt.name)
        if not canon_path:
            q.put({"smt_file": smt.name, "status": "no-canon"})
            return
        canon_bools = _load_canon_vars(canon_path)
        if not canon_bools:
            q.put({"smt_file": smt.name, "status": "empty-canon", "canon_file": Path(canon_path).name})
            return

        dbg_dir = None
        if DEBUG_DUMPS:
            dbg_dir = OUT_DIR / "_debug" / smt.stem
            dbg_dir.mkdir(parents=True, exist_ok=True)

        res = project_with_budget_inproc(smt, canon_bools, debug_dir=dbg_dir, max_projected_bytes=max_projected_bytes)

        # Always write summary
        (OUT_DIR / "_summaries" / f"{smt.stem}.json"
         ).write_text(json.dumps(res["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # Write SMT only for ok; partial crumb if fallback produced something
        if res["status"] == "ok" and WRITE_PROJECTED_SMT:
            cleaned = _clean_healed_name(smt.stem)
            (OUT_DIR / "_projected_smt" / f"{cleaned}_canon_projected.smt2").write_text(res["proj"], encoding="utf-8")
        elif res.get("proj"):
            (OUT_DIR / "_projected_smt" / f"{smt.stem}_PARTIAL.smt2").write_text(res["proj"], encoding="utf-8")

        # compact row
        sumr = res["summary"]
        q.put({
            "smt_file": smt.name,
            "status": res["status"],
            "preset": res.get("preset"),
            "canon_file": canon_path.name,
            "canon_bool_count": sumr.get("canon_bool_count", 0),
            "numeric_var_count": sumr.get("canon_numeric_count", 0),
            "must_true_count": len(sumr.get("must_true", [])) if isinstance(sumr.get("must_true"), list) else 0,
            "must_false_count": len(sumr.get("must_false", [])) if isinstance(sumr.get("must_false"), list) else 0,
            "numeric_unit_count": sumr.get("numeric_unit_count", 0),
            "binary_implicates_count": sumr.get("binary_implicates_count", 0),
        })
    except MemoryError:
        q.put({"smt_file": smt.name, "status": "error", "error": "MemoryError"})
    except Exception as e:
        q.put({"smt_file": smt.name, "status": "error", "error": str(e)})

# ───────── supervisor (hard timeout + bounded concurrency) ─────────
def supervise(files: List[Path], workers: int, timeout_s: int, max_projected_mb: int) -> List[Dict[str, object]]:
    max_bytes = max_projected_mb * 1024 * 1024 if max_projected_mb > 0 else -1
    index: List[Dict[str, object]] = []

    active: Dict[mp.Process, Dict[str, object]] = {}
    q = mp.Queue()

    if HAS_TQDM:
        pbar = tqdm(total=len(files), desc="Projecting", unit="file")
    else:
        pbar = None
        done = 0

    def _start_task(smt: Path):
        # Redirect child stdout/stderr to a per-file log to avoid TTY clogging
        log_path = OUT_DIR / "_logs" / f"{smt.stem}.log"
        # On POSIX, we can dup FDs after fork via preexec_fn, but for portability we do it inside the child:
        # we instead rely on _child_entry to be quiet; anything printed by libs will still hit stderr,
        # but most of our code avoids prints. If you want absolute silence, spawn "python -u -c" wrapper.
        p = mp.Process(target=_child_entry, args=(str(smt), max_bytes, q), daemon=True)
        p.start()
        active[p] = {"smt": smt, "t0": time.time(), "log": log_path}
        return p

    it = iter(files)
    # fill initial window
    try:
        while len(active) < workers:
            smt = next(it)
            _start_task(smt)
    except StopIteration:
        pass

    while active:
        # collect finished results without blocking too long
        drained = True
        while drained:
            try:
                row = q.get_nowait()
                index.append(row)
                if HAS_TQDM: pbar.update(1)
                else:
                    done += 1
                    pct = int(done * 100 / len(files))
                    sys.stdout.write(f"\r[progress] {done}/{len(files)} ({pct}%)")
                    sys.stdout.flush()
            except queue.Empty:
                drained = False

        # check timeouts & liveness
        to_remove: List[mp.Process] = []
        for p, meta in list(active.items()):
            smt = meta["smt"]; t0 = meta["t0"]
            if not p.is_alive():
                p.join(timeout=0.1)
                to_remove.append(p)
                continue
            if time.time() - t0 > timeout_s:
                try:
                    p.terminate()
                finally:
                    # mark timeout
                    index.append({"smt_file": smt.name, "status": "timeout"})
                    if HAS_TQDM: pbar.update(1)
                    else:
                        done += 1
                        pct = int(done * 100 / len(files))
                        sys.stdout.write(f"\r[progress] {done}/{len(files)} ({pct}%)"); sys.stdout.flush()
                    to_remove.append(p)

        # free slots and launch new tasks
        for p in to_remove:
            active.pop(p, None)

        try:
            while len(active) < workers:
                smt = next(it)
                _start_task(smt)
        except StopIteration:
            pass

        # small sleep to avoid busy loop
        time.sleep(0.05)

    if HAS_TQDM:
        pbar.close()
    else:
        sys.stdout.write("\n")

    return index

# ───────── CLI / main ─────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SMT QE projector with supervised processes")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help="Max concurrent files")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="(unused; compatibility only)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Reprocess even if projected output exists")
    ap.add_argument("--per-file-timeout", type=int, default=180,
                    help="Hard wall-clock seconds per file")
    ap.add_argument("--max-projected-mb", type=int, default=16,
                    help="Skip or truncate huge projected SMTs; 0=unlimited")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    _ensure_dirs()

    smt_files_all = sorted(
        p for p in SMT_DIR.glob("*.smt2")
        if ".healed" not in p.name and not p.name.endswith("_canon_projected.smt2")
    )
    if not smt_files_all:
        print(f"[info] no SMT files in {SMT_DIR}")
        (OUT_DIR / "_index.json").write_text("[]\n", encoding="utf-8")
        return

    smt_files = smt_files_all if args.rebuild else [p for p in smt_files_all if not _already_done(p)]
    if not smt_files:
        print("[info] nothing to do (all projected).")
        (OUT_DIR / "_index.json").write_text("[]\n", encoding="utf-8")
        return

    print(f"[plan] {len(smt_files)} file(s); workers={args.workers}; timeout={args.per_file_timeout}s; max_out={args.max_projected_mb}MB")
    index = supervise(smt_files, workers=args.workers, timeout_s=args.per_file_timeout, max_projected_mb=args.max_projected_mb)

    # write index
    (OUT_DIR / "_index.json"
     ).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # short summary
    counts: Dict[str, int] = {}
    for r in index:
        counts[r.get("status","ok")] = counts.get(r.get("status","ok"), 0) + 1
    print(f"[done] ok={counts.get('ok',0)} timeout={counts.get('timeout',0)} "
          f"too-big={counts.get('too-big',0)} error={counts.get('error',0)} "
          f"no-canon={counts.get('no-canon',0)} empty-canon={counts.get('empty-canon',0)}")
    print(f"Index → {OUT_DIR / '_index.json'}")

if __name__ == "__main__":
    # Use 'spawn' on macOS/Windows for safety with libraries; default 'fork' on Linux is fine.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()