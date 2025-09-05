# ───────────────────────── profiling utils ─────────────────────────
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import resource  # Unix/macOS
except Exception:  # pragma: no cover
    resource = None  # type: ignore

try:
    import psutil  # optional; improves RSS reporting
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

import tracemalloc
import os
import pathlib

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _rss_mb() -> float:
    """Best-effort RSS (MB). Uses psutil if available, else ru_maxrss snapshot when possible."""
    if psutil:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    # resource gives peak, not instantaneous RSS; still useful context
    if resource:
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is KB on Linux, bytes on macOS depending on Python build; normalize
            peak_kb = float(ru.ru_maxrss)
            # Heuristic: if value looks too large to be KB (e.g., mac returns bytes),
            # convert bytes->MB; otherwise KB->MB
            if peak_kb > 10**8:  # likely bytes
                return peak_kb / (1024 * 1024)
            return peak_kb / 1024.0
        except Exception:
            pass
    # last resort: tracemalloc gives Python heap only, not total RSS
    try:
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)
    except Exception:
        return float("nan")

def _peak_rss_mb() -> float:
    """Best-effort peak RSS (MB)."""
    if psutil:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    if resource:
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            peak_kb = float(ru.ru_maxrss)
            if peak_kb > 10**8:
                return peak_kb / (1024 * 1024)
            return peak_kb / 1024.0
        except Exception:
            pass
    try:
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)
    except Exception:
        return float("nan")

@dataclass
class LLMStats:
    llm_calls: int = 0
    llm_retries: int = 0
    llm_errors: int = 0
    llm_tok_in: int = 0
    llm_tok_out: int = 0
    llm_tok_total: int = 0
    llm_last_latency_s: float = 0.0
    llm_cum_latency_s: float = 0.0

@dataclass
class StageProfile:
    stage: str
    start: str
    end: str
    wall_s: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    peak_rss_mb: float
    llm: LLMStats

@dataclass
class RunProfile:
    trial_id: str
    side: str
    started: str
    ended: Optional[str]
    wall_s: Optional[float]
    stages: list
    peak_rss_mb: float

class RunProfiler:
    """
    Collects per-stage + overall metrics and writes them to
    <log_dir>/profiles/<trial_id>/<side>/<trial_id>_<side>_<ts>.json
    and also updates a 'latest.json' in that directory.
    """
    def __init__(self, profile_root: pathlib.Path, trial_id: str, side: str, ts: str):
        self.trial_id = trial_id
        self.side = side
        self.ts = ts
        self.root = profile_root / trial_id / side
        self.root.mkdir(parents=True, exist_ok=True)
        self._run_t0 = None
        self._stage_t0 = None
        self._rss0 = None
        self._stages: list[StageProfile] = []
        # start tracemalloc if not already
        try:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
        except Exception:
            pass
        self._run_started_iso = _now_iso()

    def start_run(self):
        self._run_t0 = time.perf_counter()

    def start_stage(self, stage_name: str):
        self._stage_t0 = time.perf_counter()
        self._rss0 = _rss_mb()

    def end_stage(self, stage_name: str, llm_stats_dict: dict):
        t1 = time.perf_counter()
        rss1 = _rss_mb()
        prof = StageProfile(
            stage=stage_name,
            start=self._run_started_iso,  # stage-level start timestamp (approx run start)
            end=_now_iso(),
            wall_s=round(t1 - (self._stage_t0 or t1), 6),
            rss_start_mb=round((self._rss0 or rss1), 3),
            rss_end_mb=round(rss1, 3),
            rss_delta_mb=round(rss1 - (self._rss0 or rss1), 3),
            peak_rss_mb=round(_peak_rss_mb(), 3),
            llm=LLMStats(**llm_stats_dict),
        )
        self._stages.append(prof)

    def finish_and_save(self):
        t1 = time.perf_counter()
        ended = _now_iso()
        run = RunProfile(
            trial_id=self.trial_id,
            side=self.side,
            started=self._run_started_iso,
            ended=ended,
            wall_s=round(t1 - (self._run_t0 or t1), 6),
            stages=[asdict(s) for s in self._stages],
            peak_rss_mb=round(_peak_rss_mb(), 3),
        )
        out_path = self.root / f"{self.trial_id}_{self.side}_{self.ts}.json"
        latest_path = self.root / "latest.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        tmp.replace(out_path)
        # also refresh latest
        latest_tmp = latest_path.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        latest_tmp.replace(latest_path)
        return out_path
