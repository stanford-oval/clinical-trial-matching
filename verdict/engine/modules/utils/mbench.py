# src/utils/mbench.py
from __future__ import annotations

import json, os, re, time
import datetime as dt
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _slug(s: str) -> str:
    """Filesystem-safe, compact slug."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_")[:80]


def _now() -> str:
    """Wall-clock timestamp (YYYYmmddTHHMMSS)."""
    return time.strftime("%Y%m%dT%H%M%S")


def _to_jsonable(obj: Any):
    """Best-effort conversion to JSON-serializable."""
    try:
        json.dumps(obj)  # fast path
        return obj
    except Exception:
        if isinstance(obj, set):
            return sorted(list(obj))
        if hasattr(obj, "sexpr"):
            try:
                return obj.sexpr()
            except Exception:
                pass
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"


class Microbench:
    """
    Small logger for microbench + I/O snapshots.

    Directory layout:
      mbench/<run_id>/
        meta.json
        SMTMatcher/
          summary.json
          timings.jsonl
          *.json / *.txt
        SMTLeafCollector/
        SMTVariableValueMiner/
        SingleRequirementEvaluator/
    """

    def __init__(self, root: str = "mbench", run_id: Optional[str] = None):
        self.root = Path(root)
        self.run_id = run_id or _now()
        self.base = self.root / self.run_id
        self.base.mkdir(parents=True, exist_ok=True)

    def step_dir(self, step: str) -> Path:
        p = self.base / _slug(step)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _write(self, step: str, name: str, data: bytes):
        d = self.step_dir(step)
        (d / name).write_bytes(data)

    def log_json(self, step: str, name: str, obj: Any):
        payload = _to_jsonable(obj)
        fname = name if str(name).endswith(".json") else f"{name}.json"
        self._write(step, _slug(fname), json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))

    def log_text(self, step: str, name: str, text: str):
        fname = name if str(name).endswith(".txt") else f"{name}.txt"
        self._write(step, _slug(fname), (text or "").encode("utf-8"))

    def log_bytes(self, step: str, name: str, data: bytes):
        self._write(step, _slug(name), data)

    def write_meta(self, meta: Dict[str, Any]):
        # Write in the run root ('.' resolves to the base dir via step_dir)
        self.log_json(".", "meta", meta)

    def append_timing(self, step: str, record: Dict[str, Any]):
        d = self.step_dir(step)
        fp = d / "timings.jsonl"
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")

    @contextmanager
    def timeit(self, step: str, name: str, extra: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rec = {"at": _now(), "block": name, "ms": round(dt_ms, 3)}
            if extra:
                rec.update(extra)
            self.append_timing(step, rec)


def mbench_enabled(ctx: Dict[str, Any]) -> bool:
    """Default: enabled. Override with context['MBENCH_ENABLED']=False."""
    return bool(ctx.get("MBENCH_ENABLED", True))


def get_mbench(ctx: dict) -> Microbench:
    """
    Build a Microbench using ctx/env settings.

    Controls:
      • MBENCH_ROOT      (ctx/env)  : base folder (default 'mbench')
      • MBENCH_RUN_ID    (ctx/env)  : exact run folder name (bypasses timestamp/label logic)
      • MBENCH_RUN_LABEL (ctx/env)  : human label added to run_id when no explicit RUN_ID
      • MBENCH_TS        (ctx/env)  : '1'|'0' (default '1'): include timestamp in run_id when no RUN_ID
    """
    root = ctx.get("MBENCH_ROOT", os.getenv("MBENCH_ROOT", "mbench"))
    label = ctx.get("MBENCH_RUN_LABEL") or os.getenv("MBENCH_RUN_LABEL", "")
    run_id = ctx.get("MBENCH_RUN_ID") or os.getenv("MBENCH_RUN_ID")

    # Toggle timestamps (default ON for back-compat)
    add_ts = ctx.get("MBENCH_TS")
    if add_ts is None:
        add_ts = os.getenv("MBENCH_TS", "1").lower() not in {"0", "false", "no"}

    if not run_id:
        ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S") if add_ts else ""
        if ts and label:
            run_id = f"{ts}_{_slug(label)}"
        elif label:
            run_id = _slug(label)
        elif ts:
            run_id = ts
        else:
            run_id = "run"

    mb = Microbench(root=root, run_id=run_id)

    # Best-effort meta (non-fatal)
    try:
        mb.write_meta(
            {
                "run_id": run_id,
                "root": str(mb.root),
                "label": label,
                "ts_in_name": bool(add_ts),
            }
        )
    except Exception:
        pass

    return mb


def shorten(text: str, max_len: int = 800) -> str:
    """Trim long strings but preserve head/tail context."""
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    half = max_len // 2
    head = s[:half]
    tail = s[-half:]
    return f"{head}\n…[+{len(s) - max_len} chars]…\n{tail}"
