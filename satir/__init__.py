"""SatIR — constraint-satisfaction clinical-trial retrieval and compilation.

    import satir

    satir.config()                        # resolved SatIRConfig
    satir.compile_trial("NCT00337116")    # trial text  -> SMT constraints
    satir.compile_patient("sigir-20141")  # patient note -> coded facts
    satir.index()                         # constraints -> clause database
    satir.retrieve()                      # SQL constraint-satisfaction retrieval
    satir.match("sigir-20141", "NCT00337116")   # SMT eligibility check

The compiler and indexer stages are argv-driven underneath: their upstream
`main(argv)` functions own the argument parsing, so these wrappers pass extra
options straight through rather than re-declaring a parallel schema that could
drift. Anything you would type after the subcommand goes in `*args`:

    satir.compile_trial("NCT00337116", "--side", "inclusion", "--stop-after", "ir")

For matching, the underlying module exposes genuinely reusable pieces and they
are re-exported here: `load_patient`, `build_ctx_from_persisted`,
`run_match_for_side`.

Imports are lazy: `import satir` works without the optional heavy dependencies
(torch, elasticsearch, matplotlib...). You only pay for what you call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, List, Optional

__all__ = ["config", "compile_trial", "compile_patient", "index", "retrieve",
           "Candidate",
           "match", "load_patient", "build_ctx_from_persisted",
           "run_match_for_side", "services", "benchmark"]


def config() -> Any:
    """The resolved SatIRConfig (paths, services, retrieval, keys)."""
    from smt_core.config import get_config
    return get_config()


def _run_argv(module_path: str, func: str, argv: list[str]) -> Any:
    """Call an argv-driven main() without leaking our own sys.argv into it."""
    import importlib, sys
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func)
    saved = sys.argv
    sys.argv = [module_path.split(".")[-1]] + argv
    try:
        return fn()
    finally:
        sys.argv = saved


def compile_trial(trial_id: str, *args: str) -> Any:
    """Compile one trial's eligibility criteria into SMT constraints.

    Args:
        trial_id: NCT identifier.
        *args:    passed through, e.g. "--side", "inclusion".
    """
    return _run_argv("trial_compiler.compile_trial", "main", [trial_id, *args])


def compile_patient(patient_id: str, *args: str) -> Any:
    """Compile one patient's note into canonical constraint variables."""
    return _run_argv("patient_compiler.compile_patient", "main", [patient_id, *args])


def index(*args: str) -> Any:
    """Index compiled constraints into the clause database."""
    return _run_argv("db_indexer.cli", "main", list(args))


@dataclass
class Candidate:
    """One trial SatIR considers worth a closer look, for one patient."""
    nct_id: str                 # canonical NCT, e.g. NCT02357212
    status: str                 # 'survivor' | 'eliminated'
    label: str                  # why, e.g. 'all_satisfied'
    rank: Optional[int] = None  # best rank across this trial's subcohorts
    sub_nct_ids: List[str] = field(default_factory=list)

    @property
    def survived(self) -> bool:
        return self.status == "survivor"


def retrieve(patient_id: str, *args: str, db: Optional[str] = None,
             out: Optional[str] = None, survivors_only: bool = True,
             ) -> List["Candidate"]:
    """Retrieve candidate trials for one patient. Returns them, ranked.

    Pure SQL over the clause database -- no LLM, no services. Defaults come
    from `config().retrieval`; `*args` are appended to the underlying command.

    Args:
        patient_id:     e.g. "sigir-20141".
        db:             clause database; defaults to config().paths.build_dir.
        out:            where to write results; a temporary directory if unset.
        survivors_only: drop candidates retrieval already eliminated.

    Raises:
        RuntimeError: retrieval exited non-zero.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    cfg = config()
    root = pathlib.Path(__file__).resolve().parent.parent
    out_dir = pathlib.Path(out) if out else pathlib.Path(tempfile.mkdtemp())
    cmd = [sys.executable, "-m", "sql_retrieval.ops.constraint_retrieval",
           "--db", db or str(pathlib.Path(cfg.paths.build_dir) / "trial.db"),
           "--patient", patient_id,
           "--out", str(out_dir),
           "--scope", cfg.retrieval.scope,
           "--important-mode", cfg.retrieval.important_mode,
           "--alt-mode", cfg.retrieval.alt_mode,
           "--parallel", str(cfg.retrieval.parallel), "--quiet"]
    if cfg.retrieval.enable_prevention:
        cmd.append("--enable-prevention-hits")
    cmd += list(args)
    r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(root)},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"retrieval failed ({r.returncode}):\n"
                           f"{r.stderr[-800:] or r.stdout[-800:]}")

    # retrieval writes list_to_match__<mode>__<prevent>__<alt>/<patient>__*.json
    hits = sorted(out_dir.glob("list_to_match__*/*.json"))
    if not hits:
        return []
    payload = json.loads(hits[0].read_text())
    cands: List[Candidate] = []
    for c in payload.get("canonical_trials", []):
        ranks = [s.get("rank") for s in c.get("subcohorts", [])
                 if s.get("rank") is not None]
        cands.append(Candidate(
            nct_id=c.get("canonical_nct_id", ""),
            status=c.get("status", ""),
            label=c.get("label", ""),
            rank=min(ranks) if ranks else None,
            sub_nct_ids=list(c.get("sub_nct_ids") or []),
        ))
    if survivors_only:
        cands = [c for c in cands if c.survived]
    cands.sort(key=lambda c: (c.rank is None, c.rank))
    return cands


def match(patient_id: str, trial_id: str, *args: str) -> Any:
    """SMT eligibility check for one patient--trial pair.

    For programmatic control over a single side, prefer `run_match_for_side`.
    """
    return _run_argv("smt_matcher.match_patient_to_trial", "main",
                     [patient_id, trial_id, *args])


def services(*args: str) -> int:
    """Manage Elasticsearch + Snowstorm (start/stop/status/install/import/check)."""
    import pathlib, subprocess
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "services.sh"
    return subprocess.run([str(script), *args]).returncode


def benchmark(*args: str) -> Any:
    """Benchmark retrieval performance."""
    return _run_argv("tests.benchmark_retrieval", "main", list(args))


def __getattr__(name: str) -> Any:
    """Lazily re-export the reusable matcher functions.

    Kept lazy so `import satir` does not pull in z3/azure just to read config.
    """
    if name in ("load_patient", "build_ctx_from_persisted", "run_match_for_side"):
        try:
            from smt_matcher import match_patient_to_trial as m
        except ImportError as e:
            raise ImportError(
                f"satir.{name} needs the optional inference backends: "
                f"pip install -e '.[llm]'   (missing: {e.name})") from e
        return getattr(m, name)
    raise AttributeError(name)
