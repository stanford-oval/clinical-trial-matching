"""End-to-end screening: retrieve candidate trials, then decide each one.

    import pipeline

    for r in pipeline.screen("sigir-20141", db="build/trial.db", limit=10):
        print(r.rank, r.nct_id, r.decision or "(no data)")

This is the only module that touches both systems. SatIR and VERDICT must not
import each other -- coupling them would mean you could not run retrieval
without the matcher, or the matcher without a database -- so the join lives
here, one layer above both.

    parser  ->  SatIR      (which trials are worth a look?)
            ->  VERDICT    (does this patient meet this one, and why?)
                    \\
                     pipeline: the two in sequence

What each stage needs
---------------------
SatIR    a clause database. Pure SQL: no LLM, no services.
VERDICT  per-pair stage-1 artifacts under $VERDICT_PAIR_DATA. Candidates
         without them come back with decision=None rather than a guess --
         "not evaluated" must never be reportable as "not eligible".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ScreenResult", "screen"]


@dataclass
class ScreenResult:
    """One candidate trial, retrieved and (where possible) decided."""
    nct_id: str
    rank: Optional[int]
    retrieval_label: str
    decision: Optional[str] = None      # None == VERDICT had no data for it
    reasoning: str = ""
    #: rho as actionable statements -- see Artifacts.assumptions_report().
    #: Never the raw solver witnesses: under MAXSMT those are arbitrary points
    #: in the satisfying region, and surfacing one would invent a finding.
    assumptions: List[Dict[str, Any]] = field(default_factory=list)
    pivotal: List[str] = field(default_factory=list)

    def assumptions_text(self) -> str:
        """The assumptions as plain text, pivotal ones first."""
        if not self.assumptions:
            return "No assumptions: every condition was resolved from the chart."
        recs = sorted(self.assumptions,
                      key=lambda r: (not r.get("pivotal"), r.get("label", "")))
        out = []
        for r in recs:
            out.append(("!" if r.get("pivotal") else "-") + " "
                       + r.get("statement", "") + " " + r.get("basis", ""))
            out.append("    " + r.get("action", ""))
        return "\n".join(out)

    @property
    def eligible(self) -> Optional[bool]:
        """True / False, or None when the pair could not be evaluated."""
        return None if self.decision is None else self.decision == "eligible"

    @property
    def evaluated(self) -> bool:
        return self.decision is not None


def screen(patient_id: str, *, db: Optional[str] = None,
           limit: Optional[int] = None, system: str = "verdict",
           artifacts: bool = True) -> List[ScreenResult]:
    """Retrieve candidates for a patient, then decide each with VERDICT.

    Args:
        patient_id: e.g. "sigir-20141".
        db:         clause database for retrieval.
        limit:      stop after this many candidates (they are rank-ordered).
        system:     matcher variant; see verdict.systems().
        artifacts:  also compute assumptions and pivotal conditions.

    Returns:
        One ScreenResult per candidate, in retrieval rank order. Candidates
        VERDICT cannot evaluate keep decision=None.
    """
    import satir
    import verdict

    candidates = satir.retrieve(patient_id, db=db)
    if limit is not None:
        candidates = candidates[:limit]

    results: List[ScreenResult] = []
    for c in candidates:
        r = ScreenResult(nct_id=c.nct_id, rank=c.rank,
                         retrieval_label=c.label)
        pair = f"{patient_id}__{c.nct_id}"
        try:
            d = verdict.match(pair, system=system)      # strict: raises if absent
        except verdict.MissingPairData:
            results.append(r)                            # decision stays None
            continue
        r.decision, r.reasoning = d.decision, d.reasoning
        if artifacts:
            try:
                from verdict.artifacts import artifacts_for
                a = artifacts_for(pair)
                if a is not None:
                    phi = getattr(a, "phi_lines", None) or _phi_for(pair)
                    r.assumptions = a.assumptions_report(phi)
                    r.pivotal = a.pivotal
            except Exception:
                pass                                     # artifacts are optional
        results.append(r)
    return results



def _phi_for(pair: str) -> List[str]:
    """The trial program for a pair, for rendering requirements. [] if absent."""
    try:
        from verdict.artifacts import conditions_from_pair  # noqa: F401
        from verdict import data as _d
        import json
        base = _d.pair_root() / "cmsrc_out"
        lines: List[str] = []
        for f in sorted((base / pair.split("__")[0]).glob(f"*{pair.split('__')[-1]}*full.json")):
            raw = json.loads(f.read_text())
            for side in ("inclusion", "exclusion"):
                r = (raw.get(side) or {}).get("raw") or {}
                lines += r.get("smt_program_lines") or []
        return lines
    except Exception:
        return []

# --------------------------------------------------------------- compile chain
#: The trial-side stages, in order, between raw trial text and the program the
#: matcher reads. Each reads the previous stage's output under the one build
#: root that smt_core.buildroot resolves.
COMPILE_STAGES = ("compile", "normalize", "slice", "link")


def compile_trial_program(trial_id: str, *, stages=COMPILE_STAGES,
                          verbose: bool = True) -> "pathlib.Path":
    """Compile one trial into the SMT program `verdict run` consumes.

    Returns the path to the linked IR for `trial_id`.

    The four stages were previously separate scripts a user had to know the
    order of, two of which pointed at a hard-coded directory outside the
    repository. Running them as one chain is what makes an end-to-end match
    possible from a clone.

    Needs an LLM endpoint: the compile stage calls a model.
    """
    import importlib.util
    import pathlib
    import subprocess
    import sys

    from smt_core.buildroot import build_root, describe

    root = pathlib.Path(__file__).resolve().parent
    scripts = root / "db_indexer" / "trial_side" / "scripts"
    build = build_root()

    def _say(msg):
        if verbose:
            print(msg, flush=True)

    def _load(path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(path.stem, mod)
        spec.loader.exec_module(mod)
        return mod

    _say(f"build root: {describe()}")

    if "compile" in stages:
        _say(f"[1/4] compile   {trial_id}")
        rc = subprocess.call([sys.executable, "-m", "trial_compiler.cli", trial_id],
                             cwd=root)
        if rc != 0:
            raise SystemExit(f"trial_compiler failed for {trial_id} (exit {rc})")

    if "normalize" in stages:
        _say("[2/4] normalize")
        try:
            _load(scripts / "normalize_units.py").run(
                in_dir=build / "ir", out_dir=build / "ir_normalized")
        except (ImportError, RuntimeError) as exc:
            # unit handling lives behind an extra, not the base install
            raise SystemExit(
                "the compile chain needs the compile extra: "
                "pip install '.[compile]'\n(" + str(exc) + ")")

    if "slice" in stages:
        _say("[3/4] slice")
        _load(scripts / "batch_slice_ir.py").main()

    if "link" in stages:
        _say("[4/4] link")
        _load(scripts / "batch_link_qualifiers.py").main()

    linked = build / "noslice_ir_linked"
    out = linked / f"{trial_id}.smt2"
    if not out.exists():
        cand = sorted(linked.glob(f"{trial_id}*"))
        out = cand[0] if cand else None
    if out is None:
        # Never report a path that was not produced -- an empty build tree
        # would otherwise read as success.
        raise SystemExit(
            f"compile chain produced no program for {trial_id} under {linked}.\n"
            "Ran stages: " + ", ".join(stages) + ".\n"
            "If you skipped the 'compile' stage, there was no IR to normalize.")
    _say(f"done -> {out}")
    return out
