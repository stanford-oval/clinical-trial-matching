"""
solver_based_validator.py
-------------------------
Run Z3 over the whole SMT program, optionally invoke a refiner on UNSAT.
Outputs (added to context)
    • solver_check_attempts : list[dict]
    • solver_check          : dict        (result of last attempt)
    • solver_ok             : bool        (True ↔ SAT)
"""
from __future__ import annotations
import time
import pprint
from typing import Dict, Any, Optional

import dspy
from z3 import Solver, parse_smt2_string, Z3Exception

from .SMTOnepassSolverBasedUnsatCoreRefiner import SMTOnepassSolverBasedUnsatCoreRefiner

__all__ = ["SMTOnepassSolverBasedValidator"]

# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────
_WRAP_RE  = r"</?output_complete_smt_program\s*>"
_META_RE  = r"<!-- BEGIN META -->.*<!-- END META -->"
import re
_UNWRAP_RE = re.compile(f"{_WRAP_RE}|{_META_RE}", re.I | re.S)

def _clean_smt(text: str) -> str:
    """Remove the XML wrapper + metadata block, return pure SMT-LIB."""
    return _UNWRAP_RE.sub("", text).strip()

def _print_solver_result(label: str, res: Dict[str, Any]) -> None:
    hdr = f"[Z3 {label}] status={res['status']}"
    if res.get("message"):
        hdr += f" | msg: {res['message']}"
    if res.get("unsat_core"):
        hdr += f" | unsat-core: {res['unsat_core']}"
    print(hdr)

    if res["status"] == "sat" and res.get("model"):
        print("  model:\n  " + "\n  ".join(res["model"].splitlines()))
    if res.get("stats"):
        print("  stats:")
        pprint.pp(res["stats"], indent=4, width=80)
    print()

# ───────────────────────────────────────────────────────────────────
# Validator
# ───────────────────────────────────────────────────────────────────
class SMTOnepassSolverBasedValidator(dspy.Module):
    """Run Z3 once over the entire program; optionally call global refiner."""

    def __init__(
        self,
        engine,
        *,
        produce_model: bool = False,
        produce_unsat_core: bool = True,
        max_global_attempts: int = 1,
    ):
        super().__init__()
        self.engine = engine
        self.produce_model = produce_model
        self.produce_unsat_core = produce_unsat_core
        self.max_global_attempts = max_global_attempts
        self.refiner =  SMTOnepassSolverBasedUnsatCoreRefiner(engine)

    # ------------------------------------------------------
    def _run_solver(
        self,
        smt_text: str,
        tag_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Return dict with keys: status, model, unsat_core, unsat_reqs, stats."""
        out: Dict[str, Any] = {
            "status": "error",
            "message": "",
            "model": None,
            "unsat_core": None,
            "unsat_reqs": None,
            "stats": {},
        }

        solver = Solver()
        if self.produce_unsat_core:
            solver.set(unsat_core=True)

        try:
            ast = parse_smt2_string(smt_text)
            t0 = time.time()
            solver.add(ast)
            res = solver.check()
            elapsed = time.time() - t0

            zstats = solver.statistics()
            out["stats"] = {k: zstats.get_key_value(k) for k in zstats.keys()}
            out["stats"]["elapsed_sec"] = round(elapsed, 4)

            if res.r == 1:                                            # sat
                out["status"] = "sat"
                if self.produce_model:
                    out["model"] = str(solver.model())

            elif res.r == -1:                                         # unsat
                out["status"] = "unsat"
                if self.produce_unsat_core:
                    core_tags = [str(t) for t in solver.unsat_core()]
                    out["unsat_core"] = core_tags
                    if tag_map:
                        reqs = {tag_map.get(t) for t in core_tags if t in tag_map}
                        out["unsat_reqs"] = sorted(r for r in reqs if r is not None)

            else:                                                     # unknown
                out["status"] = "unknown"
                out["message"] = solver.reason_unknown()

        except Z3Exception as e:
            out["status"] = "error"
            out["message"] = str(e)

        return out

    # ------------------------------------------------------
    def forward(self, context: dict) -> dict:                       # type: ignore[override]
        print("\n=== SMTOnepassSolverBasedValidator ===\n")

        smt_lines = context.get("smt_program_lines")
        if not smt_lines:
            raise ValueError("Validator expects 'smt_program_lines' in context")

        pure_smt = _clean_smt("\n".join(smt_lines))
        tag_map   = context.get("assertion_tags", {})

        context["solver_check_attempts"] = []
        attempt = 0

        result = self._run_solver(pure_smt, tag_map)
        _print_solver_result(attempt, result)
        context["solver_check_attempts"].append({"attempt": attempt, **result})

        # ------------- global refinement loop ---------------------
        while result["status"] == "unsat" and attempt < self.max_global_attempts:
            attempt += 1
            context = self.refiner.forward(context, solver_result=result)

            patched_lines = context.get("patched_smt_program_lines")
            if not patched_lines:            # refiner made no change
                break

            pure_smt = _clean_smt("\n".join(patched_lines))
            result   = self._run_solver(pure_smt, tag_map)
            _print_solver_result(attempt, result)
            context["solver_check_attempts"].append({"attempt": attempt, **result})

        context["solver_check"] = result
        context["solver_ok"]    = (result["status"] == "sat")
        return context
