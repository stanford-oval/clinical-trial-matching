# SMTOnepassRequirementChecker.py
import z3, dspy
from typing import Dict, List, Any, Tuple

class SMTOnepassRequirementChecker(dspy.Module):
    """
    For each requirement slice, run Z3 and record:

        context["req_solver_report"][idx] = {
            "status"     : "sat" | "unsat" | "unknown" | "error",
            "unsat_core" : [tag, ...],          # only if unsat
            "z3_msg"     : str
        }
    """

    def __init__(self, timeout_ms: int = 5000):
        super().__init__()
        self.timeout_ms = timeout_ms

    # ---------------- internal helpers --------------------------------
    @staticmethod
    def _separate_decl_assert(lines: List[str]) -> Tuple[List[str], List[str]]:
        decl, assert_ = [], []
        for ln in lines:
            if ln.lstrip().startswith("(declare"):
                decl.append(ln)
            elif ln.lstrip().startswith("(assert"):
                assert_.append(ln)
        return decl, assert_

    @staticmethod
    def _make_solver(timeout_ms: int) -> z3.Solver:
        s = z3.Solver()
        s.set(unsat_core=True, timeout=timeout_ms)
        return s

    # ---------------- main --------------------------------------------
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        prog_lines: List[str] = ctx["smt_program_lines"]
        req_blocks: Dict[int, Tuple[int, int]] = ctx["req_blocks"]

        decl_lines, _ = self._separate_decl_assert(prog_lines)

        reports = {}
        for rid, (lo, hi) in req_blocks.items():
            slice_lines = prog_lines[lo:hi]

            # 1) build solver with everything EXCEPT this slice
            solver = self._make_solver(self.timeout_ms)

            # add declarations
            solver.add(z3.parse_smt2_string("\n".join(decl_lines)))

            # add all other assertions
            for rj, (sj, ej) in req_blocks.items():
                if rj == rid:
                    continue
                _, other_assert = self._separate_decl_assert(prog_lines[sj:ej])
                if other_assert:
                    solver.add(z3.parse_smt2_string("\n".join(other_assert)))

            # 2) push slice-i assertions
            _, slice_assert = self._separate_decl_assert(slice_lines)
            if slice_assert:
                solver.push()
                solver.add(z3.parse_smt2_string("\n".join(slice_assert)))

            # 3) check
            try:
                st = solver.check()
            except z3.Z3Exception as e:
                reports[rid] = {"status": "error", "unsat_core": [], "z3_msg": str(e)}
                continue

            if st == z3.sat:
                reports[rid] = {"status": "sat", "unsat_core": [], "z3_msg": ""}
            elif st == z3.unknown:
                reports[rid] = {"status": "unknown", "unsat_core": [],
                                "z3_msg": solver.reason_unknown()}
            else:  # unsat
                tags = [b.decl().name() for b in solver.unsat_core()]
                reports[rid] = {"status": "unsat", "unsat_core": tags, "z3_msg": ""}

        ctx["req_solver_report"] = reports
        # convenience overall flag
        ctx["all_reqs_ok"] = all(r["status"] == "sat" for r in reports.values())
        return ctx
