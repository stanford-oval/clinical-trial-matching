from typing import Dict, Any, Tuple, List, Optional, Callable

import dspy

# ---------------------------------------------------------------------------
# Rule type & helpers
# ---------------------------------------------------------------------------

RuleFn = Callable[[List[str]], Tuple[bool, Optional[str]]]
Rule    = Tuple[str, RuleFn]


# -------------------- built‑in rules ---------------------------------------

def _no_duplicate_decl(lines: List[str]) -> Tuple[bool, Optional[str]]:
    seen = set()
    for ln in lines:
        if ln.startswith("(declare-const"):
            name = ln.split()[1]
            if name in seen:
                return False, f"duplicate declaration of {name}"
            seen.add(name)
    return True, None


def _percent_within_bounds(lines: List[str]) -> Tuple[bool, Optional[str]]:
    for ln in lines:
        if "_percent" in ln and ln.startswith("(declare-const"):
            parts = ln.split()
            if len(parts) >= 3 and parts[2] != "Int":
                return False, f"{parts[1]} should be Int for percentage values"
    return True, None


DEFAULT_RULES: List[Rule] = [
    ("duplicate-decl", _no_duplicate_decl),
    ("pct-bounds",      _percent_within_bounds),
]


# ---------------------------------------------------------------------------
# Global semantic checker (batch‑aware)
# ---------------------------------------------------------------------------

class SMTOnepassSemanticChecker(dspy.Module):
    """Run lightweight domain/typing sanity checks over requirement slices.

    Parameters
    ----------
    batch_size : int | None, optional
        * `None`   → check each requirement slice independently (default).
        * `0` or <=0 → concatenate **all** requirements and check once.
        * `N > 0` → group slices into batches of at most *N* requirements and
          run the rules on each batch.
    rules : list[Rule], optional
        Custom rule list (falls back to DEFAULT_RULES if omitted).
    """

    # --------------------------------------------------------------
    def __init__(self, engine=None, *, batch_size: Optional[int] = None, rules: Optional[List[Rule]] = None):
        super().__init__()
        self.batch_size = batch_size  # None → per‑requirement
        self.rules: List[Rule] = rules or DEFAULT_RULES

    # --------------------------------------------------------------
    def _run_rules(self, lines: List[str]) -> List[Dict[str, Any]]:
        report: List[Dict[str, Any]] = []
        for rule_name, fn in self.rules:
            ok, msg = fn(lines)
            report.append({"rule": rule_name, "ok": ok, "message": msg})
        return report

    # --------------------------------------------------------------
    def forward(self, context: dict) -> dict:  # type: ignore[override]
        print("In SMTOnepassSemanticChecker\n")

        program = context.get("smt_program_lines", [])
        blocks  = context.get("req_blocks", {})
        if not program or not blocks:
            raise ValueError("Semantic checker expects 'smt_program_lines' and 'req_blocks' in context")

        # ------------------------------------------------------------------
        # Decide batching strategy
        # ------------------------------------------------------------------
        if self.batch_size is None:  # one slice at a time (status quo)
            batches: List[Tuple[List[int], List[str]]] = []
            for idx, (lo, hi) in blocks.items():
                batches.append(([idx], program[lo:hi]))
        elif self.batch_size <= 0:   # single mega‑batch
            all_idx   = sorted(blocks.keys())
            all_lines = []
            for idx in all_idx:
                lo, hi = blocks[idx]
                all_lines.extend(program[lo:hi])
            batches = [(all_idx, all_lines)]
        else:                        # fixed‑size batches
            sorted_idx = sorted(blocks.keys())
            batches = []
            for i in range(0, len(sorted_idx), self.batch_size):
                batch_idx = sorted_idx[i : i + self.batch_size]
                lines: List[str] = []
                for idx in batch_idx:
                    lo, hi = blocks[idx]
                    lines.extend(program[lo:hi])
                batches.append((batch_idx, lines))

        # ------------------------------------------------------------------
        # Run checks
        # ------------------------------------------------------------------
        overall_ok = True
        reports: Dict[int | Tuple[int, ...], Any] = {}

        for idx_list, slice_lines in batches:
            rep = self._run_rules(slice_lines)
            if not all(r["ok"] for r in rep):
                overall_ok = False
            key = idx_list[0] if len(idx_list) == 1 else tuple(idx_list)
            reports[key] = rep

        context["semantic_reports"] = reports
        context["semantic_ok"] = overall_ok
        return context
