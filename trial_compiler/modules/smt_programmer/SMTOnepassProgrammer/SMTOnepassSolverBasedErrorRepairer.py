"""
SMTOnepassErrorRepairer.py
--------------------------
Fix *parsing / typing* errors that make Z3 return status "error" or "unknown".

Input (in `context`)
    smt_program_lines : list[str]                  – current program
    solver_check      : dict{status,message,…}     – produced by the validator

Output (added to / updated in `context`)
    patched_smt_program_lines : list[str]          – modified program
"""
from __future__ import annotations
import re, textwrap, logging
from typing import List, Set
import dspy

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
_LINE_RE = re.compile(r"line (\d+)", re.I)          # matches “line 42”
_COL_RE  = re.compile(r"column (\d+)")              # optional

ERROR_FIX_TEMPLATE = textwrap.dedent("""\
    You are an expert SMT-LIB engineer.

    # Problem
    Z3 rejected the file with the following diagnostic:

    ```
    #ERROR_MSG#
    ```

    The relevant lines are shown below with their original line numbers:

    ```
    #BAD_SNIPPET#
    ```

    # Task
    *Return **only** the corrected replacement lines (one per line, **no
    extra commentary**).*
    * Keep the same order and same line count as in the snippet.
    * Preserve all :named tags and REQUIREMENT headers.
    * The result **must parse in Z3 4.13.0**.

    Begin your reply with the first replacement line; do not wrap in markdown.
    """)

def _extract_bad_lines(smt_lines: List[str], z3_message: str) -> List[int]:
    """Return sorted list of 1-based line numbers mentioned in Z3’s message."""
    nums: Set[int] = {int(m.group(1)) for m in _LINE_RE.finditer(z3_message)}
    return sorted(n for n in nums if 1 <= n <= len(smt_lines))

# ────────────────────────────────────────────────────────────────────
class SMTOnepassSolverBasedErrorRepairer(dspy.Module):
    """Repair syntactic / typing errors until the program at least parses."""

    MAX_ATTEMPTS = 2

    def __init__(self, engine):
        super().__init__()
        self.engine = engine                      # e.g. dspy.OpenAI("gpt-4o")

    # ------------------------------------------------------------------
    def _build_prompt(self,
                      bad_idxs: List[int],
                      smt_lines: List[str],
                      z3_message: str) -> str:
        snippet = "\n".join(f"{i:>4}: {smt_lines[i-1]}" for i in bad_idxs)
        return (ERROR_FIX_TEMPLATE
                .replace("#ERROR_MSG#",   z3_message.strip())
                .replace("#BAD_SNIPPET#", snippet))

    # ------------------------------------------------------------------
    def forward(self, context: dict) -> dict:         # type: ignore[override]
        sc          = context["solver_check"]
        smt_lines   = context["smt_program_lines"]
        z3_message  = sc.get("message", "")

        bad_idxs = _extract_bad_lines(smt_lines, z3_message)
        if not bad_idxs:
            logger.warning("ErrorRepairer: no line numbers found in Z3 message")
            return context

        prompt = self._build_prompt(bad_idxs, smt_lines, z3_message)
        logger.info("ErrorRepairer prompt:\n%s", prompt)

        replacement_block = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                replacement_block = self.engine(prompt)[0].splitlines()
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                continue

            if len(replacement_block) == len(bad_idxs):
                break
            logger.warning("LLM did not return the expected number of lines "
                           "(got %d, expected %d)",
                           len(replacement_block), len(bad_idxs))
            replacement_block = None

        if replacement_block is None:
            logger.error("ErrorRepairer failed; no patch applied.")
            return context

        # splice
        for idx, newln in zip(bad_idxs, replacement_block):
            smt_lines[idx-1] = newln.rstrip()

        context["patched_smt_program_lines"] = smt_lines
        logger.info("ErrorRepairer: patched %d line(s).", len(bad_idxs))
        return context
