import re
import json
import textwrap
from typing import Dict, Any, List

import dspy
from smt_core.parse_functions import parse_corrected_program

# ---------------------------------------------------------------------------
# Fallback prompt template (uses #PLACEHOLDERS# so users can override easily)
# ---------------------------------------------------------------------------
FALLBACK_PROMPT = textwrap.dedent("""\
    # instruction
    You are an expert SMT‑LIB engineer and Z3 debugger.
    The full SMT program below returned **{status}** in Z3.

    Your job: rewrite ONLY the assertions (or declarations) named in the
    unsat‑core so that the program becomes **sat**.  Keep the meaning of the
    clinical‑trial eligibility constraints as intact as possible.

    ## Context
    • Z3 message  : {message}
    • unsat‑core  : {unsat_core}

    ## Output format (STRICT)
    <scratchpad>
    (you may think step‑by‑step here – ignored by the parser)
    </scratchpad>

    >>> Corrected Program:
    <corrected_program>
    … replacement lines here …
    </corrected_program>
""")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r":named\s+([^)\s]+)")


def _line_tag(line: str) -> str | None:
    """Return the :named TAG if present in *line*, else None."""
    m = _TAG_RE.search(line)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Global refiner
# ---------------------------------------------------------------------------

class SMTOnepassSolverBasedUnsatCoreRefiner(dspy.Module):
    """LLM‑powered fixer that patches *only* the assertions named in the
       unsat‑core.  The new lines are appended at the end of the program; the
       validator will re‑run Z3 afterwards.
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    # ------------------------------------------------------------------
    def forward(self, context: Dict[str, Any], solver_result: Dict[str, Any]):
        print("In SMTOnepassSolverBasedRefiner\n")

        smt_lines: List[str] = context.get("smt_program_lines", [])
        if not smt_lines:
            raise ValueError("Global refiner expects 'smt_program_lines' in context")

        core_tags = set(solver_result.get("unsat_core") or [])
        if not core_tags:
            print("No unsat‑core provided – nothing to refine.")
            return context

        # ------------------------------------------------------------------
        # 1) Build LLM prompt
        # ------------------------------------------------------------------
        user_tmpl = context.get("SolverbasedOnepassUnsatCoreRefiner_prompt")
        tmpl = user_tmpl or FALLBACK_PROMPT

        prompt = (
            tmpl.replace("{status}", solver_result.get("status", "error"))
                .replace("{message}", solver_result.get("message", ""))
                .replace("{unsat_core}", json.dumps(sorted(core_tags)))
        )

        prompt += "\n\n<full_program>\n" + "\n".join(smt_lines) + "\n</full_program>\n"

        llm_out = self.engine(prompt)[0]

        new_block = parse_corrected_program(llm_out)
        if new_block is None:
            context.setdefault("refiner_failures", []).append(
                {"idx": "global", "reason": "no_corrected_program_tag", "raw": llm_out[:250]}
            )
            return context

        # ------------------------------------------------------------------
        # 2) Remove ONLY the core‑implicated assertions/declarations
        # ------------------------------------------------------------------
        patched: List[str] = []
        for ln in smt_lines:
            tag = _line_tag(ln)
            if tag and tag in core_tags:
                continue  # drop
            patched.append(ln)

        # ------------------------------------------------------------------
        # 3) Append the corrected lines (LLM output)
        # ------------------------------------------------------------------
        patched.extend(new_block)
        patched.append("")

        context["patched_smt_program_lines"] = patched
        return context
