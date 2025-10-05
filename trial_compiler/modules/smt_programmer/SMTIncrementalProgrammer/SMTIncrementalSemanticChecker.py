# modules/smt_semantic_checker.py
import re
import json
import dspy
from typing import Dict, Any

# ------------------------------------------------------------------ #
# Fallback prompt (used if caller did NOT load a custom one)         #
# ------------------------------------------------------------------ #
SEMANTIC_CHECK_PROMPT_FALLBACK = r"""
# instruction
You are an expert formal‑methods reviewer.
…
Return **only** JSON like this:

{{ "ok": false, "issues": ["...", "..."] }}
(no markdown)

#REQUIREMENT#
---
{requirement}
#VARIABLES#
---
{variables}
#ASSERTIONS#
---
{assertions}
""".strip()

# ------------------------------------------------------------------ #
# Tag patterns expected in the generated SMT slice                   #
# ------------------------------------------------------------------ #
# The generation pipeline now wraps fresh code in the tags
#   <new_variables> … </new_variables>
#   <new_requirement_implementation> … </new_requirement_implementation>
# Update the regexes accordingly so we can reliably extract them.
_VAR_RE = re.compile(r"<new_variables>(.*?)</new_variables>", re.S)
_EXPR_RE = re.compile(r"<new_requirement_implementation>(.*?)</new_requirement_implementation>", re.S)

# ------------------------------------------------------------------ #
# Utilities                                                          #
# ------------------------------------------------------------------ #

def _run_semantic_llm(engine, prompt: str) -> Dict[str, Any]:
    """Call the LLM twice at most, forcing JSON if the first call fails."""
    for _ in range(2):
        raw = engine(prompt)
        # Some engines return a list/tuple of messages, others a string.
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        raw = str(raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # On the second iteration we prepend an explicit instruction.
            prompt = "Respond ONLY with JSON:\n" + prompt
    # If all else fails, surface a clear error for downstream handling.
    return {"ok": False, "issues": ["Could not parse LLM output as JSON"]}


class SMTIncrementalSemanticChecker(dspy.Module):
    """Check each newly‑added SMT slice for semantic soundness via an LLM."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    # ------------------------------------------------------------------ #
    def forward(self, context: dict) -> dict:  # type: ignore[override]
        print("In SMTSemanticChecker")

        idx: int = context["current_requirement_index"]

        # --------------------------------------------------------------
        # 1. Fetch the requirement text
        # --------------------------------------------------------------
        req_entry = context["requirements"][idx]
        requirement = (
            req_entry["requirement"] if isinstance(req_entry, dict) else str(req_entry)
        )

        # --------------------------------------------------------------
        # 2. Extract the SMT slice corresponding to this requirement
        # --------------------------------------------------------------
        start, end = context["req_blocks"][idx]
        slice_lines = context["smt_program_lines"][start:end]
        slice_text = "\n".join(slice_lines)

        # Try to pull the tagged blocks first; fall back to last two lines.
        var_match = _VAR_RE.search(slice_text)
        expr_match = _EXPR_RE.search(slice_text)

        if var_match and expr_match:
            var_block = var_match.group(1).strip()
            assert_block = expr_match.group(1).strip()
        elif len(slice_lines) >= 2:
            var_block, assert_block = slice_lines[-2].strip(), slice_lines[-1].strip()
        else:
            var_block = slice_lines[0].strip() if slice_lines else ""
            assert_block = ""

        # --------------------------------------------------------------
        # 3. Assemble the prompt
        # --------------------------------------------------------------
        prompt_tmpl = context.get(
            "semantic_checker_prompt", SEMANTIC_CHECK_PROMPT_FALLBACK
        )

        try:
            prompt = prompt_tmpl.format(
                requirement=requirement,
                variables=var_block,
                assertions=assert_block,
            )
        except KeyError as e:
            # Surface mis‑matched placeholders clearly
            raise ValueError(
                f"Missing placeholder in semantic‑checker template: {e}"
            )

        # --------------------------------------------------------------
        # 4. Query the LLM and record the verdict
        # --------------------------------------------------------------
        verdict = _run_semantic_llm(self.engine, prompt)
        ok: bool = bool(verdict.get("ok"))
        issues = verdict.get("issues", [])

        # --------------------------------------------------------------
        # 5. Persist results into the shared context
        # --------------------------------------------------------------
        context.setdefault("semantic_check", {})[idx] = {"ok": ok, "issues": issues}
        context["semantic_ok"] = ok
        return context
