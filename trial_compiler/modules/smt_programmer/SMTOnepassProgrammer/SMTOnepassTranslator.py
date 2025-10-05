"""
onepass_translator.py
---------------------
Build a single pre-tagged SMT-LIB program (keep LLM output verbatim).

context outputs
---------------
smt_program_lines : list[str]
smt_program_text  : str
req_blocks        : dict[int, (start, end)]
assertion_tags    : dict[str, int]
"""
from __future__ import annotations
import re, logging
from typing import List, Dict, Tuple
import dspy

# ────────── logging ──────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                        level=logging.INFO)

# ────────── regex helpers ──────────
_REQ_HDR_RE = re.compile(r";;\s*REQ(?:UIREMENT)?\s+#?(\d+)", re.I)
_TAG_RE     = re.compile(r":named\s+(req(\d+)_(\d+))\)", re.I)
_WRAP_RE    = re.compile(r"</?output_complete_smt_program\s*>", re.I)
_FENCE_RE   = re.compile(r"^```(?:smt)?|```$", re.I | re.M)

# ────────── utilities ──────────
def _unwrap_program(text: str) -> str:
    """Strip XML wrapper + markdown fences, return raw SMT text."""
    text = _WRAP_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    return text.strip()

def _scan_and_partition(lines: List[str]
        ) -> Tuple[Dict[int, Tuple[int, int]], Dict[str, int]]:
    """Map requirement slices and :named tags."""
    req_blocks, tag_map = {}, {}
    cur_req, req_start  = 0, 0
    for idx, ln in enumerate(lines):
        if (m := _REQ_HDR_RE.match(ln.strip())):
            req_blocks[cur_req] = (req_start, idx)
            cur_req, req_start = int(m.group(1)), idx + 1
        elif (m := _TAG_RE.search(ln)):
            tag_map[m.group(1)] = int(m.group(2))
    req_blocks[cur_req] = (req_start, len(lines))
    return req_blocks, tag_map

# ────────── translator ──────────
class SMTOnepassTranslator(dspy.Module):
    """One-shot translator that keeps the LLM’s SMT verbatim."""
    MAX_ATTEMPTS = 1

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def _assemble_prompt(self, ctx: dict) -> str:
        tmpl = ctx.get("SMTOnepassTranslator_prompt", "")
        if not tmpl:
            raise ValueError("Missing `SMTOnepassTranslator_prompt`")

        req_lines = []
        for idx, r in enumerate(ctx.get("requirements", []), 1):
            req_text = r["requirement"] if isinstance(r, dict) else str(r)
            req_lines.append(
                f";; REQUIREMENT {idx}\n"
                f";; {req_text.strip()}\n"
                f"(assert (! <fill-me> :named req{idx}_1))"
            )

        return (tmpl
                .replace("#CONTEXTUAL_TEXT#",    ctx.get("contextual_text", ""))
                .replace("#REQ_BLOCKS#",         "\n".join(req_lines))
                .replace
                .replace("#SMT_PROGRAM_BY_FAR#", ctx.get("smt_program_so_far", "")))

    # ------------------------------------------------------------------
    def forward(self, context: dict) -> dict:            # type: ignore[override]
        prompt     = self._assemble_prompt(context)
        raw_reply  = self.engine(prompt)[0]
        logger.info("LLM reply:\n%s", raw_reply)

        smt_text = _unwrap_program(raw_reply)
        lines    = smt_text.splitlines()

        # Ensure `set-logic` line exists
        if not any(ln.lstrip().startswith("(set-logic") for ln in lines):
            lines.insert(0, "(set-logic ALL)")

        req_blocks, tag_map = _scan_and_partition(lines)

        # Pretty-print for humans
        print("\n===== SMT-LIB program (%d lines) =====" % len(lines))
        for i, ln in enumerate(lines, 1):
            print(f"{i:4}: {ln}")

        # Store in context
        context.update(
            smt_program_lines = lines,
            smt_program_text  = "\n".join(lines),
            req_blocks        = req_blocks,
            assertion_tags    = tag_map,
        )
        return context
