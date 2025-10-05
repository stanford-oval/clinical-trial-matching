# ========================= entity_span_expander.py =========================
import json
import re
from typing import List, Dict, Union, Tuple
import dspy
from pathlib import Path


# ---------------------------------------------------------------------------
# ★ Prompt template ----------------------------------------------------------
# ---------------------------------------------------------------------------
from .RequirementEntitySurfaceExpanderVerifier import RequirementEntitySurfaceExpanderVerifier

_SPAN_PROMPT = (
    "You are a careful, deterministic clinical-trial text editor.\n"
    "Goal: rewrite each requirement so that every potential medical entity "
    "inside a coordinated list (comma, 'and', 'or') becomes a single, contiguous "
    "span.  If you see a shared-head construction, duplicate the missing head "
    "so each item stands alone, while keeping conjunctions, punctuation, "
    "capitalisation, and meaning unchanged.\n\n"

    "Examples—rewrite the *underlined* part only:\n"
    "  • retropharyngeal *or* buccal _cellulitis_ → retropharyngeal cellulitis "
    "or buccal cellulitis\n"
    "  • fracture _of_ tibia *and* fibula        → fracture of tibia and "
    "fracture of fibula\n"
    "  • tonsil *and/or* pharyngeal _erythema_ *and/or* exudate → tonsil erythema or tonsil exudate or pharyngeal erythema or pharyngeal exudate\n\n"

    "#CONTEXTUAL_TEXT#\n\n"
    "Original requirement list:\n#REQUIREMENT_TEXT#\n\n"

    "# === OUTPUT FORMAT (output exactly these two blocks) ===\n"
    "In the <rewritten_requirement_list> block below. Return the SAME number of lines, each prefixed with its original 0-based index in square brackets, e.g.:\n"
    "    \"[00] rewritten requirement …\n"
    "     [01] rewritten requirement …\n\n"
    "<rewritten_requirement_list>\n"
    "[00] …\n"
    "</rewritten_requirement_list>\n"
)

# Block tags expected from the LLM
_OPEN_TAG = "<rewritten_requirement_list>"
_CLOSE_TAG = "</rewritten_requirement_list>"

# ---------------------------------------------------------------------------
# ★ Helper: minimal parser for the model output -----------------------------
# ---------------------------------------------------------------------------

def _extract_block(raw: str) -> str | None:
    """
    Return the raw text between <rewritten_requirement_list> ... </rewritten_requirement_list>,
    case-insensitive, tolerant to extra whitespace/newlines around tags.
    """
    open_pat  = r"<\s*rewritten_requirement_list\s*>"
    close_pat = r"<\s*/\s*rewritten_requirement_list\s*>"
    m = re.search(open_pat + r"(.*?)" + close_pat, raw, re.S | re.I)
    if m:
        return m.group(1).strip()
    return None


def parse_rewrite_output(raw: str, expect_n: int) -> Union[List[str], bool]:
    """Parse LLM output in the expected tag-wrapped, index-prefixed format."""
    body = _extract_block(raw)
    if body is None:
        return False

    # Strip stray code fences if present
    body = re.sub(r"^```.*?$|```$", "", body, flags=re.M)

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    pattern = re.compile(r"^\[(\d+)\]\s*(.+)$")

    collected: List[Tuple[int, str]] = []
    for ln in lines:
        m = pattern.match(ln)
        if not m:
            return False
        idx, text = int(m.group(1)), m.group(2).strip()
        collected.append((idx, text))

    if len(collected) != expect_n:
        return False

    collected.sort(key=lambda t: t[0])
    have_idxs = [i for i, _ in collected]
    want_idxs = list(range(expect_n))

    if have_idxs != want_idxs:
        # Repair only if indices are unique, contiguous, but offset/padded
        if len(set(have_idxs)) == expect_n and (max(have_idxs) - min(have_idxs) + 1) == expect_n:
            collected = list(zip(range(expect_n), [t[1] for t in collected]))
        else:
            return False

    return [txt for _, txt in collected]

# ---------------------------------------------------------------------------
# ★ Main DSPy Module ---------------------------------------------------------
# ---------------------------------------------------------------------------

class RequirementEntitySpanExpander(dspy.Module):
    """
    LLM-driven span expander **with built-in consistency verification**.
    """

    MAX_ATTEMPTS = 4      # how many end-to-end tries (rewrite + verify)
    PARSE_RETRY  = 3      # within one attempt, how many times to re-query if format parse fails

    def __init__(self, engine: dspy.Module):
        super().__init__()
        self.engine   = engine
        self.verifier = RequirementEntitySurfaceExpanderVerifier(engine, max_attempts=3)  # <<< CHANGED (3 attempts)

    def _build_prompt(
        self,
        ctx_txt: str,
        req_lines: List[str],
        ctx: Dict,
    ) -> str:
        prompt = ctx.get("RequirementEntitySpanExpander_prompt", _SPAN_PROMPT)
        n = len(req_lines)
        index_list = " ".join(f"[{i:02d}]" for i in range(n))

        strict_header = (
            f"\n# === STRICT OUTPUT RULES ===\n"
            f"- You MUST output exactly {n} lines in the block.\n"
            f"- Each line MUST start with one of these indices exactly once: {index_list}\n"
            f"- Use the SAME order as the originals.\n"
            f"- Do NOT add commentary, code fences, or any text outside the block.\n"
            f"- Do NOT skip or duplicate indices.\n"
        )

        prompt = prompt + strict_header
        return (prompt
                .replace("#CONTEXTUAL_TEXT#",  ctx_txt)
                .replace("#REQUIREMENT_TEXT#", "\n".join(req_lines)))

    def forward(self, context: Dict, use_full_context: bool = True) -> Dict:
        ctx_txt = context.get("contextual_text", "")
        req_items: List[Union[str, Dict]] = context.get("requirements", [])
        orig_lines: List[str] = [
            itm["requirement"] if isinstance(itm, dict) else str(itm)
            for itm in req_items
        ]

        attempt_logs: List[Dict] = []
        best_attempt: Dict | None = None

        for att in range(1, self.MAX_ATTEMPTS + 1):
            prompt = self._build_prompt(ctx_txt, orig_lines, context)
            # print(f"expander prompt is {prompt}")

            parse_ok = False
            raw = ""
            for _ in range(self.PARSE_RETRY):
                raw = self.engine(prompt)[0]
                # print(f"expander raw is {raw}")
                rewritten = parse_rewrite_output(raw, expect_n=len(orig_lines))
                if rewritten is not False:
                    parse_ok = True
                    break

            if not parse_ok:
                print("[EntitySpanExpander] parse failed; falling back to originals.")
                rewritten = orig_lines

            ok, failed, reasons = self.verifier(
                ctx=context,
                original_reqs=orig_lines,
                expanded_reqs=rewritten,
                ctx_txt=ctx_txt,
            )

            attempt_log = {
                "attempt":        att,
                "parse_ok":       parse_ok,
                "all_passed":     ok,
                "failed_indices": failed,
                "reasons":        reasons,
                "mapping":        dict(zip(orig_lines, rewritten)),
                "rewritten":      rewritten,
                "raw":            (raw[:400] if isinstance(raw, str) else ""),
            }
            attempt_logs.append(attempt_log)

            if ok:
                best_attempt = attempt_log
                print(f"[EntitySpanExpander] verifier PASSED at attempt {att}\n")
                break
            else:
                if (best_attempt is None or
                    len(failed) < len(best_attempt["failed_indices"])):
                    best_attempt = attempt_log

                print(f"[EntitySpanExpander] attempt {att} FAILED")
                print("  failed indices :", failed or "—")
                print("  reasons        :", "; ".join(reasons) or "—\n")

        if best_attempt is None:
            best_attempt = attempt_logs[-1]

        final = best_attempt
        final_mapping = dict(zip(orig_lines, final["rewritten"]))

        context["requirements"] = [
            {"requirement": new, "source": old}
            for old, new in zip(orig_lines, final["rewritten"])
        ]
        context["span_mapping"] = final_mapping
        context["span_expansion_metrics"] = {
            k: v for k, v in final.items()
            if k in {"attempt", "parse_ok", "all_passed", "failed_indices", "reasons"}
        }
        context["span_attempt_logs"] = attempt_logs

        # --- minimal persistent logging (per trial) -------------------
        # Directory:  mbench/req_mbench/span_logs/<trial>_<inc_exc>/
        trial_id = str(context.get("trial_id", "unknown_trial"))
        list_type = str(context.get("inc_exc", "inclusion"))
        base_dir = Path("mbench/req_mbench/expansion_maps") / f"{trial_id}_{list_type}"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Filenames
        attempts_path = base_dir / "span_attempt_logs.json"
        final_map_path = base_dir / "span_final_mapping.json"
        final_reqs_path = base_dir / "span_final_requirements.json"
        metrics_path = base_dir / "span_expansion_metrics.json"

        # Write logs
        try:
            attempts_path.write_text(json.dumps(attempt_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            final_map_path.write_text(json.dumps(final_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
            final_reqs_path.write_text(json.dumps(context["requirements"], ensure_ascii=False, indent=2), encoding="utf-8")
            metrics_path.write_text(json.dumps(context["span_expansion_metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[EntitySpanExpander] logs written → {base_dir.resolve()}")
        except Exception as e:
            print(f"[EntitySpanExpander] log write failed: {e}")

        return context



# ---------------------------------------------------------------------------
# ★ Minimal standalone convenience runner -----------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    class EchoEngine(dspy.Module):
        """Mock engine that echoes prompt for quick CLI debug."""
        def forward(self, prompt):
            print("\nPROMPT→\n", prompt)
            manual = input("\nPaste model answer then hit Enter twice: \n")
            return [manual]

    dummy_ctx = {
        "requirements": [
            "retropharyngeal or buccal cellulitis may occur in children",
            "infection of lung or pleura should be treated promptly",
        ]
    }

    expander = RequirementEntitySpanExpander(engine=EchoEngine())
    result = expander(dummy_ctx)
    print("\nRESULT→", json.dumps(result, indent=2, ensure_ascii=False))
