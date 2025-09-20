# expansion_consistency_verifier.py
import json, textwrap, re
import dspy
from typing import Dict, List, Any, Tuple


def _unwrap(txt: str) -> str:
    """提取 ```json … ``` 或原始文本"""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S)
    return m.group(1).strip() if m else txt

# ---------------------------------------------------------------------------

_EXPANSION_VERIFIER_PROMPT = """
# === GOAL ===
You are an expert clinical-text auditor.

In the last module, for each patient fact, coordinated medical-entity spans have been expanded in noun form so every medically relevant entity becomes a single, contiguous phrase (duplicate shared heads where needed).
Your task is two-fold:
Verify that every medically relevant entity has been correctly rewritten as a single, contiguous phrase (shared heads duplicated as needed).
Ensure each rewritten patient fact is semantically identical to its original, with absolutely no new errors or changes in meaning.

# === INPUTS ===

ORIGINAL patient facts:
{{ORIGINAL_LIST}}

EXPANDED patient facts (after entity-span expansion):
{{EXPANDED_LIST}}


# === CHECKLIST ===

Check each pair line-by-line according to ALL six rules:

1. No information is lost.
2. No extra information is added.
3. Shared-head / coordinated entities are fully expanded.
4. Abbreviations are fully expanded.
5. Pronouns/references are resolved to their explicit referents.
6. The original fact and the rewritten fact describe the same patient state (no change to assertion, scope, polarity, or temporality).

# === GUIDELINES ===
1. For each fact in the list, walk through the checklist one rule at a time, totalling the above 6 rules. Write the result of each to the corresponding "rule x": <true|false> field.
2. If all are satisfied, we call it a good expansion, and mark the "good_expansion" field as true. If any rule is not satisfied, we mark good_expansion as false.
3. In the field of "explanation", explain why you came up with the final decision on "good_expansion".

===  OUTPUT FORMAT  ===
Return **valid JSON** with these keys ONLY:
{
    "<fact_index_1>":
        {
        "rule 1": <true|false>,
        "rule 2": <true|false>,
        "rule 3": <true|false>,
        "rule 4": <true|false>,
        "rule 5": <true|false>,
        "rule 6": <true|false>,
        "good_expansion": <true|false>,          // true → every line passes every rule
        "explanation": "...your explanation..."
        "corrected_fact": "<candidate rewrite that makes all items YES>",
        },
    "<fact_index_2>":
    ...
}

Respond with the JSON alone — no extra commentary.
"""

# ---------------------------------------------------------------------------

class PatientStateEntitySurfaceExpanderVerifier(dspy.Module):
    """Return parsed JSON to feed iterative correction loop."""

    def __init__(self, engine: dspy.Module, max_attempts: int = 2, debug: bool = False):
        super().__init__()
        self.engine, self.max_attempts, self.debug = engine, max_attempts, debug

    def _d(self, *msg):
        if self.debug:
            print(*msg, flush=True)

    def forward(
        self,
        ctx: dict,
        original_reqs: List[str],
        expanded_reqs: List[str],
    ) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
        prompt = (
            _EXPANSION_VERIFIER_PROMPT
            .replace("{{ORIGINAL_LIST}}", textwrap.indent(json.dumps(original_reqs, ensure_ascii=False, indent=2), "  "))
            .replace("{{EXPANDED_LIST}}", textwrap.indent(json.dumps(expanded_reqs, ensure_ascii=False, indent=2), "  "))
        )
        raw, parsed, last_err = None, {}, None
        for _ in range(self.max_attempts):
            raw = self.engine(prompt)[0]
            try:
                parsed = json.loads(_unwrap(raw))
                if not isinstance(parsed, dict):
                    raise ValueError("verifier output must be dict")
                break
            except Exception as e:
                last_err = e
        if not parsed:  # fabricate fully negative
            parsed = {str(i): {"good_expansion": False, "explanation": str(last_err), "corrected_fact": expanded_reqs[i]} for i in range(len(original_reqs))}
        failed, reasons = [], []

        for idx, det in parsed.items():
            i = int(idx)
            # attach raw text for easier diffing in logs
            det.setdefault("original", original_reqs[i] if i < len(original_reqs) else "")
            det.setdefault("rewritten", expanded_reqs[i] if i < len(expanded_reqs) else "")
            if not det.get("good_expansion", False):
                failed.append(idx)
                reasons.append(det.get("explanation", ""))

        return len(failed) == 0, failed, reasons, parsed