import json
import re
from typing import Dict, List, Tuple, Any

import dspy

def _pairs_from_context(context: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    pairs = []
    reqs = context.get("requirements", [])
    for i, r in enumerate(reqs):
        if isinstance(r, dict) and "source" in r and "requirement" in r:
            orig = str(r["source"]).strip()
            rew = str(r["requirement"]).strip()
        else:
            orig = str(r)
            rew = str(r)
        pairs.append((i, orig, rew))
    return pairs

def _render_pairs(pairs: List[Tuple[int, str, str]]) -> str:
    lines = []
    for i, orig, rew in pairs:
        lines.append(f"[{i:02d}] ORIGINAL: {orig}")
        lines.append(f"[{i:02d}] REWRITTEN: {rew}")
    return "\n".join(lines)

def _extract_json_from_block(text: str) -> Dict[str, Any]:
    m = re.search(r"<rewritten_requirement_list>\s*(.*?)\s*</rewritten_requirement_list>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("Missing <rewritten_requirement_list> block")
    inner = m.group(1).strip()
    inner = re.sub(r"//.*", "", inner)
    start = inner.find("{")
    end = inner.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON braces not found in verifier output")
    candidate = inner[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)

class RequirementLogicalPrecisionRewriterVerifier(dspy.Module):
    def __init__(self, engine, max_attempts: int = 3):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mode = context.get("inc_exc")
        assert mode in {"inclusion", "exclusion"}

        pairs = _pairs_from_context(context)
        pairs_text = _render_pairs(pairs)

        if mode == "inclusion":
            prompt = context["RequirementLogicalPrecisionRewriterVerifierInclusion_prompt"]
        else:
            prompt = context["RequirementLogicalPrecisionRewriterVerifierExclusion_prompt"]

        prompt = prompt.replace("#ORIGINAL_REWRITE_PAIRS#", pairs_text)

        # print(f"RequirementLogiclaPrecisionRewriterVerifier \n prompt is {prompt}")

        attempts = 0
        last_error = None
        raw = None
        parsed: Dict[str, Any] = {}
        while attempts < self.max_attempts:
            raw = self.engine(prompt)[0]
            # print(f"In RequirementLogicalPrecisionRewriter, raw is \n{raw}")
            try:
                parsed = _extract_json_from_block(raw)
                if not isinstance(parsed, dict) or "by_index" not in parsed:
                    raise ValueError("Missing 'by_index' in parsed output")
                break
            except Exception as e:
                last_error = e
                attempts += 1
                print(f"[rewrite-verifier] retry {attempts}/{self.max_attempts}: {e}")

        if not parsed:
            by_index: Dict[str, Any] = {}
            for i, _o, _r in pairs:
                by_index[str(i)] = {
                    k: "NO" for k in [
                        "HIERARCHIES_CAPTURED", "SAME_PATIENT_POOL_MAPPING", "AND_OR_XOR_NOT",
                        "ABBREVIATIONS_EXPANDED", "REMOVE_REDUNDANT_CLAUSES", "DUPLICATE_QUALIFIERS",
                        "PRESERVE_SEMANTIC_TEMPLATE", "NO_NEW_INFORMATION", "ALL_GOOD"
                    ]
                }
                by_index[str(i)]["explanation"] = f"Verifier failed: {last_error}"
                by_index[str(i)]["corrected_requirement"] = ""
            parsed = {"by_index": by_index}

        yes = sum(1 for v in parsed.get("by_index", {}).values() if v.get("ALL_GOOD") == "YES")
        no = sum(1 for v in parsed.get("by_index", {}).values() if v.get("ALL_GOOD") == "NO")

        context["verification"] = {
            "mode": mode,
            "raw": raw,
            "parsed": parsed,
            "all_good_counts": {"YES": yes, "NO": no},
        }
        return context
