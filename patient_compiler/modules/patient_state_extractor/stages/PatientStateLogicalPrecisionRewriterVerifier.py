import json
import re
from typing import Dict, List, Tuple, Any

import dspy

# ---------- pairs extraction (dual-mode) ----------
def _pairs_from_context(context: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """
    Returns [(idx, original_text, rewritten_text), ...]
    Works with either:
      - context["requirements"]   items having "requirement"
      - context["patient_facts"]  items having "fact"
    """
    slot_key = "patient_facts" if context.get("patient_facts") else "requirements"
    text_key = "fact" if slot_key == "patient_facts" else "requirement"

    pairs: List[Tuple[int, str, str]] = []
    items = context.get(slot_key, [])
    for i, r in enumerate(items):
        if isinstance(r, dict):
            orig = str(r.get("source", r.get(text_key, ""))).strip()
            rew  = str(r.get(text_key, "")).strip()
        else:
            orig = str(r).strip()
            rew  = str(r).strip()
        pairs.append((i, orig, rew))
    return pairs

def _render_pairs(pairs: List[Tuple[int, str, str]]) -> str:
    lines = []
    for i, orig, rew in pairs:
        lines.append(f"[{i:02d}] ORIGINAL: {orig}")
        lines.append(f"[{i:02d}] REWRITTEN: {rew}")
    return "\n".join(lines)

# ---------- robust JSON extractor (supports both tags or bare JSON) ----------
def _extract_json_from_block(text: str) -> Dict[str, Any]:
    """
    Try to extract JSON from a tagged block first; if no known tags are found,
    fall back to parsing the first top-level JSON object present in the text.
    """
    tag_patterns = [
        r"<rewritten_requirement_list>\s*(.*?)\s*</rewritten_requirement_list>",
        r"<rewritten_patient_fact_list>\s*(.*?)\s*</rewritten_patient_fact_list>",
        r"<precision_verifier>\s*(.*?)\s*</precision_verifier>",
        r"<verifier_output>\s*(.*?)\s*</verifier_output>",
    ]

    m = None
    for pat in tag_patterns:
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            break

    if m:
        inner = m.group(1).strip()
        # strip // comments (common in LLM outputs)
        inner = re.sub(r"//.*", "", inner)
        start = inner.find("{")
        end = inner.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON braces not found in verifier output")
        candidate = inner[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)

    # Fallback: try to parse the first JSON object anywhere in the text
    # (crude but works well in practice)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)

    raise ValueError("Could not find JSON in verifier output")

class PatientStateLogicalPrecisionRewriterVerifier(dspy.Module):
    def __init__(self, engine, max_attempts: int = 3):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pairs = _pairs_from_context(context)
        pairs_text = _render_pairs(pairs)

        prompt = context["PatientStateLogicalPrecisionRewriterVerifier_prompt"]
        prompt = prompt.replace("#ORIGINAL_REWRITE_PAIRS#", pairs_text)

        attempts = 0
        last_error = None
        raw = None
        parsed: Dict[str, Any] = {}

        while attempts < self.max_attempts:
            raw = self.engine(prompt)[0]
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
            # fabricate a fully-negative result so the caller can fall back
            by_index: Dict[str, Any] = {}
            for i, _o, _r in pairs:
                rec = {
                    k: "NO" for k in [
                        "HIERARCHIES_CAPTURED",
                        "ABBREVIATIONS_EXPANDED", "REMOVE_REDUNDANT_CLAUSES", "DUPLICATE_QUALIFIERS",
                        "PRESERVE_SEMANTIC_MEANING", "NO_NEW_INFORMATION", "ALL_GOOD"
                    ]
                }
                rec["explanation"] = f"Verifier failed: {last_error}"
                # provide both keys so downstream code can consume either
                rec["corrected_requirement"] = ""
                rec["corrected_fact"] = ""
                by_index[str(i)] = rec
            parsed = {"by_index": by_index}

        for idx, orig, rew in pairs:
            rec = parsed.setdefault("by_index", {}).setdefault(str(idx), {})
            rec.setdefault("original", orig)
            rec.setdefault("rewritten", rew)

        yes = sum(1 for v in parsed.get("by_index", {}).values() if v.get("ALL_GOOD") == "YES")
        no  = sum(1 for v in parsed.get("by_index", {}).values() if v.get("ALL_GOOD") == "NO")

        context["verification"] = {
            "raw": raw,
            "parsed": parsed,
            "all_good_counts": {"YES": yes, "NO": no},
        }
        return context
