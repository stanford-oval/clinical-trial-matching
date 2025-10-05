# ========================= expansion_consistency_verifier.py =========================
import json, textwrap, re, time
import dspy

def _unwrap(txt: str) -> str:
    """提取 ```json … ``` 或原始文本"""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S)
    return m.group(1).strip() if m else txt

_NUM_KEY_RE = re.compile(r"^\d+$")

def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true","yes","y","1"}: return True
        if s in {"false","no","n","0"}: return False
    return False

def _normalize_item(item) -> dict:
    if isinstance(item, dict):
        rules = {f"rule {i}": _to_bool(item.get(f"rule {i}", False)) for i in range(1,7)}
        ge = item.get("good_expansion")
        ge = all(rules.values()) if ge is None else _to_bool(ge)
        expl = item.get("explanation") if isinstance(item.get("explanation"), str) else ""
        return {**rules, "good_expansion": ge, "explanation": expl.strip()}
    if isinstance(item, (bool,int,float,str)):
        b = _to_bool(item)
        rules = {f"rule {i}": b for i in range(1,7)}
        return {**rules, "good_expansion": b, "explanation": "Coerced non-object entry from model response."}
    rules = {f"rule {i}": False for i in range(1,7)}
    return {**rules, "good_expansion": False, "explanation": "Invalid entry type; treated as failure."}

# ---------------------------------------------------------------------------
# <<< CHANGED: mirror Guideline #4 (universally-true constraint_clauses okay to omit)
_EXPANSION_VERIFIER_PROMPT = """
# === GOAL ===
You are an expert clinical-trial text auditor.

In the last module, for each requirement, coordinated medical-entity spans have been expanded in noun form so every medically relevant entity becomes a single, contiguous phrase (duplicate shared heads where needed). 
Your task is two-fold:
Verify that every medically relevant entity has been correctly rewritten as a single, contiguous phrase (shared heads duplicated as needed).
Ensure each rewritten requirement is semantically and logically identical to its original, with absolutely no new errors or changes in meaning.

# === INPUTS ===

ORIGINAL trial requirements:
{{ORIGINAL_LIST}}

EXPANDED trial requirements (after entity-span expansion):
{{EXPANDED_LIST}}

# === CHECKLIST ===
Check each pair line-by-line according to ALL six rules:
1. No information is lost.
2. No extra information is added.
3. Shared-head / coordinated entities are fully expanded.
4. Abbreviations are fully expanded.
5. Pronouns are resolved to their explicit referents.
6. The original requirement and the rewritten requirements map to the same pool of patients.

# === GUIDELINES ===
- Respect these clarifications when judging equivalence:
  1) Clauses that are universally true (e.g., “of any race”, “of any sex/gender”, “male and female”) are informationally neutral and MAY be omitted without counting as information loss.
  2) Preserve original logical relationships (AND stays AND, OR stays OR). Expand “and/or” as “or”.
  3) Do NOT import details across requirements or from context.
  4) Shared-head expansions should duplicate the missing head so each item stands alone.
  5) Convert adjectival medical descriptors to explicit noun phrases when appropriate.

# === OUTPUT FORMAT ===
Return valid JSON with ONLY numeric string keys 0..N-1, where N is the number of requirements.
Example:
{
  "0": {"rule 1": true, "rule 2": true, "rule 3": true, "rule 4": true, "rule 5": true, "rule 6": true, "good_expansion": true, "explanation": "…"},
  "1": { … }
}
Respond with the JSON alone — no extra commentary.
"""

class RequirementEntitySurfaceExpanderVerifier(dspy.Module):
    """
    Verifies that RequirementEntitySpanExpander output complies with
    logical-equivalence and coverage criteria, with retry + bypass.
    """
    def __init__(
        self,
        engine: dspy.Module,
        max_attempts: int = 3,
        debug: bool = False,
        bypass_on_failure: bool = True,
        ctx_flag_key: str = "span_expansion_verifier",
    ):
        super().__init__()
        self.engine           = engine
        self.max_attempts     = max_attempts
        self.debug            = debug
        self.bypass_on_failure= bypass_on_failure
        self.ctx_flag_key     = ctx_flag_key

    def _d(self, *msg):
        if self.debug:
            print(*msg, flush=True)

    def forward(
        self,
        ctx: dict,
        original_reqs: list[str],
        expanded_reqs: list[str],
        ctx_txt: str,
    ) -> tuple[bool, list[str], list[str]]:
        """
        Returns
        -------
        all_passed   : bool                 # True if every rule passes
        failed_reqs  : list[str]            # indices (as strings) that failed
        reasons      : list[str]            # brief explanations
        """
        n = len(original_reqs)

        # Allow caller to override prompt; otherwise use our robust default
        prompt = ctx.get("RequirementEntitySurfaceExpanderVerifier_prompt", _EXPANSION_VERIFIER_PROMPT)  # <<< CHANGED

        def _build_prompt(attempt: int) -> str:
            base = (
                prompt
                .replace(
                    "{{ORIGINAL_LIST}}",
                    textwrap.indent(json.dumps(original_reqs, indent=2, ensure_ascii=False), "  "),
                )
                .replace(
                    "{{EXPANDED_LIST}}",
                    textwrap.indent(json.dumps(expanded_reqs, indent=2, ensure_ascii=False), "  "),
                )
                .replace("#CONTEXTUAL_TEXT#",  ctx_txt)
            )
            if attempt == 1:
                return base
                print("[debug] raw verifier prompt:\n", base)
            if attempt >= 2:
                base += "\nSTRICT OUTPUT: Return a single JSON object with ONLY numeric string keys '0'..'"
                base += str(n-1) + ". No code fences, no prose."
            if attempt == self.max_attempts:
                base += "\nFINAL ATTEMPT: Minify the JSON (no extra whitespace)."
            return base

        last_err = None
        for attempt in range(1, self.max_attempts + 1):
            ptxt = _build_prompt(attempt)
            raw = self.engine(ptxt)[0]
            self._d(f"[Verifier attempt {attempt}] raw →", str(raw)[:200], "…")

            try:
                data = json.loads(_unwrap(str(raw)))

                # --- normalize to per-index dict ---------------------------
                per_index = {}

                if isinstance(data, list):
                    # List → enumerate to indices 0..N-1
                    for i, item in enumerate(data[:n]):
                        per_index[str(i)] = item

                elif isinstance(data, dict):
                    numeric_items = {str(k): v for k, v in data.items() if _NUM_KEY_RE.match(str(k))}
                    if numeric_items:
                        # <<< CHANGED: auto-shift 1..N → 0..N-1 if necessary
                        keys_int = sorted(int(k) for k in numeric_items.keys())
                        if keys_int == list(range(1, n+1)):                 # [1,2,...,n]
                            for k, v in numeric_items.items():
                                i = int(k) - 1
                                if 0 <= i < n:
                                    per_index[str(i)] = v
                        else:
                            for k, v in numeric_items.items():
                                i = int(k)
                                if 0 <= i < n:
                                    per_index[str(i)] = v
                    else:
                        # Wrapped in a container key
                        for key in ("results", "items", "output"):
                            if key in data:
                                val = data[key]
                                if isinstance(val, list):
                                    for i, item in enumerate(val[:n]):
                                        per_index[str(i)] = item
                                elif isinstance(val, dict):
                                    sub_numeric = {str(k): v for k, v in val.items() if _NUM_KEY_RE.match(str(k))}
                                    if sub_numeric:
                                        keys_int = sorted(int(k) for k in sub_numeric.keys())
                                        if keys_int == list(range(1, n+1)): # <<< CHANGED: same auto-shift in nested dict
                                            for k, v in sub_numeric.items():
                                                i = int(k) - 1
                                                if 0 <= i < n:
                                                    per_index[str(i)] = v
                                        else:
                                            for k, v in sub_numeric.items():
                                                i = int(k)
                                                if 0 <= i < n:
                                                    per_index[str(i)] = v
                                break
                else:
                    raise ValueError("Unsupported JSON top-level type")

                # Fill any missing indices with explicit failure entries
                failed, reasons = [], []
                for i in range(n):
                    s = str(i)
                    if s not in per_index:
                        per_index[s] = {"good_expansion": False,
                                        "explanation": "Missing index in model response."}

                # Normalize and score
                for i in range(n):
                    s = str(i)
                    detail = _normalize_item(per_index[s])
                    if detail["good_expansion"] and not all(detail[f"rule {r}"] for r in range(1,7)):
                        detail["good_expansion"] = False
                        if not detail["explanation"]:
                            detail["explanation"] = "Inconsistent rules vs verdict; downgraded."
                    if not detail["good_expansion"]:
                        failed.append(s)
                        reasons.append(detail.get("explanation","") or "Failed one or more rules.")

                all_passed = (len(failed) == 0)
                return all_passed, failed, reasons

            except (json.JSONDecodeError, ValueError) as err:
                last_err = err
                self._d(f"  JSON parse/shape error: {err}")
                time.sleep(0.4 * attempt)
                continue

        # ---- All attempts failed to produce usable JSON → BYPASS or FAIL ---
        if self.bypass_on_failure:
            ver = ctx.setdefault("verifier", {}).setdefault(self.ctx_flag_key, {})
            ver.update({
                "bypassed": True,
                "reason": f"LLM verifier failed after {self.max_attempts} attempts",
                "error": str(last_err) if last_err else "unknown",
                "original_count": n,
            })
            self._d("[Verifier] BYPASS engaged — proceeding as pass.")
            return True, [], []

        return False, [f"{i:02d}" for i in range(n)], ["LLM verifier failed"]
