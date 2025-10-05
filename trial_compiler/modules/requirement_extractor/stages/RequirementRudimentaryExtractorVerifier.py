# bullet_coverage_verifier.py
import json, textwrap, re
import dspy

def _unwrap(txt: str) -> str:
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S)
    return m.group(1).strip() if m else txt

class RequirementRudimentaryExtractorVerifier(dspy.Module):
    """
    Runs coverage & logical equivalence check for inclusion/exclusion criteria
    and returns the **raw parsed JSON** from the LLM.
    """

    def __init__(self, engine, max_attempts: int = 3, debug: bool = False):
        super().__init__()
        self.engine, self.max_attempts, self.debug = engine, max_attempts, debug

    def _d(self, *msg):
        if self.debug:
            print(*msg, flush=False)

    def forward(
        self,
        ctx: dict,
        criteria_txt: str,
        extracted_reqs: list[dict],
    ) -> dict:
        # Build prompt
        if ctx["inc_exc"] == "inclusion":
            prompt_tmpl = ctx.get("RequirementRudimentaryExtractorVerifierInclusion_prompt")
        else:
            prompt_tmpl = ctx.get("RequirementRudimentaryExtractorVerifierExclusion_prompt")

        if not prompt_tmpl:
            raise KeyError("Missing verifier prompt in ctx")

        prompt = (
            prompt_tmpl
            .replace("{{CRITERIA}}", criteria_txt)
            .replace(
                "{{EXTRACTED}}",
                textwrap.indent(
                    json.dumps(extracted_reqs, indent=2, ensure_ascii=False),
                    "  "
                )
            )
        )

        last_error = None
        for k in range(1, self.max_attempts + 1):
            raw = self.engine(prompt)[0]
            self._d(f"[coverage-check attempt {k}] raw →", raw[:200], "…")
            try:
                return json.loads(_unwrap(raw))  # return raw parsed JSON directly
            except Exception as e:
                last_error = e
                self._d(f"  JSON parse failed: {e}")

        return {"error": f"LLM verifier failed: {last_error}", "raw": raw}
