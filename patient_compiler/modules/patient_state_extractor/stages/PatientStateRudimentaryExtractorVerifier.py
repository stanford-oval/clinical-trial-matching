# patient_fact_coverage_verifier.py
import json, textwrap, re
import dspy
from typing import List, Tuple

def _unwrap(txt: str) -> str:
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S)
    return m.group(1).strip() if m else txt

class PatientStateRudimentaryExtractorVerifier(dspy.Module):
    """
    A thin wrapper around the LLM-based verifier.

    The class feeds the note and extracted facts into an LLM
    according to the provided prompt template.  It then parses the
    JSON response and returns whether the extraction is acceptable.

    If the response cannot be parsed or indicates failure, the method
    returns `False` together with a list of human-readable messages
    that describe the problem (useful for logging).
    """

    def __init__(self, engine, max_attempts: int = 2, debug: bool = False):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts
        self.debug = debug

    # ------------------------------------------------------------------
    def _d(self, *msg) -> None:
        """Lightweight debug print helper."""
        if self.debug:
            print(*msg, flush=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        ctx: dict,
        note_txt: str,
        extracted_facts: List[dict],
    ) -> Tuple[bool, List[str]]:
        """
        Parameters
        ----------
        ctx : dict
            Pipeline context; must include the verifier prompt template
            under one of these keys:
              * "PatientFactRudimentaryExtractorVerifier_prompt"
              * "PatientStateRudimentaryExtractorVerifier_prompt"
        note_txt : str
            The raw patient note.
        extracted_facts : List[dict]
            The JSON list of machine-extracted patient facts.

        Returns
        -------
        Tuple[bool, List[str]]
            (verification_passed, list_of_missing_or_error_messages)
        """
        prompt_tmpl = (
            ctx.get("PatientFactRudimentaryExtractorVerifier_prompt")
            or ctx.get("PatientStateRudimentaryExtractorVerifier_prompt")
        )
        if not prompt_tmpl:
            raise KeyError(
                "Verifier prompt template missing: expected "
                "'PatientFactRudimentaryExtractorVerifier_prompt' "
                "in the context."
            )

        # Fill NOTE / EXTRACTED placeholders inside the template
        prompt = (
            prompt_tmpl.replace("{{NOTE}}", note_txt)
            .replace(
                "{{EXTRACTED}}",
                textwrap.indent(
                    json.dumps(extracted_facts, indent=2, ensure_ascii=False),
                    "  ",
                ),
            )
        )

        # Run up to `max_attempts` in case of transient LLM errors
        for attempt in range(1, self.max_attempts + 1):
            raw = self.engine(prompt)[0]
            self._d(f"[coverage-check attempt {attempt}] raw →", raw[:200], "…")

            # Safely unwrap and parse JSON
            try:
                data = json.loads(_unwrap(raw))
            except (json.JSONDecodeError, TypeError):
                # Parsing failed → retry
                continue

            passed = str(data.get("OVERALL_GOOD_EXTRACTION", "")).upper() == "YES"

            if passed:
                # Successful verification
                return True, []

            # Failed: gather explanation sentences for logging
            explanation = data.get("explanation", "")
            issues = [
                s.strip()
                for s in re.split(r"[.;。]\s*|\n+", explanation)
                if s.strip()
            ] or ["Verifier reported extraction problems"]
            return False, issues

        # All attempts failed (unparsable responses)
        return False, ["LLM verifier failed to return valid JSON"]