#!/usr/bin/env python3
"""
requirement_rudimentary_extractor.py
────────────────────────────────────────────────────────────────────────────
Extract eligibility-criteria constraint_clauses, run verification, and dump both
the extraction and verification outputs as JSON logs.

Adds:
- Extraction retries with configurable attempts, exponential backoff, and jitter.
"""

from __future__ import annotations

import json, pathlib, re, time, random
from typing import Dict, List, Optional, Any

import dspy
from smt_core.parse_functions import parse_extract_requirements_output
from .RequirementRudimentaryExtractorVerifier import RequirementRudimentaryExtractorVerifier

# ────────────────────────────────────────────────────────────────
_HEADER_PATTERNS = (
    re.compile(r'^\s*inclusion criteria', re.I),
    re.compile(r'^\s*exclusion criteria', re.I),
    re.compile(r'^\s*criteria[:\s]',      re.I),
)

def _coerce_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    """Best-effort coercion of various JSON-y truthy/falsey to bool, else default."""
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return default

def _mk_round_tag(i: int) -> str:
    return f"round_{i:02d}"


class RequirementRudimentaryExtractor(dspy.Module):
    """Simple requirement extractor + verifier with raw JSON output logging.

    New kwargs:
      - extract_max_attempts: int = 3
      - extract_backoff_s: float = 0.5
      - extract_jitter_s: float = 0.25
      - verifier_max_attempts: int = 3  (passed to the verifier)
      - debug: bool = True
    """

    def __init__(
        self,
        engine,
        *,
        log_dir: str | pathlib.Path | None = "extract_logs",
        extract_max_attempts: int = 3,
        extract_backoff_s: float = 0.5,
        extract_jitter_s: float = 0.25,
        verifier_max_attempts: int = 3,
        overall_rounds: int = 3,
        revise_prompt_with_feedback: bool = True,
        debug: bool = True,
    ):
        super().__init__()
        self.engine = engine
        self.log_dir = pathlib.Path(log_dir).expanduser() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # retry knobs
        self.extract_max_attempts = max(1, int(extract_max_attempts))
        self.extract_backoff_s = max(0.0, float(extract_backoff_s))
        self.extract_jitter_s = max(0.0, float(extract_jitter_s))

        # outer (verifier-gated) rounds
        self.overall_rounds = max(1, int(overall_rounds))
        self.revise_prompt_with_feedback = bool(revise_prompt_with_feedback)

        self.debug = debug
        self.verifier = RequirementRudimentaryExtractorVerifier(
            engine, max_attempts=verifier_max_attempts, debug=debug
        )

    def _d(self, *msg):
        if self.debug:
            print(*msg, flush=True)

    @staticmethod
    def _strip_headers(txt: str) -> str:
        lines = txt.splitlines()
        if lines and any(pat.match(lines[0]) for pat in _HEADER_PATTERNS):
            lines = lines[1:]
        return "\n".join(lines)

    def _retry_sleep(self, attempt_idx: int) -> None:
        """Exponential backoff with ± jitter."""
        if self.extract_backoff_s <= 0:
            return
        base = self.extract_backoff_s * (2 ** (attempt_idx - 1))
        jitter = random.uniform(-self.extract_jitter_s, self.extract_jitter_s) if self.extract_jitter_s > 0 else 0.0
        time.sleep(max(0.0, base + jitter))

    # ---- extraction (inner loop) -------------------------------------------
    def _run_single_extraction(self, prompt: str) -> tuple[List[dict], str, Optional[Exception], int]:
        """
        Returns: (reqs, raw_text, last_exc, attempts_used)
        """
        raw_extract = ""
        reqs: List[dict] = []
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.extract_max_attempts + 1):
            self._d(f"[Extractor] JSON/parse attempt {attempt}/{self.extract_max_attempts}")
            raw_extract = self.engine(prompt)[0]
            try:
                reqs = parse_extract_requirements_output(raw_extract) or []
                self._d(f"[Extractor] parsed {len(reqs)} requirements.")
                return reqs, raw_extract, None, attempt
            except Exception as e:
                last_exc = e
                self._d(f"[Extractor] parse failed on attempt {attempt}: {e}")
                if attempt < self.extract_max_attempts:
                    self._retry_sleep(attempt)
                else:
                    self._d("[Extractor] giving up after max attempts; returning empty extraction.")
        return reqs, raw_extract, last_exc, self.extract_max_attempts

    # ---- feedback → prompt revision ---------------------------------------
    def _revise_prompt(self, base_prompt: str, verification: dict) -> str:
        """
        Append a compact, model-friendly instruction block using the verifier's critique.
        Safe to call even if verification lacks fields.
        """
        if not self.revise_prompt_with_feedback:
            return base_prompt

        problems: list[str] = []
        def flag(name: str, label: str):
            val = _coerce_bool(verification.get(name), None)
            if val is False:
                problems.append(label)

        flag("all_covered", "Some original bullets/constraint_clauses were missed or partially covered.")
        flag("oring_list_gives_same_meaning", "Logical equivalence to the original section was broken.")
        flag("starts_with_semantic_template", "Outputs did not follow the required semantic template.")
        flag("no_duplicate_information_representation", "Duplicate/overlapping requirements were produced.")

        expl = str(verification.get("explanation", "") or "").strip()
        critique = "\n".join(f"- {p}" for p in problems) if problems else ""
        feedback = "\n".join(x for x in [critique, (f"- Notes: {expl}" if expl else "")] if x)

        if not feedback:
            # nothing explicit to fix; keep original prompt
            return base_prompt

        repair_block = (
            "\n\n"
            "### VERIFIER FEEDBACK (system-added)\n"
            "The previous draft failed verification. Carefully fix the issues below **without** omitting any\n"
            "information from the original criteria. Keep the output format identical.\n"
            f"{feedback}\n"
            "Re-output the corrected requirements only.\n"
        )
        return base_prompt + repair_block

    def forward(self, context: Dict, *, use_full_context: bool) -> Dict:
        """Run extraction (with retries) and verifier, then save logs."""
        inc_exc = context["inc_exc"]
        ctx_txt = context["contextual_text"]
        req_txt = context["requirement_text"]

        # Build extraction prompt
        if inc_exc == "inclusion":
            prompt_t = context["RequirementRudimentaryExtractorInclusion_prompt"]
        else:
            prompt_t = context["RequirementRudimentaryExtractorExclusion_prompt"]

        base_prompt = (
            prompt_t.replace("#CONTEXTUAL_TEXT#", ctx_txt)
                    .replace("#REQUIREMENT_TEXT#", req_txt)
        )

        # Prepare log base dir (per trial/side like decomposer)
        trial_id = str(context.get("trial_id", "unknown"))
        side = inc_exc
        round_dir: Optional[pathlib.Path] = None
        if self.log_dir:
            round_dir = self.log_dir / f"{trial_id}_{side}"
            round_dir.mkdir(parents=True, exist_ok=True)

        history: List[Dict[str, Any]] = []
        final_reqs: List[dict] = []
        final_verif: Dict[str, Any] = {}

        current_prompt = base_prompt

        for r in range(1, self.overall_rounds + 1):
            tag = _mk_round_tag(r)
            self._d(f"===== extractor round {r}/{self.overall_rounds} =====")

            # 1) save prompt for this round
            if round_dir is not None:
                (round_dir / f"{tag}__extraction_prompt.txt").write_text(current_prompt, encoding="utf-8")

            # 2) inner extraction attempts
            reqs, raw, parse_err, used_attempts = self._run_single_extraction(current_prompt)

            # 3) log raw + parsed
            if round_dir is not None:
                (round_dir / f"{tag}__extraction_raw.txt").write_text(str(raw), encoding="utf-8")
                payload = {
                    "trial_id": trial_id,
                    "side": side,
                    "requirement_text": req_txt,
                    "requirements": reqs,
                    "raw_extraction_output": raw,
                    "parse_attempts_used": used_attempts,
                }
                if parse_err:
                    payload["parse_error"] = str(parse_err)
                (round_dir / f"{tag}__extraction_parsed.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            # 4) verification (single logical pass; the verifier itself may retried internally)
            verif = self.verifier(context, req_txt, reqs)

            if round_dir is not None:
                (round_dir / f"{tag}__verification.json").write_text(
                    json.dumps(verif, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            history.append({"round": r, "requirements": reqs, "verification": verif})

            good = _coerce_bool(verif.get("overall_good_extraction"), None)
            if good is True:
                self._d("  ✓ verifier approved – done")
                final_reqs, final_verif = reqs, verif
                break

            if good is False and r < self.overall_rounds:
                self._d("  ✗ verifier flagged issues – revising prompt and retrying…")
                current_prompt = self._revise_prompt(base_prompt, verif)
                continue

            # If verifier didn't give a clear boolean, or we've hit the limit, accept last
            self._d("  ⚠ stopping: no explicit FALSE from verifier or reached round limit.")
            final_reqs, final_verif = reqs, verif
            break
        else:
            # fell through all rounds without break – take last
            last = history[-1]
            final_reqs, final_verif = last["requirements"], last["verification"]

        # 5) top-level summary logs (backward compatible)
        if self.log_dir:
            extraction_fp = self.log_dir / f"{trial_id}_{side}.extraction.json"
            verification_fp = self.log_dir / f"{trial_id}_{side}.verification.json"
            extraction_fp.write_text(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "side": side,
                        "requirement_text": req_txt,
                        "requirements": final_reqs,
                        "history_rounds": len(history),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            verification_fp.write_text(
                json.dumps(final_verif, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[Extractor] extraction log saved → {extraction_fp}")
            print(f"[Extractor] verification log saved → {verification_fp}")

        # 6) expose results in context
        context["requirements"] = final_reqs
        context["verification"] = final_verif
        context["extraction_history"] = history
        return context
