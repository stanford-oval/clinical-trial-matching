from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import dspy

def _unwrap_code_fence(txt: str) -> str:
    """Strip ```json … ``` fencing if present."""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S | re.I)
    return m.group(1).strip() if m else txt



class RequirementDecomposerVerifier(dspy.Module):
    """
    Audits each (original-text, components) pair produced by RequirementDecomposer.
    Context keys required:
        * inc_exc  – "inclusion" | "exclusion"
        * verification_pairs – list[dict] with keys requirement_index, text, components
    """
    def __init__(self, engine, max_attempts: int = 3, debug: bool = False):
        super().__init__()
        self.engine, self.max_attempts, self.debug = engine, max_attempts, debug

    def _prompt(self, context, list_type, payload_json):
        tmpl = context["RequirementDecomposerVerifierInclusion_prompt"] if list_type == "inclusion" else context["RequirementDecomposerVerifierExclusion_prompt"]
        return tmpl.replace("#ORIGINAL_DECOMPOSED_PAIR#", payload_json)


    def forward(self, context: Dict, *, use_full_context: bool = True) -> Dict:
        pairs: List[Dict[str, Any]] = context["verification_pairs"]
        payload_json = json.dumps(pairs, ensure_ascii=False, indent=2)

        # === NEW: determine current round (default 1) ===
        round_no: int = int(context.get("verification_round", 1))

        prompt = self._prompt(context, context["inc_exc"].lower(), payload_json)
        raw = self.engine(prompt)[0]  # single-shot call

        try:
            verdicts = json.loads(_unwrap_code_fence(raw))
        except Exception as e:
            if self.debug:
                print("[Verifier] JSON parse error:", e)
            verdicts = []

        # === NEW: ensure each verdict carries a round number and a minimal snapshot ===
        if isinstance(verdicts, list):
            # map index -> input pair for quick lookup
            idx2pair = {p.get("requirement_index"): p for p in pairs}
            for v in verdicts:
                # Stamp round number if missing
                if isinstance(v, dict) and "round_number" not in v:
                    v["round_number"] = round_no  # === NEW ===
                # Ensure a compact echo exists
                ridx = isinstance(v, dict) and v.get("requirement_index")
                if isinstance(v, dict) and "round_snapshot" not in v:
                    inp = idx2pair.get(ridx) if ridx in idx2pair else None
                    if inp:
                        v["round_snapshot"] = {            # === NEW ===
                            "input_pair": {
                                "requirement_index": inp.get("requirement_index"),
                                "text": inp.get("text"),
                                "components": inp.get("components", []),
                            },
                            "notes": "auto-added by verifier for sanity check"
                        }

        # === existing next-round assembly (kept), with small additions to help caller ===
        next_round: List[Dict[str, Any]] = []
        if verdicts and isinstance(verdicts, list):
            index_to_pair = {p.get("requirement_index"): p for p in pairs}
            for v in verdicts:
                try:
                    ridx = v.get("requirement_index")
                    if not ridx:
                        continue
                    original = index_to_pair.get(ridx)
                    if not original:
                        continue

                    overall_ok = (v.get("overall_good_decomposition") == "YES")
                    pass_next = (v.get("pass_to_next_round") == "YES")
                    rewrite_pair = v.get("next_round_rewrite_pair") or {}

                    if (not overall_ok) and pass_next and rewrite_pair and isinstance(rewrite_pair, dict):
                        orig_tagged = dict(original)
                        orig_tagged["variant"] = "original"
                        rewrite_tagged = {
                            "requirement_index": rewrite_pair.get("requirement_index", original.get("requirement_index")),
                            "text": rewrite_pair.get("text", original.get("text")),
                            "components": rewrite_pair.get("components", []),
                            "variant": "rewrite"
                        }
                        if rewrite_tagged["components"]:
                            next_round.extend([orig_tagged, rewrite_tagged])
                except Exception as e:
                    if self.debug:
                        print("[Verifier] next-round assembly error:", e)

        # === CHANGED: always attach parsed verdicts; add round snapshots/history ===
        context["verification_results"] = verdicts

        # === NEW: per-round compact snapshot for sanity checks
        round_snapshot = {
            "round": round_no,
            "inc_exc": context.get("inc_exc"),
            "input_pairs": pairs,
            "raw_model_text": raw,             # useful when JSON parsing needed inspection
        }
        context["verification_round_snapshot"] = round_snapshot  # === NEW ===

        # === NEW: append to history
        hist: List[Dict[str, Any]] = context.get("verification_history", [])
        hist.append(round_snapshot)
        context["verification_history"] = hist

        if next_round:
            context["verification_pairs_next_round"] = next_round
            context["next_verification_round"] = round_no + 1  # === NEW === convenience
        else:
            context["verification_pairs_next_round"] = []
            context["next_verification_round"] = round_no      # stays the same

        return context