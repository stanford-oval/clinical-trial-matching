from __future__ import annotations

"""requirement_decomposer.py  —  *rev-4*

Adds support for the new verifier schema that returns
``overall_good_decomposition`` instead of ``verdict``, improves robustness
of JSON parsing, normalizes LLM output handling, and enhances debug output.

This revision also adds a *post-pass reconciliation* policy:
- If the final round marks a requirement as bad (overall_good_decomposition != "YES"
  or legacy verdict not "correct"), we search backwards in the verification history
  for the most recent round where that requirement was marked good and restore
  the components from that round.
- If no round ever marked it good, we fall back to the original input requirement
  as a single-component list.

Changes
~~~~~~~
* Accepts either field name when judging success:
  ``verdict`` **or** ``overall_good_decomposition`` (explicit YES).
* Treats any present ``overall_good_decomposition`` value other than YES as failure.
* Makes the decomposed JSON parser tolerant of extra keys (only requires
  'requirement_index' and 'components').
* Normalizes engine return shape via a small helper (no more [0] vs [0][0]).
* Clearer debug printing with ✓/✗ markers.
* **New**: Post-pass reconciliation to backfill "bad" finals with last-known-good,
  otherwise original requirement.
"""

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


import dspy


# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────

def _unwrap_code_fence(txt: str) -> str:
    """Strip a ```json fenced block``` if present; otherwise return as-is."""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S | re.I)
    return m.group(1).strip() if m else txt


def _parse_decomp_json(raw: str, expect_n: int) -> Tuple[Optional[List[Dict]], str]:
    """Parse the decomposer JSON output.

    Requirements:
      - top-level must be a list of length `expect_n`
      - each entry must contain keys: requirement_index, components
      - components must be list[str]
    Extra keys are tolerated.
    """
    raw = _unwrap_code_fence(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON error: {e}"

    if not isinstance(data, list) or len(data) != expect_n:
        return None, "top-level list length mismatch"

    for ent in data:
        if not isinstance(ent, dict):
            return None, "each entry must be an object"
        mandatory = {"requirement_index", "components"}
        if not mandatory.issubset(ent):
            return None, "missing required key(s): 'requirement_index', 'components'"
        if not isinstance(ent["components"], list) or not all(
            isinstance(c, str) for c in ent["components"]
        ):
            return None, "'components' must be list[str]"
    return data, ""


# ────────────────────────────────────────────────────────────────
# module
# ────────────────────────────────────────────────────────────────
class RequirementDecomposer(dspy.Module):
    """Decompose + self-verify clinical-trial criteria in one shot."""

    def __init__(
        self,
        engine,
        *,
        decomp_attempts: int = 3,
        overall_rounds: int = 5,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.decomp_attempts = decomp_attempts
        self.overall_rounds = overall_rounds
        self.debug = debug

    # ------------------------------------------------------------
    def _d(self, *msg: object) -> None:
        if self.debug:
            print(*msg, flush=False)

    # ------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        """Normalize different engine return shapes to a single string."""
        out = self.engine(prompt)
        try:
            first = out[0]
            if isinstance(first, (list, tuple)):
                return str(first[0])
            return str(first)
        except Exception:
            # Best-effort fallback
            return str(out)

    # ------------------------------------------------------------
    @staticmethod
    def _build_verification_pairs(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "requirement_index": idx + 1,
                "text": req["requirement"],
                "components": req["components"],
            }
            for idx, req in enumerate(reqs)
        ]

    # ------------------------------------------------------------
    def _is_bad(self, verdict_entry: Dict[str, Any]) -> bool:
        """Return True if the entry marks the decomposition as bad/unclear.

        New schema: overall_good_decomposition == "YES"  → good
                    anything else (when present)         → bad
        Legacy:     verdict in {"incorrect","unclear","no"} → bad
        """
        # Prefer new schema if present
        g = verdict_entry.get("overall_good_decomposition", None)
        if g is not None:
            return str(g).strip().upper() != "YES"

        # Fallback to legacy schema
        v = verdict_entry.get("verdict", None)
        if v is None:
            # If neither field exists, don't block progress here.
            return False
        return str(v).strip().lower() in {"incorrect", "unclear", "no"}

    # ------------------------------------------------------------
    def _is_good(self, verdict_entry: Dict[str, Any]) -> bool:
        """Return True if the entry explicitly marks the decomposition as good."""
        g = verdict_entry.get("overall_good_decomposition", None)
        if g is not None:
            return str(g).strip().upper() == "YES"
        v = verdict_entry.get("verdict", None)
        return v is not None and str(v).strip().lower() == "correct"

    # ------------------------------------------------------------
    def forward(self, context: Dict, *, use_full_context: bool = True) -> Dict:  # noqa: D401
        trial_id  = str(context.get("trial_id", "unknown"))
        list_type = str(context.get("inc_exc", "inclusion")).lower()
        if list_type not in {"inclusion", "exclusion"}:
            raise ValueError("context['inc_exc'] must be 'inclusion' or 'exclusion'")
        
        base_dir  = Path("mbench/req_mbench/decomposition_maps") / f"{trial_id}_{list_type}"
        base_dir.mkdir(parents=True, exist_ok=True)
        def _w(name: str, content: str) -> None:
            try:
                (base_dir / name).write_text(content, encoding="utf-8")
            except Exception:
                pass

        prompt_key = (
            "RequirementDecomposerInclusion_prompt"
            if list_type == "inclusion"
            else "RequirementDecomposerExclusion_prompt"
        )
        prompt_tmpl = context.get(prompt_key)
        if not prompt_tmpl or "##REQUIREMENTS_JSON##" not in prompt_tmpl:
            raise ValueError("missing or malformed decomposer prompt template")

        verifier_key = (
            "RequirementDecomposerVerifierInclusion_prompt"
            if list_type == "inclusion"
            else "RequirementDecomposerVerifierExclusion_prompt"
        )
        verifier_tmpl = context.get(verifier_key)
        if not verifier_tmpl or "#ORIGINAL_DECOMPOSED_PAIR#" not in verifier_tmpl:
            raise ValueError("missing or malformed verifier prompt template")

        # ---------- normalise requirements -----------------------
        norm_reqs: List[Dict[str, Any]] = []
        texts: List[str] = []
        for item in context["requirements"]:
            if isinstance(item, str):
                norm_reqs.append({"requirement": item})
                texts.append(item)
            else:
                norm_reqs.append(dict(item))
                texts.append(item.get("requirement", str(item)))
        const_texts = list(texts)

        # ---------- outer loop ----------------------------------
        history: List[Dict[str, Any]] = []  # keeps verdicts + per-round components

        for r in range(1, self.overall_rounds + 1):
            self._d(f"===== round {r}/{self.overall_rounds} =====")

            # 1) build decomposer prompt
            req_json = json.dumps(
                [{"requirement_index": i + 1, "text": t} for i, t in enumerate(const_texts)],
                ensure_ascii=False,
                indent=2,
            )
            decomp_prompt = prompt_tmpl.replace(
                "##REQUIREMENTS_JSON##", textwrap.indent(req_json, "  ")
            )
            
            round_tag = f"round_{r:02d}"

            # save decomposer prompt
            _w(f"{round_tag}__decomposer_prompt.txt", decomp_prompt)

            # 2) inner retries for well-formed JSON
            parsed: Optional[List[Dict[str, Any]]] = None
            for a in range(1, self.decomp_attempts + 1):
                raw = self._call_llm(decomp_prompt)
                # raw output per attempt
                _w(f"{round_tag}__decomposer_attempt_{a:02d}_raw.txt", str(raw))

                parsed, err = _parse_decomp_json(raw, len(const_texts))
                if parsed:
                    _w(f"{round_tag}__decomposer_attempt_{a:02d}_parsed.json",
                       json.dumps(parsed, ensure_ascii=False, indent=2))
                    break
                else:
                    _w(f"{round_tag}__decomposer_attempt_{a:02d}_parse_error.txt", err)

                self._d(f"  ⤷ decomp attempt {a} failed → {err}")
            if not parsed:
                parsed = [
                    {"requirement_index": i + 1, "components": [t]} for i, t in enumerate(const_texts)
                ]
                _w(f"{round_tag}__decomposer_fallback_parsed.json",
                   json.dumps(parsed, ensure_ascii=False, indent=2))
                
            # 3) merge components
            for base, ent in zip(norm_reqs, parsed):
                base["components"] = ent["components"]

            # Snapshot per-round components so we can restore later if needed
            round_components = [list(req["components"]) for req in norm_reqs]

            # 4) verifier
            pairs = self._build_verification_pairs(norm_reqs)
            verifier_prompt = verifier_tmpl.replace(
                "#ORIGINAL_DECOMPOSED_PAIR#", json.dumps(pairs, ensure_ascii=False, indent=2)
            )
            _w(f"{round_tag}__verifier_prompt.txt", verifier_prompt)

            raw_v = self._call_llm(verifier_prompt)
            
            _w(f"{round_tag}__verifier_raw.txt", str(raw_v))
            try:
                verdicts = json.loads(_unwrap_code_fence(raw_v))
            except Exception as exc:
                self._d(f"  ⤷ verifier JSON error → {exc}; accepting result")
                verdicts = []

            context["verification_results"] = verdicts
            history.append({"round": r, "verdicts": verdicts, "components": round_components})

            # >>> debug print of outcome
            if self.debug:
                self._d("  — verifier outcome —")
                for v in verdicts:
                    idx = v.get("requirement_index")
                    good = v.get("overall_good_decomposition", v.get("verdict"))
                    note = v.get("issues", v.get("explanation", ""))
                    is_ok = (
                        str(good).strip().upper() == "YES"
                        or str(good).strip().lower() == "correct"
                    )
                    status = "✓" if is_ok else "✗"
                    self._d(f"    [{status}] #{idx}: {good} — {note}")

            # 5) decide if we need another round
            flawed = [v for v in verdicts if self._is_bad(v)]
            if not flawed:
                self._d("  ✓ all pairs verified – done")
                break
            self._d(f"  ✗ {len(flawed)} flawed pair(s) – retrying…")
        else:
            self._d("⚠ reached overall retry limit – returning last decomposition")

        # ---------- post-pass reconciliation ---------------------
        # Apply the policy:
        # (1) If final round marks req as bad, search history backwards for last-known-good.
        # (2) If none, fallback to the original input requirement as [original_text].
        final_verdicts: List[Dict[str, Any]] = history[-1]["verdicts"] if history else []
        verdicts_by_idx = {v.get("requirement_index"): v for v in final_verdicts}

        for i, req in enumerate(norm_reqs, start=1):
            v = verdicts_by_idx.get(i)
            if v and self._is_bad(v):
                restored: Optional[List[str]] = None
                for h in reversed(history):
                    hv = next((x for x in h["verdicts"] if x.get("requirement_index") == i), None)
                    if hv and self._is_good(hv):
                        # restore components from that round's snapshot
                        restored = list(h["components"][i - 1])
                        break
                if restored is None:
                    restored = [const_texts[i - 1]]
                if self.debug:
                    self._d(
                        f"  ↻ restoring components for #{i}: "
                        f"{'last-known-good' if restored != [const_texts[i - 1]] else 'original requirement'}"
                    )
                req["components"] = restored

        # ---------- finalize context -----------------------------
        context["requirements"] = norm_reqs
        context["verification_history"] = history
        context["decomposition_mapping"] = {
            req["requirement"]: list(req["components"]) for req in norm_reqs
        }
        return context
