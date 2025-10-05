# requirement_extractor.py
# ===============================================================
# End-to-end preprocessing pipeline with an additional
# RequirementExplicitPositivityClassifier for inclusion criteria.
# ===============================================================

from __future__ import annotations

import json
import re
import textwrap
from itertools import islice
from typing import Any, Dict, List, Optional, Set

import dspy

# ────────────────────────────────────────────────────────────────
# New module: RequirementExplicitPositivityClassifier
# ────────────────────────────────────────────────────────────────

# helpers (self-contained)
def _unwrap_code_fence(txt: str) -> str:
    """Strip ```json … ``` fencing if present."""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S | re.I)
    return m.group(1).strip() if m else txt

def _chunked(iterable, size):
    """Yield successive <size>-element lists from *iterable*."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

# label set for positivity
_VALID_POS: Set[str] = {"explicit", "optional", "irrelevant"}

def _parse_pos_label_json(raw: str, expect_n: int) -> tuple[Optional[List], str]:
    """
    Return (parsed_json, err_msg). err_msg is '' on success.
    Expected schema:
      [
        {"requirement_index": <int>, "components": [{"text": "...", "label": "explicit|optional|irrelevant", ...}, ...]},
        ...
      ]  length == expect_n
    """
    raw = _unwrap_code_fence(raw)
    try:
        dat = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON error: {e}"

    if not isinstance(dat, list) or len(dat) != expect_n:
        return None, "top-level list length mismatch"

    for entry in dat:
        if "requirement_index" not in entry:
            return None, "missing requirement_index"
        comps = entry.get("components")
        if not isinstance(comps, list) or not comps:
            return None, "'components' must be a non-empty list"
        for c in comps:
            lab = c.get("label")
            if lab not in _VALID_POS:
                return None, f"unknown label {lab!r} (valid: {_VALID_POS})"
    return dat, ""

_DEFAULT_EXPLICIT_POSITIVITY_PROMPT = """\
You are labeling inclusion requirement *components* for information retrieval value.

Task:
- For each component, assign a single label:
  - "explicit": A known positive fact is required to assert meaningful relevance.
               Example: specific diagnosis, histology, biomarker presence, stage, ECOG range, measurable disease, age range, etc.
  - "optional": Helpful if present but not strictly needed for initial retrieval; absence/unknown should not block recall.
  - "irrelevant": Wording or administrative fragments with negligible retrieval value (e.g., boilerplate, consent verbiage without substance).

Guidelines:
- Prefer "explicit" for core phenotyping anchors and inclusion gates that strongly determine trial-topic relevance.
- Prefer "optional" for ancillary items (e.g., willingness-to-comply) unless clearly gating.
- Use "irrelevant" sparingly, only for content that should not influence retrieval at all.

Return ONLY JSON in the form:
[
  {
    "requirement_index": <int>,
    "components": [
      { "text": "...", "label": "explicit" | "optional" | "irrelevant" },
      ...
    ]
  },
  ...
]

Components to label:
##COMPONENTS_JSON##
"""

class RequirementExplicitPositivityClassifier(dspy.Module):
    """
    Classify inclusion components as explicit / optional / irrelevant (batched).
    - Runs ONLY when inc_exc == 'inclusion'.
    - Mirrors the structure of RequirementHardSoftClassifier.
    """

    def __init__(self, engine, *, max_attempts: int = 3, debug: bool = False):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts
        self.debug = debug

    # small debug helper
    def _d(self, *msg):
        if self.debug:
            print(*msg, flush=False)

    def forward(self, context: Dict, *, use_full_context: bool = True) -> Dict:  # noqa: D401
        list_type: str = str(context.get("inc_exc", "inclusion")).lower()
        if list_type != "inclusion":
            # This classifier is no-op for exclusion lists
            return context

        if "requirements" not in context:
            raise ValueError("context['requirements'] missing (did decomposer run?)")
        for r in context["requirements"]:
            if not r.get("components"):
                raise ValueError("each requirement must have non-empty 'components' list")

        # Fetch prompt template (with safe default)
        prompt_tmpl: str = context.get("RequirementExplicitPositivity_prompt") or _DEFAULT_EXPLICIT_POSITIVITY_PROMPT
        if "##COMPONENTS_JSON##" not in prompt_tmpl:
            raise ValueError("positivity prompt must contain '##COMPONENTS_JSON##' placeholder")

        # batching controls
        MAX_REQ_PER_BATCH: int = int(context.get("positivity_batch_size", 10))

        all_parsed: List[Dict[str, Any]] = []
        req_iter = enumerate(context["requirements"], start=1)

        for batch in _chunked(list(req_iter), MAX_REQ_PER_BATCH):
            # Build JSON slice for this batch
            comp_json = json.dumps(
                [
                    {"requirement_index": global_idx, "components": req["components"]}
                    for global_idx, req in batch
                ],
                ensure_ascii=False,
                indent=2,
            )
            prompt = prompt_tmpl.replace("##COMPONENTS_JSON##", textwrap.indent(comp_json, "  "))
            self._d("\n=== Positivity prompt (truncated) ===\n", prompt[:600], "…")

            parsed: Optional[List] = None
            err_msg = ""
            for attempt in range(1, self.max_attempts + 1):
                raw = self.engine(prompt)[0]
                self._d(f"[attempt {attempt}] raw →", raw[:300], "…")
                parsed, err_msg = _parse_pos_label_json(raw, len(batch))
                if parsed:
                    break
                # repair prompt
                prompt = (
                    "The previous response could not be parsed because of:\n"
                    f"{err_msg}\n\n"
                    "Return ONLY the corrected JSON, without extra text.\n\n"
                    + prompt_tmpl.replace("##COMPONENTS_JSON##", textwrap.indent(comp_json, "  "))
                )

            if not parsed:
                raise RuntimeError(
                    "Explicit-positivity classifier: failed on a batch after "
                    f"{self.max_attempts} attempts. Last error: {err_msg}"
                )

            all_parsed.extend(parsed)

        # Merge labels into context
        merged: List[Dict[str, Any]] = []
        for base, entry in zip(context["requirements"], all_parsed):
            comps_out = []
            for c in entry["components"]:
                out = {k: v for k, v in c.items() if k != "label"}
                out["positivity"] = c["label"]  # rename 'label' → 'positivity'
                comps_out.append(out)

            lbls = sorted({c["positivity"] for c in comps_out})
            new_base = dict(base)
            new_base["components"] = comps_out
            new_base["positivity_labels"] = lbls
            new_base["positivity_constraint"] = ",".join(lbls)
            merged.append(new_base)

        context["requirements"] = merged
        return context
