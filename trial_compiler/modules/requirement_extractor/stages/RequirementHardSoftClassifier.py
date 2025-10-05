from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

import dspy

# ────────────────────────────────────────────────────────────────
# helpers & constants
# ────────────────────────────────────────────────────────────────

def _unwrap_code_fence(txt: str) -> str:
    """Strip ```json … ``` fencing if present."""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S | re.I)
    return m.group(1).strip() if m else txt

# === Label sets ===
# Inclusion (your new scheme)
_VALID_INCL: Set[str] = {
    "PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE",
    "NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION",
    "OTHER_REQUIREMENTS",
}
# Only CAN_ALWAYS… should require "correction"
_CORR_INCL: Set[str] = {"NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION"}

# Exclusion (unchanged from your prior code)
_VALID_EXCL: Set[str] = {
    "NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION",
    "OTHER_REQUIREMENTS",
}
_CORR_EXCL: Set[str] = _VALID_EXCL - {"OTHER_REQUIREMENTS"}

# Defaults used only if the LLM fails after retries.
# NOTE: 'POSSBILE...' spelling matches your _VALID_EXCL set.
_DEFAULT_INCL_LABEL = "OTHER_REQUIREMENTS"
_DEFAULT_EXCL_LABEL = "OTHER_REQUIREMENTS"

def _parse_label_json_lenient(
    raw: str,
    valid_labels: Set[str],
    correctable_labels: Set[str],
) -> tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Lenient parser:
      - Accepts list of any length (>=1).
      - Validates labels & 'correction' when required.
      - Allows dict → wrap to list.
    Returns (parsed_list, err_msg). err_msg == '' on success.
    """
    raw = _unwrap_code_fence(raw)
    try:
        dat = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON error: {e}"

    if isinstance(dat, dict):
        dat = [dat]  # wrap single object

    if not isinstance(dat, list) or not dat:
        return None, "top-level must be a non-empty list"

    for entry in dat:
        if not isinstance(entry, dict):
            return None, "each entry must be an object"
        if "requirement_index" not in entry:
            return None, "missing 'requirement_index' in an entry"
        comps = entry.get("components")
        if not isinstance(comps, list) or not comps:
            return None, "'components' must be a non-empty list"
        for c in comps:
            lab = c.get("label")
            if lab not in valid_labels:
                return None, f"unknown label {lab!r}"
            if lab in correctable_labels and not c.get("correction"):
                return None, f"'correction' required for label {lab}"
    return dat, ""

def _build_prompt_with_context(prompt_tmpl: str, comp_json: str, trial_ctx: str) -> str:
    return (
        prompt_tmpl
        .replace("##COMPONENTS_JSON##", textwrap.indent(comp_json, "  "))
        .replace("##TRIAL_CONTEXT##", trial_ctx)
    )

def _validate_batch_coverage(
    parsed: List[Dict[str, Any]],
    batch_indices: List[int],
    reqs: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Ensure:
      - Every batch index appears exactly once.
      - Each entry has the same number of components as its source requirement.
    """
    expected = set(batch_indices)
    seen: Set[int] = set()

    for entry in parsed:
        try:
            idx = int(entry.get("requirement_index", -1))
        except Exception:
            return False, "a 'requirement_index' was not an integer"

        if idx not in expected:
            return False, f"unexpected requirement_index {idx}; expected one of {sorted(expected)}"
        if idx in seen:
            return False, f"duplicate requirement_index {idx}"
        seen.add(idx)

        comps = entry.get("components")
        if not isinstance(comps, list):
            return False, f"'components' for requirement_index {idx} must be a list"
        expected_len = len(reqs[idx - 1]["components"])
        if len(comps) != expected_len:
            return False, (
                f"components length mismatch for requirement_index {idx} "
                f"(got {len(comps)}, expected {expected_len})"
            )

    missing = expected - seen
    if missing:
        return False, f"missing requirement_index entries: {sorted(missing)}"

    return True, ""

# ────────────────────────────────────────────────────────────────
# module
# ────────────────────────────────────────────────────────────────

class RequirementHardSoftClassifier(dspy.Module):
    """Classify requirement components (batched) with trial-context awareness.

    Retries up to `max_attempts` (default 3). If the model still fails to produce a
    valid batch, we fall back to a per-mode default:
      • Inclusion → OTHER_REQUIREMENTS
      • Exclusion → OTHER_REQUIREMENTS
    """

    def __init__(self, engine, *, max_attempts: Optional[int] = 3, debug: bool = False):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts  # None => unlimited
        self.debug = debug

    def _d(self, *msg):
        if self.debug:
            print(*msg, flush=False)

    def forward(self, context: Dict, *, use_full_context: bool = True) -> Dict:  # noqa: D401
        # Determine inclusion vs exclusion list
        list_type: str = str(context.get("inc_exc", "inclusion")).lower()
        if list_type not in {"inclusion", "exclusion"}:
            raise ValueError("context['inc_exc'] must be 'inclusion' or 'exclusion'")

        # Fetch prompt template
        prompt_key = (
            "RequirementHardSoftClassifierInclusion_prompt"
            if list_type == "inclusion"
            else "RequirementHardSoftClassifierExclusion_prompt"
        )
        prompt_tmpl: str | None = context.get(prompt_key)
        if not prompt_tmpl:
            raise ValueError(f"missing classifier prompt in context: {prompt_key}")
        if "##COMPONENTS_JSON##" not in prompt_tmpl:
            raise ValueError("classifier prompt must contain '##COMPONENTS_JSON##' placeholder")

        # Trial context (supports either key)
        trial_ctx: str = str(context.get("contextual_text") or context.get("trial_context") or "")

        # Ensure decomposed components present (SOFT-FIX instead of hard-fail)
        if "requirements" not in context:
            raise ValueError("context['requirements'] missing (did decomposer run?)")

        auto_filled_indices: List[int] = []
        for i, r in enumerate(context["requirements"], start=1):
            comps = r.get("components")
            if not isinstance(comps, list) or len(comps) == 0:
                raw_txt = (
                    r.get("text")
                    or r.get("raw")
                    or r.get("full_text")
                    or ""
                )
                r["components"] = [{"text": str(raw_txt).strip() or "(empty)", "_autofilled": True}]
                auto_filled_indices.append(i)
        if auto_filled_indices:
            self._d(
                f"[classifier] Soft-filled empty 'components' for requirement_index "
                f"{auto_filled_indices}. Proceeding with classification."
            )

        # Label sets
        valid = _VALID_INCL if list_type == "inclusion" else _VALID_EXCL
        corr  = _CORR_INCL if list_type == "inclusion" else _CORR_EXCL

        # Choose per-mode default
        default_label = _DEFAULT_INCL_LABEL if list_type == "inclusion" else _DEFAULT_EXCL_LABEL

        # ── Batching (fixed size = 1; pure count-based split) ──────────────────
        max_per_batch: int = 1

        reqs: List[Dict[str, Any]] = context["requirements"]

        # --- persistent logging setup (per trial/list) ---------------
        trial_id  = str(context.get("trial_id", "unknown"))
        list_type_out = str(context.get("inc_exc", "inclusion"))
        base_dir  = Path("mbench/req_mbench/classification_maps") / f"{trial_id}_{list_type_out}"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Build batches of (global_idx, req)
        indices = list(range(1, len(reqs) + 1))
        batches: List[List[int]] = [indices[i:i+max_per_batch] for i in range(0, len(indices), max_per_batch)]

        all_entries: Dict[int, Dict[str, Any]] = {}

        for batch_indices in batches:
            # JSON for this batch
            comp_json = json.dumps(
                [
                    {"requirement_index": i, "components": reqs[i-1]["components"]}
                    for i in batch_indices
                ],
                ensure_ascii=False,
                indent=2,
            )
            base_prompt = _build_prompt_with_context(prompt_tmpl, comp_json, trial_ctx)
            prompt = base_prompt
            self._d("\n=== Classifier prompt (truncated) ===\n", prompt[:900], "…")

            parsed: Optional[List] = None
            err_msg = ""
            attempt = 0

            allowed_labels_str = ", ".join(sorted(valid))

            # one tag per batch, e.g. "batch_0001" or "batch_0001-0002" if multiple indices
            batch_tag = "batch_" + "-".join(f"{i:04d}" for i in batch_indices)

            def _write(name: str, content: str) -> None:
                try:
                    (base_dir / f"{batch_tag}__{name}").write_text(content, encoding="utf-8")
                except Exception:
                    pass

            _write("batch_components_input.json", comp_json)
            _write("prompt_initial.txt", prompt)

            while True:
                attempt += 1
                raw = self.engine(prompt)[0]
                _write(f"attempt_{attempt:02d}_raw.txt", str(raw))

                self._d(f"[attempt {attempt}] raw →", raw[:400], "…")

                parsed, err_msg = _parse_label_json_lenient(raw, valid, corr)
                if parsed:
                    ok, cov_err = _validate_batch_coverage(parsed, batch_indices, reqs)
                    if ok:
                        _write(f"attempt_{attempt:02d}_parsed.json", json.dumps(parsed, ensure_ascii=False, indent=2))
                        break
                    else:
                        _write(f"attempt_{attempt:02d}_coverage_error.txt", cov_err)
                        parsed = None
                        err_msg = cov_err
                else:
                    _write(f"attempt_{attempt:02d}_parse_error.txt", err_msg)

                # Respect an optional hard cap → fallback on last attempt
                if self.max_attempts is not None and attempt >= self.max_attempts:
                    if self.debug:
                        print(
                            f"[classifier] Falling back to default '{default_label}' "
                            f"for batch {batch_indices} after {self.max_attempts} attempts. "
                            f"Last error: {err_msg}"
                        )
                    parsed = []  # trigger default fill below
                    break

                # Prepare repair prompt; keep placeholders filled
                repair_header = (
                    "The previous response could not be accepted because:\n"
                    f"{err_msg}\n\n"
                    "Return ONLY the corrected JSON array with:\n"
                    f"  • Exactly these requirement_index values: {sorted(batch_indices)}\n"
                    "  • For each index, 'components' array length must exactly match the provided input\n"
                    f"  • Labels must be one of: " + allowed_labels_str + "\n"
                    "  • Include 'correction' ONLY when required by the label\n\n"
                )
                prompt = repair_header + base_prompt

            # Merge parsed entries by requirement_index (strict 1:1 when present)
            for entry in parsed:
                idx = int(entry.get("requirement_index", -1))
                all_entries[idx] = entry

            # Fill any missing indices in this batch with per-mode defaults
            for idx in batch_indices:
                if idx not in all_entries:
                    comps = []
                    for c in reqs[idx-1]["components"]:
                        if isinstance(c, dict):
                            comps.append({
                                **{k: v for k, v in c.items() if k != "label"},
                                "label": default_label
                            })
                        else:
                            comps.append({"text": str(c), "label": default_label})
                    all_entries[idx] = {"requirement_index": idx, "components": comps}

        # Final merge back into context (preserve original order)
        merged: List[Dict[str, Any]] = []
        for idx, base in enumerate(reqs, start=1):
            if idx not in all_entries:
                raise RuntimeError(f"internal error: missing parsed entry for requirement_index {idx}")
            entry = all_entries[idx]
            comps_out = []
            for c in entry["components"]:
                out = {k: v for k, v in c.items() if k != "label"}
                out["constraint"] = c["label"]  # keep your field name
                comps_out.append(out)

            lbls = sorted({c["constraint"] for c in comps_out})
            new_base = dict(base)
            new_base["components"] = comps_out
            new_base["labels"] = lbls
            new_base["constraint"] = ",".join(lbls)
            merged.append(new_base)

        context["requirements"] = merged

        try:
            (base_dir / "final_requirements.json").write_text(
                json.dumps(context["requirements"], ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

        return context
