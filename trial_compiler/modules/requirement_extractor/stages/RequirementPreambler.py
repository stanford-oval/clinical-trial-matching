#!/usr/bin/env python3
"""
RequirementPreambler.py
────────────────────────────────────────────────────────────────────────────
Rewrite pre-supplied eligibility constraint_clauses so each starts with a canonical
preamble, then dump input/output pairs as JSON logs.

• inclusion → "To be included, a patient must …"
• exclusion → "A patient is excluded if the patient …"

Context requirements:
- context["requirements"] : list[str|dict] (dicts must have "requirement")
- context["inc_exc"]      : "inclusion" | "exclusion"
- Optional prompt keys (fallbacks provided if absent):
    - context["RequirementPreambleInclusion_prompt"]
    - context["RequirementPreambleExclusion_prompt"]
"""

from __future__ import annotations

import json, pathlib, re
from typing import Dict, List, Any, Optional

import dspy

# Default prompt template (uses #PREAMBLE# and #ITEMS_JSON# placeholders)
_DEFAULT_PROMPT = """You are editing clinical trial eligibility criteria.

Task:
- For each input clause, rewrite it as ONE sentence that starts EXACTLY with the preamble below.
- Keep meaning identical. No added/removed constraints. Fix grammar minimally.
- Do not enumerate, merge, or split items; keep order 1:1.
- Return STRICT JSON only, of the form: {"rewritten":["...","...", "..."]}

Preamble:
#PREAMBLE#

items_json:
#ITEMS_JSON#
"""

# Simple code-fence stripper for robust JSON parsing
_CODEFENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

def _strip_codefence(s: str) -> str:
    return _CODEFENCE_RE.sub("", s.strip())

def _ensure_period(s: str) -> str:
    return s if s.endswith((".", "!", "?")) else s + "."

def _already_prefixed(s: str, preamble: str) -> bool:
    return s.strip().lower().startswith(preamble.strip().lower())

class RequirementPreambler(dspy.Module):
    """Rewrite each requirement to start with a canonical preamble + log results."""

    def __init__(self, engine, *, log_dir: str | pathlib.Path | None = "preamble_logs",
                 enable_llm: bool = True, verbose: bool = True):
        super().__init__()
        self.engine = engine
        self.log_dir = pathlib.Path(log_dir).expanduser() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_llm = enable_llm
        self.verbose = verbose

    def _preamble_for(self, side: str) -> str:
        if side == "inclusion":
            return "To be included, a patient must "
        return "A patient is excluded if the patient "

    def _prompt_template(self, context: Dict, side: str) -> str:
        key = "RequirementPreamblerInclusion_prompt" if side == "inclusion" \
              else "RequirementPreamblerExclusion_prompt"
        return context.get(key) or _DEFAULT_PROMPT

    def _extract_plain(self, reqs: List[Any]) -> tuple[List[str], List[Optional[dict]]]:
        items, envs = [], []
        for r in reqs:
            if isinstance(r, dict):
                envs.append(r)
                items.append(str(r.get("requirement", "")).strip())
            else:
                envs.append(None)
                items.append(str(r).strip())
        return items, envs

    def _fallback_prefix(self, items: List[str], preamble: str) -> List[str]:
        out: List[str] = []
        for s in items:
            if not s:
                out.append(preamble.rstrip())
                continue
            if _already_prefixed(s, preamble):
                out.append(_ensure_period(s))
            else:
                # avoid awkward capitalization at the join
                s2 = s[0].lower() + s[1:] if s[:1].isupper() else s
                out.append(_ensure_period(preamble + s2))
        return out

    def _rewrite_with_llm(self, prompt: str) -> Optional[List[str]]:
        if not self.enable_llm:
            return None
        try:
            raw = self.engine(prompt)[0]
            txt = _strip_codefence(raw)
            obj = json.loads(txt)
            arr = obj.get("rewritten")
            if isinstance(arr, list):
                # normalize trailing punctuation
                return [_ensure_period(str(x).strip()) for x in arr]
        except Exception as e:
            if self.verbose:
                print(f"[Preambler] LLM rewrite failed; falling back. Reason: {e}")
        return None

    def forward(self, context: Dict, *, use_full_context: bool = True) -> Dict:  # type: ignore[override]
        side = context["inc_exc"]
        tid  = context.get("trial_id", "unknown")
        reqs = context.get("requirements", [])
        if not isinstance(reqs, list) or not reqs:
            if self.verbose:
                print("[Preambler] No requirements to rewrite; skipping.")
            return context

        preamble = self._preamble_for(side)
        items, envelopes = self._extract_plain(reqs)

        # Build prompt (template from context if present, else default)
        prompt_t = self._prompt_template(context, side)
        prompt = (prompt_t
                  .replace("#PREAMBLE#", preamble)
                  .replace("#ITEMS_JSON#", json.dumps(items, ensure_ascii=False)))

        # Try LLM rewrite; if it fails, fall back to deterministic prefixing
        rewritten = self._rewrite_with_llm(prompt)
        if rewritten is None or len(rewritten) != len(items):
            rewritten = self._fallback_prefix(items, preamble)

        # Reassemble envelopes
        new_reqs: List[Any] = []
        for env, new_txt in zip(envelopes, rewritten):
            if env is None:
                new_reqs.append(new_txt)
            else:
                new_env = dict(env)
                new_env["requirement"] = new_txt
                new_reqs.append(new_env)

        # Update context
        context["requirements"] = new_reqs
        context["preamble_mapping"] = [
            {"original": o, "rewritten": n} for o, n in zip(items, rewritten)
        ]
        context["preamble"] = preamble

        # Save logs
        if self.log_dir:
            try:
                fp = self.log_dir / f"{tid}_{side}.preamble.json"
                payload = {
                    "trial_id": tid,
                    "side": side,
                    "preamble": preamble,
                    "mapping": context["preamble_mapping"],
                    "prompt_template_used": "custom" if (("RequirementPreambleInclusion_prompt" in context)
                                                         or ("RequirementPreambleExclusion_prompt" in context)) else "default"
                }
                fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[Preambler] log saved → {fp}")
            except Exception as exc:
                print(f"[Preambler] could not write log: {exc}")

        return context
