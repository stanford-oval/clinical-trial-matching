#!/usr/bin/env python3
"""gpt-5 chart modifier v2 for NL-rationale systems (v5/tg/shah/v5_blockers/v5_gpt5).

Mirror of cf_modifier_gpt5.py (SMT/typed-atom version) but accepts a
free-text rationale instead of typed atom targets.
"""
from __future__ import annotations
import json, os, pathlib, urllib.request

KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODIFIER_MODEL", "gpt-5")  # "gpt-5" (default) or "gpt-4.1"
if MODEL == "gpt-4.1":
    EP = os.environ.get("OPENAI_ENDPOINT", "")
else:
    EP = os.environ.get("OPENAI_ENDPOINT_GPT5") or os.environ.get("OPENAI_ENDPOINT", "")
BASE = EP.split("/openai/")[0] if EP else ""
EP_GPT5 = EP  # back-compat


PROMPT = """# ROLE
You are an expert clinical chart editor. A trial matcher said this patient is
INELIGIBLE for the trial. The matcher cited specific reasons (below).
Rewrite the chart so that EVERY cited reason is neutralized — preserving
everything else.

# HARD RULES

R1. MINIMALITY. Make the smallest edit that neutralizes each cited reason.
    For a numeric blocker (e.g., "patient is 82, trial requires 18-65"):
    REPLACE the number in place. Do not delete the surrounding sentence.
    For a categorical blocker (e.g., "patient has lung disease"): remove
    the smallest phrase that established the fact, not the whole section.

R2. COHERENCE. The rewritten chart MUST read as a real, internally
    consistent clinical note. No dangling references, no broken pronouns,
    no orphaned dates, no obvious contradictions. Re-read your own output
    and fix any inconsistency before returning.

R3. NO COLLATERAL DAMAGE. Every fact NOT named as a blocker must remain
    supported. In particular, if the chart contains text that supports an
    INCLUSION criterion (e.g., a diagnosis the trial requires), you may
    NOT delete or alter that text — even if it is adjacent to a cited
    blocker.

R4. ADDRESS EVERY CITED REASON. The system gave you the complete set of
    reasons it found this patient ineligible. Each one must be neutralized
    by the edit. Do not address only a subset.

R5. NO NEW DIAGNOSES. Do not invent findings unrelated to the cited
    reasons. Do not add labs, imaging, dates, or comorbidities the chart
    did not have.

# SELF-CHECK (perform internally before returning)

  Q1. Did every cited reason get neutralized? (Point to the chart line
      that now establishes the new value.)
  Q2. Did any inclusion-supporting fact get deleted or altered? (If yes — revise.)
  Q3. Does the chart read coherently end-to-end? (If not — revise.)
  Q4. Did you introduce content not strictly necessary for a cited-reason fix?
      (If yes — revise.)

# INPUTS

ORIGINAL CHART:
{{CHART}}

TRIAL (for context):
{{TRIAL}}

SYSTEM-CITED REASONS for ineligibility (these are ALL the reasons —
neutralize every one):
{{RATIONALE}}

INCLUSION-SUPPORTING FACTS to PRESERVE (do NOT delete or alter chart text
that establishes these — the patient must remain eligible on the inclusion
side after your edit):
{{SUPPORTS}}

# OUTPUT
Return ONLY the rewritten chart text. No preamble, no markdown, no commentary.
"""


def _gpt5_call(prompt: str, max_tokens: int = 6000, max_reasoning: int = 4000) -> str:
    if not BASE or not KEY:
        raise RuntimeError("OPENAI_ENDPOINT(_GPT5) + OPENAI_API_KEY required")
    if MODEL == "gpt-4.1":
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        url = f"{BASE}/chat/completions?api-version=2024-08-01-preview"
    else:
        body = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens + max_reasoning,
        }
        url = f"{BASE}/openai/deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"api-key": KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return (resp["choices"][0]["message"]["content"] or "").strip()


def generate_cf_gpt5_rationale(chart: str, trial: str, rationale: str,
                                supports: str | None = None) -> str:
    if not rationale: return chart
    sup = supports if supports else "(no explicit support list — preserve any chart content that supports an inclusion criterion)"
    prompt = (PROMPT
              .replace("{{CHART}}", chart[:3500])
              .replace("{{TRIAL}}", (trial or "")[:2500])
              .replace("{{RATIONALE}}", rationale[:2500])
              .replace("{{SUPPORTS}}", sup[:2500]))
    return _gpt5_call(prompt)


if __name__ == "__main__":
    chart = "Patient is a 82-year-old male with stage IV lung adenocarcinoma."
    trial = "Inclusion: age 18-65. Exclusion: metastatic disease."
    rationale = "[Excluded] Age 82 exceeds the upper limit of 65. [Excluded] Stage IV indicates metastatic disease."
    out = generate_cf_gpt5_rationale(chart, trial, rationale)
    print("ORIG:", chart)
    print("CF:  ", out)
