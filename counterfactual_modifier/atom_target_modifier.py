#!/usr/bin/env python3
"""gpt-5 chart modifier v2 — sharpened coherence + minimality.

Differences from cf_aegis_targets.prompt + cf_generator.py:
  - gpt-5 instead of gpt-4.1 (more capable for compositional surgery)
  - Explicit coherence rule (chart must read as a real clinical note end-to-end)
  - Explicit minimality rule (smallest edit that flips the target)
  - Explicit "in-place replacement" guidance for numeric atoms (replace the
    value, do not delete the surrounding context)
  - Self-check step at the end (model verifies its own edit preserves coherence)

Reads the same target/preserve/other-atoms shape used by the v1 modifier
so we can drop it into the existing run.py pipeline.
"""
from __future__ import annotations
import json, os, pathlib, urllib.request

ROOT = pathlib.Path(os.environ.get("VERDICT_ROOT",
                                   pathlib.Path(__file__).resolve().parents[1]))
KEY = os.environ.get("OPENAI_API_KEY", "")
# MODIFIER_MODEL env var: "gpt-5" (default) or "gpt-4.1" — controls which
# Azure deployment is called. Setting to "gpt-4.1" replicates the
# mbench_modifier_gpt4_1 published numbers.
MODEL = os.environ.get("MODIFIER_MODEL", "gpt-5")
if MODEL == "gpt-4.1":
    EP = os.environ.get("OPENAI_ENDPOINT", "")
else:
    EP = os.environ.get("OPENAI_ENDPOINT_GPT5") or os.environ.get("OPENAI_ENDPOINT", "")
BASE = EP.split("/openai/")[0] if EP else ""
# Back-compat alias used elsewhere in this file
EP_GPT5 = EP


PROMPT = """# ROLE
You are an expert clinical chart editor. Rewrite a patient chart so that
specified atomic facts ("TARGETS") become supported, while preserving every
other fact in the original chart.

# HARD RULES (do not violate any of these)

R1. MINIMALITY. Make the smallest edit that flips the TARGET atom's polarity.
    For a numeric atom (e.g., "age = 82 -> 18"): REPLACE the number in place.
    Do not delete the surrounding sentence, paragraph, or chart context.
    For a categorical atom (e.g., "has_lung_disease = TRUE -> FALSE"): remove
    the smallest phrase that established the fact (e.g., one finding line),
    not the whole respiratory section.

R2. COHERENCE. The rewritten chart MUST read as a real, internally
    consistent clinical note. No dangling references, no broken pronouns,
    no orphaned dates, no obvious contradictions ("denies chest pain" after
    you removed an MI mention but the chart still says "post-MI status").
    Re-read your own output and fix any inconsistency before returning.

R3. NO COLLATERAL DAMAGE. Every fact in the PRESERVE list and every fact
    NOT in the TARGETS list must remain supported. If an inclusion criterion
    is supported by chart text (e.g., "critical limb ischemia" supporting an
    inclusion atom), you may NOT delete or alter that text — even if it is
    adjacent to a TARGET line.

R4. SILENT QUALIFIERS STAY SILENT. If OTHER ATOMS shows
    `X@@<qualifier> = NULL`, do not introduce a qualifier when flipping X.
    Write "patient is referred for coronary angiography" — NOT "patient is
    referred for elective coronary angiography" or "urgent coronary
    angiography." The qualifier was silent; keep it silent.

R5. NO NEW DIAGNOSES. Do not invent findings unrelated to the targets.
    Do not add labs, imaging, dates, or comorbidities the chart did not have.

# SELF-CHECK (perform internally before returning)

Before you output the final chart, mentally walk these four questions:
  Q1. Did each TARGET flip? (For each target, point to the chart line that
      now establishes the new value.)
  Q2. Did any PRESERVE fact get deleted or altered? (If yes — revise.)
  Q3. Does the chart read coherently end-to-end? (If not — revise.)
  Q4. Did you introduce content not in the original? (If yes that wasn't
      strictly necessary for a target flip — revise.)

If any answer is "no/yes-broken", silently fix and re-check before output.

# INPUTS

ORIGINAL CHART:
{{CHART}}

TARGETS to flip (atom = desired_value):
{{TARGETS}}

PRESERVE (atomic facts that must stay supported, with chart evidence):
{{PRESERVE}}

OTHER ATOMS (must remain at current value when re-mined; silents stay silent):
{{OTHER_ATOMS}}

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


def _fmt_targets(targets):
    lines = []
    for t in targets:
        atom = t["atom"]
        tv = t.get("target_value")
        cv = t.get("current_value")
        readable = atom.replace("_", " ").replace("@@", " qualified by ")
        line = f"- atom `{atom}`: target value = {tv}"
        if cv is not None: line += f" (current: {cv})"
        line += f'\n  meaning: "{readable}"'
        if t.get("assessment"):
            line += f"\n  current evidence/assessment: {t['assessment'][:200]}"
        lines.append(line)
    return "\n".join(lines)[:2500]


def _fmt_preserve(preserve):
    if not preserve: return "_(no atoms to preserve specified)_"
    lines = []
    for p in preserve:
        atom = p["atom"]
        readable = atom.replace("_", " ").replace("@@", " qualified by ")
        cv = p.get("current_value")
        line = f"- atom `{atom}` = `{cv}` (KEEP THIS)\n  meaning: \"{readable}\""
        if p.get("evidence"):
            line += f'\n  chart evidence to preserve: "{(p["evidence"] or "")[:200]}"'
        lines.append(line)
    return "\n".join(lines)[:3000]


def _fmt_other(other_atoms):
    if not other_atoms: return "_(no other atoms specified)_"
    lines = []
    for a in other_atoms:
        atom = a["atom"]
        readable = atom.replace("_", " ").replace("@@", " qualified by ")
        cv = a.get("current_value")
        cv_label = "NULL (silent in original chart — must remain silent)" if cv is None else f"`{cv}`"
        lines.append(f"- `{atom}` = {cv_label}  ({readable})")
    return "\n".join(lines)[:4000]


def generate_cf_gpt5_targets(chart: str, targets: list, preserve: list = None,
                              other_atoms: list = None) -> str:
    if not targets: return chart
    prompt = (PROMPT
              .replace("{{CHART}}", chart[:3500])
              .replace("{{TARGETS}}", _fmt_targets(targets))
              .replace("{{PRESERVE}}", _fmt_preserve(preserve or []))
              .replace("{{OTHER_ATOMS}}", _fmt_other(other_atoms or [])))
    return _gpt5_call(prompt)


if __name__ == "__main__":
    # Smoke test
    chart = "Patient is a 82-year-old male with stage IV lung adenocarcinoma."
    targets = [{"atom": "age_in_years", "target_value": 65, "current_value": 82,
                "assessment": "82-year-old"}]
    other = [{"atom": "has_lung_cancer", "current_value": "TRUE"}]
    result = generate_cf_gpt5_targets(chart, targets, other_atoms=other)
    print("ORIGINAL:", chart)
    print("REWRITE: ", result)
