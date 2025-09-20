#!/usr/bin/env python3
"""
patient_fact_rudimentary_extractor.py
────────────────────────────────────────────────────────────────────────────
Extract atomic patient facts from a patient note, verify that every sentence
in the source note is represented (fuzzy), and (optionally) dump a JSON
debug log.

If all validation attempts fail, the attempt with the **largest number of
extracted facts** is used as the final output.

Expected LLM output format:

<patient_fact_extraction_process>
...your step-by-step notes...
</patient_fact_extraction_process>

<patient_facts>
[
  {"fact": "<Fact 1>", "text_span": "<exact supporting text>"},
  {"fact": "<Fact 2>", "text_span": "<exact supporting text>"}
]
</patient_facts>
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Dict, List

import dspy
from .PatientStateRudimentaryExtractorVerifier import PatientStateRudimentaryExtractorVerifier


# ────────────────────────────────────────────────────────────────
# Regexes & helpers
_MARKER_RE = re.compile(r'^[\s]*(?:[-•–]|\(\d+\)|\d+[.)])\s*(.+)$', flags=re.MULTILINE)
_WS_RE = re.compile(r'\s+')
# Reasonable sentence splitter: split on ., !, ? followed by whitespace/newline; also split on newlines
_SENT_SPLIT_RE = re.compile(r'(?<!\b[A-Z])[.!?]\s+|\n+')

def _norm(txt: str) -> str:
    return _WS_RE.sub(" ", (txt or "")).strip().lower()

def _strip_headers_patient_note(txt: str) -> str:
    # For patient notes we *do not* strip headers by default.
    return "\n".join((txt or "").splitlines())


# ────────────────────────────────────────────────────────────────
# Minimal parser for the model's output
def parse_extract_patient_facts_output(raw: str) -> List[Dict[str, str]]:
    """
    Pull JSON list from inside <patient_facts> ... </patient_facts>.
    Returns: [{"fact": "...", "text_span": "..."}, ...] or [] if not found/parsed.
    """
    if not isinstance(raw, str):
        return []
    m = re.search(r"<patient_facts>\s*(\[.*?\])\s*</patient_facts>", raw, flags=re.S | re.I)
    if not m:
        return []
    blob = m.group(1)
    try:
        arr = json.loads(blob)
    except Exception:
        return []
    out = []
    for x in arr if isinstance(arr, list) else []:
        fact = x.get("fact") or x.get("requirement") or ""
        span = x.get("text_span") or ""
        if fact and span:
            out.append({"fact": fact, "text_span": span})
    return out


class PatientStateRudimentaryExtractor(dspy.Module):
    MAX_ATTEMPTS = 3

    def __init__(self, engine, *, log_dir: str | pathlib.Path | None = "extract_logs"):
        super().__init__()
        self.engine = engine
        self.log_dir = pathlib.Path(log_dir).expanduser() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.coverage_llm = PatientStateRudimentaryExtractorVerifier(engine, debug=True)

    # ------------------------------------------------------------------ main
    def forward(self, context: Dict, *, use_full_context: bool) -> Dict:
        """
        Expects `context` to contain:
          - "contextual_text": str (optional; can be "")
          - "patient_note": str (preferred) OR "requirement_text": str (fallback)
          - "PatientFactRudimentaryExtractor_prompt": str (preferred)
            OR "PatientStateRudimentaryExtractor_prompt" / *_v2 (fallback)

        The prompt should include placeholders:
          #CONTEXTUAL_TEXT#, #PATIENT_NOTE# (and optionally #REQUIREMENT_TEXT# for compat)
        """
        ctx_txt  = context.get("contextual_text", "")
        note_txt = context.get("patient_note") or context.get("requirement_text", "")
        if not isinstance(note_txt, str):
            note_txt = str(note_txt or "")

        prompt_t = (context.get("PatientFactRudimentaryExtractor_prompt")
                    or context.get("PatientStateRudimentaryExtractor_prompt")
                    or context.get("PatientStateRudimentaryExtractor_prompt_v2"))

        if not prompt_t:
            raise ValueError("Missing prompt template: provide 'PatientFactRudimentaryExtractor_prompt' in context.")

        prompt = (prompt_t.replace("#CONTEXTUAL_TEXT#", ctx_txt)
                           .replace("#PATIENT_NOTE#",  note_txt)
                           .replace("#REQUIREMENT_TEXT#", note_txt))  # backward compatibility

        attempts: List[dict] = []       # store metadata of every try
        best_attempt: dict | None = None   # highest #facts so far

        for i in range(1, self.MAX_ATTEMPTS + 1):
            raw = self.engine(prompt)[0]
            facts = parse_extract_patient_facts_output(raw) or []

            sent_ok, reasons = self.coverage_llm(context, note_txt, facts)

            attempt_rec = {
                "attempt":            i,
                "n_facts":            len(facts),
                "sentence_ok":        sent_ok,
                "fail_reasons":  reasons[:10],  # cap preview
                "facts":              facts,         # store extraction
                "raw":                raw,
            }
            attempts.append(attempt_rec)

            # track best (#facts) even if fail
            if (best_attempt is None) or (len(facts) > best_attempt["n_facts"]):
                best_attempt = attempt_rec

            if sent_ok:
                break
            if i < self.MAX_ATTEMPTS:
                print(f"[Extractor] retrying {i}/{self.MAX_ATTEMPTS} …")

        # choose successful attempt or the best-coverage fallback
        final = attempts[-1] if attempts[-1]["sentence_ok"] else best_attempt

        # ---- summary ----
        print("\n[Extractor] sentence-coverage summary")
        for r in attempts:
            status = "PASS" if r["sentence_ok"] else "FAIL"
            mb = "; ".join(r["fail_reasons"]) or "—"
            print(f"  • attempt {r['attempt']:>2}: {status:4} | "
                  f"#facts={r['n_facts']:>2} | fail reasons: {mb}")
        if not final["sentence_ok"]:
            print(f"\n[Extractor] All attempts failed – using attempt "
                  f"{final['attempt']} with the most extracted facts "
                  f"({final['n_facts']}).")

        # ---- optional JSON log ----
        if self.log_dir:
            try:
                # Use note_id if available; fallback to trial_id; otherwise "unknown"
                nid = context.get("note_id") or context.get("trial_id") or "unknown"

                side = "patientfacts"
                # Use attempt number instead of timestamp
                attempt_num = final["attempt"]

                fp = self.log_dir / f"{nid}_{side}_attempt{attempt_num}.json"
                fp.write_text(json.dumps({
                    "note_id": nid,
                    "side": side,
                    "patient_note": note_txt,
                    "attempt_log": attempts,
                    "best_attempt": final["attempt"],
                }, indent=2), encoding="utf-8")
                print(f"[Extractor] log saved → {fp}")
            except Exception as exc:
                print(f"[Extractor] could not write log: {exc}")


        
        # ---- expose results ----
        context["patient_facts"] = final["facts"]
        context["extraction_metrics"] = {
            k: v for k, v in final.items()
            if k in {"n_facts", "sentence_ok", "missing_sentences", "attempt"}
        }
        return context

