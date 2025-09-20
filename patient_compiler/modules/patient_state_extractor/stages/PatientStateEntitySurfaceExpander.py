# -*- coding: utf-8 -*-
"""Patient‑fact entity‑span expander with verifier feedback, `best_good` caching,
**and full logging support**.

Differences to prior quick‑fix version
-------------------------------------
1. Restored `log_dir` handling – pass `log_dir="path/to/out"` when constructing
   to dump:
   * `span_attempt_logs.json` — every regen try & verifier result
   * `span_final_pairs.txt`   — ORIGINAL vs REWRITTEN pairs (final)
2. `_build_prompt()` again uses **"\n".join** to preserve line breaks.
3. Added `_pairs_text()` helper for readable txt export.
4. Minor tidy‑ups (typing, comments).
"""
from __future__ import annotations

from pathlib import Path
import json, re
from typing import List, Dict, Union, Tuple, Any

import dspy

from .PatientStateEntitySurfaceExpanderVerifier import (
    PatientStateEntitySurfaceExpanderVerifier,
)

_OPEN_TAG = "<rewritten_patient_fact_list>"
_CLOSE_TAG = "</rewritten_patient_fact_list>"
_idx_line_re = re.compile(r"^\[(\d+)\]\s*(.+)")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_block(raw: str) -> str | None:
    m = re.search(re.escape(_OPEN_TAG) + r"(.*?)" + re.escape(_CLOSE_TAG), raw, re.S)
    return m.group(1).strip() if m else None

def parse_rewrite_output(raw: str, expect_n: int) -> Union[List[str], bool]:
    body = _extract_block(raw)
    if body is None:
        return False
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    collected: List[Tuple[int, str]] = []
    for ln in lines:
        m = _idx_line_re.match(ln)
        if not m:
            return False
        collected.append((int(m.group(1)), m.group(2).strip()))
    if len(collected) != expect_n:
        return False
    collected.sort(key=lambda t: t[0])
    if [i for i, _ in collected] != list(range(expect_n)):
        return False
    return [txt for _, txt in collected]

# ---------------------------------------------------------------------------
# Span expander main class
# ---------------------------------------------------------------------------
class PatientStateEntitySpanExpander(dspy.Module):
    MAX_REGEN_ATTEMPTS = 3  # outer LLM regenerate
    PARSE_RETRY        = 2  # attempts to get parsable output per regen
    MAX_VERIFY_PASSES  = 2  # inner correction passes

    def __init__(self, engine: dspy.Module, log_dir: str | Path | None = None):
        super().__init__()
        self.engine   = engine
        self.verifier = PatientStateEntitySurfaceExpanderVerifier(engine, max_attempts=2)
        self.log_dir  = Path(log_dir) if log_dir else None

    # ---------------- helpers ----------------
    @staticmethod
    def _build_prompt(ctx_txt: str, fact_lines: List[str], ctx: Dict) -> str:
        tpl = ctx.get("PatientStateEntitySpanExpander_prompt")
        return tpl.replace("#CONTEXTUAL_TEXT#", ctx_txt).replace("#PATIENT_FACT_TEXT#", "\n".join(fact_lines))

    @staticmethod
    def _pairs_text(pf_items: List[Dict[str, Any]]) -> str:
        parts = []
        for i, r in enumerate(pf_items):
            orig = r.get("source", r.get("fact", "")).strip()
            rew  = r.get("fact", "").strip()
            parts += [f"[{i:02d}] ORIGINAL: {orig}", f"[{i:02d}] REWRITTEN: {rew}"]
        return "\n".join(parts)

    # ---------------- main -------------------
    def forward(self, context: Dict, use_full_context: bool = True) -> Dict:
        ctx_txt  = context.get("contextual_text", "")
        pf_items = context.get("patient_facts", [])
        orig_lines = [(it.get("fact") if isinstance(it, dict) else str(it)).strip() for it in pf_items]

        best_good: Dict[int, str] = {}
        attempt_logs: List[Dict[str, Any]] = []

        # -------- regen loop --------
        for regen in range(1, self.MAX_REGEN_ATTEMPTS + 1):
            prompt = self._build_prompt(ctx_txt, orig_lines, context)

            # --- obtain parsable rewrite ---
            for _ in range(self.PARSE_RETRY):
                raw = self.engine(prompt)[0]
                rewritten = parse_rewrite_output(raw, expect_n=len(orig_lines))
                if rewritten is not False:
                    print("succeed to parse the expander raw output!!!")
                    break
                else:
                    rewritten = orig_lines
            

            # --- verify & iterative corrections ---
            for verify_pass in range(1, self.MAX_VERIFY_PASSES + 1):
                ok, failed, reasons, parsed = self.verifier(ctx={}, original_reqs=orig_lines, expanded_reqs=rewritten)

                # log
                attempt_logs.append({
                    "regen": regen,
                    "verify_pass": verify_pass,
                    "all_passed": ok,
                    "failed_indices": failed,
                    "reasons": reasons,
                    "parsed_results": parsed
                })

                # update best_good
                for idx, txt in enumerate(rewritten):
                    if str(idx) not in failed:
                        best_good[idx] = txt

                if ok:
                    break  # done with verify loop

                # apply corrections
                for idx_str in failed:
                    corr = (parsed.get(idx_str, {}).get("corrected_fact") or "").strip()
                    idx = int(idx_str)
                    if corr and corr != rewritten[idx]:
                        rewritten[idx] = corr



            if ok:
                break  # all good, exit regen loop

        # -------- assemble final result --------
        final_lines = [best_good.get(i, orig_lines[i]) for i in range(len(orig_lines))]
        mapping     = dict(zip(orig_lines, final_lines))

        new_pf: List[Dict[str, Any]] = []
        for old, new, obj in zip(orig_lines, final_lines, pf_items):
            upd = dict(obj) if isinstance(obj, dict) else {}
            upd.update({"fact": new, "source": old})
            new_pf.append(upd)

        context.update({
            "patient_facts": new_pf,
            "span_mapping": mapping,
            "span_attempt_logs": attempt_logs,
            "best_good_indices": sorted(best_good),
        })

        # -------- logging --------
        expansion_log_out = self.log_dir
        if expansion_log_out:
            base = Path(expansion_log_out)
            out_dir = base.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = base.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{stem}.span_attempt_logs.json").write_text(json.dumps(attempt_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / f"{stem}.span_final_pairs.txt").write_text(self._pairs_text(new_pf), encoding="utf-8")

        return context