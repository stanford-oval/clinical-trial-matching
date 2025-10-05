"""
SMTIncrementalSolverBasedNaiveRefiner
-------------------------------------
A “naïve” refiner that rewrites the *entire* SMT‑LIB program returned
by the LLM until the global solver reports SAT.

Key points
~~~~~~~~~~
• Keeps a short strategy/outcome history in `context`.
• After each rewrite it updates *both*
      – context["smt_program_lines"]   (full program)
      – context["new_smt_lines"]       (slice for CURRENT requirement)  ← NEW
  so the verifier can inspect the fresh fragment.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Any, List
import os, json

import dspy

from smt_core.parse_functions import parse_naive_refiner_output   # your existing parser

# ────────────────── helper: strategy/outcome history ──────────────
def _push_strategy(context: dict, comment: str, outcome: str) -> None:
    hist = context.setdefault("strategy_outcome_history", [])
    hist.append(
        {
            "timestamp": time.time(),
            "comment": comment,
            "outcome": outcome,
        }
    )


def _format_strategy_map(hist: List[dict], limit: int = 5) -> str:
    if not hist:
        return "_none yet_"
    rows = [f"* {h['comment']}  ⇒  {h['outcome']}" for h in hist[-limit:][::-1]]
    return "\n".join(rows)


# ────────────────────────── slice helper ──────────────────────────
def _extract_slice(lines: List[str], req_idx: int) -> List[str]:
    """Return only the lines whose :named tag belongs to `req_idx`."""
    tag_prefix = f"R{req_idx}_"
    return [ln for ln in lines if tag_prefix in ln]


# ───────────────────────── main module ────────────────────────────
class SMTIncrementalSolverBasedNaiveRefiner(dspy.Module):
    """Refines the SMT program with an LLM until the solver returns SAT."""

    _FENCE_RE = re.compile(r"```(?:smt2)?\s*([\s\S]*?)```", re.I)

    # ------------------------------------------------------------------
    def __init__(self, engine, *, target_outcome: str = "sat", log_dir: str | None = "./validator_logs"):
        super().__init__()
        self.engine = engine
        self.target_outcome = target_outcome
        # mbench 根目录（与 validator 默认为同一个根，保持一致）
        self.log_dir = log_dir or "./validator_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        context: Dict[str, Any],
        solver_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 1️⃣ Prepare prompt ------------------------------------------------
        program_text = "\n".join(context.get("smt_program_lines", []))
        prompt_tpl = context["SMTIncrementalSolverBasedNaiveRefiner_prompt"]

        prompt = prompt_tpl.format(
            original_whole_program=program_text,
            solver_status=solver_result["status"],
            solver_message=solver_result.get("message", ""),
            unsat_core=solver_result.get("unsat_core", "∅"),
            strategy_map=_format_strategy_map(
                context.get("strategy_outcome_history", [])
            ),
        )


        # mbench layout: <log_root>/<trial_id>/<inclusion|exclusion>/reqNNN/
        trial_id = context.get("trial_id", "unknown_trial")
        side  = context.get("inc_exc", "unknown")
        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc","inclusion","include","in"}: return "inclusion"
            if s in {"exc","exclusion","exclude","ex"}: return "exclusion"
            return s or "unknown"
        req_idx = int(context.get("current_requirement_index", 0))
        req_dir = os.path.join(self.log_dir, str(trial_id), _bucket(side), f"req{req_idx:03d}")
        os.makedirs(req_dir, exist_ok=True)
        # 以已有策略历史长度作为当前 refiner 调用的序号（1-based）
        attempt_no = len(context.get("strategy_outcome_history", [])) + 1
        p_log = os.path.join(req_dir, f"refiner_attempt{attempt_no:02d}_prompt.txt")
        r_log = os.path.join(req_dir, f"refiner_attempt{attempt_no:02d}_raw.txt")
        try:
            with open(p_log, "w", encoding="utf-8") as fh:
                fh.write(prompt)
        except Exception:
            pass


        # 2️⃣ Call LLM ------------------------------------------------------
        llm_out = self.engine(prompt)[0].strip()
        try:
            with open(r_log, "w", encoding="utf-8") as fh:
                fh.write(llm_out)
        except Exception:
            pass

        parsed = parse_naive_refiner_output(llm_out)

        smt_lines = parsed["corrected_whole_program"]
        strategy = parsed["strategy_description"].strip() or "<naive‑rewrite>"

        # Fallback if the parser didn’t find fenced code
        if not smt_lines:
            m = self._FENCE_RE.search(llm_out)
            smt_text = (m.group(1) if m else llm_out).strip()
            smt_lines = smt_text.splitlines()
            strategy = "<raw‑rewrite‑no‑tags>"

        _push_strategy(context, strategy, outcome="pending")

        # 3️⃣ Update context -----------------------------------------------
        context["smt_program_lines"] = smt_lines

        # ——— NEW: expose slice for the verifier ————————————————
        req_idx = context.get("current_requirement_index")
        if req_idx is not None:
            context["new_smt_lines"] = _extract_slice(smt_lines, req_idx)  # NEW

        context["registry_refiner_comment"] = strategy

        # mbench: 落盘当前重写后的完整程序与 strategy
        try:
            prog_path = os.path.join(req_dir, f"refiner_attempt{attempt_no:02d}_program.smt2")
            with open(prog_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(smt_lines))
            meta_path = os.path.join(req_dir, f"refiner_attempt{attempt_no:02d}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"strategy": strategy}, fh, indent=2, ensure_ascii=False)
        except Exception:
            pass


        return context
