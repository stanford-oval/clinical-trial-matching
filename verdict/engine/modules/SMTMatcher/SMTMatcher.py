from __future__ import annotations

import copy
from typing import Dict, Any, List

import dspy

from .stages import (
    SMTLeafCollector,
    SMTVariableValueMiner,
    SMTProgramEvaluator,
    SMTCanonicalVariableMiner,
    SMTVariableScopeMiner,
    SMTVariableProjectionRewriter,
    SMTVariableAliasRemapper,
)

from ..utils.mbench import get_mbench, mbench_enabled


# ────────────────────────────────────────────────────────────────────────
# Whole-program stats helper
# ────────────────────────────────────────────────────────────────────────

def _summarize_whole_program(whole: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a compact summary of the whole-program solver result:

        {
          'status':            'sat' | 'unsat' | 'unknown' | 'error' | None,
          'unsat_assertions':  [label...],   # from label_status['unsat']
          'unsat_core':        [sexpr...],   # optional Z3 unsat core
          # (optional on error)
          'message':           str|None,
          'line':              int|None,
          'column':            int|None,
          'snippet':           str|None,
        }
    """
    st = None
    unsat_assertions: List[str] = []
    unsat_core: List[str] = []

    msg = None
    line = None
    col = None
    snip = None

    if isinstance(whole, dict):
        st = whole.get("status")
        ls = whole.get("label_status") or {}
        if isinstance(ls, dict):
            unsat_assertions = list(ls.get("unsat") or [])
        uc = whole.get("unsat_core")
        if isinstance(uc, list):
            unsat_core = uc

        msg = whole.get("message")
        line = whole.get("line")
        col = whole.get("column")
        snip = whole.get("snippet")

    out = {
        "status": st,
        "unsat_assertions": sorted(set(unsat_assertions)),
        "unsat_core": unsat_core,
    }

    if st == "error":
        out.update({"message": msg, "line": line, "column": col, "snippet": snip})

    return out


# ────────────────────────────────────────────────────────────────────────
#                                  Runner
# ────────────────────────────────────────────────────────────────────────

class SMTMatcher(dspy.Module):
    """
    Minimal matcher pipeline:

      1) Collect SMT leaves
      2) Scope-filter variables
      3) Rewrite surviving variables into extraction aliases / projected meanings
      4) Mine patient-specific variable values
      5) Remap alias-keyed miner outputs back to original SMT variable names
      6) Run whole-program evaluation (SMTProgramEvaluator)
      7) Attach:

         - context['eval_results']['whole_program']  (raw solver result)
         - context['stats']['whole_program']         (compact summary)
         - context['eval_result']                    (alias of whole_program)

    Notes:
    - Canonical miner path is preserved; canonical miner continues to operate on
      original variables.
    - Alias remapping is only applied to the LLM miner path.

    NEW:
    - Scope miner and projection rewriter are FROZEN BY DEFAULT.
    - Both stages support stage-level prompt caching.
    - To unfreeze:
          context["FREEZE_SCOPE_MINER"] = False
          context["FREEZE_PROJECTION_REWRITER"] = False
    - Cache defaults:
          context["CACHE_SCOPE_MINER"] = True
          context["CACHE_PROJECTION_REWRITER"] = True
    """

    def __init__(self, engine, *, per_requirement: bool = True):
        # per_requirement flag kept for backward compatibility but unused.
        super().__init__()
        self.smt_leaf_collector = SMTLeafCollector()
        self.smt_variable_scope_miner = SMTVariableScopeMiner(engine)
        self.smt_variable_projection_rewriter = SMTVariableProjectionRewriter(engine)
        self.smt_variable_miner_llm = SMTVariableValueMiner(engine)
        self.smt_variable_alias_remapper = SMTVariableAliasRemapper()
        self.smt_variable_miner_canon = SMTCanonicalVariableMiner()
        self._whole_eval = SMTProgramEvaluator(engine)

    def forward(self, context: dict) -> dict:
        # Stage-cache / freeze defaults
        context.setdefault("CACHE_SCOPE_MINER", True)
        context.setdefault("CACHE_PROJECTION_REWRITER", True)

        # Default to FROZEN as requested
        context.setdefault("FREEZE_SCOPE_MINER", True)
        context.setdefault("FREEZE_PROJECTION_REWRITER", True)

        # Default frozen cache-miss policies
        # scope miner: safer to preserve vars rather than delete everything
        context.setdefault("FROZEN_SCOPE_MISS_POLICY", "all_in_scope")
        # projection rewriter: identity rewrite
        context.setdefault("FROZEN_PROJECTION_MISS_POLICY", "identity")

        # 1) Collect leaf information from the SMT program
        context = self.smt_leaf_collector.forward(context)

        # 2) LLM scope filter (cached / frozen-aware)
        context = self.smt_variable_scope_miner.forward(context)

        # 3) Projection rewrite on surviving variables (cached / frozen-aware)
        context = self.smt_variable_projection_rewriter.forward(context)

        # 4) Mine patient-specific variable values
        if context.get("USE_CANONICAL_MINER", False):
            context = self.smt_variable_miner_canon.forward(context)
        else:
            context = self.smt_variable_miner_llm.forward(context)
            # 5) Remap alias-keyed outputs back to original SMT variable names
            context = self.smt_variable_alias_remapper.forward(context)

        # 6) Whole-program evaluation
        ctx_whole = copy.deepcopy(context)
        ctx_whole = self._whole_eval.forward(ctx_whole)
        whole_er = ctx_whole.get("eval_result") or {}

        stats = {
            "whole_program": _summarize_whole_program(whole_er),
        }

        context["eval_results"] = {
            "whole_program": whole_er,
        }
        context["stats"] = stats
        context["eval_result"] = whole_er

        # 6b) Threshold-ablation dual eval: re-run the whole-program evaluator
        # on the SAME mined patient values but with EVAL_USE_RICH_PATIENT_VALUES
        # flipped. This isolates the threshold dual-extraction (raw value +
        # range bounds + projected threshold assessment) from mining
        # nondeterminism: both verdicts share one mine.
        if context.get("DUAL_EVAL_THRESHOLD_ABLATION"):
            primary_rich = context.get("EVAL_USE_RICH_PATIENT_VALUES")
            if primary_rich is None:
                primary_rich = context.get("ENABLE_NUMERIC_RANGE_ASSERTS", True)
            ctx_alt = copy.deepcopy(context)
            ctx_alt["EVAL_USE_RICH_PATIENT_VALUES"] = (not primary_rich)
            ctx_alt["ENABLE_NUMERIC_RANGE_ASSERTS"] = (not primary_rich)
            ctx_alt.pop("eval_result", None)
            ctx_alt.pop("eval_results", None)
            ctx_alt = self._whole_eval.forward(ctx_alt)
            alt_er = ctx_alt.get("eval_result") or {}
            context["eval_results"]["whole_program_ablation"] = alt_er
            context["eval_result_ablation"] = alt_er
            context["ablation_meta"] = {
                "primary_rich": bool(primary_rich),
                "ablation_rich": bool(not primary_rich),
            }

        if mbench_enabled(context):
            mb = get_mbench(context)
            mb.log_json("SMTMatcher", "eval_results/whole_program", stats["whole_program"])
            mb.log_json(
                "SMTMatcher",
                "stage_cache_meta.json",
                context.get("stage_cache_meta") or {},
            )

        return context