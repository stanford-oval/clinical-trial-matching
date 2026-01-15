# modules/SMTMatcher.py
from __future__ import annotations

import json
import re
import dspy
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict

from .stages import (
    SMTLeafCollector,
    SMTVariableValueMiner,
    SMTProgramEvaluator,
    SingleRequirementEvaluator,
)

from ..utils.mbench import get_mbench, mbench_enabled


# ────────────────────────────────────────────────────────────────────────
#                                   Utils
# ────────────────────────────────────────────────────────────────────────

def _split_req_label(lbl: str) -> Tuple[str, str]:
    """
    Split 'REQ8_COMPONENT0_FOO' → ('REQ8','COMPONENT0').
    If no part code exists, returns ('REQ8','').
    NOTE: We ignore DEFINITION asserts; callers should filter them out.
    """
    parts = lbl.split("_", 2)
    if not parts:
        return (lbl, "")
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], parts[1])  # ('REQk', 'COMPONENT{m}' or 'DEFINITION{m}')



def _aggregate_per_requirement(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer existing grouping if present (so we don't overwrite upstream fixes).
    Otherwise (fallback):
      * Group by REQk
      * Consider ONLY labels that include '_COMPONENT' (ignore DEFINITIONs)
      * Aggregation over chosen statuses:
          - 'unsat' if ANY unsat
          - else 'sat' if ANY sat
          - else 'unknown'
    Also computes overall_status_grouped with 3-valued fold.
    """
    if "per_requirement_grouped" in res and isinstance(res["per_requirement_grouped"], dict) and res["per_requirement_grouped"]:
        grouped = res["per_requirement_grouped"]
        if "overall_status_grouped" not in res:
            res["overall_status_grouped"] = (
                "unsat" if any(s == "unsat" for s in grouped.values())
                else ("sat" if grouped and all(s == "sat" for s in grouped.values()) else "unknown")
            )
        return res

    if "per_requirement" not in res:
        return res

    pr = res["per_requirement"]

    # Build buckets: { 'REQ8': {'COMPONENT0':[st..], 'COMPONENT1':[st..]}, ... }
    buckets: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for lbl, st in pr.items():
        # Ignore anything that isn't a component (e.g., REQ…_DEFINITION…)
        if "_COMPONENT" not in lbl:
            continue
        rid, part = _split_req_label(lbl)
        buckets[rid][part].append(st)

    grouped: Dict[str, str] = {}
    for rid, parts in buckets.items():
        # Flatten all COMPONENT parts for this requirement
        sts = [s for arr in parts.values() for s in arr]

        if any(s == "unsat" for s in sts):
            grouped[rid] = "unsat"
        elif any(s == "sat" for s in sts):
            grouped[rid] = "sat"
        else:
            grouped[rid] = "unknown"

    res["per_requirement_grouped"] = grouped
    res["overall_status_grouped"] = (
        "unsat" if any(s == "unsat" for s in grouped.values())
        else ("sat" if grouped and all(s == "sat" for s in grouped.values()) else "unknown")
    )
    return res


def _sorted_req_id(req_id: str) -> int:
    m = re.match(r"^REQ(\d+)$", req_id, re.I)
    return int(m.group(1)) if m else 10**9


def _extract_labels_single(context: dict) -> Tuple[List[str], List[str]]:
    lab = context.get("labels") or {}
    return list(lab.get("gpt4") or []), list(lab.get("expert") or [])


def _extract_labels_per_patient(context: dict) -> Dict[str, Tuple[List[str], List[str]]]:
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    epp = context.get("eligibility_labels_per_patient") or {}
    if isinstance(epp, dict):
        for pid, d in epp.items():
            if isinstance(d, dict):
                out[pid] = (list(d.get("gpt4") or []), list(d.get("expert") or []))
    return out


def _make_alignment_from_grouped_raw_labels(
    grouped: Dict[str, str],
    gpt4_labels: List[str] | None,
    expert_labels: List[str] | None,
    criteria: List[str] | None = None,   # ⬅ may be provided by caller
) -> Dict[str, Any]:
    """
    Strict sequential alignment: row i ↔ requirement Ri (sorted numerically).
    Keeps raw labels; no mapping or agreement flags. Optionally includes criterion text.
    """
    gpt4_labels = gpt4_labels or []
    expert_labels = expert_labels or []
    criteria = criteria or []

    sorted_rids = sorted(grouped.keys(), key=_sorted_req_id)
    rows: List[Dict[str, Any]] = []

    for i, rid in enumerate(sorted_rids):
        ours_status = grouped.get(rid)  # "sat" | "unsat" | "unknown"
        g_label = gpt4_labels[i] if i < len(gpt4_labels) else None
        e_label = expert_labels[i] if i < len(expert_labels) else None
        crit = criteria[i] if i < len(criteria) else None

        rows.append(
            {
                "idx": i,
                "id": rid,
                "criterion": crit,
                "ours_status": ours_status,
                "gpt4_label": g_label,
                "expert_label": e_label,
            }
        )

    return {
        "rows": rows,
        "lengths": {
            "requirements": len(sorted_rids),
            "gpt4_labels": len(gpt4_labels),
            "expert_labels": len(expert_labels),
            "criteria": len(criteria),
        },
    }


# ────────────────────────────────────────────────────────────────────────
#                                  Runner
# ────────────────────────────────────────────────────────────────────────

class SMTMatcher(dspy.Module):
    """
    Minimal runner that:
      1) Runs the three stages.
      2) Groups requirement judgments to Rk (respect upstream grouping if present, else A1-first any-sat).
      3) Builds sequential alignment (ours_status | raw GPT-4 | raw Expert | optional criterion).
      4) Prints it and saves ONLY alignment_sequential.json to mbench.
    """

    def __init__(self, engine, *, per_requirement: bool = True):
        super().__init__()
        self.smt_leaf_collector = SMTLeafCollector()
        self.smt_variable_miner = SMTVariableValueMiner(engine)
        self.smt_program_evaluator = (
            SingleRequirementEvaluator(engine) if per_requirement else SMTProgramEvaluator(engine)
        )
        self._per_req = per_requirement

    def forward(self, context: dict) -> dict:
        # 1) Stages (no extra logging)
        context = self.smt_leaf_collector.forward(context)
        context = self.smt_variable_miner.forward(context)
        context = self.smt_program_evaluator.forward(context)

        if "eval_result" not in context:
            return context

        er = context["eval_result"]

        # 2) Single-patient
        if isinstance(er, dict) and "per_requirement" in er:
            context["eval_result"] = _aggregate_per_requirement(er)
            grouped = context["eval_result"].get("per_requirement_grouped", {}) or {}
            gpt4_labels, expert_labels = _extract_labels_single(context)
            criteria = list(context.get("criteria") or [])

            alignment = _make_alignment_from_grouped_raw_labels(grouped, gpt4_labels, expert_labels, criteria)
            context["alignment_sequential"] = alignment

            # 3) Print alignment only
            print(json.dumps(alignment, indent=2, ensure_ascii=False))

            # 4) Save ONLY alignment to mbench
            if mbench_enabled(context):
                get_mbench(context).log_json("SMTMatcher", "alignment_sequential", alignment)

            return context

        # 3) Multi-patient: {pid: {per_requirement: ...}, ...}
        if isinstance(er, dict):
            labels_pp = _extract_labels_per_patient(context)
            aligned_all: Dict[str, Any] = {}

            # Re-aggregate (respect upstream grouping if present for each patient)
            reagg: Dict[str, Any] = {}
            for pid, res in er.items():
                reagg[pid] = _aggregate_per_requirement(res)
                grouped = reagg[pid].get("per_requirement_grouped", {}) or {}
                gpt4_labels, expert_labels = labels_pp.get(pid, ([], []))
                criteria: List[str] = []  # per-patient criteria not wired; leave empty
                aligned_all[pid] = _make_alignment_from_grouped_raw_labels(grouped, gpt4_labels, expert_labels, criteria)

            context["eval_result"] = reagg
            context["alignment_sequential"] = aligned_all

            print(json.dumps(aligned_all, indent=2, ensure_ascii=False))

            if mbench_enabled(context):
                get_mbench(context).log_json("SMTMatcher", "alignment_sequential", aligned_all)

            return context

        return context
