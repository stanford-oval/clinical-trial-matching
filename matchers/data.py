"""Data loaders.

For the evaluation in the paper, the per-pair (chart, atoms, mined values, LM
rationales, judges) are precomputed and stored in the experiment directories.
This module provides clean loaders so each variant can be inspected on any
pair without re-running the full pipeline.
"""
from __future__ import annotations
import json, os, pathlib
from typing import Any, Dict, List, Optional

# Repo root: this file is <root>/matchers/data.py. $VERDICT_ROOT overrides, so a
# checkout can point at pair data held outside the working tree.
ROOT = pathlib.Path(os.environ.get("VERDICT_ROOT",
                                   pathlib.Path(__file__).resolve().parents[1]))
def _pair_root() -> pathlib.Path:
    """data/pairs if present, else the historical experiments/53_v2_full."""
    if os.environ.get("VERDICT_PAIR_DATA"):
        return pathlib.Path(os.environ["VERDICT_PAIR_DATA"])
    preferred = ROOT / "data" / "pairs"
    return preferred if (preferred / "cmsrc_out").exists() else \
        ROOT / "experiments" / "53_v2_full"


EXP53 = _pair_root()


def load_pair_data(pair_id: str) -> Optional[Dict[str, Any]]:
    """Load all per-pair data: chart, criteria, atom mining results, blocking
    atoms, LM-judge decision and rationale.

    Returns None if pair not found.
    """
    pid, nct = pair_id.split("__", 1)
    # Which mining snapshot to read. Several exist (cmsrc_out,
    # cmsrc_out_REMINE_v10_full, ... ) and they do NOT agree pair-for-pair, so
    # the choice changes the verdicts. $VERDICT_SNAPSHOT selects one.
    snapshot = os.environ.get("VERDICT_SNAPSHOT", "cmsrc_out")
    pdir = EXP53 / snapshot / pid
    if not pdir.exists(): return None

    # Use first variant of the trial (a/b/c) — paper aggregates if any variant accepts
    full_files = sorted(pdir.glob(f"{nct}*__full.json"))
    if not full_files: return None

    full = json.loads(full_files[0].read_text())
    inc_d = full.get("inclusion") or {}
    exc_d = full.get("exclusion") or {}
    inc_raw = inc_d.get("raw") or {}
    exc_raw = exc_d.get("raw") or {}

    note = inc_raw.get("patient_contextual_text", "")
    if not note:
        pat = inc_raw.get("patient", {})
        note = (pat.get("text") or "")[:1800]   # match experiments/96 + 97 truncation

    inc_text = inc_raw.get("trial_inclusion_criteria", "")
    exc_text = exc_raw.get("trial_exclusion_criteria", "")

    inc_summ = inc_d.get("summary") or {}
    exc_summ = exc_d.get("summary") or {}
    inc_pv = inc_raw.get("patient_var_values_rich") or {}
    exc_pv = exc_raw.get("patient_var_values_rich") or {}

    def extract_blockers(side_name, side_summ, pv):
        if side_summ.get("status") != "unsat": return []
        out = []
        for atom in side_summ.get("unsat_core", []):
            if atom.startswith("REQ"): continue
            n = atom[8:] if atom.startswith("patient_") else atom
            e = pv.get(atom) or pv.get(n) or {}
            out.append({
                "atom": atom,
                "side": side_name,
                "value": e.get("value"),
                "rationale": (e.get("assessment") or "")[:400],
                "evidence": (e.get("evidence") or "")[:300],
            })
        return out

    inc_blockers = extract_blockers("inclusion", inc_summ, inc_pv)
    exc_blockers = extract_blockers("exclusion", exc_summ, exc_pv)

    # Get LM-judge rationale and decision from overall.json
    overall_files = sorted(pdir.glob(f"{nct}*__overall.json"))
    lm_decision = None
    lm_reasoning = ""
    lm_explanation = ""
    smt_decision = None
    if overall_files:
        try:
            o = json.loads(overall_files[0].read_text())
            smt_decision = "eligible" if o.get("eligible") else "ineligible"
            lm_decision = "eligible" if o.get("llm_eligible") else "ineligible"
            lm_reasoning = (o.get("llm_reasoning") or "")
            lm_explanation = (o.get("llm_explanation") or "")
        except Exception:
            pass

    # Format lm_rsn the way experiments/96+97 scripts do (used for cache lookup).
    # Note: limits are :400 each, not :300 — match run_symmetric.get_trial_data exactly.
    parts = []
    if lm_reasoning: parts.append(f"reasoning: {lm_reasoning[:400]}")
    if lm_explanation: parts.append(f"explanation: {lm_explanation[:400]}")
    lm_rsn_for_cache = ("\n".join(parts) if parts else "")[:800]  # final cap matches run_symmetric

    return {
        "pair_id": pair_id,
        "patient_note": note[:1800],   # match pre-existing pipeline truncation
        "inclusion_criteria": inc_text[:1500],
        "exclusion_criteria": exc_text[:1500],
        "smt_decision": smt_decision,
        "smt_inclusion_status": inc_summ.get("status"),
        "smt_exclusion_status": exc_summ.get("status"),
        "smt_blocking_atoms": inc_blockers + exc_blockers,
        "n_total_atoms": len(inc_pv) + len(exc_pv),
        "lm_decision": lm_decision,
        "lm_reasoning": lm_reasoning,
        "lm_explanation": lm_explanation,
        "lm_rsn_for_cache": lm_rsn_for_cache,
    }


def load_judge_verdict(pair_id: str, judge: str = "clinician_v2") -> Optional[str]:
    """Load the gold verdict from a specific LM-based judge."""
    judges_dir = EXP53 / f"judges_{judge}"
    pid, nct = pair_id.split("__", 1)
    for jf in judges_dir.glob(f"{pid}__{nct}*.json"):
        try:
            j = json.loads(jf.read_text())
            v = j.get("judge_verdict")
            if v in ("eligible", "ineligible"):
                return v
        except Exception:
            pass
    return None


def list_disagreement_pairs() -> List[str]:
    """Return all pair IDs in the failure-mode study (122 SMT-LM disagreements)."""
    cases_file = ROOT / "experiments" / "98_failure_study" / "cases.jsonl"
    if not cases_file.exists(): return []
    return [json.loads(l)["pair"] for l in cases_file.read_text().splitlines() if l.strip()]
