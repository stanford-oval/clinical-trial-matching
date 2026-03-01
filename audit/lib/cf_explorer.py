"""Counterfactual explorer: answer 'what would need to change for this
patient to become eligible?'

Uses the v2 heuristic parser + Z3 MaxSAT to identify the minimum set of
variable changes that would flip an ineligible decision. Returns a
clinician-friendly list of actionable changes.
"""
from __future__ import annotations
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from evaluation.explainability.smtlib_parser import extract_named_assertions, extract_targets
from evaluation.explainability.z3_target_solver import extract_targets_via_optimize
from audit.lib.verbalize import variable_meaning, value_display, _extract_meaning


def _bool_label(val) -> str:
    if val is True: return "yes"
    if val is False: return "no"
    if isinstance(val, str):
        s = val.lower().strip()
        if s in ("true", "1", "t", "yes"): return "yes"
        if s in ("false", "0", "f", "no"): return "no"
    return str(val) if val is not None else "not documented"


def find_cf_targets(smt_decision: dict, method: str = "z3") -> dict:
    """Return {var: {target_value, current_value, meaning, side}} for vars
    that need to change for eligibility to flip. method='z3' uses Z3 MaxSAT;
    method='heuristic' uses unsat-core walking.

    If the pair is already eligible, returns {}."""
    if smt_decision.get("eligible") is True:
        return {}

    inc_raw = (smt_decision.get("inclusion") or {}).get("raw") or {}
    exc_raw = (smt_decision.get("exclusion") or {}).get("raw") or {}
    inc_prog = "\n".join(inc_raw.get("smt_program_lines") or [])
    exc_prog = "\n".join(exc_raw.get("smt_program_lines") or [])
    inc_er = inc_raw.get("eval_result") or {}
    exc_er = exc_raw.get("eval_result") or {}
    inc_unsat = inc_er.get("status") == "unsat"
    exc_unsat = exc_er.get("status") == "unsat"

    # Gather per-side variable indices + current mined values
    inc_vindex = inc_raw.get("variable_index") or {}
    exc_vindex = exc_raw.get("variable_index") or {}
    vindex = {**inc_vindex, **exc_vindex}
    inc_pvv = inc_raw.get("patient_var_values") or {}
    exc_pvv = exc_raw.get("patient_var_values") or {}
    pvv = {**inc_pvv, **exc_pvv}

    targets = {}
    if method == "z3":
        try:
            z3_targets, status = extract_targets_via_optimize(
                inc_prog or None, exc_prog or None,
                inc_pvv, exc_pvv,
                inc_was_unsat=inc_unsat, exc_was_unsat=exc_unsat,
            )
            if status == "ok":
                targets = z3_targets
        except Exception:
            targets = {}

    # Fallback: heuristic unsat-core walking
    if not targets:
        for side_raw, side_er in ((inc_raw, inc_er), (exc_raw, exc_er)):
            if side_er.get("status") != "unsat": continue
            labels = (side_er.get("label_status") or {}).get("unsat") or []
            smt_src = "\n".join(side_raw.get("smt_program_lines") or [])
            named = extract_named_assertions(smt_src)
            for lbl in labels:
                if "_AUXILIARY" in lbl: continue
                body = named.get(lbl)
                if body is None: continue
                targets.update(extract_targets(body))

    # Enrich with metadata
    out = {}
    for var, tgt in targets.items():
        cur_raw = pvv.get(var)
        cur_val = cur_raw.get("value") if isinstance(cur_raw, dict) else cur_raw
        side = "inclusion" if var in inc_vindex else "exclusion" if var in exc_vindex else "unknown"
        out[var] = {
            "target_value": tgt,
            "current_value": cur_val,
            "meaning": variable_meaning(var, vindex),
            "side": side,
            "evidence": (cur_raw.get("evidence") if isinstance(cur_raw, dict) else "") or "",
        }
    return out


def _target_phrasing(var: str, info: dict) -> str:
    """Render a single target as an actionable sentence for the clinician."""
    target = info["target_value"]
    current = info["current_value"]
    meaning = info["meaning"]
    side = info["side"]
    # Determine phrasing based on types
    if isinstance(target, bool) or (isinstance(target, str) and target.lower() in ("true","false")):
        tgt_bool = target if isinstance(target, bool) else (target.lower() == "true")
        cur_bool = None
        if isinstance(current, bool): cur_bool = current
        elif isinstance(current, str) and current.strip().lower() in ("true","false"):
            cur_bool = current.lower() == "true"
        if cur_bool is None:
            # Chart is silent — clinician must document
            if tgt_bool:
                return f"Document that the patient has: <b>{meaning}</b>"
            else:
                return f"Confirm the patient does NOT have: <b>{meaning}</b>"
        else:
            # Actual value contradicts what's needed
            if tgt_bool and not cur_bool:
                return f"Patient would need to have: <b>{meaning}</b> (chart currently says no)"
            if (not tgt_bool) and cur_bool:
                return f"Patient would need to NOT have: <b>{meaning}</b> (chart currently says yes)"
    # Numeric or other
    cur_show = value_display(current)
    return f"<b>{meaning}</b> would need to be {target} (currently {cur_show})"


def render_cf_section_html(cf_targets: dict) -> str:
    """Render the 'what would change this' section as HTML."""
    if not cf_targets:
        return ""
    rows_html = []
    for var, info in cf_targets.items():
        phrasing = _target_phrasing(var, info)
        side_tag = f'<span style="font-size:0.85em;color:#6b7280">({info["side"]})</span>'
        rows_html.append(f'<li style="margin:6px 0">{phrasing} {side_tag}</li>')
    n = len(cf_targets)
    return f"""
    <section style="margin:24px 0;padding:16px;background:#fef3c7;border-left:4px solid #f59e0b;border-radius:4px">
      <h2 style="margin:0 0 8px 0;color:#92400e">What would change this decision?</h2>
      <p style="color:#78350f;margin:0 0 12px 0">
        Based on the trial's requirements, {n} chart fact{'s' if n != 1 else ''} would need to change
        for this patient to potentially meet the criteria. A clinician may want to confirm each
        item during full enrollment screening.
      </p>
      <ol style="margin:8px 0;padding-left:24px;color:#451a03">
        {"".join(rows_html)}
      </ol>
    </section>
    """
