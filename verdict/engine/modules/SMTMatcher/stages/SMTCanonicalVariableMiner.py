# modules/stages/SMTCanonicalVariableMiner.py
from __future__ import annotations

import math, re, json, pathlib
from typing import Dict, Any, List, Optional, Tuple
import dspy

from ...utils.mbench import get_mbench, mbench_enabled

_HRS = dict(
    minute=1/60, minutes=1/60,
    hour=1, hours=1,
    day=24, days=24,
    week=24*7, weeks=24*7,
    month=24*30, months=24*30,   # simple, deterministic
    year=24*365, years=24*365,
)

def _parse_timeframe_to_hours(tf: str) -> Tuple[float, float, Optional[float]]:
    tf = (tf or "").strip().lower()
    if tf == "now": return (0.0, 0.0, None)
    if tf == "inthehistory": return (-math.inf, 0.0, None)
    if tf == "inthefuture": return (0.0, math.inf, None)
    m = re.fullmatch(r"inthepast(\d+)([a-z]+)", tf)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return (-n * _HRS[unit], 0.0, None)
    m = re.fullmatch(r"inthefuture(\d+)([a-z]+)", tf)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return (0.0, n * _HRS[unit], None)
    m = re.fullmatch(r"foradurationof(\d+)([a-z]+)", tf)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        D = n * _HRS[unit]
        return (-D, 0.0, D)
    raise ValueError(f"Unrecognized timeframe: {tf}")

def _intervals_overlap(RL: float, RR: float, PL: float, PR: float, incP_L: bool, incP_R: bool) -> bool:
    if RR < PL: return False
    if RR == PL and not incP_L: return False
    if PR < RL: return False
    if PR == RL and not incP_R: return False
    return True

def _duration_satisfied(PL: float, PR: float, incP_L: bool, incP_R: bool, D: float) -> bool:
    RL, RR = -D, 0.0
    if not _intervals_overlap(RL, RR, PL, PR, incP_L, incP_R): return False
    L = max(RL, PL); R = min(RR, PR)
    return (R - L) >= D - 1e-9

def _pick_best_numeric(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands: return None
    return sorted(cands, key=lambda c: (c.get("end_time_in_hours", -math.inf),
                                        c.get("start_time_in_hours", -math.inf)))[-1]

def _scalar_value_from_fact(f: Dict[str, Any], vtype: str) -> Any:
    if (vtype or "").lower() in {"int", "integer", "real", "float", "number"}:
        for k in ("value", "numeric_value", "measured_value"):
            if k in f and f[k] is not None:
                return f[k]
        return None
    if "extracted_value" in f: return bool(f["extracted_value"])
    if "value" in f and isinstance(f["value"], bool): return f["value"]
    return None

def _iter_jsonl(path: pathlib.Path):
    if not path.exists(): return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except json.JSONDecodeError: continue

class SMTCanonicalVariableMiner(dspy.Module):
    """
    Deterministic miner that sets values **only** for canonical variables present in:
      build/canon/{trial_id}_{side}_canonical_variables.json
    or in context["canonical_variables"].

    Matches by exact entity_variable_name and enforces timeframe overlap.
    """

    def __init__(self):
        super().__init__()

    def _load_canonical(self, context: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        import sys
        def _dbg(*args):
            if context.get("VERBOSE"):
                print(*args, file=sys.stderr)

        if isinstance(context.get("canonical_variables"), list):
            cvs = context["canonical_variables"]
            _dbg("[CanonMiner] canonical_variables provided in context. count=", len(cvs))
            return cvs, "context"

        trial_id = context.get("trial_id")
        side = context.get("inc_exc") or context.get("side")
        build_root = pathlib.Path(context.get("build_root", "../build"))
        if not (trial_id and side):
            _dbg("[CanonMiner] ERROR: Missing trial_id/side in context.")
            return None, "missing-trial/side"

        canon_path = build_root / "canon" / f"{trial_id}_{side}_canonical_variables.json"
        _dbg(f"[CanonMiner] Trying canonical path: {canon_path}  exists={canon_path.exists()}")
        if not canon_path.exists(): return None, str(canon_path)

        try:
            obj = json.loads(canon_path.read_text(encoding="utf-8"))
        except Exception as e:
            _dbg(f"[CanonMiner] ERROR reading canonical JSON: {e}")
            return None, str(canon_path)

        cvs = obj.get("canonical_variables", obj)
        if isinstance(cvs, list):
            _dbg(f"[CanonMiner] Loaded canonical_variables. count={len(cvs)}")
            return cvs, str(canon_path)

        _dbg("[CanonMiner] canonical JSON did not contain a list; treating as empty list.")
        return [], str(canon_path)

    def _repo_patient_paths(self, context: Dict[str, Any]) -> Tuple[pathlib.Path, pathlib.Path]:
        root = pathlib.Path(context.get("project_root", ".")).resolve()
        side = (context.get("inc_exc") or "inclusion").lower()
        pid = context.get("patient_id") or ""
        canonical_jsonl = root / f"patient_build_{side}" / f"patient_facts_export_{side}" / pid / side / "canonical.jsonl"
        demographics_jsonl = root / f"patient_build_{side}" / "patient_coded_results" / pid / "demographics.jsonl"
        return canonical_jsonl, demographics_jsonl

    def _collect_patient_facts(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        import sys
        def _dbg(*args):
            if context.get("VERBOSE"):
                print(*args, file=sys.stderr)

        canon_path, demo_path = self._repo_patient_paths(context)

        _dbg(f"[CanonMiner] Facts canonical.jsonl: {canon_path}  exists={canon_path.exists()}")
        for obj in _iter_jsonl(canon_path):
            if isinstance(obj, dict): facts.append(obj)

        _dbg(f"[CanonMiner] Facts demographics.jsonl: {demo_path}  exists={demo_path.exists()}")
        for obj in _iter_jsonl(demo_path):
            if isinstance(obj, dict): facts.append(obj)

        _dbg(f"[CanonMiner] Total facts loaded so far: {len(facts)}")

        if isinstance(context.get("patient_facts"), list):
            facts.extend(context["patient_facts"])
        pat = context.get("patient") or {}
        if isinstance(pat.get("facts"), list):
            facts.extend(pat["facts"])
        fj = pat.get("facts_jsonl") or context.get("patient_facts_jsonl")
        if isinstance(fj, str) and fj.strip():
            for line in fj.splitlines():
                line = line.strip()
                if not line: continue
                try: facts.append(json.loads(line))
                except Exception: pass
        return facts

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mb = get_mbench(context)
        cvs, src = self._load_canonical(context)  # REQUIRED file (may be empty list)
        if cvs is None:
            import sys
            print(f"[CanonMiner] ERROR: canonical_variables file missing/unreadable from {src}; aborting.", file=sys.stderr)
            raise FileNotFoundError("canonical_variables file missing or unreadable.")

        facts = self._collect_patient_facts(context)

        # limit to variables the SMT actually needs (leaf_detail) if present
        leaf_detail: Dict[str, Dict[str, str]] = context.get("leaf_detail") or {}
        leaf_vars = set(leaf_detail.keys()) if leaf_detail else None

        # Index facts by variable name
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for f in facts:
            v = f.get("entity_variable_name")
            if not isinstance(v, str): continue
            by_name.setdefault(v, []).append(f)

        out: Dict[str, Any] = {}

        for cv in cvs:
            var = cv.get("entity_variable_name")
            if not var or (leaf_vars is not None and var not in leaf_vars):
                continue

            timeframe = cv.get("timeframe", "now")
            RL, RR, dur = _parse_timeframe_to_hours(timeframe)

            vmeta = leaf_detail.get(var, {})
            vtype = (vmeta.get("type") or "").lower()

            cands = []
            for f in by_name.get(var, []):
                PL = float(f.get("start_time_in_hours", -math.inf))
                PR = float(f.get("end_time_in_hours",  math.inf))
                incL = bool(f.get("start_time_inclusive", True))
                incR = bool(f.get("end_time_inclusive",   True))

                if not _intervals_overlap(RL, RR, PL, PR, incL, incR): continue
                if dur is not None and not _duration_satisfied(PL, PR, incL, incR, dur): continue

                val = _scalar_value_from_fact(f, vtype)
                cands.append({**f, "_value": val})

            value, evidence = None, ""
            if (vtype in {"real", "float", "number", "int", "integer"}):
                best = _pick_best_numeric([c for c in cands if c.get("_value") is not None])
                if best:
                    value = best["_value"]
                    evidence = best.get("span_match") or best.get("preferred_term") or best.get("fact_id", "")
            else:
                truthy = [c for c in cands if c.get("_value") is True]
                if truthy:
                    value = True
                    latest = _pick_best_numeric(truthy) or truthy[-1]
                    evidence = latest.get("span_match") or latest.get("preferred_term") or latest.get("fact_id", "")
                else:
                    value = None
                    evidence = ""

            out[var] = {"value": value, "evidence": evidence}
            if context.get("VERBOSE"):
                import sys
                print(f"[CanonMiner] var={var} type={vtype} value={value} evidence={evidence}", file=sys.stderr)

        # Ensure SMT has entries for all leaf variables (canonical or not) → set missing to None
        if leaf_detail:
            for v in leaf_detail:
                out.setdefault(v, {"value": None, "evidence": ""})

        context["patient_var_values"] = out

        if mbench_enabled(context):
            mb.log_json("SMTCanonicalVariableMiner", "outputs", out)
        if context.get("VERBOSE"):
            import sys
            print(f"[CanonMiner] Final patient_var_values keys={len(out)}", file=sys.stderr)
        return context