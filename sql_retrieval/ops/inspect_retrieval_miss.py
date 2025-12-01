#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_retrieval_miss.py

Inspect why a (patient, trial) pair is NOT retrieved by:
  - disease_hits()
  - positive_literal_hits()
  - prevention_hits()  (optional)

Subcohort-aware:
  - If you pass an NCT like "NCT01234567" and --expand-subcohorts is enabled,
    it will inspect all trials whose trials.nct_id starts with that canonical NCT
    (e.g. NCT01234567a, NCT01234567b, ...), PLUS the exact match if present.

Root-gate diagnostics:
  - For hop>0 matches (accepted-alternatives / expanded tables),
    if --allow-hop-without-root is NOT set and the patient-side table has is_root,
    we report which matches were blocked because is_root!=1.

Defaults:
  --db defaults to ../../build/trial.db (so you can omit it)

Examples:
  python inspect_retrieval_miss.py --patient sigir-201418 --trial NCT01260259
  python inspect_retrieval_miss.py --patient sigir-201418 --trial NCT01260259 --important-mode chief
  python inspect_retrieval_miss.py --patient sigir-201418 --trial NCT01260259 --enable-prevention-hits
  python inspect_retrieval_miss.py --patient sigir-201418 --trial 12345

Output:
  - Prints a header JSON and then a per-trial report JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# -----------------------
# Regex / constants
# -----------------------

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CANON_NCT_RE = re.compile(r"^(NCT\d{8})", re.IGNORECASE)

# -----------------------
# Small SQL helpers
# -----------------------

def _safe_ident(name: str) -> str:
    name = (name or "").strip()
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name

def table_has(cur: sqlite3.Cursor, table: str) -> bool:
    return bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone())

def cols_of(cur: sqlite3.Cursor, table: str) -> List[str]:
    try:
        rows = cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()
    except sqlite3.Error:
        return []
    return [r[1] for r in rows]

def pick_col(cur: sqlite3.Cursor, table: str, candidates: Sequence[str]) -> Optional[str]:
    have = set(cols_of(cur, table))
    for c in candidates:
        if c in have:
            return c
    return None

def canon_nct(nct: str) -> str:
    m = _CANON_NCT_RE.match((nct or "").strip())
    return (m.group(1).upper() if m else (nct or "").strip().upper())

def important_table_for_mode(mode: str) -> str:
    mode = (mode or "all").strip()
    if mode not in {"chief", "ccr", "all"}:
        raise ValueError(f"bad important mode: {mode}")
    return f"patient_inclusion_constraints_important_{mode}"

def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) == 1.0
    except Exception:
        s = str(value).strip().upper()
        return s in {"1", "TRUE", "T", "Y", "YES"}

# -----------------------
# Trial resolution (subcohort aware)
# -----------------------

def resolve_trials(cur: sqlite3.Cursor, trial_arg: str, expand_subcohorts: bool) -> List[Dict[str, Any]]:
    """
    Returns list of {trial_id, nct_id} from trials table.

    If trial_arg is numeric => match trials.id
    If starts with NCT => match trials.nct_id exact; if expand_subcohorts, also prefix match canon NCT
    """
    trial_arg = (trial_arg or "").strip()
    out: List[Dict[str, Any]] = []

    if trial_arg.isdigit():
        row = cur.execute("SELECT id, nct_id FROM trials WHERE id=? LIMIT 1", (int(trial_arg),)).fetchone()
        if row:
            out.append({"trial_id": int(row[0]), "nct_id": row[1]})
        return out

    nct = trial_arg.upper()
    core = canon_nct(nct)

    # exact
    rows = cur.execute("SELECT id, nct_id FROM trials WHERE UPPER(nct_id)=UPPER(?)", (nct,)).fetchall()
    for tid, nid in rows:
        out.append({"trial_id": int(tid), "nct_id": nid})

    # prefix (subcohorts)
    if expand_subcohorts and core.startswith("NCT") and len(core) == 11:
        rows2 = cur.execute(
            "SELECT id, nct_id FROM trials WHERE UPPER(nct_id) LIKE UPPER(?) ORDER BY nct_id, id",
            (core + "%",),
        ).fetchall()
        seen = {x["trial_id"] for x in out}
        for tid, nid in rows2:
            if int(tid) not in seen:
                out.append({"trial_id": int(tid), "nct_id": nid})

    return out

# -----------------------
# Patient facts lookup
# -----------------------

def fetch_patient_facts(cur: sqlite3.Cursor, important_table: str, patient_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Map: base_var -> {truthy: bool, is_root: Optional[int], raw_value: Any}
    Only kind='bool' rows considered (matches your retrieval logic).
    """
    if not table_has(cur, important_table):
        return {}
    cols = set(cols_of(cur, important_table))
    has_is_root = "is_root" in cols

    sql = f"""
    SELECT base_var, value{", COALESCE(is_root,0) AS is_root" if has_is_root else ""}
    FROM {_safe_ident(important_table)}
    WHERE patient_id = :patient AND kind='bool'
    """
    rows = cur.execute(sql, {"patient": patient_id}).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        base_var = row[0]
        value = row[1]
        is_root = int(row[2]) if has_is_root else None
        truthy = _is_truthy(value)

        prev = out.get(base_var)
        if prev is None:
            out[base_var] = {"truthy": truthy, "is_root": is_root, "raw_value": value}
        else:
            # prefer truthy, then prefer is_root=1
            if (not prev["truthy"]) and truthy:
                out[base_var] = {"truthy": truthy, "is_root": is_root, "raw_value": value}
            elif prev["truthy"] == truthy and (prev.get("is_root") or 0) < (is_root or 0):
                out[base_var] = {"truthy": truthy, "is_root": is_root, "raw_value": value}

    return out

# -----------------------
# Source inspectors
# -----------------------

def inspect_disease_source(
    cur: sqlite3.Cursor,
    patient_facts: Dict[str, Dict[str, Any]],
    important_table: str,
    trial_nct: str,
    require_root_for_hops: bool,
) -> Dict[str, Any]:
    """
    Explain whether disease_* would hit this trial_nct for this patient_facts.
    """
    out: Dict[str, Any] = {"hit": False, "details": {}}

    if not table_has(cur, "disease_constraint_atoms") and not table_has(cur, "disease_constraint_alternatives"):
        out["details"]["note"] = "No disease tables present."
        return out

    # disease_constraint_atoms: exact subcohort match by trial key
    dli_vars: List[str] = []
    if table_has(cur, "disease_constraint_atoms"):
        dli_cols = set(cols_of(cur, "disease_constraint_atoms"))
        dli_trial_col = "trial_id" if "trial_id" in dli_cols else ("nct_id" if "nct_id" in dli_cols else None)
        dli_var_col = next((c for c in ["stem_var", "var_name", "base_var", "var_name_notime"] if c in dli_cols), None)
        if dli_trial_col and dli_var_col:
            rows = cur.execute(
                f"SELECT DISTINCT {dli_var_col} FROM disease_constraint_atoms WHERE {dli_trial_col}=?",
                (trial_nct,),
            ).fetchall()
            dli_vars = sorted({r[0] for r in rows if r and r[0] is not None})

    # disease_constraint_alternatives: hop/reason aware
    daa_items: List[Dict[str, Any]] = []
    if table_has(cur, "disease_constraint_alternatives"):
        daa_cols = set(cols_of(cur, "disease_constraint_alternatives"))
        daa_trial_col = "trial_id" if "trial_id" in daa_cols else ("nct_id" if "nct_id" in daa_cols else None)
        cand = [c for c in ["alt_var_name_notime", "stem_var", "alt_var_name", "var_name_notime", "var_name"] if c in daa_cols]
        hop_col = "hop" if "hop" in daa_cols else None
        reason_col = "reason" if "reason" in daa_cols else None
        if daa_trial_col and cand:
            coalesce = "COALESCE(" + ", ".join(cand) + ")"
            sql = f"""
            SELECT DISTINCT
              {coalesce} AS base_var,
              COALESCE({hop_col},0) AS hop,
              COALESCE({reason_col},'self') AS reason
            FROM disease_constraint_alternatives
            WHERE {daa_trial_col} = ?
              AND {coalesce} IS NOT NULL
            """
            rows = cur.execute(sql, (trial_nct,)).fetchall()
            for base_var, hop, reason in rows:
                daa_items.append({"base_var": base_var, "hop": int(hop or 0), "reason": str(reason or "self")})

    def eval_plain(vars_: List[str]) -> Dict[str, List[str]]:
        matched, missing = [], []
        for v in vars_:
            pf = patient_facts.get(v)
            if pf and pf.get("truthy"):
                matched.append(v)
            else:
                missing.append(v)
        return {"matched": sorted(set(matched)), "missing": sorted(set(missing))}

    dli_eval = eval_plain(dli_vars)

    daa_matched, daa_missing, daa_blocked = [], [], []
    for it in daa_items:
        v = it["base_var"]
        hop = int(it.get("hop", 0))
        reason = str(it.get("reason", "self"))

        pf = patient_facts.get(v)
        if not pf or not pf.get("truthy"):
            daa_missing.append(v)
            continue

        if require_root_for_hops and hop > 0 and reason != "self":
            is_root = int(pf.get("is_root") or 0)
            if is_root != 1:
                daa_blocked.append(v)
                continue

        daa_matched.append(v)

    out["details"] = {
        "trial_nct_id": trial_nct,
        "dli_vars": dli_vars,
        "dli_matched_truthy": dli_eval["matched"],
        "dli_missing_or_not_truthy": dli_eval["missing"],
        "daa_items": daa_items,
        "daa_matched_truthy_and_passed_root_gate": sorted(set(daa_matched)),
        "daa_missing_or_not_truthy": sorted(set(daa_missing)),
        "daa_blocked_by_root_gate": sorted(set(daa_blocked)),
    }
    out["hit"] = (len(dli_eval["matched"]) > 0) or (len(daa_matched) > 0)
    return out

def inspect_positive_literal_source(
    cur: sqlite3.Cursor,
    patient_facts: Dict[str, Dict[str, Any]],
    important_table: str,
    trial_nct: str,
    require_root_for_hops: bool,
) -> Dict[str, Any]:
    """
    Explain whether positive_literal_hits would hit this trial_nct for this patient_facts.
    """
    out: Dict[str, Any] = {"hit": False, "details": {}}

    has_pl = table_has(cur, "positive_constraint_literals")
    has_pla = table_has(cur, "positive_constraint_alternatives")
    if not has_pl and not has_pla:
        out["details"]["note"] = "No positive literal tables present."
        return out

    def stem_key_expr_for(table: str) -> Optional[str]:
        cols = set(cols_of(cur, table))
        if table == "positive_constraint_alternatives":
            priority = ["lifted_var_stem", "lifted_var", "base_var_stem", "var_name_notime", "base_var", "var_name"]
        else:
            priority = ["base_var_stem", "var_name_notime", "base_var", "var_name", "lifted_var_stem", "lifted_var"]
        present = [c for c in priority if c in cols]
        if not present:
            return None
        return "COALESCE(" + ", ".join(present) + ")"

    def trial_key_col(table: str) -> Optional[str]:
        cols = set(cols_of(cur, table))
        if "nct_id" in cols:
            return "nct_id"
        if "trial_id" in cols:
            return "trial_id"
        return None

    def hop_col(table: str) -> Optional[str]:
        return "hop" if "hop" in set(cols_of(cur, table)) else None

    def reason_col(table: str) -> Optional[str]:
        return "reason" if "reason" in set(cols_of(cur, table)) else None

    src_items: List[Dict[str, Any]] = []

    for table in ["positive_constraint_literals", "positive_constraint_alternatives"]:
        if not table_has(cur, table):
            continue
        tcol = trial_key_col(table)
        sk = stem_key_expr_for(table)
        if not tcol or not sk:
            continue
        hc = hop_col(table)
        rc = reason_col(table)
        hop_expr = f"COALESCE({hc},0)" if hc else "0"
        reason_expr = f"COALESCE({rc},'self')" if rc else "'self'"

        sql = f"""
        SELECT DISTINCT {sk} AS stem_key, {hop_expr} AS hop, {reason_expr} AS reason
        FROM {table}
        WHERE {tcol} = ?
          AND {sk} IS NOT NULL
        """
        rows = cur.execute(sql, (trial_nct,)).fetchall()
        for stem_key, hop, reason in rows:
            src_items.append({
                "table": table,
                "stem_key": stem_key,
                "hop": int(hop or 0),
                "reason": str(reason or "self"),
            })

    matched, missing, blocked = [], [], []
    for it in src_items:
        v = it["stem_key"]
        hop = int(it["hop"])
        reason = str(it["reason"])

        pf = patient_facts.get(v)
        if not pf or not pf.get("truthy"):
            missing.append(v)
            continue

        if require_root_for_hops and hop > 0 and reason != "self":
            is_root = int(pf.get("is_root") or 0)
            if is_root != 1:
                blocked.append(v)
                continue

        matched.append(v)

    out["details"] = {
        "trial_nct_id": trial_nct,
        "src_items": src_items,
        "matched_truthy_and_passed_root_gate": sorted(set(matched)),
        "missing_or_not_truthy": sorted(set(missing)),
        "blocked_by_root_gate": sorted(set(blocked)),
    }
    out["hit"] = len(matched) > 0
    return out

def inspect_prevention_source(
    cur: sqlite3.Cursor,
    patient_id: str,
    trial_nct: str,
    require_root_for_hops: bool,
) -> Dict[str, Any]:
    """
    Explain whether prevention_hits would hit this trial_nct for this patient.
    Tables (as in your primitives):
      - patient_prevention_constraints
      - disease_constraint_alternatives_prevent
      - positive_constraint_alternatives_expanded_prevention
    """
    out: Dict[str, Any] = {"hit": False, "details": {}}

    patient_table = "patient_prevention_constraints"
    t1 = "disease_constraint_alternatives_prevent"
    t2 = "positive_constraint_alternatives_expanded_prevention"

    if not table_has(cur, patient_table):
        out["details"]["note"] = f"Missing patient prevention table {patient_table}."
        return out
    if not table_has(cur, t1) and not table_has(cur, t2):
        out["details"]["note"] = f"Missing both prevention trial tables ({t1}, {t2})."
        return out

    pcols = set(cols_of(cur, patient_table))
    p_entity = next((c for c in ["entity_var", "base_var", "var_name", "var_name_notime"] if c in pcols), None)
    p_kind = "kind" if "kind" in pcols else None
    p_val = "value" if "value" in pcols else None
    p_is_root = "is_root" if "is_root" in pcols else None
    if not p_entity or not p_val:
        out["details"]["note"] = f"{patient_table} missing entity/value columns."
        return out

    kind_guard = f"AND {p_kind}='bool'" if p_kind else ""
    psql = f"""
    SELECT DISTINCT {p_entity} AS key, value, COALESCE({p_is_root},0) AS is_root
    FROM {patient_table}
    WHERE patient_id = :patient {kind_guard}
    """
    prow = cur.execute(psql, {"patient": patient_id}).fetchall()

    patient_keys: Dict[str, Dict[str, Any]] = {}
    for key, value, is_root in prow:
        patient_keys[key] = {"truthy": _is_truthy(value), "is_root": int(is_root or 0), "raw_value": value}

    def inspect_trial_table(table: str, key_priority: List[str]) -> List[Dict[str, Any]]:
        if not table_has(cur, table):
            return []
        tcols = set(cols_of(cur, table))
        nct_col = "nct_id" if "nct_id" in tcols else ("trial_id" if "trial_id" in tcols else None)
        if not nct_col:
            return []
        present = [c for c in key_priority if c in tcols]
        if not present:
            return []
        key_expr = "COALESCE(" + ", ".join(present) + ")"
        hop_col = "hop" if "hop" in tcols else None
        reason_col = "reason" if "reason" in tcols else None
        hop_expr = f"COALESCE({hop_col},0)" if hop_col else "0"
        reason_expr = f"COALESCE({reason_col},'self')" if reason_col else "'self'"

        sql = f"""
        SELECT DISTINCT {key_expr} AS key, {hop_expr} AS hop, {reason_expr} AS reason
        FROM {table}
        WHERE {nct_col} = ?
          AND {key_expr} IS NOT NULL
        """
        rows = cur.execute(sql, (trial_nct,)).fetchall()
        return [{"table": table, "key": key, "hop": int(hop or 0), "reason": str(reason or "self")} for key, hop, reason in rows]

    t1_items = inspect_trial_table(t1, ["alt_var_name_notime", "stem_var", "var_name_notime", "alt_var_name", "var_name"])
    t2_items = inspect_trial_table(t2, ["lifted_var_stem", "base_var_stem", "lifted_var", "base_var"])
    src_items = t1_items + t2_items

    matched, missing, blocked = [], [], []
    for it in src_items:
        key = it["key"]
        hop = int(it["hop"])
        reason = str(it["reason"])

        pf = patient_keys.get(key)
        if not pf or not pf.get("truthy"):
            missing.append(key)
            continue

        if require_root_for_hops and hop > 0 and reason != "self":
            if int(pf.get("is_root") or 0) != 1:
                blocked.append(key)
                continue

        matched.append(key)

    out["details"] = {
        "trial_nct_id": trial_nct,
        "patient_truthy_keys_count": sum(1 for v in patient_keys.values() if v.get("truthy")),
        "src_items": src_items,
        "matched_truthy_and_passed_root_gate": sorted(set(matched)),
        "missing_or_not_truthy": sorted(set(missing)),
        "blocked_by_root_gate": sorted(set(blocked)),
    }
    out["hit"] = len(matched) > 0
    return out

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Inspect why a (patient, trial) pair is not retrieved (subcohort-aware).")
    ap.add_argument("--db", type=Path, default=Path("../../build/trial.db"))
    ap.add_argument("--patient", required=True)
    ap.add_argument("--trial", required=True, help="Either trials.id (numeric) or NCT... (e.g., NCT01234567 or NCT01234567a)")
    ap.add_argument("--important-mode", choices=["chief", "ccr", "all"], default="all")
    ap.add_argument("--allow-hop-without-root", action="store_true", default=False)
    ap.add_argument("--enable-prevention-hits", action="store_true", default=False)
    ap.add_argument("--expand-subcohorts", action="store_true", default=True,
                    help="If trial is canonical NCT########, also inspect NCT########a/b/c... variants.")
    ap.add_argument("--max-show", type=int, default=50, help="Max list entries to show per field (defensive truncation).")
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    cur = conn.cursor()

    important_table = important_table_for_mode(args.important_mode)
    require_root_for_hops = not args.allow_hop_without_root

    if not table_has(cur, "trials"):
        print("[error] missing trials table", file=sys.stderr)
        sys.exit(2)

    trials = resolve_trials(cur, args.trial, expand_subcohorts=args.expand_subcohorts)
    if not trials:
        print(json.dumps({
            "patient_id": args.patient,
            "trial_arg": args.trial,
            "error": "trial not found in trials table (by id or nct_id/prefix)",
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    patient_facts = fetch_patient_facts(cur, important_table, args.patient)

    header = {
        "patient_id": args.patient,
        "important_mode": args.important_mode,
        "important_table": important_table,
        "require_root_for_hops": require_root_for_hops,
        "enable_prevention_hits": args.enable_prevention_hits,
        "trial_arg": args.trial,
        "n_resolved_trials": len(trials),
        "resolved_trials": trials,
        "patient_fact_count_bool": len(patient_facts),
        "patient_truthy_fact_count_bool": sum(1 for v in patient_facts.values() if v.get("truthy")),
    }
    print(f"# inspecting patient={args.patient} trial={args.trial} -> {len(trials)} resolved trial(s)")
    print(json.dumps(header, ensure_ascii=False, indent=2))

    reports = []

    def trunc_obj(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: trunc_obj(v) for k, v in o.items()}
        if isinstance(o, list):
            if len(o) > args.max_show:
                return [trunc_obj(v) for v in o[:args.max_show]] + [f"... ({len(o)-args.max_show} more)"]
            return [trunc_obj(v) for v in o]
        return o

    for t in trials:
        tid = int(t["trial_id"])
        nct = t.get("nct_id") or ""

        disease = inspect_disease_source(cur, patient_facts, important_table, nct, require_root_for_hops)
        poslit = inspect_positive_literal_source(cur, patient_facts, important_table, nct, require_root_for_hops)
        prevent = (
            inspect_prevention_source(cur, args.patient, nct, require_root_for_hops)
            if args.enable_prevention_hits
            else {"hit": False, "details": {"note": "disabled"}}
        )

        report = {
            "merged_trial_id": tid,
            "nct_id": nct,
            "hits": {
                "disease": bool(disease["hit"]),
                "positive_literal": bool(poslit["hit"]),
                "prevention": bool(prevent["hit"]) if args.enable_prevention_hits else False,
                "overall_union": bool(disease["hit"] or poslit["hit"] or (prevent["hit"] if args.enable_prevention_hits else False)),
            },
            "disease": trunc_obj(disease["details"]),
            "positive_literal": trunc_obj(poslit["details"]),
            "prevention": trunc_obj(prevent["details"]) if args.enable_prevention_hits else {"note": "disabled"},
        }
        reports.append(report)

    print("\n# per-trial report")
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))

    conn.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)