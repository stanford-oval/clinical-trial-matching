#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_nonact_all_why_not_hit.py

Debug WHY a target trial is or is not retrieved under:
  important_mode = all
  alt_mode       = nonact

It inspects retrieval only:
  - disease_hits(...)
  - positive_literal_hits(...)
  - prevention_hits(...)

For one patient + one trial, it reports:
  1) hit membership
  2) trial-side candidate vars from relevant NONACT tables
  3) patient-side matching facts
  4) key probing for each table:
       - which key columns exist
       - which actual key values produce rows
  5) likely miss reasons

This is intended to answer:
  "I know the needed rows exist somewhere; which key is missing / mismatched?"
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sql_retrieval.ops import constraint_primitives as tep


# ============================================================
# Basic helpers
# ============================================================

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_ident(name: str) -> str:
    name = (name or "").strip()
    if not IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def table_has(cur: sqlite3.Cursor, table: str) -> bool:
    return bool(
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
    )


def table_cols(cur: sqlite3.Cursor, table: str) -> List[str]:
    if not table_has(cur, table):
        return []
    return [r[1] for r in cur.execute(f"PRAGMA table_info({safe_ident(table)})").fetchall()]


def pick_col(cur: sqlite3.Cursor, table: str, candidates: List[str]) -> Optional[str]:
    cols = set(table_cols(cur, table))
    for c in candidates:
        if c in cols:
            return c
    return None


def truthy_value(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip().upper()
    if s in {"1", "TRUE", "T", "Y", "YES"}:
        return True
    try:
        return float(v) == 1.0
    except Exception:
        return False


def canon_nct(nct: Optional[str]) -> Optional[str]:
    if not nct:
        return None
    nct = nct.strip().upper()
    m = re.match(r"^(NCT\d{8})", nct)
    return m.group(1) if m else nct


def unique_dict_rows(rows: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ============================================================
# Trial lookup
# ============================================================

def resolve_trial(conn: sqlite3.Connection, trial: str) -> Dict[str, Any]:
    cur = conn.cursor()

    if trial.isdigit():
        row = cur.execute(
            """
            SELECT id, nct_id, inclusion_trial_side_id, assumed_trial_side_id, exclusion_trial_side_id
            FROM trials
            WHERE id=?
            LIMIT 1
            """,
            (int(trial),),
        ).fetchone()
        if row:
            return {
                "merged_trial_id": int(row[0]),
                "nct_id": row[1],
                "canonical_nct_id": canon_nct(row[1]),
                "inclusion_trial_side_id": row[2],
                "assumed_trial_side_id": row[3],
                "exclusion_trial_side_id": row[4],
            }

    trial_up = trial.strip().upper()
    row = cur.execute(
        """
        SELECT id, nct_id, inclusion_trial_side_id, assumed_trial_side_id, exclusion_trial_side_id
        FROM trials
        WHERE UPPER(nct_id)=?
        LIMIT 1
        """,
        (trial_up,),
    ).fetchone()
    if row:
        return {
            "merged_trial_id": int(row[0]),
            "nct_id": row[1],
            "canonical_nct_id": canon_nct(row[1]),
            "inclusion_trial_side_id": row[2],
            "assumed_trial_side_id": row[3],
            "exclusion_trial_side_id": row[4],
        }

    core = canon_nct(trial_up)
    row = cur.execute(
        """
        SELECT id, nct_id, inclusion_trial_side_id, assumed_trial_side_id, exclusion_trial_side_id
        FROM trials
        WHERE UPPER(SUBSTR(nct_id,1,11))=?
        ORDER BY id
        LIMIT 1
        """,
        (core,),
    ).fetchone()
    if row:
        return {
            "merged_trial_id": int(row[0]),
            "nct_id": row[1],
            "canonical_nct_id": canon_nct(row[1]),
            "inclusion_trial_side_id": row[2],
            "assumed_trial_side_id": row[3],
            "exclusion_trial_side_id": row[4],
        }

    raise RuntimeError(f"Could not resolve trial: {trial}")


# ============================================================
# Patient fact helpers
# ============================================================

def fetch_patient_bool_facts(
    conn: sqlite3.Connection,
    patient_id: str,
    table: str,
    base_var_col: str = "base_var",
) -> Dict[str, List[Dict[str, Any]]]:
    cur = conn.cursor()
    if not table_has(cur, table):
        return {}

    cols = set(table_cols(cur, table))
    if base_var_col not in cols:
        return {}

    kind_col = "kind" if "kind" in cols else None
    value_col = "value" if "value" in cols else None
    is_root_col = "is_root" if "is_root" in cols else None

    where = ["patient_id = ?"]
    params: List[Any] = [patient_id]
    if kind_col:
        where.append(f"{kind_col} = 'bool'")

    sql = f"""
        SELECT *
        FROM {safe_ident(table)}
        WHERE {" AND ".join(where)}
    """
    rows = cur.execute(sql, params).fetchall()
    names = [d[0] for d in cur.description]

    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rec = dict(zip(names, row))
        key = rec.get(base_var_col)
        if key is None:
            continue
        rec["_norm_var"] = key
        rec["_truthy"] = truthy_value(rec.get(value_col)) if value_col else None
        rec["_is_root"] = int(rec.get(is_root_col) or 0) if is_root_col else 0
        out.setdefault(str(key), []).append(rec)
    return out


def summarize_patient_var(facts_by_var: Dict[str, List[Dict[str, Any]]], var: str) -> Dict[str, Any]:
    rows = facts_by_var.get(var, [])
    if not rows:
        return {
            "exists": False,
            "n_rows": 0,
            "any_truthy": False,
            "any_root": False,
            "rows": [],
        }
    return {
        "exists": True,
        "n_rows": len(rows),
        "any_truthy": any(bool(r.get("_truthy")) for r in rows),
        "any_root": any(int(r.get("_is_root", 0)) == 1 for r in rows),
        "rows": rows,
    }


# ============================================================
# Key probing
# ============================================================

def get_trial_probe_values(trial_info: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [
        ("merged_trial_id", trial_info.get("merged_trial_id")),
        ("nct_id", trial_info.get("nct_id")),
        ("canonical_nct_id", trial_info.get("canonical_nct_id")),
        ("inclusion_trial_side_id", trial_info.get("inclusion_trial_side_id")),
        ("assumed_trial_side_id", trial_info.get("assumed_trial_side_id")),
        ("exclusion_trial_side_id", trial_info.get("exclusion_trial_side_id")),
    ]


def debug_trial_table_keys(conn: sqlite3.Connection, trial_info: Dict[str, Any]) -> Dict[str, Any]:
    cur = conn.cursor()

    tables = [
        "disease_constraint_atoms",
        "disease_constraint_alternatives_nonact",
        "positive_constraint_literals",
        "positive_constraint_alternatives_expanded_nonact",
        "disease_constraint_alternatives_prevent_nonact",
        "positive_constraint_alternatives_expanded_prevention_nonact",
    ]

    probes = get_trial_probe_values(trial_info)
    out: Dict[str, Any] = {}

    for table in tables:
        if not table_has(cur, table):
            out[table] = {"exists": False}
            continue

        cols = table_cols(cur, table)
        candidate_key_cols = [c for c in ["trial_id", "nct_id", "trial_side_id", "side_id"] if c in cols]

        info: Dict[str, Any] = {
            "exists": True,
            "columns": cols,
            "matches": {},
        }

        for key_col in candidate_key_cols:
            key_matches: Dict[str, Any] = {}
            for label, value in probes:
                if value is None:
                    continue
                try:
                    row = cur.execute(
                        f"SELECT COUNT(*) FROM {safe_ident(table)} WHERE {safe_ident(key_col)} = ?",
                        (value,),
                    ).fetchone()
                    key_matches[label] = int(row[0] or 0)
                except Exception as e:
                    key_matches[label] = f"ERROR: {e}"
            info["matches"][key_col] = key_matches

        out[table] = info

    return out


def pick_best_key_for_table(
    conn: sqlite3.Connection,
    table: str,
    trial_info: Dict[str, Any],
) -> Optional[Tuple[str, str, Any, int]]:
    """
    Returns best matching key as:
      (table_key_col, probe_label, probe_value, count)
    """
    cur = conn.cursor()
    if not table_has(cur, table):
        return None

    cols = table_cols(cur, table)
    candidate_key_cols = [c for c in ["nct_id", "trial_id", "trial_side_id", "side_id"] if c in cols]
    probes = get_trial_probe_values(trial_info)

    best: Optional[Tuple[str, str, Any, int]] = None
    for key_col in candidate_key_cols:
        for probe_label, probe_value in probes:
            if probe_value is None:
                continue
            try:
                row = cur.execute(
                    f"SELECT COUNT(*) FROM {safe_ident(table)} WHERE {safe_ident(key_col)} = ?",
                    (probe_value,),
                ).fetchone()
                count = int(row[0] or 0)
            except Exception:
                continue
            if count <= 0:
                continue
            cand = (key_col, probe_label, probe_value, count)
            if best is None or count > best[3]:
                best = cand
    return best


# ============================================================
# Generic row fetcher
# ============================================================

def sample_rows_for_trial_key(
    conn: sqlite3.Connection,
    table: str,
    key_col: str,
    key_val: Any,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    if not table_has(cur, table):
        return []
    sql = f"""
        SELECT *
        FROM {safe_ident(table)}
        WHERE {safe_ident(key_col)} = ?
        LIMIT {int(limit)}
    """
    rows = cur.execute(sql, (key_val,)).fetchall()
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in rows]


# ============================================================
# Trial-side source extraction with auto key detection
# ============================================================

def get_disease_sources_for_trial(
    conn: sqlite3.Connection,
    trial_info: Dict[str, Any],
) -> Dict[str, Any]:
    cur = conn.cursor()
    out: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {}

    # disease_constraint_atoms
    table = "disease_constraint_atoms"
    if table_has(cur, table):
        var_col = pick_col(cur, table, ["stem_var", "var_name", "base_var", "var_name_notime"])
        hop_col = pick_col(cur, table, ["hop"])
        best = pick_best_key_for_table(conn, table, trial_info)
        debug[table] = {
            "best_key": {
                "key_col": best[0],
                "probe_label": best[1],
                "probe_value": best[2],
                "count": best[3],
            } if best else None
        }
        if best and var_col:
            key_col, probe_label, probe_value, count = best
            hop_expr = hop_col if hop_col else "0"
            sql = f"""
                SELECT DISTINCT
                    {var_col} AS base_var,
                    {hop_expr} AS hop,
                    'self' AS reason,
                    '{table}' AS source_table,
                    '{key_col}' AS matched_key_col,
                    ? AS matched_probe_label,
                    ? AS matched_probe_value
                FROM {safe_ident(table)}
                WHERE {safe_ident(key_col)} = ?
                ORDER BY 1
            """
            for base_var, hop, reason, source_table, matched_key_col, mpl, mpv in cur.execute(
                sql, (probe_label, str(probe_value), probe_value)
            ).fetchall():
                out.append({
                    "base_var": base_var,
                    "hop": int(hop or 0),
                    "reason": reason,
                    "source_table": source_table,
                    "source_kind": "disease",
                    "matched_key_col": matched_key_col,
                    "matched_probe_label": mpl,
                    "matched_probe_value": mpv,
                })

    # disease_constraint_alternatives_nonact
    table = "disease_constraint_alternatives_nonact"
    if table_has(cur, table):
        var_col = pick_col(cur, table, ["alt_var_name_notime", "stem_var", "var_name_notime", "alt_var_name", "var_name"])
        hop_col = pick_col(cur, table, ["hop"])
        reason_col = pick_col(cur, table, ["reason"])
        best = pick_best_key_for_table(conn, table, trial_info)
        debug[table] = {
            "best_key": {
                "key_col": best[0],
                "probe_label": best[1],
                "probe_value": best[2],
                "count": best[3],
            } if best else None
        }
        if best and var_col:
            key_col, probe_label, probe_value, count = best
            hop_expr = hop_col if hop_col else "0"
            reason_expr = reason_col if reason_col else "'self'"
            sql = f"""
                SELECT DISTINCT
                    {var_col} AS base_var,
                    {hop_expr} AS hop,
                    {reason_expr} AS reason,
                    '{table}' AS source_table,
                    '{key_col}' AS matched_key_col,
                    ? AS matched_probe_label,
                    ? AS matched_probe_value
                FROM {safe_ident(table)}
                WHERE {safe_ident(key_col)} = ?
                ORDER BY 1
            """
            for base_var, hop, reason, source_table, matched_key_col, mpl, mpv in cur.execute(
                sql, (probe_label, str(probe_value), probe_value)
            ).fetchall():
                out.append({
                    "base_var": base_var,
                    "hop": int(hop or 0),
                    "reason": reason,
                    "source_table": source_table,
                    "source_kind": "disease",
                    "matched_key_col": matched_key_col,
                    "matched_probe_label": mpl,
                    "matched_probe_value": mpv,
                })

    out = unique_dict_rows(
        out,
        ["base_var", "hop", "reason", "source_table", "source_kind", "matched_key_col", "matched_probe_label", "matched_probe_value"],
    )
    out.sort(key=lambda x: (
        str(x["source_table"]),
        str(x["base_var"]),
        int(x["hop"]),
        str(x["reason"]),
    ))
    return {"rows": out, "debug": debug}


def get_positive_sources_for_trial(
    conn: sqlite3.Connection,
    trial_info: Dict[str, Any],
) -> Dict[str, Any]:
    cur = conn.cursor()
    out: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {}

    for table in [
        "positive_constraint_literals",
        "positive_constraint_alternatives_expanded_nonact",
    ]:
        if not table_has(cur, table):
            continue

        cols = set(table_cols(cur, table))
        stem_priority = [
            "lifted_var_stem",
            "lifted_var",
            "base_var_stem",
            "var_name_notime",
            "base_var",
            "var_name",
        ]
        present = [c for c in stem_priority if c in cols]
        if not present:
            debug[table] = {"best_key": None, "note": "no usable var column"}
            continue

        stem_expr = "COALESCE(" + ", ".join(present) + ")"
        hop_col = pick_col(cur, table, ["hop"])
        reason_col = pick_col(cur, table, ["reason"])
        best = pick_best_key_for_table(conn, table, trial_info)

        debug[table] = {
            "best_key": {
                "key_col": best[0],
                "probe_label": best[1],
                "probe_value": best[2],
                "count": best[3],
            } if best else None
        }

        if not best:
            continue

        key_col, probe_label, probe_value, count = best
        hop_expr = hop_col if hop_col else "0"
        reason_expr = reason_col if reason_col else "'self'"

        sql = f"""
            SELECT DISTINCT
                {stem_expr} AS base_var,
                {hop_expr} AS hop,
                {reason_expr} AS reason,
                '{table}' AS source_table,
                '{key_col}' AS matched_key_col,
                ? AS matched_probe_label,
                ? AS matched_probe_value
            FROM {safe_ident(table)}
            WHERE {safe_ident(key_col)} = ?
            ORDER BY 1
        """
        for base_var, hop, reason, source_table, matched_key_col, mpl, mpv in cur.execute(
            sql, (probe_label, str(probe_value), probe_value)
        ).fetchall():
            out.append({
                "base_var": base_var,
                "hop": int(hop or 0),
                "reason": reason,
                "source_table": source_table,
                "source_kind": "positive_literal",
                "matched_key_col": matched_key_col,
                "matched_probe_label": mpl,
                "matched_probe_value": mpv,
            })

    out = unique_dict_rows(
        out,
        ["base_var", "hop", "reason", "source_table", "source_kind", "matched_key_col", "matched_probe_label", "matched_probe_value"],
    )
    out.sort(key=lambda x: (
        str(x["source_table"]),
        str(x["base_var"]),
        int(x["hop"]),
        str(x["reason"]),
    ))
    return {"rows": out, "debug": debug}


def get_prevention_sources_for_trial(
    conn: sqlite3.Connection,
    trial_info: Dict[str, Any],
) -> Dict[str, Any]:
    cur = conn.cursor()
    out: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {}

    # disease prevention
    table = "disease_constraint_alternatives_prevent_nonact"
    if table_has(cur, table):
        var_col = pick_col(cur, table, ["alt_var_name_notime", "stem_var", "var_name_notime", "alt_var_name", "var_name"])
        hop_col = pick_col(cur, table, ["hop"])
        reason_col = pick_col(cur, table, ["reason"])
        best = pick_best_key_for_table(conn, table, trial_info)
        debug[table] = {
            "best_key": {
                "key_col": best[0],
                "probe_label": best[1],
                "probe_value": best[2],
                "count": best[3],
            } if best else None
        }
        if best and var_col:
            key_col, probe_label, probe_value, count = best
            hop_expr = hop_col if hop_col else "0"
            reason_expr = reason_col if reason_col else "'self'"
            sql = f"""
                SELECT DISTINCT
                    {var_col} AS base_var,
                    {hop_expr} AS hop,
                    {reason_expr} AS reason,
                    '{table}' AS source_table,
                    '{key_col}' AS matched_key_col,
                    ? AS matched_probe_label,
                    ? AS matched_probe_value
                FROM {safe_ident(table)}
                WHERE {safe_ident(key_col)} = ?
                ORDER BY 1
            """
            for base_var, hop, reason, source_table, matched_key_col, mpl, mpv in cur.execute(
                sql, (probe_label, str(probe_value), probe_value)
            ).fetchall():
                out.append({
                    "base_var": base_var,
                    "hop": int(hop or 0),
                    "reason": reason,
                    "source_table": source_table,
                    "source_kind": "prevention_disease",
                    "matched_key_col": matched_key_col,
                    "matched_probe_label": mpl,
                    "matched_probe_value": mpv,
                })

    # prevention positive
    table = "positive_constraint_alternatives_expanded_prevention_nonact"
    if table_has(cur, table):
        cols = set(table_cols(cur, table))
        present = [c for c in ["lifted_var_stem", "base_var_stem", "lifted_var", "base_var"] if c in cols]
        hop_col = pick_col(cur, table, ["hop"])
        reason_col = pick_col(cur, table, ["reason"])
        best = pick_best_key_for_table(conn, table, trial_info)
        debug[table] = {
            "best_key": {
                "key_col": best[0],
                "probe_label": best[1],
                "probe_value": best[2],
                "count": best[3],
            } if best else None
        }
        if best and present:
            stem_expr = "COALESCE(" + ", ".join(present) + ")"
            key_col, probe_label, probe_value, count = best
            hop_expr = hop_col if hop_col else "0"
            reason_expr = reason_col if reason_col else "'self'"
            sql = f"""
                SELECT DISTINCT
                    {stem_expr} AS base_var,
                    {hop_expr} AS hop,
                    {reason_expr} AS reason,
                    '{table}' AS source_table,
                    '{key_col}' AS matched_key_col,
                    ? AS matched_probe_label,
                    ? AS matched_probe_value
                FROM {safe_ident(table)}
                WHERE {safe_ident(key_col)} = ?
                ORDER BY 1
            """
            for base_var, hop, reason, source_table, matched_key_col, mpl, mpv in cur.execute(
                sql, (probe_label, str(probe_value), probe_value)
            ).fetchall():
                out.append({
                    "base_var": base_var,
                    "hop": int(hop or 0),
                    "reason": reason,
                    "source_table": source_table,
                    "source_kind": "prevention_positive_literal",
                    "matched_key_col": matched_key_col,
                    "matched_probe_label": mpl,
                    "matched_probe_value": mpv,
                })

    out = unique_dict_rows(
        out,
        ["base_var", "hop", "reason", "source_table", "source_kind", "matched_key_col", "matched_probe_label", "matched_probe_value"],
    )
    out.sort(key=lambda x: (
        str(x["source_table"]),
        str(x["base_var"]),
        int(x["hop"]),
        str(x["reason"]),
    ))
    return {"rows": out, "debug": debug}


# ============================================================
# Explanation logic
# ============================================================

def explain_source_rows(
    rows: List[Dict[str, Any]],
    important_all_facts: Dict[str, List[Dict[str, Any]]],
    root_facts: Dict[str, List[Dict[str, Any]]],
    prevention_facts: Dict[str, List[Dict[str, Any]]],
    source_family: str,
    require_root_for_hops: bool,
) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        var = r["base_var"]
        hop = int(r.get("hop", 0) or 0)
        reason = r.get("reason", "self")

        if source_family in {"disease", "positive_literal"}:
            main = summarize_patient_var(important_all_facts, var)
            root = summarize_patient_var(root_facts, var)
            fact_source = "patient_inclusion_constraints_important_all"
        elif source_family == "prevention":
            main = summarize_patient_var(prevention_facts, var)
            root = summarize_patient_var(prevention_facts, var)
            fact_source = "patient_prevention_constraints"
        else:
            raise ValueError(source_family)

        needs_root = require_root_for_hops and hop > 0 and reason != "self"

        if not main["exists"]:
            would_match = False
            failure = "no_patient_fact_for_var"
        elif not main["any_truthy"]:
            would_match = False
            failure = "patient_fact_exists_but_not_truthy"
        elif needs_root and not root["any_root"]:
            would_match = False
            failure = "hop_requires_root_but_patient_var_not_root"
        else:
            would_match = True
            failure = None

        out.append({
            **r,
            "fact_source": fact_source,
            "patient_fact_exists": main["exists"],
            "patient_fact_truthy": main["any_truthy"],
            "patient_fact_root": root["any_root"],
            "needs_root_gate": needs_root,
            "would_match": would_match,
            "failure_reason": failure,
            "patient_rows": main["rows"],
        })

    return out


def build_summary(
    disease_expl: List[Dict[str, Any]],
    positive_expl: List[Dict[str, Any]],
    prevention_expl: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        would = [r for r in rows if r["would_match"]]
        miss = [r for r in rows if not r["would_match"]]
        by_reason: Dict[str, int] = {}
        by_key: Dict[str, int] = {}
        for r in rows:
            key = f"{r.get('source_table')}::{r.get('matched_key_col')}={r.get('matched_probe_label')}"
            by_key[key] = by_key.get(key, 0) + 1
        for r in miss:
            key = r["failure_reason"] or "unknown"
            by_reason[key] = by_reason.get(key, 0) + 1
        return {
            "n_candidate_vars": len(rows),
            "n_would_match": len(would),
            "n_would_not_match": len(miss),
            "miss_reasons": by_reason,
            "matching_vars": sorted({r["base_var"] for r in would}),
            "matched_key_usage": by_key,
        }

    return {
        "disease": summarize(disease_expl),
        "positive_literal": summarize(positive_expl),
        "prevention": summarize(prevention_expl),
    }


# ============================================================
# Inspection
# ============================================================

def inspect_nonact_all(
    conn: sqlite3.Connection,
    patient_id: str,
    trial: str,
    require_root_for_hops: bool = True,
) -> Dict[str, Any]:
    cur = conn.cursor()
    trial_info = resolve_trial(conn, trial)
    merged_trial_id = trial_info["merged_trial_id"]

    disease_hit_ids = tep.satisfy_disease_constraints(
        conn,
        patient_id,
        daa_table="disease_constraint_alternatives_nonact",
        require_root_for_hops=require_root_for_hops,
        important_table="patient_inclusion_constraints_important_all",
    )

    positive_hit_ids = tep.satisfy_positive_literal_constraints(
        conn,
        patient_id,
        require_root_for_hops=require_root_for_hops,
        important_table="patient_inclusion_constraints_important_all",
        alt_mode="nonact",
    )

    prevention_hit_ids = tep.satisfy_prevention_constraints(
        conn,
        patient_id,
        require_root_for_hops=require_root_for_hops,
        alt_mode="nonact",
    )

    important_all_facts = fetch_patient_bool_facts(
        conn,
        patient_id,
        "patient_inclusion_constraints_important_all",
        base_var_col="base_var",
    )
    root_facts = fetch_patient_bool_facts(
        conn,
        patient_id,
        "patient_inclusion_constraints",
        base_var_col="base_var",
    )

    prevention_var_col = pick_col(
        cur,
        "patient_prevention_constraints",
        ["entity_var", "base_var", "var_name", "var_name_notime"],
    ) or "base_var"

    prevention_facts = fetch_patient_bool_facts(
        conn,
        patient_id,
        "patient_prevention_constraints",
        base_var_col=prevention_var_col,
    )

    key_debug = debug_trial_table_keys(conn, trial_info)

    disease_pack = get_disease_sources_for_trial(conn, trial_info)
    positive_pack = get_positive_sources_for_trial(conn, trial_info)
    prevention_pack = get_prevention_sources_for_trial(conn, trial_info)

    disease_expl = explain_source_rows(
        disease_pack["rows"],
        important_all_facts=important_all_facts,
        root_facts=root_facts,
        prevention_facts=prevention_facts,
        source_family="disease",
        require_root_for_hops=require_root_for_hops,
    )
    positive_expl = explain_source_rows(
        positive_pack["rows"],
        important_all_facts=important_all_facts,
        root_facts=root_facts,
        prevention_facts=prevention_facts,
        source_family="positive_literal",
        require_root_for_hops=require_root_for_hops,
    )
    prevention_expl = explain_source_rows(
        prevention_pack["rows"],
        important_all_facts=important_all_facts,
        root_facts=root_facts,
        prevention_facts=prevention_facts,
        source_family="prevention",
        require_root_for_hops=require_root_for_hops,
    )

    result = {
        "patient_id": patient_id,
        "trial": trial_info,
        "mode": {
            "important_mode": "all",
            "important_table": "patient_inclusion_constraints_important_all",
            "alt_mode": "nonact",
            "require_root_for_hops": require_root_for_hops,
        },
        "hit_membership": {
            "disease_hit": merged_trial_id in set(disease_hit_ids),
            "positive_literal_hit": merged_trial_id in set(positive_hit_ids),
            "prevention_hit": merged_trial_id in set(prevention_hit_ids),
            "union_hit": merged_trial_id in (set(disease_hit_ids) | set(positive_hit_ids) | set(prevention_hit_ids)),
        },
        "raw_hit_sets": {
            "disease_hit_ids": disease_hit_ids,
            "positive_literal_hit_ids": positive_hit_ids,
            "prevention_hit_ids": prevention_hit_ids,
        },
        "table_key_debug": key_debug,
        "table_source_debug": {
            "disease": disease_pack["debug"],
            "positive_literal": positive_pack["debug"],
            "prevention": prevention_pack["debug"],
        },
        "trial_side_debug": {
            "disease": disease_expl,
            "positive_literal": positive_expl,
            "prevention": prevention_expl,
        },
        "summary": build_summary(disease_expl, positive_expl, prevention_expl),
    }
    return result


# ============================================================
# Pretty print
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_hit_membership(result: Dict[str, Any]):
    hm = result["hit_membership"]
    print_section("Hit membership")
    for k, v in hm.items():
        print(f"{k}: {v}")


def print_key_debug(result: Dict[str, Any]):
    print_section("Table key debug")
    tkd = result["table_key_debug"]
    for table, info in tkd.items():
        print(f"\n[{table}]")
        if not info.get("exists"):
            print("  table missing")
            continue
        print(f"  columns: {', '.join(info.get('columns', []))}")
        matches = info.get("matches", {})
        if not matches:
            print("  no recognized key columns")
            continue
        for key_col, by_probe in matches.items():
            print(f"  key_col={key_col}")
            for probe_label, count in by_probe.items():
                print(f"    {probe_label}: {count}")


def print_source_debug(result: Dict[str, Any]):
    print_section("Best key chosen per source table")
    sdbg = result["table_source_debug"]
    for family, info in sdbg.items():
        print(f"\n<{family}>")
        for table, x in info.items():
            print(f"  [{table}] {json.dumps(x, ensure_ascii=False)}")


def print_explanations(name: str, rows: List[Dict[str, Any]], show_rows: bool):
    print_section(name)
    if not rows:
        print("(none)")
        return

    for i, r in enumerate(rows, 1):
        print(f"[{i}] var={r['base_var']}")
        print(f"    source_table       : {r['source_table']}")
        print(f"    matched_key        : {r['matched_key_col']} = {r['matched_probe_label']} ({r['matched_probe_value']})")
        print(f"    hop / reason       : {r['hop']} / {r['reason']}")
        print(f"    fact_source        : {r['fact_source']}")
        print(f"    patient_fact_exist : {r['patient_fact_exists']}")
        print(f"    patient_fact_truthy: {r['patient_fact_truthy']}")
        print(f"    patient_fact_root  : {r['patient_fact_root']}")
        print(f"    needs_root_gate    : {r['needs_root_gate']}")
        print(f"    would_match        : {r['would_match']}")
        print(f"    failure_reason     : {r['failure_reason']}")
        if show_rows and r["patient_rows"]:
            print("    patient_rows:")
            for pr in r["patient_rows"]:
                slim = {
                    k: pr.get(k)
                    for k in [
                        "patient_id", "base_var", "entity_var", "kind", "value", "is_root",
                        "tf_token", "tf_lb_hours", "tf_ub_hours"
                    ]
                    if k in pr
                }
                print("      " + json.dumps(slim, ensure_ascii=False))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Inspect why a trial is / is not hit under important_mode=all and alt_mode=nonact"
    )
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--trial", required=True, help="merged trial id or NCT id")
    ap.add_argument("--allow-hop-without-root", action="store_true")
    ap.add_argument("--json", action="store_true", help="Emit full JSON instead of pretty text")
    ap.add_argument("--show-patient-rows", action="store_true", help="Print matching patient fact rows")
    args = ap.parse_args()

    try:
        conn = sqlite3.connect(str(args.db))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        result = inspect_nonact_all(
            conn,
            patient_id=args.patient,
            trial=args.trial,
            require_root_for_hops=not args.allow_hop_without_root,
        )
    except Exception as e:
        print(f"[error] inspection failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print_section("Target")
    print(json.dumps({
        "patient_id": result["patient_id"],
        "trial": result["trial"],
        "mode": result["mode"],
    }, ensure_ascii=False, indent=2))

    print_hit_membership(result)

    print_section("Summary")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    print_key_debug(result)
    print_source_debug(result)

    ts = result["trial_side_debug"]
    print_explanations("Disease sources", ts["disease"], args.show_patient_rows)
    print_explanations("Positive literal sources", ts["positive_literal"], args.show_patient_rows)
    print_explanations("Prevention sources", ts["prevention"], args.show_patient_rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)