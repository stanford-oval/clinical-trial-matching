#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_trial_label_why.py

Inspect why a patient-trial pair is labeled as:
  - all_satisfied
  - unsatisfied_inclusion
  - explicit_contradiction

Designed to match the current logic in:
  - compose_trial_eval.py
  - trial_eval_primitives.py

Usage examples
--------------
python inspect_trial_label_why.py \
  --db ../../build/trial.db \
  --patient sigir-201416 \
  --trial-id 12345 \
  --mode ccr \
  --scope any \
  --alt-mode act \
  --pretty

python inspect_trial_label_why.py \
  --db ../../build/trial.db \
  --patient sigir-201416 \
  --nct NCT01723358 \
  --mode all \
  --scope any \
  --alt-mode nonact \
  --pretty

Notes
-----
1) Final label is computed using trial_eval_primitives.classify_trials().
2) explicit_contradiction explanation follows eliminate():
     - knowledge = patient_exclusion_constraints + demographics
     - checks BOTH inclusion constraint_clauses and exclusion constraint_clauses
3) unsatisfied_inclusion / all_satisfied explanation follows inclusion_gap():
     - BOOL inclusion + assumed-inclusion constraint_clauses only
     - semantic lifting for inclusion-gap
     - root gating comes from patient_inclusion_constraints.is_root (not important_table.is_root)
     - optional fallback to default_predicate_values(kind='inclusion')
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sql_retrieval.ops import constraint_primitives as tep


# ============================================================
# Shared helpers
# ============================================================

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SENTINEL_NEG_INF = -1e15
SENTINEL_POS_INF = 1e15


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


def pick_col(cur: sqlite3.Cursor, table: str, candidates: Sequence[str]) -> Optional[str]:
    rows = cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()
    have = {r[1] for r in rows}
    for c in candidates:
        if c in have:
            return c
    return None


def _scope_pred(alias: str, scope: str) -> str:
    if scope == "any":
        return ""
    return (
        f"AND COALESCE({alias}.tf_lb_hours, {SENTINEL_NEG_INF}) <= 0 "
        f"AND COALESCE({alias}.tf_ub_hours,  {SENTINEL_POS_INF}) >= 0"
    )


def _inc_cols(cur: sqlite3.Cursor, table: str, alias: str) -> tuple[str, str]:
    try:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
    except sqlite3.Error:
        cols = set()
    lb = f"{alias}.tf_lb_inclusive" if "tf_lb_inclusive" in cols else "1"
    ub = f"{alias}.tf_ub_inclusive" if "tf_ub_inclusive" in cols else "1"
    return lb, ub


def _overlaps_pred_inc(
    fact_alias: str,
    fact_lb_inc_col: str, fact_ub_inc_col: str,
    lit_lb: str, lit_ub: str,
    lit_lb_inc_col: str, lit_ub_inc_col: str,
) -> str:
    flb = f"COALESCE({fact_alias}.tf_lb_hours, {SENTINEL_NEG_INF})"
    fub = f"COALESCE({fact_alias}.tf_ub_hours,  {SENTINEL_POS_INF})"
    llb = f"COALESCE({lit_lb}, {SENTINEL_NEG_INF})"
    lub = f"COALESCE({lit_ub}, {SENTINEL_POS_INF})"

    f_lb_inc = f"COALESCE({fact_alias}.{fact_lb_inc_col}, 1)"
    f_ub_inc = f"COALESCE({fact_alias}.{fact_ub_inc_col}, 1)"
    l_lb_inc = f"COALESCE({lit_lb_inc_col}, 1)"
    l_ub_inc = f"COALESCE({lit_ub_inc_col}, 1)"

    return (
        "("
        f"(({flb} < {lub}) OR ({flb} = {lub} AND {f_lb_inc}=1 AND {l_ub_inc}=1))"
        " AND "
        f"(({fub} > {llb}) OR ({fub} = {llb} AND {f_ub_inc}=1 AND {l_lb_inc}=1))"
        ")"
    )


def _materialize(cur: sqlite3.Cursor, name: str, select_sql: str, params: Dict[str, Any]) -> None:
    cur.execute(f"DROP TABLE IF EXISTS tmp_{name}")
    cur.execute(f"CREATE TEMP TABLE tmp_{name} AS {select_sql}", params)


def important_table_for_mode(mode: str) -> str:
    mode = (mode or "all").strip()
    if mode not in {"chief", "ccr", "all"}:
        raise ValueError(f"bad important mode: {mode}")
    return f"patient_inclusion_constraints_important_{mode}"


def alt_mode_norm(m: str | None) -> str:
    m = (m or "act").strip().lower()
    return "nonact" if m == "nonact" else "act"


def truthy_sql(alias: str) -> str:
    return (
        f"({alias}.value IS NOT NULL) AND ("
        f"CAST({alias}.value AS NUMERIC)=1 "
        f"OR UPPER(CAST({alias}.value AS TEXT)) IN ('1','TRUE','T','Y','YES'))"
    )


def falsy_sql(alias: str) -> str:
    return (
        f"({alias}.value IS NOT NULL) AND ("
        f"CAST({alias}.value AS NUMERIC)=0 "
        f"OR UPPER(CAST({alias}.value AS TEXT)) IN ('0','FALSE','F','N','NO'))"
    )


def _try_num_eq(x: Any, y: float) -> bool:
    try:
        return float(x) == y
    except Exception:
        return False


def _value_satisfies_literal(value: Any, is_neg: int) -> bool:
    sval = str(value).strip().upper()
    truthy = sval in {"1", "TRUE", "T", "Y", "YES"} or _try_num_eq(value, 1.0)
    falsy = sval in {"0", "FALSE", "F", "N", "NO"} or _try_num_eq(value, 0.0)
    return falsy if is_neg else truthy


# ============================================================
# Trial resolution + label
# ============================================================

def resolve_trial_id(conn: sqlite3.Connection, trial_id: int | None, nct_id: str | None) -> Tuple[int, str]:
    cur = conn.cursor()
    if trial_id is not None:
        row = cur.execute(
            "SELECT id, nct_id FROM trials WHERE id=? LIMIT 1",
            (int(trial_id),),
        ).fetchone()
        if not row:
            raise RuntimeError(f"trial id not found: {trial_id}")
        return int(row[0]), str(row[1])

    if not nct_id:
        raise RuntimeError("must provide --trial-id or --nct")

    rows = cur.execute(
        "SELECT id, nct_id FROM trials WHERE nct_id=? ORDER BY id",
        (nct_id,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"nct not found: {nct_id}")

    if len(rows) > 1:
        ids = [int(r[0]) for r in rows]
        raise RuntimeError(
            f"nct {nct_id} maps to multiple merged trial ids: {ids}. "
            f"Please use --trial-id."
        )
    return int(rows[0][0]), str(rows[0][1])


def classify_one(
    conn: sqlite3.Connection,
    patient_id: str,
    trial_id: int,
    scope: str,
    important_table: str,
) -> Dict[str, Any]:
    rows = tep.classify_trials(
        conn,
        patient_id=patient_id,
        candidate_trial_ids=[trial_id],
        scope=scope,
        important_table=important_table,
    )
    if not rows:
        raise RuntimeError("classify_trials returned no rows")
    return rows[0]


# ============================================================
# Part A: explicit_contradiction explanation (matches eliminate)
# ============================================================

def _build_demo_temp_tables(cur: sqlite3.Cursor) -> None:
    demo_bool = """
      SELECT d.patient_id,
             ('patient_sex_is_'||d.sex) AS base_var,
             d.tf_token, 1.0 AS value,
             d.tf_lb_hours, d.tf_ub_hours,
             d.tf_lb_inclusive, d.tf_ub_inclusive
      FROM patient_demographic_constraints d
      WHERE d.sex IS NOT NULL
    """
    demo_num = """
      SELECT patient_id, 'patient_age_value_recorded_in_years'  AS base_var, tf_token, age_years  AS value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_demographic_constraints WHERE age_years IS NOT NULL
      UNION ALL
      SELECT patient_id, 'patient_age_value_recorded_in_months' AS base_var, tf_token, age_months AS value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_demographic_constraints WHERE age_months IS NOT NULL
      UNION ALL
      SELECT patient_id, 'patient_age_value_recorded_in_days'   AS base_var, tf_token, age_days   AS value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_demographic_constraints WHERE age_days IS NOT NULL
    """
    ex_bool = """
      SELECT patient_id, base_var, tf_token, value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_exclusion_constraints WHERE kind='bool'
      UNION ALL
      SELECT patient_id, base_var, tf_token, value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM tmp_demo_bool
    """
    ex_num = """
      SELECT patient_id, base_var, tf_token, value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM patient_exclusion_constraints WHERE kind='num'
      UNION ALL
      SELECT patient_id, base_var, tf_token, value,
             tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive
      FROM tmp_demo_num
    """

    _materialize(cur, "demo_bool", demo_bool, {})
    _materialize(cur, "demo_num", demo_num, {})
    _materialize(cur, "ex_knowledge_bool", ex_bool, {})
    _materialize(cur, "ex_knowledge_num", ex_num, {})


def _collect_bool_clause_members(
    conn: sqlite3.Connection,
    patient_id: str,
    clause_id: int,
    scope: str,
    knowledge_table: str,
) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cl_lb_inc, cl_ub_inc = _inc_cols(cur, "constraint_clause_atoms", alias="cl")
    kb_lb_inc, kb_ub_inc = _inc_cols(cur, knowledge_table, alias="kb")
    kb_scope = _scope_pred("kb", scope)

    sql = f"""
      SELECT
        cl.literal_index,
        cl.base_var,
        cl.is_neg,
        cl.tf_lb_hours,
        cl.tf_ub_hours,
        {cl_lb_inc} AS cl_tf_lb_inclusive,
        {cl_ub_inc} AS cl_tf_ub_inclusive,

        kb.tf_token,
        kb.value,
        kb.tf_lb_hours,
        kb.tf_ub_hours,
        {kb_lb_inc} AS kb_tf_lb_inclusive,
        {kb_ub_inc} AS kb_tf_ub_inclusive

      FROM constraint_clause_atoms cl
      LEFT JOIN {_safe_ident(knowledge_table)} kb
        ON kb.patient_id=:patient_id
       AND kb.base_var=cl.base_var
       AND {_overlaps_pred_inc(
            "kb",
            "tf_lb_inclusive", "tf_ub_inclusive",
            "cl.tf_lb_hours", "cl.tf_ub_hours",
            cl_lb_inc, cl_ub_inc
       )} {kb_scope}
      WHERE cl.clause_id=:clause_id
      ORDER BY cl.literal_index, kb.tf_token
    """
    rows = cur.execute(sql, {"patient_id": patient_id, "clause_id": clause_id}).fetchall()

    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        lit_idx = int(row[0])
        base = grouped.setdefault(lit_idx, {
            "literal_index": lit_idx,
            "base_var": row[1],
            "is_neg": int(row[2]),
            "literal_tf_lb_hours": row[3],
            "literal_tf_ub_hours": row[4],
            "literal_tf_lb_inclusive": row[5],
            "literal_tf_ub_inclusive": row[6],
            "facts": [],
        })
        if row[7] is not None:
            val = row[8]
            sat = _value_satisfies_literal(val, base["is_neg"])
            base["facts"].append({
                "tf_token": row[7],
                "value": val,
                "tf_lb_hours": row[9],
                "tf_ub_hours": row[10],
                "tf_lb_inclusive": row[11],
                "tf_ub_inclusive": row[12],
                "satisfies_literal": sat,
            })

    out: List[Dict[str, Any]] = []
    for lit_idx in sorted(grouped):
        g = grouped[lit_idx]
        g["known"] = len(g["facts"]) > 0
        g["satisfied"] = any(f["satisfies_literal"] for f in g["facts"])
        out.append(g)
    return out


def _collect_numeric_clause_members(
    conn: sqlite3.Connection,
    patient_id: str,
    clause_id: int,
    knowledge_table: str,
) -> List[Dict[str, Any]]:
    cur = conn.cursor()

    sql = f"""
      SELECT
        cnr.member_index,
        cnr.base_var,
        cnr.lb,
        cnr.lb_inc,
        cnr.ub,
        cnr.ub_inc,
        kb.tf_token,
        kb.value
      FROM constraint_clause_numeric_range cnr
      LEFT JOIN {_safe_ident(knowledge_table)} kb
        ON kb.patient_id=:patient_id
       AND kb.base_var=cnr.base_var
      WHERE cnr.clause_id=:clause_id
      ORDER BY cnr.member_index, kb.tf_token
    """
    rows = cur.execute(sql, {"patient_id": patient_id, "clause_id": clause_id}).fetchall()

    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        idx = int(row[0])
        g = grouped.setdefault(idx, {
            "member_index": idx,
            "base_var": row[1],
            "lb": row[2],
            "lb_inc": row[3],
            "ub": row[4],
            "ub_inc": row[5],
            "facts": [],
        })
        if row[6] is not None:
            val = float(row[7])
            lb = row[2]
            ub = row[4]
            lb_ok = (lb is None) or (val > lb) or (val == lb and int(row[3] or 0) == 1)
            ub_ok = (ub is None) or (val < ub) or (val == ub and int(row[5] or 0) == 1)
            g["facts"].append({
                "tf_token": row[6],
                "value": row[7],
                "satisfies_member": bool(lb_ok and ub_ok),
            })

    out: List[Dict[str, Any]] = []
    for idx in sorted(grouped):
        g = grouped[idx]
        g["known"] = len(g["facts"]) > 0
        g["satisfied"] = any(f["satisfies_member"] for f in g["facts"])
        out.append(g)
    return out


def _inspect_clause_against_exclusion_knowledge(
    conn: sqlite3.Connection,
    patient_id: str,
    clause_id: int,
    scope: str,
) -> Dict[str, Any]:
    bool_members = _collect_bool_clause_members(
        conn, patient_id, clause_id, scope, knowledge_table="tmp_ex_knowledge_bool"
    )
    num_members = _collect_numeric_clause_members(
        conn, patient_id, clause_id, knowledge_table="tmp_ex_knowledge_num"
    )

    all_members = []
    for x in bool_members:
        all_members.append({
            "kind": "bool",
            "index": x["literal_index"],
            "known": x["known"],
            "satisfied": x["satisfied"],
            "detail": x,
        })
    for x in num_members:
        all_members.append({
            "kind": "num",
            "index": x["member_index"],
            "known": x["known"],
            "satisfied": x["satisfied"],
            "detail": x,
        })

    all_members.sort(key=lambda z: (z["kind"], z["index"]))

    n_members = len(all_members)
    known_count = sum(1 for x in all_members if x["known"])
    sat_count = sum(1 for x in all_members if x["satisfied"])
    fully_known = (n_members > 0 and known_count == n_members)
    unsatisfied_clause = fully_known and sat_count == 0

    return {
        "clause_id": clause_id,
        "n_members": n_members,
        "known_count": known_count,
        "sat_count": sat_count,
        "fully_known": fully_known,
        "unsatisfied_clause": unsatisfied_clause,
        "members": all_members,
    }


def inspect_explicit_contradiction_details(
    conn: sqlite3.Connection,
    patient_id: str,
    merged_trial_id: int,
    scope: str,
) -> Dict[str, Any]:
    cur = conn.cursor()
    _build_demo_temp_tables(cur)

    row = cur.execute("""
      SELECT inclusion_trial_side_id, assumed_trial_side_id, exclusion_trial_side_id, nct_id
      FROM trials
      WHERE id=?
      LIMIT 1
    """, (merged_trial_id,)).fetchone()
    if not row:
        raise RuntimeError(f"trial id not found: {merged_trial_id}")

    inclusion_side_id, assumed_side_id, exclusion_side_id, nct_id = row

    assumed_is_inclusion = False
    if assumed_side_id is not None:
        k = cur.execute(
            "SELECT kind FROM trial_constraint_sides WHERE id=? LIMIT 1",
            (assumed_side_id,),
        ).fetchone()
        assumed_is_inclusion = bool(k and k[0] == "inclusion")

    inclusion_clause_ids = [
        int(r[0]) for r in cur.execute(
            "SELECT clause_id FROM trial_constraint_clauses WHERE trial_id=? ORDER BY clause_id",
            (inclusion_side_id,),
        ).fetchall()
    ]
    assumed_clause_ids = []
    if assumed_is_inclusion:
        assumed_clause_ids = [
            int(r[0]) for r in cur.execute(
                "SELECT clause_id FROM trial_constraint_clauses WHERE trial_id=? ORDER BY clause_id",
                (assumed_side_id,),
            ).fetchall()
        ]

    exclusion_clause_ids = [
        int(r[0]) for r in cur.execute(
            "SELECT clause_id FROM trial_constraint_clauses WHERE trial_id=? ORDER BY clause_id",
            (exclusion_side_id,),
        ).fetchall()
    ]

    violated_inclusion = []
    for cid in inclusion_clause_ids + assumed_clause_ids:
        info = _inspect_clause_against_exclusion_knowledge(conn, patient_id, cid, scope)
        if info["unsatisfied_clause"]:
            violated_inclusion.append(info)

    violated_exclusion = []
    for cid in exclusion_clause_ids:
        info = _inspect_clause_against_exclusion_knowledge(conn, patient_id, cid, scope)
        if info["unsatisfied_clause"]:
            violated_exclusion.append(info)

    return {
        "trial_id": merged_trial_id,
        "nct_id": nct_id,
        "violated_inclusion_or_assumed_constraint_clauses": violated_inclusion,
        "violated_exclusion_constraint_clauses": violated_exclusion,
        "has_explicit_contradiction": bool(violated_inclusion or violated_exclusion),
    }


# ============================================================
# Part B: inclusion-gap explanation (matches inclusion_gap)
# ============================================================

def _find_lifted_source(cur: sqlite3.Cursor) -> str:
    if table_has(cur, "constraint_literal_alternatives"):
        return "constraint_literal_alternatives"
    if table_has(cur, "constraint_lifted_atoms"):
        return "constraint_lifted_atoms"
    return ""


def _build_inclusion_gap_temp_tables(
    conn: sqlite3.Connection,
    patient_id: str,
    merged_trial_id: int,
    scope: str,
    important_table: str,
) -> None:
    cur = conn.cursor()
    important_table = _safe_ident(important_table)

    cur.execute("DROP TABLE IF EXISTS tmp_trials_base_gap")
    cur.execute("""
      CREATE TEMP TABLE tmp_trials_base_gap AS
      SELECT id AS merged_trial_id
      FROM trials
      WHERE id=?
    """, (merged_trial_id,))

    has_demo = table_has(cur, "patient_demographic_constraints")
    has_fi = table_has(cur, "patient_inclusion_constraints")
    has_fii = table_has(cur, important_table)

    demo_bool = (
        """
        SELECT d.patient_id, ('patient_sex_is_'||d.sex) AS base_var,
               d.tf_token, 1.0 AS value,
               d.tf_lb_hours, d.tf_ub_hours,
               d.tf_lb_inclusive, d.tf_ub_inclusive
        FROM patient_demographic_constraints d
        WHERE d.sex IS NOT NULL
        """
        if has_demo else
        """
        SELECT NULL AS patient_id, NULL AS base_var, NULL AS tf_token, NULL AS value,
               NULL AS tf_lb_hours, NULL AS tf_ub_hours,
               NULL AS tf_lb_inclusive, NULL AS tf_ub_inclusive
        WHERE 0
        """
    )
    _materialize(cur, "demo_bool", demo_bool, {})

    if has_fi:
        in_bool_fi_demo = """
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM patient_inclusion_constraints
          WHERE kind='bool'
          UNION ALL
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM tmp_demo_bool
        """
    else:
        in_bool_fi_demo = """
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM tmp_demo_bool
        """
    _materialize(cur, "in_bool_fi_demo", in_bool_fi_demo, {})

    if has_fi:
        patient_root_bool = """
          SELECT patient_id, base_var
          FROM patient_inclusion_constraints
          WHERE kind='bool' AND COALESCE(is_root,0)=1
          GROUP BY patient_id, base_var
        """
    else:
        patient_root_bool = """
          SELECT NULL AS patient_id, NULL AS base_var
          WHERE 0
        """
    _materialize(cur, "patient_root_bool", patient_root_bool, {})

    if has_fii:
        in_bool_fii_root = f"""
          SELECT fii.patient_id, fii.base_var, fii.tf_token, fii.value,
                 fii.tf_lb_hours, fii.tf_ub_hours,
                 fii.tf_lb_inclusive, fii.tf_ub_inclusive
          FROM {important_table} fii
          JOIN tmp_patient_root_bool pr
            ON pr.patient_id = fii.patient_id
           AND pr.base_var   = fii.base_var
          WHERE fii.kind='bool'
        """
        _materialize(cur, "in_bool_fii_root", in_bool_fii_root, {})
    else:
        cur.execute("DROP TABLE IF EXISTS tmp_in_bool_fii_root")
        cur.execute("""
          CREATE TEMP TABLE tmp_in_bool_fii_root AS
          SELECT NULL AS patient_id, NULL AS base_var, NULL AS tf_token, NULL AS value,
                 NULL AS tf_lb_hours, NULL AS tf_ub_hours,
                 NULL AS tf_lb_inclusive, NULL AS tf_ub_inclusive
          WHERE 0
        """)

    incl_constraint_clauses = """
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
      WHERE mt.id = :trial_id

      UNION ALL

      SELECT mt.id AS merged_trial_id, tc2.clause_id
      FROM trials mt
      JOIN trial_constraint_sides ta
        ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
      JOIN trial_constraint_clauses tc2
        ON tc2.trial_id = ta.id
      WHERE mt.id = :trial_id
    """
    _materialize(cur, "inclusion_trial_constraint_clauses_gap", incl_constraint_clauses, {"trial_id": merged_trial_id})

    cur.execute("DROP TABLE IF EXISTS tmp_inclusion_trial_constraint_clauses_gap_nonum")
    cur.execute("""
      CREATE TEMP TABLE tmp_inclusion_trial_constraint_clauses_gap_nonum AS
      SELECT itc.merged_trial_id, itc.clause_id
      FROM tmp_inclusion_trial_constraint_clauses_gap itc
      WHERE NOT EXISTS (
        SELECT 1
        FROM constraint_clause_numeric_range cnr
        WHERE cnr.clause_id = itc.clause_id
      )
    """)

    cur.execute("DROP TABLE IF EXISTS tmp_clause_bool_size")
    cur.execute("""
      CREATE TEMP TABLE tmp_clause_bool_size AS
      SELECT cl.clause_id,
             COUNT(DISTINCT cl.literal_index) AS n_bool_members
      FROM constraint_clause_atoms cl
      GROUP BY cl.clause_id
    """)

    lifted_source = _find_lifted_source(cur)
    if lifted_source:
        hop_col = pick_col(cur, lifted_source, ["hop"])
        reason_col = pick_col(cur, lifted_source, ["reason"])

        hop_expr = f"x.{hop_col}" if hop_col else "NULL"
        reason_expr = f"x.{reason_col}" if reason_col else "NULL"

        lifted_sql = f"""
          SELECT
            x.trial_id,
            x.clause_id,
            x.literal_index,
            x.lifted_var,
            x.lifted_var_stem,
            x.base_var_stem,
            COALESCE(x.lifted_var_stem, x.base_var_stem) AS base_var_stem_key,
            COALESCE({hop_expr}, 0) AS hop,
            COALESCE({reason_expr}, 'self') AS reason
          FROM {_safe_ident(lifted_source)} x
        """

        semantic_sql = f"""
          WITH la AS (
            SELECT
              mt.id AS merged_trial_id,
              sub.trial_id,
              sub.clause_id,
              sub.literal_index,
              sub.base_var_stem,
              sub.lifted_var_stem,
              sub.base_var_stem_key,
              sub.hop,
              sub.reason
            FROM trials mt
            JOIN ({lifted_sql}) AS sub
              ON sub.trial_id IN (
                   mt.inclusion_trial_side_id,
                   mt.assumed_trial_side_id
                 )
            WHERE mt.id = :trial_id
          )
          SELECT
            itc.merged_trial_id,
            cl.clause_id,
            cl.literal_index,
            COALESCE(la.base_var_stem_key, cl.base_var) AS sem_base_var,
            CASE
              WHEN la.hop IS NOT NULL
                   AND la.hop > 0
                   AND COALESCE(la.reason, 'self') <> 'self'
              THEN 1 ELSE 0
            END AS sem_is_lifted
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          LEFT JOIN la
            ON la.merged_trial_id = itc.merged_trial_id
           AND la.clause_id      = cl.clause_id
           AND la.literal_index  = cl.literal_index
        """
        _materialize(cur, "literal_semantic_gap", semantic_sql, {"trial_id": merged_trial_id})
    else:
        semantic_sql = """
          SELECT
            itc.merged_trial_id,
            cl.clause_id,
            cl.literal_index,
            cl.base_var AS sem_base_var,
            0 AS sem_is_lifted
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
        """
        _materialize(cur, "literal_semantic_gap", semantic_sql, {})

    kib_scope = _scope_pred("kib_fd", scope)
    cl_lb_inc, cl_ub_inc = _inc_cols(cur, "constraint_clause_atoms", alias="cl")
    has_default = table_has(cur, "default_predicate_values")

    if has_default:
        canon_trial_nct = (
            "CASE WHEN t.nct_id LIKE 'NCT________%' "
            "THEN SUBSTR(t.nct_id, 1, 11) ELSE t.nct_id END"
        )
        canon_dv_nct = (
            "CASE WHEN dv.nct_id LIKE 'NCT________%' "
            "THEN SUBSTR(dv.nct_id, 1, 11) ELSE dv.nct_id END"
        )

        ikb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          LEFT JOIN trials t
            ON t.id = itc.merged_trial_id
          LEFT JOIN default_predicate_values dv
            ON {canon_dv_nct} = {canon_trial_nct}
           AND dv.base_var     = sem.sem_base_var
           AND dv.kind         = 'inclusion'
          WHERE
              (
                sem.sem_is_lifted = 0
                AND (
                     kib_fd.patient_id IS NOT NULL
                  OR kib_root.patient_id IS NOT NULL
                  OR dv.base_var IS NOT NULL
                )
              )
           OR (
                sem.sem_is_lifted = 1
                AND (
                     kib_root.patient_id IS NOT NULL
                  OR dv.base_var IS NOT NULL
                )
              )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """
    else:
        ikb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          WHERE
              (
                sem.sem_is_lifted = 0
                AND (
                     kib_fd.patient_id IS NOT NULL
                  OR kib_root.patient_id IS NOT NULL
                )
              )
           OR (
                sem.sem_is_lifted = 1
                AND kib_root.patient_id IS NOT NULL
              )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """
    _materialize(cur, "gap_ikb", ikb, {"patient_id": patient_id})

    if has_default:
        canon_trial_nct = (
            "CASE WHEN t.nct_id LIKE 'NCT________%' "
            "THEN SUBSTR(t.nct_id, 1, 11) ELSE t.nct_id END"
        )
        canon_dv_nct = (
            "CASE WHEN dv.nct_id LIKE 'NCT________%' "
            "THEN SUBSTR(dv.nct_id, 1, 11) ELSE dv.nct_id END"
        )

        isb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          LEFT JOIN trials t
            ON t.id = itc.merged_trial_id
          LEFT JOIN default_predicate_values dv
            ON {canon_dv_nct} = {canon_trial_nct}
           AND dv.base_var     = sem.sem_base_var
           AND dv.kind         = 'inclusion'
          WHERE
            (
              sem.sem_is_lifted = 0 AND (
                (
                  kib_fd.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('kib_fd')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('kib_fd')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('kib_root')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NULL
                  AND dv.base_var IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('dv')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('dv')})
                  )
                )
              )
            )
            OR
            (
              sem.sem_is_lifted = 1 AND (
                (
                  kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('kib_root')})
                  )
                )
                OR
                (
                  kib_root.patient_id IS NULL
                  AND dv.base_var IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('dv')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('dv')})
                  )
                )
              )
            )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """
    else:
        isb = f"""
          SELECT
            itc.merged_trial_id,
            sem.clause_id,
            COUNT(DISTINCT sem.literal_index) AS n
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
          JOIN constraint_clause_atoms cl
            ON cl.clause_id = itc.clause_id
          JOIN tmp_literal_semantic_gap sem
            ON sem.merged_trial_id = itc.merged_trial_id
           AND sem.clause_id       = cl.clause_id
           AND sem.literal_index   = cl.literal_index
          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}
          WHERE
            (
              sem.sem_is_lifted = 0 AND (
                (
                  kib_fd.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('kib_fd')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('kib_fd')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_sql('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_sql('kib_root')})
                  )
                )
              )
            )
            OR
            (
              sem.sem_is_lifted = 1 AND
              kib_root.patient_id IS NOT NULL AND
              (
                (cl.is_neg = 0 AND {truthy_sql('kib_root')})
                OR
                (cl.is_neg = 1 AND {falsy_sql('kib_root')})
              )
            )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """
    _materialize(cur, "gap_isb", isb, {"patient_id": patient_id})


def inspect_inclusion_gap_details(
    conn: sqlite3.Connection,
    patient_id: str,
    merged_trial_id: int,
    scope: str,
    important_table: str,
) -> Dict[str, Any]:
    cur = conn.cursor()
    _build_inclusion_gap_temp_tables(
        conn=conn,
        patient_id=patient_id,
        merged_trial_id=merged_trial_id,
        scope=scope,
        important_table=important_table,
    )

    clause_ids = [
        int(r[0]) for r in cur.execute("""
          SELECT clause_id
          FROM tmp_inclusion_trial_constraint_clauses_gap_nonum
          ORDER BY clause_id
        """).fetchall()
    ]

    has_default = table_has(cur, "default_predicate_values")
    cl_lb_inc, cl_ub_inc = _inc_cols(cur, "constraint_clause_atoms", alias="cl")
    kib_scope = _scope_pred("kib_fd", scope)

    out_rows: List[Dict[str, Any]] = []

    for cid in clause_ids:
        sql = f"""
          SELECT
            cl.literal_index,
            cl.base_var,
            cl.is_neg,
            cl.tf_lb_hours,
            cl.tf_ub_hours,
            {cl_lb_inc} AS cl_tf_lb_inclusive,
            {cl_ub_inc} AS cl_tf_ub_inclusive,

            sem.sem_base_var,
            sem.sem_is_lifted,

            kib_fd.tf_token,
            kib_fd.value,

            kib_root.tf_token,
            kib_root.value,

            dv.value

          FROM constraint_clause_atoms cl
          JOIN tmp_literal_semantic_gap sem
            ON sem.clause_id = cl.clause_id
           AND sem.literal_index = cl.literal_index

          LEFT JOIN tmp_in_bool_fi_demo kib_fd
            ON kib_fd.patient_id = :patient_id
           AND kib_fd.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_fd",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}

          LEFT JOIN tmp_in_bool_fii_root kib_root
            ON kib_root.patient_id = :patient_id
           AND kib_root.base_var   = sem.sem_base_var
           AND {_overlaps_pred_inc(
                "kib_root",
                "tf_lb_inclusive", "tf_ub_inclusive",
                "cl.tf_lb_hours", "cl.tf_ub_hours",
                cl_lb_inc, cl_ub_inc
           )} {kib_scope}

          LEFT JOIN trials t
            ON t.id = :trial_id

          LEFT JOIN default_predicate_values dv
            ON (
                 CASE WHEN dv.nct_id LIKE 'NCT________%'
                      THEN SUBSTR(dv.nct_id, 1, 11)
                      ELSE dv.nct_id
                 END
               ) = (
                 CASE WHEN t.nct_id LIKE 'NCT________%'
                      THEN SUBSTR(t.nct_id, 1, 11)
                      ELSE t.nct_id
                 END
               )
           AND dv.base_var = sem.sem_base_var
           AND dv.kind = 'inclusion'

          WHERE cl.clause_id = :clause_id
          ORDER BY cl.literal_index
        """
        rows = cur.execute(sql, {
            "patient_id": patient_id,
            "trial_id": merged_trial_id,
            "clause_id": cid,
        }).fetchall()

        grouped: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            lit_idx = int(row[0])
            g = grouped.setdefault(lit_idx, {
                "literal_index": lit_idx,
                "raw_base_var": row[1],
                "is_neg": int(row[2]),
                "literal_tf_lb_hours": row[3],
                "literal_tf_ub_hours": row[4],
                "literal_tf_lb_inclusive": row[5],
                "literal_tf_ub_inclusive": row[6],
                "semantic_base_var": row[7],
                "semantic_is_lifted": int(row[8]),
                "chosen_source": None,
                "chosen_value": None,
                "known": False,
                "satisfied": False,
                "all_sources": {
                    "patient_inclusion_constraints_or_demo": [],
                    "important_root_only": [],
                    "default_predicate_values": [],
                },
            })

            if row[9] is not None:
                g["all_sources"]["patient_inclusion_constraints_or_demo"].append({
                    "tf_token": row[9],
                    "value": row[10],
                })
            if row[11] is not None:
                g["all_sources"]["important_root_only"].append({
                    "tf_token": row[11],
                    "value": row[12],
                })
            if row[13] is not None:
                g["all_sources"]["default_predicate_values"].append({
                    "value": row[13],
                })

        for lit_idx in sorted(grouped):
            g = grouped[lit_idx]
            fd = g["all_sources"]["patient_inclusion_constraints_or_demo"]
            root = g["all_sources"]["important_root_only"]
            dv = g["all_sources"]["default_predicate_values"]

            if g["semantic_is_lifted"] == 0:
                if fd:
                    g["chosen_source"] = "patient_inclusion_constraints_or_demo"
                    g["chosen_value"] = fd[0]["value"]
                elif root:
                    g["chosen_source"] = "important_root_only"
                    g["chosen_value"] = root[0]["value"]
                elif has_default and dv:
                    g["chosen_source"] = "default_predicate_values"
                    g["chosen_value"] = dv[0]["value"]
            else:
                if root:
                    g["chosen_source"] = "important_root_only"
                    g["chosen_value"] = root[0]["value"]
                elif has_default and dv:
                    g["chosen_source"] = "default_predicate_values"
                    g["chosen_value"] = dv[0]["value"]

            g["known"] = g["chosen_source"] is not None
            if g["known"]:
                g["satisfied"] = _value_satisfies_literal(g["chosen_value"], g["is_neg"])

        members = [grouped[k] for k in sorted(grouped)]
        n_members = len(members)
        known_count = sum(1 for m in members if m["known"])
        sat_count = sum(1 for m in members if m["satisfied"])
        fully_evaluable = (n_members > 0 and known_count == n_members)
        violated_evaluable = fully_evaluable and sat_count == 0

        out_rows.append({
            "clause_id": cid,
            "n_members": n_members,
            "known_count": known_count,
            "sat_count": sat_count,
            "fully_evaluable": fully_evaluable,
            "violated_evaluable": violated_evaluable,
            "members": members,
        })

    total_bool_constraint_clauses = len(out_rows)
    evaluable_constraint_clauses = sum(1 for r in out_rows if r["fully_evaluable"])
    violated_evaluable_constraint_clauses = sum(1 for r in out_rows if r["violated_evaluable"])
    frac_unsat_any = (violated_evaluable_constraint_clauses / evaluable_constraint_clauses) if evaluable_constraint_clauses else 0.0

    return {
        "trial_id": merged_trial_id,
        "total_bool_constraint_clauses": total_bool_constraint_clauses,
        "evaluable_constraint_clauses": evaluable_constraint_clauses,
        "violated_evaluable_constraint_clauses": violated_evaluable_constraint_clauses,
        "frac_unsat_any": round(frac_unsat_any, 6),
        "violated_clause_ids": [r["clause_id"] for r in out_rows if r["violated_evaluable"]],
        "constraint_clauses": out_rows,
    }


# ============================================================
# Pretty output
# ============================================================

def _short_member_summary(m: Dict[str, Any]) -> str:
    if m.get("kind") == "num":
        d = m["detail"]
        return (
            f"num[{d['member_index']}] {d['base_var']} "
            f"known={d['known']} sat={d['satisfied']}"
        )

    if "detail" in m:
        d = m["detail"]
        sign = "NOT " if d["is_neg"] else ""
        return (
            f"bool[{d['literal_index']}] {sign}{d['base_var']} "
            f"known={d['known']} sat={d['satisfied']}"
        )

    sign = "NOT " if m["is_neg"] else ""
    return (
        f"bool[{m['literal_index']}] {sign}{m['semantic_base_var']} "
        f"(raw={m['raw_base_var']}, lifted={m['semantic_is_lifted']}) "
        f"src={m['chosen_source']} known={m['known']} sat={m['satisfied']}"
    )


def pretty_print_report(report: Dict[str, Any]) -> None:
    print("=" * 88)
    print(f"patient: {report['patient_id']}")
    print(f"trial_id: {report['trial_id']}")
    print(f"nct_id:   {report['nct_id']}")
    print(f"label:    {report['label']}")
    print(f"scope:    {report['scope']}")
    print(f"mode:     {report['important_mode']}")
    print("=" * 88)

    classify_row = report["classify_row"]
    print("\n[final label row]")
    print(json.dumps(classify_row, ensure_ascii=False, indent=2))

    ec = report["explicit_contradiction_explanation"]
    print("\n[explicit contradiction inspection]")
    print(f"has_explicit_contradiction: {ec['has_explicit_contradiction']}")
    print(f"violated inclusion/assumed constraint_clauses: {len(ec['violated_inclusion_or_assumed_constraint_clauses'])}")
    for c in ec["violated_inclusion_or_assumed_constraint_clauses"]:
        print(f"  - clause {c['clause_id']}  members={c['n_members']} known={c['known_count']} sat={c['sat_count']}")
        for m in c["members"]:
            print(f"      {_short_member_summary(m)}")

    print(f"violated exclusion constraint_clauses: {len(ec['violated_exclusion_constraint_clauses'])}")
    for c in ec["violated_exclusion_constraint_clauses"]:
        print(f"  - clause {c['clause_id']}  members={c['n_members']} known={c['known_count']} sat={c['sat_count']}")
        for m in c["members"]:
            print(f"      {_short_member_summary(m)}")

    ig = report["inclusion_gap_explanation"]
    print("\n[inclusion-gap inspection]")
    print(f"total_bool_constraint_clauses:           {ig['total_bool_constraint_clauses']}")
    print(f"evaluable_constraint_clauses:            {ig['evaluable_constraint_clauses']}")
    print(f"violated_evaluable_constraint_clauses:   {ig['violated_evaluable_constraint_clauses']}")
    print(f"frac_unsat_any:               {ig['frac_unsat_any']}")
    print(f"violated_clause_ids:          {ig['violated_clause_ids']}")

    print("\n[violated evaluable inclusion constraint_clauses]")
    for c in ig["constraint_clauses"]:
        if not c["violated_evaluable"]:
            continue
        print(f"  - clause {c['clause_id']}  members={c['n_members']} known={c['known_count']} sat={c['sat_count']}")
        for m in c["members"]:
            print(f"      {_short_member_summary(m)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect why a patient-trial pair is all_satisfied / unsatisfied_inclusion / explicit_contradiction."
    )
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--patient", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--trial-id", type=int, default=None)
    g.add_argument("--nct", default=None)
    ap.add_argument("--scope", choices=["now", "any"], default="any")
    ap.add_argument(
        "--important-mode", "--mode",
        dest="important_mode",
        choices=["chief", "ccr", "all"],
        default="all",
    )
    ap.add_argument("--alt-mode", choices=["act", "nonact"], default="act")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    try:
        conn = sqlite3.connect(str(args.db))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    important_table = important_table_for_mode(args.important_mode)
    _ = alt_mode_norm(args.alt_mode)

    try:
        trial_id, nct_id = resolve_trial_id(conn, args.trial_id, args.nct)

        classify_row = classify_one(
            conn=conn,
            patient_id=args.patient,
            trial_id=trial_id,
            scope=args.scope,
            important_table=important_table,
        )
        label = classify_row["label"]

        explicit_detail = inspect_explicit_contradiction_details(
            conn=conn,
            patient_id=args.patient,
            merged_trial_id=trial_id,
            scope=args.scope,
        )

        inclusion_gap_detail = inspect_inclusion_gap_details(
            conn=conn,
            patient_id=args.patient,
            merged_trial_id=trial_id,
            scope=args.scope,
            important_table=important_table,
        )

        report = {
            "patient_id": args.patient,
            "trial_id": trial_id,
            "nct_id": nct_id,
            "scope": args.scope,
            "important_mode": args.important_mode,
            "important_table": important_table,
            "alt_mode": args.alt_mode,
            "label": label,
            "classify_row": classify_row,
            "explicit_contradiction_explanation": explicit_detail,
            "inclusion_gap_explanation": inclusion_gap_detail,
        }

        if args.pretty:
            pretty_print_report(report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)