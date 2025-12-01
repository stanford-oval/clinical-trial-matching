#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
why_eliminated.py — explain why specific trials were eliminated.

Policy mirrored from eliminate():
- Knowledge source: patient_exclusion_constraints ONLY (no demographics).
- BOOL timeframe applicability: inclusive-overlap between fact and requirement literal intervals.
- BOOL satisfaction: respect clause polarity (is_neg) and truthy/falsy value.
- NUM timeframes: disabled; NUM satisfaction uses value within [lb,ub] (with lb_inc/ub_inc).

Usage:
  python why_eliminated.py --db PATH/triage.sqlite --patient P123 \
      --trials sigir-20141,NCT02532699 --scope any

Outputs a JSON report to stdout.
"""

from __future__ import annotations
import argparse, json, sqlite3, sys
from typing import List, Dict, Any, Optional

SENTINEL_NEG_INF = -1e15
SENTINEL_POS_INF =  1e15

TRUTHY_KEB = """
(keb.value IS NOT NULL) AND (
  CAST(keb.value AS NUMERIC) = 1
  OR UPPER(CAST(keb.value AS TEXT)) IN ('1','TRUE','T','Y','YES')
)
"""

FALSY_KEB = """
(keb.value IS NOT NULL) AND (
  CAST(keb.value AS NUMERIC) = 0
  OR UPPER(CAST(keb.value AS TEXT)) IN ('0','FALSE','F','N','NO')
)
"""

def table_has(cur: sqlite3.Cursor, table: str) -> bool:
    return bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone())

def pick_col(cur: sqlite3.Cursor, table: str, candidates) -> Optional[str]:
    try:
        have = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return None
    for c in candidates:
        if c in have:
            return c
    return None

def inc_cols_or_one(cur: sqlite3.Cursor, table: str, alias: str) -> tuple[str, str]:
    """Return expressions for inclusive flags; default to 1 if missing."""
    try:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        cols = set()
    lb = f"{alias}.tf_lb_inclusive" if "tf_lb_inclusive" in cols else "1"
    ub = f"{alias}.tf_ub_inclusive" if "tf_ub_inclusive" in cols else "1"
    return lb, ub

def overlaps_pred_inc(fact_alias: str, lit_alias: str,
                      fact_lb_inc: str, fact_ub_inc: str,
                      lit_lb_inc: str, lit_ub_inc: str) -> str:
    """Inclusive overlap between [fact.lb, fact.ub] and [lit.lb, lit.ub]."""
    flb = f"COALESCE({fact_alias}.tf_lb_hours, {SENTINEL_NEG_INF})"
    fub = f"COALESCE({fact_alias}.tf_ub_hours,  {SENTINEL_POS_INF})"
    llb = f"COALESCE({lit_alias}.tf_lb_hours,   {SENTINEL_NEG_INF})"
    lub = f"COALESCE({lit_alias}.tf_ub_hours,   {SENTINEL_POS_INF})"
    return (
        "("
        f"(({flb} < {lub}) OR ({flb} = {lub} AND {fact_lb_inc}=1 AND {lit_ub_inc}=1))"
        " AND "
        f"(({fub} > {llb}) OR ({fub} = {llb} AND {fact_ub_inc}=1 AND {lit_lb_inc}=1))"
        ")"
    )

def scope_pred(alias: str, scope: str) -> str:
    if scope == "any":
        return ""
    # scope == now: fact must include 0
    return (
        f"AND COALESCE({alias}.tf_lb_hours, {SENTINEL_NEG_INF}) <= 0 "
        f"AND COALESCE({alias}.tf_ub_hours,  {SENTINEL_POS_INF}) >= 0"
    )

def resolve_trial_ids(cur: sqlite3.Cursor, keys: List[str]) -> List[int]:
    out = []
    for k in keys:
        k = k.strip()
        if not k:
            continue
        if k.isdigit():
            row = cur.execute("SELECT id FROM trials WHERE id=?", (int(k),)).fetchone()
            if row: out.append(int(row[0])); continue
        # try trials.nct_id
        row = cur.execute("SELECT id FROM trials WHERE nct_id=?", (k,)).fetchone()
        if row: out.append(int(row[0])); continue
        # fallback: best-effort exact match in any table that has trial_id/nct_id pointing back to trials
        # (no-op if not found)
        sys.stderr.write(f"[warn] could not resolve trial key: {k}\n")
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--patient", required=True, help="patient_id to evaluate")
    ap.add_argument("--trials", default="sigir-20141,NCT02532699", help="comma-separated trial keys (id or NCT)")
    ap.add_argument("--scope", choices=["now","any"], default="any")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # sanity checks
    required = ["trials","trial_constraint_clauses","constraint_clauses","constraint_clause_atoms","patient_exclusion_constraints"]
    for t in required:
        if not table_has(cur, t):
            sys.stderr.write(f"[error] missing table: {t}\n")
            sys.exit(2)

    # resolve trials
    trial_ids = resolve_trial_ids(cur, args.trials.split(","))
    if not trial_ids:
        sys.stderr.write("[error] no trials resolved from --trials\n")
        sys.exit(2)

    # build candidate temp table
    cur.execute("DROP TABLE IF EXISTS tmp_candidates_inspect")
    cur.execute("CREATE TEMP TABLE tmp_candidates_inspect(id INTEGER PRIMARY KEY)")
    cur.executemany("INSERT INTO tmp_candidates_inspect(id) VALUES (?)", [(tid,) for tid in trial_ids])

    # clause sets
    cur.execute("DROP TABLE IF EXISTS tmp_inclusion_trial_constraint_clauses_why")
    cur.execute("""
      CREATE TEMP TABLE tmp_inclusion_trial_constraint_clauses_why AS
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN tmp_candidates_inspect c ON c.id = mt.id
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
      UNION ALL
      SELECT mt.id AS merged_trial_id, tc2.clause_id
      FROM trials mt
      JOIN tmp_candidates_inspect c ON c.id = mt.id
      JOIN trial_constraint_sides ta ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
      JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
    """)
    cur.execute("DROP TABLE IF EXISTS tmp_exclusion_trial_constraint_clauses_why")
    cur.execute("""
      CREATE TEMP TABLE tmp_exclusion_trial_constraint_clauses_why AS
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN tmp_candidates_inspect c ON c.id = mt.id
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.exclusion_trial_side_id
    """)

    # inclusivity flags (clause side)
    cl_lb_inc, cl_ub_inc = inc_cols_or_one(cur, "constraint_clause_atoms", "cl")
    keb_scope = scope_pred("keb", args.scope)

    # ---- Inclusion: known/sat counts (BOOL + NUM) ----
    # BOOL known
    ikb_sql = f"""
      SELECT itc.merged_trial_id, cl.clause_id,
             COUNT(DISTINCT CASE WHEN keb.patient_id IS NOT NULL THEN cl.literal_index END) AS n
      FROM tmp_inclusion_trial_constraint_clauses_why itc
      JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
      LEFT JOIN patient_exclusion_constraints keb
        ON keb.patient_id=? AND keb.kind='bool' AND keb.base_var=cl.base_var
       AND {overlaps_pred_inc('keb','cl', 'COALESCE(keb.tf_lb_inclusive,1)', 'COALESCE(keb.tf_ub_inclusive,1)', cl_lb_inc, cl_ub_inc)} {keb_scope}
      GROUP BY itc.merged_trial_id, cl.clause_id
    """
    # BOOL sat
    isb_sql = f"""
      SELECT itc.merged_trial_id, cl.clause_id, COUNT(DISTINCT cl.literal_index) AS n
      FROM tmp_inclusion_trial_constraint_clauses_why itc
      JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
      JOIN patient_exclusion_constraints keb
        ON keb.patient_id=? AND keb.kind='bool' AND keb.base_var=cl.base_var
       AND {overlaps_pred_inc('keb','cl', 'COALESCE(keb.tf_lb_inclusive,1)', 'COALESCE(keb.tf_ub_inclusive,1)', cl_lb_inc, cl_ub_inc)} {keb_scope}
      WHERE (cl.is_neg=0 AND {TRUTHY_KEB}) OR (cl.is_neg=1 AND {FALSY_KEB})
      GROUP BY itc.merged_trial_id, cl.clause_id
    """
    # NUM known (no timeframe)
    ikn_sql = """
      SELECT itc.merged_trial_id, cnr.clause_id,
             COUNT(DISTINCT CASE WHEN ken.patient_id IS NOT NULL THEN cnr.member_index END) AS n
      FROM tmp_inclusion_trial_constraint_clauses_why itc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = itc.clause_id
      LEFT JOIN patient_exclusion_constraints ken
        ON ken.patient_id=? AND ken.kind='num' AND ken.base_var=cnr.base_var
      GROUP BY itc.merged_trial_id, cnr.clause_id
    """
    # NUM sat (value within range; inclusive flags respected)
    esn_val_pred = """
      (cnr.lb IS NULL OR CAST(ken.value AS NUMERIC) > cnr.lb OR (CAST(ken.value AS NUMERIC) = cnr.lb AND cnr.lb_inc=1))
      AND (cnr.ub IS NULL OR CAST(ken.value AS NUMERIC) < cnr.ub OR (CAST(ken.value AS NUMERIC) = cnr.ub AND cnr.ub_inc=1))
    """
    isn_sql = f"""
      SELECT itc.merged_trial_id, cnr.clause_id, COUNT(DISTINCT cnr.member_index) AS n
      FROM tmp_inclusion_trial_constraint_clauses_why itc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = itc.clause_id
      JOIN patient_exclusion_constraints ken
        ON ken.patient_id=? AND ken.kind='num' AND ken.base_var=cnr.base_var
      WHERE {esn_val_pred}
      GROUP BY itc.merged_trial_id, cnr.clause_id
    """

    # materialize inclusion kv
    cur.execute("DROP TABLE IF EXISTS tmp_ikb_why")
    cur.execute("DROP TABLE IF EXISTS tmp_ikn_why")
    cur.execute("DROP TABLE IF EXISTS tmp_isb_why")
    cur.execute("DROP TABLE IF EXISTS tmp_isn_why")
    cur.execute(f"CREATE TEMP TABLE tmp_ikb_why AS {ikb_sql}", (args.patient,))
    cur.execute(f"CREATE TEMP TABLE tmp_isb_why AS {isb_sql}", (args.patient,))
    cur.execute(f"CREATE TEMP TABLE tmp_ikn_why AS {ikn_sql}", (args.patient,))
    cur.execute(f"CREATE TEMP TABLE tmp_isn_why AS {isn_sql}", (args.patient,))

    cur.execute("DROP TABLE IF EXISTS tmp_inc_kv_why")
    cur.execute("""
      CREATE TEMP TABLE tmp_inc_kv_why AS
      SELECT itc.merged_trial_id, c.id AS clause_id, c.number_of_clause_members AS n_members,
             COALESCE(ikb.n,0)+COALESCE(ikn.n,0) AS known_count,
             COALESCE(isb.n,0)+COALESCE(isn.n,0) AS sat_count
      FROM tmp_inclusion_trial_constraint_clauses_why itc
      JOIN constraint_clauses c ON c.id = itc.clause_id
      LEFT JOIN tmp_ikb_why ikb ON ikb.merged_trial_id=itc.merged_trial_id AND ikb.clause_id=itc.clause_id
      LEFT JOIN tmp_ikn_why ikn ON ikn.merged_trial_id=itc.merged_trial_id AND ikn.clause_id=itc.clause_id
      LEFT JOIN tmp_isb_why isb ON isb.merged_trial_id=itc.merged_trial_id AND isb.clause_id=itc.clause_id
      LEFT JOIN tmp_isn_why isn ON isn.merged_trial_id=itc.merged_trial_id AND isn.clause_id=itc.clause_id
    """)

    # ---- Exclusion: known/sat counts (BOOL + NUM) ----
    ekb_sql = f"""
      SELECT etc.merged_trial_id, cl.clause_id,
             COUNT(DISTINCT CASE WHEN keb.patient_id IS NOT NULL THEN cl.literal_index END) AS n
      FROM tmp_exclusion_trial_constraint_clauses_why etc
      JOIN constraint_clause_atoms cl ON cl.clause_id = etc.clause_id
      LEFT JOIN patient_exclusion_constraints keb
        ON keb.patient_id=? AND keb.kind='bool' AND keb.base_var=cl.base_var
       AND {overlaps_pred_inc('keb','cl', 'COALESCE(keb.tf_lb_inclusive,1)', 'COALESCE(keb.tf_ub_inclusive,1)', cl_lb_inc, cl_ub_inc)} {keb_scope}
      GROUP BY etc.merged_trial_id, cl.clause_id
    """
    esb_sql = f"""
      SELECT etc.merged_trial_id, cl.clause_id, COUNT(DISTINCT cl.literal_index) AS n
      FROM tmp_exclusion_trial_constraint_clauses_why etc
      JOIN constraint_clause_atoms cl ON cl.clause_id = etc.clause_id
      JOIN patient_exclusion_constraints keb
        ON keb.patient_id=? AND keb.kind='bool' AND keb.base_var=cl.base_var
       AND {overlaps_pred_inc('keb','cl', 'COALESCE(keb.tf_lb_inclusive,1)', 'COALESCE(keb.tf_ub_inclusive,1)', cl_lb_inc, cl_ub_inc)} {keb_scope}
      WHERE (cl.is_neg=0 AND {TRUTHY_KEB}) OR (cl.is_neg=1 AND {FALSY_KEB})
      GROUP BY etc.merged_trial_id, cl.clause_id
    """
    ekn_sql = """
      SELECT etc.merged_trial_id, cnr.clause_id,
             COUNT(DISTINCT CASE WHEN ken.patient_id IS NOT NULL THEN cnr.member_index END) AS n
      FROM tmp_exclusion_trial_constraint_clauses_why etc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = etc.clause_id
      LEFT JOIN patient_exclusion_constraints ken
        ON ken.patient_id=? AND ken.kind='num' AND ken.base_var=cnr.base_var
      GROUP BY etc.merged_trial_id, cnr.clause_id
    """
    esn_sql = f"""
      SELECT etc.merged_trial_id, cnr.clause_id, COUNT(DISTINCT cnr.member_index) AS n
      FROM tmp_exclusion_trial_constraint_clauses_why etc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = etc.clause_id
      JOIN patient_exclusion_constraints ken
        ON ken.patient_id=? AND ken.kind='num' AND ken.base_var=cnr.base_var
      WHERE {esn_val_pred}
      GROUP BY etc.merged_trial_id, cnr.clause_id
    """

    cur.execute("DROP TABLE IF EXISTS tmp_ekb_why")
    cur.execute("DROP TABLE IF EXISTS tmp_ekn_why")
    cur.execute("DROP TABLE IF EXISTS tmp_esb_why")
    cur.execute("DROP TABLE IF EXISTS tmp_esn_why")
    cur.execute(f"CREATE TEMP TABLE tmp_ekb_why AS {ekb_sql}", (args.patient,))
    cur.execute(f"CREATE TEMP TABLE tmp_esb_why AS {esb_sql}", (args.patient,))
    cur.execute(f"CREATE TEMP TABLE tmp_ekn_why AS {ekn_sql}", (args.patient,))
    cur.execute(f"CREATE TEMP TABLE tmp_esn_why AS {esn_sql}", (args.patient,))

    cur.execute("DROP TABLE IF EXISTS tmp_exc_kv_why")
    cur.execute("""
      CREATE TEMP TABLE tmp_exc_kv_why AS
      SELECT etc.merged_trial_id, c.id AS clause_id, c.number_of_clause_members AS n_members,
             COALESCE(ekb.n,0)+COALESCE(ekn.n,0) AS known_count,
             COALESCE(esb.n,0)+COALESCE(esn.n,0) AS sat_count
      FROM tmp_exclusion_trial_constraint_clauses_why etc
      JOIN constraint_clauses c ON c.id = etc.clause_id
      LEFT JOIN tmp_ekb_why ekb ON ekb.merged_trial_id=etc.merged_trial_id AND ekb.clause_id=etc.clause_id
      LEFT JOIN tmp_ekn_why ekn ON ekn.merged_trial_id=etc.merged_trial_id AND ekn.clause_id=etc.clause_id
      LEFT JOIN tmp_esb_why esb ON esb.merged_trial_id=etc.merged_trial_id AND esb.clause_id=etc.clause_id
      LEFT JOIN tmp_esn_why esn ON esn.merged_trial_id=etc.merged_trial_id AND esn.clause_id=etc.clause_id
    """)

    # contradiction sets
    cur.execute("DROP TABLE IF EXISTS tmp_inc_contra_why")
    cur.execute("""
      CREATE TEMP TABLE tmp_inc_contra_why AS
      SELECT DISTINCT merged_trial_id, clause_id
      FROM tmp_inc_kv_why
      WHERE n_members > 0 AND known_count = n_members AND sat_count = 0
    """)
    cur.execute("DROP TABLE IF EXISTS tmp_exc_contra_why")
    cur.execute("""
      CREATE TEMP TABLE tmp_exc_contra_why AS
      SELECT DISTINCT merged_trial_id, clause_id
      FROM tmp_exc_kv_why
      WHERE n_members > 0 AND known_count = n_members AND sat_count = 0
    """)

    # helpers to pretty-print literals/numerics for a clause (with known/sat flags)
    def clause_detail_rows(side: str, tid: int, clause_id: int) -> Dict[str, Any]:
        # BOOL literals with known/sat flags and example overlapping values
        bool_rows = cur.execute(f"""
          WITH lit AS (
            SELECT cl.clause_id, cl.literal_index, cl.base_var, cl.is_neg,
                   cl.tf_lb_hours, cl.tf_ub_hours,
                   {cl_lb_inc} AS tf_lb_inc, {cl_ub_inc} AS tf_ub_inc
            FROM constraint_clause_atoms cl
            WHERE cl.clause_id = ?
          ),
          over AS (
            SELECT l.literal_index,
                   COUNT(DISTINCT keb.tf_token) AS known_hits,
                   SUM(CASE WHEN (l.is_neg=0 AND {TRUTHY_KEB}) OR (l.is_neg=1 AND {FALSY_KEB}) THEN 1 ELSE 0 END) AS sat_hits,
                   GROUP_CONCAT(DISTINCT CAST(keb.value AS TEXT)) AS values_seen
            FROM lit l
            LEFT JOIN patient_exclusion_constraints keb
              ON keb.patient_id=? AND keb.kind='bool' AND keb.base_var=l.base_var
             AND {overlaps_pred_inc('keb','l', 'COALESCE(keb.tf_lb_inclusive,1)', 'COALESCE(keb.tf_ub_inclusive,1)', 'l.tf_lb_inc', 'l.tf_ub_inc')} {keb_scope}
            GROUP BY l.literal_index
          )
          SELECT l.literal_index, l.base_var, l.is_neg,
                 COALESCE(over.known_hits,0) AS known_hits,
                 COALESCE(over.sat_hits,0)   AS sat_hits,
                 COALESCE(over.values_seen,'') AS values_seen
          FROM lit l
          LEFT JOIN over ON over.literal_index = l.literal_index
          ORDER BY l.literal_index
        """, (clause_id, args.patient)).fetchall()

        bool_details = [{
            "literal_index": r[0],
            "base_var": r[1],
            "is_neg": int(r[2]),
            "known_hits": int(r[3]),
            "sat_hits": int(r[4]),
            "values_seen": r[5].split(",") if r[5] else []
        } for r in bool_rows]

        # NUM members with known/sat flags and example numeric values
        # (timeframe ignored)
        num_rows = cur.execute("""
          WITH mem AS (
            SELECT cnr.member_index, cnr.base_var, cnr.lb, cnr.lb_inc, cnr.ub, cnr.ub_inc
            FROM constraint_clause_numeric_range cnr
            WHERE cnr.clause_id = ?
          ),
          kn AS (
            SELECT m.member_index,
                   COUNT(DISTINCT ken.tf_token) AS known_hits,
                   SUM(CASE WHEN
                      (m.lb IS NULL OR CAST(ken.value AS NUMERIC) > m.lb OR (CAST(ken.value AS NUMERIC) = m.lb AND m.lb_inc=1))
                      AND
                      (m.ub IS NULL OR CAST(ken.value AS NUMERIC) < m.ub OR (CAST(ken.value AS NUMERIC) = m.ub AND m.ub_inc=1))
                   THEN 1 ELSE 0 END) AS sat_hits,
                   GROUP_CONCAT(DISTINCT CAST(ken.value AS TEXT)) AS values_seen
            FROM mem m
            LEFT JOIN patient_exclusion_constraints ken
              ON ken.patient_id=? AND ken.kind='num' AND ken.base_var=m.base_var
            GROUP BY m.member_index
          )
          SELECT m.member_index, m.base_var, m.lb, m.lb_inc, m.ub, m.ub_inc,
                 COALESCE(kn.known_hits,0), COALESCE(kn.sat_hits,0),
                 COALESCE(kn.values_seen,'')
          FROM mem m LEFT JOIN kn ON kn.member_index=m.member_index
          ORDER BY m.member_index
        """, (clause_id, args.patient)).fetchall()

        num_details = [{
            "member_index": r[0],
            "base_var": r[1],
            "lb": r[2], "lb_inc": int(r[3]) if r[3] is not None else None,
            "ub": r[4], "ub_inc": int(r[5]) if r[5] is not None else None,
            "known_hits": int(r[6]),
            "sat_hits": int(r[7]),
            "values_seen": r[8].split(",") if r[8] else []
        } for r in num_rows]

        return {"bool_literals": bool_details, "num_members": num_details}

    # build report
    report: List[Dict[str, Any]] = []
    for tid in trial_ids:
        nct = cur.execute("SELECT nct_id FROM trials WHERE id=?", (tid,)).fetchone()
        nct_id = nct[0] if nct else None

        # pull kv rows for this trial
        inc_kv = cur.execute("""
          SELECT clause_id, n_members, known_count, sat_count
          FROM tmp_inc_kv_why
          WHERE merged_trial_id=?
          ORDER BY clause_id
        """, (tid,)).fetchall()
        exc_kv = cur.execute("""
          SELECT clause_id, n_members, known_count, sat_count
          FROM tmp_exc_kv_why
          WHERE merged_trial_id=?
          ORDER BY clause_id
        """, (tid,)).fetchall()

        inc_contra = {(r[0]) for r in cur.execute(
            "SELECT clause_id FROM tmp_inc_contra_why WHERE merged_trial_id=?", (tid,)
        ).fetchall()}
        exc_contra = {(r[0]) for r in cur.execute(
            "SELECT clause_id FROM tmp_exc_contra_why WHERE merged_trial_id=?", (tid,)
        ).fetchall()}

        # details for contradicted constraint_clauses
        inc_details = []
        for (clause_id, nm, kc, sc) in inc_kv:
            contrad = (nm > 0 and kc == nm and sc == 0)
            if contrad:
                inc_details.append({
                    "clause_id": clause_id,
                    "n_members": int(nm),
                    "known_count": int(kc),
                    "sat_count": int(sc),
                    **clause_detail_rows("inclusion", tid, clause_id)
                })
        exc_details = []
        for (clause_id, nm, kc, sc) in exc_kv:
            contrad = (nm > 0 and kc == nm and sc == 0)
            if contrad:
                exc_details.append({
                    "clause_id": clause_id,
                    "n_members": int(nm),
                    "known_count": int(kc),
                    "sat_count": int(sc),
                    **clause_detail_rows("exclusion", tid, clause_id)
                })

        eliminated = bool(inc_contra or exc_contra)
        report.append({
            "trial_id": tid,
            "nct_id": nct_id,
            "eliminated": eliminated,
            "eliminated_by": [s for s,flag in [
                ("inclusion_contradiction", bool(inc_contra)),
                ("exclusion_contradiction", bool(exc_contra)),
            ] if flag],
            "inclusion": {
                "total_clauses": len(inc_kv),
                "contradicted_count": len(inc_contra),
                "contradicted_constraint_clauses": inc_details
            },
            "exclusion": {
                "total_clauses": len(exc_kv),
                "contradicted_count": len(exc_contra),
                "contradicted_constraint_clauses": exc_details
            }
        })

    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
