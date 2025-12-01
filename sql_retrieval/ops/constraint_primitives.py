#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trial_eval_primitives.py — MODE-AWARE + ALT-MODE (act/nonact) version (COMPLETE).

Key points:
- inclusion_gap() root gating DOES NOT use important_table.is_root.
  Root gating comes from patient_inclusion_constraints(kind='bool' AND is_root=1), joined onto important_table
  to form tmp_in_bool_fii_root.

ALT-MODE (NEW):
- Adds --alt-mode {act,nonact} to relevant subcommands.
- nonact switches accepted-alternatives tables:
    disease_hits: daa_table = disease_constraint_alternatives_nonact
    prevention_disease_hits: trial_table = disease_constraint_alternatives_prevent_nonact
    prevention_positive_literal_hits: trial_table = positive_constraint_alternatives_expanded_prevention_nonact
    positive_literal_hits: prefers positive_constraint_alternatives_expanded_nonact

Positive literal retrieval:
- Prefers *_expanded* accepted alternatives tables; falls back to legacy
  positive_constraint_alternatives if expanded doesn’t exist.

Everything else is the same as your pasted version.
"""

from __future__ import annotations
import argparse, csv, json, logging, sqlite3, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal
import re

LOGGER = logging.getLogger("trial_eval_primitives")

# =========================
# Mode/table helpers
# =========================

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _safe_ident(name: str) -> str:
    name = (name or "").strip()
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name

def important_table_for_mode(mode: str) -> str:
    mode = (mode or "all").strip()
    if mode not in {"chief", "ccr", "all"}:
        raise ValueError(f"bad important mode: {mode}")
    return f"patient_inclusion_constraints_important_{mode}"

# =========================
# ALT-MODE helpers (act vs nonact)
# =========================

AltMode = Literal["act", "nonact"]

def _alt_mode_norm(m: str | None) -> AltMode:
    m = (m or "act").strip().lower()
    return "nonact" if m == "nonact" else "act"

def disease_daa_table_for_alt(alt_mode: AltMode) -> str:
    return "disease_constraint_alternatives_nonact" if alt_mode == "nonact" else "disease_constraint_alternatives"

def disease_daa_prevent_table_for_alt(alt_mode: AltMode) -> str:
    return "disease_constraint_alternatives_prevent_nonact" if alt_mode == "nonact" else "disease_constraint_alternatives_prevent"

def pla_expanded_table_for_alt(alt_mode: AltMode) -> str:
    return "positive_constraint_alternatives_expanded_nonact" if alt_mode == "nonact" else "positive_constraint_alternatives_expanded"

def pla_expanded_prevention_table_for_alt(alt_mode: AltMode) -> str:
    return "positive_constraint_alternatives_expanded_prevention_nonact" if alt_mode == "nonact" else "positive_constraint_alternatives_expanded_prevention"

# =========================
# Logging / Dirs
# =========================

def ensure_dirs(out_dir: Optional[Path]) -> None:
    if not out_dir:
        return
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

def configure_logging(out_dir: Optional[Path], verbose: bool) -> None:
    ensure_dirs(out_dir)
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir:
        handlers.append(logging.FileHandler(out_dir / "logs" / "run.log", mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s", handlers=handlers)

# =========================
# Shared helpers
# =========================

TRUTHY_FII = """
(fii.value IS NOT NULL) AND (
  CAST(fii.value AS NUMERIC) = 1
  OR UPPER(CAST(fii.value AS TEXT)) IN ('1','TRUE','T','Y','YES')
)
"""
TRUTHY_FI  = """
(fi.value IS NOT NULL) AND (
  CAST(fi.value AS NUMERIC) = 1
  OR UPPER(CAST(fi.value AS TEXT)) IN ('1','TRUE','T','Y','YES')
)
"""
FALSY_FII = """
(fii.value IS NOT NULL) AND (
  CAST(fii.value AS NUMERIC) = 0
  OR UPPER(CAST(fii.value AS TEXT)) IN ('0','FALSE','F','N','NO')
)
"""

SENTINEL_NEG_INF = -1e15
SENTINEL_POS_INF =  1e15

def _scope_pred(alias: str, scope: str) -> str:
    if scope == "any":
        return ""
    # scope == now → fact interval must include 0
    return (
        f"AND COALESCE({alias}.tf_lb_hours, {SENTINEL_NEG_INF}) <= 0 "
        f"AND COALESCE({alias}.tf_ub_hours,  {SENTINEL_POS_INF}) >= 0"
    )

def table_has(cur: sqlite3.Cursor, table: str) -> bool:
    return bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())

def pick_col(cur: sqlite3.Cursor, table: str, candidates: Sequence[str]) -> Optional[str]:
    rows = cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()
    have = {r[1] for r in rows}
    for c in candidates:
        if c in have:
            return c
    return None

def _materialize(cur: sqlite3.Cursor, name: str, select_sql: str, params: Dict[str, Any]) -> None:
    cur.execute(f"DROP TABLE IF EXISTS tmp_{name}")
    cur.execute(f"CREATE TEMP TABLE tmp_{name} AS {select_sql}", params)

# -------- Inclusive overlap helper (BOOL only) --------
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

def _inc_cols(cur: sqlite3.Cursor, table: str, alias: str) -> tuple[str, str]:
    try:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
    except sqlite3.Error:
        cols = set()
    lb = f"{alias}.tf_lb_inclusive" if "tf_lb_inclusive" in cols else "1"
    ub = f"{alias}.tf_ub_inclusive" if "tf_ub_inclusive" in cols else "1"
    return lb, ub

# =========================
# Primitive 0: geo_race_screen (unchanged)
# =========================

def geo_race_screen(conn: sqlite3.Connection, patient_id: str,
                    candidate_trial_ids: Optional[Iterable[int]] = None) -> List[int]:
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS tmp_candidates")
    cur.execute("CREATE TEMP TABLE tmp_candidates(id INTEGER PRIMARY KEY)")
    if candidate_trial_ids:
        cur.executemany("INSERT INTO tmp_candidates(id) VALUES (?)",
                        [(int(tid),) for tid in set(int(x) for x in candidate_trial_ids)])
    else:
        cur.execute("INSERT INTO tmp_candidates(id) SELECT id FROM trials")

    has_pb   = table_has(cur, "patient_birth_country")
    has_trb  = table_has(cur, "trial_birth_country_allowed")
    has_prc  = table_has(cur, "patient_resident_country")
    has_trrc = table_has(cur, "trial_resident_country_allowed")
    has_pr   = table_has(cur, "patient_race")
    has_trra = table_has(cur, "trial_race_allowed")

    viol_ctes: List[str] = []
    elim_parts: List[str] = []

    if has_pb and has_trb:
        viol_ctes.append("""
        viol_birth AS (
          SELECT i.trial_id
          FROM itc i
          WHERE EXISTS (SELECT 1 FROM patient_birth_country p WHERE p.patient_id = :patient)
            AND EXISTS (SELECT 1 FROM trial_birth_country_allowed a WHERE a.trial_id = i.nct_id)
            AND NOT EXISTS (
              SELECT 1
              FROM patient_birth_country p
              JOIN trial_birth_country_allowed a
                    ON a.trial_id = i.nct_id
                   AND a.country_index = p.country_index
              WHERE p.patient_id = :patient
            )
        )""")
        elim_parts.append("SELECT trial_id, 'birth_country' AS reason FROM viol_birth")

    if has_prc and has_trrc:
        viol_ctes.append("""
        viol_resident AS (
          SELECT i.trial_id
          FROM itc i
          WHERE EXISTS (SELECT 1 FROM patient_resident_country p WHERE p.patient_id = :patient)
            AND EXISTS (SELECT 1 FROM trial_resident_country_allowed a WHERE a.trial_id = i.nct_id)
            AND NOT EXISTS (
              SELECT 1
              FROM patient_resident_country p
              JOIN trial_resident_country_allowed a
                    ON a.trial_id = i.nct_id
                   AND a.country_index = p.country_index
              WHERE p.patient_id = :patient
            )
        )""")
        elim_parts.append("SELECT trial_id, 'resident_country' AS reason FROM viol_resident")

    if has_pr and has_trra:
        viol_ctes.append("""
        viol_race AS (
          SELECT i.trial_id
          FROM itc i
          WHERE EXISTS (SELECT 1 FROM patient_race pr WHERE pr.patient_id = :patient)
            AND EXISTS (SELECT 1 FROM trial_race_allowed tr WHERE tr.trial_id = i.nct_id)
            AND NOT EXISTS (
              SELECT 1
              FROM patient_race pr
              JOIN trial_race_allowed tr
                ON tr.trial_id = i.nct_id
               AND (
                     (pr.race_id   IS NOT NULL AND tr.race_id   IS NOT NULL AND tr.race_id = pr.race_id)
                  OR (pr.race_name IS NOT NULL AND tr.race_name IS NOT NULL
                      AND UPPER(TRIM(pr.race_name)) = UPPER(TRIM(tr.race_name)))
               )
              WHERE pr.patient_id = :patient
            )
        )""")
        elim_parts.append("SELECT trial_id, 'race' AS reason FROM viol_race")

    if not elim_parts:
        rows = cur.execute("SELECT id FROM tmp_candidates ORDER BY id").fetchall()
        return [int(r[0]) for r in rows]

    elim_union = "\nUNION ALL\n".join(elim_parts)

    sql = f"""
    WITH itc AS (
      SELECT t.id AS trial_id, t.nct_id
      FROM trials t
      JOIN tmp_candidates c ON c.id = t.id
    ),
    {",\n".join(viol_ctes)},
    elim AS (
      {elim_union}
    )
    SELECT i.trial_id
    FROM itc i
    LEFT JOIN elim e ON e.trial_id = i.trial_id
    WHERE e.trial_id IS NULL
    ORDER BY i.trial_id
    """
    rows = cur.execute(sql, {"patient": patient_id}).fetchall()
    return [int(r[0]) for r in rows]

def geo_race_violations(conn: sqlite3.Connection, patient_id: str,
                        candidate_trial_ids: Optional[Iterable[int]] = None) -> List[Tuple[int, str]]:
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS tmp_candidates")
    cur.execute("CREATE TEMP TABLE tmp_candidates(id INTEGER PRIMARY KEY)")
    if candidate_trial_ids:
        cur.executemany("INSERT INTO tmp_candidates(id) VALUES (?)",
                        [(int(tid),) for tid in set(int(x) for x in candidate_trial_ids)])
    else:
        cur.execute("INSERT INTO tmp_candidates(id) SELECT id FROM trials")

    has_pb   = table_has(cur, "patient_birth_country")
    has_trb  = table_has(cur, "trial_birth_country_allowed")
    has_prc  = table_has(cur, "patient_resident_country")
    has_trrc = table_has(cur, "trial_resident_country_allowed")
    has_pr   = table_has(cur, "patient_race")
    has_trra = table_has(cur, "trial_race_allowed")

    viol_ctes: List[str] = []
    elim_parts: List[str] = []

    if has_pb and has_trb:
        viol_ctes.append("""
        viol_birth AS (
          SELECT i.trial_id
          FROM itc i
          WHERE EXISTS (SELECT 1 FROM patient_birth_country p WHERE p.patient_id = :patient)
            AND EXISTS (SELECT 1 FROM trial_birth_country_allowed a WHERE a.trial_id = i.nct_id)
            AND NOT EXISTS (
              SELECT 1
              FROM patient_birth_country p
              JOIN trial_birth_country_allowed a
                    ON a.trial_id = i.nct_id
                   AND a.country_index = p.country_index
              WHERE p.patient_id = :patient
            )
        )""")
        elim_parts.append("SELECT trial_id, 'birth_country' AS reason FROM viol_birth")

    if has_prc and has_trrc:
        viol_ctes.append("""
        viol_resident AS (
          SELECT i.trial_id
          FROM itc i
          WHERE EXISTS (SELECT 1 FROM patient_resident_country p WHERE p.patient_id = :patient)
            AND EXISTS (SELECT 1 FROM trial_resident_country_allowed a WHERE a.trial_id = i.nct_id)
            AND NOT EXISTS (
              SELECT 1
              FROM patient_resident_country p
              JOIN trial_resident_country_allowed a
                    ON a.trial_id = i.nct_id
                   AND a.country_index = p.country_index
              WHERE p.patient_id = :patient
            )
        )""")
        elim_parts.append("SELECT trial_id, 'resident_country' AS reason FROM viol_resident")

    if has_pr and has_trra:
        viol_ctes.append("""
        viol_race AS (
          SELECT i.trial_id
          FROM itc i
          WHERE EXISTS (SELECT 1 FROM patient_race pr WHERE pr.patient_id = :patient)
            AND EXISTS (SELECT 1 FROM trial_race_allowed tr WHERE tr.trial_id = i.nct_id)
            AND NOT EXISTS (
              SELECT 1
              FROM patient_race pr
              JOIN trial_race_allowed tr
                ON tr.trial_id = i.nct_id
               AND (
                     (pr.race_id   IS NOT NULL AND tr.race_id   IS NOT NULL AND tr.race_id = pr.race_id)
                  OR (pr.race_name IS NOT NULL AND tr.race_name IS NOT NULL
                      AND UPPER(TRIM(pr.race_name)) = UPPER(TRIM(tr.race_name)))
               )
              WHERE pr.patient_id = :patient
            )
        )""")
        elim_parts.append("SELECT trial_id, 'race' AS reason FROM viol_race")

    if not elim_parts:
        return []

    elim_union = "\nUNION ALL\n".join(elim_parts)
    sql = f"""
    WITH itc AS (
      SELECT t.id AS trial_id, t.nct_id
      FROM trials t
      JOIN tmp_candidates c ON c.id = t.id
    ),
    {",\n".join(viol_ctes)},
    elim AS (
      {elim_union}
    )
    SELECT trial_id, reason
    FROM elim
    ORDER BY trial_id, reason
    """
    rows = cur.execute(sql, {"patient": patient_id}).fetchall()
    return [(int(tid), str(reason)) for (tid, reason) in rows]

# =========================
# Primitive 1: disease_hits (MODE-AWARE via important_table; ALT via daa_table)
# =========================

def satisfy_disease_constraints(conn: sqlite3.Connection, patient_id: str,
                 dli_table: str = "disease_constraint_atoms",
                 daa_table: Optional[str] = "disease_constraint_alternatives",
                 trials_id_col: str = "id",
                 trials_nct_col: str = "nct_id",
                 require_root_for_hops: bool = True,
                 important_table: str = "patient_inclusion_constraints_important_all") -> List[int]:
    important_table = _safe_ident(important_table)
    cur = conn.cursor()

    if not table_has(cur, important_table):
        raise RuntimeError(f"Missing table {important_table} (required for disease_hits).")

    def _join_target_for_key(cur2: sqlite3.Cursor, table: str, key_col: Optional[str]) -> str:
        if not key_col:
            return "itc.nct_id"
        try:
            nct_like = cur2.execute(
                f"SELECT 1 FROM {_safe_ident(table)} WHERE {key_col} LIKE 'NCT%' LIMIT 1;"
            ).fetchone()
            numeric_like = cur2.execute(
                f"SELECT 1 FROM {_safe_ident(table)} WHERE {key_col} GLOB '[0-9]*' AND {key_col} NOT LIKE 'NCT%' LIMIT 1;"
            ).fetchone()
        except sqlite3.Error:
            return "itc.nct_id"
        if nct_like and not numeric_like:
            return "itc.nct_id"
        if numeric_like and not nct_like:
            return "itc.merged_trial_id"
        return "itc.nct_id"

    if not table_has(cur, dli_table):
        raise RuntimeError(f"Missing table {dli_table}")

    dli_trial_col = pick_col(cur, dli_table, ["trial_id", "nct_id"]) or "trial_id"
    dli_var_col   = pick_col(cur, dli_table, ["stem_var", "var_name", "base_var", "var_name_notime"]) or "stem_var"

    fii_has_is_root = pick_col(cur, important_table, ["is_root"]) is not None

    use_daa = False
    daa_trial_col = None
    daa_alt_col = None
    if daa_table and table_has(cur, daa_table):
        daa_trial_col = pick_col(cur, daa_table, ["trial_id", "nct_id"])
        daa_alt_col   = pick_col(cur, daa_table, ["alt_var_name_notime"])
        use_daa = (daa_trial_col is not None) and (daa_alt_col is not None)

    if use_daa:
        daa_has_hop    = pick_col(cur, daa_table, ["hop"]) is not None
        daa_has_reason = pick_col(cur, daa_table, ["reason"]) is not None
        daa_join_right = _join_target_for_key(cur, daa_table, daa_trial_col)

        hop_expr    = f"{_safe_ident(daa_table)}.hop" if daa_has_hop else "NULL"
        reason_expr = f"{_safe_ident(daa_table)}.reason" if daa_has_reason else "NULL"

        if require_root_for_hops and fii_has_is_root:
            hop_gate_sql = "((dv.hop <= 0 OR dv.reason = 'self') OR COALESCE(fii.is_root,0)=1)"
        else:
            hop_gate_sql = "1=1"

        if daa_join_right == "itc.nct_id":
            daa_join_cond = "dv.trial_key = itc.nct_id"
        else:
            daa_join_cond = "dv.trial_key = itc.merged_trial_id"

        sql = f"""
          WITH itc AS (
            SELECT {trials_id_col} AS merged_trial_id, {trials_nct_col} AS nct_id
            FROM trials
          ),
          disease_vars AS (
            SELECT
              {_safe_ident(daa_table)}.{daa_trial_col} AS trial_key,
              {_safe_ident(daa_table)}.{daa_alt_col}   AS base_var,
              COALESCE({hop_expr}, 0)     AS hop,
              COALESCE({reason_expr}, 'self') AS reason
            FROM {_safe_ident(daa_table)}
            WHERE {_safe_ident(daa_table)}.{daa_alt_col} IS NOT NULL
            GROUP BY {_safe_ident(daa_table)}.{daa_trial_col}, {_safe_ident(daa_table)}.{daa_alt_col}
          )
          SELECT DISTINCT itc.merged_trial_id
          FROM itc
          JOIN disease_vars dv ON {daa_join_cond}
          LEFT JOIN {important_table} fii
            ON fii.patient_id=:patient_id
           AND fii.kind='bool'
           AND {TRUTHY_FII}
           AND fii.base_var=dv.base_var
           AND {hop_gate_sql}
          WHERE fii.base_var IS NOT NULL
        """
    else:
        dli_join_right = _join_target_for_key(cur, dli_table, dli_trial_col)

        if dli_join_right == "itc.nct_id":
            dli_join_cond = f"dli.{dli_trial_col} = itc.nct_id"
        else:
            dli_join_cond = f"dli.{dli_trial_col} = itc.merged_trial_id"

        sql = f"""
          WITH itc AS (
            SELECT {trials_id_col} AS merged_trial_id, {trials_nct_col} AS nct_id
            FROM trials
          )
          SELECT DISTINCT itc.merged_trial_id
          FROM itc
          JOIN {_safe_ident(dli_table)} dli ON {dli_join_cond}
          LEFT JOIN {important_table} fii
            ON fii.patient_id=:patient_id
           AND fii.kind='bool'
           AND {TRUTHY_FII}
           AND fii.base_var=dli.{dli_var_col}
          WHERE fii.base_var IS NOT NULL
        """

    rows = cur.execute(sql, {"patient_id": patient_id}).fetchall()
    return sorted({int(r[0]) for r in rows})

# =========================
# Primitive 2: literal_hits (MODE-AWARE via important_table) [legacy]
# =========================

def literal_hits(conn: sqlite3.Connection, patient_id: str, scope: str = "any",
                 require_root_for_hops: bool = True,
                 important_table: str = "patient_inclusion_constraints_important_all") -> List[int]:
    important_table = _safe_ident(important_table)
    cur = conn.cursor()
    if not table_has(cur, important_table):
        raise RuntimeError(f"Missing table {important_table} (required for literal_hits).")

    has_laa = table_has(cur, "constraint_literal_alternatives")
    has_llm = table_has(cur, "constraint_lifted_atoms")
    lifted_source = "constraint_literal_alternatives" if has_laa else ("constraint_lifted_atoms" if has_llm else "")

    fii_has_is_root = pick_col(cur, important_table, ["is_root"]) is not None

    if lifted_source:
        hop_col    = "hop"    if pick_col(cur, lifted_source, ["hop"])    else None
        reason_col = "reason" if pick_col(cur, lifted_source, ["reason"]) else None

        hop_expr    = f"x.{hop_col}"    if hop_col    else "NULL"
        reason_expr = f"x.{reason_col}" if reason_col else "NULL"

        if require_root_for_hops and fii_has_is_root:
            hop_gate_sql = "((la.hop <= 0 OR la.reason = 'self') OR COALESCE(fii.is_root,0)=1)"
        else:
            hop_gate_sql = "1=1"

        lifted_sql = (
            f"""
            SELECT x.trial_id,
                   x.clause_id,
                   x.literal_index,
                   x.timeframe AS timeframe,
                   x.lifted_var,
                   x.lifted_var_stem,
                   x.base_var_stem,
                   COALESCE(x.lifted_var_stem, x.base_var_stem) AS base_var_stem_key,
                   COALESCE({hop_expr}, 0) AS hop,
                   COALESCE({reason_expr}, 'self') AS reason
            FROM {_safe_ident(lifted_source)} x
            """
        )

        sql = f"""
          WITH la AS (
            SELECT mt.id AS merged_trial_id,
                   mt.nct_id,
                   sub.trial_id,
                   sub.clause_id,
                   sub.literal_index,
                   sub.timeframe,
                   sub.lifted_var,
                   sub.lifted_var_stem,
                   sub.base_var_stem,
                   sub.base_var_stem_key,
                   sub.hop,
                   sub.reason
            FROM trials mt
            JOIN ({lifted_sql}) AS sub
              ON sub.trial_id IN (mt.inclusion_trial_side_id, mt.assumed_trial_side_id)
          )
          SELECT DISTINCT la.merged_trial_id
          FROM la
          LEFT JOIN {important_table} fii
            ON fii.patient_id = :patient_id
           AND fii.kind = 'bool'
           AND {TRUTHY_FII}
           AND fii.base_var = la.base_var_stem_key
           AND {hop_gate_sql}
          WHERE fii.base_var IS NOT NULL
        """
    else:
        sql = f"""
          WITH itc AS (
            SELECT mt.id AS merged_trial_id,
                   tc.clause_id,
                   cl.literal_index,
                   cl.is_neg,
                   cl.base_var
            FROM trials mt
            JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
            JOIN constraint_clause_atoms cl ON cl.clause_id = tc.clause_id
            UNION ALL
            SELECT mt.id AS merged_trial_id,
                   tc2.clause_id,
                   cl2.literal_index,
                   cl2.is_neg,
                   cl2.base_var
            FROM trials mt
            JOIN trial_constraint_sides ta
                 ON ta.id = mt.assumed_trial_side_id AND ta.kind = 'inclusion'
            JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
            JOIN constraint_clause_atoms cl2 ON cl2.clause_id = tc2.clause_id
          )
          SELECT DISTINCT itc.merged_trial_id
          FROM itc
          JOIN {important_table} fii
            ON fii.patient_id = :patient_id
           AND fii.kind = 'bool'
           AND fii.base_var = itc.base_var
          WHERE (itc.is_neg = 0 AND {TRUTHY_FII})
             OR (itc.is_neg = 1 AND {FALSY_FII})
        """

    rows = cur.execute(sql, {"patient_id": patient_id}).fetchall()
    return sorted({int(r[0]) for r in rows})

# =========================
# Primitive 2b: positive_literal_hits (MODE-AWARE via important_table; ALT via alt_mode)
# =========================

def satisfy_positive_literal_constraints(conn: sqlite3.Connection, patient_id: str,
                          require_root_for_hops: bool = True,
                          important_table: str = "patient_inclusion_constraints_important_all",
                          alt_mode: str = "act") -> List[int]:
    important_table = _safe_ident(important_table)
    cur = conn.cursor()
    if not table_has(cur, important_table):
        raise RuntimeError(f"Missing table {important_table} (required for positive_literal_hits).")

    am: AltMode = _alt_mode_norm(alt_mode)

    pla_exp = pla_expanded_table_for_alt(am)
    pla_legacy = "positive_constraint_alternatives"

    has_pl  = table_has(cur, "positive_constraint_literals")
    pla_table: Optional[str] = None
    if table_has(cur, pla_exp):
        pla_table = pla_exp
    elif table_has(cur, pla_legacy):
        pla_table = pla_legacy

    if not has_pl and not pla_table:
        return []

    fii_has_is_root = pick_col(cur, important_table, ["is_root"]) is not None

    if require_root_for_hops and fii_has_is_root:
        hop_gate_tpl = "((s.hop <= 0 OR s.reason = 'self') OR COALESCE(fii.is_root,0)=1)"
    else:
        hop_gate_tpl = "1=1"

    def _stem_key_expr(table: str) -> str:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
        if table in {pla_legacy, pla_exp}:
            priority = ["lifted_var_stem","lifted_var","base_var_stem","var_name_notime","base_var","var_name"]
        else:
            priority = ["base_var_stem","var_name_notime","base_var","var_name","lifted_var_stem","lifted_var"]
        present = [c for c in priority if c in cols]
        if not present:
            return "NULL"
        return "COALESCE(" + ",".join(present) + ")"

    def _trial_col(table: str) -> Optional[str]:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
        if "nct_id" in cols:
            return "nct_id"
        if "trial_id" in cols:
            return "trial_id"
        return None

    def _hop_expr(table: str) -> str:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
        return "hop" if "hop" in cols else "NULL"

    def _reason_expr(table: str) -> str:
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()}
        return "reason" if "reason" in cols else "NULL"

    union_parts: List[str] = []

    if has_pl:
        tcol = _trial_col("positive_constraint_literals")
        if tcol:
            union_parts.append(
                f"""
                SELECT
                  {tcol} AS trial_key,
                  {_stem_key_expr("positive_constraint_literals")} AS stem_key,
                  COALESCE({_hop_expr("positive_constraint_literals")}, 0) AS hop,
                  COALESCE({_reason_expr("positive_constraint_literals")}, 'self') AS reason
                FROM positive_constraint_literals
                """
            )

    if pla_table:
        tcol = _trial_col(pla_table)
        if tcol:
            union_parts.append(
                f"""
                SELECT
                  {tcol} AS trial_key,
                  {_stem_key_expr(pla_table)} AS stem_key,
                  COALESCE({_hop_expr(pla_table)}, 0) AS hop,
                  COALESCE({_reason_expr(pla_table)}, 'self') AS reason
                FROM {_safe_ident(pla_table)}
                """
            )

    if not union_parts:
        return []

    union_sql = "\nUNION ALL\n".join(union_parts)

    sql = f"""
      WITH src AS (
        {union_sql}
      )
      SELECT DISTINCT mt.id AS merged_trial_id
      FROM trials mt
      JOIN src s ON s.trial_key = mt.nct_id
      LEFT JOIN {important_table} fii
        ON fii.patient_id = :patient_id
       AND fii.kind = 'bool'
       AND {TRUTHY_FII}
       AND fii.base_var = s.stem_key
       AND {hop_gate_tpl}
      WHERE fii.base_var IS NOT NULL
    """

    rows = cur.execute(sql, {"patient_id": patient_id}).fetchall()
    return sorted({int(r[0]) for r in rows})

# =========================
# Primitive 3: eliminate (UNCHANGED)
# =========================

def check_constraint_contradictions(conn: sqlite3.Connection, patient_id: str, candidate_trial_ids: Iterable[int], scope: str = "any") -> List[int]:
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS tmp_candidates")
    cur.execute("CREATE TEMP TABLE tmp_candidates(id INTEGER PRIMARY KEY)")
    cur.executemany("INSERT INTO tmp_candidates(id) VALUES (?)", [(int(tid),) for tid in set(candidate_trial_ids)])

    keb_scope = _scope_pred("keb", scope)
    cl_lb_inc, cl_ub_inc = _inc_cols(cur, "constraint_clause_atoms", alias="cl")

    demo_bool = """
      SELECT d.patient_id, ('patient_sex_is_'||d.sex) AS base_var, d.tf_token, 1.0 AS value, d.tf_lb_hours, d.tf_ub_hours, d.tf_lb_inclusive, d.tf_ub_inclusive
      FROM patient_demographic_constraints d WHERE d.sex IS NOT NULL
    """
    demo_num = """
      SELECT patient_id, 'patient_age_value_recorded_in_years'  AS base_var, tf_token, age_years  AS value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM patient_demographic_constraints WHERE age_years  IS NOT NULL
      UNION ALL
      SELECT patient_id, 'patient_age_value_recorded_in_months' AS base_var, tf_token, age_months AS value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM patient_demographic_constraints WHERE age_months IS NOT NULL
      UNION ALL
      SELECT patient_id, 'patient_age_value_recorded_in_days'   AS base_var, tf_token, age_days   AS value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM patient_demographic_constraints WHERE age_days   IS NOT NULL
    """
    ex_bool = """
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM patient_exclusion_constraints WHERE kind='bool' AND patient_id = :patient_id
      UNION ALL
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM tmp_demo_bool
    """
    ex_num = """
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM patient_exclusion_constraints WHERE kind='num' AND patient_id = :patient_id
      UNION ALL
      SELECT patient_id, base_var, tf_token, value, tf_lb_hours, tf_ub_hours, tf_lb_inclusive, tf_ub_inclusive FROM tmp_demo_num
    """

    _materialize(cur, "demo_bool", demo_bool, {})
    _materialize(cur, "demo_num",  demo_num,  {})
    _materialize(cur, "ex_knowledge_bool", ex_bool, {"patient_id": patient_id})
    _materialize(cur, "ex_knowledge_num",  ex_num,  {"patient_id": patient_id})

    incl_constraint_clauses = """
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN tmp_candidates c ON c.id = mt.id
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
      UNION ALL
      SELECT mt.id AS merged_trial_id, tc2.clause_id
      FROM trials mt
      JOIN tmp_candidates c ON c.id = mt.id
      JOIN trial_constraint_sides ta ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
      JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
    """
    excl_constraint_clauses = """
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      JOIN tmp_candidates c ON c.id = mt.id
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.exclusion_trial_side_id
    """
    _materialize(cur, "inclusion_trial_constraint_clauses", incl_constraint_clauses, {})
    _materialize(cur, "exclusion_trial_constraint_clauses", excl_constraint_clauses, {})

    ikb = f"""
      SELECT itc.merged_trial_id, cl.clause_id,
             COUNT(DISTINCT CASE WHEN keb.patient_id IS NOT NULL THEN cl.literal_index END) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
      LEFT JOIN tmp_ex_knowledge_bool keb ON keb.patient_id=:patient_id AND keb.base_var=cl.base_var
        AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      GROUP BY itc.merged_trial_id, cl.clause_id
    """
    isb = f"""
      SELECT itc.merged_trial_id, cl.clause_id, COUNT(DISTINCT cl.literal_index) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
      JOIN tmp_ex_knowledge_bool keb ON keb.patient_id=:patient_id AND keb.base_var=cl.base_var
        AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      WHERE (cl.is_neg=0 AND CAST(keb.value AS NUMERIC)=1.0)
         OR (cl.is_neg=1 AND CAST(keb.value AS NUMERIC)=0.0)
      GROUP BY itc.merged_trial_id, cl.clause_id
    """

    ikn = f"""
      SELECT itc.merged_trial_id, cnr.clause_id,
             COUNT(DISTINCT CASE WHEN ken.patient_id IS NOT NULL THEN cnr.member_index END) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = itc.clause_id
      LEFT JOIN tmp_ex_knowledge_num ken ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      GROUP BY itc.merged_trial_id, cnr.clause_id
    """
    esn_overlap_val_pred = """
      (cnr.lb IS NULL OR CAST(ken.value AS NUMERIC) > cnr.lb OR (CAST(ken.value AS NUMERIC) = cnr.lb AND cnr.lb_inc=1))
      AND (cnr.ub IS NULL OR CAST(ken.value AS NUMERIC) < cnr.ub OR (CAST(ken.value AS NUMERIC) = cnr.ub AND cnr.ub_inc=1))
    """
    isn = f"""
      SELECT itc.merged_trial_id, cnr.clause_id, COUNT(DISTINCT cnr.member_index) AS n
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = itc.clause_id
      JOIN tmp_ex_knowledge_num ken ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      WHERE {esn_overlap_val_pred}
      GROUP BY itc.merged_trial_id, cnr.clause_id
    """

    _materialize(cur, "ikb", ikb, {"patient_id": patient_id})
    _materialize(cur, "isb", isb, {"patient_id": patient_id})
    _materialize(cur, "ikn", ikn, {"patient_id": patient_id})
    _materialize(cur, "isn", isn, {"patient_id": patient_id})

    inc_kvss = """
      SELECT itc.merged_trial_id, c.id AS clause_id, c.number_of_clause_members AS number_of_clause_members,
             COALESCE(ikb.n,0)+COALESCE(ikn.n,0) AS known_count,
             COALESCE(isb.n,0)+COALESCE(isn.n,0) AS sat_count
      FROM tmp_inclusion_trial_constraint_clauses itc
      JOIN constraint_clauses c ON c.id = itc.clause_id
      LEFT JOIN tmp_ikb ikb ON ikb.merged_trial_id=itc.merged_trial_id AND ikb.clause_id=itc.clause_id
      LEFT JOIN tmp_ikn ikn ON ikn.merged_trial_id=itc.merged_trial_id AND ikn.clause_id=itc.clause_id
      LEFT JOIN tmp_isb isb ON isb.merged_trial_id=itc.merged_trial_id AND isb.clause_id=itc.clause_id
      LEFT JOIN tmp_isn isn ON isn.merged_trial_id=itc.merged_trial_id AND isn.clause_id=itc.clause_id
    """
    _materialize(cur, "inc_known_vs_sat", inc_kvss, {})

    inc_contra = """
      SELECT DISTINCT merged_trial_id
      FROM tmp_inc_known_vs_sat
      WHERE number_of_clause_members > 0 AND known_count = number_of_clause_members AND sat_count = 0
    """
    _materialize(cur, "inclusion_contradicted_trials", inc_contra, {})

    ekb = f"""
      SELECT etc.merged_trial_id, cl.clause_id,
             COUNT(DISTINCT CASE WHEN keb.patient_id IS NOT NULL THEN cl.literal_index END) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_atoms cl ON cl.clause_id = etc.clause_id
      LEFT JOIN tmp_ex_knowledge_bool keb ON keb.patient_id=:patient_id AND keb.base_var=cl.base_var
        AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      GROUP BY etc.merged_trial_id, cl.clause_id
    """
    esb = f"""
      SELECT etc.merged_trial_id, cl.clause_id, COUNT(DISTINCT cl.literal_index) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_atoms cl ON cl.clause_id = etc.clause_id
      JOIN tmp_ex_knowledge_bool keb ON keb.patient_id=:patient_id AND keb.base_var=cl.base_var
        AND {_overlaps_pred_inc('keb','tf_lb_inclusive','tf_ub_inclusive','cl.tf_lb_hours','cl.tf_ub_hours', cl_lb_inc, cl_ub_inc)} {keb_scope}
      WHERE (cl.is_neg=0 AND CAST(keb.value AS NUMERIC)=1.0)
         OR (cl.is_neg=1 AND CAST(keb.value AS NUMERIC)=0.0)
      GROUP BY etc.merged_trial_id, cl.clause_id
    """

    ekn = f"""
      SELECT etc.merged_trial_id, cnr.clause_id,
             COUNT(DISTINCT CASE WHEN ken.patient_id IS NOT NULL THEN cnr.member_index END) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = etc.clause_id
      LEFT JOIN tmp_ex_knowledge_num ken ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      GROUP BY etc.merged_trial_id, cnr.clause_id
    """
    esn = f"""
      SELECT etc.merged_trial_id, cnr.clause_id, COUNT(DISTINCT cnr.member_index) AS n
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = etc.clause_id
      JOIN tmp_ex_knowledge_num ken ON ken.patient_id=:patient_id AND ken.base_var=cnr.base_var
      WHERE {esn_overlap_val_pred}
      GROUP BY etc.merged_trial_id, cnr.clause_id
    """

    _materialize(cur, "ekb", ekb, {"patient_id": patient_id})
    _materialize(cur, "esb", esb, {"patient_id": patient_id})
    _materialize(cur, "ekn", ekn, {"patient_id": patient_id})
    _materialize(cur, "esn", esn, {"patient_id": patient_id})

    exc_kvss = """
      SELECT etc.merged_trial_id, c.id AS clause_id, c.number_of_clause_members AS number_of_clause_members,
             COALESCE(ekb.n,0)+COALESCE(ekn.n,0) AS known_count,
             COALESCE(esb.n,0)+COALESCE(esn.n,0) AS sat_count
      FROM tmp_exclusion_trial_constraint_clauses etc
      JOIN constraint_clauses c ON c.id = etc.clause_id
      LEFT JOIN tmp_ekb ekb ON ekb.merged_trial_id=etc.merged_trial_id AND ekb.clause_id=etc.clause_id
      LEFT JOIN tmp_ekn ekn ON ekn.merged_trial_id=etc.merged_trial_id AND ekn.clause_id=etc.clause_id
      LEFT JOIN tmp_esb esb ON esb.merged_trial_id=etc.merged_trial_id AND esb.clause_id=etc.clause_id
      LEFT JOIN tmp_esn esn ON esn.merged_trial_id=etc.merged_trial_id AND esn.clause_id=etc.clause_id
    """
    _materialize(cur, "exc_known_vs_sat", exc_kvss, {})

    exc_contra = """
      SELECT DISTINCT merged_trial_id
      FROM tmp_exc_known_vs_sat
      WHERE number_of_clause_members > 0 AND known_count = number_of_clause_members AND sat_count = 0
    """
    _materialize(cur, "exclusion_contradicted_trials_explicit", exc_contra, {})

    sql_survivors = """
      WITH elim AS (
        SELECT merged_trial_id FROM tmp_inclusion_contradicted_trials
        UNION
        SELECT merged_trial_id FROM tmp_exclusion_contradicted_trials_explicit
      )
      SELECT c.id
      FROM tmp_candidates c
      LEFT JOIN elim e ON e.merged_trial_id=c.id
      WHERE e.merged_trial_id IS NULL
      ORDER BY c.id
    """
    rows = cur.execute(sql_survivors).fetchall()
    return [int(r[0]) for r in rows]

# =========================
# Primitive 4: inclusion_gap (UPDATED ROOT SOURCE)
# =========================

def evaluate_constraint_satisfaction_gap(
    conn: sqlite3.Connection,
    patient_id: str,
    scope: str = "any",
    candidate_trial_ids: Optional[Iterable[int]] = None,
    important_table: str = "patient_inclusion_constraints_important_all",
) -> List[Dict[str, Any]]:
    """
    Rank trials by fraction of violated constraint_clauses among all fully-evaluable BOOL constraint_clauses
    (using inclusion + assumed-inclusion constraint_clauses).

    IMPORTANT CHANGE:
    - Root gating DOES NOT come from important_table.is_root anymore.
    - Root gating comes from patient facts table:
        patient_inclusion_constraints(kind='bool' AND is_root=1)

    So:
      tmp_in_bool_fii_root = important_table (bool) JOIN patient_inclusion_constraints_root(bool)
    """
    important_table = _safe_ident(important_table)
    cur = conn.cursor()

    if candidate_trial_ids:
        cur.execute("DROP TABLE IF EXISTS tmp_candidates_gap")
        cur.execute("CREATE TEMP TABLE tmp_candidates_gap(id INTEGER PRIMARY KEY)")
        cur.executemany(
            "INSERT INTO tmp_candidates_gap(id) VALUES (?)",
            [(int(tid),) for tid in set(candidate_trial_ids)],
        )
        trial_filter_join = "JOIN tmp_candidates_gap c ON c.id = mt.id"
    else:
        trial_filter_join = ""

    cur.execute("DROP TABLE IF EXISTS tmp_trials_base_gap")
    cur.execute(
        f"""
        CREATE TEMP TABLE tmp_trials_base_gap AS
        SELECT mt.id AS merged_trial_id
        FROM trials mt
        {trial_filter_join}
        """
    )

    kib_scope = _scope_pred("kib", scope)

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
        if has_demo
        else
        "SELECT NULL AS patient_id, NULL AS base_var, NULL AS tf_token, NULL AS value, "
        "NULL AS tf_lb_hours, NULL AS tf_ub_hours, "
        "NULL AS tf_lb_inclusive, NULL AS tf_ub_inclusive WHERE 0"
    )
    _materialize(cur, "demo_bool", demo_bool, {})

    if has_fi:
        in_bool_fi_demo = """
          SELECT patient_id, base_var, tf_token, value,
                 tf_lb_hours, tf_ub_hours,
                 tf_lb_inclusive, tf_ub_inclusive
          FROM patient_inclusion_constraints
          WHERE kind='bool' AND patient_id = :patient_id
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
    _materialize(cur, "in_bool_fi_demo", in_bool_fi_demo, {"patient_id": patient_id})

    # UPDATED: root gate from patient_inclusion_constraints.is_root, not important_table.is_root.
    if has_fi:
        patient_root_bool = """
          SELECT patient_id, base_var
          FROM patient_inclusion_constraints
          WHERE kind='bool' AND COALESCE(is_root,0)=1
            AND patient_id = :patient_id
          GROUP BY patient_id, base_var
        """
    else:
        patient_root_bool = """
          SELECT NULL AS patient_id, NULL AS base_var
          WHERE 0
        """
    _materialize(cur, "patient_root_bool", patient_root_bool, {"patient_id": patient_id})

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
            AND fii.patient_id = :patient_id
        """
        _materialize(cur, "in_bool_fii_root", in_bool_fii_root, {"patient_id": patient_id})
    else:
        cur.execute("DROP TABLE IF EXISTS tmp_in_bool_fii_root")
        cur.execute(
            """
            CREATE TEMP TABLE tmp_in_bool_fii_root AS
            SELECT NULL AS patient_id, NULL AS base_var, NULL AS tf_token, NULL AS value,
                   NULL AS tf_lb_hours, NULL AS tf_ub_hours,
                   NULL AS tf_lb_inclusive, NULL AS tf_ub_inclusive
            WHERE 0
            """
        )

    incl_constraint_clauses = f"""
      SELECT mt.id AS merged_trial_id, tc.clause_id
      FROM trials mt
      {trial_filter_join}
      JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
      UNION ALL
      SELECT mt.id AS merged_trial_id, tc2.clause_id
      FROM trials mt
      {trial_filter_join}
      JOIN trial_constraint_sides ta
           ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
      JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
    """
    _materialize(cur, "inclusion_trial_constraint_clauses_gap", incl_constraint_clauses, {})

    cur.execute("DROP TABLE IF EXISTS tmp_inclusion_trial_constraint_clauses_gap_nonum")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_inclusion_trial_constraint_clauses_gap_nonum AS
        SELECT itc.merged_trial_id, itc.clause_id
        FROM tmp_inclusion_trial_constraint_clauses_gap itc
        WHERE NOT EXISTS (
          SELECT 1
          FROM constraint_clause_numeric_range cnr
          WHERE cnr.clause_id = itc.clause_id
        )
        """
    )

    cur.execute("DROP TABLE IF EXISTS tmp_clause_bool_size")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_clause_bool_size AS
        SELECT cl.clause_id,
               COUNT(DISTINCT cl.literal_index) AS n_bool_members
        FROM constraint_clause_atoms cl
        GROUP BY cl.clause_id
        """
    )

    has_laa = table_has(cur, "constraint_literal_alternatives")
    has_llm = table_has(cur, "constraint_lifted_atoms")
    lifted_source = "constraint_literal_alternatives" if has_laa else (
        "constraint_lifted_atoms" if has_llm else ""
    )

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
          JOIN constraint_clause_atoms cl ON cl.clause_id = itc.clause_id
          LEFT JOIN la
            ON la.merged_trial_id = itc.merged_trial_id
           AND la.clause_id      = cl.clause_id
           AND la.literal_index  = cl.literal_index
        """
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

    if has_default:
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

    truthy_expr = lambda alias: (
        f"(CAST({alias}.value AS NUMERIC) = 1 "
        f"OR UPPER(CAST({alias}.value AS TEXT)) IN ('1','TRUE','T','Y','YES'))"
    )
    falsy_expr = lambda alias: (
        f"(CAST({alias}.value AS NUMERIC) = 0 "
        f"OR UPPER(CAST({alias}.value AS TEXT)) IN ('0','FALSE','F','N','NO'))"
    )

    if has_default:
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
                    (cl.is_neg = 0 AND {truthy_expr('kib_fd')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_fd')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_root')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NULL
                  AND dv.base_var IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('dv')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('dv')})
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
                    (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_root')})
                  )
                )
                OR
                (
                  kib_root.patient_id IS NULL
                  AND dv.base_var IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('dv')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('dv')})
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
                    (cl.is_neg = 0 AND {truthy_expr('kib_fd')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_fd')})
                  )
                )
                OR
                (
                  kib_fd.patient_id IS NULL
                  AND kib_root.patient_id IS NOT NULL AND
                  (
                    (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                    OR
                    (cl.is_neg = 1 AND {falsy_expr('kib_root')})
                  )
                )
              )
            )
            OR
            (
              sem.sem_is_lifted = 1 AND
              kib_root.patient_id IS NOT NULL AND
              (
                (cl.is_neg = 0 AND {truthy_expr('kib_root')})
                OR
                (cl.is_neg = 1 AND {falsy_expr('kib_root')})
              )
            )
          GROUP BY itc.merged_trial_id, sem.clause_id
        """

    _materialize(cur, "gap_isb", isb, {"patient_id": patient_id})

    inc_kvss = """
      SELECT
        itc.merged_trial_id,
        c.id AS clause_id,
        COALESCE(cbs.n_bool_members, 0) AS number_of_clause_members,
        COALESCE(ikb.n, 0) AS known_count,
        COALESCE(isb.n, 0) AS sat_count
      FROM tmp_inclusion_trial_constraint_clauses_gap_nonum itc
      JOIN constraint_clauses c
        ON c.id = itc.clause_id
      LEFT JOIN tmp_clause_bool_size cbs
        ON cbs.clause_id = itc.clause_id
      LEFT JOIN tmp_gap_ikb ikb
        ON ikb.merged_trial_id = itc.merged_trial_id
       AND ikb.clause_id       = itc.clause_id
      LEFT JOIN tmp_gap_isb isb
        ON isb.merged_trial_id = itc.merged_trial_id
       AND isb.clause_id       = itc.clause_id
    """
    _materialize(cur, "inc_known_vs_sat_gap", inc_kvss, {})

    agg_sql = """
      SELECT
        t.merged_trial_id,
        COALESCE(COUNT(kv.clause_id), 0) AS total_bool_constraint_clauses,
        COALESCE(SUM(
          CASE
            WHEN kv.number_of_clause_members > 0
             AND kv.known_count = kv.number_of_clause_members
            THEN 1 ELSE 0
          END
        ), 0) AS evaluable_constraint_clauses,
        COALESCE(SUM(
          CASE
            WHEN kv.number_of_clause_members > 0
             AND kv.known_count = kv.number_of_clause_members
             AND kv.sat_count = 0
            THEN 1 ELSE 0
          END
        ), 0) AS violated_evaluable_constraint_clauses
      FROM tmp_trials_base_gap t
      LEFT JOIN tmp_inc_known_vs_sat_gap kv
        ON kv.merged_trial_id = t.merged_trial_id
      GROUP BY t.merged_trial_id
    """
    rows = cur.execute(agg_sql).fetchall()

    results: List[Dict[str, Any]] = []
    for tid, total_bool, evaluable, violated in rows:
        total_bool = int(total_bool)
        evaluable = int(evaluable)
        violated = int(violated)
        frac_any = (violated / evaluable) if evaluable > 0 else 0.0
        pct_any = 100.0 * frac_any
        results.append(
            {
                "trial_id": int(tid),
                "total_clauses": evaluable,
                "unsat_any": violated,
                "unsat_explicit": violated,
                "frac_unsat_any": round(frac_any, 6),
                "pct_unsat_any": round(pct_any, 3),
                "pct_unsat_explicit": round(pct_any, 3),
            }
        )

    results.sort(key=lambda r: (r["frac_unsat_any"], -r["total_clauses"], r["trial_id"]))
    return results

# =========================
# Prevention primitives (ALT aware via trial_table selection)
# =========================

TRUTHY_FDP = """
(fdp.value IS NOT NULL) AND (
  CAST(fdp.value AS NUMERIC) = 1
  OR UPPER(CAST(fdp.value AS TEXT)) IN ('1','TRUE','T','Y','YES')
)
"""

def satisfy_prevention_disease_constraints(
    conn: sqlite3.Connection,
    patient_id: str,
    *,
    patient_table: str = "patient_prevention_constraints",
    trial_table: str = "disease_constraint_alternatives_prevent",
    require_root_for_hops: bool = True,
) -> List[int]:
    cur = conn.cursor()

    if not table_has(cur, patient_table) or not table_has(cur, trial_table):
        return []

    p_entity = pick_col(cur, patient_table, ["entity_var", "base_var", "var_name", "var_name_notime"])
    p_is_root = pick_col(cur, patient_table, ["is_root"])
    p_kind = pick_col(cur, patient_table, ["kind"])
    p_val  = pick_col(cur, patient_table, ["value"])

    if not p_entity or not p_val:
        return []

    t_trial = pick_col(cur, trial_table, ["trial_id", "nct_id"])
    if not t_trial:
        return []

    t_key_expr_parts = []
    for c in ["alt_var_name_notime", "stem_var", "var_name_notime", "alt_var_name", "var_name"]:
        if pick_col(cur, trial_table, [c]):
            t_key_expr_parts.append(f"{_safe_ident(trial_table)}.{c}")
    if not t_key_expr_parts:
        return []

    t_key_expr = "COALESCE(" + ", ".join(t_key_expr_parts) + ")"

    t_hop    = pick_col(cur, trial_table, ["hop"])
    t_reason = pick_col(cur, trial_table, ["reason"])
    hop_expr = f"COALESCE({_safe_ident(trial_table)}.{t_hop}, 0)" if t_hop else "0"
    reason_expr = f"COALESCE({_safe_ident(trial_table)}.{t_reason}, 'self')" if t_reason else "'self'"

    kind_guard = f"AND fdp.{p_kind}='bool'" if p_kind else ""
    truthy_pred = TRUTHY_FDP

    if require_root_for_hops and p_is_root:
        root_gate = f"((src.hop <= 0 OR src.reason = 'self') OR COALESCE(fdp.{p_is_root},0)=1)"
    else:
        root_gate = "1=1"

    sql = f"""
      WITH src AS (
        SELECT
          {_safe_ident(trial_table)}.{t_trial} AS nct_id,
          {t_key_expr}            AS stem_key,
          {hop_expr}              AS hop,
          {reason_expr}           AS reason
        FROM {_safe_ident(trial_table)}
        WHERE {t_key_expr} IS NOT NULL
      )
      SELECT DISTINCT t.id
      FROM trials t
      JOIN src
        ON src.nct_id = t.nct_id
      JOIN {_safe_ident(patient_table)} fdp
        ON fdp.patient_id = :patient_id
       {kind_guard}
       AND fdp.{p_entity} = src.stem_key
       AND {truthy_pred}
       AND {root_gate}
    """
    rows = cur.execute(sql, {"patient_id": patient_id}).fetchall()
    return sorted({int(r[0]) for r in rows})

def satisfy_prevention_positive_literal_constraints(
    conn: sqlite3.Connection,
    patient_id: str,
    *,
    patient_table: str = "patient_prevention_constraints",
    trial_table: str = "positive_constraint_alternatives_expanded_prevention",
    require_root_for_hops: bool = True,
) -> List[int]:
    cur = conn.cursor()

    if not table_has(cur, patient_table) or not table_has(cur, trial_table):
        return []

    p_entity = pick_col(cur, patient_table, ["entity_var", "base_var", "var_name", "var_name_notime"])
    p_is_root = pick_col(cur, patient_table, ["is_root"])
    p_kind = pick_col(cur, patient_table, ["kind"])
    p_val  = pick_col(cur, patient_table, ["value"])

    if not p_entity or not p_val:
        return []

    t_nct = pick_col(cur, trial_table, ["nct_id", "trial_id"])
    if not t_nct:
        return []

    key_cols = []
    for c in ["lifted_var_stem", "base_var_stem", "lifted_var", "base_var"]:
        if pick_col(cur, trial_table, [c]):
            key_cols.append(f"{_safe_ident(trial_table)}.{c}")
    if not key_cols:
        return []
    t_key_expr = "COALESCE(" + ", ".join(key_cols) + ")"

    t_hop    = pick_col(cur, trial_table, ["hop"])
    t_reason = pick_col(cur, trial_table, ["reason"])
    hop_expr = f"COALESCE({_safe_ident(trial_table)}.{t_hop}, 0)" if t_hop else "0"
    reason_expr = f"COALESCE({_safe_ident(trial_table)}.{t_reason}, 'self')" if t_reason else "'self'"

    kind_guard = f"AND fdp.{p_kind}='bool'" if p_kind else ""
    truthy_pred = TRUTHY_FDP

    if require_root_for_hops and p_is_root:
        root_gate = f"((src.hop <= 0 OR src.reason = 'self') OR COALESCE(fdp.{p_is_root},0)=1)"
    else:
        root_gate = "1=1"

    sql = f"""
      WITH src AS (
        SELECT
          {_safe_ident(trial_table)}.{t_nct} AS nct_id,
          {t_key_expr}          AS stem_key,
          {hop_expr}            AS hop,
          {reason_expr}         AS reason
        FROM {_safe_ident(trial_table)}
        WHERE {t_key_expr} IS NOT NULL
      )
      SELECT DISTINCT t.id
      FROM trials t
      JOIN src
        ON src.nct_id = t.nct_id
      JOIN {_safe_ident(patient_table)} fdp
        ON fdp.patient_id = :patient_id
       {kind_guard}
       AND fdp.{p_entity} = src.stem_key
       AND {truthy_pred}
       AND {root_gate}
    """
    rows = cur.execute(sql, {"patient_id": patient_id}).fetchall()
    return sorted({int(r[0]) for r in rows})

def satisfy_prevention_constraints(
    conn: sqlite3.Connection,
    patient_id: str,
    *,
    require_root_for_hops: bool = True,
    alt_mode: str = "act",
) -> List[int]:
    am: AltMode = _alt_mode_norm(alt_mode)
    a = satisfy_prevention_disease_constraints(conn, patient_id,
                               trial_table=disease_daa_prevent_table_for_alt(am),
                               require_root_for_hops=require_root_for_hops)
    b = satisfy_prevention_positive_literal_constraints(conn, patient_id,
                                        trial_table=pla_expanded_prevention_table_for_alt(am),
                                        require_root_for_hops=require_root_for_hops)
    return sorted(set(a) | set(b))

# =========================
# Composition: 3-way labels (MODE-AWARE via important_table)
# =========================

def classify_trials(
    conn: sqlite3.Connection,
    patient_id: str,
    candidate_trial_ids: Iterable[int],
    scope: str = "any",
    important_table: str = "patient_inclusion_constraints_important_all",
) -> List[Dict[str, Any]]:
    cand = sorted({int(t) for t in candidate_trial_ids})
    if not cand:
        return []

    survivors = set(eliminate(conn, patient_id, cand, scope=scope))

    gap_rows = inclusion_gap(
        conn,
        patient_id,
        scope=scope,
        candidate_trial_ids=cand,
        important_table=important_table,
    )
    gap_by_tid: Dict[int, Dict[str, Any]] = {int(r["trial_id"]): r for r in gap_rows}

    results: List[Dict[str, Any]] = []
    for tid in cand:
        tid_int = int(tid)
        is_survivor = tid_int in survivors
        r = gap_by_tid.get(
            tid_int,
            {
                "total_clauses": 0,
                "unsat_any": 0,
                "unsat_explicit": 0,
                "frac_unsat_any": 0.0,
                "pct_unsat_any": 0.0,
                "pct_unsat_explicit": 0.0,
            },
        )

        if not is_survivor:
            label = "explicit_contradiction"
        else:
            label = "all_satisfied" if r["unsat_any"] == 0 else "unsatisfied_inclusion"

        results.append({
            "trial_id": tid_int,
            "label": label,
            "total_clauses": int(r["total_clauses"]),
            "unsat_any": int(r["unsat_any"]),
            "unsat_explicit": int(r["unsat_explicit"]),
            "frac_unsat_any": float(r["frac_unsat_any"]),
            "pct_unsat_any": float(r["pct_unsat_any"]),
            "pct_unsat_explicit": float(r["pct_unsat_explicit"]),
        })

    results.sort(key=lambda x: x["trial_id"])
    return results

# =========================
# CLI glue
# =========================

def _read_trials_arg(arg: str) -> List[int]:
    if arg.startswith("@"):
        with open(arg[1:], "r", encoding="utf-8") as f:
            return [int(x.strip()) for x in f if x.strip()]
    if not arg:
        return []
    return [int(x) for x in arg.split(",") if x]

def _emit(ids: List[int], out_dir: Optional[Path], quiet: bool) -> None:
    if quiet:
        for tid in ids:
            print(tid)
        return
    payload = {"trial_ids": ids}
    print(json.dumps(payload, indent=2))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "trials.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(out_dir / "trials.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["trial_id"]); w.writerows([[tid] for tid in ids])

def _emit_gap(rows: List[Dict[str, Any]], out_dir: Optional[Path], quiet: bool) -> None:
    if quiet:
        for r in rows:
            print(f"{r['trial_id']}\t{r['frac_unsat_any']}\t{r['pct_unsat_any']}\t{r['pct_unsat_explicit']}")
        return
    payload = {"rows": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "inclusion_gap.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(out_dir / "inclusion_gap.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial_id","total_clauses","unsat_any","unsat_explicit",
                "frac_unsat_any","pct_unsat_any","pct_unsat_explicit"
            ])
            for r in rows:
                w.writerow([
                    r["trial_id"], r["total_clauses"], r["unsat_any"], r["unsat_explicit"],
                    r["frac_unsat_any"], r["pct_unsat_any"], r["pct_unsat_explicit"]
                ])

def _emit_classify(rows: List[Dict[str, Any]], out_dir: Optional[Path], quiet: bool) -> None:
    if quiet:
        for r in rows:
            print(
                f"{r['trial_id']}\t{r['label']}\t"
                f"{r['frac_unsat_any']}\t{r['pct_unsat_any']}\t{r['pct_unsat_explicit']}"
            )
        return
    payload = {"rows": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "classify.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(out_dir / "classify.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial_id", "label",
                "total_clauses", "unsat_any", "unsat_explicit",
                "frac_unsat_any", "pct_unsat_any", "pct_unsat_explicit",
            ])
            for r in rows:
                w.writerow([
                    r["trial_id"], r["label"],
                    r["total_clauses"], r["unsat_any"], r["unsat_explicit"],
                    r["frac_unsat_any"], r["pct_unsat_any"], r["pct_unsat_explicit"],
                ])

def main():
    ap = argparse.ArgumentParser(description="Five standalone primitives for trial filtering (mode-aware)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_root_toggle(sp: argparse.ArgumentParser):
        sp.add_argument("--allow-hop-without-root", default=False, action="store_true",
                        help="Allow hop>0 accepted-alternative matches without requiring important_table.is_root=1")

    def add_mode(sp: argparse.ArgumentParser):
        sp.add_argument("--important-mode", choices=["chief","ccr","all"], default="all",
                        help="Select important facts mode (table patient_inclusion_constraints_important_{mode})")

    def add_alt(sp: argparse.ArgumentParser):
        sp.add_argument("--alt-mode", choices=["act","nonact"], default="act",
                        help="Accepted-alternatives variant: act vs nonact (*_nonact tables)")

    sp0 = sub.add_parser("geo_race_screen", help="Screen candidates by patient geography & race against trial allowed lists")
    sp0.add_argument("--db", type=Path, required=True)
    sp0.add_argument("--patient", required=True)
    sp0.add_argument("--trials", default="", help="Optional comma list of trial_ids or @file path (default: ALL trials)")
    sp0.add_argument("--explain", action="store_true", help="Also emit eliminated trials with reasons")
    sp0.add_argument("--out", type=Path, default=None)
    sp0.add_argument("--quiet", action="store_true")
    sp0.add_argument("--verbose", action="store_true")

    sp1 = sub.add_parser("disease_hits", help="Positive hits from disease_* tables (mode-aware important_table)")
    sp1.add_argument("--db", type=Path, required=True)
    sp1.add_argument("--patient", required=True)
    add_mode(sp1); add_root_toggle(sp1); add_alt(sp1)
    sp1.add_argument("--out", type=Path, default=None)
    sp1.add_argument("--quiet", action="store_true")
    sp1.add_argument("--verbose", action="store_true")

    sp2 = sub.add_parser("literal_hits", help="Positive literal hits (mode-aware important_table) [legacy]")
    sp2.add_argument("--db", type=Path, required=True)
    sp2.add_argument("--patient", required=True)
    sp2.add_argument("--scope", choices=["now","any"], default="any")
    add_mode(sp2); add_root_toggle(sp2)
    sp2.add_argument("--out", type=Path, default=None)
    sp2.add_argument("--quiet", action="store_true")
    sp2.add_argument("--verbose", action="store_true")

    sp2b = sub.add_parser("positive_literal_hits", help="Positive hits from positive_* tables (mode-aware important_table)")
    sp2b.add_argument("--db", type=Path, required=True)
    sp2b.add_argument("--patient", required=True)
    add_mode(sp2b); add_root_toggle(sp2b); add_alt(sp2b)
    sp2b.add_argument("--out", type=Path, default=None)
    sp2b.add_argument("--quiet", action="store_true")
    sp2b.add_argument("--verbose", action="store_true")

    sp3 = sub.add_parser("eliminate", help="Explicit-contradiction elimination over candidates (patient_exclusion_constraints + demographics)")
    sp3.add_argument("--db", type=Path, required=True)
    sp3.add_argument("--patient", required=True)
    sp3.add_argument("--trials", required=True, help="Comma list of trial_ids or @file path")
    sp3.add_argument("--scope", choices=["now","any"], default="any")
    sp3.add_argument("--out", type=Path, default=None)
    sp3.add_argument("--quiet", action="store_true")
    sp3.add_argument("--verbose", action="store_true")

    sp4 = sub.add_parser("inclusion_gap", help="Rank by violated evaluable inclusion BOOL constraint_clauses (mode-aware important_table)")
    sp4.add_argument("--db", type=Path, required=True)
    sp4.add_argument("--patient", required=True)
    sp4.add_argument("--scope", choices=["now","any"], default="any")
    sp4.add_argument("--trials", default="", help="Optional comma list of trial_ids or @file path")
    add_mode(sp4)
    sp4.add_argument("--out", type=Path, default=None)
    sp4.add_argument("--quiet", action="store_true")
    sp4.add_argument("--verbose", action="store_true")

    sp5 = sub.add_parser("classify", help="Classify: all_satisfied / unsatisfied_inclusion / explicit_contradiction (mode-aware)")
    sp5.add_argument("--db", type=Path, required=True)
    sp5.add_argument("--patient", required=True)
    sp5.add_argument("--scope", choices=["now","any"], default="any")
    sp5.add_argument("--trials", required=True, help="Comma list of trial_ids or @file path")
    add_mode(sp5)
    sp5.add_argument("--out", type=Path, default=None)
    sp5.add_argument("--quiet", action="store_true")
    sp5.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    configure_logging(getattr(args, "out", None), getattr(args, "verbose", False))

    try:
        conn = sqlite3.connect(str(args.db))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    def root_req_from(ns) -> bool:
        return not getattr(ns, "allow_hop_without_root", False)

    def imp_table_from(ns) -> str:
        mode = getattr(ns, "important_mode", "all")
        return important_table_for_mode(mode)

    if args.cmd == "geo_race_screen":
        cand = _read_trials_arg(getattr(args, "trials", "")) if getattr(args, "trials", "") else []
        survivors = geo_race_screen(conn, args.patient, cand if cand else None)
        if getattr(args, "explain", False):
            eliminated_rows = geo_race_violations(conn, args.patient, cand if cand else None)
            elim_map: Dict[int, List[str]] = {}
            for tid, reason in eliminated_rows:
                elim_map.setdefault(tid, []).append(reason)
            eliminated = [{"trial_id": tid, "reasons": sorted(set(rs))} for tid, rs in sorted(elim_map.items())]
            payload = {"patient_id": args.patient,
                       "n_candidates": (len(survivors) + len(elim_map)),
                       "n_survivors": len(survivors),
                       "survivor_trial_ids": survivors,
                       "eliminated": eliminated}
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _emit(survivors, args.out, args.quiet)

    elif args.cmd == "disease_hits":
        am: AltMode = _alt_mode_norm(getattr(args, "alt_mode", "act"))
        ids = disease_hits(conn, args.patient,
                          daa_table=disease_daa_table_for_alt(am),
                          require_root_for_hops=root_req_from(args),
                          important_table=imp_table_from(args))
        _emit(ids, args.out, args.quiet)

    elif args.cmd == "literal_hits":
        ids = literal_hits(conn, args.patient, scope=args.scope,
                           require_root_for_hops=root_req_from(args),
                           important_table=imp_table_from(args))
        _emit(ids, args.out, args.quiet)

    elif args.cmd == "positive_literal_hits":
        ids = positive_literal_hits(conn, args.patient,
                                   require_root_for_hops=root_req_from(args),
                                   important_table=imp_table_from(args),
                                   alt_mode=getattr(args, "alt_mode", "act"))
        _emit(ids, args.out, args.quiet)

    elif args.cmd == "eliminate":
        cand = _read_trials_arg(args.trials)
        if not cand:
            print("[error] --trials is empty", file=sys.stderr)
            sys.exit(2)
        ids = eliminate(conn, args.patient, cand, scope=args.scope)
        _emit(ids, args.out, args.quiet)

    elif args.cmd == "classify":
        cand = _read_trials_arg(args.trials)
        if not cand:
            print("[error] --trials is empty", file=sys.stderr)
            sys.exit(2)
        rows = classify_trials(conn, args.patient, cand, scope=args.scope, important_table=imp_table_from(args))
        _emit_classify(rows, args.out, args.quiet)

    else:  # inclusion_gap
        cand = _read_trials_arg(getattr(args, "trials", "")) if getattr(args, "trials", "") else []
        cand_iter = cand if cand else None
        rows = inclusion_gap(conn, args.patient, scope=args.scope,
                             candidate_trial_ids=cand_iter,
                             important_table=imp_table_from(args))
        _emit_gap(rows, args.out, args.quiet)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)