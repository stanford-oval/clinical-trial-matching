#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_why_hits.py

FAST post-hoc "why hits" recomputation from compose_trial_eval outputs.

UPDATED:
- Adds --alt-mode {act,nonact} and matches compose suffix/layout:
    <retrieved_root>/retrieved_mappings__{mode}__{prevent_tag}__{alt_tag}/merged_json/*.json
- suffix is now:
    __{mode}__{prevent_tag}__{alt_tag}

Everything else unchanged.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------
# Suffix helpers (must match compose_trial_eval.py)
# ---------------------------

def _prevent_tag(enable_prevention_hits: bool) -> str:
    return "prevent" if enable_prevention_hits else "noprevent"

def _alt_tag(alt_mode: str) -> str:
    return "nonact" if (alt_mode or "").strip().lower() == "nonact" else "act"

def _run_suffix(mode: str, enable_prevention_hits: bool, alt_mode: str) -> str:
    return f"__{mode}__{_prevent_tag(enable_prevention_hits)}__{_alt_tag(alt_mode)}"

def _scoped_dirname(prefix: str, mode: str, enable_prevention_hits: bool, alt_mode: str) -> str:
    return f"{prefix}{_run_suffix(mode, enable_prevention_hits, alt_mode)}"


# ---------------------------
# Truthy predicates (match compose script)
# ---------------------------

TRUTHY_FII = """
(fii.value IS NOT NULL) AND (
  CAST(fii.value AS NUMERIC) = 1
  OR UPPER(CAST(fii.value AS TEXT)) IN ('1','TRUE','T','Y','YES')
)
"""
TRUTHY_FINI = """
(fni.value IS NOT NULL) AND (
  CAST(fni.value AS NUMERIC) = 1
  OR UPPER(CAST(fni.value AS TEXT)) IN ('1','TRUE','T','Y','YES')
)
"""


# ---------------------------
# Canonical NCT helper (SQL expr)
# ---------------------------

def _canon_nct_sql(col_sql: str) -> str:
    """
    Canonicalize to the first 11 chars for NCT-like strings (NCT########...),
    else UPPER(TRIM(x)).
    """
    return f"""
    CASE
      WHEN {col_sql} IS NOT NULL
       AND UPPER(SUBSTR(TRIM({col_sql}),1,3))='NCT'
       AND LENGTH(TRIM({col_sql})) >= 11
      THEN UPPER(SUBSTR(TRIM({col_sql}), 1, 11))
      ELSE UPPER(TRIM({col_sql}))
    END
    """


# ---------------------------
# Candidate temp table (id + canon_nct)
# ---------------------------

def _setup_tmp_candidates(cur: sqlite3.Cursor, ids: List[int]) -> None:
    """
    TEMP table:
      tmp_candidates_why(id INTEGER PRIMARY KEY, canon_nct TEXT)
    """
    cur.execute("DROP TABLE IF EXISTS tmp_candidates_why")
    cur.execute("CREATE TEMP TABLE tmp_candidates_why(id INTEGER PRIMARY KEY, canon_nct TEXT)")

    if not ids:
        return

    uniq = sorted(set(int(x) for x in ids))
    cur.executemany("INSERT INTO tmp_candidates_why(id) VALUES (?)", [(x,) for x in uniq])

    cur.execute(
        f"""
        UPDATE tmp_candidates_why
        SET canon_nct = (
          SELECT {_canon_nct_sql("t.nct_id")}
          FROM trials t
          WHERE t.id = tmp_candidates_why.id
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tmp_candidates_why_canon ON tmp_candidates_why(canon_nct)")


def _expand_subcohort_candidates(conn: sqlite3.Connection, merged_ids: List[int]) -> List[int]:
    """
    Given merged trial ids (often rep_trial_id parent ids), expand to include:
      - the ids themselves
      - any child/subcohort trials where trials.rep_trial_id is in merged_ids
    """
    if not merged_ids:
        return []
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tmp_parent_ids")
    cur.execute("CREATE TEMP TABLE tmp_parent_ids(id INTEGER PRIMARY KEY)")
    cur.executemany(
        "INSERT INTO tmp_parent_ids(id) VALUES (?)",
        [(int(x),) for x in sorted(set(merged_ids))],
    )
    rows = cur.execute(
        """
        SELECT id FROM trials
        WHERE id IN (SELECT id FROM tmp_parent_ids)
           OR rep_trial_id IN (SELECT id FROM tmp_parent_ids)
        """
    ).fetchall()
    return sorted({int(r[0]) for r in rows})


# ---------------------------
# Explain: disease hits → edges (restricted to tmp_candidates_why)
# ---------------------------

def explain_disease_hits(
    conn: sqlite3.Connection,
    patient: str,
    important_table: str,
    *,
    alt_mode: str = "act",
) -> Dict[int, Dict[str, Any]]:
    """
    Uses existing tmp_candidates_why temp table (must be set up by caller).

    Semantics:
      - trial_literal is ALWAYS hop=0 (trial-side) concept
      - trial_literal_hit is what matched patient fact
    """
    cur = conn.cursor()

    have_fii = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (important_table,),
    ).fetchone())
    if not have_fii:
        return {}

    have_fi_noisa = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_inclusion_constraints_noisa'"
    ).fetchone())

    if not cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_atoms'"
    ).fetchone():
        return {}

    dli_cols = {r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_atoms)").fetchall()}
    dli_trial_col = "trial_id" if "trial_id" in dli_cols else ("nct_id" if "nct_id" in dli_cols else None)
    dli_var_col = next((c for c in ["stem_var", "var_name", "base_var", "var_name_notime"] if c in dli_cols), None)
    if not dli_trial_col or not dli_var_col:
        return {}

    # ALT-aware accepted-alternatives table
    atag = _alt_tag(alt_mode)
    daa_table = "disease_constraint_alternatives_nonact" if atag == "nonact" else "disease_constraint_alternatives"

    has_daa = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (daa_table,),
    ).fetchone())

    daa_sql = ""
    if has_daa:
        daa_cols = {r[1] for r in cur.execute(f"PRAGMA table_info({daa_table})").fetchall()}
        daa_trial_col = "trial_id" if "trial_id" in daa_cols else ("nct_id" if "nct_id" in daa_cols else None)

        hop0_cands = [c for c in ["stem_var", "var_name_notime", "var_name", "base_var"] if c in daa_cols]
        hit_cands  = [c for c in ["alt_var_name_notime", "alt_var_name", "var_name_notime", "var_name", "stem_var"] if c in daa_cols]

        if daa_trial_col and hit_cands:
            daa_hop0 = "COALESCE(" + ", ".join([f"daa.{c}" for c in hop0_cands]) + ")" if hop0_cands else "COALESCE(" + ", ".join([f"daa.{c}" for c in hit_cands]) + ")"
            daa_hit  = "COALESCE(" + ", ".join([f"daa.{c}" for c in hit_cands]) + ")"
            daa_hop = "COALESCE(daa.hop,0)" if "hop" in daa_cols else "0"
            daa_reason = "COALESCE(daa.reason,'self')" if "reason" in daa_cols else "'self'"

            daa_sql = f"""
            UNION ALL
            SELECT
              daa.{daa_trial_col} AS trial_key,
              {daa_hop0}          AS trial_literal_hop0,
              {daa_hit}           AS trial_literal_hit,
              {daa_hop}           AS hop,
              {daa_reason}        AS reason
            FROM {daa_table} daa
            WHERE {daa_hit} IS NOT NULL
            """

    fii_cols = {r[1] for r in cur.execute(f"PRAGMA table_info({important_table})").fetchall()}
    fii_is_root_col = "is_root" if "is_root" in fii_cols else None

    fni_is_root_col = None
    if have_fi_noisa:
        fni_cols = {r[1] for r in cur.execute("PRAGMA table_info(patient_inclusion_constraints_noisa)").fetchall()}
        fni_is_root_col = "is_root" if "is_root" in fni_cols else None

    fii_root_gate = "0"
    if fii_is_root_col:
        fii_root_gate = "COALESCE(fii.is_root,0)=1"

    sql = f"""
      WITH disease_vars_raw AS (
        SELECT
          dli.{dli_trial_col} AS trial_key,
          dli.{dli_var_col}   AS trial_literal_hop0,
          dli.{dli_var_col}   AS trial_literal_hit,
          0                   AS hop,
          'self'              AS reason
        FROM disease_constraint_atoms dli
        WHERE dli.{dli_var_col} IS NOT NULL

        {daa_sql}
      ),
      disease_vars AS (
        SELECT
          trial_key,
          {_canon_nct_sql("trial_key")} AS canon_nct,
          trial_literal_hop0,
          trial_literal_hit,
          COALESCE(hop,0) AS hop,
          COALESCE(reason,'self') AS reason
        FROM disease_vars_raw
        WHERE trial_literal_hit IS NOT NULL
      )
      SELECT
        c.id AS merged_trial_id,
        t.nct_id,
        dv.trial_key,
        dv.canon_nct,
        dv.trial_literal_hop0,
        dv.trial_literal_hit,
        dv.hop,
        dv.reason,

        CASE
          WHEN fii.base_var IS NOT NULL THEN 'important'
          WHEN fni.base_var IS NOT NULL THEN 'noisa'
          ELSE NULL
        END AS patient_table_tag,

        COALESCE(fii.base_var, fni.base_var) AS patient_base_var,
        COALESCE(fii.value,    fni.value)    AS patient_value,
        COALESCE(fii.tf_token, fni.tf_token) AS patient_tf_token,
        COALESCE(fii.tf_lb_hours, fni.tf_lb_hours) AS patient_tf_lb_hours,
        COALESCE(fii.tf_ub_hours, fni.tf_ub_hours) AS patient_tf_ub_hours,

        {("fii.is_root" if fii_is_root_col else "NULL")} AS patient_is_root_important,
        {("fni.is_root" if (have_fi_noisa and fni_is_root_col) else "NULL")} AS patient_is_root_noisa

      FROM tmp_candidates_why c
      JOIN trials t
        ON t.id = c.id
      JOIN disease_vars dv
        ON dv.canon_nct = c.canon_nct

      LEFT JOIN {important_table} fii
        ON fii.patient_id=:patient AND fii.kind='bool' AND {TRUTHY_FII}
       AND fii.base_var = dv.trial_literal_hit

      {(
        f"""
      LEFT JOIN patient_inclusion_constraints_noisa fni
        ON fni.patient_id=:patient AND fni.kind='bool' AND {TRUTHY_FINI}
       AND fni.base_var = dv.trial_literal_hit
        """ if have_fi_noisa else
        """
      LEFT JOIN (
        SELECT NULL AS patient_id, NULL AS kind, NULL AS base_var, NULL AS value,
               NULL AS tf_token, NULL AS tf_lb_hours, NULL AS tf_ub_hours, NULL AS is_root
        WHERE 0
      ) fni ON 1=0
        """
      )}

      WHERE
        (
          COALESCE(dv.hop,0)=0
          AND fii.base_var IS NOT NULL
        )
        OR
        (
          COALESCE(dv.hop,0)<>0
          AND (
            fni.base_var IS NOT NULL
            OR (fii.base_var IS NOT NULL AND ({fii_root_gate}))
          )
        )
    """

    out: Dict[int, Dict[str, Any]] = {}
    rows = cur.execute(sql, {"patient": patient}).fetchall()

    for (
        tid, nct, tkey, canon_nct, tlit0, tlit_hit, hop, reason,
        ptag, pvar, pval, ptf, plb, pub, proot_imp, proot_noisa
    ) in rows:
        tid_int = int(tid)
        bucket = out.setdefault(tid_int, {"nct_id": nct, "edges": []})

        patient_table = (
            important_table if ptag == "important"
            else ("patient_inclusion_constraints_noisa" if ptag == "noisa" else None)
        )
        patient_is_root = proot_imp if ptag == "important" else proot_noisa

        hop_i = int(hop) if hop is not None else 0
        reason_s = str(reason) if reason is not None else "self"

        edge = {
            "source": "disease",
            "trial_key_raw": tkey,
            "trial_key_canon": canon_nct,
            "trial_literal": tlit0,
            "trial_literal_hit": tlit_hit,
            "hop": hop_i,
            "reason": reason_s,
            "patient_table": patient_table,
            "patient_base_var": pvar,
            "patient_value": pval,
            "patient_tf_token": ptf,
            "patient_tf_lb_hours": plb,
            "patient_tf_ub_hours": pub,
            "patient_is_root": patient_is_root,
            "hop_gated": bool(hop_i > 0 and reason_s != "self"),
        }
        bucket["edges"].append(edge)

    # de-dupe edges per trial
    for v in out.values():
        seen = set()
        dedup = []
        for e in v["edges"]:
            key = (
                e.get("source"),
                e.get("trial_key_raw"),
                e.get("trial_key_canon"),
                e.get("trial_literal"),
                e.get("trial_literal_hit"),
                e.get("patient_table"),
                e.get("patient_base_var"),
                str(e.get("patient_value")),
                str(e.get("patient_tf_token")),
                str(e.get("patient_tf_lb_hours")),
                str(e.get("patient_tf_ub_hours")),
                str(e.get("patient_is_root")),
                str(e.get("hop")),
                str(e.get("reason")),
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(e)
        v["edges"] = dedup

    return out


# ---------------------------
# Explain: positive literal hits → edges (restricted to tmp_candidates_why)
# ---------------------------

def explain_positive_literal_hits(
    conn: sqlite3.Connection,
    patient: str,
    important_table: str,
    *,
    alt_mode: str = "act",
) -> Dict[int, Dict[str, Any]]:
    """
    Uses existing tmp_candidates_why temp table (must be set up by caller).

    ALT-aware:
      prefers positive_constraint_alternatives_expanded{_nonact}
      falls back to positive_constraint_alternatives
    """
    cur = conn.cursor()

    if not cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (important_table,),
    ).fetchone():
        return {}

    atag = _alt_tag(alt_mode)
    pla_exp = "positive_constraint_alternatives_expanded_nonact" if atag == "nonact" else "positive_constraint_alternatives_expanded"
    pla_legacy = "positive_constraint_alternatives"

    has_pl = bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='positive_constraint_literals'").fetchone())
    use_pla: Optional[str] = None
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (pla_exp,)).fetchone():
        use_pla = pla_exp
    elif cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (pla_legacy,)).fetchone():
        use_pla = pla_legacy

    if not has_pl and not use_pla:
        return {}

    fii_cols = {r[1] for r in cur.execute(f"PRAGMA table_info({important_table})").fetchall()}
    fii_is_root_col = "is_root" if "is_root" in fii_cols else None

    def _have_cols(table: str) -> Set[str]:
        return {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}

    parts: List[str] = []

    if has_pl:
        have = _have_cols("positive_constraint_literals")
        trial_col = "nct_id" if "nct_id" in have else ("trial_id" if "trial_id" in have else None)
        lit_col = next((c for c in ["base_var_stem", "var_name_notime", "base_var", "var_name"] if c in have), None)
        if trial_col and lit_col:
            parts.append(
                f"""
                SELECT
                  {trial_col} AS trial_key,
                  {lit_col}   AS trial_literal_hop0,
                  {lit_col}   AS trial_literal_hit,
                  0           AS hop,
                  'self'      AS reason
                FROM positive_constraint_literals
                WHERE {lit_col} IS NOT NULL
                """
            )

    if use_pla:
        have = _have_cols(use_pla)
        trial_col = "nct_id" if "nct_id" in have else ("trial_id" if "trial_id" in have else None)

        hop0_cols = [f"{c}" for c in ["base_var_stem", "var_name_notime", "base_var", "var_name"] if c in have]
        hit_cols  = [f"{c}" for c in ["lifted_var_stem", "lifted_var", "alt_var_name_notime", "alt_var_name", "var_name_notime", "base_var_stem", "base_var"] if c in have]

        if trial_col and hit_cols:
            hop0_expr = "COALESCE(" + ", ".join(hop0_cols) + ")" if hop0_cols else "COALESCE(" + ", ".join(hit_cols) + ")"
            hit_expr  = "COALESCE(" + ", ".join(hit_cols) + ")"

            hop_expr = "COALESCE(hop,0)" if "hop" in have else "0"
            reason_expr = "COALESCE(reason,'self')" if "reason" in have else "'self'"

            parts.append(
                f"""
                SELECT
                  {trial_col} AS trial_key,
                  {hop0_expr} AS trial_literal_hop0,
                  {hit_expr}  AS trial_literal_hit,
                  {hop_expr}  AS hop,
                  {reason_expr} AS reason
                FROM {use_pla}
                WHERE {hit_expr} IS NOT NULL
                """
            )

    if not parts:
        return {}

    union_sql = "\nUNION ALL\n".join(parts)

    sql = f"""
      WITH src_raw AS ({union_sql}),
      src AS (
        SELECT
          trial_key,
          {_canon_nct_sql("trial_key")} AS canon_nct,
          trial_literal_hop0,
          trial_literal_hit,
          COALESCE(hop,0) AS hop,
          COALESCE(reason,'self') AS reason
        FROM src_raw
        WHERE trial_literal_hit IS NOT NULL
      )
      SELECT
        c.id AS merged_trial_id,
        t.nct_id,
        s.trial_key,
        s.canon_nct,
        s.trial_literal_hop0,
        s.trial_literal_hit,
        s.hop,
        s.reason,

        fii.base_var AS patient_base_var,
        fii.value AS patient_value,
        fii.tf_token AS patient_tf_token,
        fii.tf_lb_hours AS patient_tf_lb_hours,
        fii.tf_ub_hours AS patient_tf_ub_hours,
        {("fii.is_root" if fii_is_root_col else "NULL")} AS patient_is_root

      FROM tmp_candidates_why c
      JOIN trials t
        ON t.id = c.id
      JOIN src s
        ON s.canon_nct = c.canon_nct

      JOIN {important_table} fii
        ON fii.patient_id=:patient AND fii.kind='bool' AND {TRUTHY_FII}
       AND fii.base_var = s.trial_literal_hit
    """

    out: Dict[int, Dict[str, Any]] = {}
    for (
        tid, nct, tkey, canon_nct, tlit0, tlit_hit, hop, reason,
        pvar, pval, ptf, plb, pub, proot
    ) in cur.execute(sql, {"patient": patient}).fetchall():

        tid_int = int(tid)
        bucket = out.setdefault(tid_int, {"nct_id": nct, "edges": []})

        hop_i = int(hop) if hop is not None else 0
        reason_s = str(reason) if reason is not None else "self"

        edge = {
            "source": "positive_literal",
            "trial_key_raw": tkey,
            "trial_key_canon": canon_nct,
            "trial_literal": tlit0,
            "trial_literal_hit": tlit_hit,
            "hop": hop_i,
            "reason": reason_s,
            "patient_table": important_table,
            "patient_base_var": pvar,
            "patient_value": pval,
            "patient_tf_token": ptf,
            "patient_tf_lb_hours": plb,
            "patient_tf_ub_hours": pub,
            "patient_is_root": proot,
            "hop_gated": bool(hop_i > 0 and reason_s != "self"),
        }
        bucket["edges"].append(edge)

    # de-dupe edges per trial
    for v in out.values():
        seen = set()
        dedup = []
        for e in v["edges"]:
            key = (
                e.get("trial_key_raw"),
                e.get("trial_key_canon"),
                e.get("trial_literal"),
                e.get("trial_literal_hit"),
                e.get("patient_base_var"),
                str(e.get("patient_value")),
                str(e.get("patient_tf_token")),
                str(e.get("patient_tf_lb_hours")),
                str(e.get("patient_tf_ub_hours")),
                str(e.get("patient_is_root")),
                str(e.get("hop")),
                str(e.get("reason")),
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(e)
        v["edges"] = dedup

    return out


# ---------------------------
# Read merged_json and get candidate trial ids
# ---------------------------

def _extract_candidate_ids(merged_payload: Dict[str, Any], use: str) -> List[int]:
    trials = merged_payload.get("trials", []) or []
    ids: Set[int] = set()

    for row in trials:
        if use == "rep_trial_id":
            rt = row.get("rep_trial_id")
            if rt is not None:
                try:
                    ids.add(int(rt))
                except Exception:
                    pass
        elif use == "all_trial_ids":
            for t in (row.get("trial_ids", []) or []):
                try:
                    ids.add(int(t))
                except Exception:
                    pass
        else:
            raise ValueError(f"bad --use: {use}")

    return sorted(ids)


# ---------------------------
# Emit artifacts
# ---------------------------

def _write_hits_outputs(
    out_dir: Path,
    patient_id: str,
    suffix: str,
    mode: str,
    prevent_tag: str,
    alt_tag: str,
    important_table: str,
    disease_why: Dict[int, Dict[str, Any]],
    poslit_why: Dict[int, Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{patient_id}{suffix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fo:

        def emit_edge(trial_id: int, nct_id: Optional[str], edge: Dict[str, Any]) -> None:
            row = {
                "patient_id": patient_id,
                "mode": mode,
                "prevent_tag": prevent_tag,
                "alt_mode": alt_tag,
                "important_table": important_table,
                "trial_id": int(trial_id),
                "nct_id": nct_id,
                **edge,
            }
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")

        for tid, info in sorted(disease_why.items()):
            nct = info.get("nct_id")
            for e in info.get("edges", []) or []:
                emit_edge(tid, nct, e)

        for tid, info in sorted(poslit_why.items()):
            nct = info.get("nct_id")
            for e in info.get("edges", []) or []:
                emit_edge(tid, nct, e)

    agg: Dict[str, Any] = {}
    trial_ids = sorted(set(disease_why.keys()) | set(poslit_why.keys()))
    for tid in trial_ids:
        nct = (disease_why.get(tid, {}).get("nct_id") or poslit_why.get(tid, {}).get("nct_id"))
        agg[str(tid)] = {
            "trial_id": tid,
            "nct_id": nct,
            "disease_edges": disease_why.get(tid, {}).get("edges", []) or [],
            "positive_literal_edges": poslit_why.get(tid, {}).get("edges", []) or [],
        }

    json_path = out_dir / f"{patient_id}{suffix}.json"
    json_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "mode": mode,
                "prevent_tag": prevent_tag,
                "alt_mode": alt_tag,
                "important_table": important_table,
                "by_trial_id": agg,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompute and save 'why hits' edges for retrieved trials by reading merged_json outputs."
    )
    ap.add_argument("--db", type=Path, default="../../build/trial.db", help="Path to trial.db")
    ap.add_argument(
        "--retrieved-root",
        type=Path,
        default="./out_compose",
        help="The same --out directory you used for compose_trial_eval.py",
    )
    ap.add_argument(
        "--important-mode",
        choices=["chief", "ccr", "all"],
        default="all",
        help="Select important facts table: patient_inclusion_constraints_important_{mode}",
    )
    ap.add_argument(
        "--enable-prevention-hits",
        action="store_true",
        help="Use prevent suffix to locate retrieved artifacts (does not change why logic here).",
    )
    ap.add_argument(
        "--alt-mode",
        choices=["act", "nonact"],
        default="act",
        help="Match compose suffix/layout: __{mode}__{prevent_tag}__{alt_tag}",
    )
    ap.add_argument(
        "--use",
        choices=["rep_trial_id", "all_trial_ids"],
        default="rep_trial_id",
        help="Which trial ids from merged_json to explain.",
    )
    ap.add_argument(
        "--expand-subcohorts",
        action="store_true",
        help="If using rep_trial_id candidates, also include child trials where trials.rep_trial_id matches.",
    )
    ap.add_argument(
        "--patients",
        default="",
        help="Optional comma list of patient_ids to process; default: all files in merged_json/",
    )

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-overwrite", action="store_true", help="If set, skip patients that already have outputs.")
    g.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (default behavior unless --no-overwrite is set).")

    args = ap.parse_args()

    mode = args.important_mode
    prevent_tag = _prevent_tag(args.enable_prevention_hits)
    alt_tag = _alt_tag(args.alt_mode)
    suffix = _run_suffix(mode, args.enable_prevention_hits, alt_tag)
    important_table = f"patient_inclusion_constraints_important_{mode}"

    scoped = args.retrieved_root / _scoped_dirname("retrieved_mappings", mode, args.enable_prevention_hits, alt_tag)
    merged_json_dir = scoped / "merged_json"
    if not merged_json_dir.exists():
        print(f"[error] missing merged_json dir: {merged_json_dir}", file=sys.stderr)
        sys.exit(2)

    hits_out_dir = scoped / "hits"
    hits_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(args.db))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    want_patients: Optional[Set[str]] = None
    if args.patients.strip():
        want_patients = {p.strip() for p in args.patients.split(",") if p.strip()}

    files = sorted(merged_json_dir.glob(f"*{suffix}.json"))

    if want_patients is not None:
        filtered: List[Path] = []
        for p in files:
            stem = p.stem
            if not stem.endswith(suffix):
                continue
            pid = stem[:-len(suffix)]
            if pid in want_patients:
                filtered.append(p)
        files = filtered

    if not files:
        print(f"[warn] no merged_json files found matching *{suffix}.json in {merged_json_dir}", file=sys.stderr)
        conn.close()
        sys.exit(0)

    n_ok = 0
    for fp in files:
        patient_id = fp.stem[:-len(suffix)]  # strip suffix from stem

        out_jsonl = hits_out_dir / f"{patient_id}{suffix}.jsonl"
        out_json = hits_out_dir / f"{patient_id}{suffix}.json"
        if args.no_overwrite and out_jsonl.exists() and out_json.exists():
            continue

        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] failed to parse {fp.name}: {e}", file=sys.stderr)
            continue

        cand_ids = _extract_candidate_ids(payload, use=args.use)
        if args.expand_subcohorts:
            cand_ids = _expand_subcohort_candidates(conn, cand_ids)

        _setup_tmp_candidates(conn.cursor(), cand_ids)

        if not cand_ids:
            _write_hits_outputs(
                hits_out_dir,
                patient_id,
                suffix,
                mode,
                prevent_tag,
                alt_tag,
                important_table,
                disease_why={},
                poslit_why={},
            )
            n_ok += 1
            continue

        disease_why = explain_disease_hits(conn, patient_id, important_table, alt_mode=alt_tag)
        poslit_why = explain_positive_literal_hits(conn, patient_id, important_table, alt_mode=alt_tag)

        _write_hits_outputs(
            hits_out_dir,
            patient_id,
            suffix,
            mode,
            prevent_tag,
            alt_tag,
            important_table,
            disease_why=disease_why,
            poslit_why=poslit_why,
        )
        n_ok += 1

    conn.close()
    print(f"[ok] wrote why-hit edges for {n_ok} patients into: {hits_out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)