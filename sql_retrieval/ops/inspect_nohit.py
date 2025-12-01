#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_nohit.py — Inspect why a (patient, trial) pair did *not* retrieve.

Focus: retrieval stage only
  disease_hits / literal_hits / positive_literal_hits

For a given patient_id + trial, show for each path:
  - candidate base vars (from disease_* / literal_* / positive_* tables)
  - matching patient_inclusion_constraints_important rows (if any)
  - whether they are truthy
  - whether they pass the hop/is_root gate

Usage:
  # Single specific trial row (by trials.id)
  python inspect_nohit.py --db path/to/trial.db --patient P --trial-id 341 --pretty

  # Canonical NCT: will inspect all subcohorts whose nct_id matches / starts with this
  python inspect_nohit.py --db path/to/trial.db --patient P --trial-id NCT00118651 --pretty

  # Toggle hop-root requirement
  python inspect_nohit.py --db ... --patient P --trial-id NCT00118651 --allow-hop-without-root
"""

from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

from sql_retrieval.ops import constraint_primitives as tep

# ----------------------------
# Shared helpers
# ----------------------------

TRUTHY_STRINGS = {"1", "TRUE", "T", "Y", "YES"}


def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    try:
        if float(s) == 1.0:
            return True
    except ValueError:
        pass
    return s.upper() in TRUTHY_STRINGS


def _table_has(cur: sqlite3.Cursor, name: str) -> bool:
    return tep.table_has(cur, name)


def _pick_col(cur: sqlite3.Cursor, table: str, candidates) -> str | None:
    return tep.pick_col(cur, table, candidates)


def _join_target_for_key(cur: sqlite3.Cursor, table: str, key_col: str) -> str:
    """
    Copy of the heuristic used inside disease_hits(): decide whether table.{key_col}
    is NCT-like or numeric-like, and therefore should be joined to trials.nct_id or trials.id.
    """
    try:
        nct_like = cur.execute(
            f"SELECT 1 FROM {table} WHERE {key_col} LIKE 'NCT%' LIMIT 1"
        ).fetchone()
        numeric_like = cur.execute(
            f"SELECT 1 FROM {table} WHERE {key_col} GLOB '[0-9]*' "
            f"AND {key_col} NOT LIKE 'NCT%' LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return "nct_id"
    if nct_like and not numeric_like:
        return "nct_id"
    if numeric_like and not nct_like:
        return "id"
    return "nct_id"


def _get_trial_row(cur: sqlite3.Cursor, trial_id: int) -> Dict[str, Any]:
    row = cur.execute(
        "SELECT id, nct_id, inclusion_trial_side_id, assumed_trial_side_id "
        "FROM trials WHERE id = ?",
        (trial_id,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"trial_id={trial_id} not found in trials")
    return {
        "id": int(row[0]),
        "nct_id": row[1],
        "inc_side": row[2],
        "ass_side": row[3],
    }


def _has_is_root(cur: sqlite3.Cursor) -> bool:
    return _pick_col(cur, "patient_inclusion_constraints_important", ["is_root"]) is not None


def _canon_nct(s: str | None) -> str | None:
    if not s:
        return s
    s = s.strip()
    # NCT + 8 chars (usually digits)
    if s.upper().startswith("NCT") and len(s) >= 11:
        return s[:11].upper()
    return s.upper()


# ----------------------------
# Disease path inspection
# ----------------------------

def inspect_disease_path(
    conn: sqlite3.Connection,
    patient_id: str,
    trial: Dict[str, Any],
    require_root_for_hops: bool,
) -> Dict[str, Any]:
    cur = conn.cursor()
    out: Dict[str, Any] = {
        "supported": False,
        "use_daa": False,
        "candidates": [],
        "hit_possible": False,
    }

    if not _table_has(cur, "patient_inclusion_constraints_important"):
        out["reason"] = "no patient_inclusion_constraints_important table"
        return out
    if not _table_has(cur, "disease_constraint_atoms"):
        out["reason"] = "no disease_constraint_atoms table"
        return out

    out["supported"] = True
    fii_has_is_root = _has_is_root(cur)
    canon_trial_nct = _canon_nct(trial["nct_id"])

    dli_table = "disease_constraint_atoms"
    dli_trial_col = _pick_col(cur, dli_table, ["trial_id", "nct_id"])
    dli_var_col = _pick_col(
        cur, dli_table, ["stem_var", "var_name", "base_var", "var_name_notime"]
    )
    if not dli_trial_col or not dli_var_col:
        out["reason"] = "disease_constraint_atoms missing trial/var columns"
        return out

    # Does DAA override DLI?
    use_daa = False
    daa_table = "disease_constraint_alternatives"
    daa_trial_col = None
    daa_alt_col = None
    if _table_has(cur, daa_table):
        daa_trial_col = _pick_col(cur, daa_table, ["trial_id", "nct_id"])
        daa_alt_col = _pick_col(
            cur, daa_table, ["alt_var_name_notime", "alt_var_name"]
        )
        if daa_trial_col and daa_alt_col:
            use_daa = True
            out["use_daa"] = True
        else:
            out["note"] = (
                "disease_constraint_alternatives exists but lacks trial/alt columns; "
                "falling back to disease_constraint_atoms"
            )

    candidates: List[Dict[str, Any]] = []

    # -------- DAA path (canonical NCT-aware join) --------
    if use_daa:
        daa_cols = {
            r[1] for r in cur.execute(f"PRAGMA table_info({daa_table})").fetchall()
        }
        hop_col = "hop" if "hop" in daa_cols else None
        reason_col = "reason" if "reason" in daa_cols else None
        hop_expr = hop_col or "NULL"
        reason_expr = reason_col or "NULL"

        # Decide whether daa.trial_id looks like an NCT string or numeric id
        join_target = _join_target_for_key(cur, daa_table, daa_trial_col)

        if join_target == "nct_id":
            trial_match_expr = f"""
              CASE
                WHEN {daa_trial_col} LIKE 'NCT________%' THEN SUBSTR({daa_trial_col}, 1, 11)
                ELSE {daa_trial_col}
              END
            """
            key_param = canon_trial_nct
        else:
            trial_match_expr = daa_trial_col
            key_param = trial["id"]

        rows = cur.execute(
            f"""
            SELECT
              {daa_alt_col} AS base_var,
              COALESCE({hop_expr}, 0) AS hop,
              COALESCE({reason_expr}, 'self') AS reason
            FROM {daa_table}
            WHERE {daa_alt_col} IS NOT NULL
              AND {trial_match_expr} = ?
            """,
            (key_param,),
        ).fetchall()

        for base_var, hop, reason in rows:
            if base_var is None:
                continue
            candidates.append(
                {
                    "base_var": base_var,
                    "source": "disease_constraint_alternatives",
                    "hop": int(hop),
                    "reason": str(reason),
                }
            )

    # -------- DLI path (fallback; canonical NCT-aware join) --------
    dli_cols = {
        r[1] for r in cur.execute(f"PRAGMA table_info({dli_table})").fetchall()
    }
    dli_hop_col = "hop" if "hop" in dli_cols else None
    dli_hop_expr = dli_hop_col or "NULL"

    # Decide whether disease_constraint_atoms.trial_id looks NCT-like or numeric
    dli_join_target = _join_target_for_key(cur, dli_table, dli_trial_col)

    if dli_join_target == "nct_id":
        dli_trial_match_expr = f"""
          CASE
            WHEN {dli_trial_col} LIKE 'NCT________%' THEN SUBSTR({dli_trial_col}, 1, 11)
            ELSE {dli_trial_col}
          END
        """
        dli_key_param = canon_trial_nct
    else:
        dli_trial_match_expr = dli_trial_col
        dli_key_param = trial["id"]

    rows = cur.execute(
        f"""
        SELECT
          {dli_var_col} AS base_var,
          COALESCE({dli_hop_expr}, 0) AS hop
        FROM {dli_table}
        WHERE {dli_var_col} IS NOT NULL
          AND {dli_trial_match_expr} = ?
        """,
        (dli_key_param,),
    ).fetchall()

    for base_var, hop in rows:
        if base_var is None:
            continue
        candidates.append(
            {
                "base_var": base_var,
                "source": "disease_constraint_atoms" if not use_daa else "dli_fallback",
                "hop": int(hop),
                "reason": "self",
            }
        )

    # -------- Enrich with FII + gating --------
    for c in candidates:
        base_var = c["base_var"]
        fii_row = cur.execute(
            """
            SELECT value,
                   {} AS is_root
            FROM patient_inclusion_constraints_important
            WHERE patient_id = ?
              AND kind = 'bool'
              AND base_var = ?
            LIMIT 1
            """.format("is_root" if fii_has_is_root else "NULL"),
            (patient_id, base_var),
        ).fetchone()

        if fii_row:
            value, is_root = fii_row
            truthy = _is_truthy(value)
            if require_root_for_hops and fii_has_is_root:
                passes_hop_gate = (
                    c["hop"] <= 0 or c["reason"] == "self" or (is_root == 1)
                )
            else:
                passes_hop_gate = True
            contributes = bool(truthy and passes_hop_gate)
        else:
            value, is_root = None, None
            truthy = False
            passes_hop_gate = False
            contributes = False

        c.update(
            {
                "fii_present": fii_row is not None,
                "fii_value": value,
                "fii_is_root": is_root,
                "truthy": truthy,
                "passes_hop_gate": passes_hop_gate,
                "would_contribute_hit": contributes,
            }
        )

    out["candidates"] = candidates
    out["hit_possible"] = any(c["would_contribute_hit"] for c in candidates)
    if not candidates:
        out.setdefault(
            "reason", "no disease vars for this trial (after canonical NCT join)"
        )
    elif not out["hit_possible"]:
        out.setdefault(
            "reason",
            "have disease vars but none are both truthy and pass hop/is_root gate",
        )
    return out


# ----------------------------
# Literal path inspection (lifted path only)
# ----------------------------

def inspect_literal_path(
    conn: sqlite3.Connection,
    patient_id: str,
    trial: Dict[str, Any],
    require_root_for_hops: bool,
) -> Dict[str, Any]:
    cur = conn.cursor()
    out: Dict[str, Any] = {
        "supported": False,
        "uses_lifted_source": None,
        "candidates": [],
        "hit_possible": False,
    }

    if not _table_has(cur, "patient_inclusion_constraints_important"):
        out["reason"] = "no patient_inclusion_constraints_important table"
        return out

    has_laa = _table_has(cur, "constraint_literal_alternatives")
    has_llm = _table_has(cur, "constraint_lifted_atoms")
    lifted_source = "constraint_literal_alternatives" if has_laa else (
        "constraint_lifted_atoms" if has_llm else None
    )
    if not lifted_source:
        out["reason"] = (
            "no lifted literal tables "
            "(constraint_literal_alternatives / constraint_lifted_atoms)"
        )
        # Non-lifted path is not implemented here (constraint_clause_atoms + is_neg)
        return out

    out["supported"] = True
    out["uses_lifted_source"] = lifted_source
    fii_has_is_root = _has_is_root(cur)

    cols = {r[1] for r in cur.execute(f"PRAGMA table_info({lifted_source})").fetchall()}
    has_base_stem = "base_var_stem" in cols
    has_lifted_stem = "lifted_var_stem" in cols
    has_hop = "hop" in cols
    has_reason = "reason" in cols

    # IMPORTANT: prefer lifted_var_stem over base_var_stem (matches literal_hits)
    if has_base_stem and has_lifted_stem:
        base_expr = "COALESCE(x.lifted_var_stem, x.base_var_stem)"
    elif has_lifted_stem:
        base_expr = "x.lifted_var_stem"
    elif has_base_stem:
        base_expr = "x.base_var_stem"
    else:
        out["reason"] = (
            f"{lifted_source} has no base_var_stem / lifted_var_stem columns"
        )
        return out

    hop_expr = "x.hop" if has_hop else "0"
    reason_expr = "x.reason" if has_reason else "'self'"

    # Collect lifted candidates restricted to this trial
    rows = cur.execute(
        f"""
        WITH la AS (
          SELECT
            mt.id AS merged_trial_id,
            mt.nct_id,
            x.clause_id,
            x.literal_index,
            {base_expr}       AS base_var,
            COALESCE({hop_expr}, 0)    AS hop,
            COALESCE({reason_expr}, 'self') AS reason
          FROM trials mt
          JOIN {lifted_source} x
            ON x.trial_id IN (mt.inclusion_trial_side_id, mt.assumed_trial_side_id)
          WHERE mt.id = :trial_id
        )
        SELECT DISTINCT base_var, hop, reason
        FROM la
        WHERE base_var IS NOT NULL
        """,
        {"trial_id": trial["id"]},
    ).fetchall()

    candidates: List[Dict[str, Any]] = []
    for base_var, hop, reason in rows:
        candidates.append(
            {
                "base_var": base_var,
                "hop": int(hop),
                "reason": str(reason),
            }
        )

    # Enrich with FII
    for c in candidates:
        base_var = c["base_var"]
        fii_row = cur.execute(
            """
            SELECT value,
                   {} As is_root
            FROM patient_inclusion_constraints_important
            WHERE patient_id = ?
              AND kind = 'bool'
              AND base_var = ?
            LIMIT 1
            """.format("is_root" if fii_has_is_root else "NULL"),
            (patient_id, base_var),
        ).fetchone()

        if fii_row:
            value, is_root = fii_row
            truthy = _is_truthy(value)
            if require_root_for_hops and fii_has_is_root:
                passes_hop_gate = (
                    c["hop"] <= 0 or c["reason"] == "self" or (is_root == 1)
                )
            else:
                passes_hop_gate = True
            contributes = bool(truthy and passes_hop_gate)
        else:
            value, is_root = None, None
            truthy = False
            passes_hop_gate = False
            contributes = False

        c.update(
            {
                "fii_present": fii_row is not None,
                "fii_value": value,
                "fii_is_root": is_root,
                "truthy": truthy,
                "passes_hop_gate": passes_hop_gate,
                "would_contribute_hit": contributes,
            }
        )

    out["candidates"] = candidates
    out["hit_possible"] = any(c["would_contribute_hit"] for c in candidates)
    if not candidates:
        out.setdefault("reason", "no lifted literal vars for this trial")
    elif not out["hit_possible"]:
        out.setdefault(
            "reason",
            "have literal vars but none are both truthy and pass hop/is_root gate",
        )
    return out


# ----------------------------
# Positive literal path inspection
# ----------------------------

def inspect_positive_path(
    conn: sqlite3.Connection,
    patient_id: str,
    trial: Dict[str, Any],
    require_root_for_hops: bool,
) -> Dict[str, Any]:
    cur = conn.cursor()
    out: Dict[str, Any] = {
        "supported": False,
        "candidates": [],
        "hit_possible": False,
    }

    if not _table_has(cur, "patient_inclusion_constraints_important"):
        out["reason"] = "no patient_inclusion_constraints_important table"
        return out

    has_pl = _table_has(cur, "positive_constraint_literals")
    has_pla = _table_has(cur, "positive_constraint_alternatives")
    if not has_pl and not has_pla:
        out["reason"] = (
            "no positive_constraint_literals / positive_constraint_alternatives tables"
        )
        return out

    out["supported"] = True
    fii_has_is_root = _has_is_root(cur)

    def _stem_expr(table: str) -> str | None:
        """
        Build the same kind of stem key used in positive_literal_hits():
          - positive_constraint_literals → original concept (base_var_stem etc.)
          - positive_constraint_alternatives → lifted concept (lifted_var_stem first)
        """
        cols = {
            r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()
        }

        if table == "positive_constraint_alternatives":
            priority = [
                "lifted_var_stem",
                "lifted_var",
                "base_var_stem",
                "var_name_notime",
                "base_var",
                "var_name",
            ]
        else:  # positive_constraint_literals
            priority = [
                "base_var_stem",
                "var_name_notime",
                "base_var",
                "var_name",
                "lifted_var_stem",
                "lifted_var",
            ]

        present = [c for c in priority if c in cols]
        if not present:
            return None
        # qualify with table name since we don't alias it
        parts = [f"{table}.{c}" for c in present]
        return "COALESCE(" + ",".join(parts) + ")"

    def _trial_col(table: str) -> str | None:
        cols = {
            r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if "nct_id" in cols:
            return "nct_id"
        if "trial_id" in cols:
            return "trial_id"
        return None

    def _hop_col(table: str) -> str | None:
        cols = {
            r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()
        }
        return f"{table}.hop" if "hop" in cols else None

    def _reason_col(table: str) -> str | None:
        cols = {
            r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()
        }
        return f"{table}.reason" if "reason" in cols else None

    candidates: List[Dict[str, Any]] = []

    for table in ("positive_constraint_literals", "positive_constraint_alternatives"):
        if not _table_has(cur, table):
            continue

        trial_col = _trial_col(table)
        stem_expr = _stem_expr(table)
        hop_col = _hop_col(table)
        reason_col = _reason_col(table)

        if not trial_col or not stem_expr:
            continue

        rows = cur.execute(
            f"""
            SELECT
              {stem_expr} AS base_var,
              COALESCE({hop_col or '0'}, 0)        AS hop,
              COALESCE({reason_col or "'self'"}, 'self') AS reason
            FROM {table}
            WHERE {table}.{trial_col} = :trial_key
              AND {stem_expr} IS NOT NULL
            """,
            {"trial_key": trial["nct_id"]},
        ).fetchall()

        for base_var, hop, reason in rows:
            candidates.append(
                {
                    "base_var": base_var,
                    "hop": int(hop),
                    "reason": str(reason),
                    "source": table,
                }
            )

    # Enrich with FII
    for c in candidates:
        base_var = c["base_var"]
        fii_row = cur.execute(
            """
            SELECT value,
                   {} AS is_root
            FROM patient_inclusion_constraints_important
            WHERE patient_id = ?
              AND kind = 'bool'
              AND base_var = ?
            LIMIT 1
            """.format("is_root" if fii_has_is_root else "NULL"),
            (patient_id, base_var),
        ).fetchone()

        if fii_row:
            value, is_root = fii_row
            truthy = _is_truthy(value)
            if require_root_for_hops and fii_has_is_root:
                passes_hop_gate = (
                    c["hop"] <= 0 or c["reason"] == "self" or (is_root == 1)
                )
            else:
                passes_hop_gate = True
            contributes = bool(truthy and passes_hop_gate)
        else:
            value, is_root = None, None
            truthy = False
            passes_hop_gate = False
            contributes = False

        c.update(
            {
                "fii_present": fii_row is not None,
                "fii_value": value,
                "fii_is_root": is_root,
                "truthy": truthy,
                "passes_hop_gate": passes_hop_gate,
                "would_contribute_hit": contributes,
            }
        )

    out["candidates"] = candidates
    out["hit_possible"] = any(c["would_contribute_hit"] for c in candidates)
    if not candidates:
        out.setdefault("reason", "no positive_* vars for this trial")
    elif not out["hit_possible"]:
        out.setdefault(
            "reason",
            "have positive_* vars but none are both truthy and pass hop/is_root gate",
        )
    return out


# ----------------------------
# Main CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Inspect why a (patient, trial) pair did not retrieve in "
            "disease/literal/positive paths."
        )
    )
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument(
        "--trial-id",
        required=True,
        help="trials.id (int) OR canonical NCT id (e.g. NCT00118651)",
    )
    ap.add_argument(
        "--allow-hop-without-root",
        action="store_true",
        help="Match hop>0 accepted alternatives even if "
             "patient_inclusion_constraints_important.is_root is 0/NULL",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    try:
        conn = sqlite3.connect(str(args.db))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    raw_trial = args.trial_id
    trials_to_inspect: List[Dict[str, Any]] = []

    # Case 1: user passed a numeric trials.id
    try:
        trial_id = int(raw_trial)
        trial = _get_trial_row(cur, trial_id)
        trials_to_inspect.append(trial)
    except ValueError:
        # Case 2: user passed an NCT id (canonical or with suffix)
        nct = raw_trial.strip()

        # Exact matches
        rows = cur.execute(
            "SELECT id, nct_id FROM trials WHERE nct_id = ?",
            (nct,),
        ).fetchall()

        # If no exact matches, try prefix to include subcohorts (NCTxxxxxxa/b/…)
        if not rows:
            rows = cur.execute(
                "SELECT id, nct_id FROM trials WHERE nct_id LIKE ?",
                (nct + "%",),
            ).fetchall()

        if not rows:
            print(f"[error] no trials found with nct_id '{nct}'", file=sys.stderr)
            sys.exit(2)

        # For NCT, we now inspect *all* matching rows (subcohorts)
        for tid, tid_nct in rows:
            trials_to_inspect.append(_get_trial_row(cur, int(tid)))

    require_root_for_hops = not args.allow_hop_without_root

    per_trial_results: List[Dict[str, Any]] = []
    for trial in trials_to_inspect:
        disease_info = inspect_disease_path(
            conn, args.patient, trial, require_root_for_hops
        )
        literal_info = inspect_literal_path(
            conn, args.patient, trial, require_root_for_hops
        )
        positive_info = inspect_positive_path(
            conn, args.patient, trial, require_root_for_hops
        )

        per_trial_results.append(
            {
                "trial_id": trial["id"],
                "trial_nct_id": trial["nct_id"],
                "paths": {
                    "disease": disease_info,
                    "literal": literal_info,
                    "positive": positive_info,
                },
            }
        )

    summary = {
        "db": str(args.db),
        "patient_id": args.patient,
        "query_trial_id": raw_trial,
        "require_root_for_hops": require_root_for_hops,
        "trials": per_trial_results,
    }

    if args.pretty:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[warn] interrupted", file=sys.stderr)
        sys.exit(130)
