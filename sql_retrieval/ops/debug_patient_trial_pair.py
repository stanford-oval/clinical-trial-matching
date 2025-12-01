#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_patient_trial_pair.py

Inspect why a patient–trial pair did or did not "hit" in retrieval and how it
is classified by the current pipeline.

Inputs:
  --db        path to trial.db
  --patient   patient_id
  --trial-id  EITHER:
                * numeric trials.id
                * OR trials.nct_id (e.g. NCT00118651)
  --scope     now | any (used for eliminate/classify; retrieval ignores timeframe)
  --allow-hop-without-root  mirror of primitives/compose flags

Outputs:
  - JSON blob on stdout with:
      * retrieval flags (disease / literal / positive)
      * classify_trials label & metrics
      * disease / literal / positive mappings for THIS trial
      * relevant patient_inclusion_constraints_important rows for the mapped stem keys
  - Human-readable summary lines to stderr for quick inspection

Typical usage:

  python debug_patient_trial_pair.py \\
    --db /path/to/trial.db \\
    --patient sigir-201524 \\
    --trial-id NCT00118651 \\
    --pretty
"""

from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sql_retrieval.ops import constraint_primitives as tep


TRUTHY_STRINGS = {"1", "TRUE", "T", "Y", "YES"}
FALSY_STRINGS  = {"0", "FALSE", "F", "N", "NO"}


def _truthy(value: Any) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip()
    # numeric
    try:
        num = float(s)
        if num == 1.0:
            return True
        if num == 0.0:
            return False
    except ValueError:
        pass
    # string
    up = s.upper()
    if up in TRUTHY_STRINGS:
        return True
    if up in FALSY_STRINGS:
        return False
    return None


def _row_to_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> Dict[str, Any]:
    return {desc[0]: row[idx] for idx, desc in enumerate(cursor.description)}


def _get_trial_basic(conn: sqlite3.Connection, trial_id_or_nct: str) -> Dict[str, Any]:
    """
    Accept either a numeric trials.id *or* an NCT ID string (e.g. 'NCT00118651').
    Returns a dict with: id, nct_id, inclusion_trial_side_id, exclusion_trial_side_id, assumed_trial_side_id.
    """
    cur = conn.cursor()
    key = trial_id_or_nct.strip()

    if key.isdigit():
        # Interpret as trials.id
        cur.execute(
            """
            SELECT id, nct_id,
                   inclusion_trial_side_id,
                   exclusion_trial_side_id,
                   assumed_trial_side_id
            FROM trials
            WHERE id = ?
            """,
            (int(key),),
        )
    else:
        # Interpret as trials.nct_id
        cur.execute(
            """
            SELECT id, nct_id,
                   inclusion_trial_side_id,
                   exclusion_trial_side_id,
                   assumed_trial_side_id
            FROM trials
            WHERE nct_id = ?
            """,
            (key,),
        )

    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"trial identifier={trial_id_or_nct!r} not found in trials")
    cols = [d[0] for d in cur.description]
    return {c: row[i] for i, c in enumerate(cols)}


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None


def _stem_key_expr(cols: List[str]) -> List[str]:
    """
    Priority list to mirror positive_literal_hits() and friends.
    Returns ordered list of columns to COALESCE over.
    """
    priority = [
        "lifted_var_stem",
        "base_var_stem",
        "var_name_notime",
        "base_var",
        "var_name",
        "lifted_var",
    ]
    return [c for c in priority if c in cols]


def _collect_disease_mappings(
    conn: sqlite3.Connection,
    trial_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Collect disease_* rows for this trial and compute a "stem_key" that should
    line up with patient_inclusion_constraints_important.base_var.
    """
    cur = conn.cursor()
    mappings: List[Dict[str, Any]] = []

    nct_id = trial_info.get("nct_id")
    trial_id = trial_info.get("id")

    # disease_constraint_atoms
    if _table_exists(cur, "disease_constraint_atoms"):
        cur.execute("PRAGMA table_info(disease_constraint_atoms)")
        cols = [r[1] for r in cur.fetchall()]
        key_cols = []
        if "trial_id" in cols:
            key_cols.append("trial_id")
        if "nct_id" in cols:
            key_cols.append("nct_id")
        var_cols = _stem_key_expr(cols)
        hop_col = "hop" if "hop" in cols else None
        reason_col = "reason" if "reason" in cols else None

        for key_col in key_cols:
            q = f"SELECT * FROM disease_constraint_atoms WHERE {key_col} = ?"
            key_val = trial_id if key_col == "trial_id" else nct_id
            if key_val is None:
                continue
            cur.execute(q, (key_val,))
            for row in cur.fetchall():
                d = _row_to_dict(cur, row)
                stem_key = None
                for c in var_cols:
                    if d.get(c) is not None:
                        stem_key = d[c]
                        break
                mappings.append(
                    {
                        "source_table": "disease_constraint_atoms",
                        "key_col": key_col,
                        "key_val": key_val,
                        "raw": d,
                        "stem_key": stem_key,
                        "hop": d.get(hop_col) if hop_col else 0,
                        "reason": d.get(reason_col) if reason_col else "self",
                    }
                )

    # disease_constraint_alternatives
    if _table_exists(cur, "disease_constraint_alternatives"):
        cur.execute("PRAGMA table_info(disease_constraint_alternatives)")
        cols = [r[1] for r in cur.fetchall()]
        key_cols = []
        if "trial_id" in cols:
            key_cols.append("trial_id")
        if "nct_id" in cols:
            key_cols.append("nct_id")
        var_cols = _stem_key_expr(cols)
        hop_col = "hop" if "hop" in cols else None
        reason_col = "reason" if "reason" in cols else None

        for key_col in key_cols:
            q = f"SELECT * FROM disease_constraint_alternatives WHERE {key_col} = ?"
            key_val = trial_id if key_col == "trial_id" else nct_id
            if key_val is None:
                continue
            cur.execute(q, (key_val,))
            for row in cur.fetchall():
                d = _row_to_dict(cur, row)
                stem_key = None
                for c in var_cols:
                    if d.get(c) is not None:
                        stem_key = d[c]
                        break
                mappings.append(
                    {
                        "source_table": "disease_constraint_alternatives",
                        "key_col": key_col,
                        "key_val": key_val,
                        "raw": d,
                        "stem_key": stem_key,
                        "hop": d.get(hop_col) if hop_col else 0,
                        "reason": d.get(reason_col) if reason_col else "self",
                    }
                )

    return mappings


def _collect_literal_mappings(
    conn: sqlite3.Connection,
    trial_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Collect literal_* rows for this trial (from inclusion + assumed side) and
    compute stem_key(s).
    """
    cur = conn.cursor()
    mappings: List[Dict[str, Any]] = []

    incl_id = trial_info.get("inclusion_trial_side_id")
    assumed_id = trial_info.get("assumed_trial_side_id")

    # constraint_literal_alternatives OR constraint_lifted_atoms (both use trial-side IDs)
    for table in ("constraint_literal_alternatives", "constraint_lifted_atoms"):
        if not _table_exists(cur, table):
            continue
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        if "trial_id" not in cols:
            continue
        var_cols = _stem_key_expr(cols)
        hop_col = "hop" if "hop" in cols else None
        reason_col = "reason" if "reason" in cols else None

        for side_id in (incl_id, assumed_id):
            if side_id is None:
                continue
            cur.execute(
                f"SELECT * FROM {table} WHERE trial_id = ?",
                (side_id,),
            )
            for row in cur.fetchall():
                d = _row_to_dict(cur, row)
                stem_key = None
                for c in var_cols:
                    if d.get(c) is not None:
                        stem_key = d[c]
                        break
                mappings.append(
                    {
                        "source_table": table,
                        "trial_side_id": side_id,
                        "raw": d,
                        "stem_key": stem_key,
                        "hop": d.get(hop_col) if hop_col else 0,
                        "reason": d.get(reason_col) if reason_col else "self",
                    }
                )

    # Fallback: constraint_clause_atoms directly (base_var)
    if _table_exists(cur, "trial_constraint_clauses") and _table_exists(cur, "constraint_clause_atoms"):
        for side_id in (incl_id, assumed_id):
            if side_id is None:
                continue
            cur.execute(
                """
                SELECT cl.*
                FROM trial_constraint_clauses tc
                JOIN constraint_clause_atoms cl ON cl.clause_id = tc.clause_id
                WHERE tc.trial_id = ?
                """,
                (side_id,),
            )
            for row in cur.fetchall():
                d = _row_to_dict(cur, row)
                stem_key = d.get("base_var")
                mappings.append(
                    {
                        "source_table": "constraint_clause_atoms",
                        "trial_side_id": side_id,
                        "raw": d,
                        "stem_key": stem_key,
                        "hop": 0,
                        "reason": "self",
                    }
                )

    return mappings


def _collect_positive_mappings(
    conn: sqlite3.Connection,
    trial_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Collect positive_* rows for this trial and compute stem_key(s).
    """
    cur = conn.cursor()
    mappings: List[Dict[str, Any]] = []

    nct_id = trial_info.get("nct_id")
    trial_id = trial_info.get("id")

    for table in ("positive_constraint_literals", "positive_constraint_alternatives"):
        if not _table_exists(cur, table):
            continue
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]

        tcol = None
        if "nct_id" in cols:
            tcol = "nct_id"
        elif "trial_id" in cols:
            tcol = "trial_id"
        if tcol is None:
            continue

        var_cols = _stem_key_expr(cols)
        hop_col = "hop" if "hop" in cols else None
        reason_col = "reason" if "reason" in cols else None

        key_val = nct_id if tcol == "nct_id" else trial_id
        if key_val is None:
            continue

        cur.execute(f"SELECT * FROM {table} WHERE {tcol} = ?", (key_val,))
        for row in cur.fetchall():
            d = _row_to_dict(cur, row)
            stem_key = None
            for c in var_cols:
                if d.get(c) is not None:
                    stem_key = d[c]
                    break
            mappings.append(
                {
                    "source_table": table,
                    "key_col": tcol,
                    "key_val": key_val,
                    "raw": d,
                    "stem_key": stem_key,
                    "hop": d.get(hop_col) if hop_col else 0,
                    "reason": d.get(reason_col) if reason_col else "self",
                }
            )

    return mappings


def _collect_fii_for_stem_keys(
    conn: sqlite3.Connection,
    patient_id: str,
    stem_keys: List[str],
) -> List[Dict[str, Any]]:
    """
    Fetch patient_inclusion_constraints_important rows for the given base_var stem_keys for one patient.
    """
    if not stem_keys:
        return []
    cur = conn.cursor()
    if not _table_exists(cur, "patient_inclusion_constraints_important"):
        return []

    placeholders = ",".join(["?"] * len(stem_keys))
    params: List[Any] = [patient_id] + stem_keys
    cur.execute(
        f"""
        SELECT *
        FROM patient_inclusion_constraints_important
        WHERE patient_id = ?
          AND kind = 'bool'
          AND base_var IN ({placeholders})
        ORDER BY base_var
        """,
        params,
    )
    rows: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        d = _row_to_dict(cur, row)
        truth = _truthy(d.get("value"))
        d["debug_truthy"] = truth
        d["debug_is_root"] = d.get("is_root")
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Inspect why a patient–trial pair did or did not hit."
    )
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--patient", required=True)
    ap.add_argument(
        "--trial-id",
        required=True,
        type=str,
        help="Either trials.id (numeric) or trials.nct_id (e.g. NCT00118651)",
    )
    ap.add_argument(
        "--scope",
        choices=["now", "any"],
        default="any",
        help="Scope for eliminate/classify (retrieval ignores timeframe).",
    )
    ap.add_argument(
        "--allow-hop-without-root",
        action="store_true",
        help="Mirror primitives flag: allow hop>0 matches even if is_root != 1.",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    args = ap.parse_args()

    try:
        conn = sqlite3.connect(str(args.db))
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve trial from id or NCT
    trial_info = _get_trial_basic(conn, args.trial_id)
    internal_trial_id = int(trial_info["id"])  # canonical trials.id
    require_root_for_hops = not args.allow_hop_without_root

    # 1) Retrieval membership using the same primitives as compose_trial_eval
    disease_ids = tep.satisfy_disease_constraints(
        conn,
        args.patient,
        require_root_for_hops=require_root_for_hops,
    )
    literal_ids = tep.literal_hits(
        conn,
        args.patient,
        scope=args.scope,
        require_root_for_hops=require_root_for_hops,
    )
    positive_ids = tep.satisfy_positive_literal_constraints(
        conn,
        args.patient,
        require_root_for_hops=require_root_for_hops,
    )

    union_ids = sorted(set(disease_ids) | set(literal_ids) | set(positive_ids))
    in_disease = internal_trial_id in disease_ids
    in_literal = internal_trial_id in literal_ids
    in_positive = internal_trial_id in positive_ids
    in_union = internal_trial_id in union_ids

    # 2) Classification label for this trial
    classify_rows = tep.classify_trials(
        conn,
        args.patient,
        candidate_trial_ids=[internal_trial_id],
        scope=args.scope,
    )
    classify_info = classify_rows[0] if classify_rows else None

    # 3) Source-level mappings for THIS trial
    disease_maps = _collect_disease_mappings(conn, trial_info)
    literal_maps = _collect_literal_mappings(conn, trial_info)
    positive_maps = _collect_positive_mappings(conn, trial_info)

    stem_keys = sorted(
        {
            m["stem_key"]
            for m in (disease_maps + literal_maps + positive_maps)
            if m.get("stem_key") is not None
        }
    )
    fii_rows = _collect_fii_for_stem_keys(conn, args.patient, stem_keys)

    # 4) Build payload
    payload: Dict[str, Any] = {
        "patient_id": args.patient,
        "trial": {
            "trial_id": trial_info["id"],
            "nct_id": trial_info.get("nct_id"),
            "inclusion_trial_side_id": trial_info.get("inclusion_trial_side_id"),
            "exclusion_trial_side_id": trial_info.get("exclusion_trial_side_id"),
            "assumed_trial_side_id": trial_info.get("assumed_trial_side_id"),
        },
        "flags": {
            "scope": args.scope,
            "require_root_for_hops": require_root_for_hops,
        },
        "retrieval_membership": {
            "in_disease_hits": in_disease,
            "in_literal_hits": in_literal,
            "in_positive_literal_hits": in_positive,
            "in_union": in_union,
        },
        "retrieval_counts_for_patient": {
            "disease_hits_total_trials": len(disease_ids),
            "literal_hits_total_trials": len(literal_ids),
            "positive_hits_total_trials": len(positive_ids),
            "union_total_trials": len(union_ids),
        },
        "classification": classify_info,
        "mappings": {
            "disease": disease_maps,
            "literal": literal_maps,
            "positive": positive_maps,
        },
        "patient_inclusion_constraints_important_for_stem_keys": {
            "stem_keys": stem_keys,
            "rows": fii_rows,
        },
    }

    # 5) Human-readable summary for quick glance (stderr)
    print(
        f"[summary] patient={args.patient} "
        f"trial_arg={args.trial_id!r} "
        f"resolved_trial_id={internal_trial_id} "
        f"nct_id={trial_info.get('nct_id')}",
        file=sys.stderr,
    )
    print(
        f"[summary] retrieval: disease={in_disease} "
        f"literal={in_literal} "
        f"positive={in_positive} "
        f"union={in_union}",
        file=sys.stderr,
    )
    if classify_info:
        print(
            f"[summary] classify: label={classify_info['label']} "
            f"frac_unsat_any={classify_info['frac_unsat_any']} "
            f"total_clauses={classify_info['total_clauses']}",
            file=sys.stderr,
        )
    else:
        print("[summary] classify: no row returned", file=sys.stderr)

    print(
        f"[summary] mapped_stem_keys={stem_keys}",
        file=sys.stderr,
    )
    print(
        f"[summary] fii_rows_for_mapped_keys={len(fii_rows)}",
        file=sys.stderr,
    )

    # 6) JSON to stdout (so you can paste back to ChatGPT)
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)
