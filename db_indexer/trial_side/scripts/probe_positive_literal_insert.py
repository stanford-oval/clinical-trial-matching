#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_positive_literal_insert.py

Run the same logic as ingest_positive_constraint_literals.py for ONE json file and
optionally insert into SQLite, then verify the row exists.

Example:
  python probe_positive_literal_insert.py \
    --db <SATIR_ROOT>/build/trial.db \
    --json <SATIR_ROOT>/build/positive_constraint_literals_categorized/per_file/NCT00629356b_inclusion_program.smt2.json \
    --table positive_constraint_literals_nonact \
    --direction sat_on_true \
    --do-insert
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

TABLE_MAIN = "positive_constraint_literals_nonact"
TABLE_PREV = "positive_constraint_literals_prevention_nonact"

CATS_PREVENTION = {
    "Prevention (finding)",
}

CATS_MAIN = {
    "Improves effectiveness of (procedure)",
    "Clinically address (finding)",
    "Not of clinical interest",
    "Other (finding)",
    "Reduce exposure/use (substance/product)",
}

_FILE_RE = re.compile(
    r"^(NCT\d{8})([A-Za-z0-9_-]*?)_(inclusion|exclusion)_program(?:\.assumed)?\.smt2(?:\.json)?$",
    re.I,
)

_TIMEFRAME_TOKEN_PAT = (
    r"(now|inthehistory|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture(?:\d+)?(?:minutes|hours|days|weeks|months|years)?|"
    r"inthefuture)"
)
_TF_FINDER_RE = re.compile(r"_" + _TIMEFRAME_TOKEN_PAT + r"(?:_|$)")

_ENTITY_PATTERNS = [
    re.compile(r"^patient_has_(?:finding|diagnosis)_of_(.+)$"),
]


def parse_fname(name: str):
    base_name = Path(name).name if name else ""
    m = _FILE_RE.match(base_name)
    if not m:
        return None, None, "main"
    nct_base, suffix, kind = m.group(1), (m.group(2) or ""), m.group(3).lower()
    nct = nct_base + suffix
    variant = (
        "assumed"
        if ".assumed." in base_name or base_name.endswith(".assumed.smt2") or base_name.endswith(".assumed.smt2.json")
        else "main"
    )
    return nct, kind, variant


def split_base_timeframe(var_name: str):
    last = None
    for m in _TF_FINDER_RE.finditer(var_name or ""):
        last = m
    if not last:
        return var_name, None
    tf = last.group(1)
    base = (var_name[: last.start()] + var_name[last.end() :]).strip("_")
    base = re.sub(r"_+", "_", base)
    return base, tf


def tf_window_hours(token):
    if token == "now":
        return 0.0, 0.0
    if token is None:
        return None, None
    return None, None


def extract_entity_from_base(base: str):
    for p in _ENTITY_PATTERNS:
        m = p.match(base)
        if m:
            return m.group(1)
    return None


def rewrite_prevention_var(var_name: str):
    base, tf = split_base_timeframe(var_name)
    if not tf:
        return var_name, None
    ent = extract_entity_from_base(base)
    if not ent:
        return var_name, None
    return f"patient_wants_to_prevent_{ent}_{tf}", f"patient_wants_to_prevent_{ent}"


def normalize_categories(cat: Any) -> list[str]:
    if cat is None:
        return []
    if isinstance(cat, str):
        s = cat.strip()
        return [s] if s else []
    if isinstance(cat, list):
        out = []
        for c in cat:
            if isinstance(c, str):
                s = c.strip()
                if s:
                    out.append(s)
        return out
    return []


def iter_reps_allowed_and_uncategorized(js: dict, allowed_categories: set[str]):
    allowed, uncategorized = [], []
    for r in (js.get("representatives") or []):
        if not isinstance(r, dict):
            continue
        rep = (r.get("rep") or "").strip()
        if not rep:
            continue

        cats = normalize_categories(r.get("category"))
        raw_cat = r.get("category")

        if not raw_cat or not cats:
            uncategorized.append(rep)
            continue

        if any(c in allowed_categories for c in cats):
            allowed.append(rep)

    return list(dict.fromkeys(allowed)), list(dict.fromkeys(uncategorized))


def ddl(table: str, with_orig: bool):
    extra = ", orig_var_name TEXT" if with_orig else ""
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
      nct_id TEXT,
      kind TEXT,
      variant TEXT,
      direction TEXT,
      var_name TEXT,
      smt2_file TEXT,
      base_var TEXT,
      timeframe TEXT,
      tf_lb_hours REAL,
      tf_ub_hours REAL{extra},
      PRIMARY KEY (nct_id, kind, variant, direction, var_name)
    );
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--table", choices=[TABLE_MAIN, TABLE_PREV], required=True)
    ap.add_argument("--direction", default="sat_on_true")
    ap.add_argument("--allow-uncategorized", action="store_true")
    ap.add_argument("--do-insert", action="store_true")
    args = ap.parse_args()

    is_prev = args.table == TABLE_PREV
    allowed_categories = CATS_PREVENTION if is_prev else CATS_MAIN

    p = Path(args.json)
    js = json.loads(p.read_text())

    print(f"DB:   {args.db}")
    print(f"JSON: {p}")
    print(f"TABLE: {args.table}")
    print()

    nct, kind, variant = parse_fname(js.get("smt2_file", ""))
    print("Parse from js['smt2_file']:")
    print(" ", js.get("smt2_file"))
    print(" ", {"nct_id": nct, "kind": kind, "variant": variant})

    if kind != "inclusion":
        print(f"[STOP] kind={kind!r} != 'inclusion'")
        return

    if js.get("direction") != args.direction:
        print(f"[STOP] direction mismatch: js={js.get('direction')!r} expected={args.direction!r}")
        return

    allowed, uncategorized = iter_reps_allowed_and_uncategorized(js, allowed_categories)
    vars_to_ingest = list(allowed)
    if args.allow_uncategorized:
        vars_to_ingest = list(dict.fromkeys(vars_to_ingest + uncategorized))

    print(f"allowed={allowed}")
    print(f"uncategorized={uncategorized}")
    print(f"vars_to_ingest={vars_to_ingest}")
    print()

    if not vars_to_ingest:
        print("[STOP] no vars to ingest")
        return

    conn = sqlite3.connect(args.db)
    try:
        conn.executescript(ddl(args.table, with_orig=is_prev))

        for orig in vars_to_ingest:
            var = orig
            base_forced = None

            if is_prev:
                new, new_base = rewrite_prevention_var(orig)
                if new_base:
                    var, base_forced = new, new_base

            base, tf = split_base_timeframe(var)
            if base_forced:
                base = base_forced
            tf_lb, tf_ub = tf_window_hours(tf)

            row = {
                "nct_id": nct,
                "kind": kind,
                "variant": variant,
                "direction": js["direction"],
                "var_name": var,
                "smt2_file": js["smt2_file"],
                "base_var": base,
                "timeframe": tf,
                "tf_lb_hours": tf_lb,
                "tf_ub_hours": tf_ub,
            }
            if is_prev:
                row["orig_var_name"] = orig

            print("[CANDIDATE ROW]")
            print(json.dumps(row, indent=2, ensure_ascii=False))

            if args.do_insert:
                if is_prev:
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {args.table}
                        (nct_id, kind, variant, direction, var_name, smt2_file, base_var, timeframe, tf_lb_hours, tf_ub_hours, orig_var_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["nct_id"], row["kind"], row["variant"], row["direction"],
                            row["var_name"], row["smt2_file"], row["base_var"], row["timeframe"],
                            row["tf_lb_hours"], row["tf_ub_hours"], row["orig_var_name"]
                        )
                    )
                else:
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {args.table}
                        (nct_id, kind, variant, direction, var_name, smt2_file, base_var, timeframe, tf_lb_hours, tf_ub_hours)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["nct_id"], row["kind"], row["variant"], row["direction"],
                            row["var_name"], row["smt2_file"], row["base_var"], row["timeframe"],
                            row["tf_lb_hours"], row["tf_ub_hours"]
                        )
                    )
                    conn.commit()

                got = conn.execute(
                    f"""
                    SELECT *
                    FROM {args.table}
                    WHERE nct_id=? AND kind=? AND variant=? AND direction=? AND var_name=?
                    """,
                    (row["nct_id"], row["kind"], row["variant"], row["direction"], row["var_name"]),
                ).fetchall()

                print(f"[VERIFY] matched rows after insert: {len(got)}")
                for g in got:
                    print(" ", g)

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise