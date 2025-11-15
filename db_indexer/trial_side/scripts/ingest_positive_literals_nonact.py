#!/usr/bin/env python3
# scripts/ingest_positive_constraint_literals.py
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# DEFAULTS
DEFAULT_PER_FILE_DIR = "<SATIR_ROOT>/build/positive_constraint_literals_categorized/per_file"

TABLE_MAIN = "positive_constraint_literals_nonact"
TABLE_PREV = "positive_constraint_literals_prevention_nonact"

# NOTE:
# Categories now come from PositiveLiteralCategorizer.CATEGORY_NAME_MAP (human strings),
# and each representative may have MULTIPLE categories (List[str]) or legacy single str.
# We ingest a rep if AT LEAST ONE of its categories matches the allowed set.
#
# Optional behavior:
#   - If --allow-uncategorized is set, also ingest reps with missing/empty/invalid category.
#   - Reps whose category is present/valid but disjoint from allowed_categories are not ingested.

CATS_PREVENTION = {
    # findings template
    "Prevention (finding)",
    # optionally include if you want broader prevention-ish ingestion:
    # "Other (finding)",
}

CATS_MAIN = {
    # procedures template
    "Improves effectiveness of (procedure)",
    # "Reduces procedure-related adverse effect (procedure)",
    # "Other (procedure)",

    # findings template
    "Clinically address (finding)",
    "Not of clinical interest",
    # "Prevention (finding)",
    "Other (finding)",

    # other templates (substance/product)
    "Reduce exposure/use (substance/product)",
    # "Mitigates harms of exposure/use (substance/product)",
    # NOTE: spelling matches your categorizer constant ("venefits")
    # "Enhances benefits of exposure/use (substance/product)",
    # "Other (substance/product)",
}

# ────────────────────────────────────────────────────────────────
# Filename parsing
_FILE_RE = re.compile(
    r"^(NCT\d{8})([A-Za-z0-9_-]*?)_(inclusion|exclusion)_program(?:\.assumed)?\.smt2$",
    re.I,
)


def _parse_fname(name: str):
    # If upstream stored a full path, keep only basename
    base_name = Path(name).name if name else ""
    m = _FILE_RE.match(base_name)
    if not m:
        return None, None, "main"
    nct_base, suffix, kind = m.group(1), (m.group(2) or ""), m.group(3).lower()
    nct = nct_base + suffix
    variant = (
        "assumed"
        if ".assumed." in base_name or base_name.endswith(".assumed.smt2")
        else "main"
    )
    return nct, kind, variant


# ────────────────────────────────────────────────────────────────
# Timeframe parsing
_TIMEFRAME_TOKEN_PAT = (
    r"(now|inthehistory|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture(\d+)?(?:minutes|hours|days|weeks|months|years)?|"
    r"inthefuture)"
)
_TF_FINDER_RE = re.compile(r"_" + _TIMEFRAME_TOKEN_PAT + r"(?:_|$)")


def _split_base_timeframe(var_name: str):
    last = None
    for m in _TF_FINDER_RE.finditer(var_name or ""):
        last = m
    if not last:
        return var_name, None
    tf = last.group(1)
    base = (var_name[: last.start()] + var_name[last.end() :]).strip("_")
    base = re.sub(r"_+", "_", base)
    return base, tf


def _tf_window_hours(token):
    # Placeholder for future expansion: turn token -> numeric window.
    if token == "now":
        return 0.0, 0.0
    if token is None:
        return None, None
    return None, None


# ────────────────────────────────────────────────────────────────
# Categorized reps (QUALIFIERS STRIPPED)
def _iter_reps_allowed_and_uncategorized(js: dict, allowed_categories: set[str]):
    """
    Returns:
      allowed: reps whose category (string or list[str]) intersects allowed_categories
      uncategorized: reps with missing/empty/invalid category

    representative["category"] may be:
      - List[str]  (new behavior)
      - str        (legacy behavior)

    A rep is "allowed" if ANY of its categories matches allowed_categories.
    A rep is "uncategorized" if its category is missing/empty/invalid.
    A rep with a present/valid category that does NOT intersect allowed_categories
    is ignored.
    """
    allowed, uncategorized = [], []
    reps = js.get("representatives") or []

    for r in reps:
        if not isinstance(r, dict):
            continue
        rep = (r.get("rep") or "").strip()
        if not rep:
            continue

        cat = r.get("category")

        # Missing category => uncategorized
        if not cat:
            uncategorized.append(rep)
            continue

        # Normalize categories to List[str]
        if isinstance(cat, str):
            cats = [cat.strip()] if cat.strip() else []
        elif isinstance(cat, list):
            cats = [
                (c or "").strip()
                for c in cat
                if isinstance(c, str) and (c or "").strip()
            ]
        else:
            cats = []

        if not cats:
            uncategorized.append(rep)
            continue

        # Allow if ANY matches
        if any(c in allowed_categories for c in cats):
            allowed.append(rep)

    # De-dupe while preserving order
    return list(dict.fromkeys(allowed)), list(dict.fromkeys(uncategorized))


# ────────────────────────────────────────────────────────────────
# Prevention rewrite
_ENTITY_PATTERNS = [
    re.compile(r"^patient_has_(?:finding|diagnosis)_of_(.+)$"),
]


def _extract_entity_from_base(base: str):
    for p in _ENTITY_PATTERNS:
        m = p.match(base)
        if m:
            return m.group(1)
    return None


def _rewrite_prevention_var(var_name: str):
    base, tf = _split_base_timeframe(var_name)
    if not tf:
        return var_name, None
    ent = _extract_entity_from_base(base)
    if not ent:
        return var_name, None
    return f"patient_wants_to_prevent_{ent}_{tf}", f"patient_wants_to_prevent_{ent}"


# ────────────────────────────────────────────────────────────────
# DDL
def _ddl(table: str, with_orig: bool):
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


# ────────────────────────────────────────────────────────────────
def _ingest_into_table(
    *,
    conn,
    table,
    per_file_dir,
    direction_filter,
    allowed_categories,
    recreate,
    allow_uncategorized,
):
    is_prev = (table == TABLE_PREV)

    mode = "APPEND (no-recreate)" if not recreate else "RECREATE (drop+create)"
    print(
        f"[{table}] start | mode={mode} | dir={per_file_dir} | "
        f"direction={direction_filter} | allow_uncategorized={allow_uncategorized}"
    )

    existed_before = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )

    before_count = 0
    if existed_before and not recreate:
        try:
            before_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            before_count = 0

    if recreate:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.executescript(_ddl(table, with_orig=is_prev))

    cur = conn.cursor()
    files = sorted(Path(per_file_dir).glob("*.json"))

    files_total = len(files)
    files_used = 0
    rows_upserted = 0
    reps_seen_allowed = 0
    reps_seen_uncategorized = 0
    reps_ingested_uncategorized = 0
    reps_skipped_uncategorized = 0
    reps_ingested_total = 0

    for f in files:
        js = json.loads(f.read_text())

        if js.get("direction") != direction_filter:
            continue

        nct, kind, variant = _parse_fname(js.get("smt2_file", ""))
        if kind != "inclusion":
            continue

        files_used += 1

        allowed, uncategorized = _iter_reps_allowed_and_uncategorized(js, allowed_categories)

        reps_seen_allowed += len(allowed)
        reps_seen_uncategorized += len(uncategorized)

        vars_to_ingest = list(allowed)
        if allow_uncategorized:
            vars_to_ingest = list(dict.fromkeys(vars_to_ingest + uncategorized))
            reps_ingested_uncategorized += len(uncategorized)
        else:
            reps_skipped_uncategorized += len(uncategorized)

        reps_ingested_total += len(vars_to_ingest)

        if not vars_to_ingest:
            continue

        for orig in vars_to_ingest:
            var = orig
            base_forced = None

            if is_prev:
                new, new_base = _rewrite_prevention_var(orig)
                if new_base:
                    var, base_forced = new, new_base

            base, tf = _split_base_timeframe(var)
            if base_forced:
                base = base_forced

            tf_lb, tf_ub = _tf_window_hours(tf)

            cols = "(nct_id,kind,variant,direction,var_name,smt2_file,base_var,timeframe,tf_lb_hours,tf_ub_hours"
            vals = "?,?,?,?,?,?,?,?,?,?"
            args = [nct, kind, variant, js["direction"], var, js["smt2_file"], base, tf, tf_lb, tf_ub]

            if is_prev:
                cols += ",orig_var_name"
                vals += ",?"
                args.append(orig)

            cur.execute(
                f"INSERT OR REPLACE INTO {table} {cols}) VALUES ({vals})",
                args,
            )
            rows_upserted += 1

    conn.commit()

    after_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    dropped = recreate and existed_before
    created = (not existed_before) or recreate

    print(
        f"[{table}] done | existed_before={existed_before} | dropped={dropped} | created={created} | "
        f"files_total={files_total} files_used={files_used} rows_upserted={rows_upserted} | "
        f"count_before={before_count} count_after={after_count} | "
        f"allowed_reps_seen={reps_seen_allowed} uncategorized_reps_seen={reps_seen_uncategorized} | "
        f"uncategorized_reps_ingested={reps_ingested_uncategorized} "
        f"uncategorized_reps_skipped={reps_skipped_uncategorized} | "
        f"reps_ingested_total={reps_ingested_total}"
    )


# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../../../build/trial.db")
    ap.add_argument("--per-file-dir", default=DEFAULT_PER_FILE_DIR)
    ap.add_argument("--direction", default="sat_on_true")
    ap.add_argument("--only", choices=["both", "main", "prevention"], default="both")
    ap.add_argument("--no-recreate", action="store_true")
    ap.add_argument(
        "--allow-uncategorized",
        action="store_true",
        help="Also ingest representatives with missing/empty/invalid category.",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.only in ("both", "main"):
            _ingest_into_table(
                conn=conn,
                table=TABLE_MAIN,
                per_file_dir=args.per_file_dir,
                direction_filter=args.direction,
                allowed_categories=CATS_MAIN,
                recreate=not args.no_recreate,
                allow_uncategorized=args.allow_uncategorized,
            )
        if args.only in ("both", "prevention"):
            _ingest_into_table(
                conn=conn,
                table=TABLE_PREV,
                per_file_dir=args.per_file_dir,
                direction_filter=args.direction,
                allowed_categories=CATS_PREVENTION,
                recreate=not args.no_recreate,
                allow_uncategorized=args.allow_uncategorized,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
        print("[OK] ingest_positive_constraint_literals finished successfully.")
    except Exception as e:
        print(f"[ERROR] ingest_positive_constraint_literals failed: {type(e).__name__}: {e}", file=sys.stderr)
        raise