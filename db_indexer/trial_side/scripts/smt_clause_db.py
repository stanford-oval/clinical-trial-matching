#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smt_clause_db_build.py  —  ASSUMED→EXCLUSION EDITION
────────────────────────────────────────────────────────────────────────────
Build + materialize a SQLite database of CNF-style constraint_clauses extracted from
projected SMT files (Booleans + numeric comparisons), with timeframe-aware
metadata, signatures (full/base/stem), usage analytics, and filtered subsets.

Key policy change
─────────────────
• Files ending in ".assumed.smt2" are still stored as variant='assumed' in trial_constraint_sides,
  BUT for *all downstream logic* we *treat them as exclusion*:
    effective_kind := (variant='assumed') ? 'exclusion' : kind
  This applies to:
    - Top-N computations (full and base-pattern)
    - filtered_trial_constraint_clauses building
    - merged trial materialization (exclusion side includes main ∪ assumed)
    - all usage analytics (TYPE 1 / TYPE 2 / TYPE 2½ / TYPE 3)

Merged trials
─────────────
• Inclusion picks ONLY main inclusion (no assumed fallback).
• Exclusion picks union of main exclusion (if any) and assumed (if any).
• A merged trial row is created iff inclusion(main) exists AND (exclusion(main) or assumed) exists
  when --merge-require-both is given.

CLI
───
Minimal and opinionated:
  --projected-dir     Directory of projected SMT2 files (default: ../../build/canon_projection/_projected_smt)
  --db, --db-path     SQLite path (default: ../../build/trial.db)
  --merge-require-both  Require inclusion(main) and at least one of exclusion(main)/assumed for merged rows
  --skip-filtered
  --skip-merge
  --skip-reuse
  --top-n-incl        Top-N inclusion patterns/constraint_clauses (default: very large)
  --top-n-excl        Top-N exclusion patterns/constraint_clauses (default: very large)
  --filtered-mode     base | full  (default: base)

Everything else is unchanged from a user perspective.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, Union
import argparse
import re
import sqlite3
import sys

# ============================== Minimal S-expression parsing ==============================

Token = str
S = Any  # Atom=str, List[S]=list

def _tokenize(s: str) -> List[Token]:
    s = re.sub(r";[^\n]*", "", s)
    s = re.sub(r'("([^"\\]|\\.)*")', r' \1 ', s)
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]

def _parse(tokens: List[Token], i: int = 0) -> Tuple[S, int]:
    if i >= len(tokens): raise ValueError("unexpected EOF")
    t = tokens[i]
    if t == "(":
        i += 1
        out: List[S] = []
        while i < len(tokens) and tokens[i] != ")":
            node, i = _parse(tokens, i)
            out.append(node)
        if i >= len(tokens) or tokens[i] != ")":
            raise ValueError("missing ')'")
        return out, i + 1
    if t == ")":
        raise ValueError("unexpected ')'")
    return t, i + 1

def parse_sexpr(src: str) -> List[S]:
    toks = _tokenize(src)
    i = 0
    out: List[S] = []
    while i < len(toks):
        node, i = _parse(toks, i)
        out.append(node)
    return out

def sym(x: S) -> Optional[str]:
    return x if isinstance(x, str) else None

# ============================== Clause member types & helpers ==============================

@dataclass(frozen=True)
class Literal:
    var: str
    is_neg: bool  # True means ~var

@dataclass(frozen=True)
class NumPred:
    var: str
    op: str         # '>=','<=','>','<','='
    rhs: float

ClauseMember = Union[Literal, NumPred]
_NUM_OPS = {">=", "<=", ">", "<", "="}

def _canon_num(x: float) -> str:
    return f"{float(x):.9g}"

def _member_sig(m: ClauseMember) -> str:
    if isinstance(m, Literal):
        return ("~" if m.is_neg else "") + m.var
    else:
        return f"{m.var}{m.op}{_canon_num(m.rhs)}"

def _member_sort_key(m: ClauseMember):
    if isinstance(m, Literal):
        return ("B", m.var, 1 if m.is_neg else 0, "", 0.0)
    else:
        return ("N", m.var, 0, m.op, m.rhs)

def _normalize_members(members: Iterable[ClauseMember]) -> Tuple[str, List[Literal], List[NumPred]]:
    seen_b: dict[Tuple[str,int], Literal] = {}
    seen_n: dict[Tuple[str,str,float], NumPred] = {}
    for m in members:
        if isinstance(m, Literal):
            seen_b[(m.var, 1 if m.is_neg else 0)] = m
        else:
            seen_n[(m.var, m.op, float(m.rhs))] = m
    blits = list(seen_b.values())
    nlits = list(seen_n.values())
    all_sorted: List[ClauseMember] = sorted([*blits, *nlits], key=_member_sort_key)
    sig = "|".join(_member_sig(m) for m in all_sorted)
    return sig, blits, nlits

def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _parse_num_pred(form: S) -> Optional[NumPred]:
    if isinstance(form, list) and form and sym(form[0]) in _NUM_OPS and len(form) == 3:
        op = sym(form[0])  # type: ignore
        v = sym(form[1])
        rhs = _as_float(form[2])
        if v and rhs is not None:
            return NumPred(v, op, rhs)
    return None

def _complement_num(pred: NumPred) -> List[NumPred]:
    match pred.op:
        case ">=": return [NumPred(pred.var, "<",  pred.rhs)]
        case ">":  return [NumPred(pred.var, "<=", pred.rhs)]
        case "<=": return [NumPred(pred.var, ">",  pred.rhs)]
        case "<":  return [NumPred(pred.var, ">=", pred.rhs)]
        case "=":  return [NumPred(pred.var, "<",  pred.rhs), NumPred(pred.var, ">", pred.rhs)]
    return []

def members_from_assert(form: S) -> Optional[List[ClauseMember]]:
    if not (isinstance(form, list) and form and sym(form[0]) == "assert"):
        return None
    body = form[1] if len(form) >= 2 else None
    if isinstance(body, list) and body and sym(body[0]) == "!":
        body = body[1]
    if isinstance(body, str):
        return [Literal(body, False)]
    if not (isinstance(body, list) and body):
        return None
    head = sym(body[0])
    if head == "not" and len(body) == 2:
        inner = body[1]
        # Case 1: (assert (not p)) where p is a boolean atom
        if isinstance(inner, str):
            return [Literal(inner, True)]
        # Case 2: (assert (not (<|<=|>|>=|= v c)))  → complement numeric predicate(s)
        if isinstance(inner, list):
            npi = _parse_num_pred(inner)
            if npi:
                return _complement_num(npi)
            # Case 3: (assert (not (and ...))) handled below (De Morgan to a single OR-clause)
            if inner and sym(inner[0]) == "and":
                members: List[ClauseMember] = []
                for arg in inner[1:]:
                    if isinstance(arg, str):
                        members.append(Literal(arg, True))
                    else:
                        npa = _parse_num_pred(arg)
                        if npa: members.extend(_complement_num(npa))
                        else:   return None
                return members
        return None
    np = _parse_num_pred(body)
    if np:
        return [np]
    if head == "=>" and len(body) == 3 and isinstance(body[1], str) and isinstance(body[2], str):
        return [Literal(body[1], True), Literal(body[2], False)]
    if head == "or" and len(body) >= 2:
        members: List[ClauseMember] = []
        for arg in body[1:]:
            if isinstance(arg, str):
                members.append(Literal(arg, False)); continue
            if isinstance(arg, list) and arg and sym(arg[0]) == "not":
                inner = arg[1] if len(arg) == 2 else None
                if isinstance(inner, str):
                    members.append(Literal(inner, True)); continue
                npi = _parse_num_pred(inner) if isinstance(inner, list) else None
                if npi:
                    members.extend(_complement_num(npi)); continue
                return None
            npi = _parse_num_pred(arg)
            if npi:
                members.append(npi); continue
            return None
        return members
    if head == "not" and len(body) == 2 and isinstance(body[1], list) and body[1]:
        inner = body[1]
        if sym(inner[0]) == "and":
            members: List[ClauseMember] = []
            for arg in inner[1:]:
                if isinstance(arg, str):
                    members.append(Literal(arg, True))
                else:
                    npi = _parse_num_pred(arg)
                    if npi: members.extend(_complement_num(npi))
                    else:   return None
            return members
    return None

# ============================== Timeframe parsing ==============================

_TIMEFRAME_TOKEN_PAT = r'(now|inthehistory|inthepast\d+(?:minutes|hours|days|weeks|months|years)|inthefuture\d*(?:minutes|hours|days|weeks|months|years)?|inthefuture)'
_TF_FINDER_RE = re.compile(r'_' + _TIMEFRAME_TOKEN_PAT + r'(?:_|$)')

_UNIT_HOURS = {
    "minutes": 1.0/60.0, "hours": 1.0, "days": 24.0, "weeks": 24.0*7.0,
    "months": 24.0*30.0, "years": 24.0*365.0,
}

def _split_base_timeframe(var_name: str) -> tuple[str, Optional[str]]:
    last: Optional[re.Match[str]] = None
    for m in _TF_FINDER_RE.finditer(var_name):
        last = m
    if not last:
        return var_name, None
    tf = last.group(1)
    matched = last.group(0)
    if matched.endswith("_"):
        base_var = var_name[:last.start()] + "_" + var_name[last.end():]
    else:
        base_var = var_name[:last.start()] + var_name[last.end():]
    return base_var, tf

def _tf_window_hours(token: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if token is None: return None, None
    if token == "now": return 0.0, 0.0
    if token == "inthehistory": return -1.0e9, 0.0
    m = re.match(r"inthepast(\d+)(minutes|hours|days|weeks|months|years)", token)
    if m:
        n, u = int(m.group(1)), m.group(2)
        return -(n * _UNIT_HOURS[u]), 0.0
    m = re.match(r"inthefuture(\d+)?(minutes|hours|days|weeks|months|years)?", token)
    if m:
        if m.group(1) and m.group(2):
            n, u = int(m.group(1)), m.group(2)
            return 0.0, (n * _UNIT_HOURS[u])
        return 0.0, 1.0e9
    return None, None

def _tf_direction(token: Optional[str]) -> Optional[str]:
    if token is None: return None
    if token == "now": return "now"
    if token == "inthehistory" or token.startswith("inthepast"): return "history"
    if token.startswith("inthefuture"): return "future"
    return None

# ============================== Signatures ==============================

def _signature_from_members(blits: List[Literal], nlits: List[NumPred]) -> str:
    parts: List[str] = []
    parts += [("~" if l.is_neg else "") + l.var for l in blits]
    parts += [f"{n.var}{n.op}{_canon_num(n.rhs)}" for n in nlits]
    parts.sort()
    return "|".join(parts)

def _signature_base_from_members(blits: List[Literal], nlits: List[NumPred]) -> str:
    parts: List[str] = []
    for l in blits:
        base_var, _ = _split_base_timeframe(l.var)
        parts.append(("~" if l.is_neg else "") + base_var)
    for n in nlits:
        base_var, _ = _split_base_timeframe(n.var)
        parts.append(f"{base_var}{n.op}{_canon_num(n.rhs)}")
    parts.sort()
    return "|".join(parts)

def _signature_stem_from_members(blits: List[Literal], nlits: List[NumPred]) -> str:
    items: List[str] = []
    for l in blits:
        base_var, _ = _split_base_timeframe(l.var)
        items.append("B" + ("~" if l.is_neg else "") + base_var)
    for n in nlits:
        base_var, _ = _split_base_timeframe(n.var)
        items.append("N" + base_var + n.op)
    items.sort()
    return "|".join(items)

# ============================== DB schema ==============================

DDL_BASE = [
    """
    CREATE TABLE IF NOT EXISTS trial_constraint_sides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nct_id  TEXT,
        kind    TEXT CHECK (kind in ('inclusion','exclusion')),
        variant TEXT NOT NULL CHECK (variant in ('main','assumed')) DEFAULT 'main',
        file_name TEXT UNIQUE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trial_constraint_sides_nct_kind_variant ON trial_constraint_sides(nct_id, kind, variant)",
    """
    CREATE TABLE IF NOT EXISTS constraint_clauses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        number_of_clause_members INTEGER NOT NULL,
        signature TEXT NOT NULL UNIQUE,
        signature_base TEXT,
        signature_stem TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_constraint_clauses_sig ON constraint_clauses(signature)",
    "CREATE INDEX IF NOT EXISTS idx_constraint_clauses_sig_base ON constraint_clauses(signature_base)",
    "CREATE INDEX IF NOT EXISTS idx_constraint_clauses_sig_stem ON constraint_clauses(signature_stem)",
    """
    CREATE TABLE IF NOT EXISTS constraint_clause_atoms (
        clause_id     INTEGER NOT NULL,
        literal_index INTEGER NOT NULL,
        var_name      TEXT NOT NULL,
        base_var      TEXT,
        timeframe     TEXT,
        tf_lb_hours   REAL,
        tf_ub_hours   REAL,
        is_neg        INTEGER NOT NULL CHECK (is_neg in (0,1)),
        PRIMARY KEY (clause_id, literal_index),
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_constraint_clause_atoms_var ON constraint_clause_atoms(var_name, is_neg)",
    "CREATE INDEX IF NOT EXISTS idx_constraint_clause_atoms_base ON constraint_clause_atoms(base_var, is_neg)",
    """
    CREATE TABLE IF NOT EXISTS numerical_constraint_predicates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        var_name TEXT NOT NULL,
        op TEXT NOT NULL CHECK (op IN ('>=','<=','>','<','=')),
        rhs REAL NOT NULL,
        UNIQUE (var_name, op, rhs)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_numerical_constraint_predicates_var ON numerical_constraint_predicates(var_name)",
    """
    CREATE TABLE IF NOT EXISTS numerical_constraint_clauses (
        clause_id INTEGER NOT NULL,
        numpred_id INTEGER NOT NULL,
        member_index INTEGER NOT NULL,
        PRIMARY KEY (clause_id, numpred_id),
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE,
        FOREIGN KEY (numpred_id) REFERENCES numerical_constraint_predicates(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_numerical_constraint_clauses_clause ON numerical_constraint_clauses(clause_id)",
    """
    CREATE TABLE IF NOT EXISTS trial_constraint_clauses (
        trial_id INTEGER NOT NULL,
        clause_id INTEGER NOT NULL,
        PRIMARY KEY (trial_id, clause_id),
        FOREIGN KEY (trial_id) REFERENCES trial_constraint_sides(id) ON DELETE CASCADE,
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trial_constraint_clauses_trial ON trial_constraint_clauses(trial_id)",
]

DDL_NUMERIC_RANGES = [
    """
    CREATE TABLE IF NOT EXISTS constraint_clause_numeric_range (
        clause_id    INTEGER NOT NULL,
        member_index INTEGER NOT NULL,
        var_name     TEXT    NOT NULL,
        lb           REAL,
        lb_inc       INTEGER NOT NULL,
        ub           REAL,
        ub_inc       INTEGER NOT NULL,
        base_var     TEXT,
        timeframe    TEXT,
        tf_lb_hours  REAL,
        tf_ub_hours  REAL,
        PRIMARY KEY (clause_id, member_index, var_name),
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cnr_var ON constraint_clause_numeric_range(var_name, lb, ub)",
    "CREATE INDEX IF NOT EXISTS idx_cnr_clause ON constraint_clause_numeric_range(clause_id, member_index)",
    "CREATE INDEX IF NOT EXISTS idx_cnr_base ON constraint_clause_numeric_range(base_var, lb, ub)",
]

DDL_MERGED = [
    """
    CREATE TABLE IF NOT EXISTS trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nct_id TEXT NOT NULL,
        inclusion_trial_side_id INTEGER,
        exclusion_trial_side_id INTEGER,   -- main exclusion (if any)
        assumed_trial_side_id   INTEGER,   -- assumed (treated as exclusion) (if any)
        label TEXT UNIQUE,
        FOREIGN KEY (inclusion_trial_side_id) REFERENCES trial_constraint_sides(id) ON DELETE CASCADE,
        FOREIGN KEY (exclusion_trial_side_id) REFERENCES trial_constraint_sides(id) ON DELETE CASCADE,
        FOREIGN KEY (assumed_trial_side_id)   REFERENCES trial_constraint_sides(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trials_nct ON trials(nct_id)",
    """
    CREATE TABLE IF NOT EXISTS trial_constraint_clauses (
        merged_trial_id INTEGER NOT NULL,
        clause_id INTEGER NOT NULL,
        PRIMARY KEY (merged_trial_id, clause_id),
        FOREIGN KEY (merged_trial_id) REFERENCES trials(id) ON DELETE CASCADE,
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trial_constraint_clauses_merged ON trial_constraint_clauses(merged_trial_id)",
]

DDL_FILTERED = [
    """
    CREATE TABLE IF NOT EXISTS top_common_constraint_clauses (
        kind TEXT NOT NULL CHECK (kind in ('inclusion','exclusion')),
        clause_id INTEGER NOT NULL,
        n_trials INTEGER NOT NULL,
        PRIMARY KEY (kind, clause_id),
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_top_common_kind ON top_common_constraint_clauses(kind)",
    """
    CREATE TABLE IF NOT EXISTS filtered_trial_constraint_clauses (
        trial_id INTEGER NOT NULL,
        clause_id INTEGER NOT NULL,
        PRIMARY KEY (trial_id, clause_id),
        FOREIGN KEY (trial_id) REFERENCES trial_constraint_sides(id) ON DELETE CASCADE,
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_filtered_trial ON filtered_trial_constraint_clauses(trial_id)",
    """
    CREATE TABLE IF NOT EXISTS filtered_merged_trial_constraint_clauses (
        merged_trial_id INTEGER NOT NULL,
        clause_id INTEGER NOT NULL,
        PRIMARY KEY (merged_trial_id, clause_id),
        FOREIGN KEY (merged_trial_id) REFERENCES trials(id) ON DELETE CASCADE,
        FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_f_mtc_merged ON filtered_merged_trial_constraint_clauses(merged_trial_id)",
]

DDL_FILTERED_PATTERNS = [
    """
    CREATE TABLE IF NOT EXISTS top_common_clause_patterns (
        kind TEXT NOT NULL CHECK (kind in ('inclusion','exclusion')),
        signature_base TEXT NOT NULL,
        n_trials INTEGER NOT NULL,
        PRIMARY KEY (kind, signature_base)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_top_common_patterns_kind ON top_common_clause_patterns(kind)",
]

DDL_VAR_CATALOG = [
    """
    CREATE TABLE IF NOT EXISTS predicate_catalog (
        var_name    TEXT PRIMARY KEY,
        base_var    TEXT,
        timeframe   TEXT,
        tf_lb_hours REAL,
        tf_ub_hours REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_predicate_catalog_base ON predicate_catalog(base_var)",
    "CREATE INDEX IF NOT EXISTS idx_predicate_catalog_tf   ON predicate_catalog(timeframe)",
]

DDL_STEM_DIRECTION = [
    """
    CREATE TABLE IF NOT EXISTS constraint_clause_usage_stem_direction (
        signature_stem TEXT NOT NULL,
        direction TEXT NOT NULL CHECK (direction in ('history','now','future')),
        number_of_clause_members INTEGER NOT NULL,
        n_sides_total        INTEGER NOT NULL,
        n_sides_inclusion    INTEGER NOT NULL,
        n_sides_exclusion    INTEGER NOT NULL,
        n_trials_total       INTEGER NOT NULL,
        n_trials_inclusion   INTEGER NOT NULL,
        n_trials_exclusion   INTEGER NOT NULL,
        PRIMARY KEY (signature_stem, direction)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cusd_stem ON constraint_clause_usage_stem_direction(signature_stem)",
    "CREATE INDEX IF NOT EXISTS idx_cusd_dir  ON constraint_clause_usage_stem_direction(direction)",
]

def _maybe_add_column(cur: sqlite3.Cursor, table: str, col_name: str, col_def: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")

def ensure_timeaware_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    _maybe_add_column(cur, "constraint_clauses", "signature_base", "TEXT")
    _maybe_add_column(cur, "constraint_clauses", "signature_stem", "TEXT")
    _maybe_add_column(cur, "constraint_clause_atoms",      "base_var",    "TEXT")
    _maybe_add_column(cur, "constraint_clause_atoms",      "timeframe",   "TEXT")
    _maybe_add_column(cur, "constraint_clause_atoms",      "tf_lb_hours", "REAL")
    _maybe_add_column(cur, "constraint_clause_atoms",      "tf_ub_hours", "REAL")
    _maybe_add_column(cur, "constraint_clause_numeric_range", "base_var",    "TEXT")
    _maybe_add_column(cur, "constraint_clause_numeric_range", "timeframe",   "TEXT")
    _maybe_add_column(cur, "constraint_clause_numeric_range", "tf_lb_hours", "REAL")
    _maybe_add_column(cur, "constraint_clause_numeric_range", "tf_ub_hours", "REAL")
    _maybe_add_column(cur, "trial_constraint_sides",          "variant",     "TEXT DEFAULT 'main'")
    # merged table extra column (assumed link)
    _maybe_add_column(cur, "trials", "assumed_trial_side_id", "INTEGER")
    _maybe_add_column(cur, "trials", "inclusion_trial_side_id", "INTEGER")
    _maybe_add_column(cur, "trials", "exclusion_trial_side_id", "INTEGER")

    conn.commit()

def ensure_views(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("DROP VIEW IF EXISTS vw_member_timeframes")
    cur.execute("""
    CREATE VIEW IF NOT EXISTS vw_member_timeframes AS
      SELECT 'bool' AS kind, cl.clause_id, cl.literal_index AS member_index,
             cl.var_name, cl.base_var, cl.timeframe, cl.tf_lb_hours, cl.tf_ub_hours
      FROM constraint_clause_atoms cl
      UNION ALL
      SELECT 'num'  AS kind, cnr.clause_id, cnr.member_index,
             cnr.var_name, cnr.base_var, cnr.timeframe, cnr.tf_lb_hours, cnr.tf_ub_hours
      FROM constraint_clause_numeric_range cnr
    """)
    conn.commit()

def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for stmt in DDL_BASE: cur.execute(stmt)
    for stmt in DDL_NUMERIC_RANGES: cur.execute(stmt)
    for stmt in DDL_MERGED: cur.execute(stmt)
    for stmt in DDL_FILTERED: cur.execute(stmt)
    for stmt in DDL_FILTERED_PATTERNS: cur.execute(stmt)
    for stmt in DDL_VAR_CATALOG: cur.execute(stmt)
    for stmt in DDL_STEM_DIRECTION: cur.execute(stmt)
    conn.commit()
    ensure_timeaware_schema(conn)
    ensure_views(conn)

# ============================== Filenames → trial meta ==============================

# UPDATED: allow optional suffix after digits (letters, numbers, '_' or '-')
_FILE_RE = re.compile(r'^(NCT\d+[A-Za-z0-9_-]*)_(inclusion|exclusion)_', re.I)

def _trial_meta_from_filename(fname: str) -> Tuple[str, str]:
    m = _FILE_RE.match(fname)
    if not m:
        return ("UNKNOWN", "inclusion")
    return (m.group(1), m.group(2).lower())

def _variant_from_filename(fname: str) -> str:
    # match .assumed_, _assumed_, -assumed-, or ending with .assumed(.smt2)?
    return "assumed" if re.search(r'(?:^|[._-])assumed(?:[._-]|$)', fname) else "main"

# ============================== Insert helpers ==============================

def upsert_predicate_catalog(conn: sqlite3.Connection, var_name: str) -> None:
    base_var, tf = _split_base_timeframe(var_name)
    tf_lb_h, tf_ub_h = _tf_window_hours(tf)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predicate_catalog(var_name, base_var, timeframe, tf_lb_hours, tf_ub_hours)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(var_name) DO UPDATE SET
            base_var=excluded.base_var,
            timeframe=excluded.timeframe,
            tf_lb_hours=excluded.tf_lb_hours,
            tf_ub_hours=excluded.tf_ub_hours
    """, (var_name, base_var, tf, tf_lb_h, tf_ub_h))

# UPDATED: make this self-healing on conflict (updates UNKNOWN, kind, and variant)
def upsert_trial_side(conn: sqlite3.Connection, file_name: str) -> int:
    nct, kind = _trial_meta_from_filename(file_name)
    variant = _variant_from_filename(file_name)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trial_constraint_sides (nct_id, kind, variant, file_name)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_name) DO UPDATE SET
            nct_id  = excluded.nct_id,
            kind    = excluded.kind,
            variant = excluded.variant
    """, (nct, kind, variant, file_name))
    cur.execute("SELECT id FROM trial_constraint_sides WHERE file_name = ?", (file_name,))
    return cur.fetchone()[0]

def upsert_numerical_predicate(conn: sqlite3.Connection, np: NumPred) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO numerical_constraint_predicates(var_name, op, rhs) VALUES (?, ?, ?)",
                (np.var, np.op, float(np.rhs)))
    cur.execute("SELECT id FROM numerical_constraint_predicates WHERE var_name=? AND op=? AND rhs=?",
                (np.var, np.op, float(np.rhs)))
    return cur.fetchone()[0]

def _range_from_pred(np: NumPred) -> Tuple[Optional[float], int, Optional[float], int]:
    if np.op == ">=":  return (float(np.rhs), 1, None, 1)
    if np.op ==  ">":  return (float(np.rhs), 0, None, 1)
    if np.op == "<=":  return (None, 1, float(np.rhs), 1)
    if np.op ==  "<":  return (None, 1, float(np.rhs), 0)
    if np.op ==  "=":  c = float(np.rhs); return (c, 1, c, 1)
    raise ValueError(f"unknown op {np.op}")

def _insert_numeric_ranges(cur: sqlite3.Cursor, clause_id: int, nlits: List[NumPred]) -> None:
    if not nlits: return
    rows = []
    for i, np in enumerate(nlits):
        lb, lb_inc, ub, ub_inc = _range_from_pred(np)
        base_var, tf = _split_base_timeframe(np.var)
        tf_lb_h, tf_ub_h = _tf_window_hours(tf)
        rows.append((clause_id, i, np.var, lb, lb_inc, ub, ub_inc, base_var, tf, tf_lb_h, tf_ub_h))
    cur.executemany("""
        INSERT OR IGNORE INTO constraint_clause_numeric_range
            (clause_id, member_index, var_name, lb, lb_inc, ub, ub_inc,
             base_var, timeframe, tf_lb_hours, tf_ub_hours)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

def upsert_clause_full(conn: sqlite3.Connection, blits: List[Literal], nlits: List[NumPred]) -> int:
    cur = conn.cursor()
    for l in blits: upsert_predicate_catalog(conn, l.var)
    for n in nlits: upsert_predicate_catalog(conn, n.var)
    sig_full  = _signature_from_members(blits, nlits)
    sig_base  = _signature_base_from_members(blits, nlits)
    sig_stem  = _signature_stem_from_members(blits, nlits)
    num_members = len(blits) + len(nlits)
    cur.execute("""
        INSERT OR IGNORE INTO constraint_clauses (number_of_clause_members, signature, signature_base, signature_stem)
        VALUES (?, ?, ?, ?)
    """, (num_members, sig_full, sig_base, sig_stem))
    cur.execute("SELECT id FROM constraint_clauses WHERE signature = ?", (sig_full,))
    cid = cur.fetchone()[0]
    cur.execute("SELECT 1 FROM constraint_clause_atoms WHERE clause_id = ? LIMIT 1", (cid,))
    if cur.fetchone() is None and blits:
        rows = []
        for i, l in enumerate(blits):
            base_var, tf = _split_base_timeframe(l.var)
            tf_lb_h, tf_ub_h = _tf_window_hours(tf)
            rows.append((cid, i, l.var, base_var, tf, tf_lb_h, tf_ub_h, 1 if l.is_neg else 0))
        cur.executemany(
            """INSERT INTO constraint_clause_atoms
               (clause_id, literal_index, var_name, base_var, timeframe, tf_lb_hours, tf_ub_hours, is_neg)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
    cur.execute("SELECT 1 FROM numerical_constraint_clauses WHERE clause_id = ? LIMIT 1", (cid,))
    already = cur.fetchone() is not None
    if not already and nlits:
        rows = []
        for i, n in enumerate(nlits):
            nid = upsert_numerical_predicate(conn, n)
            rows.append((cid, nid, i))
        cur.executemany(
            "INSERT OR IGNORE INTO numerical_constraint_clauses (clause_id, numpred_id, member_index) VALUES (?, ?, ?)",
            rows
        )
    cur.execute("SELECT 1 FROM constraint_clause_numeric_range WHERE clause_id = ? LIMIT 1", (cid,))
    if cur.fetchone() is None and nlits:
        _insert_numeric_ranges(cur, cid, nlits)
    return cid

def link_trial_side_clause(conn: sqlite3.Connection, trial_id: int, clause_id: int) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO trial_constraint_clauses (trial_id, clause_id) VALUES (?, ?)",
        (int(trial_id), int(clause_id)),
    )

# ============================== Parse projected files into constraint_clauses ==============================

def parse_projected_file_to_constraint_clauses(path: Path) -> List[Tuple[List[Literal], List[NumPred]]]:
    text = path.read_text(encoding="utf-8")
    ast = parse_sexpr(text)
    results: List[Tuple[List[Literal], List[NumPred]]] = []
    for form in ast:
        members = members_from_assert(form)
        if not members:
            continue
        _sig, blits, nlits = _normalize_members(members)
        results.append((blits, nlits))
    return results

# ============================== Build DB from projected SMT ==============================

def build_db(*, projected_dir: str | Path, db_path: str | Path, glob: str = "*.smt2") -> None:
    projected_dir = Path(projected_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    try:
        # --- PRAGMAs outside any transaction (autocommit) ---
        old_iso = conn.isolation_level
        conn.isolation_level = None            # enter autocommit
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        conn.isolation_level = old_iso         # restore transactional mode

        # Now it’s safe to do schema + inserts inside normal transactions
        ensure_schema(conn)

        cur = conn.cursor()
        files = sorted(projected_dir.glob(glob))
        if not files:
            print(f"[info] No projected SMT files in {projected_dir}")
            return

        for f in files:
            trial_side_id = upsert_trial_side(conn, f.name)
            pairs = parse_projected_file_to_constraint_clauses(f)
            if not pairs:
                print(f"[warn] No parsable asserts in {f.name}")
                # still recorded as a side with 0 constraint_clauses
                continue
            for blits, nlits in pairs:
                cid = upsert_clause_full(conn, blits, nlits)
                link_trial_side_clause(conn, trial_side_id, cid)
            conn.commit()
            print(f"[ok] {f.name}: {len(pairs)} constraint_clauses")

    finally:
        conn.close()

# ============================== Seeding trials from other sources ==============================

def seed_trials_from_sources(*, db_path: str | Path) -> None:
    """
    Ensure we have a trials row for every NCT id known from:
      • trial_constraint_sides (even if that side produced 0 constraint_clauses)
      • disease_constraint_alternatives / disease_constraint_atoms
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS tmp_candidate_ncts")
        cur.execute("CREATE TEMP TABLE tmp_candidate_ncts(nct_id TEXT PRIMARY KEY)")

        # From trial_constraint_sides
        cur.execute("""
            INSERT OR IGNORE INTO tmp_candidate_ncts(nct_id)
            SELECT DISTINCT nct_id FROM trial_constraint_sides
            WHERE nct_id IS NOT NULL AND nct_id <> ''
        """)

        # From disease_constraint_alternatives
        if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_alternatives'").fetchone():
            cols = {r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_alternatives)")}
            if "nct_id" in cols:
                cur.execute("""
                    INSERT OR IGNORE INTO tmp_candidate_ncts(nct_id)
                    SELECT DISTINCT nct_id FROM disease_constraint_alternatives
                    WHERE nct_id IS NOT NULL AND nct_id <> ''
                """)
            elif "trial_id" in cols:
                cur.execute("""
                    INSERT OR IGNORE INTO tmp_candidate_ncts(nct_id)
                    SELECT DISTINCT trial_id FROM disease_constraint_alternatives
                    WHERE trial_id IS NOT NULL AND trial_id <> ''
                """)

        # From disease_constraint_atoms
        if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_atoms'").fetchone():
            cols = {r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_atoms)")}
            if "nct_id" in cols:
                cur.execute("""
                    INSERT OR IGNORE INTO tmp_candidate_ncts(nct_id)
                    SELECT DISTINCT nct_id FROM disease_constraint_atoms
                    WHERE nct_id IS NOT NULL AND nct_id <> ''
                """)
            elif "trial_id" in cols:
                cur.execute("""
                    INSERT OR IGNORE INTO tmp_candidate_ncts(nct_id)
                    SELECT DISTINCT trial_id FROM disease_constraint_atoms
                    WHERE trial_id IS NOT NULL AND trial_id <> ''
                """)

        # Backfill trials (bare rows; merged links filled later)
        cur.execute("""
            INSERT OR IGNORE INTO trials (nct_id, label)
            SELECT nct_id, nct_id || '_merged'
            FROM tmp_candidate_ncts
            WHERE nct_id IS NOT NULL AND nct_id <> ''
        """)
        conn.commit()
        print("[ok] seeded trials from trial_constraint_sides/disease tables (even with 0 constraint_clauses)")
    finally:
        conn.close()

# ============================== Merged materialization (ASSUMED→EXCLUSION) ==============================

def materialize_merged_trials(*, db_path: str | Path, require_both: bool = True, clear_existing: bool = True) -> None:
    """
    Create/refresh trials:
      inclusion := main inclusion ONLY
      exclusion := (main exclusion) ∪ (assumed)   [both linked distinctly; unioned in constraint_clauses]
    Notes:
      • Does NOT clear the `trials` table; only clears `trial_constraint_clauses` when requested,
        so bare seeded trial rows remain.
      • Upserts rows by `label` to fill in side links.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DELETE FROM trial_constraint_clauses")
            conn.commit()

        cur.execute("SELECT id, nct_id, kind, variant FROM trial_constraint_sides")
        rows = cur.fetchall()
        by_nct: Dict[str, Dict[str, Dict[str, Optional[int]]]] = {}
        for tid, nct, kind, variant in rows:
            d = by_nct.setdefault(nct, {"inclusion": {"main": None}, "exclusion": {"main": None}, "assumed": {"assumed": None}})
            if variant == "assumed":
                d["assumed"]["assumed"] = tid
            else:
                d[kind]["main"] = tid

        created = 0
        for nct, sides in by_nct.items():
            inc_main = sides["inclusion"]["main"]
            exc_main = sides["exclusion"]["main"]
            assumed  = sides["assumed"]["assumed"]

            if require_both and not (inc_main and (exc_main or assumed)):
                continue
            if not (inc_main or exc_main or assumed):
                continue  # nothing to merge at all

            label = f"{nct}_merged"
            # Upsert into trials without deleting seeded rows
            cur.execute(f"""
                INSERT INTO trials (nct_id, inclusion_trial_side_id, exclusion_trial_side_id, assumed_trial_side_id, label)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(label) DO UPDATE SET
                  nct_id=excluded.nct_id,
                  inclusion_trial_side_id=excluded.inclusion_trial_side_id,
                  exclusion_trial_side_id=excluded.exclusion_trial_side_id,
                  assumed_trial_side_id=excluded.assumed_trial_side_id
            """, (nct, inc_main, exc_main, assumed, label))
            cur.execute("SELECT id FROM trials WHERE label = ?", (label,))
            mt_id = cur.fetchone()[0]
            created += 1

            clause_ids: Set[int] = set()
            for side_id in (inc_main, exc_main, assumed):
                if side_id:
                    cur.execute("SELECT clause_id FROM trial_constraint_clauses WHERE trial_id = ?", (side_id,))
                    clause_ids.update(c[0] for c in cur.fetchall())
            cur.executemany(
                "INSERT OR IGNORE INTO trial_constraint_clauses (merged_trial_id, clause_id) VALUES (?, ?)",
                [(mt_id, cid) for cid in clause_ids]
            )
        conn.commit()
        print(f"[ok] materialized {created} merged NCT rows (assumed→exclusion; require_both={require_both})")
    finally:
        conn.close()

# ============================== Filtered subset materialization (ASSUMED→EXCLUSION) ==============================

# Helper: CASE to map assumed→exclusion
EFFECTIVE_KIND = "CASE WHEN t.variant='assumed' THEN 'exclusion' ELSE t.kind END"

COMMON_BY_KIND_SQL = f"""
SELECT c.id AS clause_id, COUNT(*) AS n_trials
FROM constraint_clauses c
JOIN trial_constraint_clauses tc ON tc.clause_id = c.id
JOIN trial_constraint_sides t ON t.id = tc.trial_id
GROUP BY c.id
HAVING {EFFECTIVE_KIND} = :kind
ORDER BY n_trials DESC, c.signature
LIMIT :limit_n
"""

COMMON_BY_KIND_BASE_SQL = f"""
SELECT c.signature_base AS signature_base, COUNT(*) AS n_trials
FROM constraint_clauses c
JOIN trial_constraint_clauses tc ON tc.clause_id = c.id
JOIN trial_constraint_sides t ON t.id = tc.trial_id
GROUP BY c.signature_base
HAVING {EFFECTIVE_KIND} = :kind
ORDER BY n_trials DESC, c.signature_base
LIMIT :limit_n
"""

def rebuild_top_common_constraint_clauses(*, db_path: str | Path, top_n_inclusion: int, top_n_exclusion: int, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DELETE FROM top_common_constraint_clauses")
        for kind, limit_n in (("inclusion", top_n_inclusion), ("exclusion", top_n_exclusion)):
            cur.execute(COMMON_BY_KIND_SQL, {"kind": kind, "limit_n": int(limit_n)})
            rows = cur.fetchall()
            cur.executemany(
                "INSERT OR REPLACE INTO top_common_constraint_clauses(kind, clause_id, n_trials) VALUES (?,?,?)",
                [(kind, r[0], r[1]) for r in rows]
            )
        conn.commit()
        print(f"[ok] top_common_constraint_clauses rebuilt (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_filtered_trial_constraint_clauses(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DELETE FROM filtered_trial_constraint_clauses")
        # Use effective kind for matching to top_common_constraint_clauses
        cur.execute(f"""
        INSERT OR IGNORE INTO filtered_trial_constraint_clauses(trial_id, clause_id)
        SELECT t.id, tc.clause_id
        FROM trial_constraint_sides t
        JOIN trial_constraint_clauses tc ON tc.trial_id = t.id
        JOIN top_common_constraint_clauses top ON top.kind = ({EFFECTIVE_KIND}) AND top.clause_id = tc.clause_id
        """)
        conn.commit()
        print(f"[ok] filtered_trial_constraint_clauses rebuilt (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_top_common_clause_patterns(*, db_path: str | Path, top_n_inclusion: int, top_n_exclusion: int, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DELETE FROM top_common_clause_patterns")
        for kind, limit_n in (("inclusion", top_n_inclusion), ("exclusion", top_n_exclusion)):
            cur.execute(COMMON_BY_KIND_BASE_SQL, {"kind": kind, "limit_n": int(limit_n)})
            rows = cur.fetchall()
            cur.executemany(
                "INSERT OR REPLACE INTO top_common_clause_patterns(kind, signature_base, n_trials) VALUES (?,?,?)",
                [(kind, r[0], r[1]) for r in rows]
            )
        conn.commit()
        print(f"[ok] top_common_clause_patterns rebuilt (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_filtered_trial_constraint_clauses_by_pattern(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DELETE FROM filtered_trial_constraint_clauses")
        cur.execute(f"""
        INSERT OR IGNORE INTO filtered_trial_constraint_clauses(trial_id, clause_id)
        SELECT t.id, tc.clause_id
        FROM trial_constraint_sides t
        JOIN trial_constraint_clauses tc ON tc.trial_id = t.id
        JOIN constraint_clauses c ON c.id = tc.clause_id
        JOIN top_common_clause_patterns p ON p.kind = ({EFFECTIVE_KIND}) AND p.signature_base = c.signature_base
        """)
        conn.commit()
        print(f"[ok] filtered_trial_constraint_clauses rebuilt (by pattern; assumed→exclusion)")
    finally:
        conn.close()

def materialize_filtered_merged_trial_constraint_clauses(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DELETE FROM filtered_merged_trial_constraint_clauses")
        # inclusion(main)
        cur.execute("""
        INSERT OR IGNORE INTO filtered_merged_trial_constraint_clauses(merged_trial_id, clause_id)
        SELECT mt.id, ftc.clause_id
        FROM trials mt
        JOIN trial_constraint_sides ti ON ti.id = mt.inclusion_trial_side_id
        JOIN filtered_trial_constraint_clauses ftc ON ftc.trial_id = ti.id
        """)
        # exclusion(main)
        cur.execute("""
        INSERT OR IGNORE INTO filtered_merged_trial_constraint_clauses(merged_trial_id, clause_id)
        SELECT mt.id, ftc.clause_id
        FROM trials mt
        JOIN trial_constraint_sides te ON te.id = mt.exclusion_trial_side_id
        JOIN filtered_trial_constraint_clauses ftc ON ftc.trial_id = te.id
        """)
        # assumed (as exclusion)
        cur.execute("""
        INSERT OR IGNORE INTO filtered_merged_trial_constraint_clauses(merged_trial_id, clause_id)
        SELECT mt.id, ftc.clause_id
        FROM trials mt
        JOIN trial_constraint_sides ta ON ta.id = mt.assumed_trial_side_id
        JOIN filtered_trial_constraint_clauses ftc ON ftc.trial_id = ta.id
        """)
        conn.commit()
        print("[ok] filtered_merged_trial_constraint_clauses materialized (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_filtered_subset_full(*, db_path: str | Path, top_n_inclusion: int, top_n_exclusion: int, require_both_for_merge: bool = True) -> None:
    rebuild_top_common_constraint_clauses(db_path=db_path, top_n_inclusion=top_n_inclusion, top_n_exclusion=top_n_exclusion, clear_existing=True)
    rebuild_filtered_trial_constraint_clauses(db_path=db_path, clear_existing=True)
    materialize_merged_trials(db_path=db_path, require_both=require_both_for_merge, clear_existing=False)
    materialize_filtered_merged_trial_constraint_clauses(db_path=db_path, clear_existing=True)

def rebuild_filtered_subset_by_pattern(*, db_path: str | Path, top_n_inclusion: int, top_n_exclusion: int, require_both_for_merge: bool = True) -> None:
    rebuild_top_common_clause_patterns(db_path=db_path, top_n_inclusion=top_n_inclusion, top_n_exclusion=top_n_exclusion, clear_existing=True)
    rebuild_filtered_trial_constraint_clauses_by_pattern(db_path=db_path, clear_existing=True)
    materialize_merged_trials(db_path=db_path, require_both=require_both_for_merge, clear_existing=False)
    materialize_filtered_merged_trial_constraint_clauses(db_path=db_path, clear_existing=True)

# ============================== Usage analytics (ASSUMED→EXCLUSION) ==============================

def rebuild_constraint_clause_usage(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("DROP VIEW IF EXISTS vw_clause_reuse")
        if clear_existing:
            cur.execute("DROP TABLE IF EXISTS constraint_clause_usage")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS constraint_clause_usage (
            clause_id INTEGER PRIMARY KEY,
            signature TEXT NOT NULL,
            number_of_clause_members INTEGER NOT NULL,
            n_sides_total INTEGER NOT NULL,
            n_sides_inclusion INTEGER NOT NULL,
            n_sides_exclusion INTEGER NOT NULL,
            n_trials_total INTEGER NOT NULL,
            n_trials_inclusion INTEGER NOT NULL,
            n_trials_exclusion INTEGER NOT NULL,
            FOREIGN KEY (clause_id) REFERENCES constraint_clauses(id) ON DELETE CASCADE
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_constraint_clause_usage_sig ON constraint_clause_usage(signature)")
        cur.execute(f"""
        INSERT OR REPLACE INTO constraint_clause_usage (
            clause_id, signature, number_of_clause_members,
            n_sides_total, n_sides_inclusion, n_sides_exclusion,
            n_trials_total, n_trials_inclusion, n_trials_exclusion
        )
        SELECT
            c.id,
            c.signature,
            c.number_of_clause_members,
            COUNT(*) AS n_sides_total,
            SUM(CASE WHEN ({EFFECTIVE_KIND})='inclusion' THEN 1 ELSE 0 END) AS n_sides_inclusion,
            SUM(CASE WHEN ({EFFECTIVE_KIND})='exclusion' THEN 1 ELSE 0 END) AS n_sides_exclusion,
            COUNT(DISTINCT t.nct_id) AS n_trials_total,
            COUNT(DISTINCT CASE WHEN ({EFFECTIVE_KIND})='inclusion' THEN t.nct_id END) AS n_trials_inclusion,
            COUNT(DISTINCT CASE WHEN ({EFFECTIVE_KIND})='exclusion' THEN t.nct_id END) AS n_trials_exclusion
        FROM trial_constraint_clauses tc
        JOIN trial_constraint_sides t  ON t.id = tc.trial_id
        JOIN constraint_clauses c ON c.id = tc.clause_id
        GROUP BY c.id, c.signature, c.number_of_clause_members
        """)
        conn.commit()
        cur.execute("""
        CREATE VIEW IF NOT EXISTS vw_clause_reuse AS
        SELECT
          cu.clause_id, cu.signature, cu.number_of_clause_members,
          cu.n_sides_total, cu.n_sides_inclusion, cu.n_sides_exclusion,
          cu.n_trials_total, cu.n_trials_inclusion, cu.n_trials_exclusion
        FROM constraint_clause_usage cu
        """)
        conn.commit()
        print(f"[ok] constraint_clause_usage rebuilt (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_constraint_clause_usage_stem(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DROP VIEW IF EXISTS vw_clause_reuse_stem")
            cur.execute("DROP TABLE IF EXISTS constraint_clause_usage_stem")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS constraint_clause_usage_stem (
            signature_stem TEXT PRIMARY KEY,
            number_of_clause_members  INTEGER NOT NULL,
            n_sides_total        INTEGER NOT NULL,
            n_sides_inclusion    INTEGER NOT NULL,
            n_sides_exclusion    INTEGER NOT NULL,
            n_trials_total       INTEGER NOT NULL,
            n_trials_inclusion   INTEGER NOT NULL,
            n_trials_exclusion   INTEGER NOT NULL
        )
        """)
        cur.execute(f"""
        INSERT OR REPLACE INTO constraint_clause_usage_stem (
            signature_stem, number_of_clause_members,
            n_sides_total, n_sides_inclusion, n_sides_exclusion,
            n_trials_total, n_trials_inclusion, n_trials_exclusion
        )
        SELECT
            c.signature_stem,
            MIN(c.number_of_clause_members) AS number_of_clause_members,
            COUNT(DISTINCT t.id) AS n_sides_total,
            COUNT(DISTINCT CASE WHEN ({EFFECTIVE_KIND})='inclusion' THEN t.id END) AS n_sides_inclusion,
            COUNT(DISTINCT CASE WHEN ({EFFECTIVE_KIND})='exclusion' THEN t.id END) AS n_sides_exclusion,
            COUNT(DISTINCT t.nct_id) AS n_trials_total,
            COUNT(DISTINCT CASE WHEN ({EFFECTIVE_KIND})='inclusion' THEN t.nct_id END) AS n_trials_inclusion,
            COUNT(DISTINCT CASE WHEN ({EFFECTIVE_KIND})='exclusion' THEN t.nct_id END) AS n_trials_exclusion
        FROM trial_constraint_clauses tc
        JOIN trial_constraint_sides  t ON t.id = tc.trial_id
        JOIN constraint_clauses c ON c.id = tc.clause_id
        GROUP BY c.signature_stem
        """)
        cur.execute("CREATE VIEW IF NOT EXISTS vw_clause_reuse_stem AS SELECT * FROM constraint_clause_usage_stem")
        conn.commit()
        print(f"[ok] constraint_clause_usage_stem rebuilt (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_constraint_clause_usage_stem_direction(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DROP VIEW IF EXISTS vw_clause_reuse_stem_direction")
            cur.execute("DELETE FROM constraint_clause_usage_stem_direction")
        cur.execute(f"""
        WITH lit AS (
            SELECT
              t.id      AS trial_id,
              t.nct_id  AS nct_id,
              ({EFFECTIVE_KIND}) AS kind,
              c.signature_stem AS stem,
              c.number_of_clause_members AS number_of_clause_members,
              CASE
                WHEN cl.timeframe = 'now' THEN 'now'
                WHEN cl.timeframe = 'inthehistory' OR cl.timeframe LIKE 'inthepast%' THEN 'history'
                WHEN cl.timeframe LIKE 'inthefuture%' OR cl.timeframe = 'inthefuture' THEN 'future'
              END AS direction
            FROM trial_constraint_clauses tc
            JOIN trial_constraint_sides t  ON t.id = tc.trial_id
            JOIN constraint_clauses c ON c.id = tc.clause_id
            JOIN constraint_clause_atoms cl ON cl.clause_id = tc.clause_id
        ),
        num AS (
            SELECT
              t.id      AS trial_id,
              t.nct_id  AS nct_id,
              ({EFFECTIVE_KIND}) AS kind,
              c.signature_stem AS stem,
              c.number_of_clause_members AS number_of_clause_members,
              CASE
                WHEN cnr.timeframe = 'now' THEN 'now'
                WHEN cnr.timeframe = 'inthehistory' OR cnr.timeframe LIKE 'inthepast%' THEN 'history'
                WHEN cnr.timeframe LIKE 'inthefuture%' OR cnr.timeframe = 'inthefuture' THEN 'future'
              END AS direction
            FROM trial_constraint_clauses tc
            JOIN trial_constraint_sides t  ON t.id = tc.trial_id
            JOIN constraint_clauses c ON c.id = tc.clause_id
            JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = tc.clause_id
        ),
        members AS (
            SELECT * FROM lit
            UNION ALL
            SELECT * FROM num
        ),
        dedup AS (
            SELECT DISTINCT trial_id, nct_id, kind, stem, number_of_clause_members, direction
            FROM members
            WHERE direction IS NOT NULL
        )
        INSERT OR REPLACE INTO constraint_clause_usage_stem_direction (
            signature_stem, direction, number_of_clause_members,
            n_sides_total, n_sides_inclusion, n_sides_exclusion,
            n_trials_total, n_trials_inclusion, n_trials_exclusion
        )
        SELECT
            stem AS signature_stem,
            direction,
            MIN(number_of_clause_members) AS number_of_clause_members,
            COUNT(DISTINCT trial_id) AS n_sides_total,
            COUNT(DISTINCT CASE WHEN kind='inclusion' THEN trial_id END) AS n_sides_inclusion,
            COUNT(DISTINCT CASE WHEN kind='exclusion' THEN trial_id END) AS n_sides_exclusion,
            COUNT(DISTINCT nct_id) AS n_trials_total,
            COUNT(DISTINCT CASE WHEN kind='inclusion' THEN nct_id END) AS n_trials_inclusion,
            COUNT(DISTINCT CASE WHEN kind='exclusion' THEN nct_id END) AS n_trials_exclusion
        FROM dedup
        GROUP BY stem, direction
        """)
        cur.execute("CREATE VIEW IF NOT EXISTS vw_clause_reuse_stem_direction AS SELECT * FROM constraint_clause_usage_stem_direction")
        conn.commit()
        print(f"[ok] constraint_clause_usage_stem_direction rebuilt (assumed→exclusion)")
    finally:
        conn.close()

def rebuild_directional_usage(*, db_path: str | Path, clear_existing: bool = True) -> None:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        if clear_existing:
            cur.execute("DROP VIEW IF EXISTS vw_predicate_usage_directional")
            cur.execute("DROP VIEW IF EXISTS vw_constraint_direction_usage")
            cur.execute("DROP TABLE IF EXISTS predicate_usage_directional")
            cur.execute("DROP TABLE IF EXISTS constraint_direction_usage")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predicate_usage_directional (
            base_var TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction in ('history','now','future')),
            n_sides_total INTEGER NOT NULL,
            n_sides_inclusion INTEGER NOT NULL,
            n_sides_exclusion INTEGER NOT NULL,
            n_trials_total INTEGER NOT NULL,
            n_trials_inclusion INTEGER NOT NULL,
            n_trials_exclusion INTEGER NOT NULL,
            PRIMARY KEY (base_var, direction)
        )
        """)
        cur.execute(f"""
        WITH m_bool AS (
            SELECT t.id AS trial_id, t.nct_id, ({EFFECTIVE_KIND}) AS kind, cl.base_var, cl.timeframe
            FROM trial_constraint_clauses tc
            JOIN trial_constraint_sides t ON t.id = tc.trial_id
            JOIN constraint_clause_atoms cl ON cl.clause_id = tc.clause_id
        ),
        m_num AS (
            SELECT t.id AS trial_id, t.nct_id, ({EFFECTIVE_KIND}) AS kind, cnr.base_var, cnr.timeframe
            FROM trial_constraint_clauses tc
            JOIN trial_constraint_sides t ON t.id = tc.trial_id
            JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = tc.clause_id
        ),
        members AS (
            SELECT * FROM m_bool
            UNION ALL
            SELECT * FROM m_num
        ),
        tagged AS (
            SELECT
              trial_id, nct_id, kind, base_var,
              CASE
                WHEN timeframe = 'now' THEN 'now'
                WHEN timeframe = 'inthehistory' OR timeframe LIKE 'inthepast%' THEN 'history'
                WHEN timeframe LIKE 'inthefuture%' OR timeframe = 'inthefuture' THEN 'future'
              END AS direction
            FROM members
        )
        INSERT OR REPLACE INTO predicate_usage_directional (
            base_var, direction,
            n_sides_total, n_sides_inclusion, n_sides_exclusion,
            n_trials_total, n_trials_inclusion, n_trials_exclusion
        )
        SELECT
          base_var, direction,
          COUNT(DISTINCT trial_id) AS n_sides_total,
          COUNT(DISTINCT CASE WHEN kind='inclusion' THEN trial_id END) AS n_sides_inclusion,
          COUNT(DISTINCT CASE WHEN kind='exclusion' THEN trial_id END) AS n_sides_exclusion,
          COUNT(DISTINCT nct_id) AS n_trials_total,
          COUNT(DISTINCT CASE WHEN kind='inclusion' THEN nct_id END) AS n_trials_inclusion,
          COUNT(DISTINCT CASE WHEN kind='exclusion' THEN nct_id END) AS n_trials_exclusion
        FROM tagged
        WHERE direction IS NOT NULL AND base_var IS NOT NULL
        GROUP BY base_var, direction
        """)
        cur.execute("CREATE VIEW IF NOT EXISTS vw_predicate_usage_directional AS SELECT * FROM predicate_usage_directional")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS constraint_direction_usage (
            direction TEXT PRIMARY KEY CHECK (direction in ('history','now','future')),
            n_sides_total INTEGER NOT NULL,
            n_sides_inclusion INTEGER NOT NULL,
            n_sides_exclusion INTEGER NOT NULL,
            n_trials_total INTEGER NOT NULL,
            n_trials_inclusion INTEGER NOT NULL,
            n_trials_exclusion INTEGER NOT NULL
        )
        """)
        cur.execute("DELETE FROM constraint_direction_usage")
        cur.execute(f"""
        INSERT INTO constraint_direction_usage (
            direction, n_sides_total, n_sides_inclusion, n_sides_exclusion,
            n_trials_total, n_trials_inclusion, n_trials_exclusion
        )
        SELECT
          direction,
          COUNT(DISTINCT trial_id) AS n_sides_total,
          COUNT(DISTINCT CASE WHEN kind='inclusion' THEN trial_id END) AS n_sides_inclusion,
          COUNT(DISTINCT CASE WHEN kind='exclusion' THEN trial_id END) AS n_sides_exclusion,
          COUNT(DISTINCT nct_id) AS n_trials_total,
          COUNT(DISTINCT CASE WHEN kind='inclusion' THEN nct_id END) AS n_trials_inclusion,
          COUNT(DISTINCT CASE WHEN kind='exclusion' THEN nct_id END) AS n_trials_exclusion
        FROM (
            SELECT t.id AS trial_id, t.nct_id, ({EFFECTIVE_KIND}) AS kind,
                   CASE
                     WHEN cl.timeframe = 'now' THEN 'now'
                     WHEN cl.timeframe = 'inthehistory' OR cl.timeframe LIKE 'inthepast%' THEN 'history'
                     WHEN cl.timeframe LIKE 'inthefuture%' OR cl.timeframe = 'inthefuture' THEN 'future'
                   END AS direction
            FROM trial_constraint_clauses tc
            JOIN trial_constraint_sides t ON t.id = tc.trial_id
            JOIN constraint_clause_atoms cl ON cl.clause_id = tc.clause_id
            UNION ALL
            SELECT t.id, t.nct_id, ({EFFECTIVE_KIND}) AS kind,
                   CASE
                     WHEN cnr.timeframe = 'now' THEN 'now'
                     WHEN cnr.timeframe = 'inthehistory' OR cnr.timeframe LIKE 'inthepast%' THEN 'history'
                     WHEN cnr.timeframe LIKE 'inthefuture%' OR cnr.timeframe = 'inthefuture' THEN 'future'
                   END AS direction
            FROM trial_constraint_clauses tc
            JOIN trial_constraint_sides t ON t.id = tc.trial_id
            JOIN constraint_clause_numeric_range cnr ON cnr.clause_id = tc.clause_id
        ) AS all_dirs
        WHERE direction IS NOT NULL
        GROUP BY direction
        """)
        cur.execute("CREATE VIEW IF NOT EXISTS vw_constraint_direction_usage AS SELECT * FROM constraint_direction_usage")
        conn.commit()
        print(f"[ok] directional usage rebuilt (assumed→exclusion)")
    finally:
        conn.close()

# ============================== Optional helpers ==============================

def clause_members_as_text(*, db_path: str | Path, clause_id: int) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT var_name, is_neg
        FROM constraint_clause_atoms
        WHERE clause_id = ?
        ORDER BY literal_index ASC
        """, (clause_id,))
        blits = [(r[0], r[1]) for r in cur.fetchall()]
        cur.execute("""
        SELECT np.var_name, np.op, np.rhs, cn.member_index
        FROM numerical_constraint_clauses cn
        JOIN numerical_constraint_predicates np ON np.id = cn.numpred_id
        WHERE cn.clause_id = ?
        ORDER BY cn.member_index ASC
        """, (clause_id,))
        nlits = [(r[0], r[1], r[2]) for r in cur.fetchall()]
        parts = []
        parts.extend([("~" if is_neg else "") + var for (var, is_neg) in blits])
        parts.extend([f"{v}{op}{rhs:g}" for (v, op, rhs) in nlits])
        return " OR ".join(parts) if parts else "(empty)"
    finally:
        conn.close()

def get_top_reused_expressions(*, db_path: str | Path, limit: int = 50, order_by: str = "n_trials_total", min_members: int = 1) -> List[Dict[str, Any]]:
    valid_cols = {
        "n_trials_total","n_trials_inclusion","n_trials_exclusion",
        "n_sides_total","n_sides_inclusion","n_sides_exclusion"
    }
    if order_by not in valid_cols:
        raise ValueError(f"order_by must be one of {sorted(valid_cols)}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(f"""
        SELECT clause_id, signature, number_of_clause_members,
               n_sides_total, n_sides_inclusion, n_sides_exclusion,
               n_trials_total, n_trials_inclusion, n_trials_exclusion
        FROM constraint_clause_usage
        WHERE number_of_clause_members >= ?
        ORDER BY {order_by} DESC, n_sides_total DESC, signature ASC
        LIMIT ?
        """, (int(min_members), int(limit)))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

# ============================== CLI ==============================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build/materialize SMT clause DB (timeframe-aware; stems; directional usage) with assumed→exclusion.")
    ap.add_argument("--projected-dir", type=Path, required=False, default="../../build/canon_projection/_projected_smt",
                    help="Directory containing projected SMT2 files")
    ap.add_argument("--db", "--db-path", dest="db_path", type=Path, required=False, default="../../build/trial.db",
                    help="SQLite DB path to create/update")

    # Materialization toggles
    ap.add_argument("--merge-require-both", action="store_true",
                    help="Only create merged rows when inclusion(main) exists and at least one of exclusion(main)/assumed exists")
    ap.add_argument("--skip-filtered", action="store_true", help="Skip building filtered subset")
    ap.add_argument("--skip-merge", action="store_true", help="Skip merged materialization")
    ap.add_argument("--skip-reuse", action="store_true", help="Skip usage analytics (all kinds)")

    # Filtered knobs
    ap.add_argument("--top-n-incl", type=int, default=10_000_000, help="Top-N inclusion (filtered subset)")
    ap.add_argument("--top-n-excl", type=int, default=10_000_000, help="Top-N exclusion (filtered subset)")
    ap.add_argument("--filtered-mode", choices=["base","full"], default="base",
                    help="Filtered mode: 'base' (time-agnostic patterns) or 'full' (time-aware constraint_clauses)")

    args = ap.parse_args()

    build_db(projected_dir=args.projected_dir, db_path=args.db_path)

    # NEW: ensure every known NCT has a row in `trials` even if no constraint_clauses were produced
    seed_trials_from_sources(db_path=args.db_path)

    if not args.skip_merge:
        materialize_merged_trials(db_path=args.db_path, require_both=args.merge_require_both, clear_existing=True)

    if not args.skip_filtered:
        if args.filtered_mode == "base":
            rebuild_filtered_subset_by_pattern(db_path=args.db_path,
                                               top_n_inclusion=args.top_n_incl,
                                               top_n_exclusion=args.top_n_excl,
                                               require_both_for_merge=args.merge_require_both)
        else:
            rebuild_filtered_subset_full(db_path=args.db_path,
                                         top_n_inclusion=args.top_n_incl,
                                         top_n_exclusion=args.top_n_excl,
                                         require_both_for_merge=args.merge_require_both)

    if not args.skip_reuse:
        rebuild_constraint_clause_usage(db_path=args.db_path)                # TYPE 1
        rebuild_constraint_clause_usage_stem(db_path=args.db_path)           # TYPE 2
        rebuild_directional_usage(db_path=args.db_path)           # TYPE 3
        rebuild_constraint_clause_usage_stem_direction(db_path=args.db_path) # TYPE 2½

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted", file=sys.stderr)
        sys.exit(130)
