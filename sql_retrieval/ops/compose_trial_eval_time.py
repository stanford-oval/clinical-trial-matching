#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compose_trial_eval.py — Compose the primitives with structured logging:
  (1) union(disease_hits, literal_hits, positive_literal_hits) → candidates
  (2) eliminate(candidates) → survivors
  (3) rank: survivors first (best inclusion satisfaction first), then all eliminated
      Ranking uses trial_eval_primitives.inclusion_gap (patient_inclusion_constraints ∪ demographics):
        ASC by frac_unsat_any, ties: MORE total_clauses, then trial_id.
  (4) export retrieved mappings per patient (ON by default) under retrieved_mappings/

MERGING OF SUB-COHORTS
----------------------
Final "clean" and merged artifacts collapse subcohorts like NCT00465907a/b/c/d
into a single canonical NCT00465907, while retaining:
  • union of disease_vars / literal_vars that caused hits
  • list of contributing sub-NCTs and trial_ids
Representative metrics (frac_unsat_any, etc.) are taken from the “best” subcohort:
lowest frac_unsat_any, tie → MORE total_clauses, tie → smaller trial_id.

Outputs
-------
• stdout: JSON (unless --quiet) OR CSV-ish lines with --quiet
• out/logs/compose.jsonl: structured per-event logs (one JSON object per line)
• out/survivors__*.csv, out/compose_results__*.json
• out/ranked__*.csv (ranked list with metrics)
• retrieved_mappings/
    json/{patient}.json           (detailed, per subcohort)
    csv/{patient}.csv             (detailed, per subcohort)
    clean/{patient}.txt           (MERGED, canonical NCTs, ordered)
    merged_json/{patient}.json    (MERGED, rich rows)
    merged_csv/{patient}.csv      (MERGED, rich rows)
    labeled_clean/{patient}.txt   (MERGED canonical NCTs + label; always ON)
    labeled_csv/{patient}.txt     (per-subcohort trial rows + label; always ON)

Requires: trial_eval_primitives.py to be importable.

— Minimal-change extension —
- include positive_literal_hits() in retrieval union.
- WHY helper for positive_* sources merged into literal WHY so downstream stays unchanged.
- NOTE: No geo/country-of-residence prefilter is applied; all retrieval candidates proceed to elimination.
- NEW: --allow-hop-without-root disables the is_root gate for hop>0 in disease/literal/positive hits.

— Profiling additions —
- SQL latency for every execute/executemany (dur_ms, rowcount if feasible, SQL preview)
- Phase timings for: hits.*, why.*, eliminate, rank.inclusion_gap, export.mappings
- Per-patient wall clock with --show-time (prints and logs as {"type":"patient_wall"...})
"""

from __future__ import annotations
import argparse, csv, json, logging, sqlite3, sys, re, time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager

from sql_retrieval.ops import constraint_primitives as tep

LOGGER = logging.getLogger("compose_trial_eval")

# ---------------------------
# Lightweight SQL profiler (Connection/Cursor wrappers)
# ---------------------------

class ProfilingCursor(sqlite3.Cursor):
    def execute(self, sql, parameters=()):
        _t0 = time.perf_counter()
        rows = None
        try:
            res = super().execute(sql, parameters)
            # Best-effort rowcount for SELECTs: fetch and re-execute so callers still iterate normally.
            try:
                if sql.strip().lower().startswith("select"):
                    rows = super().fetchall()
                    super().execute(sql, parameters)
            except Exception:
                rows = None
            return res
        finally:
            _t1 = time.perf_counter()
            conn = getattr(self, "connection", None)
            log_event = getattr(conn, "_log_event", None)
            if log_event:
                preview = " ".join(sql.strip().split())[:300]
                rowcount = None
                try:
                    if rows is not None:
                        rowcount = len(rows)
                    else:
                        rowcount = self.rowcount if self.rowcount != -1 else None
                except Exception:
                    rowcount = None
                log_event({
                    "type": "sql_timing",
                    "dur_ms": round((_t1 - _t0) * 1000.0, 3),
                    "rowcount": rowcount,
                    "sql": preview,
                })

    def executemany(self, sql, seq_of_parameters):
        _t0 = time.perf_counter()
        try:
            return super().executemany(sql, seq_of_parameters)
        finally:
            _t1 = time.perf_counter()
            conn = getattr(self, "connection", None)
            log_event = getattr(conn, "_log_event", None)
            if log_event:
                preview = " ".join(sql.strip().split())[:300]
                log_event({
                    "type": "sql_timing",
                    "dur_ms": round((_t1 - _t0) * 1000.0, 3),
                    "rowcount": None,
                    "sql": preview + "  --executemany",
                })

class ProfilingConnection(sqlite3.Connection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._log_event = None

    def cursor(self, *a, **kw):
        kw["factory"] = ProfilingCursor
        return super().cursor(*a, **kw)

    def set_profiler(self, log_event_callable):
        """Provide the JSONL writer so cursors can emit timings."""
        self._log_event = log_event_callable

@contextmanager
def phase_timer(name: str, log_event):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        if log_event:
            log_event({"type": "phase_timing", "name": name, "dur_ms": round((t1 - t0) * 1000.0, 3)})

# ---------------------------
# Logging / FS
# ---------------------------

def ensure_dirs(out_dir: Path | None) -> None:
    if not out_dir: return
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

def configure_logging(out_dir: Path | None, verbose: bool):
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir:
        ensure_dirs(out_dir)
        handlers.append(logging.FileHandler(out_dir / "logs" / "run.log", mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(message)s",
                        handlers=handlers)
    # JSONL writer (append)
    jsonl_path = (out_dir / "logs" / "compose.jsonl") if out_dir else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")  # truncate

    def log_event(event: Dict) -> None:
        if not jsonl_path: return
        with open(jsonl_path, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(event, ensure_ascii=False) + "\n")

    return jsonl_path, log_event

# ---------------------------
# Patient list
# ---------------------------

SQL_ALL_PATIENTS = """
SELECT patient_id FROM (
  SELECT DISTINCT patient_id FROM patient_inclusion_constraints
  UNION
  SELECT DISTINCT patient_id FROM patient_inclusion_constraints_important
  UNION
  SELECT DISTINCT patient_id FROM patient_exclusion_constraints
  UNION
  SELECT DISTINCT patient_id FROM patient_demographic_constraints
)
ORDER BY patient_id
"""

def _get_all_patients(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute(SQL_ALL_PATIENTS)
    return [r[0] for r in cur.fetchall()]

# ---------------------------
# Small helpers shared with WHY sections
# ---------------------------

SENTINEL_NEG_INF = -1e15
SENTINEL_POS_INF =  1e15

def _inside_pred(fact_alias: str, lb: str, ub: str) -> str:
    # kept for back-compat; not used by the WHY explainers now that timeframe is ignored
    return (
        f"COALESCE({fact_alias}.tf_lb_hours, {SENTINEL_NEG_INF}) >= COALESCE({lb}, {SENTINEL_NEG_INF}) "
        f"AND COALESCE({fact_alias}.tf_ub_hours,  {SENTINEL_POS_INF}) <= COALESCE({ub},  {SENTINEL_POS_INF})"
    )

def _scope_pred(alias: str, scope: str) -> str:
    if scope == "any": return ""
    return (
        f"AND COALESCE({alias}.tf_lb_hours, {SENTINEL_NEG_INF}) <= 0 "
        f"AND COALESCE({alias}.tf_ub_hours,  {SENTINEL_POS_INF}) >= 0"
    )

# ---------------------------
# WHY helpers (mirror primitives: use patient_inclusion_constraints_important; ignore timeframe)
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

def _explain_disease_hits(conn: sqlite3.Connection, patient: str) -> Dict[int, Dict]:
    """
    WHY mirror for disease_hits:
      • uses patient_inclusion_constraints_important (BOOL)
      • timeframe ignored
      • respects hop (prefer hop==0; allow hop>0 only if present in _noisa)
    """
    cur = conn.cursor()

    have_fii = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_inclusion_constraints_important'"
    ).fetchone())
    if not have_fii:
        return {}

    have_fi_noisa = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_inclusion_constraints_noisa'"
    ).fetchone())

    if not cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_atoms'").fetchone():
        return {}

    # Discover columns in disease tables
    dli_cols = {r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_atoms)").fetchall()}
    dli_trial_col = "trial_id" if "trial_id" in dli_cols else ("nct_id" if "nct_id" in dli_cols else None)
    dli_var_col   = next((c for c in ["stem_var","var_name","base_var","var_name_notime"] if c in dli_cols), None)
    dli_hop       = "hop" if "hop" in dli_cols else None
    if not dli_trial_col or not dli_var_col:
        return {}

    # Optional DAA
    has_daa = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_alternatives'"
    ).fetchone())
    daa_sql = ""
    if has_daa:
        daa_cols = {r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_alternatives)").fetchall()}
        daa_trial_col = "trial_id" if "trial_id" in daa_cols else ("nct_id" if "nct_id" in daa_cols else None)
        cand = [c for c in ["stem_var","alt_var_name","var_name_notime","alt_var_name_notime"] if c in daa_cols]
        if daa_trial_col and cand:
            daa_coalesce = "COALESCE(" + ", ".join(cand) + ")"
            daa_hop = "COALESCE(daa.hop,0)" if "hop" in daa_cols else "0"
            daa_sql = f"""
            UNION ALL
            SELECT daa.{daa_trial_col} AS trial_nct_id, {daa_coalesce} AS base_var, {daa_hop} AS hop
            FROM disease_constraint_alternatives daa
            """

    dli_hop_expr = f"COALESCE(dli.{dli_hop},0)" if dli_hop else "0"

    # If hop>0 we allow via patient_inclusion_constraints_noisa (legacy behavior)
    fini_join = (
        f"""
        LEFT JOIN patient_inclusion_constraints_noisa fni
          ON fni.patient_id=:patient AND fni.kind='bool' AND {TRUTHY_FINI}
         AND fni.base_var = dv.base_var
        """ if have_fi_noisa else ""
    )
    fini_guard = " OR (COALESCE(dv.hop,0)<>0 AND fni.base_var IS NOT NULL)" if have_fi_noisa else ""

    sql = f"""
      WITH disease_vars AS (
        SELECT dli.{dli_trial_col} AS trial_nct_id, dli.{dli_var_col} AS base_var, {dli_hop_expr} AS hop
        FROM disease_constraint_atoms dli
        {daa_sql}
      ), itc AS (
        SELECT id AS merged_trial_id, nct_id FROM trials
      )
      SELECT DISTINCT itc.merged_trial_id, itc.nct_id, dv.base_var
      FROM itc
      JOIN disease_vars dv ON dv.trial_nct_id = itc.nct_id
      LEFT JOIN patient_inclusion_constraints_important fii
        ON fii.patient_id=:patient AND fii.kind='bool' AND {TRUTHY_FII}
       AND fii.base_var=dv.base_var
      {fini_join}
      WHERE (COALESCE(dv.hop,0)=0 AND fii.base_var IS NOT NULL){fini_guard}
    """
    out: Dict[int, Dict] = {}
    for tid, nct, base_var in cur.execute(sql, {"patient": patient}).fetchall():
        d = out.setdefault(int(tid), {"nct_id": nct, "vars": []})
        d["vars"].append(base_var)
    for v in out.values():
        v["vars"] = sorted(set(v["vars"]))
    return out

def _explain_literal_hits(conn: sqlite3.Connection, patient: str, scope: str) -> Dict[int, Dict]:
    """
    WHY mirror for literal_hits:
      • uses patient_inclusion_constraints_important (BOOL)
      • timeframe ignored
      • supports lifted sources
    """
    cur = conn.cursor()
    if not cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_inclusion_constraints_important'").fetchone():
        return {}

    has_laa = bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='constraint_literal_alternatives'").fetchone())
    has_llm = bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='constraint_lifted_atoms'").fetchone())
    lifted_source = "constraint_literal_alternatives" if has_laa else ("constraint_lifted_atoms" if has_llm else "")

    if lifted_source:
        lifted_sql = (
            """
            SELECT x.trial_id, x.clause_id, x.literal_index,
                   COALESCE(x.base_var_stem, x.lifted_var_stem) AS base_var
            FROM constraint_literal_alternatives x
            """ if lifted_source == "constraint_literal_alternatives" else
            """
            SELECT x.trial_id, x.clause_id, x.literal_index,
                   COALESCE(x.base_var_stem, x.lifted_var_stem) AS base_var
            FROM constraint_lifted_atoms x
            """
        )
        sql = f"""
          WITH la AS (
            SELECT mt.id AS merged_trial_id, mt.nct_id,
                   sub.clause_id, sub.literal_index, sub.base_var
            FROM trials mt
            JOIN ({lifted_sql}) AS sub
              ON sub.trial_id IN (mt.inclusion_trial_side_id, mt.assumed_trial_side_id)
          )
          SELECT DISTINCT la.merged_trial_id, la.nct_id, la.base_var
          FROM la
          JOIN patient_inclusion_constraints_important fii
            ON fii.patient_id=:patient AND fii.kind='bool' AND {TRUTHY_FII}
           AND fii.base_var=la.base_var
        """
    else:
        sql = """
          WITH itc AS (
            SELECT mt.id AS merged_trial_id, mt.nct_id, tc.clause_id, cl.literal_index, cl.is_neg, cl.base_var
            FROM trials mt
            JOIN trial_constraint_clauses tc ON tc.trial_id = mt.inclusion_trial_side_id
            JOIN constraint_clause_atoms cl ON cl.clause_id = tc.clause_id
            UNION ALL
            SELECT mt.id AS merged_trial_id, mt.nct_id, tc2.clause_id, cl2.literal_index, cl2.is_neg, cl2.base_var
            FROM trials mt
            JOIN trial_constraint_sides ta ON ta.id = mt.assumed_trial_side_id AND ta.kind='inclusion'
            JOIN trial_constraint_clauses tc2 ON tc2.trial_id = ta.id
            JOIN constraint_clause_atoms cl2 ON cl2.clause_id = tc2.clause_id
          )
          SELECT DISTINCT itc.merged_trial_id, itc.nct_id, itc.base_var
          FROM itc
          JOIN patient_inclusion_constraints_important fii
            ON fii.patient_id=:patient AND fii.kind='bool'
           AND fii.base_var=itc.base_var
           AND (
                (itc.is_neg=0 AND CAST(fii.value AS NUMERIC)=1.0)
             OR (itc.is_neg=1 AND CAST(fii.value AS NUMERIC)=0.0)
           )
        """

    out: Dict[int, Dict] = {}
    for tid, nct, base_var in cur.execute(sql, {"patient": patient}).fetchall():
        d = out.setdefault(int(tid), {"nct_id": nct, "vars": []})
        d["vars"].append(base_var)
    for v in out.values():
        v["vars"] = sorted(set(v["vars"]))
    return out

def _explain_positive_literal_hits(conn: sqlite3.Connection, patient: str) -> Dict[int, Dict]:
    """
    WHY mirror for positive_literal_hits:
      • uses patient_inclusion_constraints_important (BOOL)
      • timeframe ignored
    """
    cur = conn.cursor()
    if not cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_inclusion_constraints_important'").fetchone():
        return {}
    has_pl  = bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='positive_constraint_literals'").fetchone())
    has_pla = bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='positive_constraint_alternatives'").fetchone())
    if not has_pl and not has_pla: return {}

    def _cols(table: str):
        have = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        trial_col = "nct_id" if "nct_id" in have else ("trial_id" if "trial_id" in have else None)
        base_var_col = next((c for c in ["base_var","base_var_stem","var_name","var_name_notime"] if c in have), None)
        return trial_col, base_var_col

    parts = []
    if has_pl:
        tcol, vcol = _cols("positive_constraint_literals")
        if tcol and vcol:
            parts.append(f"SELECT {tcol} AS trial_key, {vcol} AS base_var FROM positive_constraint_literals")
    if has_pla:
        tcol, vcol = _cols("positive_constraint_alternatives")
        if tcol and vcol:
            parts.append(f"SELECT {tcol} AS trial_key, {vcol} AS base_var FROM positive_constraint_alternatives")
    if not parts: return {}
    union_sql = "\nUNION ALL\n".join(parts)

    sql = f"""
      WITH src AS ({union_sql})
      SELECT DISTINCT mt.id AS merged_trial_id, mt.nct_id, s.base_var
      FROM trials mt
      JOIN src s ON s.trial_key = mt.nct_id
      JOIN patient_inclusion_constraints_important fii
        ON fii.patient_id=:patient AND fii.kind='bool' AND {TRUTHY_FII}
       AND fii.base_var = s.base_var
    """
    out: Dict[int, Dict] = {}
    for tid, nct, base_var in cur.execute(sql, {"patient": patient}).fetchall():
        d = out.setdefault(int(tid), {"nct_id": nct, "vars": []})
        d["vars"].append(base_var)
    for v in out.values():
        v["vars"] = sorted(set(v["vars"]))
    return out

# ---------------------------
# Canonical NCT merging helpers
# ---------------------------

_CANON_NCT_RE = re.compile(r'^(NCT\d{8})', re.IGNORECASE)

def _canon_nct(nct: str | None) -> str | None:
    if not nct: return None
    m = _CANON_NCT_RE.match(nct.strip())
    return m.group(1).upper() if m else nct.strip().upper()

# ---------------------------
# Export: retrieved mappings (DETAILED + MERGED + ALWAYS-ON LABELED)
# ---------------------------

def _write_retrieved_mappings(conn: sqlite3.Connection,
                              base_out: Path | None,
                              patient_id: str,
                              ranked_rows: List[Dict[str, Any]],
                              disease_why: Dict[int, Dict],
                              literal_why: Dict[int, Dict]) -> None:
    """
    Writes:
      - retrieved_mappings/json/{patient}.json  (detailed, per-subcohort)
      - retrieved_mappings/csv/{patient}.csv    (detailed, per-subcohort)
      - retrieved_mappings/clean/{patient}.txt  (MERGED, canonical NCTs; SURVIVOR-FIRST)
      - retrieved_mappings/merged_json/{patient}.json  (MERGED, rich; SURVIVOR-FIRST)
      - retrieved_mappings/merged_csv/{patient}.csv    (MERGED, rich; SURVIVOR-FIRST)
      - retrieved_mappings/labeled_clean/{patient}.txt (MERGED canonical NCT + label; always ON)
      - retrieved_mappings/labeled_csv/{patient}.txt   (per-subcohort trial rows + label; always ON)
    """
    base = (base_out / "retrieved_mappings") if base_out else Path("retrieved_mappings")
    json_dir          = base / "json"
    csv_dir           = base / "csv"
    clean_dir         = base / "clean"
    mjson_dir         = base / "merged_json"
    mcsv_dir          = base / "merged_csv"
    labeled_clean_dir = base / "labeled_clean"
    labeled_csv_dir   = base / "labeled_csv"
    for d in (json_dir, csv_dir, clean_dir, mjson_dir, mcsv_dir, labeled_clean_dir, labeled_csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Fallback lookup: trial_id -> nct_id
    def _lookup_nct(trial_id: int) -> str | None:
        cur = conn.cursor()
        row = cur.execute("SELECT nct_id FROM trials WHERE id=? LIMIT 1", (trial_id,)).fetchone()
        return row[0] if row and row[0] is not None else None

    # -------- Detailed rows (per subcohort) --------
    rows_out: List[Dict[str, Any]] = []
    for idx, r in enumerate(ranked_rows, 1):
        tid = int(r["trial_id"])
        d_info = disease_why.get(tid, {})
        l_info = literal_why.get(tid, {})
        nct_id = d_info.get("nct_id") or l_info.get("nct_id") or _lookup_nct(tid)

        rows_out.append({
            "rank": idx,
            "status": "survivor" if r.get("is_survivor") else "eliminated",
            "trial_id": tid,
            "nct_id": nct_id,
            "disease_vars": sorted(set(d_info.get("vars", []))),
            "literal_vars": sorted(set(l_info.get("vars", []))),
            "total_clauses": r.get("total_clauses"),
            "unsat_any": r.get("unsat_any"),
            "unsat_explicit": r.get("unsat_explicit"),
            "frac_unsat_any": r.get("frac_unsat_any"),
            "pct_unsat_any": r.get("pct_unsat_any"),
            "pct_unsat_explicit": r.get("pct_unsat_explicit"),
        })

    # 1) JSON (detailed)
    (json_dir / f"{patient_id}.json").write_text(
        json.dumps({"patient_id": patient_id, "trials": rows_out}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 2) CSV (detailed)
    with (csv_dir / f"{patient_id}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id","rank","status","trial_id","nct_id",
                    "disease_vars","literal_vars",
                    "total_clauses","unsat_any","unsat_explicit",
                    "frac_unsat_any","pct_unsat_any","pct_unsat_explicit"])
        for rr in rows_out:
            w.writerow([
                patient_id, rr["rank"], rr["status"], rr["trial_id"], rr.get("nct_id",""),
                ";".join(rr["disease_vars"]), ";".join(rr["literal_vars"]),
                rr["total_clauses"], rr["unsat_any"], rr["unsat_explicit"],
                rr["frac_unsat_any"], rr["pct_unsat_any"], rr["pct_unsat_explicit"]
            ])

    # -------- MERGED view by canonical NCT --------
    merged: Dict[str, Dict[str, Any]] = {}
    for rr in rows_out:
        nct = rr.get("nct_id")
        core = _canon_nct(nct) if nct else None
        if not core:
            core = f"TRIAL_{rr['trial_id']}"
        bucket = merged.setdefault(core, {
            "canonical_nct_id": core if core.startswith("NCT") else "",
            "sub_nct_ids": [],
            "trial_ids": [],
            "statuses": set(),
            "best_row": None,  # best by sorter
            "disease_vars": set(),
            "literal_vars": set(),
        })
        if rr.get("nct_id"): bucket["sub_nct_ids"].append(rr["nct_id"])
        bucket["trial_ids"].append(rr["trial_id"])
        bucket["statuses"].add(rr["status"])
        bucket["disease_vars"].update(rr["disease_vars"])
        bucket["literal_vars"].update(rr["literal_vars"])

        def row_key(x):
            return (x["frac_unsat_any"], -(x["total_clauses"] or 0), x["trial_id"])
        if (bucket["best_row"] is None) or (row_key(rr) < row_key(bucket["best_row"])):
            bucket["best_row"] = rr

    merged_rows: List[Dict[str, Any]] = []
    for core, b in merged.items():
        br = b["best_row"]
        if b["statuses"] == {"survivor"}:
            merged_status = "survivor"
        elif b["statuses"] == {"eliminated"}:
            merged_status = "eliminated"
        else:
            merged_status = "mixed"

        merged_rows.append({
            "canonical_nct_id": b["canonical_nct_id"] or core,
            "sub_nct_ids": sorted(set(b["sub_nct_ids"])),
            "trial_ids": sorted(set(b["trial_ids"])),
            "status": merged_status,
            "rep_trial_id": br["trial_id"],
            "rep_status": br["status"],
            "total_clauses": br["total_clauses"],
            "unsat_any": br["unsat_any"],
            "unsat_explicit": br["unsat_explicit"],
            "frac_unsat_any": br["frac_unsat_any"],
            "pct_unsat_any": br["pct_unsat_any"],
            "pct_unsat_explicit": br["pct_unsat_explicit"],
            "disease_vars": sorted(b["disease_vars"]),
            "literal_vars": sorted(b["literal_vars"]),
        })

    def _status_bucket(s: str) -> int:
        return {"survivor": 0, "mixed": 1, "eliminated": 2}.get(s, 1)

    def _safe_frac(v):
        return v if v is not None else float("inf")

    merged_rows.sort(
        key=lambda r: (
            _status_bucket(r["status"]),
            _safe_frac(r["frac_unsat_any"]),
            -(r["total_clauses"] or 0),
            int(str(r["rep_trial_id"]))
        )
    )

    # CLEAN (merged NCTs)
    clean_dir = (base_out / "retrieved_mappings" / "clean") if base_out else Path("retrieved_mappings/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    with (clean_dir / f"{patient_id}.txt").open("w", encoding="utf-8") as f:
        for mr in merged_rows:
            cid = str(mr["canonical_nct_id"])
            if cid.upper().startswith("NCT"):
                f.write(f"{cid}\n")

    # LABELED_CLEAN (always ON)
    labeled_clean_dir = (base_out / "retrieved_mappings" / "labeled_clean") if base_out else Path("retrieved_mappings/labeled_clean")
    labeled_clean_dir.mkdir(parents=True, exist_ok=True)
    with (labeled_clean_dir / f"{patient_id}.txt").open("w", encoding="utf-8") as f:
        for mr in merged_rows:
            cid = str(mr["canonical_nct_id"])
            if cid.upper().startswith("NCT"):
                f.write(f"{cid}\t{mr['status']}\n")

    # MERGED JSON/CSV (survivor-first)
    mjson_dir = (base_out / "retrieved_mappings" / "merged_json") if base_out else Path("retrieved_mappings/merged_json")
    mcsv_dir  = (base_out / "retrieved_mappings" / "merged_csv")  if base_out else Path("retrieved_mappings/merged_csv")
    mjson_dir.mkdir(parents=True, exist_ok=True)
    mcsv_dir.mkdir(parents=True, exist_ok=True)

    (mjson_dir / f"{patient_id}.json").write_text(
        json.dumps({"patient_id": patient_id, "trials": merged_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    with (mcsv_dir / f"{patient_id}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "patient_id","canonical_nct_id","status",
            "rep_trial_id","rep_status",
            "total_clauses","unsat_any","unsat_explicit",
            "frac_unsat_any","pct_unsat_any","pct_unsat_explicit",
            "sub_nct_ids","trial_ids","disease_vars","literal_vars"
        ])
        for mr in merged_rows:
            w.writerow([
                patient_id, mr["canonical_nct_id"], mr["status"],
                mr["rep_trial_id"], mr["rep_status"],
                mr["total_clauses"], mr["unsat_any"], mr["unsat_explicit"],
                mr["frac_unsat_any"], mr["pct_unsat_any"], mr["pct_unsat_explicit"],
                ";".join(mr["sub_nct_ids"]), ";".join(map(str, mr["trial_ids"])),
                ";".join(mr["disease_vars"]), ";".join(mr["literal_vars"]),
            ])

    # LABELED_CSV (per-subcohort + label; always ON)
    labeled_csv_dir = (base_out / "retrieved_mappings" / "labeled_csv") if base_out else Path("retrieved_mappings/labeled_csv")
    labeled_csv_dir.mkdir(parents=True, exist_ok=True)
    with (labeled_csv_dir / f"{patient_id}.txt").open("w", encoding="utf-8") as f:
        for rr in rows_out:
            f.write(f"{rr['trial_id']}\t{rr.get('nct_id','')}\t{rr['status']}\n")

# ---------------------------
# Emit helpers
# ---------------------------

def _emit_single(payload: Dict, survivors: List[int], ranked_rows: List[Dict[str, Any]],
                 out_dir: Path | None, quiet: bool) -> None:
    if quiet:
        for idx, r in enumerate(ranked_rows, 1):
            status = "survivor" if r.get("is_survivor") else "eliminated"
            print(f"{payload['patient_id']},{r['trial_id']},{status},{idx},{r['frac_unsat_any']},{r['pct_unsat_any']},{r['pct_unsat_explicit']}")
        return

    payload_out = dict(payload)
    payload_out["ranked"] = ranked_rows
    print(json.dumps(payload_out, ensure_ascii=False, indent=2))

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"compose_results__{payload['patient_id']}.json", "w", encoding="utf-8") as f:
            json.dump(payload_out, f, ensure_ascii=False, indent=2)
        with open(out_dir / f"survivors__{payload['patient_id']}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["patient_id","trial_id"]); w.writerows([[payload['patient_id'], t] for t in survivors])
        with open(out_dir / f"ranked__{payload['patient_id']}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id","rank","trial_id","status","total_clauses","unsat_any","unsat_explicit","frac_unsat_any","pct_unsat_any","pct_unsat_explicit"])
            for idx, r in enumerate(ranked_rows, 1):
                w.writerow([
                    payload['patient_id'], idx, r["trial_id"],
                    "survivor" if r.get("is_survivor") else "eliminated",
                    r["total_clauses"], r["unsat_any"], r["unsat_explicit"],
                    r["frac_unsat_any"], r["pct_unsat_any"], r["pct_unsat_explicit"]
                ])

# ---------------------------
# Core run (with ranking + mappings export)
# ---------------------------

def run_for_patient(conn: sqlite3.Connection, log_event, patient: str, scope: str,
                    base_out: Path | None, require_root_for_hops: bool = True
                   ) -> Tuple[Dict, List[int], List[Dict[str, Any]]]:
    # 1) Hits (now includes positive_* sources)
    with phase_timer("hits.disease", log_event):
        disease_ids   = tep.satisfy_disease_constraints(conn, patient, require_root_for_hops=require_root_for_hops)
    with phase_timer("hits.literal", log_event):
        literal_ids   = tep.literal_hits(conn, patient, scope=scope, require_root_for_hops=require_root_for_hops)
    with phase_timer("hits.positive_literal", log_event):
        positive_ids  = tep.satisfy_positive_literal_constraints(conn, patient, require_root_for_hops=require_root_for_hops)
    union_ids     = sorted(set(disease_ids) | set(literal_ids) | set(positive_ids))

    # No Geography/Country prefilter — pass all candidates forward unchanged
    geo_screened_ids = union_ids
    geo_eliminated_set = set()

    LOGGER.info("patient=%s  disease=%d  literal=%d  positive=%d  union=%d  geo_after=%d  geo_elim=%d",
                patient, len(disease_ids), len(literal_ids), len(positive_ids), len(union_ids),
                len(geo_screened_ids), 0)
    log_event({"type":"summary_stage_counts","patient_id":patient,
               "counts":{"disease":len(disease_ids),
                         "literal":len(literal_ids),
                         "positive":len(positive_ids),
                         "union":len(union_ids),
                         "geo_after":len(geo_screened_ids),
                         "geo_elim":0}})

    # WHY: disease & literal & positive (mirrors primitives: patient_inclusion_constraints_important, no timeframe)
    with phase_timer("why.disease", log_event):
        disease_why  = _explain_disease_hits(conn, patient)
    with phase_timer("why.literal", log_event):
        literal_why  = _explain_literal_hits(conn, patient, scope)
    with phase_timer("why.positive_literal", log_event):
        positive_why = _explain_positive_literal_hits(conn, patient)

    # Merge positive vars into literal WHY so exports remain "literal_vars"
    for tid, info in positive_why.items():
        bucket = literal_why.setdefault(tid, {"nct_id": info.get("nct_id"), "vars": []})
        if not bucket.get("nct_id"):
            bucket["nct_id"] = info.get("nct_id")
        bucket["vars"].extend(info.get("vars", []))

    for tid, info in sorted(disease_why.items()):
        log_event({"type":"why_hit","source":"disease","patient_id":patient,
                   "trial_id":tid,"nct_id":info.get("nct_id"),"base_vars":info.get("vars",[])})
    for tid, info in sorted(literal_why.items()):
        log_event({"type":"why_hit","source":"literal","patient_id":patient,
                   "trial_id":tid,"nct_id":info.get("nct_id"),"base_vars":info.get("vars",[])})

    # 2) Eliminate explicit contradictions (run on all candidates)
    with phase_timer("eliminate", log_event):
        survivors = tep.check_constraint_contradictions(conn, patient, geo_screened_ids, scope=scope)
    LOGGER.info("patient=%s  survivors=%d", patient, len(survivors))

    eliminated_set = set(union_ids) - set(survivors)  # no separate geo elimination now
    for tid in sorted(eliminated_set - geo_eliminated_set):
        log_event({"type":"eliminated","patient_id":patient,"trial_id":tid})

    # 3) Rank by inclusion criteria satisfaction (survivors first)
    with phase_timer("rank.inclusion_gap", log_event):
        gap_rows = tep.evaluate_constraint_satisfaction_gap(conn, patient, scope=scope, candidate_trial_ids=union_ids)
    survivors_ordered = [r for r in gap_rows if r["trial_id"] in survivors]
    eliminated_ordered = [r for r in gap_rows if r["trial_id"] in (set(union_ids) - set(survivors))]

    ranked_rows: List[Dict[str, Any]] = []
    for r in survivors_ordered:
        ranked_rows.append(dict(r, is_survivor=True))
    for r in eliminated_ordered:
        ranked_rows.append(dict(r, is_survivor=False))

    # 4) Export retrieved mappings (DETAILED + MERGED + LABELED)
    with phase_timer("export.mappings", log_event):
        _write_retrieved_mappings(conn, base_out, patient, ranked_rows, disease_why, literal_why)

    # logs for ranking
    for idx, r in enumerate(ranked_rows, 1):
        log_event({
            "type":"rank",
            "patient_id": patient,
            "rank": idx,
            "trial_id": r["trial_id"],
            "status": "survivor" if r["is_survivor"] else "eliminated",
            "frac_unsat_any": r["frac_unsat_any"],
            "total_clauses": r["total_clauses"],
            "unsat_any": r["unsat_any"],
            "unsat_explicit": r["unsat_explicit"],
        })

    payload = {
        "patient_id": patient,
        "counts": {
            "disease": len(disease_ids),
            "literal": len(literal_ids),
            "positive": len(positive_ids),
            "union": len(union_ids),
            "geo_after": len(geo_screened_ids),
            "survivors": len(survivors),
        },
        "disease_trial_ids": disease_ids,
        "literal_trial_ids": literal_ids,
        "positive_trial_ids": positive_ids,
        "union_trial_ids": union_ids,
        "geo_screened_trial_ids": geo_screened_ids,
        "survivor_trial_ids": survivors,
    }
    return payload, survivors, ranked_rows

# ---------------------------
# Multi-patient emitter
# ---------------------------

def _emit_multi(all_payloads: Dict[str, Dict], out_dir: Path | None, quiet: bool) -> None:
    if quiet:
        for pid, pld in all_payloads.items():
            for idx, r in enumerate(pld.get("ranked", []), 1):
                status = "survivor" if r.get("is_survivor") else "eliminated"
                print(f"{pid},{r['trial_id']},{status},{idx},{r['frac_unsat_any']},{r['pct_unsat_any']},{r['pct_unsat_explicit']}")
        return
    print(json.dumps({"patients": all_payloads}, ensure_ascii=False, indent=2))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "compose_results__all.json", "w", encoding="utf-8") as f:
            json.dump({"patients": all_payloads}, f, ensure_ascii=False, indent=2)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Union disease+literal(+positive) hits → eliminate → rank (survivors first). Also exports per-patient retrieved mappings by default. Subcohorts are merged by canonical NCT."
    )
    ap.add_argument("--db", default="../../build/trial.db", type=Path)
    ap.add_argument("--patient", default=None, help="Run for a single patient_id; omit to run for ALL patients")
    ap.add_argument("--scope", choices=["now","any"], default="any", help="Range-aware scope for elimination and ranking (retrieval ignores timeframe)")
    ap.add_argument("--out", type=Path, default=None, help="Write results & logs to this directory (retrieved_mappings/ is created here). If omitted, ./retrieved_mappings is used for mapping exports.")
    ap.add_argument("--quiet", action="store_true", help="CSV lines: patient_id,trial_id,status,rank,frac_unsat_any,pct_unsat_any,pct_unsat_explicit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show-time", action="store_true",
                    help="Print wall-clock time per patient (also logged as patient_wall in compose.jsonl)")
    # Toggle: allow hop>0 matches without requiring fii.is_root=1
    ap.add_argument("--allow-hop-without-root", action="store_true",
                    help="Let accepted-alternative (hop>0) matches count even if patient_inclusion_constraints_important.is_root is 0/NULL")

    args = ap.parse_args()
    _, log_event = configure_logging(args.out, args.verbose)

    try:
        conn = sqlite3.connect(str(args.db), factory=ProfilingConnection)
        conn.set_profiler(log_event)
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    require_root_for_hops = not args.allow_hop_without_root

    if args.patient:
        log_event({"type":"run_start","mode":"single","patient_id":args.patient})
        t0 = time.perf_counter()
        payload, survivors, ranked_rows = run_for_patient(conn, log_event, args.patient, args.scope, args.out,
                                                          require_root_for_hops=require_root_for_hops)
        t1 = time.perf_counter()
        wall_ms = round((t1 - t0) * 1000.0, 3)
        log_event({"type":"patient_wall","patient_id":args.patient,"wall_ms":wall_ms})
        if args.show_time:
            print(f"TIME,patient={args.patient},wall_ms={wall_ms},wall_s={wall_ms/1000.0:.3f}")

        if not args.quiet:
            payload_out = dict(payload)
            payload_out["ranked"] = ranked_rows
            print(json.dumps(payload_out, ensure_ascii=False, indent=2))
        else:
            for idx, r in enumerate(ranked_rows, 1):
                status = "survivor" if r.get("is_survivor") else "eliminated"
                print(f"{payload['patient_id']},{r['trial_id']},{status},{idx},{r['frac_unsat_any']},{r['pct_unsat_any']},{r['pct_unsat_explicit']}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            with open(args.out / f"ranked__{payload['patient_id']}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["patient_id","rank","trial_id","status","total_clauses","unsat_any","unsat_explicit","frac_unsat_any","pct_unsat_any","pct_unsat_explicit"])
                for idx, r in enumerate(ranked_rows, 1):
                    w.writerow([payload['patient_id'], idx, r["trial_id"],
                                "survivor" if r.get("is_survivor") else "eliminated",
                                r["total_clauses"], r["unsat_any"], r["unsat_explicit"],
                                r["frac_unsat_any"], r["pct_unsat_any"], r["pct_unsat_explicit"]])
        log_event({"type":"run_end","patient_id":args.patient,"survivors":len(survivors)})
    else:
        patients = _get_all_patients(conn)
        if not patients:
            LOGGER.info("No patients found.")
            print(json.dumps({"patients": {}}, ensure_ascii=False))
            return
        log_event({"type":"run_start","mode":"all","num_patients":len(patients)})
        all_payloads: Dict[str, Dict] = {}
        for idx, pid in enumerate(patients, 1):
            LOGGER.info("[%d/%d] patient=%s", idx, len(patients), pid)
            log_event({"type":"patient_start","index":idx,"total":len(patients),"patient_id":pid})
            t0 = time.perf_counter()
            payload, survivors, ranked_rows = run_for_patient(conn, log_event, pid, args.scope, args.out,
                                                              require_root_for_hops=require_root_for_hops)
            t1 = time.perf_counter()
            wall_ms = round((t1 - t0) * 1000.0, 3)
            log_event({"type":"patient_wall","patient_id":pid,"wall_ms":wall_ms})
            if args.show_time:
                print(f"TIME,patient={pid},wall_ms={wall_ms},wall_s={wall_ms/1000.0:.3f}")

            payload_with_rank = dict(payload)
            payload_with_rank["ranked"] = ranked_rows
            all_payloads[pid] = payload_with_rank
            log_event({"type":"patient_end","patient_id":pid})
        _emit_multi(all_payloads, args.out, args.quiet)
        log_event({"type":"run_end","mode":"all","num_patients":len(patients)})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[warn] interrupted", file=sys.stderr)
        sys.exit(130)
