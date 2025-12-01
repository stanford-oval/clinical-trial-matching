#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compose_trial_eval.py — MODE-AWARE + PREVENTION-SUFFIX + ALT-MODE (act/nonact) version.

Adds a third suffix dimension so outputs never collide across:
  - --important-mode {chief,ccr,all}
  - --enable-prevention-hits (prevent vs noprevent)
  - --alt-mode {act,nonact}

Suffix scheme (used everywhere files/dirs are written):
  __{mode}__{prevent_tag}__{alt_tag}
where:
  prevent_tag ∈ {prevent, noprevent}
  alt_tag     ∈ {act, nonact}

Dirs created under --out:
  logs__{mode}__{prevent_tag}__{alt_tag}/
  retrieved_mappings__{mode}__{prevent_tag}__{alt_tag}/
  clean_eval__{mode}__{prevent_tag}__{alt_tag}/
  list_to_match__{mode}__{prevent_tag}__{alt_tag}/

and filenames also include __{mode}__{prevent_tag}__{alt_tag}.

MODIFIED (per request):
- Retrieval no longer uses literal_hits().
- Candidates = disease_hits ∪ positive_literal_hits ∪ (optional) prevention_hits.
- "literal_vars" in outputs now correspond to positive-literal why-vars (keeps schema stable).

NEW (per request):
- --alt-mode nonact switches accepted-alternatives tables:
    main disease:      disease_constraint_alternatives_nonact
    prevent disease:   disease_constraint_alternatives_prevent_nonact
    main pos-lit:      positive_constraint_alternatives_expanded_nonact (preferred)
    prevent pos-lit:   positive_constraint_alternatives_expanded_prevention_nonact
"""

from __future__ import annotations
import argparse, csv, json, logging, sqlite3, sys, re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from sql_retrieval.ops import constraint_primitives as tep

LOGGER = logging.getLogger("compose_trial_eval")

# ---------------------------
# Suffix helpers
# ---------------------------

def _prevent_tag(enable_prevention_hits: bool) -> str:
    return "prevent" if enable_prevention_hits else "noprevent"

def _alt_tag(alt_mode: str) -> str:
    return "nonact" if (alt_mode or "").strip().lower() == "nonact" else "act"

def _run_suffix(mode: str, enable_prevention_hits: bool, alt_mode: str) -> str:
    return f"__{mode}__{_prevent_tag(enable_prevention_hits)}__{_alt_tag(alt_mode)}"

def _logs_dirname(mode: str, enable_prevention_hits: bool, alt_mode: str) -> str:
    return f"logs{_run_suffix(mode, enable_prevention_hits, alt_mode)}"

def _scoped_dirname(prefix: str, mode: str, enable_prevention_hits: bool, alt_mode: str) -> str:
    return f"{prefix}{_run_suffix(mode, enable_prevention_hits, alt_mode)}"

# ---------------------------
# Logging / FS
# ---------------------------

def ensure_dirs(out_dir: Path | None, mode: str, enable_prevention_hits: bool, alt_mode: str) -> None:
    if not out_dir:
        return
    (out_dir / _logs_dirname(mode, enable_prevention_hits, alt_mode)).mkdir(parents=True, exist_ok=True)

def configure_logging(out_dir: Path | None,
                      verbose: bool,
                      truncate_jsonl: bool = True,
                      mode: str = "all",
                      enable_prevention_hits: bool = False,
                      alt_mode: str = "act"):
    """
    Configure logging + JSONL writer.

    truncate_jsonl=True  - main process: clear previous logs
    truncate_jsonl=False - worker processes: append only, don't wipe logs
    """
    # Reset handlers for this process only
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    run_log_path = None
    if out_dir:
        ensure_dirs(out_dir, mode, enable_prevention_hits, alt_mode)
        run_log_path = out_dir / _logs_dirname(mode, enable_prevention_hits, alt_mode) / "run.log"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        if truncate_jsonl:
            run_log_path.write_text("", encoding="utf-8")
        handlers.append(logging.FileHandler(run_log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=handlers,
    )

    jsonl_path = (out_dir / _logs_dirname(mode, enable_prevention_hits, alt_mode) / "compose.jsonl") if out_dir else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if truncate_jsonl:
            jsonl_path.write_text("", encoding="utf-8")

    def log_event(event: Dict) -> None:
        if not jsonl_path:
            return
        with open(jsonl_path, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(event, ensure_ascii=False) + "\n")

    return jsonl_path, log_event

# ---------------------------
# Patient list (mode-aware)
# ---------------------------

def sql_all_patients(important_table: str) -> str:
    return f"""
    SELECT patient_id FROM (
      SELECT DISTINCT patient_id FROM patient_inclusion_constraints
      UNION
      SELECT DISTINCT patient_id FROM {important_table}
      UNION
      SELECT DISTINCT patient_id FROM patient_exclusion_constraints
      UNION
      SELECT DISTINCT patient_id FROM patient_demographic_constraints
    )
    ORDER BY patient_id
    """

def _get_all_patients(conn: sqlite3.Connection, important_table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(sql_all_patients(important_table))
    return [r[0] for r in cur.fetchall()]

# ---------------------------
# WHY helpers (mode-aware; ignore timeframe)
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

def _explain_disease_hits(conn: sqlite3.Connection, patient: str, important_table: str) -> Dict[int, Dict]:
    cur = conn.cursor()

    have_fii = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (important_table,)
    ).fetchone())
    if not have_fii:
        return {}

    have_fi_noisa = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patient_inclusion_constraints_noisa'"
    ).fetchone())

    if not cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_atoms'").fetchone():
        return {}

    dli_cols = {r[1] for r in cur.execute("PRAGMA table_info(disease_constraint_atoms)").fetchall()}
    dli_trial_col = "trial_id" if "trial_id" in dli_cols else ("nct_id" if "nct_id" in dli_cols else None)
    dli_var_col   = next((c for c in ["stem_var","var_name","base_var","var_name_notime"] if c in dli_cols), None)
    dli_hop       = "hop" if "hop" in dli_cols else None
    if not dli_trial_col or not dli_var_col:
        return {}

    # Prefer nonact if present? (WHY is for debugging; we union whatever exists)
    has_daa = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_alternatives'"
    ).fetchone())
    has_daa_nonact = bool(cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='disease_constraint_alternatives_nonact'"
    ).fetchone())

    daa_sql = ""
    def _build_daa_union(table: str) -> str:
        daa_cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        daa_trial_col = "trial_id" if "trial_id" in daa_cols else ("nct_id" if "nct_id" in daa_cols else None)
        cand = [c for c in ["stem_var","alt_var_name","var_name_notime","alt_var_name_notime"] if c in daa_cols]
        if not (daa_trial_col and cand):
            return ""
        daa_coalesce = "COALESCE(" + ", ".join(cand) + ")"
        daa_hop = "COALESCE(daa.hop,0)" if "hop" in daa_cols else "0"
        return f"""
            UNION ALL
            SELECT daa.{daa_trial_col} AS trial_nct_id, {daa_coalesce} AS base_var, {daa_hop} AS hop
            FROM {table} daa
        """

    if has_daa:
        daa_sql += _build_daa_union("disease_constraint_alternatives")
    if has_daa_nonact:
        daa_sql += _build_daa_union("disease_constraint_alternatives_nonact")

    dli_hop_expr = f"COALESCE(dli.{dli_hop},0)" if dli_hop else "0"

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
      LEFT JOIN {important_table} fii
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

def _explain_positive_literal_hits(conn: sqlite3.Connection, patient: str, important_table: str, alt_mode: str) -> Dict[int, Dict]:
    cur = conn.cursor()
    if not cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (important_table,)
    ).fetchone():
        return {}

    alt_tag = _alt_tag(alt_mode)

    # Prefer expanded accepted-alts. For WHY we can union multiple sources if present.
    main_exp = "positive_constraint_alternatives_expanded_nonact" if alt_tag == "nonact" else "positive_constraint_alternatives_expanded"
    legacy   = "positive_constraint_alternatives"

    tables = []
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='positive_constraint_literals'").fetchone():
        tables.append("positive_constraint_literals")
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(main_exp,)).fetchone():
        tables.append(main_exp)
    elif cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(legacy,)).fetchone():
        tables.append(legacy)

    if not tables:
        return {}

    def _cols(table: str):
        have = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        trial_col = "nct_id" if "nct_id" in have else ("trial_id" if "trial_id" in have else None)
        base_var_col = next((c for c in ["base_var","base_var_stem","var_name","var_name_notime","lifted_var_stem","lifted_var"] if c in have), None)
        return trial_col, base_var_col

    parts = []
    for t in tables:
        tcol, vcol = _cols(t)
        if tcol and vcol:
            parts.append(f"SELECT {tcol} AS trial_key, {vcol} AS base_var FROM {t}")
    if not parts:
        return {}

    union_sql = "\nUNION ALL\n".join(parts)

    sql = f"""
      WITH src AS ({union_sql})
      SELECT DISTINCT mt.id AS merged_trial_id, mt.nct_id, s.base_var
      FROM trials mt
      JOIN src s ON s.trial_key = mt.nct_id
      JOIN {important_table} fii
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
    if not nct:
        return None
    m = _CANON_NCT_RE.match(nct.strip())
    return m.group(1).upper() if m else nct.strip().upper()

# ---------------------------
# Export: retrieved mappings (MODE+PREVENTION+ALT aware)
# ---------------------------

def _write_retrieved_mappings(conn: sqlite3.Connection,
                              base_out: Path | None,
                              patient_id: str,
                              ranked_rows: List[Dict[str, Any]],
                              disease_why: Dict[int, Dict],
                              literal_why: Dict[int, Dict],
                              mode: str,
                              enable_prevention_hits: bool,
                              alt_mode: str
                             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mode_suffix = _run_suffix(mode, enable_prevention_hits, alt_mode)
    base = (base_out / _scoped_dirname("retrieved_mappings", mode, enable_prevention_hits, alt_mode)) if base_out else Path(_scoped_dirname("retrieved_mappings", mode, enable_prevention_hits, alt_mode))
    json_dir          = base / "json"
    csv_dir           = base / "csv"
    clean_dir         = base / "clean"
    mjson_dir         = base / "merged_json"
    mcsv_dir          = base / "merged_csv"
    labeled_clean_dir = base / "labeled_clean"
    labeled_csv_dir   = base / "labeled_csv"
    for d in (json_dir, csv_dir, clean_dir, mjson_dir, mcsv_dir, labeled_clean_dir, labeled_csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _lookup_nct(trial_id: int) -> str | None:
        cur = conn.cursor()
        row = cur.execute("SELECT nct_id FROM trials WHERE id=? LIMIT 1", (trial_id,)).fetchone()
        return row[0] if row and row[0] is not None else None

    rows_out: List[Dict[str, Any]] = []
    for idx, r in enumerate(ranked_rows, 1):
        tid = int(r["trial_id"])
        d_info = disease_why.get(tid, {})
        l_info = literal_why.get(tid, {})
        nct_id = d_info.get("nct_id") or l_info.get("nct_id") or _lookup_nct(tid)

        label = r.get("label", "explicit_contradiction")
        status = "survivor" if label != "explicit_contradiction" else "eliminated"

        rows_out.append({
            "rank": idx,
            "status": status,
            "label": label,
            "trial_id": tid,
            "nct_id": nct_id,
            "disease_vars": sorted(set(d_info.get("vars", []))),
            "literal_vars": sorted(set(l_info.get("vars", []))),  # NOTE: now stores positive-literal vars
            "total_clauses": r.get("total_clauses"),
            "unsat_any": r.get("unsat_any"),
            "unsat_explicit": r.get("unsat_explicit"),
            "frac_unsat_any": r.get("frac_unsat_any"),
            "pct_unsat_any": r.get("pct_unsat_any"),
            "pct_unsat_explicit": r.get("pct_unsat_explicit"),
        })

    (json_dir / f"{patient_id}{mode_suffix}.json").write_text(
        json.dumps({"patient_id": patient_id, "mode": mode, "prevent_tag": _prevent_tag(enable_prevention_hits), "alt_mode": _alt_tag(alt_mode), "trials": rows_out}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    with (csv_dir / f"{patient_id}{mode_suffix}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id","mode","prevent_tag","alt_mode","rank","trial_id","nct_id",
                    "status","label",
                    "disease_vars","literal_vars",
                    "total_clauses","unsat_any","unsat_explicit",
                    "frac_unsat_any","pct_unsat_any","pct_unsat_explicit"])
        for rr in rows_out:
            w.writerow([
                patient_id, mode, _prevent_tag(enable_prevention_hits), _alt_tag(alt_mode), rr["rank"], rr["trial_id"], rr.get("nct_id",""),
                rr["status"], rr["label"],
                ";".join(rr["disease_vars"]), ";".join(rr["literal_vars"]),
                rr["total_clauses"], rr["unsat_any"], rr["unsat_explicit"],
                rr["frac_unsat_any"], rr["pct_unsat_any"], rr["pct_unsat_explicit"]
            ])

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
            "labels": set(),
            "best_row": None,
            "disease_vars": set(),
            "literal_vars": set(),
        })
        if rr.get("nct_id"):
            bucket["sub_nct_ids"].append(rr["nct_id"])
        bucket["trial_ids"].append(rr["trial_id"])
        bucket["statuses"].add(rr["status"])
        bucket["labels"].add(rr["label"])
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
        merged_label = br.get("label", "explicit_contradiction")

        merged_rows.append({
            "canonical_nct_id": b["canonical_nct_id"] or core,
            "sub_nct_ids": sorted(set(b["sub_nct_ids"])),
            "trial_ids": sorted(set(b["trial_ids"])),
            "status": merged_status,
            "label": merged_label,
            "rep_trial_id": br["trial_id"],
            "rep_status": br["status"],
            "rep_label": merged_label,
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

    with (clean_dir / f"{patient_id}{mode_suffix}.txt").open("w", encoding="utf-8") as f:
        for mr in merged_rows:
            cid = str(mr["canonical_nct_id"])
            if cid.upper().startswith("NCT"):
                f.write(f"{cid}\n")

    with (labeled_clean_dir / f"{patient_id}{mode_suffix}.txt").open("w", encoding="utf-8") as f:
        for mr in merged_rows:
            cid = str(mr["canonical_nct_id"])
            if cid.upper().startswith("NCT"):
                label = mr.get("label", mr["status"])
                f.write(f"{cid}\t{label}\n")

    (mjson_dir / f"{patient_id}{mode_suffix}.json").write_text(
        json.dumps({"patient_id": patient_id, "mode": mode, "prevent_tag": _prevent_tag(enable_prevention_hits), "alt_mode": _alt_tag(alt_mode), "trials": merged_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    with (mcsv_dir / f"{patient_id}{mode_suffix}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "patient_id","mode","prevent_tag","alt_mode","canonical_nct_id","status","label",
            "rep_trial_id","rep_status","rep_label",
            "total_clauses","unsat_any","unsat_explicit",
            "frac_unsat_any","pct_unsat_any","pct_unsat_explicit",
            "sub_nct_ids","trial_ids","disease_vars","literal_vars"
        ])
        for mr in merged_rows:
            w.writerow([
                patient_id, mode, _prevent_tag(enable_prevention_hits), _alt_tag(alt_mode), mr["canonical_nct_id"], mr["status"], mr["label"],
                mr["rep_trial_id"], mr["rep_status"], mr["rep_label"],
                mr["total_clauses"], mr["unsat_any"], mr["unsat_explicit"],
                mr["frac_unsat_any"], mr["pct_unsat_any"], mr["pct_unsat_explicit"],
                ";".join(mr["sub_nct_ids"]), ";".join(map(str, mr["trial_ids"])),
                ";".join(mr["disease_vars"]), ";".join(mr["literal_vars"]),
            ])

    with (labeled_csv_dir / f"{patient_id}{mode_suffix}.txt").open("w", encoding="utf-8") as f:
        for rr in rows_out:
            label = rr.get("label", rr["status"])
            f.write(f"{rr['trial_id']}\t{rr.get('nct_id','')}\t{label}\n")

    return rows_out, merged_rows

# ---------------------------
# MODE+PREVENTION+ALT: clean per-patient labels export
# ---------------------------

def _export_patient_labels_json(base_out: Path | None,
                                patient_id: str,
                                merged_rows: List[Dict[str, Any]],
                                mode: str,
                                enable_prevention_hits: bool,
                                alt_mode: str) -> None:
    mode_suffix = _run_suffix(mode, enable_prevention_hits, alt_mode)
    clean_root = (base_out or Path(".")) / _scoped_dirname("clean_eval", mode, enable_prevention_hits, alt_mode)
    out_dir = clean_root / "patient_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    for idx, mr in enumerate(merged_rows, 1):
        canonical_id = mr.get("canonical_nct_id") or str(mr.get("rep_trial_id"))
        label = mr.get("label", "explicit_contradiction")
        trials.append({
            "trial_id": canonical_id,
            "rank": idx,
            "label": label,
        })

    data = {"patient_id": patient_id, "mode": mode, "prevent_tag": _prevent_tag(enable_prevention_hits), "alt_mode": _alt_tag(alt_mode), "trials": trials}

    (out_dir / f"{patient_id}{mode_suffix}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ---------------------------
# MODE+PREVENTION+ALT: list_to_match export
# ---------------------------

def _export_list_to_match(base_out: Path | None,
                          patient_id: str,
                          detailed_rows: List[Dict[str, Any]],
                          merged_rows: List[Dict[str, Any]],
                          mode: str,
                          enable_prevention_hits: bool,
                          alt_mode: str) -> None:
    mode_suffix = _run_suffix(mode, enable_prevention_hits, alt_mode)
    root = (base_out or Path(".")) / _scoped_dirname("list_to_match", mode, enable_prevention_hits, alt_mode)
    root.mkdir(parents=True, exist_ok=True)

    by_canonical: Dict[str, List[Dict[str, Any]]] = {}
    for rr in detailed_rows:
        nct_id = rr.get("nct_id")
        if nct_id:
            core = _canon_nct(nct_id) or nct_id.upper()
        else:
            core = f"TRIAL_{rr['trial_id']}"
        by_canonical.setdefault(core, []).append({
            "trial_id": rr["trial_id"],
            "nct_id": nct_id,
            "status": rr["status"],
            "label": rr["label"],
            "rank": rr["rank"],
        })

    def _status_bucket_2way(s: str) -> int:
        return 0 if s == "survivor" else 1

    canonical_trials: List[Dict[str, Any]] = []
    for mr in merged_rows:
        cid = mr["canonical_nct_id"]
        subcohorts = by_canonical.get(cid, [])
        subcohorts_sorted = sorted(subcohorts, key=lambda s: (_status_bucket_2way(s["status"]), s["rank"]))
        canonical_trials.append({
            "canonical_nct_id": cid,
            "status": mr["status"],
            "label": mr["label"],
            "sub_nct_ids": mr["sub_nct_ids"],
            "subcohorts": subcohorts_sorted,
        })

    data = {"patient_id": patient_id, "mode": mode, "prevent_tag": _prevent_tag(enable_prevention_hits), "alt_mode": _alt_tag(alt_mode), "canonical_trials": canonical_trials}
    (root / f"{patient_id}{mode_suffix}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------
# Emit helpers (MODE+PREVENTION+ALT)
# ---------------------------

def _emit_single(payload: Dict,
                 survivors: List[int],
                 ranked_rows: List[Dict[str, Any]],
                 out_dir: Path | None,
                 quiet: bool,
                 mode: str,
                 enable_prevention_hits: bool,
                 alt_mode: str) -> None:
    mode_suffix = _run_suffix(mode, enable_prevention_hits, alt_mode)
    ptag = _prevent_tag(enable_prevention_hits)
    atag = _alt_tag(alt_mode)

    if quiet:
        for idx, r in enumerate(ranked_rows, 1):
            label = r.get("label", "explicit_contradiction")
            status = "survivor" if label != "explicit_contradiction" else "eliminated"
            print(f"{payload['patient_id']},{mode},{ptag},{atag},{r['trial_id']},{status},{label},{idx},"
                  f"{r['frac_unsat_any']},{r['pct_unsat_any']},{r['pct_unsat_explicit']}")
        return

    payload_out = dict(payload)
    payload_out["mode"] = mode
    payload_out["prevent_tag"] = ptag
    payload_out["alt_mode"] = atag
    payload_out["ranked"] = ranked_rows
    print(json.dumps(payload_out, ensure_ascii=False, indent=2))

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"compose_results__{payload['patient_id']}{mode_suffix}.json", "w", encoding="utf-8") as f:
            json.dump(payload_out, f, ensure_ascii=False, indent=2)
        with open(out_dir / f"survivors__{payload['patient_id']}{mode_suffix}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id","mode","prevent_tag","alt_mode","trial_id"])
            w.writerows([[payload['patient_id'], mode, ptag, atag, t] for t in survivors])
        with open(out_dir / f"ranked__{payload['patient_id']}{mode_suffix}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id","mode","prevent_tag","alt_mode","rank","trial_id","status","label",
                        "total_clauses","unsat_any","unsat_explicit",
                        "frac_unsat_any","pct_unsat_any","pct_unsat_explicit"])
            for idx, r in enumerate(ranked_rows, 1):
                label = r.get("label", "explicit_contradiction")
                status = "survivor" if label != "explicit_contradiction" else "eliminated"
                w.writerow([
                    payload['patient_id'], mode, ptag, atag, idx, r["trial_id"],
                    status, label,
                    r["total_clauses"], r["unsat_any"], r["unsat_explicit"],
                    r["frac_unsat_any"], r["pct_unsat_any"], r["pct_unsat_explicit"]
                ])

# ---------------------------
# Core run (MODE-AWARE) — MODIFIED: no literal_hits()
# ---------------------------

def run_for_patient(conn: sqlite3.Connection, log_event, patient: str, scope: str,
                    base_out: Path | None,
                    important_table: str,
                    mode: str,
                    require_root_for_hops: bool = True,
                    disease_only: bool = False,
                    enable_prevention_hits: bool = False,
                    alt_mode: str = "act"
                   ) -> Tuple[Dict, List[int], List[Dict[str, Any]]]:

    atag = _alt_tag(alt_mode)

    daa_main = "disease_constraint_alternatives_nonact" if atag == "nonact" else "disease_constraint_alternatives"

    disease_ids   = tep.satisfy_disease_constraints(conn, patient,
                                    daa_table=daa_main,
                                    require_root_for_hops=require_root_for_hops,
                                    important_table=important_table)

    positive_ids  = tep.satisfy_positive_literal_constraints(conn, patient,
                                             require_root_for_hops=require_root_for_hops,
                                             important_table=important_table,
                                             alt_mode=atag)

    prevention_ids: List[int] = []
    if enable_prevention_hits:
        prevention_ids = tep.satisfy_prevention_constraints(conn, patient,
                                             require_root_for_hops=require_root_for_hops,
                                             alt_mode=atag)

    if disease_only:
        union_ids = sorted(set(disease_ids))
    else:
        union_ids = sorted(set(disease_ids) | set(positive_ids) | set(prevention_ids))

    geo_screened_ids = union_ids
    geo_eliminated_set = set()

    LOGGER.info("patient=%s  mode=%s  prevent=%s  alt=%s  disease=%d  positive=%d  prevention=%d  union=%d  geo_after=%d  geo_elim=%d",
                patient, mode, _prevent_tag(enable_prevention_hits), atag,
                len(disease_ids), len(positive_ids), len(prevention_ids),
                len(union_ids), len(geo_screened_ids), 0)
    log_event({"type":"summary_stage_counts","patient_id":patient,"mode":mode,"prevent_tag":_prevent_tag(enable_prevention_hits),"alt_mode":atag,
               "counts":{"disease":len(disease_ids),
                         "positive":len(positive_ids),
                         "prevention":len(prevention_ids),
                         "union":len(union_ids),
                         "geo_after":len(geo_screened_ids),
                         "geo_elim":0}})

    disease_why  = _explain_disease_hits(conn, patient, important_table)
    # NOTE: keep the name "literal_why" to preserve downstream schema ("literal_vars"),
    # but it now stores positive-literal why-vars.
    literal_why  = _explain_positive_literal_hits(conn, patient, important_table, alt_mode=atag)

    for tid, info in sorted(disease_why.items()):
        log_event({"type":"why_hit","source":"disease","patient_id":patient,"mode":mode,"prevent_tag":_prevent_tag(enable_prevention_hits),"alt_mode":atag,
                   "trial_id":tid,"nct_id":info.get("nct_id"),"base_vars":info.get("vars",[])})
    for tid, info in sorted(literal_why.items()):
        log_event({"type":"why_hit","source":"literal","patient_id":patient,"mode":mode,"prevent_tag":_prevent_tag(enable_prevention_hits),"alt_mode":atag,
                   "trial_id":tid,"nct_id":info.get("nct_id"),"base_vars":info.get("vars",[])})

    survivors = tep.check_constraint_contradictions(conn, patient, geo_screened_ids, scope=scope)
    LOGGER.info("patient=%s  mode=%s  prevent=%s  alt=%s  survivors=%d", patient, mode, _prevent_tag(enable_prevention_hits), atag, len(survivors))

    eliminated_set = set(union_ids) - set(survivors)
    for tid in sorted(eliminated_set - geo_eliminated_set):
        log_event({"type":"eliminated","patient_id":patient,"mode":mode,"prevent_tag":_prevent_tag(enable_prevention_hits),"alt_mode":atag,"trial_id":tid})

    gap_rows = tep.evaluate_constraint_satisfaction_gap(conn, patient, scope=scope, candidate_trial_ids=union_ids, important_table=important_table)

    labels_by_tid: Dict[int, Dict[str, Any]] = {}
    survivors_set = set(survivors)
    for r in gap_rows:
        tid = int(r["trial_id"])
        is_survivor = tid in survivors_set
        if not is_survivor:
            label = "explicit_contradiction"
        else:
            label = "all_satisfied" if r["unsat_any"] == 0 else "unsatisfied_inclusion"
        labels_by_tid[tid] = {"is_survivor": is_survivor, "label": label}

    survivors_ordered = [r for r in gap_rows if int(r["trial_id"]) in survivors_set]
    eliminated_ordered = [r for r in gap_rows if int(r["trial_id"]) in (set(union_ids) - survivors_set)]

    ranked_rows: List[Dict[str, Any]] = []
    for r in survivors_ordered:
        info = labels_by_tid[int(r["trial_id"])]
        ranked_rows.append(dict(r, is_survivor=info["is_survivor"], label=info["label"]))
    for r in eliminated_ordered:
        info = labels_by_tid[int(r["trial_id"])]
        ranked_rows.append(dict(r, is_survivor=info["is_survivor"], label=info["label"]))

    detailed_rows, merged_rows = _write_retrieved_mappings(
        conn, base_out, patient, ranked_rows, disease_why, literal_why, mode, enable_prevention_hits, atag
    )
    _export_patient_labels_json(base_out, patient, merged_rows, mode, enable_prevention_hits, atag)
    _export_list_to_match(base_out, patient, detailed_rows, merged_rows, mode, enable_prevention_hits, atag)

    for idx, r in enumerate(ranked_rows, 1):
        log_event({
            "type":"rank",
            "patient_id": patient,
            "mode": mode,
            "prevent_tag": _prevent_tag(enable_prevention_hits),
            "alt_mode": atag,
            "rank": idx,
            "trial_id": r["trial_id"],
            "status": "survivor" if r["label"] != "explicit_contradiction" else "eliminated",
            "label": r["label"],
            "frac_unsat_any": r["frac_unsat_any"],
            "total_clauses": r["total_clauses"],
            "unsat_any": r["unsat_any"],
            "unsat_explicit": r["unsat_explicit"],
        })

    payload = {
        "patient_id": patient,
        "mode": mode,
        "prevent_tag": _prevent_tag(enable_prevention_hits),
        "alt_mode": atag,
        "important_table": important_table,
        "counts": {
            "disease": len(disease_ids),
            "positive": len(positive_ids),
            "prevention": len(prevention_ids),
            "union": len(union_ids),
            "geo_after": len(geo_screened_ids),
            "survivors": len(survivors),
        },
        "disease_trial_ids": disease_ids,
        "positive_trial_ids": positive_ids,
        "prevention_trial_ids": prevention_ids,
        "union_trial_ids": union_ids,
        "geo_screened_trial_ids": geo_screened_ids,
        "survivor_trial_ids": survivors,
    }
    return payload, survivors, ranked_rows

# ---------------------------
# Worker helper for parallelism (MODE+PREVENTION+ALT)
# ---------------------------

def _process_patient_worker(db_path: Path,
                            patient_id: str,
                            scope: str,
                            out_dir: Path | None,
                            require_root_for_hops: bool,
                            verbose: bool,
                            disease_only: bool,
                            enable_prevention_hits: bool,
                            important_table: str,
                            mode: str,
                            alt_mode: str):
    conn = sqlite3.connect(str(db_path))
    _, worker_log_event = configure_logging(
        out_dir,
        verbose,
        truncate_jsonl=False,
        mode=mode,
        enable_prevention_hits=enable_prevention_hits,
        alt_mode=alt_mode,
    )
    payload, survivors, ranked_rows = run_for_patient(
        conn,
        worker_log_event,
        patient_id,
        scope,
        out_dir,
        important_table=important_table,
        mode=mode,
        require_root_for_hops=require_root_for_hops,
        disease_only=disease_only,
        enable_prevention_hits=enable_prevention_hits,
        alt_mode=alt_mode,
    )
    conn.close()
    return patient_id, payload, survivors, ranked_rows

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Union disease+(positive)(+prevention) hits → eliminate → rank (survivors first). "
                    "MODE-AWARE + PREVENTION-SUFFIX + ALT-MODE outputs. (No literal_hits)"
    )
    ap.add_argument("--db", default="../../build/trial.db", type=Path)
    ap.add_argument("--patient", default=None,
                    help="Run for a single patient_id; omit to run for ALL patients")
    ap.add_argument("--scope", choices=["now","any"], default="any",
                    help="Range-aware scope for elimination and ranking (retrieval ignores timeframe)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write results & logs to this directory (mode+prevent+alt scoped subdirs are created).")
    ap.add_argument("--quiet", action="store_true",
                    help="CSV lines: patient_id,mode,prevent_tag,alt_mode,trial_id,status,label,rank,frac_unsat_any,"
                         "pct_unsat_any,pct_unsat_explicit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-hop-without-root", action="store_true",
                    help="Let accepted-alternative (hop>0) matches count even if is_root gate is not enforced")
    ap.add_argument("--disease-only", action="store_true",
                    help="Use only disease_hits() for retrieval candidates (ignore positive/prevention hits).")
    ap.add_argument("--enable-prevention-hits", action="store_true",
                    help="Also include prevention_hits() in retrieval union.")
    ap.add_argument("--alt-mode", choices=["act","nonact"], default="act",
                    help="Accepted-alternatives variant: act (default) vs nonact (*_nonact tables)")
    ap.add_argument("--parallel", type=int, default=8,
                    help="Number of workers for per-patient parallelism (>=1)")
    ap.add_argument("--important-mode", choices=["chief","ccr","all"], default="all",
                    help="Select important facts table: patient_inclusion_constraints_important_{mode}")

    args = ap.parse_args()

    mode = args.important_mode
    important_table = f"patient_inclusion_constraints_important_{mode}"
    alt_mode = _alt_tag(args.alt_mode)

    _, log_event = configure_logging(
        args.out,
        args.verbose,
        truncate_jsonl=True,
        mode=mode,
        enable_prevention_hits=args.enable_prevention_hits,
        alt_mode=alt_mode
    )

    try:
        conn = sqlite3.connect(str(args.db))
    except Exception as e:
        print(f"[error] failed to open DB: {e}", file=sys.stderr)
        sys.exit(2)

    require_root_for_hops = not args.allow_hop_without_root

    if args.patient:
        log_event({"type":"run_start","mode":"single","patient_id":args.patient,
                   "important_mode":mode,"important_table":important_table,
                   "prevent_tag":_prevent_tag(args.enable_prevention_hits),
                   "alt_mode":alt_mode})
        payload, survivors, ranked_rows = run_for_patient(
            conn,
            log_event,
            args.patient,
            args.scope,
            args.out,
            important_table=important_table,
            mode=mode,
            require_root_for_hops=require_root_for_hops,
            disease_only=args.disease_only,
            enable_prevention_hits=args.enable_prevention_hits,
            alt_mode=alt_mode,
        )
        _emit_single(payload, survivors, ranked_rows, args.out, args.quiet, mode, args.enable_prevention_hits, alt_mode)
        log_event({"type":"run_end","patient_id":args.patient,"mode":mode,"prevent_tag":_prevent_tag(args.enable_prevention_hits),"alt_mode":alt_mode,"survivors":len(survivors)})

    else:
        patients = _get_all_patients(conn, important_table)
        if not patients:
            LOGGER.info("No patients found.")
            print(json.dumps({"patients": {}}, ensure_ascii=False))
            return

        log_event({"type":"run_start","mode":"all","num_patients":len(patients),
                   "important_mode":mode,"important_table":important_table,
                   "prevent_tag":_prevent_tag(args.enable_prevention_hits),
                   "alt_mode":alt_mode})
        num_workers = max(1, int(args.parallel or 1))

        if num_workers == 1:
            for idx, pid in enumerate(patients, 1):
                LOGGER.info("[%d/%d] patient=%s mode=%s prevent=%s alt=%s", idx, len(patients), pid, mode, _prevent_tag(args.enable_prevention_hits), alt_mode)
                log_event({"type":"patient_start","index":idx,"total":len(patients),"patient_id":pid,"mode":mode,"prevent_tag":_prevent_tag(args.enable_prevention_hits),"alt_mode":alt_mode})
                payload, survivors, ranked_rows = run_for_patient(
                    conn,
                    log_event,
                    pid,
                    args.scope,
                    args.out,
                    important_table=important_table,
                    mode=mode,
                    require_root_for_hops=require_root_for_hops,
                    disease_only=args.disease_only,
                    enable_prevention_hits=args.enable_prevention_hits,
                    alt_mode=alt_mode,
                )
                _emit_single(payload, survivors, ranked_rows, args.out, args.quiet, mode, args.enable_prevention_hits, alt_mode)
                log_event({"type":"patient_end","patient_id":pid,"mode":mode,"prevent_tag":_prevent_tag(args.enable_prevention_hits),"alt_mode":alt_mode})

        else:
            conn.close()

            tasks = [
                (args.db, pid, args.scope, args.out,
                 require_root_for_hops, args.verbose,
                 args.disease_only, args.enable_prevention_hits,
                 important_table, mode, alt_mode)
                for pid in patients
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_process_patient_worker, *t) for t in tasks]
                for fut in as_completed(futures):
                    try:
                        _, payload, survivors, ranked_rows = fut.result()
                    except Exception as e:
                        print(f"[error] worker failed: {e}", file=sys.stderr)
                        continue
                    _emit_single(payload, survivors, ranked_rows, args.out, args.quiet, mode, args.enable_prevention_hits, alt_mode)

        log_event({"type":"run_end","mode":"all","num_patients":len(patients),
                   "important_mode":mode,"prevent_tag":_prevent_tag(args.enable_prevention_hits),
                   "alt_mode":alt_mode})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[warn] interrupted", file=sys.stderr)
        sys.exit(130)