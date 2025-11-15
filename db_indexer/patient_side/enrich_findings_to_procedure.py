#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_findings_to_procedure.py — streaming + on-disk dedupe
────────────────────────────────────────────────────────────────────────
From TRUE Finding variables (incl. implied names) derive Procedure variables.

Inputs (per patient folder under IN_ROOT):
  • Prefer: canonical.enriched.annotated.jsonl  (line-by-line)
  • Fallbacks: canonical.enriched.jsonl / canonical.jsonl / *.enriched.jsonl

Outputs (per side):
  • finding_to_procedure_enriched.annotated.jsonl   (streamed, per row)
  • finding_to_procedure_enrichment.structured.json (SMALL summary only)
  • finding_to_procedure_enrichment.report.json     (counts)
  • finding_to_procedure_enrichment.per_var.jsonl   (streamed)
  • finding_to_procedure_enrichment.flat.jsonl.tmp  (streamed, pre-dedupe)
  • finding_to_procedure_enrichment.flat.json       (FINAL array, deduped on disk)

Memory safety:
  • No giant lists in Python. We stream rows and dedupe later using a tiny sqlite PK table.
"""

from __future__ import annotations
import os, sys, json, csv, re, argparse
from typing import Dict, Any, List, Optional, Tuple
from glob import glob
import requests
import sqlite3

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Roots / services
# 正常 pipeline：应通过 --in-root / --out-root 显式传入。
# 单独跑脚本：可以设置环境变量 F2P_INPUT_ROOT / F2P_OUTPUT_ROOT 作为 fallback。
IN_ROOT  = os.environ.get("F2P_INPUT_ROOT", "")
OUT_ROOT = os.environ.get("F2P_OUTPUT_ROOT", "")

SNOMED_BASE   = os.getenv("SNOMED_BASE", "http://localhost:8080")
SNOMED_BRANCH = os.getenv("SNOMED_BRANCH", "MAIN")

# Mapping CSV (required, term-based hasInterpretation → status flags)
PROC_MAP_CSV  = os.getenv(
    "FINDING_TO_PROC_MAPPING_CSV",
    os.path.join(MODULE_DIR, "../rules/mapping/finding_interprets_procedure_has_interpretation_qualifier_value_mapping.csv")
).strip()

# SNOMED constants
CID_INTERPRETS         = "363714003"
CID_HAS_INTERPRETATION = "363713009"

# Templates / regex
TIMEFRAME_UNITS = r"(?:minutes|hours|days|weeks|months|years)"
TIMEFRAME_RE = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+" + TIMEFRAME_UNITS + r"|"
    r"inthefuture\d+" + TIMEFRAME_UNITS + r")"
)
QUAL_SUFFIX_RE = r"(?:@@[a-z0-9_]+(?:@@[a-z0-9_]+)*)?$"

FINDING_STEMS_ALL = [
    "has_diagnosis_of_{e}_{t}",
    "has_finding_of_{e}_{t}",
    "has_symptoms_of_{e}_{t}",
    "has_clinical_signs_of_{e}_{t}",
    "has_suspicion_of_{e}_{t}",
]
FINDING_STEMS_EXC = [
    "has_diagnosis_of_{e}_{t}",
    "has_finding_of_{e}_{t}",
]

UNDERGONE_TPL = "patient_has_undergone_{e}_{t}"
UNDERGONE_OUTCOME_TPL = {
    "positive":  "patient_has_undergone_{e}_{t}_outcome_is_positive",
    "negative":  "patient_has_undergone_{e}_{t}_outcome_is_negative",
    "normal":    "patient_has_undergone_{e}_{t}_outcome_is_normal",
    "abnormal":  "patient_has_undergone_{e}_{t}_outcome_is_abnormal",
}

DERIVATION_STAGE = "f2p"
DERIVATION_RULE  = "interprets(363714003)"

def _dedupe_key(name: str, time_ctx: Dict[str, Any]) -> str:
    """name + 4 time fields → unique key; hours rounded to 3 decimals."""
    def r(v):
        if v is None: return "null"
        try: return f"{round(float(v), 3):.3f}"
        except Exception: return str(v)
    def b(v): return "1" if bool(v) else "0"
    return "|".join([
        name,
        r(time_ctx.get("start_time_in_hours")),
        r(time_ctx.get("end_time_in_hours")),
        b(time_ctx.get("start_time_inclusive")),
        b(time_ctx.get("end_time_inclusive")),
    ])


def _ensure_dir(p: str) -> None: os.makedirs(p, exist_ok=True)
def _boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() == "true" if v.strip().lower() in ("true","false") else None
    return None


def _compile_finding_regexes(stems: List[str]) -> List[re.Pattern]:
    """
    Compile regexes where {t} is OPTIONAL, and if stem contains '_{t}' we
    make that whole segment optional. {e} remains greedy.
    """
    out = []
    for stem in stems:
        esc = re.escape(stem)
        # optional timeframe token with optional leading underscore
        if r"_\{t\}" in esc:
            esc = esc.replace(r"_\{t\}", rf"(?:_(?P<t>{TIMEFRAME_RE}))?")
        else:
            esc = esc.replace(r"\{t\}",  rf"(?P<t>{TIMEFRAME_RE})?")
        # entity
        esc = esc.replace(r"\{e\}", r"(?P<e>.+)")
        out.append(re.compile(r"^(?:patient_)?"+esc+QUAL_SUFFIX_RE, re.I))
    return out



def _extract_timeframe_from_var(name: str, fallback: Optional[str]) -> Optional[str]:
    m = re.search(TIMEFRAME_RE, name); return m.group(0) if m else (fallback or None)
def _canon(s: str) -> str: s=re.sub(r"\s*\([^)]*\)\s*$","",s or "").strip(); return re.sub(r"\s+","_",s.lower())


def _render(tpl: str, e: str, t: Optional[str]) -> str:
    s = tpl.replace("{e}", e or "")
    # remove "_{t}" or "{t}" entirely; collapse underscores and trim
    s = re.sub(r"_\{t\}", "", s)
    s = s.replace("{t}", "")
    s = re.sub(r"__+", "_", s).strip("_")
    return s


def load_proc_mapping(csv_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    if not os.path.isfile(csv_path):
        print(f"[err] mapping CSV not found: {csv_path}", file=sys.stderr); sys.exit(2)
    m: Dict[str, Dict[str, Dict[str, int]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            term = (row.get("hasInterpretation_destTerm") or "").strip().lower()
            if not term: continue
            entry = m.setdefault(term, {"inc":{"normal":0,"abnormal":0,"positive":0,"negative":0},
                                        "exc":{"normal":0,"abnormal":0,"positive":0,"negative":0}})
            def flag(col:str)->int:
                v=(row.get(col) or "").strip()
                if v=="": return 0
                try: return 1 if int(float(v))!=0 else 0
                except Exception: return 1 if v.lower() in ("y","yes","true","t") else 0
            for k in ("normal","abnormal","positive","negative"):
                entry["inc"][k] = max(entry["inc"][k], flag(f"{k}(inc)"))
                entry["exc"][k] = max(entry["exc"][k], flag(f"{k}(exc)"))
    return m

# ---- Time-window (NEW) ----
TIME_KEYS = (
    "start_time_in_hours",
    "end_time_in_hours",
    "start_time_inclusive",
    "end_time_inclusive",
)

def _copy_time_fields(src: Dict[str, Any], dst: Dict[str, Any]) -> None:
    for k in TIME_KEYS:
        if k in src:
            dst[k] = src[k]


# Snowstorm
_sess = requests.Session()
def snowstorm_get_concept(cid: str, timeout: float = 100.0) -> Optional[Dict[str, Any]]:
    try:
        r = _sess.get(f"{SNOMED_BASE.rstrip('/')}/browser/{SNOMED_BRANCH}/concepts/{cid}", timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return None

def extract_groups_interprets_procedure_and_hi(doc: Dict[str, Any]) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    groups: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
    for rel in (doc or {}).get("relationships") or []:
        if not rel.get("active", True): continue
        gid = int(rel.get("groupId") or 0)
        bucket = groups.setdefault(gid, {"interprets_proc": [], "has_interpretation": []})
        rtype = (rel.get("type") or {}).get("conceptId")
        tgt   = rel.get("target") or {}
        fsn   = ((tgt.get("fsn") or {}) or {}).get("term") or ""
        pt    = ((tgt.get("pt")  or {}) or {}).get("term") or ""
        if rtype == CID_INTERPRETS:
            if "(procedure)" in (fsn or "").lower():
                bucket["interprets_proc"].append((tgt.get("conceptId") or "", _canon(pt or fsn)))
        elif rtype == CID_HAS_INTERPRETATION:
            term = pt or fsn
            bucket["has_interpretation"].append((tgt.get("conceptId") or "", term))
    return groups

def _collect_candidate_vars_for_side(row: Dict[str, Any], side: str, rx_list: List[re.Pattern]) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    def _maybe(name: str):
        name = (name or "").strip()
        if not name: return
        for rx in rx_list:
            if rx.match(name):
                out.append((name, None))  # timeframe no longer extracted from name
                break
    if _boolish(row.get("extracted_value")) is True:
        _maybe(str(row.get("entity_variable_name") or ""))
    imp = (row.get("implications") or {})
    for si in (imp.get("schema_new_only") or []):   _maybe(str(si.get("name") or ""))
    for ti in (imp.get("timeframe_new_only") or []): _maybe(str(ti.get("name") or ""))
    seen, uniq = set(), []
    for v, tf in out:
        if v in seen: continue
        seen.add(v); uniq.append((v, tf))
    return uniq


# ---------- on-disk dedupe ----------
def _spill_open(dbp: str) -> sqlite3.Connection:
    _ensure_dir(os.path.dirname(dbp))
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY)")
    conn.commit()
    return conn

def process_side(in_root: str, out_root: str, side: str, patients: List[str], input_filename: str) -> None:
    if not PROC_MAP_CSV:
        print("[err] FINDING_TO_PROC_MAPPING_CSV not set.", file=sys.stderr); sys.exit(2)
    mapping = load_proc_mapping(PROC_MAP_CSV)
    stems = FINDING_STEMS_ALL if side=="inclusion" else FINDING_STEMS_EXC
    rx_list = _compile_finding_regexes(stems)

    for pid in patients:
        pin  = os.path.join(in_root, pid)
        pout = os.path.join(out_root, pid, side); _ensure_dir(pout)

        # locate input (prefer annotated.jsonl)
        def find_input(patient_dir: str, preferred: str) -> Optional[str]:
            p = os.path.join(patient_dir, preferred)
            if os.path.isfile(p): return p
            for c in ["canonical.enriched.annotated.jsonl",
                      "canonical.enriched.jsonl",
                      "canonical.jsonl",
                      "canonical.enriched.healed.jsonl"]:
                q = os.path.join(patient_dir, c)
                if os.path.isfile(q): return q
            hits = sorted(glob(os.path.join(patient_dir, "*.jsonl")))
            return hits[0] if hits else None

        inp = find_input(pin, input_filename)
        if not inp:
            print(f"[{side:^9}] {pid:<24} skipped (no input jsonl found; tried '{input_filename}')")
            continue

        # outputs (streamed)
        out_annot   = os.path.join(pout, "finding_to_procedure_enriched.annotated.jsonl")
        out_pervar  = os.path.join(pout, "finding_to_procedure_enrichment.per_var.jsonl")
        out_flat_tmp= os.path.join(pout, "finding_to_procedure_enrichment.flat.jsonl.tmp")
        out_flat    = os.path.join(pout, "finding_to_procedure_enrichment.flat.json")
        out_struct  = os.path.join(pout, "finding_to_procedure_enrichment.structured.json")
        out_rep     = os.path.join(pout, "finding_to_procedure_enrichment.report.json")

        # reset files
        for p in (out_annot, out_pervar, out_flat_tmp):
            open(p, "w", encoding="utf-8").close()

        # counters only (no big lists)
        c_rows = c_rows_with_prod = c_prod_vars = 0
        c_conflict = c_not_match = c_not_true = c_no_interprets = 0

        # sqlite “set” for per-row produced name dedupe
        seen_db = os.path.join(pout, ".tmp", "seen.sqlite")
        if os.path.exists(seen_db):
            os.remove(seen_db)          # 确保每轮干净开始
        conn = _spill_open(seen_db); cur = conn.cursor()


        with open(inp, "r", encoding="utf-8") as f, \
             open(out_annot, "a", encoding="utf-8") as fa, \
             open(out_pervar, "a", encoding="utf-8") as fpv, \
             open(out_flat_tmp, "a", encoding="utf-8") as fft:

            for ln in f:
                s = ln.strip()
                if not s: continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue

                c_rows += 1
                cid = str(row.get("conceptId") or "").strip()
                src_term = row.get("preferred_term") or row.get("fully_specified_name") or ""
                base_var = str(row.get("entity_variable_name") or "")
                timeframe0 = _extract_timeframe_from_var(base_var, str(row.get("timeframe") or ""))

                time_ctx = {
                    "start_time_in_hours": row.get("start_time_in_hours"),
                    "end_time_in_hours": row.get("end_time_in_hours"),
                    "start_time_inclusive": row.get("start_time_inclusive"),
                    "end_time_inclusive": row.get("end_time_inclusive"),
                }
                
                candidates = _collect_candidate_vars_for_side(row, side, rx_list)
                if not candidates:
                    reason = None
                    if any(rx.match(base_var) for rx in rx_list):
                        if _boolish(row.get("extracted_value")) is not True:
                            reason = "extracted_value_not_true"; c_not_true += 1
                        else:
                            reason = "no_candidates"
                    else:
                        reason = "not_matching_template"; c_not_match += 1

                    enr = {"side": side, "eligible": False, "produced": [], "conflict_flags": [], "reason": reason}
                    row["finding_to_procedure_enrichment"] = enr
                    fa.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fpv.write(json.dumps({
                        "patient_id": pid, "fact_id": row.get("fact_id"), "conceptId": row.get("conceptId"),
                        "var_name": base_var, "timeframe": timeframe0, "converted": 0, "reason": reason,
                        "procedure": "", "hi_terms": "", "produced_names": "", "produced_statuses": ""
                    }, ensure_ascii=False) + "\n")
                    continue

                if not cid:
                    enr = {"side": side, "eligible": True, "produced": [], "conflict_flags": [], "reason": "no_conceptId"}
                    row["finding_to_procedure_enrichment"] = enr
                    fa.write(json.dumps(row, ensure_ascii=False) + "\n")
                    for var_name, tf in candidates:
                        fpv.write(json.dumps({
                            "patient_id": pid, "fact_id": row.get("fact_id"), "conceptId": "",
                            "var_name": var_name, "timeframe": tf, "converted": 0, "reason": "no_conceptId",
                            "procedure": "", "hi_terms": "", "produced_names": "", "produced_statuses": ""
                        }, ensure_ascii=False) + "\n")
                    continue

                doc = snowstorm_get_concept(cid)
                groups = extract_groups_interprets_procedure_and_hi(doc or {})
                any_proc = any(bool(g["interprets_proc"]) for g in groups.values())
                if not any_proc:
                    c_no_interprets += 1

                produced_names_for_row: List[str] = []
                produced_items_for_row: List[Dict[str, Any]] = []
                conflict_flags: List[str] = []


                for var_name, timeframe in candidates:
                    # match to get timeframe if missing
                    mtf = timeframe
                    for rx in rx_list:
                        m = rx.match(var_name)
                        if m: mtf = mtf or m.group("t"); break

                    for g in groups.values():
                        if not g["interprets_proc"]: continue
                        for proc_id, proc_canon in g["interprets_proc"]:
                            if not g["has_interpretation"]:
                                name = _render(UNDERGONE_TPL, proc_canon, None)  # ensure no timeframe in name
                                key  = _dedupe_key(name, time_ctx)
                                cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                                if cur.rowcount == 1:
                                    produced_names_for_row.append(name)
                                    produced_items_for_row.append({
                                        "name": name,
                                        "procedureId": proc_id,
                                        "procedureCanonical": proc_canon
                                    })

                                    rec = {
                                        "patient_id": pid,
                                        "class": "finding_to_procedure",
                                        "new_variable_name": name,
                                        "target_variable_name": name,
                                        # no "timeframe"
                                        "source_variable_name": var_name,
                                        "source_conceptId": cid,
                                        "derivation_stage": DERIVATION_STAGE,
                                        "derivation_rule": DERIVATION_RULE,
                                        "derived_from_variable": base_var,
                                        "derived_from_conceptId": cid,
                                        "derived_from_entity_term": src_term,
                                        "derived_conceptId": proc_id,
                                        "derived_entity_term": proc_canon,
                                        "derived_qualifierId": None,
                                        "derived_qualifier_term": None,
                                        "fact_id": row.get("fact_id"),
                                    }
                                    rec.update(time_ctx)
                                    fft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                # else: 同名同时间窗已存在，跳过写出

                                continue

                            for hi_id, hi_term in g["has_interpretation"]:
                                flags = mapping.get((hi_term or "").strip().lower())
                                # unknown HI term → fallback to base undergone
                                if not flags:
                                    name = _render(UNDERGONE_TPL, proc_canon, None)
                                    key  = _dedupe_key(name, time_ctx)
                                    cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                                    if cur.rowcount == 1:
                                        produced_names_for_row.append(name)
                                        produced_items_for_row.append({
                                            "name": name,
                                            "procedureId": proc_id,
                                            "procedureCanonical": proc_canon
                                        })
                                        rec = {
                                            "patient_id": pid,
                                            "class": "finding_to_procedure",
                                            "new_variable_name": name,
                                            "target_variable_name": name,
                                            "source_variable_name": var_name,
                                            "source_conceptId": cid,
                                            "derivation_stage": DERIVATION_STAGE,
                                            "derivation_rule": DERIVATION_RULE,
                                            "derived_from_variable": base_var,
                                            "derived_from_conceptId": cid,
                                            "derived_from_entity_term": src_term,
                                            "derived_conceptId": proc_id,
                                            "derived_entity_term": proc_canon,
                                            "derived_qualifierId": None,
                                            "derived_qualifier_term": None,
                                            "fact_id": row.get("fact_id"),
                                        }
                                        rec.update(time_ctx)
                                        fft.write(json.dumps(rec, ensure_ascii=False) + "\n")

                                    continue

                                use = flags["inc" if side=="inclusion" else "exc"]
                                if (use["normal"] and use["abnormal"]) or (use["positive"] and use["negative"]):
                                    conflict_flags.append(f"conflict:{hi_term}")
                                    name = _render(UNDERGONE_TPL, proc_canon, None)
                                    key  = _dedupe_key(name, time_ctx)
                                    cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                                    if cur.rowcount == 1:
                                        produced_names_for_row.append(name)
                                        produced_items_for_row.append({
                                            "name": name,
                                            "procedureId":     proc_id,
                                            "procedureCanonical": proc_canon,
                                        })


                                        rec = {
                                            "patient_id": pid,
                                            "class": "finding_to_procedure",
                                            "new_variable_name": name,
                                            "target_variable_name": name,
                                            "source_variable_name": var_name,
                                            "source_conceptId": cid,
                                            "derivation_stage": DERIVATION_STAGE,
                                            "derivation_rule": DERIVATION_RULE,  # 如你此处有自定义规则，可换回去
                                            "derived_from_variable": base_var,
                                            "derived_from_conceptId": cid,
                                            "derived_from_entity_term": src_term,
                                            "derived_conceptId": proc_id,
                                            "derived_entity_term": proc_canon,
                                            "derived_qualifierId": None,
                                            "derived_qualifier_term": None,
                                            "fact_id": row.get("fact_id"),
                                        }
                                        rec.update(time_ctx)
                                        fft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    # else: 已存在，跳过


                                    continue

                                for status in ("positive","negative","normal","abnormal"):
                                    if use[status]:
                                        name = _render(UNDERGONE_OUTCOME_TPL[status], proc_canon, None)
                                        key  = _dedupe_key(name, time_ctx)
                                        cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                                        if cur.rowcount == 1:
                                            produced_names_for_row.append(name)
                                            produced_items_for_row.append({
                                                "name": name,
                                                "procedureId":     proc_id,
                                                "procedureCanonical": proc_canon,
                                                "hasInterpretationId":   hi_id,
                                                "hasInterpretationTerm": hi_term,
                                                "status":                status
                                            })

                                            rec = {
                                                "patient_id": pid,
                                                "class": "finding_to_procedure",
                                                "new_variable_name": name,
                                                "target_variable_name": name,
                                                "source_variable_name": var_name,
                                                "source_conceptId": cid,
                                                "derivation_stage": DERIVATION_STAGE,
                                                "derivation_rule": DERIVATION_RULE,
                                                "derived_from_variable": base_var,
                                                "derived_from_conceptId": cid,
                                                "derived_from_entity_term": src_term,
                                                "derived_conceptId": proc_id,
                                                "derived_entity_term": proc_canon,
                                                "derived_qualifierId": hi_id,          # ← outcome 分支保留 HI
                                                "derived_qualifier_term": hi_term,     # ← outcome 分支保留 HI
                                                "fact_id": row.get("fact_id"),
                                            }
                                            rec.update(time_ctx)
                                            fft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                        # else: 已存在，跳过



                # finalize per-base row
                produced_names_for_row = sorted(set(produced_names_for_row))
                if produced_names_for_row:
                    c_rows_with_prod += 1
                    c_prod_vars += len(produced_names_for_row)
                if conflict_flags:
                    c_conflict += 1

                fa.write(json.dumps({
                    **row,
                    "finding_to_procedure_enrichment": {
                        "side": side,
                        "eligible": True,
                        "produced": produced_items_for_row,
                        "conflict_flags": conflict_flags,
                        "reason": "ok" if produced_names_for_row else ("no_interprets_to_procedure" if not any_proc else "no_output_after_mapping")
                    }
                }, ensure_ascii=False) + "\n")

                fpv.write(json.dumps({
                    "patient_id": pid, "fact_id": row.get("fact_id"), "conceptId": row.get("conceptId"),
                    "var_name": base_var, "timeframe": timeframe0,
                    "converted": 1 if produced_names_for_row else 0,
                    "reason": "ok" if produced_names_for_row else ("no_interprets_to_procedure" if not any_proc else "no_output_after_mapping"),
                    "procedure": "", "hi_terms": "",  # kept minimal
                    "produced_names": ";".join(produced_names_for_row),
                    "produced_statuses": ""  # omitted for memory
                }, ensure_ascii=False) + "\n")

        conn.commit(); conn.close()

        # finalize reports (SMALL)
        summary = {
            "rows": c_rows,
            "rows_with_produced": c_rows_with_prod,
            "produced_variables": c_prod_vars,
            "conflicted_rows": c_conflict,
            "rows_not_matching_template": c_not_match,
            "rows_extracted_value_not_true": c_not_true,
            "rows_missing_interprets_procedure": c_no_interprets,
        }
        with open(out_struct, "w", encoding="utf-8") as fs:
            json.dump({"patient_id": pid, "side": side, "summary": summary}, fs, ensure_ascii=False, indent=2)
        with open(out_rep, "w", encoding="utf-8") as fr:
            json.dump({"patient_id": pid, "side": side, "counts": summary}, fr, ensure_ascii=False, indent=2)

        # convert tmp jsonl → final array json without loading all (stream write)
        first = True
        with open(out_flat, "w", encoding="utf-8") as fout, open(out_flat_tmp, "r", encoding="utf-8") as fin:
            fout.write("[\n")
            for line in fin:
                s=line.strip()
                if not s: continue
                if not first: fout.write(",\n")
                fout.write(s)
                first=False
            fout.write("\n]\n")
        # keep the *.tmp for debugging? remove if you like:
        # os.remove(out_flat_tmp)

        print(f"[{side:^9}] {pid:<24} ok (produced={summary['produced_variables']}, conflicts={summary['conflicted_rows']}, not_true={summary['rows_extracted_value_not_true']})")

# CLI
def main():
    ap = argparse.ArgumentParser(description="Finding → Procedure enricher (memory-safe streaming).")
    ap.add_argument("--patient", action="append",
                    help="Only process this patient id. If omitted, process all under IN_ROOT.")
    ap.add_argument("--input-filename", default="canonical.enriched.annotated.jsonl",
                    help="Preferred per-patient input filename (default).")
    ap.add_argument("--side", choices=["inclusion","exclusion","both"], default="inclusion")
    ap.add_argument(
        "--in-root",
        dest="in_root",
        required=False,
        default=None,
        help="Input root that contains per-patient dirs. Required unless F2P_INPUT_ROOT is set.",
    )
    ap.add_argument(
        "--out-root",
        dest="out_root",
        required=False,
        default=None,
        help="Output root to write results. Required unless F2P_OUTPUT_ROOT is set.",
    )
    args = ap.parse_args()

    # 解析 in_root / out_root：优先命令行，其次环境变量，缺失就报错
    if args.in_root:
        in_root = os.path.abspath(args.in_root)
    elif IN_ROOT:
        in_root = os.path.abspath(IN_ROOT)
    else:
        print("[f2p] --in-root is required unless F2P_INPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if args.out_root:
        out_root = os.path.abspath(args.out_root)
    elif OUT_ROOT:
        out_root = os.path.abspath(OUT_ROOT)
    else:
        print("[f2p] --out-root is required unless F2P_OUTPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(in_root):
        print(f"[err] IN_ROOT not found: {in_root}", file=sys.stderr); sys.exit(2)
    _ensure_dir(out_root)

    all_entries = sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))])
    patients = all_entries if not args.patient else [e for e in all_entries if e in set(args.patient)]
    if not patients:
        print(f"[warn] no matching patients under {in_root}"); return
    if not PROC_MAP_CSV:
        print("[err] FINDING_TO_PROC_MAPPING_CSV not set.", file=sys.stderr); sys.exit(2)

    if args.side in ("inclusion","exclusion"):
        process_side(in_root, out_root, args.side, patients, args.input_filename)
    else:
        process_side(in_root, out_root, "inclusion", patients, args.input_filename)
        process_side(in_root, out_root, "exclusion", patients, args.input_filename)

    print("[done] all requested sides processed.")

if __name__ == "__main__":
    main()
