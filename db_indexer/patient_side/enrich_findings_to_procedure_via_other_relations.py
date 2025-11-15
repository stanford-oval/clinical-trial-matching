#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_findings_to_procedure_via_other_relations.py — streaming
────────────────────────────────────────────────────────────────────────
From TRUE Finding variables, via attributes (During/After/Due to/Associated with/Temporally related to)
derive Procedure variables (base undergone only).

Inputs per patient (prefer jsonl):
  • canonical.enriched.annotated.jsonl (or fallback *enriched*.jsonl / *.jsonl)

Outputs per side:
  • finding_to_procedure_via_relations.annotated.jsonl
  • finding_to_procedure_via_relations.structured.json  (SMALL summary)
  • finding_to_procedure_via_relations.report.json
  • finding_to_procedure_via_relations.per_var.jsonl
  • finding_to_procedure_via_relations.flat.json        (FINAL array built from a streamed tmp)
"""

from __future__ import annotations
import os, sys, json, re, argparse
from typing import Dict, Any, List, Optional, Tuple
from glob import glob
import requests, sqlite3

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# IN_ROOT / OUT_ROOT 仅从环境变量读取，供单独跑脚本时偷懒；
# 正常 pipeline 应通过命令行 --in-root / --out-root 显式传入。
IN_ROOT  = os.environ.get("F2P_REL_INPUT_ROOT", "")
OUT_ROOT = os.environ.get("F2P_REL_OUTPUT_ROOT", "")

SNOMED_BASE   = os.getenv("SNOMED_BASE", "http://localhost:8080")
SNOMED_BRANCH = os.getenv("SNOMED_BRANCH", "MAIN")

CID_PROCEDURE          = "71388002"
CID_DURING             = "371881003"
CID_AFTER              = "255234002"
CID_DUE_TO             = "42752001"
CID_ASSOCIATED_WITH    = "47429007"
CID_TEMPORALLY_RELATED = "726633004"

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
ATTR_CLASS_MAP = {
    CID_DURING:             "finding_to_procedure_during",
    CID_AFTER:              "finding_to_procedure_after",
    CID_DUE_TO:             "finding_to_procedure_due_to",
    CID_ASSOCIATED_WITH:    "finding_to_procedure_associated_with",
    CID_TEMPORALLY_RELATED: "finding_to_procedure_temporally_related_to",
}
RELATION_LABEL = {
    CID_DURING:             "during",
    CID_AFTER:              "after",
    CID_DUE_TO:             "due_to",
    CID_ASSOCIATED_WITH:    "associated_with",
    CID_TEMPORALLY_RELATED: "temporally_related_to",
}
EXCLUSION_DISABLED_ATTRS = { CID_ASSOCIATED_WITH }

DERIVATION_STAGE = "f2p_rel"


# ---- Time-window (NEW) ----
TIME_KEYS = (
    "start_time_in_hours",
    "end_time_in_hours",
    "start_time_inclusive",
    "end_time_inclusive",
)

def _time_ctx(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_time_in_hours": row.get("start_time_in_hours"),
        "end_time_in_hours": row.get("end_time_in_hours"),
        "start_time_inclusive": row.get("start_time_inclusive"),
        "end_time_inclusive": row.get("end_time_inclusive"),
    }

def _dedupe_key(name: str, tc: Dict[str, Any]) -> str:
    def r(v):
        if v is None: return "null"
        try: return f"{round(float(v), 3):.3f}"
        except Exception: return str(v)
    def b(v): return "1" if bool(v) else "0"
    return "|".join([
        name,
        r(tc.get("start_time_in_hours")),
        r(tc.get("end_time_in_hours")),
        b(tc.get("start_time_inclusive")),
        b(tc.get("end_time_inclusive")),
    ])


def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool): return v
    if isinstance(v, str):
        s=v.strip().lower()
        if s in ("true","false"): return s=="true"
    return None

def _compile_finding_regexes(stems: List[str]) -> List[re.Pattern]:
    """
    Compile regexes where {t} is OPTIONAL (whole `_{t}` segment optional);
    {e} is greedy. This supports var names without timeframe.
    """
    out = []
    for stem in stems:
        esc = re.escape(stem)
        # make "_{t}" optional if exists; else make "{t}" optional
        if r"_\{t\}" in esc:
            esc = esc.replace(r"_\{t\}", r"(?:_(?P<t>[a-z0-9_]+))?")
        else:
            esc = esc.replace(r"\{t\}",  r"(?P<t>[a-z0-9_]+)?")
        esc = esc.replace(r"\{e\}", r"(?P<e>.+)")
        out.append(re.compile(r"^(?:patient_)?"+esc+QUAL_SUFFIX_RE, re.I))
    return out


def _extract_timeframe_from_var(name: str, fallback: Optional[str]) -> Optional[str]:
    m = re.search(TIMEFRAME_RE, name); return m.group(0) if m else (fallback or None)
def _canon_term(s: str) -> str:
    s=re.sub(r"\s*\([^)]*\)\s*$","",s or "").strip()
    return re.sub(r"\s+","_",s.lower())

def _render(tpl: str, e: str, t: Optional[str]) -> str:
    s = tpl.replace("{e}", e or "")
    # strip "_{t}" / "{t}" and collapse underscores
    s = re.sub(r"_\{t\}", "", s)
    s = s.replace("{t}", "")
    return re.sub(r"__+", "_", s).strip("_")


def snowstorm_get_concept(cid: str, timeout: float = 100.0) -> Optional[Dict[str, Any]]:
    try:
        r=requests.get(f"{SNOMED_BASE.rstrip('/')}/browser/{SNOMED_BRANCH}/concepts/{cid}", timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return None

def _is_procedure_target(tgt: Dict[str, Any]) -> bool:
    fsn = ((tgt or {}).get("fsn") or {}).get("term") or ""
    return "(procedure)" in fsn.lower()

def collect_proc_relations(doc: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    wanted = {CID_DURING, CID_AFTER, CID_DUE_TO, CID_ASSOCIATED_WITH, CID_TEMPORALLY_RELATED}
    out: List[Tuple[str,str,str]]=[]
    for rel in (doc or {}).get("relationships") or []:
        if not rel.get("active", True): continue
        rtype = ((rel.get("type") or {}).get("conceptId") or "").strip()
        if rtype not in wanted: continue
        tgt = rel.get("target") or {}
        if not _is_procedure_target(tgt): continue
        proc_id = (tgt.get("conceptId") or "").strip()
        term = ((tgt.get("pt") or {}) or {}).get("term") or (tgt.get("fsn") or {}).get("term") or ""
        out.append((rtype, proc_id, _canon_term(term)))
    return out

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
    stems = FINDING_STEMS_ALL if side=="inclusion" else FINDING_STEMS_EXC
    rx_list = _compile_finding_regexes(stems)
    disabled_attrs = EXCLUSION_DISABLED_ATTRS if side=="exclusion" else set()

    for pid in patients:
        pin  = os.path.join(in_root, pid)
        pout = os.path.join(out_root, pid, side); _ensure_dir(pout)

        # locate input (prefer annotated jsonl)
        def find_input(patient_dir: str, preferred: str) -> Optional[str]:
            p=os.path.join(patient_dir, preferred)
            if os.path.isfile(p): return p
            for c in ["canonical.enriched.annotated.jsonl",
                      "canonical.enriched.jsonl",
                      "canonical.jsonl",
                      "canonical.enriched.healed.jsonl"]:
                q=os.path.join(patient_dir,c)
                if os.path.isfile(q): return q
            hits=sorted(glob(os.path.join(patient_dir,"*.jsonl"))); return hits[0] if hits else None

        inp=find_input(pin, input_filename)
        if not inp:
            print(f"[{side:^9}] {pid:<24} skipped (no input jsonl found; tried '{input_filename}')")
            continue

        out_annot   = os.path.join(pout,"finding_to_procedure_via_relations.annotated.jsonl")
        out_pervar  = os.path.join(pout,"finding_to_procedure_via_relations.per_var.jsonl")
        out_flat_tmp= os.path.join(pout,"finding_to_procedure_via_relations.flat.jsonl.tmp")
        out_flat    = os.path.join(pout,"finding_to_procedure_via_relations.flat.json")
        out_struct  = os.path.join(pout,"finding_to_procedure_via_relations.structured.json")
        out_rep     = os.path.join(pout,"finding_to_procedure_via_relations.report.json")

        for p in (out_annot,out_pervar,out_flat_tmp): open(p,"w",encoding="utf-8").close()

        c_rows=c_not_match=c_not_true=c_no_rel=0
        c_rows_with_prod=c_prod=0

        seen_db = os.path.join(pout, ".tmp", "seen.sqlite")
        if os.path.exists(seen_db):
            os.remove(seen_db)
        conn = _spill_open(seen_db); cur = conn.cursor()

        with open(inp,"r",encoding="utf-8") as f, \
             open(out_annot,"a",encoding="utf-8") as fa, \
             open(out_pervar,"a",encoding="utf-8") as fpv, \
             open(out_flat_tmp,"a",encoding="utf-8") as fft:

            for ln in f:
                s=ln.strip()
                if not s: continue
                try: row=json.loads(s)
                except Exception: continue

                c_rows+=1
                time_ctx = _time_ctx(row)

                var = str(row.get("entity_variable_name") or "")
                exb = _boolish(row.get("extracted_value"))
                timeframe = _extract_timeframe_from_var(var, str(row.get("timeframe") or ""))

                matched=None
                for rx in rx_list:
                    m=rx.match(var)
                    if m: matched=m; break
                if matched is None:
                    c_not_match+=1
                    fa.write(json.dumps({**row,"finding_to_procedure_via_relations":{
                        "side":side,"eligible":False,"produced":[],"reason":"not_matching_template"
                    }}, ensure_ascii=False)+"\n")
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": row.get("conceptId"),
                        "var_name": var,"timeframe": timeframe,"converted": 0,"reason": "not_matching_template",
                        "procedure": "","attr_used": "","produced_names": ""
                    }, ensure_ascii=False)+"\n")
                    continue

                if exb is not True:
                    c_not_true+=1
                    fa.write(json.dumps({**row,"finding_to_procedure_via_relations":{
                        "side":side,"eligible":False,"produced":[],"reason":"extracted_value_not_true"
                    }}, ensure_ascii=False)+"\n")
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": row.get("conceptId"),
                        "var_name": var,"timeframe": timeframe,"converted": 0,"reason": "extracted_value_not_true",
                        "procedure": "","attr_used": "","produced_names": ""
                    }, ensure_ascii=False)+"\n")
                    continue

                cid = str(row.get("conceptId") or "").strip()
                if not cid:
                    fa.write(json.dumps({**row,"finding_to_procedure_via_relations":{
                        "side":side,"eligible":True,"produced":[],"reason":"no_conceptId"
                    }}, ensure_ascii=False)+"\n")
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": "",
                        "var_name": var,"timeframe": timeframe,"converted": 0,"reason": "no_conceptId",
                        "procedure": "","attr_used": "","produced_names": ""
                    }, ensure_ascii=False)+"\n")
                    continue

                doc=snowstorm_get_concept(cid)
                if doc is None:
                    fa.write(json.dumps({**row,"finding_to_procedure_via_relations":{
                        "side":side,"eligible":True,"produced":[],"reason":"snomed_fetch_failed"
                    }}, ensure_ascii=False)+"\n")
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": cid,
                        "var_name": var,"timeframe": timeframe,"converted": 0,"reason": "snomed_fetch_failed",
                        "procedure": "","attr_used": "","produced_names": ""
                    }, ensure_ascii=False)+"\n")
                    continue

                rels = [r for r in collect_proc_relations(doc) if r[0] not in disabled_attrs]
                if not rels:
                    c_no_rel+=1
                    fa.write(json.dumps({**row,"finding_to_procedure_via_relations":{
                        "side":side,"eligible":True,"produced":[],"reason":"no_target_procedure_relations"
                    }}, ensure_ascii=False)+"\n")
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": cid,
                        "var_name": var,"timeframe": timeframe,"converted": 0,"reason": "no_target_procedure_relations",
                        "procedure": "","attr_used": "","produced_names": ""
                    }, ensure_ascii=False)+"\n")
                    continue

                produced_names=[]; produced_items=[]
                src_term = row.get("preferred_term") or row.get("fully_specified_name") or ""
                tf_use = timeframe or (matched.group("t") if matched else None)

                for attr_cid, proc_id, proc_canon in rels:
                    # 统一不带 timeframe 的变量名
                    name = _render(UNDERGONE_TPL, proc_canon, None)

                    # 用 “名字 + 四个时间键” 去重（同名不同时间窗并存）
                    key = _dedupe_key(name, time_ctx)
                    cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                    if cur.rowcount == 1:
                        produced_names.append(name)
                        produced_items.append({
                            "name": name,
                            "procedureCanonical": proc_canon,
                            "procedureId": proc_id,
                            "attrId": attr_cid,
                            "attrClass": ATTR_CLASS_MAP.get(attr_cid, "finding_to_procedure_other"),
                            "source_variable_name": var,
                            "source_conceptId": cid,
                            "target_variable_name": name,
                            "derivation_stage": DERIVATION_STAGE,
                            "derivation_rule": RELATION_LABEL.get(attr_cid,"other"),
                            "relation": RELATION_LABEL.get(attr_cid,"other"),
                        })
                        # flat row（附上四个时间键；不再写 timeframe）
                        rec = {
                            "patient_id": pid,
                            "class": ATTR_CLASS_MAP.get(attr_cid, "finding_to_procedure_other"),
                            "new_variable_name": name,
                            "target_variable_name": name,
                            "source_variable_name": var,
                            "source_conceptId": cid,
                            "derivation_stage": DERIVATION_STAGE,
                            "derivation_rule": RELATION_LABEL.get(attr_cid,"other"),
                            "relation": RELATION_LABEL.get(attr_cid,"other"),
                            "derived_from_variable": var,
                            "derived_from_conceptId": cid,
                            "derived_from_entity_term": src_term,
                            "derived_conceptId": proc_id,
                            "derived_entity_term": proc_canon,
                            "fact_id": row.get("fact_id"),
                            "attrId": attr_cid
                        }
                        rec.update(time_ctx)
                        fft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # else: 同名同时间窗已存在，跳过写出



                if produced_names:
                    c_rows_with_prod+=1
                    c_prod+=len(set(produced_names))

                fa.write(json.dumps({**row,"finding_to_procedure_via_relations":{
                    "side":side,"eligible":True,"produced":produced_items,"reason":"ok" if produced_names else "no_output_after_filter"
                }}, ensure_ascii=False)+"\n")

                fpv.write(json.dumps({
                    "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": cid,
                    "var_name": var,"timeframe": tf_use,"converted": 1 if produced_names else 0,
                    "reason": "ok" if produced_names else "no_output_after_filter",
                    "procedure": ",".join(sorted({p["procedureCanonical"] for p in produced_items})) if produced_items else "",
                    "attr_used": ",".join(sorted({p["attrClass"] for p in produced_items})) if produced_items else "",
                    "produced_names": ";".join(sorted(set(produced_names))) if produced_names else "",
                }, ensure_ascii=False)+"\n")

        conn.commit(); conn.close()

        summary={
            "rows": c_rows,
            "rows_with_produced": c_rows_with_prod,
            "produced_variables": c_prod,
            "rows_not_matching_template": c_not_match,
            "rows_extracted_value_not_true": c_not_true,
            "rows_missing_target_relations": c_no_rel,
        }
        with open(out_struct,"w",encoding="utf-8") as fs: json.dump({"patient_id":pid,"side":side,"summary":summary}, fs, ensure_ascii=False, indent=2)
        with open(out_rep,"w",encoding="utf-8") as fr: json.dump({"patient_id":pid,"side":side,"counts":summary}, fr, ensure_ascii=False, indent=2)

        # finalize array from tmp
        first=True
        with open(out_flat,"w",encoding="utf-8") as fout, open(out_flat_tmp,"r",encoding="utf-8") as fin:
            fout.write("[\n")
            for line in fin:
                s=line.strip()
                if not s: continue
                if not first: fout.write(",\n")
                fout.write(s); first=False
            fout.write("\n]\n")

        print(f"[{side:^9}] {pid:<24} ok (produced={summary['produced_variables']}, missing_target_relations={summary['rows_missing_target_relations']})")

# ───────── CLI ─────────
def main():
    ap=argparse.ArgumentParser(description="Finding → Procedure via relations (memory-safe streaming).")
    ap.add_argument("--patient", action="append", help="Only process this patient id (folder).")
    ap.add_argument("--input-filename", default="canonical.enriched.annotated.jsonl", help="Preferred input filename.")
    ap.add_argument("--side", choices=["inclusion","exclusion","both"], default="inclusion")
    ap.add_argument(
        "--in-root",
        dest="in_root",
        required=False,
        default=None,
        help="Input root that contains per-patient dirs. Required unless F2P_REL_INPUT_ROOT is set.",
    )
    ap.add_argument(
        "--out-root",
        dest="out_root",
        required=False,
        default=None,
        help="Output root to write results. Required unless F2P_REL_OUTPUT_ROOT is set.",
    )
    args=ap.parse_args()

    # 解析 in_root / out_root：优先命令行，其次环境变量，缺失则报错
    if args.in_root:
        in_root = os.path.abspath(args.in_root)
    elif IN_ROOT:
        in_root = os.path.abspath(IN_ROOT)
    else:
        print("[f2p_rel] --in-root is required unless F2P_REL_INPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if args.out_root:
        out_root = os.path.abspath(args.out_root)
    elif OUT_ROOT:
        out_root = os.path.abspath(OUT_ROOT)
    else:
        print("[f2p_rel] --out-root is required unless F2P_REL_OUTPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(in_root):
        print(f"[err] IN_ROOT not found: {in_root}", file=sys.stderr); sys.exit(2)
    _ensure_dir(out_root)

    all_entries=sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root,d))])
    patients = all_entries if not args.patient else [e for e in all_entries if e in set(args.patient)]
    if not patients:
        print(f"[warn] no matching patients under {in_root}"); return

    if args.side in ("inclusion","exclusion"):
        process_side(in_root, out_root, args.side, patients, args.input_filename)
    else:
        process_side(in_root, out_root, "inclusion", patients, args.input_filename)
        process_side(in_root, out_root, "exclusion", patients, args.input_filename)

    print("[done] all requested sides processed.")

if __name__ == "__main__":
    main()
