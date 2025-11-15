#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_findings_to_observable_entity.py — streaming
────────────────────────────────────────────────────────────────────────
From TRUE Finding variables, derive Observable-Entity status variables:

  patients_{entity}_is_{normal|abnormal|positive|negative}_{timeframe}

Inputs per side (prefer jsonl):
  • finding_to_procedure_enriched.annotated.jsonl  (or fallback *enriched*.jsonl / *.jsonl)

Outputs per side:
  • finding_to_observable_entity_enriched.annotated.jsonl
  • finding_to_observable_entity_enrichment.structured.json  (SMALL summary)
  • finding_to_observable_entity_enrichment.report.json
  • finding_to_observable_entity_enrichment.per_var.jsonl
  • finding_to_observable_entity_enrichment.flat.jsonl.tmp   (pre-dedupe stream)
  • finding_to_observable_entity_enrichment.flat.json        (FINAL array, deduped)
"""

from __future__ import annotations
import os, sys, json, csv, re, argparse, sqlite3
from typing import Dict, Any, List, Optional, Tuple
from glob import glob
import requests

# ---------- on-disk spill set ----------
def _spill_open(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY)")
    conn.commit()
    return conn

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

def _dedupe_key(name: str, time_ctx: Dict[str, Any]) -> str:
    """Make a unique key using name + 4 time-window fields; round hours to 3 decimals."""
    def r(v):
        if v is None: return "null"
        try:
            return f"{round(float(v), 3):.3f}"
        except Exception:
            return str(v)
    def b(v): return "1" if bool(v) else "0"
    return "|".join([
        name,
        r(time_ctx.get("start_time_in_hours")),
        r(time_ctx.get("end_time_in_hours")),
        b(time_ctx.get("start_time_inclusive")),
        b(time_ctx.get("end_time_inclusive")),
    ])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# IN_ROOT / OUT_ROOT：只从环境变量读，正常使用应通过 --in-root / --out-root 显式传入
IN_ROOT  = os.environ.get("F2OE_INPUT_ROOT", "")
OUT_ROOT = os.environ.get("F2OE_OUTPUT_ROOT", "")

SNOMED_BASE   = os.getenv("SNOMED_BASE", "http://localhost:8080")
SNOMED_BRANCH = os.getenv("SNOMED_BRANCH", "MAIN")

MAPPING_CSV   = os.getenv(
    "FINDING_TO_OBS_MAPPING_CSV",
    os.path.join(SCRIPT_DIR, "../rules/mapping/finding_interprets_observable_entity_has_interpretation_qualifier_value_mapping.csv"),
).strip()

CID_INTERPRETS         = "363714003"
CID_HAS_INTERPRETATION = "363713009"

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

OBS_STATUS_TPL = {
    "normal":   "patients_{e}_is_normal_{t}",
    "abnormal": "patients_{e}_is_abnormal_{t}",
    "positive": "patients_{e}_is_positive_{t}",
    "negative": "patients_{e}_is_negative_{t}",
}

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool): return v
    if isinstance(v, str):
        s=v.strip().lower()
        if s in ("true","false"): return s=="true"
    return None

def _compile_finding_regexes(stems: List[str]) -> List[re.Pattern]:
    """
    Compile regexes where {t} is OPTIONAL. If stem contains '_{t}', make that
    whole segment optional. {e} 仍然是贪婪匹配。
    """
    out = []
    for stem in stems:
        esc = re.escape(stem)
        if r"_\{t\}" in esc:
            esc = esc.replace(r"_\{t\}", rf"(?:_(?P<t>{TIMEFRAME_RE}))?")
        else:
            esc = esc.replace(r"\{t\}",  rf"(?P<t>{TIMEFRAME_RE})?")
        esc = esc.replace(r"\{e\}", r"(?P<e>.+)")
        out.append(re.compile(r"^(?:patient_)?"+esc+QUAL_SUFFIX_RE, re.I))
    return out


def _extract_timeframe_from_var(name: str, fallback: Optional[str]) -> Optional[str]:
    m = re.search(TIMEFRAME_RE, name); return m.group(0) if m else (fallback or None)

def _render_status_name(tpl: str, e: str, t: Optional[str]) -> str:
    s = tpl.replace("{e}", e or "")
    s = re.sub(r"_\{t\}", "", s)   # 去掉 "_{t}"
    s = s.replace("{t}", "")       # 去掉裸 "{t}"
    s = re.sub(r"__+", "_", s).strip("_")  # 折叠下划线
    return s


def _canonicalize_oe_term(term: str) -> str:
    term = re.sub(r"\s*\([^)]*\)\s*$","", term or "").strip()
    return re.sub(r"\s+","_", term.lower())

def load_interpretation_mapping(csv_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(csv_path):
        print(f"[err] mapping CSV not found: {csv_path}", file=sys.stderr); sys.exit(2)
    m: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            hid = (row.get("hasInterpretationId") or "").strip()
            if not hid: continue
            def flag(col: str) -> int:
                v=(row.get(col) or "").strip()
                if v=="": return 0
                try: return 1 if int(float(v))!=0 else 0
                except Exception: return 1 if v.lower() in ("y","yes","true","t") else 0
            entry = m.setdefault(hid, {
                "hasInterpretationTerm": row.get("hasInterpretationTerm") or "",
                "inc": {"normal":0,"abnormal":0,"positive":0,"negative":0},
                "exc": {"normal":0,"abnormal":0,"positive":0,"negative":0},
            })
            for k in ("normal","abnormal","positive","negative"):
                entry["inc"][k] = max(entry["inc"][k], flag(f"{k}(inc)"))
                entry["exc"][k] = max(entry["exc"][k], flag(f"{k}(exc)"))
    return m

_sess = requests.Session()
def snowstorm_get_concept(cid: str, timeout: float = 100.0) -> Optional[Dict[str, Any]]:
    try:
        r=_sess.get(f"{SNOMED_BASE.rstrip('/')}/browser/{SNOMED_BRANCH}/concepts/{cid}", timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return None

def pick_interprets_oe(doc: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    for rel in (doc or {}).get("relationships") or []:
        if not rel.get("active", True): continue
        if (rel.get("type") or {}).get("conceptId") != CID_INTERPRETS: continue
        tgt = rel.get("target") or {}
        fsn = ((tgt.get("fsn") or {}) or {}).get("term") or ""
        pt  = ((tgt.get("pt")  or {}) or {}).get("term") or ""
        if "(observable entity)" in fsn.lower():
            name = pt or fsn
            return tgt.get("conceptId") or "", _canonicalize_oe_term(name)
    return None

def pick_has_interpretation(doc: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    for rel in (doc or {}).get("relationships") or []:
        if not rel.get("active", True): continue
        if (rel.get("type") or {}).get("conceptId") != CID_HAS_INTERPRETATION: continue
        tgt = rel.get("target") or {}
        term = ((tgt.get("pt") or {}) or {}).get("term") or (tgt.get("fsn") or {}).get("term") or ""
        return (tgt.get("conceptId") or "", term)
    return None

class SnowCache:
    def __init__(self): self.mem: Dict[str, Dict[str, Any]] = {}
    def get(self,k:str)->Optional[Dict[str,Any]]: return self.mem.get(k)
    def put(self,k:str,v:Dict[str,Any])->None: self.mem[k]=v

def find_input_file(patient_dir: str, preferred: str) -> Optional[str]:
    p=os.path.join(patient_dir, preferred)
    if os.path.isfile(p): return p
    for c in ["finding_to_procedure_enriched.annotated.jsonl","canonical.enriched.annotated.jsonl",
              "canonical.enriched.jsonl","canonical.jsonl","canonical.enriched.healed.jsonl"]:
        q=os.path.join(patient_dir,c)
        if os.path.isfile(q): return q
    hits=sorted(glob(os.path.join(patient_dir,"*.jsonl")))
    return hits[0] if hits else None

def _collect_candidate_vars_for_side(row: Dict[str, Any], rx_list: List[re.Pattern]) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    def _maybe(name: str):
        name = (name or "").strip()
        if not name: return
        for rx in rx_list:
            if rx.match(name):
                out.append((name, None))  # timeframe 不再使用
                break
    if _boolish(row.get("extracted_value")) is True:
        _maybe(str(row.get("entity_variable_name") or ""))
    imp = (row.get("implications") or {})
    for si in (imp.get("schema_new_only") or []):   _maybe(str(si.get("name") or ""))
    for ti in (imp.get("timeframe_new_only") or []):_maybe(str(ti.get("name") or ""))
    seen, uniq = set(), []
    for v, tf in out:
        if v in seen: continue
        seen.add(v); uniq.append((v, tf))
    return uniq


DERIVATION_STAGE = "f2oe"
DERIVATION_RULE  = "interprets(363714003)+hasInterpretation(363713009)"

def process_side(in_root: str, out_root: str, side: str, patients: List[str], input_filename: str) -> None:
    if not MAPPING_CSV:
        print("[err] FINDING_TO_OBS_MAPPING_CSV not set in environment.", file=sys.stderr); sys.exit(2)
    mapping = load_interpretation_mapping(MAPPING_CSV)

    stems = FINDING_STEMS_ALL if side=="inclusion" else FINDING_STEMS_EXC
    rx_list = _compile_finding_regexes(stems)
    cache = SnowCache()

    for pid in patients:
        # 输入是在 in_root/<patient>/<side> 下
        pin = os.path.join(in_root, pid, side)
        # 输出是在 out_root/<patient>/<side> 下
        pout= os.path.join(out_root, pid, side); _ensure_dir(pout)

        inp = find_input_file(pin, input_filename)
        if not inp:
            print(f"[{side:^9}] {pid:<24} skipped (no input jsonl found; tried '{input_filename}')")
            continue

        out_annot   = os.path.join(pout,"finding_to_observable_entity_enriched.annotated.jsonl")
        out_pervar  = os.path.join(pout,"finding_to_observable_entity_enrichment.per_var.jsonl")
        out_flat_tmp= os.path.join(pout,"finding_to_observable_entity_enrichment.flat.jsonl.tmp")
        out_flat    = os.path.join(pout,"finding_to_observable_entity_enrichment.flat.json")
        out_struct  = os.path.join(pout,"finding_to_observable_entity_enrichment.structured.json")
        out_rep     = os.path.join(pout,"finding_to_observable_entity_enrichment.report.json")

        for p in (out_annot,out_pervar,out_flat_tmp): open(p,"w",encoding="utf-8").close()

        c_rows=c_rows_with_prod=c_prod=0
        c_not_match=c_not_true=c_no_interp=c_no_hid=0

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
                time_ctx = {
                    "start_time_in_hours": row.get("start_time_in_hours"),
                    "end_time_in_hours": row.get("end_time_in_hours"),
                    "start_time_inclusive": row.get("start_time_inclusive"),
                    "end_time_inclusive": row.get("end_time_inclusive"),
                }
                candidates = _collect_candidate_vars_for_side(row, rx_list)

                if not candidates:
                    base_var = str(row.get("entity_variable_name") or "")
                    timeframe = _extract_timeframe_from_var(base_var, str(row.get("timeframe") or ""))
                    if any(rx.match(base_var) for rx in rx_list):
                        if _boolish(row.get("extracted_value")) is not True:
                            reason="extracted_value_not_true"; c_not_true+=1
                        else:
                            reason="no_candidates"
                    else:
                        reason="not_matching_template"; c_not_match+=1

                    fa.write(json.dumps({**row,"finding_to_observable_entity_enrichment":{
                        "side":side,"eligible":False,"produced":[],"conflict_flags":[],"reason":reason
                    }}, ensure_ascii=False)+"\n")
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": row.get("conceptId"),
                        "var_name": base_var,"timeframe": timeframe,"converted": 0,"reason": reason,
                        "interprets_entity": "","hasInterpretationId": "","hasInterpretationTerm": "",
                        "produced_count": 0,"produced_names": "","produced_statuses": ""
                    }, ensure_ascii=False)+"\n")
                    continue

                cid=str(row.get("conceptId") or "").strip()
                if not cid:
                    for var_name, tf in candidates:
                        fa.write(json.dumps({**row,"finding_to_observable_entity_enrichment":{
                            "side":side,"eligible":True,"produced":[],"conflict_flags":[],"reason":"no_conceptId"
                        }}, ensure_ascii=False)+"\n")
                        fpv.write(json.dumps({
                            "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": "",
                            "var_name": var_name,"timeframe": tf,"converted": 0,"reason":"no_conceptId",
                            "interprets_entity": "","hasInterpretationId": "","hasInterpretationTerm": "",
                            "produced_count": 0,"produced_names": "","produced_statuses": ""
                        }, ensure_ascii=False)+"\n")
                    continue

                doc = cache.get(cid) or snowstorm_get_concept(cid)
                if doc is None:
                    for var_name, tf in candidates:
                        fa.write(json.dumps({**row,"finding_to_observable_entity_enrichment":{
                            "side":side,"eligible":True,"produced":[],"conflict_flags":[],"reason":"snomed_fetch_failed"
                        }}, ensure_ascii=False)+"\n")
                        fpv.write(json.dumps({
                            "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": cid,
                            "var_name": var_name,"timeframe": tf,"converted": 0,"reason":"snomed_fetch_failed",
                            "interprets_entity": "","hasInterpretationId": "","hasInterpretationTerm": "",
                            "produced_count": 0,"produced_names": "","produced_statuses": ""
                        }, ensure_ascii=False)+"\n")
                    continue
                cache.put(cid, doc)

                produced_names_row: List[str]=[]
                produced_items_row: List[Dict[str,Any]]=[]

                for var_name, tf in candidates:
                    oe = pick_interprets_oe(doc)
                    hi = pick_has_interpretation(doc)

                    if not oe:
                        c_no_interp+=1
                        fpv.write(json.dumps({
                            "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": row.get("conceptId"),
                            "var_name": var_name,"timeframe": tf,"converted": 0,"reason":"no_interprets_to_observable_entity",
                            "interprets_entity": "","hasInterpretationId": "","hasInterpretationTerm": "",
                            "produced_count": 0,"produced_names": "","produced_statuses": ""
                        }, ensure_ascii=False)+"\n")
                        continue

                    oe_id, oe_canon = oe
                    if not hi or hi[0] not in mapping:
                        c_no_hid+=1
                        fpv.write(json.dumps({
                            "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": row.get("conceptId"),
                            "var_name": var_name,"timeframe": tf,"converted": 0,"reason":"no_hasInterpretationId_or_not_in_mapping",
                            "interprets_entity": oe_canon,"hasInterpretationId": hi[0] if hi else "","hasInterpretationTerm": hi[1] if hi else "",
                            "produced_count": 0,"produced_names": "","produced_statuses": ""
                        }, ensure_ascii=False)+"\n")
                        continue

                    flags = mapping[hi[0]]["inc" if side=="inclusion" else "exc"]
                    tf_use = tf
                    statuses_produced: List[str] = []

                    for status in ("normal","abnormal","positive","negative"):
                        if not int(flags.get(status,0)):  # 简单起见，只按 mapping 的 0/1 来
                            continue

                        new_name = _render_status_name(OBS_STATUS_TPL[status], oe_canon, None)

                        key = _dedupe_key(new_name, time_ctx)
                        cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                        if cur.rowcount == 1:
                            produced_names_row.append(new_name)
                            statuses_produced.append(status)
                            produced_items_row.append({
                                "name": new_name,
                                "status": status,
                                "observableEntityCanonical": oe_canon,
                                "observableEntityId": oe_id,
                                "hasInterpretationId": hi[0],
                                "hasInterpretationTerm": hi[1] or mapping[hi[0]]["hasInterpretationTerm"],
                            })
                            rec = {
                                "patient_id": pid,
                                "class": "finding_to_observable_entity",
                                "new_variable_name": new_name,
                                "target_variable_name": new_name,
                                "source_variable_name": var_name,
                                "source_conceptId": row.get("conceptId"),
                                "derivation_stage": DERIVATION_STAGE,
                                "derivation_rule": DERIVATION_RULE,
                                "derived_from_variable": var_name,
                                "derived_from_conceptId": row.get("conceptId"),
                                "derived_conceptId": oe_id,
                                "derived_entity_term": oe_canon,
                                "derived_qualifierId": hi[0],
                                "derived_qualifier_term": hi[1] or mapping[hi[0]]["hasInterpretationTerm"],
                                "fact_id": row.get("fact_id"),
                            }
                            rec.update(time_ctx)
                            fft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        # else: 同名同时间窗已存在，跳过

                    # 写 per-var 记录
                    fpv.write(json.dumps({
                        "patient_id": pid,"fact_id": row.get("fact_id"),"conceptId": row.get("conceptId"),
                        "var_name": var_name,"timeframe": tf_use,"converted": 1 if statuses_produced else 0,
                        "reason": "ok" if statuses_produced else "no_active_flags_for_context",
                        "interprets_entity": oe_canon,
                        "hasInterpretationId": hi[0],
                        "hasInterpretationTerm": hi[1] or mapping[hi[0]]["hasInterpretationTerm"],
                        "produced_count": len(set(statuses_produced)),
                        "produced_names": ";".join(sorted(set(produced_names_row))) if produced_names_row else "",
                        "produced_statuses": ";".join(sorted(set(statuses_produced))) if statuses_produced else "",
                    }, ensure_ascii=False)+"\n")

                if produced_names_row:
                    c_rows_with_prod+=1
                    c_prod+=len(set(produced_names_row))

                fa.write(json.dumps({**row,"finding_to_observable_entity_enrichment":{
                    "side": side,
                    "eligible": True,
                    "produced": produced_items_row,
                    "reason": "ok" if produced_items_row else "no_active_flags_for_context"
                }}, ensure_ascii=False)+"\n")

        conn.commit(); conn.close()

        summary={
            "rows": c_rows,
            "rows_with_produced": c_rows_with_prod,
            "produced_variables": c_prod,
            "rows_not_matching_template": c_not_match,
            "rows_extracted_value_not_true": c_not_true,
            "rows_missing_interprets": c_no_interp,
            "rows_missing_has_interpretation_in_mapping": c_no_hid,
        }
        with open(out_struct,"w",encoding="utf-8") as fs:
            json.dump({"patient_id":pid,"side":side,"summary":summary}, fs, ensure_ascii=False, indent=2)
        with open(out_rep,"w",encoding="utf-8") as fr:
            json.dump({"patient_id":pid,"side":side,"counts":summary}, fr, ensure_ascii=False, indent=2)

        # jsonl.tmp → array json
        first=True
        with open(out_flat,"w",encoding="utf-8") as fout, open(out_flat_tmp,"r",encoding="utf-8") as fin:
            fout.write("[\n")
            for line in fin:
                s=line.strip()
                if not s: continue
                if not first: fout.write(",\n")
                fout.write(s); first=False
            fout.write("\n]\n")

        print(f"[{side:^9}] {pid:<24} ok (produced={summary['produced_variables']}, missing_interprets={summary['rows_missing_interprets']})")

# ───────── CLI ─────────
def main():
    ap = argparse.ArgumentParser(description="Finding → Observable-Entity status enricher (memory-safe streaming).")
    ap.add_argument("--patient", action="append", help="Only process this patient id (folder name).")
    ap.add_argument("--input-filename", default="finding_to_procedure_enriched.annotated.jsonl",
                    help="Preferred per-patient input filename.")
    ap.add_argument("--side", choices=["inclusion","exclusion","both"], default="inclusion")
    ap.add_argument(
        "--in-root",
        dest="in_root",
        required=False,
        default=None,
        help="Input root that contains per-patient dirs. Required unless F2OE_INPUT_ROOT is set.",
    )
    ap.add_argument(
        "--out-root",
        dest="out_root",
        required=False,
        default=None,
        help="Output root to write results. Required unless F2OE_OUTPUT_ROOT is set.",
    )
    args = ap.parse_args()

    # 解析 in_root / out_root：优先命令行，其次环境变量，缺失就报错
    if args.in_root:
        in_root = os.path.abspath(args.in_root)
    elif IN_ROOT:
        in_root = os.path.abspath(IN_ROOT)
    else:
        print("[f2oe] --in-root is required unless F2OE_INPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if args.out_root:
        out_root = os.path.abspath(args.out_root)
    elif OUT_ROOT:
        out_root = os.path.abspath(OUT_ROOT)
    else:
        print("[f2oe] --out-root is required unless F2OE_OUTPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(in_root):
        print(f"[err] IN_ROOT not found: {in_root}", file=sys.stderr); sys.exit(2)
    _ensure_dir(out_root)

    all_patients=sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root,d))])
    patients = all_patients if not args.patient else [e for e in all_patients if e in set(args.patient)]
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
