#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_procedure_to_finding.py — streaming + on-disk dedupe
────────────────────────────────────────────────────────────────────────
From undergone-outcome Procedure variables derive normalized Finding variables.

Inputs per side (per patient folder under IN_ROOT):
  • finding_to_procedure_enriched.annotated.jsonl

Outputs per side (streamed):
  • procedure_to_finding_enriched.annotated.jsonl
  • procedure_to_finding_enrichment.structured.json  (SMALL summary)
  • procedure_to_finding_enrichment.report.json      (counts)
  • procedure_to_finding_enrichment.per_var.jsonl
  • procedure_to_finding_enrichment.flat.jsonl.tmp   (pre-dedupe stream)
  • procedure_to_finding_enrichment.flat.json        (FINAL array, deduped)
"""

from __future__ import annotations
import os, sys, json, csv, re, argparse
from typing import Dict, Any, List, Optional, Tuple
import requests, sqlite3
from urllib.parse import quote

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))

# IN_ROOT / OUT_ROOT：只从环境变量读。
# 正常使用时推荐通过 --in-root / --out-root 显式传入；
# 若命令行没给、环境变量也没设，则直接报错。
IN_ROOT        = os.environ.get("P2F_INPUT_ROOT", "")
OUT_ROOT       = os.environ.get("P2F_OUTPUT_ROOT", "")

SNOMED_BASE    = os.getenv("SNOMED_BASE", "http://localhost:8080")
SNOMED_BRANCH  = os.getenv("SNOMED_BRANCH", "MAIN")

MAPPING_CSV    = os.getenv(
    "FINDING_TO_OBS_MAPPING_CSV",
    os.path.join(SCRIPT_DIR, "../rules/mapping/finding_interprets_procedure_has_interpretation_qualifier_value_mapping.csv"),
).strip()

CID_INTERPRETS          = "363714003"
CID_HAS_INTERPRETATION  = "363713009"
CID_ISA                 = "116680003"
DEF_DEFINED_ID          = "900000000000073002"

DERIVATION_STAGE = "p2f"
DERIVATION_RULE  = "interprets(363714003)+hasInterpretation(363713009)"

PROC_OUTCOME_RX = re.compile(
    r"^patient_has_undergone_(?P<e>.+?)_outcome_is_(?P<status>positive|negative|normal|abnormal)(?:@@[a-z0-9_]+(?:@@[a-z0-9_]+)*)?$",
    re.I
)

# ---- Time-window aware dedupe key (NEW) ----
def _dedupe_key(name: str, time_ctx: Dict[str, Any]) -> str:
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
    if isinstance(v, str): return v.strip().lower()=="true" if v.strip().lower() in ("true","false") else None
    return None
def _canon_term(term: str) -> str:
    term = re.sub(r"\s*\([^)]*\)\s*$","", term or "").strip()
    return re.sub(r"\s+","_", term.lower())

def load_hid_mapping(csv_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
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
    return m  # type: ignore[return-value]

_sess = requests.Session()
def snowstorm_get_concept(cid: str, timeout: float = 100.0) -> Optional[Dict[str, Any]]:
    try:
        r = _sess.get(f"{SNOMED_BASE.rstrip('/')}/browser/{SNOMED_BRANCH}/concepts/{cid}", timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return None

def snowstorm_ecl_search(ecl: str, limit: int = 1000, timeout: float = 100.0) -> List[Dict[str, Any]]:
    try:
        r = _sess.get(f"{SNOMED_BASE.rstrip('/')}/{SNOMED_BRANCH}/concepts?ecl={quote(ecl)}&limit={limit}", timeout=timeout)
        if r.status_code==200:
            js = r.json() or {}
            return js.get("items") or js.get("concepts") or []
    except Exception:
        pass
    return []

def _active_rels(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in (doc or {}).get("relationships") or [] if r.get("active", True)]

def _same_group_non_isa(doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    groups: Dict[int, Dict[str, Any]] = {}
    for rel in _active_rels(doc):
        rtype = (rel.get("type") or {}).get("conceptId")
        if rtype == CID_ISA:  # skip isa
            continue
        gid = int(rel.get("groupId") or 0)
        g = groups.setdefault(gid, {"types": [], "rels": []})
        g["types"].append(rtype)
        g["rels"].append(rel)
    return groups

def _group_interprets_proc_and_his(group: Dict[str, Any], proc_id: str) -> List[Tuple[str, str]]:
    has_proc = False
    his: List[Tuple[str,str]] = []
    for rel in group["rels"]:
        if (rel.get("type") or {}).get("conceptId") == CID_INTERPRETS:
            tgt = rel.get("target") or {}
            if tgt.get("conceptId") == proc_id:
                has_proc = True
    if not has_proc: return []
    for rel in group["rels"]:
        if (rel.get("type") or {}).get("conceptId") == CID_HAS_INTERPRETATION:
            tgt = rel.get("target") or {}
            term = ((tgt.get("pt") or {}) or {}).get("term") or (tgt.get("fsn") or {}) or {}
            if isinstance(term, dict):
                term = term.get("term") or ""
            if not isinstance(term, str):
                term = ""
            his.append((tgt.get("conceptId") or "", term))
    return his

def _status_match(flags: Dict[str,int], want: str) -> bool:
    if int(flags.get(want,0))==1: return True
    if want in ("normal","abnormal"): return int(flags.get("normal",0))==1 and int(flags.get("abnormal",0))==1
    if want in ("positive","negative"): return int(flags.get("positive",0))==1 and int(flags.get("negative",0))==1
    return False

def _is_minimal_single_group(doc: Dict[str, Any], required_gid: int) -> bool:
    groups = _same_group_non_isa(doc)
    if set(groups.keys()) != {required_gid}: return False
    tys = set(groups[required_gid]["types"])
    return tys.issubset({CID_INTERPRETS, CID_HAS_INTERPRETATION}) and CID_INTERPRETS in tys and CID_HAS_INTERPRETATION in tys

def _is_fully_defined(doc: Dict[str, Any]) -> bool:
    if str(doc.get("definitionStatusId") or "") == DEF_DEFINED_ID: return True
    ds = doc.get("definitionStatus")
    if isinstance(ds, str) and ds.strip().lower().replace(" ","_") in ("fully_defined","sufficiently_defined","defined"): return True
    if isinstance(ds, dict):
        sid = str(ds.get("id") or ds.get("conceptId") or "")
        if sid == DEF_DEFINED_ID: return True
        term = ((ds.get("pt") or {}) or {}).get("term") or (ds.get("fsn") or {}) or {}
        if isinstance(term, dict):
            term = term.get("term") or ""
        if isinstance(term, str) and "fully defined" in term.lower(): return True
    return False

def _spill_open(dbp: str) -> sqlite3.Connection:
    _ensure_dir(os.path.dirname(dbp))
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY)")
    conn.commit()
    return conn

def find_input_file(patient_side_dir: str, preferred: str) -> Tuple[str, str]:
    """
    Force using annotated JSONL:
      <in_root>/<patient>/<side>/finding_to_procedure_enriched.annotated.jsonl
    Returns ("jsonl", path) if exists, else ("", "").
    """
    p = os.path.join(patient_side_dir, "finding_to_procedure_enriched.annotated.jsonl")
    if os.path.isfile(p):
        return ("jsonl", p)
    return ("", "")


def process_side(
    in_root: str,
    out_root: str,
    side: str,
    patients: List[str],
    input_filename: str,
) -> None:
    if not MAPPING_CSV:
        print(f"[err] mapping CSV not found: {MAPPING_CSV}", file=sys.stderr); sys.exit(2)
    mapping = load_hid_mapping(MAPPING_CSV)

    for pid in patients:
        pin  = os.path.join(in_root, pid, side)
        pout = os.path.join(out_root, pid, side); _ensure_dir(pout)

        mode, inp = find_input_file(pin, input_filename)
        if not inp:
            print(f"[{side:^9}] {pid:<24} skipped (no input found; tried '{input_filename}')")
            continue

        out_annot   = os.path.join(pout, "procedure_to_finding_enriched.annotated.jsonl")
        out_pervar  = os.path.join(pout, "procedure_to_finding_enrichment.per_var.jsonl")
        out_flat_tmp= os.path.join(pout, "procedure_to_finding_enrichment.flat.jsonl.tmp")
        out_flat    = os.path.join(pout, "procedure_to_finding_enrichment.flat.json")
        out_struct  = os.path.join(pout, "procedure_to_finding_enrichment.structured.json")
        out_rep     = os.path.join(pout, "procedure_to_finding_enrichment.report.json")

        for p in (out_annot, out_pervar, out_flat_tmp):
            open(p, "w", encoding="utf-8").close()

        c_rows = c_rows_with_prod = c_prod = 0
        c_not_match = c_missing_pid = c_no_cands = c_filtered_min = c_filtered_def = 0

        seen_db = os.path.join(pout, ".tmp", "seen.sqlite")
        if os.path.exists(seen_db):
            os.remove(seen_db)
        conn = _spill_open(seen_db); cur = conn.cursor()

        # choose an iterator
        def _iter_rows():
            if mode == "jsonl":
                with open(inp, "r", encoding="utf-8") as f:
                    for ln in f:
                        s=ln.strip()
                        if s:
                            try: yield json.loads(s)
                            except Exception: continue
            else:  # flat_json (array) — LAST resort; may be heavy
                try:
                    arr = json.load(open(inp,"r",encoding="utf-8")) or []
                    for it in arr:
                        if isinstance(it, dict): yield it
                except Exception:
                    return

        with open(out_annot, "a", encoding="utf-8") as fa, \
             open(out_pervar, "a", encoding="utf-8") as fpv, \
             open(out_flat_tmp, "a", encoding="utf-8") as fft:

            for row in _iter_rows():
                c_rows += 1
                time_ctx = {
                    "start_time_in_hours": row.get("start_time_in_hours"),
                    "end_time_in_hours": row.get("end_time_in_hours"),
                    "start_time_inclusive": row.get("start_time_inclusive"),
                    "end_time_inclusive": row.get("end_time_inclusive"),
                }

                var = str(row.get("new_variable_name") or row.get("entity_variable_name") or row.get("var_name") or "")
                if not var:
                    continue
                m = PROC_OUTCOME_RX.match(var)
                if not m:
                    c_not_match += 1
                    fa.write(json.dumps({**row, "procedure_to_finding_enrichment": {
                        "side": side, "eligible": False, "produced": [], "reason": "not_matching_undergone_outcome_template"
                    }}, ensure_ascii=False) + "\n")
                    continue

                proc_canon = m.group("e")
                status     = m.group("status")
                tf         = None  # timeframe 不再使用
                proc_id = str(row.get("derived_conceptId") or row.get("procedureId") or "").strip()
                if not proc_id:
                    c_missing_pid += 1
                    fa.write(json.dumps({**row, "procedure_to_finding_enrichment": {
                        "side": side, "eligible": True, "produced": [], "reason": "missing_procedure_id"
                    }}, ensure_ascii=False) + "\n")
                    fpv.write(json.dumps({
                        "patient_id": row.get("patient_id") or "",
                        "var_name": var, "timeframe": tf,
                        "converted": 0, "reason": "missing_procedure_id",
                        "procedure_canonical": proc_canon, "procedure_id": "",
                        "produced_count": 0, "produced_names": ""
                    }, ensure_ascii=False) + "\n")
                    continue

                ecl = f"<<404684003 : {{ {CID_INTERPRETS} = <<{proc_id} , {CID_HAS_INTERPRETATION} = * }}"
                candidates = snowstorm_ecl_search(ecl, limit=2000)
                if not candidates:
                    c_no_cands += 1
                    fa.write(json.dumps({**row, "procedure_to_finding_enrichment": {
                        "side": side, "eligible": True, "produced": [], "reason": "no_candidate_findings"
                    }}, ensure_ascii=False) + "\n")
                    fpv.write(json.dumps({
                        "patient_id": row.get("patient_id") or "",
                        "var_name": var, "timeframe": tf,
                        "converted": 0, "reason": "no_candidate_findings",
                        "procedure_canonical": proc_canon, "procedure_id": proc_id,
                        "produced_count": 0, "produced_names": ""
                    }, ensure_ascii=False) + "\n")
                    continue

                produced_names: List[str] = []
                produced_items: List[Dict[str, Any]] = []
                filtered_min = filtered_def = 0

                for cand in candidates:
                    fid = str(cand.get("conceptId") or "")
                    if not fid: continue
                    fdoc = snowstorm_get_concept(fid)
                    if not fdoc: continue
                    groups = _same_group_non_isa(fdoc)
                    ok_any = False
                    for gid, g in groups.items():
                        his = _group_interprets_proc_and_his(g, proc_id)
                        if not his: continue
                        chosen_hid = None; chosen_term = ""
                        for hid, hterm in his:
                            flags = mapping.get(hid)
                            if not flags: continue
                            use = flags["inc" if side=="inclusion" else "exc"]
                            if _status_match(use, status):
                                chosen_hid = hid; chosen_term = hterm or flags["hasInterpretationTerm"]; break
                        if not chosen_hid: continue
                        if not _is_minimal_single_group(fdoc, gid):
                            filtered_min += 1; continue
                        if not _is_fully_defined(fdoc):
                            filtered_def += 1; continue

                        f_pt = ((fdoc.get("pt") or {}) or {}).get("term") or (fdoc.get("fsn") or {}) or {}
                        if isinstance(f_pt, dict):
                            f_pt = f_pt.get("term") or ""
                        if not isinstance(f_pt, str):
                            f_pt = ""
                        f_canon = _canon_term(f_pt)

                        new_name = f"patient_has_finding_of_{f_canon}"  # 名字不含 timeframe

                        # 用“名字 + 四个时间字段”做唯一键，避免同名不同时间窗被合并
                        key = _dedupe_key(new_name, time_ctx)
                        cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
                        if cur.rowcount == 1:
                            produced_names.append(new_name)
                            produced_items.append({
                                "name": new_name,
                                "findingCanonical": f_canon,
                                "findingId": fid,
                                "procedureId": proc_id,
                                "relationshipGroup": gid,
                                "hasInterpretationId": chosen_hid,
                                "hasInterpretationTerm": chosen_term,
                                "source_variable_name": var,
                                "source_conceptId": proc_id,
                                "target_variable_name": new_name,
                                "derivation_stage": DERIVATION_STAGE,
                                "derivation_rule": DERIVATION_RULE,
                            })
                            rec = {
                                "patient_id": row.get("patient_id") or "",
                                "class": "procedure_to_finding",
                                "new_variable_name": new_name,
                                "target_variable_name": new_name,
                                # no "timeframe" field anymore
                                "source_variable_name": var,
                                "source_conceptId": proc_id,
                                "derivation_stage": DERIVATION_STAGE,
                                "derivation_rule": DERIVATION_RULE,
                                "derived_from_variable": var,
                                "derived_from_procedureId": proc_id,
                                "derived_findingId": fid,
                                "derived_finding_term": f_canon,
                                "derived_qualifierId": chosen_hid,
                                "derived_qualifier_term": chosen_term,
                                "relationship_group": gid,
                            }
                            rec.update(time_ctx)  # ← 带上四个时间字段
                            fft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            ok_any = True

                    if not ok_any:
                        pass

                if produced_names:
                    c_rows_with_prod += 1
                    c_prod += len(set(produced_names))

                reason = ("ok" if produced_names else
                          ("filtered_by_definition_status" if filtered_def>0 else
                           ("filtered_by_minimality" if filtered_min>0 else "no_status_consistent_pairs")))
                if filtered_min: c_filtered_min += 1
                if filtered_def: c_filtered_def += 1

                fa.write(json.dumps({**row, "procedure_to_finding_enrichment": {
                    "side": side, "eligible": True, "produced": produced_items, "reason": reason
                }}, ensure_ascii=False) + "\n")

                fpv.write(json.dumps({
                    "patient_id": row.get("patient_id") or "",
                    "var_name": var,
                    "timeframe": tf,
                    "converted": 1 if produced_names else 0,
                    "reason": reason,
                    "procedure_canonical": proc_canon,
                    "procedure_id": proc_id,
                    "produced_count": len(set(produced_names)),
                    "produced_names": ";".join(sorted(set(produced_names)))
                }, ensure_ascii=False) + "\n")

        conn.commit(); conn.close()

        # jsonl → pretty array json (streamed; no in-RAM accumulation)
        out_pretty = out_annot.replace(".jsonl", ".pretty.json")
        first = True
        with open(out_pretty, "w", encoding="utf-8") as pf, open(out_annot, "r", encoding="utf-8") as fa:
            pf.write("[\n")
            for line in fa:
                s = line.strip()
                if not s:
                    continue
                if not first:
                    pf.write(",\n")
                pf.write(s)
                first = False
            pf.write("\n]\n")
        print(f"[info] also wrote pretty JSON: {out_pretty}")
        
        summary = {
            "rows": c_rows,
            "rows_with_produced": c_rows_with_prod,
            "produced_variables": c_prod,
            "rows_not_matching_template": c_not_match,
            "rows_missing_procedure_id": c_missing_pid,
            "rows_no_candidate_findings": c_no_cands,
            "rows_filtered_by_minimality": c_filtered_min,
            "rows_filtered_by_definition_status": c_filtered_def,
        }
        with open(out_struct, "w", encoding="utf-8") as fs:
            json.dump({"patient_id": pid, "side": side, "summary": summary}, fs, ensure_ascii=False, indent=2)
        with open(out_rep, "w", encoding="utf-8") as fr:
            json.dump({"patient_id": pid, "side": side, "counts": summary}, fr, ensure_ascii=False, indent=2)

        # jsonl.tmp → array json
        first=True
        with open(out_flat, "w", encoding="utf-8") as fout, open(out_flat_tmp, "r", encoding="utf-8") as fin:
            fout.write("[\n")
            for line in fin:
                s=line.strip()
                if not s: continue
                if not first: fout.write(",\n")
                fout.write(s)
                first=False
            fout.write("\n]\n")

        print(f"[{side:^9}] {pid:<24} ok (produced={summary['produced_variables']}, filtered_min={summary['rows_filtered_by_minimality']}, filtered_def={summary['rows_filtered_by_definition_status']})")

def main():
    ap = argparse.ArgumentParser(description="Procedure → Finding enricher (memory-safe streaming).")
    ap.add_argument("--patient", action="append", help="Only process this patient id (can repeat).")
    ap.add_argument(
        "--input-filename",
        default="finding_to_procedure_enrichment.flat.json",
        help="Preferred per-patient input filename; we auto-prefer *.flat.jsonl.",
    )
    ap.add_argument("--side", choices=["inclusion","exclusion","both"], default="inclusion")
    ap.add_argument(
        "--in-root",
        dest="in_root",
        required=False,
        default=None,
        help="Input root that contains per-patient dirs. Required unless P2F_INPUT_ROOT is set.",
    )
    ap.add_argument(
        "--out-root",
        dest="out_root",
        required=False,
        default=None,
        help="Output root to write results. Required unless P2F_OUTPUT_ROOT is set.",
    )
    args = ap.parse_args()

    # 解析 in_root / out_root：优先命令行，其次环境变量，缺失则报错
    if args.in_root:
        in_root = os.path.abspath(args.in_root)
    elif IN_ROOT:
        in_root = os.path.abspath(IN_ROOT)
    else:
        print("[p2f] --in-root is required unless P2F_INPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if args.out_root:
        out_root = os.path.abspath(args.out_root)
    elif OUT_ROOT:
        out_root = os.path.abspath(OUT_ROOT)
    else:
        print("[p2f] --out-root is required unless P2F_OUTPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(in_root):
        print(f"[err] IN_ROOT not found: {in_root}", file=sys.stderr); sys.exit(2)
    _ensure_dir(out_root)

    all_patients = sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))])
    patients = all_patients if not args.patient else [p for p in all_patients if p in set(args.patient)]
    if not patients:
        print(f"[warn] no matching patients under {in_root}")
        return

    if args.side in ("inclusion","exclusion"):
        process_side(in_root, out_root, args.side, patients, args.input_filename)
    else:
        process_side(in_root, out_root, "inclusion", patients, args.input_filename)
        process_side(in_root, out_root, "exclusion", patients, args.input_filename)

    print("[done] all requested sides processed.")


if __name__ == "__main__":
    main()
