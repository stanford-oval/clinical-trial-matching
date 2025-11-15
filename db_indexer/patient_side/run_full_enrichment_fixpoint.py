#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_enrichment_fixpoint.py
STRICT booleans in snapshots + alias-projection ingestion + qualifier stripping
────────────────────────────────────────────────────────────────────────────
Patient-by-patient enrichment fixpoint with:
  • Strict chaining (mirrors union into per-side inputs)
  • Per-stage provenance augmentation
  • Dedicated fine-grained entailment snapshots whose `source_value`:
        - Are computed ONLY from explicit booleans (authoritative facts first)
        - Use strict lookup (no heuristics) but
        - KV stores deterministic alias keys that map to the SAME explicit boolean
          (cross-stem + tail normalization with SAME timeframe), so strict lookups
          still hit for syntactic variants used as `source_variable_name`.
  • Qualifier-insensitive names (strip trailing @@qualifiers in normalization)
  • Persist facts BEFORE each snapshot so KV sees current booleans
  • Optional CLOSED-WORLD snapshot default (fill missing with False):
        --snapshot-default-false

Core enrich outputs are unchanged.

Stages: isa, imp, f2p, f2oe, p2f, oe2f, f2p_rel
"""

from __future__ import annotations
import os, sys, json, re, argparse, datetime as dt
from typing import List, Tuple, Optional, Iterable, Dict, Any, Set

# =========================
# ======= CONFIG ==========
# =========================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Inputs/Outputs (stage roots)
ISA_IN_ROOT   = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results"))
ISA_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_isa"))

IMP_IN_ROOT   = ISA_OUT_ROOT
IMP_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_isa_imp"))

F2P_IN_ROOT   = IMP_OUT_ROOT
F2P_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_finding_to_procedure_enrichment"))

F2OE_IN_ROOT  = F2P_IN_ROOT
F2OE_OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_finding_to_observable_entity_enrichment"))

P2F_IN_ROOT   = F2P_OUT_ROOT
P2F_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_procedure_to_finding_enrichment"))

OE2F_IN_ROOT  = F2OE_OUT_ROOT
OE2F_OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_observable_entity_to_finding_enrichment"))

F2P_OTHER_IN_ROOT  = IMP_OUT_ROOT
F2P_OTHER_OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_finding_to_procedure_enrichment_other"))

ENTAIL_OUT_ROOT    = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_entailments"))
ROUND_SEED_ROOT    = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_round_seeds"))

# Demographics/Diagnosis (side-agnostic)
DEMO_ROOT     = ISA_IN_ROOT
DIAG_ROOT_IN  = (os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_build/patient_diagnosis_coded_results"))
                 if os.path.isdir(os.path.join(THIS_DIR, "../../patient_build/patient_build/patient_diagnosis_coded_results"))
                 else os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_diagnosis_coded_results")))

# Fixpoint/passes
GLOBAL_MAX_ROUNDS = 1
INNER_MAX_PASSES  = 2  # per-patient
RUN_SIDES = ["inclusion", "exclusion"]

# FS-only persistence
PERSIST_DB_PATH   = None
PERSIST_FS_ROOT   = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_facts_export"))
PERSIST_FROM      = ["demo", "diag", "isa_imp", "proc", "obs", "p2f", "oe2f", "proc_rel", "isa"]

# DEDICATED fine-grained entailment snapshot root (per patient/stage)
SNAPSHOT_ROOT          = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/snapshots"))
ENTAIL_SNAPSHOT_ROOT   = os.path.join(SNAPSHOT_ROOT, "entail_inspect")  # dedicated

# Version banner for logs
SNAPSHOT_IMPL_VERSION = "entail-snapshot-v11-strict-alias-qual-strip-prepersist-2025-10-06"

# Snapshot behavior flag (set by CLI)
SNAPSHOT_DEFAULT_FALSE = False  # closed-world fill for missing booleans (snapshot-only)

# =========================
# ===== Imports ===========
# =========================

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Stage modules (unchanged)
import enrich_with_isa as isa_mod
import apply_schema_implications as impl_mod
import enrich_findings_to_procedure as f2p_mod
import enrich_findings_to_observable_entity as f2oe_mod
import enrich_procedure_to_finding as p2f_mod
import enrich_observable_entity_to_finding as oe2f_mod
import enrich_findings_to_procedure_via_other_relations as f2p_rel_mod

# Bind module roots (avoid drift)
impl_mod.ROOT_IN  = IMP_IN_ROOT
impl_mod.ROOT_OUT = IMP_OUT_ROOT
f2p_mod.IN_ROOT   = F2P_IN_ROOT
f2oe_mod.IN_ROOT  = F2OE_IN_ROOT
p2f_mod.IN_ROOT   = P2F_IN_ROOT
oe2f_mod.IN_ROOT  = OE2F_IN_ROOT
f2p_rel_mod.IN_ROOT  = F2P_OTHER_IN_ROOT
f2p_rel_mod.OUT_ROOT = F2P_OTHER_OUT_ROOT

# Implication rules
ABS_RULES_PATH = "../rules/schema_implications.json"
impl_mod.RULES_PATH = ABS_RULES_PATH

# =========================
# == Helpers & Utilities ==
# =========================

def _print_roots(args):
    print("[roots]")
    print("  ISA in/out:          ", ISA_IN_ROOT, "->", ISA_OUT_ROOT)
    print("  Imp in/out:          ", IMP_IN_ROOT, "->", IMP_OUT_ROOT)
    print("  F→P in/out:          ", F2P_IN_ROOT, "->", F2P_OUT_ROOT)
    print("  F→OE in/out:         ", F2OE_IN_ROOT, "->", F2OE_OUT_ROOT)
    print("  P→F in/out:          ", P2F_IN_ROOT, "->", P2F_OUT_ROOT)
    print("  OE→F in/out:         ", OE2F_IN_ROOT, "->", OE2F_OUT_ROOT)
    print("  F→P_rel in/out:      ", F2P_OTHER_IN_ROOT, "->", F2P_OTHER_OUT_ROOT)
    print("  Entailments out:     ", ENTAIL_OUT_ROOT)
    print("  Seed root:           ", ROUND_SEED_ROOT)
    print("  Persist FS root:     ", PERSIST_FS_ROOT)
    print("  Snapshot root:       ", ENTAIL_SNAPSHOT_ROOT, "(DEDICATED)")

def _list_dirs(root: str) -> List[str]:
    if not os.path.isdir(root): return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def _list_patients_under(root: str) -> List[str]:
    return sorted(_list_dirs(root))

def _filter_patients(candidates: List[str], only: Optional[str]) -> List[str]:
    return candidates if not only else [p for p in candidates if p == only]

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path): return
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try: yield json.loads(s)
            except Exception: continue

def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _read_json_list(path: str) -> list:
    try: return json.load(open(path, "r", encoding="utf-8"))
    except Exception: return []

def _write_json_list(path: str, arr: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

def _sha1_file(path: str) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

# 放在辅助函数区域
_QUAL_RX = re.compile(r"(?:@@[a-z0-9_]+)+$", re.I)

def _strip_tf_suffix(name: str) -> str:
    """从变量名里去掉结尾的 _{timeframe}，并保留原有 @@qualifier，最后折叠多余下划线。"""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # 去掉末尾的 _{timeframe}
    m = re.search(rf"_(?:{TIMEFRAME_RE})$", s)
    if m:
        s = s[:m.start()]
    # 保留结尾 qualifier，但去掉多余下划线
    m2 = _QUAL_RX.search(s)
    qual = m2.group(0) if m2 else ""
    stem = s if not m2 else s[:m2.start()]
    stem = re.sub(r"__+", "_", stem).strip("_")
    return stem + (qual or "")

def _replace_entity_in_finding_var(base_var: str, new_entity_canon: str) -> str:
    """
    把形如 '..._of_{entity}[_...][@@qual]*' 的变量里 entity 段替换为 new_entity_canon。
    若不含 `_of_`，退化为 'patient_has_finding_of_{new_entity_canon}'。
    """
    if not base_var:
        return f"patient_has_finding_of_{new_entity_canon}"
    s = _strip_tf_suffix(base_var)
    mqual = _QUAL_RX.search(s)
    qual = mqual.group(0) if mqual else ""
    stem = s if not mqual else s[:mqual.start()]
    m = re.match(r"^(?:(?P<prefix>.+?_of_))(?P<ent>[a-z0-9_]+)(?P<tail>(?:_.+)?)$", stem, re.I)
    if m:
        prefix = m.group("prefix")
        tail   = m.group("tail") or ""
        out = f"{prefix}{new_entity_canon}{tail}"
        return re.sub(r"__+", "_", out).strip("_") + qual
    return f"patient_has_finding_of_{new_entity_canon}{qual}"


TIMEFRAME_UNITS = r"(?:minutes|hours|days|weeks|months|years)"
TIMEFRAME_RE = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+" + TIMEFRAME_UNITS + r"|"
    r"inthefuture\d+" + TIMEFRAME_UNITS + r"|"
    r"foradurationof\d+" + TIMEFRAME_UNITS + r")"
)
TF_RX = re.compile(TIMEFRAME_RE)

try:
    import psutil
except Exception:
    psutil = None

def mem_ok(max_gb: Optional[float]) -> bool:
    if not max_gb or not psutil: return True
    vm = psutil.virtual_memory()
    used_gb = (vm.total - vm.available) / (1024**3)
    return used_gb <= max_gb

# =========================
# == Implications helpers ==
# =========================

def _flatten_implications_structured(doc: dict) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(doc, dict): return out
    gobj = doc.get("groups")
    if not isinstance(gobj, dict): return out
    for key in ("schema", "timeframe"):
        rows = gobj.get(key)
        if not isinstance(rows, list): continue
        for r in rows:
            if not isinstance(r, dict): continue
            nm = r.get("name")
            if isinstance(nm, str) and nm:
                out.append({"entity_variable_name": nm, "extracted_value": bool(r.get("extracted_value", True)), "type": "Bool"})
            for it in (r.get("implied") or []):
                nm2 = (it or {}).get("name")
                if isinstance(nm2, str) and nm2:
                    out.append({"entity_variable_name": nm2, "extracted_value": bool((it or {}).get("value", True)), "type": "Bool"})
    return out

# ---- streaming union helpers ----
def _spillset_open(db_path: str):
    import sqlite3
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY)")
    conn.commit()
    return conn

def _canonicalize_term(term: Optional[str]) -> str:
    """
    规范化 SNOMED 术语为变量名可用的 canonical 形式：
    - 去掉括号里的词性尾缀，如 "xxx (finding)"
    - 全小写
    - 非字母数字替换为下划线，折叠多下划线，去首尾下划线
    """
    t = (term or "").strip()
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)  # strip trailing parenthetical
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _build_implication_union_for_patient(pid: str, side: str) -> tuple[int, int]:
    """
    读 IMP_OUT_ROOT/<pid>/<side>/canonical.enriched.annotated.jsonl
    写 IMP_OUT_ROOT/<pid>/<side>/canonical.enriched.union.jsonl
    """
    pdir = os.path.join(IMP_OUT_ROOT, pid, side); os.makedirs(pdir, exist_ok=True)
    annotated_path = os.path.join(pdir, "canonical.enriched.annotated.jsonl")
    union_path     = os.path.join(pdir, "canonical.enriched.union.jsonl")

    seen_db = os.path.join(pdir, ".tmp", "union_seen.sqlite")
    if os.path.exists(seen_db): os.remove(seen_db)
    conn = _spillset_open(seen_db); cur = conn.cursor()

    def _ins(rec: Dict[str, Any], cnt: Dict[str,int]):
        nm = str(rec.get("entity_variable_name") or "").strip()
        if not nm: return
        st = rec.get("start_time_in_hours"); en = rec.get("end_time_in_hours")
        si = 1 if rec.get("start_time_inclusive") else 0
        ei = 1 if rec.get("end_time_inclusive") else 0
        ex = 1 if str(rec.get("extracted_value")).lower() in ("1","true","t","yes") else 0
        key = f"{nm}|{(0.0 if st is None else round(float(st),3))}|{(0.0 if en is None else round(float(en),3))}|{si}|{ei}|{ex}"
        cur.execute("INSERT OR IGNORE INTO s(k) VALUES (?)", (key,))
        if cur.rowcount == 1:
            cnt["union"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counters = {"expanded": 0, "union": 0}
    if not os.path.isfile(annotated_path): return (0, 0)

    with open(union_path, "w", encoding="utf-8") as fout, open(annotated_path, "r", encoding="utf-8") as fin:
        for ln in fin:
            s = ln.strip()
            if not s: continue
            try: row = json.loads(s)
            except Exception: continue

            base_name = _strip_tf_suffix(str(row.get("entity_variable_name") or ""))
            if not base_name: continue
            exb = True if str(row.get("extracted_value")).lower() in ("true","t","1","yes") else False
            tc = {
                "start_time_in_hours":  row.get("start_time_in_hours"),
                "end_time_in_hours":    row.get("end_time_in_hours"),
                "start_time_inclusive": row.get("start_time_inclusive"),
                "end_time_inclusive":   row.get("end_time_inclusive"),
            }

            base_rec = {
                "entity_variable_name": base_name,
                "type": "Bool",
                "extracted_value": bool(exb),
                "conceptId": row.get("conceptId"),
                "fact_id": row.get("fact_id"),
            }
            base_rec.update(tc); _ins(base_rec, counters)

            stem_for_proj = base_name

            if exb:
                for bucket, rel_label in (("parents_inferred","parent"), ("ancestors_inferred","ancestor")):
                    for it in (row.get(bucket) or []):
                        canon = _canonicalize_term(it.get("preferred_term") or it.get("term") or "")
                        if not canon: continue
                        nm = _replace_entity_in_finding_var(stem_for_proj, canon)
                        rec = {"entity_variable_name": nm, "type": "Bool", "extracted_value": True,
                               "conceptId": it.get("conceptId"), "fact_id": row.get("fact_id"), "isa_relation": rel_label}
                        rec.update(tc); _ins(rec, counters); counters["expanded"] += 1

            for it in (row.get("descendants_inferred") or []):
                canon = _canonicalize_term(it.get("preferred_term") or it.get("term") or "")
                if not canon: continue
                nm = _replace_entity_in_finding_var(stem_for_proj, canon)
                rec = {"entity_variable_name": nm, "type": "Bool", "extracted_value": False,
                       "conceptId": it.get("conceptId"), "fact_id": row.get("fact_id"), "isa_relation": "descendant"}
                rec.update(tc); _ins(rec, counters); counters["expanded"] += 1

            imp = row.get("implications") or {}
            for bucket in ("schema_new_only", "timeframe_new_only"):
                for it in (imp.get(bucket) or []):
                    nm = _strip_tf_suffix(str(it.get("name") or ""))
                    if not nm: continue
                    rec = {"entity_variable_name": nm, "type": "Bool", "extracted_value": True,
                           "conceptId": it.get("conceptId"), "fact_id": row.get("fact_id")}
                    rec.update({
                        "start_time_in_hours":  it.get("start_time_in_hours", tc["start_time_in_hours"]),
                        "end_time_in_hours":    it.get("end_time_in_hours", tc["end_time_in_hours"]),
                        "start_time_inclusive": it.get("start_time_inclusive", tc["start_time_inclusive"]),
                        "end_time_inclusive":   it.get("end_time_inclusive", tc["end_time_inclusive"]),
                    })
                    _ins(rec, counters); counters["expanded"] += 1

    conn.commit(); conn.close()
    return (counters["expanded"], counters["union"])



def _compat_load_impl_rules():
    tf_cfg_default = getattr(impl_mod, "TF_CFG_DEFAULT", {"collapse_timeframes": True})
    primary_path = getattr(impl_mod, "RULES_PATH", None)
    if primary_path and os.path.isfile(primary_path):
        with open(primary_path, "r", encoding="utf-8") as f:
            rules_doc = json.load(f)
        return rules_doc, rules_doc.get("timeframe_implication", tf_cfg_default), f"file:{primary_path}"
    rules_doc = getattr(impl_mod, "RULES_DOC", None)
    if isinstance(rules_doc, dict) and "rules" in rules_doc:
        return rules_doc, rules_doc.get("timeframe_implication", tf_cfg_default), "impl_mod.RULES_DOC"
    rules_list = getattr(impl_mod, "RULES", None)
    if isinstance(rules_list, list):
        return {"rules": rules_list}, getattr(impl_mod, "TF_CFG", tf_cfg_default), "impl_mod.RULES"
    print("[warn] no implication rules found, using empty set.")
    return {"rules": []}, tf_cfg_default, "empty"

def _implications_passthrough(pin: str, pout: str) -> tuple[int,int,int]:
    os.makedirs(pout, exist_ok=True)
    src = None
    for cand in ("canonical.enriched.jsonl", "canonical.jsonl"):
        p = os.path.join(pin, cand)
        if os.path.isfile(p): src = p; break
    rows = 0
    if src:
        dst = os.path.join(pout, "canonical.enriched.jsonl")
        with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for ln in fin:
                s = ln.strip()
                if s: rows += 1
                fout.write(ln)
    else:
        open(os.path.join(pout, "canonical.enriched.jsonl"), "w", encoding="utf-8").close()
    struct_path = os.path.join(pout, "canonical.enriched.implications.structured.json")
    if not os.path.isfile(struct_path):
        with open(struct_path, "w", encoding="utf-8") as f:
            json.dump({"groups": {"schema": [], "timeframe": []}}, f, ensure_ascii=False)
    return rows, 0, 0

def _impl_process_patient_compat(pin: str, pout: str, rules: list, tf_cfg: dict) -> tuple[int,int,int]:
    for name in ("process_patient", "process_one_patient", "process_patient_dir", "apply_to_patient", "apply_rules_to_patient"):
        func = getattr(impl_mod, name, None)
        if not callable(func): continue
        try: res = func(pin, pout, rules, tf_cfg)
        except TypeError:
            try: res = func(pin, pout, rules)
            except TypeError:
                try: res = func(pin, pout)
                except Exception: continue
        except Exception: continue
        if isinstance(res, tuple):
            a = int(res[0]) if len(res)>0 else 0
            b = int(res[1]) if len(res)>1 else 0
            c = int(res[2]) if len(res)>2 else 0
            return a,b,c
        if isinstance(res, (int,float)): return int(res),0,0
        return 0,0,0
    print("[warn] apply_schema_implications.* had no compatible entrypoint; skipping.")
    return 0,0,0

# =========================
# === Union Mirroring =====
# =========================

def _mirror_union_into_side_trees(pids: List[str], dest_root: str) -> None:
    """
    将 IMP_OUT_ROOT/<pid>/<side>/canonical.enriched.union.jsonl 镜像到 <dest_root>/<pid>/<side>/...。
    若源目标同路径（例如 dest_root == IMP_OUT_ROOT），跳过以避免自我截断。
    """
    if not pids or not os.path.isdir(IMP_OUT_ROOT): 
        return
    for pid in pids:
        for side in RUN_SIDES:
            src = os.path.join(IMP_OUT_ROOT, pid, side, "canonical.enriched.union.jsonl")
            dst = os.path.join(dest_root,  pid, side, "canonical.enriched.union.jsonl")

            if not os.path.isfile(src):
                continue

            # 如果目标与源完全相同，直接跳过，避免 open("w") 把自己清空
            if os.path.abspath(src) == os.path.abspath(dst):
                # 可选：打印一次 debug
                # print(f"[mirror-union] skip self-copy: {src}")
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # 安全拷贝：先读入内存，再一次性写入，避免边读边写的截断问题
            with open(src, "rb") as fin:
                data = fin.read()

            with open(dst, "wb") as fout:
                fout.write(data)

            if len(data) == 0:
                print(f"[warn][mirror-union] {pid}/{side} mirrored 0 bytes from {src} → {dst}")



# =========================
# == Provenance augmenters ==
# =========================

SOURCE_ID_KEYS = {
    "f2p":     ["source_conceptId", "findingId", "derived_findingId", "conceptId"],
    "f2oe":    ["source_conceptId", "findingId", "derived_findingId", "conceptId"],
    "p2f":     ["source_conceptId", "procedureId", "derived_procedureId", "conceptId"],
    "oe2f":    ["source_conceptId", "observableEntityId", "derived_observableEntityId", "conceptId"],
    "f2p_rel": ["source_conceptId", "findingId", "derived_findingId", "conceptId"],
}

DEFAULT_RULE = {
    "f2p": "interprets(363714003)",
    "f2oe": "interprets(363714003)",
    "p2f": "interprets(363714003)+hasInterpretation(363713009)",
    "oe2f": "interprets(363714003)",
    "f2p_rel": "via_relation",
}

def _ensure_provenance_fields(stage_key: str, root: str, pid: str, side: str,
                              out_fname: str, rule_hint_key: Optional[str] = None) -> int:
    p = os.path.join(root, pid, side, out_fname)
    arr = _read_json_list(p)
    if not isinstance(arr, list) or not arr:
        return 0

    updated = 0
    missing_src = 0

    for rec in arr:
        if not isinstance(rec, dict): continue

        dst = (rec.get("new_variable_name") or rec.get("name") or rec.get("entity_variable_name"))
        if not isinstance(dst, str) or not dst.strip(): continue

        # source_variable_name
        src = rec.get("source_variable_name")
        if not (isinstance(src, str) and src.strip()):
            dv = rec.get("derived_from_variable")
            if isinstance(dv, str) and dv.strip():
                rec["source_variable_name"] = dv.strip()
                updated += 1
            else:
                missing_src += 1

        # source_conceptId
        if not rec.get("source_conceptId"):
            for k in SOURCE_ID_KEYS.get(stage_key, []):
                cid = rec.get(k)
                if isinstance(cid, (int, float)): cid = str(cid)
                if isinstance(cid, str) and cid.strip():
                    rec["source_conceptId"] = cid.strip(); updated += 1; break

        if not rec.get("derivation_stage"):
            rec["derivation_stage"] = stage_key; updated += 1

        if not rec.get("derivation_rule"):
            if rule_hint_key and isinstance(rec.get(rule_hint_key), str) and rec[rule_hint_key].strip():
                rec["derivation_rule"] = rec[rule_hint_key].strip()
            else:
                rec["derivation_rule"] = DEFAULT_RULE.get(stage_key, "rule")
            updated += 1

    if updated:
        _write_json_list(p, arr)
        print(f"[prov] augmented {updated} row(s) → {p}")
    if missing_src:
        print(f"[prov][warn] {stage_key}:{pid}:{side} {missing_src} row(s) still missing explicit source_variable_name")
    return updated

# =========================
# ====== Persistence ======
# =========================

def _normalize_demo_varname(varname: str) -> str:
    if not isinstance(varname, str): return varname or ""
    return re.sub(r"^age_value_recorded_", "patient_age_value_recorded_", varname)

def _split_base_and_tf(varname: str) -> tuple[str,str]:
    m = TF_RX.search(varname or ""); 
    if not m: return varname, "now"
    tf = m.group(0); s,e = m.span()
    base = (varname[:s] + varname[e:]).strip("_"); base = re.sub(r"__+","_", base)
    return base, tf

def _iter_records_file(path: str) -> Iterable[dict]:
    if path.endswith(".jsonl"):
        yield from _iter_jsonl(path); return
    try: doc = json.load(open(path,"r",encoding="utf-8"))
    except Exception: return
    if isinstance(doc, list):
        for it in doc:
            if isinstance(it, dict): yield it
        return
    if not isinstance(doc, dict): return
    groups = doc.get("groups")
    if isinstance(groups, list):
        for g in groups:
            for it in (g.get("produced_variables") or []):
                if isinstance(it, dict): yield it
    gobj = doc.get("groups")
    if isinstance(gobj, dict):
        for key in ("schema","timeframe"):
            for r in (gobj.get(key) or []):
                nm = (r or {}).get("name")
                if isinstance(nm,str) and nm:
                    yield {"name": nm, "extracted_value": bool(r.get("extracted_value", True))}
                for it in ((r or {}).get("implied") or []):
                    nm2 = (it or {}).get("name")
                    if isinstance(nm2,str) and nm2:
                        yield {"name": nm2, "extracted_value": bool((it or {}).get("value", True))}

def _find_patient_files_for_root(root: str, pid: str) -> list[tuple[str,str,Optional[str]]]:
    out: list[tuple[str,str,Optional[str]]] = []
    ab = os.path.abspath(root)
    if ab == os.path.abspath(DEMO_ROOT):
        p = os.path.join(root, pid, "demographics.jsonl")
        if os.path.isfile(p): out.append(("demo", p, None))
        return out
    if ab == os.path.abspath(DIAG_ROOT_IN):
        p = os.path.join(root, pid, "diagnosis.export.jsonl")
        if os.path.isfile(p): out.append(("diag", p, None))
        else:
            p2 = os.path.join(root, pid, "diagnosis.jsonl")
            if os.path.isfile(p2): out.append(("diag", p2, None))
        return out

    # side-based roots（包含 isa/imp/f2*）
    for side in ("inclusion","exclusion"):
        d = os.path.join(root, pid, side)
        if not os.path.isdir(d): continue

        if ab == os.path.abspath(ISA_OUT_ROOT):
            p = os.path.join(d, "isa_enriched.flat.json")
            if os.path.isfile(p): out.append(("isa", p, side))
            p = os.path.join(d, "canonical.enriched.jsonl")
            if os.path.isfile(p): out.append(("isa_imp", p, side))  # ISA enriched canonical 也作为 isa_imp 输入的一部分

        elif ab == os.path.abspath(IMP_OUT_ROOT):
            p = os.path.join(d, "canonical.enriched.jsonl")
            if os.path.isfile(p): out.append(("isa_imp", p, side))
            p_imp_struct = os.path.join(d, "canonical.enriched.implications.structured.json")
            if os.path.isfile(p_imp_struct): out.append(("isa_imp", p_imp_struct, side))

        # 下游 side-based
        p = os.path.join(d, "finding_to_observable_entity_enrichment.flat.json")
        if os.path.isfile(p): out.append(("obs", p, side))
        p = os.path.join(d, "finding_to_procedure_enrichment.flat.json")
        if os.path.isfile(p): out.append(("proc", p, side))
        p = os.path.join(d, "finding_to_procedure_enriched.annotated.jsonl")
        if os.path.isfile(p): out.append(("proc", p, side))
        p = os.path.join(d, "observable_entity_to_finding_enrichment.flat.json")
        if os.path.isfile(p): out.append(("oe2f", p, side))
        p = os.path.join(d, "procedure_to_finding_enrichment.flat.json")
        if os.path.isfile(p): out.append(("p2f", p, side))
        p = os.path.join(d, "finding_to_procedure_via_relations.flat.json")
        if os.path.isfile(p): out.append(("proc_rel", p, side))

    # 非 side（原始 ISA_IN_ROOT / ROUND_SEED_ROOT 的 base）
    d = os.path.join(root, pid)
    for nm in ("canonical.jsonl",):
        p = os.path.join(d, nm)
        if os.path.isfile(p):
            out.append(("isa_imp", p, None))
    return out


def _extract_var_and_value(rec: dict) -> tuple[Optional[str], Optional[float]]:
    for k in ("name","new_variable_name"):
        v = rec.get(k)
        if isinstance(v,str) and v: return _normalize_demo_varname(v), None
    vname = rec.get("entity_variable_name")
    if isinstance(vname, str) and vname:
        vname = _normalize_demo_varname(vname)
        if isinstance(rec.get("value"), (int,float)): return vname, float(rec["value"])
        rtype = (rec.get("type") or "").strip().lower()
        if rtype in {"int","float","double","number","numeric","real"}:
            ev = rec.get("extracted_value")
            if isinstance(ev,(int,float)): return vname, float(ev)
            if isinstance(ev,str):
                try: return vname, float(ev.strip())
                except Exception: pass
        if str(rec.get("extracted_value")).strip().lower() == "true": return vname, None
        return None, None
    pv = rec.get("produced_variables")
    if isinstance(pv, list):
        for it in pv:
            if isinstance(it, dict) and isinstance(it.get("name"), str):
                return _normalize_demo_varname(it["name"]), None
    return None, None

def persist_all_facts(patients: List[str], stage_roots: dict[str,str], persist_from: List[str],
                      persist_db_path: Optional[str], persist_fs_root: Optional[str], round_idx: int) -> int:
    total = 0
    if persist_fs_root: os.makedirs(persist_fs_root, exist_ok=True)
    ordered: List[tuple[str,str]] = []
    for tag in ("demo","diag","isa_imp","proc","obs","p2f","oe2f","proc_rel","isa"):
        if tag in persist_from and tag in stage_roots: ordered.append((tag, stage_roots[tag]))
    demo_count = 0; demo_patients: Set[str] = set()
    for pid in patients:
        seen: set[tuple[str,str,str,str,str]] = set()
        fh: dict[str, Optional[Any]] = {"inclusion": None, "exclusion": None}
        if persist_fs_root:
            for side in ("inclusion","exclusion"):
                pdir = os.path.join(persist_fs_root, pid, side); os.makedirs(pdir, exist_ok=True)
                fh[side] = open(os.path.join(pdir, f"facts.round{round_idx}.jsonl"), "w", encoding="utf-8")
        def _emit(side: str, base: str, tf: str, kind: str, value: float|int, src_tag: str):
            nonlocal total, demo_count
            key = (pid, base, tf, kind, side)
            if key in seen: return
            seen.add(key)
            if fh.get(side):
                fh[side].write(json.dumps({
                    "patient_id": pid, "base_var": base, "tf_token": tf, "value": value,
                    "kind": kind, "source": src_tag, "side": side
                }, ensure_ascii=False) + "\n")
            total += 1
        try:
            for tag, root in ordered:
                if not os.path.isdir(root): continue
                for src_tag, path, side_hint in _find_patient_files_for_root(root, pid):
                    if src_tag not in persist_from: continue
                    sides_to_write = [side_hint] if side_hint in ("inclusion","exclusion") else ["inclusion","exclusion"]
                    for rec in _iter_records_file(path):
                        if isinstance(rec.get("extracted_value"), bool):
                            varname = rec.get("entity_variable_name") or rec.get("name") or rec.get("new_variable_name")
                            if isinstance(varname,str) and varname:
                                base, tf = _split_base_and_tf(_normalize_demo_varname(varname))
                                for s in sides_to_write: _emit(s, base, tf, "bool", 1 if rec["extracted_value"] else 0, src_tag)
                                if src_tag == "demo": demo_count += 1; demo_patients.add(pid)
                                continue
                        var, num = _extract_var_and_value(rec)
                        if not var: continue
                        base, tf = _split_base_and_tf(var)
                        if num is not None:
                            for s in sides_to_write: _emit(s, base, tf, "num", float(num), src_tag)
                        else:
                            for s in sides_to_write: _emit(s, base, tf, "bool", 1, src_tag)
        finally:
            for s in ("inclusion","exclusion"):
                if fh.get(s): fh[s].close()
    if demo_count:
        print(f"[persist] demo rows: {demo_count} across {len(demo_patients)} patient(s)")
    else:
        print("[persist] WARNING: no demographics rows seen; check DEMO_ROOT")
    return total

# =========================
# ===== Seed canonical ====
# =========================

def _extract_tf_token(name: str) -> Optional[str]:
    m = TF_RX.search(name or ""); return m.group(0) if m else None

def _iter_prev_seed_rows_for_patient(pid: str, side: str) -> Iterable[Dict[str, Any]]:
    assert side in RUN_SIDES

    def _add_from_flat(root: str, filename: str, cid_keys: List[str]) -> Iterable[Dict[str, Any]]:
        p = os.path.join(root, pid, side, filename)
        if not os.path.isfile(p): return
        try: arr = json.load(open(p,"r",encoding="utf-8"))
        except Exception: arr = None
        if not isinstance(arr, list): return
        for row in arr:
            if not isinstance(row, dict): continue
            name = str(row.get("new_variable_name") or row.get("name") or "")
            if not name: continue
            cid = None
            for ck in cid_keys:
                v = row.get(ck)
                if isinstance(v,str) and v: cid = v.strip(); break
            yield {"entity_variable_name": name, "conceptId": cid, "extracted_value": True, "type": "Bool",
                   "timeframe": _extract_tf_token(name), "fact_id": f"seed_from_union_{side}"}

    # 仅吸收对应侧的下游产物
    yield from _add_from_flat(F2P_OUT_ROOT,     "finding_to_procedure_enrichment.flat.json",            ["derived_conceptId","procedureId"])
    yield from _add_from_flat(F2OE_OUT_ROOT,    "finding_to_observable_entity_enrichment.flat.json",     ["derived_conceptId","observableEntityId"])
    yield from _add_from_flat(P2F_OUT_ROOT,     "procedure_to_finding_enrichment.flat.json",            ["derived_findingId","findingId"])
    yield from _add_from_flat(OE2F_OUT_ROOT,    "observable_entity_to_finding_enrichment.flat.json",     ["derived_findingId","findingId"])
    yield from _add_from_flat(F2P_OTHER_OUT_ROOT,"finding_to_procedure_via_relations.flat.json",         ["derived_conceptId","procedureId"])

    # ISA（已侧化的 ISA 输出）
    isa_flat = os.path.join(ISA_OUT_ROOT, pid, side, "isa_enriched.flat.json")
    if os.path.isfile(isa_flat):
        try: arr = json.load(open(isa_flat,"r",encoding="utf-8"))
        except Exception: arr = None
        if isinstance(arr, list):
            for row in arr:
                name = str((row or {}).get("new_variable_name") or (row or {}).get("target_variable_name") or "")
                if not name: continue
                bv = row.get("extracted_value")
                if isinstance(bv, str): bv = bv.strip().lower() == "true"
                exb = True if bv is None else bool(bv)
                yield {"entity_variable_name": name,
                       "conceptId": (row or {}).get("derived_conceptId") or None,
                       "extracted_value": exb,
                       "type":"Bool",
                       "timeframe": _extract_tf_token(name),
                       "fact_id":f"seed_from_isa_{side}"}

    # Implications（已侧化）
    imp_struct = os.path.join(IMP_OUT_ROOT, pid, side, "canonical.enriched.implications.structured.json")
    if os.path.isfile(imp_struct):
        try: doc = json.load(open(imp_struct,"r",encoding="utf-8"))
        except Exception: doc = None
        flat = _flatten_implications_structured(doc) if isinstance(doc, dict) else []
        for it in flat:
            nm = it["entity_variable_name"]
            yield {"entity_variable_name": nm, "conceptId": None, "extracted_value": bool(it.get("extracted_value", True)),
                   "type":"Bool", "timeframe": _extract_tf_token(nm), "fact_id":f"seed_from_implication_{side}"}


def _state_hash_path(pid: str, side: str) -> str:
    return os.path.join(ROUND_SEED_ROOT, pid, side, ".seed.sha1")

def _build_round_seed_canonical(patients: List[str], round_idx: int) -> Set[tuple[str,str]]:
    """
    为每个 patient 的每个 side 生成独立 seed：
    读取 base（ISA_IN_ROOT/<pid>/canonical.jsonl），合并本侧历史/上轮产物。
    返回发生变化的 (pid, side) 集合。
    """
    print(f"\n=== Build side seeds (round {round_idx}) for {len(patients)} patient(s) × {len(RUN_SIDES)} side(s) ===")
    os.makedirs(ROUND_SEED_ROOT, exist_ok=True)
    changed: Set[tuple[str,str]] = set()
    for pid in patients:
        orig_path = os.path.join(ISA_IN_ROOT, pid, "canonical.jsonl")
        for side in RUN_SIDES:
            out_dir  = os.path.join(ROUND_SEED_ROOT, pid, side); os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "canonical.jsonl")
            seen=set(); written=base=seed=0
            with open(out_path, "w", encoding="utf-8") as fout:
                # base（side 无关，直接并入）
                for r in _iter_jsonl(orig_path):
                    nm = str(r.get("entity_variable_name") or "")
                    if not nm: continue
                    base += 1
                    if nm in seen: continue
                    seen.add(nm); fout.write(json.dumps(r, ensure_ascii=False) + "\n"); written += 1
                # 补种（本侧）
                for r in _iter_prev_seed_rows_for_patient(pid, side):
                    nm = str(r.get("entity_variable_name") or "")
                    if not nm: continue
                    seed += 1
                    if nm in seen: continue
                    seen.add(nm); fout.write(json.dumps(r, ensure_ascii=False) + "\n"); written += 1
            new_hash = _sha1_file(out_path)
            st = _state_hash_path(pid, side)
            old_hash = None
            if os.path.isfile(st):
                try: old_hash = open(st,"r",encoding="utf-8").read().strip()
                except Exception: old_hash = None
            open(st, "w", encoding="utf-8").write(new_hash)
            is_changed = (old_hash != new_hash)
            if is_changed: changed.add((pid, side))
            print(f"[seed:{side}] {pid:<24} base={base} +seeds={seed} -> written={written}  ({'changed' if is_changed else 'unchanged'})")
    return changed


# =========================
# === Entailments export ==
# =========================

def _ent_write_for_side(pid: str, side: str) -> int:
    out_dir = os.path.join(ENTAIL_OUT_ROOT, pid, side); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "entailments.vars.jsonl")
    seen: Set[str] = set(); written = 0
    def _emit(varname: Optional[str], source: str):
        nonlocal written
        if not isinstance(varname, str) or not varname.strip(): return
        nm = varname.strip()
        if nm in seen: return
        seen.add(nm)
        open(out_path, "a", encoding="utf-8").write(json.dumps({
            "patient_id": pid, "entity_variable_name": nm, "extracted_value": True, "type":"Bool", "source": source
        }, ensure_ascii=False) + "\n")
        written += 1
    open(out_path, "w", encoding="utf-8").close()
    union = os.path.join(IMP_OUT_ROOT, pid, side, "canonical.enriched.union.jsonl")
    for obj in _iter_jsonl(union):
        nm = obj.get("entity_variable_name") or obj.get("name")
        if isinstance(nm,str) and str(obj.get("extracted_value", True)).lower()=="true": _emit(nm,"isa_imp_union")
    for it in _read_json_list(os.path.join(F2P_OUT_ROOT, pid, side, "finding_to_procedure_enrichment.flat.json")):
        _emit(it.get("new_variable_name"), "f2p")
    for it in _read_json_list(os.path.join(F2OE_OUT_ROOT, pid, side, "finding_to_observable_entity_enrichment.flat.json")):
        _emit(it.get("new_variable_name"), "f2oe")
    for it in _read_json_list(os.path.join(P2F_OUT_ROOT, pid, side, "procedure_to_finding_enrichment.flat.json")):
        _emit(it.get("new_variable_name"), "p2f")
    for it in _read_json_list(os.path.join(OE2F_OUT_ROOT, pid, side, "observable_entity_to_finding_enrichment.flat.json")):
        _emit(it.get("new_variable_name"), "oe2f")
    for it in _read_json_list(os.path.join(F2P_OTHER_OUT_ROOT, pid, side, "finding_to_procedure_via_relations.flat.json")):
        _emit(it.get("new_variable_name"), "f2p_rel")
    return written

def write_entailments_for_patients(patients: List[str]) -> None:
    if not patients: return
    os.makedirs(ENTAIL_OUT_ROOT, exist_ok=True)
    for pid in patients:
        c_in = _ent_write_for_side(pid, "inclusion")
        c_ex = _ent_write_for_side(pid, "exclusion")
        print(f"[entail] {pid:<24} inclusion={c_in}  exclusion={c_ex}")

# =========================
# === Snapshot KV helpers (STRICT LOOKUP + ALIAS PROJECTION) ==
# =========================

_NAME_RX_MULTI_US = re.compile(r"__+")
_QUAL_SUFFIX_RX   = re.compile(r"@@[a-z0-9_]+(?:@@[a-z0-9_]+)*$", re.IGNORECASE)
_STEM_PREFIXES = [
    "patient_has_diagnosis_of_",
    "patient_has_finding_of_",
    "patient_has_symptoms_of_",
    "patient_has_clinical_signs_of_",
    "patient_has_suspicion_of_",
]

def _norm_name(nm: Optional[str]) -> Optional[str]:
    """Normalize keys: strip trailing @@qualifiers and collapse underscores."""
    if not isinstance(nm, str): return None
    nm = nm.strip()
    if not nm: return None
    nm = _QUAL_SUFFIX_RX.sub("", nm)
    nm = _NAME_RX_MULTI_US.sub("_", nm)
    return nm

def _make_alias_keys_from_fact_key(base_var: str, tf: str) -> list[str]:
    """Deterministic aliases (same timeframe): cross-stem + tail normalization."""
    keys = set()
    def add_key(b: str):
        if not b: return
        keys.add(_norm_name(f"{b}_{tf}"))

    # original
    add_key(base_var)

    # detect stem
    stem, tail0 = None, None
    for s in _STEM_PREFIXES:
        if base_var.startswith(s):
            stem, tail0 = s, base_var[len(s):]
            break

    def tail_norms(t: str) -> list[str]:
        out = {t}
        out.add(re.sub(r"_test_finding(?=$|_)", "_test", t))
        out.add(re.sub(r"_level_finding(?=$|_)", "_level", t))
        out.add(re.sub(r"_measurement_finding(?=$|_)", "_measurement", t))
        out.add(re.sub(r"_finding(?=$|_)", "", t))
        return [x for x in out if x]

    if stem and tail0:
        # cross-stem projections
        for alt in _STEM_PREFIXES:
            add_key(alt + tail0)
        # tail-normalized + cross-stem
        for tnorm in tail_norms(tail0):
            for alt in _STEM_PREFIXES:
                add_key(alt + tnorm)
    else:
        # No recognized stem—still normalize tail-like suffixes on full string
        b0 = base_var
        for bnorm in {
            b0,
            re.sub(r"_test_finding(?=$|_)", "_test", b0),
            re.sub(r"_level_finding(?=$|_)", "_level", b0),
            re.sub(r"_measurement_finding(?=$|_)", "_measurement", b0),
            re.sub(r"_finding(?=$|_)", "", b0),
        }:
            add_key(bnorm)

    return sorted(k for k in keys if k)

def _ingest_patient_facts_into_kv(pid: str, cur) -> int:
    """Authoritative booleans from facts; project aliases with same value."""
    import glob
    total = 0

    def _ingest_dir(side: str) -> int:
        n = 0
        d = os.path.join(PERSIST_FS_ROOT, pid, side)
        if not os.path.isdir(d): return 0
        files = sorted(glob.glob(os.path.join(d, "facts.round*.jsonl")))
        for path in files:  # earlier rounds first; later overwrite
            for obj in _iter_jsonl(path):
                if (obj or {}).get("kind") != "bool":  continue
                base = _norm_name((obj or {}).get("base_var"))
                tf   = _norm_name((obj or {}).get("tf_token"))
                if not base or not tf: continue
                vraw = (obj or {}).get("value")
                if isinstance(vraw, (int, float)): val = 1 if int(vraw)!=0 else 0
                elif isinstance(vraw, bool):       val = 1 if vraw else 0
                else: continue
                # exact key
                exact_key = _norm_name(f"{base}_{tf}")
                if exact_key:
                    try:
                        cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) "
                                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (exact_key, val))
                    except Exception:
                        pass
                    n += 1
                # alias keys
                for alias_key in _make_alias_keys_from_fact_key(base, tf):
                    try:
                        cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) "
                                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (alias_key, val))
                    except Exception:
                        pass
                    n += 1
        return n

    total += _ingest_dir("inclusion")
    total += _ingest_dir("exclusion")
    return total

def _ingest_patient_facts_into_kv_side(pid: str, side: str, cur) -> int:
    """
    仅摄入指定 side 的 persisted facts 到 KV（并做同值 alias 投影）。
    side ∈ {"inclusion", "exclusion"}
    """
    import glob
    assert side in ("inclusion", "exclusion")
    total = 0
    d = os.path.join(PERSIST_FS_ROOT or "", pid, side)
    if not d or not os.path.isdir(d): 
        return 0

    files = sorted(glob.glob(os.path.join(d, "facts.round*.jsonl")))
    for path in files:  # earlier rounds first; later overwrite
        for obj in _iter_jsonl(path):
            if (obj or {}).get("kind") != "bool":
                continue
            base = _norm_name((obj or {}).get("base_var"))
            tf   = _norm_name((obj or {}).get("tf_token"))
            if not base or not tf:
                continue

            vraw = (obj or {}).get("value")
            if isinstance(vraw, (int, float)): val = 1 if int(vraw) != 0 else 0
            elif isinstance(vraw, bool):       val = 1 if vraw else 0
            else:                               continue

            # exact key
            exact_key = _norm_name(f"{base}_{tf}")
            if exact_key:
                try:
                    cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) "
                                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (exact_key, val))
                except Exception:
                    pass
                total += 1

            # alias keys（与原逻辑一致：同 timeframe 的 cross-stem + tail 归一化）
            for alias_key in _make_alias_keys_from_fact_key(base, tf):
                try:
                    cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) "
                                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (alias_key, val))
                except Exception:
                    pass
                total += 1
    return total


def _kv_lookup_bool(dbp: str, key: str) -> Optional[bool]:
    """
    STRICT lookup for exact inferred boolean:
      • exact key, as written in facts KV (or its deterministic aliases that were pre-inserted)
      • minimal convenience: if key lacks a timeframe token, try appending '_now' ONLY
    No heuristic matching at lookup. If not found, return None.
    """
    import sqlite3
    k = _norm_name(key)
    if not k:
        return None

    def variants(kstr: str):
        yield kstr
        if not TF_RX.search(kstr):
            yield kstr + "_now"

    conn = sqlite3.connect(dbp); cur = conn.cursor()
    try:
        for v in variants(k):
            cur.execute("SELECT v FROM kv WHERE k=? LIMIT 1", (v,))
            row = cur.fetchone()
            if row is not None:
                return bool(row[0])
    finally:
        conn.close()
    return None

def _ensure_union_kv_db(pid: str, side: str) -> str:
    """
    构建指定 patient/side 的 KV（name -> {1,0}）：
      0) Persisted facts（仅当前 side）
      1) Side implications union（IMP_OUT_ROOT/<pid>/<side>/canonical.enriched.union.jsonl）
      2) Side round-seed canonical（ROUND_SEED_ROOT/<pid>/<side>/canonical.jsonl）
      3) Diagnosis（side 无关，照旧并入）
      4) Side mirrors（容错，通常 1) 已覆盖）
    """
    import sqlite3

    def _to_bool_opt(v) -> Optional[int]:
        if isinstance(v, bool): return 1 if v else 0
        if isinstance(v, (int, float)): return 1 if int(v) != 0 else 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true","t","1","yes","y"): return 1
            if s in ("false","f","0","no","n"): return 0
        return None

    def _nm(obj):
        raw = obj.get("entity_variable_name") or obj.get("name")
        return _norm_name(raw)

    def _ingest_jsonl(path: str, cur) -> int:
        if not os.path.isfile(path): return 0
        n = 0
        for obj in _iter_jsonl(path):
            key = _nm(obj)
            if not key: continue
            val = _to_bool_opt(obj.get("extracted_value"))
            if val is None: continue
            try:
                cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) "
                            "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, val))
            except Exception:
                try:
                    cur.execute("UPDATE kv SET v=? WHERE k=?", (val, key))
                    if cur.rowcount == 0:
                        cur.execute("INSERT OR IGNORE INTO kv(k, v) VALUES (?, ?)", (key, val))
                except Exception:
                    pass
            n += 1
        return n

    tmpd = os.path.join(ENTAIL_SNAPSHOT_ROOT, pid, ".tmp", side); os.makedirs(tmpd, exist_ok=True)
    dbp = os.path.join(tmpd, "union_kv.sqlite")
    if os.path.isfile(dbp):  # already built for this run
        return dbp

    conn = sqlite3.connect(dbp); cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;"); cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v INTEGER)"); conn.commit()

    total = 0

    # 0) facts（严格当前 side）
    total += _ingest_patient_facts_into_kv_side(pid, side, cur)

    # 1) side union
    total += _ingest_jsonl(os.path.join(IMP_OUT_ROOT, pid, side, "canonical.enriched.union.jsonl"), cur)

    # 2) side round seed
    total += _ingest_jsonl(os.path.join(ROUND_SEED_ROOT, pid, side, "canonical.jsonl"), cur)

    # 3) diagnosis（side 无关）
    diag_dir = os.path.join(DIAG_ROOT_IN, pid)
    for fname in ("diagnosis.export.jsonl", "diagnosis.jsonl"):
        total += _ingest_jsonl(os.path.join(diag_dir, fname), cur)

    # 4) mirrors（容错）
    for root in (F2P_OUT_ROOT, F2OE_OUT_ROOT, P2F_OUT_ROOT, OE2F_OUT_ROOT, F2P_OTHER_OUT_ROOT):
        total += _ingest_jsonl(os.path.join(root, pid, side, "canonical.enriched.union.jsonl"), cur)

    conn.commit(); conn.close()
    return dbp



# =========================
# === SNAPSHOTS (dedicated)
# =========================

def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")

def _parse_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true","t","1","yes","y"): return True
        if s in ("false","f","0","no","n"): return False
    return None

def _guess_source_var(rec: Dict[str, Any]) -> Optional[str]:
    sv = rec.get("source_variable_name")
    if isinstance(sv, str) and sv.strip(): return sv.strip()
    dv = rec.get("derived_from_variable")
    if isinstance(dv, str) and dv.strip(): return dv.strip()
    return None  # explicit only

def _get_derived_var(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("new_variable_name") or rec.get("name") or rec.get("entity_variable_name")
    return v.strip() if isinstance(v, str) and v.strip() else None

def _entail_inspect_dir(stage_key: str, round_idx: int, inner_pass: int, pid: str, side: Optional[str]) -> str:
    parts = [ENTAIL_SNAPSHOT_ROOT, pid, f"round{round_idx:02d}", f"pass{inner_pass:02d}", stage_key]
    if side: parts.append(side)
    d = os.path.join(*parts); os.makedirs(d, exist_ok=True)
    print(f"[inspect] dir → {d}")
    return d

def _iter_imp_pairs_from_structured(struct_path: str) -> Iterable[Tuple[str,str,bool]]:
    try: doc = json.load(open(struct_path,"r",encoding="utf-8"))
    except Exception: return
    g = doc.get("groups") if isinstance(doc, dict) else None
    if not isinstance(g, dict): return
    for key in ("schema","timeframe"):
        rows = g.get(key) or []
        for r in rows:
            src = (r or {}).get("name")
            if not isinstance(src, str) or not src.strip(): continue
            for it in ((r or {}).get("implied") or []):
                dst = (it or {}).get("name")
                val = _parse_bool((it or {}).get("value"))
                if isinstance(dst,str) and dst.strip():
                    yield src.strip(), dst.strip(), (True if val is None else bool(val))

def _collect_stage_mapping_by_patient(stage_key: str, patients: List[str], sides: Optional[List[str]]=None
) -> Dict[tuple[str, Optional[str]], Dict[str, Dict[str, bool]]]:
    by_ps: Dict[tuple[str, Optional[str]], Dict[str, Dict[str, bool]]] = {}

    def ensure_src(pid: str, side: Optional[str], src: Optional[str]) -> Optional[Dict[str, bool]]:
        if not isinstance(src, str) or not src.strip(): return None
        key = (pid, side)
        m = by_ps.setdefault(key, {})
        return m.setdefault(src.strip(), {})

    def add_pair(pid: str, side: Optional[str], src: Optional[str], dst: Optional[str], val: Optional[bool]):
        dstv = (dst or "").strip()
        if not dstv:
            ensure_src(pid, side, src); return
        bucket = ensure_src(pid, side, src)
        if bucket is None: return
        bucket[dstv] = True if val is None else bool(val)

    stage_to_file = {
        "isa":     (ISA_OUT_ROOT, "isa_enriched.flat.json"),
        "imp":     (IMP_OUT_ROOT, "canonical.enriched.implications.structured.json"),
        "f2p":     (F2P_OUT_ROOT, "finding_to_procedure_enrichment.flat.json"),
        "f2oe":    (F2OE_OUT_ROOT,"finding_to_observable_entity_enrichment.flat.json"),
        "p2f":     (P2F_OUT_ROOT, "procedure_to_finding_enrichment.flat.json"),
        "oe2f":    (OE2F_OUT_ROOT,"observable_entity_to_finding_enrichment.flat.json"),
        "f2p_rel": (F2P_OTHER_OUT_ROOT,"finding_to_procedure_via_relations.flat.json"),
    }
    stage_is_side = stage_key in stage_to_file
    use_sides = sides or (RUN_SIDES if stage_is_side else RUN_SIDES)  # isa/imp 也按侧

    if stage_key == "imp":
        for pid in patients:
            for side in use_sides:
                struct_path = os.path.join(IMP_OUT_ROOT, pid, side, "canonical.enriched.implications.structured.json")
                for src, dst, val in _iter_imp_pairs_from_structured(struct_path):
                    add_pair(pid, side, src, dst, val)
                try: doc = json.load(open(struct_path, "r", encoding="utf-8"))
                except Exception: doc = None
                g = (doc or {}).get("groups") if isinstance(doc, dict) else None
                if isinstance(g, dict):
                    for key in ("schema", "timeframe"):
                        for r in (g.get(key) or []):
                            ensure_src(pid, side, (r or {}).get("name"))
        return by_ps

    # 统一处理（包括 isa 与下游各阶段）
    root, out_fname = stage_to_file.get(stage_key, (None, None))
    if not root: return by_ps
    for pid in patients:
        for side in use_sides:
            if stage_key == "isa":
                p = os.path.join(root, pid, side, out_fname)
                arr = _read_json_list(p)
                if isinstance(arr, list) and arr:
                    for rec in arr:
                        if not isinstance(rec, dict): continue
                        src = _guess_source_var(rec)
                        dst = _get_derived_var(rec)
                        val = _parse_bool(rec.get("extracted_value"))
                        add_pair(pid, side, src, dst, val)
                else:
                    seed = os.path.join(ROUND_SEED_ROOT, pid, side, "canonical.jsonl")
                    for obj in _iter_jsonl(seed):
                        nm = obj.get("entity_variable_name") or obj.get("name")
                        if isinstance(nm, str) and nm.strip():
                            ensure_src(pid, side, nm)
            else:
                p = os.path.join(root, pid, side, out_fname)
                arr = _read_json_list(p)
                produced_any = False
                if isinstance(arr, list) and arr:
                    for rec in arr:
                        if not isinstance(rec, dict): continue
                        src = _guess_source_var(rec)
                        dst = _get_derived_var(rec)
                        val = _parse_bool(rec.get("extracted_value"))
                        add_pair(pid, side, src, dst, val)
                        produced_any = True
                if not produced_any:
                    union = os.path.join(IMP_OUT_ROOT, pid, side, "canonical.enriched.union.jsonl")
                    for obj in _iter_jsonl(union):
                        nm = obj.get("entity_variable_name") or obj.get("name")
                        if isinstance(nm, str) and nm.strip():
                            ensure_src(pid, side, nm)
    return by_ps


def _precreate_snapshot_dirs(stage_key: str, round_idx: int, inner_pass: int,
                             patients: List[str], sides: Optional[List[str]] = None):
    use_sides = sides or RUN_SIDES
    for pid in patients:
        for side in use_sides:
            _entail_inspect_dir(stage_key, round_idx, inner_pass, pid, side)

def _write_stage_entail_snapshot(stage_key: str, round_idx: int, inner_pass: int,
                                 patients: List[str], sides: Optional[List[str]] = None):
    print(f"[inspect] impl={SNAPSHOT_IMPL_VERSION} stage={stage_key} begin")
    use_sides = sides or RUN_SIDES
    discovered = _collect_stage_mapping_by_patient(stage_key, patients, use_sides)

    expected_slices = len(patients) * len(use_sides)
    total_rows = 0

    for pid in patients:
        for side in use_sides:
            kv_path = _ensure_union_kv_db(pid, side)
            mapping = discovered.get((pid, side), {})
            d = _entail_inspect_dir(stage_key, round_idx, inner_pass, pid, side)
            mp = os.path.join(d, "mapping.jsonl")

            rows = 0
            with open(mp, "w", encoding="utf-8") as fp:
                for src, tgt_map in mapping.items():
                    src_val = _kv_lookup_bool(kv_path, src)
                    if src_val is None and SNAPSHOT_DEFAULT_FALSE:
                        src_val = False
                    entailed_list = [{"variable": t, "value": bool(v)} for t, v in sorted(tgt_map.items())]
                    fp.write(json.dumps(
                        {"source_variable_name": src, "source_value": src_val, "entailed": entailed_list},
                        ensure_ascii=False
                    ) + "\n")
                    rows += 1; total_rows += 1
                if rows == 0:
                    src_val = False if SNAPSHOT_DEFAULT_FALSE else None
                    fp.write(json.dumps({"source_variable_name": None, "source_value": src_val, "entailed": []},
                                        ensure_ascii=False) + "\n")

            with open(os.path.join(d, "info.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "stage": stage_key, "round": round_idx, "pass": inner_pass,
                    "patient": pid, "side": side, "rows": rows,
                    "timestamp": _now_stamp(), "impl_version": SNAPSHOT_IMPL_VERSION,
                    "snapshot_default_false": SNAPSHOT_DEFAULT_FALSE,
                }, f, ensure_ascii=False, indent=2)

    print(f"[inspect] stage={stage_key} wrote {total_rows} structured rows across {expected_slices} patient-slices")

# =========================
# ===== Substage runners ==
# =========================

def run_isa_all_from_shadow(patients: List[str]) -> None:
    if not patients: return
    print("\n=== ISA stage (from side round seed canonical) ===")
    tot_r=tot_w=tot_s=0
    for pid in patients:
        for side in RUN_SIDES:
            pin  = os.path.join(ROUND_SEED_ROOT, pid, side)
            pout = os.path.join(ISA_OUT_ROOT,    pid, side)
            os.makedirs(pout, exist_ok=True)
            r,w,s = isa_mod.process_patient_dir(pin, pout)
            tot_r+=r; tot_w+=w; tot_s+=s
            st = f"ok (wrote {w}/{r}; skipped {s} non-bool)" if r>0 else "skipped (empty)"
            print(f"[patient] {pid:<24} [{side}] {st}")
    print(f"[done][ISA] read={tot_r} wrote={tot_w} skipped_nonbool={tot_s}")


def run_implications_fixpoint_once(patients: Optional[List[str]]=None) -> Tuple[int,int,int]:
    rules_doc, tf_cfg, src = _compat_load_impl_rules()
    rules = rules_doc.get("rules", [])
    rin = os.path.abspath(getattr(impl_mod, "ROOT_IN", IMP_IN_ROOT))   # 逻辑上仍指向 ISA_OUT_ROOT，但我们按侧传 pin/pout
    rout= os.path.abspath(getattr(impl_mod, "ROOT_OUT", IMP_OUT_ROOT))
    if not os.path.isdir(ISA_OUT_ROOT):
        print(f"[err] ISA out root not found: {ISA_OUT_ROOT}"); return (0,0,0)
    os.makedirs(rout, exist_ok=True)
    run_list = patients if patients else _list_patients_under(ISA_OUT_ROOT)
    if not run_list:
        print("[Implications] nothing changed; skipping"); return (0,0,0)

    print(f"[Implications] rules source: {src}; rules={len(rules)}")
    tot_rows=tot_schema=tot_tf=0
    for pid in run_list:
        for side in RUN_SIDES:
            pin  = os.path.join(ISA_OUT_ROOT, pid, side)
            pout = os.path.join(IMP_OUT_ROOT, pid, side)
            os.makedirs(pout, exist_ok=True)
            rows=sch=tf=0
            if rules:
                rows,sch,tf = _impl_process_patient_compat(pin, pout, rules, tf_cfg)
            if rows==sch==tf==0 and len(rules)==0:
                rows,sch,tf = _implications_passthrough(pin, pout)

            expanded_n, union_n = _build_implication_union_for_patient(pid, side)
            tot_rows+=rows; tot_schema+=sch; tot_tf+=tf
            status = (f"ok (schema_new_rc={sch}, timeframe_new_rc={tf}; expanded={expanded_n}; union_rows={union_n})"
                      if rows or sch or tf else f"passthrough (expanded={expanded_n}; union_rows={union_n})")
            print(f"[patient] {pid:<24} [{side}] {status}")

    print(f"[done][Implications] rows={tot_rows} schema_new_recursive={tot_schema} timeframe_new_recursive={tot_tf}")
    return (tot_rows,tot_schema,tot_tf)


# =========================
# ====== Pruning (keep) ===
# =========================

def _prune_step_outputs(stage: str, patients: List[str], sides: Optional[List[str]]=None):
    root_map = {
        "isa": ISA_OUT_ROOT, "imp": IMP_OUT_ROOT, "f2p": F2P_OUT_ROOT,
        "f2oe": F2OE_OUT_ROOT, "p2f": P2F_OUT_ROOT, "oe2f": OE2F_OUT_ROOT, "f2p_rel": F2P_OTHER_OUT_ROOT,
    }
    keep_map = {
        "isa": {"inclusion": {"canonical.enriched.jsonl", "isa_enriched.flat.json"},
                "exclusion": {"canonical.enriched.jsonl", "isa_enriched.flat.json"}},
        "imp": {"inclusion": {"canonical.enriched.union.jsonl", "canonical.enriched.implications.structured.json"},
                "exclusion": {"canonical.enriched.union.jsonl", "canonical.enriched.implications.structured.json"}},

        # ↓ 只保留产出，不再保留 union（union 是“输入”，且在 IMP_OUT_ROOT）
        "f2p": {"inclusion": {"finding_to_procedure_enrichment.flat.json", "finding_to_procedure_enrichment.report.json"},
                "exclusion": {"finding_to_procedure_enrichment.flat.json", "finding_to_procedure_enrichment.report.json"}},
        "f2oe": {"inclusion": {"finding_to_observable_entity_enrichment.flat.json", "finding_to_observable_entity_enrichment.report.json"},
                "exclusion": {"finding_to_observable_entity_enrichment.flat.json", "finding_to_observable_entity_enrichment.report.json"}},
        "p2f": {"inclusion": {"procedure_to_finding_enrichment.flat.json", "procedure_to_finding_enrichment.report.json"},
                "exclusion": {"procedure_to_finding_enrichment.flat.json", "procedure_to_finding_enrichment.report.json"}},
        "oe2f": {"inclusion": {"observable_entity_to_finding_enrichment.flat.json", "observable_entity_to_finding_enrichment.report.json"},
                "exclusion": {"observable_entity_to_finding_enrichment.flat.json", "observable_entity_to_finding_enrichment.report.json"}},
        "f2p_rel": {"inclusion": {"finding_to_procedure_via_relations.flat.json", "finding_to_procedure_via_relations.report.json"},
                    "exclusion": {"finding_to_procedure_via_relations.flat.json", "finding_to_procedure_via_relations.report.json"}},
    }


    root = root_map.get(stage)
    if not root or not os.path.isdir(root): return
    use_sides = sides or RUN_SIDES

    for pid in patients:
        for side in use_sides:
            d = os.path.join(root, pid, side)
            if not os.path.isdir(d): continue
            keep = keep_map[stage][side]
            for fn in os.listdir(d):
                if fn not in keep:
                    fp = os.path.join(d, fn)
                    try:
                        if os.path.isfile(fp): os.remove(fp)
                    except Exception: pass


def _archive_stage_outputs(stage: str, patients: List[str], round_idx: int, inner_pass: int,
                           archive_root: Optional[str], mode: str, sides: Optional[List[str]] = None) -> None:
    if not archive_root: return
    root_map = {
        "isa": ISA_OUT_ROOT, "imp": IMP_OUT_ROOT, "f2p": F2P_OUT_ROOT,
        "f2oe": F2OE_OUT_ROOT, "p2f": P2F_OUT_ROOT, "oe2f": OE2F_OUT_ROOT, "f2p_rel": F2P_OTHER_OUT_ROOT,
    }
    out_root = root_map.get(stage)
    if not out_root: return
    use_sides = sides or RUN_SIDES

    def _copy_file(src: str, dst: str):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if mode == "link":
            try:
                if os.path.exists(dst): return
                os.link(src, dst); return
            except Exception:
                pass
        import shutil
        shutil.copy2(src, dst)

    for pid in patients:
        for side in use_sides:
            src_dir = os.path.join(out_root, pid, side)
            if not os.path.isdir(src_dir): continue
            dst_dir = os.path.join(archive_root, pid, f"round{round_idx:02d}", f"pass{inner_pass:02d}", stage, side)
            for base, _, files in os.walk(src_dir):
                for fn in files:
                    src_fp = os.path.join(base, fn)
                    rel    = os.path.relpath(src_fp, src_dir)
                    dst_fp = os.path.join(dst_dir, rel)
                    try:
                        _copy_file(src_fp, dst_fp)
                    except Exception:
                        pass



# =========================
# ===== Substage runners ==
# =========================

def run_finding_to_procedure(side: str, patients: Optional[List[str]]=None) -> Tuple[int,int,int]:
    if not patients: return (0,0,0)
    print(f"\n=== Finding→Procedure [{side}] ===")
    os.makedirs(F2P_OUT_ROOT, exist_ok=True)
    f2p_mod.process_side(side, patients, "canonical.enriched.union.jsonl")
    for pid in patients:
        _ensure_provenance_fields("f2p", F2P_OUT_ROOT, pid, side, "finding_to_procedure_enrichment.flat.json")
    rwo=tot=conf=0
    for pid in patients:
        rep = os.path.join(F2P_OUT_ROOT, pid, side, "finding_to_procedure_enrichment.report.json")
        if not os.path.isfile(rep): continue
        c = (json.load(open(rep,"r",encoding="utf-8")) or {}).get("counts",{})
        rwo += int(c.get("rows_with_produced",0)); tot += int(c.get("produced_variables",0)); conf += int(c.get("conflicted_rows",0))
    print(f"[done][F→P:{side}] rows_with_output={rwo} produced_variables={tot} conflicts={conf}")
    return rwo,tot,conf

def run_finding_to_observable_entity(side: str, patients: Optional[List[str]]=None) -> Tuple[int,int,int]:
    if not patients: return (0,0,0)
    print(f"\n=== Finding→Observable-Entity [{side}] ===")
    os.makedirs(F2OE_OUT_ROOT, exist_ok=True)
    f2oe_mod.process_side(side, patients, "canonical.enriched.union.jsonl")
    for pid in patients:
        _ensure_provenance_fields("f2oe", F2OE_OUT_ROOT, pid, side, "finding_to_observable_entity_enrichment.flat.json")
    rwo=tot=conf=0
    for pid in patients:
        rep = os.path.join(F2OE_OUT_ROOT, pid, side, "finding_to_observable_entity_enrichment.report.json")
        if not os.path.isfile(rep): continue
        c = (json.load(open(rep,"r",encoding="utf-8")) or {}).get("counts",{})
        rwo += int(c.get("rows_with_produced",0)); tot += int(c.get("produced_variables",0)); conf += int(c.get("conflicted_rows",0))
    print(f"[done][F→OE:{side}] rows_with_output={rwo} produced_variables={tot} conflicts={conf}")
    return rwo,tot,conf

def run_finding_to_procedure_via_relations(side: str, patients: Optional[List[str]]=None) -> Tuple[int,int]:
    if not patients: return (0,0)
    print(f"\n=== Finding→Procedure via relations [{side}] ===")
    os.makedirs(F2P_OTHER_OUT_ROOT, exist_ok=True)
    f2p_rel_mod.process_side(side, patients, "canonical.enriched.annotated.jsonl")
    for pid in patients:
        _ensure_provenance_fields("f2p_rel", F2P_OTHER_OUT_ROOT, pid, side, "finding_to_procedure_via_relations.flat.json", rule_hint_key="relation")
    rwo=tot=0
    for pid in patients:
        rep = os.path.join(F2P_OTHER_OUT_ROOT, pid, side, "finding_to_procedure_via_relations.report.json")
        if not os.path.isfile(rep): continue
        c = (json.load(open(rep,"r",encoding="utf-8")) or {}).get("counts",{})
        rwo += int(c.get("rows_with_produced",0)); tot += int(c.get("produced_variables",0))
    print(f"[done][F→P_rel:{side}] rows_with_output={rwo} produced_variables={tot}")
    return rwo,tot

def run_procedure_to_finding(side: str, patients: Optional[List[str]]=None) -> Tuple[int,int]:
    if not patients: return (0,0)
    print(f"\n=== Procedure→Finding [{side}] ===")
    os.makedirs(P2F_OUT_ROOT, exist_ok=True)
    p2f_mod.process_side(side, patients, "finding_to_procedure_enrichment.flat.json")
    for pid in patients:
        _ensure_provenance_fields("p2f", P2F_OUT_ROOT, pid, side, "procedure_to_finding_enrichment.flat.json")
    rwo=tot=0
    for pid in patients:
        rep = os.path.join(P2F_OUT_ROOT, pid, side, "procedure_to_finding_enrichment.report.json")
        if not os.path.isfile(rep): continue
        c = (json.load(open(rep,"r",encoding="utf-8")) or {}).get("counts",{})
        rwo += int(c.get("rows_with_produced",0)); tot += int(c.get("produced_variables",0))
    print(f"[done][P→F:{side}] rows_with_output={rwo} produced_variables={tot}")
    return rwo,tot

def run_observable_entity_to_finding(side: str, patients: Optional[List[str]]=None) -> Tuple[int,int]:
    if not patients: return (0,0)
    print(f"\n=== Observable-Entity→Finding [{side}] ===")
    os.makedirs(OE2F_OUT_ROOT, exist_ok=True)
    oe2f_mod.process_side(side, patients, "finding_to_observable_entity_enrichment.flat.json")
    for pid in patients:
        _ensure_provenance_fields("oe2f", OE2F_OUT_ROOT, pid, side, "observable_entity_to_finding_enrichment.flat.json")
    rwo=tot=0
    for pid in patients:
        rep = os.path.join(OE2F_OUT_ROOT, pid, side, "observable_entity_to_finding_enrichment.report.json")
        if not os.path.isfile(rep): continue
        c = (json.load(open(rep,"r",encoding="utf-8")) or {}).get("counts",{})
        rwo += int(c.get("rows_with_produced",0)); tot += int(c.get("produced_variables",0))
    print(f"[done][OE→F:{side}] rows_with_output={rwo} produced_variables={tot}")
    return rwo,tot

# =========================
# ====== Main driver ======
# =========================

def _persist_and_log(pid: str, round_idx: int, stage_roots: dict[str,str]) -> None:
    """Persist facts for the patient before taking a snapshot so KV sees current booleans."""
    wrote = persist_all_facts([pid], stage_roots, PERSIST_FROM, PERSIST_DB_PATH, PERSIST_FS_ROOT, round_idx)
    print(f"[persist][{pid}] round {round_idx}: wrote {wrote} facts (pre-snapshot)")

def main():
    global SNAPSHOT_DEFAULT_FALSE
    parser = argparse.ArgumentParser(description="Patient-by-patient enrichment fixpoint with dedicated entailment snapshots + provenance (STRICT booleans, SIDE-ISOLATED).")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--max-ram-gb", type=float, default=float(os.environ.get("FIXPOINT_MAX_RAM_GB","0")))
    parser.add_argument("--skip-reverse", action="store_true")
    parser.add_argument("--skip-rel", action="store_true")
    parser.add_argument("--single-output", action="store_true", default=True)
    parser.add_argument("--snapshot-default-false", action="store_true")
    parser.add_argument("--archive-root", type=str, default=os.environ.get("ARCHIVE_ROOT", ""))
    parser.add_argument("--archive-mode", type=str, choices=["copy","link"], default="copy")

    args = parser.parse_args()
    SNAPSHOT_DEFAULT_FALSE = bool(args.snapshot_default_false)

    print(f"[boot] running {__file__}")
    print(f"[boot] snapshot impl = {SNAPSHOT_IMPL_VERSION}")
    print(f"[boot] snapshot_default_false = {SNAPSHOT_DEFAULT_FALSE}")

    _print_roots(args)

    candidates = _list_patients_under(ISA_IN_ROOT)
    if not candidates:
        print("[warn] no patients under ISA_IN_ROOT"); return
    patients_all = _filter_patients(candidates, args.patient)
    if args.patient and not patients_all:
        print(f"[warn] patient '{args.patient}' not found"); return

    stage_roots = {
        "demo": DEMO_ROOT, "diag": DIAG_ROOT_IN, "isa_imp": IMP_OUT_ROOT,
        "proc": F2P_OUT_ROOT, "obs": F2OE_OUT_ROOT, "p2f": P2F_OUT_ROOT, "oe2f": OE2F_OUT_ROOT, "proc_rel": F2P_OTHER_OUT_ROOT, "isa": ISA_OUT_ROOT,
    }

    for round_idx in range(1, GLOBAL_MAX_ROUNDS+1):
        print("\n\n==========================")
        print(f" Global round {round_idx} (patient-by-patient, side-isolated)")
        print("==========================")
        for pid in patients_all:
            print(f"\n— Patient {pid} —")
            for inner_pass in range(1, INNER_MAX_PASSES+1):
                if args.max_ram_gb and not mem_ok(args.max_ram_gb):
                    print("  [guard] memory high → break"); break

                # 逐侧构建种子，记录哪些侧发生变化
                changed = _build_round_seed_canonical([pid], round_idx)  # 返回 {(pid, side)}
                changed_sides = [s for (p,s) in changed if p == pid]

                if not changed_sides:
                    print("  [seed] both sides unchanged → skip stages"); break

                # ISA（仅对发生变化的侧运行）
                run_needed_sides = changed_sides
                print(f"  [isa] run sides: {run_needed_sides}")
                # 为简洁复用 run_isa_all_from_shadow（它会对两侧都跑），这里手动只跑需要的侧：
                for side in run_needed_sides:
                    pin  = os.path.join(ROUND_SEED_ROOT, pid, side)
                    pout = os.path.join(ISA_OUT_ROOT,    pid, side)
                    os.makedirs(pout, exist_ok=True)
                    r,w,s = isa_mod.process_patient_dir(pin, pout)
                    st = f"ok (wrote {w}/{r}; skipped {s} non-bool)" if r>0 else "skipped (empty)"
                    print(f"  [isa:{side}] {pid:<24} {st}")

                _archive_stage_outputs("isa", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                if args.single_output: _prune_step_outputs("isa", [pid], RUN_SIDES)
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("isa", round_idx, inner_pass, [pid], RUN_SIDES)
                _write_stage_entail_snapshot("isa", round_idx, inner_pass, [pid], RUN_SIDES)

                # Implications（两侧都按侧跑；是否只跑 changed_sides 可按需裁剪）
                run_implications_fixpoint_once([pid])
                _archive_stage_outputs("imp", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                if args.single_output: _prune_step_outputs("imp", [pid], RUN_SIDES)
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("imp", round_idx, inner_pass, [pid], RUN_SIDES)
                _write_stage_entail_snapshot("imp", round_idx, inner_pass, [pid], RUN_SIDES)

                # Mirror union → 各 side 下游 IN_ROOT
                _mirror_union_into_side_trees([pid], F2P_IN_ROOT)
                _mirror_union_into_side_trees([pid], F2OE_IN_ROOT)
                _mirror_union_into_side_trees([pid], P2F_IN_ROOT)
                _mirror_union_into_side_trees([pid], OE2F_IN_ROOT)
                _mirror_union_into_side_trees([pid], F2P_OTHER_IN_ROOT)

                # Forward: F→P / F→OE（两侧）
                for s in RUN_SIDES: run_finding_to_procedure(s, [pid])
                _archive_stage_outputs("f2p", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                if args.single_output: _prune_step_outputs("f2p", [pid], RUN_SIDES)
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("f2p", round_idx, inner_pass, [pid], RUN_SIDES)
                _write_stage_entail_snapshot("f2p", round_idx, inner_pass, [pid], RUN_SIDES)

                for s in RUN_SIDES: run_finding_to_observable_entity(s, [pid])
                _archive_stage_outputs("f2oe", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                if args.single_output: _prune_step_outputs("f2oe", [pid], RUN_SIDES)
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("f2oe", round_idx, inner_pass, [pid], RUN_SIDES)
                _write_stage_entail_snapshot("f2oe", round_idx, inner_pass, [pid], RUN_SIDES)

                # Forward: F→P_rel（可选）
                if not args.skip_rel:
                    for s in RUN_SIDES: run_finding_to_procedure_via_relations(s, [pid])
                    _archive_stage_outputs("f2p_rel", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                    if args.single_output: _prune_step_outputs("f2p_rel", [pid], RUN_SIDES)
                    _persist_and_log(pid, round_idx, stage_roots)
                    _precreate_snapshot_dirs("f2p_rel", round_idx, inner_pass, [pid], RUN_SIDES)
                    _write_stage_entail_snapshot("f2p_rel", round_idx, inner_pass, [pid], RUN_SIDES)

                # Reverse（可选）
                if not args.skip_reverse:
                    for s in RUN_SIDES: run_procedure_to_finding(s, [pid])
                    _archive_stage_outputs("p2f", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                    if args.single_output: _prune_step_outputs("p2f", [pid], RUN_SIDES)
                    _persist_and_log(pid, round_idx, stage_roots)
                    _precreate_snapshot_dirs("p2f", round_idx, inner_pass, [pid], RUN_SIDES)
                    _write_stage_entail_snapshot("p2f", round_idx, inner_pass, [pid], RUN_SIDES)

                    for s in RUN_SIDES: run_observable_entity_to_finding(s, [pid])
                    _archive_stage_outputs("oe2f", [pid], round_idx, inner_pass, args.archive_root, args.archive_mode, RUN_SIDES)
                    if args.single_output: _prune_step_outputs("oe2f", [pid], RUN_SIDES)
                    _persist_and_log(pid, round_idx, stage_roots)
                    _precreate_snapshot_dirs("oe2f", round_idx, inner_pass, [pid], RUN_SIDES)
                    _write_stage_entail_snapshot("oe2f", round_idx, inner_pass, [pid], RUN_SIDES)

                if INNER_MAX_PASSES == 1: break

        print(f"\n[fixpoint] Finished round {round_idx} (patient-by-patient, side-isolated).")

    wrote_final = persist_all_facts(patients_all, stage_roots, PERSIST_FROM, PERSIST_DB_PATH, PERSIST_FS_ROOT, 9999)
    print(f"[persist] final: wrote {wrote_final} facts")
    print("\n[done] End-to-end driver finished.")


if __name__ == "__main__":
    main()