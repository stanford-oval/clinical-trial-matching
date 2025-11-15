#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_enrichment_no_isa.py
STRICT booleans in snapshots + alias-projection ingestion + qualifier stripping
────────────────────────────────────────────────────────────────────────────
No-ISA variant (updated to match script2 behaviors):
  • Skips the ISA stage entirely (no is-a entailments, no ISA snapshots)
  • Uses round-seed canonical as the starting point
  • Ensures per-patient seed has a minimal canonical.enriched.jsonl (passthrough)
  • Persists facts to a different root: patient_facts_export_noisa
  • **All outputs are written to folders with suffix `_noisa` to avoid confusion**
  • All other stages unchanged: imp, f2p, f2oe, f2p_rel, p2f, oe2f

What’s new vs the older no-ISA runner (parity with script2):
  • Snapshot KV uses STRICT lookup keyed by full normalized variable names
    (no timeframe split or fallback), with deterministic alias projection.
  • Time-window metadata (start/end/inclusive flags) is carried through:
      - flattened implications,
      - persistence export, and
      - round-seed seeds passthrough.
  • Implication union building streams & dedupes on-disk (sqlite spillset).
  • Qualifier-insensitive names (strip trailing @@qualifiers in normalization).
  • Persist facts BEFORE each snapshot so KV sees current booleans.
  • Optional CLOSED-WORLD snapshot default (fill missing with False):
        --snapshot-default-false

Stages executed: imp, f2p, f2oe, f2p_rel, p2f, oe2f
"""

from __future__ import annotations
import os, sys, json, re, argparse, datetime as dt
from typing import List, Tuple, Optional, Iterable, Dict, Any, Set

# =========================
# ======= CONFIG ==========
# =========================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Inputs/Outputs (stage roots)
ISA_IN_ROOT   = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results"))  # seed source

# Implications read directly from ROUND_SEED_ROOT (we ensure enriched passthrough)
IMP_IN_ROOT   = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_round_seeds_noisa"))
IMP_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_isa_imp_noisa"))

F2P_IN_ROOT   = IMP_OUT_ROOT
F2P_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_finding_to_procedure_enrichment_noisa"))

# Chain inputs similarly to script2 (both read mirrored union)
F2OE_IN_ROOT  = F2P_IN_ROOT
F2OE_OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_finding_to_observable_entity_enrichment_noisa"))

P2F_IN_ROOT   = F2P_OUT_ROOT
P2F_OUT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_procedure_to_finding_enrichment_noisa"))

OE2F_IN_ROOT  = F2OE_IN_ROOT
OE2F_OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_observable_entity_to_finding_enrichment_noisa"))

F2P_OTHER_IN_ROOT  = IMP_OUT_ROOT
F2P_OTHER_OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_finding_to_procedure_enrichment_other_noisa"))

ENTAIL_OUT_ROOT    = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_entailments_noisa"))
ROUND_SEED_ROOT    = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_coded_results_round_seeds_noisa"))

# Demographics/Diagnosis (side-agnostic)
DEMO_ROOT     = ISA_IN_ROOT
DIAG_ROOT_IN  = (os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_build/patient_diagnosis_coded_results"))
                 if os.path.isdir(os.path.join(THIS_DIR, "../../patient_build/patient_build/patient_diagnosis_coded_results"))
                 else os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_diagnosis_coded_results")))

# Fixpoint/passes
GLOBAL_MAX_ROUNDS = 1
INNER_MAX_PASSES  = 3
RUN_SIDES = ["inclusion", "exclusion"]

# FS-only persistence
PERSIST_DB_PATH   = None
PERSIST_FS_ROOT   = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/patient_facts_export_noisa"))

# Include round-seed canonical in facts export (this was missing before)
PERSIST_FROM      = ["demo", "diag", "seed", "isa_imp", "proc", "obs", "p2f", "oe2f", "proc_rel"]

# DEDICATED entailment snapshot root (per patient/stage)
SNAPSHOT_ROOT          = os.path.abspath(os.path.join(THIS_DIR, "../../patient_build/snapshots_noisa"))
ENTAIL_SNAPSHOT_ROOT   = os.path.join(SNAPSHOT_ROOT, "entail_inspect")  # dedicated

# Version banner for logs
SNAPSHOT_IMPL_VERSION = "entail-snapshot-v11-strict-alias-qual-strip-prepersist-noisa-2025-10-24"

# Snapshot behavior flag (set by CLI)
SNAPSHOT_DEFAULT_FALSE = False  # closed-world fill for missing booleans (snapshot-only)

# =========================
# ===== Imports ===========
# =========================

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Stage modules
import apply_schema_implications as impl_mod
import enrich_findings_to_procedure as f2p_mod
import enrich_findings_to_observable_entity as f2oe_mod
import enrich_procedure_to_finding as p2f_mod
import enrich_observable_entity_to_finding as oe2f_mod
import enrich_findings_to_procedure_via_other_relations as f2p_rel_mod

# Bind module roots (NO-ISA ONLY OUTPUTS)
impl_mod.ROOT_IN  = IMP_IN_ROOT
impl_mod.ROOT_OUT = IMP_OUT_ROOT

f2p_mod.IN_ROOT   = F2P_IN_ROOT
f2p_mod.OUT_ROOT  = F2P_OUT_ROOT

f2oe_mod.IN_ROOT  = F2OE_IN_ROOT
f2oe_mod.OUT_ROOT = F2OE_OUT_ROOT

p2f_mod.IN_ROOT   = P2F_IN_ROOT
p2f_mod.OUT_ROOT  = P2F_OUT_ROOT

oe2f_mod.IN_ROOT  = OE2F_IN_ROOT
oe2f_mod.OUT_ROOT = OE2F_OUT_ROOT

f2p_rel_mod.IN_ROOT  = F2P_OTHER_IN_ROOT
f2p_rel_mod.OUT_ROOT = F2P_OTHER_OUT_ROOT

def _assert_noisa_outputs():
    """Abort if any stage is configured to write to a non-_noisa path."""
    outs = [
        ("impl.ROOT_OUT", getattr(impl_mod, "ROOT_OUT", "")),
        ("f2p.OUT_ROOT", getattr(f2p_mod, "OUT_ROOT", "")),
        ("f2oe.OUT_ROOT", getattr(f2oe_mod, "OUT_ROOT", "")),
        ("p2f.OUT_ROOT", getattr(p2f_mod, "OUT_ROOT", "")),
        ("oe2f.OUT_ROOT", getattr(oe2f_mod, "OUT_ROOT", "")),
        ("f2p_rel.OUT_ROOT", getattr(f2p_rel_mod, "OUT_ROOT", "")),
    ]
    bad = [(k, v) for k, v in outs if not str(v).endswith("_noisa")]
    if bad:
        raise RuntimeError(f"Non-_noisa output path detected: {bad}")

# Implication rules
ABS_RULES_PATH = "../rules/schema_implications.json"
impl_mod.RULES_PATH = ABS_RULES_PATH

# =========================
# == Helpers & Utilities ==
# =========================

def _print_roots(args):
    print("[roots]")
    print("  Seed in (ISA_IN_ROOT):", ISA_IN_ROOT)
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

# Accept legacy timeframe tokens in names
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
    """Preserve time-window fields from both sources and implied rows."""
    out: List[Dict[str, Any]] = []
    if not isinstance(doc, dict): return out
    gobj = doc.get("groups")
    if not isinstance(gobj, dict): return out

    TIME_META_KEYS = ("start_time_in_hours","end_time_in_hours",
                      "start_time_inclusive","end_time_inclusive")

    def _copy_time_meta(src: dict, dst: dict):
        for k in TIME_META_KEYS:
            if k in src:
                dst[k] = src[k]

    for key in ("schema", "timeframe"):
        rows = gobj.get(key)
        if not isinstance(rows, list): continue
        for r in rows:
            if not isinstance(r, dict): continue
            nm = r.get("name")
            if isinstance(nm, str) and nm:
                base = {
                    "entity_variable_name": nm,
                    "extracted_value": bool(r.get("extracted_value", True)),
                    "type": "Bool"
                }
                _copy_time_meta(r, base)
                out.append(base)
            for it in (r.get("implied") or []):
                if not isinstance(it, dict): continue
                nm2 = (it or {}).get("name")
                if isinstance(nm2, str) and nm2:
                    row = {
                        "entity_variable_name": nm2,
                        "extracted_value": bool((it or {}).get("value", True)),  # <-- fixed key
                        "type": (it or {}).get("type") or "Bool"
                    }
                    _copy_time_meta(it, row)
                    out.append(row)
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

def _build_implication_union_for_patient(pid: str) -> tuple[int,int]:
    """Build flat implications & union for a patient (NO ISA targets added here)."""
    pdir = os.path.join(IMP_OUT_ROOT, pid); os.makedirs(pdir, exist_ok=True)
    base_path   = os.path.join(pdir, "canonical.enriched.jsonl")
    struct_path = os.path.join(pdir, "canonical.enriched.implications.structured.json")
    flat_path   = os.path.join(pdir, "canonical.enriched.implications.flat.jsonl")
    union_path  = os.path.join(pdir, "canonical.enriched.union.jsonl")

    # flatten structured → jsonl (streaming)
    flat_rows = 0
    with open(flat_path, "w", encoding="utf-8") as fflat:
        if os.path.isfile(struct_path):
            try: doc = json.load(open(struct_path, "r", encoding="utf-8"))
            except Exception: doc = None
            for r in _flatten_implications_structured(doc or {}):
                fflat.write(json.dumps(r, ensure_ascii=False) + "\n")
                flat_rows += 1

    # union write with on-disk dedupe
    seen_db = os.path.join(pdir, ".tmp", "union_seen.sqlite"); os.makedirs(os.path.dirname(seen_db), exist_ok=True)
    conn = _spillset_open(seen_db); cur = conn.cursor()
    written = 0
    with open(union_path, "w", encoding="utf-8") as fout:
        # 1) base canonical.enriched.jsonl
        if os.path.isfile(base_path):
            for r in _iter_jsonl(base_path):
                nm = str(r.get("entity_variable_name") or r.get("name") or "").strip()
                if not nm: continue
                try:
                    cur.execute("INSERT INTO s(k) VALUES (?)", (nm,))
                    rec = {"entity_variable_name": nm, **{k:v for k,v in r.items() if k!='name'}}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                except Exception:
                    pass

        # 2) flat implications (jsonl)
        if os.path.isfile(flat_path):
            for r in _iter_jsonl(flat_path):
                nm = str(r.get("entity_variable_name") or "").strip()
                if not nm: continue
                try:
                    cur.execute("INSERT INTO s(k) VALUES (?)", (nm,))
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += 1
                except Exception:
                    pass

    conn.commit(); conn.close()
    return flat_rows, written

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
    if not pids or not os.path.isdir(IMP_OUT_ROOT): return
    for pid in pids:
        src = os.path.join(IMP_OUT_ROOT, pid, "canonical.enriched.union.jsonl")
        if not os.path.isfile(src): continue
        for side in ("inclusion","exclusion"):
            d = os.path.join(dest_root, pid, side); os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, "canonical.enriched.union.jsonl")
            with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                for ln in fin: fout.write(ln)

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

_NAME_RX_MULTI_US = re.compile(r"__+")
_QUAL_SUFFIX_RX   = re.compile(r"@@[a-z0-9_]+(?:@@[a-z0-9_]+)*$", re.IGNORECASE)

def _normalize_demo_varname(varname: str) -> str:
    if not isinstance(varname, str): return varname or ""
    return re.sub(r"^age_value_recorded_", "patient_age_value_recorded_", varname)

def _extract_time_window(rec: dict) -> Optional[dict]:
    s = rec.get("start_time_in_hours")
    e = rec.get("end_time_in_hours")
    si = rec.get("start_time_inclusive")
    ei = rec.get("end_time_inclusive")
    if isinstance(s, (int, float)) and isinstance(e, (int, float)):
        win = {
            "start_time_in_hours": float(s),
            "end_time_in_hours": float(e),
            "start_time_inclusive": bool(si) if isinstance(si, bool) else None,
            "end_time_inclusive": bool(ei) if isinstance(ei, bool) else None,
        }
        return {k: v for k, v in win.items() if v is not None}
    return None

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
        # Preserve window fields when yielding both source and implied rows
        for key in ("schema","timeframe"):
            for r in (gobj.get(key) or []):
                nm = (r or {}).get("name")
                if isinstance(nm,str) and nm:
                    row = {"name": nm, "extracted_value": bool(r.get("extracted_value", True))}
                    for k in ("start_time_in_hours","end_time_in_hours","start_time_inclusive","end_time_inclusive"):
                        if k in r: row[k] = r[k]
                    yield row
                for it in ((r or {}).get("implied") or []):
                    if not isinstance(it, dict): continue
                    nm2 = (it or {}).get("name")
                    if isinstance(nm2,str) and nm2:
                        row = {"name": nm2, "extracted_value": bool((it or {}).get("value", True))}
                        for k in ("start_time_in_hours","end_time_in_hours","start_time_inclusive","end_time_inclusive"):
                            if k in it: row[k] = it[k]
                        yield row

def _find_patient_files_for_root(root: str, pid: str) -> list[tuple[str,str,Optional[str]]]:
    out: list[tuple[str,str,Optional[str]]] = []
    ab = os.path.abspath(root)

    # Demographics
    if ab == os.path.abspath(DEMO_ROOT):
        p = os.path.join(root, pid, "demographics.jsonl")
        if os.path.isfile(p): out.append(("demo", p, None))
        return out

    # Diagnosis
    if ab == os.path.abspath(DIAG_ROOT_IN):
        p = os.path.join(root, pid, "diagnosis.export.jsonl")
        if os.path.isfile(p): out.append(("diag", p, None))
        else:
            p2 = os.path.join(root, pid, "diagnosis.jsonl")
            if os.path.isfile(p2): out.append(("diag", p2, None))
        return out

    # Round-seed canonical (NEW: make sure these are persisted)
    if ab == os.path.abspath(ROUND_SEED_ROOT):
        d = os.path.join(root, pid)
        for nm in ("canonical.enriched.jsonl", "canonical.jsonl"):
            p = os.path.join(d, nm)
            if os.path.isfile(p): out.append(("seed", p, None))
        return out

    # Side-based stage outputs
    for side in ("inclusion","exclusion"):
        d = os.path.join(root, pid, side)
        if not os.path.isdir(d): continue
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

    # Non-side files in stage roots (imp outputs)
    d = os.path.join(root, pid)
    for nm in ("canonical.enriched.union.jsonl","isa_enriched.flat.json","canonical.enriched.annotated.jsonl",
               "canonical.enriched.jsonl","canonical.jsonl"):
        p = os.path.join(d, nm)
        if os.path.isfile(p):
            if nm.startswith("isa_enriched"): out.append(("isa", p, None))
            else: out.append(("isa_imp", p, None))
    p_imp_struct = os.path.join(d, "canonical.enriched.implications.structured.json")
    if os.path.isfile(p_imp_struct): out.append(("isa_imp", p_imp_struct, None))
    return out

def persist_all_facts(patients: List[str], stage_roots: dict[str,str], persist_from: List[str],
                      persist_db_path: Optional[str], persist_fs_root: Optional[str], round_idx: int) -> int:
    """
    Persist authoritative facts (booleans & numbers) BEFORE each snapshot.
    Uses full normalized variable names (no timeframe split) and carries time windows.
    """
    total = 0
    if persist_fs_root: os.makedirs(persist_fs_root, exist_ok=True)

    # Preserve a deterministic order; include seed early so base facts are present
    ordered: List[tuple[str,str]] = []
    for tag in ("demo","diag","seed","isa_imp","proc","obs","p2f","oe2f","proc_rel"):
        if tag in persist_from and tag in stage_roots:
            ordered.append((tag, stage_roots[tag]))

    demo_count = 0; demo_patients: Set[str] = set()
    for pid in patients:
        seen: set[tuple[str,str,str]] = set()  # (var,kind,side)
        fh: dict[str, Optional[Any]] = {"inclusion": None, "exclusion": None}
        if persist_fs_root:
            for side in ("inclusion","exclusion"):
                pdir = os.path.join(persist_fs_root, pid, side); os.makedirs(pdir, exist_ok=True)
                fh[side] = open(os.path.join(pdir, f"facts.round{round_idx}.jsonl"), "w", encoding="utf-8")

        def _emit(side: str, varname: str, kind: str, value: float|int, src_tag: str,
                  *, win: Optional[dict], raw_var: Optional[str]):
            nonlocal total, demo_count
            key = (varname, kind, side)
            if key in seen: return
            seen.add(key)
            if fh.get(side):
                row = {
                    "patient_id": pid,
                    "entity_variable_name": varname,
                    "value": value,
                    "kind": kind,
                    "source": src_tag,
                    "side": side
                }
                if isinstance(raw_var, str) and raw_var:
                    row["source_variable_name"] = raw_var
                if win:
                    row.update(win)
                fh[side].write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

        try:
            for tag, root in ordered:
                if not os.path.isdir(root): continue
                for src_tag, path, side_hint in _find_patient_files_for_root(root, pid):
                    if src_tag not in persist_from: continue
                    sides_to_write = [side_hint] if side_hint in ("inclusion","exclusion") else ["inclusion","exclusion"]
                    for rec in _iter_records_file(path):
                        # Booleans with explicit extracted_value
                        if isinstance(rec.get("extracted_value"), bool):
                            varname = rec.get("entity_variable_name") or rec.get("name") or rec.get("new_variable_name")
                            if isinstance(varname, str) and varname:
                                varname_norm = _normalize_demo_varname(varname)
                                win = _extract_time_window(rec)
                                for s in sides_to_write:
                                    _emit(s, varname_norm, "bool", 1 if rec["extracted_value"] else 0,
                                          src_tag, win=win, raw_var=varname)
                                if src_tag == "demo": demo_count += 1; demo_patients.add(pid)
                                continue
                        # Numeric or presence-as-true
                        varname = None
                        for k in ("name","new_variable_name","entity_variable_name"):
                            v = rec.get(k)
                            if isinstance(v,str) and v:
                                varname = _normalize_demo_varname(v); break
                        if not varname:
                            pv = rec.get("produced_variables")
                            if isinstance(pv, list):
                                for it in pv:
                                    if isinstance(it, dict) and isinstance(it.get("name"), str):
                                        varname = _normalize_demo_varname(it["name"])
                                        break
                        if not varname: continue
                        win = _extract_time_window(rec)
                        ev = rec.get("value")
                        if isinstance(ev,(int,float)):
                            for s in sides_to_write:
                                _emit(s, varname, "num", float(ev), src_tag, win=win, raw_var=rec.get("entity_variable_name") or rec.get("name") or rec.get("new_variable_name"))
                        else:
                            ev2 = rec.get("extracted_value")
                            if isinstance(ev2, (int,float)):
                                for s in sides_to_write:
                                    _emit(s, varname, "num", float(ev2), src_tag, win=win, raw_var=rec.get("entity_variable_name") or rec.get("name") or rec.get("new_variable_name"))
                            else:
                                for s in sides_to_write:
                                    _emit(s, varname, "bool", 1, src_tag, win=win, raw_var=rec.get("entity_variable_name") or rec.get("name") or rec.get("new_variable_name"))
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

TIME_META_KEYS = ("start_time_in_hours", "end_time_in_hours",
                  "start_time_inclusive", "end_time_inclusive")

def _copy_time_meta(src: dict, dst: dict) -> None:
    for k in TIME_META_KEYS:
        if k in src:
            dst[k] = src[k]

def _extract_tf_token(name: str) -> Optional[str]:
    m = TF_RX.search(name or ""); return m.group(0) if m else None

def _iter_prev_seed_rows_for_patient(pid: str) -> Iterable[Dict[str, Any]]:
    def _add_from_flat(root: str, side: str, filename: str, cid_keys: List[str], term_keys: List[str]=[]) -> Iterable[Dict[str, Any]]:
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
            out = {"entity_variable_name": name,
                   "conceptId": cid,
                   "extracted_value": True,
                   "type": "Bool",
                   "timeframe": _extract_tf_token(name),
                   "fact_id": "seed_from_union"}
            _copy_time_meta(row, out)
            yield out

    for side in RUN_SIDES:
        yield from _add_from_flat(F2P_OUT_ROOT, side, "finding_to_procedure_enrichment.flat.json", ["derived_conceptId","procedureId"])
        yield from _add_from_flat(F2OE_OUT_ROOT, side, "finding_to_observable_entity_enrichment.flat.json", ["derived_conceptId","observableEntityId"])
        yield from _add_from_flat(P2F_OUT_ROOT, side, "procedure_to_finding_enrichment.flat.json", ["derived_findingId","findingId"])
        yield from _add_from_flat(OE2F_OUT_ROOT, side, "observable_entity_to_finding_enrichment.flat.json", ["derived_findingId","findingId"])
        yield from _add_from_flat(F2P_OTHER_OUT_ROOT, side, "finding_to_procedure_via_relations.flat.json", ["derived_conceptId","procedureId"])

    imp_struct = os.path.join(IMP_OUT_ROOT, pid, "canonical.enriched.implications.structured.json")
    if os.path.isfile(imp_struct):
        try: doc = json.load(open(imp_struct,"r",encoding="utf-8"))
        except Exception: doc = None
        flat = _flatten_implications_structured(doc) if isinstance(doc, dict) else []
        for it in flat:
            nm = it["entity_variable_name"]
            out = {"entity_variable_name": nm,
                   "conceptId": None,
                   "extracted_value": bool(it.get("extracted_value", True)),
                   "type":"Bool",
                   "timeframe": _extract_tf_token(nm),
                   "fact_id":"seed_from_union_implication"}
            _copy_time_meta(it, out)
            yield out

def _state_hash_path(pid: str) -> str:
    return os.path.join(ROUND_SEED_ROOT, pid, ".seed.sha1")

def _build_round_seed_canonical(patients: List[str], round_idx: int) -> Set[str]:
    print(f"\n=== Build seed canonical (round {round_idx}) for {len(patients)} patient(s) ===")
    os.makedirs(ROUND_SEED_ROOT, exist_ok=True)
    changed: Set[str] = set()
    for pid in patients:
        orig_path = os.path.join(ISA_IN_ROOT, pid, "canonical.jsonl")
        out_dir   = os.path.join(ROUND_SEED_ROOT, pid); os.makedirs(out_dir, exist_ok=True)
        out_path  = os.path.join(out_dir, "canonical.jsonl")
        seen=set(); written=base=seed=0
        with open(out_path, "w", encoding="utf-8") as fout:
            for r in _iter_jsonl(orig_path):
                nm = str(r.get("entity_variable_name") or "")
                if not nm: continue
                base += 1
                if nm in seen: continue
                seen.add(nm); fout.write(json.dumps(r, ensure_ascii=False) + "\n"); written += 1
            for r in _iter_prev_seed_rows_for_patient(pid):
                nm = str(r.get("entity_variable_name") or "")
                if not nm: continue
                seed += 1
                if nm in seen: continue
                seen.add(nm); fout.write(json.dumps(r, ensure_ascii=False) + "\n"); written += 1
        new_hash = _sha1_file(out_path)
        st = _state_hash_path(pid)
        old_hash = None
        if os.path.isfile(st):
            try: old_hash = open(st,"r",encoding="utf-8").read().strip()
            except Exception: old_hash = None
        open(st, "w", encoding="utf-8").write(new_hash)
        is_changed = (old_hash != new_hash)
        if is_changed: changed.add(pid)
        print(f"[seed] {pid:<32} base={base} +seeds={seed} -> written={written}  ({'changed' if is_changed else 'unchanged'})")
    return changed

def _ensure_seed_enriched_passthrough(pid: str) -> None:
    """ROUND_SEED_ROOT/<pid>/canonical.enriched.jsonl exists (passthrough)."""
    pdir = os.path.join(ROUND_SEED_ROOT, pid)
    src = os.path.join(pdir, "canonical.jsonl")
    dst = os.path.join(pdir, "canonical.enriched.jsonl")
    if not os.path.isfile(src): return
    if os.path.isfile(dst): return
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for ln in fin: fout.write(ln)

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
    union = os.path.join(IMP_OUT_ROOT, pid, "canonical.enriched.union.jsonl")
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
# === Snapshot KV (STRICT) ==
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
    if not isinstance(nm, str): return None
    nm = nm.strip()
    if not nm: return None
    nm = _QUAL_SUFFIX_RX.sub("", nm)
    nm = _NAME_RX_MULTI_US.sub("_", nm)
    return nm

def _tail_norms(full_base: str) -> list[str]:
    t = full_base
    out = {
        t,
        re.sub(r"_test_finding(?=$|_)", "_test", t),
        re.sub(r"_level_finding(?=$|_)", "_level", t),
        re.sub(r"_measurement_finding(?=$|_)", "_measurement", t),
        re.sub(r"_finding(?=$|_)", "", t),
    }
    return [x for x in out if x]

def _make_alias_keys_from_full_name(full_name: str) -> list[str]:
    name = _norm_name(full_name) or ""
    keys = set()
    if not name: return []
    keys.add(name)
    stem = None
    for s in _STEM_PREFIXES:
        if name.startswith(s):
            stem = s
            tail = name[len(s):]
            for alt in _STEM_PREFIXES:
                keys.add(_norm_name(alt + tail))
            for tnorm in _tail_norms(tail):
                for alt in _STEM_PREFIXES:
                    keys.add(_norm_name(alt + tnorm))
            break
    if not stem:
        for bnorm in _tail_norms(name):
            keys.add(_norm_name(bnorm))
    return sorted(k for k in keys if k)

def _ingest_patient_facts_into_kv(pid: str, cur) -> int:
    import glob
    total = 0
    d_incl = os.path.join(PERSIST_FS_ROOT, pid, "inclusion")
    d_excl = os.path.join(PERSIST_FS_ROOT, pid, "exclusion")
    files = []
    for d in (d_incl, d_excl):
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, "facts.round*.jsonl"))))
    for path in files:
        for obj in _iter_jsonl(path):
            if (obj or {}).get("kind") != "bool":
                continue
            nm = _norm_name((obj or {}).get("entity_variable_name"))
            if not nm: continue
            vraw = (obj or {}).get("value")
            if isinstance(vraw, (int, float)):
                val = 1 if int(vraw) != 0 else 0
            elif isinstance(vraw, bool):
                val = 1 if vraw else 0
            else:
                continue
            try:
                cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (nm, val))
            except Exception:
                pass
            total += 1
            for alias_key in _make_alias_keys_from_full_name(nm):
                try:
                    cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (alias_key, val))
                except Exception:
                    pass
    return total

def _kv_lookup_bool(dbp: str, key: str) -> Optional[bool]:
    import sqlite3
    k = _norm_name(key)
    if not k: return None
    conn = sqlite3.connect(dbp); cur = conn.cursor()
    try:
        cur.execute("SELECT v FROM kv WHERE k=? LIMIT 1", (k,))
        row = cur.fetchone()
        if row is not None:
            return bool(row[0])
    finally:
        conn.close()
    return None

def _ensure_union_kv_db(pid: str) -> str:
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
                cur.execute("INSERT INTO kv(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, val))
            except Exception:
                try:
                    cur.execute("UPDATE kv SET v=? WHERE k=?", (val, key))
                    if cur.rowcount == 0:
                        cur.execute("INSERT OR IGNORE INTO kv(k, v) VALUES (?, ?)", (key, val))
                except Exception:
                    pass
            n += 1
        return n

    tmpd = os.path.join(ENTAIL_SNAPSHOT_ROOT, pid, ".tmp"); os.makedirs(tmpd, exist_ok=True)
    dbp = os.path.join(tmpd, "union_kv.sqlite")
    if os.path.isfile(dbp):  # already built for this run
        return dbp

    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v INTEGER)")
    conn.commit()

    total = 0
    total += _ingest_patient_facts_into_kv(pid, cur)
    total += _ingest_jsonl(os.path.join(IMP_OUT_ROOT, pid, "canonical.enriched.union.jsonl"), cur)
    total += _ingest_jsonl(os.path.join(ROUND_SEED_ROOT, pid, "canonical.jsonl"), cur)
    diag_dir = os.path.join(DIAG_ROOT_IN, pid)
    for fname in ("diagnosis.export.jsonl", "diagnosis.jsonl"):
        total += _ingest_jsonl(os.path.join(diag_dir, fname), cur)
    for root in (F2P_OUT_ROOT, F2OE_OUT_ROOT, P2F_OUT_ROOT, OE2F_OUT_ROOT, F2P_OTHER_OUT_ROOT):
        for side in ("inclusion","exclusion"):
            total += _ingest_jsonl(os.path.join(root, pid, side, "canonical.enriched.union.jsonl"), cur)

    conn.commit(); conn.close()
    return dbp

# =========================
# ===== SNAPSHOTS =========
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
    return None

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
        "f2p":     (F2P_OUT_ROOT,      "finding_to_procedure_enrichment.flat.json"),
        "f2oe":    (F2OE_OUT_ROOT,     "finding_to_observable_entity_enrichment.flat.json"),
        "p2f":     (P2F_OUT_ROOT,      "procedure_to_finding_enrichment.flat.json"),
        "oe2f":    (OE2F_OUT_ROOT,     "observable_entity_to_finding_enrichment.flat.json"),
        "f2p_rel": (F2P_OTHER_OUT_ROOT,"finding_to_procedure_via_relations.flat.json"),
    }
    stage_is_side = stage_key in stage_to_file
    use_sides = sides or (["inclusion", "exclusion"] if stage_is_side else [None])

    if stage_key == "imp":
        for pid in patients:
            struct_path = os.path.join(IMP_OUT_ROOT, pid, "canonical.enriched.implications.structured.json")
            for src, dst, val in _iter_imp_pairs_from_structured(struct_path):
                add_pair(pid, None, src, dst, val)
            try:
                doc = json.load(open(struct_path, "r", encoding="utf-8"))
            except Exception:
                doc = None
            g = (doc or {}).get("groups") if isinstance(doc, dict) else None
            if isinstance(g, dict):
                for key in ("schema", "timeframe"):
                    for r in (g.get(key) or []):
                        ensure_src(pid, None, (r or {}).get("name"))

    elif stage_is_side:
        root, out_fname = stage_to_file[stage_key]
        for pid in patients:
            for side in use_sides:
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
                    union = os.path.join(IMP_OUT_ROOT, pid, "canonical.enriched.union.jsonl")
                    for obj in _iter_jsonl(union):
                        nm = obj.get("entity_variable_name") or obj.get("name")
                        if isinstance(nm, str) and nm.strip():
                            ensure_src(pid, side, nm)

    return by_ps

def _precreate_snapshot_dirs(stage_key: str, round_idx: int, inner_pass: int,
                             patients: List[str], sides: Optional[List[str]] = None):
    stage_is_side_based = stage_key in {"f2p","f2oe","p2f","oe2f","f2p_rel"}
    use_sides = (sides if sides is not None else (["inclusion","exclusion"] if stage_is_side_based else [None]))
    for pid in patients:
        for side in use_sides:
            _entail_inspect_dir(stage_key, round_idx, inner_pass, pid, side)

def _write_stage_entail_snapshot(stage_key: str, round_idx: int, inner_pass: int,
                                 patients: List[str], sides: Optional[List[str]] = None):
    print(f"[inspect] impl={SNAPSHOT_IMPL_VERSION} stage={stage_key} begin")
    discovered = _collect_stage_mapping_by_patient(stage_key, patients, sides)
    stage_is_side_based = stage_key in {"f2p", "f2oe", "p2f", "oe2f", "f2p_rel"}
    use_sides = (sides if sides is not None else (["inclusion", "exclusion"] if stage_is_side_based else [None]))
    expected_slices = len(patients) * len(use_sides)
    total_rows = 0

    union_kv_paths: Dict[str, str] = {pid: _ensure_union_kv_db(pid) for pid in patients}

    for pid in patients:
        for side in use_sides:
            mapping = discovered.get((pid, side), {})
            d = _entail_inspect_dir(stage_key, round_idx, inner_pass, pid, side)
            mp = os.path.join(d, "mapping.jsonl")

            rows = 0
            with open(mp, "w", encoding="utf-8") as fp:
                for src, tgt_map in mapping.items():
                    src_val = _kv_lookup_bool(union_kv_paths[pid], src)
                    if src_val is None and SNAPSHOT_DEFAULT_FALSE:
                        src_val = False
                    entailed_list = [{"variable": t, "value": bool(v)} for t, v in sorted(tgt_map.items())]
                    fp.write(json.dumps(
                        {"source_variable_name": src, "source_value": src_val, "entailed": entailed_list},
                        ensure_ascii=False
                    ) + "\n")
                    rows += 1
                    total_rows += 1

                if rows == 0:
                    src_val = False if SNAPSHOT_DEFAULT_FALSE else None
                    fp.write(json.dumps({"source_variable_name": None, "source_value": src_val, "entailed": []}, ensure_ascii=False) + "\n")

            with open(os.path.join(d, "info.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "stage": stage_key, "round": round_idx, "pass": inner_pass,
                    "patient": pid, "side": side, "rows": rows, "timestamp": _now_stamp(),
                    "impl_version": SNAPSHOT_IMPL_VERSION, "snapshot_default_false": SNAPSHOT_DEFAULT_FALSE,
                }, f, ensure_ascii=False, indent=2)

    print(f"[inspect] stage={stage_key} wrote {total_rows} structured rows across {expected_slices} patient-slices")

# =========================
# ===== Substage runners ==
# =========================

def run_implications_fixpoint_once(patients: Optional[List[str]]=None) -> Tuple[int,int,int]:
    rules_doc, tf_cfg, src = _compat_load_impl_rules()
    rules = rules_doc.get("rules", [])
    rin = os.path.abspath(getattr(impl_mod, "ROOT_IN", IMP_IN_ROOT))
    rout= os.path.abspath(getattr(impl_mod, "ROOT_OUT", IMP_OUT_ROOT))
    if not os.path.isdir(rin):
        print(f"[err] implications input root not found: {rin}"); return (0,0,0)
    os.makedirs(rout, exist_ok=True)
    run_list = patients if patients else _list_patients_under(rin)
    if not run_list:
        print("[Implications] nothing changed; skipping"); return (0,0,0)

    print(f"[Implications] rules source: {src}; rules={len(rules)}")
    tot_rows=tot_schema=tot_tf=0
    for entry in run_list:
        pin = os.path.join(rin, entry)
        pout= os.path.join(rout, entry)
        rows=sch=tf=0
        if rules: rows,sch,tf = _impl_process_patient_compat(pin, pout, rules, tf_cfg)
        if rows==sch==tf==0 and len(rules)==0:
            rows,sch,tf = _implications_passthrough(pin, pout)
        implied_n, union_n = _build_implication_union_for_patient(entry)
        tot_rows+=rows; tot_schema+=sch; tot_tf+=tf
        status = (f"ok (schema_new_rc={sch}, timeframe_new_rc={tf}; implied_flat={implied_n}; union_rows={union_n})"
                  if rows or sch or tf else f"passthrough (implied_flat={implied_n}; union_rows={union_n})")
        print(f"[patient] {entry:<32} {status}")
    print(f"[done][Implications] rows={tot_rows} schema_new_recursive={tot_schema} timeframe_new_recursive={tot_tf}")
    return (tot_rows,tot_schema,tot_tf)

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
# ====== Pruning (keep) ===
# =========================

def _prune_step_outputs(stage: str, patients: List[str], sides: Optional[List[str]]=None):
    root_map = {
        "imp": IMP_OUT_ROOT, "f2p": F2P_OUT_ROOT, "f2oe": F2OE_OUT_ROOT, "p2f": P2F_OUT_ROOT, "oe2f": OE2F_OUT_ROOT, "f2p_rel": F2P_OTHER_OUT_ROOT,
    }
    keep_map = {
        "imp": {"": {"canonical.enriched.union.jsonl", "canonical.enriched.implications.structured.json"}},
        "f2p": {"inclusion": {"finding_to_procedure_enrichment.flat.json", "canonical.enriched.union.jsonl"},
                "exclusion": {"finding_to_procedure_enrichment.flat.json", "canonical.enriched.union.jsonl"}},
        "f2oe": {"inclusion": {"finding_to_observable_entity_enrichment.flat.json", "canonical.enriched.union.jsonl"},
                 "exclusion": {"finding_to_observable_entity_enrichment.flat.json", "canonical.enriched.union.jsonl"}},
        "p2f": {"inclusion": {"procedure_to_finding_enrichment.flat.json", "canonical.enriched.union.jsonl"},
                "exclusion": {"procedure_to_finding_enrichment.flat.json", "canonical.enriched.union.jsonl"}},
        "oe2f": {"inclusion": {"observable_entity_to_finding_enrichment.flat.json", "canonical.enriched.union.jsonl"},
                 "exclusion": {"observable_entity_to_finding_enrichment.flat.json", "canonical.enriched.union.jsonl"}},
        "f2p_rel": {"inclusion": {"finding_to_procedure_via_relations.flat.json", "canonical.enriched.union.jsonl"},
                    "exclusion": {"finding_to_procedure_via_relations.flat.json", "canonical.enriched.union.jsonl"}},
    }

    root = root_map.get(stage)
    if not root or not os.path.isdir(root): return
    is_side = stage in {"f2p","f2oe","p2f","oe2f","f2p_rel"}
    sides = sides or (["inclusion","exclusion"] if is_side else [""])

    for pid in patients:
        if is_side:
            for side in sides:
                d = os.path.join(root, pid, side)
                if not os.path.isdir(d): continue
                keep = keep_map[stage][side]
                for fn in os.listdir(d):
                    if fn not in keep:
                        fp = os.path.join(d, fn)
                        try:
                            if os.path.isfile(fp): os.remove(fp)
                        except Exception: pass
        else:
            d = os.path.join(root, pid)
            if not os.path.isdir(d): continue
            keep = keep_map[stage][""]
            for fn in os.listdir(d):
                if fn not in keep:
                    fp = os.path.join(d, fn)
                    try:
                        if os.path.isfile(fp): os.remove(fp)
                    except Exception: pass

# =========================
# ====== Main driver ======
# =========================

def _persist_and_log(pid: str, round_idx: int, stage_roots: dict[str,str]) -> None:
    wrote = persist_all_facts([pid], stage_roots, PERSIST_FROM, PERSIST_DB_PATH, PERSIST_FS_ROOT, round_idx)
    print(f"[persist][{pid}] round {round_idx}: wrote {wrote} facts (pre-snapshot)")

def main():
    global SNAPSHOT_DEFAULT_FALSE
    parser = argparse.ArgumentParser(description="Patient-by-patient enrichment fixpoint (NO ISA) with dedicated entailment snapshots + provenance (STRICT booleans).")
    parser.add_argument("--patient", type=str, default=None, help="Only process this patient (default: all).")
    parser.add_argument("--max-ram-gb", type=float, default=float(os.environ.get("FIXPOINT_MAX_RAM_GB","0")))
    parser.add_argument("--skip-reverse", action="store_true")
    parser.add_argument("--skip-rel", action="store_true")
    parser.add_argument("--single-output", action="store_true", default=True)
    parser.add_argument("--snapshot-default-false", action="store_true",
                        help="In snapshots, use False when no explicit boolean is found for a source.")
    args = parser.parse_args()

    SNAPSHOT_DEFAULT_FALSE = bool(args.snapshot_default_false)

    print(f"[boot] running {__file__}")
    print(f"[boot] snapshot impl = {SNAPSHOT_IMPL_VERSION}")
    print(f"[boot] snapshot_default_false = {SNAPSHOT_DEFAULT_FALSE}")

    # Hard-stop if any OUT_ROOT points to non-_noisa tree
    _assert_noisa_outputs()

    _print_roots(args)

    candidates = _list_patients_under(ISA_IN_ROOT)
    if not candidates:
        print("[warn] no patients under ISA_IN_ROOT"); return
    patients_all = _filter_patients(candidates, args.patient)
    if args.patient and not patients_all:
        print(f"[warn] patient '{args.patient}' not found"); return

    # Include ROUND_SEED_ROOT in stage_roots so its canonical gets persisted
    stage_roots = {
        "demo": DEMO_ROOT,
        "diag": DIAG_ROOT_IN,
        "seed": ROUND_SEED_ROOT,        # <— NEW
        "isa_imp": IMP_OUT_ROOT,
        "proc": F2P_OUT_ROOT,
        "obs": F2OE_OUT_ROOT,
        "p2f": P2F_OUT_ROOT,
        "oe2f": OE2F_OUT_ROOT,
        "proc_rel": F2P_OTHER_OUT_ROOT,
    }

    for round_idx in range(1, GLOBAL_MAX_ROUNDS+1):
        print("\n\n==========================")
        print(f" Global round {round_idx} (patient-by-patient, NO ISA)")
        print("==========================")
        for pid in patients_all:
            print(f"\n— Patient {pid} —")
            for inner_pass in range(1, INNER_MAX_PASSES+1):
                if args.max_ram_gb and not mem_ok(args.max_ram_gb):
                    print("  [guard] memory high → break"); break

                # Build seed canonical; only proceed if changed
                changed_pids = _build_round_seed_canonical([pid], round_idx)
                if pid not in changed_pids:
                    print("  [seed] unchanged → skip stages"); break

                # Ensure enriched passthrough for implications input
                _ensure_seed_enriched_passthrough(pid)

                # Implications
                run_implications_fixpoint_once([pid])
                if args.single_output: _prune_step_outputs("imp", [pid])
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("imp", round_idx, inner_pass, [pid])
                _write_stage_entail_snapshot("imp", round_idx, inner_pass, [pid])

                # Mirror union into side trees for next steps
                _mirror_union_into_side_trees([pid], F2P_IN_ROOT)
                _mirror_union_into_side_trees([pid], F2OE_IN_ROOT)
                _mirror_union_into_side_trees([pid], P2F_IN_ROOT)
                _mirror_union_into_side_trees([pid], OE2F_IN_ROOT)
                _mirror_union_into_side_trees([pid], F2P_OTHER_IN_ROOT)

                # Forward: F→P
                for s in RUN_SIDES: run_finding_to_procedure(s, [pid])
                if args.single_output: _prune_step_outputs("f2p", [pid], RUN_SIDES)
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("f2p", round_idx, inner_pass, [pid], RUN_SIDES)
                _write_stage_entail_snapshot("f2p", round_idx, inner_pass, [pid], RUN_SIDES)

                # Forward: F→OE
                for s in RUN_SIDES: run_finding_to_observable_entity(s, [pid])
                if args.single_output: _prune_step_outputs("f2oe", [pid], RUN_SIDES)
                _persist_and_log(pid, round_idx, stage_roots)
                _precreate_snapshot_dirs("f2oe", round_idx, inner_pass, [pid], RUN_SIDES)
                _write_stage_entail_snapshot("f2oe", round_idx, inner_pass, [pid], RUN_SIDES)

                # Forward: F→P_rel (optional)
                if not args.skip_rel:
                    for s in RUN_SIDES: run_finding_to_procedure_via_relations(s, [pid])
                    if args.single_output: _prune_step_outputs("f2p_rel", [pid], RUN_SIDES)
                    _persist_and_log(pid, round_idx, stage_roots)
                    _precreate_snapshot_dirs("f2p_rel", round_idx, inner_pass, [pid], RUN_SIDES)
                    _write_stage_entail_snapshot("f2p_rel", round_idx, inner_pass, [pid], RUN_SIDES)

                # Reverse (optional): P→F and OE→F
                if not args.skip_reverse:
                    for s in RUN_SIDES: run_procedure_to_finding(s, [pid])
                    if args.single_output: _prune_step_outputs("p2f", [pid], RUN_SIDES)
                    _persist_and_log(pid, round_idx, stage_roots)
                    _precreate_snapshot_dirs("p2f", round_idx, inner_pass, [pid], RUN_SIDES)
                    _write_stage_entail_snapshot("p2f", round_idx, inner_pass, [pid], RUN_SIDES)

                    for s in RUN_SIDES: run_observable_entity_to_finding(s, [pid])
                    if args.single_output: _prune_step_outputs("oe2f", [pid], RUN_SIDES)
                    _persist_and_log(pid, round_idx, stage_roots)
                    _precreate_snapshot_dirs("oe2f", round_idx, inner_pass, [pid], RUN_SIDES)
                    _write_stage_entail_snapshot("oe2f", round_idx, inner_pass, [pid], RUN_SIDES)

                if INNER_MAX_PASSES == 1: break

        print(f"\n[fixpoint] Finished round {round_idx} (patient-by-patient, NO ISA).")

    # Final (round 9999 marker)
    wrote_final = persist_all_facts(patients_all, stage_roots, PERSIST_FROM, PERSIST_DB_PATH, PERSIST_FS_ROOT, 9999)
    print(f"[persist] final: wrote {wrote_final} facts")
    print("\n[done] End-to-end driver (NO ISA) finished.")

if __name__ == "__main__":
    main()