#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_isa.py — qualifier-aware + streaming + on-disk dedupe

Changes in this edition (cache-corrected + timeframe-hardened):
  • Minis from hier_edge_cache are normalized to ALWAYS carry a label:
      - prefer preferred_term / fully_specified_name from mini_to_row
      - otherwise fall back to pt.term / fsn.term / term present in raw minis
      - otherwise one-shot backfill via hier_edge_cache.get_concept_label (if available)
  • Emission guards: skip any derived variable whose entity_canonical_form would be empty.
  • Sorting/dedup respects backfilled labels.
  • NEW: Robust timeframe recovery even if the variable name slightly deviates from the template.
    - If regex parsing misses timeframe, recover it from the var text (e.g., "now", "inthehistory", "inthepast#days").
    - Fallback to row['timeframe'] if still missing.
"""

from __future__ import annotations
import os, sys, json, re, argparse, hashlib
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Incremental hierarchy cache (persistent)
from hier_edge_cache import (
    descendants_cached, parents_cached, ancestors_cached,
    mini_to_row
)

# Optional label backfill if hier_edge_cache exposes it
try:
    from hier_edge_cache import get_concept_label as _cache_get_concept_label  # cid -> str|None
except Exception:
    _cache_get_concept_label = None  # gracefully degrade

# ==================== FIXED CONFIG (edit here) ====================
# 输入/输出根目录：
# - 正常 pipeline：推荐通过 --in-root / --out-root 传入；
# - 单独跑脚本：可以设置环境变量 ISA_INPUT_ROOT / ISA_OUTPUT_ROOT 作为 fallback。
INPUT_ROOT   = os.environ.get("ISA_INPUT_ROOT", "")
OUTPUT_ROOT  = os.environ.get("ISA_OUTPUT_ROOT", "")

SNOWSTORM_BASE   = os.getenv("SNOWSTORM_BASE", "http://localhost:8080")
SNOWSTORM_BRANCH = os.getenv("SNOWSTORM_BRANCH", "MAIN")
HIER_FORM        = os.getenv("SNOWSTORM_FORM", "inferred")  # inferred | stated

INPUT_FILENAME              = "canonical.jsonl"
OUTPUT_FILENAME_JSONL       = "canonical.enriched.jsonl"
OUTPUT_FILENAME_STRUCT      = "canonical.enriched.structured.json"   # small summary
OUTPUT_FILENAME_ISA_DERIVED = "isa_enriched.flat.json"               # final ARRAY (deduped)

ISA_EMIT_FALSE_DESCENDANTS = True
ISA_DESC_MAX_PER_VAR = int(os.getenv("ISA_DESC_MAX_PER_VAR", "100000000"))
# 限制 canonical.enriched.* 每一行 descendants 列表的最大条数；<=0 表示不限制
ISA_DESC_LIST_LIMIT = int(os.getenv("ISA_DESC_LIST_LIMIT", "100"))

# 限制向下（descendants）最远跳数；<=0 表示不限制
ISA_DESC_MAX_HOPS = int(os.getenv("ISA_DESC_MAX_HOPS", "10"))
# ================================================================

# ==================== Templates ====================
STEMS_FINDINGS = [
    "patient_has_diagnosis_of_{entity_canonical_form}_{timeframe}",
    "patient_has_finding_of_{entity_canonical_form}_{timeframe}",
    "patient_has_symptoms_of_{entity_canonical_form}_{timeframe}",
    "patient_has_clinical_signs_of_{entity_canonical_form}_{timeframe}",
    "patient_has_suspicion_of_{entity_canonical_form}_{timeframe}",
]
STEMS_PROCEDURES = [
    "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_positive",
    "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_negative",
    "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_normal",
    "patient_has_undergone_{entity_canonical_form}_{timeframe}_outcome_is_abnormal",
    "patient_has_undergone_{entity_canonical_form}_{timeframe}",
    "patient_is_undergoing_{entity_canonical_form}_{timeframe}",
    "patient_will_undergo_{entity_canonical_form}_{timeframe}",
    "patient_can_undergo_{entity_canonical_form}_{timeframe}",
]
STEMS_OBS_NUM = [
    "patient_{entity_canonical_form}_value_recorded_{timeframe}_withunit_{unit}",
]
STEMS_OBS_STATUS = [
    "patients_{entity_canonical_form}_is_positive_{timeframe}",
    "patients_{entity_canonical_form}_is_negative_{timeframe}",
    "patients_{entity_canonical_form}_is_normal_{timeframe}",
    "patients_{entity_canonical_form}_is_abnormal_{timeframe}",
]
STEMS_PRODUCT = [
    "patient_is_taking_{entity_canonical_form}_{timeframe}",
    "patient_has_taken_{entity_canonical_form}_{timeframe}",
    "patient_has_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
    "patient_has_intolerance_to_{entity_canonical_form}_{timeframe}",
    "patient_has_allergy_to_{entity_canonical_form}_{timeframe}",
    "patient_has_nonimmune_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
]
STEMS_SUBSTANCE = [
    "patient_is_exposed_to_{entity_canonical_form}_{timeframe}",
    "patient_has_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
    "patient_has_intolerance_to_{entity_canonical_form}_{timeframe}",
    "patient_has_allergy_to_{entity_canonical_form}_{timeframe}",
    "patient_has_nonimmune_hypersensitivity_to_{entity_canonical_form}_{timeframe}",
]

ALL_TEMPLATES = (
    STEMS_FINDINGS + STEMS_PROCEDURES + STEMS_OBS_NUM +
    STEMS_OBS_STATUS + STEMS_PRODUCT + STEMS_SUBSTANCE
)

def _category_for_template(tpl: str) -> str:
    if tpl in STEMS_FINDINGS: return "findings"
    if tpl in STEMS_PROCEDURES: return "procedures"
    if tpl in STEMS_OBS_NUM: return "observable_entities_numeric"
    if tpl in STEMS_OBS_STATUS: return "observable_entities_status"
    if tpl in STEMS_PRODUCT: return "product"
    if tpl in STEMS_SUBSTANCE: return "substance"
    return "unknown"

OUTPUT_FILENAME_JSON_ARRAY   = "canonical.enriched.json" 

# ==================== utils: canonicalize & compose ====================

def _canonize_entity_form(term: Optional[str]) -> str:
    s = (term or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

# Qualifier suffix like @@severe (possibly multiple, chained)
_QUAL_RX = re.compile(r"(?:@@[a-z0-9_]+)+$", re.I)

# Recognize our standard timeframe tokens; extend as needed
_TF_TOKEN_RX = re.compile(r"(now|inthehistory|inthepast\d+days)$", re.I)

def _recover_timeframe_from_text(vn: str) -> str:
    m = _TF_TOKEN_RX.search(vn or "")
    return (m.group(1) if m else "").strip().lower()


def _split_qual_suffix(vn: str) -> tuple[str, str]:
    m = _QUAL_RX.search(vn or "")
    if not m:
        return vn, ""
    return vn[:m.start()], m.group(0)


def _append_qual(vn: str, qual: str) -> str:
    return (vn + (qual or "")).strip("_")


def _compose_var_name(template_str: str, entity_form: str, timeframe: Optional[str],
                      unit: Optional[str], *, qual_suffix: str = "") -> str:
    tf = (timeframe or "").strip()
    u  = (unit or "").strip()
    name = (template_str or "").replace("{entity_canonical_form}", entity_form)\
                                .replace("{timeframe}", tf)\
                                .replace("{unit}", u)
    name = re.sub(r"__+", "_", name).strip("_")
    return _append_qual(name, qual_suffix)

# ==================== utils: detect template from varname ====================

def _template_to_regex(tpl: str) -> re.Pattern:
    """
    Build a regex for the template where {timeframe} is OPTIONAL, including its surrounding underscores.
    This supports variable names both with and without timeframe parts.
    """
    esc = re.escape(tpl)
    tf_esc = re.escape("{timeframe}")

    # Make the {timeframe} segment optional, together with adjacent '_' if present in the template
    if tf_esc in esc:
        idx = esc.find(tf_esc)
        left = esc[idx-1] if idx > 0 else ""
        right = esc[idx+len(tf_esc)] if idx + len(tf_esc) < len(esc) else ""
        # remove the underscore from the literal part and include it inside the optional group
        start = idx - (1 if left == "_" else 0)
        end   = idx + len(tf_esc) + (1 if right == "_" else 0)
        prefix = "_" if left == "_" else ""
        suffix = "_" if right == "_" else ""
        opt_tf = f"(?:{re.escape(prefix)}(?P<tf>[a-z0-9_]+){re.escape(suffix)})?"
        esc = esc[:start] + opt_tf + esc[end:]

    # Replace other placeholders
    esc = esc.replace(re.escape("{entity_canonical_form}"), r"(?P<entity>[a-z0-9_]+)")
    esc = esc.replace(re.escape("{unit}"), r"(?P<unit>[a-z0-9_]+)")

    return re.compile(rf"^{esc}$", re.I)


def _literal_len(tpl: str) -> int:
    # 去掉占位符后的字面量长度，越长越具体
    return len(re.sub(r"\{[^\}]+\}", "", tpl))

_TEMPLATE_REGEXES = [(tpl, _template_to_regex(tpl)) for tpl in ALL_TEMPLATES]
_TEMPLATE_REGEXES.sort(key=lambda x: _literal_len(x[0]), reverse=True)

def _detect_template_from_varname(varname: str) -> Optional[Tuple[str, Dict[str, str]]]:
    vn = (varname or "").strip()
    base_vn, qual = _split_qual_suffix(vn)
    if not (base_vn.startswith("patient_") or base_vn.startswith("patients_")):
        return None
    for tpl, rgx in _TEMPLATE_REGEXES:
        m = rgx.match(base_vn)
        if m:
            g = m.groupdict()
            g["qual_suffix"] = qual
            return tpl, {
                "entity": g.get("entity",""),
                "tf": g.get("tf",""),
                "unit": g.get("unit",""),
                "qual_suffix": qual,
            }
    return None

# ----- Time window helpers (NEW) -----
TIME_KEYS = (
    "start_time_in_hours",
    "end_time_in_hours",
    "start_time_inclusive",
    "end_time_inclusive",
)

def _copy_time_fields(src: Dict[str, Any], dst: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the 4 time-window fields from src row to dst record."""
    for k in TIME_KEYS:
        if k in src:
            dst[k] = src[k]
    return dst

# ==================== Cache label helpers ====================

def _label_from_raw_mini(m: Dict[str, Any]) -> Optional[str]:
    # Try common Snowstorm shapes
    return (
        m.get("preferred_term")
        or (m.get("pt") or {}).get("term")
        or m.get("term")
        or m.get("fully_specified_name")
        or (m.get("fsn") or {}).get("term")
        or None
    )


def _mini_enrich_labels(raw_mini: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw mini via mini_to_row, and ensure it carries a label.
    Backfill once via cache if needed.
    """
    row = dict(mini_to_row(raw_mini) or {})
    cid = str(raw_mini.get("conceptId") or row.get("conceptId") or "").strip()
    # Prefer labels already present (row then raw)
    term = row.get("preferred_term") or row.get("fully_specified_name") or _label_from_raw_mini(raw_mini)
    if not term and _cache_get_concept_label and cid:
        try:
            term = _cache_get_concept_label(cid) or ""
        except Exception:
            term = ""
    # Attach best-effort fields back
    if term:
        if not row.get("preferred_term"):
            row["preferred_term"] = term
        if not row.get("fully_specified_name"):
            # if fsn missing, at least mirror term so downstream has something
            row["fully_specified_name"] = row.get("fully_specified_name") or None
    # Ensure conceptId present
    if cid and not row.get("conceptId"):
        row["conceptId"] = cid
    return row

# ==================== Upward hops (parents/ancestors) ====================
_PARENTS_IDS_CACHE: Dict[str, List[str]] = {}

def _get_parent_ids(cid: str) -> List[str]:
    """Use parents_cached() to avoid new HTTP; only store IDs here for hop calc."""
    if cid in _PARENTS_IDS_CACHE:
        return _PARENTS_IDS_CACHE[cid]
    minis = parents_cached(SNOWSTORM_BRANCH, HIER_FORM, cid) or []
    ids = [str((m.get("conceptId") or "")) for m in minis if m.get("conceptId")]
    _PARENTS_IDS_CACHE[cid] = ids
    return ids


def compute_upward_hops(start_cid: str, target_ids: List[str]) -> Dict[str, int]:
    target_set = set(target_ids)
    found: Dict[str, int] = {}
    seen = set([start_cid])
    q = deque([(start_cid, 0)])
    while q and len(found) < len(target_set):
        cur, h = q.popleft()
        parents = _get_parent_ids(cur)
        for p in parents:
            if not p or p in seen: continue
            seen.add(p)
            hop = h + 1
            if p in target_set and p not in found:
                found[p] = hop
            q.append((p, hop))
    return found

def hop_to_ancestor(desc_cid: str, anc_cid: str) -> Optional[int]:
    """
    返回从后代 desc_cid 向上到祖先 anc_cid 的最小跳数；不可达则返回 None。
    复用已有的父边 BFS（_get_parent_ids）。
    """
    if not desc_cid or not anc_cid: 
        return None
    if desc_cid == anc_cid:
        return 0
    seen = {desc_cid}
    q = deque([(desc_cid, 0)])
    while q:
        cur, h = q.popleft()
        for p in _get_parent_ids(cur):
            if not p or p in seen:
                continue
            if p == anc_cid:
                return h + 1
            seen.add(p)
            q.append((p, h + 1))
    return None

# 每个起点(anc_cid)一份 memo：desc_cid -> hop
_HOP_MEMO_BY_START: Dict[str, Dict[str, int]] = {}

def compute_desc_hops_bulk(anc_cid: str, desc_minis: List[Dict[str, Any]], cutoff: int) -> Dict[str, int]:
    # 1) ids for descendants
    desc_ids = [str(m.get("conceptId") or "") for m in desc_minis if m.get("conceptId")]
    desc_set = set(desc_ids)

    # 2) parent map (ID-only; cached calls)
    parent_ids = {d: _get_parent_ids(d) for d in desc_ids}

    # 3) children map (reverse edges) but only within this induced subgraph
    from collections import defaultdict, deque
    children = defaultdict(list)
    for d, ps in parent_ids.items():
        for p in ps:
            children[p].append(d)

    # 4) downward BFS from ancestor within induced subgraph
    INF = 10**9
    bound = cutoff if (cutoff and cutoff > 0) else INF
    hop = {}  # d -> distance
    q = deque([(anc_cid, 0)])
    seen = {anc_cid}

    while q:
        cur, h = q.popleft()
        if h == bound:
            continue
        for ch in children.get(cur, []):
            if ch in seen:
                continue
            seen.add(ch)
            nh = h + 1
            if ch in desc_set:
                hop[ch] = nh
            q.append((ch, nh))
    return hop



# ==================== structure helpers ====================

def _dedup_and_sort(minis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for m in minis:
        cid = str(m.get("conceptId") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        # If label still missing, last-chance backfill (cheap, memoized in cache)
        if not (m.get("preferred_term") or m.get("fully_specified_name")) and _cache_get_concept_label:
            try:
                lbl = _cache_get_concept_label(cid) or None
                if lbl and not m.get("preferred_term"):
                    m["preferred_term"] = lbl
            except Exception:
                pass
        out.append(m)
    out.sort(key=lambda x: ((x.get("preferred_term") or x.get("fully_specified_name") or "").lower(),
                            str(x.get("conceptId") or "")))
    return out


def to_flat_enriched(row: Dict[str, Any], *, form: str,
                     parents=None, ancestors=None, descendants=None, mode: Optional[str]=None) -> Dict[str, Any]:
    flat = dict(row)
    if parents is not None: flat[f"parents_{form}"] = parents
    if ancestors is not None: flat[f"ancestors_{form}"] = ancestors
    if descendants is not None: flat[f"descendants_{form}"] = descendants
    if mode: flat["enrichment_mode"] = mode
    return flat


def to_structured(row: Dict[str, Any], *, form: str,
                  parents=None, ancestors=None, descendants=None, mode: Optional[str]=None) -> Dict[str, Any]:
    concept = {
        "conceptId": row.get("conceptId"),
        "preferred_term": row.get("preferred_term"),
        "fully_specified_name": row.get("fully_specified_name"),
        "span_match": row.get("span_match"),
    }
    variable = {
        "name": row.get("entity_variable_name"),
        "type": row.get("type"),
        "template": row.get("template"),
        "meaning": row.get("entity_variable_meaning"),
        "start_time_in_hours": row.get("start_time_in_hours"),
        "end_time_in_hours": row.get("end_time_in_hours"),
        "start_time_inclusive": row.get("start_time_inclusive"),
        "end_time_inclusive": row.get("end_time_inclusive"),
    }

    context = {"fact_id": row.get("fact_id")}
    hierarchy = {"form": form, "direction": mode or "none", "counts": {}}
    if parents is not None: hierarchy["parents"] = parents; hierarchy["counts"]["parents"] = len(parents)
    if ancestors is not None: hierarchy["ancestors"] = ancestors; hierarchy["counts"]["ancestors"] = len(ancestors)
    if descendants is not None: hierarchy["descendants"] = descendants; hierarchy["counts"]["descendants"] = len(descendants)
    return {"concept": concept, "variable": variable, "context": context, "hierarchy": hierarchy}

# ==================== per-row enrich & derive ====================

def _parse_extracted_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true","t","1","yes","y"): return True
        if s in ("false","f","0","no","n"): return False
    return None


def enrich_row_with_isa(row: Dict[str, Any], base: str, branch: str, form: str, mode: str
                        ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    cid = str(row.get("conceptId") or "").strip()
    if not cid:
        return to_flat_enriched(row, form=form, mode=mode), to_structured(row, form=form, mode=mode), {}, {}

    if mode == "ancestors":
        parents   = _dedup_and_sort([_mini_enrich_labels(m) for m in (parents_cached(branch, form, cid) or [])])
        ancestors = _dedup_and_sort([_mini_enrich_labels(m) for m in (ancestors_cached(branch, form, cid) or [])])
        target_ids = [str(m.get("conceptId") or "") for m in (parents + ancestors) if m.get("conceptId")]
        hops = compute_upward_hops(cid, target_ids)
        return (
            to_flat_enriched(row, form=form, parents=parents, ancestors=ancestors, mode=mode),
            to_structured(row, form=form, parents=parents, ancestors=ancestors, mode=mode),
            {"parents": parents, "ancestors": ancestors},
            hops
        )

    if mode == "descendants":
        all_desc = [_mini_enrich_labels(m) for m in (descendants_cached(branch, form, cid) or [])]
        hop_map = compute_desc_hops_bulk(cid, all_desc, ISA_DESC_MAX_HOPS)

        kept = []
        for m in all_desc:
            d_id = str(m.get("conceptId") or "")
            if not d_id:
                continue
            h = hop_map.get(d_id)
            if h is None:
                continue
            m["_down_hop"] = h
            kept.append(m)

        # 去重 + 按 hop 升序（近→远），再按 label 稳定排序
        kept = _dedup_and_sort(kept)
        kept.sort(key=lambda x: (x.get("_down_hop", 10**9),
                                 (x.get("preferred_term") or x.get("fully_specified_name") or "").lower(),
                                 str(x.get("conceptId") or "")))

        if ISA_DESC_LIST_LIMIT > 0:
            kept = kept[:ISA_DESC_LIST_LIMIT]

        return (
            to_flat_enriched(row, form=form, descendants=kept, mode=mode),
            to_structured(row, form=form, descendants=kept, mode=mode),
            {"descendants": kept},
            {}
        )

    return to_flat_enriched(row, form=form, mode="none"), to_structured(row, form=form, mode="none"), {}, {}


def _derive_variables_from_hierarchy(row: Dict[str, Any],
                                     derived_sets: Dict[str, List[Dict[str, Any]]],
                                     hops_map: Dict[str, int],
                                     patient_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    source_var = (row.get("entity_variable_name") or "").strip()
    if not source_var:
        return out

    det = _detect_template_from_varname(source_var)
    if not det:
        return out

    matched_tpl, groups = det
    tf_from_var   = groups.get("tf") or (row.get("timeframe") or "")
    unit_from_var = groups.get("unit") or (row.get("unit") or "")
    qual_suffix   = groups.get("qual_suffix", "")

    # Hardening: if tf still empty, try to recover from the var text itself
    if not tf_from_var:
        tf_from_var = _recover_timeframe_from_text(source_var)
    # If still empty, keep as empty (some numeric templates might not require it),
    # but most of our templates do use timeframe, so this prevents silent dropping.

    row_type    = (row.get("type") or "").strip()
    source_cid  = row.get("conceptId")
    source_term = row.get("preferred_term") or row.get("fully_specified_name")
    fact_id     = row.get("fact_id")
    tpl_category = _category_for_template(matched_tpl)

    def _emit(kind: str, minis: List[Dict[str, Any]]):
        rule_label = f"isa_{kind}"
        for m in minis:
            term  = m.get("preferred_term") or m.get("fully_specified_name") or ""
            d_cid = str(m.get("conceptId") or "")
            canon = _canonize_entity_form(term)
            if not canon:              # ← guard: never emit empty-entity variables
                continue
            target_var = _compose_var_name(matched_tpl, canon, tf_from_var, unit_from_var, qual_suffix=qual_suffix)
            hop_val = hops_map.get(d_cid)
            if hop_val is None:
                hop_val = 1 if kind == "parents" else None

            rec = {
                "patient_id": patient_id,
                "class": kind,
                "type": row_type,
                # timeframe retained for backward-compat, will be None in the new world
                "timeframe": (tf_from_var or None),
                "unit": (unit_from_var or None),
                "template_used": tpl_category,
                "new_variable_name": target_var,
                "target_variable_name": target_var,
                "derived_conceptId": d_cid or None,
                "derived_entity_term": term,
                "hop": hop_val,
                "source_variable_name": source_var,
                "source_conceptId": source_cid,
                "derived_from_variable": source_var,
                "derived_from_conceptId": source_cid,
                "derived_from_entity_term": source_term,
                "derivation_stage": "isa",
                "derivation_rule": rule_label,
                "fact_id": fact_id,
            }
            _copy_time_fields(row, rec)   # ← NEW: carry 4 time-window fields
            out.append(rec)


    if "parents" in derived_sets:   _emit("parents",   derived_sets["parents"])
    if "ancestors" in derived_sets: _emit("ancestors", derived_sets["ancestors"])
    return out

# ==================== tiny on-disk set for dedupe ====================

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


def _spillset_add(cur, key: str) -> bool:
    try:
        cur.execute("INSERT INTO s(k) VALUES (?)", (key,))
        return True
    except Exception:
        return False


def _json_dump(fp, obj, first_flag, *, indent=None, sort_keys=False):
    if not first_flag[0]:
        fp.write(",\n")
    fp.write(json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys))
    first_flag[0] = False


# ==================== patient-level hashing (skip unchanged) ====================

def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def _isa_input_stamp(patient_out_dir: str) -> str:
    return os.path.join(patient_out_dir, ".isa.input.sha1")

# ==================== streaming impl ====================

def process_patient_dir(patient_in_dir: str, patient_out_dir: str) -> Tuple[int, int, int]:
    in_path   = os.path.join(patient_in_dir, INPUT_FILENAME)
    out_jsonl = os.path.join(patient_out_dir, OUTPUT_FILENAME_JSONL)
    out_json  = os.path.join(patient_out_dir, OUTPUT_FILENAME_STRUCT)
    out_isa   = os.path.join(patient_out_dir, OUTPUT_FILENAME_ISA_DERIVED)
    out_json_arr   = os.path.join(patient_out_dir, OUTPUT_FILENAME_JSON_ARRAY)   

    if not os.path.isfile(in_path): return (0, 0, 0)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    tmp_dir = os.path.join(patient_out_dir, ".tmp"); os.makedirs(tmp_dir, exist_ok=True)
    tmp_struct_jsonl = os.path.join(tmp_dir, "structured.tmp.jsonl")
    tmp_isa_jsonl    = os.path.join(tmp_dir, "isa.tmp.jsonl")
    tmp_dedupe_db    = os.path.join(tmp_dir, "dedupe.sqlite")

    patient_id = os.path.basename(os.path.normpath(patient_in_dir))

    cfg_fingerprint = json.dumps({
        "snowstorm": [SNOWSTORM_BASE, SNOWSTORM_BRANCH, HIER_FORM],
        "emit_false_desc": ISA_EMIT_FALSE_DESCENDANTS,
        "desc_cap": ISA_DESC_MAX_PER_VAR,
        "templates_version": len(ALL_TEMPLATES),
    }, sort_keys=True)
    h_in = _sha1_file(in_path)
    h_all = hashlib.sha1((h_in + "|" + cfg_fingerprint).encode("utf-8")).hexdigest()
    stamp_path = _isa_input_stamp(patient_out_dir)
    os.makedirs(patient_out_dir, exist_ok=True)
    old = None
    if os.path.isfile(stamp_path):
        try: old = open(stamp_path, "r", encoding="utf-8").read().strip()
        except Exception: pass
    if old == h_all and os.path.isfile(out_isa) and os.path.isfile(out_jsonl) and os.path.isfile(out_json) and os.path.isfile(out_json_arr):
        return (0, 0, 0)

    read_n = write_n = skip_n = 0
    struct_summary = {"form": HIER_FORM, "rows": 0, "parents_rows": 0, "ancestors_rows": 0, "desc_rows": 0}

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_jsonl, "w", encoding="utf-8") as fout_jsonl, \
         open(tmp_struct_jsonl, "w", encoding="utf-8") as f_struct, \
         open(tmp_isa_jsonl, "w", encoding="utf-8") as f_isa, \
         open(out_json_arr, "w", encoding="utf-8") as f_jsonarr:
        
        f_jsonarr.write("[\n")
        first_flat = [True]  

        for ln in fin:
            s = ln.strip()
            if not s: continue
            read_n += 1
            try:
                row = json.loads(s)
            except Exception:
                skip_n += 1; continue

            exb = _parse_extracted_bool(row.get("extracted_value"))
            if exb is None:
                skip_n += 1; continue

            mode = "ancestors" if exb else "descendants"
            flat, struct, sets_, hops_map = enrich_row_with_isa(row, SNOWSTORM_BASE, SNOWSTORM_BRANCH, HIER_FORM, mode)

            # stream flat canonical enriched
            fout_jsonl.write(json.dumps(flat, ensure_ascii=False) + "\n")
            _json_dump(f_jsonarr, flat, first_flat, indent=2)
            write_n += 1

            # stream structured as jsonl (1 per source)
            f_struct.write(json.dumps(struct, ensure_ascii=False) + "\n")
            struct_summary["rows"] += 1
            if (struct.get("hierarchy") or {}).get("parents"):     struct_summary["parents_rows"] += 1
            if (struct.get("hierarchy") or {}).get("ancestors"):   struct_summary["ancestors_rows"] += 1
            if (struct.get("hierarchy") or {}).get("descendants"): struct_summary["desc_rows"] += 1

            # TRUE → derived parents/ancestors → tmp isa jsonl
            if mode == "ancestors" and sets_:
                for rec in _derive_variables_from_hierarchy(row, sets_, hops_map, patient_id):
                    f_isa.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # FALSE → derived descendants as negatives (bounded)
            source_var = (row.get("entity_variable_name") or "").strip()
            if mode == "descendants" and ISA_EMIT_FALSE_DESCENDANTS and source_var and sets_.get("descendants"):
                det = _detect_template_from_varname(source_var)
                if det:
                    matched_tpl, groups = det
                    tf_from_var   = groups.get("tf") or (row.get("timeframe") or "")
                    unit_from_var = groups.get("unit") or (row.get("unit") or "")
                    qual_suffix   = groups.get("qual_suffix", "")
                    # Hardening: recover timeframe from text if still empty
                    if not tf_from_var:
                        tf_from_var = _recover_timeframe_from_text(source_var)

                    tpl_category  = _category_for_template(matched_tpl)

                    emitted = 0
                    for m in sets_["descendants"]:
                        if emitted >= ISA_DESC_MAX_PER_VAR:
                            break
                        term = m.get("preferred_term") or m.get("fully_specified_name") or ""
                        d_cid = str(m.get("conceptId") or "")
                        canon = _canonize_entity_form(term)
                        if not canon:
                            continue

                        hop_val = m.get("_down_hop")  # ← 直接复用
                        if ISA_DESC_MAX_HOPS > 0 and (hop_val is None or hop_val > ISA_DESC_MAX_HOPS):
                            continue

                        target_var = _compose_var_name(matched_tpl, canon, tf_from_var, unit_from_var, qual_suffix=qual_suffix)

                        rec = {
                            "patient_id": patient_id,
                            "class": "descendants",
                            "type": row.get("type"),
                            "timeframe": (tf_from_var or None),
                            "unit": (unit_from_var or None),
                            "template_used": tpl_category,
                            "new_variable_name": target_var,
                            "target_variable_name": target_var,
                            "derived_conceptId": d_cid or None,
                            "derived_entity_term": term,
                            "extracted_value": False,
                            "source_variable_name": source_var,
                            "source_conceptId": row.get("conceptId"),
                            "derived_from_variable": source_var,
                            "derived_from_conceptId": row.get("conceptId"),
                            "derived_from_entity_term": row.get("preferred_term") or row.get("fully_specified_name"),
                            "derivation_stage": "isa",
                            "derivation_rule": "isa_descendants_false",
                            "fact_id": row.get("fact_id"),
                            "hop": hop_val,
                        }
                        _copy_time_fields(row, rec)
                        f_isa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        emitted += 1

        f_jsonarr.write("\n]\n")

    # small structured summary JSON
    with open(out_json, "w", encoding="utf-8") as fjson:
        json.dump(struct_summary, fjson, ensure_ascii=False, indent=2)

    # Deduplicate tmp isa jsonl to final JSON array w/ on-disk set
    conn = _spillset_open(tmp_dedupe_db); cur = conn.cursor()
    first = [True]
    with open(out_isa, "w", encoding="utf-8") as fout, open(tmp_isa_jsonl, "r", encoding="utf-8") as fin:
        fout.write("[\n")
        for line in fin:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = f"{obj.get('new_variable_name')}|{obj.get('class')}|{obj.get('derived_from_variable')}"
            if _spillset_add(cur, key):
                _json_dump(fout, obj, first)
        conn.commit()
        fout.write("\n]\n")
    conn.close()

    try: os.remove(tmp_isa_jsonl)
    except Exception: pass

    # Write the stamp after successful outputs
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(h_all)
    except Exception:
        pass

    print(f"[isa] {patient_id}: derived written (deduped) → {out_isa}")

    try:
        _HOP_MEMO_BY_START.clear()
    except Exception:
        pass

    return (read_n, write_n, skip_n)

# ==================== entry (auto-detect + --patient) ====================

def main():
    parser = argparse.ArgumentParser(
        description="Enrich canonical facts with ISA hierarchy and derive parent/ancestor variables (and negative descendants for False)."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Patient ID to process (e.g., sigir-20141). If omitted, process all patients under the input root.",
    )
    parser.add_argument(
        "--in-root",
        dest="in_root",
        required=False,
        default=None,
        help="Input root containing per-patient dirs. Required unless ISA_INPUT_ROOT is set.",
    )
    parser.add_argument(
        "--out-root",
        dest="out_root",
        required=False,
        default=None,
        help="Output root for ISA-enriched results. Required unless ISA_OUTPUT_ROOT is set.",
    )
    args = parser.parse_args()

    # 解析 base_in：优先命令行，其次环境变量；都没有就报错
    if args.in_root:
        base_in = os.path.abspath(args.in_root)
    elif INPUT_ROOT:
        base_in = os.path.abspath(INPUT_ROOT)
    else:
        print("[isa] --in-root is required unless ISA_INPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    # 解析 base_out：同样逻辑
    if args.out_root:
        base_out = os.path.abspath(args.out_root)
    elif OUTPUT_ROOT:
        base_out = os.path.abspath(OUTPUT_ROOT)
    else:
        print("[isa] --out-root is required unless ISA_OUTPUT_ROOT is set.", file=sys.stderr)
        sys.exit(2)

    if args.patient:
        root_in = os.path.join(base_in, args.patient)
    else:
        root_in = base_in

    print(f"[info] Snowstorm: {SNOWSTORM_BASE}  branch={SNOWSTORM_BRANCH}  form={HIER_FORM}")
    print(f"[info] IN ROOT : {base_in}")
    print(f"[info] OUT ROOT: {base_out}")
    if args.patient:
        print(f"[info] Only patient: {args.patient}")
    print(f"[info] ISA_EMIT_FALSE_DESCENDANTS={ISA_EMIT_FALSE_DESCENDANTS}  ISA_DESC_MAX_PER_VAR={ISA_DESC_MAX_PER_VAR}\n")

    if not os.path.exists(root_in):
        print(f"[err] Input path not found: {root_in}", file=sys.stderr); sys.exit(2)

    total_read = total_written = total_skipped = 0

    if args.patient and os.path.isfile(os.path.join(root_in, INPUT_FILENAME)):
        out_dir = os.path.join(base_out, args.patient)
        r, w, s = process_patient_dir(root_in, out_dir)
        total_read += r; total_written += w; total_skipped += s
        status = f"ok (wrote {w}/{r}; skipped {s} non-boolean)" if r>0 else "skipped (empty canonical.jsonl or unchanged)"
        print(f"[patient] {args.patient:<32} {status}")
    elif args.patient is None and os.path.isdir(root_in):
        for entry in sorted(os.listdir(root_in)):
            patient_in_dir = os.path.join(root_in, entry)
            if not os.path.isdir(patient_in_dir): continue
            if not os.path.isfile(os.path.join(patient_in_dir, INPUT_FILENAME)):
                print(f"[patient] {entry:<32} skipped (no canonical.jsonl)")
                continue
            out_dir = os.path.join(base_out, entry)
            r, w, s = process_patient_dir(patient_in_dir, out_dir)
            total_read += r; total_written += w; total_skipped += s
            status = f"ok (wrote {w}/{r}; skipped {s} non-boolean)" if r>0 else "skipped (empty canonical.jsonl or unchanged)"
            print(f"[patient] {entry:<32} {status}")
    else:
        print(f"[err] Unsupported input layout for: {root_in}", file=sys.stderr); sys.exit(2)

    print(f"\n[done] lines read: {total_read}, written: {total_written}, skipped (non-boolean): {total_skipped}")

if __name__ == "__main__":
    main()
