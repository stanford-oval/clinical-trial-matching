#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_isa_prevent_disease.py — ISA ancestors for "patient_wants_to_prevent_*" variables

用途：
- 输入每个病人的 canonical.jsonl（行类似）：
    {
      "conceptId": "50570003",
      "preferred_term": "Aneurysm of coronary vessels",
      "fully_specified_name": "Aneurysm of coronary vessels (disorder)",
      "span_match": "Coronary artery aneurysm",
      "entity_variable_name": "patient_wants_to_prevent_aneurysm_of_coronary_vessels",
      "start_time_in_hours": 0.0,
      "end_time_in_hours": 1000000000.0,
      "start_time_inclusive": true,
      "end_time_inclusive": true,
      "entity_variable_meaning": "...",
      ...
    }

- 对每个 conceptId，从 SNOMED ISA 层级中找 parents + ancestors。
- 在 canonical.enriched.jsonl 里写入 parents_inferred / ancestors_inferred 等结构化信息。
- 在 isa_prevent_enriched.flat.json 里写出派生变量：
    patient_wants_to_prevent_{ancestor_canonical_form}
  带上时间窗、conceptId、hop 等信息。

注意：
- 只做 “向上”（parents/ancestors），不处理 descendants。
- 变量名 composition 固定为 "patient_wants_to_prevent_{entity_canonical_form}"。
"""

from __future__ import annotations
import os, sys, json, re, argparse, hashlib
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Incremental hierarchy cache (persistent)
from hier_edge_cache import (
    parents_cached, ancestors_cached, mini_to_row
)

try:
    from hier_edge_cache import get_concept_label as _cache_get_concept_label  # cid -> str|None
except Exception:
    _cache_get_concept_label = None  # gracefully degrade


# ==================== ROOT CONFIG ====================
INPUT_ROOT   = os.getenv("ISA_PREVENT_INPUT_ROOT")
OUTPUT_ROOT  = os.getenv("ISA_PREVENT_OUTPUT_ROOT")

SNOWSTORM_BASE   = os.getenv("SNOWSTORM_BASE", "http://localhost:8080")
SNOWSTORM_BRANCH = os.getenv("SNOWSTORM_BRANCH", "MAIN")
HIER_FORM        = os.getenv("SNOWSTORM_FORM", "inferred")  # inferred | stated

INPUT_FILENAME              = "disease_prevention.jsonl"
OUTPUT_FILENAME_JSONL       = "canonical.enriched.jsonl"
OUTPUT_FILENAME_STRUCT      = "canonical.enriched.structured.json"
OUTPUT_FILENAME_ISA_DERIVED = "isa_prevent_enriched.flat.json"   # 避免和普通 ISA 冲突
OUTPUT_FILENAME_JSON_ARRAY  = "canonical.enriched.prevent.json"  # array 版


# ==================== utils: canonicalize & compose ====================

_PREVENT_PREFIX = "patient_wants_to_prevent_"

def _canonize_entity_form(term: Optional[str]) -> str:
    """SNOMED label -> canonical token: lower, alnum + '_' only."""
    s = (term or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _parse_prevent_var(varname: str) -> Optional[str]:
    """
    从变量名中抽出 entity_canonical_form：
      patient_wants_to_prevent_{entity}
    返回 entity 部分（原样，不再处理 timeframe/qualifier）。
    """
    vn = (varname or "").strip()
    if not vn.startswith(_PREVENT_PREFIX):
        return None
    return vn[len(_PREVENT_PREFIX):]


def _compose_prevent_var(entity_form: str) -> str:
    """给 canonical entity form 拼 prevent 变量名。"""
    entity_form = (entity_form or "").strip("_")
    return f"{_PREVENT_PREFIX}{entity_form}"


# ----- Time window helpers -----
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


# ==================== Cache label helpers ====================

def _label_from_raw_mini(m: Dict[str, Any]) -> Optional[str]:
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
    Normalize mini and ensure it has preferred_term / fully_specified_name;
    backfill once via cache label if needed.
    """
    row = dict(mini_to_row(raw_mini) or {})
    cid = str(raw_mini.get("conceptId") or row.get("conceptId") or "").strip()
    term = row.get("preferred_term") or row.get("fully_specified_name") or _label_from_raw_mini(raw_mini)
    if not term and _cache_get_concept_label and cid:
        try:
            term = _cache_get_concept_label(cid) or ""
        except Exception:
            term = ""
    if term:
        if not row.get("preferred_term"):
            row["preferred_term"] = term
        if not row.get("fully_specified_name"):
            row["fully_specified_name"] = row.get("fully_specified_name") or None
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
            if not p or p in seen:
                continue
            seen.add(p)
            hop = h + 1
            if p in target_set and p not in found:
                found[p] = hop
            q.append((p, hop))
    return found


# ==================== structure helpers ====================

def _dedup_and_sort(minis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for m in minis:
        cid = str(m.get("conceptId") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if not (m.get("preferred_term") or m.get("fully_specified_name")) and _cache_get_concept_label:
            try:
                lbl = _cache_get_concept_label(cid) or None
                if lbl and not m.get("preferred_term"):
                    m["preferred_term"] = lbl
            except Exception:
                pass
        out.append(m)
    out.sort(
        key=lambda x: (
            (x.get("preferred_term") or x.get("fully_specified_name") or "").lower(),
            str(x.get("conceptId") or ""),
        )
    )
    return out


def to_flat_enriched(
    row: Dict[str, Any],
    *,
    parents=None,
    ancestors=None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    flat = dict(row)
    if parents is not None:
        flat["parents_inferred"] = parents
    if ancestors is not None:
        flat["ancestors_inferred"] = ancestors
    if mode:
        flat["enrichment_mode"] = mode
    return flat


def to_structured(
    row: Dict[str, Any],
    *,
    parents=None,
    ancestors=None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    concept = {
        "conceptId": row.get("conceptId"),
        "preferred_term": row.get("preferred_term"),
        "fully_specified_name": row.get("fully_specified_name"),
        "span_match": row.get("span_match"),
    }
    variable = {
        "name": row.get("entity_variable_name"),
        "type": row.get("type"),
        "template": "patient_wants_to_prevent_{entity_canonical_form}",
        "meaning": row.get("entity_variable_meaning"),
        "start_time_in_hours": row.get("start_time_in_hours"),
        "end_time_in_hours": row.get("end_time_in_hours"),
        "start_time_inclusive": row.get("start_time_inclusive"),
        "end_time_inclusive": row.get("end_time_inclusive"),
    }

    context = {"fact_id": row.get("fact_id")}
    hierarchy = {"form": HIER_FORM, "direction": mode or "ancestors", "counts": {}}
    if parents is not None:
        hierarchy["parents"] = parents
        hierarchy["counts"]["parents"] = len(parents)
    if ancestors is not None:
        hierarchy["ancestors"] = ancestors
        hierarchy["counts"]["ancestors"] = len(ancestors)
    return {"concept": concept, "variable": variable, "context": context, "hierarchy": hierarchy}


# ==================== per-row enrich & derive ====================

def _parse_extracted_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def enrich_row_with_isa_prevent(
    row: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """
    只做 ancestors/parents：
      - 不区分 True/False 模式；如果 extracted_value 明确为 False，则跳过。
    """
    cid = str(row.get("conceptId") or "").strip()
    if not cid:
        return to_flat_enriched(row, mode="none"), to_structured(row, mode="none"), {}, {}

    # 获取 parents + ancestors minis
    parents = _dedup_and_sort(
        [_mini_enrich_labels(m) for m in (parents_cached(SNOWSTORM_BRANCH, HIER_FORM, cid) or [])]
    )
    ancestors = _dedup_and_sort(
        [_mini_enrich_labels(m) for m in (ancestors_cached(SNOWSTORM_BRANCH, HIER_FORM, cid) or [])]
    )
    target_ids = [str(m.get("conceptId") or "") for m in (parents + ancestors) if m.get("conceptId")]
    hops = compute_upward_hops(cid, target_ids)

    flat = to_flat_enriched(row, parents=parents, ancestors=ancestors, mode="ancestors")
    struct = to_structured(row, parents=parents, ancestors=ancestors, mode="ancestors")
    return flat, struct, {"parents": parents, "ancestors": ancestors}, hops


def _derive_prevent_from_hierarchy(
    row: Dict[str, Any],
    derived_sets: Dict[str, List[Dict[str, Any]]],
    hops_map: Dict[str, int],
    patient_id: str,
) -> List[Dict[str, Any]]:
    """
    从 parents + ancestors minis 里派生 Prevent 变量：
      patient_wants_to_prevent_{ancestor_canonical_form}
    """
    out: List[Dict[str, Any]] = []
    source_var = (row.get("entity_variable_name") or "").strip()
    if not source_var:
        return out

    # 只接受 patient_wants_to_prevent_* 变量
    if not source_var.startswith(_PREVENT_PREFIX):
        return out

    row_type    = (row.get("type") or "Bool").strip()
    source_cid  = row.get("conceptId")
    source_term = row.get("preferred_term") or row.get("fully_specified_name")
    fact_id     = row.get("fact_id")

    def _emit(kind: str, minis: List[Dict[str, Any]]):
        for m in minis:
            term  = m.get("preferred_term") or m.get("fully_specified_name") or ""
            d_cid = str(m.get("conceptId") or "")
            if not d_cid:
                continue
            canon = _canonize_entity_form(term)
            if not canon:
                continue
            target_var = _compose_prevent_var(canon)
            hop_val = hops_map.get(d_cid)
            if hop_val is None:
                hop_val = 1 if kind == "parents" else None

            rec: Dict[str, Any] = {
                "patient_id": patient_id,
                "class": f"prevent_{kind}",
                "type": row_type,
                "new_variable_name": target_var,
                "target_variable_name": target_var,
                "derived_conceptId": d_cid,
                "derived_entity_term": term,
                "hop": hop_val,
                "source_variable_name": source_var,
                "source_conceptId": source_cid,
                "derived_from_variable": source_var,
                "derived_from_conceptId": source_cid,
                "derived_from_entity_term": source_term,
                "derivation_stage": "isa_prevent",
                "derivation_rule": f"isa_prevent_{kind}",
                "fact_id": fact_id,
            }
            _copy_time_fields(row, rec)
            out.append(rec)

    if "parents" in derived_sets:
        _emit("parents", derived_sets["parents"])
    if "ancestors" in derived_sets:
        _emit("ancestors", derived_sets["ancestors"])
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
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _isa_input_stamp(patient_out_dir: str) -> str:
    return os.path.join(patient_out_dir, ".isa_prevent.input.sha1")


# ==================== streaming impl ====================

def process_patient_dir(patient_in_dir: str, patient_out_dir: str) -> Tuple[int, int, int]:
    in_path   = os.path.join(patient_in_dir, INPUT_FILENAME)
    out_jsonl = os.path.join(patient_out_dir, OUTPUT_FILENAME_JSONL)
    out_json  = os.path.join(patient_out_dir, OUTPUT_FILENAME_STRUCT)
    out_isa   = os.path.join(patient_out_dir, OUTPUT_FILENAME_ISA_DERIVED)
    out_json_arr = os.path.join(patient_out_dir, OUTPUT_FILENAME_JSON_ARRAY)

    if not os.path.isfile(in_path):
        return (0, 0, 0)

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    tmp_dir = os.path.join(patient_out_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_struct_jsonl = os.path.join(tmp_dir, "prevent.structured.tmp.jsonl")
    tmp_isa_jsonl    = os.path.join(tmp_dir, "prevent.isa.tmp.jsonl")
    tmp_dedupe_db    = os.path.join(tmp_dir, "prevent.dedupe.sqlite")

    patient_id = os.path.basename(os.path.normpath(patient_in_dir))

    cfg_fingerprint = json.dumps(
        {
            "snowstorm": [SNOWSTORM_BASE, SNOWSTORM_BRANCH, HIER_FORM],
            "templates": ["patient_wants_to_prevent_{entity_canonical_form}"],
        },
        sort_keys=True,
    )
    h_in = _sha1_file(in_path)
    h_all = hashlib.sha1((h_in + "|" + cfg_fingerprint).encode("utf-8")).hexdigest()
    stamp_path = _isa_input_stamp(patient_out_dir)
    os.makedirs(patient_out_dir, exist_ok=True)
    old = None
    if os.path.isfile(stamp_path):
        try:
            old = open(stamp_path, "r", encoding="utf-8").read().strip()
        except Exception:
            pass
    if (
        old == h_all
        and os.path.isfile(out_isa)
        and os.path.isfile(out_jsonl)
        and os.path.isfile(out_json)
        and os.path.isfile(out_json_arr)
    ):
        return (0, 0, 0)

    read_n = write_n = skip_n = 0
    struct_summary = {
        "form": HIER_FORM,
        "rows": 0,
        "parents_rows": 0,
        "ancestors_rows": 0,
    }

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_jsonl, "w", encoding="utf-8") as fout_jsonl, \
         open(tmp_struct_jsonl, "w", encoding="utf-8") as f_struct, \
         open(tmp_isa_jsonl,    "w", encoding="utf-8") as f_isa, \
         open(out_json_arr,     "w", encoding="utf-8") as f_jsonarr:

        f_jsonarr.write("[\n")
        first_flat = [True]

        for ln in fin:
            s = ln.strip()
            if not s:
                continue
            read_n += 1
            try:
                row = json.loads(s)
            except Exception:
                skip_n += 1
                continue

            # 只处理 patient_wants_to_prevent_* 变量
            var = (row.get("entity_variable_name") or "").strip()
            if not var.startswith(_PREVENT_PREFIX):
                skip_n += 1
                continue

            exb = _parse_extracted_bool(row.get("extracted_value"))
            # 如果明确标 False，就跳过；None/True 都当作需要 prevent
            if exb is False:
                skip_n += 1
                continue

            flat, struct, sets_, hops_map = enrich_row_with_isa_prevent(row)

            # 写 flat enriched
            fout_jsonl.write(json.dumps(flat, ensure_ascii=False) + "\n")
            _json_dump(f_jsonarr, flat, first_flat, indent=2)
            write_n += 1

            # structured 行
            f_struct.write(json.dumps(struct, ensure_ascii=False) + "\n")
            struct_summary["rows"] += 1
            if (struct.get("hierarchy") or {}).get("parents"):
                struct_summary["parents_rows"] += 1
            if (struct.get("hierarchy") or {}).get("ancestors"):
                struct_summary["ancestors_rows"] += 1

            # 从 parents/ancestors 派生 Prevent 变量
            if sets_:
                for rec in _derive_prevent_from_hierarchy(row, sets_, hops_map, patient_id):
                    f_isa.write(json.dumps(rec, ensure_ascii=False) + "\n")

        f_jsonarr.write("\n]\n")

    # structured summary
    with open(out_json, "w", encoding="utf-8") as fjson:
        json.dump(struct_summary, fjson, ensure_ascii=False, indent=2)

    # 去重派生变量 → JSON array
    conn = _spillset_open(tmp_dedupe_db)
    cur = conn.cursor()
    first = [True]
    with open(out_isa, "w", encoding="utf-8") as fout, open(tmp_isa_jsonl, "r", encoding="utf-8") as fin:
        fout.write("[\n")
        for line in fin:
            if not line.strip():
                continue
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

    try:
        os.remove(tmp_isa_jsonl)
    except Exception:
        pass

    # 记录 stamp，便于下次跳过
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(h_all)
    except Exception:
        pass

    print(f"[isa_prevent] {patient_id}: derived written (deduped) → {out_isa}")
    return (read_n, write_n, skip_n)


# ==================== entry (auto-detect + --patient) ====================

def main():
    parser = argparse.ArgumentParser(
        description="ISA ancestors enrichment for patient_wants_to_prevent_* variables."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help=(
            "Patient ID to process (e.g., sigir-20141). "
            "If omitted, process all patients under the input root."
        ),
    )
    parser.add_argument(
        "--in-root",
        type=str,
        default=None,
        help=(
            "Input root directory containing patient subdirs. "
            "Required unless ISA_PREVENT_INPUT_ROOT environment variable is set."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help=(
            "Output root directory for ISA-prevent-enriched results. "
            "Required unless ISA_PREVENT_OUTPUT_ROOT environment variable is set."
        ),
    )

    args = parser.parse_args()

    # 解析 IN ROOT：优先命令行，其次环境变量
    if args.in_root:
        base_in = os.path.abspath(args.in_root)
    elif INPUT_ROOT:
        base_in = os.path.abspath(INPUT_ROOT)
    else:
        parser.error("--in-root is required unless ISA_PREVENT_INPUT_ROOT is set")

    # 解析 OUT ROOT：优先命令行，其次环境变量
    if args.out_root:
        base_out = os.path.abspath(args.out_root)
    elif OUTPUT_ROOT:
        base_out = os.path.abspath(OUTPUT_ROOT)
    else:
        parser.error("--out-root is required unless ISA_PREVENT_OUTPUT_ROOT is set")

    if args.patient:
        root_in = os.path.join(base_in, args.patient)
    else:
        root_in = base_in

    print(f"[info] Snowstorm: {SNOWSTORM_BASE}  branch={SNOWSTORM_BRANCH}  form={HIER_FORM}")
    print(f"[info] IN ROOT : {base_in}")
    print(f"[info] OUT ROOT: {base_out}")
    if args.patient:
        print(f"[info] Only patient: {args.patient}")

    if not os.path.exists(root_in):
        print(f"[err] Input path not found: {root_in}", file=sys.stderr)
        sys.exit(2)

    total_read = total_written = total_skipped = 0

    if args.patient and os.path.isfile(os.path.join(root_in, INPUT_FILENAME)):
        out_dir = os.path.join(base_out, args.patient)
        r, w, s = process_patient_dir(root_in, out_dir)
        total_read += r
        total_written += w
        total_skipped += s
        status = (
            f"ok (wrote {w}/{r}; skipped {s} non-prevent rows or malformed)"
            if r > 0
            else "skipped (empty disease_prevention.jsonl or unchanged)"
        )
        print(f"[patient] {args.patient:<32} {status}")
    elif args.patient is None and os.path.isdir(root_in):
        for entry in sorted(os.listdir(root_in)):
            patient_in_dir = os.path.join(root_in, entry)
            if not os.path.isdir(patient_in_dir):
                continue
            if not os.path.isfile(os.path.join(patient_in_dir, INPUT_FILENAME)):
                print(f"[patient] {entry:<32} skipped (no disease_prevention.jsonl)")
                continue
            out_dir = os.path.join(base_out, entry)
            r, w, s = process_patient_dir(patient_in_dir, out_dir)
            total_read += r
            total_written += w
            total_skipped += s
            status = (
                f"ok (wrote {w}/{r}; skipped {s} non-prevent rows or malformed)"
                if r > 0
                else "skipped (empty disease_prevention.jsonl or unchanged)"
            )
            print(f"[patient] {entry:<32} {status}")
    else:
        print(f"[err] Unsupported input layout for: {root_in}", file=sys.stderr)
        sys.exit(2)

    print(
        f"\n[done] lines read: {total_read}, "
        f"written: {total_written}, skipped (non-prevent or malformed): {total_skipped}"
    )


if __name__ == "__main__":
    main()
