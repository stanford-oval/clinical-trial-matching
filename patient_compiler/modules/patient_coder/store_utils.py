# modules/recorders/store_utils.py
# =================================
# Per-patient appenders:
#   ./coded_results/<patient_id>/canonical.jsonl
#   ./coded_results/<patient_id>/demographics.jsonl

from __future__ import annotations
import os, json
from typing import Dict, Any, List, Tuple, Optional

# ---------- shared helpers ----------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _slug(x: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(x or ""))
    return s.strip("_") or "unknown"

def _patient_id(ctx: Dict[str, Any]) -> str:
    pid = ctx.get("patient_id") or ctx.get("note_id") or "unknown_patient"
    return _slug(pid)

def _ensure_meaning(d: dict) -> str:
    m = (d.get("entity_variable_meaning") or "").strip()
    if m:
        return m
    stem = str(d.get("entity_variable_name", ""))
    tf   = str(d.get("timeframe", ""))
    return f"Variable '{stem}' refers to the patient {tf}."

def _canon_decls(context: Dict[str, Any]) -> List[dict]:
    vr = context.get("coder_verifier_results")
    if isinstance(vr, dict):
        lst = vr.get("new_canonical_variable_declarations")
        if isinstance(lst, list) and lst:
            return [x for x in lst if isinstance(x, dict)]
    lst2 = context.get("new_canonical_variable_declarations") or []
    return [x for x in lst2 if isinstance(x, dict)]

def _asps_decls(context: Dict[str, Any]) -> List[dict]:
    vr = context.get("coder_verifier_results")
    if isinstance(vr, dict):
        lst = vr.get("new_age_sex_pregnancystatus_declarations")
        if isinstance(lst, list) and lst:
            return [x for x in lst if isinstance(x, dict)]
    lst2 = context.get("new_age_sex_pregnancystatus_declarations") or []
    return [x for x in lst2 if isinstance(x, dict)]

def _entities_for_idx(ctx: Dict[str, Any], i: int) -> List[dict]:
    """Return entities for fact i from valid_entities_by_req or requirement_bundles."""
    ver_all = ctx.get("valid_entities_by_req")
    if isinstance(ver_all, dict):
        d = ver_all.get(str(i)) or ver_all.get(i) or {}
        if isinstance(d, dict):
            return [x for x in d.values() if isinstance(x, dict)]
        if isinstance(d, list):
            return [x for x in d if isinstance(x, dict)]
    bundles = ctx.get("requirement_bundles") or []
    if isinstance(bundles, list):
        for b in bundles:
            if isinstance(b, dict) and (b.get("req_index") == i):
                ents = b.get("entities") or []
                if isinstance(ents, list):
                    return [x for x in ents if isinstance(x, dict)]
    return []

def _snomed_lookups(ents: List[dict]) -> Tuple[dict, dict, dict]:
    by_pref, by_fsn, by_span = {}, {}, {}
    for ent in ents:
        pref = str(ent.get("preferred_term", "")).strip().lower()
        fsn  = str(ent.get("fully_specified_name", "")).strip().lower()
        span = str(ent.get("extracted_span", "")).strip().lower()
        if pref: by_pref[pref] = ent
        if fsn:  by_fsn[fsn] = ent
        if span: by_span[span] = ent
    return by_pref, by_fsn, by_span


# ---------- recorder 1: per-patient canonical rows (SNOMED-enriched) ----------
def append_flat_fact_canon_rows(
    context: Dict[str, Any],
    *,
    base_dir: str = "./coded_results",
    filename: str = "canonical.jsonl",
) -> Dict[str, List[dict]]:
    """
    Append SNOMED-enriched canonical rows to two files per patient:
      - {base_dir}/{<patient_id>_inclusive}/{<filename_root>}.jsonl
      - {base_dir}/{<patient_id>_exclusive}/{<filename_root>}.jsonl

    Returns:
      {
        "inclusive": <rows written to *_inclusive.jsonl>,
        "exclusive": <rows written to *_exclusive.jsonl>,
      }
    """
    idx = int(context.get("current_requirement_index", 0))
    fact_id = f"fact{idx:03d}"
    patient_id = _patient_id(context)

    canon = _canon_decls(context)
    if not canon:
        return {"inclusive": [], "exclusive": []}

    ents = _entities_for_idx(context, idx)
    by_pref, by_fsn, by_span = _snomed_lookups(ents)

    def _pick_ent(d: dict) -> Optional[dict]:
        canon_used = str(d.get("entity_canonical_form_used", "")).strip().lower()
        span = str(d.get("span", "")).strip().lower()
        return by_pref.get(canon_used) or by_fsn.get(canon_used) or by_span.get(span)


    rows_inclusive: List[dict] = []
    rows_exclusive: List[dict] = []

    for d in canon:
        # conservative/exclusive
        start_excl_h = d.get("smallest_timewindow_start_time_in_hours")
        end_excl_h = d.get("smallest_timewindow_end_time_in_hours")
        start_excl_inc = d.get("smallest_timewindow_start_time_inclusive")
        end_excl_inc = d.get("smallest_timewindow_end_time_inclusive")

        # possible/inclusive
        start_incl_h = d.get("largest_timewindow_start_time_in_hours")
        end_incl_h = d.get("largest_timewindow_end_time_in_hours")
        start_incl_inc = d.get("largest_timewindow_start_time_inclusive")
        end_incl_inc = d.get("largest_timewindow_end_time_inclusive")

        ent = _pick_ent(d)
        base_fields = {
            "conceptId": (str(ent.get("conceptId")) if ent and ent.get("conceptId") is not None else None),
            "preferred_term": (ent.get("preferred_term") if ent else d.get("entity_canonical_form_used")),
            "fully_specified_name": (ent.get("fully_specified_name") if ent else None),
            "span_match": (ent.get("extracted_span") if ent else d.get("span")),
            "entity_variable_name": d.get("entity_variable_name"),
            "type": d.get("type"),
            "fact_id": fact_id,
            "template": d.get("template"),
            "entity_variable_meaning": _ensure_meaning(d),
            "extracted_value": d.get("extracted_value"),
        }

        rows_exclusive.append({
            **base_fields,
            "start_time_in_hours": start_excl_h,
            "end_time_in_hours": end_excl_h,
            "start_time_inclusive": start_excl_inc,
            "end_time_inclusive": end_excl_inc,
        })

        rows_inclusive.append({
            **base_fields,
            "start_time_in_hours": start_incl_h,
            "end_time_in_hours": end_incl_h,
            "start_time_inclusive": start_incl_inc,
            "end_time_inclusive": end_incl_inc,
        })

    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jsonl"

    fname = f"{root}{ext}"
    dir_inclusive = os.path.join(base_dir, f"{patient_id}_inclusion")
    dir_exclusive = os.path.join(base_dir, f"{patient_id}_exclusion")
    out_inclusive = os.path.join(dir_inclusive, fname)
    out_exclusive = os.path.join(dir_exclusive, fname)

    for path, rows in ((out_inclusive, rows_inclusive), (out_exclusive, rows_exclusive)):
        _ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[info] Appended {len(rows)} canonical variable rows to {path}")
        print(f"[debug] Sample row ({os.path.basename(path)}): {rows[0] if rows else 'N/A'}")

    return {"inclusive": rows_inclusive, "exclusive": rows_exclusive}

# ---------- recorder 2: per-patient demographics-only (ASPS) ----------
def append_demographics_rows(
    context: Dict[str, Any],
    *,
    base_dir: str = "./coded_results",
    filename: str = "demographics.jsonl",
) -> Dict[str, List[dict]]:
    """
    Append one JSON object per demographics (ASPS) variable for the current fact### to:
      {base_dir}/{patient_id}/{<root>}_inclusive.jsonl
      {base_dir}/{patient_id}/{<root>}_exclusive.jsonl

    Inclusive window:   possible_earliest_start_time → possible_latest_end_time
    Exclusive window:   confirmable_latest_start_time → confirmable_earliest_end_time

    Returns:
      {"inclusive": [...], "exclusive": [...]}
    """
    idx = int(context.get("current_requirement_index", 0))
    fact_id = f"fact{idx:03d}"
    patient_id = _patient_id(context)

    asps = _asps_decls(context)
    if not asps:
        return {"inclusive": [], "exclusive": []}

    rows_inclusive: List[dict] = []
    rows_exclusive: List[dict] = []

    for d in asps:
        # # Inclusive (possible)
        start_excl_h = d.get("smallest_timewindow_start_time_in_hours")
        end_excl_h = d.get("smallest_timewindow_end_time_in_hours")
        start_excl_inc = d.get("smallest_timewindow_start_time_inclusive")
        end_excl_inc = d.get("smallest_timewindow_end_time_inclusive")
    

        # possible/inclusive
        start_incl_h = d.get("largest_timewindow_start_time_in_hours")
        end_incl_h = d.get("largest_timewindow_end_time_in_hours")
        start_incl_inc = d.get("largest_timewindow_start_time_inclusive")
        end_incl_inc = d.get("largest_timewindow_end_time_inclusive")

        base_fields = {
            "entity_variable_name": d.get("entity_variable_name"),
            "type": d.get("type"),
            "template": d.get("template"),
            # "timeframe": d.get("timeframe"),
            "entity_variable_meaning": _ensure_meaning(d),
            "extracted_value": d.get("extracted_value"),
            "fact_id": fact_id,
        }

        rows_exclusive.append({
            **base_fields,
            "start_time_in_hours": start_excl_h,
            "end_time_in_hours": end_excl_h,
            "start_time_inclusive": start_excl_inc,
            "end_time_inclusive": end_excl_inc,
        })

        rows_inclusive.append({
            **base_fields,
            "start_time_in_hours": start_incl_h,
            "end_time_in_hours": end_incl_h,
            "start_time_inclusive": start_incl_inc,
            "end_time_inclusive": end_incl_inc,
        })

    # Filenames with suffixes
    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jsonl"

    fname = f"{root}{ext}"
    dir_inclusive = os.path.join(base_dir, f"{patient_id}_inclusion")
    dir_exclusive = os.path.join(base_dir, f"{patient_id}_exclusion")
    out_inclusive = os.path.join(dir_inclusive, fname)
    out_exclusive = os.path.join(dir_exclusive, fname)

    for path, rows in ((out_inclusive, rows_inclusive), (out_exclusive, rows_exclusive)):
        _ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[info] Appended {len(rows)} demographics rows to {path}")
        print(f"[debug] Sample row ({os.path.basename(path)}): {rows[0] if rows else 'N/A'}")

    return {"inclusive": rows_inclusive, "exclusive": rows_exclusive}


__all__ = [
    "append_flat_fact_canon_rows",
    "append_demographics_rows",
]
