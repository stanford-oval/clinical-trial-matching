# modules/smt_programmer.py
# =========================
# NAMER-ONLY orchestrator with per-patient appenders for
# demographics (ASPS), canonical entities, and GLOBAL diagnosis-coded findings.
#
# Verifier-free variant: no PatientSMTVariableCoderChecker calls, no diagnosis verification.

from __future__ import annotations
from typing import Optional, List, Dict, Any
import os, json, re, shutil
import dspy

# ── pipeline pieces (split namers + diagnosis coder; NO verifier) ────────────
from .stages import (
    PatientCanonicalEntityEnricher,
    PatientDemographicsVariableCoder,
    PatientCanonicalVariableCoder,
    PatientCanonicalVariableOtherCandidatesCoder,

    PatientDiagnoseClassifier,
    PatientDiagnosisCoder,   # deterministic diagnosis→vars coder (exports kept separate)
)

from smt_core.utils.z3_helpers import _log
from .store_utils import (
    append_flat_fact_canon_rows,
    append_demographics_rows,
)

# ──────────────────────────────────────────────────────────────────────────────
# Patient ID resolution (avoid "unknown_patient/")
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_patient_id(ctx: Dict[str, Any]) -> str:
    candidates: List[Optional[str]] = [
        ctx.get("patient_id"),
        ctx.get("patientId"),
        (ctx.get("patient") or {}).get("id") if isinstance(ctx.get("patient"), dict) else None,
        (ctx.get("patient_meta") or {}).get("id") if isinstance(ctx.get("patient_meta"), dict) else None,
        ctx.get("doc_id"),
        ctx.get("note_id"),
        ctx.get("record_id"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            pid = c.strip()
            if pid.lower() != "unknown_patient":
                return pid
    raise ValueError(
        "[smt_programmer] Missing patient_id in context. "
        "Please set context['patient_id'] (or patientId/patient.id/doc_id/note_id)."
    )

def fanout_diagnosis_and_remove_plain_patient_dir(
    base_dir: str,
    actual_pid: str,
    diagnosis_jsonl_name: str = "diagnosis.jsonl",
) -> int:
    """
    Copy (append if exists) diagnosis.jsonl from:
        {base_dir}/{actual_pid}/{diagnosis_jsonl_name}
    to:
        {base_dir}/{actual_pid}_inclusion/{diagnosis_jsonl_name}
        {base_dir}/{actual_pid}_exclusion/{diagnosis_jsonl_name}
    and then delete the original {base_dir}/{actual_pid} directory.

    Returns:
        int: number of lines copied from the source file (0 if source missing).
    """
    src_dir = os.path.join(base_dir, actual_pid)
    src = os.path.join(src_dir, diagnosis_jsonl_name)
    if not os.path.isfile(src):
        # Nothing to migrate; try to remove the plain dir if empty.
        try:
            if os.path.isdir(src_dir) and not os.listdir(src_dir):
                os.rmdir(src_dir)
        except Exception:
            pass
        return 0

    # Prepare destinations
    dst_dirs = [
        os.path.join(base_dir, f"{actual_pid}_inclusion"),
        os.path.join(base_dir, f"{actual_pid}_exclusion"),
    ]
    for d in dst_dirs:
        os.makedirs(d, exist_ok=True)
    dst_paths = [os.path.join(d, diagnosis_jsonl_name) for d in dst_dirs]

    # Copy/append while counting lines
    lines_copied = 0
    with open(src, "r", encoding="utf-8") as in_f:
        buf = in_f.readlines()
        lines_copied = len(buf)
        for dp in dst_paths:
            if os.path.exists(dp):
                with open(dp, "a", encoding="utf-8") as out_f:
                    out_f.writelines(buf)
            else:
                # write new file
                with open(dp, "w", encoding="utf-8") as out_f:
                    out_f.writelines(buf)

    # Remove the original patient_id directory (entire tree)
    try:
        shutil.rmtree(src_dir)
    except Exception:
        # Best-effort cleanup; ignore failures
        pass

    return lines_copied


def _maybe_migrate_unknown_patient(
    base_dir: str,
    actual_pid: str,
    diagnosis_jsonl_name: str = "diagnosis.jsonl",
) -> None:
    """
    Ensure diagnosis.jsonl appears under BOTH suffixed patient folders:
        {base_dir}/{<patient_id>_inclusion}/{diagnosis_jsonl_name}
        {base_dir}/{<patient_id>_exclusion}/{diagnosis_jsonl_name}

    Sources we will copy/append from (use any that exist):
        {base_dir}/unknown_patient/{diagnosis_jsonl_name}   (legacy pre-ID; will be removed after migration)
        {base_dir}/{<patient_id>}/{diagnosis_jsonl_name}    (unsuffixed target; kept)

    Behavior:
      • If destination file exists: append lines.
      • If destination file doesn't exist: copy file.
      • If source is unknown_patient: delete the source after migration (and prune dir if empty).
      • If source is {patient_id}/...: keep it (do not delete).
    """
    import os, shutil

    # Candidate sources
    src_unknown = os.path.join(base_dir, "unknown_patient", diagnosis_jsonl_name)
    src_plain   = os.path.join(base_dir, actual_pid, diagnosis_jsonl_name)

    sources = []
    if os.path.isfile(src_unknown):
        sources.append(("unknown", src_unknown))
    if os.path.isfile(src_plain):
        sources.append(("plain", src_plain))

    if not sources:
        return  # nothing to migrate

    # Destinations: only _inclusion / _exclusion
    dest_dirs = [
        os.path.join(base_dir, f"{actual_pid}_inclusion"),
        os.path.join(base_dir, f"{actual_pid}_exclusion"),
    ]

    def _append_or_copy(src_path: str, dst_path: str):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(dst_path):
            with open(dst_path, "a", encoding="utf-8") as out_f, open(src_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
        else:
            shutil.copy2(src_path, dst_path)

    # Perform migrations
    for src_kind, src in sources:
        for ddir in dest_dirs:
            dst = os.path.join(ddir, diagnosis_jsonl_name)
            _append_or_copy(src, dst)

        # Clean up legacy unknown_patient source
        if src_kind == "unknown":
            try:
                os.remove(src)
                unk_dir = os.path.dirname(src)
                try:
                    if not os.listdir(unk_dir):
                        os.rmdir(unk_dir)
                except Exception:
                    pass
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis helpers: repair (join to ctx['diagnosis_canonical']) + filter
# ──────────────────────────────────────────────────────────────────────────────

def _is_diag_family(name: str) -> bool:
    return name.startswith("patient_has_diagnosis_of_") or name.startswith("patient_has_finding_of_")

def _to_var_snake(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

_TIMEFRAME_TOK_RE = re.compile(
    r"_(now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"\d+(?:minutes|hours|days|weeks|months|years)(?:ago|slater))$",
    re.I,
)

def _strip_timeframe_token(stem: str) -> str:
    return _TIMEFRAME_TOK_RE.sub("", stem)

def _repair_and_dedupe_canonical_rows(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only repairs entries living in new_canonical_variable_declarations.
    Diagnosis rows are now kept separate and are not modified here.
    """
    rows = list(ctx.get("new_canonical_variable_declarations") or [])
    if not rows:
        return ctx
    ddx = ctx.get("diagnosis_canonical", []) or []
    diag_map_by_snake: Dict[str, Dict[str, Any]] = {}
    for d in ddx:
        m = d.get("mapping") or {}
        pt  = (m.get("preferred_term") or "").strip()
        fsn = (m.get("fully_specified_name") or "").strip()
        if not (pt or fsn):
            continue
        diag_map_by_snake[_to_var_snake(pt or fsn)] = {
            "conceptId": (m.get("conceptId") or "").strip(),
            "preferred_term": pt,
            "fully_specified_name": fsn,
            "mapping_type": (m.get("type") or "").strip(),
        }

    fixed: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for r in rows:
        name = str(r.get("entity_variable_name") or "")
        if not name:
            continue
        if name in seen_names:
            # prefer mapped version
            existing_idx = next((i for i, x in enumerate(fixed) if x.get("entity_variable_name") == name), None)
            if existing_idx is not None and (not fixed[existing_idx].get("conceptId")) and r.get("conceptId"):
                fixed[existing_idx] = r
            continue
        seen_names.add(name)

        # Historically this tried to backfill diagnosis mapping here.
        # With separate diagnosis storage, this branch should rarely trigger.
        if _is_diag_family(name) and not r.get("conceptId"):
            base = _strip_timeframe_token(name)
            base = re.sub(r"^(patient_has_diagnosis_of_|patient_has_finding_of_)", "", base)
            m = diag_map_by_snake.get(base)
            if m and m.get("conceptId"):
                r = dict(r)
                r["conceptId"] = m["conceptId"]
                r["preferred_term"] = m.get("preferred_term") or r.get("preferred_term")
                r["fully_specified_name"] = m.get("fully_specified_name") or r.get("fully_specified_name")
                if "mapping_type" not in r and m.get("mapping_type"):
                    r["mapping_type"] = m["mapping_type"]
                if not r.get("mapping"):
                    r["mapping"] = {
                        "conceptId": r["conceptId"],
                        "preferred_term": r.get("preferred_term", ""),
                        "fully_specified_name": r.get("fully_specified_name", ""),
                        "type": r.get("mapping_type", ""),
                    }
                r["provenance"] = (r.get("provenance") or []) + ["repair:diagnosis_map"]

        fixed.append(r)

    ctx["new_canonical_variable_declarations"] = fixed
    asps_existing = ctx.get("new_age_sex_pregnancystatus_declarations", []) or []
    ctx["new_variable_declarations"] = (asps_existing or []) + fixed
    return ctx

def _temporarily_filter_unmapped_diag_for_export(ctx: Dict[str, Any]):
    """
    Temporarily remove unmapped diagnosis-family rows from the global canonical list
    for the purpose of per-req export only. Returns a restore lambda.
    (Verifier references removed.)
    """
    global_rows = ctx.get("new_canonical_variable_declarations")

    def _filter(rows):
        if not rows: return rows
        out = []
        for r in rows:
            name = str(r.get("entity_variable_name") or "")
            if _is_diag_family(name) and not r.get("conceptId"):
                continue
            out.append(r)
        return out

    if global_rows is not None:
        ctx["new_canonical_variable_declarations"] = _filter(global_rows)

    def _restore():
        if global_rows is not None:
            ctx["new_canonical_variable_declarations"] = global_rows
    return _restore

# Keep for safety; with separate diagnosis storage this usually becomes a no-op.
def _temporarily_remove_diag_for_canon_export(ctx: Dict[str, Any]):
    """
    Temporarily remove ALL diagnosis-family rows (diagnosis_of_ / finding_of_)
    from ctx['new_canonical_variable_declarations'] for the purpose of writing
    canonical.jsonl. Returns a restore() lambda to revert the change.
    """
    saved = ctx.get("new_canonical_variable_declarations")
    if saved is None:
        return lambda: None
    filtered = [
        r for r in saved
        if not _is_diag_family(str(r.get("entity_variable_name") or ""))
    ]
    ctx["new_canonical_variable_declarations"] = filtered
    def _restore():
        ctx["new_canonical_variable_declarations"] = saved
    return _restore

# ──────────────────────────────────────────────────────────────────────────────
def _write_embedding_search_jsonl(
    context: Dict[str, Any],
    *,
    base_dir: str,
    filename: str = "embedding_search_other_candidate_canonical.jsonl",
) -> Dict[str, int]:
    """
    将 context['embedding_search_other_candidate_variable_declarations'] 映射为精简结构后写入：
      <base>/<patient_id>_inclusion/<filename>   （使用 largest_* 时间窗）
      <base>/<patient_id>_exclusion/<filename>   （使用 smallest_* 时间窗）
    仅输出以下字段：
      conceptId, preferred_term, fully_specified_name, span_match,
      entity_variable_name, type, fact_id, template, extracted_value,
      start_time_in_hours, end_time_in_hours, start_time_inclusive, end_time_inclusive
    返回写入条数：{"inclusion": n1, "exclusion": n2}
    """
    pid = _resolve_patient_id(context)
    src = context.get("embedding_search_other_candidate_variable_declarations") or []
    # 兜底：没有就不写
    counts = {"inclusion": 0, "exclusion": 0}
    if not src:
        return counts

    # 组装函数：按侧挑选时间字段
    def _project(d: Dict[str, Any], side: str) -> Dict[str, Any]:
        # 时间窗：exclusion 用 smallest_*，inclusion 用 largest_*
        if side == "exclusion":
            start_h = d.get("smallest_timewindow_start_time_in_hours")
            end_h   = d.get("smallest_timewindow_end_time_in_hours")
            s_inc   = d.get("smallest_timewindow_start_time_inclusive")
            e_inc   = d.get("smallest_timewindow_end_time_inclusive")
        else:  # inclusion
            start_h = d.get("largest_timewindow_start_time_in_hours")
            end_h   = d.get("largest_timewindow_end_time_in_hours")
            s_inc   = d.get("largest_timewindow_start_time_inclusive")
            e_inc   = d.get("largest_timewindow_end_time_inclusive")

        # span_match 优先使用 span / extracted_span / entity_name
        span_match = (
            d.get("span")
            or d.get("extracted_span")
            or d.get("entity_name")
            or d.get("span_match")
            or ""
        )

        return {
            "conceptId": str(d.get("conceptId") or ""),
            "preferred_term": d.get("preferred_term") or "",
            "fully_specified_name": d.get("fully_specified_name") or "",
            "span_match": span_match,
            "entity_variable_name": d.get("entity_variable_name") or "",
            "type": d.get("type") or "",
            "fact_id": d.get("fact_id") or "",
            "template": d.get("template") or "",
            "extracted_value": d.get("extracted_value", None),

            "start_time_in_hours": start_h,
            "end_time_in_hours": end_h,
            "start_time_inclusive": bool(s_inc) if s_inc is not None else None,
            "end_time_inclusive": bool(e_inc) if e_inc is not None else None,
        }

    # 两侧分别写：_inclusion 用 largest_*, _exclusion 用 smallest_*
    for suffix, key, side in [("_inclusion", "inclusion", "inclusion"),
                              ("_exclusion", "exclusion", "exclusion")]:
        out_dir = os.path.join(base_dir, f"{pid}{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "a", encoding="utf-8") as fh:
            for row in src:
                proj = _project(row, side)
                fh.write(json.dumps(proj, ensure_ascii=False) + "\n")
                counts[key] += 1

    return counts




def _write_diagnosis_only_jsonl(context: Dict[str, Any], *, base_dir: str, filename: str = "diagnosis.jsonl") -> int:
    """
    Append diagnosis rows to JSONL under correct patient.
    Prefers context['diagnosis_variable_declarations'] (new separate store).
    Falls back to scanning context['new_canonical_variable_declarations'] for legacy runs.
    """
    pid = _resolve_patient_id(context)
    out_dir = os.path.join(base_dir, pid)
    os.makedirs(out_dir, exist_ok=True)

    diag_list = [r for r in (context.get("diagnosis_variable_declarations") or []) if isinstance(r, dict)]

    if not diag_list:
        # Legacy fallback
        merged = context.get("new_canonical_variable_declarations") or []
        def _is_diag_or_finding(r: Dict[str, Any]) -> bool:
            name = str(r.get("entity_variable_name", ""))
            return name.startswith("patient_has_diagnosis_of_") or name.startswith("patient_has_finding_of_")
        diag_list = [r for r in merged if isinstance(r, dict) and _is_diag_or_finding(r)]

    if not diag_list:
        return 0

    path = os.path.join(out_dir, filename)
    with open(path, "a", encoding="utf-8") as fh:
        for row in diag_list:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(diag_list)


class PatientCoder(dspy.Module):
    """
    NAMER-ONLY orchestrator (per-note), verifier-free.
    """

    def __init__(
        self,
        engine,
        *,
        max_reqs: Optional[int] = 50,
        namer_log_dir: Optional[str] = None,
        diagnosis_coder_log_dir: Optional[str] = None,
        record_enriched_facts: bool = True,
        recorder_base_dir: str = "../patient_build/patient_coded_results",
        persist_base_for_pt_maps: Optional[str] = None,
        prefer_diagnosis_stem: bool = True,
        write_diagnosis_jsonl: bool = True,
        diagnosis_jsonl_name: str = "diagnosis.jsonl",
        # <<< NEW: 只跑 diagnosis coder 的开关
        diagnosis_only: bool = False,
    ) -> None:
        super().__init__()
        self.max_reqs = max_reqs
        self.record_enriched_facts = record_enriched_facts
        self.recorder_base_dir = recorder_base_dir
        self.write_diagnosis_jsonl = write_diagnosis_jsonl
        self.diagnosis_jsonl_name = diagnosis_jsonl_name
        self.diagnosis_only = diagnosis_only  # <<< NEW

        self.entity_enricher = PatientCanonicalEntityEnricher(
            persist_base=persist_base_for_pt_maps
        )

        self.demographics_namer = PatientDemographicsVariableCoder(engine, log_dir=namer_log_dir)
        self.canonical_namer    = PatientCanonicalVariableCoder(engine, log_dir=namer_log_dir)
        self.other_candidates_namer = PatientCanonicalVariableOtherCandidatesCoder(
            engine, log_dir=namer_log_dir
        )

        self.diagnosis_coder = PatientDiagnosisCoder(
            prefer_diagnosis_stem=prefer_diagnosis_stem,
            log_dir=diagnosis_coder_log_dir,
        )

        self.diagnosis_classifier = PatientDiagnoseClassifier(
            engine=engine,
            log_dir=diagnosis_coder_log_dir,
        )


    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]

        # <<< NEW: 只跑 diagnosis coder 的快速路径 ------------------------
        if self.diagnosis_only:
            # 全局诊断变量编码（一次 per note）
            context = self.diagnosis_classifier.forward(context)
            context = self.diagnosis_coder.forward(context)
            added = len(context.get("diagnosis_variable_declarations") or [])
            _log("diagnosis-coder ✓", -1, f"prepared {added} diagnosis vars (kept separate)")

            # 只写 diagnosis.jsonl + fanout（如果允许写盘）
            if self.record_enriched_facts and self.write_diagnosis_jsonl:
                try:
                    pid = _resolve_patient_id(context)
                    _maybe_migrate_unknown_patient(self.recorder_base_dir, pid, self.diagnosis_jsonl_name)

                    n_diag = _write_diagnosis_only_jsonl(
                        context, base_dir=self.recorder_base_dir, filename=self.diagnosis_jsonl_name
                    )
                    _log("recorder ✓", -1, f"diagnosis-only rows appended: {n_diag}")

                    n = fanout_diagnosis_and_remove_plain_patient_dir(
                        base_dir=self.recorder_base_dir,
                        actual_pid=pid,
                        diagnosis_jsonl_name=self.diagnosis_jsonl_name,
                    )
                    _log("recorder ✓", -1, f"diagnosis fanout (inclusion/exclusion) lines: {n}")
                except Exception as e:
                    _log("recorder ✗", -1, f"{e}")

            return context
        # ---------------------------------------------------------------

        if self.max_reqs and "requirements" in context:
            context["requirements"] = context["requirements"][: self.max_reqs]

        reqs = context.get("requirements", []) or []

        # PRE: enrich patient-side entities with stable snake_case canonical forms
        context = self.entity_enricher.enrich_ctx(context)

        # === Per-requirement pass (no verifier) ===
        for i, _ in enumerate(reqs):
            context["current_requirement_index"] = i

            context = self.demographics_namer.forward(context)
            _log("namer ✓", i, "demographics")

            context = self.canonical_namer.forward(context)
            _log("namer ✓", i, "canonical")
            _log("namer ✓", i)

            # 仅当本 req 有“其他候选”时才触发
            has_other = (context.get("entity_other_canonical_candidates") or {}).get(str(i)) \
                        or (context.get("entity_other_canonical_candidates") or {}).get(i)
            if has_other:
                context = self.other_candidates_namer.forward(context)
                _log("namer ✓", i, "other-candidates")

            # Try to repair (may be no-op early on if diagnosis_canonical not built yet)
            context = _repair_and_dedupe_canonical_rows(context)

            # Export per-req rows WITHOUT unmapped diagnosis vars
            if self.record_enriched_facts:
                try:
                    try:
                        pid = _resolve_patient_id(context)
                        _maybe_migrate_unknown_patient(self.recorder_base_dir, pid, self.diagnosis_jsonl_name)
                    except Exception as e:
                        _log("patient-id ⚠︎", i, f"{e}")

                    _restore = _temporarily_filter_unmapped_diag_for_export(context)
                    try:
                        canon_rows_both = append_flat_fact_canon_rows(
                            context, base_dir=self.recorder_base_dir, filename="canonical.jsonl"
                        )
                        demog_rows_both = append_demographics_rows(
                            context, base_dir=self.recorder_base_dir, filename="demographics.jsonl"
                        )
                    finally:
                        _restore()

                    # Split for clarity
                    canon_rows_exclusive = canon_rows_both["exclusive"]
                    canon_rows_inclusive = canon_rows_both["inclusive"]
                    demog_rows_exclusive = demog_rows_both["exclusive"]
                    demog_rows_inclusive = demog_rows_both["inclusive"]

                    context.setdefault("enriched_records", {})[f"fact{i:03d}"] = {
                        "canonical_rows_exclusive": canon_rows_exclusive,
                        "canonical_rows_inclusive": canon_rows_inclusive,
                        "demographics_rows_exclusive": demog_rows_exclusive,
                        "demographics_rows_inclusive": demog_rows_inclusive,
                    }
                    _log(
                        "recorder ✓", i,
                        f"canonical rows — exclusive: {len(canon_rows_exclusive)}, "
                        f"inclusive: {len(canon_rows_inclusive)}; "
                        f"demographics — exclusive: {len(demog_rows_exclusive)}, "
                        f"inclusive: {len(demog_rows_inclusive)}"
                    )
                except Exception as e:
                    _log("recorder ✗", i, f"{e}")

        # === Global diagnosis pass (once per note) — still no verifier ===
        context = self.diagnosis_classifier.forward(context)
        context = self.diagnosis_coder.forward(context)
        added = len(context.get("diagnosis_variable_declarations") or [])
        _log("diagnosis-coder ✓", -1, f"prepared {added} diagnosis vars (kept separate)")

        # Final repair for canonical rows (diagnosis already separate)
        context = _repair_and_dedupe_canonical_rows(context)

        # Final export
        if self.record_enriched_facts:
            try:
                pid = _resolve_patient_id(context)
                _maybe_migrate_unknown_patient(self.recorder_base_dir, pid, self.diagnosis_jsonl_name)

                # Exclude any lingering diagnosis-family rows from canonical.jsonl (safety)
                _restore_diag_strip = _temporarily_remove_diag_for_canon_export(context)
                try:
                    canon_rows_both = append_flat_fact_canon_rows(
                        context, base_dir=self.recorder_base_dir, filename="canonical.jsonl"
                    )
                finally:
                    _restore_diag_strip()

                canon_rows_exclusive = canon_rows_both["exclusive"]
                canon_rows_inclusive = canon_rows_both["inclusive"]
                _log(
                    "recorder ✓", -1,
                    f"canonical rows (post-diagnosis; diagnosis excluded) — exclusive: {len(canon_rows_exclusive)}, "
                    f"inclusive: {len(canon_rows_inclusive)}"
                )

                # 并列输出 embedding_search_other_candidate_canonical.jsonl
                emb_counts = _write_embedding_search_jsonl(
                    context,
                    base_dir=self.recorder_base_dir,
                    filename="embedding_search_other_candidate_canonical.jsonl"
                )
                _log(
                    "recorder ✓", -1,
                    f"embedding_search rows appended — exclusion: {emb_counts.get('exclusion', 0)}, "
                    f"inclusion: {emb_counts.get('inclusion', 0)}"
                )

                if self.write_diagnosis_jsonl:
                    n_diag = _write_diagnosis_only_jsonl(
                        context, base_dir=self.recorder_base_dir, filename=self.diagnosis_jsonl_name
                    )
                    _log("recorder ✓", -1, f"diagnosis-only rows appended: {n_diag}")
            except Exception as e:
                _log("recorder ✗", -1, f"{e}")

        if self.record_enriched_facts and self.write_diagnosis_jsonl:
            try:
                pid = _resolve_patient_id(context)
                n = fanout_diagnosis_and_remove_plain_patient_dir(
                    base_dir=self.recorder_base_dir,
                    actual_pid=pid,
                    diagnosis_jsonl_name=self.diagnosis_jsonl_name,  # e.g., "diagnosis.jsonl"
                )
                _log("recorder ✓", -1, f"diagnosis fanout (inclusion/exclusion) lines: {n}")
            except Exception as e:
                _log("recorder ⚠︎", -1, f"diagnosis fanout failed: {e}")

        return context
