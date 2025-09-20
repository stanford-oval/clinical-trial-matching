# modules/PatientDiagnosisCoder.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json, re

# PatientDiagnosisCoder — emits diagnosis-coded vars aligned with canonical.jsonl schema,
# but stores them separately under context["diagnosis_variable_declarations"] (no merging).

_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")

def _strip_tag(term: str | None) -> str:
    return _TAG_RE.sub("", (term or "").strip())

def _to_var(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def _canon_from_pt(pt_or_fsn: str, concept_id: str | None, used: set[str]) -> str:
    base = _to_var(_strip_tag(pt_or_fsn))
    if base and base not in used:
        used.add(base)
        return base
    if concept_id:
        cand = f"{base}_{concept_id}"
        if cand not in used:
            used.add(cand)
            return cand
    i = 2
    while True:
        cand = f"{base}_{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1

_TIMEFRAME_SUFFIX_RE = re.compile(
    r"_(now|inthehistory|inthefuture|"
    r"inthepast\d+(?:minutes|hours|days|weeks|months|years)|"
    r"inthefuture\d+(?:minutes|hours|days|weeks|months|years)|"
    r"\d+(?:minutes|hours|days|weeks|months|years)(?:ago|slater))$",
    re.I,
)

def _existing_declared_symbols(program_lines: List[str] | None) -> set[str]:
    if not program_lines:
        return set()
    names = set()
    decl_const = re.compile(r"^\s*\(\s*declare-const\s+([^\s()]+)\b", re.I)
    decl_fun0  = re.compile(r"^\s*\(\s*declare-fun\s+([^\s()]+)\s*\(\s*\)\s+", re.I)
    for ln in program_lines:
        m = decl_const.match(ln) or decl_fun0.match(ln)
        if m:
            names.add(m.group(1))
    return names

def _meaning_sentence(pt_or_fsn: str, stem_family: str, timeframe: str) -> str:
    pt_clean = _strip_tag(pt_or_fsn)
    when_txt = {
        "now": "currently",
        "inthehistory": "in the medical history",
        "inthefuture": "in the (unspecified) future"
    }.get(timeframe, "in the specified timeframe")
    if stem_family == "diagnosis":
        return f"The variable asserts that the patient {when_txt} has a diagnosis of {pt_clean}."
    else:
        return f"The variable asserts that the patient {when_txt} has a clinical finding of {pt_clean}."

class PatientDiagnosisCoder:
    def __init__(
        self,
        *,
        prefer_diagnosis_stem: bool = True,
        fact_id_prefix: str = "diag_global",
        log_dir: Optional[str] = None,
    ):
        self.prefer_diagnosis_stem = bool(prefer_diagnosis_stem)
        self.fact_id_prefix = fact_id_prefix
        self.log_dir = Path(log_dir) if log_dir else None

    def _stem_prefix(self) -> str:
        return "patient_has_diagnosis_of" if self.prefer_diagnosis_stem else "patient_has_finding_of"

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads context['diagnosis_canonical'] and produces a list of diagnosis variables.
        Stores them in context['diagnosis_variable_declarations'] WITHOUT merging into canonical.
        Also keeps context['new_variable_declarations'] unchanged (no implicit union).

        现在同时使用（如果存在）:
          - row['can_be_used_for_exclusion']: bool
          - row['exclusion_reason']: str
        这些通常由 PatientDiagnoseClassifier 事先写入 diagnosis_canonical。
        """
        diags: List[Dict[str, Any]] = context.get("diagnosis_canonical", []) or []
        if not diags:
            context["diagnosis_variable_declarations"] = []
            return context

        existing_prog = _existing_declared_symbols(context.get("smt_program_lines"))
        # also avoid collisions with already-declared (per-req) canonical/asps, but don't merge
        already_named = set()
        for blk in ("new_age_sex_pregnancystatus_declarations", "new_canonical_variable_declarations"):
            for o in (context.get(blk) or []):
                if isinstance(o, dict) and o.get("entity_variable_name"):
                    already_named.add(str(o["entity_variable_name"]))

        out: List[Dict[str, Any]] = []
        used_canons: set[str] = set()
        k = 0

        for row in diags:
            mapping = row.get("mapping") or {}
            cid = (mapping.get("conceptId") or "").strip()
            pt  = (mapping.get("preferred_term") or "").strip()
            fsn = (mapping.get("fully_specified_name") or "").strip()
            ctyp = (mapping.get("type") or "").strip()

            if not cid or not (pt or fsn):
                continue
            if ctyp and ctyp.lower() not in {"clinical finding", "situation with explicit context"}:
                continue

            # LLM classifier 产生的标记（可选）
            can_excl = bool(row.get("can_be_used_for_exclusion", False))
            excl_reason = str(row.get("exclusion_reason") or "").strip()


            canon = _canon_from_pt(pt or fsn, cid, used_canons)

            tf = str(row.get("timeframe") or "").strip().lower().replace(" ", "") or "now"
            stem = f"{self._stem_prefix()}_{canon}"
            # var_name = _enforce_timeframe(stem, tf)
            var_name = stem

            # skip if already declared in SMT or in per-req sets
            if var_name in existing_prog or var_name in already_named:
                continue

            k += 1
            fact_id = f"{self.fact_id_prefix}_{k:03d}"
            stem_family = "diagnosis" if self.prefer_diagnosis_stem else "finding"
            meaning = _meaning_sentence(pt or fsn, stem_family, tf)
            span_match = str(row.get("diagnosis") or "").strip() or _strip_tag(pt or fsn)
            print("[debug diagnosis var]:", row.keys())
            start_time_in_hours = row.get("start_time_in_hours")
            end_time_in_hours = row.get("end_time_in_hours")
            start_time_inclusive = row.get("start_time_inclusive")
            end_time_inclusive = row.get("end_time_inclusive")

            out.append({
                "conceptId": cid,
                "preferred_term": pt or _strip_tag(fsn),
                "fully_specified_name": fsn or pt,
                "span_match": span_match,
                "entity_variable_name": var_name,
                "type": "Bool",
                "fact_id": fact_id,
                "template": "findings",
                # "timeframe": tf,
                "start_time_in_hours": start_time_in_hours,
                "end_time_in_hours": end_time_in_hours,
                "start_time_inclusive": start_time_inclusive,
                "end_time_inclusive": end_time_inclusive,
                "entity_variable_meaning": meaning,
                "extracted_value": None,  # keep as-is; change to "True" if you want non-null values
                "mapping": {
                    "conceptId": cid,
                    "preferred_term": pt,
                    "fully_specified_name": fsn,
                    "type": ctyp,
                },
                # <<< NEW: 直接传播 LLM 判定字段
                "can_be_used_for_exclusion": can_excl,
                "exclusion_reason": excl_reason,
            })

        out.sort(key=lambda o: o.get("entity_variable_name", ""))

        # ← store separately; do NOT merge
        context["diagnosis_variable_declarations"] = out

        # keep existing per-req union intact
        asps = context.get("new_age_sex_pregnancystatus_declarations", []) or []
        canon = context.get("new_canonical_variable_declarations", []) or []
        context["new_variable_declarations"] = (asps or []) + (canon or [])

        if self.log_dir:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "diagnosis_coder_rows.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

        return context
