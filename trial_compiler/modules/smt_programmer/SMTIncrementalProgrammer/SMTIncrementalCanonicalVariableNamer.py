from __future__ import annotations

"""
SMTIncrementalCanonicalVariableNamer (canonical-only)
────────────────────────────────────────────────────
This module declares **canonicalizable entity variables only** for ONE requirement.
Demographics (age/sex/pregnancy) have been split into a separate module and are
NOT handled here anymore.

Inputs expected in `context`:
  - requirements: List[str|dict]
  - current_requirement_index: int
  - requirements_entities_attributes_top_level (preferred), or fallbacks
  - reusable_variables: List[{"variable_name": str, "why": str}]
  - SMTIncrementalCanonicalVariableNamer_prompt: str
      (the prompt mirrors the user's latest spec and yields ONLY
       <new_canonical_variable_declarations> block)

Outputs written into `context`:
  - new_canonical_variable_declarations: List[dict] (validated; qualifiers in detailed+string aliases)
  - errors: List[dict]
  - namer_stage: "canonical_only"

Notably removed vs previous version:
  - No parsing/writing of <new_age_sex_pregnancystatus_declarations>
  - No writing to context["new_variable_declarations"] (prevents cross-req bleed)

Compatibility niceties:
  - Qualifiers are returned in **detailed** form under
      qualifier_predicates_detailed = [{qualifier_variable_snake_case_form, qualifier_meaning}]
    AND mirrored into the **string** alias lists
      qualifier_predicates_for_semantics_not_already_captured_with_stem
      qualifier_predicates
    to keep downstream code working.
"""

import json
import logging
import os
import re
import sys
import warnings
from typing import Dict, List, Any, Optional, Tuple

import dspy

from .namer_checks import (
    _get_canon,
    _NEWCANON_RE,
    _ERRORS_RE,
    _parse_json_array_relaxed,
    _validate_decl_list,
    _extract_declared_symbols,
    _schema_log,
    _default_timeframe_for_type,
    _synthesize_stem,
    _template_for_type,
    _enforce_timeframe_in_stem,
    _sort_key,
)

# ---------------------------------------------------------------------------
# Fallback logger (mirrors translator)
# ---------------------------------------------------------------------------
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snake(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _qual_sanitize(s: str) -> str:
    return str(s or "").strip()


def _collect_entity_qualifiers(entrec: dict, *, only_canonical: bool = False) -> list[str]:
    """Collect raw qualifier strings from the top-level enriched block."""
    out: list[str] = []
    if not only_canonical:
        for q in (entrec.get("qualifiers") or []):
            qn = _qual_sanitize(q)
            if qn:
                out.append(qn)
    for a in (entrec.get("attributes") or []):
        if only_canonical and not bool(a.get("is_canonical")):
            continue
        oq = _qual_sanitize(a.get("original_qualifier"))
        if oq:
            out.append(oq)
    # dedup, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for q in out:
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _find_top_level_block_for_req(ctx: Dict[str, Any], idx: int) -> Optional[dict]:
    arr = ctx.get("requirements_entities_attributes_top_level") or []
    if not isinstance(arr, list):
        return None
    for b in arr:
        if str(b.get("requirement_id")) == str(idx):
            return b
    try:
        if 0 <= idx < len(arr):
            return arr[idx]
    except Exception:
        pass
    return None


def _minify_from_top_level_block(block: dict | None, *, include_index: bool = True) -> Tuple[str, List[dict], Dict[str, List[str]]]:
    if not block:
        return "", [], {}
    ents = block.get("entities") or []
    out_rows: List[dict] = []
    slim: List[dict] = []
    bycanon: Dict[str, List[str]] = {}
    for e in ents:
        ent = (e or {}).get("entity", {}) or {}
        if not ent.get("is_canonical_entity", True):
            continue
        canon = ent.get("canonical_form") or ent.get("preferred_term") or ent.get("surface_string") or ""
        if not canon:
            continue
        quals = _collect_entity_qualifiers(e, only_canonical=False)
        node = {
            "entity_canonical_form_from_entity_canonical_forms_block": canon,
            "type": ent.get("type"),
            "span": ent.get("surface_string") or canon,
            "start": ent.get("start"),
            "end": ent.get("end"),
            "qualifiers": list(quals or []),
        }
        out_rows.append(node)
        bycanon[canon] = node["qualifiers"]
        slim_node = dict(node)
        if include_index:
            slim_node["canon_index"] = len(slim)
        slim.append(slim_node)
    return json.dumps(slim, ensure_ascii=False, indent=2), out_rows, bycanon


# ---------------------------------------------------------------------------
# Merge helpers for meanings / qualifiers
# ---------------------------------------------------------------------------

def _extract_meaning_and_qual_maps(raw_list: List[dict]) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, str]]]]:
    """
    聚合 variable_meaning，并把 qualifiers 统一成 detailed 形态：
      [{qualifier_variable_snake_case_form, qualifier_meaning, qualifier_variable_declaration}]
    其中 declaration 为可选；优先保留最先出现的非空值。
    """
    var_mean: Dict[str, str] = {}
    # qual_map: stem -> form -> {"qualifier_meaning": ..., "qualifier_variable_declaration": ...}
    qual_map: Dict[str, Dict[str, Dict[str, str]]] = {}

    def _add_qual(stem: str, form: str, meaning: str, declaration: str = "") -> None:
        if not form:
            return
        f = form.strip()
        if not f:
            return
        slot = qual_map.setdefault(stem, {}).setdefault(f, {"qualifier_meaning": "", "qualifier_variable_declaration": ""})
        # first-non-empty wins
        if meaning and not slot["qualifier_meaning"]:
            slot["qualifier_meaning"] = meaning.strip()
        if declaration and not slot["qualifier_variable_declaration"]:
            slot["qualifier_variable_declaration"] = declaration.strip()

    for obj in raw_list or []:
        if not isinstance(obj, dict):
            continue
        stem = (obj.get("entity_variable_name") or "").strip()
        if not stem:
            continue

        # variable_meaning: first non-empty wins
        vm = obj.get("variable_meaning")
        if isinstance(vm, str) and vm.strip() and stem not in var_mean:
            var_mean[stem] = vm.strip()

        # 1) detailed objects（优先）
        for d in (obj.get("qualifier_predicates_detailed") or []):
            if isinstance(d, dict):
                _add_qual(
                    stem,
                    (d.get("qualifier_variable_snake_case_form") or "").strip(),
                    (d.get("qualifier_meaning") or "").strip(),
                    # 兼容两种键名
                    (d.get("qualifier_variable_declaration") or d.get("qualifier_declaration") or "").strip(),
                )

        # 2) for_semantics: 可能是 dict 或 str
        for q in (obj.get("qualifier_predicates_for_semantics_not_already_captured_with_stem") or []):
            if isinstance(q, dict):
                _add_qual(
                    stem,
                    (q.get("qualifier_variable_snake_case_form") or "").strip(),
                    (q.get("qualifier_meaning") or "").strip(),
                    (q.get("qualifier_variable_declaration") or q.get("qualifier_declaration") or "").strip(),
                )
            elif isinstance(q, str):
                _add_qual(stem, q.strip(), "")

        # 3) simple string list
        for s in (obj.get("qualifier_predicates") or []):
            if isinstance(s, str):
                _add_qual(stem, s.strip(), "")

    # 转回 detailed 列表
    qual_det: Dict[str, List[Dict[str, str]]] = {}
    for stem, fmap in qual_map.items():
        items: List[Dict[str, str]] = []
        for form, payload in fmap.items():
            if form:
                node = {
                    "qualifier_variable_snake_case_form": form,
                    "qualifier_meaning": payload.get("qualifier_meaning", "") or ""
                }
                decl = payload.get("qualifier_variable_declaration", "") or ""
                if decl:
                    node["qualifier_variable_declaration"] = decl
                items.append(node)
        qual_det[stem] = items
    return var_mean, qual_det



def _merge_meanings_and_qualifiers(validated: List[dict], var_mean: Dict[str, str], qual_det: Dict[str, List[Dict[str, str]]]) -> None:
    for o in validated or []:
        if not isinstance(o, dict):
            continue
        stem = (o.get("entity_variable_name") or "").strip()
        if not stem:
            continue
        if stem in var_mean:
            o["variable_meaning"] = var_mean[stem]
        det = qual_det.get(stem) or []
        if det:
            o["qualifier_predicates_detailed"] = det
            # also keep simple string aliases for downstream compatibility
            forms = [d.get("qualifier_variable_snake_case_form") for d in det if isinstance(d, dict) and d.get("qualifier_variable_snake_case_form")]
            o["qualifier_predicates_for_semantics_not_already_captured_with_stem"] = forms
            o["qualifier_predicates"] = list(forms)
        else:
            # ensure keys exist as empty lists
            o.setdefault("qualifier_predicates_detailed", [])
            o.setdefault("qualifier_predicates_for_semantics_not_already_captured_with_stem", [])
            o.setdefault("qualifier_predicates", [])

def _merge_canonical_duplicates(canon_list: List[dict]) -> List[dict]:
    """
    同 stem 合并：合并 qualifier_predicates_detailed（按 form 去重；优先保留最早出现的非空 meaning/声明）。
    同时重建两个字符串别名列表。
    """
    merged: Dict[str, dict] = {}
    order: List[str] = []

    def _build_qmap(lst: List[dict]) -> Dict[str, Dict[str, str]]:
        # form -> {"qualifier_meaning": ..., "qualifier_variable_declaration": ...}
        qmap: Dict[str, Dict[str, str]] = {}
        for q in lst or []:
            if not isinstance(q, dict):
                continue
            form = (q.get("qualifier_variable_snake_case_form") or "").strip()
            if not form:
                continue
            cur = qmap.setdefault(form, {"qualifier_meaning": "", "qualifier_variable_declaration": ""})
            m = (q.get("qualifier_meaning") or "").strip()
            d = (q.get("qualifier_variable_declaration") or q.get("qualifier_declaration") or "").strip()
            if m and not cur["qualifier_meaning"]:
                cur["qualifier_meaning"] = m
            if d and not cur["qualifier_variable_declaration"]:
                cur["qualifier_variable_declaration"] = d
        return qmap

    for o in canon_list or []:
        if not isinstance(o, dict):
            continue
        stem = (o.get("entity_variable_name") or "").strip()
        if not stem:
            continue

        if stem not in merged:
            base = dict(o)
            base.setdefault("qualifier_predicates_detailed", [])
            base.setdefault("qualifier_predicates_for_semantics_not_already_captured_with_stem", [])
            base.setdefault("qualifier_predicates", [])
            merged[stem] = base
            order.append(stem)
        else:
            base = merged[stem]
            if o.get("__reuse_existing_symbol"):
                base["__reuse_existing_symbol"] = True
            if o.get("__listed_as_reusable"):
                base["__listed_as_reusable"] = True

            qmap = _build_qmap(base.get("qualifier_predicates_detailed") or [])
            for q in (o.get("qualifier_predicates_detailed") or []):
                if not isinstance(q, dict):
                    continue
                form = (q.get("qualifier_variable_snake_case_form") or "").strip()
                if not form:
                    continue
                cur = qmap.setdefault(form, {"qualifier_meaning": "", "qualifier_variable_declaration": ""})
                m = (q.get("qualifier_meaning") or "").strip()
                d = (q.get("qualifier_variable_declaration") or q.get("qualifier_declaration") or "").strip()
                if m and not cur["qualifier_meaning"]:
                    cur["qualifier_meaning"] = m
                if d and not cur["qualifier_variable_declaration"]:
                    cur["qualifier_variable_declaration"] = d

            unified_det = []
            for f, payload in qmap.items():
                node = {"qualifier_variable_snake_case_form": f,
                        "qualifier_meaning": payload.get("qualifier_meaning", "") or ""}
                if payload.get("qualifier_variable_declaration"):
                    node["qualifier_variable_declaration"] = payload["qualifier_variable_declaration"]
                unified_det.append(node)

            base["qualifier_predicates_detailed"] = unified_det
            forms = [d["qualifier_variable_snake_case_form"] for d in unified_det]
            base["qualifier_predicates_for_semantics_not_already_captured_with_stem"] = forms
            base["qualifier_predicates"] = list(forms)

    return [merged[s] for s in order]


# ---------------------------------------------------------------------------
# New helpers for this customization
# ---------------------------------------------------------------------------

def _has_any_qualifiers(o: dict) -> bool:
    """Return True if the object carries any qualifier content in any of the supported fields."""
    if not isinstance(o, dict):
        return False
    return bool(
        (o.get("qualifier_predicates_detailed") or [])
        or (o.get("qualifier_predicates_for_semantics_not_already_captured_with_stem") or [])
        or (o.get("qualifier_predicates") or [])
    )


# ---------------------------------------------------------------------------
# Enrichment for all_linked_variales: concept & PT lookup
# ---------------------------------------------------------------------------

def _build_canon_info_lookup(context: Dict[str, Any], idx: int) -> Dict[str, Dict[str, Any]]:
    """
    Build a map: canonical_form -> {preferred_term, concept_id, entity_type, span}
    Prefer data from requirements_entities_attributes_top_level; fallback to
    preferred_term_canonical_form_map to at least recover preferred_term.
    """
    info: Dict[str, Dict[str, Any]] = {}

    # Prefer enriched top-level block (EntityEnricher output)
    tlb = _find_top_level_block_for_req(context, idx)
    if tlb:
        for rec in (tlb.get("entities") or []):
            ent = (rec or {}).get("entity", {}) or {}
            if not bool(ent.get("is_canonical_entity", True)):
                continue
            cf = (ent.get("canonical_form") or ent.get("preferred_term") or ent.get("surface_string") or "").strip()
            if not cf:
                continue
            slot = info.setdefault(cf, {})
            pt = ent.get("preferred_term")
            if pt and not slot.get("preferred_term"):
                slot["preferred_term"] = pt
            cid = ent.get("conceptId")
            if cid is not None and not slot.get("concept_id"):
                slot["concept_id"] = str(cid)
            et = ent.get("type")
            if et and not slot.get("entity_type"):
                slot["entity_type"] = et
            sp = ent.get("surface_string") or pt or cf
            if sp and not slot.get("span"):
                slot["span"] = sp

    # Fallback: PT→form mapping lets us at least backfill preferred_term
    pt2form = context.get("preferred_term_canonical_form_map") or {}
    if isinstance(pt2form, dict):
        for pt, cf in pt2form.items():
            cf = (cf or "").strip()
            if not cf:
                continue
            slot = info.setdefault(cf, {})
            if pt and not slot.get("preferred_term"):
                slot["preferred_term"] = pt

    return info


# ---------------------------------------------------------------------------
# all_linked_variales helper (augments with preferred_term / concept_id)
# ---------------------------------------------------------------------------

def _extend_all_linked_variales(
    context: Dict[str, Any],
    items: List[Dict[str, Any]],
    *,
    idx: int,
    canon_info: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """
    Append `items` into context["all_linked_variales"], creating the list if needed.
    Deduplicate by entity_variable_name. Augment each item with:
      - preferred_term
      - concept_id
      - entity_type
      - span
    when resolvable via canonical_form.
    """
    arr = context.get("all_linked_variales")
    if not isinstance(arr, list):
        arr = []
    context["all_linked_variales"] = arr

    existing_by_name: Dict[str, Dict[str, Any]] = {}
    for e in arr:
        if isinstance(e, dict):
            n = e.get("entity_variable_name")
            if isinstance(n, str) and n and n not in existing_by_name:
                existing_by_name[n] = e

    trial_id = context.get("trial_id")
    inc_exc  = context.get("inc_exc")
    cinfo    = canon_info or {}

    def _augment_with_concepts(it: Dict[str, Any]) -> None:
        # Identify canonical_form key in the variable payload
        cf = (it.get("entity_canonical_form_from_entity_canonical_forms_block") or "").strip()
        meta = cinfo.get(cf, {}) if cf else {}

        # Set kind/provenance
        it.setdefault("kind", "variable")
        it.setdefault("requirement_id", idx)
        if trial_id and not it.get("trial_id"):
            it["trial_id"] = trial_id
        if inc_exc and not it.get("inc_exc"):
            it["inc_exc"] = inc_exc

        # Fill concept/meta fields if missing
        for src_key, dst_key in [
            ("preferred_term", "preferred_term"),
            ("concept_id", "concept_id"),
            ("entity_type", "entity_type"),
            ("span", "span"),
        ]:
            if meta.get(src_key) and not it.get(dst_key):
                it[dst_key] = meta[src_key]

    added = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = it.get("entity_variable_name")
        if not isinstance(name, str) or not name.strip():
            continue

        _augment_with_concepts(it)

        if name in existing_by_name:
            base = existing_by_name[name]
            # Merge shallowly: fill only missing fields in base
            for k, v in it.items():
                if k == "entity_variable_name":
                    continue
                if base.get(k) is None or base.get(k) == "":
                    base[k] = v
            # Re-run augmentation in case base still lacks concept fields
            _augment_with_concepts(base)
        else:
            arr.append(it)
            existing_by_name[name] = it
            added += 1

    if added:
        _log("canon→link ✓", idx, f"added {added} canonical vars (total linked: {len(arr)})")


# ---------------------------------------------------------------------------
# Canonical-only namer
# ---------------------------------------------------------------------------
class SMTIncrementalCanonicalVariableNamer(dspy.Module):
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        engine,
        *,
        log_dir: str | None = None,
        entities_only: bool | None = True,
        enforce_suffix_in_stem: bool = True,
        allow_mixed_case: bool = False,
        require_canonical_coverage: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self._entities_only = entities_only
        self._enforce_suffix_in_stem = enforce_suffix_in_stem
        self._allow_mixed_case = allow_mixed_case
        self._require_canonical_coverage = require_canonical_coverage

    @staticmethod
    def _env_truthy(name: str, default: bool = False) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "on"}

    def _build_canonical_bundle_for_req(
        self,
        context: Dict[str, Any],
        idx: int,
        *,
        entities_only: bool,
    ) -> Tuple[str, set[str], Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        """Prefer top-level enriched block; fall back to older structures."""
        allowed_canon_forms: set[str] = set()
        allowed_info: Dict[str, Dict[str, str]] = {}
        canon_to_quals: Dict[str, List[str]] = {}

        tlb = _find_top_level_block_for_req(context, idx)
        if tlb:
            bundle_json, rows, bycanon_quals = _minify_from_top_level_block(tlb)
            for r in rows:
                canon = _get_canon(r)
                if canon:
                    allowed_canon_forms.add(canon)
                    allowed_info[canon] = {
                        "type": r.get("type", "") or "",
                        "span": r.get("span", canon) or canon,
                    }
            canon_to_quals = bycanon_quals or {}
            if bundle_json and allowed_canon_forms:
                return bundle_json, allowed_canon_forms, allowed_info, canon_to_quals

        # Fallback: valid_entities_by_req (no qualifiers in this legacy path)
        ver_all: dict = context.get("valid_entities_by_req", {}) or {}
        ver_for_req = ver_all.get(str(idx)) or {}
        if ver_for_req:
            rows_sorted = sorted(
                ver_for_req.values(),
                key=lambda e: (e.get("start", 10**9), e.get("end", 10**9), e.get("extracted_span", "")),
            )
            out = []
            for i, e in enumerate(rows_sorted):
                item = {
                    "entity_canonical_form_from_entity_canonical_forms_block": _get_canon(e),
                    "type": e.get("type"),
                    "span": e.get("extracted_span"),
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "qualifiers": [],
                    "canon_index": i,
                }
                out.append(item)
                s = _get_canon(e)
                if s:
                    allowed_canon_forms.add(s)
                    allowed_info[s] = {"type": e.get("type", ""), "span": e.get("extracted_span", s)}
            return json.dumps(out, ensure_ascii=False, indent=2), allowed_canon_forms, allowed_info, canon_to_quals

        # Fallback: requirement_bundles (if present)
        bundles: List[dict] = context.get("requirement_bundles", [])
        bmap = {b.get("req_index"): b for b in bundles if isinstance(b, dict)}
        b = bmap.get(idx)
        out = []
        if b:
            for i, ent in enumerate((b.get("entities") or [])):
                s = _get_canon(ent)
                if not s:
                    continue
                out.append({
                    "entity_canonical_form_from_entity_canonical_forms_block": s,
                    "type": ent.get("type"),
                    "span": ent.get("span", s),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "qualifiers": [],
                    "canon_index": i,
                })
                allowed_canon_forms.add(s)
                allowed_info[s] = {"type": ent.get("type", ""), "span": ent.get("span", s)}
        return json.dumps(out, ensure_ascii=False, indent=2), allowed_canon_forms, allowed_info, canon_to_quals

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]

        if not context.get("requirements"):
            return context
        idx: int = int(context["current_requirement_index"])

        entities_only = (
            self._entities_only
            if self._entities_only is not None
            else (bool(context.get("namer_entities_only")) or self._env_truthy("SMT_NAMER_ENTITIES_ONLY", default=False))
        )

        cov_ctx = context.get("namer_require_canonical_coverage", None)
        if cov_ctx is None:
            coverage_required = self._env_truthy("SMT_NAMER_REQUIRE_CANONICAL_COVERAGE", default=self._require_canonical_coverage)
        else:
            coverage_required = bool(cov_ctx)

        partial_ok_ctx = context.get("namer_allow_partial_canonical_coverage", None)
        if partial_ok_ctx is None:
            partial_coverage_ok = self._env_truthy("SMT_NAMER_ALLOW_PARTIAL_COVERAGE", True)
        else:
            partial_coverage_ok = bool(partial_ok_ctx)

        # Full coverage is only required when explicitly requested AND partial is not allowed
        required_full_coverage = bool(coverage_required and not partial_coverage_ok)
        # If full coverage isn't required, don't waste attempts on retries
        max_attempts = self.MAX_ATTEMPTS if required_full_coverage else 1

        req_entry = context["requirements"][idx]
        requirement_txt = req_entry.get("requirement") if isinstance(req_entry, dict) else str(req_entry)

        tpl = context.get("SMTIncrementalCanonicalVariableNamer_prompt", "")
        if not tpl:
            _log("namer ✗", idx, "prompt template missing")
            return context

        canon_bundle, allowed_canon_forms, allowed_info, canon_to_quals = self._build_canonical_bundle_for_req(
            context, idx, entities_only=bool(entities_only)
        )
        if not canon_bundle:
            _log("namer", idx, "canonical bundle empty (no enriched top-level/valid_entities/bundle)")

        # Build once: lookup for preferred_term / concept_id enrichment into linked list
        canon_info = _build_canon_info_lookup(context, idx)

        reusable_json = json.dumps(context.get("reusable_variables", []), ensure_ascii=False, indent=2)
        prompt = (tpl
                  .replace("#CANONICAL_FORMS#", canon_bundle or "[]")
                  .replace("#REQUIREMENT#", requirement_txt)
                  .replace("#REUSABLE_VARIABLES#", reusable_json))

        trial_id = context.get("trial_id", "unknown_trial")
        side = context.get("inc_exc", "unknown")

        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc", "inclusion", "include", "in"}:  return "inclusion"
            if s in {"exc", "exclusion", "exclude", "ex"}:  return "exclusion"
            return s or "unknown"
        stage_dir = os.path.join(self.log_dir, str(trial_id), _bucket(side), f"req{idx:03d}")
        os.makedirs(stage_dir, exist_ok=True)
        p_log    = os.path.join(stage_dir, "2canon_prompt.txt")
        r_log    = os.path.join(stage_dir, "2canon_raw.txt")
        plan_log = os.path.join(stage_dir, "2canon_plan.json")

        with open(p_log, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        _log(
            "namer",
            idx,
            f"mode={'ENTITIES_ONLY' if entities_only else 'FULL'}; "
            f"coverage_required={coverage_required}; partial_ok={partial_coverage_ok}; "
            f"required_full_coverage={required_full_coverage}; max_attempts={max_attempts}"
        )

        existing_vars = _extract_declared_symbols(context.get("smt_program_lines", []))
        reusable_vars: List[Dict[str, Any]] = context.get("reusable_variables", []) or []
        reused_names = {(rv or {}).get("variable_name") for rv in reusable_vars if isinstance(rv, dict)}
        dup_targets = set(existing_vars) | set(reused_names)

        succeeded = False
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            diagnostics: list = []
            p_log_attempt = os.path.join(stage_dir, f"2canon_attempt{attempt:02d}_prompt.txt")
            r_log_attempt = os.path.join(stage_dir, f"2canon_attempt{attempt:02d}_raw.txt")
            plan_log_attempt = os.path.join(stage_dir, f"2canon_attempt{attempt:02d}_plan.json")

            with open(p_log_attempt, "w", encoding="utf-8") as fh:
                fh.write(prompt)

            llm_out: str = self.engine(prompt)[0]
            with open(r_log_attempt, "w", encoding="utf-8") as fh:
                fh.write(llm_out)
            with open(r_log, "a", encoding="utf-8") as fh:
                fh.write(f"--- attempt {attempt} ---{llm_out}")

            m_canon = _NEWCANON_RE.search(llm_out)
            if not m_canon:
                _schema_log(diagnostics, idx=idx, block="top_level", item_index=None,
                            code="MISSING_BLOCKS",
                            message="Could not find <new_canonical_variable_declarations> block.")
                last_error = RuntimeError(f"Missing <new_canonical_variable_declarations> for req#{idx} attempt {attempt}")
                if attempt == max_attempts:
                    break
                _log("namer", idx, f"missing block; retry ({attempt}/{max_attempts})")
                continue

            try:
                canon_list_raw = _parse_json_array_relaxed(m_canon.group(1))
                m_err = _ERRORS_RE.search(llm_out)
                errors_list = _parse_json_array_relaxed(m_err.group(1)) if m_err else []
            except Exception as exc:
                _schema_log(diagnostics, idx=idx, block="top_level", item_index=None,
                            code="JSON_PARSE", message="Failed to parse JSON blocks.", detail=str(exc))
                last_error = exc
                if attempt == max_attempts:
                    break
                _log("namer", idx, f"JSON error; retry ({attempt}/{max_attempts})")
                continue

            # Pass-through unchanged (no auto-attached qualifiers)
            canon_list_raw_aug = list(canon_list_raw)

            # === Rule 1: For stems in existing/reusable: drop items with NO qualifiers; keep ones WITH qualifiers ===
            pre_filtered: List[dict] = []
            for i, o in enumerate(canon_list_raw_aug):
                stem = (o.get("entity_variable_name") or "").strip()
                if stem in dup_targets and not _has_any_qualifiers(o):
                    _schema_log(
                        diagnostics, idx=idx, block="dup_filter", item_index=i,
                        code="DROP_DUP_NO_QUAL",
                        message="Duplicate of existing/reusable stem without qualifiers; dropped.",
                        stem=stem,
                    )
                    continue
                pre_filtered.append(o)
            canon_list_raw_aug = pre_filtered

            # Extract meaning + normalize qualifiers to detailed objects (from *filtered* raw list)
            var_mean, qual_det = _extract_meaning_and_qual_maps(canon_list_raw_aug)

            try:
                # Validate canonical declarations
                canon_list = _validate_decl_list(
                    canon_list_raw_aug,
                    kind="canonical",
                    allow_backcompat=True,
                    enforce_suffix_in_stem=self._enforce_suffix_in_stem,
                    allow_mixed_case=self._allow_mixed_case,
                    collect_sanitize_corrections=[],
                    enable_template_checks=True,
                    diag=diagnostics, context_idx=idx, block_label="new_canonical_variable_declarations"
                )

                # I3 gating: only entities present in canonical forms allowed
                if allowed_canon_forms:
                    bad = [c for c in canon_list if _get_canon(c) not in allowed_canon_forms]
                    if bad:
                        _schema_log(diagnostics, idx=idx, block="new_canonical_variable_declarations", item_index=None,
                                    code="CANON_NOT_IN_ALLOWED",
                                    message="Canonical entity not present in #CANONICAL_FORMS#.",
                                    offending=[_get_canon(b) for b in bad], allowed_sorted=sorted(allowed_canon_forms))
                        raise ValueError("canonical declarations include entities not in #CANONICAL_FORMS#")

                # I1 timeframe token check
                _enforce_timeframe_in_stem(canon_list, allow_mixed_case=self._allow_mixed_case,
                                           diag=diagnostics, context_idx=idx, block_label="new_canonical_variable_declarations")

                # Dedup vs existing SMT declarations (KEEP entries, mark as reuse)
                new_names = [o.get("entity_variable_name") for o in canon_list if isinstance(o, dict)]
                dups_existing = sorted({n for n in new_names if n in existing_vars})
                if dups_existing:
                    _schema_log(
                        diagnostics, idx=idx, block="dedup", item_index=None,
                        code="REUSE_EXISTING_KEEP_ENTRY",
                        message="Stem already declared; keeping entry to preserve qualifiers/meaning; downstream must not re-declare.",
                        affected=dups_existing,
                    )
                    for o in canon_list:
                        if o.get("entity_variable_name") in dups_existing:
                            o["__reuse_existing_symbol"] = True

                # Collision with reusable inputs (KEEP entries, mark as reusable)
                dups_reused = sorted(n for n in new_names if n in reused_names)
                if dups_reused:
                    _schema_log(
                        diagnostics, idx=idx, block="reused_collision", item_index=None,
                        code="REUSED_KEEP_ENTRY",
                        message="Stem is listed as reusable; keeping entry for qualifiers/meaning; downstream must reuse symbol.",
                        names=dups_reused,
                    )
                    for o in canon_list:
                        if o.get("entity_variable_name") in dups_reused:
                            o["__listed_as_reusable"] = True

                # === Rule 2: merge meanings/qualifiers; duplicates by stem become one item with all qualifiers ===
                _merge_meanings_and_qualifiers(canon_list, var_mean, qual_det)
                canon_list = _merge_canonical_duplicates(canon_list)

                # === Rule 3: mark redeclared stems so persistence layer does NOT update their variable_meaning ===
                for o in canon_list:
                    stem = (o.get("entity_variable_name") or "").strip()
                    if stem in dup_targets and _has_any_qualifiers(o):
                        o["__do_not_update_variable_meaning"] = True

                # Optional coverage handling: only retry if full coverage is *required*
                if allowed_canon_forms:
                    present = {_get_canon(c) for c in canon_list}
                    missing = sorted(allowed_canon_forms - present)
                    if missing:
                        _schema_log(diagnostics, idx=idx, block="coverage", item_index=None,
                                    code="COVERAGE_PARTIAL", message="Partial canonical coverage.",
                                    missing_canonical_forms=missing)
                        if required_full_coverage:
                            if attempt < max_attempts:
                                last_error = ValueError("missing canonical: " + ", ".join(missing))
                                _log("namer", idx, f"coverage partial; retry ({attempt}/{max_attempts}): {', '.join(missing)}")
                                raise last_error
                            # last attempt and still missing → hard failure
                            raise ValueError("missing canonical declarations: " + ", ".join(missing))
                        else:
                            # Partial coverage is allowed → accept and continue without retrying
                            warnings.warn(
                                f"SMT namer partial coverage allowed (req#{idx}): missing {', '.join(missing)}",
                                RuntimeWarning
                            )

            except ValueError as ve:
                last_error = ve
                if attempt == max_attempts:
                    break
                continue

            # Success path
            for _o in canon_list:
                _o.pop("_original_entity_variable_name", None)

            plan_payload = {
                "new_canonical_variable_declarations": canon_list,
                "errors": diagnostics or [],
                "stage": "canonical_only",
            }
            with open(plan_log_attempt, "w", encoding="utf-8") as fh:
                json.dump(plan_payload, fh, indent=2, ensure_ascii=False)
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan_payload, fh, indent=2, ensure_ascii=False)

            # Context writes (canonical only)
            context["new_canonical_variable_declarations"] = canon_list
            context["errors"] = diagnostics or []
            context["namer_mode"] = "entities_only" if entities_only else "full"
            context["namer_stage"] = "canonical_only"

            # Append to linked list with PT / concept enrichment
            _extend_all_linked_variales(context, canon_list, idx=idx, canon_info=canon_info)

            _log("namer ✓", idx, f"accepted on attempt {attempt}")
            succeeded = True
            break

        # Fallback synthesize
        if not succeeded:
            _log("namer ⚠ fallback", idx, f"due to: {last_error!r}")
            warnings.warn(f"SMT canonical namer fallback (req#{idx}): {last_error}", RuntimeWarning)

            errors_list = [{
                "invariant": "FALLBACK",
                "problem": "LLM outputs could not be parsed/validated; synthesized canonical declarations",
                "detail": str(last_error) if last_error else "unknown",
            }]

            canon_list: List[dict] = []
            existing = _extract_declared_symbols(context.get("smt_program_lines", []))
            for canon in sorted(allowed_canon_forms):
                info = allowed_info.get(canon, {"type": "", "span": canon})
                typ = info.get("type", "")
                timeframe = _default_timeframe_for_type(typ)
                stem = _synthesize_stem(canon, typ, timeframe, allow_mixed_case=self._allow_mixed_case)
                template = _template_for_type(typ)
                if (stem in existing) or any((rv.get("variable_name") == stem) for rv in (context.get("reusable_variables") or [])):
                    continue
                canon_list.append({
                    "span": info.get("span", canon),
                    "entity_variable_name": stem,
                    "template": template,
                    "timeframe": timeframe,
                    "entity_canonical_form_from_entity_canonical_forms_block": canon,
                    "entity_type": info.get("type", "") or "unknown",
                    "variable_meaning": f"auto-synthesized variable for {canon} ({typ or 'unknown type'}) with timeframe {timeframe}",
                    "usage_description": "fallback synthesized declaration to preserve canonical coverage",
                    "qualifier_predicates": [],
                    "qualifier_predicates_for_semantics_not_already_captured_with_stem": [],
                    "qualifier_predicates_detailed": [],
                })

            canon_list.sort(key=_sort_key)
            try:
                _enforce_timeframe_in_stem(canon_list, allow_mixed_case=self._allow_mixed_case)
            except Exception as e:
                errors_list.append({
                    "invariant": "I1-relaxed-fallback",
                    "problem": "timeframe token check failed in synthesized stem; proceeding",
                    "detail": str(e),
                })

            plan_log_attempt = os.path.join(stage_dir, f"2canon_attempt{attempt:02d}_plan.json")
            with open(plan_log_attempt, "w", encoding="utf-8") as fh:
                json.dump({
                    "new_canonical_variable_declarations": canon_list,
                    "errors": errors_list,
                    "stage": "canonical_only",
                    "fallback": True,
                }, fh, indent=2, ensure_ascii=False)

            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump({
                    "new_canonical_variable_declarations": canon_list,
                    "errors": errors_list,
                    "stage": "canonical_only",
                    "fallback": True,
                }, fh, indent=2, ensure_ascii=False)

            context["new_canonical_variable_declarations"] = canon_list
            context["errors"] = errors_list
            context["namer_mode"] = "entities_only" if entities_only else "full"
            context["namer_stage"] = "canonical_only"

            # Append synthesized vars to linked list with enrichment
            _extend_all_linked_variales(context, canon_list, idx=idx, canon_info=canon_info)

        return context
