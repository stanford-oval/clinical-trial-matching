# modules/PatientCanonicalEntityEnricher.py
# -*- coding: utf-8 -*-
"""
PatientCanonicalEntityEnricher
────────────────────────────────────────────────────────────────────────────
Enrich *patient-side* entities and assign a stable snake_case
entity_canonical_form derived from Preferred Term (PT).

Outputs (into context):
  - preferred_term_canonical_form_map
  - requirement_bundles[*].entities[*].entity_canonical_form set
  - valid_entities_by_req[*][*].entity_canonical_form  ← backfilled for canonical coder
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json, re, hashlib

# ───── helpers ─────

_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")

def _norm(x: Any) -> str:
    return str(x or "").strip()

def _to_int(x: Any) -> Optional[int]:
    if x is None: return None
    try: return int(x)
    except Exception:
        try: return int(str(x).strip())
        except Exception: return None

def _strip_tag(term: Optional[str]) -> str:
    return _TAG_RE.sub("", term or "").strip()

def _to_var(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def _sha12(s: str) -> str:
    return hashlib.sha256((_norm(s)).encode("utf-8")).hexdigest()[:12]

def _maybe_set(dst: Dict[str, Any], key: str, val: Any) -> None:
    if val in (None, "", [], {}): return
    cur = dst.get(key)
    if cur in (None, "", [], {}):
        dst[key] = val

# ───── valid-entity index & match ─────

def _build_valid_index_for_req(valid_for_req: Dict[str, Any]) -> Dict[Tuple[Optional[int], Optional[int], Optional[str]], Dict[str, Any]]:
    idx: Dict[Tuple[Optional[int], Optional[int], Optional[str]], Dict[str, Any]] = {}
    for _, node in (valid_for_req or {}).items():
        st, ed = _to_int(node.get("start")), _to_int(node.get("end"))
        span_l = _norm(node.get("extracted_span")).lower() or None
        idx[(st, ed, span_l)] = node
        idx[(st, ed, None)] = node
        if span_l:
            idx[(None, None, span_l)] = node
    return idx

def _find_valid_match(idx, surface: Any, start: Any, end: Any) -> Optional[Dict[str, Any]]:
    st, ed = _to_int(start), _to_int(end)
    span_l = (_norm(surface).lower() or None)
    return idx.get((st, ed, span_l)) or idx.get((st, ed, None)) or idx.get((None, None, span_l))

# ───── registry ─────

class _CanonicalFormRegistry:
    """PreferredTerm(no tag) → snake_case; reuse across calls; persistable."""
    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self._pt2form: Dict[str, str] = dict(mapping or {})
        self._form2pts: Dict[str, set] = {}
        for pt, form in self._pt2form.items():
            self._form2pts.setdefault(form, set()).add(pt)

    def to_dict(self) -> Dict[str, str]:
        return dict(self._pt2form)

    @staticmethod
    def _base(pt: str) -> str:
        return _to_var(_strip_tag(pt))

    def get_or_assign(self, preferred_term: str, concept_id: Optional[str] = None) -> str:
        pt_key = _strip_tag(preferred_term or "")
        base = self._base(pt_key)
        if not pt_key:
            pt_key = preferred_term or ""
        if pt_key in self._pt2form:
            return self._pt2form[pt_key]

        if base not in self._form2pts:
            self._pt2form[pt_key] = base
            self._form2pts.setdefault(base, set()).add(pt_key)
            return base

        cand = _to_var(f"{base}_{concept_id}") if concept_id else _to_var(f"{base}_{_sha12(pt_key)[:6]}")
        if cand in self._form2pts and pt_key not in self._form2pts[cand]:
            k = 2
            while True:
                c2 = f"{cand}_{k}"
                if c2 not in self._form2pts:
                    cand = c2
                    break
                k += 1
        self._pt2form[pt_key] = cand
        self._form2pts.setdefault(cand, set()).add(pt_key)
        return cand

    @classmethod
    def load(cls, path: Optional[Path]) -> "_CanonicalFormRegistry":
        if path is None or not path.exists():
            return cls({})
        try:
            mapping = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                mapping = {}
        except Exception:
            mapping = {}
        return cls(mapping)

    def save(self, path: Optional[Path]) -> None:
        if path is None: return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._pt2form, ensure_ascii=False, indent=2), encoding="utf-8")

# ───── main enricher ─────

class PatientCanonicalEntityEnricher:
    """
    Enrich requirement_bundles with stable snake_case `entity_canonical_form`
    and backfill `valid_entities_by_req` so the canonical coder can gate on it.
    """

    def __init__(self, persist_base: Optional[str | Path] = None) -> None:
        self._persist_base = Path(persist_base).expanduser() if persist_base else None
        self._resolved_path: Optional[Path] = None
        self.registry = _CanonicalFormRegistry({})

    # — internal: resolve per-patient path & load/save —
    def _resolve_map_path(self, ctx: Dict[str, Any]) -> Optional[Path]:
        if self._persist_base is None:
            return None
        pid = _norm(ctx.get("patient_id")) or _norm(ctx.get("note_id")) or "unknown_patient"
        return self._persist_base / pid / "pt2form.json"

    def _ensure_registry_for_ctx(self, ctx: Dict[str, Any]) -> None:
        path = self._resolve_map_path(ctx)
        if path is None:
            self._resolved_path = None
            self.registry = _CanonicalFormRegistry({})
            return
        if self._resolved_path is None or path != self._resolved_path:
            self.registry = _CanonicalFormRegistry.load(path)
            self._resolved_path = path

    def _maybe_save(self) -> None:
        _CanonicalFormRegistry.save(self.registry, self._resolved_path)

    # — core enrichment —
    def _enrich_bundle_entities(self, bundle: Dict[str, Any], ver_for_req: Dict[str, Any]) -> None:
        idx = _build_valid_index_for_req(ver_for_req or {})
        for e in (bundle.get("entities") or []):
            ent = e.get("entity") if isinstance(e.get("entity"), dict) else e
            if not isinstance(ent, dict):
                continue

            match = _find_valid_match(
                idx,
                ent.get("span") or ent.get("surface_string") or e.get("span"),
                ent.get("start") or e.get("start"),
                ent.get("end") or e.get("end"),
            )

            if match:
                _maybe_set(ent, "entity_name",        match.get("entity_name"))
                _maybe_set(ent, "preferred_term",       match.get("preferred_term"))
                _maybe_set(ent, "fully_specified_name", match.get("fully_specified_name"))
                _maybe_set(ent, "type",                 match.get("type"))
                _maybe_set(ent, "conceptId",            match.get("conceptId"))
                _maybe_set(ent, "select_reason",        match.get("select_reason"))

            pt = _strip_tag(
                ent.get("preferred_term")
                or ent.get("fully_specified_name")
                or ent.get("span")
                or e.get("span")
                or ""
            )
            if pt:
                form = self.registry.get_or_assign(pt, str(ent.get("conceptId") or "") or None)

                e["entity_canonical_form"] = form
                if "entity" in e and isinstance(e["entity"], dict):
                    e["entity"]["entity_canonical_form"] = form

                if match is not None and isinstance(match, dict):
                    match["entity_canonical_form"] = form
                    # 若 bundle 有 select_reason 而 match 没有，也补回去（双向一致）
                    if ent.get("select_reason") and not match.get("select_reason"):
                        match["select_reason"] = ent.get("select_reason")
                    if ent.get("entity_name") and not match.get("entity_name"):
                        match["entity_name"] = ent.get("entity_name")

    # — public API —
    def enrich_ctx(
        self,
        ctx: Dict[str, Any],
        *,
        bundles_key: str = "requirement_bundles",
        valid_key: str = "valid_entities_by_req",
        mapping_key: str = "preferred_term_canonical_form_map",
    ) -> Dict[str, Any]:
        """Enrich context in-place and return it."""
        bundles: List[Dict[str, Any]] = ctx.get(bundles_key) or []
        ver_all: Dict[str, Any] = ctx.get(valid_key) or {}

        # Load registry per patient
        self._ensure_registry_for_ctx(ctx)

        # Enrich each bundle using its corresponding valid-entities bucket
        for b in bundles:
            rid = b.get("req_index")
            ver_for_req = ver_all.get(str(rid)) or ver_all.get(rid) or {}
            self._enrich_bundle_entities(b, ver_for_req)

        # Second pass: ensure every valid entity gets a canonical form even without bundles
        for _, ents in (ver_all.items() if isinstance(ver_all, dict) else []):
            if not isinstance(ents, dict):
                continue
            for node in ents.values():
                if not isinstance(node, dict):
                    continue
                if node.get("entity_canonical_form"):
                    continue
                pt = _strip_tag(
                    node.get("preferred_term")
                    or node.get("fully_specified_name")
                    or node.get("extracted_span")
                    or ""
                )
                if not pt:
                    continue
                form = self.registry.get_or_assign(pt, str(node.get("conceptId") or "") or None)
                node["entity_canonical_form"] = form

        # Normalize valid_entities_by_req keys to strings (avoid int/str key mismatch downstream)
        if isinstance(ver_all, dict) and any(not isinstance(k, str) for k in ver_all.keys()):
            ctx[valid_key] = {str(k): v for k, v in ver_all.items()}

        # Save mapping if a path was configured
        self._maybe_save()

        # Expose PT→form map for prompts/diagnostics
        ctx[mapping_key] = self.registry.to_dict()
        return ctx
