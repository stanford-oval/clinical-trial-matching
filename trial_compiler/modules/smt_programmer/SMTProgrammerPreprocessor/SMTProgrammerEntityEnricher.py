#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMTProgrammerEntityEnricher (optimized)
- Prebuild valid-entity indices once per call
- Only save PT→form map if modified (dirty flag)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json, re, hashlib

# ────────────────────────── helpers ──────────────────────────

_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")  # strip trailing “ ( … )” semantic tag

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

def _lower_or_none(x: Any) -> Optional[str]:
    s = _norm(x)
    return s.lower() if s else None

def _truthy(x: Any) -> bool:
    return bool(_norm(x))

def _maybe_set(dst: Dict[str, Any], key: str, val: Any) -> None:
    if not _truthy(val): return
    cur = dst.get(key)
    if not _truthy(cur):
        dst[key] = val

def _resolve_persist_base(p: Optional[str | Path]) -> Optional[Path]:
    """
    Return a DIRECTORY to serve as the base for trial/side nesting.
    - If p endswith .json → return parent directory of p.
    - Else → return p as a directory (even if it doesn't exist yet).
    """
    if p is None:
        return None
    pp = Path(p).expanduser()
    if pp.suffix.lower() == ".json":
        return pp.parent
    return pp

def _resolve_mapping_path(base_dir: Optional[Path], trial_id: Optional[str], inc_exc: Optional[str]) -> Optional[Path]:
    """
    Build <base_dir>/<trial_id>/<inc_exc>/pt2form.json.
    If base_dir is None, return None (no persistence).
    """
    if base_dir is None:
        return None
    tid = (trial_id or "unknown_trial").strip() or "unknown_trial"
    sie = (inc_exc  or "side").strip() or "side"
    return base_dir / tid / sie / "pt2form.json"

# ────────────────────────── entity matching ──────────────────────────

def _build_valid_index_for_req(valid_for_req: Dict[str, Any]) -> Dict[Tuple[Optional[int], Optional[int], Optional[str]], Dict[str, Any]]:
    """
    Index by (start, end, lower(span)) with fallbacks for (start,end,None) and (None,None,lower(span)).
    """
    idx: Dict[Tuple[Optional[int], Optional[int], Optional[str]], Dict[str, Any]] = {}
    for _, node in (valid_for_req or {}).items():
        st, ed = _to_int(node.get("start")), _to_int(node.get("end"))
        span_l = _lower_or_none(node.get("extracted_span"))
        idx[(st, ed, span_l)] = node
        idx[(st, ed, None)] = node
        if span_l:
            idx[(None, None, span_l)] = node
    return idx

def _find_valid_match(
    idx: Dict[Tuple[Optional[int], Optional[int], Optional[str]], Dict[str, Any]],
    surface: Any,
    start: Any,
    end: Any
) -> Optional[Dict[str, Any]]:
    st, ed, span_l = _to_int(start), _to_int(end), _lower_or_none(surface)
    return idx.get((st, ed, span_l)) or idx.get((st, ed, None)) or idx.get((None, None, span_l))

# ────────────────────────── registry ──────────────────────────

class CanonicalFormRegistry:
    """Stable mapping: PreferredTerm(no tag) → snake_case canonical_form."""

    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self._pt2form: Dict[str, str] = dict(mapping or {})
        self._form2pts: Dict[str, set] = {}
        for pt, form in self._pt2form.items():
            self._form2pts.setdefault(form, set()).add(pt)

    def to_dict(self) -> Dict[str, str]:
        return dict(self._pt2form)

    @staticmethod
    def _base_form_from_pt(pt: str) -> str:
        return _to_var(_strip_tag(pt))

    def get_or_assign(self, preferred_term: str, concept_id: Optional[str] = None) -> str:
        pt_key = _strip_tag(preferred_term or "")
        base = self._base_form_from_pt(pt_key)
        if not pt_key:
            pt_key = preferred_term or ""
        if pt_key in self._pt2form:
            return self._pt2form[pt_key]

        # free base
        if base not in self._form2pts:
            self._pt2form[pt_key] = base
            self._form2pts.setdefault(base, set()).add(pt_key)
            return base

        # collide → add disambiguator
        candidate = _to_var(f"{base}_{concept_id}") if concept_id else _to_var(f"{base}_{_sha12(pt_key)[:6]}")
        if candidate in self._form2pts and pt_key not in self._form2pts[candidate]:
            k = 2
            while True:
                cand2 = f"{candidate}_{k}"
                if cand2 not in self._form2pts:
                    candidate = cand2
                    break
                k += 1
        self._pt2form[pt_key] = candidate
        self._form2pts.setdefault(candidate, set()).add(pt_key)
        return candidate

    @classmethod
    def load(cls, path: Optional[Path]) -> "CanonicalFormRegistry":
        if path is None:
            return cls({})
        if path.exists():
            try:
                mapping = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(mapping, dict): mapping = {}
            except Exception:
                mapping = {}
        else:
            mapping = {}
        return cls(mapping)

    @staticmethod
    def save(registry: "CanonicalFormRegistry", path: Optional[Path]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

# ────────────────────────── Enricher class ──────────────────────────

class SMTProgrammerEntityEnricher:
    """
    Enrich canonical entities from valid_entities_by_req and attach canonical_form.
    Also ensures trial context ("trial_id", "inc_exc") is present on each block if available in ctx.

    Optimizations:
      * Prebuild valid-entity indices once per call
      * Save PT→form map only when modified (dirty flag)
    """

    def __init__(self, persist_path: Optional[str | Path] = None) -> None:
        # Store only the BASE directory; actual file is resolved per trial/side
        self._persist_base: Optional[Path] = _resolve_persist_base(persist_path)
        self._resolved_path: Optional[Path] = None  # last resolved file path
        self.registry = CanonicalFormRegistry()     # loaded lazily per resolved path
        self._dirty = False

    # ------------- core enrichment -------------

    def _enrich_entities_for_req(self, entities: List[Dict[str, Any]], valid_for_req_idx: Dict, *, changed_flag: List[bool]) -> None:
        idx = valid_for_req_idx
        for e in entities or []:
            ent = (e or {}).get("entity", {}) or {}
            # Only enrich canonical entities
            if not bool(ent.get("is_canonical_entity", True)):
                continue

            match = _find_valid_match(idx, ent.get("surface_string"), ent.get("start"), ent.get("end")) \
                 or _find_valid_match(idx, ent.get("surface_string") or ent.get("text"), ent.get("start"), ent.get("end"))

            if match:
                _maybe_set(ent, "preferred_term",       match.get("preferred_term"))
                _maybe_set(ent, "fully_specified_name", match.get("fully_specified_name"))
                _maybe_set(ent, "type",                 match.get("type"))
                _maybe_set(ent, "conceptId",            match.get("conceptId"))

            # canonical_form from PT (fallback to surface_string)
            pt = _strip_tag(ent.get("preferred_term") or ent.get("surface_string") or "")
            if pt:
                before = self.registry.to_dict().get(_strip_tag(ent.get("preferred_term") or ent.get("surface_string") or ""), None)
                ent["canonical_form"] = self.registry.get_or_assign(pt, ent.get("conceptId"))
                after = self.registry.to_dict().get(_strip_tag(ent.get("preferred_term") or ent.get("surface_string") or ""), None)
                if before != after:
                    changed_flag[0] = True

    def _ensure_block_context(self, block: Dict[str, Any], trial_id: Optional[str], inc_exc: Optional[str]) -> None:
        """Attach trial context to the block if not already present."""
        if trial_id:
            _maybe_set(block, "trial_id", trial_id)
        if inc_exc:
            _maybe_set(block, "inc_exc", inc_exc)

    def _ensure_registry_for(self, trial_id: Optional[str], inc_exc: Optional[str]) -> None:
        """
        Ensure `self.registry` is loaded for the resolved per-trial/side path.
        """
        path = _resolve_mapping_path(self._persist_base, trial_id, inc_exc)
        if path is None:
            # No persistence requested; keep in-memory only.
            self._resolved_path = None
            return
        if self._resolved_path is None or path != self._resolved_path:
            # (Re)load for this specific file
            self.registry = CanonicalFormRegistry.load(path)
            self._resolved_path = path

    def _maybe_save_registry(self, changed: bool) -> None:
        if changed:
            CanonicalFormRegistry.save(self.registry, self._resolved_path)

    # ------------- public API -------------

    def enrich_ctx(
        self,
        ctx: Dict[str, Any],
        *,
        src_key: str = "requirements_entities_attributes_top_level",
        valid_key: str = "valid_entities_by_req",
        mapping_key: str = "preferred_term_canonical_form_map",
    ) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = ctx.get(src_key) or []
        valid_by_req: Dict[str, Any] = ctx.get(valid_key) or {}

        # pull context once
        trial_id = _norm(ctx.get("trial_id"))
        inc_exc  = _norm(ctx.get("inc_exc"))

        # Resolve and (re)load per-trial/side mapping
        self._ensure_registry_for(trial_id, inc_exc)

        # prebuild indices once
        prebuilt: Dict[str, Dict] = {
            str(k): _build_valid_index_for_req(v or {}) for k, v in (valid_by_req or {}).items()
        }
        changed_flag = [False]

        for block in blocks:
            # ensure block context is present
            self._ensure_block_context(block, trial_id, inc_exc)

            rid_str = str(block.get("requirement_id", ""))
            idx = prebuilt.get(str(rid_str)) or prebuilt.get(str(_to_int(rid_str))) or {}
            self._enrich_entities_for_req(block.get("entities") or [], idx, changed_flag=changed_flag)

        # Persist mapping for this trial/side if a path is set (only when changed)
        self._maybe_save_registry(changed_flag[0])

        ctx[mapping_key] = self.registry.to_dict()
        return ctx

    def enrich_blocks(
        self,
        blocks: List[Dict[str, Any]],
        valid_by_req: Dict[str, Any],
        *,
        trial_id: Optional[str] = None,
        inc_exc: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Standalone enrichment for a list of blocks. Optionally pass trial_id/inc_exc to attach to blocks.
        Persistence path is resolved as <persist_base>/<trial_id>/<inc_exc>/pt2form.json.
        """
        trial_id = _norm(trial_id)
        inc_exc  = _norm(inc_exc)

        # Resolve and (re)load per-trial/side mapping
        self._ensure_registry_for(trial_id, inc_exc)

        prebuilt: Dict[str, Dict] = {
            str(k): _build_valid_index_for_req(v or {}) for k, v in (valid_by_req or {}).items()
        }
        changed_flag = [False]

        for block in blocks or []:
            self._ensure_block_context(block, trial_id, inc_exc)
            rid_str = str(block.get("requirement_id", ""))
            idx = prebuilt.get(str(rid_str)) or prebuilt.get(str(_to_int(rid_str))) or {}
            self._enrich_entities_for_req(block.get("entities") or [], idx, changed_flag=changed_flag)

        # Persist mapping for this trial/side if a path is set (only when changed)
        self._maybe_save_registry(changed_flag[0])

        return blocks, self.registry.to_dict()
