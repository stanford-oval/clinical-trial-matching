from __future__ import annotations
from typing import Any, Dict, List
import re

# ────────────────────────── stdlib ──────────────────────────
from pathlib import Path
from copy import deepcopy
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import hashlib

# ────────────────────────── typing helpers ──────────────────────────
class _ChatEngine(Protocol):
    def __call__(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> Tuple[str, ...]:
        ...

# ────────────────────────── misc helpers ──────────────────────────
def _chunks(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

# ────────────────────────── SNOMED helpers ──────────────────────────
import requests
import pandas as pd

BASE_URL = "http://localhost:8080"
BRANCH = "MAIN"
FORM = "inferred"


def load_domain_attribute_map(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Return two dicts: domain→attrIds and attrId→attrName (FSN)."""
    df = (
        pd.read_excel(path)
        if path.lower().endswith((".xls", ".xlsx"))
        else pd.read_csv(path)
    )

    required = {"domainId", "referencedComponentId", "attributeFSN"}
    if not required.issubset(df.columns):
        raise ValueError(f"File must contain columns: {required}")

    df = df.astype(str)

    domain_attr_ids = (
        df.groupby("domainId")["referencedComponentId"].apply(list).to_dict()
    )
    attr_id_to_name = dict(zip(df["referencedComponentId"], df["attributeFSN"]))
    return domain_attr_ids, attr_id_to_name


# --- terminology queries ----------------------------------------------------
def concept_name_to_id(name: str) -> str:
    """Return the first *active* conceptId for a term via SNOMED browser API."""
    url = f"{BASE_URL}/browser/{BRANCH}/descriptions"
    params = {"term": name, "activeFilter": True, "limit": 5}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to query terminology service: {exc}") from exc

    items = resp.json().get("items", [])
    active_item = next(
        (item for item in items if item.get("concept", {}).get("active")), None
    )
    if not active_item:
        raise ValueError(f"No *active* concept matches ‘{name}’")
    return active_item["concept"]["conceptId"]


def fetch_ancestors(concept_id: str, form: str = FORM) -> List[str]:
    url = f"{BASE_URL}/browser/{BRANCH}/concepts/{concept_id}/ancestors?form={form}"
    r = requests.get(url, timeout=10)
    if r.status_code == 400 and form == "inferred":
        return fetch_ancestors(concept_id, form="stated")
    if r.status_code == 404:
        raise ValueError(f"Concept {concept_id} is missing on branch {BRANCH}")
    r.raise_for_status()
    return [item["conceptId"] for item in r.json()]


# --- attribute helpers ------------------------------------------------------
def allowed_attributes(
    concept_name: str,
    domain_attribute_map: Dict[str, List[str]],
    attr_id_to_name: Dict[str, str],
    *,
    return_ids: bool = False,
) -> Optional[List[str]]:
    """Return the list of allowed attribute names (or IDs if *return_ids*)."""
    concept_id = concept_name_to_id(concept_name)
    ancestors = fetch_ancestors(concept_id)
    collected: List[str] = []
    for anc in [concept_id] + ancestors:
        if anc in domain_attribute_map:
            collected.extend(domain_attribute_map[anc])
    if return_ids:
        return collected
    return [attr_id_to_name.get(aid, aid) for aid in collected]


def hit_ancestors(
    concept_name: str,
    domain_attribute_map: Dict[str, List[str]],
    *,
    include_self: bool = True,
) -> List[str]:
    concept_id = concept_name_to_id(concept_name)
    lineage = [concept_id] if include_self else []
    lineage.extend(fetch_ancestors(concept_id))
    return [anc for anc in lineage if anc in domain_attribute_map]


# ────────────────────────── JSON‑safe ctx helpers ──────────────────────────
def _strip_unserialisable(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow‑copy *ctx* dropping obvious non‑serialisable entries."""
    clean = deepcopy(ctx)
    for k, v in list(clean.items()):
        if k.endswith("engine") or k == "model" or callable(v):
            clean.pop(k, None)
    return clean


def save_ctx(ctx: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_strip_unserialisable(ctx), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_ctx(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _write_mbench_json(p: str | Path, obj: Any) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)   # ensure dir exists
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ────────────────────────── var‑name utilities ──────────────────────────
_slug_re = re.compile(r"[^A-Za-z0-9]+")
_multire = re.compile(r"_+")

# --- PT helpers -------------------------------------------------------------
_TAG_RE = re.compile(r"\s*\([^)]*\)$")          # matches trailing “ ( … )”

def _strip_tag(term: str | None) -> str:
    """Return Preferred Term without the trailing semantic tag."""
    return _TAG_RE.sub("", term or "").strip()

def _to_var(term: str) -> str:
    """Preferred‑term → snake_case variable name (tag already stripped)."""
    clean = _strip_tag(term).lower()
    parts = re.split(r"[^a-z0-9]+", clean)
    return "_".join(p for p in parts if p) or "unnamed"


def _clean_attr_name(attr_type: str | None) -> str:
    # kept for backward‑compat elsewhere; now uses PT helpers
    if not attr_type:
        return ""
    core = attr_type.rstrip().removesuffix("(attribute)").strip()
    return _to_var(core)


# ────────────────────────── bundling logic ──────────────────────────
def _best_value(av: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    best = av.get("best_attribute_value") or []
    if best:
        return best[0].get("conceptId"), best[0].get("preferred_term")
    for block in av.get("potential_matches_by_attribute", {}).values():
        for cand in block.get("filtered_matches", []):
            if cand.get("keep", "").lower() == "yes":
                return cand.get("conceptId"), cand.get("preferred_term")
    return None, None


def bundle_requirements(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform *ctx* into serialisable requirement bundles.

    • All variable names come from Preferred Terms (PT) with semantic tags removed.
    """
    bundles: Dict[str, Dict[str, Any]] = {}

    ent_lookup      = ctx.setdefault("entity_var_lookup", {})
    attr_lookup     = ctx.setdefault("attr_var_lookup", {})
    attr_val_lookup = ctx.setdefault("attr_val_var_lookup", {})

    for group in ctx.get("groups", []):
        for mem in group.get("members", []):
            trial_full = mem["trial"]                      # e.g. NCT123456_req002
            trial_id, _, req_idx = trial_full.partition("_req")
            key = f"{trial_id}::{req_idx or '000'}"

            req_obj = bundles.setdefault(
                key,
                {
                    "trial_id"  : trial_id,
                    "req_index" : int(req_idx) if req_idx else 0,
                    "requirement": mem["requirement"],
                    "entities"  : [],
                },
            )

            # ─── entity symbol ─────────────────────────────────────
            e        = mem["entity"]
            ent_pt   = _strip_tag(e.get("preferred_term") or e["surface_string"])
            ent_var  = _to_var(ent_pt)

            ent_lookup.setdefault(
                ent_var,
                {
                    "preferred_term"      : ent_pt,
                    "fully_specified_name": e.get("fully_specified_name"),
                    "conceptId"           : e.get("conceptId"),
                },
            )

            ent_obj = {
                "entity_canonical_form"            : ent_var,
                "span"                : e["surface_string"],
                "preferred_term"      : ent_pt,
                "fully_specified_name": e.get("fully_specified_name"),
                "type"                : e.get("type"),
                "conceptId"           : e.get("conceptId"),
                "definition"          : e.get("definition"),
                "start"               : e.get("start"),
                "end"                 : e.get("end"),
                "attributes"          : [],
            }

            # ─── attribute/value pairs ────────────────────────────
            for av in mem.get("attribute_value_pairs", []):
                val_cid, val_term = _best_value(av)
                if not val_cid:
                    continue

                attr_cid = next(
                    (
                        blk.get("id")
                        for blk in av.get("potential_matches_by_attribute", {}).values()
                        if blk.get("id")
                    ),
                    None,
                )

                # Preferred‑term for the attribute
                attr_pt  = _strip_tag(
                    (av.get("attribute_type") or "").removesuffix("(attribute)").strip()
                )
                attr_var = _to_var(attr_pt)

                # Preferred‑term for the value
                val_var = _to_var(val_term or "")

                # populate look‑ups once
                attr_lookup.setdefault(
                    attr_var,
                    {
                        "attr_type": attr_pt,
                        "attr_cid" : attr_cid,
                    },
                )
                attr_val_lookup.setdefault(
                    val_var,
                    {
                        "value_term": val_term,
                        "value_cid" : val_cid,
                    },
                )

                ent_obj["attributes"].append(
                    {
                        "attribute_class_canonical_form"  : attr_var,
                        "attribute_value_canonical_form" : val_var,
                        "attr_type" : attr_pt,
                        "attr_cid"  : attr_cid,
                        "value_cid" : val_cid,
                        "value_term": val_term,
                    }
                )

            req_obj["entities"].append(ent_obj)

    # deterministic order for downstream diffing / testing
    return sorted(bundles.values(), key=lambda r: (r["trial_id"], r["req_index"]))


# ────────────────────────────────────────────────────────────────────────────
# helpers
_strip_tag = lambda s: re.sub(r"\s*\([^)]+\)\s*$", "", s or "").strip()
_to_var    = lambda s: re.sub(r"\W+", "_", s).lower()

from typing import Dict, Any, List, Tuple

def bundle_requirements_from_finals(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Derive requirement-bundles **after** consolidate_final_attributes[_free].

    Reads:
        ctx["groups"][i]["final_attribute_values"]
        ctx["free_groups"][i]["final_attribute_values_free"]
    Writes nothing – it only returns the bundle list.

    Adds:
        For each entity, includes:
          - "offsets": list[[start, end], ...]
          - "start"/"end" if exactly one offset is known
    """

    bundles: Dict[str, Dict[str, Any]] = {}

    # ────────────────────────────────────────────────────────────────────
    # Build an offsets index from `members` blocks:
    # key = (trial_id, req_idx_str, surface_string) → set[(start,end)]
    # ────────────────────────────────────────────────────────────────────
    def _trial_parts(trial_full: str) -> Tuple[str, str]:
        trial_id, _, req_idx = trial_full.partition("_req")
        return trial_id, (req_idx or "000")

    entity_offsets: Dict[Tuple[str, str, str], set] = {}

    def _ingest_members(members: List[Dict[str, Any]]) -> None:
        for m in members or []:
            trial_id, req_idx = _trial_parts(m.get("trial", ""))
            ent = m.get("entity", {})
            if not isinstance(ent, dict):
                continue
            surf = ent.get("surface_string") or ent.get("preferred_term") or ""
            if not surf:
                continue
            try:
                s = int(ent.get("start"))
                e = int(ent.get("end"))
            except (TypeError, ValueError):
                continue
            entity_offsets.setdefault((trial_id, req_idx, surf), set()).add((s, e))

    for g in ctx.get("groups", []) or []:
        _ingest_members(g.get("members", []))

    for g in ctx.get("free_groups", []) or []:
        # Some pipelines use "members" even for free groups; others "members_free".
        _ingest_members(g.get("members", []))
        _ingest_members(g.get("members_free", []))

    # Helper: look up offsets for a given (trial_id, req_idx, entity_surface)
    def _lookup_offsets(trial_id: str, req_idx: str, surface: str) -> List[List[int]]:
        offs = entity_offsets.get((trial_id, req_idx, surface))
        if not offs:
            return []
        return [[s, e] for (s, e) in sorted(offs)]

    # ------------------------------------------------------------------ #
    def _add(av_rec: Dict[str, Any], is_free: bool = False) -> None:
        """Ingest one attribute-value record (free or SNOMED)."""
        trial_full = av_rec["trial"]                     # e.g. NCT02509286_req007
        trial_id, req_idx = _trial_parts(trial_full)
        key = f"{trial_id}::{req_idx}"

        bundle = bundles.setdefault(
            key,
            {
                "trial_id":   trial_id,
                "req_index":  int(req_idx),
                "requirement": av_rec["requirement"],
                "entities":   [],
            },
        )

        # ── entity block ──
        ent_span   = av_rec["entity"]                    # surface string in finals
        ent_pt     = _strip_tag(ent_span)
        ent_var    = _to_var(ent_pt)

        # fetch offsets using the ORIGINAL surface string stored in finals
        offsets = _lookup_offsets(trial_id, req_idx, ent_span)

        ent_block: Dict[str, Any] = next(
            (e for e in bundle["entities"] if e["entity_canonical_form"] == ent_var),
            None,
        )
        if ent_block is None:
            ent_block = {
                "entity_canonical_form": ent_var,
                "span": ent_span,               # original surface string
                "preferred_term": ent_pt,
                "attributes": [],
                "offsets": offsets,             # ← NEW
            }
            if len(offsets) == 1:
                ent_block["start"], ent_block["end"] = offsets[0]  # ← convenience
            bundle["entities"].append(ent_block)
        else:
            # merge offsets if we see this entity again
            if offsets:
                # ensure deterministic + deduped merge
                existing = {tuple(x) for x in ent_block.get("offsets", [])}
                merged = sorted(existing.union({tuple(x) for x in offsets}))
                ent_block["offsets"] = [list(x) for x in merged]
                if len(merged) == 1:
                    ent_block["start"], ent_block["end"] = merged[0]
                else:
                    ent_block.pop("start", None)
                    ent_block.pop("end", None)

        # Place-holder rows (attribute_type is None) carry *no* attribute
        attr_type = av_rec.get("attribute_type")
        if attr_type:
            attr_pt  = _strip_tag(attr_type)
            attr_var = _to_var(attr_pt)
            val_term = av_rec.get("original_attribute_value") or av_rec.get("attribute_value") or ""
            val_var  = _to_var(val_term)

            ent_block["attributes"].append(
                {
                    "attribute_class_canonical_form": attr_var,
                    "attribute_value_canonical_form": val_var,
                    "attr_type": attr_pt,
                    "value_term": val_term,
                    "qualifier": av_rec.get("qualifier"),
                    "qualifier_span": av_rec.get("qualifier_span"),
                    "qualifier_description": av_rec.get("qualifier_description") if is_free else None,
                    "best_attribute_value": av_rec.get("best_attribute_value", []),
                }
            )

    # ------------------------------------------------------------------ #
    # ingest *all* groups
    for g in ctx.get("groups", []) or []:
        for rec in g.get("final_attribute_values", []) or []:
            _add(rec, is_free=False)

    for g in ctx.get("free_groups", []) or []:
        for rec in g.get("final_attribute_values_free", []) or []:
            _add(rec, is_free=True)

    # deterministic order → easier diffing / testing
    return sorted(bundles.values(), key=lambda b: (b["trial_id"], b["req_index"]))






# ────────────────────────────────────────────────────────────────────────────
# helpers
_strip_tag = lambda s: re.sub(r"\s*\([^)]+\)\s*$", "", s or "").strip()
_to_var    = lambda s: re.sub(r"\W+", "_", s).lower()

# ────────────────────────────────────────────────────────────────────────────
def _to_var(s: str) -> str:
    """小写化，非字母数字→下划线，合并多余下划线。"""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _split_note_id(note: str) -> Tuple[str, int]:
    """trec-20214_req007 → ('trec-20214', 7)；兜底 req_index=0。"""
    if not isinstance(note, str):
        return "", 0
    head, _, tail = note.partition("_req")
    try:
        req_idx = int(re.match(r"\d+", tail or "0").group(0)) if tail else 0
    except Exception:
        req_idx = 0
    return head, req_idx

def bundle_requirements_from_verified(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    只从 member['final_attribute_type_canonical_value'] 聚合，跳过没有 final 的 member。
    返回稳定排序的 bundle 列表。
    """
    bundles: Dict[str, Dict[str, Any]] = {}

    for group in ctx.get("groups", []) or []:
        for member in group.get("members", []) or []:
            finals = member.get("final_attribute_type_canonical_value") or []
            if not isinstance(finals, list) or not finals:
                continue  # 只用 final，没就跳过

            trial_full: str = member.get("trial", "") or ""
            trial_id, req_idx = _split_note_id(trial_full)
            key = f"{trial_id}::{req_idx:03d}"

            b = bundles.setdefault(
                key,
                {
                    "trial_id": trial_id,
                    "req_index": req_idx,
                    "requirement": member.get("requirement", ""),
                    "entities": [],
                },
            )

            ent = member.get("entity", {}) or {}
            ent_span = ent.get("surface_string", "") or ""
            ent_pref = ent.get("preferred_term") or ent_span
            ent_key  = _to_var(ent_pref or ent_span)

            # 找到 / 创建实体块
            ent_block = None
            for e in b["entities"]:
                if e.get("entity_canonical_form") == ent_key:
                    ent_block = e
                    break
            if ent_block is None:
                ent_block = {
                    "entity_canonical_form": ent_key,
                    "span": ent_span,
                    "preferred_term": ent_pref,
                    "fully_specified_name": ent.get("fully_specified_name", ""),
                    "conceptId": ent.get("conceptId", ""),
                    "attributes": [],
                }
                b["entities"].append(ent_block)

            # 去重：同实体内 (attr_type, canonical_value, conceptID, qualifier_span)
            seen = set()
            for it in finals:
                attr_type = it.get("attribute_type", "") or ""
                canon_val = it.get("canonical_attribute_value", "") or ""
                canon_cid = it.get("canonical_attribute_value_conceptID", "") or ""
                qspan     = it.get("qualifier_span", "") or ""
                dkey = (attr_type, canon_val, canon_cid, qspan)
                if dkey in seen:
                    continue
                seen.add(dkey)

                ent_block["attributes"].append(
                    {
                        "attribute_class_canonical_form": _to_var(attr_type),
                        "attribute_value_canonical_form": _to_var(canon_val),
                        "attribute_type": attr_type,
                        "attribute_value_original": it.get("attribute_value", ""),
                        "canonical_attribute_value": canon_val,
                        "canonical_attribute_value_conceptID": canon_cid,
                        "qualifier": it.get("qualifier", ""),
                        "qualifier_span": qspan,
                        "reason": it.get("reason", ""),
                        "COMPOSED_QUALIFIER": it.get("COMPOSED_QUALIFIER", ""),
                    }
                )

    # 稳定排序
    out = sorted(bundles.values(), key=lambda b: (b["trial_id"], b["req_index"]))
    for b in out:
        b["entities"].sort(key=lambda e: e["entity_canonical_form"])
        for e in b["entities"]:
            e["attributes"].sort(
                key=lambda a: (a["attribute_class_canonical_form"],
                               a["attribute_value_canonical_form"],
                               a["qualifier_span"])
            )
    return out


def _safe_id(s: str) -> str:
    # 文件夹名做个清洗，避免奇怪字符
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s))

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]