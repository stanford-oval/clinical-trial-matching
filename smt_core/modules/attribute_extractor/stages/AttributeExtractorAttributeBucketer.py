# modules/stages/AttributeExtractorAttributeBucketer.py
from __future__ import annotations
from typing import Dict, List, Any, Iterable
import logging, dspy, json
import pandas as pd

# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────
def _iter_entity_fields(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield every value whose key starts with `entity_`."""
    for k, v in doc.items():
        if k.startswith("entity_"):
            yield v


def _auto_build_documents(ctx: Dict[str, Any]) -> None:
    """
    If ctx lacks 'documents' but *does* contain
        • ctx['requirements']            (list[str | dict])
        • ctx['valid_entities_by_req']   (mapping idx → {'entity_*': {...}})
    then synthesise the list that later stages expect.

    Tolerates empty inputs: if both are empty, leave 'documents' unset and let
    the Bucketer's no-op guard handle it.
    """
    if "documents" in ctx:
        return

    if not {"requirements", "valid_entities_by_req"} <= ctx.keys():
        return

    reqs      = ctx.get("requirements") or []
    ents_by_i = ctx.get("valid_entities_by_req") or {}

    # Nothing to do: keep 'documents' absent and let caller no-op.
    if not reqs and not ents_by_i:
        return

    documents: List[Dict[str, Any]] = []

    for idx, req in enumerate(reqs):
        trial_id = ctx.get("trial_id", f"trial_{idx:03d}")
        requirement_text = (
            (req.get("requirement")         if isinstance(req, dict) else None)
            or (req.get("text")             if isinstance(req, dict) else None)
            or (req.get("requirement_text") if isinstance(req, dict) else None)
            or (req.get("sentence")         if isinstance(req, dict) else None)
            or str(req)
        )

        doc: Dict[str, Any] = {
            "trial":       f"{trial_id}_req{idx:03d}",
            "requirement": requirement_text,
        }

        # Accept both int and str keys (JSON round-trips often stringify indices).
        ents = (ents_by_i.get(idx) or ents_by_i.get(str(idx)) or {})
        for j, ent in enumerate(ents.values(), start=1):
            doc[f"entity_{j}"] = {
                "surface_string":       ent.get("extracted_span", ""),
                "preferred_term":       ent.get("preferred_term", ""),
                "fully_specified_name": ent.get("fully_specified_name", ""),
                "type":                 ent.get("type", ""),
                "definition":           ent.get("definition", ""),
                # keep positional / ID metadata
                "conceptId":            ent.get("conceptId"),
                "start":                ent.get("start"),
                "end":                  ent.get("end"),
            }

        documents.append(doc)

    ctx["documents"] = documents
    logging.getLogger("AttributeBucketer").debug("Auto-built %d document(s)", len(documents))


class AttributeExtractorAttributeBucketer(dspy.Module):
    """
    Group raw *trial-level* documents into buckets that share the same
    set of **allowed attributes**.

    Empty-input semantics:
      • If there are 0 documents (no requirements & no entities), this stage
        returns with `ctx["groups"] = []` instead of raising.
    """

    def __init__(
        self,
        *,
        domain_attribute_map,
        attribute_id_to_name,
        attr_map_path,
        return_ids: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.return_ids = return_ids
        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

        # Lazy imports to avoid heavy deps at import-time
        from ..utils import (
            allowed_attributes,
            hit_ancestors,
            fetch_ancestors,
            concept_name_to_id,
        )
        self._allowed_attributes = allowed_attributes
        self._hit_ancestors      = hit_ancestors
        self._fetch_ancestors    = fetch_ancestors
        self._concept_name2id    = concept_name_to_id

        self._domain_attr_map = domain_attribute_map
        self._attr_id_to_name = attribute_id_to_name

        # Load attribute definitions so we can enrich group metadata
        df = pd.read_excel(attr_map_path, dtype=str)
        self._attr_def: dict[str, str] = {
            row["attributeFSN"]: row["attributeDefinition"]
            for _, row in df.iterrows()
        } | {
            row["attributeId"]:  row["attributeDefinition"]
            for _, row in df.iterrows()
        }

    # ------------------------------------------------------------------
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        print("in AttributeBucketer forward, before _auto_build_documents")
        _auto_build_documents(ctx)  # may leave 'documents' absent if nothing to do
        print("documents is built!")

        docs = ctx.get("documents") or []  # tolerate None/absent
        if not docs:
            # Allow “no requirements/entities” as a valid no-op.
            self.log.info("AttributeBucketer: 0 documents → returning empty groups (no-op).")
            ctx["groups"] = []
            print("After bucketer ctx['groups']: ")
            print(json.dumps(ctx["groups"], indent=4, ensure_ascii=False))
            return ctx

        domain_attr_map = self._domain_attr_map
        attr_id_to_name = self._attr_id_to_name

        buckets: Dict[frozenset[str], List[Dict[str, Any]]] = {}

        for doc in docs:
            trial       = doc["trial"]
            requirement = doc["requirement"]

            # each document can carry multiple entity_* fields
            for entity in _iter_entity_fields(doc):
                concept = entity.get("fully_specified_name", "") or entity.get("preferred_term", "")

                attrs = self._allowed_attributes(
                    concept,
                    domain_attr_map,
                    attr_id_to_name,
                    return_ids=self.return_ids,
                )

                member = {
                    "trial":       trial,
                    "requirement": requirement,
                    "entity":      entity,
                }
                buckets.setdefault(frozenset(attrs), []).append(member)

        # Materialize groups with enriched attribute definitions (stable ordering)
        groups: List[Dict[str, Any]] = []
        for attr_set, members in sorted(buckets.items(), key=lambda p: tuple(sorted(p[0]))):
            sorted_attrs = sorted(attr_set)
            groups.append(
                {
                    "allowed_attributes": [
                        {
                            "AttributeType": attr,
                            "definition": self._attr_def.get(attr, ""),
                        }
                        for attr in sorted_attrs
                    ],
                    "members": members,
                }
            )

        ctx["groups"] = groups
        print("After bucketer ctx['groups']: ")
        print(json.dumps(ctx["groups"], indent=4, ensure_ascii=False))
        return ctx
