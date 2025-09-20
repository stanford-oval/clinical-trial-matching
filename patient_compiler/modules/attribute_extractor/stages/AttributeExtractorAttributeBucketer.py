# modules/stages/attribute_bucketer.py
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
    """
    if "documents" in ctx:
        return

    if not {"requirements", "valid_entities_by_req"} <= ctx.keys():
        return

    documents: List[Dict[str, Any]] = []
    reqs      = ctx["requirements"]
    ents_by_i = ctx["valid_entities_by_req"]

    for idx, req in enumerate(reqs):
        patient_note_id = ctx.get("note_id")
        requirement_text = (
            (req.get("requirement")        if isinstance(req, dict) else None)
            or (req.get("text")            if isinstance(req, dict) else None)
            or (req.get("requirement_text")if isinstance(req, dict) else None)
            or (req.get("sentence")        if isinstance(req, dict) else None)
            or str(req)
        )

        doc: Dict[str, Any] = {
            "patient_note":       f"{patient_note_id}_req{idx:03d}",
            "requirement": requirement_text,
        }

        # --- key fix: accept both int and str keys from JSON ---
        ents = ents_by_i.get(idx) or ents_by_i.get(str(idx), {})
        for j, ent in enumerate(ents.values(), start=1):
            doc[f"entity_{j}"] = {
                "surface_string":       ent.get("extracted_span", ""),
                "preferred_term":       ent.get("preferred_term", ""),
                "fully_specified_name": ent.get("fully_specified_name", ""),
                "type":                 ent.get("type", ""),
                "definition":           ent.get("definition", ""),
                # --- keep all positional / ID metadata so it flows downstream ---
                "conceptId":            ent.get("conceptId"),
                "start":                ent.get("start"),
                "end":                  ent.get("end"),
            }
        documents.append(doc)

    ctx["documents"] = documents
    logging.getLogger("AttributeBucketer").debug(
        "Auto‑built %d document(s)", len(documents)
    )


class AttributeExtractorAttributeBucketer(dspy.Module):
    """
    Group raw *trial‑level* documents into buckets that share the same
    set of **allowed attributes**.
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

        # Lazy imports to avoid heavy deps at import‑time
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
        _auto_build_documents(ctx)                              # ensure docs exist

        print("documents is built!")

        docs = ctx.get("documents", [])
        if not docs:
            raise RuntimeError("AttributeBucketer: no documents to bucket")

        domain_attr_map = self._domain_attr_map
        attr_id_to_name = self._attr_id_to_name

        buckets: Dict[frozenset[str], List[Dict[str, Any]]] = {}

        for doc in docs:
            patient_note       = doc["patient_note"]
            requirement = doc["requirement"]

            for entity in _iter_entity_fields(doc):
                concept = entity["fully_specified_name"]

                attrs = self._allowed_attributes(
                    concept,
                    domain_attr_map,
                    attr_id_to_name,
                    return_ids=self.return_ids,
                )

                member = {
                    "patient_note":       patient_note,
                    "requirement": requirement,
                    "entity":      entity,
                }
                buckets.setdefault(frozenset(attrs), []).append(member)

        # ctx["groups"] = [
        #     {"allowed_attributes": sorted(attr_set), "members": members}
        #     for attr_set, members in sorted(
        #         buckets.items(), key=lambda p: tuple(sorted(p[0]))
        #     )
        # ]

        groups = []
        for attr_set, members in sorted(buckets.items(), key=lambda p: tuple(sorted(p[0]))):
            sorted_attrs = sorted(attr_set)
            groups.append(
                {
                    "allowed_attributes": [
                        {
                            "AttributeType": attr,
                            "definition": self._attr_def.get(attr, "")
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
