# modules/DiseaseCanonicalizer.py
from __future__ import annotations
from typing import Callable, Dict, Any
import dspy

from .stages import (
    VectorEmbeddingConceptSearch,
    LLMBasedMedicalEntityFilter,
)

from .report_utils import write_entity_report


class DiseaseCanonicalizer(dspy.Module):
    """
    End-to-end pipeline

        DictMatcher ➜ LLM-NER ➜ Vector search ➜ Filter
    """

    # --------------------------------------------------------------
    def __init__(
        self,
        engine: Callable[[str], str] | Callable[[list[str]], list[str]],
        *,
        # dictionary matcher
        fuzzy_threshold: float | str = 0.9,
        # vector stage
        es_url: str = "http://localhost:9200",
        vec_top_k: int = 3,
        vec_score_cut: float = 0.30,
        # validator
        validator_batch: int = 5,
        # misc
        report_dir: str | None = "entity_reports",
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose    = True
        self.report_dir = "mbench/entity_mbench/entity_reports"

        self.vec   = VectorEmbeddingConceptSearch(
            es_url=es_url,
            index="snomed_vectors",
            top_k=vec_top_k,
            score_cut=vec_score_cut,
            verbose=verbose,
        )

        self.filter = LLMBasedMedicalEntityFilter(engine)

    # --------------------------------------------------------------
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        # ---- tolerate empty/blank requirements ---------------------------------
        diseases = context.get("target_disease") or []

        if len(diseases) == 0:
            # Seed the views downstream expects, skip heavy stages (LLM/ES/torch)
            context.setdefault("llm_surface_entities_by_req", {})
            context.setdefault("valid_entities_by_req", {})
            context.setdefault("verifier", {})
            context["verifier"].setdefault("entity_canonicalizer", {})
            context["verifier"]["entity_canonicalizer"].update({
                "skipped": True,
                "reason": "no requirements",
            })
            if self.verbose:
                print("[EC] 0 requirements → skipping canonicalization, seeding empty views")
            return context
        # ------------------------------------------------------------------------


        context = self.vec(context)
        context = self.filter(context)
        if self.report_dir:
            write_entity_report(context, self.report_dir)
        return context