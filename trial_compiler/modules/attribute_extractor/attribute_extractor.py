# modules/stages/qualifier_input_builder.py
from __future__ import annotations
from typing import Any, Dict
import dspy

from smt_core.modules.attribute_extractor.stages import (
    AttributeExtractorRequirementEntityCollector,
    AttributeExtractorQualifierIdentifier
)


class AttributeExtractor(dspy.Module):
    """
    End‑to‑end attribute pipeline
        Bucketer  ➜  LLM extractor  ➜  Vector search  ➜  Canonical filter
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        engine,
        *,
        verbose: bool = False,
        mbench_path: str | None = "mbench/attr_mbench/",
    ):
        
        super().__init__()
        self.verbose = verbose
        self.mbench_path = mbench_path

        self.requirement_entity_collector = AttributeExtractorRequirementEntityCollector(verbose=True)
        self.qualifier_identifier = AttributeExtractorQualifierIdentifier(
            batch_size=5,
            verbose=verbose,
        )
    

    def forward(
        self,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:  
    
        ctx = self.requirement_entity_collector(ctx)
        ctx = self.qualifier_identifier(ctx)


        return ctx