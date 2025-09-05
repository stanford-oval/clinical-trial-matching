# modules/EntityCanonicalizer.py
from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import dspy
import contextlib

from .stages import (
    LLMBasedMedicalEntityRecognizer,
    VectorEmbeddingConceptSearch,
    LLMBasedMedicalEntityFilter,
)

from .report_utils import write_entity_report


class EntityCanonicalizer(dspy.Module):
    """
    End-to-end pipeline:

        DictMatcher ➜ LLM-NER ➜ Vector search ➜ Filter

    Fine-grained profiling:
      CANON/NER, CANON/VEC/encode, CANON/VEC/search, CANON/VEC/enrich, CANON/FILTER
    """

    # --------------------------------------------------------------
    def __init__(
        self,
        engine: Callable[[str], str] | Callable[[list[str]], list[str]],
        *,
        # dictionary matcher
        exact_jsonl: str,
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
        # NEW: vector-stage performance knobs
        encode_batch: int = 32,
        es_workers: int = 8,
        query_cache_size: int = 4096,
        fetch_definitions: bool = False,
        defs_top_n: int = 1,
        # NEW: profiler wiring
        profiler: Optional[Any] = None,
        profiler_run_id: str = "run",
    ):
        super().__init__()
        # Keep your preferred defaults but allow override
        self.verbose    = bool(verbose)
        self.report_dir = report_dir or "mbench/entity_mbench/entity_reports"

        self.profiler = profiler
        self.run_id   = profiler_run_id

        # LLM NER
        self.ner = LLMBasedMedicalEntityRecognizer(engine)

        # Vector stage (optimized + profiled)
        self.vec = VectorEmbeddingConceptSearch(
            es_url=es_url,
            index="snomed_vectors",
            top_k=vec_top_k,
            score_cut=vec_score_cut,
            verbose=verbose,
            encode_batch=encode_batch,
            es_workers=es_workers,
            query_cache_size=query_cache_size,
            fetch_definitions=fetch_definitions,
            defs_top_n=defs_top_n,
            profiler=profiler,
            profiler_run_id=profiler_run_id,
        )

        # LLM-based validator/filter
        self.filter = LLMBasedMedicalEntityFilter(engine)

    # small helper for spans
    def _span(self, stage: str, ctx: Dict[str, Any], **extra):
        if not self.profiler:
            return contextlib.nullcontext()
        return self.profiler.span(
            run_id=self.run_id,
            trial_id=str(ctx.get("trial_id", "?")),
            side=str(ctx.get("inc_exc", "?")),
            stage=stage,
            cohort_id=str(ctx.get("trial_id", "?")),  # cohort-id if you suffix trial_id upstream
            **extra
        )

    # --------------------------------------------------------------
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        # ---- tolerate empty/blank requirements ---------------------------------
        reqs = context.get("requirements") or []

        def _is_blank(r):
            if isinstance(r, str):
                return not r.strip()
            if isinstance(r, dict):
                s = (r.get("requirement") or r.get("text") or "").strip()
                return not s
            return True

        effective = [r for r in reqs if not _is_blank(r)]
        if len(effective) == 0:
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

        # Run LLM-NER once over effective reqs
        if self.verbose:
            print(f"[EC] NER — batching {len(effective)} requirements")

        with self._span("CANON/NER", context):
            context["requirements"] = effective  # keep only non-blank
            context = self.ner(context)  # (LLM calls timed separately by engine if wired)

        # Vector search + filter, per requirement
        for idx, _ in enumerate(effective):
            context["current_requirement_index"] = idx

            if self.verbose:
                print(f"[EC] req {idx} — Vector search")
            # Vector stage has its own internal subspans (encode/search/enrich)
            context = self.vec(context)

            if self.verbose:
                print(f"[EC] req {idx} — Validator")
            with self._span("CANON/FILTER", context, requirement_index=idx):
                context = self.filter(context)

        if self.report_dir:
            write_entity_report(context, self.report_dir)

        return context
