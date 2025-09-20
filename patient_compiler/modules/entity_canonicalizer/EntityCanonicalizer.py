# modules/EntityCanonicalizer.py
from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import dspy

from .stages import (
    LLMBasedMedicalEntityRecognizer,
    VectorEmbeddingConceptSearch,
    LLMBasedMedicalEntityFilter,
    DiagnosisCanonicalizer,
)

from .report_utils import write_entity_report


class EntityCanonicalizer(dspy.Module):
    """
    End-to-end pipeline

        DictMatcher ➜ LLM-NER ➜ Vector search ➜ Filter  [+ optional Dx→SNOMED mapping]
    """

    def __init__(
        self,
        engine: Callable[[str], str] | Callable[[list[str]], list[str]],
        *,
        # dictionary matcher (kept for signature compatibility; used elsewhere upstream)
        exact_jsonl: str,
        fuzzy_threshold: float | str = 0.9,

        # vector stage
        es_url: str = "http://localhost:9200",
        vec_top_k: int = 3,
        vec_score_cut: float = 0.30,

        # validator
        validator_batch: int = 5,

        # misc
        report_dir: Optional[str] = "entity_reports",
        verbose: bool = False,

        # ===== Diagnosis canonicalization knobs =====
        ddx_enabled: bool = True,
        ddx_es_url: str = "http://localhost:9200",
        ddx_index: str = "snomed_vectors",
        ddx_model_name: str = "cambridgeltl/sapbert-from-pubmedbert-fulltext",
        ddx_snowstorm_url: Optional[str] = "http://localhost:8080",
        ddx_branch: str = "MAIN",
        ddx_top_k: int = 5,
        ddx_score_cut: float = 0.30,
        ddx_overshoot: int = 4,
        ddx_allowed_types: tuple[str, ...] = ("Clinical finding",),
        ddx_strict_only_findings: bool = True,
        ddx_min_es_score_for_verify: float = 0.0,      # <<< optional numeric floor
        ddx_log_dir: Optional[str] = None,

        # <<< NEW: 只跑诊断 canonicalization 的开关
        diagnosis_only: bool = False,
    ):
        super().__init__()
        self.verbose    = True
        self.report_dir = "mbench/entity_mbench/entity_reports"

        # ── Entity stages ────────────────────────────────────────────────
        self.ner   = LLMBasedMedicalEntityRecognizer(engine)
        self.vec   = VectorEmbeddingConceptSearch(
            es_url=es_url,
            index="snomed_vectors",
            top_k=vec_top_k,
            score_cut=vec_score_cut,
            verbose=verbose,
        )
        self.filter = LLMBasedMedicalEntityFilter(engine)
        # self.schematizer = EntitySchematizer(
        #     engine=engine,
        #     max_attempts=3,
        #     verbose=verbose,
        #     log_dir="mbench/entity_mbench/entity_schematizer_logs",
        # )

        # ── Diagnosis canonicalizer (LLM link + LLM verify) ─────────────
        self.ddx_enabled = ddx_enabled
        if ddx_enabled:
            self.ddx_canon = DiagnosisCanonicalizer(
                engine=engine,                              # <<< PASS ENGINE
                es_url=ddx_es_url,
                index=ddx_index,
                model_name=ddx_model_name,
                snowstorm_url=ddx_snowstorm_url,
                branch=ddx_branch,
                top_k=ddx_top_k,
                score_cut=ddx_score_cut,
                overshoot=ddx_overshoot,
                allowed_types=ddx_allowed_types,
                strict_only_findings=ddx_strict_only_findings,
                min_es_score_for_verify=ddx_min_es_score_for_verify,
                verbose=verbose,
                log_dir=ddx_log_dir,
            )
        else:
            self.ddx_canon = None

        # <<< NEW
        self.diagnosis_only = diagnosis_only

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]

        # <<< NEW: 只跑诊断 canonicalization 的快速路径
        if self.diagnosis_only:
            # 只跑 Dx → SNOMED canonicalization，不跑 NER / vec / filter
            if self.ddx_enabled and self.ddx_canon and context.get("diagnosis_candidates"):
                if self.verbose:
                    print("[EC] diagnosis canonicalization — LLM link + verify (findings only)")
                context = self.ddx_canon.forward(context)
            return context
        # --------------------------------------------------------------

        reqs = context.get("requirements", [])
        if not reqs:
            raise ValueError("context must contain non-empty 'requirements'")
                # Diagnosis → SNOMED mapping (runs if diagnoser populated candidates)

        for idx, curr_req in enumerate(reqs):
            print(f"Current requirement is {curr_req}")
            context["current_requirement_index"] = idx

            # if self.verbose:
            #     print(f"[EC] req {idx} — Schematize entities")
            # context = self.schematizer(context)

            if self.verbose:
                print(f"[EC] req {idx} — LLM-NER")
            context = self.ner(context)

            if self.verbose:
                print(f"[EC] req {idx} — Vector search")
            context = self.vec(context)

            if self.verbose:
                print(f"[EC] req {idx} — Validator")
            context = self.filter(context)

        # optional markdown report for entity side
        if self.report_dir:
            write_entity_report(context, self.report_dir)

        if self.ddx_enabled and self.ddx_canon and context.get("diagnosis_candidates"):
            if self.verbose:
                print("[EC] diagnosis canonicalization — LLM link + verify (findings only)")
            context = self.ddx_canon.forward(context)

        return context