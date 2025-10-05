#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# compile_trial_to_smt.py
"""
Refactored, modular version of the clinical-trial SMT pipeline **with automatic
prompt reloading from specified file paths when resuming from checkpoints**, plus
optional **requirements injection** from a JSONL file to skip rudimentary extraction.
==========================================================================
Key improvements (over previous refactor)
----------------------------------------
* On every resume (`--resume-from`), the pipeline **overwrites all prompt
  templates** in `ctx` by re-reading them from file paths you specify via:
    - `--prompt-root`  (base directory; defaults to ./prompts/clinical_trial under repo root)
    - `--prompt-map`   (JSON { key: absolute_or_relative_path }, overrides per-key)
* Can **inject inclusion/exclusion requirements** from a JSONL row and skip
  the rudimentary extraction step (and optionally the entire extractor) via:
    - `--requirements-jsonl <file>`
    - `--patient-id <id>` (if multiple rows per trial)
    - `--skip-extractor` (bypass the whole RequirementExtractor)
    - `--side both` (run inclusion & exclusion in one command)
* Fallback to the legacy search roots unchanged if no explicit file is found.

NEW in this revision
--------------------
* The RequirementContextPreprocessor is a **standalone, first-class stage**
  that always runs before extraction, and splits into **cohort subcontexts**.
* All later modules (canon → attr → program → final) process **each cohort**
  subcontext if present, otherwise they process the parent context once.

PLUS (this file): Always-on **profiling** (opt-in) that records wall-clock durations
per stage and per LLM call to JSONL (and an optional CSV summary).

NEW flags in this version
------------------------
* --skip-built-cohorts: when a trial has cohort subcontexts, only process those
  whose IR artifacts are **not** present yet (saves re-computation on re-runs).

* --preproc-source {ckpt,llm,canonical,auto}:
    - ckpt: prefer PREPROC checkpoints (same-side then opposite), else run LLM preproc
    - llm: always run LLM preproc (ignore ckpts/canonical)
    - canonical: load <canonical-subcohort-dir>/<trial_id>.json as source of truth (error if missing)
    - auto: canonical if present, else ckpt, else llm
* --canonical-subcohort-dir <dir>: directory containing canonical preproc/subcohort JSONs
"""

from __future__ import annotations

# ────────────────────── standard library ──────────────────────
import argparse
import contextlib
import datetime as dt
import json
import logging
import os
import pathlib
from pathlib import Path
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Sequence, Optional

# ───────────────────── 3ʳᵈ-party dependencies ─────────────────────
from openai import AzureOpenAI  # type: ignore  # noqa: F401 – retained for future use
import dspy  # type: ignore

# ──────────────────────── local modules ─────────────────────────
from smt_core.checkpoint_io import (
    load_canon_ckpt,
    # load_noncanon_ckpt,
    load_attr_ckpt,
    load_final_ckpt,
    load_preproc_ckpt,
    load_program_ckpt,
    save_canon_ckpt,
    # save_noncanon_ckpt,
    save_attr_ckpt,
    save_final_ckpt,
    save_preproc_ckpt,
    save_program_ckpt,
    json_sanitize,
)

# ──────────────────────── import inference engine ────────────────────
from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
from smt_core.inference_engine import AzureInferenceEngine

print(f"[INFO] Using AzureInferenceEngine for {ENGINE_VERSION} (model={MODEL_NAME})")


# ──────────────────────── import other modules ─────────────────────────
from smt_core.modules.entity_canonicalizer import EntityCanonicalizer

# from modules.NonCanonEntityIdentifier import NonCanonEntityIdentifier
from trial_compiler.modules.requirement_extractor.requirement_extractor import RequirementExtractor

# Preprocessor is now called explicitly as a first-class step
from trial_compiler.modules.requirement_extractor.stages import RequirementContextPreprocessor

# from modules.SMTMatcher.SMTMatcher import SMTMatcher
from trial_compiler.modules.smt_programmer.SMTProgrammer import SMTProgrammer
from trial_compiler.modules.attribute_extractor import AttributeExtractor
from smt_core.utils.text_utils import dict_to_readable_string, _load_patient_notes  # noqa: F401 – future use
from smt_core.helpers_entities import load_entity_annotations

from trial_compiler.modules.requirement_extractor.stages.RequirementContradictCriterionRewriter import (
    RequirementContradictCriterionRewriter,
)

# ─────────────────────── global constants ───────────────────────
dspy.settings.configure(lm_cache=None)
logging.getLogger("azure").setLevel(logging.WARNING)

# All required prompt keys → relative file paths (under a prompt root)
REQUIRED_PROMPTS: Dict[str, str] = {
    # ⬇︎ RequirementExtractor prompts
    "RequirementContextPreprocessor_prompt": "RequirementExtractor/RequirementContextPreprocessor.prompt",
    "RequirementRudimentaryExtractorInclusion_prompt": "RequirementExtractor/RequirementRudimentaryExtractorInclusion.prompt",
    "RequirementRudimentaryExtractorExclusion_prompt": "RequirementExtractor/RequirementRudimentaryExtractorExclusion.prompt",
    "RequirementPreamblerInclusion_prompt": "RequirementExtractor/RequirementPreamblerInclusion.prompt",
    "RequirementPreamblerExclusion_prompt": "RequirementExtractor/RequirementPreamblerExclusion.prompt",
    "RequirementRudimentaryExtractorVerifierInclusion_prompt": "RequirementExtractor/RequirementRudimentaryExtractorVerifierInclusion.prompt",
    "RequirementRudimentaryExtractorVerifierExclusion_prompt": "RequirementExtractor/RequirementRudimentaryExtractorVerifierExclusion.prompt",
    "RequirementEntitySpanExpander_prompt": "RequirementExtractor/RequirementEntitySpanExpander.prompt",
    "RequirementEntitySurfaceExpanderVerifier_prompt": "RequirementExtractor/RequirementEntitySurfaceExpanderVerifier.prompt",
    "RequirementLogicalPrecisionRewriterInclusion_prompt": "RequirementExtractor/RequirementLogicalPrecisionRewriterInclusion.prompt",
    "RequirementLogicalPrecisionRewriterExclusion_prompt": "RequirementExtractor/RequirementLogicalPrecisionRewriterExclusion.prompt",
    "RequirementLogicalPrecisionRewriterVerifierInclusion_prompt": "RequirementExtractor/RequirementLogicalPrecisionRewriterVerifierInclusion.prompt",
    "RequirementLogicalPrecisionRewriterVerifierExclusion_prompt": "RequirementExtractor/RequirementLogicalPrecisionRewriterVerifierExclusion.prompt",
    "RequirementDecomposerInclusion_prompt": "RequirementExtractor/RequirementDecomposerInclusion.prompt",
    "RequirementDecomposerExclusion_prompt": "RequirementExtractor/RequirementDecomposerExclusion.prompt",
    "RequirementDecomposerVerifierInclusion_prompt": "RequirementExtractor/RequirementDecomposerVerifierInclusion.prompt",
    "RequirementDecomposerVerifierExclusion_prompt": "RequirementExtractor/RequirementDecomposerVerifierExclusion.prompt",
    # "RequirementExplicitPositivityClassifier_prompt": "RequirementExtractor/RequirementExplicitPositivityClassfier.prompt"
    "RequirementHardSoftClassifierInclusion_prompt": "RequirementExtractor/RequirementHardSoftClassifierInclusion.prompt",
    "RequirementHardSoftClassifierExclusion_prompt": "RequirementExtractor/RequirementHardSoftClassifierExclusion.prompt",
    # ⬇︎ SMT-programmer prompts
    "SMTProgrammerFreeEntityExtractor_prompt": "SMTProgrammer/SMTPreprocessor/SMTProgrammerFreeEntityExtractor.prompt",
    "SMTProgrammerFreeEntityQualifierIdentifier_prompt": "SMTProgrammer/SMTPreprocessor/SMTProgrammerFreeEntityQualifierIdentifier.prompt",
    "SMTProgrammerFreeEntityQualifierIdentifierVerifier_prompt": "SMTProgrammer/SMTPreprocessor/SMTProgrammerFreeEntityQualifierIdentifierVerifier.prompt",
    "SMTProgrammerTopLevelEntityFilter_prompt": "SMTProgrammer/SMTPreprocessor/SMTProgrammerTopLevelEntityFilter.prompt",
    "SMTIncrementalReusableVariableIdentifier_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalReusableVariableIdentifier.prompt",
    "SMTIncrementalDemographicsVariableNamer_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalDemographicsVariableNamer.prompt",
    "SMTIncrementalCanonicalVariableNamer_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalCanonicalVariableNamer.prompt",
    "SMTIncrementalFreeVariableNamer_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalFreeVariableNamer.prompt",
    "SMTIncrementalTranslatorInclusion_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalTranslatorInclusion.prompt",
    "SMTIncrementalTranslatorExclusion_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalTranslatorExclusion.prompt",
    "SMTIncrementalSolverBasedNaiveRefiner_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalSolverBasedNaiveRefiner.prompt",
    "SMTIncrementalVerifierInclusion_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalVerifierInclusion.prompt",
    "SMTIncrementalVerifierExclusion_prompt": "SMTProgrammer/SMTIncrementalProgrammer/SMTIncrementalVerifierExclusion.prompt",
    # ⬇︎ Matcher
    "SMTVariableValueMiner_prompt": "SMTMatcher/SMTVariableValueMiner.prompt",
    # ⬇︎ Entity-canonicalizer prompts
    "LLMBasedMedicalEntityRecognizer_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityRecognizer.prompt",
    "LLMBasedMedicalEntityFilterLinker_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityFilterLinker.prompt",
    "LLMBasedMedicalEntityFilterVerifier_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityFilterVerifier.prompt",
    "LLMBasedMedicalEntityFilterArbiter_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityFilterArbiter.prompt",
    # ⬇︎ Attribute-canonicalizer prompts
    "AttributeExtractorQualifierIdentifier_prompt": "AttributeExtractor/AttributeExtractorQualifierIdentifier.prompt",
    "AttributeExtractorQualifierIdentifierVerifier_prompt": "AttributeExtractor/AttributeExtractorQualifierIdentifierVerifier.prompt",
    "AttributeExtractorAttributeTranslator_prompt": "AttributeExtractor/AttributeExtractorAttributeTranslator.prompt",
    "AttributeExtractorFreeAttributeTranslator_prompt": "AttributeExtractor/AttributeExtractorFreeAttributeTranslator.prompt",
    "AttributeExtractorCanonicalAttributeValueFilterCanon_prompt": "AttributeExtractor/AttributeExtractorCanonicalAttributeValueFilterCanon.prompt",
    "AttributeExtractorCanonicalAttributeValueFilterFree_prompt": "AttributeExtractor/AttributeExtractorCanonicalAttributeValueFilterFree.prompt",
    "AttributeExtractorCanonicalAttributeValueVerifier_prompt": "AttributeExtractor/AttributeExtractorCanonicalAttributeValueVerifier.prompt",
    "AttributeExtractorNonCanonQualifierIdentifier_prompt": "AttributeExtractor/AttributeExtractorNonCanonQualifierIdentifier.prompt",
    "AttributeExtractorNonCanonFreeAttributeTranslator_prompt": "AttributeExtractor/AttributeExtractorNonCanonFreeAttributeTranslator.prompt",
}

# ------------------------------------------------------------------------
#  Lightweight profiling
# ------------------------------------------------------------------------


from smt_core.buildroot import build_root as _shared_build_root


@dataclass
class ProfileEvent:
    run_id: str
    trial_id: str
    side: str
    stage: str
    cohort_id: Optional[str] = None
    action: str = "stage"  # "stage" | "llm_call" | "misc"
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Append-only JSONL profiler."""

    def __init__(self, jsonl_path: pathlib.Path):
        self.jsonl_path = jsonl_path
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def span(
        self,
        *,
        run_id: str,
        trial_id: str,
        side: str,
        stage: str,
        cohort_id: str | None = None,
        **extra,
    ):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            evt = ProfileEvent(
                run_id=run_id,
                trial_id=trial_id,
                side=side,
                stage=stage,
                cohort_id=cohort_id,
                started_at=t0,
                ended_at=t1,
                duration_s=t1 - t0,
                extra=extra,
            )
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(evt.__dict__) + "\n")

    def log_llm_call(
        self,
        *,
        run_id: str,
        trial_id: str,
        side: str,
        stage: str,
        cohort_id: str | None,
        model: str,
        duration_s: float,
        **extra,
    ):
        evt = ProfileEvent(
            run_id=run_id,
            trial_id=trial_id,
            side=side,
            stage=stage,
            cohort_id=cohort_id,
            started_at=time.time() - duration_s,
            ended_at=time.time(),
            duration_s=duration_s,
            action="llm_call",
            extra={"model": model, **extra},
        )
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt.__dict__) + "\n")


# ------------------------------------------------------------------------
#  Config & utility layer
# ------------------------------------------------------------------------


@dataclass
class Config:
    """Holds every tunable path/flag so we avoid magic constants."""

    # root dirs (may come from env vars)
    repo_root: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
    data_root: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(os.getenv("TRIAL_DATA", "../dataset/clinical_trial"))
    )

    # checkpoint / log dirs
    ckpt_dir: pathlib.Path = pathlib.Path("checkpoints")
    log_dir: pathlib.Path = pathlib.Path("run_logs")
    status_file: pathlib.Path = pathlib.Path("run_logs/status.jsonl")

    # profiling
    profile_jsonl: pathlib.Path = pathlib.Path("run_logs/profile.jsonl")
    enable_profile: bool = True

    # misc
    max_retries: int = int(os.getenv("MAX_RETRIES", 3))
    consent_sentence: str = (
        "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
    )

    # azure
    azure_endpoint: str = os.getenv("OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name = MODEL_NAME

    build_root: pathlib.Path = field(default_factory=_shared_build_root)  # $VERDICT_BUILD
    ir_dir: pathlib.Path = field(init=False)
    symtab_dir: pathlib.Path = field(init=False)
    canon_dir: pathlib.Path = field(init=False)
    linkmap_dir: pathlib.Path = field(init=False)
    requirements_dir: pathlib.Path = field(init=False)

    # entity annotations
    entity_jsonl: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(
            os.getenv(
                "ENTITY_JSONL",
                "../dataset/clinical_trial/criterion_level/ner_labels/trial_criteria_train_yufei_labelled_128.jsonl",
            )
        )
    )
    mbench_root: pathlib.Path = pathlib.Path("mbench/smt_mbench")

    # ── prompt source controls ───────────────────────────────────────────
    prompt_root: pathlib.Path = pathlib.Path(__file__).resolve().parent / "prompts" / "clinical_trial"
    prompt_map_json: pathlib.Path | None = None

    # ── NEW: preproc source controls ─────────────────────────────────────
    # ckpt | llm | canonical | auto
    preproc_source: str = "ckpt"
    canonical_subcohort_dir: pathlib.Path = pathlib.Path("../canonical_subcohort_results")

    # helper dirs -----------------------------------------------------------

    def free_entity_extractor_dir(self) -> pathlib.Path:
        path = self.mbench_root / "free_entity_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def free_entity_qualifier_identifier_dir(self) -> pathlib.Path:
        """Folder for SMT-Programmer prompt / raw / fragment logs."""
        path = self.mbench_root / "free_qualifier_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def top_level_entity_filter_dir(self) -> pathlib.Path:
        path = self.mbench_root / "entity_filter_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def entity_enricher_dir(self) -> pathlib.Path:
        path = self.mbench_root / "entity_enricher_mappings"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def namer_log_dir(self) -> pathlib.Path:
        path = self.mbench_root / "namer_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def translator_log_dir(self) -> pathlib.Path:
        """Folder for SMT-Programmer prompt / raw / fragment logs."""
        path = self.mbench_root / "translator_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validator_log_dir(self) -> pathlib.Path:
        path = self.mbench_root / "validator_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def verifier_log_dir(self) -> pathlib.Path:
        path = self.mbench_root / "verifier_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def report_dir(self) -> pathlib.Path:
        path = self.mbench_root / "integrated_report_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ckpt_path(self, stage: str, trial_id: str, side: str) -> pathlib.Path:
        """Generate <ckpt_dir>/<stage>/<trial_id>_<side>_<stage>.chkpt.json"""
        root = self.ckpt_dir / stage
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{trial_id}_{side}_{stage}.chkpt.json"

    def prompt_roots(self) -> List[pathlib.Path]:
        """Legacy fallback search roots (used if pathmap doesn't specify a file)."""
        base = self.prompt_root
        return [base, base / "inc", base / "exc"]

    def prompt_sources(self) -> Dict[str, pathlib.Path]:
        """
        Build a concrete mapping of prompt keys → file paths.
        Priority: explicit JSON map > prompt_root + REQUIRED_PROMPTS.
        Relative paths in JSON map are resolved under prompt_root.
        """
        mapping: Dict[str, pathlib.Path] = {}

        # 1) Start from default (root + relative file path)
        for k, rel in REQUIRED_PROMPTS.items():
            mapping[k] = (self.prompt_root / rel).resolve()

        # 2) Override by JSON map if provided
        if self.prompt_map_json:
            raw = json.loads(pathlib.Path(self.prompt_map_json).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("prompt_map JSON must be an object of {key: path}.")
            for k, v in raw.items():
                p = pathlib.Path(v)
                if not p.is_absolute():
                    p = (self.prompt_root / p)
                mapping[k] = p.resolve()
        return mapping

    def __post_init__(self):
        # Normalize prompt_root relative to repo_root if not absolute
        if not self.prompt_root.is_absolute():
            self.prompt_root = (self.repo_root / self.prompt_root).resolve()

        # Normalize canonical_subcohort_dir relative to repo_root if not absolute
        if not self.canonical_subcohort_dir.is_absolute():
            self.canonical_subcohort_dir = (self.repo_root / self.canonical_subcohort_dir).resolve()

        # derive the three siblings once
        self.ir_dir = (self.build_root / "ir")
        self.symtab_dir = (self.build_root / "symtab")
        self.canon_dir = (self.build_root / "canon")
        self.linkmap_dir = (self.build_root / "linkmap")
        self.requirements_dir = (self.build_root / "requirements")
        for d in (self.ir_dir, self.symtab_dir, self.linkmap_dir, self.requirements_dir, self.canon_dir):
            d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------
#  Stream duplicator util
# ------------------------------------------------------------------------


class Tee(contextlib.AbstractContextManager):
    """Redirect stdout+stderr to a per-run log while keeping console output."""

    def __init__(self, log_path: pathlib.Path):
        self.log_path = log_path
        self._fh = log_path.open("w", encoding="utf-8")
        self._streams = (sys.stdout, self._fh)

    def __enter__(self):
        tee = _StreamFork(*self._streams)
        self._ctx = contextlib.ExitStack()
        self._ctx.enter_context(contextlib.redirect_stdout(tee))
        self._ctx.enter_context(contextlib.redirect_stderr(tee))
        print(f"✓ Log → {self.log_path}")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


class _StreamFork:
    """Simple file-like duplicator used by `Tee`."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ------------------------------------------------------------------------
#  Pipeline stage enumeration
# ------------------------------------------------------------------------


class Stage(Enum):
    PREPROC = auto()
    CANON = auto()
    # NONCANON = auto()
    ATTR = auto()
    PROGRAM = auto()
    FINAL = auto()

    @classmethod
    def from_str(cls, s: str | None) -> "Stage | None":
        if s is None:
            return None
        return cls[s.upper()]


# ------------------------------------------------------------------------
#  Small helpers for cohort-skipping
# ------------------------------------------------------------------------


def _build_artifacts_exist(build_dir: Path, eff_tid: str) -> bool:
    inc = build_dir / f"{eff_tid}_inclusion_program.smt2"
    exc = build_dir / f"{eff_tid}_exclusion_program.smt2"
    return inc.exists() and exc.exists()


def _list_missing_effective_ids(build_dir: Path, effective_ids: List[str]) -> List[str]:
    return [tid for tid in effective_ids if not _build_artifacts_exist(build_dir, tid)]


# ------------------------------------------------------------------------
#  Core pipeline orchestrator
# ------------------------------------------------------------------------


class TrialPipeline:
    """End-to-end driver with checkpointing, resumability, and profiling."""

    def __init__(self, cfg: Config, profiler: Optional[Profiler] = None, run_id: str = "run"):
        self.cfg = cfg
        self.profiler = profiler
        self.run_id = run_id
        self.entity_table = load_entity_annotations(cfg.entity_jsonl)
        logging.info("Entity annotations loaded: %s criteria", len(self.entity_table))

        self.engine = AzureInferenceEngine(
            endpoint=cfg.azure_endpoint,
            api_key_env_var="OPENAI_API_KEY",
            model_name=cfg.model_name,
            profiler=profiler,  # pass-through for LLM call timing
            profiler_run_id=run_id,
        )

    def _load_existing_preproc_ckpt(self, tid: str, side: str) -> Optional[dict]:
        """
        Prefer same-side PREPROC ckpt; fall back to the opposite side.
        Return loaded ctx or None if neither exists / load fails.
        """
        other = "exclusion" if side == "inclusion" else "inclusion"
        candidates = [
            self.cfg.ckpt_path(Stage.PREPROC.name.lower(), tid, side),
            self.cfg.ckpt_path(Stage.PREPROC.name.lower(), tid, other),
        ]
        for p in candidates:
            if p.exists():
                try:
                    return load_preproc_ckpt(self.engine, p)
                except Exception as e:
                    logging.warning("Failed loading PREPROC ckpt %s: %s", p, e)
        return None

    def _load_canonical_subcohort_preproc(self, tid: str) -> Optional[dict]:
        """
        Load canonical preprocessor output from:
            <cfg.canonical_subcohort_dir>/<tid>.json

        Supports two schemas:
          A) Unified schema:
             {"trial_id":..., "extracted": {"effective_trial_ids":..., "parent_trial_id":...,
                                            "preprocessor_normalized": {...}}}
          B) Raw ctx-like dump with top-level "preprocessor_normalized".

        Returns a dict shaped like a PREPROC ctx fragment with keys:
          shared_context, enrollment_cohorts, preprocessor_normalized,
          effective_trial_ids, parent_trial_id, preprocessor_raw_output
        """
        p = (self.cfg.canonical_subcohort_dir / f"{tid}.json").resolve()
        if not p.exists():
            return None

        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Canonical subcohort file {p} is not a JSON object.")

        extracted = obj.get("extracted") if isinstance(obj.get("extracted"), dict) else None

        norm = None
        eff_ids = None
        parent_id = None

        if extracted:
            norm = extracted.get("preprocessor_normalized")
            eff_ids = extracted.get("effective_trial_ids")
            parent_id = extracted.get("parent_trial_id")

        if norm is None:
            norm = obj.get("preprocessor_normalized")
        if eff_ids is None:
            eff_ids = obj.get("effective_trial_ids")
        if parent_id is None:
            parent_id = obj.get("parent_trial_id") or obj.get("trial_id") or tid

        if not isinstance(norm, dict):
            raise ValueError(f"Canonical subcohort file {p} missing dict preprocessor_normalized.")

        shared = norm.get("shared_context") or obj.get("shared_context") or ""
        cohorts = norm.get("enrollment_cohorts") or obj.get("enrollment_cohorts") or []
        if not isinstance(cohorts, list):
            raise ValueError(f"Canonical subcohort file {p} has non-list enrollment_cohorts.")

        # If effective_trial_ids absent, derive from cohorts
        if not eff_ids:
            eff_ids = [
                c.get("trial_id_effective")
                for c in cohorts
                if isinstance(c, dict) and c.get("trial_id_effective")
            ]

        return {
            "trial_id": tid,
            "shared_context": shared,
            "enrollment_cohorts": cohorts,
            "preprocessor_normalized": {"shared_context": shared, "enrollment_cohorts": cohorts},
            "effective_trial_ids": eff_ids,
            "parent_trial_id": parent_id,
            "preprocessor_raw_output": json.dumps(obj.get("sources_used", []), ensure_ascii=False),
        }

    def _retarget_preproc_ctx(self, prev_ctx: dict, base_ctx: dict, side: str) -> dict:
        """
        Overwrite ALL preprocessor-owned fields on `base_ctx` using `prev_ctx`,
        but swap cohort requirement_texts to match `side`, and rebuild subcontexts.
        """
        # 1) pull normalized pieces from previous ctx
        shared = (
            prev_ctx.get("shared_context")
            or (prev_ctx.get("preprocessor_normalized") or {}).get("shared_context")
            or base_ctx.get("contextual_text", "")
        )
        cohorts_prev = (
            prev_ctx.get("enrollment_cohorts")
            or (prev_ctx.get("preprocessor_normalized") or {}).get("enrollment_cohorts")
            or []
        )

        # 2) rebuild cohorts with correct side’s requirement_text, preserve trial_id_effective
        new_cohorts = []
        for i, c in enumerate(cohorts_prev):
            inc = (c.get("inclusion_criteria") or "").strip()
            exc = (c.get("exclusion_criteria") or "").strip()
            ctx_local = (c.get("context") or "").strip()
            req_txt = inc if side == "inclusion" else exc
            if not req_txt:
                req_txt = base_ctx.get("requirement_text", "")
            merged_ctx = (shared + ("\n\n" + ctx_local if ctx_local else "")) or base_ctx.get(
                "contextual_text", ""
            )
            new_cohorts.append(
                {
                    "id": c.get("id") or f"C{i+1}",
                    "label": c.get("label") or f"Cohort {i+1}",
                    "inclusion_criteria": inc,
                    "exclusion_criteria": exc,
                    "context": ctx_local,
                    "requirement_text": req_txt,
                    "contextual_text": merged_ctx,
                    "trial_id_effective": c.get("trial_id_effective") or base_ctx["trial_id"],
                }
            )

        # 3) rebuild subcohort contexts for this side
        subctxs = []
        total = len(new_cohorts) or 1
        for i, co in enumerate(
            new_cohorts
            or [
                {
                    "id": "default",
                    "label": "Overall",
                    "requirement_text": base_ctx.get("requirement_text", ""),
                    "contextual_text": base_ctx.get("contextual_text", ""),
                    "trial_id_effective": base_ctx["trial_id"],
                }
            ]
        ):
            sc = deepcopy(base_ctx)
            eff_tid = co["trial_id_effective"]
            sc["requirement_text"] = co["requirement_text"]
            sc["contextual_text"] = co["contextual_text"]
            sc["cohort_id"] = co["id"]
            sc["cohort_label"] = co["label"]
            sc["substudy_id"] = co["id"]
            sc["substudy_label"] = co["label"]
            sc["__preprocessed__"] = True
            sc["trial_id_parent"] = base_ctx["trial_id"]
            sc["trial_id"] = eff_tid
            sc["cohort_index"] = i
            sc["cohort_count"] = total
            sc["cohort_context_raw"] = co.get("context", "")
            sc["shared_context"] = shared
            sc["substudy_index"] = i
            sc["substudy_count"] = total
            sc["substudy_context_raw"] = co.get("context", "")
            sc["inc_exc"] = side
            # wipe side-coupled payloads
            for k in ("requirements", "requirements_by_substudy", "verification_by_substudy", "requirement_entities"):
                sc.pop(k, None)
            subctxs.append(sc)

        # 4) overwrite ALL preprocessor-owned fields on parent
        parent = base_ctx
        parent["shared_context"] = shared
        parent["enrollment_cohorts"] = new_cohorts
        parent["has_cohorts"] = len(new_cohorts) > 1
        parent["num_cohorts"] = len(new_cohorts)
        parent["__cohort_contexts__"] = subctxs
        parent["effective_trial_ids"] = [c["trial_id_effective"] for c in new_cohorts]
        parent["parent_trial_id"] = base_ctx["trial_id"]
        parent["preprocessor_raw_output"] = prev_ctx.get("preprocessor_raw_output", "")
        parent["preprocessor_normalized"] = {"shared_context": shared, "enrollment_cohorts": new_cohorts}
        parent["__preprocessed__"] = True
        parent["inc_exc"] = side
        for k in ("requirements", "requirements_by_substudy", "verification_by_substudy", "requirement_entities"):
            parent.pop(k, None)

        # 只有当复用的是“同侧” ckpt，才标记为 reused；对侧→当前侧的 retarget 需要落一个新的侧向 ckpt
        parent["__preproc_reused__"] = (prev_ctx.get("inc_exc") == side)

        return parent

    # ───────────────────────── stages ──────────────────────────
    def preprocess(self, trial: dict, side: str, extractor_opts: dict | None = None) -> dict:
        extractor_opts = extractor_opts or {}
        tid = trial.get("_id") or trial.get("trial_id")
        used_canonical = False

        with (
            self.profiler.span(run_id=self.run_id, trial_id=tid, side=side, stage="PREPROC")
            if self.profiler
            else contextlib.nullcontext()
        ):
            # Fresh base context (always) + fresh prompts
            ctx = self._construct_context(trial, side)
            ctx = ensure_prompt_templates(
                ctx,
                force_reload=True,
                pathmap=self.cfg.prompt_sources(),
                roots=self.cfg.prompt_roots(),
            )

            # 1) Canonical subcohort results as source of truth
            if self.cfg.preproc_source in ("auto", "canonical"):
                canon = self._load_canonical_subcohort_preproc(tid)
                if canon is not None:
                    canon = dict(canon)
                    canon["inc_exc"] = side  # so _retarget marks reused=True
                    ctx = self._retarget_preproc_ctx(canon, ctx, side)
                    ctx["__preproc_reused__"] = True
                    ctx["__preproc_source__"] = "canonical_subcohort_results"
                    used_canonical = True
                elif self.cfg.preproc_source == "canonical":
                    raise FileNotFoundError(
                        f"--preproc-source canonical but no file found: {self.cfg.canonical_subcohort_dir}/{tid}.json"
                    )

            # 2) Otherwise, fall back to ckpt (unless forcing llm)
            if not used_canonical:
                prev = None
                if self.cfg.preproc_source != "llm":
                    prev = self._load_existing_preproc_ckpt(tid, side)
                if prev is not None:
                    ctx = self._retarget_preproc_ctx(prev, ctx, side)
                    ctx["__preproc_source__"] = "preproc_ckpt"
                else:
                    # 3) Last resort: run the preprocessor LLM
                    pre = RequirementContextPreprocessor(self.engine, log_dir="mbench/preproc_logs")
                    ctx = pre.forward(ctx, use_full_context=True)
                    ctx["__preproc_source__"] = "llm_preproc"
                    # Do NOT set __preproc_reused__ here (we want run() to save this initial ckpt)

            # ── Optional: only write snapshot when NOT using canonical as truth ──
            if not used_canonical:
                try:
                    out_dir = pathlib.Path("../subcohort_results")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{tid}.json"
                    with out_path.open("w", encoding="utf-8") as fh:
                        json.dump(json_sanitize(ctx), fh, ensure_ascii=False, indent=2)
                    logging.info("Wrote subcohort snapshot → %s", out_path)
                except Exception as e:
                    logging.warning("Could not write subcohort snapshot: %s", e)

            # Attach entities on parent + each cohort subcontext
            ctx = self._attach_entities_for_context(ctx, tid, side)
            for sc in ctx.get("__cohort_contexts__", []):
                stid = sc.get("trial_id", sc.get("trial_id_effective", tid))
                self._attach_entities_for_context(sc, stid, side)

            return ctx

    def extract_requirements(self, ctx: dict, side: str, extractor_opts: dict | None = None) -> dict:
        """
        Run RequirementExtractor over cohort subcontexts if present; otherwise over parent.
        Honors `pre_extracted` and `__skip_entire_extractor__` in extractor_opts.
        """
        extractor_opts = extractor_opts or {}
        run_extractor = not extractor_opts.get("__skip_entire_extractor__", False)

        with (
            self.profiler.span(run_id=self.run_id, trial_id=ctx.get("trial_id"), side=side, stage="EXTRACT")
            if self.profiler
            else contextlib.nullcontext()
        ):
            if not run_extractor:
                return ctx

            re_extractor = RequirementExtractor(engine=self.engine, list_type=side, **extractor_opts)

            # Prefer cohort contexts if present
            subctxs = ctx.get("__cohort_contexts__") or ctx.get("__substudy_contexts__", [])
            if subctxs:
                all_requirements = []
                reqs_by_tid = {}
                verif_by_tid = {}

                for sc in subctxs:
                    sc_out = re_extractor.forward(sc)
                    reqs = sc_out.get("requirements", []) or []
                    all_requirements.extend(reqs)
                    reqs_by_tid[sc_out["trial_id"]] = reqs
                    if "verification" in sc_out:
                        verif_by_tid[sc_out["trial_id"]] = sc_out["verification"]

                ctx["requirements"] = all_requirements
                ctx["requirements_by_substudy"] = reqs_by_tid  # legacy key remains
                if verif_by_tid:
                    ctx["verification_by_substudy"] = verif_by_tid
                return ctx

            # No cohorts → run once on parent
            ctx = re_extractor.forward(ctx)
            return ctx

    def _extract_until_precision(self, ctx: dict, side: str, extractor_opts: dict | None = None) -> dict:
        """把单侧 extractor 跑到 precision 结束，并把子上下文的变更写回。"""
        opts = dict(extractor_opts or {})
        opts["stop_after"] = "precision"
        rex = RequirementExtractor(engine=self.engine, list_type=side, **opts)

        subctxs = ctx.get("__cohort_contexts__") or ctx.get("__substudy_contexts__", [])
        if subctxs:
            for i, sc in enumerate(subctxs):

                def _subctx_key(sc, idx):
                    return (
                        sc.get("trial_id")
                        or sc.get("trial_id_effective")
                        or sc.get("substudy_id")
                        or sc.get("id")
                        or f"IDX{idx}"
                    )

                sc_out = rex.forward(sc)
                if isinstance(sc_out, dict):
                    # 只写回“加工产物”，不覆盖身份字段
                    for k, v in sc_out.items():
                        if k in {
                            "trial_id",
                            "trial_id_effective",
                            "cohort_id",
                            "substudy_id",
                            "cohort_label",
                            "substudy_label",
                            "inc_exc",
                        }:
                            continue
                        sc[k] = v
            # 也可以像原来一样聚合父级 requirements，这里可选
            ctx["requirements"] = [r for sc in subctxs for r in (sc.get("requirements") or [])]
            ctx["requirements_by_substudy"] = {
                _subctx_key(sc, i): sc.get("requirements", []) for i, sc in enumerate(subctxs)
            }
        else:
            ctx = rex.forward(ctx)
        return ctx

    def _extract_after_precision(self, ctx: dict, side: str, extractor_opts: dict | None = None) -> dict:
        """从 precision 之后继续跑 decompose + hard/soft。"""
        opts = dict(extractor_opts or {})
        opts["resume_from"] = "precision"
        # 继续到结尾（classifier）
        opts["stop_after"] = "classifier"
        rex = RequirementExtractor(engine=self.engine, list_type=side, **opts)

        subctxs = ctx.get("__cohort_contexts__") or ctx.get("__substudy_contexts__", [])
        if subctxs:
            for i, sc in enumerate(subctxs):

                def _subctx_key(sc, idx):
                    return (
                        sc.get("trial_id")
                        or sc.get("trial_id_effective")
                        or sc.get("substudy_id")
                        or sc.get("id")
                        or f"IDX{idx}"
                    )

                sc_out = rex.forward(sc)
                if isinstance(sc_out, dict):
                    # 只写回“加工产物”，不覆盖身份字段
                    for k, v in sc_out.items():
                        if k in {
                            "trial_id",
                            "trial_id_effective",
                            "cohort_id",
                            "substudy_id",
                            "cohort_label",
                            "substudy_label",
                            "inc_exc",
                        }:
                            continue
                        sc[k] = v
            ctx["requirements"] = [r for sc in subctxs for r in (sc.get("requirements") or [])]
            ctx["requirements_by_substudy"] = {
                _subctx_key(sc, i): sc.get("requirements", []) for i, sc in enumerate(subctxs)
            }
        else:
            ctx = rex.forward(ctx)
        return ctx

    def run_both_with_contradiction_barrier(
        self,
        trial_inc: dict,
        trial_exc: dict,
        *,
        resume_from: "Stage | None" = None,
        stop_after: "Stage | None" = None,
        extractor_opts_inc: dict | None = None,
        extractor_opts_exc: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        同一个 trial 的 inclusion/exclusion：
        PREPROC → (两侧直到 precision) → 跨侧矛盾修复 → (两侧从 precision 继续)
        → 若需要，再跑 CANON/ATTR/PROGRAM（与原逻辑一致）
        """
        # 预处理（各自侧、可复用 ckpt / canonical）
        ctx_inc = self.preprocess(trial_inc, "inclusion", extractor_opts_inc)
        if not ctx_inc.get("__preproc_reused__", False):
            save_preproc_ckpt(
                ctx_inc, self.cfg.ckpt_path(Stage.PREPROC.name.lower(), ctx_inc["trial_id"], "inclusion")
            )

        ctx_exc = self.preprocess(trial_exc, "exclusion", extractor_opts_exc)
        if not ctx_exc.get("__preproc_reused__", False):
            save_preproc_ckpt(
                ctx_exc, self.cfg.ckpt_path(Stage.PREPROC.name.lower(), ctx_exc["trial_id"], "exclusion")
            )

        # 两侧直到 precision
        ctx_inc = self._extract_until_precision(ctx_inc, "inclusion", extractor_opts_inc)
        ctx_exc = self._extract_until_precision(ctx_exc, "exclusion", extractor_opts_exc)

        # 跨侧矛盾修复（per-cohort）
        rewriter = RequirementContradictCriterionRewriter(self.engine)
        ctx_inc, ctx_exc = rewriter.forward_pair(ctx_inc, ctx_exc)

        # 从 precision 继续（decompose + classifier）
        ctx_inc = self._extract_after_precision(ctx_inc, "inclusion", extractor_opts_inc)
        ctx_exc = self._extract_after_precision(ctx_exc, "exclusion", extractor_opts_exc)

        save_preproc_ckpt(
            ctx_inc, self.cfg.ckpt_path(Stage.PREPROC.name.lower(), ctx_inc["trial_id"], "inclusion")
        )
        save_preproc_ckpt(
            ctx_exc, self.cfg.ckpt_path(Stage.PREPROC.name.lower(), ctx_exc["trial_id"], "exclusion")
        )

        # 如果还要继续跑后续阶段（与原来 run() 一致）
        if stop_after and stop_after.value <= Stage.PREPROC.value:
            return ctx_inc, ctx_exc

        # ── inclusion 后续（如存在 cohort：逐 cohort 跑 CANON→ATTR→PROGRAM）
        subctxs = ctx_inc.get("__cohort_contexts__")
        if subctxs:
            contexts_by_cohort: Dict[str, dict] = {}
            keys_by: Dict[str, List[str]] = {}
            for sc in subctxs:
                stid = sc.get("trial_id", ctx_inc.get("trial_id"))
                sc = ensure_prompt_templates(
                    sc,
                    force_reload=True,
                    pathmap=self.cfg.prompt_sources(),
                    roots=self.cfg.prompt_roots(),
                )
                sc = self._attach_entities_for_context(sc, stid, "inclusion")
                sc_out = self._run_stages_for_single_context(
                    sc,
                    side="inclusion",
                    resume_from=Stage.PREPROC,
                    stop_after=stop_after,
                )
                contexts_by_cohort[stid] = sc_out
                keys_by[stid] = list(sc_out.keys())
            ctx_inc["contexts_by_cohort"] = contexts_by_cohort
            ctx_inc["final_keys_by_cohort"] = keys_by

        # ── exclusion 后续
        subctxs = ctx_exc.get("__cohort_contexts__")
        if subctxs:
            contexts_by_cohort = {}
            keys_by = {}
            for sc in subctxs:
                stid = sc.get("trial_id", ctx_exc.get("trial_id"))
                sc = ensure_prompt_templates(
                    sc,
                    force_reload=True,
                    pathmap=self.cfg.prompt_sources(),
                    roots=self.cfg.prompt_roots(),
                )
                sc = self._attach_entities_for_context(sc, stid, "exclusion")
                sc_out = self._run_stages_for_single_context(
                    sc,
                    side="exclusion",
                    resume_from=Stage.PREPROC,
                    stop_after=stop_after,
                )
                contexts_by_cohort[stid] = sc_out
                keys_by[stid] = list(sc_out.keys())
            ctx_exc["contexts_by_cohort"] = contexts_by_cohort
            ctx_exc["final_keys_by_cohort"] = keys_by

        return ctx_inc, ctx_exc

    def canonicalise(self, ctx: dict) -> dict:
        with (
            self.profiler.span(
                run_id=self.run_id, trial_id=ctx.get("trial_id"), side=ctx.get("inc_exc", "?"), stage="CANON"
            )
            if self.profiler
            else contextlib.nullcontext()
        ):
            canon = EntityCanonicalizer(
                self.engine,
                report_dir="entity_reports",
                exact_jsonl="src/modules/EntityCanonicalizer/setup/data/exact_terms.jsonl",
                fuzzy_threshold=0.86,
                verbose=False,
            )
            return canon(ctx)

    def extract_attributes(self, ctx: dict) -> dict:
        with (
            self.profiler.span(
                run_id=self.run_id, trial_id=ctx.get("trial_id"), side=ctx.get("inc_exc", "?"), stage="ATTR"
            )
            if self.profiler
            else contextlib.nullcontext()
        ):
            attr = AttributeExtractor(
                engine=self.engine,
                verbose=False,
            )
            return attr(ctx)

    def program_smt(self, ctx: dict, *, mode: str = "incremental") -> dict:
        with (
            self.profiler.span(
                run_id=self.run_id, trial_id=ctx.get("trial_id"), side=ctx.get("inc_exc", "?"), stage="PROGRAM"
            )
            if self.profiler
            else contextlib.nullcontext()
        ):
            return SMTProgrammer(
                engine=self.engine,
                # mode=mode,
                free_entity_out_dir=str(self.cfg.free_entity_extractor_dir()),
                free_qualifier_out_dir=str(self.cfg.free_entity_qualifier_identifier_dir()),
                entity_filter_out_dir=str(self.cfg.top_level_entity_filter_dir()),
                entity_enricher_out_dir=str(self.cfg.entity_enricher_dir()),
                namer_log_dir=str(self.cfg.namer_log_dir()),
                translator_log_dir=str(self.cfg.translator_log_dir()),
                validator_log_dir=str(self.cfg.validator_log_dir()),
                verifier_log_dir=str(self.cfg.verifier_log_dir()),
                report_dir=str(self.cfg.report_dir()),
                smt_out_dir=self.cfg.ir_dir,
                var_out_dir=self.cfg.symtab_dir,
                canon_out_dir=self.cfg.canon_dir,
                ent_out_dir=self.cfg.linkmap_dir,
                requirements_out_dir=self.cfg.requirements_dir,
            ).forward(ctx)

    # def match_patients(self, ctx: dict) -> dict:
    #     return SMTMatcher(engine=self.engine).forward(ctx)

    # ──────────────────── helper utilities ────────────────────
    def _attach_entities_for_context(self, ctx: dict, tid: str, side: str) -> dict:
        """Attach requirement_entities to *ctx* based on existing ctx['requirements']."""
        entities_per_req = []
        for idx, _ in enumerate(ctx.get("requirements", [])):
            key = (tid, side, idx)
            entities_per_req.append(
                self.entity_table.get(
                    key,
                    {"entities_recall": [], "entities_precision": [], "entities_old": []},
                )
            )
        ctx["requirement_entities"] = entities_per_req
        return ctx

    def _run_stages_for_single_context(
        self,
        ctx: dict,
        *,
        side: str,
        resume_from: "Stage | None",
        stop_after: "Stage | None",
    ) -> dict:
        """
        Run CANON → ATTR → PROGRAM → FINAL for a single context.
        Respects stop_after; assumes PREPROC & EXTRACT already done for this context.
        """
        cursor = resume_from or Stage.PREPROC

        # CANON
        if cursor.value < Stage.CANON.value:
            ctx = self.canonicalise(ctx)
            cursor = Stage.CANON
        if stop_after == Stage.CANON:
            return ctx

        # ATTR
        if cursor.value < Stage.ATTR.value:
            ctx = self.extract_attributes(ctx)
            cursor = Stage.ATTR
        if stop_after == Stage.ATTR:
            return ctx

        # PROGRAM
        if cursor.value < Stage.PROGRAM.value:
            ctx = self.program_smt(ctx)
            cursor = Stage.PROGRAM
        if stop_after == Stage.PROGRAM:
            return ctx

        # FINAL (matcher) – safe if method exists
        if cursor.value < Stage.FINAL.value and hasattr(self, "match_patients"):
            ctx = self.match_patients(ctx)  # type: ignore[attr-defined]
            cursor = Stage.FINAL

        return ctx

    # ──────────────────── main pipeline runner ────────────────────
    def run(
        self,
        trial: dict,
        side: str,
        *,
        resume_from: Stage | None = None,
        stop_after: Stage | None = None,
        extractor_opts: dict | None = None,
        skip_built_cohorts: bool = False,  # ← NEW
    ) -> dict:
        tid = trial.get("_id") or trial.get("trial_id")
        ckpt = self.cfg.ckpt_path

        # 0) ── load checkpoint if we're resuming ───────────────────────────
        ctx = None
        if resume_from is not None:
            # If user resumes from PREPROC but wants canonical as truth, rebuild from canonical instead of ckpt.
            if resume_from == Stage.PREPROC and self.cfg.preproc_source in ("auto", "canonical"):
                base = self._construct_context(trial, side)
                base = ensure_prompt_templates(
                    base,
                    force_reload=True,
                    pathmap=self.cfg.prompt_sources(),
                    roots=self.cfg.prompt_roots(),
                )
                canon = self._load_canonical_subcohort_preproc(tid)
                if canon is not None:
                    canon = dict(canon)
                    canon["inc_exc"] = side
                    ctx = self._retarget_preproc_ctx(canon, base, side)
                    ctx["__preproc_reused__"] = True
                    ctx["__preproc_source__"] = "canonical_subcohort_results"
                elif self.cfg.preproc_source == "canonical":
                    raise FileNotFoundError(
                        f"--preproc-source canonical but no file found: {self.cfg.canonical_subcohort_dir}/{tid}.json"
                    )

            if ctx is None:
                loader = {
                    Stage.PREPROC: load_preproc_ckpt,
                    Stage.CANON: load_canon_ckpt,
                    # Stage.NONCANON: load_noncanon_ckpt,
                    Stage.ATTR: load_attr_ckpt,
                    Stage.PROGRAM: load_program_ckpt,
                    Stage.FINAL: load_final_ckpt,
                }[resume_from]
                ctx = loader(self.engine, ckpt(resume_from.name.lower(), tid, side))

                # Force **overwrite** of ALL prompts from specified file paths
                ctx = ensure_prompt_templates(
                    ctx,
                    force_reload=True,
                    pathmap=self.cfg.prompt_sources(),
                    roots=self.cfg.prompt_roots(),
                )

        # Helper that commits a stage *only if* we're past it ---------------
        def save(stage: Stage, _ctx: dict):
            if stop_after is None or stage.value <= stop_after.value:
                {
                    Stage.PREPROC: save_preproc_ckpt,
                    Stage.CANON: save_canon_ckpt,
                    # Stage.NONCANON: save_noncanon_ckpt,
                    Stage.ATTR: save_attr_ckpt,
                    Stage.PROGRAM: save_program_ckpt,
                    Stage.FINAL: save_final_ckpt,
                }[stage](_ctx, ckpt(stage.name.lower(), tid, side))

        # 1) ── PREPROC (standalone) ────────────────────────────────────────
        if ctx is None:
            ctx = self.preprocess(trial, side, extractor_opts)
            cursor = Stage.PREPROC
            if not ctx.get("__preproc_reused__", False):
                save(cursor, ctx)  # only save the first time this trial is seen
        else:
            cursor = resume_from  # type: ignore[assignment]

        # 1b) ── EXTRACT (explicit step right after PREPROC) ───────────────
        ctx = self.extract_requirements(ctx, side, extractor_opts)
        if not ctx.get("__preproc_reused__", False):
            save(Stage.PREPROC, ctx)  # don’t mutate an existing PREPROC ckpt

        if stop_after == Stage.PREPROC:
            return ctx

        # 2) ── If we have cohort subcontexts, run later stages per-cohort ──
        cohort_subctxs = ctx.get("__cohort_contexts__")
        if cohort_subctxs:
            # Recompute cursor start
            cursor = resume_from or Stage.PREPROC

            # If requested, keep only subcohorts whose IR isn’t built yet
            if skip_built_cohorts:
                eff_ids = ctx.get("effective_trial_ids") or [
                    sc.get("trial_id") for sc in cohort_subctxs if sc.get("trial_id")
                ]
                missing = set(_list_missing_effective_ids(self.cfg.ir_dir, eff_ids))
                cohort_subctxs = [sc for sc in cohort_subctxs if sc.get("trial_id") in missing]
                if not cohort_subctxs:
                    # All subcohorts done — nothing to run
                    return ctx

            contexts_by_cohort: Dict[str, dict] = {}
            final_keys_by_cohort: Dict[str, List[str]] = {}

            # Ensure each subcontext has freshest prompt templates and run stages
            for sc in cohort_subctxs:
                stid = sc.get("trial_id", sc.get("trial_id_effective", ctx.get("trial_id")))
                with (
                    self.profiler.span(
                        run_id=self.run_id,
                        trial_id=stid,
                        side=side,
                        stage="COHORT_PIPELINE",
                        cohort_id=stid,
                    )
                    if self.profiler
                    else contextlib.nullcontext()
                ):
                    sc = ensure_prompt_templates(
                        sc,
                        force_reload=True,
                        pathmap=self.cfg.prompt_sources(),
                        roots=self.cfg.prompt_roots(),
                    )

                    # Attach/refresh entities for this subcontext (id is already suffixed)
                    sc = self._attach_entities_for_context(sc, stid, side)

                    # Run the remaining stages for this cohort
                    sc_out = self._run_stages_for_single_context(
                        sc,
                        side=side,
                        resume_from=cursor,
                        stop_after=stop_after,
                    )

                # Collect results
                contexts_by_cohort[stid] = sc_out
                final_keys_by_cohort[stid] = list(sc_out.keys())

            # Aggregate back to parent and return
            ctx["contexts_by_cohort"] = contexts_by_cohort
            ctx["final_keys_by_cohort"] = final_keys_by_cohort
            return ctx

        # 3) ── No cohorts → keep the original single-context flow ──────────
        if cursor.value < Stage.CANON.value:
            ctx = self.canonicalise(ctx)
            cursor = Stage.CANON
            save(cursor, ctx)
        if stop_after == Stage.CANON:
            return ctx

        if cursor.value < Stage.ATTR.value:
            ctx = self.extract_attributes(ctx)
            cursor = Stage.ATTR
            save(cursor, ctx)
        if stop_after == Stage.ATTR:
            return ctx

        if cursor.value < Stage.PROGRAM.value:
            ctx = self.program_smt(ctx)
            cursor = Stage.PROGRAM
            save(cursor, ctx)
        if stop_after == Stage.PROGRAM:
            return ctx

        if cursor.value < Stage.FINAL.value and hasattr(self, "match_patients"):
            ctx = self.match_patients(ctx)  # type: ignore[attr-defined]
            cursor = Stage.FINAL
            save(cursor, ctx)
        return ctx

    # ───────────────────── internal helpers ────────────────────
    def _construct_context(self, trial: dict, side: str) -> dict:
        # Build requirement_text from metadata when present, else derive from pre-supplied lists.
        def _derive_req_text(t: dict, s: str) -> str:
            meta = t.get("metadata", {})
            if isinstance(meta, dict) and f"{s}_criteria" in meta:
                return str(meta[f"{s}_criteria"])
            raw = t.get(f"{s}_criteria")
            if isinstance(raw, list):

                def _as_str(x: Any) -> str:
                    if isinstance(x, str):
                        return x
                    if isinstance(x, dict):
                        return x.get("requirement") or x.get("text") or ""
                    return str(x)

                return "\n".join(_as_str(x) for x in raw)
            return ""

        return {
            "engine": self.engine,
            "contextual_text": dict_to_readable_string(trial),
            "requirement_text": _derive_req_text(trial, side),
            "trial_id": trial.get("_id") or trial.get("trial_id"),
            "inc_exc": side,
        }


# ------------------------------------------------------------------------
#  prompt utilities (now support explicit path maps & overwrite)
# ------------------------------------------------------------------------


def ensure_prompt_templates(
    ctx: Dict[str, Any],
    *,
    force_reload: bool = True,
    pathmap: Dict[str, pathlib.Path] | None = None,
    roots: List[pathlib.Path] | None = None,
) -> Dict[str, Any]:
    """
    Attach every required *_prompt template to *ctx*.

    When `force_reload=True`, the function **overwrites** existing entries
    by re-reading from disk. Loading order per key:
      1) If `pathmap[key]` exists, use it.
      2) Else, try each directory in `roots` joined with the relative path.
      3) Else, fallback to default module-relative search:
         ./prompts/clinical_trial[/(inc|exc)]/<relative>

    Raises FileNotFoundError if no candidate path exists for a key.
    """
    if not force_reload and all(k in ctx for k in REQUIRED_PROMPTS):
        return ctx

    # normalize roots fallback if not provided
    if roots is None:
        base = pathlib.Path(__file__).resolve().parent / "prompts" / "clinical_trial"
        roots = [base, base / "inc", base / "exc"]

    updates: Dict[str, str] = {}
    tried_paths: Dict[str, List[str]] = {}

    for key, rel in REQUIRED_PROMPTS.items():
        candidates: List[pathlib.Path] = []
        if pathmap and key in pathmap:
            candidates.append(pathmap[key])
        candidates.extend([(r / rel) for r in roots])

        tried_paths[key] = [str(c) for c in candidates]

        loaded = False
        for cand in candidates:
            try:
                text = cand.read_text(encoding="utf-8")
                updates[key] = text
                loaded = True
                break
            except FileNotFoundError:
                continue

        if not loaded:
            tried = " | ".join(tried_paths[key])
            raise FileNotFoundError(f"Prompt not found for key '{key}'. Tried: {tried}")

    # Overwrite (or add) into ctx
    ctx.update(updates)

    return ctx


# ------------------------------------------------------------------------
#  Requirement injection utilities (JSONL loader)
# ------------------------------------------------------------------------


def _normalize_req(x: Any) -> dict:
    """Normalize a requirement item into the dict shape the pipeline expects."""
    if isinstance(x, str):
        return {"requirement": x, "text_span": "", "source": "manual"}
    if isinstance(x, dict):
        req = x.get("requirement") or x.get("text") or x.get("requirement_text")
        if not req:
            raise ValueError("Requirement dict must include 'requirement' (or 'text').")
        y = dict(x)
        y["requirement"] = req
        y.setdefault("text_span", "")
        y.setdefault("source", y.get("source", "manual"))
        return y
    raise ValueError(f"Unsupported requirement item: {type(x)}")


def _load_requirements_for_trial_from_jsonl(
    jsonl_path: pathlib.Path,
    trial_id: str,
    side: str,
    patient_id: str | None = None,
) -> List[dict]:
    """
    Return a normalized list of requirements for (trial_id, side) from a JSONL.

    Expected JSONL schema includes 'trial_id' and '<side>_criteria' keys.
    If multiple rows exist for the same trial_id, you may pass patient_id to disambiguate.
    """
    key = f"{side}_criteria"
    chosen: dict | None = None
    with jsonl_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            if rec.get("trial_id") != trial_id:
                continue
            if patient_id and rec.get("patient_id") != patient_id:
                continue
            chosen = rec
            break
    if not chosen:
        if patient_id:
            raise ValueError(f"No record found in {jsonl_path} for trial_id={trial_id} and patient_id={patient_id}")
        raise ValueError(f"No record found in {jsonl_path} for trial_id={trial_id}")

    if key not in chosen or not isinstance(chosen[key], list):
        raise KeyError(f"Record found but missing '{key}' list in JSONL for trial_id={trial_id}")

    return [_normalize_req(item) for item in chosen[key]]


# ------------------------------------------------------------------------
#  CLI entry-point
# ------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the refactored trial pipeline (prompts reload/overwrite on resume; optional requirements injection from JSONL)."
    )
    p.add_argument("trial_id", help="NCT identifier to run")
    p.add_argument("--side", choices=["inclusion", "exclusion", "both"], default="both")
    p.add_argument("--resume-from", choices=[s.name.lower() for s in Stage], default=None)
    p.add_argument("--stop-after", choices=[s.name.lower() for s in Stage], default=None)
    p.add_argument("--log-dir", default="run_logs")

    # ── prompt source arguments ───────────────────────────────────────────
    p.add_argument(
        "--prompt-root",
        default="src/prompts/clinical_trial",
        help="Base directory for relative prompt paths (default: src/prompts/clinical_trial under repo root).",
    )
    p.add_argument(
        "--prompt-map",
        default=None,
        help="JSON file mapping prompt keys to explicit file paths; overrides per-key.",
    )

    # ── requirements JSONL injection ──────────────────────────────────────
    p.add_argument(
        "--requirements-jsonl",
        default=None,
        help="Path to a JSONL file that contains inclusion_criteria / exclusion_criteria per trial_id.",
    )
    p.add_argument(
        "--patient-id",
        default=None,
        help="If your JSONL has multiple rows per trial_id, choose the one with this patient_id.",
    )
    p.add_argument(
        "--skip-extractor",
        action="store_true",
        help="If set and requirements are pre-injected, bypass RequirementExtractor.forward().",
    )

    # ── profiling flags ───────────────────────────────────────────────────
    p.add_argument("--profile", default=True, action="store_true", help="Enable wall-clock profiling (JSONL).")
    p.add_argument("--profile-jsonl", default=None, help="Path to write profile JSONL (default run_logs/profile.jsonl).")

    # ── cohort skipping flag ──────────────────────────────────────────────
    p.add_argument(
        "--skip-built-cohorts",
        default=True,
        action="store_true",
        help="If the trial has cohort subcontexts, only run the subcohorts whose IR artifacts are missing.",
    )

    # ── NEW: preproc source control ───────────────────────────────────────
    p.add_argument(
        "--preproc-source",
        choices=["ckpt", "llm", "canonical", "auto"],
        default="ckpt",
        help="Where to get PREPROC: ckpt (default), llm, canonical, or auto (canonical→ckpt→llm).",
    )
    p.add_argument(
        "--canonical-subcohort-dir",
        default=os.getenv("SATIR_BUILD", "build") + "/../canonical_subcohort_results",
        help="Directory containing <trial_id>.json canonical subcohort/preproc outputs.",
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = Config(
        log_dir=pathlib.Path(args.log_dir),
        prompt_root=pathlib.Path(args.prompt_root),
        prompt_map_json=pathlib.Path(args.prompt_map) if args.prompt_map else None,
        enable_profile=bool(args.profile),
        profile_jsonl=pathlib.Path(args.profile_jsonl) if args.profile_jsonl else pathlib.Path("run_logs/profile.jsonl"),
        preproc_source=str(args.preproc_source),
        canonical_subcohort_dir=pathlib.Path(args.canonical_subcohort_dir),
    )

    # --- load corpus (simplified single-file fetch) --------------------
    corpus_path = cfg.data_root / "sigir/corpus.jsonl"
    with corpus_path.open(encoding="utf-8") as fh:
        trial = next(json.loads(l) for l in fh if f'"_id": "{args.trial_id}"' in l)

    # Prepare run(s)
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")  # stable id for this invocation
    ts = run_id
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    profiler = Profiler(cfg.profile_jsonl) if cfg.enable_profile else None
    pipeline = TrialPipeline(cfg, profiler=profiler, run_id=run_id)

    if args.side == "both":
        # ——— 为两个侧各自准备 trial + extractor_opts ———
        trial_inc = deepcopy(trial)
        trial_exc = deepcopy(trial)

        extractor_opts_inc: dict = {}
        extractor_opts_exc: dict = {}

        if args.requirements_jsonl:
            inc_reqs = _load_requirements_for_trial_from_jsonl(
                pathlib.Path(args.requirements_jsonl), args.trial_id, "inclusion", args.patient_id
            )
            exc_reqs = _load_requirements_for_trial_from_jsonl(
                pathlib.Path(args.requirements_jsonl), args.trial_id, "exclusion", args.patient_id
            )
            trial_inc["inclusion_criteria"] = inc_reqs
            trial_exc["exclusion_criteria"] = exc_reqs
            extractor_opts_inc["pre_extracted"] = True
            extractor_opts_exc["pre_extracted"] = True
            if args.skip_extractor:
                # ⚠ 跳过 extractor 会绕过 precision/跨侧修复，不建议在该模式下使用
                extractor_opts_inc["__skip_entire_extractor__"] = True
                extractor_opts_exc["__skip_entire_extractor__"] = True

        # ——— 跑双侧 + 屏障 ———
        with Tee(cfg.log_dir / f"{args.trial_id}_both_{ts}.log"):
            ctx_inc, ctx_exc = pipeline.run_both_with_contradiction_barrier(
                trial_inc,
                trial_exc,
                resume_from=Stage.from_str(args.resume_from),
                stop_after=Stage.from_str(args.stop_after),
                extractor_opts_inc=extractor_opts_inc or None,
                extractor_opts_exc=extractor_opts_exc or None,
            )
            print(f"✓ done (inclusion) – final keys:", list(ctx_inc.keys()))
            print(f"✓ done (exclusion) – final keys:", list(ctx_exc.keys()))

    else:
        # —— 保留原来的单侧路径 ——
        sides = [args.side]
        for side in sides:
            trial_side = deepcopy(trial)
            extractor_opts: dict = {}
            if args.requirements_jsonl:
                reqs = _load_requirements_for_trial_from_jsonl(
                    pathlib.Path(args.requirements_jsonl),
                    args.trial_id,
                    side,
                    args.patient_id,
                )
                trial_side[side + "_criteria"] = reqs
                extractor_opts["pre_extracted"] = True
                if args.skip_extractor:
                    extractor_opts["__skip_entire_extractor__"] = True

            log_path = cfg.log_dir / f"{args.trial_id}_{side}_{ts}.log"
            with Tee(log_path):
                ctx = pipeline.run(
                    trial_side,
                    side,
                    resume_from=Stage.from_str(args.resume_from),
                    stop_after=Stage.from_str(args.stop_after),
                    extractor_opts=extractor_opts or None,
                    skip_built_cohorts=bool(args.skip_built_cohorts),
                )
                print(f"✓ done ({side}) – final keys:", list(ctx.keys()))
                if "__cohort_contexts__" in ctx or "contexts_by_cohort" in ctx:
                    print("• Cohort run detected.")
                    for tid_eff, sc in (ctx.get("contexts_by_cohort") or {}).items():
                        print(f"  - {tid_eff}: keys = {list(sc.keys())}")

    # Optional: tiny CSV stage summary for this run
    if profiler:
        try:
            import csv

            rows = []
            with cfg.profile_jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("run_id") == run_id and rec.get("action") == "stage":
                        rows.append(
                            {
                                "trial_id": rec["trial_id"],
                                "side": rec["side"],
                                "stage": rec["stage"],
                                "cohort_id": rec.get("cohort_id"),
                                "duration_s": rec["duration_s"],
                            }
                        )
            out = cfg.log_dir / f"profile_summary_{run_id}.csv"
            with out.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["trial_id", "side", "stage", "cohort_id", "duration_s"]
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"✓ Wrote profile summary CSV → {out}")
        except Exception as e:
            print(f"[WARN] Could not produce CSV summary: {e}")


if __name__ == "__main__":
    main()
