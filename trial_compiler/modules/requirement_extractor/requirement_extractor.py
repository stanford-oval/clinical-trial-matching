#!/usr/bin/env python3
"""
requirement_extractor.py  –  cohort-aware requirement extraction pipeline
"""

from __future__ import annotations

import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import dspy

# ----------------------------------------------------------------------
# Stages (no internal preprocessor here; it is a separate stage upstream)
# ----------------------------------------------------------------------

from .stages import (
    RequirementRudimentaryExtractor,
    RequirementEntitySpanExpander,          # optional
    RequirementLogicalPrecisionRewriter,
    RequirementDecomposer,                  # splits logical ORs/ANDs into atomic components
    RequirementHardSoftClassifier,
    RequirementPreambler,
    # RequirementAtomicityEnhancer,
    # RequirementSelfcontainednessEnhancer,
    # RequirementDeduplicator,
)

# ----------------------------------------------------------------------
# Pretty-print helpers (self-contained)
# ----------------------------------------------------------------------

def _pretty_requirements_raw(reqs: List) -> str:
    def _txt(r):
        return r["requirement"] if isinstance(r, dict) else str(r)
    return "\n".join(_txt(r) for r in reqs)

def _pretty_requirements_tagged(reqs: List) -> str:
    def _txt(r):
        if isinstance(r, dict):
            tag = r.get("constraint")
            prefix = "[H]" if tag == "hard" else "[S]" if tag == "soft" else "[-]"
            return f"{prefix} {r['requirement']}"
        return str(r)
    return "\n".join(_txt(r) for r in reqs)

def _pretty_requirements_components(reqs: List) -> str:
    out_lines: List[str] = []
    for r in reqs:
        if not isinstance(r, dict):
            out_lines.append(str(r))
            continue
        req_tag = r.get("constraint")
        req_prefix = "[H]" if req_tag == "hard" else "[S]" if req_tag == "soft" else "[-]"
        out_lines.append(f"{req_prefix} {r.get('requirement', str(r))}")
        for c in r.get("components", []):
            if isinstance(c, dict):
                comp_tag = c.get("constraint")
                comp_prefix = "[H]" if comp_tag == "hard" else "[S]" if comp_tag == "soft" else "[-]"
                comp_text = c.get("text", str(c))
            else:
                comp_prefix = "[-]"
                comp_text = str(c)
            out_lines.append(f"  └─ {comp_prefix} {comp_text}")
    return "\n".join(out_lines)

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def _export_mapping(mapping: Dict | List | Any, path: str | Path, label: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{label}] mapping exported → {path.resolve()}")

# ----------------------------------------------------------------------
# Pipeline wrapper
# ----------------------------------------------------------------------

class RequirementExtractor(dspy.Module):
    """Cohort-aware requirement extraction with debug printouts."""

    def __init__(
        self,
        engine,
        *,
        pre_extracted: bool = False,
        verbose: bool = True,
        compare_logs: bool = True,
        diff_out: str | None = None,
        extraction_map_out: str | None = None,
        preproc_map_out: str | None = None,  # kept for API compat; unused here
        span_map_out: str | None = None,
        precision_map_out: str | None = None,
        decomposer_map_out: str | None = None,
        classifier_map_out: str | None = None,

        preamble_map_out: str | None = None,
        preamble_enable_llm: bool = True,
        preamble_batch_size: int = 40,
        stop_after: str | None = None,
        list_type: str = "inclusion",

        # NEW: passthrough knobs for rudimentary extractor retries
        extract_max_attempts: int = 3,
        extract_backoff_s: float = 0.5,
        extract_jitter_s: float = 0.25,
        verifier_max_attempts: int = 3,
        debug: bool = True,

        resume_from: Optional[str] = None,   # None | "precision"

    ):
        super().__init__()
        self.engine = engine
        self.verbose      = verbose
        self.compare_logs = compare_logs
        self.diff_out_dir = Path(diff_out) if diff_out else None

        # ── stages -----------------------------------------------------
        log_dir = (
            Path(extraction_map_out).parent
            if extraction_map_out else Path("mbench/req_mbench/extraction_maps")
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        if not pre_extracted:
            # Use the rudimentary extractor stack (now with retry knobs)
            self.extractor = RequirementRudimentaryExtractor(
                engine,
                log_dir=log_dir,
                extract_max_attempts=extract_max_attempts,
                extract_backoff_s=extract_backoff_s,
                extract_jitter_s=extract_jitter_s,
                verifier_max_attempts=verifier_max_attempts,
                debug=debug,
            )
            self.preambler = None
        else:
            self.extractor = None
            self.preambler = RequirementPreambler(
                engine,
                log_dir=preamble_map_out,
                enable_llm=preamble_enable_llm,
                verbose=verbose,
            )
            self._preamble_map_out = preamble_map_out
            self._preamble_batch_size = preamble_batch_size

        self.span_exp   = RequirementEntitySpanExpander(engine)
        self.precision  = RequirementLogicalPrecisionRewriter(engine, log_dir=precision_map_out)
        self.decomposer = RequirementDecomposer(engine)
        self.classifier = RequirementHardSoftClassifier(engine)

        self.extraction_map_out  = extraction_map_out
        self.span_map_out        = span_map_out
        self.precision_map_out   = precision_map_out
        self.decomposer_map_out  = decomposer_map_out
        self.classifier_map_out  = classifier_map_out
        self.preamble_map_out    = preamble_map_out

        self.stop_after = stop_after or "classifier"
        _VALID = {None, "extract", "preamble", "span", "precision", "decompose", "classifier"}
        if self.stop_after not in _VALID:
            raise ValueError(f"stop_after must be one of {_VALID}, got {self.stop_after!r}")

        self.list_type = list_type

        self.resume_from = resume_from
        _RESUME_VALID = {None, "precision"}
        if self.resume_from not in _RESUME_VALID:
            raise ValueError(f"resume_from must be one of {_RESUME_VALID}, got {self.resume_from!r}")
        
    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, stage: str, context: Dict):
        if not self.verbose:
            return
        reqs = context.get("requirements", [])
        if stage.lower().startswith("hard-soft") or stage.lower().startswith("decomposition"):
            txt = _pretty_requirements_components(reqs)
        else:
            txt = _pretty_requirements_raw(reqs)
        print(f"► After {stage}:\n{txt}\n")

    @staticmethod
    def _as_text_list(reqs):
        return [r["requirement"] if isinstance(r, dict) else str(r) for r in reqs]

    def _side_by_side_str(self, before, after) -> str:
        left  = sorted(self._as_text_list(before), key=str.lower)
        right = sorted(self._as_text_list(after), key=str.lower)
        cols, _ = shutil.get_terminal_size(fallback=(120, 24))
        gap = 4
        width_left = min(max(map(len, left)) if left else 0, (cols - gap) // 2)
        lines = ["-" * cols]
        for i in range(max(len(left), len(right))):
            ltxt = left[i]  if i < len(left)  else ""
            rtxt = right[i] if i < len(right) else ""
            lines.append(f"{ltxt:<{width_left}}{' ' * gap}| {rtxt}")
        lines.append("-" * cols)
        return "\n".join(lines)

    def _emit_diff(self, stage: str, before, after):
        if not self.compare_logs:
            return
        diff = self._side_by_side_str(before, after)
        print(f"► {stage} (alphabetical diff)\n{diff}\n")
        if self.diff_out_dir:
            self.diff_out_dir.mkdir(parents=True, exist_ok=True)
            p = self.diff_out_dir / f"{stage.replace(' ', '_')}.diff.txt"
            p.write_text(diff, encoding="utf-8")
            if self.verbose:
                print(f"[{stage}] diff written → {p.resolve()}\n")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def forward(self, context: Dict) -> Dict:  # type: ignore[override]
        resume_from = getattr(self, "resume_from", None)

        # ───────────────────── path A: 从头跑到 precision ─────────────────────
        if resume_from != "precision":
            # 1) Extraction -------------------------------------------------
            if self.extractor and "requirements" not in context:
                subctxs = context.get("__cohort_contexts__") or context.get("__substudy_contexts__", [])
                if subctxs:
                    all_requirements = []
                    requirements_by_substudy = {}
                    verification_by_substudy = {}
                    for sc in subctxs:
                        sc_res = self.extractor.forward(sc, use_full_context=True)
                        reqs = sc_res.get("requirements", []) or []
                        all_requirements.extend(reqs)
                        requirements_by_substudy[sc["trial_id"]] = reqs
                        if "verification" in sc_res:
                            verification_by_substudy[sc["trial_id"]] = sc_res["verification"]
                    context["requirements"] = all_requirements
                    context["requirements_by_substudy"] = requirements_by_substudy
                    if verification_by_substudy:
                        context["verification_by_substudy"] = verification_by_substudy
                    context["effective_trial_ids"] = [sc["trial_id"] for sc in subctxs]
                    context["parent_trial_id"] = context.get("parent_trial_id", context.get("trial_id"))
                    context["__from_substudies__"] = True
                else:
                    context = self.extractor.forward(context, use_full_context=True)
            else:
                self._log("extraction (skipped)", context)

            self._log("extraction", context)
            if self.stop_after == "extract":
                return context

            if self.extraction_map_out:
                entries = {
                    (r["requirement"] if isinstance(r, dict) else str(r)): {
                        "text_span": (r.get("text_span", "") if isinstance(r, dict) else "")
                    }
                    for r in context.get("requirements", [])
                }
                payload = {
                    "requirement_text": context.get("requirement_text", ""),
                    "extractions": entries,
                }
                _export_mapping(payload, self.extraction_map_out, "extraction")

            # 1b) Preamble rewrite (only when pre_extracted → preambler is set)
            if getattr(self, "preambler", None) is not None:
                if self.preamble_map_out:
                    context["__preamble_map_out__"] = self.preamble_map_out
                if hasattr(self, "_preamble_batch_size"):
                    context["__preamble_batch_size__"] = self._preamble_batch_size
                before = list(context.get("requirements", []))
                context = self.preambler.forward(context, use_full_context=True)
                self._emit_diff("preamble rewrite", before, context.get("requirements", []))
                self._log("preamble rewrite", context)
                if self.stop_after == "preamble":
                    return context

            # 2) Entity-span expansion
            before = list(context["requirements"])
            context = self.span_exp.forward(context, use_full_context=True)
            self._emit_diff("span expansion", before, context["requirements"])
            self._log("span expansion", context)

            if self.span_map_out and "span_mapping" in context:
                _export_mapping(context["span_mapping"], self.span_map_out, "span")

            if self.verbose:
                print("► span-expansion metrics:",
                    json.dumps(context.get("span_expansion_metrics", {}), indent=2), "\n")

            if self.stop_after == "span":
                return context

            # 3) Logical precision rewrite
            context = self.precision.forward(context, use_full_context=True)
            self._log("logical precision", context)
            if self.precision_map_out and "precision_mapping" in context:
                _export_mapping(context["precision_mapping"], self.precision_map_out, "logical-precision")
            if self.stop_after == "precision":
                return context

        # ───────────────────── path B: 从 precision 之后继续 ────────────────────
        else:
            # 要求：context 内已具备 precision 后的 requirements
            if "requirements" not in context:
                # 容错：没有就直接返回，不强行继续
                return context
            
        # 4) Requirement decomposition
        before = list(context["requirements"])
        context = self.decomposer.forward(context, use_full_context=True)
        self._emit_diff("decomposition", before, context["requirements"])
        self._log("decomposition", context)

        if self.decomposer_map_out and "decomposition_mapping" in context:
            _export_mapping(context["decomposition_mapping"], self.decomposer_map_out, "decomposition")
            base = Path(self.decomposer_map_out)
            out_dir = base.parent
            stem = base.stem
            if "verification_history" in context:
                _export_mapping(context["verification_history"], out_dir / f"{stem}.verification_history.json",
                                "decomposition-verification-history")
            if "verification_results" in context:
                _export_mapping(context["verification_results"], out_dir / f"{stem}.verification_results.json",
                                "decomposition-verification-results")

        if self.stop_after == "decompose":
            return context

        # 5) Hard/Soft classification
        before = list(context["requirements"])
        context = self.classifier.forward(context, use_full_context=True)
        self._emit_diff("hard-soft classification", before, context["requirements"])
        self._log("hard-soft classification", context)

        if self.classifier_map_out:
            c_map = {
                r["requirement"]: {
                    "constraint": r.get("constraint"),
                    "components": r.get("components", []),
                }
                for r in context["requirements"]
                if isinstance(r, dict) and "requirement" in r
            }
            _export_mapping(c_map, self.classifier_map_out, "hard-soft")

        return context