"""
requirement_extractor.py  –  end-to-end preprocessing pipeline
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Any

import dspy

# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

from .stages import (
    PatientStateRudimentaryExtractor,
    PatientStateEntitySpanExpander,          # optional
    PatientStateLogicalPrecisionRewriter,
    # PatientStateAtomicityEnhancer,
    # PatientStateSelfcontainednessEnhancer,
    # PatientStateDeduplicator,
    PatientStateDifferentialDiagnoser,       # <<< NEW
)

# ----------------------------------------------------------------------
# Pretty-print helpers (self-contained)
# ----------------------------------------------------------------------

def _req_text(r: Any) -> str:
    """Return a human-readable line for either a requirement or a patient fact."""
    if isinstance(r, dict):
        return r.get("requirement") or r.get("fact") or r.get("text") or str(r)
    return str(r)

def _pretty_requirements_raw(reqs: List) -> str:
    """Legacy view – one line per requirement/fact, no tags."""
    return "\n".join(_req_text(r) for r in reqs)

def _pretty_requirements_tagged(reqs: List) -> str:
    """Same list but prefix [H]/[S] when a constraint tag is present."""
    def _txt(r):
        if isinstance(r, dict):
            tag = r.get("constraint")
            prefix = "[H]" if tag == "hard" else "[S]" if tag == "soft" else "[-]"
            return f"{prefix} {_req_text(r)}"
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
        out_lines.append(f"{req_prefix} {_req_text(r)}")

        for c in r.get("components", []):
            if isinstance(c, dict):
                comp_tag = c.get("constraint")
                comp_prefix = "[H]" if comp_tag == "hard" else "[S]" if comp_tag == "soft" else "[-]"
                comp_text = c.get("text") or c.get("requirement") or c.get("fact") or str(c)
            else:
                comp_prefix = "[-]"
                comp_text = str(c)
            out_lines.append(f"  └─ {comp_prefix} {comp_text}")
    return "\n".join(out_lines)

# <<< NEW: pretty printer for diagnoses
def _pretty_diagnoses(diags: List[Dict[str, Any]]) -> str:
    lines = []
    for d in diags or []:
        name = d.get("diagnosis", "?")
        status = d.get("status", "?")
        conf = d.get("confidence", 0)
        support = "; ".join((d.get("support") or [])[:3])
        lines.append(f"• {name}  [{status}, conf={conf:.2f}]  support={support}")
    return "\n".join(lines) or "(no candidates)"

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def _export_mapping(mapping: Dict | List | Any, path: str | Path, label: str) -> None:
    """Dump any JSON-serialisable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{label}] mapping exported → {path.resolve()}")

# <<< NEW: very small mbench helper (JSONL, one record per run)
def _mbench_log(record: Dict[str, Any], path: str | Path) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[mbench] appended → {p.resolve()}")

# ----------------------------------------------------------------------
# Pipeline wrapper
# ----------------------------------------------------------------------

class PatientStateExtractor(dspy.Module):
    """End-to-end requirement preprocessing with debug printouts."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        engine,
        *,
        pre_extracted: bool = False,
        verbose: bool = True,
        compare_logs: bool = True,
        diff_out: str | None = None,
        extraction_map_out: str | None = None,
        span_map_out: str | None = None,
        precision_map_out: str | None = None,
        # <<< NEW: diagnosis/mbench outputs (match export pattern of other stages)
        diagnosis_map_out: str | None = None,
        mbench_out: str | None = None,              # JSONL file to append eval records
        mbench_task_prefix: str = "patient_ddx",    # task name prefix
        stop_after: str | None = None,
        list_type: str = "inclusion",
        # <<< NEW: 只跑诊断的开关
        diagnosis_only: bool = False,
    ):
        super().__init__()
        self.verbose      = verbose
        self.compare_logs = compare_logs
        self.diff_out_dir = Path(diff_out) if diff_out else None

        # ── stages -----------------------------------------------------
        if not pre_extracted:
            log_dir = (
                Path(extraction_map_out).parent           # same folder
                if extraction_map_out else Path("extract_logs") # fallback default
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            self.extractor = PatientStateRudimentaryExtractor(engine, log_dir=log_dir)
        else:
            self.extractor = None

        self.span_exp   = PatientStateEntitySpanExpander(engine, log_dir=span_map_out)
        self.precision  = PatientStateLogicalPrecisionRewriter(engine, log_dir=precision_map_out)
        self.diagnosis  = PatientStateDifferentialDiagnoser(engine, log_dir=diagnosis_map_out)  # <<< NEW

        # ── optional exports ------------------------------------------
        self.extraction_map_out  = extraction_map_out
        self.span_map_out        = span_map_out
        self.precision_map_out   = precision_map_out
        self.diagnosis_map_out   = diagnosis_map_out        # <<< NEW
        self.mbench_out          = mbench_out               # <<< NEW
        self.mbench_task_prefix  = mbench_task_prefix       # <<< NEW

        # ── stop-after -------------------------------------------------
        self.stop_after = stop_after or "classifier"
        _VALID = {None, "extract", "span", "precision", "diagnosis", "decompose", "classifier"}  # <<< NEW
        if self.stop_after not in _VALID:
            raise ValueError(f"stop_after must be one of {_VALID}, got {self.stop_after!r}")

        self.list_type = list_type
        self.diagnosis_only = diagnosis_only   # <<< NEW

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, stage: str, context: Dict):
        if not self.verbose:
            return
        reqs = context.get("patient_facts", [])
        if stage.lower().startswith("hard-soft") or stage.lower().startswith("decomposition"):
            txt = _pretty_requirements_components(reqs)
        elif stage.lower().startswith("differential"):
            txt = _pretty_diagnoses(context.get("diagnosis_candidates", []))
        else:
            txt = _pretty_requirements_raw(reqs)
        print(f"► After {stage}:\n{txt}\n")

    @staticmethod
    def _as_text_list(reqs):
        return [_req_text(r) for r in reqs]

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

        # 如果只跑诊断，直接跳过 1/2/3 步 ------------------------------<<< NEW
        if self.diagnosis_only:
            # 4) Differential diagnosis (NEW)
            context = self.diagnosis.forward(context, use_full_context=True)
            self._log("differential diagnosis", context)

            # Optional export mirroring other stages
            if self.diagnosis_map_out and "diagnosis_candidates" in context:
                _export_mapping(context["diagnosis_candidates"], self.diagnosis_map_out, "diagnosis")

            # Optional mbench JSONL record
            if self.mbench_out:
                rec = {
                    "task": f"{self.mbench_task_prefix}.differential",
                    "note_id": context.get("note_id") or context.get("trial_id") or "",
                    "n_inputs": len(context.get("patient_facts", [])),
                    "inputs": [
                        r.get("fact", r.get("requirement", "")) if isinstance(r, dict) else str(r)
                        for r in context.get("patient_facts", [])
                    ],
                    "outputs": context.get("diagnosis_candidates", []),
                    "summary": context.get("diagnosis_summary", {}),
                }
                _mbench_log(rec, self.mbench_out)

            # diagnosis_only 下，诊断后直接返回
            return context
        # ----------------------------------------------------------------

        # 1) Extraction -------------------------------------------------
        if self.extractor and "patient_facts" not in context:
            context = self.extractor.forward(context, use_full_context=True)
        else:
            self._log("extraction (skipped)", {"patient_facts": context.get("patient_facts", [])})

        self._log("extraction", {"patient_facts": context.get("patient_facts", [])})
        if self.stop_after == "extract":
            return context

        if self.extraction_map_out:
            extraction_map = {
                (r.get("requirement") or r.get("fact") or str(r)):
                    {"text_span": (r.get("text_span") or "")}
                for r in context.get("patient_facts", [])
                if isinstance(r, dict) or r
            }
            _export_mapping(extraction_map, self.extraction_map_out, "extraction")

        # 2) Entity-span expansion -------------------------------------
        before = list(context["patient_facts"])  # shallow copy for diff
        context = self.span_exp.forward(context, use_full_context=True)

        self._emit_diff("span expansion", before, context["patient_facts"])
        self._log("span expansion", context)

        if self.span_map_out and "span_mapping" in context:
            _export_mapping(context["span_mapping"], self.span_map_out, "span")

        if self.verbose:
            print("► span-expansion metrics:",
                  json.dumps(context.get("span_attempt_logs", {}), indent=2), "\n")

        if self.stop_after == "span":
            return context

        # 3) Logical precision rewrite ---------------------------------
        context = self.precision.forward(context, use_full_context=True)
        self._log("logical precision", context)
        if self.precision_map_out and "precision_mapping" in context:
            _export_mapping(context["precision_mapping"], self.precision_map_out, "logical-precision")
        if self.stop_after == "precision":
            return context

        # 4) Differential diagnosis (NEW) -------------------------------
        # Inputs: context["patient_facts"] (rewritten), context.get("contextual_text")
        context = self.diagnosis.forward(context, use_full_context=True)
        self._log("differential diagnosis", context)

        # Optional export mirroring other stages
        if self.diagnosis_map_out and "diagnosis_candidates" in context:
            _export_mapping(context["diagnosis_candidates"], self.diagnosis_map_out, "diagnosis")

        # Optional mbench JSONL record
        if self.mbench_out:
            # keep the record small and auditable; no raw model outputs here
            rec = {
                "task": f"{self.mbench_task_prefix}.differential",
                "note_id": context.get("note_id") or context.get("trial_id") or "",
                "n_inputs": len(context.get("patient_facts", [])),
                "inputs": [
                    r.get("fact", r.get("requirement", "")) if isinstance(r, dict) else str(r)
                    for r in context.get("patient_facts", [])
                ],
                "outputs": context.get("diagnosis_candidates", []),
                "summary": context.get("diagnosis_summary", {}),
            }
            _mbench_log(rec, self.mbench_out)

        if self.stop_after == "diagnosis":
            return context

        # hand off for downstream
        context["requirements"] = list(context.get("precision_mapping", {}).values())
        return context
