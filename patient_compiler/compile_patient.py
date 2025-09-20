"""
Refactored, modular version of the **patient-notes SMT pipeline**
with **automatic prompt reloading when resuming from checkpoints**.
=================================================================
Key changes vs. clinical-trial variant
-------------------------------------
* Domain switched from clinical trials to *patient notes*.
* **Removed** requirement decomposition and hard/soft classification steps.
* **Removed** variable miner / matcher stage (pipeline ends at PROGRAM).
* Prompts list trimmed accordingly (no Decomposer / HardSoft / VariableMiner).
* Input corpus now expects JSONL with fields: {"_id": <str>, "text": <str>, "metadata": {...}}
* Checkpoint file names and logging updated for note IDs (no inclusion/exclusion side).
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
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Sequence, Tuple

# ───────────────────── 3ʳᵈ-party dependencies ─────────────────────
from openai import AzureOpenAI  # type: ignore  # noqa: F401 – retained for future use
import dspy  # type: ignore

# ──────────────────────── local modules ─────────────────────────
from smt_core.checkpoint_io import (
    load_canon_ckpt,
    load_attr_ckpt,
    load_preproc_ckpt,
    load_program_ckpt,
    save_canon_ckpt,
    save_attr_ckpt,
    save_preproc_ckpt,
    save_program_ckpt,
)
from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
from smt_core.inference_engine import AzureInferenceEngine
from smt_core.modules.entity_canonicalizer import EntityCanonicalizer
from patient_compiler.modules.patient_state_extractor.PatientStateExtractor import PatientStateExtractor
from patient_compiler.modules.patient_coder import PatientCoder
from patient_compiler.modules.attribute_extractor import AttributeExtractor
from smt_core.helpers_entities import load_entity_annotations

from patient_compiler.modules.patient_state_extractor.stages.PatientStateExplicitDiagnoseExtractor import (
    PatientStateExplicitDiagnoseExtractor,   # <<< NEW
)


# ─────────────────────── global constants ───────────────────────
dspy.settings.configure(lm_cache=None)
logging.getLogger("azure").setLevel(logging.WARNING)

# ------------------------------------------------------------------------
#  Config & utility layer
# ------------------------------------------------------------------------

@dataclass
class Config:
    """Holds every tunable path/flag so we avoid magic constants."""

    # root dirs (may come from env vars)
    repo_root: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
    data_root: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(os.getenv("PN_DATA", "dataset/patient_notes"))
    )

    # checkpoint / log dirs
    ckpt_dir: pathlib.Path = pathlib.Path("checkpoints")
    log_dir: pathlib.Path = pathlib.Path("run_logs")
    status_file: pathlib.Path = pathlib.Path("run_logs/status.jsonl")

    # misc
    max_retries: int = int(os.getenv("MAX_RETRIES", 3))

    # azure
    azure_endpoint: str = os.getenv("OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("OPENAI_API_KEY", "")
    #model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    model_name = MODEL_NAME

    build_root: pathlib.Path = pathlib.Path(os.getenv("SATIR_BUILD", "build"))  # override via SATIR_BUILD env var
    ir_dir: pathlib.Path = field(init=False)
    symtab_dir: pathlib.Path = field(init=False)
    linkmap_dir: pathlib.Path = field(init=False)

    # entity annotations (optional for notes)
    entity_jsonl: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(
            os.getenv(
                "PN_ENTITY_JSONL",
                "dataset/patient_notes/ner_labels/patient_notes_entities.jsonl",
            )
        )
    )

    # mbench roots (req_mbench keeps extractor/rewriter/diagnosis exports)
    mbench_root: pathlib.Path = pathlib.Path(os.getenv("SMT_MBENCH_ROOT", "mbench/smt_mbench"))
    req_mbench_root: pathlib.Path = pathlib.Path(os.getenv("REQ_MBENCH_ROOT", "mbench/req_mbench"))
    ddx_mbench_file: pathlib.Path = pathlib.Path(
        os.getenv("DDX_MBENCH_FILE", "mbench/req_mbench/ddx.jsonl")
    )

    diagnosis_only: bool = False

    # helper dirs -----------------------------------------------------------

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

    def diagnosis_coder_log_dir(self)  -> pathlib.Path:
        path = self.mbench_root / "diagnosis_coder_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ckpt_path(self, stage: str, note_id: str) -> pathlib.Path:
        """Generate <ckpt_dir>/<stage>/<note_id>_<stage>.chkpt.json"""
        root = self.ckpt_dir / stage
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{note_id}_{stage}.chkpt.json"

    def __post_init__(self):
        # derive the three siblings once
        self.ir_dir = self.build_root / "ir"
        self.symtab_dir = self.build_root / "symtab"
        self.linkmap_dir = self.build_root / "linkmap"
        for d in (self.ir_dir, self.symtab_dir, self.linkmap_dir):
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
#  Pipeline stage enumeration (matcher removed)
# ------------------------------------------------------------------------

class Stage(Enum):
    PREPROC = auto()
    CANON = auto()
    ATTR = auto()
    PROGRAM = auto()

    @classmethod
    def from_str(cls, s: str | None) -> "Stage | None":
        if s is None:
            return None
        return cls[s.upper()]


# ------------------------------------------------------------------------
#  Core pipeline orchestrator (Patient Notes)
# ------------------------------------------------------------------------

class PatientNotesPipeline:
    """End-to-end driver with checkpointing and resumability for patient notes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # entity annotations are optional for notes; load if present
        try:
            self.entity_table = load_entity_annotations(cfg.entity_jsonl)
            logging.info("Entity annotations loaded: %s entries", len(self.entity_table))
        except Exception:
            self.entity_table = {}
            logging.info("Entity annotations not found; proceeding without them.")

        self.engine = AzureInferenceEngine(
            endpoint=cfg.azure_endpoint,
            api_key_env_var="OPENAI_API_KEY",
            model_name=cfg.model_name,
        )

    # ───────────────────────── stages ──────────────────────────
    def preprocess(self, note: dict, extractor_opts: dict | None = None) -> dict:
        extractor_opts = extractor_opts or {}

        # micro-benchmark paths (note-centric)
        nid = note.get("_id") or note.get("note_id")

        # requirement-stage exports together under req_mbench_root
        mbench_root = self.cfg.req_mbench_root
        extractor_opts.setdefault(
            "extraction_map_out",
            str(mbench_root / "extraction_maps" / f"{nid}_extraction.json"),
        )
        extractor_opts.setdefault(
            "span_map_out", str(mbench_root / "expansion_maps" / f"{nid}_expansion.json")
        )
        extractor_opts.setdefault(
            "precision_map_out",
            str(mbench_root / "precision_maps" / f"{nid}_precision.json"),
        )
        extractor_opts.setdefault(
            "diagnosis_map_out",
            str(mbench_root / "diagnosis_maps" / f"{nid}_diagnosis.json"),
        )
        extractor_opts.setdefault(
            "mbench_out",
            str(self.cfg.ddx_mbench_file),
        )
        extractor_opts.setdefault("mbench_task_prefix", "patient_ddx")

        ctx = self._construct_context(note)

        # guarantee prompt templates are present (always reload on resume or fresh)
        ctx = ensure_prompt_templates(ctx, force_reload=True)

        # Pre-extracted requirements are unlikely for raw notes; allow passthrough
        if "requirements" in note:
            ctx["requirements"] = note["requirements"]
            extractor_opts.setdefault("pre_extracted", True)

        # list_type is a freeform tag for downstream logging; "note" by default
        re_extractor = PatientStateExtractor(
            engine=self.engine,
            list_type="note",
            diagnosis_only=self.cfg.diagnosis_only,  # <<< NEW
            **extractor_opts,
        )
        ctx = re_extractor.forward(ctx)



        # --- NEW: explicit diagnoses extractor --------------------------------
        # 给 explicit 诊断单独一个 log 目录（可选）
        explicit_log_dir = self.cfg.req_mbench_root / "explicit_diagnose_maps" / f"{nid}"
        explicit_extractor = PatientStateExplicitDiagnoseExtractor(
            engine=self.engine,
            log_dir=str(explicit_log_dir),
        )
        ctx = explicit_extractor.forward(ctx)
        # 现在 ctx 里多了:
        #   ctx["explicit_diagnose"]
        #   ctx["explicit_diagnose_summary"]
        # ----------------------------------------------------------------------



        # Optional: attach entity lists if available
        entities_per_req: List[dict] = []
        for idx, _ in enumerate(ctx.get("requirements", [])):
            key = (nid, "note", idx)
            entities_per_req.append(
                self.entity_table.get(
                    key,
                    {
                        "entities_recall": [],
                        "entities_precision": [],
                        "entities_old": [],
                    },
                )
            )
        ctx["requirement_entities"] = entities_per_req
        return ctx

    def canonicalise(self, ctx: dict) -> dict:
        nid = ctx.get("note_id", "unknown")
        canon = EntityCanonicalizer(
            self.engine,
            report_dir="entity_reports",
            exact_jsonl="src/modules/EntityCanonicalizer/setup/data/exact_terms.jsonl",
            fuzzy_threshold=0.86,
            verbose=False,
            ddx_log_dir=f"mbench/diagnosis_mbench/ddx_runs/{nid}/ddx",
            diagnosis_only=self.cfg.diagnosis_only,  # <<< NEW
        )
        return canon(ctx)


    def extract_attributes(self, ctx: dict) -> dict:
        attr = AttributeExtractor(
            engine=self.engine,
            attr_proj_map_path="modules/AttributeExtractor/backup/domain_attribute_pairs.xlsx",
            attr_map_path="modules/AttributeExtractor/backup/attribute_value.xlsx",
            es_url="http://localhost:9200",
            top_k=3,
            verbose=False,
        )
        return attr(ctx)



    def _explicit_pipeline(self, ctx: dict) -> dict:
        """
        Run explicit-diagnose pipeline end-to-end:
          explicit_diagnose
            → DiagnosisCanonicalizer (SNOMED)
            → PatientCoder (diagnosis_only, explicit_diagnose.jsonl).
        """
        nid = ctx.get("note_id", "unknown")

        # --- 1) SNOMED canonicalization for explicit diagnoses -------------
        explicit_candidates = ctx.get("explicit_diagnose", []) or []
        if not explicit_candidates:
            # 没有 explicit 诊断就啥也不做
            return ctx

        # 备份 diff 线的 diagnose/canonical
        saved_candidates = ctx.get("diagnosis_candidates")
        saved_canonical = ctx.get("diagnosis_canonical")

        # 用 explicit_diagnose 当成 DiagnosisCanonicalizer 的输入
        ctx["diagnosis_candidates"] = explicit_candidates

        # diagnosis_only=True: 只跑 Dx → SNOMED，不跑 NER / vec / filter
        explicit_canon = EntityCanonicalizer(
            self.engine,
            report_dir="entity_reports",
            exact_jsonl="src/modules/EntityCanonicalizer/setup/data/exact_terms.jsonl",
            fuzzy_threshold=0.86,
            verbose=False,
            ddx_es_url="http://localhost:9200",
            ddx_index="snomed_vectors",
            ddx_log_dir=f"mbench/diagnosis_mbench/explicit_runs/{nid}/ddx",
            diagnosis_only=True,
        )
        ctx = explicit_canon.forward(ctx)
        # 保存 explicit 的 canonical 结果
        ctx["explicit_diagnosis_canonical"] = ctx.get("diagnosis_canonical", [])

        # 恢复 diff 线的 candidates / canonical（如果之前有）
        ctx["diagnosis_candidates"] = saved_candidates
        if saved_canonical is not None:
            ctx["diagnosis_canonical"] = saved_canonical

        # --- 2) SMT coding + export for explicit diagnoses -----------------
        explicit_canon_list = ctx.get("explicit_diagnosis_canonical", []) or []
        if not explicit_canon_list:
            return ctx

        # 备份当前 diagnosis_canonical（diff 线）
        saved_canonical2 = ctx.get("diagnosis_canonical")

        # 用 explicit 的 canonical 作为 coder 输入
        ctx["diagnosis_canonical"] = explicit_canon_list

        explicit_coder = PatientCoder(
            engine=self.engine,
            namer_log_dir=str(self.cfg.namer_log_dir()),
            diagnosis_coder_log_dir=str(self.cfg.diagnosis_coder_log_dir()),
            diagnosis_only=True,                      # 只跑诊断 coder
            write_diagnosis_jsonl=True,
            diagnosis_jsonl_name="explicit_diagnose.jsonl",  # <<< 输出文件名
        )
        ctx = explicit_coder.forward(ctx)

        # 保存 explicit 的变量声明
        ctx["explicit_diagnosis_variable_declarations"] = ctx.get(
            "diagnosis_variable_declarations", []
        )

        # 恢复 diff 线 canonical
        if saved_canonical2 is not None:
            ctx["diagnosis_canonical"] = saved_canonical2

        return ctx
    


    def program_smt(self, ctx: dict, *, mode: str = "namer") -> dict:
        # 1) 原来的 diff 诊断/变量 pipeline
        ctx = PatientCoder(
            engine=self.engine,
            namer_log_dir=str(self.cfg.namer_log_dir()),
            diagnosis_coder_log_dir=str(self.cfg.diagnosis_coder_log_dir()),
            diagnosis_only=self.cfg.diagnosis_only,
        ).forward(ctx)

        # 2) explicit pipeline: explicit_diagnose → canonicalizer → coder → explicit_diagnose.jsonl
        ctx = self._explicit_pipeline(ctx)

        return ctx



    # ──────────────────── main pipeline runner ────────────────────
    def run(
        self,
        note: dict,
        *,
        resume_from: Stage | None = None,
        stop_after: Stage | None = None,
        extractor_opts: dict | None = None,
    ) -> dict:

        nid = note.get("_id") or note.get("note_id")
        ckpt = self.cfg.ckpt_path

        # 0) ── load checkpoint if we're resuming ───────────────────────────
        ctx = None
        if resume_from is not None:
            loader = {
                Stage.PREPROC: load_preproc_ckpt,
                Stage.CANON: load_canon_ckpt,
                Stage.ATTR: load_attr_ckpt,
                Stage.PROGRAM: load_program_ckpt,
            }[resume_from]
            ctx = loader(self.engine, ckpt(resume_from.name.lower(), nid))

            # NEW ➜ ensure that **all** prompt templates are reloaded
            ctx = ensure_prompt_templates(ctx, force_reload=True)

        # Helper that commits a stage *only if* we're past it ---------------
        def save(stage: Stage, _ctx: dict):
            if stop_after is None or stage.value <= stop_after.value:
                {
                    Stage.PREPROC: save_preproc_ckpt,
                    Stage.CANON: save_canon_ckpt,
                    Stage.ATTR: save_attr_ckpt,
                    Stage.PROGRAM: save_program_ckpt,
                }[stage](_ctx, ckpt(stage.name.lower(), nid))

        # 1) ── PREPROC ─────────────────────────────────────────────────────
        if ctx is None:
            ctx = self.preprocess(note, extractor_opts)
            cursor = Stage.PREPROC
            save(cursor, ctx)
        else:
            cursor = resume_from  # type: ignore[assignment]

        if stop_after == Stage.PREPROC:
            return ctx

        # 2) ── CANON ───────────────────────────────────────────────────────
        if cursor.value < Stage.CANON.value:
            ctx = self.canonicalise(ctx)
            cursor = Stage.CANON
            save(cursor, ctx)
        if stop_after == Stage.CANON:
            return ctx

        # 3) ── ATTR ────────────────────────────────────────────────────────
        if cursor.value < Stage.ATTR.value:
            if not self.cfg.diagnosis_only:
                ctx = self.extract_attributes(ctx)
            # diagnosis_only 模式下跳过 ATTR 抽取，但仍然推进光标并保存当前 ctx
            cursor = Stage.ATTR
            save(cursor, ctx)
        if stop_after == Stage.ATTR:
            return ctx


        # 4) ── PROGRAM ─────────────────────────────────────────────────────
        if cursor.value < Stage.PROGRAM.value:
            ctx = self.program_smt(ctx)
            cursor = Stage.PROGRAM
            save(cursor, ctx)
        return ctx

    # ───────────────────── internal helpers ────────────────────
    def _construct_context(self, note: dict) -> dict:
        # minimal context: the raw note text becomes the working corpus
        return {
            "engine": self.engine,
            "contextual_text": note.get("text", ""),
            "requirement_text": note.get("text", ""),  # PatientStateExtractor expects this key
            "note_id": note.get("_id") or note.get("note_id"),
            "source": "patient_note",
        }


# ------------------------------------------------------------------------
#  prompt utilities (trimmed to drop Decomposer/HardSoft/VariableMiner)
# ------------------------------------------------------------------------

def ensure_prompt_templates(ctx: Dict[str, Any], *, force_reload: bool = False) -> Dict[str, Any]:
    """Attach every required *_prompt template to *ctx* if missing.

    For patient notes, we keep rudimentary extraction, verification,
    span expansion, and logical-precision rewriter prompts.
    Intentionally **omit** Decomposer, Hard/Soft Classifier, and Variable Miner prompts.
    """

    required = {
        # ⬇︎ PatientStateExtractor prompts (note domain)
        "PatientStateRudimentaryExtractor_prompt": "PatientStateExtractor/PatientStateRudimentaryExtractor.prompt",
        "PatientStateRudimentaryExtractorVerifier_prompt": "PatientStateExtractor/PatientStateRudimentaryExtractorVerifier.prompt",
        "PatientStateEntitySpanExpander_prompt": "PatientStateExtractor/PatientStateEntitySpanExpander.prompt",
        "PatientStateLogicalPrecisionRewriter_prompt": "PatientStateExtractor/PatientStateLogicalPrecisionRewriter.prompt",
        "PatientStateLogicalPrecisionRewriterVerifier_prompt": "PatientStateExtractor/PatientStateLogicalPrecisionRewriterVerifier.prompt",
        "PatientStateDifferentialDiagnoser_prompt": "PatientStateExtractor/PatientStateDifferentialDiagnoser.prompt",
        "PatientStateExplicitDiagnoseExtractor_prompt": "PatientStateExtractor/PatientStateExplicitDiagnoseExtractor.prompt",


        # ⬇︎ SMT-programmer prompts (unchanged)
        "PatientCanonicalVariableCoder_prompt": "PatientCoder/PatientCanonicalVariableCoder.prompt",
        "PatientCanonicalVariableOtherCandidatesCoder_prompt": "PatientCoder/PatientCanonicalVariableOtherCandidatesCoder.prompt",
        "PatientDemographicsVariableCoder_prompt": "PatientCoder/PatientDemographicsVariableCoder.prompt",
        "PatientSMTVariableCoderChecker_prompt": "PatientCoder/PatientSMTVariableCoderChecker.prompt",
        "PatientDiagnoseClassifier_prompt": "PatientCoder/PatientDiagnoseClassifier.prompt",

        # ⬇︎ Entity-canonicalizer prompts
        "LLMBasedMedicalEntityRecognizer_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityRecognizer.prompt",
        "LLMBasedMedicalEntityFilterLinker_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityFilterLinker.prompt",
        "LLMBasedMedicalEntityFilterVerifier_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityFilterVerifier.prompt",
        "LLMBasedMedicalEntityFilterArbiter_prompt": "EntityCanonicalizer/LLMBasedMedicalEntityFilterArbiter.prompt",

        "DiagnosisFilterLinker_prompt": "EntityCanonicalizer/DiagnosisFilterLinker.prompt",
        "DiagnosisFilterVerifier_prompt": "EntityCanonicalizer/DiagnosisFilterVerifier.prompt",

        # ⬇︎ Attribute-extractor prompts
        "AttributeExtractorQualifierIdentifier_prompt": "AttributeExtractor/AttributeExtractorQualifierIdentifier.prompt",
        "AttributeExtractorQualifierIdentifierVerifier_prompt": "AttributeExtractor/AttributeExtractorQualifierIdentifierVerifier.prompt",
        "AttributeExtractorAttributeTranslator_prompt": "AttributeExtractor/AttributeExtractorAttributeTranslator.prompt",
        "AttributeExtractorFreeAttributeTranslator_prompt": "AttributeExtractor/AttributeExtractorFreeAttributeTranslator.prompt",
        "AttributeExtractorCanonicalAttributeValueFilterCanon_prompt": "AttributeExtractor/AttributeExtractorCanonicalAttributeValueFilterCanon.prompt",
        "AttributeExtractorCanonicalAttributeValueFilterFree_prompt": "AttributeExtractor/AttributeExtractorCanonicalAttributeValueFilterFree.prompt",
        "AttributeExtractorCanonicalAttributeValueVerifier_prompt": "AttributeExtractor/AttributeExtractorCanonicalAttributeValueVerifier.prompt"
    }

    if not force_reload and all(k in ctx for k in required):
        return ctx  # old guard still useful for cold runs

    roots = [
        "prompts/patient_notes",           # preferred new root
        "prompts/clinical_trial",          # fallback while migrating
        "prompts/clinical_trial/inc",
        "prompts/clinical_trial/exc",
    ]

    cache: dict[str, str] = {}

    def read_from_roots(filename: str) -> str:
        if filename in cache:
            return cache[filename]
        for root in roots:
            path = pathlib.Path(root) / filename
            if path.exists():
                cache[filename] = path.read_text(encoding="utf-8")
                return cache[filename]
        raise FileNotFoundError(filename)

    ctx.update({k: read_from_roots(v) for k, v in required.items() if (force_reload or k not in ctx)})
    return ctx


# ------------------------------------------------------------------------
#  Multi-source note loader
# ------------------------------------------------------------------------

def _find_note_by_id(note_id: str, paths: List[pathlib.Path]) -> Tuple[dict, pathlib.Path]:
    """
    Search the provided JSONL paths in order and return (note_obj, source_path)
    for the first object whose `_id` or `note_id` matches `note_id`.
    """
    checked: List[str] = []
    for p in paths:
        if not p.exists():
            checked.append(f"{p} (missing)")
            continue
        checked.append(str(p))
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_id") == note_id or obj.get("note_id") == note_id:
                    return obj, p
    raise SystemExit(
        f"[not found] note_id='{note_id}' not present in any source.\nChecked:\n  - " +
        "\n  - ".join(checked)
    )


# ------------------------------------------------------------------------
#  CLI entry-point
# ------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the patient-notes pipeline (prompts reload on resume)")
    p.add_argument("note_id", help="Note identifier to run (e.g., trec-20211)")
    p.add_argument("--log-dir", default="run_logs")
    p.add_argument("--resume-from", choices=[s.name.lower() for s in Stage], default=None)
    p.add_argument("--stop-after", choices=[s.name.lower() for s in Stage], default=None)
    p.add_argument(
        "--diagnosis-only",
        action="store_true",
        help="Run diagnosis-only subpipeline (differential diagnoser + diagnosis canonicalizer + diagnosis coder)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config(log_dir=pathlib.Path(args.log_dir),
                 diagnosis_only=args.diagnosis_only,)

    # --- load corpus (search across all three sources) -----------------------
    sources = [
        pathlib.Path("../dataset/clinical_trial/sigir/queries.jsonl"),
        pathlib.Path("../dataset/clinical_trial/trec_2021/queries.jsonl"),
        pathlib.Path("../dataset/clinical_trial/trec_2022/queries.jsonl"),
    ]
    note, source_path = _find_note_by_id(args.note_id, sources)
    print(f"✓ Loaded note '{args.note_id}' from: {source_path}")

    # --- tee logging ---------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = cfg.log_dir / f"{args.note_id}_{ts}.log"
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PatientNotesPipeline(cfg)
    with Tee(log_path):
        ctx = pipeline.run(
            note,
            resume_from=Stage.from_str(args.resume_from),
            stop_after=Stage.from_str(args.stop_after),
        )
        print("✓ done – final keys:", list(ctx.keys()))


if __name__ == "__main__":
    main()
