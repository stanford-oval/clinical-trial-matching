#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# ───────── stdlib ─────────
import argparse
import contextlib
import datetime as dt
import json
import logging
import os
import pathlib
from pathlib import Path
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence
import pdb

# ───────── 3rd-party ─────────
import dspy  # type: ignore

# ───────── local modules ─────────
from smt_core.engine_factory import detect_engine_and_model
ENGINE_VERSION, MODEL_NAME = detect_engine_and_model()
if ENGINE_VERSION == "gpt-5":
    from smt_core.inference_engine_5 import AzureInferenceEngine
else:
    from smt_core.inference_engine import AzureInferenceEngine

print(f"[INFO] Using AzureInferenceEngine for {ENGINE_VERSION}")

# Core stages
from trial_compiler.disease_compiler.modules.DiseaseCanonicalizer import DiseaseCanonicalizer
from trial_compiler.disease_compiler.modules.DiseaseListPreprocessor.DiseaseListPreprocessor import (
    DiseaseListPreprocessor,
    DiseasePreprocConfig,
)
from trial_compiler.disease_compiler.modules.DiseaseListPreprocessor.stages.DiseaseListRevisor import DiseaseListRevisor, RevisorConfig
from trial_compiler.disease_compiler.modules.DiseaseListPreprocessor.stages.DiseaseLogicCapturer import DiseaseLogicCapturer, LogicConfig
from trial_compiler.disease_compiler.modules.DiseaseListPreprocessor.stages.DiseaseListExtractor import DiseaseListExtractor
from trial_compiler.disease_compiler.modules.DiseaseListPreprocessor.stages.DiseaseListEliminator import DiseaseListEliminator

from trial_compiler.disease_compiler.utils import dict_to_readable_string

dspy.settings.configure(lm_cache=None)
logging.getLogger("azure").setLevel(logging.WARNING)
_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ───────────────────── Prompts ─────────────────────
DISEASE_REQUIRED_PROMPTS: Dict[str, str] = {
    # Canonicalizer stages
    "LLMBasedMedicalEntityFilterLinker_prompt":   "DiseaseCanonicalizer/LLMBasedMedicalEntityFilterLinker.prompt",
    "LLMBasedMedicalEntityFilterVerifier_prompt": "DiseaseCanonicalizer/LLMBasedMedicalEntityFilterVerifier.prompt",
    # New Revisor + Logic (optional LLM modes)
    "DiseaseListRevisor_prompt":                  "DiseaseListPreprocessor/DiseaseListRevisor.prompt",
    "DiseaseLogicCapturer_prompt":                "DiseaseListPreprocessor/DiseaseLogicCapturer.prompt",

    "DiseaseListExtractor_prompt": "RequirementExtractor/RequirementDiseaseListExtractor.prompt",

    # disease list extractor and eliminator for subcohort-specific context (if applicable)
    "DiseaseListExtractor_cohort_prompt": "RequirementExtractor/RequirementDiseaseListExtractor_Subcohort.prompt",
    "DiseaseListEliminator_cohort_prompt": "RequirementExtractor/RequirementDiseaseListEliminator_Subcohort.prompt",

}

# ───────────────────── Config ─────────────────────

@dataclass
class Config:
    repo_root: pathlib.Path = pathlib.Path(__file__).resolve().parent
    data_root: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(os.getenv("TRIAL_DATA", "../dataset/clinical_trial"))
    )
    ckpt_dir: pathlib.Path = pathlib.Path("checkpoints_target_disease")
    log_dir: pathlib.Path = pathlib.Path("run_logs")
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    azure_endpoint: str = os.getenv("OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # prompt 源控制
    prompt_root: pathlib.Path = pathlib.Path("./prompts/clinical_trial")
    prompt_map_json: pathlib.Path | None = None
    load_prompts: bool = True  # 是否加载疾病 prompts

    # resources for preprocess/revisor
    alias_map: Optional[pathlib.Path] = None
    synonym_map: Optional[pathlib.Path] = None           # for preprocessor (expansion list)
    synonym_keep_map: Optional[pathlib.Path] = None      # for revisor (preferred term map)
    whitelist: Optional[pathlib.Path] = None
    blacklist: Optional[pathlib.Path] = None

    # feature toggles
    enable_llm_revision: bool = False
    enable_llm_logic: bool = False

    def prompt_sources(self) -> Dict[str, pathlib.Path]:
        mapping: Dict[str, pathlib.Path] = {
            k: (self.prompt_root / rel).resolve()
            for k, rel in DISEASE_REQUIRED_PROMPTS.items()
        }
        if self.prompt_map_json:
            raw = json.loads(pathlib.Path(self.prompt_map_json).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("prompt_map JSON must be an object of {key: path}.")
            for k, v in raw.items():
                if k in DISEASE_REQUIRED_PROMPTS:
                    p = pathlib.Path(v)
                    if not p.is_absolute():
                        p = (self.prompt_root / p)
                    mapping[k] = p.resolve()
        return mapping

    def __post_init__(self):
        if not self.prompt_root.is_absolute():
            self.prompt_root = (self.repo_root / self.prompt_root).resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ───────────────────── Tee logger ─────────────────────

class Tee(contextlib.AbstractContextManager):
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
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):  # noqa
        for s in self._streams:
            s.write(data)
    def flush(self):  # noqa
        for s in self._streams:
            s.flush()


# ───────────────────── Stages ─────────────────────

class Stage(Enum):
    PREPROC = auto()  # includes: Preprocessor → Revisor → Logic
    CANON = auto()

    @classmethod
    def from_str(cls, s: str | None) -> "Stage | None":
        if s is None:
            return None
        return cls[s.upper()]


# ───────────────────── Core Pipeline ─────────────────────

class TrialTargetDiseasePipeline:
    """PREPROC (Preprocessor → Revisor → Logic) + CANON（Canonicalizer）."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = AzureInferenceEngine(
            endpoint=cfg.azure_endpoint,
            api_key_env_var="OPENAI_API_KEY",
            model_name=cfg.model_name,
        )

    # ========= PREPROC =========
    def preprocess(
        self,
        trial: Dict[str, Any],
        cohort: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self._construct_context(trial, cohort=cohort)

        trial_id = ctx.get("trial_id") or "UNKNOWN"
        # 00: 刚构造完 context
        self._save_step_ckpt(trial_id, "00_context", ctx)

        extractor = DiseaseListExtractor(self.engine, verbose=False)
        ctx = extractor(ctx)
        self._save_step_ckpt(trial_id, "01_extractor", ctx)

        eliminator = DiseaseListEliminator(self.engine, verbose=False)
        ctx = eliminator(ctx)
        self._save_step_ckpt(trial_id, "02_eliminator", ctx)

        dlp_cfg = DiseasePreprocConfig(
            alias_map_path=self.cfg.alias_map,
            synonym_map_path=self.cfg.synonym_map,
            whitelist_path=self.cfg.whitelist,
            blacklist_path=self.cfg.blacklist,
            report_dir=Path("entity_reports"),
        )
        dlp = DiseaseListPreprocessor(dlp_cfg, verbose=False)
        ctx = dlp(ctx)
        self._save_step_ckpt(trial_id, "03_preprocessor", ctx)

        print(ctx.get("target_disease"))

        rev_cfg = RevisorConfig(
            alias_map_path=self.cfg.alias_map,
            synonym_keep_map_path=self.cfg.synonym_keep_map,
            whitelist_path=self.cfg.whitelist,
            blacklist_path=self.cfg.blacklist,
            report_dir=Path("entity_reports"),
            enable_llm_revision=self.cfg.enable_llm_revision,
        )
        revisor = DiseaseListRevisor(
            rev_cfg,
            engine=(self.engine if self.cfg.enable_llm_revision else None),
            verbose=False,
        )
        ctx = revisor(ctx)
        self._save_step_ckpt(trial_id, "04_revisor", ctx)

        logic_cfg = LogicConfig(
            report_dir=Path("entity_reports"),
            enable_llm_logic=self.cfg.enable_llm_logic,
        )
        logic = DiseaseLogicCapturer(
            logic_cfg,
            engine=(self.engine if self.cfg.enable_llm_logic else None),
            verbose=False,
        )
        ctx = logic(ctx)
        self._save_step_ckpt(trial_id, "05_logic", ctx)

        if self.cfg.load_prompts:
            self._ensure_disease_prompts(ctx)
            self._save_step_ckpt(trial_id, "06_prompts_loaded", ctx)

        return ctx


    # ========= CANON =========
    def canonicalise(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx.setdefault("canonical_diseases", [])
        trial_id = ctx.get("trial_id") or "UNKNOWN"

        canon = DiseaseCanonicalizer(
            self.engine,
            report_dir="entity_reports",
            verbose=False,
        )

        ctx = canon(ctx)
        ctx["canonical_diseases"] = ctx.get("final_selected_concept_by_disease", {})
        # 07: canonicalizer 之后
        self._save_step_ckpt(trial_id, "07_canon", ctx)


        return ctx

    # ========= RUN =========
    def run(
        self,
        trial: Dict[str, Any],
        *,
        resume_from: Stage | None = None,
        stop_after: Stage | None = None,
    ) -> Dict[str, Any]:

        base_trial_id = trial.get("_id") or trial.get("trial_id") or "UNKNOWN"

        # 先看是否有多 cohort 设置
        cohorts = self._load_enrollment_cohorts(base_trial_id)
        #print(cohorts)

        # ───────── 多 cohort 模式：对每个 cohort 分别跑 ─────────
        if len(cohorts) > 1:
            _LOG.info("[RUN] Detected %d enrollment_cohorts for %s; running all cohorts.",
                      len(cohorts), base_trial_id)
            results: Dict[str, Dict[str, Any]] = {}

            for coh in cohorts:
                # 为每个 cohort 构造/跑一套 pipeline
                ctx = None
                trial_id_eff = coh.get("trial_id_effective") or (
                    f"{base_trial_id}_{coh.get('id','')}"
                )

                # 这里为了简单，先不支持 resume_from；需要的话可以按 trial_id_eff 做 checkpoint
                ctx = self.preprocess(trial, cohort=coh)
                cursor = Stage.PREPROC
                self._save_ckpt(Stage.PREPROC, ctx, trial_id_eff)

                if stop_after == Stage.PREPROC:
                    results[trial_id_eff] = ctx
                    continue

                if cursor.value < Stage.CANON.value:
                    ctx = self.canonicalise(ctx)
                    cursor = Stage.CANON
                    self._save_ckpt(Stage.CANON, ctx, trial_id_eff)

                # 如果你有 _save_mbench，这里也可以调用：
                # self._save_mbench(trial_id_eff, ctx)

                results[trial_id_eff] = ctx

            return results

        # ───────── 单 cohort / 无 subcohort：保持原行为 ─────────
        trial_id = base_trial_id

        # ── try checkpoint resume
        ctx: Dict[str, Any] | None = None
        if resume_from is not None:
            try:
                ctx = self._load_ckpt(resume_from, trial_id)
                cursor = resume_from
            except FileNotFoundError:
                _LOG.warning("[CKPT] resume_from=%s but no checkpoint found; starting fresh.", resume_from.name)
                ctx = None

        # ── PREPROC
        if ctx is None:
            ctx = self.preprocess(trial, cohort=None)
            cursor = Stage.PREPROC
            self._save_ckpt(Stage.PREPROC, ctx, trial_id)

        if stop_after == Stage.PREPROC:
            return ctx

        # ── CANON
        if cursor.value < Stage.CANON.value:
            ctx = self.canonicalise(ctx)
            cursor = Stage.CANON
            self._save_ckpt(Stage.CANON, ctx, trial_id)

        # 如果你已经实现了 _save_mbench，这里可以：
        # self._save_mbench(trial_id, ctx)

        return ctx



    # ========= helpers =========

    def _inject_contextual_parsed(self, ctx: Dict[str, Any]) -> None:
        """Parse ctx['contextual_text'] and project keys back into ctx. 
        For trial with subcohorts, the shared contextual_text is parsed (same as single-cohort trial)"""
        raw = ctx.get("contextual_text")
        # print("Raw contextual_text:", raw)
        if not isinstance(raw, str) or not raw.strip():
            _LOG.warning("[PREPROC] contextual_text is empty or not a string; skip parsing.")
            return

        parsed: Dict[str, Any] | None = None

        # Try strict JSON first
        try:
            parsed = json.loads(raw)
        except Exception:
            # Try to handle quoted JSON-as-string
            try:
                parsed = json.loads(json.loads(raw) if raw.strip().startswith("{") is False else raw)
            except Exception:
                # Last resort: literal_eval for quasi-JSON
                try:
                    import ast
                    parsed = ast.literal_eval(raw)
                except Exception as e:
                    _LOG.warning("[PREPROC] failed to parse contextual_text: %s", e)
                    return

        if not isinstance(parsed, dict):
            _LOG.warning("[PREPROC] contextual_text parsed to non-dict; skip flatten.")
            return

        ctx["contextual"] = parsed

    def _construct_context(
        self,
        trial: Dict[str, Any],
        cohort: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        - cohort 为 None：单 cohort 或不走 subcohort，行为和原来一样。
        - cohort 不为 None：用该 cohort 的 trial_id_effective + contextual_text +
          inclusion_criteria / exclusion_criteria 构造上下文。
        """
        base_trial_id = trial.get("_id") or trial.get("trial_id")
        original_contextual_text = dict_to_readable_string(trial)

        if cohort is None:
            # 原始行为
            ctx = {
                "engine": self.engine,
                "contextual_text": original_contextual_text,
                "trial_id": base_trial_id,
            }
        else:
            # 针对某个具体 cohort 构建 context
            trial_id_eff = cohort.get("trial_id_effective") or base_trial_id
            coh_ctx_text = cohort.get("contextual_text") or dict_to_readable_string(trial)
            label = cohort.get("label") or ""
            coh_ctx_text_clean = self._strip_shared_criteria_block(coh_ctx_text)

            incl = cohort.get("inclusion_criteria") or ""
            excl = cohort.get("exclusion_criteria") or ""

            parts = [label, coh_ctx_text_clean.rstrip()]
            if incl:
                parts.append("Inclusion criteria (cohort-specific):\n" + incl)
            if excl:
                parts.append("Exclusion criteria (cohort-specific):\n" + excl)

            contextual_text = "\n\n".join(p for p in parts if p)

            ctx = {
                "engine": self.engine,
                "contextual_text": original_contextual_text,    # keep the original full contextual text for parsing metadata (used in _inject_contextual_parsed)
                "cohort_contextual_text": contextual_text,      # this cohort_contextual_text is used during disease item extraction
                "trial_id": trial_id_eff,
                "cohort_id": cohort.get("id"),
                "cohort_label": cohort.get("label"),
            }

        self._inject_contextual_parsed(ctx)
        return ctx


    def _ensure_disease_prompts(self, ctx: Dict[str, Any]) -> None:
        for key, path in self.cfg.prompt_sources().items():
            try:
                ctx[key] = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                _LOG.warning("[PREPROC] prompt not found for key=%s path=%s", key, path)

    @staticmethod
    def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
        seen, out = set(), []
        for it in items:
            key = it.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    @staticmethod
    def _list_repr_single_quotes(lst: List[str]) -> str:
        return "[" + ", ".join(f"'{x}'" for x in lst) + "]"

    def _ckpt_path(self, stage: Stage, trial_id: str) -> Path:
        ckpt_root: Path = Path(self.cfg.ckpt_dir)
        stage_name = stage.name.lower()
        stage_dir = ckpt_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"{trial_id}_{stage_name}.chkpt.json"

    def _save_ckpt(self, stage: Stage, ctx: Dict[str, Any], trial_id: str) -> None:
        path = self._ckpt_path(stage, trial_id)
        to_save = dict(ctx)
        to_save.pop("engine", None)  # drop non-serializable
        path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        _LOG.info("[CKPT] saved %s", path)

    def _load_ckpt(self, stage: Stage, trial_id: str) -> Dict[str, Any]:
        path = self._ckpt_path(stage, trial_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["engine"] = self.engine
        _LOG.info("[CKPT] loaded %s", path)
        return data

    # ========= subcohort helpers =========

    def _load_enrollment_cohorts(self, base_trial_id: str) -> List[Dict[str, Any]]:
        """
        从 <SATIR_ROOT>/subcohort_results/<NCT>.json 读取 enrollment_cohorts。
        如果文件不存在或格式不对，返回 []。
        """
        root = Path("../canonical_subcohort_results")
        path = root / f"{base_trial_id}.json"
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ecs = data.get("extracted").get("preprocessor_normalized").get("enrollment_cohorts") or []
            if isinstance(ecs, list):
                return ecs
            return []
        except Exception as e:
            _LOG.warning("[SUBCOHORT] Failed to load %s: %s", path, e)
            return []

    @staticmethod
    def _strip_shared_criteria_block(text: str) -> str:
        """
        从 subcohort 的 contextual_text 里删掉
        'Shared inclusion criteria ...' 到文本结尾（通常包括 Shared exclusion）。
        """
        if not isinstance(text, str):
            text = str(text)

        marker = "Shared inclusion criteria"
        idx = text.find(marker)
        if idx == -1:
            return text
        return text[:idx].rstrip()

    # ========= extra step checkpoints & mbench =========

    def _step_ckpt_path(self, trial_id: str, step_name: str) -> Path:
        """
        Per-step checkpoint path under ./checkpoints.
        e.g. ./checkpoints/NCT01234567_02_extractor.json
        """
        root = Path("checkpoints")
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{trial_id}_{step_name}.json"

    def _save_step_ckpt(self, trial_id: str, step_name: str, ctx: Dict[str, Any]) -> None:
        """
        Save full ctx (minus engine) after each step.
        """
        path = self._step_ckpt_path(trial_id, step_name)
        to_save = dict(ctx)
        to_save.pop("engine", None)  # non-serializable
        path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        _LOG.info("[STEP-CKPT] saved %s", path)

    def _save_mbench(self, trial_id: str, ctx: Dict[str, Any]) -> None:
        """
        Dump all recorded LLM calls in ctx['mbench_llm_calls'] to ./mbench.
        """
        calls = ctx.get("mbench_llm_calls") or []
        if not calls:
            return
        root = Path("mbench")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{trial_id}_llm_calls.json"
        path.write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8")
        _LOG.info("[MBENCH] saved %s", path)



# ───────────────────── CLI ─────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TrialTargetDiseasePipeline (PREPROC → CANON).")
    p.add_argument("trial_id", help="NCT identifier to run")
    p.add_argument("--side", choices=["inclusion", "exclusion", "both"], default="both")

    # Paths & prompts
    p.add_argument("--log-dir", default="run_logs")
    p.add_argument("--prompt-root", default="./prompts/clinical_trial")
    p.add_argument("--prompt-map", default=None)
    p.add_argument("--no-prompts", action="store_true", help="Do not load prompts into ctx.")

    # Resource files for Preprocessor/Revisor
    p.add_argument("--alias-map", default=None, help="JSON mapping alias->preferred")
    p.add_argument("--synonym-map", default=None, help="JSON mapping base->[synonyms] (preprocessor expansion)")
    p.add_argument("--synonym-keep-map", default=None, help="JSON mapping variant->preferred (revisor keep choice)")
    p.add_argument("--whitelist", default=None, help="Text file, one term per line")
    p.add_argument("--blacklist", default=None, help="Text file, one term per line")

    # Feature toggles
    p.add_argument("--enable-llm-revision", action="store_true", help="Use LLM pass for revising disease strings.")
    p.add_argument("--enable-llm-logic", action="store_true", help="Use LLM to capture logic across disease items.")

    # Flow control
    p.add_argument(
        "--stop-after",
        choices=["preproc", "canon"],
        default=None,
        help="Stop the pipeline after the given stage."
    )
    p.add_argument(
        "--resume-from",
        choices=["preproc", "canon"],
        default=None,
        help="Resume the pipeline from a saved checkpoint of the given stage."
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config(
        log_dir=pathlib.Path(args.log_dir),
        prompt_root=pathlib.Path(args.prompt_root),
        prompt_map_json=pathlib.Path(args.prompt_map) if args.prompt_map else None,
        load_prompts=(not args.no_prompts),
        alias_map=(pathlib.Path(args.alias_map) if args.alias_map else None),
        synonym_map=(pathlib.Path(args.synonym_map) if args.synonym_map else None),
        synonym_keep_map=(pathlib.Path(args.synonym_keep_map) if args.synonym_keep_map else None),
        whitelist=(pathlib.Path(args.whitelist) if args.whitelist else None),
        blacklist=(pathlib.Path(args.blacklist) if args.blacklist else None),
        enable_llm_revision=bool(args.enable_llm_revision),
        enable_llm_logic=bool(args.enable_llm_logic),
    )

    # 从 corpus 简单取出该 trial（与原脚本相同的默认组织）
    corpus_path = cfg.data_root / "sigir/corpus.jsonl"  #"trec_2022_subset_padded/corpus.jsonl"
    with corpus_path.open(encoding="utf-8") as fh:
        trial = next(json.loads(l) for l in fh if f'"_id": "{args.trial_id}"' in l)

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = cfg.log_dir / f"{args.trial_id}_target_disease_{ts}.log"

    pipeline = TrialTargetDiseasePipeline(cfg)

    with Tee(log_path):
        result = pipeline.run(
            trial,
            resume_from=Stage.from_str(args.resume_from),
            stop_after=Stage.from_str(args.stop_after)
        )

        # 单 ctx 情况：保持原行为
        if isinstance(result, dict) and "target_disease" in result:
            ctx = result
            td = ctx.get("target_disease") or []
            logic_expr = (ctx.get("disease_logic") or {}).get("expr", "")
            print("\n— Summary —")
            print("trial_id:", ctx.get("trial_id"))
            print("target_disease:", [e.get("disease") for e in td if isinstance(e, dict)])
            if logic_expr:
                print("logic:", logic_expr)
            kept = ctx.get("final_selected_concept_by_disease") or {}
            if kept:
                print("final concepts:", kept)
        else:
            # 多 cohort：result 预期是 {trial_id_eff -> ctx}
            print("\n— Multi-cohort Summary —")
            for eff_id, ctx in result.items():
                td = ctx.get("target_disease") or []
                logic_expr = (ctx.get("disease_logic") or {}).get("expr", "")
                print(f"\n[cohort trial_id_effective={eff_id}]")
                print("  target_disease:", [e.get("disease") for e in td if isinstance(e, dict)])
                if logic_expr:
                    print("  logic:", logic_expr)
                kept = ctx.get("final_selected_concept_by_disease") or {}
                if kept:
                    print("  final concepts:", kept)

        print("\n✓ done.")



if __name__ == "__main__":
    main()
