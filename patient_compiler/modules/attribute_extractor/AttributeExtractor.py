from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import os, dspy, pathlib
from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Dict

from .stages import (
    AttributeExtractorAttributeBucketer,
    AttributeExtractorQualifierIdentifier,
    AttributeExtractorAttributeTranslator,
    AttributeExtractorFreeAttributeTranslator,
    # AttributeExtractorLLMAttributeExtractor,
    AttributeExtractorVectorAttributeValueSearch,
    AttributeExtractorCanonicalAttributeValueFilter,
    AttributeExtractorCanonicalAttributeValueVerifier
)
from .utils import (load_domain_attribute_map,_write_mbench_json, bundle_requirements_from_verified)


def consolidate_final_attributes(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    After all earlier stages have run, create a compact list of
    {attribute_type, qualifier_aspect, value, …} and attach it under each group.

    Adds:
    -------
    ctx["groups"][i]["final_attribute_values"]  # List[Dict[str, Any]]
    """
    for g in ctx.get("groups", []):
        finals: List[Dict[str, Any]] = []

        for m in g.get("members", []):
            requirement = m.get("requirement", "")
            trial_id    = m.get("trial")           # 可选，便于追踪
            entity      = m.get("entity", {})      # 若需要更多实体信息

            for p in m.get("attribute_value_pairs", []):
                # —— 取字段 ——
                attr_type   = p.get("attribute_type")        # None 允许保留
                qualifier_span   = p.get("qualifier_span")
                qualifier = p.get("qualifier")
                best_vals   = p.get("best_attribute_value")  # 期望是列表
                # —— 选 value ——
                value = p.get("attribute_value")

                finals.append(
                    {
                        "trial":          trial_id,
                        "requirement":    requirement,
                        "entity":         entity.get("surface_string"),  # 或 entity 整块
                        "qualifier_span": qualifier_span,
                        "qualifier": qualifier,
                        "attribute_type": attr_type,
                        "best_attribute_value": best_vals,
                        "original_attribute_value":          value,
                    }
                )

        g["final_attribute_values"] = finals

    return ctx

def consolidate_final_attributes_free(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    After all earlier stages have run, create a compact list of
    {attribute_type, qualifier_aspect, value, …} and attach it under each group.

    Adds:
    -------
    ctx["groups"][i]["final_attribute_values"]  # List[Dict[str, Any]]
    """
    for g in ctx.get("free_groups", []):
        finals: List[Dict[str, Any]] = []

        for m in g.get("members", []):
            requirement = m.get("requirement", "")
            trial_id    = m.get("trial")           # 可选，便于追踪
            entity      = m.get("entity", {})      # 若需要更多实体信息

            for p in m.get("attribute_value_pairs", []):
                # —— 取字段 ——
                attr_type   = p.get("attribute_type")        # None 允许保留
                qualifier_span   = p.get("qualifier_span")
                qualifier_description = p.get("qualifier_description")
                qualifier = p.get("qualifier")
                best_vals   = p.get("best_attribute_value")  # 期望是列表
                # —— 选 value ——
                value = p.get("attribute_value")

                finals.append(
                    {
                        "trial":          trial_id,
                        "requirement":    requirement,
                        "entity":         entity.get("surface_string"),  # 或 entity 整块
                        "qualifier_span": qualifier_span,
                        "qualifier": qualifier,
                        "qualifier_description": qualifier_description,
                        "attribute_type": attr_type,
                        "best_attribute_value": best_vals,
                        "original_attribute_value":          value,
                    }
                )

        g["final_attribute_values_free"] = finals

    return ctx

def _add_entities_without_qualifier(ctx: Dict, gid: int) -> Dict:
    """
    若某个 member 的 **attribute_value_pairs 为空**，
    就在 ctx["groups"][gid]["final_attribute_values"] 里
    补一条占位记录，避免后续 trial_groups 丢失该实体。
    """
    g    = ctx["groups"][gid]
    favs = g.setdefault("final_attribute_values", [])
    seen = {(it["trial"], it["entity"]) for it in favs}

    for m in g.get("members", []):
        trial = m["trial"]
        ent   = m["entity"]["surface_string"]

        # ----------- 关键判断：attribute_value_pairs 为空 ----------- #
        has_no_av_pairs = len(m.get("attribute_value_pairs", [])) == 0

        if has_no_av_pairs and (trial, ent) not in seen:
            placeholder = {
                "trial"              : trial,
                "requirement"        : m["requirement"],
                "entity"             : ent,
                # qualifier / attribute 信息均设为 None
                "qualifier_span"     : None,
                "qualifier"          : None,
                "attribute_type"     : None,
                "best_attribute_value": [],
                "attribute_value"    : None,
            }
            favs.append(placeholder)
            seen.add((trial, ent))

    return ctx


def _add_entities_without_qualifier_free(ctx: Dict, gid: int) -> Dict:
    """
    若某个 member 的 **attribute_value_pairs 为空**，
    就在 ctx["groups"][gid]["final_attribute_values"] 里
    补一条占位记录，避免后续 trial_groups 丢失该实体。
    """
    g    = ctx["free_groups"][gid]
    favs = g.setdefault("final_attribute_values_free", [])
    seen = {(it["trial"], it["entity"]) for it in favs}

    for m in g.get("members", []):
        trial = m["trial"]
        ent   = m["entity"]["surface_string"]

        # ----------- 关键判断：attribute_value_pairs 为空 ----------- #
        has_no_av_pairs = len(m.get("attribute_value_pairs", [])) == 0

        if has_no_av_pairs and (trial, ent) not in seen:
            placeholder = {
                "trial"              : trial,
                "requirement"        : m["requirement"],
                "entity"             : ent,
                # qualifier / attribute 信息均设为 None
                "qualifier_span"     : None,
                "qualifier"          : None,
                "qualifier_description": None,
                "attribute_type"     : None,
                "best_attribute_value": [],
                "attribute_value"    : None,
            }
            favs.append(placeholder)
            seen.add((trial, ent))

    return ctx





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
        attr_proj_map_path,
        return_attr_ids: bool = False,
        extractor_batch: int = 5,
        translator_batch: int = 1,
        es_url: str = "http://localhost:9200",
        attr_map_path: os.PathLike | str,
        top_k: int = 3,
        canon_batch: int = 2,
        verbose: bool = False,
        mbench_path: str | None = "mbench/attr_mbench/sub_mbench",
    ):
        super().__init__()
        self.verbose        = verbose
        self.mbench_path = mbench_path

        # --- static maps ------------------------------------------------
        domain_attr_map, attr_id2name = load_domain_attribute_map(
            attr_proj_map_path
        )
        self._domain_attr_map  = domain_attr_map
        self._attr_id2name     = attr_id2name

        # --- stages -----------------------------------------------------
        self.bucketer  = AttributeExtractorAttributeBucketer(
            domain_attribute_map=domain_attr_map,
            attribute_id_to_name=attr_id2name,
            attr_map_path=attr_map_path,
            return_ids=return_attr_ids,
            verbose=verbose,
        )
        self.qualifier_identifier = AttributeExtractorQualifierIdentifier(
            # prompt_file=qualifier_prompt_file,
            batch_size=extractor_batch,
            verbose=verbose,
        )

        # self.extractor = LLMAttributeExtractor(
        #     prompt_file=av_prompt_file,
        #     batch_size=extractor_batch,
        #     verbose=verbose,
        # )

        self.extractor_snomed = AttributeExtractorAttributeTranslator(
            # prompt_file=av_prompt_file,
            batch_size=translator_batch,
            engine = engine,
            verbose=verbose,
        )

        self.extractor_free = AttributeExtractorFreeAttributeTranslator(
            # prompt_file="<SATIR_ROOT>/src/modules/AttributeExtractor/backup/FreeAttributeValueExtraction.prompt",  # 与上面模板格式一致
            batch_size=extractor_batch,
            verbose=verbose,
        )

        self.vec       = AttributeExtractorVectorAttributeValueSearch(
            es_url=es_url,
            attr_map_path=attr_map_path,
            top_k=top_k,
            verbose=verbose,
        )
        self.canon     = AttributeExtractorCanonicalAttributeValueFilter(
            # prompt_file=canon_prompt_file,
            # free_prompt_file="<SATIR_ROOT>/src/modules/AttributeExtractor/backup/CanonicalAttributeValues_free.prompt",
            # batch_size=canon_batch,
            engine = engine,
            verbose=verbose,
        )

        self.verifier = AttributeExtractorCanonicalAttributeValueVerifier(
            engine=engine,
            max_retries=3,
        )

    # ------------------------------------------------------------------ #
    def _ck_path(self, stage: str, grp: int, tid) -> Path:
        """
        Compute the checkpoint path for *stage* and group *grp*.

        *grp* == ‑1 is reserved for the global Bucketer output.
        """
        # if self.mbench_path is None:
        #     # checkpointing disabled → return path that never exists
        #     return Path("__NO_CHECKPOINTS__")
        # return Path(self.mbench_path) / (
        #     f"{tid}/bucketer.json" if grp == -1 else f"{tid}/grp{grp:03d}_{stage}.json"
        # )

        root = Path(self.mbench_path) / tid
        if grp == -1:
            # 全局 artefact：{mbench_path}/{tid}/{stage}.json
            return root / f"{stage}.json"
        else:
            # group-specific：{mbench_path}/{tid}/grp{gid}_{stage}.json
            return root / f"grp{grp:03d}_{stage}.json"


    def _dump_mbench_attr(self, ctx: Dict[str, Any]) -> None:
        """
        将 ctx["groups"] 中的成员按 trial 号聚类，分别写到 mbench/attr/{trial}.json
        """
        out_root = pathlib.Path("../../mbench/attr_mbench/attr_logs")
        per_trial: Dict[str, list] = {}
        for grp in ctx.get("groups", []):
            # 每条 member 都带 trial
            for m in grp.get("members", []):
                # trial 字段形如 NCT02509286_req009
                tid = m.get("trial", "")
                # 记录：也把所在 group 的 allowed_attributes 带过去，便于调试
                per_trial.setdefault(tid, []).append({
                    "allowed_attributes": grp.get("allowed_attributes", []),
                    **m,
                })

        # 写文件
        for tid, items in per_trial.items():
            _write_mbench_json(f"mbench/attr_mbench/attr_logs/{tid}_inclusion_attr.json", items)



    # ------------------------------------------------------------------ #
    def forward(
        self,
        ctx: Dict[str, Any],
        index: int | None = None,
        *,
        resume: bool = False,
    ) -> Dict[str, Any]:  # type: ignore[override]
        """
        Parameters
        ----------
        ctx : dict
            Driver‑supplied DSPy context (will be mutated in‑place).
        index : int | None
            If given, run **only** that group and return the group‑level dict.
            Otherwise run end‑to‑end for every group and return the whole ctx.
        resume : bool
            If True (default) load cached JSON checkpoints when present.
        """

        # ----------------------------------------------------------------
        # Inject static lookup tables once (harmless on subsequent calls).
        # ----------------------------------------------------------------
        ctx.setdefault("domain_attribute_map", self._domain_attr_map)
        ctx.setdefault("attribute_id_to_name", self._attr_id2name)



        print("=== DEBUG ===")
        print("len(requirements):", len(ctx.get("requirements", [])))
        print("len(valid_entities_by_req):", len(ctx.get("valid_entities_by_req", {})))
        print("valid_entities_by_req",ctx.get("valid_entities_by_req",{}))
        print("keys(valid_entities_by_req[0]):",
            list(ctx.get("valid_entities_by_req", {}).get(0, {}).keys())[:5])
        print("ctx contains 'documents':", "documents" in ctx)
        print("len(documents):", len(ctx.get("documents", [])))
        print("================")

        # ----------------------------------------------------------------
        # Stage 0 – Bucketer (global, hence one checkpoint only)
        # ----------------------------------------------------------------
        
        
        # ----------------------------------------------------------------
        # Stage 0 – Bucketer (global, hence one checkpoint only)
        # ----------------------------------------------------------------
        
        
        ctx = self.bucketer(ctx)

        if "groups" not in ctx or not ctx["groups"]:
            raise RuntimeError("Bucketer produced no groups")

        # Decide which groups to process
        group_ids = [index] if index is not None else range(len(ctx["groups"]))

        if "trial_groups" not in ctx:
            ctx["trial_groups"] = defaultdict(list)



        # # with open("<SATIR_ROOT>/src/checkpoints/attr/NCT02509286_inclusion_attr.chkpt.json", "r", encoding="utf-8") as f:
        # #     ctx = json.load(f)   # data 是 dict 或 list，取决于文件内容
        # # group_ids = [index] if index is not None else range(len(ctx["groups"]))
        # # ----------------------------------------------------------------
        # # Loop over groups
        # # ----------------------------------------------------------------
        # for gid in group_ids:
        #     # -------- Stage 1 : LLM extraction --------

            
        #     ctx = self.qualifier_identifier(ctx, gid)
        #     patient_note_id = str(ctx["note_id"])
        #     # inc_exc  = str(ctx["inc_exc"])
        #     # stage    = "qualifier"  # 在不同阶段改成：extractor / vec / canon / verifier

        #     # d = Path("<SATIR_ROOT>/src/checkpoints/attr") / patient_note_id / stage
        #     # d.mkdir(parents=True, exist_ok=True)

        #     # fp = d / f"{time.strftime('%Y%m%d-%H%M%S')}_g{gid:04d}.json"
        #     # fp.write_text(json.dumps(ctx["groups"][gid], ensure_ascii=False, indent=2), encoding="utf-8")




        #     ctx = self.extractor_snomed(ctx, gid)
        #     patient_note_id = str(ctx["note_id"])
        #     # inc_exc  = str(ctx["inc_exc"])
        #     # stage    = "extractor"  # 在不同阶段改成：extractor / vec / canon / verifier

        #     # d = Path("<SATIR_ROOT>/src/checkpoints/attr") / patient_note_id / stage
        #     # d.mkdir(parents=True, exist_ok=True)

        #     # fp = d / f"{time.strftime('%Y%m%d-%H%M%S')}_g{gid:04d}.json"
        #     # fp.write_text(json.dumps(ctx["groups"][gid], ensure_ascii=False, indent=2), encoding="utf-8")




        #     ctx = self.vec(ctx, gid)
        #     patient_note_id = str(ctx["note_id"])
        #     # inc_exc  = str(ctx["inc_exc"])
        #     # stage    = "vec"  # 在不同阶段改成：extractor / vec / canon / verifier

        #     # d = Path("<SATIR_ROOT>/src/checkpoints/attr") / patient_note_id / stage
        #     # d.mkdir(parents=True, exist_ok=True)

        #     # fp = d / f"{time.strftime('%Y%m%d-%H%M%S')}_g{gid:04d}.json"
        #     # fp.write_text(json.dumps(ctx["groups"][gid], ensure_ascii=False, indent=2), encoding="utf-8")




        #     ctx = self.canon(ctx, gid)
        #     # save_attr_checkpoint(ctx, gid, "canon")
        #     patient_note_id = str(ctx["note_id"])
        #     # inc_exc  = str(ctx["inc_exc"])
        #     # stage    = "canon"  # 在不同阶段改成：extractor / vec / canon / verifier

        #     # d = Path("<SATIR_ROOT>/src/checkpoints/attr") / patient_note_id / stage
        #     # d.mkdir(parents=True, exist_ok=True)

        #     # fp = d / f"{time.strftime('%Y%m%d-%H%M%S')}_g{gid:04d}.json"
        #     # fp.write_text(json.dumps(ctx["groups"][gid], ensure_ascii=False, indent=2), encoding="utf-8")





        #     ctx = self.verifier(ctx, gid)
        #     patient_note_id = str(ctx["note_id"])
        #     # inc_exc  = str(ctx["inc_exc"])
        #     # stage    = "verifier"  # 在不同阶段改成：extractor / vec / canon / verifier

        #     # d = Path("<SATIR_ROOT>/src/checkpoints/attr") / patient_note_id / stage
        #     # d.mkdir(parents=True, exist_ok=True)

        #     # fp = d / f"{time.strftime('%Y%m%d-%H%M%S')}_g{gid:04d}.json"
        #     # fp.write_text(json.dumps(ctx["groups"][gid], ensure_ascii=False, indent=2), encoding="utf-8")





        ctx["requirement_bundles"] = bundle_requirements_from_verified(ctx)
        return ctx  

        # # for _k in ("documents", "attribute_id_to_name", "domain_attribute_map","span_mapping","precision_mapping","label_counts","constraint_counts","requirement_entities","llm_surface_entities_by_req","valid_entities_by_req","extraction_metrics","groups"):
        # #     ctx.pop(_k, None)
        # # # ----------------------------------------------------------------
        # # Return value mirrors original semantics
        # # ----------------------------------------------------------------
        # if index is not None:
        #     return ctx["groups"][index]
        
        # self._dump_mbench_attr(ctx)
        # if self.verbose:
        #     print("[AE] mbench/attr printed")
        # return ctx
