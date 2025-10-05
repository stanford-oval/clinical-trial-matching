#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concept_first_filter.py — target_disease linker + verifier (no Stage 3)

适配新 ctx 结构：
  ctx["target_disease"] = [
    {
      "disease": "Menopause, Premature",
      "candidates": [ {... 完整候选 ...} ],
      "provenance": "..."  # 可选
    },
    ...
  ]

流程：
  • Stage 1 (LINKER): 对每个 disease 的候选，用 LLM 在 candidates 中选择 EXACTLY ONE（或 null）。
  • Stage 2 (VERIFY): 对 (disease, 选择的 concept) 做 KEEP/REJECT（传完整字段，不裁剪）。
产出：
  • ctx["linked_result"]             — linker 选中的（可能为 None）
  • ctx["valid_concepts_by_disease"]     — verifier KEEP 后的列表（本质上 0/1 个，保列表以兼容）
  • ctx["final_selected_concept_by_disease"]           — 每个 disease 的最终 best（若 KEEP）
"""

from __future__ import annotations
from typing import List, Dict, Any, Callable, Iterable, Optional, Tuple
import json, pathlib, logging, datetime as dt
import dspy

# ================= Config =================
LINKER_BATCH_SIZE  = 30
VERIFY_BATCH_SIZE  = 40
MAX_LLM_ATTEMPTS   = 3
_LOG_TO_STDOUT     = True

def _log(msg: str) -> None:
    (print if _LOG_TO_STDOUT else logging.getLogger("entity_filter").info)(msg)

def _write_txt(p: pathlib.Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _chunks(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

class LLMBasedMedicalEntityFilter(dspy.Module):
    def __init__(
        self,
        engine: Callable[[str], List[str]],
        *,
        linker_batch_size: int = LINKER_BATCH_SIZE,
        verify_batch_size: int = VERIFY_BATCH_SIZE,
        max_attempts: int = MAX_LLM_ATTEMPTS,
        verbose: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.verbose = verbose
        self.linker_batch_size = linker_batch_size
        self.verify_batch_size = verify_batch_size
        self.max_attempts = max_attempts

    # ---------- call LLM & log ----------
    def _call_and_log(self, prompt: str, expect_array: bool, prefix: pathlib.Path) -> Any:
        _write_txt(prefix.parent / f"{prefix.name}_prompt.txt", prompt)
        for attempt in range(self.max_attempts):
            out = self.engine(prompt)[0]
            if attempt == 0:
                _write_txt(prefix.parent / f"{prefix.name}_raw.txt", out)
            if expect_array:
                first, last = out.find("["), out.rfind("]")
                if first != -1 and last != -1 and last > first:
                    out = out[first:last+1]
            try:
                return json.loads(out.strip())
            except Exception:
                continue
        raise RuntimeError("LLM returned ill-formed JSON after retries")

    # ================= main =================
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        trial_id = ctx.get("trial_id", "trial")
        td = ctx.get("target_disease") or []
        if not isinstance(td, list) or not td:
            ctx.setdefault("linked_result", {})
            ctx.setdefault("valid_concepts_by_disease", {})
            ctx.setdefault("final_selected_concept_by_disease", {})
            return ctx

        # ===== Stage 1: LINKER（在 candidates 中单选；用 disease） =====
        def _mk_linker_payload(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            payload = []
            for e in items:
                if not isinstance(e, dict):
                    continue
                disease = str(e.get("disease") or "").strip()
                if not disease:
                    continue
                # 不筛选字段：把 candidates 的对象原样传给 LLM
                cands = e.get("candidates") or []
                if not isinstance(cands, list):
                    cands = []
                payload.append({"disease": disease, "candidates": cands})
            return payload

        linked_result: Dict[str, Optional[Dict[str, Any]]] = {}
        for b_no, batch in enumerate(_chunks(td, self.linker_batch_size), 1):
            payload = _mk_linker_payload(batch)
            if not payload:
                continue
            # 使用 ctx 中的 linker prompt；不需要 criterion，占位只替换 #CANDIDATES#
            linker_tpl = ctx.get("LLMBasedMedicalEntityFilterLinker_prompt", "")
            prompt = linker_tpl.replace("#CANDIDATES#", json.dumps(payload, ensure_ascii=False, indent=2))
            pfx = pathlib.Path("mbench/entity_mbench/entity_logs", f"{trial_id}_disease_stage1b{b_no:02d}")
            out = self._call_and_log(prompt, True, pfx)

            # 解析 linker 输出（保持 disease 语义）
            for item in out or []:
                disease = str(item.get("disease") or "").strip()
                concept = item.get("concept")
                if not disease:
                    continue
                if concept is not None and isinstance(concept, dict):
                    linked_result[disease] = concept  # 完整对象，不筛字段
                else:
                    linked_result[disease] = None

        ctx["linked_result"] = linked_result

        # ===== Stage 2: VERIFIER（对 linker 选择结果做 KEEP/REJECT；不筛字段） =====
        pairs_all: List[Dict[str, Any]] = []
        for e in td:
            if not isinstance(e, dict):
                continue
            disease = str(e.get("disease") or "").strip()
            if not disease:
                continue
            chosen = linked_result.get(disease)
            if not chosen:
                continue
            # 把完整 concept 字段与 disease 一并传入，不做裁剪
            pair = {"disease": disease}
            if isinstance(chosen, dict):
                pair.update({
                    "conceptId":           chosen.get("conceptId"),
                    "preferred_term":      chosen.get("preferred_term"),
                    "fully_specified_name":chosen.get("fully_specified_name"),
                    "type":                chosen.get("type") or chosen.get("top_type"),
                    "definition":          chosen.get("definition"),
                    "best_match_term":     chosen.get("best_match_term"),
                    # "match_reason":        chosen.get("match_reason"),
                })
            pairs_all.append(pair)

        kept_by_text: Dict[str, List[Dict[str, Any]]] = {}
        final_selected_concept_by_disease: Dict[str, Dict[str, Any]] = {}

        if pairs_all:
            for b_no, batch in enumerate(_chunks(pairs_all, self.verify_batch_size), 1):
                # 构造 verifier prompt。按照你的模板，包含 <criterion>；若没有则传空字符串。
                verifier_tpl = ctx.get("LLMBasedMedicalEntityFilterVerifier_prompt", "")
                criterion_text = str(ctx.get("criterion_text") or ctx.get("requirement_text") or "")
                prompt = (
                    verifier_tpl
                    .replace("#CRITERION#", criterion_text)
                    .replace("#PAIRS#", json.dumps(batch, ensure_ascii=False, indent=2))
                )
                pfx = pathlib.Path("mbench/entity_mbench/entity_logs", f"{trial_id}_disease_stage2b{b_no:02d}")
                verdicts = self._call_and_log(prompt, True, pfx)

                # KEEP 集合：基于 (disease, conceptId)
                keep_keys: set[Tuple[str, str]] = set()
                for v in verdicts or []:
                    if v.get("decision") == "KEEP":
                        t  = str(v.get("disease") or "").strip()
                        cid = str(v.get("conceptId") or "").strip()
                        if t and cid:
                            keep_keys.add((t, cid))

                # 收集 KEEP 的完整 pair（不筛字段）
                for p in batch:
                    t = str(p.get("disease") or "").strip()
                    cid = str(p.get("conceptId") or "").strip()
                    if (t, cid) in keep_keys:
                        kept_by_text.setdefault(t, []).append(p)

            # Linker 已“单选”，这里 KEEP 要么 0 要么 1；final_selected_concept_by_disease 填入该对象（原样）
            for t, lst in kept_by_text.items():
                if not lst:
                    continue
                final_selected_concept_by_disease[t] = lst[0]

        ctx["valid_concepts_by_disease"] = kept_by_text
        ctx["final_selected_concept_by_disease"] = final_selected_concept_by_disease

        # 汇总日志（可选）
        _write_json(
            pathlib.Path("mbench/entity_mbench/entity_logs", f"{trial_id}_disease_link_filter_summary.json"),
            {
                "generated": dt.datetime.now().isoformat(timespec="seconds"),
                "trial_id": trial_id,                          # ← 新增
                "contextual": ctx.get("contextual"),           # ← 新增（解析后的上下文）
                "linked_result": linked_result,
                "kept_counts": {k: len(v) for k, v in kept_by_text.items()},
                "final_selected_concept_by_disease": final_selected_concept_by_disease,
            },
        )
        return ctx
