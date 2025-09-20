# modules/stages/free_attribute_Translator.py
from __future__ import annotations
import copy, json, logging, re
from typing import Any, Dict, List

import dspy
from ..utils import _chunks, _ChatEngine


class AttributeExtractorFreeAttributeTranslator(dspy.Module):
    """
    Stage 1-b: LLM“自由”推断 qualifier→attribute_value_pairs
    - 与 AttributeTranslator 并行，但不传 allowed_attributes
    - 输出写入 ctx["free_groups"][gid]
    - prompt 模板仍需 {entity_qualifier} 与 {allowed_attributes} 占位
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        engine: _ChatEngine | None = None,
        *,
        model: str = "gpt-4o",
        batch_size: int = 10,
        max_retries: int = 2,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self._default_engine = engine
        self.batch_size   = batch_size
        self.max_retries  = max_retries
        self.temperature  = temperature
        self.model        = model

        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # ------------------------------------------------------------------ #
    #                         Helper functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_tasks(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = []
        for m in members:
            for q in m.get("all_qualifying_information_related_to_the_entity", []):
                if q.get("qualifier_span_is_in_entity_surface_string", "").upper() == "YES":
                    continue
                tasks.append(
                    {
                        "task_id": f"{m['trial']}|{m['entity']['surface_string'].lower()}|{q['qualifier_span'].lower()}",
                        "trial": m["trial"],
                        "requirement": m["requirement"],
                        "entity": m["entity"],
                        "qualifier": q,
                    }
                )
        return tasks

    def _build_prompt(self, batch: List[Dict[str, Any]],prompt_text: str,) -> str:
        """allowed_attributes 恒为空列表，让 LLM 随意发挥。"""
        return prompt_text.format(
            entity_qualifier=json.dumps(batch, ensure_ascii=False, indent=2),
        )

    def _infer(self, engine: _ChatEngine, prompt: str) -> List[Dict[str, Any]]:
        msg = prompt
        for attempt in range(1, self.max_retries + 1):
            raw = engine(msg, temperature=self.temperature)[0]
            raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.S)
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("top-level JSON must be an array")
                return parsed
            except Exception as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM reply not valid JSON: {exc}") from exc
                msg += [
                    {"role": "assistant", "content": raw[:1_000]},
                    {"role": "user", "content": "Return **valid JSON array** only."},
                ]

    # ------------------------------------------------------------------ #
    def forward(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:  # type: ignore[override]
        engine: _ChatEngine | None = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine provided.")

        # ① 取源 group，做深拷贝免得改动原数据
        src_group = ctx["groups"][index]
        group     = copy.deepcopy(src_group)
        members   = group.get("members", [])
        for m in members:
            m.pop("attribute_value_pairs", None)
        for noisy_key in ("final_attribute_values", "final_attribute_values_free"):
            group.pop(noisy_key, None)

        # ② 展平为 qualifier-level tasks
        tasks = self._make_tasks(members)
        if not tasks:
            self.log.info("FreeGroup %d 无可映射 qualifier – 跳过", index)
            return ctx

        # ③ 批处理调用 LLM
        batches = _chunks(tasks, self.batch_size)
        self.log.info("FreeGroup %d – %d qualifiers, batch=%d", index, len(tasks), self.batch_size)

        # LUT：task_id → (member, qualifier)
        key2ptr: Dict[str, Dict[str, Any]] = {}
        for m in members:
            for q in m.get("all_qualifying_information_related_to_the_entity", []):
                kid = f"{m['trial']}|{m['entity']['surface_string'].lower()}|{q['qualifier_span'].lower()}"
                key2ptr[kid] = {"member": m, "qualifier": q}

        for b in batches:
            prompt = self._build_prompt(b,ctx["AttributeExtractorFreeAttributeTranslator_prompt"])
            preds  = self._infer(engine, prompt)

            for p in preds:
                kid   = p.get("task_id")
                pairs = p.get("attribute_value_pairs", [])
                if not kid or not pairs:
                    continue
                ptr = key2ptr.get(kid)
                if not ptr:
                    continue
                ptr["member"].setdefault("attribute_value_pairs", []).extend(pairs)

        group["members"] = sorted(members, key=lambda m: (m["trial"], m["requirement"]))

        # ④ 存入 ctx["free_groups"]
        # ctx.setdefault("free_groups", {})[index] = group

        lst = ctx.setdefault("free_groups", [])
        # 保证长度 ≥ index+1
        while len(lst) <= index:
            lst.append({})
        lst[index] = group

        return ctx
