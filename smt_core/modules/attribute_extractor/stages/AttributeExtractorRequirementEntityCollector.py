# modules/stages/qualifier_input_builder.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging

# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────

_REQ_TEXT_KEYS = ("requirement", "text", "requirement_text", "sentence", "raw")

def _extract_requirement_text(req: Any) -> str:
    if isinstance(req, str):
        return req
    if isinstance(req, dict):
        for k in _REQ_TEXT_KEYS:
            if k in req and req[k] is not None:
                return str(req[k])
    return str(req)

def _extract_requirement_id(fallback_idx: int, req: Any) -> int:
    if isinstance(req, dict):
        for k in ("requirement_id", "id", "index", "idx"):
            if k in req:
                try:
                    return int(req[k])
                except Exception:
                    pass
    return int(fallback_idx)

def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None

def _sort_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 start 升序排序（缺失的排后）。
    """
    def _key(e: Dict[str, Any]):
        s = e.get("start")
        return (0, s) if isinstance(s, int) else (1, float("inf"))
    return sorted(entities, key=_key)

def _reorder_entity_keys(e: Dict[str, Any], entity_id: int) -> Dict[str, Any]:
    """
    统一实体字段的输出顺序（dict 保持插入序）。
    优先顺序：
      entity_id → extracted_span → start → end → preferred_term
      → fully_specified_name → type → 其它原字段（按原顺序）
    """
    # 需要强制排除的键（可按需扩展）
    drop_keys = {"conceptId", "concept_id", "preferred_term", "fully_specified_name"}

    priority = [
        "entity_id",
        "extracted_span",
        "start",
        "end",
        "type",
    ]
    out: Dict[str, Any] = {}
    for k in priority:
        if k == "entity_id":
            out[k] = entity_id
        elif k in e and k not in drop_keys:
            out[k] = e[k]
    # 把剩余字段按原来出现顺序追加，且排除 drop_keys
    for k, v in e.items():
        if k not in out and k not in drop_keys:
            out[k] = v
    return out

# ────────────────────────────────────────────────────────────────
#  Core
# ────────────────────────────────────────────────────────────────

def build_requirement_entities_for_qualifier_identification(
    ctx: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    仅依赖：
      - ctx["requirements"]: list[str | dict]
      - ctx["valid_entities_by_req"]: dict[int|str → dict 或 list 的实体集合]
    产出：
      - ctx["requirement_entities_for_qualifier_identification"]: List[dict]
        格式：
        [
          {
            "requirement_id": int,
            "requirement": str,
            "entities": [
              { "entity_id": 0, "surface_string": ..., "start": ..., "end": ..., ...other keys },
              ...
            ]
          },
          ...
        ]
    说明：
      - 会为每条实体重新分配连续的 entity_id（从 0 开始，覆盖原有同名键）。
      - 只读取 valid_entities_by_req，不会触碰其它 ctx 字段。
    """
    log = logger or logging.getLogger("RequirementEntityCollector")

    reqs = ctx.get("requirements")
    if not isinstance(reqs, list):
        raise RuntimeError("requirements must be a list[str|dict]")

    ver = ctx.get("valid_entities_by_req")
    if not isinstance(ver, dict):
        raise RuntimeError("valid_entities_by_req must be a dict")

    results: List[Dict[str, Any]] = []

    for i, req in enumerate(reqs):
        rid  = _extract_requirement_id(i, req)
        rtxt = _extract_requirement_text(req)

        # 取该 requirement 的实体容器（兼容 int/str key；兼容 dict 或 list）
        container = ver.get(i) or ver.get(str(i)) or {}

        raw_entities: List[Dict[str, Any]] = []
        if isinstance(container, dict):
            raw_entities = [v for v in container.values() if isinstance(v, dict)]
        elif isinstance(container, list):
            raw_entities = [e for e in container if isinstance(e, dict)]
        else:
            raw_entities = []

        entities = _sort_entities(raw_entities)
        for eid, e in enumerate(entities):
            entities[eid] = _reorder_entity_keys(e, eid)

        results.append(
            {
                "requirement_id": rid,
                "requirement": rtxt,
                "entities": entities,
            }
        )

    ctx["requirement_entities_for_qualifier_identification"] = results
    log.debug(
        "Built requirement_entities_for_qualifier_identification for %d requirements",
        len(results),
    )
    return results

class AttributeExtractorRequirementEntityCollector:
    def __init__(self, *, verbose: bool = False):
        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    def __call__(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        build_requirement_entities_for_qualifier_identification(ctx, logger=self.log)
        return ctx

    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self.__call__(ctx)
