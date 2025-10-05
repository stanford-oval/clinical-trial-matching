from __future__ import annotations
from typing import Any, Dict, List, Set, Optional
import json, os

import dspy  # type: ignore

try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [typed-decls] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)

# 复用 translator 内的工具，避免逻辑漂移
from .SMTIncrementalTranslator import (
    _normalize_decl,
    _merged_new_declarations,
    _filter_into_reusables,
)

def _bucket(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"inc", "inclusion", "include", "in"}:  return "inclusion"
    if s in {"exc", "exclusion", "exclude", "ex"}:  return "exclusion"
    return s or "unknown"

def _collect_canonical_stems(context: Dict[str, Any]) -> Set[str]:
    stems: Set[str] = set()
    for c in (context.get("new_canonical_variable_declarations") or []):
        if isinstance(c, dict):
            nm = c.get("entity_variable_name")
            if isinstance(nm, str) and nm.strip():
                stems.add(nm.strip())
    return stems

def _collect_demographic_names(context: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    demo = context.get("new_age_sex_pregnancystatus_declarations") or []
    for obj in demo:
        if isinstance(obj, dict):
            nd = _normalize_decl(obj)
            vn = nd.get("variable_name")
            if isinstance(vn, str) and vn.strip():
                names.add(vn.strip())
    return names

def _classify_type(var_name: str, canon_stems: Set[str], demo_names: Set[str]) -> str:
    if var_name in demo_names:
        return "demographic"
    if "@@" in var_name:
        base = var_name.split("@@", 1)[0]
        return "canon_qualifier" if base in canon_stems else "noncanon"
    if var_name in canon_stems:
        return "canon"
    return "noncanon"

def _declared_ids_from_smt(lines: List[str]) -> Set[str]:
    ids: Set[str] = set()
    for ln in (lines or []):
        if not isinstance(ln, str):
            continue
        s = ln.lstrip()
        if s.startswith("(declare-const"):
            parts = s.split()
            if len(parts) > 1:
                ids.add(parts[1])
    return ids

class SMTIncrementalNewVariableFilter(dspy.Module):
    """
    Build NEW_VARIABLE_DECLARATIONS with an extra 'type' field, consistent with
    the translator's merge & reuse-filter rules.

    Inputs (context):
      - reusable_variables                (List[str|dict], optional)
      - smt_program_lines                 (List[str], optional)
      - new_age_sex_pregnancystatus_declarations
      - new_canonical_variable_declarations
      - new_remaining_variable_declarations / new_noncanonical_variable_declarations
      - trial_id, inc_exc, current_requirement_index (optional; for logging)

    Outputs (context):
      - new_variable_declarations_typed_all          : List[dict]
      - new_variable_declarations_typed_for_prompt   : List[dict]
      - NEW_VARIABLE_DECLARATIONS_TYPED_JSON         : str (pretty JSON for prompt)
      - reusable_variables                           : List[dict] (updated, normalized)
    """

    def __init__(self, *, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        idx: int = int(context.get("current_requirement_index", 0))
        side = context.get("inc_exc", "unknown")
        trial_id = context.get("trial_id", "unknown_trial")

        stage_dir = os.path.join(self.log_dir, str(trial_id), _bucket(side), f"req{idx:03d}")
        os.makedirs(stage_dir, exist_ok=True)

        # 1) 收集分类所需集合
        canon_stems = _collect_canonical_stems(context)
        demo_names  = _collect_demographic_names(context)

        # 2) 生成与 translator 一致的合并列表（不含类型）
        merged_newdecls: List[Dict[str, Any]] = _merged_new_declarations(context)

        # 3) 做“已声明/已可复用”过滤，得到应进入 prompt 的子集
        declared_ids = _declared_ids_from_smt(context.get("smt_program_lines", []))
        existing_reusables = context.get("reusable_variables", [])
        newdecls_for_prompt, reusables_for_prompt, moved = _filter_into_reusables(
            merged_newdecls, declared_ids, existing_reusables
        )

        if moved:
            _log("typed-decls", idx, f"moved {len(moved)} vars to reusables: {', '.join(sorted(moved))}")

        # 4) 添加 type 字段（对 all 与 for_prompt 分别做）
        def _with_type(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for obj in lst or []:
                if not isinstance(obj, dict):
                    continue
                vn = obj.get("variable_name")
                if not isinstance(vn, str) or not vn.strip():
                    continue
                t = _classify_type(vn.strip(), canon_stems, demo_names)
                o2 = dict(obj)
                o2["type"] = t
                out.append(o2)
            return out

        typed_all        = _with_type(merged_newdecls)
        typed_for_prompt = _with_type(newdecls_for_prompt)

        # 5) 写回 context
        context["new_variable_declarations_typed_all"] = typed_all
        context["new_variable_declarations_typed_for_prompt"] = typed_for_prompt
        context["NEW_VARIABLE_DECLARATIONS_TYPED_JSON"] = json.dumps(
            typed_for_prompt, indent=2, ensure_ascii=False
        )
        context["reusable_variables"] = reusables_for_prompt  # 与 translator 保持一致

        # 6) 可选落盘，便于对照/调试
        try:
            with open(os.path.join(stage_dir, "1typed_all.json"), "w", encoding="utf-8") as fh:
                json.dump(typed_all, fh, indent=2, ensure_ascii=False)
            with open(os.path.join(stage_dir, "1typed_for_prompt.json"), "w", encoding="utf-8") as fh:
                json.dump(typed_for_prompt, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            _log("typed-decls ⚠", idx, f"failed to write typed decl files: {e}")

        _log("typed-decls ✓", idx, f"typed_all={len(typed_all)}; for_prompt={len(typed_for_prompt)}")
        return context

