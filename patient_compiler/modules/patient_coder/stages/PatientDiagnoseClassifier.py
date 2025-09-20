# modules/PatientDiagnoseClassifier.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re

import dspy  # 保持和其他模块一致的风格


_OPEN = "<diagnosis_list>"
_CLOSE = "</diagnosis_list>"


def _extract_block(raw: str) -> Optional[str]:
    """从 <diagnosis_list>...</diagnosis_list> 中抽出主体."""
    m = re.search(re.escape(_OPEN) + r"(.*?)" + re.escape(_CLOSE), raw, flags=re.S | re.I)
    return m.group(1).strip() if m else None


def _parse_json_block(raw: str) -> Optional[List[Dict[str, Any]]]:
    """抽出 diagnosis_list block 并解析成 list[dict]。宽松处理一些常见 JSON 小错误。"""
    body = _extract_block(raw)
    if not body:
        return None

    # normalize quotes
    body = (
        body.replace("\u201c", '"').replace("\u201d", '"')  # curly double
        .replace("\u2018", "'").replace("\u2019", "'")     # curly single
    )
    body = re.sub(r"[ \t]+\n", "\n", body)

    # tolerate trailing commas
    body = re.sub(r",\s*([}\]])", r"\1", body)

    stripped = body.strip()
    # 确保顶层是 array
    if not (stripped.startswith("[") and stripped.endswith("]")):
        if stripped.startswith("{") or re.search(r"^\s*\{", stripped, flags=re.M):
            body = "[\n" + body + "\n]"
            stripped = body.strip()

    try:
        arr = json.loads(stripped)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def _norm_diag_name(s: str) -> str:
    return (s or "").strip().lower()


class PatientDiagnoseClassifier(dspy.Module):
    """
    基于 LLM 的诊断强度判断模块。

    输入:
      - context["patient_note"] 或 context["requirement_text"] / context["contextual_text"]
      - context["diagnosis_canonical"]: list[dict]，每个 dict 至少有 diagnosis 或 mapping 里的术语
      - context["PatientDiagnoseClassifier_prompt"]: prompt 模板，包含
          #PATIENT_NOTE#
          #POSSIBLE_DIAGNOSES#

    行为:
      - 调用给定 prompt，对每个 diagnosis 生成:
          can_be_used_for_exclusion: bool
          reason: str
      - 将结果写回:
          row["can_be_used_for_exclusion"]
          row["exclusion_reason"]
        并存到 context["diagnosis_exclusion_judgments"] 里。
    """

    def __init__(
        self,
        engine,
        *,
        log_dir: Optional[str] = None,
    ):
        super().__init__()
        self.engine = engine
        self.log_dir = Path(log_dir) if log_dir else None

    # ---------------- internal helpers ----------------

    def _call_engine(self, prompt: str) -> str:
        """调用 engine，兼容 engine(prompt) 或 engine(prompt, **kwargs) 两种签名。"""
        try:
            # 尽量用确定性解码
            resp = self.engine(prompt, temperature=0.0, top_p=1.0)
        except TypeError:
            resp = self.engine(prompt)
        # AzureInferenceEngine 一般返回 list[str]
        if isinstance(resp, (list, tuple)):
            return str(resp[0])
        return str(resp)

    # ---------------- main API ----------------

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        diags: List[Dict[str, Any]] = context.get("diagnosis_canonical", []) or []
        if not diags:
            # 没有诊断就什么都不做
            context["diagnosis_exclusion_judgments"] = []
            return context

        # 1) 准备 patient note 文本
        note_txt = (
            context.get("patient_note")
            or context.get("requirement_text")
            or context.get("contextual_text")
            or ""
        )
        if not isinstance(note_txt, str):
            note_txt = str(note_txt)

        # 2) 准备 POSSIBLE_DIAGNOSES 列表 —— 这里只传诊断名称
        possible: List[Dict[str, Any]] = []
        names: List[str] = []
        for row in diags:
            mapping = row.get("mapping") or {}
            name = (
                row.get("diagnosis")
                or mapping.get("preferred_term")
                or mapping.get("fully_specified_name")
                or ""
            )
            name = str(name).strip()
            names.append(name)
            possible.append({"diagnosis": name})

        # 3) 从 context 里读取 prompt 模板
        tmpl = context.get("PatientDiagnoseClassifier_prompt")
        if not tmpl:
            raise KeyError("Missing 'PatientDiagnoseClassifier_prompt' in context")

        # 只把 {"diagnosis": name} 列表序列化进去
        possible_str = json.dumps(possible, ensure_ascii=False, indent=2)

        prompt = (
            tmpl.replace("#PATIENT_NOTE#", note_txt)
                .replace("#POSSIBLE_DIAGNOSES#", possible_str)
        )

        # 4) 调用 LLM
        raw = self._call_engine(prompt)

        # 4.1 写 debug 文件 prompt.txt / raw.txt -----------------------
        if self.log_dir:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                (self.log_dir / "raw.txt").write_text(raw, encoding="utf-8")
            except Exception:
                # debug 辅助，失败就悄悄忽略
                pass
        # ----------------------------------------------------------------

        # 然后再解析输出
        arr = _parse_json_block(raw) or []

        # 5) 对齐: 优先按 index 对齐；长度不一致时再 fallback 用诊断名字匹配
        judgments: List[Dict[str, Any]] = arr
        by_name: Dict[str, Dict[str, Any]] = {}
        for obj in arr:
            dn = _norm_diag_name(str(obj.get("diagnosis", "")))
            if dn:
                by_name[dn] = obj

        enriched_diags: List[Dict[str, Any]] = []
        for idx, row in enumerate(diags):
            # 先尝试 index 对齐
            obj: Optional[Dict[str, Any]] = arr[idx] if idx < len(arr) else None
            # 再尝试名字对齐
            if not obj or not obj.get("diagnosis"):
                mapping = row.get("mapping") or {}
                key = (
                    row.get("diagnosis")
                    or mapping.get("preferred_term")
                    or mapping.get("fully_specified_name")
                    or ""
                )
                norm = _norm_diag_name(str(key))
                if norm in by_name:
                    obj = by_name[norm]

            can_excl = False
            reason = ""
            if obj:
                can_excl = bool(obj.get("can_be_used_for_exclusion", False))
                reason = str(obj.get("reason", "") or "").strip()

            row = dict(row)  # 防御性 copy
            row["can_be_used_for_exclusion"] = can_excl
            # 单独起个 key，避免和别的字段冲突
            row["exclusion_reason"] = reason
            enriched_diags.append(row)

        context["diagnosis_canonical"] = enriched_diags
        context["diagnosis_exclusion_judgments"] = judgments

        # 可选结构化日志
        if self.log_dir:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "diagnosis_exclusion_judgments.json").write_text(
                    json.dumps(judgments, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return context
