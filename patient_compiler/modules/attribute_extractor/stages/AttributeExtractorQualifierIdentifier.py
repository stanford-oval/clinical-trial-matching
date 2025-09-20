# modules/stages/attribute_qualifier_identifier.py
from __future__ import annotations
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import time

import dspy

from ..utils import _ChatEngine, _safe_id


class AttributeExtractorQualifierIdentifier(dspy.Module):
    """
    Stage 1 + verification

    • 输入：ctx["groups"][index]["members"]
    • 行为：对未完成的 member 批量抽取 → 对整条预测做一次 verifier → 仅保留 ALL_GOOD == "YES" 的 qualifiers 写回。
    • 对齐键：严格使用 (entity.surface_string, entity.start, entity.end) 三元组。

    • 新版日志（使用 mbench_path 根目录；默认 mbench/attr_mbench/）：
        mbench/attr_mbench/<Trial id>/g<gid>/1qualifier_prompts_outputs/
            - {ts}_g{gid}_b{batch}_extractor_prompt.txt
            - {ts}_g{gid}_b{batch}_extractor_raw_attempt{n}.txt
            - {ts}_g{gid}_verifier_s{start}_e{end}_prompt.txt
            - {ts}_g{gid}_verifier_s{start}_e{end}_raw_attempt{n}.txt
        mbench/attr_mbench/<Trial id>/g<gid>/1qualifier.json
            - **单个对象**：等于处理完成后的 `context["groups"][gid]`
    """

    def __init__(
        self,
        engine: _ChatEngine | None = None,
        *,
        model: str = "gpt-4o",
        batch_size: int = 10,
        max_retries: int = 2,
        temperature: float = 0.0,
        verbose: bool = False,
        mbench_path: str | os.PathLike = "mbench/attr_mbench/",
    ):
        super().__init__()
        self._default_engine = engine
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.temperature = temperature
        self.model = model
        self.mbench_root = Path(mbench_path)

        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # ----------------------------- builders ----------------------------- #
    def _build_extractor_prompt(self, prompt_text: str, batch: List[Dict[str, Any]]) -> str:
        return prompt_text.format(
            patientfact_entities=json.dumps(batch, ensure_ascii=False, indent=2),
        )

    def _build_verifier_prompt(self, template: str, pair: Dict[str, Any]) -> str:
        blob = json.dumps(pair, ensure_ascii=False, indent=2)
        for tok in ("#PATIENTFACT_ENTITY_QUALIFIER#",):
            if tok in template:
                return template.replace(tok, blob, 1)
        return f"{template.rstrip()}\n\n#PATIENTFACT_ENTITY_QUALIFIER#\n{blob}\n"

    # ------------------------------ 新路径工具 --------------------------- #
    @staticmethod
    def _pick_trial_folder(ctx: Dict[str, Any], members: List[Dict[str, Any]]) -> str:
        """优先 ctx['trial_id']；否则从 member['trial'] 取 NCT 前缀（下划线前）。"""
        t = str(ctx.get("note_id") or "").strip()
        if t:
            return t
        for m in members:
            tr = str(m.get("patient_note") or "").strip()
            if tr:
                return tr.split("_", 1)[0]
        return "unknown_trial"

    def _prompts_dir_and_agg_file(self, trial_folder: str, gid: int) -> tuple[Path, Path]:
        """返回 (prompts_dir, aggregate_json_path)"""
        base = self.mbench_root / _safe_id(trial_folder) / f"g{gid:04d}"
        prompts = base / "1qualifier_prompts_outputs"
        prompts.mkdir(parents=True, exist_ok=True)
        agg = base / "1qualifier.json"
        base.mkdir(parents=True, exist_ok=True)
        return prompts, agg

    @staticmethod
    def _write_txt(dir_path: Path, filename: str, content: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(str(content), encoding="utf-8")

    # ------------------------------ LLM IO ------------------------------ #
    def _strip_code_fence(self, s: str) -> str:
        return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.S)

    def _infer_with_logging(
        self,
        engine: _ChatEngine,
        prompt: str,
        *,
        ckpt_dir: Path,  # 指向 1qualifier_prompts_outputs/
        gid: int,
        batch_idx: int,
    ) -> List[Dict[str, Any]]:
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_g{gid:04d}_b{batch_idx:03d}_extractor"
        self._write_txt(ckpt_dir, f"{base}_prompt.txt", prompt)

        msg = prompt
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            raw = engine(msg, temperature=self.temperature)[0]
            self._write_txt(ckpt_dir, f"{base}_raw_attempt{attempt}.txt", raw)

            raw_clean = self._strip_code_fence(raw)
            try:
                parsed = json.loads(raw_clean)
                if not isinstance(parsed, list):
                    raise ValueError("top-level JSON must be an array")
                return parsed
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM reply not valid JSON: {exc}") from exc
                continue
        raise RuntimeError(f"LLM reply not valid JSON: {last_exc}")

    def _verify_with_logging(
        self,
        engine: _ChatEngine,
        pair: Dict[str, Any],
        template: str,
        *,
        ckpt_dir: Path,  # 指向 1qualifier_prompts_outputs/
        gid: int,
    ) -> Dict[str, Any]:
        ent = pair.get("entity", {}) or {}
        st  = ent.get("start", "na")
        ed  = ent.get("end", "na")
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_g{gid:04d}_verifier_s{st}_e{ed}"

        prompt = self._build_verifier_prompt(template, pair)
        self._write_txt(ckpt_dir, f"{base}_prompt.txt", prompt)

        msg = prompt
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            raw = engine(msg, temperature=self.temperature)[0]
            self._write_txt(ckpt_dir, f"{base}_raw_attempt{attempt}.txt", raw)

            raw_clean = self._strip_code_fence(raw)
            try:
                out = json.loads(raw_clean)
                if isinstance(out, dict):
                    return out
                if isinstance(out, list) and out:
                    return out[0]
                raise ValueError("verifier JSON must be a dict or non-empty array")
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise RuntimeError(f"Verifier reply not valid JSON: {exc}") from exc
                continue
        raise RuntimeError(f"Verifier reply not valid JSON: {last_exc}")

    # ------------------------------ matching utils ---------------------- #
    @staticmethod
    def _norm_text(s: Any) -> str:
        return str(s or "").strip().lower()

    @staticmethod
    def _to_int(x: Any) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            try:
                return int(str(x).strip())
            except Exception:
                return None

    @classmethod
    def _entity_key(cls, ent: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[int]]:
        surf = cls._norm_text(ent.get("surface_string"))
        st = cls._to_int(ent.get("start"))
        ed = cls._to_int(ent.get("end"))
        return (surf, st, ed)

    # ------------------------------ main ------------------------------- #
    def forward(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:  # type: ignore[override]
        engine: _ChatEngine | None = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine found. Provide one via constructor or ctx['engine'].")

        extractor_prompt_text: str = ctx["AttributeExtractorQualifierIdentifier_prompt"]
        verifier_template: str = ctx["AttributeExtractorQualifierIdentifierVerifier_prompt"]
        if not verifier_template:
            raise RuntimeError("Verifier prompt template not found in ctx.")

        group = ctx["groups"][index]
        members = group.get("members", [])
        if not members:
            self.log.warning("Group %d has no members – nothing to extract.", index)
            return ctx

        # 新日志路径
        trial_folder = self._pick_trial_folder(ctx, members)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder, index)

        # batching：仅处理还没抽到 qualifier 的 member
        pending_batches: List[List[Dict[str, Any]]] = []
        batch: List[Dict[str, Any]] = []
        for m in members:
            if m.get("all_qualifying_information_related_to_the_entity"):
                continue
            batch.append(m)
            if len(batch) >= self.batch_size:
                pending_batches.append(batch)
                batch = []
        if batch:
            pending_batches.append(batch)

        self.log.info(
            "QualifierExtractor(TRIAL) group %d – processing %d/%d members (batch=%d)",
            index,
            sum(len(b) for b in pending_batches),
            len(members),
            self.batch_size,
        )

        # 抽取 + 验证
        for b_idx, b in enumerate(pending_batches):
            ext_prompt = self._build_extractor_prompt(extractor_prompt_text, b)
            predictions = self._infer_with_logging(
                engine, ext_prompt, ckpt_dir=prompts_dir, gid=index, batch_idx=b_idx
            )

            # LUT：以 (surface,start,end) 对齐
            lut_by_triplet: Dict[Tuple[str, Optional[int], Optional[int]], Dict[str, Any]] = {}
            for p in predictions:
                ent = p.get("entity", {}) or {}
                key_triplet = self._entity_key(ent)
                lut_by_triplet[key_triplet] = p

            for m in b:
                m_ent = m.get("entity", {}) or {}
                m_key_triplet = self._entity_key(m_ent)

                pred = lut_by_triplet.get(m_key_triplet)
                if not pred:
                    m["all_qualifying_information_related_to_the_entity"] = []
                    continue

                pred_quals = pred.get("all_qualifying_information_related_to_the_entity") or []
                if not pred_quals:
                    m["all_qualifying_information_related_to_the_entity"] = []
                    continue

                # verifier（写 prompt/raw 到新目录）
                try:
                    verify_out = self._verify_with_logging(
                        engine, pred, verifier_template,
                        ckpt_dir=prompts_dir, gid=index
                    )
                except Exception as exc:
                    self.log.error(
                        "Verifier failed for entity=%s in trial=%s: %s",
                        m.get("entity", {}).get("surface_string"),
                        m.get("patient_note", "<unknown>"),
                        exc,
                    )
                    m["all_qualifying_information_related_to_the_entity"] = []
                    continue

                # 回写仅保留 ALL_GOOD == YES
                quals = verify_out.get("all_qualifying_information_related_to_the_entity") or []
                good = [q for q in quals if str(q.get("ALL_GOOD", "")).strip().upper() == "YES"]
                m["all_qualifying_information_related_to_the_entity"] = good

        # 排序保持稳定性
        group["members"] = sorted(members, key=lambda m: (m.get("patient_note", ""), m.get("requirement", "")))

        # === 将处理后的 group（即 context["groups"][gid]）写入 1qualifier.json（覆盖写入） ===
        try:
            agg_json_path.write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error("Failed to write 1qualifier.json: %s", e)

        return ctx
