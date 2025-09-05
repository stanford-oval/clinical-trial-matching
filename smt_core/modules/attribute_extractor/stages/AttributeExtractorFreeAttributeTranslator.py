# modules/stages/free_attribute_Translator.py
from __future__ import annotations
import json, logging, re, time, hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import dspy
from ..utils import _chunks, _ChatEngine


def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

def _sha12(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]


class AttributeExtractorFreeAttributeTranslator(dspy.Module):
    """
    Stage 1-b (FREE): translate qualifiers → attribute_type/value *without* allowed_attributes.

    日志（与 AttributeExtractorAttributeTranslator 对齐）:
      mbench/attr_mbench/<Trial id>/<inc_exc>/g<gid>/2free_translater_prompts_outputs/
        - {ts}_g{gid}_b{batch}_free_translator_prompt.txt
        - {ts}_g{gid}_b{batch}_free_translator_raw_attempt{n}.txt
      mbench/attr_mbench/<Trial id>/<inc_exc>/g<gid>/2free_translater.json
        - **单个对象**：等于处理完成后的 context["groups"][gid]
    """

    def __init__(
        self,
        engine: _ChatEngine | None = None,
        *,
        model: str = "gpt-4o",
        batch_size: int = 1,
        max_retries: int = 2,
        temperature: float = 0.0,
        verbose: bool = False,
        mbench_path: str | Path = "mbench/attr_mbench/",
    ):
        super().__init__()
        self._default_engine = engine
        self.batch_size   = batch_size
        self.max_retries  = max_retries
        self.temperature  = temperature
        self.model        = model
        self.mbench_root  = Path(mbench_path)

        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # ----------------------------- helpers -------------------------------- #

    @staticmethod
    def _norm(s: Any) -> str:
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
    def _member_key(cls, m: Dict[str, Any]) -> Tuple[str, str, Optional[int], Optional[int]]:
        ent = m.get("entity", {}) or {}
        return (
            str(m.get("trial", "")),
            cls._norm(ent.get("surface_string")),
            cls._to_int(ent.get("start")),
            cls._to_int(ent.get("end")),
        )

    @classmethod
    def _qual_key(
        cls,
        note_id: str,
        ent_surf_norm: str,
        ent_start: Optional[int],
        ent_end: Optional[int],
        q: Dict[str, Any],
        with_offsets: bool = True,
    ) -> Tuple:
        q_span = cls._norm(q.get("qualifier_span", ""))
        if with_offsets:
            q_st = cls._to_int(q.get("start"))
            q_ed = cls._to_int(q.get("end"))
            return (note_id, ent_surf_norm, ent_start, ent_end, q_span, q_st, q_ed)
        return (note_id, ent_surf_norm, ent_start, ent_end, q_span)

    @staticmethod
    def _pick_trial_folder(ctx: Dict[str, Any], members: List[Dict[str, Any]]) -> str:
        t = str(ctx.get("trial_id") or "").strip()
        if t:
            return t
        for m in members or []:
            tr = str(m.get("trial") or "").strip()
            if tr:
                return tr.split("_", 1)[0]
        return "unknown_trial"

    def _prompts_dir_and_agg_file(self, trial_folder: str, gid: int, inc_exc: str) -> tuple[Path, Path]:
        base = self.mbench_root / _safe_id(trial_folder) / _safe_id(inc_exc) / f"g{gid:04d}"
        prompts = base / "2free_translater_prompts_outputs"
        prompts.mkdir(parents=True, exist_ok=True)
        agg = base / "2free_translater.json"
        base.mkdir(parents=True, exist_ok=True)
        return prompts, agg

    @staticmethod
    def _log_txt(dir_path: Path, filename: str, content: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(str(content), encoding="utf-8")

    def _build_prompt(self, template_prompt: str, batch_members: List[Dict[str, Any]]) -> str:
        """
        新 prompt 只有 {entity_qualifier} 占位；不传 allowed_attributes。
        传入的 member 需保留 entity 与其 qualifiers（但过滤掉“嵌在 entity surface 的 qualifier”）。
        """
        # 只保留需要的字段，避免把 pipeline 其它中间键传给 LLM
        light_members: List[Dict[str, Any]] = []
        for m in batch_members:
            ent = m.get("entity", {}) or {}
            quals = []
            for q in (m.get("all_qualifying_information_related_to_the_entity") or []):
                if str(q.get("qualifier_span_is_in_entity_surface_string", "")).upper() == "YES":
                    continue
                quals.append({
                    "entity": ent.get("surface_string", ""),
                    "start":  q.get("start"),
                    "end":    q.get("end"),
                    "qualifier_span": q.get("qualifier_span", ""),
                    # 可选把原句也带上（若已有）
                    "qualifier": q.get("qualifier", ""),
                })
            light_members.append({
                "trial": m.get("trial", ""),
                "requirement": m.get("requirement", ""),
                "entity": {
                    "surface_string": ent.get("surface_string", ""),
                    "preferred_term": ent.get("preferred_term", ""),
                    "fully_specified_name": ent.get("fully_specified_name", ""),
                    "type": ent.get("type", ""),
                    "definition": ent.get("definition", ""),
                    "conceptId": ent.get("conceptId", ""),
                    "start": ent.get("start"),
                    "end":   ent.get("end"),
                },
                "all_qualifying_information_related_to_the_entity": quals,
            })
        return template_prompt.format(
            entity_qualifier=json.dumps(light_members, ensure_ascii=False, indent=2),
        )

    def _infer_with_prompt_raw_logging(
        self,
        engine: _ChatEngine,
        prompt: str,
        log_dir: Path,
        gid: int,
        batch_idx: int,
    ) -> List[Dict[str, Any]]:
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_g{gid:04d}_b{batch_idx:03d}_free_translator"
        self._log_txt(log_dir, f"{base}_prompt.txt", prompt)

        messages = prompt
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            raw = engine(messages, temperature=self.temperature)[0]
            self._log_txt(log_dir, f"{base}_raw_attempt{attempt}.txt", raw)

            # 清理 markdown fence
            cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.S)
            # 容错：截去首尾非 JSON
            first, last = cleaned.find("["), cleaned.rfind("]")
            if first != -1 and last != -1 and last > first:
                cleaned = cleaned[first:last+1]
            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, list):
                    raise ValueError("Top-level JSON must be an array.")
                return parsed
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM reply not valid JSON: {exc}") from exc
                try:
                    messages.append({"role": "assistant", "content": raw[:1000]})
                    messages.append({
                        "role": "user",
                        "content": "Return a VALID JSON ARRAY only. No markdown, no preface, no extra keys."
                    })
                except Exception:
                    pass
        raise RuntimeError(f"LLM reply not valid JSON: {last_exc}")

    # ----------------------------- forward -------------------------------- #
    def forward(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:  # type: ignore[override]
        engine: _ChatEngine | None = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine provided.")

        group   = ctx["groups"][index]
        members = group.get("members", [])
        if not members:
            self.log.info("FreeGroup %d: no members – skip", index)
            return ctx

        # 新日志定位
        trial_folder = self._pick_trial_folder(ctx, members)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder, index, ctx["inc_exc"])

        # 仅对“有 qualifier（且不嵌在 entity surface 内）”的 members 调用
        eligible_members: List[Dict[str, Any]] = []
        for m in members:
            quals = [q for q in (m.get("all_qualifying_information_related_to_the_entity") or [])
                     if str(q.get("qualifier_span_is_in_entity_surface_string", "")).upper() != "YES"]
            if quals:
                eligible_members.append(m)

        batches = list(_chunks(eligible_members, self.batch_size))
        self.log.info(
            "FreeGroup %d – %d members, %d eligible, batch=%d (%d batches)",
            index, len(members), len(eligible_members), self.batch_size, len(batches)
        )

        # lookups
        key2member: Dict[Tuple[str, str, Optional[int], Optional[int]], Dict[str, Any]] = {
            self._member_key(m): m for m in members
        }
        qual_index_primary: Dict[Tuple, Dict[str, Any]] = {}
        qual_index_fallback: Dict[Tuple, Dict[str, Any]] = {}

        for m in members:
            note, ent_surf, st, ed = self._member_key(m)
            for q in (m.get("all_qualifying_information_related_to_the_entity") or []):
                pk = self._qual_key(note, ent_surf, st, ed, q, with_offsets=True)
                fk = self._qual_key(note, ent_surf, st, ed, q, with_offsets=False)
                qual_index_primary[pk] = q
                qual_index_fallback[fk] = q

        # 查询并写回（仅更新 ctx；不再聚合 LLM 输出）
        for b_idx, batch_members in enumerate(batches):
            if not batch_members:
                continue
            prompt_template = ctx["AttributeExtractorFreeAttributeTranslator_prompt"]
            prompt = self._build_prompt(prompt_template, batch_members)

            preds = self._infer_with_prompt_raw_logging(
                engine, prompt, log_dir=prompts_dir, gid=index, batch_idx=b_idx
            )

            for subj in preds:
                note_id  = str(subj.get("trial", ""))
                ent_obj  = subj.get("entity", {}) or {}
                ent_surf = self._norm(ent_obj.get("surface_string", ""))
                ent_start = self._to_int(ent_obj.get("start"))
                ent_end   = self._to_int(ent_obj.get("end"))

                m_ptr = key2member.get((note_id, ent_surf, ent_start, ent_end))
                if not m_ptr:
                    # 找不到 member：跳过这个 subject
                    continue

                pred_quals = subj.get("all_qualifying_information_related_to_the_entity", []) or []
                for pq in pred_quals:
                    # 取关键解释字段
                    interp = {
                        "attribute_type":        pq.get("attribute_type", ""),
                        "attribute_value":       pq.get("attribute_value", ""),
                        "qualifier_description": pq.get("qualifier_description", ""),
                        "rationale":             pq.get("rationale", ""),
                    }

                    # 精确匹配 qualifier（优先带 offset）
                    pk = self._qual_key(
                        note_id, ent_surf, ent_start, ent_end, pq, with_offsets=True
                    )
                    q_ptr = qual_index_primary.get(pk)
                    if not q_ptr:
                        fk = self._qual_key(
                            note_id, ent_surf, ent_start, ent_end, pq, with_offsets=False
                        )
                        q_ptr = qual_index_fallback.get(fk)
                    if not q_ptr:
                        continue

                    # 写回 free 解释（不污染有约束解释）
                    q_ptr.setdefault("attribute_interpretations_free", [])
                    q_ptr["attribute_interpretations_free"].append(interp)

        # 稳定排序
        group["members"] = sorted(members, key=lambda m: (m.get("trial",""), m.get("requirement","")))

        # === 将处理后的 group（即 context["groups"][gid]）写入 2free_translater.json（覆盖写入） ===
        try:
            agg_json_path.write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error("Failed to write 2free_translater.json: %s", e)

        return ctx