# modules/stages/attribute_translator.py
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


class AttributeExtractorAttributeTranslator(dspy.Module):
    """
    Stage: translate qualifiers → attribute_type-attribute_value pairs

    新日志（使用 mbench_path 根目录；默认 mbench/attr_mbench/）：
      mbench/attr_mbench/<Trial id>/g<gid>/2translater_prompts_outputs/
        - {ts}_g{gid}_b{batch}_translator_prompt.txt
        - {ts}_g{gid}_b{batch}_translator_raw_attempt{n}.txt
      mbench/attr_mbench/<Trial id>/g<gid>/2translater.json
        - **单个对象**：等于处理完成后的 `context["groups"][gid]`
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
    def _to_allowed_dicts(raw: Any) -> List[Dict[str, str]]:
        if not raw:
            return []
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return [
                {"AttributeType": a.get("AttributeType", ""), "definition": a.get("definition", "")}
                for a in raw
            ]
        return [{"AttributeType": str(name), "definition": ""} for name in (raw or [])]

    @staticmethod
    def _norm_attr_key(name: str) -> str:
        s = (name or "").strip().casefold()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"\s*\(attribute\)\s*$", "", s)
        return s

    def _build_attr_def_lut(self, allowed: List[Dict[str, str]]) -> Dict[str, str]:
        lut: Dict[str, str] = {}
        for a in allowed:
            name = a.get("AttributeType", "") or ""
            definition = a.get("definition", "") or ""
            if not name:
                continue
            k1 = self._norm_attr_key(name)
            k2 = self._norm_attr_key(re.sub(r"\s*\(attribute\)\s*$", "", name, flags=re.I))
            for k in {k1, k2}:
                lut.setdefault(k, definition)
        return lut

    @staticmethod
    def _enrich_attr_interpretations(
        interps: List[Dict[str, Any]],
        def_lut: Dict[str, str],
        norm_fn,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in interps or []:
            atype = it.get("attribute_type")
            it["attribute_definition"] = def_lut.get(norm_fn(atype), "") if atype else ""
            out.append(it)
        return out

    def _build_prompt(
        self,
        template_prompt: str,
        batch_members: List[Dict[str, Any]],
        allowed: List[Dict[str, str]],
    ) -> str:
        return template_prompt.format(
            entity_qualifier=json.dumps(batch_members, ensure_ascii=False, indent=2),
            allowed_attributes=json.dumps(allowed, ensure_ascii=False, indent=2),
        )

    # ---------- 新的日志路径/写入工具 ---------- #
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
        prompts = base / "2translater_prompts_outputs"
        prompts.mkdir(parents=True, exist_ok=True)
        agg = base / "2translater.json"
        base.mkdir(parents=True, exist_ok=True)
        return prompts, agg

    @staticmethod
    def _log_txt(dir_path: Path, filename: str, content: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(str(content), encoding="utf-8")

    def _infer_with_prompt_raw_logging(
        self,
        engine: _ChatEngine,
        prompt: str,
        log_dir: Path,
        gid: int,
        batch_idx: int,
    ) -> List[Dict[str, Any]]:
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_g{gid:04d}_b{batch_idx:03d}_translator"
        self._log_txt(log_dir, f"{base}_prompt.txt", prompt)

        messages = prompt
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            raw = engine(messages, temperature=self.temperature)[0]
            self._log_txt(log_dir, f"{base}_raw_attempt{attempt}.txt", raw)

            cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.S)
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

    # ------------------------- match helpers（新增三元组键） ------------------------- #
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

    # ----------------------------- forward -------------------------------- #
    def forward(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:  # type: ignore[override]
        engine: _ChatEngine | None = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine provided.")

        group   = ctx["groups"][index]
        allowed = self._to_allowed_dicts(group.get("allowed_attributes", []))
        members = group.get("members", [])
        if not members:
            self.log.info("Group %d: no members – skip", index)
            return ctx

        # 新日志定位
        trial_folder = self._pick_trial_folder(ctx, members)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder, index, ctx["inc_exc"])

        # 仅对“有 qualifier 的 members”调用翻译器
        eligible_members = [m for m in members if (m.get("all_qualifying_information_related_to_the_entity") or [])]

        batches = list(_chunks(eligible_members, self.batch_size))
        self.log.info(
            "Group %d – %d members, %d eligible (have qualifiers), batch=%d (%d batches)",
            index, len(members), len(eligible_members), self.batch_size, len(batches)
        )

        # lookups
        key2member: Dict[Tuple[str, str, Optional[int], Optional[int]], Dict[str, Any]] = {
            self._member_key(m): m for m in members
        }
        qual_index: Dict[Tuple[str, str, Optional[int], Optional[int], str], Dict[str, Any]] = {}
        for m in members:
            note, ent, st, ed = self._member_key(m)
            for q in m.get("all_qualifying_information_related_to_the_entity", []) or []:
                qkey = (note, ent, st, ed, self._norm(q.get("qualifier_span", "")))
                qual_index[qkey] = q

        # Build definition LUT once per group
        def_lut = self._build_attr_def_lut(allowed)

        # 查询并写回（仅更新 ctx；不再聚合 LLM 输出）
        for b_idx, batch_members in enumerate(batches):
            if not batch_members:
                continue
            prompt_template = ctx["AttributeExtractorAttributeTranslator_prompt"]
            prompt = self._build_prompt(prompt_template, batch_members, allowed)

            preds = self._infer_with_prompt_raw_logging(
                engine, prompt, log_dir=prompts_dir, gid=index, batch_idx=b_idx
            )

            for subj in preds:
                note_id  = str(subj.get("trial", ""))
                ent_obj = subj.get("entity", {}) or {}
                ent_surf_raw = ent_obj.get("surface_string", "")
                ent_surf = self._norm(ent_surf_raw)
                ent_start = self._to_int(ent_obj.get("start"))
                ent_end   = self._to_int(ent_obj.get("end"))

                # enrich interpretations（先富化，再写回 ctx）
                pred_quals = subj.get("all_qualifying_information_related_to_the_entity", []) or []
                for pq in pred_quals:
                    interps = pq.get("attribute_interpretations", []) or []
                    pq["attribute_interpretations"] = self._enrich_attr_interpretations(
                        interps, def_lut, self._norm_attr_key
                    )

                # 回写 ctx（按三元组 + qualifier_span）
                m_ptr = key2member.get((note_id, ent_surf, ent_start, ent_end))
                if not m_ptr:
                    continue
                for pq in pred_quals:
                    q_span = self._norm(pq.get("qualifier_span", ""))
                    q_ptr = qual_index.get((note_id, ent_surf, ent_start, ent_end, q_span))
                    if not q_ptr:
                        continue
                    q_ptr["attribute_interpretations"] = pq.get("attribute_interpretations", []) or []

        # 稳定排序
        group["members"] = sorted(members, key=lambda m: (m.get("trial",""), m.get("requirement","")))

        # === 将处理后的 group（即 context["groups"][gid]）写入 2translater.json（覆盖写入） ===
        try:
            agg_json_path.write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error("Failed to write 2translater.json: %s", e)

        return ctx
