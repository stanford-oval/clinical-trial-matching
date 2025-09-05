# modules/stages/canonical_attribute_value_verifier.py
from __future__ import annotations
import json, logging, re, time, hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import dspy
from ..utils import _ChatEngine  # 保持与过去文件一致的引擎类型

# ============================ 小工具 ============================

def _strip_code_fence(s: str) -> str:
    """去掉 ``` / ```json 包裹。"""
    return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.S)

def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

def _sha12(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]

# 归一化与键构造（用于按 entity start/end 精确定位）
def _norm(s: Any) -> str:
    return str(s or "").strip().lower()

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

def _member_key(m: Dict[str, Any]) -> Tuple[str, str, Optional[int], Optional[int]]:
    ent = m.get("entity", {}) or {}
    return (
        str(m.get("trial", "")),
        _norm(ent.get("surface_string")),
        _to_int(ent.get("start")),
        _to_int(ent.get("end")),
    )

# ============================ 主模块 ============================

class AttributeExtractorCanonicalAttributeValueVerifier(dspy.Module):
    """
    Verifies (attribute_type, canonical_attribute_value) selections for qualifiers.

    新日志布局（仅两类）：
      • mbench/attr_mbench/<Trial id>/g<gid>/5verifier_prompts_outputs/   # prompt & raw 文本
      • mbench/attr_mbench/<Trial id>/g<gid>/5verifier.json               # 写入处理后的 ctx["groups"][gid]
    """

    def __init__(
        self,
        engine: _ChatEngine | None = None,   # 可被 ctx["engine"] 覆盖
        *,
        model: str = "gpt-4o",
        max_retries: int = 3,
        temperature: float = 0.0,
        verbose: bool = False,
        mbench_path: str | Path = "mbench/attr_mbench/",
    ):
        super().__init__()
        self._default_engine = engine
        self.max_retries  = max_retries
        self.temperature  = temperature
        self.model        = model
        self.mbench_root  = Path(mbench_path)

        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # -------------------------- 新日志路径工具 --------------------------

    @staticmethod
    def _pick_trial_folder(ctx: Dict[str, Any], members: List[Dict[str, Any]]) -> str:
        """
        优先 ctx['trial_id']；否则从任一 member['trial'] 取下划线前缀（如 NCT00123456）。
        """
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
        prompts = base / "5verifier_prompts_outputs"
        prompts.mkdir(parents=True, exist_ok=True)
        agg = base / "5verifier.json"
        base.mkdir(parents=True, exist_ok=True)
        return prompts, agg

    @staticmethod
    def _write_txt(dir_path: Path, filename: str, content: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(str(content), encoding="utf-8")

    # -------------------------- 组装 prompt 输入 --------------------------

    def _iter_qualifiers(self, member: Dict[str, Any]):
        for q in member.get("all_qualifying_information_related_to_the_entity", []) or []:
            yield q

    def _find_next_verifiable_qualifier_idx_for_member(
        self, member: Dict[str, Any]
    ) -> Optional[int]:
        for qi, q in enumerate(self._iter_qualifiers(member)):
            if q.get("_cav_verified") is True:
                continue
            best_list = q.get("best_attribute_type_canonical_attribute_value", [])
            if isinstance(best_list, list) and len(best_list) > 0:
                return qi
        return None

    def _build_payload_for_qualifier(
        self,
        member: Dict[str, Any],
        qualifier: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "trial": member.get("trial", ""),
            "requirement": member.get("requirement", ""),
            "entity": member.get("entity", {}),
            "best_attribute_type_canonical_attribute_value": qualifier.get(
                "best_attribute_type_canonical_attribute_value", []
            ),
        }

    def _load_prompt_template(self, ctx: Dict[str, Any]) -> str:
        return ctx.get("AttributeExtractorCanonicalAttributeValueVerifier_prompt")

    def _build_prompt(self, payload: Dict[str, Any], template_txt: str) -> str:
        blob = json.dumps(payload, ensure_ascii=False, indent=2)
        token = "#qualifier_attributetype_canonicalattributevalue#"
        return template_txt.replace(token, blob, 1)

    # -------------------------- 调用 LLM & 解析（写 prompt/raw 到新目录） --------------------------

    def _infer_with_prompt_raw_logging(
        self,
        engine: _ChatEngine,
        prompt: str,
        *,
        log_dir: Path,        # 新：指定 prompts 输出目录
        gid: int,
        member_idx: int,
        ent_start: Optional[int],
        ent_end: Optional[int],
        qual_sha12: str,
    ) -> Dict[str, Any]:
        """
        与 _infer 同语义；额外把 prompt 和每次 raw 输出落盘到 mbench 新目录。
        文件前缀：{ts}_g{gid}_m{member}_s{start}_e{end}_q{sha12}_verifier
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        s = "na" if ent_start is None else ent_start
        e = "na" if ent_end   is None else ent_end
        base = f"{ts}_g{gid:04d}_m{member_idx:04d}_s{s}_e{e}_q{qual_sha12}_verifier"

        # prompt
        self._write_txt(log_dir, f"{base}_prompt.txt", prompt)

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            raw = engine(prompt, temperature=self.temperature)[0]
            # 原样落盘 raw（每次 attempt 一份）
            self._write_txt(log_dir, f"{base}_raw_attempt{attempt}.txt", raw)

            raw_clean = _strip_code_fence(raw)
            try:
                parsed = json.loads(raw_clean)

                if isinstance(parsed, list):
                    return {"best_attribute_type_canonical_attribute_value": parsed}

                if isinstance(parsed, dict):
                    val = parsed.get("best_attribute_type_canonical_attribute_value", None)
                    if isinstance(val, list):
                        return {"best_attribute_type_canonical_attribute_value": val}
                    if isinstance(val, dict):
                        return {"best_attribute_type_canonical_attribute_value": [val]}
                    return {"best_attribute_type_canonical_attribute_value": [parsed]}

                raise ValueError("Top-level JSON must be an object or array.")
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(f"LLM returned invalid JSON: {last_exc}")

    # -------------------------- 调用 LLM & 解析（保留以兼容） --------------------------

    def _infer(self, engine: _ChatEngine, prompt: str) -> Dict[str, Any]:
        last_exc = None
        for _ in range(self.max_retries):
            raw = engine(prompt, temperature=self.temperature)[0]
            raw = _strip_code_fence(raw)
            try:
                parsed = json.loads(raw)

                if isinstance(parsed, list):
                    return {"best_attribute_type_canonical_attribute_value": parsed}

                if isinstance(parsed, dict):
                    val = parsed.get("best_attribute_type_canonical_attribute_value", None)
                    if isinstance(val, list):
                        return {"best_attribute_type_canonical_attribute_value": val}
                    if isinstance(val, dict):
                        return {"best_attribute_type_canonical_attribute_value": [val]}
                    return {"best_attribute_type_canonical_attribute_value": [parsed]}

                raise ValueError("Top-level JSON must be an object or array.")
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"LLM returned invalid JSON: {last_exc}")

    # -------------------------- 回写到 context（按键匹配） --------------------------

    def _merge_into_qualifier(
        self,
        q_ptr: Dict[str, Any],
        llm_obj: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        enriched_best = None
        if isinstance(llm_obj, dict) and isinstance(
            llm_obj.get("best_attribute_type_canonical_attribute_value"), list
        ):
            enriched_best = llm_obj["best_attribute_type_canonical_attribute_value"]
        elif isinstance(llm_obj, list):  # 兜底
            enriched_best = llm_obj

        if enriched_best and all(isinstance(x, dict) for x in enriched_best):
            q_ptr["best_attribute_type_canonical_attribute_value"] = enriched_best

        q_ptr["_cav_verified"] = True

        kept: List[Dict[str, Any]] = []
        for item in (enriched_best or []):
            if str(item.get("KEEP", "")).upper() == "YES":
                kept.append(item)
        return kept

    # -------------------------- 处理一个 member 的一个 qualifier --------------------------

    def _verify_next_qualifier_for_member(
        self,
        engine: _ChatEngine,
        group: Dict[str, Any],
        member_idx: int,
        template_txt: str,
        *,
        prompts_dir: Path,  # 新：prompt/raw 输出目录
        gid: int,
        qual_index: Dict[Tuple[str, str, Optional[int], Optional[int], str], Dict[str, Any]],
    ) -> Tuple[bool, int]:

        members = group.get("members", []) or []
        if not (0 <= member_idx < len(members)):
            return False, 0
        member = members[member_idx]

        qi = self._find_next_verifiable_qualifier_idx_for_member(member)
        if qi is None:
            return False, 0

        qualifier = member["all_qualifying_information_related_to_the_entity"][qi]
        ent_obj = member.get("entity", {}) or {}
        ent_surf_raw = ent_obj.get("surface_string", "")
        ent_start = _to_int(ent_obj.get("start"))
        ent_end   = _to_int(ent_obj.get("end"))
        ent_surf_norm = _norm(ent_surf_raw)
        q_span_raw = qualifier.get("qualifier_span", "")
        q_span_norm = _norm(q_span_raw)
        trial_id = str(member.get("trial", "") or "unknown_trial")

        payload = self._build_payload_for_qualifier(member, qualifier)

        try:
            prompt = self._build_prompt(payload, template_txt)
            # —— 写入 prompt/raw 的推理 —— #
            llm_obj = self._infer_with_prompt_raw_logging(
                engine,
                prompt,
                log_dir=prompts_dir,
                gid=gid,
                member_idx=member_idx,
                ent_start=ent_start,
                ent_end=ent_end,
                qual_sha12=_sha12(q_span_raw or ""),
            )
        except Exception:
            # 即使异常也标记为“已处理”，避免死循环
            qualifier["_cav_verified"] = True
            return True, 0

        q_ptr = qual_index.get((trial_id, ent_surf_norm, ent_start, ent_end, q_span_norm), qualifier)
        kept = self._merge_into_qualifier(q_ptr, llm_obj)
        return True, len(kept)

    # ------------------------------- forward ------------------------------

    def forward(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:  # type: ignore[override]
        engine: _ChatEngine | None = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine found. Provide one via constructor or ctx['engine'].")

        try:
            template_txt = self._load_prompt_template(ctx)
        except Exception as e:
            raise RuntimeError(f"Failed to load AttributeExtractorCanonicalAttributeValueVerifier prompt: {e}")

        group = ctx["groups"][index]
        members = group.get("members", []) or []

        # 新日志定位
        trial_folder = self._pick_trial_folder(ctx, members)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder, index, ctx["inc_exc"])

        # ---- 构建 (trial, surface, start, end, qualifier_span) → qualifier 指针索引 ----
        qual_index: Dict[Tuple[str, str, Optional[int], Optional[int], str], Dict[str, Any]] = {}
        for m in members:
            note, ent, st, ed = _member_key(m)
            for q in m.get("all_qualifying_information_related_to_the_entity", []) or []:
                qkey = (note, ent, st, ed, _norm(q.get("qualifier_span", "")))
                qual_index[qkey] = q

        total = 0
        total_kept = 0

        for mi in range(len(members)):
            member_kept: List[Dict[str, Any]] = []
            local = 0
            while True:
                handled, kept_n = self._verify_next_qualifier_for_member(
                    engine, group, mi, template_txt,
                    prompts_dir=prompts_dir, gid=index, qual_index=qual_index
                )
                if not handled:
                    break
                total += 1
                local += 1
                total_kept += kept_n

                qlist = members[mi]["all_qualifying_information_related_to_the_entity"]
                for q in qlist:
                    for item in q.get("best_attribute_type_canonical_attribute_value", []) or []:
                        if str(item.get("KEEP", "")).upper() == "YES":
                            member_kept.append(item)

            if member_kept:
                # 去重后写回 final_attribute_type_canonical_value
                seen = set()
                deduped = []
                for it in member_kept:
                    key = (
                        it.get("entity", ""),
                        it.get("attribute_type", ""),
                        it.get("canonical_attribute_value", ""),
                        it.get("canonical_attribute_value_conceptID", ""),
                        it.get("qualifier_span", ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(it)
                members[mi]["final_attribute_type_canonical_value"] = deduped

            self.log.info("Group[%s] Member[%s]: verified %d qualifier(s).", index, mi, local)

        if total == 0:
            self.log.info("Group[%s]: no verifiable qualifier found.", index)
        else:
            self.log.info("Group[%s]: verified %d qualifier(s), KEEP==YES total=%d.", index, total, total_kept)

        # === 把“处理后的” ctx["groups"][gid] 写入新聚合文件（覆盖写入） ===
        try:
            agg_json_path.write_text(
                json.dumps(group, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            self.log.error("Failed to write 5verifier.json: %s", e)

        return ctx
