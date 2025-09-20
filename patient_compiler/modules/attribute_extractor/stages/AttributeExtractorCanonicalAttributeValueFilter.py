# modules/stages/canonical_attribute_value_filter.py
from __future__ import annotations
import json, logging, re, time, hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import dspy
from ..utils import _ChatEngine

# ============================ 工具函数 ============================

def _strip_code_fence(s: str):
    """去掉 ``` / ```json 包裹。"""
    return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.S)

def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _pick_canonical_value(cand: dict):
    """
    从向量检索候选中挑 canonical 名称与 conceptId。
    优先 best_match_term，其次 preferred_term / fully_specified_name / term。
    """
    name = (
        cand.get("best_match_term")
        or cand.get("preferred_term")
        or cand.get("fully_specified_name")
        or cand.get("term")
        or ""
    )
    cid = cand.get("conceptId") or cand.get("sctid") or cand.get("concept_id") or ""
    return name, str(cid) if cid is not None else ""

def _safe_id(s: str):
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

def _sha12(s: str):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]

# 归一化与键构造
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
        str(m.get("patient_note", "")),
        _norm(ent.get("surface_string")),
        _to_int(ent.get("start")),
        _to_int(ent.get("end")),
    )

# ============================ 主模块 ============================

class AttributeExtractorCanonicalAttributeValueFilter(dspy.Module):
    """
    Stage: 过滤/选择 canonical attribute value（每次对一个 member 的一个 pending qualifier）
    日志迁移：
      • Prompt & raw → mbench/attr_mbench/<Trial id>/g<gid>/4filter_canonical_attribute_value_prompts_outputs/
      • 聚合结果 → mbench/attr_mbench/<Trial id>/g<gid>/4filter_canonical_attribute_value.json
        （内容为处理后的 ctx["groups"][gid]）
    其余旧的 meta/jsonl/分散日志已移除。
    """

    def __init__(
        self,
        engine: _ChatEngine | None = None,
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
        t = str(ctx.get("note_id") or "").strip()
        if t:
            return t
        for m in members or []:
            tr = str(m.get("patient_note") or "").strip()
            if tr:
                return tr.split("_", 1)[0]
        return "unknown_trial"

    def _prompts_dir_and_agg_file(self, trial_folder: str, gid: int) -> tuple[Path, Path]:
        base = self.mbench_root / _safe_id(trial_folder) / f"g{gid:04d}"
        prompts = base / "4filter_canonical_attribute_value_prompts_outputs"
        prompts.mkdir(parents=True, exist_ok=True)
        agg = base / "4filter_canonical_attribute_value.json"
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

    def _find_next_pending_qualifier_idx_for_member(
        self, member: Dict[str, Any]
    ):
        """
        返回该 member 内下一个尚未拥有 "best_attribute_type_canonical_attribute_value" 的 qualifier 索引。
        若该 member 本就没有 qualifiers，则返回 None（保持原行为：不做后续操作）。
        """
        for qi, q in enumerate(self._iter_qualifiers(member)):
            if "best_attribute_type_canonical_attribute_value" not in q:
                return qi
        return None

    def _expand_attribute_interpretations_to_candidates(
        self, ai_list: List[Dict[str, Any]], limit_attr: int = 3, limit_cands: int = 3
    ):
        """
        将 attribute_interpretations 扩展为“每条一个 canonical 候选”的扁平列表。
        最多取 limit_attr 个属性；每个属性最多取 limit_cands 个候选。
        忽略 attribute_type 为空/缺失的项（避免“伪候选”）。
        """
        out: List[Dict[str, Any]] = []
        for ai in (ai_list or [])[:limit_attr]:
            attr_type = ai.get("attribute_type")
            if not attr_type or str(attr_type).strip() == "":
                continue

            attr_val  = ai.get("attribute_value", "")
            attr_def  = ai.get("attribute_definition", "")

            pmba = ai.get("potential_matches_by_attribute", {}) or {}
            block = pmba.get(attr_type) or {}
            cands = (block or {}).get("candidates", []) or []

            for cand in cands[:limit_cands]:
                canon_name, canon_cid = _pick_canonical_value(cand)
                out.append(
                    {
                        "attribute_type": attr_type,
                        "attribute_value": attr_val,
                        "attribute_definition": attr_def,
                        "canonical_attribute_value": canon_name,
                        "canonical_attribute_value_conceptID": canon_cid,
                    }
                )
        return out

    def _build_single_payload_for_qualifier(
        self,
        member: Dict[str, Any],
        qualifier: Dict[str, Any],
    ):
        expanded_ai = self._expand_attribute_interpretations_to_candidates(
            qualifier.get("attribute_interpretations", []),
            limit_attr=3,
            limit_cands=3,
        )

        payload = {
            "patient_note": member.get("patient_note", ""),
            "requirement": member.get("requirement", ""),
            "entity": member.get("entity", {}),
            "candidate_attributetype_canonicalattributevalue": [
                {
                    "entity": _safe_get(member, "entity", "surface_string", default=""),
                    "qualifier_span": qualifier.get("qualifier_span", ""),
                    "qualifier": qualifier.get("qualifier", ""),
                    "rationale": qualifier.get("rationale", ""),
                    "attribute_interpretations": expanded_ai,
                }
            ],
        }
        return payload

    def _build_prompt(self, payload: Dict[str, Any], template_txt: str):
        blob = json.dumps(payload, ensure_ascii=False, indent=2)
        prompt = template_txt
        prompt = prompt.replace("#candidate_attributetype_canonicalattributevalue#", blob)
        prompt = re.sub(
            r"<attribute_value_pairs_list>\s*#attribute_value_pairs_list#\s*</attribute_value_pairs_list>",
            blob,
            prompt,
            flags=re.S,
        )
        prompt = re.sub(r"</?attribute_value_pairs_list>", "", prompt)
        return prompt

    # -------------------------- 调用 LLM & 解析（落盘 prompt/raw） --------------------------

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
        期望返回 **单个对象**（不是数组），同时把 prompt 和每次 raw 写到新目录：
        mbench/attr_mbench/<Trial id>/g<gid>/4filter_canonical_attribute_value_prompts_outputs/
        文件前缀：{ts}_g{gid}_m{member}_s{start}_e{end}_q{sha12}_canon
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        s = "na" if ent_start is None else ent_start
        e = "na" if ent_end   is None else ent_end
        base = f"{ts}_g{gid:04d}_m{member_idx:04d}_s{s}_e{e}_q{qual_sha12}_canon"

        # 写 prompt
        self._write_txt(log_dir, f"{base}_prompt.txt", prompt)

        messages = prompt
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            raw = engine(messages, temperature=self.temperature)[0]
            # 每次 attempt 的原始输出都落盘
            self._write_txt(log_dir, f"{base}_raw_attempt{attempt}.txt", raw)

            cleaned = _strip_code_fence(raw)
            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    raise ValueError("Top-level JSON must be an object (dict).")
                return parsed
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(f"LLM returned invalid JSON: {last_exc}")

    # -------------------------- 合并 LLM 输出到上下文 --------------------------

    def _merge_output_by_key(
        self,
        q_ptr: Dict[str, Any],
        llm_obj: Dict[str, Any],
    ) -> None:
        """
        将 LLM 输出合并进目标 qualifier（已通过 key 定位到 q_ptr）。
        """
        cand_key = "candidate_attributetype_canonicalattributevalue"
        if isinstance(llm_obj, dict) and cand_key in llm_obj:
            lst = llm_obj.get(cand_key) or []
            cand_block = lst[0] if lst else {}
        else:
            cand_block = {
                "attribute_interpretations": llm_obj.get("attribute_interpretations", []),
                "best_attribute_type_canonical_attribute_value": llm_obj.get(
                    "best_attribute_type_canonical_attribute_value", []
                ),
            }

        if "attribute_interpretations" in cand_block:
            q_ptr["attribute_interpretations"] = cand_block["attribute_interpretations"]

        raw_best = cand_block.get("best_attribute_type_canonical_attribute_value", [])

        def _looks_like_best_item(x: dict) -> bool:
            return any(k in x for k in ("attribute_type", "canonical_attribute_value", "KEEP"))

        if isinstance(raw_best, dict):
            best_list = [raw_best] if _looks_like_best_item(raw_best) else []
        elif isinstance(raw_best, list):
            best_list = [x for x in raw_best if isinstance(x, dict) and _looks_like_best_item(x)]
        else:
            best_list = []

        if not best_list and isinstance(raw_best, dict) and "rationale" in raw_best and not _looks_like_best_item(raw_best):
            q_ptr["no_best_reason"] = raw_best.get("rationale")

        q_ptr["best_attribute_type_canonical_attribute_value"] = best_list

    # -------------------------- 处理一个 member 的一个 qualifier --------------------------

    def _canonicalise_one_qualifier_for_member(
        self,
        engine: _ChatEngine,
        group: Dict[str, Any],
        member_idx: int,
        template_txt: str,
        *,
        prompts_dir: Path,  # 新：prompt/raw 输出目录
        gid: int,
        qual_index: Dict[Tuple[str, str, Optional[int], Optional[int], str], Dict[str, Any]],
    ):
        """只处理指定 member 的一个 pending qualifier；处理成功返回 True，否则 False。"""
        members = group.get("members", []) or []
        if not (0 <= member_idx < len(members)):
            return False
        member = members[member_idx]

        qi = self._find_next_pending_qualifier_idx_for_member(member)
        if qi is None:
            return False

        qualifier = member["all_qualifying_information_related_to_the_entity"][qi]

        # 关键信息（用于日志命名）
        patient_note_id = str(member.get("patient_note", "") or "unknown_trial")
        ent_obj = member.get("entity", {}) or {}
        ent_surf_raw = ent_obj.get("surface_string", "")
        ent_start = _to_int(ent_obj.get("start"))
        ent_end   = _to_int(ent_obj.get("end"))
        ent_surf_norm = _norm(ent_surf_raw)
        q_span_raw = qualifier.get("qualifier_span", "")
        q_span_norm = _norm(q_span_raw)

        # 先展开；若无候选 → 不调用 LLM（满足你的“展开后为空就不要调用 prompt”）
        expanded_ai = self._expand_attribute_interpretations_to_candidates(
            qualifier.get("attribute_interpretations", []),
            limit_attr=3,
            limit_cands=3,
        )
        if not expanded_ai:
            q_ptr = qual_index.get((patient_note_id, ent_surf_norm, ent_start, ent_end, q_span_norm)) or qualifier
            q_ptr.setdefault("attribute_interpretations", [])
            q_ptr["best_attribute_type_canonical_attribute_value"] = []
            return True  # 视为已处理

        # 正常路径：有候选 → 组装 payload 并调用 LLM（并把 prompt/raw 落盘在新目录）
        payload = {
            "patient_note": member.get("patient_note", ""),
            "requirement": member.get("requirement", ""),
            "entity": member.get("entity", {}),
            "candidate_attributetype_canonicalattributevalue": [
                {
                    "entity": _safe_get(member, "entity", "surface_string", default=""),
                    "qualifier_span": qualifier.get("qualifier_span", ""),
                    "qualifier": qualifier.get("qualifier", ""),
                    "rationale": qualifier.get("rationale", ""),
                    "attribute_interpretations": expanded_ai,
                }
            ],
        }
        prompt  = self._build_prompt(payload, template_txt)

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
        q_ptr = qual_index.get((patient_note_id, ent_surf_norm, ent_start, ent_end, q_span_norm))
        if q_ptr is not None:
            self._merge_output_by_key(q_ptr, llm_obj)
        else:
            self._merge_output_by_key(qualifier, llm_obj)

        return True

    # ------------------------------- forward ------------------------------

    def forward(self, ctx: Dict[str, Any], index: int):  # type: ignore[override]
        engine: _ChatEngine | None = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine found. Provide one via constructor or ctx['engine'].")

        prompt_txt = ctx["AttributeExtractorCanonicalAttributeValueFilterCanon_prompt"]

        group = ctx["groups"][index]
        members = group.get("members", []) or []

        # 新日志路径（以 trial + gid 组织）
        trial_folder = self._pick_trial_folder(ctx, members)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder, index)

        # 构建 (trial, surface, start, end, qualifier_span) → qualifier 指针索引
        qual_index: Dict[Tuple[str, str, Optional[int], Optional[int], str], Dict[str, Any]] = {}
        for m in members:
            note, ent, st, ed = _member_key(m)
            for q in m.get("all_qualifying_information_related_to_the_entity", []) or []:
                qkey = (note, ent, st, ed, _norm(q.get("qualifier_span", "")))
                qual_index[qkey] = q

        total = 0
        for mi in range(len(members)):
            local = 0
            while self._canonicalise_one_qualifier_for_member(
                engine, group, mi, prompt_txt,
                prompts_dir=prompts_dir, gid=index, qual_index=qual_index
            ):
                total += 1
                local += 1
            self.log.info("Group[%s] Member[%s]: canonicalised %d qualifier(s).", index, mi, local)

        if total == 0:
            self.log.info("Group[%s]: no pending qualifier to canonicalise for any member.", index)
        else:
            self.log.info("Group[%s]: canonicalised %d qualifier(s) across %d member(s).", index, total, len(members))

        # 稳定排序并落一份本 gid 的聚合 JSON（内容为处理后的 group）
        group["members"] = sorted(members, key=lambda m: (m.get("patient_note", ""), m.get("requirement", "")))
        try:
            agg_json_path.write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error("Failed to write 4filter_canonical_attribute_value.json: %s", e)

        return ctx
