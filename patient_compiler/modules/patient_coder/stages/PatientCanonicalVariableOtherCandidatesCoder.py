# PatientCanonicalVariableOtherCandidatesCoder.py
from __future__ import annotations
"""
PatientCanonicalVariableOtherCandidatesCoder
────────────────────────────────────────────────────────────────────────────
对 ctx["entity_other_canonical_candidates"][str(idx)] 进行“命名/变量声明”编码，
使用独立的 prompt：context["PatientCanonicalVariableOtherCandidatesCoder_prompt"]。

占位词与 PatientCanonicalVariableCoder 相同：
  #SMT_PROGRAM_BY_FAR#, #CANONICAL_FORMS#, #PATIENT_FACT#, #PATIENT_NOTE#

输入（来自 LLMBasedMedicalEntityFilter 收集到的 payload）示例：
{
  "generated": "2025-11-05T12:34:56",
  "requirement": "...",
  "count_pairs": 12,
  "items": [
    {
      "extracted_span": "diabetes",
      "entity_name": "diabetes",
      "offset": [123,131],
      "kept_conceptId": "44054006",
      "candidate": {
         "conceptId": "XXXX",
         "preferred_term": "...",
         "fully_specified_name": "...",
         "type": "Clinical finding",
         "match_score": 0.83,
         "best_match_term": "...",
         "matched_alternative_synonyms": [...],
         "all_terms": [...],
         "definition": "..."
      }
    },
    ...
  ]
}

LLM 期望输出（JSON 数组）放在：
  <all_other_variable_declarations> ... </all_other_variable_declarations>

写回 context：
- 追加到 context["all_other_variable_declarations"]
- 并在 context["other_candidate_variable_declarations_by_req"][str(idx)] 单独存一份
- 记录 errors 到 context["errors"]（若有）

日志：
- {log_dir}/{note_id}/req{idx:03d}_namer_other_candidates_prompt.txt
- {log_dir}/{note_id}/req{idx:03d}_namer_other_candidates_raw.txt
- {log_dir}/{note_id}/req{idx:03d}_namer_other_candidates_plan.json
"""

import json, os, warnings, re, copy
from typing import Dict, List, Any

import dspy
from .namer_common import (
    _log,
    _parse_json_array_relaxed,
    _ERRORS_RE,
    _NEWOTHER_RE,
    _validate_decl_list,
    _sort_key,
)

_INF_HOURS: float = 1_000_000_000.0
# PatientCanonicalVariableOtherCandidatesCoder.py

class PatientCanonicalVariableOtherCandidatesCoder(dspy.Module):
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        engine,
        *,
        log_dir: str | None = None,
        max_per_span: int = 8,               # 每个 occurrence 纳入多少“其他候选”
        allow_mixed_case: bool = False,      # 与 canonical coder 保持风格
        max_prompt_items: int = 5,           # ★ 新增：放入 prompt 的全局上限（默认 5 条）
    ):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.max_per_span = max_per_span
        self._allow_mixed_case = allow_mixed_case
        self.max_prompt_items = max_prompt_items   # ★ 新增


    def _meta_key(self, span: str | None, start: int | None, end: int | None, cid: str | None) -> tuple:
        """构造稳定的匹配键（都允许为 None）"""
        s = (span or "").strip().lower() or None
        c = (str(cid).strip() if cid is not None else None) or None
        return (start if isinstance(start, int) else None,
                end   if isinstance(end, int)   else None,
                s, c)

    def _build_batch_meta_index(self, batch_rows: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
        """
        从本批 prompt 的源 rows 建立多种粒度的索引：
        1) (start,end,span_lower,conceptId)
        2) (start,end,span_lower,None)
        3) (None,None,span_lower,conceptId)
        4) (None,None,span_lower,None)
        5) (None,None,None,conceptId)
        """
        idx: Dict[tuple, Dict[str, Any]] = {}
        for e in batch_rows:
            off = e.get("offset") or []
            st  = off[0] if len(off) > 0 else None
            ed  = off[1] if len(off) > 1 else None
            span = e.get("span") or e.get("extracted_span") or ""
            cand = e.get("candidate") or {}
            cid  = str(cand.get("conceptId") or "") or None
            meta = {
                "conceptId": cid or "",
                "preferred_term": cand.get("preferred_term") or "",
                "fully_specified_name": cand.get("fully_specified_name") or "",
            }
            # 按优先级建立多把钥匙
            keys = [
                self._meta_key(span, st, ed, cid),
                self._meta_key(span, st, ed, None),
                self._meta_key(span, None, None, cid),
                self._meta_key(span, None, None, None),
                self._meta_key(None, None, None, cid),
            ]
            for k in keys:
                idx.setdefault(k, meta)
        return idx

    def _enrich_with_meta(self, item: Dict[str, Any], meta_idx: Dict[tuple, Dict[str, Any]], fact_id: str) -> None:
        """
        尝试用 (span,start,end,conceptId) 匹配并补全字段；不覆盖已有值。
        """
        span  = item.get("span") or item.get("extracted_span") or item.get("entity_name") or ""
        st    = item.get("start")
        ed    = item.get("end")
        cid_i = item.get("conceptId")
        # 按优先级尝试命中
        for key in [
            self._meta_key(span, st, ed, cid_i),
            self._meta_key(span, st, ed, None),
            self._meta_key(span, None, None, cid_i),
            self._meta_key(span, None, None, None),
            self._meta_key(None, None, None, cid_i),
        ]:
            meta = meta_idx.get(key)
            if not meta:
                continue
            # 补全但不强行覆盖
            if not item.get("conceptId") and meta.get("conceptId"):
                item["conceptId"] = meta["conceptId"]
            if not item.get("preferred_term") and meta.get("preferred_term"):
                item["preferred_term"] = meta["preferred_term"]
            if not item.get("fully_specified_name") and meta.get("fully_specified_name"):
                item["fully_specified_name"] = meta["fully_specified_name"]
            break
        # 无论是否命中，都补上 fact_id
        if "fact_id" not in item:
            item["fact_id"] = fact_id


    # === timewindow normalization helpers ===
    _INF_HOURS = 1_000_000_000.0  # 与 canonical 一致的“近似无穷”

    def _unit_to_hours(self, units: str | None) -> float:
        u = (units or "").strip().lower()
        if u in {"minute", "minutes", "mins", "min"}: return 1.0 / 60.0
        if u in {"hour", "hours", "hr", "hrs"}:      return 1.0
        if u in {"day", "days"}:                     return 24.0
        if u in {"week", "weeks"}:                   return 24.0 * 7.0
        if u in {"month", "months"}:                 return 24.0 * 30.0    # 30天约定
        if u in {"year", "years"}:                   return 24.0 * 365.0   # 365天约定
        return 1.0  # 默认按小时

    def _bound_to_hours(self, bound: Dict[str, Any]) -> tuple[float, bool]:
        """
        将单端点 {temporal_direction, temporal_magnitude, units, inclusive}
        归一到“相对现在的小时”并返回 (hours, inclusive)。
        约定：past→负号，future→正号，now→0；"Inf"→±_INF_HOURS。
        """
        if not isinstance(bound, dict):
            return 0.0, True
        direction = str(bound.get("temporal_direction") or "").strip().lower()
        mag_raw   = bound.get("temporal_magnitude", 0)
        units     = bound.get("units", "hours")
        inclusive = bool(bound.get("inclusive", True))

        # 解析 magnitude
        is_inf = isinstance(mag_raw, str) and mag_raw.strip().lower() == "inf"
        try:
            mag_val = float(mag_raw) if not is_inf else float("inf")
        except Exception:
            mag_val = 0.0
            is_inf = False

        # 单位转小时
        hours = self._unit_to_hours(units) * (mag_val if not (direction == "now") else 0.0)

        # 方向与无穷
        if direction == "past":
            if is_inf: return -_INF_HOURS, inclusive
            return -abs(hours), inclusive
        if direction == "future":
            if is_inf: return  _INF_HOURS, inclusive
            return  abs(hours), inclusive
        # now / 其它异常情况 → 0
        return 0.0, inclusive

    def _derive_timewindows_in_place(self, item: Dict[str, Any]) -> None:
        """
        对单条 other-candidate 声明：
        - 复制两段时间窗到 *_raw
        - 计算 smallest_* 与 largest_* 的小时数与包含性布尔位
        字段名与 canonical 派生保持一致。
        """
        sm_key = "timewindow_this_patient_fact_certainly_holds"
        lg_key = "largest_timewindow_this_patient_fact_may_hold"

        sm = item.get(sm_key)
        lg = item.get(lg_key)

        # 保留原始（deepcopy 防止后续修改）
        if sm is not None and ("_" + sm_key + "_raw") not in item:
            item["_" + sm_key + "_raw"] = copy.deepcopy(sm)
        if lg is not None and ("_" + lg_key + "_raw") not in item:
            item["_" + lg_key + "_raw"] = copy.deepcopy(lg)

        # 计算 smallest（certainly holds）
        if isinstance(sm, dict):
            s_b = sm.get("start_time", {})
            e_b = sm.get("end_time", {})
            s_h, s_inc = self._bound_to_hours(s_b)
            e_h, e_inc = self._bound_to_hours(e_b)
            item["smallest_timewindow_start_time_in_hours"]   = float(s_h)
            item["smallest_timewindow_end_time_in_hours"]     = float(e_h)
            item["smallest_timewindow_start_time_inclusive"]  = bool(s_inc)
            item["smallest_timewindow_end_time_inclusive"]    = bool(e_inc)

        # 计算 largest（may hold）
        if isinstance(lg, dict):
            s_b = lg.get("start_time", {})
            e_b = lg.get("end_time", {})
            s_h, s_inc = self._bound_to_hours(s_b)
            e_h, e_inc = self._bound_to_hours(e_b)
            item["largest_timewindow_start_time_in_hours"]    = float(s_h)
            item["largest_timewindow_end_time_in_hours"]      = float(e_h)
            item["largest_timewindow_start_time_inclusive"]   = bool(s_inc)
            item["largest_timewindow_end_time_inclusive"]     = bool(e_inc)



    def _to_snake_case(self, text: str) -> str:
        """
        将文本规范化为 snake_case：
        - 非字母数字字符统一替换为下划线
        - 连续下划线合并
        - 去除首尾下划线
        - 若 allow_mixed_case=False（默认），统一转小写
        """
        s = (text or "").strip()
        if not s:
            return ""
        s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
        s = re.sub(r"_{2,}", "_", s).strip("_")
        if not self._allow_mixed_case:
            s = s.lower()
        return s


    def _collect_compact_rows(self, ctx: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
        """
        从 context 收集并去重 other-candidates，返回“行”列表（不做全局截断）。
        每一行结构与 _build_other_bundle 原 compact2 的元素一致：
        {"span","entity_name","offset","kept_conceptId","candidate":{...}}
        """
        store_all = ctx.get("entity_other_canonical_candidates") or {}
        payload = store_all.get(str(idx)) or store_all.get(idx) or {}
        items = payload.get("items") or []

        compact: List[Dict[str, Any]] = []
        for it in items:
            cand = it.get("candidate") or {}
            compact.append({
                "span": it.get("extracted_span") or "",
                "entity_name": it.get("entity_name") or "",
                "offset": it.get("offset") or [],
                "kept_conceptId": it.get("kept_conceptId") or "",
                "candidate": {
                    "conceptId": str(cand.get("conceptId") or cand.get("concept_id") or cand.get("sctid") or ""),
                    "preferred_term": cand.get("preferred_term") or "",
                    "fully_specified_name": cand.get("fully_specified_name") or "",
                    "type": cand.get("type") or "",
                    "match_score": cand.get("match_score"),
                    "best_match_term": cand.get("best_match_term") or "",
                    "synonyms_sample": (cand.get("all_terms") or [])[:10],
                    "definition": cand.get("definition") or "",
                }
            })

        # occurrence + conceptId 去重并保序；每个 occurrence 内部再按 max_per_span 截断
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        order_keys: List[str] = []
        for e in compact:
            k = f"{e.get('span','')}|{tuple(e.get('offset') or [])}"
            if k not in grouped:
                grouped[k] = []
                order_keys.append(k)
            cid = e["candidate"]["conceptId"]
            if cid and not any(x["candidate"]["conceptId"] == cid for x in grouped[k]):
                grouped[k].append(e)

        compact2: List[Dict[str, Any]] = []
        for k in order_keys:
            rows = grouped[k][: self.max_per_span] if self.max_per_span > 0 else grouped[k]
            compact2.extend(rows)
        return compact2


    def _rows_to_prompt_array(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        把去重后的行转换成喂给 LLM 的 JSON 项（含 snake_case 的 candidate_entity_canonical_form）。
        """
        arr: List[Dict[str, Any]] = []
        for e in rows:
            offset = e.get("offset") or []
            start = offset[0] if len(offset) > 0 else None
            end   = offset[1] if len(offset) > 1 else None
            cand  = e.get("candidate") or {}

            cand_name_src = (
                cand.get("preferred_term")
                or cand.get("fully_specified_name")
                or e.get("entity_name")
                or e.get("span")
                or ""
            )
            cand_form_snake = self._to_snake_case(cand_name_src)
            concept_id = str(
                cand.get("conceptId")
                or cand.get("concept_id")
                or cand.get("sctid")
                or e.get("kept_conceptId")
                or ""
            )

            arr.append({
                "candidate_entity_canonical_form": cand_form_snake,
                "candidate_entity_canonical_name": cand.get("preferred_term") or "",
                "candidate_canonical_entity_type": cand.get("type") or "",
                "entity_name": e.get("entity_name") or e.get("span") or "",
                "extracted_span": e.get("span") or "",
                "start": start,
                "end": end,
                "conceptId": concept_id,
                "definition": cand.get("definition") or "",
            })
        return arr


    def _wrap_canonical_forms(self, arr: List[Dict[str, Any]]) -> str:
        return json.dumps(arr, indent=2, ensure_ascii=False) 


    def _batch_iter(self, rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """按 batch_size 切分 rows。batch_size<=0 时视为单批。"""
        if not rows:
            return []
        if not batch_size or batch_size <= 0:
            return [rows]
        return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]


    def _build_other_bundle(self, ctx: Dict[str, Any], idx: int, rows: List[Dict[str, Any]] | None = None) -> str:
        """
        构造单批次的 <canonical_forms>：
        - 若 rows 传入：仅用该批 rows。
        - 若 rows 未传：收集全部 rows，并按 max_prompt_items 截断到一批。
        """
        if rows is None:
            rows = self._collect_compact_rows(ctx, idx)
            if self.max_prompt_items and self.max_prompt_items > 0:
                rows = rows[: self.max_prompt_items]

        arr = self._rows_to_prompt_array(rows)
        return self._wrap_canonical_forms(arr)




    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        requirements: List = context.get("requirements", []) or []
        if not requirements:
            return context
        idx: int = context.get("current_requirement_index", 0)

        # 如果本 req 没有“其他候选”，直接返回
        store_all = context.get("entity_other_canonical_candidates") or {}
        payload = store_all.get(str(idx)) or store_all.get(idx)
        if not payload or not (payload.get("items") or []):
            return context

        # Prompt 模板
        namer_prompt_tpl = context.get("PatientCanonicalVariableOtherCandidatesCoder_prompt", "")
        if not namer_prompt_tpl:
            _log("namer-other ✗", idx, "prompt template missing (other-candidates)")
            return context

        req_entry = requirements[idx]
        requirement_txt = (
            req_entry.get("requirement") if isinstance(req_entry, dict) else str(req_entry)
        )

        # 日志基路径
        note_id = context.get("note_id", "unknown_patient_note")
        base_dir = os.path.join(self.log_dir, note_id)
        os.makedirs(base_dir, exist_ok=True)
        base_name = f"req{idx:03d}_namer_other_candidates"

        # —— 关键：按批次处理 —— #
        all_rows = self._collect_compact_rows(context, idx)
        batches = self._batch_iter(all_rows, self.max_prompt_items)

        all_other_agg: List[Dict[str, Any]] = []
        all_errors_agg: List[Dict[str, Any]] = []
        any_batch_succeeded = False

        for bidx, batch_rows in enumerate(batches):
            batch_meta_idx = self._build_batch_meta_index(batch_rows)
            fact_id = f"fact{idx:03d}"

            other_bundle = self._build_other_bundle(context, idx, rows=batch_rows)
            prompt = (
                namer_prompt_tpl
                .replace("#SMT_PROGRAM_BY_FAR#", "\n".join(context.get("smt_program_lines", [])))
                .replace("#CANONICAL_FORMS#", other_bundle)
                .replace("#PATIENT_FACT#", requirement_txt)
                .replace("#PATIENT_NOTE#", context.get("requirement_text", ""))
            )

            # 每批单独日志
            p_log = os.path.join(base_dir, f"{base_name}_b{bidx:03d}_prompt.txt")
            r_log = os.path.join(base_dir, f"{base_name}_b{bidx:03d}_raw.txt")
            plan_log = os.path.join(base_dir, f"{base_name}_b{bidx:03d}_plan.json")
            with open(p_log, "w", encoding="utf-8") as fh:
                fh.write(prompt)

            last_error: Exception | None = None
            batch_ok = False

            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                llm_out: str = self.engine(prompt)[0]
                with open(r_log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")
                try:
                    # 只解析 <all_other_variable_declarations> 区块
                    m_other = _NEWOTHER_RE.search(llm_out)
                    if not m_other:
                        raise RuntimeError("Block <all_other_variable_declarations> not found")
                    other_list_raw = _parse_json_array_relaxed(m_other.group(1))

                    # 错误区块（可选）
                    m_err = _ERRORS_RE.search(llm_out)
                    errors_list = _parse_json_array_relaxed(m_err.group(1)) if m_err else []

                    # 结构校验（宽松）
                    try:
                        other_list = _validate_decl_list(
                            other_list_raw,
                            kind="other",
                            allow_mixed_case=self._allow_mixed_case,
                            enable_template_checks=False,
                        )
                    except Exception as ve2:
                        other_list = list(other_list_raw if isinstance(other_list_raw, list) else [])
                        errors_list = (errors_list or [])
                        errors_list.append({
                            "invariant": "O-validate-relaxed",
                            "problem": "validate_decl_list(kind='other') failed; accepted relaxed",
                            "detail": str(ve2),
                        })

                    # ★ 在排序之前先补全 conceptId / preferred_term / fsn / fact_id
                    for it in (other_list or []):
                        self._enrich_with_meta(it, batch_meta_idx, fact_id)
                    # ★ 然后做时间窗派生/规范化（与 canonical 一致的字段）
                    for it in (other_list or []):
                        self._derive_timewindows_in_place(it)


                    try:
                        other_list.sort(key=_sort_key)
                    except Exception:
                        pass

                except Exception as exc:
                    last_error = exc
                    if attempt == self.MAX_ATTEMPTS:
                        break
                    _log("namer-other", idx, f"parse error; retry ({attempt}/{self.MAX_ATTEMPTS})")
                    continue

                # 本批成功：记录计划、汇总结果
                plan_obj = {
                    "this_req_pass_other_variable_declarations": other_list,
                    "errors": errors_list or [],
                    "stage": "other_candidates",
                    "batch_index": bidx,
                }
                with open(plan_log, "w", encoding="utf-8") as fh:
                    json.dump(plan_obj, fh, indent=2, ensure_ascii=False)

                all_other_agg.extend(other_list or [])
                all_errors_agg.extend(errors_list or [])
                _log("namer-other ✓", idx, f"[batch {bidx+1}/{len(batches)}] accepted ({len(other_list)} rows)")
                batch_ok = True
                break  # 结束本批尝试

            if not batch_ok:
                # 本批最终失败，记录错误但继续后续批次
                warn_msg = f"other-candidates namer batch#{bidx} fallback (req#{idx}): {last_error}"
                warnings.warn(warn_msg, RuntimeWarning)
                _log("namer-other ⚠ batch-fallback", idx, f"batch {bidx}: {last_error!r}")
                all_errors_agg.append({
                    "invariant": "O-fallback-batch",
                    "problem": "LLM outputs could not be parsed/validated after retries (other-candidates batch)",
                    "detail": str(last_error) if last_error else "unknown",
                    "batch_index": bidx,
                })
            else:
                any_batch_succeeded = True

        # —— 所有批次处理完的收尾 —— #
        # 去重（按 JSON 序列化去重，尽量稳定）
        uniq_seen = set()
        uniq_other: List[Dict[str, Any]] = []
        for obj in all_other_agg:
            key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
            if key not in uniq_seen:
                uniq_seen.add(key)
                uniq_other.append(obj)

        # 写回 context（累加）
        merged_other = (context.get("this_req_pass_other_variable_declarations") or []) + (uniq_other or [])
        context["this_req_pass_other_variable_declarations"] = merged_other

        by_req = context.get("other_candidate_variable_declarations_by_req") or {}
        prev = by_req.get(str(idx)) or []
        by_req[str(idx)] = (prev or []) + (uniq_other or [])
        context["other_candidate_variable_declarations_by_req"] = by_req

        # === Build/append embedding_search_other_candidate_variable_declarations ===
        def _is_yes_flag(v):
            if isinstance(v, str):
                return v.strip().lower() in {"yes", "true", "1"}
            return bool(v)

        emb_list = context.get("embedding_search_other_candidate_variable_declarations") or []

        # 用本轮去重后的 uniq_other 来避免批间重复
        for item in (uniq_other or []):
            if not _is_yes_flag(item.get("CAN_BE_INFERRED_FROM_PATIENT_FACT")):
                continue  # 仅收集 YES，NO 的整项跳过
            # 复制并去掉 reason（其余字段原样保留；YES 字段保留）
            cleaned = {k: v for k, v in item.items() if k != "reason"}
            emb_list.append(cleaned)

        context["embedding_search_other_candidate_variable_declarations"] = emb_list




        context["errors"] = (context.get("errors") or []) + (all_errors_agg or [])
        context["namer_stage_other_candidates"] = "completed"
        context["namer_other_total_batches"] = len(batches)
        context["namer_other_total_rows"] = len(all_rows)

        return context

