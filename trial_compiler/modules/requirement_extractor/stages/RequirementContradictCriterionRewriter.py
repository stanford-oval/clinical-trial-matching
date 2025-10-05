#!/usr/bin/env python3
"""
RequirementContradictCriterionRewriter.py
────────────────────────────────────────────────────────────────────────────
Cross-side contradiction repair inserted at the barrier after
'Until Precision Rewriter' and before 'After Precision Rewriter'.

New:
- forward_pair(inc_ctx, exc_ctx): uses inclusion-side inclusion_criteria and
  exclusion-side exclusion_criteria per cohort, runs one LLM pass per cohort,
  then writes results back into the two side contexts respectively.
- Backward compatible forward(context): legacy single-context mode.

Writes:
- inc_ctx: updated inclusion_criteria + requirement_text (+ mirrors)
- exc_ctx: updated exclusion_criteria + requirement_text (+ mirrors)
- both ctxs: contradiction_rewrite (per-cohort raw+parsed logs)

Preambles (exactly once per item):
- Inclusion: "To be included, the patient must"
- Exclusion: "A patient is excluded if the patient"
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import dspy

# ----------------------------- Utilities ------------------------------------

PREAMBLE_INCL = "To be included, the patient must"
PREAMBLE_EXCL = "A patient is excluded if the patient"


def _squash_space(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _needs_prefix(text: str, prefix: str) -> bool:
    return not (text or "").lstrip().lower().startswith(prefix.lower())


def _apply_preamble(items: List[str], *, inclusion: bool) -> List[str]:
    prefix = PREAMBLE_INCL if inclusion else PREAMBLE_EXCL
    out: List[str] = []
    for it in items:
        t = _squash_space(str(it))
        out.append(f"{prefix} {t}" if _needs_prefix(t, prefix) else t)
    return out


def _safe_json_find(s: str) -> Optional[Any]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        if s.startswith("[") and s.endswith("]"):
            s2 = re.sub(r"'([^'\\]*?(?:\\.[^'\\]*?)*)'", r'"\1"', s)
            return json.loads(s2)
    except Exception:
        pass
    first_obj = s.find("{")
    first_arr = s.find("[")
    start = min([p for p in [first_obj, first_arr] if p != -1], default=-1)
    if start == -1:
        return None
    for end in range(len(s), start, -1):
        frag = s[start:end]
        try:
            return json.loads(frag)
        except Exception:
            try:
                val = ast.literal_eval(frag)
                return val
            except Exception:
                continue
    return None

def _extract_req_list_from_cohort(c: Dict[str, Any], *, fallback_field: str) -> List[str]:
    """
    优先从 PR 后的结构化字段 c['requirements'][*]['requirement'] 读取。
    若不存在或为空，则回退到字符串化列表字段（如 'inclusion_criteria' / 'exclusion_criteria'）。
    """
    reqs = c.get("requirements")
    if isinstance(reqs, list) and reqs:
        out: List[str] = []
        for r in reqs:
            if isinstance(r, dict):
                txt = _squash_space(str(r.get("requirement") or ""))
                if txt:
                    out.append(txt)
        if out:
            return out
    # 回退：历史字符串化列表
    raw = str(c.get(fallback_field) or "")
    return _parse_criteria_list(raw)


def _parse_criteria_list(s: str) -> List[str]:
    if not s:
        return []
    s = s.strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [str(_squash_space(x)) for x in obj if str(_squash_space(x))]
    except Exception:
        pass
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(_squash_space(x)) for x in obj if str(_squash_space(x))]
    except Exception:
        pass
    parts = re.split(r"\n+|\r+|\u2022|\u2023|\u25E6|\u2043|\u2219|;", s)
    items: List[str] = []
    for p in parts:
        q = _squash_space(re.sub(r"^[\s\-•·\d\)\(\.]+", "", p))
        if q:
            items.append(q)
    return items


def _to_pylist_string(items: List[str]) -> str:
    escaped = ["'" + str(x).replace("'", "\\'") + "'" for x in items]
    return "[" + ", ".join(escaped) + "]"


DEFAULT_PROMPT = """=== ROLE ===
You are a careful, deterministic clinical-trial eligibility text editor. Your job is to repair contradictions and imprecisions between inclusion and exclusion criteria without inventing new clinical meaning.

=== BACKGROUND ===
Inclusion: If a patient meets at least one inclusion criterion (True), they are eligible unless an exclusion applies.
Exclusion: Exclusion criteria remove otherwise eligible participants.
When an inclusion and an exclusion conflict, prefer revising the exclusion to resolve the contradiction (keep the inclusion as written unless the inclusion is clearly wrong per the provided context).

=== TASK ===
Given:
<contextual_text> (protocol background: title, phase, disease, interventions, summaries, etc.)
<original_inclusion_requirement> (stringified list of inclusion items)
<original_exclusion_requirement> (stringified list of exclusion items)
For each individual criterion item (split the list items; do not merge unrelated bullets):
    1. Detect any contradiction where a patient who satisfies an inclusion item would be excluded by one or more exclusion items.
    2. If such a contradiction exists, revise the exclusion to make it precise and non-contradictory (e.g., time windows, severity qualifiers, etc.) using only information logically implied by the inclusion or explicitly present in <contextual_text>.
    3. If no contradiction exists, keep the original text unchanged in rewritten.
    4. Do not introduce new medical information (concepts, thresholds, etc.) not supported by the inclusion item or <contextual_text>.

=== INPUT ===
<contextual_text>
#CONTEXTUAL_TEXT# 
</contextual_text>

<original_inclusion_requirement>
#ORIGINAL_INCLUSION_REQUIREMENT#
</original_inclusion_requirement>

<original_exclusion_requirement>
#ORIGINAL_EXCLUSION_REQUIREMENT#
</original_exclusion_requirement>

=== GUIDELINES ===
1. Cross-check each inclusion item against all exclusion items for conflicts in Time windows, Severity/stage cut-offs, Subtypes of a disease, etc.
2. Rewrite only the conflicting exclusion(s) to resolve the conflict by adding minimal, precise qualifiers. 
3. Validate that after rewriting, any patient who meets an inclusion item is not automatically excluded by a conflicting exclusion (unless the context explicitly requires it).

=== OUTPUT (strict JSON only) ===
For each inclusion item and each exclusion item, return:
{
  "inclusion": [
    {
      "original": "<copy the original inclusion criterion item verbatim>",
      "rewritten": "<final inclusion text; identical to original unless you had to fix a clear error based on context>",
      "contradict_criterion": "<exact text of conflicting exclusion item if one existed; else \"\" >",
      "reason": "<1–2 sentences explaining the logic; if no conflict, say 'No conflict—retained as is.'>"
    }
  ],
  "exclusion": [
    {
      "original": "<copy the original exclusion criterion item verbatim>",
      "rewritten": "<final exclusion text; add clarifiers, time windows, severity, or subtype information as needed.>",
      "contradict_criterion": "<exact text of conflicting inclusion item if one existed; else \"\" >",
      "reason": "<1–2 sentences; if revised, state how the change resolves the contradiction without narrowing valid inclusions beyond context.>"
    }
  ]
}
"""


class RequirementContradictCriterionRewriter(dspy.Module):
    """LLM-powered contradiction resolver for cohort inclusion/exclusion lists."""

    def __init__(
        self,
        engine,
        *,
        prompt_template: str = DEFAULT_PROMPT,
        log_dir: str | pathlib.Path | None = "mbench/req_mbench/contradict_rewrite_logs",
        debug: bool = True,
    ):
        super().__init__()
        self.engine = engine
        self.prompt_template = prompt_template
        self.debug = debug
        self.log_dir = pathlib.Path(log_dir).expanduser() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Helpers ---------------------------------

    def _pick_prompt(self, *candidates: Dict[str, Any]) -> str:
        """Choose prompt source: cohort → top-level inc_ctx → top-level exc_ctx → default."""
        for src in candidates:
            if not isinstance(src, dict):
                continue
            p = src.get("RequirementContradictCriterionRewriter_prompt")
            if isinstance(p, str) and p.strip():
                return p
        return self.prompt_template

    def _cohort_key(self, c: Dict[str, Any], fallback_idx: int) -> str:
        return str(
            c.get("trial_id")
            or c.get("id")
            or c.get("substudy_id")
            or f"IDX{fallback_idx}"
        )

    def _cohorts_from_ctx(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Strictly use __cohort_contexts__ as source of truth for prompt inputs.
        Do NOT read enrollment_cohorts here.
        """
        sub = ctx.get("__cohort_contexts__")
        return list(sub) if isinstance(sub, list) else []


    def _build_prompt(self, tpl_src: Dict[str, Any], ctx_txt: str, inc_list: List[str], exc_list: List[str]) -> str:
        tpl = self._pick_prompt(tpl_src)
        return (
            tpl
            .replace("#CONTEXTUAL_TEXT#", ctx_txt)
            .replace("#ORIGINAL_INCLUSION_REQUIREMENT#", json.dumps(inc_list, ensure_ascii=False))
            .replace("#ORIGINAL_EXCLUSION_REQUIREMENT#", json.dumps(exc_list, ensure_ascii=False))
        )

    def _collect_rewrites(self, parsed: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        inc_items, exc_items = [], []
        inc = parsed.get("inclusion") if isinstance(parsed, dict) else None
        exc = parsed.get("exclusion") if isinstance(parsed, dict) else None
        if isinstance(inc, list):
            for obj in inc:
                inc_items.append(_squash_space(str((obj or {}).get("rewritten", "")).strip()))
        if isinstance(exc, list):
            for obj in exc:
                exc_items.append(_squash_space(str((obj or {}).get("rewritten", "")).strip()))
        return inc_items, exc_items

    def _write_back_one_side(
        self,
        *,
        ctx: Dict[str, Any],
        side: str,  # "inclusion" or "exclusion"
        cohort_idx_to_items: Dict[int, List[str]],
        raw_logs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        仅更新 __cohort_contexts__[i]["requirements"][j]["requirement"]，其余字段保持原样。
        不触碰 *_criteria 与 requirement_text。
        """
        sub = ctx.get("__cohort_contexts__")
        if isinstance(sub, list) and sub:
            sub2 = []
            for i, sc in enumerate(sub):
                sc2 = dict(sc)
                items = cohort_idx_to_items.get(i)
                if items and isinstance(sc2.get("requirements"), list):
                    # 根据 side 加前缀，再对齐写回 requirements[j]["requirement"]
                    final_items = _apply_preamble(items, inclusion=(side == "inclusion"))
                    req_list = sc2["requirements"]
                    if len(req_list) == len(final_items):
                        new_reqs = []
                        for j, r in enumerate(req_list):
                            if isinstance(r, dict):
                                r2 = dict(r)
                                r2["requirement"] = final_items[j]
                                new_reqs.append(r2)
                            else:
                                new_reqs.append(r)
                        sc2["requirements"] = new_reqs
                    # 若条数不一致则安全跳过，不修改
                sub2.append(sc2)
            ctx["__cohort_contexts__"] = sub2
            ctx["__substudy_contexts__"] = sub2  # 镜像保持一致

        # 不改 enrollment_cohorts / *_criteria / requirement_text
        ctx["contradiction_rewrite"] = raw_logs
        return ctx


    # ------------------------------ Public API -----------------------------

    def forward_pair(self, inc_ctx: Dict[str, Any], exc_ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Cross-side mode:
        - Pull inclusion list from inc_ctx, exclusion list from exc_ctx, per cohort.
        - Run LLM once per cohort.
        - Write rewritten inclusion back to inc_ctx; rewritten exclusion back to exc_ctx.
        """
        inc_cohorts = self._cohorts_from_ctx(inc_ctx)
        exc_cohorts = self._cohorts_from_ctx(exc_ctx)

        # 对齐：按 key 映射；退化为按索引
        def _index_by_key(cohorts: List[Dict[str, Any]]) -> Dict[str, Tuple[int, Dict[str, Any]]]:
            m: Dict[str, Tuple[int, Dict[str, Any]]] = {}
            for i, c in enumerate(cohorts):
                m[self._cohort_key(c, i)] = (i, c)
            return m

        inc_map = _index_by_key(inc_cohorts)
        exc_map = _index_by_key(exc_cohorts)
        # 取并集顺序：以 inclusion 的顺序为主，补上 exclusion 的孤儿
        ordered_keys: List[str] = list(inc_map.keys()) + [k for k in exc_map.keys() if k not in inc_map]

        inc_updates: Dict[int, List[str]] = {}
        exc_updates: Dict[int, List[str]] = {}
        raw_logs: List[Dict[str, Any]] = []

        for k in ordered_keys:
            i_inc, c_inc = inc_map.get(k, (None, None))
            i_exc, c_exc = exc_map.get(k, (None, None))

            # Strict: only subcontext contextual_text (inc 优先，其次 exc)，不再回退到 shared_context / 顶层
            ctx_txt = str(
                (c_inc or {}).get("contextual_text")
                or (c_exc or {}).get("contextual_text")
                or ""
            ).strip()


            # 取原始 per-side 列表
            inc_list = _extract_req_list_from_cohort((c_inc or {}), fallback_field="inclusion_criteria")
            exc_list = _extract_req_list_from_cohort((c_exc or {}), fallback_field="exclusion_criteria")


            # 构造 prompt（优先 cohort 上的 prompt，其次顶层 inc_ctx，再其次顶层 exc_ctx）
            tpl_src = (c_inc or {})
            if not tpl_src.get("RequirementContradictCriterionRewriter_prompt"):
                tpl_src = inc_ctx if inc_ctx.get("RequirementContradictCriterionRewriter_prompt") else exc_ctx
            prompt = self._build_prompt(tpl_src, ctx_txt, inc_list, exc_list)

            # LLM 调用
            raw_output = self.engine(prompt)[0]
            parsed = _safe_json_find(raw_output) or {}

            if self.debug:
                print(f"[contradict] cohort={k} parsed:", parsed)

            inc_rw, exc_rw = self._collect_rewrites(parsed)
            # 计数兜底：长度不匹配则回退
            if not inc_rw or len(inc_rw) != len(inc_list):
                inc_rw = inc_list[:]
            if not exc_rw or len(exc_rw) != len(exc_list):
                exc_rw = exc_list[:]

            # 暂存：由 _write_back_one_side 分别落地
            if i_inc is not None:
                inc_updates[i_inc] = inc_rw
            if i_exc is not None:
                exc_updates[i_exc] = exc_rw

            raw_logs.append({"cohort_key": k, "raw": str(raw_output), "parsed": parsed})

            # 可选日志落盘
            if self.log_dir:
                try:
                    stem = f"{k}.contradict"
                    (self.log_dir / f"{stem}.raw.txt").write_text(str(raw_output), encoding="utf-8")
                    (self.log_dir / f"{stem}.parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                    (self.log_dir / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
                except Exception as exc:
                    print(f"[Contradict] log write failed for {k}: {exc}")

        # 分别写回到两个 side 的上下文，并返回
        inc_ctx_out = self._write_back_one_side(ctx=inc_ctx, side="inclusion",
                                                cohort_idx_to_items=inc_updates, raw_logs=raw_logs)
        exc_ctx_out = self._write_back_one_side(ctx=exc_ctx, side="exclusion",
                                                cohort_idx_to_items=exc_updates, raw_logs=raw_logs)
        return inc_ctx_out, exc_ctx_out

    # -------- Legacy single-context API (kept for compatibility) ----------

    def _build_prompt_for_cohort(self, context, cohort: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        # Only subcontext contextual_text; no shared/top-level fallback
        ctx_txt = str(cohort.get("contextual_text") or "").strip()

        # Only structured requirements; do not read *_criteria/enrollment_cohorts
        def _reqs_only(c: Dict[str, Any]) -> List[str]:
            reqs = c.get("requirements")
            if not (isinstance(reqs, list) and reqs):
                return []
            out = []
            for r in reqs:
                if isinstance(r, dict):
                    t = _squash_space(str(r.get("requirement") or ""))
                    if t:
                        out.append(t)
            return out

        # 在单上下文模式里，cohort 就是单侧 subctx，所以两边列表相同并不影响格式要求
        inc_list = _reqs_only(cohort)
        exc_list = _reqs_only(cohort)

        # prompt 模板来源：优先 cohort 自带，其次类默认
        tpl_src = cohort if cohort.get("RequirementContradictCriterionRewriter_prompt") else context
        prompt = self._build_prompt(tpl_src, ctx_txt, inc_list, exc_list)
        return prompt, inc_list, exc_list



