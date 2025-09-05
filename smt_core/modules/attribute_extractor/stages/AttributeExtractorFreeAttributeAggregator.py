from __future__ import annotations
import json, logging, re, hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import datetime as dt

import dspy


def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

def _sha12(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]


class AttributeExtractorFreeAttributeAggregator(dspy.Module):
    """
    在所有 groups 运行完后调用：
      - 汇总 ctx["groups"]，生成 RequirementID(0..N-1) → Entity → Qualifiers 的聚合
      - 保障“空 requirement”和“无 qualifier 的 entity”也被写出
      - mbench 输出： mbench/attr_mbench/<trial_id>/<inc_exc>/free_translater_agg_by_requirement.json
      - context： ctx["free_attr_agg_by_requirement_all"]
    """

    def __init__(self, *, mbench_path: str | Path = "mbench/attr_mbench/", verbose: bool = False):
        super().__init__()
        self.mbench_root  = Path(mbench_path)
        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # ------------------------ signatures & pickers ------------------------ #
    @staticmethod
    def _entity_sig(ent: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[int]]:
        surf = str(ent.get("surface_string", "")).strip().lower()
        st   = ent.get("start")
        ed   = ent.get("end")
        return (surf, st, ed)

    @staticmethod
    def _qual_sig(q: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[int]]:
        # 允许没有 start/end：缺失时用 (text, None, None)
        span = str(q.get("qualifier_span", "")).strip().lower()
        st   = q.get("start")
        ed   = q.get("end")
        st = None if (st is None or st == "") else st
        ed = None if (ed is None or ed == "") else ed
        if st is None or ed is None:
            return (span, None, None)
        return (span, st, ed)

    @staticmethod
    def _pick_trial_folder(ctx: Dict[str, Any], groups: List[Dict[str, Any]]) -> str:
        t = str(ctx.get("trial_id") or "").strip()
        if t:
            return t
        for g in groups or []:
            for m in (g.get("members") or []):
                tr = str(m.get("trial") or "").strip()
                if tr:
                    return tr.split("_", 1)[0]
        return "unknown_trial"

    # ---------------------- requirement 索引收集/预填 ---------------------- #
    @staticmethod
    def _intish(x: Any) -> Optional[int]:
        try:
            return int(str(x).strip())
        except Exception:
            return None

    def _prefill_requirements(self, ctx: Dict[str, Any], groups: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        返回 { rid(str数字): requirement_text(str，可能为空) }，连续 0..N-1。
        优先来源顺序：
          1) ctx["requirements_echo"] 的索引和文本
          2) ctx["requirements"] 的下标和文本
          3) 从 member["trial"] 解析 reqNNN → N，再用 member["requirement"] 补文本
             并将 0..maxN 全部补齐（缺文本则置 ""）
        """
        # 1) requirements_echo
        echo = ctx.get("requirements_echo")
        if isinstance(echo, dict) and echo:
            pairs = []
            for k, v in echo.items():
                ki = self._intish(k)
                if ki is not None:
                    pairs.append((ki, str(v or "")))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                return {str(i): txt for i, txt in pairs}

        # 2) requirements 列表
        reqs = ctx.get("requirements")
        if isinstance(reqs, list) and reqs:
            out: Dict[str, str] = {}
            for i, r in enumerate(reqs):
                # 尽力提取文本
                if isinstance(r, str):
                    txt = r
                elif isinstance(r, dict):
                    txt = str(r.get("requirement") or r.get("text") or r.get("sentence") or "")
                else:
                    txt = str(r)
                out[str(i)] = txt
            return out

        # 3) 从 members 的 trial 解析 req 编号
        seen: Dict[int, str] = {}
        max_id = -1
        for g in groups or []:
            for m in (g.get("members") or []):
                trial_tag = str(m.get("trial", "") or "")
                rid = None
                if trial_tag:
                    # 优先匹配末尾/片段中的 reqNNN
                    tokens = [t for t in trial_tag.split("_") if t]
                    for tok in reversed(tokens):
                        m1 = re.match(r"(?i)^req[-_]?(\d+)\D*$", tok)
                        if m1:
                            rid = int(m1.group(1)); break
                    if rid is None:
                        m2 = re.search(r"(?i)req[-_]?(\d+)", trial_tag)
                        if m2: rid = int(m2.group(1))
                if rid is None:
                    continue
                txt = str(m.get("requirement", "") or "")
                if rid not in seen:
                    seen[rid] = txt
                elif txt and not seen[rid]:
                    seen[rid] = txt
                max_id = max(max_id, rid)

        if max_id >= 0:
            # 连续补齐 0..max_id
            return {str(i): seen.get(i, "") for i in range(max_id + 1)}

        # 实在没有任何线索：返回空
        return {}

    # ------------------------------- core -------------------------------- #
    def _aggregate(self, groups: List[Dict[str, Any]], rid2text: Dict[str, str]) -> Dict[str, Any]:
        # 先预填所有 requirement（确保空 req 也写出）
        by_req: Dict[str, Dict[str, Any]] = {
            rid: {
                "requirement_id": rid,
                "requirement_texts": [txt] if txt else [],
                "entities_by_sig": {},
                "entities": [],
            }
            for rid, txt in rid2text.items()
        }

        total_entities = 0
        total_quals = 0

        # 填充实体与 qualifiers
        for gid, group in enumerate(groups):
            for m in (group.get("members") or []):
                # 解析 requirement id（数字字符串）
                rid = None
                # 优先：从 trial 的 reqNNN 得到
                trial_tag = str(m.get("trial", "") or "")
                if trial_tag:
                    tokens = [t for t in trial_tag.split("_") if t]
                    for tok in reversed(tokens):
                        m1 = re.match(r"(?i)^req[-_]?(\d+)\D*$", tok)
                        if m1:
                            rid = str(int(m1.group(1))); break
                    if rid is None:
                        m2 = re.search(r"(?i)req[-_]?(\d+)", trial_tag)
                        if m2: rid = str(int(m2.group(1)))
                # 次优：如果 ctx 预填了 rid2text，就用 member 的 requirement 文本去匹配（不可靠，尽量避免）
                if rid is None and rid2text:
                    # 尝试根据文本找到相等的 req；否则放弃（保底加在 "0"）
                    req_txt = str(m.get("requirement", "") or "")
                    for k, v in rid2text.items():
                        if req_txt and req_txt == v:
                            rid = k; break
                if rid is None:
                    rid = "0"  # 兜底

                req_txt = str(m.get("requirement", "") or "")
                rb = by_req.setdefault(rid, {
                    "requirement_id": rid,
                    "requirement_texts": [req_txt] if req_txt else [],
                    "entities_by_sig": {},
                    "entities": [],
                })
                if req_txt and req_txt not in rb["requirement_texts"]:
                    rb["requirement_texts"].append(req_txt)

                ent = m.get("entity", {}) or {}
                esig = self._entity_sig(ent)
                eb = rb["entities_by_sig"].setdefault(esig, {
                    "entity": {
                        "surface_string": ent.get("surface_string", ""),
                        "preferred_term": ent.get("preferred_term", ""),
                        "fully_specified_name": ent.get("fully_specified_name", ""),
                        "type": ent.get("type", ""),
                        "definition": ent.get("definition", ""),
                        "conceptId": ent.get("conceptId", ""),
                        "start": ent.get("start"),
                        "end": ent.get("end"),
                    },
                    "all_qualifying_information": [],
                    "_qual_seen": set(),
                })

                for q in (m.get("all_qualifying_information_related_to_the_entity") or []):
                    qsig = self._qual_sig(q)
                    if qsig in eb["_qual_seen"]:
                        continue
                    eb["_qual_seen"].add(qsig)
                    eb["all_qualifying_information"].append({
                        "start": q.get("start"),
                        "end":   q.get("end"),
                        "qualifier_span": q.get("qualifier_span", ""),
                        "qualifier": q.get("qualifier", ""),
                        "attribute_interpretations_free": q.get("attribute_interpretations_free", []) or [],
                        "source_groups": [gid],
                    })
                    total_quals += 1

        # 收尾：把 entities_by_sig → entities 列表，稳定排序
        counts_by_requirement: Dict[str, Dict[str, int]] = {}
        # 用数值顺序输出 rid
        for rid in sorted(by_req.keys(), key=lambda k: int(k) if str(k).isdigit() else 10**9):
            rb = by_req[rid]
            ents = []
            for _, rec in rb.get("entities_by_sig", {}).items():
                rec.pop("_qual_seen", None)
                rec["all_qualifying_information"] = sorted(
                    rec["all_qualifying_information"],
                    key=lambda x: (str(x.get("qualifier_span","")).lower(),
                                   x.get("start") if x.get("start") is not None else -1,
                                   x.get("end")   if x.get("end")   is not None else -1)
                )
                ents.append(rec)
            ents = sorted(
                ents,
                key=lambda r: (str(r["entity"].get("surface_string","")).lower(),
                               r["entity"].get("start") if r["entity"].get("start") is not None else -1,
                               r["entity"].get("end")   if r["entity"].get("end")   is not None else -1)
            )
            rb["entities"] = ents
            rb.pop("entities_by_sig", None)

            counts_by_requirement[rid] = {
                "entities": len(ents),
                "qualifiers": sum(len(e["all_qualifying_information"]) for e in ents),
            }
            total_entities += len(ents)

        agg = {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "by_requirement": {rid: by_req[rid] for rid in sorted(by_req.keys(), key=lambda k: int(k) if str(k).isdigit() else 10**9)},
            "counts": {
                "requirements": len(by_req),
                "entities": total_entities,
                "qualifiers": total_quals,
            },
            "counts_by_requirement": counts_by_requirement,
        }
        return agg

    # ------------------------------- API --------------------------------- #
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        groups: List[Dict[str, Any]] = ctx.get("groups", [])
        if not groups:
            ctx["free_attr_agg_by_requirement_all"] = {
                "generated": dt.datetime.now().isoformat(timespec="seconds"),
                "by_requirement": {},
                "counts": {"requirements": 0, "entities": 0, "qualifiers": 0},
                "counts_by_requirement": {},
            }
            return ctx

        # 预收集 requirement 索引与文本，确保 0..N-1 全写
        rid2text = self._prefill_requirements(ctx, groups)

        # 输出仍按 trial/inc_exc 放目录，文件名 *_by_requirement.json
        trial_folder = self._pick_trial_folder(ctx, groups)
        inc_exc = _safe_id(str(ctx.get("inc_exc","")))
        out_dir = self.mbench_root / _safe_id(trial_folder) / inc_exc
        out_dir.mkdir(parents=True, exist_ok=True)

        agg_struct = self._aggregate(groups, rid2text)

        # 写 mbench
        out_path = out_dir / "free_translater_agg_by_requirement.json"
        try:
            out_path.write_text(json.dumps(agg_struct, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error("Failed to write free_translater_agg_by_requirement.json: %s", e)

        # 写 context
        ctx["free_attr_agg_by_requirement_all"] = agg_struct
        return ctx
