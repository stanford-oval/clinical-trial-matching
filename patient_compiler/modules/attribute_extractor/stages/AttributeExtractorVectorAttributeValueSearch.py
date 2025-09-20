# modules/stages/vector_attribute_value_search.py
from __future__ import annotations
import os, re, logging, json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional, List

import dspy
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
from elasticsearch import NotFoundError, TransportError

def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

class AttributeExtractorVectorAttributeValueSearch(dspy.Module):
    """
    Stage 2: 为每个 attribute_interpretation 的 (attribute_type, attribute_value)
    检索 SNOMED 候选，回写到：
      qualifier.attribute_interpretations[*]["potential_matches_by_attribute"]

    新日志规范：仅输出一份 group 聚合文件（覆盖写入）
      mbench/attr_mbench/<Trial id>/g<gid>/3vec_search.json
    内容为处理后的 context["groups"][gid]。
    """

    _TAG_RE = re.compile(r"\(([^()]+)\)$", re.I)
    _ATTR_CACHE: dict[str, Dict[str, str]] = {}   # path → {name: id}

    def __init__(
        self,
        *,
        es_url: str,
        attr_map_path: os.PathLike | str | None = None,
        index_tpl: str = "snomed_vectors_{attr_id}",
        model_name: str = "cambridgeltl/sapbert-from-pubmedbert-fulltext",
        top_k: int = 3,
        score_cut: float = 0.30,
        overshoot: int = 4,
        verbose: bool = False,
        mbench_path: os.PathLike | str = "mbench/attr_mbench/",
    ):
        super().__init__()
        # infra
        self.es    = Elasticsearch(es_url, request_timeout=20)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        self.top_k       = top_k
        self.cut         = score_cut
        self.raw_k       = top_k * max(1, overshoot)
        self.index_tpl   = index_tpl
        self.es_major    = int(self.es.info()["version"]["number"].split(".")[0])

        # attribute LUT (may be injected later) : name -> id
        self.attr_name2id: Dict[str, str] = {}
        if attr_map_path is not None:
            self._load_attr_map(attr_map_path)

        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

        # 新目录根
        self.mbench_root = Path(mbench_path)

    # ------------------------------ loaders ------------------------------ #
    @classmethod
    def _load_attr_map(cls, path: os.PathLike | str) -> Dict[str, str]:
        p = os.fspath(path)
        if p not in cls._ATTR_CACHE:
            df = pd.read_excel(p, engine="openpyxl", dtype=str)
            df.columns = [c.lower().strip() for c in df.columns]
            # 期望列名：attributeid, attributefsn
            id_col  = "attributeid"
            name_col = "attributefsn"
            if id_col in df.columns and name_col in df.columns:
                cls._ATTR_CACHE[p] = dict(zip(df[name_col].str.strip(), df[id_col].str.strip()))
            else:
                raise ValueError(f"Excel 缺少必要列: {df.columns}")
        return cls._ATTR_CACHE[p]

    # --------------------------- ES query builders ------------------------ #
    def _knn_query(self, vec: List[float], k: int) -> Dict[str, Any]:
        return {
            "size": k,
            "query": {
                "knn": {
                    "field": "vector",
                    "query_vector": vec,
                    "k": k,
                    "num_candidates": max(1_000, k * 10),
                }
            },
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }

    def _script_query(self, vec: List[float], k: int) -> Dict[str, Any]:
        return {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.qv, 'vector') + 1.0",
                        "params": {"qv": vec},
                    },
                }
            },
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }

    def _make_query(self, vec: List[float], k: int) -> Dict[str, Any]:
        return self._knn_query(vec, k) if self.es_major >= 8 else self._script_query(vec, k)

    # ------------------------------ PT/FSN helpers ----------------------- #
    def _fetch_pt_fsn(self, cid: str, index_name: str) -> tuple[str, str]:
        """从 ES 描述行推断 PT / FSN。"""
        term_fields = ["term", "is_pt", "is_fsn", "typeId", "description_type"]
        q = {
            "size": 1000,
            "query": {
                "bool": {
                    "should": [
                        {"term": {"concept_id.keyword": cid}},
                        {"term": {"sctid.keyword": cid}},
                        {"term": {"concept_id": cid}},
                        {"term": {"sctid": cid}},
                    ]
                }
            },
            "_source": term_fields,
        }
        hits = self.es.search(index=index_name, body=q)["hits"]["hits"]

        pt, fsn, tagged = "", "", None
        for h in hits:
            s = h["_source"]
            term = s["term"]
            is_pt  = s.get("is_pt") or s.get("description_type") == "pt" \
                     or s.get("typeId") == "900000000000013009"
            is_fsn = s.get("is_fsn") or s.get("description_type") == "fsn" \
                     or s.get("typeId") == "900000000000003001"

            if is_pt and not pt:
                pt = term
            if is_fsn and not fsn:
                fsn = term
            if not tagged and self._TAG_RE.search(term):
                tagged = term
            if pt and fsn:
                break

        if not fsn and tagged:
            fsn = tagged
        if not pt and hits:
            pt = hits[0]["_source"]["term"]
        if not fsn and pt:
            fsn = pt
        return pt, fsn

    # -------------------------- vector search core ------------------------ #
    def _search_attr_values(self, attr_id: str, text: str) -> list[dict[str, Any]]:
        """仅使用属性专属索引；索引不存在则跳过。"""
        index_name = self.index_tpl.format(attr_id=attr_id)

        if not self.es.indices.exists(index=index_name):
            self.log.warning("ES index %s missing – skip value «%s»", index_name, text)
            return []

        q_vec = self.model.encode([str(text)], normalize_embeddings=True)[0]
        try:
            hits = self.es.search(
                index=index_name,
                body=self._make_query(q_vec.tolist(), self.raw_k)
            )["hits"]["hits"]
        except (NotFoundError, TransportError) as e:
            self.log.error("ES search error on %s: %s – skip «%s»", index_name, getattr(e, "info", e), text)
            return []

        by_cid: dict[str, dict[str, Any]] = {}
        for h in hits:
            if h.get("_score", 0.0) < self.cut:
                continue
            src  = h.get("_source", {})
            cid  = src.get("concept_id") or src.get("sctid")
            if not cid:
                continue
            score = float(h["_score"])
            term  = src.get("term", text)

            rec = by_cid.setdefault(
                cid,
                dict(
                    conceptId            = cid,
                    preferred_term       = "",
                    fully_specified_name = "",
                    best_match_term      = term,
                    match_score          = score,
                ),
            )
            if score > rec["match_score"]:
                rec["match_score"]     = score
                rec["best_match_term"] = term

        if not by_cid:
            return []

        out: list[dict[str, Any]] = []
        for cid, rec in by_cid.items():
            pt, fsn = self._fetch_pt_fsn(cid, index_name)
            rec["preferred_term"]       = pt or rec["best_match_term"]
            rec["fully_specified_name"] = fsn or rec["best_match_term"]
            out.append(rec)

        out.sort(key=lambda r: (-r["match_score"], r["preferred_term"]))
        return out[: self.top_k]

    # ---------------------- iterate over new attributes ------------------- #
    def _iter_attr_items(self, member: Dict[str, Any]) -> Iterable[Tuple[Optional[str], Optional[str], Dict[str, Any]]]:
        """
        仅遍历新结构：qualifiers → attribute_interpretations
        产出 (attr_name, attr_value, sink_dict)；写回到 sink_dict["potential_matches_by_attribute"]。
        """
        for q in member.get("all_qualifying_information_related_to_the_entity", []) or []:
            for ai in q.get("attribute_interpretations", []) or []:
                yield ai.get("attribute_type"), ai.get("attribute_value"), ai

    # ----------------------- enrich a single group ------------------------ #
    def _enrich_group(self, group: Dict[str, Any], *, allowed_attrs: set[str]) -> None:
        """
        仅按 allowed_attrs（名称或 ID）过滤；只用属性专属索引。
        不再写任何中间日志。
        """
        for member in group.get("members", []) or []:
            quals = member.get("all_qualifying_information_related_to_the_entity", []) or []
            for q in quals:
                interps = q.get("attribute_interpretations", []) or []
                for ai in interps or []:
                    if not isinstance(ai, dict):
                        continue
                    if "potential_matches_by_attribute" in ai:
                        # 幂等：已有则跳过
                        continue

                    attr_name = ai.get("attribute_type")
                    val       = ai.get("attribute_value")
                    if not attr_name or not val:
                        continue

                    # 名称→ID（ctx 注入的 LUT 优先）
                    attr_id = self.attr_name2id.get(attr_name, None)

                    # allowed 既可能是名称也可能是ID
                    if (attr_name not in allowed_attrs) and (not attr_id or attr_id not in allowed_attrs):
                        continue

                    if not attr_id:
                        self.log.info("No attribute_id for «%s»; skip value «%s»", attr_name, val)
                        continue

                    index_name = self.index_tpl.format(attr_id=attr_id)
                    if not self.es.indices.exists(index=index_name):
                        self.log.warning("ES index %s missing – skip value «%s»", index_name, val)
                        continue

                    # 正常检索
                    cands = self._search_attr_values(attr_id, str(val))

                    # 写回 sink
                    ai["potential_matches_by_attribute"] = {
                        attr_name: {"id": attr_id, "candidates": cands}
                    }

    # ----------------------------- path helpers --------------------------- #
    @staticmethod
    def _pick_trial_folder(ctx: Dict[str, Any], members: list[Dict[str, Any]]) -> str:
        """
        与其它阶段保持一致：
        优先 ctx['trial_id']；否则从第一个有 trial 的 member 取 NCT 前缀（下划线前）。
        """
        t = str(ctx.get("note_id") or "").strip()
        if t:
            return t
        for m in members or []:
            tr = str(m.get("patient_note") or "").strip()
            if tr:
                return tr.split("_", 1)[0]
        return "unknown_trial"

    def _target_path(self, trial_folder: str, gid: int) -> Path:
        base = self.mbench_root / _safe_id(trial_folder) / f"g{gid:04d}"
        base.mkdir(parents=True, exist_ok=True)
        return base / "3vec_search.json"

    # ------------------------------- forward ------------------------------ #
    def forward(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:  # type: ignore[override]
        """
        仅处理 ctx["groups"][index]，检索候选并回写到
        qualifier.attribute_interpretations[*]["potential_matches_by_attribute"]。
        然后把整个 group 覆盖写入 3vec_search.json。
        """
        # 1) 注入 LUT：ctx["attribute_id_to_name"] 是 id->name，需要翻转为 name->id
        if not self.attr_name2id and "attribute_id_to_name" in ctx:
            try:
                self.attr_name2id = {v: k for k, v in ctx["attribute_id_to_name"].items()}
            except Exception as e:
                self.log.error("attribute_id_to_name malformed: %s", e)

        # 2) 取组与 allowed 集
        groups = ctx.get("groups") or []
        if not (isinstance(groups, list) and 0 <= index < len(groups)):
            self.log.warning("groups[%s] not found; skip", index)
            return ctx

        group = groups[index]
        allowed = {str(a.get("AttributeType", "")).strip()
                   for a in group.get("allowed_attributes", []) if a.get("AttributeType")}
        # allowed 也可能包含 ID；并入
        allowed |= {self.attr_name2id.get(name, name) for name in list(allowed)}

        # 3) 检索并回写到 ctx
        self._enrich_group(group, allowed_attrs=allowed)

        # 4) 稳定排序（保持与其它阶段一致）
        group["members"] = sorted(group.get("members", []),
                                  key=lambda m: (m.get("patient_note",""), m.get("requirement","")))

        # 5) 覆盖写入 3vec_search.json（内容为整个 group）
        trial_folder = self._pick_trial_folder(ctx, group.get("members", []))
        target = self._target_path(trial_folder, index)
        try:
            target.write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error("Failed to write %s: %s", target, e)

        return ctx
