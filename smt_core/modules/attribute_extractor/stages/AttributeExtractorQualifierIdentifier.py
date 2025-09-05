from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable
import copy

import dspy

from ..utils import _ChatEngine, _safe_id  # type: ignore


class AttributeExtractorQualifierIdentifier(dspy.Module):
    """
    Verifier-free qualifier identifier with batching + (NEW) verifier stage.

    需要的 ctx 键：
      - ctx["AttributeExtractorQualifierIdentifier_prompt"] : 包含 {requirement_entities}
      - ctx["requirement_entities_for_qualifier_identification"] : List[Dict]
      - ctx["AttributeExtractorQualifierIdentifierVerifier"] : 包含占位符 #REQUIREMENT_ENTITY_QUALIFIER# 的 verifier prompt（可选；缺失时跳过核查）
      - ctx["engine"] (可选) : _ChatEngine
      - ctx["trial_id"] (可选), ctx["inc_exc"] (可选) : 用于 mbench 目录

    产出：
      - ctx["requirement_entities_after_qualifier_identification"] : List[Dict]
      - （若提供 Verifier Prompt）：
          - ctx["requirement_canonical_entities_filtered_qualifiers"] : List[Dict]
      - mbench/attr_mbench/<trial_id>/<side>/
          - qualifier_identifier/prompts_outputs/{ts}_b{idx}_extractor_prompt.txt
          - qualifier_identifier/prompts_outputs/{ts}_b{idx}_extractor_raw_attempt{n}.txt
          - qualifier_identifier/requirement_entities_after_qualifier_identification.json
          - qualifier_identifier_verifier/prompts_outputs/{ts}_b{idx}_verifier_prompt.txt
          - qualifier_identifier_verifier/prompts_outputs/{ts}_b{idx}_verifier_raw_attempt{n}.txt
          - qualifier_identifier_verifier/verifier_merged_raw.json
          - qualifier_identifier_verifier/requirement_canonical_entities_filtered_qualifiers.json
    """

    def __init__(
        self,
        engine: _ChatEngine | None = None,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_retries: int = 2,
        batch_size: int = 5,
        mbench_path: str | os.PathLike = "mbench/attr_mbench/",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._default_engine = engine
        self.model = model
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.batch_size = int(batch_size)
        self.mbench_root = Path(mbench_path)
        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # ------------------------------ utils ------------------------------ #
    @staticmethod
    def _strip_code_fence(s: str) -> str:
        return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.S)

    @staticmethod
    def _best_effort_parse_array(s: str) -> List[Dict[str, Any]]:
        text = s.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass
        lidx = text.find("[")
        ridx = text.rfind("]")
        if lidx != -1 and ridx != -1 and ridx > lidx:
            candidate = text[lidx:ridx + 1].strip()
            try:
                obj = json.loads(candidate)
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass
        lidx = text.find("{")
        ridx = text.rfind("}")
        if lidx != -1 and ridx != -1 and ridx > lidx:
            candidate = text[lidx:ridx + 1].strip()
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return [obj]
            except Exception:
                pass
        raise ValueError("Failed to parse LLM output as JSON array (even after heuristics).")

    @staticmethod
    def _write_txt(dir_path: Path, filename: str, content: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(content, encoding="utf-8")

    @staticmethod
    def _write_json(file_path: Path, data: Any) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _pick_trial_folder(self, ctx: Dict[str, Any]) -> str:
        trial_id = str(ctx.get("trial_id") or "unknown_trial").strip()
        side = str(ctx.get("inc_exc") or "unknown_side").strip()
        return str(Path(_safe_id(trial_id)) / _safe_id(side))

    def _prompts_dir_and_agg_file(self, trial_folder: str) -> tuple[Path, Path]:
        base = self.mbench_root / trial_folder / "qualifier_identifier"
        prompts = base / "prompts_outputs"
        agg = base / "requirement_entities_after_qualifier_identification.json"
        prompts.mkdir(parents=True, exist_ok=True)
        base.mkdir(parents=True, exist_ok=True)
        return prompts, agg

    def _verifier_dirs(self, trial_folder: str) -> tuple[Path, Path, Path]:
        """Return (verifier_prompts_dir, verifier_root, verifier_raw_json_path)."""
        vroot = self.mbench_root / trial_folder / "qualifier_identifier_verifier"
        vprompts = vroot / "prompts_outputs"
        vraw = vroot / "verifier_merged_raw.json"
        vprompts.mkdir(parents=True, exist_ok=True)
        vroot.mkdir(parents=True, exist_ok=True)
        return vprompts, vroot, vraw

    @staticmethod
    def _chunks(items: List[Any], size: int) -> List[List[Any]]:
        if size <= 0:
            return [items]
        return [items[i:i + size] for i in range(0, len(items), size)]

    @staticmethod
    def _req_key(req_obj: Dict[str, Any], fallback_idx: Optional[int] = None) -> Any:
        for k in ("requirement_id", "id", "index", "idx"):
            if k in req_obj:
                return req_obj[k]
        if fallback_idx is not None:
            return fallback_idx
        return req_obj.get("requirement") or json.dumps(req_obj, ensure_ascii=False)

    @staticmethod
    def _dedup_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for e in ents or []:
            key = (e.get("entity_id"), e.get("extracted_span"), e.get("start"), e.get("end"), e.get("type"))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    # --------- verifier helpers --------- #
    @staticmethod
    def _iter_qualifier_entries(req_obj: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Yield (bucket_key, qualifier_dict) over all qualifier buckets in one requirement-entity."""
        buckets = ("Time", "Space", "Scale", "Source", "Cause", "Definition", "Other")
        for b in buckets:
            for q in (req_obj.get(b) or []):
                if isinstance(q, dict) and ("qualifier" in q or "qualifier_span" in q):
                    yield b, q

    @staticmethod
    def _flatten_for_verifier(requirement_entities_after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build verifier input array:
        [
          {
            "requirement_id": ...,
            "requirement": ...,
            "entity_qualifier": {
              entity fields..., qualifier, qualifier_span, rationale
            }
          },
          ...
        ]
        """
        flat: List[Dict[str, Any]] = []
        for req in requirement_entities_after or []:
            rid = req.get("requirement_id")
            rtext = req.get("requirement")
            for ent in (req.get("entities") or []):
                base_ent = {
                    "entity_id": ent.get("entity_id"),
                    "extracted_span": ent.get("extracted_span"),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "type": ent.get("type"),
                }
                for _, q in AttributeExtractorQualifierIdentifier._iter_qualifier_entries(ent):
                    flat.append({
                        "requirement_id": rid,
                        "requirement": rtext,
                        "entity_qualifier": {
                            **base_ent,
                            "qualifier": q.get("qualifier"),
                            "qualifier_span": q.get("qualifier_span"),
                            "rationale": q.get("rationale"),
                        },
                    })
        return flat

    @staticmethod
    def _vq_key(vq: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Build a stable key for a verifier item (and for merging back)."""
        eq = vq.get("entity_qualifier", {})
        return (
            vq.get("requirement_id"),
            eq.get("entity_id"),
            eq.get("start"),
            eq.get("end"),
            eq.get("type"),
            (eq.get("qualifier_span") or eq.get("qualifier")),
        )

    @staticmethod
    def _attach_verification_to_ctx_structure(
        requirement_entities_after: List[Dict[str, Any]],
        vq_items: List[Dict[str, Any]],
    ) -> None:
        """
        Mutates requirement_entities_after in place:
        For each qualifier dict, add a "verification" sub-dict with flags and explanation.
        """
        # Build lookup: for each qualifier object pointer, we need to match by the tuple key.
        index: Dict[Tuple[Any, Any, Any, Any, Any, Any], Dict[str, Any]] = {}

        for req in requirement_entities_after or []:
            rid = req.get("requirement_id")
            for ent in (req.get("entities") or []):
                ent_key_base = (rid, ent.get("entity_id"), ent.get("start"), ent.get("end"), ent.get("type"))
                for bucket_name, q in AttributeExtractorQualifierIdentifier._iter_qualifier_entries(ent):
                    key = (*ent_key_base, (q.get("qualifier_span") or q.get("qualifier")))
                    # annotate where it came from (optional)
                    q.setdefault("_bucket", bucket_name)
                    index[key] = q

        # Merge verification flags back
        for vq in vq_items or []:
            key = AttributeExtractorQualifierIdentifier._vq_key(vq)
            target = index.get(key)
            if not target:
                # try a softer fallback: match only by rid, entity_id, start, end, type (ignore qualifier span)
                soft_key = key[:-1]
                # find first with same 5-tuple
                for k2, qobj in index.items():
                    if k2[:-1] == soft_key:
                        target = qobj
                        break
            if not target:
                continue  # unmatched verifier line; skip

            eq = vq.get("entity_qualifier", {})
            verification = {
                "ALL_GOOD": eq.get("ALL_GOOD"),
                "explanation": eq.get("explanation"),
            }
            target["verification"] = verification

    @staticmethod
    def _filter_by_all_good(requirement_entities_after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return a deep-copied structure where any qualifier whose verification.ALL_GOOD != "YES" is removed.
        """
        kept = copy.deepcopy(requirement_entities_after)
        for req in kept:
            for ent in (req.get("entities") or []):
                for bucket in ("Time", "Space", "Scale", "Source", "Cause", "Definition", "Other"):
                    qlist = ent.get(bucket) or []
                    new_list = []
                    for q in qlist:
                        ok = True
                        ver = q.get("verification")
                        if ver and str(ver.get("ALL_GOOD")).upper() == "YES":
                            ok = True
                        else:
                            ok = False
                        if ok:
                            new_list.append(q)
                    ent[bucket] = new_list
        return kept

    # ------------------------------ main ------------------------------- #
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        engine: Optional[_ChatEngine] = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine found. Provide one via constructor or ctx['engine'].")

        prompt_template: str = ctx.get("AttributeExtractorQualifierIdentifier_prompt")  # must exist
        if not isinstance(prompt_template, str) or "{requirement_entities}" not in prompt_template:
            raise RuntimeError("ctx['AttributeExtractorQualifierIdentifier_prompt'] must be a str containing '{requirement_entities}'.")

        req_entities: List[Dict[str, Any]] = ctx.get("requirement_entities_for_qualifier_identification")  # type: ignore
        if not isinstance(req_entities, list):
            raise RuntimeError("ctx['requirement_entities_for_qualifier_identification'] must be a list.")

        # mbench 路径
        trial_folder = self._pick_trial_folder(ctx)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder)

        # 输入顺序（用于输出稳定排序）
        input_order_keys: List[Any] = [self._req_key(r, i) for i, r in enumerate(req_entities)]

        # 分 batch
        batches = self._chunks(req_entities, self.batch_size)

        # 合并容器：按 requirement_id（或兜底键）对齐
        merged: Dict[Any, Dict[str, Any]] = {}

        for b_idx, batch in enumerate(batches):
            # 构造本批 prompt
            payload_text = json.dumps(batch, ensure_ascii=False, indent=2)
            prompt = prompt_template.replace("{requirement_entities}", payload_text)

            ts = time.strftime("%Y%m%d-%H%M%S")
            base = f"{ts}_b{b_idx:04d}_extractor"

            # 记录 prompt
            self._write_txt(prompts_dir, f"{base}_prompt.txt", prompt)

            # LLM 调用 + 解析（带重试）
            last_err: Optional[Exception] = None
            parsed_batch: Optional[List[Dict[str, Any]]] = None

            for attempt in range(1, self.max_retries + 1):
                raw = engine(prompt, temperature=self.temperature)[0]
                self._write_txt(prompts_dir, f"{base}_raw_attempt{attempt}.txt", raw)
                try:
                    clean = self._strip_code_fence(raw)
                    parsed_batch = self._best_effort_parse_array(clean)
                    break
                except Exception as e:
                    last_err = e
                    continue

            if parsed_batch is None:
                raise RuntimeError(f"[batch {b_idx}] LLM output could not be parsed into JSON array. Last error: {last_err}")

            # 合并该 batch 的结果到 merged
            for idx_in_batch, req_obj in enumerate(parsed_batch):
                k = self._req_key(req_obj)
                cur = merged.get(k)
                if cur is None:
                    req_obj["entities"] = self._dedup_entities(req_obj.get("entities") or [])
                    merged[k] = req_obj
                else:
                    existing_entities = cur.get("entities") or []
                    new_entities = req_obj.get("entities") or []
                    cur["entities"] = self._dedup_entities(list(existing_entities) + list(new_entities))
                    if not cur.get("requirement") and req_obj.get("requirement"):
                        cur["requirement"] = req_obj["requirement"]

        # 按输入顺序输出；若有额外键（LLM 额外返回），追加在末尾
        ordered_keys = []
        seen_keys = set()
        for k in input_order_keys:
            if k in merged and k not in seen_keys:
                ordered_keys.append(k)
                seen_keys.add(k)
        for k in merged.keys():
            if k not in seen_keys:
                ordered_keys.append(k)
                seen_keys.add(k)

        final_list = [merged[k] for k in ordered_keys]

        # 写回 ctx 与 mbench（识别阶段）
        ctx["requirement_entities_after_qualifier_identification"] = final_list
        try:
            self._write_json(agg_json_path, final_list)
        except Exception as e:
            self.log.error("Failed to write aggregated JSON to %s: %s", agg_json_path, e)

        # ----------------------- NEW: Verifier 阶段 ----------------------- #
        verifier_template: Optional[str] = ctx.get("AttributeExtractorQualifierIdentifierVerifier_prompt")
        if isinstance(verifier_template, str) and "#REQUIREMENT_ENTITY_QUALIFIER#" in verifier_template:
            vprompts_dir, vroot, vraw_json = self._verifier_dirs(trial_folder)

            # 扁平化构造 verifier 输入
            flat_inputs = self._flatten_for_verifier(final_list)
            flat_batches = self._chunks(flat_inputs, self.batch_size)

            all_vq: List[Dict[str, Any]] = []
            for b_idx, vbatch in enumerate(flat_batches):
                payload = json.dumps(vbatch, ensure_ascii=False, indent=2)
                vprompt = verifier_template.replace("#REQUIREMENT_ENTITY_QUALIFIER#", payload)

                ts = time.strftime("%Y%m%d-%H%M%S")
                base = f"{ts}_b{b_idx:04d}_verifier"
                self._write_txt(vprompts_dir, f"{base}_prompt.txt", vprompt)

                last_err: Optional[Exception] = None
                parsed: Optional[List[Dict[str, Any]]] = None
                for attempt in range(1, self.max_retries + 1):
                    raw = engine(vprompt, temperature=self.temperature)[0]
                    self._write_txt(vprompts_dir, f"{base}_raw_attempt{attempt}.txt", raw)
                    try:
                        clean = self._strip_code_fence(raw)
                        parsed = self._best_effort_parse_array(clean)
                        break
                    except Exception as e:
                        last_err = e
                        continue

                if parsed is None:
                    raise RuntimeError(f"[verifier batch {b_idx}] LLM output parse failed. Last error: {last_err}")
                all_vq.extend(parsed)

            # 存所有 verifier raw 合并
            try:
                self._write_json(vraw_json, all_vq)
            except Exception as e:
                self.log.error("Failed to write verifier raw json: %s", e)

            # 合并核查标记回原结构
            self._attach_verification_to_ctx_structure(final_list, all_vq)

            # 过滤 ALL_GOOD == "YES"
            filtered = self._filter_by_all_good(final_list)

            # 写 ctx + mbench
            ctx["requirement_canonical_entities_filtered_qualifiers"] = filtered
            try:
                self._write_json(vroot / "requirement_canonical_entities_filtered_qualifiers.json", filtered)
            except Exception as e:
                self.log.error("Failed to write filtered qualifiers json: %s", e)

        else:
            # 未提供 verifier prompt：不做核查与筛选
            self.log.info("Verifier prompt missing or without placeholder; skipping verification stage.")

        return ctx
