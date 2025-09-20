# psrc/modules/EntityCanonicalizer/stages/LLMBasedMedicalEntityFilter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLMBasedMedicalEntityFilter — linker + verifier (arbiter removed)
────────────────────────────────────────────────────────────────
- Distinguishes each occurrence by (text,start,end)
- Collapses exact duplicate occurrences of (text,start,end)
- Runs LLM linker then LLM verifier
- **No arbitration**: all verifier-KEEP pairs are accepted
- Writes into ctx["valid_entities_by_req"][str(idx)] with keys "entity_1", "entity_2", ...

Schema of each entity:
{
  "extracted_span": "...",
  "entity_name": "...",
  "preferred_term": "...",
  "fully_specified_name": "...",
  "type": "Procedure|Clinical finding|...",
  "conceptId": "...",
  "start": 123,
  "end": 129
}
- 另外：把每个 mention 的向量候选（去掉最终 KEEP 的 concept）保存到
  ctx["entity_other_canonical_candidates"][str(idx)]，并写入 sigir-20141 目录。

"""

from __future__ import annotations
from typing import List, Dict, Any, Callable, Iterable, Tuple
import json, pathlib, logging, datetime as dt
import dspy

# ───────────────────────── CONFIG ─────────────────────────
BATCH1_SIZE       = 25   # linker batch size
VERIFY_BATCH_SIZE = 20   # verifier batch size
MAX_LLM_ATTEMPTS  = 3
_LOG_TO_STDOUT    = True

def _asbool(x, default=False):
    if x is None: return default
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.strip().lower() in ("1","true","yes","on","y","t")
    return default

# ────────────────────── LOGGING HELPERS ───────────────────
def _log(msg: str) -> None:
    (print if _LOG_TO_STDOUT else logging.getLogger("entity_filter").info)(msg)

def _write_txt(p: pathlib.Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ───────────────────── DEFAULT PROMPTS ────────────────────
_LINKER_PROMPT = """\
# === ROLE ===
You are a SNOMED concept linker. For EACH mention choose ONE best candidate or return null.

# === INPUT ===
<criterion>
#CRITERION#
</criterion>

<mentions>
#CANDIDATES#
</mentions>

# === OUTPUT (JSON array) ===
[
  {
    "extracted_span": "...",
    "offset": [123,129], // copy exactly
    "concept": {
      "conceptId": "123456",
      "preferred_term": "...",
      "fully_specified_name": "...",
      "top_type": "Clinical finding"
    }
  }
]
"""

_VERIFY_PROMPT = """\
# === ROLE ===
You are a SNOMED link auditor. Return "KEEP" iff the concept’s meaning matches the extracted_span **at that offset**.
Otherwise "REJECT".

# === INPUT ===
<criterion>
#CRITERION#
</criterion>

<pairs>
#PAIRS#
</pairs>

# === OUTPUT ===
[
  {
    "extracted_span": "...",
    "offset": [123,129],
    "decision": "KEEP|REJECT",
    "why": "≤20 words if REJECT else empty"
  }
]
"""

SPAN_FIELD = "extracted_span"

# ──────────────── META-LOOKUP (Python side) ───────────────
_CONCEPT_CACHE: Dict[str, Dict[str, Any]] = {}  # conceptId → {"synonyms", "definition"}

def _get_meta(concept_id: str, cand_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if concept_id in _CONCEPT_CACHE:
        return _CONCEPT_CACHE[concept_id]
    for cand in cand_list:
        if cand.get("conceptId") == concept_id:
            meta = {
                "synonyms": cand.get("all_terms", [])[:15],
                "definition": cand.get("definition", "")
            }
            _CONCEPT_CACHE[concept_id] = meta
            return meta
    meta = {"synonyms": [], "definition": ""}
    _CONCEPT_CACHE[concept_id] = meta
    return meta

# ──────────────────── SCHEMA HELPERS ──────────────────────
_Key = Tuple[str, int, int]  # (text,start,end)
def _make_key(text: str, start: int, end: int) -> _Key:
    return (text, start, end)

def _unwrap_linker_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "concept" in obj:  # flat
        return obj
    if len(obj) == 1:     # wrapped
        return next(iter(obj.values()))
    raise ValueError(f"Unrecognised linker object shape: {obj!r}")

# ────────────────────── UTILITY FUNCS ─────────────────────
def _raw(req: Any) -> str:
    if isinstance(req, str): return req
    if isinstance(req, dict):
        for k in ("requirement", "requirement_text", "sentence"):
            if req.get(k): return str(req[k])
    return json.dumps(req, ensure_ascii=False)

def _chunks(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

# ─────────────────────── MAIN MODULE ──────────────────────
class LLMBasedMedicalEntityFilter(dspy.Module):
    def __init__(
        self,
        engine: Callable[[str], List[str]],
        batch1_size: int = BATCH1_SIZE,
        verify_batch_size: int = VERIFY_BATCH_SIZE,
        max_attempts: int = MAX_LLM_ATTEMPTS,
    ):
        super().__init__()
        self.engine = engine
        self.batch1_size = batch1_size
        self.verify_batch_size = verify_batch_size
        self.max_attempts = max_attempts

    def _call_and_log(self, prompt: str, expect_array: bool, prefix: pathlib.Path) -> Any:
        _write_txt(prefix.parent / f"{prefix.name}_prompt.txt", prompt)
        for attempt in range(self.max_attempts):
            out = self.engine(prompt)[0]
            if attempt == 0:
                _write_txt(prefix.parent / f"{prefix.name}_raw.txt", out)
            if expect_array:
                first, last = out.find("["), out.rfind("]")
                if first != -1 and last != -1 and last > first:
                    out = out[first:last+1]
            try:
                return json.loads(out.strip())
            except Exception:
                continue
        raise RuntimeError("LLM returned ill-formed JSON after retries")

    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        idx       = ctx["current_requirement_index"]
        criterion = _raw(ctx["requirements"][idx])

        # surface entities (recognizer + optional dict)
        llm_ents  = ctx.get("llm_surface_entities_by_req",  {}).get(idx, [])
        dict_ents = ctx.get("dict_surface_entities_by_req", {}).get(idx, [])
        if not (llm_ents or dict_ents):
            ctx.setdefault("valid_entities_by_req", {})[str(idx)] = {}
            return ctx

        # ────── EXPLODE into one-occurrence-per-item ──────
        exploded: List[Dict[str, Any]] = []
        for ent in llm_ents + dict_ents:
            offs = ent.get("all_offsets")
            if offs:
                for off in offs:
                    exploded.append({**ent, "start": off["start"], "end": off["end"]})
            else:  # back-compat
                exploded.append(ent)

        # Collapse exact duplicates by (text,start,end)
        uniq_exploded: Dict[_Key, Dict[str, Any]] = {}
        for e in exploded:
            key = _make_key(e["text"], e["start"], e["end"])
            if key not in uniq_exploded:
                uniq_exploded[key] = e
        if len(uniq_exploded) < len(exploded):
            _log(f"[exploded] collapsed {len(exploded) - len(uniq_exploded)} duplicate occurrence(s)")
        exploded = list(uniq_exploded.values())

        # templates (ctx may override)
        linker_tpl   = ctx.get("LLMBasedMedicalEntityFilterLinker_prompt",   _LINKER_PROMPT)
        verifier_tpl = ctx.get("LLMBasedMedicalEntityFilterVerifier_prompt", _VERIFY_PROMPT)

        # identifiers for log filenames (harmless if absent)
        note_id  = ctx.get("note_id",  "note")
        side     = ctx.get("inc_exc",  "side")

        # ───────── Stage 1 : Concept linker ─────────
        concept_map: Dict[_Key, Dict[str, Any]] = {}

        def _reconcile_linker(batch: List[Dict[str, Any]], res: Any) -> Dict[_Key, Dict[str, Any] | None]:
            batch_by_key = { _make_key(e["text"], e["start"], e["end"]): e for e in batch }
            seen: Dict[_Key, Dict[str, Any] | None] = {}
            for raw in res:
                r = _unwrap_linker_obj(raw)
                if SPAN_FIELD not in r or "offset" not in r:
                    continue
                offset = r["offset"]
                if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                    continue
                start, end = int(offset[0]), int(offset[1])
                key = _make_key(r[SPAN_FIELD], start, end)
                if key not in batch_by_key:
                    continue
                concept = r.get("concept")
                if key in seen:
                    prev = seen[key]
                    if prev and concept and prev.get("conceptId") != concept.get("conceptId"):
                        raise ValueError(f"Linker produced conflicting concepts for {r[SPAN_FIELD]} [{start},{end}]")
                    if (prev is None) and concept:
                        seen[key] = concept
                else:
                    seen[key] = concept
            for key in batch_by_key:
                if key not in seen:
                    seen[key] = None
            return seen

        for b_no, batch in enumerate(_chunks(exploded, self.batch1_size), 1):
            prompt = linker_tpl.replace("#CRITERION#", criterion).replace(
                "#CANDIDATES#", json.dumps({
                    f"entity_{i}": {
                        "text": e["text"],
                        "entity_name": e["entity_name"],
                        "offset": [e["start"], e["end"]],
                        "candidates": e.get("candidates", []),
                    } for i, e in enumerate(batch, 1)
                }, indent=2, ensure_ascii=False)
            )
            pfx = pathlib.Path("mbench/entity_mbench/entity_logs",
                               f"{note_id}/{side}_req{idx:03d}_stage1b{b_no:02d}")
            res = self._call_and_log(prompt, True, pfx)
            aligned = _reconcile_linker(batch, res)
            for e in batch:
                key = _make_key(e["text"], e["start"], e["end"])
                concept = aligned.get(key)
                if not concept:
                    continue
                cand_list = e.get("candidates", [])
                c = {**concept, **_get_meta(concept.get("conceptId", ""), cand_list)}
                c["entity_name"] = e.get("entity_name", "")
                concept_map[key] = c

        if not concept_map:
            ctx.setdefault("valid_entities_by_req", {})[str(idx)] = {}
            return ctx

        # ───────── Stage 2 : Link verifier ─────────
        verified_map: Dict[_Key, Dict[str, Any]] = {}
        items = list(concept_map.items())
        for b_no, batch in enumerate(_chunks(items, self.verify_batch_size), 1):
            pairs = [{
                SPAN_FIELD: k[0],
                "entity_name": c.get("entity_name"),
                "offset": [k[1], k[2]],
                **{k2: v for k2, v in c.items() if k2 in ("conceptId", "preferred_term", "top_type", "synonyms", "definition")}
            } for k, c in batch]
            prompt = verifier_tpl.replace("#CRITERION#", criterion).replace(
                "#PAIRS#", json.dumps(pairs, indent=2, ensure_ascii=False)
            )
            pfx = pathlib.Path("mbench/entity_mbench/entity_logs",
                               f"{note_id}/{side}_req{idx:03d}_stage2b{b_no:02d}")
            verdicts = self._call_and_log(prompt, True, pfx)
            for v in verdicts:
                if v.get("decision") == "KEEP":
                    key = _make_key(v[SPAN_FIELD], *v["offset"])
                    verified_map[key] = concept_map[key]


        # ───────── Collect non-KEEP vector candidates (for analysis) ─────────
        # 规则：
        #  - 逐个 (text,start,end) occurrence 读取 vector 阶段挂在实体上的候选
        #  - 若本 occurrence 有最终 KEEP 的 conceptId，则把它从候选列表里剔除
        #  - 其余候选（含分数/同义词/定义等）作为“其他有价值候选”保存
        other_items: List[Dict[str, Any]] = []
        for e in exploded:
            key = _make_key(e["text"], e["start"], e["end"])
            kept_cid = verified_map.get(key, {}).get("conceptId") if key in verified_map else None
            # 优先用“完整候选”（若 VectorEmbeddingConceptSearch 暴露了 candidates_all）
            cand_list = (e.get("candidates_all") or e.get("candidates") or [])
            for cand in cand_list:
                cid = cand.get("conceptId") or cand.get("concept_id") or cand.get("sctid")
                if kept_cid and cid == kept_cid:
                    continue
                other_items.append({
                    SPAN_FIELD: e["text"],
                    "entity_name": e.get("entity_name", ""),
                    "offset": [int(e["start"]), int(e["end"])],
                    "kept_conceptId": kept_cid or "",
                    "candidate": cand
                })

        other_payload = {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "requirement": criterion,
            "count_pairs": len(other_items),
            "items": other_items,
        }

        # 1) 写入 context
        ctx.setdefault("entity_other_canonical_candidates", {})[str(idx)] = other_payload

        # 2) 写入相对日志目录（沿用 note_id 结构）
        _write_json(
            pathlib.Path("mbench/entity_mbench/entity_logs", f"{note_id}/{side}_req{idx:03d}_other_candidates.json"),
            other_payload
        )

        # 3) 写入你指定的绝对目录（sigir-20141）
        try:
            fixed_abs_dir = pathlib.Path("<SATIR_ROOT>/psrc/mbench/entity_mbench/entity_logs/sigir-20141")
            _write_json(fixed_abs_dir / f"{side}_req{idx:03d}_other_candidates.json", other_payload)
        except Exception as exc:
            _log(f"[other-candidates] absolute-path write skipped: {exc}")




        # 若 verifier 全拒绝，写空表并返回
        if not verified_map:
            ctx.setdefault("valid_entities_by_req", {})[str(idx)] = {}
            return ctx

        # ───────── Assemble result dict (no arbiter, no overlap checks) ─────────
        # 为了稳定性，按 (start, end, text) 排序后顺序编号 entity_1, entity_2, ...
        sorted_items = sorted(
            verified_map.items(),
            key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])  # (start, end, text)
        )

        final: Dict[str, Any] = {}
        for i, (k, c) in enumerate(sorted_items, 1):
            text, start, end = k
            final[f"entity_{i}"] = {
                SPAN_FIELD:             text,
                "entity_name":          c.get("entity_name", ""),
                "preferred_term":       c.get("preferred_term", ""),
                "fully_specified_name": c.get("fully_specified_name", ""),
                "type":                 c.get("top_type", ""),
                "conceptId":            c.get("conceptId", ""),
                "select_reason":        c.get("select_reason", ""),
                "start":                int(start),
                "end":                  int(end),
            }

        ctx.setdefault("valid_entities_by_req", {})[str(idx)] = final

        # 记录明细（便于调试）
        _write_json(
            pathlib.Path("mbench/entity_mbench/entity_logs", f"{note_id}/{side}_req{idx:03d}_detail.json"),
            {
                "generated":   dt.datetime.now().isoformat(timespec="seconds"),
                "requirement": criterion,
                "kept_count":  len(final),
                "final_entities": final,
                "note": "arbiter removed; all verifier-KEEP are accepted (no overlap checks).",
            },
        )
        return ctx
