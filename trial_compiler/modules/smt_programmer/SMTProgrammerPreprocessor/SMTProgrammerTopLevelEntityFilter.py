#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import json, re, pathlib, time
import dspy

# ───────────────────────────── helpers ─────────────────────────────

def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)

def _strip_fence(s: str) -> str:
    return _JSON_FENCE.sub("", str(s or ""))

def _norm(s: Any) -> str:
    return str(s or "").strip()

def _lower(s: Any) -> str:
    return str(s or "").strip().lower()

def _write_txt(p: pathlib.Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _rid_tag(rid_str: str) -> str:
    try:
        return f"req{int(rid_str):03d}"
    except Exception:
        return f"req_{_safe_id(rid_str)}"

def _call_engine(engine, prompt: str, *, temperature: float = 0.0) -> str:
    out = engine(prompt, temperature=temperature) if callable(engine) else engine(prompt)
    return out[0] if isinstance(out, (list, tuple)) else str(out)

def _to_int(x: Any) -> Optional[int]:
    if x is None: return None
    try: return int(x)
    except Exception:
        try: return int(str(x).strip())
        except Exception: return None

def _esig(text: str, st: Any, ed: Any) -> Tuple[str, Optional[int], Optional[int]]:
    return (_lower(text), _to_int(st), _to_int(ed))

# ─────────────────────── main: top-level filter ───────────────────────

class SMTProgrammerTopLevelEntityFilter(dspy.Module):
    """
    Batched, LLM-based top-level entity filter.

    Inputs (new):
      - ctx["requirement_canonical_entities_filtered_qualifiers"]: List[
            { requirement_id, requirement, entities: [
                { entity_id, extracted_span, start, end, type, Time|Space|Scale|Source|Cause|Definition|Other: [ {qualifier, qualifier_span, rationale, verification?} ] }
            ]}
        ]
      - ctx["requirement_free_entities_filtered_qualifiers"]: List[same-shape as above but type usually "FreeEntity"]
      - ctx["requirements"] (optional): used to backfill requirement text when missing

    (legacy; still supported if present — then new inputs are ignored):
      - ctx["requirements_entities_attributes"]: List[ { requirement_id, requirement_texts, entities:[{entity, attributes:[]}] } ]

    Optional:
      - ctx["SMTProgrammerTopLevelEntityFilter_prompt"]: custom prompt template using {REQUIREMENTS_JSON}

    Writes:
      - ctx["requirements_entities_attributes_top_level"]
      - ctx["toplevel_drop_ids_by_requirement"]
      - ctx["toplevel_decisions_by_requirement"]
    """

    def __init__(
        self,
        engine,
        *,
        temperature: float = 0.0,
        max_retries: int = 2,
        verbose: bool = True,
        microbench_dir: str | pathlib.Path | None = None,
        batch_size: int = 6,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.batch_size = max(1, int(batch_size))
        self.mbench_dir = pathlib.Path(microbench_dir).expanduser() if microbench_dir else pathlib.Path("mbench/toplevel_filter")
        self.microbench_dir = pathlib.Path(microbench_dir).expanduser() if microbench_dir else None

    # ------------------------------- prompts ------------------------------- #

    def _default_prompt_batched(self, reqs_json: str) -> str:
        return (
            "# === ROLE ===\n"
            "You are an expert in clinical NLP and ontology-grounded information extraction.\n\n"
            "# === TASK ===\n"
            "Given multiple clinical trial criteria and their entity lists (with qualifiers/attributes), decide for each entity whether it is a top-level subject OR part of the qualifier/value of another entity. If it functions as a qualifier/value to another entity, mark to DROP. Be conservative: keep if ambiguous.\n\n"
            "# === RULES ===\n"
            "1. Use (\"qualifiers\" and \"canonical_attributes\") or (\"qualifiers\" and \"free_attributes\" when \"canonical_attributes\" is None) to detect modifier/value relations (site, laterality, severity, timing, location, stage, etc.).\n"
            "2. Don’t drop an entity text just because its surface string reappears inside a qualifier. Only drop if the reappearance makes that string function as a qualifier of another subject, with no standalone fact asserted about itself.\n"
            "   Example (Keep): The entity text is “angiotensin receptor blocker (ARB)”. The qualifier says the ARB has already been administered. The entity name appears in the qualifier, but it’s just the subject; the attribute-value is administration status = already administered. So ARB is KEPT.\n"
            "3. Output JSON only. Do NOT include prose outside the JSON or any comments inside the JSON.\n"
            "4. Scratchpad usage (per-entity):\n"
            "   - Possibly modifies: name the other entity in the same requirement that this entity could be modifying; if none, write `NONE`.\n"
            "   - Evidence: copy the minimal qualifier/attribute fragment (raw phrase).\n"
            "   - Role: `TOP-LEVEL` (subject) or `MODIFIER` (qualifies another entity).\n"
            "   - Verdict: `TOP-LEVEL` or `MODIFIER` (must match Role).\n\n"
            "# === INPUT ===\n"
            "<requirements>\n"
            f"{reqs_json}\n"
            "</requirements>\n\n"
            "# === INPUT FORMAT NOTES (STRICT) ===\n"
            "- REQUIREMENTS_JSON is an array of requirement objects.\n"
            "- requirement_id MUST be an integer.\n"
            "- Each requirement object contains exactly one requirement_text string and one entities array.\n"
            "- Do NOT repeat the requirement text inside individual entities; all entities for a requirement are nested under that requirement object.\n\n"
            "# === OUTPUT ===\n"
            "[\n"
            "  {\n"
            '    "requirement_id": 0,\n'
            '    "decisions": [\n'
            '      {\n'
            '        "id": 0,\n'
            '        "text": "<entity text span>",\n'
            '        "start": 13,\n'
            '        "end": 64,\n'
            '        "scratchpad": "- Possibly modifies: <entity or NONE>\\n- Evidence: \\"<fragment>\\"\\n- Role: <TOP-LEVEL or MODIFIER>\\n- Verdict: <TOP-LEVEL|MODIFIER>",\n'
            '        "KEEP": "YES" | "NO",\n'
            '        "explanation": "<1–2 sentences consistent with the scratchpad>"\n'
            "      }\n"
            "    ]\n"
            "  }\n"
            "]\n"
        )

    def _default_prompt_single(self, requirement_text: str, entities_json: str) -> str:
        # Fallback (batch_size==1 or custom template)
        return (
            "# ROLE\n"
            "You are an expert in clinical NLP and ontology-grounded information extraction.\n\n"
            "# TASK\n"
            "Given the requirement text and a list of entities (with qualifiers/attributes), decide for each entity whether it is\n"
            "a top-level subject OR part of the qualifier/value of another entity. If it functions as a qualifier/value to another entity, mark to DROP.\n"
            "Be conservative: keep if ambiguous.\n\n"
            "# RULES\n"
            "1. Use (\"qualifiers\" and \"canonical_attributes\") or (\"qualifiers\" and \"free_attributes\" when \"canonical_attributes\" is None) to detect modifier/value relations (site, laterality, severity, timing, location, stage, etc.).\n"
            "2. Don’t drop an entity text just because its surface string reappears inside a qualifier. Only drop if the reappearance makes that string function as a qualifier of another subject, with no standalone fact asserted about itself.\n"
            "   Example (Keep): ARB already administered → ARB is KEPT.\n"
            "3. Output JSON only.\n\n"
            "# INPUT\n"
            "<requirement>\n"
            f"{requirement_text}\n"
            "</requirement>\n\n"
            "<entities>\n"
            f"{entities_json}\n"
            "</entities>\n\n"
            "# OUTPUT\n"
            "[\n"
            "  {\n"
            '    "id": 0,\n'
            '    "text": "<entity text span>",\n'
            '    "start": 13,\n'
            '    "end": 64,\n'
            '    "scratchpad": "<scratchpad for seeing if this entity is a part of a qualifier of any other entity>",\n'
            '    "KEEP": "YES" | "NO",\n'
            '    "explanation": "<explanation here>"\n'
            "  }\n"
            "]\n"
        )

    # --------------------------- input preparation ------------------------ #

    @staticmethod
    def _req_text(req: Any) -> str:
        if isinstance(req, str):
            return req
        if isinstance(req, dict):
            for k in ("requirement", "text", "requirement_text", "sentence"):
                if k in req and req[k]:
                    return str(req[k])
        return json.dumps(req, ensure_ascii=False)

    # === NEW: synthesize legacy "requirements_entities_attributes" from new filtered inputs ===
    @staticmethod
    def _iter_bucket_qualifiers(ent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten all qualifier dicts from known buckets."""
        out: List[Dict[str, Any]] = []
        for b in ("Time", "Space", "Scale", "Source", "Cause", "Definition", "Other"):
            for q in (ent.get(b) or []):
                if isinstance(q, dict):
                    # If verification present, keep only ALL_GOOD==YES (safety; inputs already filtered)
                    ver = q.get("verification") or {}
                    all_good = str(ver.get("ALL_GOOD", "YES")).upper()
                    if all_good and all_good not in ("YES",):
                        continue
                    out.append(q)
        return out

    def _synthesize_rea_from_filtered_sources(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a legacy-like structure:
          [{ "requirement_id": rid, "requirement_texts":[text], "entities":[{"entity":{...}, "attributes":[...]}] }]
        from:
          - requirement_canonical_entities_filtered_qualifiers
          - requirement_free_entities_filtered_qualifiers
        """
        can_list = ctx.get("requirement_canonical_entities_filtered_qualifiers") or []
        free_list = ctx.get("requirement_free_entities_filtered_qualifiers") or []

        if not can_list and not free_list:
            return []

        # rid -> block
        blocks: Dict[str, Dict[str, Any]] = {}

        def ensure_block(rid: Any, rtext: str) -> Dict[str, Any]:
            rid_str = str(rid)
            blk = blocks.get(rid_str)
            if not blk:
                blk = {"requirement_id": rid_str, "requirement_texts": [_norm(rtext)] if rtext else [], "entities": []}
                blocks[rid_str] = blk
            else:
                # fill missing text if available
                if (not blk.get("requirement_texts")) and rtext:
                    blk["requirement_texts"] = [_norm(rtext)]
            return blk

        # Canonical
        for rec in can_list:
            rid = rec.get("requirement_id")
            rtxt = rec.get("requirement") or ""
            blk = ensure_block(rid, rtxt)
            for ent in (rec.get("entities") or []):
                e_node = {
                    "entity": {
                        "surface_string": _norm(ent.get("extracted_span")),
                        "start": _to_int(ent.get("start")),
                        "end": _to_int(ent.get("end")),
                        "type": ent.get("type"),
                        "is_canonical_entity": True,
                    },
                    "attributes": [],
                }
                # Map qualifiers → attributes (canonical uses original_qualifier)
                for q in self._iter_bucket_qualifiers(ent):
                    q_span = _norm(q.get("qualifier_span"))
                    q_val  = _norm(q.get("qualifier") or q_span)
                    if not (q_val or q_span):
                        continue
                    e_node["attributes"].append({
                        "attribute_type": "qualifier",
                        "value": q_val,
                        "value_conceptId": None,
                        "original_qualifier": q_val,   # ← canonical branch uses this in _collect_entities_for_req_attrs
                        "qualifier_span": q_span,
                        "start": None,
                        "end": None,
                        "is_canonical": False,
                        "provenance": "canonical_filtered",
                    })
                blk["entities"].append(e_node)

        # Free
        for rec in free_list:
            rid = rec.get("requirement_id")
            rtxt = rec.get("requirement") or ""
            blk = ensure_block(rid, rtxt)
            for ent in (rec.get("entities") or []):
                e_node = {
                    "entity": {
                        "surface_string": _norm(ent.get("extracted_span")),
                        "start": _to_int(ent.get("start")),
                        "end": _to_int(ent.get("end")),
                        "type": ent.get("type"),
                        "is_canonical_entity": False,
                    },
                    "attributes": [],
                }
                # Map qualifiers → attributes (free uses self_contained_qualifier + qualifier_span)
                for q in self._iter_bucket_qualifiers(ent):
                    q_span = _norm(q.get("qualifier_span"))
                    q_val  = _norm(q.get("qualifier") or q_span)
                    if not (q_val or q_span):
                        continue
                    e_node["attributes"].append({
                        "attribute_type": "free_qualifier",
                        "value": q_val,
                        "value_conceptId": None,
                        "self_contained_qualifier": q_val,  # ← free branch uses this in _collect_entities_for_req_attrs
                        "qualifier_span": q_span,
                        "start": None,
                        "end": None,
                        "is_canonical": False,
                        "provenance": "free_filtered",
                    })
                blk["entities"].append(e_node)

        # Backfill requirement text from ctx["requirements"] if still missing
        reqs = ctx.get("requirements")
        if isinstance(reqs, list):
            for rid_str, blk in blocks.items():
                try:
                    idx = int(rid_str)
                except Exception:
                    idx = None
                if (not blk.get("requirement_texts")) and (idx is not None) and idx < len(reqs):
                    blk["requirement_texts"] = [_norm(self._req_text(reqs[idx]))]

        # Stable order by rid (int if possible)
        def _rid_sort_key(k: str) -> Tuple[int, str]:
            try:
                return (0, f"{int(k):09d}")
            except Exception:
                return (1, k)
        ordered = [blocks[k] for k in sorted(blocks.keys(), key=_rid_sort_key)]
        return ordered

    def _collect_entities_for_req_attrs(
        self, block: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """
        Build prompt-facing entity list with:
          - qualifiers (strings)    · canonical entities → ONLY from original_qualifier
          - canonical_attributes    · e.g., "type: value [conceptId]"
        """
        out_pack: List[Dict[str, Any]] = []
        id2rec: Dict[int, Dict[str, Any]] = {}

        entities = block.get("entities", []) or []
        for i, entrec in enumerate(entities):
            ent = entrec.get("entity", {}) or {}
            if "is_canonical_entity" not in ent:
                ent["is_canonical_entity"] = True
            is_canon_entity = bool(ent.get("is_canonical_entity", True))
            attrs = entrec.get("attributes", []) or []

            # qualifiers
            qualifiers: List[str] = []
            for a in attrs:
                if is_canon_entity:
                    # Canonical entries: ONLY take original_qualifier (synthesized above)
                    oq = _norm(a.get("original_qualifier"))
                    if oq:
                        qualifiers.append(oq)
                else:
                    # Non-canonical/free entries: keep permissive behavior
                    scq = _norm(a.get("self_contained_qualifier") or a.get("original_qualifier"))
                    qsp = _norm(a.get("qualifier_span"))
                    if scq:
                        qualifiers.append(scq)
                    # if qsp and qsp not in qualifiers:
                    #     qualifiers.append(qsp)
            qualifiers = sorted(list(dict.fromkeys([q for q in qualifiers if q])), key=_lower)
            qualifiers = [q for q in qualifiers if q]
            qualifiers = sorted(list(dict.fromkeys(qualifiers)), key=_lower)

            canonical_attributes: List[str] = []
            # (Optional) Preserve any canonical attributes if your upstream populates them later
            for a in attrs:
                at  = _norm(a.get("attribute_type"))
                val = _norm(a.get("value"))
                cid = _norm(a.get("value_conceptId"))
                is_can = bool(a.get("is_canonical")) if a.get("is_canonical") is not None else bool(cid)

                parts = []
                if at or val:
                    parts.append(f"{at}: {val}".strip(": ").strip())
                if cid:
                    parts.append(f"[{cid}]")
                line = " ".join([p for p in parts if p]).strip()

                if is_can and line:
                    canonical_attributes.append(line)

            canonical_attributes = sorted(list(dict.fromkeys(canonical_attributes)), key=_lower)

            pack = {
                "id": i,
                "text": _norm(ent.get("surface_string")),
                "start": ent.get("start"),
                "end": ent.get("end"),
                "is_canonical_entity": is_canon_entity,
                "qualifiers": qualifiers,
                "canonical_attributes": canonical_attributes,
            }
            out_pack.append(pack)
            id2rec[i] = entrec

        return out_pack, id2rec

    # （保留：旧的 free 合并函数将不再触发，因为 ctx 不再有 free_entity_qualifiers_by_req）
    def _merge_free_pairs_into_block(
        self,
        block: Dict[str, Any],
        rid_str: str,
        ctx: Dict[str, Any],
    ) -> None:
        free_map = ctx.get("free_entity_qualifiers_by_req") or {}
        rid_key_int: Optional[int] = None
        try:
            rid_key_int = int(rid_str)
        except Exception:
            pass

        free_list = []
        if rid_key_int is not None and rid_key_int in free_map:
            free_list = free_map.get(rid_key_int) or []
        elif rid_key_int is not None and str(rid_key_int) in free_map:
            free_list = free_map.get(str(rid_key_int)) or []
        elif rid_str in free_map:
            free_list = free_map.get(rid_str) or []

        if not free_list:
            return

        block.setdefault("entities", [])
        idx: Dict[Tuple[str, Optional[int], Optional[int]], Dict[str, Any]] = {}
        for e_rec in block["entities"]:
            ent = e_rec.get("entity", {}) or {}
            if "is_canonical_entity" not in ent:
                ent["is_canonical_entity"] = True
            sig = _esig(_norm(ent.get("surface_string","")), ent.get("start"), ent.get("end"))
            idx[sig] = e_rec

        for item in free_list:
            ent = (item or {}).get("entity", {}) or {}
            etxt = _norm(ent.get("surface_string"))
            est, eed = _to_int(ent.get("start")), _to_int(ent.get("end"))
            if not etxt: continue
            sig = _esig(etxt, est, eed)

            if sig not in idx:
                node = {
                    "entity": {
                        "surface_string": etxt,
                        "start": est,
                        "end": eed,
                        "is_canonical_entity": False,
                    },
                    "attributes": [],
                }
                block["entities"].append(node)
                idx[sig] = node

            e_node = idx[sig]
            e_attrs: List[Dict[str, Any]] = e_node.setdefault("attributes", [])

            quals = item.get("all_qualifying_information_related_to_the_entity") or []
            for q in quals:
                all_good = str(q.get("ALL_GOOD","")).strip().upper()
                if all_good and all_good not in ("YES",):
                    continue

                q_span = _norm(q.get("qualifier_span"))
                q_val  = _norm(q.get("qualifier") or q_span)
                q_st, q_ed = _to_int(q.get("start")), _to_int(q.get("end"))
                if not (q_span or q_val): continue

                e_attrs.append({
                    "attribute_type": "free_qualifier",
                    "value": q_val,
                    "value_conceptId": None,
                    "self_contained_qualifier": q_val,
                    "qualifier_span": q_span,
                    "start": q_st,
                    "end": q_ed,
                    "is_canonical": False,
                    "provenance": "free_pipeline",
                })

    # ----------------------------- LLM calls ------------------------------- #

    def _prompt_for_batch(self, ctx: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
        blob = json.dumps(items, ensure_ascii=False, indent=2)
        if "SMTProgrammerTopLevelEntityFilter_prompt" in ctx:
            return ctx["SMTProgrammerTopLevelEntityFilter_prompt"].replace("{REQUIREMENTS_JSON}", blob)
        return self._default_prompt_batched(blob)

    def _prompt_for_single(self, ctx: Dict[str, Any], req_text: str, entities_json: str) -> str:
        if "SMTProgrammerTopLevelEntityFilter_prompt" in ctx:
            return (ctx["SMTProgrammerTopLevelEntityFilter_prompt"]
                    .replace("{REQUIREMENT}", req_text)
                    .replace("{ENTITIES_JSON}", entities_json))
        return self._default_prompt_single(req_text, entities_json)

    def _parse_batched_output(self, raw: str) -> Dict[str, Dict[str, Any]]:
        def as_decisions_from_list(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
            drop_ids, keep_ids, reasons = [], [], []
            for obj in objs:
                try:
                    i = int(obj.get("id"))
                except Exception:
                    continue
                keep_flag = str(obj.get("KEEP","")).strip().upper()
                if keep_flag == "NO":
                    drop_ids.append(i); decision = "DROP"
                else:
                    keep_ids.append(i); decision = "KEEP"
                reasons.append({
                    "id": i,
                    "decision": decision,
                    "brief_reason": _norm(obj.get("explanation")) or _norm(obj.get("scratchpad")) or "",
                })
            return {"drop_ids": sorted(set(drop_ids)), "keep_ids": sorted(set(keep_ids)), "reasons": reasons}

        txt = _strip_fence(raw)
        # 1) Try batched array
        try:
            obj = json.loads(txt)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "requirement_id" in obj[0]:
                out: Dict[str, Dict[str, Any]] = {}
                for e in obj:
                    rid = str(e.get("requirement_id", ""))
                    decisions = e.get("decisions", [])
                    if isinstance(decisions, list):
                        out[rid] = as_decisions_from_list(decisions)
                return out
        except Exception:
            pass

        # 2) Try single-req entity list
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return {"__single__": as_decisions_from_list(obj)}
        except Exception:
            pass

        # 3) Legacy dict
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "drop_ids" in obj:
                drop_ids = [int(x) for x in (obj.get("drop_ids") or []) if str(x).strip().isdigit()]
                keep_ids = [int(x) for x in (obj.get("keep_ids") or []) if str(x).strip().isdigit()]
                reasons  = obj.get("reasons") or []
                return {"__single__": {"drop_ids": drop_ids, "keep_ids": keep_ids, "reasons": reasons}}
        except Exception:
            pass

        return {}

    def _infer_batch(
        self,
        engine,
        prompt: str,
        *,
        log_dir: pathlib.Path,
        batch_tag: str,
        micro_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_{batch_tag}_toplevel"
        _write_txt(log_dir / f"{base}_prompt.txt", prompt)
        if micro_dir is not None:
            _write_txt(micro_dir / f"{base}_prompt.txt", prompt)

        last_raw = ""
        parsed_map: Dict[str, Dict[str, Any]] = {}

        for attempt in range(1, self.max_retries + 1):
            raw = _call_engine(engine, prompt, temperature=self.temperature)
            last_raw = raw
            _write_txt(log_dir / f"{base}_raw_attempt{attempt}.txt", raw)
            if micro_dir is not None:
                _write_txt(micro_dir / f"{base}_raw_attempt{attempt}.txt", raw)

            parsed_map = self._parse_batched_output(raw)
            if parsed_map:
                return parsed_map

        # Repair
        repair = (
            "Return ONLY JSON in the batched format:\n"
            "[{\"requirement_id\": \"<id>\", \"decisions\": [{\"id\": int, \"text\": str, \"start\": int, \"end\": int, \"scratchpad\": str, \"KEEP\": \"YES\"|\"NO\", \"explanation\": str}]}]\n\n"
            "If not possible, return an empty array [].\n\n=== CONTENT ===\n" + last_raw
        )
        _write_txt(log_dir / f"{base}_repair_prompt.txt", repair)
        raw2 = _call_engine(engine, repair, temperature=self.temperature)
        _write_txt(log_dir / f"{base}_raw_repair.txt", raw2)
        if micro_dir is not None:
            _write_txt(micro_dir / f"{base}_raw_repair.txt", raw2)

        parsed_map = self._parse_batched_output(raw2)
        return parsed_map or {}

    # ------------------------------ forward -------------------------------- #

    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        engine = ctx.get("engine", self.engine)
        if engine is None:
            raise RuntimeError("No chat engine found. Provide one via constructor or ctx['engine'].")

        trial_id = _norm(ctx.get("trial_id", "trial"))
        inc_exc  = _norm(ctx.get("inc_exc", "side"))

        # 1) Prefer legacy if present; otherwise synthesize from new filtered inputs
        rea: List[Dict[str, Any]] = ctx.get("requirements_entities_attributes") or []
        if not rea:
            rea = self._synthesize_rea_from_filtered_sources(ctx)

        if not rea:
            ctx["requirements_entities_attributes_top_level"] = []
            ctx["toplevel_drop_ids_by_requirement"] = {}
            ctx["toplevel_decisions_by_requirement"] = {}
            return ctx

        # 2) Pre-merge free qualifiers (legacy path only; new path已在合成时纳入 free)
        prepped: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, Any]], str]] = []
        for block in rea:
            rid_str = str(block.get("requirement_id", "0"))
            req_texts = block.get("requirement_texts") or []
            req_text = req_texts[0] if req_texts else ""
            try:
                req_index = int(rid_str)
                if not req_text and isinstance(ctx.get("requirements"), list) and req_index < len(ctx["requirements"]):
                    req_text = self._req_text(ctx["requirements"][req_index])
            except Exception:
                pass
            req_text = _norm(req_text)

            # legacy merge (no-op for new path)
            self._merge_free_pairs_into_block(block, rid_str, ctx)
            entity_pack, id2rec = self._collect_entities_for_req_attrs(block)
            prepped.append((rid_str, block, entity_pack, id2rec, req_text))

        # 3) Build batches
        batches: List[List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, Any]], str]]] = []
        cur: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, Any]], str]] = []
        for item in prepped:
            cur.append(item)
            if len(cur) >= self.batch_size:
                batches.append(cur); cur = []
        if cur: batches.append(cur)

        out_blocks: List[Dict[str, Any]] = []
        drop_map: Dict[str, List[int]] = {}
        dec_map: Dict[str, Dict[str, Any]] = {}

        # 4) Process each batch with one LLM call
        for b_idx, batch in enumerate(batches, start=1):
            # Decide prompt
            if len(batch) == 1:
                rid_str, block, entity_pack, id2rec, req_text = batch[0]
                req_dir = (self.mbench_dir / _norm(trial_id) / _norm(inc_exc) /
                           (f"req{int(rid_str):03d}" if rid_str.isdigit() else f"req_{_safe_id(rid_str)}"))
                logs_dir = req_dir / "toplevel_prompts_outputs"
                logs_dir.mkdir(parents=True, exist_ok=True)

                prompt = self._prompt_for_single(ctx, req_text, json.dumps(entity_pack, ensure_ascii=False, indent=2))

                micro_dir: Optional[pathlib.Path] = None
                if self.microbench_dir is not None:
                    micro_dir = self.microbench_dir / _norm(trial_id) / _norm(inc_exc)
                    micro_dir.mkdir(parents=True, exist_ok=True)
                    tag = _rid_tag(rid_str)
                    _write_json(micro_dir / f"{tag}_input.json", {"requirement": req_text, "entities": entity_pack})
                    _write_txt(micro_dir / f"{tag}_prompt.txt", prompt)

                parsed_map = self._infer_batch(engine, prompt, log_dir=logs_dir, batch_tag=_rid_tag(rid_str), micro_dir=micro_dir)
            else:
                # Batched prompt
                def entities_for_prompt(entity_pack: Any) -> List[Dict[str, Any]]:
                    if isinstance(entity_pack, dict) and "entities" in entity_pack:
                        src = entity_pack.get("entities") or []
                    else:
                        src = entity_pack or []
                    out: List[Dict[str, Any]] = []
                    for e in src:
                        q = e.get("qualifiers")
                        if q is None:
                            q_list: List[str] = []
                        elif isinstance(q, str):
                            q_list = [q]
                        elif isinstance(q, (list, tuple)):
                            q_list = [str(x) for x in q]
                        else:
                            q_list = [str(q)]
                        out.append({
                            "id": e.get("id"),
                            "text": e.get("text"),
                            "start": e.get("start"),
                            "end": e.get("end"),
                            "is_canonical_entity": e.get("is_canonical_entity", False),
                            "all_modifying_information": q_list,
                        })
                    return out

                batch_tag = f"batch_{b_idx:03d}"
                batch_dir = self.mbench_dir / _norm(trial_id) / _norm(inc_exc) / batch_tag
                logs_dir = batch_dir / "toplevel_prompts_outputs"
                logs_dir.mkdir(parents=True, exist_ok=True)

                items_for_prompt = []
                for rid_str, block, entity_pack, id2rec, req_text in batch:
                    try:
                        rid_out = int(rid_str)
                    except Exception:
                        rid_out = rid_str
                    items_for_prompt.append({
                        "requirement_id": rid_out,
                        "requirement_text": req_text,
                        "entities": entities_for_prompt(entity_pack),
                    })

                prompt = self._prompt_for_batch(ctx, items_for_prompt)

                micro_dir: Optional[pathlib.Path] = None
                if self.microbench_dir is not None:
                    micro_dir = self.microbench_dir / _norm(trial_id) / _norm(inc_exc) / batch_tag
                    micro_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(micro_dir / "batch_input.json", items_for_prompt)
                    _write_txt(micro_dir / "batch_prompt.txt", prompt)

                parsed_map = self._infer_batch(engine, prompt, log_dir=logs_dir, batch_tag=batch_tag, micro_dir=micro_dir)

            # Apply decisions to each requirement in the batch
            for rid_str, block, entity_pack, id2rec, req_text in batch:
                decisions = parsed_map.get(rid_str)
                if decisions is None and len(batch) == 1 and "__single__" in parsed_map:
                    decisions = parsed_map["__single__"]
                if decisions is None:
                    decisions = {"drop_ids": [], "keep_ids": [], "reasons": []}

                drop_ids = set(int(x) for x in (decisions.get("drop_ids") or []))
                keep_ids = set(int(x) for x in (decisions.get("keep_ids") or []))
                valid_ids = set(range(len(entity_pack)))
                drop_ids = {i for i in drop_ids if i in valid_ids}
                keep_ids = {i for i in keep_ids if i in valid_ids and i not in drop_ids}

                kept_entities = []
                for i in range(len(entity_pack)):
                    if i in drop_ids:
                        continue
                    rec = dict(id2rec[i])  # shallow copy
                    rec.setdefault("qualifiers", entity_pack[i].get("qualifiers", []))
                    rec.setdefault("canonical_attributes", entity_pack[i].get("canonical_attributes", []))
                    kept_entities.append(rec)

                filtered_block = {
                    "requirement_id": rid_str,
                    "requirement_texts": block.get("requirement_texts", []),
                    "entities": kept_entities,
                }
                out_blocks.append(filtered_block)
                drop_map[rid_str] = sorted(list(drop_ids))
                dec_map[rid_str] = {
                    "drop_ids": sorted(list(drop_ids)),
                    "keep_ids": sorted(list(keep_ids)),
                    "reasons": decisions.get("reasons") or [],
                }

                # Per-requirement logs
                try:
                    req_dir = (self.mbench_dir / _norm(trial_id) / _norm(inc_exc) /
                               (f"req{int(rid_str):03d}" if rid_str.isdigit() else f"req_{_safe_id(rid_str)}"))
                    (req_dir / "toplevel_prompts_outputs").mkdir(parents=True, exist_ok=True)
                    (req_dir / "toplevel.json").write_text(
                        json.dumps(dec_map[rid_str], ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass

        ctx["requirements_entities_attributes_top_level"] = out_blocks
        print(f'ctx[requirements_entities_attributes_top_level] is {ctx["requirements_entities_attributes_top_level"]}')
        ctx["toplevel_drop_ids_by_requirement"] = drop_map
        ctx["toplevel_decisions_by_requirement"] = dec_map

        # ───────────────────────── persist final top-level structure ─────────────────────────
        try:
            save_root = self.mbench_dir / _norm(trial_id) / _norm(inc_exc)
            save_root.mkdir(parents=True, exist_ok=True)

            agg_path = save_root / "top_level_entities.json"
            _write_json(agg_path, out_blocks)

            for blk in out_blocks:
                rid_str = str(blk.get("requirement_id", "0"))
                req_dir = (save_root / (f"req{int(rid_str):03d}" if rid_str.isdigit()
                                        else f"req_{_safe_id(rid_str)}"))
                req_dir.mkdir(parents=True, exist_ok=True)
                _write_json(req_dir / "top_level_block.json", blk)

            summary = {
                "trial_id": trial_id,
                "side": inc_exc,
                "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_requirements_in_input": len(rea),
                "num_requirements_in_output": len(out_blocks),
                "entities_before": sum(len(b.get("entities", [])) for b in rea),
                "entities_after": sum(len(b.get("entities", [])) for b in out_blocks),
                "dropped_by_requirement": drop_map,
                "decisions_by_requirement": dec_map,
                "aggregate_path": str(agg_path),
            }
            _write_json(save_root / "top_level_summary.json", summary)
        except Exception as _persist_exc:
            if self.verbose:
                print(f"[TopLevelFilter] ⚠ failed to persist top-level outputs: {_persist_exc}")
        # ───────────────────────── end persist ─────────────────────────

        if self.verbose:
            total_before = sum(len(b.get("entities", [])) for b in rea)
            total_after  = sum(len(b.get("entities", [])) for b in out_blocks)
            print(f"[TopLevelFilter:BATCH] entities: {total_before} -> {total_after} (dropped {total_before-total_after}); batches={len(batches)} size≈{self.batch_size}")

        return ctx
