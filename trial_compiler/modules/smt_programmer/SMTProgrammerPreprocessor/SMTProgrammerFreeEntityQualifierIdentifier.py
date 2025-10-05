# SMTProgrammerFreeEntityQualifierIdentifier.py
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


# ────────────────────────────────────────────────────────────────
# Helpers to extract requirement text/id (kept local for portability)
# ────────────────────────────────────────────────────────────────
_REQ_TEXT_KEYS = ("requirement", "text", "requirement_text", "sentence", "raw")

def _extract_requirement_text(req: Any) -> str:
    if isinstance(req, str):
        return req
    if isinstance(req, dict):
        for k in _REQ_TEXT_KEYS:
            if k in req and req[k] is not None:
                return str(req[k])
    return str(req)

def _extract_requirement_id(fallback_idx: int, req: Any) -> int:
    if isinstance(req, dict):
        for k in ("requirement_id", "id", "index", "idx"):
            if k in req:
                try:
                    return int(req[k])
                except Exception:
                    pass
    return int(fallback_idx)


# ────────────────────────────────────────────────────────────────
# Free-entity builder
# ────────────────────────────────────────────────────────────────
def build_requirement_free_entities_for_qualifier_identification(
    ctx: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Inputs:
      - ctx["requirements"]: list[str | dict]
      - ctx["free_additional_spans_offsets_by_req"]: dict[int|str -> list[{"text": str, "offset": [start, end]}]]

    Output:
      - ctx["requirement_free_entities_for_qualifier_identification"]: List[dict]
        [
          {
            "requirement_id": int,
            "requirement": str,
            "entities": [
              {
                "entity_id": 0,
                "extracted_span": "...",
                "surface_string": "...",
                "start": int,
                "end": int,
                "type": "FreeEntity",
                "Time": [], "Space": [], "Scale": [],
                "Source": [], "Cause": [], "Definition": [], "Other": []
              },
              ...
            ]
          },
          ...
        ]

    Notes:
      - Accepts string or int keys for the per-requirement free span dict.
      - Entities are sorted by start offset and re-indexed with contiguous entity_id.
      - Buckets are initialized to empty lists to match the qualifier identifier schema.
    """
    log = logger or logging.getLogger("SMTProgrammerFreeRequirementEntityCollector")

    reqs = ctx.get("requirements")
    if not isinstance(reqs, list):
        raise RuntimeError("requirements must be a list[str|dict]")

    free_map = ctx.get("free_additional_spans_offsets_by_req")
    if not isinstance(free_map, dict):
        raise RuntimeError("free_additional_spans_offsets_by_req must be a dict")

    results: List[Dict[str, Any]] = []

    for i, req in enumerate(reqs):
        rid = _extract_requirement_id(i, req)
        rtxt = _extract_requirement_text(req)

        items = (
            free_map.get(i) or
            free_map.get(str(i)) or
            []
        )
        entities: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            text = it.get("text")
            off = it.get("offset")
            if not isinstance(text, str) or not isinstance(off, (list, tuple)) or len(off) != 2:
                continue
            try:
                start = int(off[0])
                end = int(off[1])
            except Exception:
                continue

            entities.append({
                "extracted_span": text,
                "surface_string": text,
                "start": start,
                "end": end,
                "type": "FreeEntity",
                # initialize qualifier buckets to match downstream schema
                "Time": [], "Space": [], "Scale": [],
                "Source": [], "Cause": [], "Definition": [], "Other": [],
            })

        # sort by start offset; assign entity_id
        entities.sort(key=lambda e: (e.get("start", 10**12), e.get("end", 10**12)))
        for eid, e in enumerate(entities):
            e["entity_id"] = eid

        results.append({
            "requirement_id": rid,
            "requirement": rtxt,
            "entities": entities,
        })

    ctx["requirement_free_entities_for_qualifier_identification"] = results
    log.debug(
        "Built requirement_free_entities_for_qualifier_identification for %d requirements",
        len(results),
    )
    return results


# ────────────────────────────────────────────────────────────────
# Qualifier Identifier (free-entity version)
# ────────────────────────────────────────────────────────────────
class SMTProgrammerFreeEntityQualifierIdentifier(dspy.Module):
    """
    Verifier-free qualifier identifier with batching; mirrors AttributeExtractorQualifierIdentifier,
    but uses *free_* ctx keys and writes to a separate mbench subtree.

    Required ctx keys:
      - ctx["SMTProgrammerFreeEntityQualifierIdentifier_prompt"]: str with "{requirement_entities}" placeholder
      - ctx["requirement_free_entities_for_qualifier_identification"]: List[Dict]
      - ctx["engine"] (optional): _ChatEngine
      - ctx["trial_id"] (optional), ctx["inc_exc"] (optional): used for mbench pathing

    Optional (enables verifier stage if present):
      - ctx["SMTProgrammerFreeEntityQualifierIdentifierVerifier_prompt"]: str with "#REQUIREMENT_ENTITY_QUALIFIER#"

    Produces:
      - ctx["requirement_free_entities_after_qualifier_identification"]: List[Dict]
      - (if verifier used) ctx["requirement_free_entities_filtered_qualifiers"]: List[Dict]

    Files:
      mbench/free_mbench/<trial_id>/<side>/
        - free_qualifier_identifier/prompts_outputs/{ts}_b{idx}_extractor_prompt.txt
        - free_qualifier_identifier/prompts_outputs/{ts}_b{idx}_extractor_raw_attempt{n}.txt
        - free_qualifier_identifier/requirement_free_entities_after_qualifier_identification.json
        - free_qualifier_identifier_verifier/prompts_outputs/{ts}_b{idx}_verifier_prompt.txt
        - free_qualifier_identifier_verifier/prompts_outputs/{ts}_b{idx}_verifier_raw_attempt{n}.txt
        - free_qualifier_identifier_verifier/verifier_merged_raw.json
        - free_qualifier_identifier_verifier/requirement_free_entities_filtered_qualifiers.json
    """

    def __init__(
        self,
        engine: _ChatEngine | None = None,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_retries: int = 2,
        batch_size: int = 5,
        mbench_path: str | os.PathLike = "mbench/smt_mbench/free_qualifier_logs/",
        verbose: bool = False,
        debug_io: bool = False,  # ← NEW: gate all file I/O
    ) -> None:
        super().__init__()
        self._default_engine = engine
        self.model = model
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.batch_size = int(batch_size)
        self.mbench_root = Path(mbench_path)
        self.debug_io = bool(debug_io)

        self.log = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.log.setLevel(logging.INFO)

    # ------------------------------ utils ------------------------------ #
    @staticmethod
    def _strip_code_fence(s: str) -> str:
        return re.sub(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$", "", s, flags=re.S)

    # ── Robust JSON parsing helpers ─────────────────────────────────── #
    @staticmethod
    def _strip_comments(t: str) -> str:
        t = re.sub(r"//.*?$", "", t, flags=re.M)
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
        return t

    @staticmethod
    def _normalize_jsonish(t: str) -> str:
        # smart quotes → straight quotes
        t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        # Python booleans/None → JSON
        t = re.sub(r"\bTrue\b", "true", t)
        t = re.sub(r"\bFalse\b", "false", t)
        t = re.sub(r"\bNone\b", "null", t)
        # If there are very few double quotes relative to single quotes, convert singles
        if t.count('"') < max(1, t.count("'") // 4):
            t = re.sub(r'(?<!\\)\'', '"', t)
        # Remove trailing commas before } or ]
        t = re.sub(r",\s*([}\]])", r"\1", t)
        return t

    @staticmethod
    def _find_top_level_json_region(t: str) -> Optional[str]:
        # Scan for first complete top-level array or object.
        def scan(opener: str, closer: str) -> Optional[str]:
            stack = 0
            start = None
            i = 0
            n = len(t)
            in_string = False
            esc = False
            while i < n:
                ch = t[i]
                if in_string:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == opener and stack == 0:
                        start = i
                        stack = 1
                    elif ch == opener and stack > 0:
                        stack += 1
                    elif ch == closer and stack > 0:
                        stack -= 1
                        if stack == 0 and start is not None:
                            return t[start:i+1]
                i += 1
            return None

        return scan("[", "]") or scan("{", "}")

    @staticmethod
    def _best_effort_parse_array(s: str) -> List[Dict[str, Any]]:
        """
        Parse model output that *should* be a JSON array (or single object).
        Handles:
          - code fences, comments, extra prose
          - single quotes, trailing commas, smart quotes
          - Python booleans/None
          - embedded examples (find first complete top-level JSON region)
        """
        text = s.strip()
        text = SMTProgrammerFreeEntityQualifierIdentifier._strip_code_fence(text)
        text = SMTProgrammerFreeEntityQualifierIdentifier._strip_comments(text)

        # 1) direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass

        # 2) extract first complete JSON region and try normalized
        region = SMTProgrammerFreeEntityQualifierIdentifier._find_top_level_json_region(text)
        if region:
            region_norm = SMTProgrammerFreeEntityQualifierIdentifier._normalize_jsonish(region)
            for candidate in (region_norm, SMTProgrammerFreeEntityQualifierIdentifier._normalize_jsonish(region_norm)):
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, list):
                        return obj
                    if isinstance(obj, dict):
                        return [obj]
                except Exception:
                    pass

        # 3) normalize whole text and retry
        try:
            obj = json.loads(SMTProgrammerFreeEntityQualifierIdentifier._normalize_jsonish(text))
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass

        raise ValueError("Failed to parse LLM output as JSON array (even after robust heuristics).")

    @staticmethod
    def _write_txt(dir_path: Path, filename: str, content: str, *, enable: bool) -> None:
        if not enable:
            return
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(content, encoding="utf-8")

    @staticmethod
    def _write_json(file_path: Path, data: Any, *, enable: bool) -> None:
        if not enable:
            return
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _pick_trial_folder(self, ctx: Dict[str, Any]) -> str:
        trial_id = str(ctx.get("trial_id") or "unknown_trial").strip()
        side = str(ctx.get("inc_exc") or "unknown_side").strip()
        return str(Path(_safe_id(trial_id)) / _safe_id(side))

    def _prompts_dir_and_agg_file(self, trial_folder: str) -> tuple[Path, Path]:
        base = self.mbench_root / trial_folder / "free_qualifier_identifier"
        prompts = base / "prompts_outputs"
        agg = base / "requirement_free_entities_after_qualifier_identification.json"
        if self.debug_io:
            prompts.mkdir(parents=True, exist_ok=True)
            base.mkdir(parents=True, exist_ok=True)
        return prompts, agg

    def _verifier_dirs(self, trial_folder: str) -> tuple[Path, Path, Path]:
        vroot = self.mbench_root / trial_folder / "free_qualifier_identifier_verifier"
        vprompts = vroot / "prompts_outputs"
        vraw = vroot / "verifier_merged_raw.json"
        if self.debug_io:
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
        buckets = ("Time", "Space", "Scale", "Source", "Cause", "Definition", "Other")
        for b in buckets:
            for q in (req_obj.get(b) or []):
                if isinstance(q, dict) and ("qualifier" in q or "qualifier_span" in q):
                    yield b, q

    @staticmethod
    def _flatten_for_verifier(requirement_entities_after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                for _, q in SMTProgrammerFreeEntityQualifierIdentifier._iter_qualifier_entries(ent):
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
        index: Dict[Tuple[Any, Any, Any, Any, Any, Any], Dict[str, Any]] = {}

        for req in requirement_entities_after or []:
            rid = req.get("requirement_id")
            for ent in (req.get("entities") or []):
                ent_key_base = (rid, ent.get("entity_id"), ent.get("start"), ent.get("end"), ent.get("type"))
                for bucket_name, q in SMTProgrammerFreeEntityQualifierIdentifier._iter_qualifier_entries(ent):
                    key = (*ent_key_base, (q.get("qualifier_span") or q.get("qualifier")))
                    q.setdefault("_bucket", bucket_name)
                    index[key] = q

        for vq in vq_items or []:
            key = SMTProgrammerFreeEntityQualifierIdentifier._vq_key(vq)
            target = index.get(key)
            if not target:
                soft_key = key[:-1]
                for k2, qobj in index.items():
                    if k2[:-1] == soft_key:
                        target = qobj
                        break
            if not target:
                continue

            eq = vq.get("entity_qualifier", {})
            verification = {
                "ALL_GOOD": eq.get("ALL_GOOD"),
                "explanation": eq.get("explanation"),
            }
            target["verification"] = verification

    @staticmethod
    def _filter_by_all_good(requirement_entities_after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept = copy.deepcopy(requirement_entities_after)
        for req in kept:
            for ent in (req.get("entities") or []):
                for bucket in ("Time", "Space", "Scale", "Source", "Cause", "Definition", "Other"):
                    qlist = ent.get(bucket) or []
                    new_list = []
                    for q in qlist:
                        ver = q.get("verification")
                        ok = bool(ver and str(ver.get("ALL_GOOD")).upper() == "YES")
                        if ok:
                            new_list.append(q)
                    ent[bucket] = new_list
        return kept

    # ------------------------------ main ------------------------------- #
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        engine: Optional[_ChatEngine] = ctx.get("engine", self._default_engine)
        if engine is None:
            raise RuntimeError("No chat engine found. Provide one via constructor or ctx['engine'].")

        build_requirement_free_entities_for_qualifier_identification(ctx, logger=self.log)

        prompt_template: str = ctx.get("SMTProgrammerFreeEntityQualifierIdentifier_prompt")
        if not isinstance(prompt_template, str) or "{requirement_entities}" not in prompt_template:
            raise RuntimeError(
                "ctx['SMTProgrammerFreeEntityQualifierIdentifier_prompt'] must be a str containing '{requirement_entities}'."
            )

        # ✳️ Strong prompt nudge to keep the model output clean JSON
        if "Return ONLY a JSON array" not in prompt_template:
            prompt_template += "\n\nReturn ONLY a JSON array. No prose, no backticks, no explanations."

        req_entities: List[Dict[str, Any]] = ctx.get("requirement_free_entities_for_qualifier_identification")  # type: ignore
        if not isinstance(req_entities, list):
            raise RuntimeError("ctx['requirement_free_entities_for_qualifier_identification'] must be a list.")

        # mbench pathing
        trial_folder = self._pick_trial_folder(ctx)
        prompts_dir, agg_json_path = self._prompts_dir_and_agg_file(trial_folder)

        # record input order for stable output
        input_order_keys: List[Any] = [self._req_key(r, i) for i, r in enumerate(req_entities)]

        batches = self._chunks(req_entities, self.batch_size)
        merged: Dict[Any, Dict[str, Any]] = {}

        for b_idx, batch in enumerate(batches):
            payload_text = json.dumps(batch, ensure_ascii=False, indent=2)
            prompt = prompt_template.replace("{requirement_entities}", payload_text)

            ts = time.strftime("%Y%m%d-%H%M%S")
            base = f"{ts}_b{b_idx:04d}_extractor"
            self._write_txt(prompts_dir, f"{base}_prompt.txt", prompt, enable=self.debug_io)

            last_err: Optional[Exception] = None
            parsed_batch: Optional[List[Dict[str, Any]]] = None

            for attempt in range(1, self.max_retries + 1):
                raw = engine(prompt, temperature=self.temperature)[0]
                self._write_txt(prompts_dir, f"{base}_raw_attempt{attempt}.txt", raw, enable=self.debug_io)
                try:
                    clean = self._strip_code_fence(raw)
                    parsed_batch = self._best_effort_parse_array(clean)
                    break
                except Exception as e:
                    last_err = e
                    if self.debug_io:
                        # Log a short preview to help triage quickly
                        preview = raw[:500].replace("\n", "\\n")
                        self.log.warning(
                            "Parse attempt failed (batch %d, attempt %d). Preview: %s",
                            b_idx, attempt, preview
                        )
                    continue

            if parsed_batch is None:
                raise RuntimeError(
                    f"[batch {b_idx}] Could not parse LLM output as JSON array. "
                    f"Enable debug_io to inspect raw outputs. Last error: {last_err}"
                )

            for req_obj in parsed_batch:
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

        # Save recognition stage outputs
        ctx["requirement_free_entities_after_qualifier_identification"] = final_list
        try:
            self._write_json(agg_json_path, final_list, enable=self.debug_io)
        except Exception as e:
            self.log.error("Failed to write aggregated JSON to %s: %s", agg_json_path, e)

        # ----------------------- Optional verifier stage ----------------------- #
        verifier_template: Optional[str] = ctx.get("SMTProgrammerFreeEntityQualifierIdentifierVerifier_prompt")
        if isinstance(verifier_template, str) and "#REQUIREMENT_ENTITY_QUALIFIER#" in verifier_template:
            # Nudge the verifier too
            if "Return ONLY a JSON array" not in verifier_template:
                verifier_template += "\n\nReturn ONLY a JSON array. No prose, no backticks, no explanations."

            vprompts_dir, vroot, vraw_json = self._verifier_dirs(trial_folder)

            flat_inputs = self._flatten_for_verifier(final_list)
            flat_batches = self._chunks(flat_inputs, self.batch_size)

            all_vq: List[Dict[str, Any]] = []
            for b_idx, vbatch in enumerate(flat_batches):
                payload = json.dumps(vbatch, ensure_ascii=False, indent=2)
                vprompt = verifier_template.replace("#REQUIREMENT_ENTITY_QUALIFIER#", payload)

                ts = time.strftime("%Y%m%d-%H%M%S")
                base = f"{ts}_b{b_idx:04d}_verifier"
                self._write_txt(vprompts_dir, f"{base}_prompt.txt", vprompt, enable=self.debug_io)

                last_err: Optional[Exception] = None
                parsed: Optional[List[Dict[str, Any]]] = None
                for attempt in range(1, self.max_retries + 1):
                    raw = engine(vprompt, temperature=self.temperature)[0]
                    self._write_txt(vprompts_dir, f"{base}_raw_attempt{attempt}.txt", raw, enable=self.debug_io)
                    try:
                        clean = self._strip_code_fence(raw)
                        parsed = self._best_effort_parse_array(clean)
                        break
                    except Exception as e:
                        last_err = e
                        if self.debug_io:
                            preview = raw[:500].replace("\n", "\\n")
                            self.log.warning(
                                "Verifier parse failed (batch %d, attempt %d). Preview: %s",
                                b_idx, attempt, preview
                            )
                        continue

                if parsed is None:
                    raise RuntimeError(f"[verifier batch {b_idx}] LLM output parse failed. Last error: {last_err}")
                all_vq.extend(parsed)

            # persist raw verifier merge
            try:
                self._write_json(vraw_json, all_vq, enable=self.debug_io)
            except Exception as e:
                self.log.error("Failed to write verifier raw json: %s", e)

            # attach verification back to qualifiers
            self._attach_verification_to_ctx_structure(final_list, all_vq)

            # keep only qualifiers with ALL_GOOD == "YES"
            filtered = self._filter_by_all_good(final_list)

            ctx["requirement_free_entities_filtered_qualifiers"] = filtered
            try:
                self._write_json(vroot / "requirement_free_entities_filtered_qualifiers.json", filtered, enable=self.debug_io)
            except Exception as e:
                self.log.error("Failed to write filtered qualifiers json: %s", e)
        else:
            self.log.info("Free verifier prompt missing or without placeholder; skipping verification stage.")

        return ctx
