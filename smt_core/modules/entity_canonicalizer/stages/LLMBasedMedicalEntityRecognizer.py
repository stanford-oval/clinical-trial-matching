# === LLMBasedMedicalEntityRecognizer (batched, resilient) ====================
from __future__ import annotations
from typing import Dict, Any, List, Callable
import json, re, pathlib
import dspy

# Match the top‑level tags exactly
TAG_BATCH = re.compile(
    r"<identified_medical_entities_by_requirement>\s*(\[\s*.*?\s*])\s*</identified_medical_entities_by_requirement>",
    flags=re.I | re.S,
)

# ────────────────────────── helpers ────────────────────────────

def _raw_text(req: Any) -> str:
    if isinstance(req, str):
        return req
    if isinstance(req, dict):
        for k in ("text", "requirement", "requirement_text", "sentence"):
            if k in req and req[k]:
                return str(req[k])
    return json.dumps(req, ensure_ascii=False)


def _write_txt(p: pathlib.Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")


def _extract_json_array(text: str) -> list | None:
    """Return parsed JSON array if we can slice [ ... ] out; else None."""
    if text is None:
        return None
    s = str(text).strip()
    first, last = s.find("["), s.rfind("]")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        return json.loads(s[first : last + 1])
    except Exception:
        return None


def _normalise_items(parsed: Any) -> List[Dict[str, Any]]:
    """Validate and normalize the parsed JSON structure."""
    if not isinstance(parsed, list):
        raise ValueError("Top-level JSON must be an array.")

    out: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        req = item.get("requirement")
        ents = item.get("entities", [])
        if not isinstance(idx, int) or not isinstance(req, str) or not isinstance(ents, list):
            continue

        norm_ents: List[Dict[str, Any]] = []
        for e in ents:
            if isinstance(e, str):
                s = e.strip()
                if s:
                    #norm_ents.append({"text": s})
                    norm_ents.append({"text": s, "entity_name": None})
            elif isinstance(e, dict):
                # prefer extracted_span (same as original text) and carry entity_name if provided
                span = (e.get("extracted_span") or e.get("text") or "").strip() if e.get("extracted_span") or e.get("text") else ""
                name = e.get("entity_name")
                if span:
                    norm_ents.append({"text": span, "entity_name": name if isinstance(name, str) and name.strip() else None})
        out.append({"index": idx, "requirement": req, "entities": norm_ents})
    return out


def _derive_offsets(text: str, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach start/end offsets for each entity mention (case‑insensitive)."""
    kept: List[Dict[str, Any]] = []
    for ent in ents:
        # span = ent.get("text", "")
        # if not span:
        #     continue
        span = ent.get("text") or ""
        if not isinstance(span, str) or not span:
            continue

        entity_name = ent.get("entity_name", None)

        matches = [m.span() for m in re.finditer(re.escape(span), text, flags=re.I)]
        if not matches:
            continue

        for s, e in matches:
            kept.append(
                {
                    "text": span,
                    "entity_name": entity_name,
                    "start": s,
                    "end": e,
                    "occurrences": [(s, e)],
                    "all_offsets": [{"start": s, "end": e}],
                }
            )
    return kept

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────── main module (batched) ─────────────────────

class LLMBasedMedicalEntityRecognizer(dspy.Module):
    """Batched LLM‑NER: splits requirements into manageable chunks before calling the LLM."""

    def __init__(
        self,
        engine: Callable[[str], List[str] | str],
        batch_size: int = 10,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.batch_size = batch_size
        self.verbose = verbose

    # ───────────────────────── private helpers ─────────────────────────

    def _run_one_batch(
        self,
        ctx: Dict[str, Any],
        req_slice: List[Any],
        start_idx: int,
        pfx: pathlib.Path,
        batch_id: int,
        max_retries: int,
    ) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
        """Call LLM on one slice of requirements and return results."""
        # Build input JSON for the prompt
        req_json = [
            {"index": i + start_idx, "requirement": _raw_text(r)}
            for i, r in enumerate(req_slice)
        ]
        req_json_str = json.dumps(
            req_json,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        )

        prompt = ctx["LLMBasedMedicalEntityRecognizer_prompt"].replace(
            "#REQ_JSON#", req_json_str
        )

        # Log prompt
        _write_txt(pfx.with_name(f"{pfx.name}_batch{batch_id}_prompt.txt"), prompt)

        parsed_json = None
        last_raw_original = ""

        for attempt in range(1, max_retries + 1):
            raw = self.engine(prompt)
            llm_raw = raw[0] if isinstance(raw, list) else raw
            last_raw_original = llm_raw
            _write_txt(
                pfx.with_name(f"{pfx.name}_batch{batch_id}_raw1_attempt{attempt}.txt"),
                llm_raw,
            )

            m = TAG_BATCH.search(llm_raw)
            json_txt = (m.group(1) if m else llm_raw).strip()
            parsed_json = _extract_json_array(json_txt)
            if parsed_json is not None:
                break

        # Self‑repair
        if parsed_json is None:
            repair_prompt = (
                "Rewrite the following EXACT content as a JSON array only.\n"
                "Schema: [{\"index\": int, \"requirement\": str, \"entities\": [str, ...]}]\n"
                "No prose. Output must start with '[' and end with ']'.\n\n=== CONTENT ===\n"
                + last_raw_original
            )
            _write_txt(
                pfx.with_name(f"{pfx.name}_batch{batch_id}_repair_prompt.txt"), repair_prompt
            )
            raw2 = self.engine(repair_prompt)
            llm_raw2 = raw2[0] if isinstance(raw2, list) else raw2
            _write_txt(
                pfx.with_name(f"{pfx.name}_batch{batch_id}_raw2_repair.txt"), llm_raw2
            )
            parsed_json = _extract_json_array(llm_raw2)

        # Strict fallback
        if parsed_json is None:
            strict_prompt = (
                "Return ONLY a JSON array per schema below. If you cannot, return [].\n"
                "Schema: [{\"index\": int, \"requirement\": str, \"entities\": [str, ...]}]"
            )
            _write_txt(
                pfx.with_name(f"{pfx.name}_batch{batch_id}_strict_prompt.txt"), strict_prompt
            )
            raw3 = self.engine(strict_prompt)
            llm_raw3 = raw3[0] if isinstance(raw3, list) else raw3
            _write_txt(
                pfx.with_name(f"{pfx.name}_batch{batch_id}_raw3_strict.txt"), llm_raw3
            )
            parsed_json = _extract_json_array(llm_raw3) or []

        parsed = _normalise_items(parsed_json)

        # Build a map from index → raw LLM item (single requirement’s block) if available
        raw_items_by_idx: Dict[int, Any] = {}
        if isinstance(parsed_json, list):
            for it in parsed_json:
                if isinstance(it, dict) and isinstance(it.get("index"), int):
                    raw_items_by_idx[it["index"]] = it

        per_req: Dict[int, List[Dict[str, Any]]] = {}
        echoed: Dict[int, str] = {}
        for block in parsed:
            idx = block["index"]
            req_text = block["requirement"]
            echoed[idx] = req_text
            # per_req[idx] = _derive_offsets(req_text, block["entities"])
            mentions = _derive_offsets(req_text, block["entities"])
            per_req[idx] = mentions

            # ── Per-requirement logging ─────────────────────────────────────
            pfx1 = pathlib.Path(str(pfx).replace("ner_logs", "recognizer_logs"))
            # 1) Raw LLM output for this requirement (as a single JSON object) → .txt
            raw_item = raw_items_by_idx.get(idx)
            _write_txt(
                pfx1.with_name(f"{pfx1.name}_req{idx:04d}_raw.txt"),
                json.dumps(raw_item if raw_item is not None else {}, ensure_ascii=False, indent=2),
            )

            # 2) Processed mentions with offsets → .json
            _write_json(
                pfx1.with_name(f"{pfx1.name}_req{idx:04d}_mentions.json"),
                {
                    "index": idx,
                    "requirement": req_text,
                    "mentions": mentions,  # each mention has text, entity_name (if any), start/end, etc.
                },
            )
            # ────────────────────────────────────────────────────────────────

        return per_req, echoed

    # ───────────────────────────── forward ─────────────────────────────

    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        reqs = ctx.get("requirements", [])
        if not reqs:
            return ctx

        batch_size = ctx.get("llm_req_batch_size", self.batch_size)
        max_retries = ctx.get("llm_retry", 3)

        trial_id = ctx.get("trial_id", "trial")
        side = ctx.get("inc_exc", "side")
        pfx = pathlib.Path("mbench/entity_mbench/ner_logs", f"{trial_id}_{side}_batched")

        per_req_all: Dict[int, List[Dict[str, Any]]] = {}
        echoed_all: Dict[int, str] = {}

        for b_id, start in enumerate(range(0, len(reqs), batch_size)):
            slice_reqs = reqs[start : start + batch_size]
            per_req, echoed = self._run_one_batch(
                ctx, slice_reqs, start, pfx, b_id, max_retries
            )
            per_req_all.update(per_req)
            echoed_all.update(echoed)

        # Fill gaps (if any)
        for i in range(len(reqs)):
            per_req_all.setdefault(i, [])
            echoed_all.setdefault(i, _raw_text(reqs[i]))

        ctx.setdefault("llm_surface_entities_by_req", {}).update(per_req_all)
        ctx["requirements_echo"] = echoed_all

        if self.verbose:
            total_mentions = sum(len(v) for v in per_req_all.values())
            print(
                f"[LLM-NER] {total_mentions} mentions across {len(reqs)} requirements "
                f"in {len(range(0, len(reqs), batch_size))} batch(es)."
            )

        return ctx
