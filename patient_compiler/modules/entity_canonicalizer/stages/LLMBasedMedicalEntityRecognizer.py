#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#  LLMBased Medical‑Entity Recognizer (one‑span‑per‑occurrence, self‑contained)
# --------------------------------------------------------------------------- #
from __future__ import annotations
from typing import Dict, Any, List, Callable
import json, re, dspy

# --------------------------------------------------------------------------- #
#  Prompt template                                                            #
# --------------------------------------------------------------------------- #
PROMPT = (
    "# === ROLE ===\n"
    "You are a Medical Entity Recognizer.\n"
    "Task: Identify EVERY medically relevant entity in the clinical‑trial "
    "eligibility criterion below. Return ONLY a JSON array of the exact spans.\n\n"
    "# === INPUT ===\n<criterion>\n#REQUIREMENT_TEXT#\n</criterion>\n\n"
    "# === EXTRACTION RULES ===\n"
    "1. Maximise recall — include drugs, procedures, symptoms, diseases, organisms, "
    "   anatomy, events, specimens, substances, etc.\n"
    "2. If spans overlap, keep them all.\n"
    "3. Keep each span **exactly** as it appears.\n\n"
    "# === OUTPUT FORMAT ===\n"
    "<identified_medical_entities>\n"
    '["non‑small cell lung cancer", "cisplatin"]\n'
    "</identified_medical_entities>\n\n"
    "Return **only** that JSON array."
)

TAG_RE = re.compile(      # handles optional wrapper tags
    r"<identified_medical_entities>\s*(\[\s*.*?\s*])\s*</identified_medical_entities>",
    flags=re.I | re.S,
)

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _raw_text(req: Any) -> str:
    """Return the criterion sentence regardless of wrapper schema."""
    if isinstance(req, str):
        return req
    if isinstance(req, dict):
        for k in ("text", "requirement", "requirement_text", "sentence"):
            if k in req and req[k]:
                return str(req[k])
    return json.dumps(req, ensure_ascii=False)   # fallback


def _derive_offsets(text: str, ents: List[Dict[str, Any]]) -> None:
    """
    Attach all offset info for each *unique* span. Removes hallucinations.
    Mutates *ents* in‑place.
    """
    kept: List[Dict[str, Any]] = []

    for ent in ents:
        matches = [m.span() for m in re.finditer(re.escape(ent["text"]), text, flags=re.I)]
        if not matches:
            continue  # hallucination → drop

        ent["occurrences"] = matches
        ent["start"], ent["end"] = matches[0]
        ent["all_offsets"] = [{"start": s, "end": e} for (s, e) in matches]
        kept.append(ent)

    ents[:] = kept


# --------------------------------------------------------------------------- #
#  Main module                                                                #
# --------------------------------------------------------------------------- #
class LLMBasedMedicalEntityRecognizer(dspy.Module):
    """
    Returns *one dictionary per mention* (i.e., per (start,end) pair).

    ctx["llm_surface_entities_by_req"][idx] => e.g.
        [
          {
            "text": "aspirin",
            "start": 26,
            "end":   33,
            "occurrences": [(26,33)],
            "all_offsets": [{"start":26,"end":33}]
          },
          {
            "text": "aspirin",
            "start": 64,
            "end":   71,
            "occurrences": [(64,71)],
            "all_offsets": [{"start":64,"end":71}]
          },
          ...
        ]
    """

    def __init__(
        self,
        engine: Callable[[str], List[str]],
        max_attempts: int = 3,
        verbose: bool = True,
    ):
        super().__init__()
        self.engine       = engine
        self.max_attempts = max_attempts
        self.verbose      = verbose

    # ------------------------------------------------------------------ #
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        import json, re, os

        idx  = ctx["current_requirement_index"]
        crit = _raw_text(ctx["requirements"][idx])

        # Build prompt (you can override PROMPT in ctx if you've updated it to the new schema)
        prompt = ctx.get("LLMBasedMedicalEntityRecognizer_prompt", PROMPT) \
                .replace("#REQUIREMENT_TEXT#", crit)

        # --- helper: tolerant parser for new LLM output -------------------------
        # Accepts:
        #   { "0": ["entity","span"], "1": ["entity","span"], ... }
        #   { "0": ("entity","span"), ... }             # tuple-like (non-JSON) -> recovered
        #   [ ["entity","span"], ... ]                   # array-of-pairs
        #   [ {"entity_name": "...", "exact_span":"..."} , ... ]
        #   codefenced or wrapped in <identified_medical_entities> ... </...>
        CODEFENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.I | re.M)
        def _sanitize(s: str) -> str:
            m = TAG_RE.search(s)
            if m:
                s = m.group(1)
            return CODEFENCE_RE.sub("", s.strip())

        TUPLE_PAIR_RE = re.compile(
            r"""["']?\s*(\d+)["']?\s*:\s*\(\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*\)""",
            flags=re.S
        )

        def _parse_pairs(raw: str) -> List[Dict[str, str]]:
            """
            Return a normalized list of {"entity_name": str, "exact_span": str}.
            Accepts many shapes:
            - [ {"entity_name": "...", "extracted_span": "..."} , ... ]
            - [ {"entity_name": "...", "exact_span": "..."} , ... ]
            - { "whatever": [ {...}, {...} ] }
            - { "0": ["entity","span"], "1": ["entity","span"], ... }
            - [ ["entity","span"], ... ] or [("entity","span"), ...]
            - Tuple-like dict with no valid JSON: {"0": ("entity","span"), ...}
            - Malformed wrapper: "{ [ {...}, {...} ] }"
            """
            s = _sanitize(raw)

            # --- helper: normalize a single dict item into {entity_name, exact_span}
            def _norm_item(d: Dict[str, Any]) -> Dict[str, str] | None:
                # Accept common variants for the span field
                span_keys = ("extracted_span", "exact_span", "span", "surface", "text", "extractedSpan", "exactSpan")
                name = d.get("entity_name") or d.get("entity") or d.get("name")
                span = None
                for k in span_keys:
                    if k in d and d[k]:
                        span = d[k]
                        break
                if name is None or span is None:
                    return None
                return {"entity_name": str(name), "exact_span": str(span)}

            # --- 0) Quick fix for "{ [ ... ] }" (invalid JSON with unkeyed array inside object)
            if s.lstrip().startswith("{") and re.match(r"^\{\s*\[", s):
                # strip the outermost braces safely
                # find first '[' after '{' and last ']' before '}'
                m_open = re.search(r"\[", s)
                m_close = re.search(r"\][^]\}] *\}$|\]\s*\}$", s)
                if m_open:
                    # naive but effective: take substring between first '[' and last ']'
                    last_br = s.rfind("]")
                    if last_br != -1 and last_br > m_open.start():
                        s = s[m_open.start(): last_br + 1]

            # --- 1) direct JSON parse
            try:
                obj = json.loads(s)
                pairs: List[Dict[str, str]] = []

                if isinstance(obj, list):
                    # list of dicts or list of 2-tuples
                    for it in obj:
                        if isinstance(it, dict):
                            normalized = _norm_item(it)
                            if normalized:
                                pairs.append(normalized)
                        elif isinstance(it, (list, tuple)) and len(it) == 2:
                            pairs.append({"entity_name": str(it[0]), "exact_span": str(it[1])})

                elif isinstance(obj, dict):
                    # (a) single top-level key whose value is a list
                    if len(obj) == 1 and isinstance(next(iter(obj.values())), list):
                        arr = next(iter(obj.values()))
                        for it in arr:
                            if isinstance(it, dict):
                                normalized = _norm_item(it)
                                if normalized:
                                    pairs.append(normalized)
                            elif isinstance(it, (list, tuple)) and len(it) == 2:
                                pairs.append({"entity_name": str(it[0]), "exact_span": str(it[1])})
                    # (b) numeric-keyed dict or arbitrary values
                    else:
                        for _, v in obj.items():
                            if isinstance(v, dict):
                                normalized = _norm_item(v)
                                if normalized:
                                    pairs.append(normalized)
                            elif isinstance(v, (list, tuple)) and len(v) == 2:
                                pairs.append({"entity_name": str(v[0]), "exact_span": str(v[1])})
                            elif isinstance(v, list):
                                for it in v:
                                    if isinstance(it, dict):
                                        normalized = _norm_item(it)
                                        if normalized:
                                            pairs.append(normalized)
                                    elif isinstance(it, (list, tuple)) and len(it) == 2:
                                        pairs.append({"entity_name": str(it[0]), "exact_span": str(it[1])})

                if pairs:
                    return pairs
            except Exception:
                pass

            # --- 2) recover tuple-like dicts that aren't valid JSON: {"0": ("entity","span"), ...}
            TUPLE_PAIR_RE = re.compile(
                r"""["']?\s*(\d+)["']?\s*:\s*\(\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*\)""",
                flags=re.S
            )
            recovered: List[Dict[str, str]] = []
            for m in TUPLE_PAIR_RE.finditer(s):
                recovered.append({"entity_name": m.group(2), "exact_span": m.group(3)})
            if recovered:
                return recovered

            # --- 3) fallback: extract array-of-dicts with entity_name + extracted_span/exact_span via regex
            #    This is lenient but helps in malformed outputs.
            OBJ_RE = re.compile(
                r'\{\s*"entity_name"\s*:\s*"(.*?)"\s*,\s*"(?:extracted_span|exact_span|span|surface|text|extractedSpan|exactSpan)"\s*:\s*"(.*?)"\s*\}',
                flags=re.S | re.I
            )
            recovered = [{"entity_name": m.group(1), "exact_span": m.group(2)} for m in OBJ_RE.finditer(s)]
            if recovered:
                return recovered

            # --- 4) last resort: extract array pairs like ["entity","span"]
            ARR_PAIR_RE = re.compile(r'\[\s*"(.*?)"\s*,\s*"(.*?)"\s*\]')
            recovered = [{"entity_name": m.group(1), "exact_span": m.group(2)} for m in ARR_PAIR_RE.finditer(s)]
            if recovered:
                return recovered

            raise ValueError("Unrecognized LLM output format for entity tuples.")


        # --- helper: derive offsets from ORIGINAL text for each exact_span ------
        def _attach_offsets_from_exact_span(text: str, pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for p in pairs:
                span = p["exact_span"]
                hits = [m.span() for m in re.finditer(re.escape(span), text, flags=re.I)]
                if not hits:
                    # skip hallucinations (span not found in text)
                    continue
                # one record per occurrence
                for s0, e0 in hits:
                    out.append({
                        "entity_name": p["entity_name"],
                        "text": span,              # keep downstream-compatible key
                        "start": s0,
                        "end":   e0,
                        "occurrences": [(s0, e0)],
                        "all_offsets": [{"start": s0, "end": e0}],
                    })
            return out

        # --- main LLM attempt loop ---------------------------------------------
        entities: List[Dict[str, Any]] = []
        raw_pairs: List[Dict[str, str]] = []   # parsed pairs (if parse succeeds)
        last_exc: Exception | None = None
        captured_llm_raw: str = ""             # << NEW: keep the literal LLM output

        for attempt in range(1, self.max_attempts + 1):
            llm_raw = self.engine(prompt)[0]
            captured_llm_raw = llm_raw  # << NEW: capture the raw text exactly as returned
            try:
                # 1) Parse raw pairs (entity_name, exact_span) from LLM output
                raw_pairs = _parse_pairs(llm_raw)

                # 2) Derive offsets from ORIGINAL text for each exact_span
                ents_with_offsets = _attach_offsets_from_exact_span(crit, raw_pairs)
                if not ents_with_offsets:
                    raise ValueError("no recognised span occurs in the criterion")

                entities = ents_with_offsets
                break
            except Exception as exc:
                last_exc = exc
                if self.verbose:
                    print(f"[LLM-NER] attempt {attempt} failed: {exc}")

        if not entities and self.verbose:
            print(f"[LLM-NER] giving up on requirement #{idx}")
            if last_exc:
                print(f"[LLM-NER] last error: {last_exc}")

        # ---------------- store in DSPy context (downstream-friendly) ----------
        ctx.setdefault("llm_surface_entities_by_req", {})[idx] = entities
        ctx.setdefault("llm_raw_pairs_by_req", {})[idx] = raw_pairs

        # ---------------- write TWO files per requirement -----------------------
        import os
        log_dir = ctx.get("llm_ner_log_dir", "mbench/entity_mbench/entity_recognizer_logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

        # 1) RAW LLM OUTPUT as plain text (no wrapping keys)
        #    If we failed to parse anything, keep a _failed suffix for clarity.
        note_id = ctx["note_id"]
        raw_txt_name = f"{note_id}_req_{idx}_raw.txt" # if entities else f"req_{idx}_raw_failed.txt"
        raw_txt_path = os.path.join(log_dir, raw_txt_name)
        try:
            with open(raw_txt_path, "w", encoding="utf-8") as f:
                f.write(captured_llm_raw if captured_llm_raw is not None else "")
        except Exception as exc:
            if self.verbose:
                print(f"[LLM-NER] failed to write raw txt {raw_txt_path}: {exc}")
                
        # 2) Entities with offsets (JSON) — keep as before
        ents_path  = os.path.join(log_dir, f"{note_id}_req_{idx}_entities.json")
        try:
            with open(ents_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "requirement_index": idx,
                        "original_text": crit,
                        "entities": entities
                    },
                    f, ensure_ascii=False, indent=2
                )
        except Exception as exc:
            if self.verbose:
                print(f"[LLM-NER] failed to write entities json {ents_path}: {exc}")

        # ---------------- human-readable log -----------------------------------
        if self.verbose:
            if not entities:
                print("\n[LLM-NER] raw LLM output saved to:", raw_txt_path)
            else:
                print("\n[LLM-NER] raw LLM output saved to:", raw_txt_path)
                print("[LLM-NER] identified entities (with offsets):")
                for ent in entities:
                    print(f'  • {ent.get("entity_name","?")}  ::  "{ent["text"]}" [{ent["start"]}:{ent["end"]}]')

        return ctx
