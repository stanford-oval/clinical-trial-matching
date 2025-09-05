#!/usr/bin/env python3
"""
concept_first_filter.py — multi-offset-aware filter with *last-try* overlap tolerance
────────────────────────────────────────────────────────────────────────────────────
This version distinguishes each occurrence by (start,end) and:
- Collapses exact duplicate occurrences of (text,start,end).
- Audits overlaps during arbitration.
- **Allows overlapping kept spans only on the final global attempt** (last try).
  - You can still force "always allow" via ctx/env if you prefer.

Controls (robust booleans):
  • Force always-allow overlaps:
      - ctx["entity_filter_allow_overlap"] = True
      - or env SMT_ENTITY_ALLOW_OVERLAP=1
  • Allow only on the last global try (default = True):
      - ctx["entity_filter_allow_overlap_last_try"] = True/False
      - or env SMT_ENTITY_ALLOW_OVERLAP_LAST_TRY=1/0
"""

from __future__ import annotations
from typing import List, Dict, Any, Callable, Iterable, Tuple
import json, pathlib, logging, datetime as dt, os
import dspy

# ───────────────────────── CONFIG ─────────────────────────
BATCH1_SIZE = 25          # linker batch size
VERIFY_BATCH_SIZE = 20    # verifier batch size
COMP_BATCH_SIZE = 10      # max connected components per arbiter prompt
MAX_LLM_ATTEMPTS = 3
_LOG_TO_STDOUT = True

def _asbool(x, default=False):
    if x is None: return default
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.strip().lower() in ("1","true","yes","on","y","t")
    return default

# Env toggles
ALLOW_OVERLAP_ALWAYS_ENV = _asbool(os.getenv("SMT_ENTITY_ALLOW_OVERLAP"), False)
ALLOW_OVERLAP_LAST_TRY_ENV = _asbool(os.getenv("SMT_ENTITY_ALLOW_OVERLAP_LAST_TRY"), True)

# Optional heuristic: prefer longer span when one is a strict substring of another
PRUNE_SUBSTRING_OVERLAPS = _asbool(os.getenv("SMT_ENTITY_PRUNE_SUBSTRING"), True)

# ────────────────────── LOGGING HELPERS ───────────────────
def _log(msg: str) -> None:
    (print if _LOG_TO_STDOUT else logging.getLogger("entity_filter").info)(msg)

def _write_txt(p: pathlib.Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ──────────────────── RECONCILE LINKER ────────────────────
def _reconcile_linker(batch, res):
    """
    Align linker outputs to the input batch using (span, offset) keys.
    - Extra outputs are ignored.
    - Missing outputs are filled with {"concept": None}.
    - Conflicting duplicates raise.

    Returns: Dict[_Key, Dict[str, Any] | None]  # key -> concept or None
    """
    batch_by_key = { _make_key(e["text"], e["start"], e["end"]): e for e in batch }
    seen: Dict[_Key, Dict[str, Any] | None] = {}
    for raw in res:
        r = _unwrap_linker_obj(raw)
        if SPAN_FIELD not in r or "offset" not in r:
            continue
        ss, (start, end) = r[SPAN_FIELD], r["offset"]
        key = _make_key(ss, start, end)
        if key not in batch_by_key:
            continue
        concept = r.get("concept")
        if key in seen:
            prev = seen[key]
            if prev and concept and prev.get("conceptId") != concept.get("conceptId"):
                raise ValueError(f"Linker produced conflicting concepts for {ss} [{start},{end}]")
            if (prev is None) and concept:
                seen[key] = concept
        else:
            seen[key] = concept
    for key in batch_by_key:
        if key not in seen:
            seen[key] = None
    return seen

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

_ARBITER_PROMPT = """\
# === ROLE ===
You are an extracted-span checker. Apply rules 1-5 and decide keep/drop.

# === INPUT ===
<criterion>
#CRITERION#
</criterion>

<spans_with_types>
#STRINGS#
</spans_with_types>

# === OUTPUT ===
[
  {
    "extracted_span": "...",
    "offset": [123,129],
    "rule1": "YES|NO",
    "rule2": "YES|NO",
    "rule3": "YES|NO",
    "rule4": "YES|NO",
    "rule5": "YES|NO",
    "keep": "YES|NO",
    "why": "<≤20 words if keep == NO else empty>"
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

# ──────────────────── SCHEMA HELPER ───────────────────────
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

def _overlap(a: Dict[str, int], b: Dict[str, int]) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]

def _components(spans: List[Dict[str, Any]]) -> List[List[int]]:
    parent = list(range(len(spans)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        parent[ri] = rj
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if _overlap(spans[i], spans[j]):
                union(i, j)
    comps: Dict[int, List[int]] = {}
    for i in range(len(spans)):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())

def _linker_input_map(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        f"entity_{i}": {
            "text": e["text"],
            "entity_name": e["entity_name"],
            "offset": [e["start"], e["end"]],
            "candidates": e.get("candidates", []),
        }
        for i, e in enumerate(batch, 1)
    }

def _abort_if_overlapping(spans: List[Dict[str, Any]], *, allow: bool = False) -> bool:
    """
    Returns True if any overlaps are present.
    If allow=True, only logs and returns True.
    If allow=False, raises ValueError on first overlap.
    """
    overlaps: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            # Ignore exactly-identical duplicates defensively
            if (
                a.get("extracted_span") == b.get("extracted_span")
                and a["start"] == b["start"]
                and a["end"] == b["end"]
            ):
                continue
            if _overlap(a, b):
                overlaps.append((a, b))
    if overlaps:
        if allow:
            for a, b in overlaps:
                _log(
                    f"[overlap tolerated] {a['extracted_span']} ({a['start']}-{a['end']}) "
                    f"↔ {b['extracted_span']} ({b['start']}-{b['end']})"
                )
            return True
        a, b = overlaps[0]
        raise ValueError(
            f"Overlapping kept spans: "
            f"{a['extracted_span']} ({a['start']}-{a['end']}) ↔ "
            f"{b['extracted_span']} ({b['start']}-{b['end']})"
        )
    return False

def _prune_substring_overlaps(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optional heuristic: remove spans that are strict substrings of a longer span
    with identical (start,end) containment.
    """
    if not spans:
        return spans
    spans_sorted = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept: List[Dict[str, Any]] = []
    for s in spans_sorted:
        if any(s["start"] >= k["start"] and s["end"] <= k["end"] for k in kept):
            continue
        kept.append(s)
    return kept

# Handy composite key
_Key = Tuple[str, int, int]  # (text,start,end)
def _make_key(text: str, start: int, end: int) -> _Key:
    return (text, start, end)

# ─────────────────────── MAIN MODULE ──────────────────────
class LLMBasedMedicalEntityFilter(dspy.Module):
    def __init__(
        self,
        engine: Callable[[str], List[str]],
        batch1_size: int = BATCH1_SIZE,
        verify_batch_size: int = VERIFY_BATCH_SIZE,
        comp_batch_size: int = COMP_BATCH_SIZE,
        max_attempts: int = MAX_LLM_ATTEMPTS,
    ):
        super().__init__()
        self.engine = engine
        self.batch1_size = batch1_size
        self.verify_batch_size = verify_batch_size
        self.comp_batch_size = comp_batch_size
        self.max_attempts = max_attempts

    # ---------- call LLM & log ----------
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

    # ---------------- main forward pass ----------------
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        idx = ctx["current_requirement_index"]
        criterion = _raw(ctx["requirements"][idx])

        # surface entities
        llm_ents = ctx.get("llm_surface_entities_by_req", {}).get(idx, [])
        dict_ents = ctx.get("dict_surface_entities_by_req", {}).get(idx, [])
        if not (llm_ents or dict_ents):
            ctx.setdefault("valid_entities_by_req", {})[idx] = {}
            return ctx

        # EXPLODE into one-occurrence-per-item
        exploded: List[Dict[str, Any]] = []
        for ent in llm_ents + dict_ents:
            offs = ent.get("all_offsets")
            if offs:
                for off in offs:
                    exploded.append({**ent, "start": off["start"], "end": off["end"]})
            else:
                exploded.append(ent)

        # collapse exact duplicates by (text,start,end)
        uniq_exploded: Dict[_Key, Dict[str, Any]] = {}
        for e in exploded:
            key = _make_key(e["text"], e["start"], e["end"])
            if key not in uniq_exploded:
                uniq_exploded[key] = e
        if len(uniq_exploded) < len(exploded):
            _log(f"[exploded] collapsed {len(exploded) - len(uniq_exploded)} duplicate occurrence(s)")
        exploded = list(uniq_exploded.values())

        # templates (ctx may override)
        linker_tpl = ctx.get("LLMBasedMedicalEntityFilterLinker_prompt", _LINKER_PROMPT)
        verifier_tpl = ctx.get("LLMBasedMedicalEntityFilterVerifier_prompt", _VERIFY_PROMPT)
        arbiter_tpl = ctx.get("LLMBasedMedicalEntityFilterArbiter_prompt", _ARBITER_PROMPT)

        trial_id = ctx.get("trial_id", "trial")
        side = ctx.get("inc_exc", "side")

        # Config: always-allow vs last-try-only
        always_allow = _asbool(ctx.get("entity_filter_allow_overlap"), ALLOW_OVERLAP_ALWAYS_ENV)
        allow_last_try = _asbool(ctx.get("entity_filter_allow_overlap_last_try"), ALLOW_OVERLAP_LAST_TRY_ENV)
        _log(f"[cfg] allow_overlap(always)={always_allow} allow_overlap(last_try)={allow_last_try}")

        # ───────── Stage 1 : Concept linker ─────────
        concept_map: Dict[_Key, Dict[str, Any]] = {}
        for b_no, batch in enumerate(_chunks(exploded, self.batch1_size), 1):
            prompt = linker_tpl.replace("#CRITERION#", criterion).replace(
                "#CANDIDATES#", json.dumps(_linker_input_map(batch), indent=2, ensure_ascii=False)
            )
            pfx = pathlib.Path("mbench/entity_mbench/entity_logs",
                               f"{trial_id}_{side}_req{idx:03d}_stage1b{b_no:02d}")
            res = self._call_and_log(prompt, True, pfx)

            aligned = _reconcile_linker(batch, res)
            for e in batch:
                key = _make_key(e["text"], e["start"], e["end"])
                concept = aligned.get(key)
                if not concept:
                    continue  # treat as unlinked
                cand_list = e.get("candidates", [])
                c = {**concept, **_get_meta(concept.get("conceptId", ""), cand_list)}
                concept_map[key] = c

        if not concept_map:
            ctx.setdefault("valid_entities_by_req", {})[idx] = {}
            return ctx

        # ───────── Stage 2 : Link verifier ─────────
        verified_map: Dict[_Key, Dict[str, Any]] = {}
        items = list(concept_map.items())
        for b_no, batch in enumerate(_chunks(items, self.verify_batch_size), 1):
            pairs = [{
                SPAN_FIELD: k[0],
                "offset": [k[1], k[2]],
                **{k2: v for k2, v in c.items() if k2 in ("conceptId", "preferred_term", "top_type", "synonyms", "definition")}
            } for k, c in batch]
            prompt = verifier_tpl.replace("#CRITERION#", criterion).replace(
                "#PAIRS#", json.dumps(pairs, indent=2, ensure_ascii=False)
            )
            pfx = pathlib.Path("mbench/entity_mbench/entity_logs",
                               f"{trial_id}_{side}_req{idx:03d}_stage2b{b_no:02d}")
            verdicts = self._call_and_log(prompt, True, pfx)
            for v in verdicts:
                if v.get("decision") == "KEEP":
                    key = _make_key(v[SPAN_FIELD], *v["offset"])
                    verified_map[key] = concept_map[key]

        if not verified_map:
            ctx.setdefault("valid_entities_by_req", {})[idx] = {}
            return ctx

        # ───────── Stage 3 : Arbiter with last-try overlap tolerance ─────────
        spans = [{
            "extracted_span": e["text"],
            "entity_name": e["entity_name"],
            "start": e["start"],
            "end": e["end"],
            "concept": verified_map[_make_key(e["text"], e["start"], e["end"])],
        } for e in exploded if _make_key(e["text"], e["start"], e["end"]) in verified_map]

        for global_try in range(self.max_attempts):
            keep_key: set[_Key] = set()
            # Allow only if forced, or if this is the **final global attempt** and last-try is enabled
            allow_now_global = bool(always_allow or (allow_last_try and (global_try + 1 == self.max_attempts)))

            # Process connected components
            needs_global_retry = False
            for b_no, comp_batch in enumerate(_chunks(_components(spans), self.comp_batch_size), 1):
                if needs_global_retry and not allow_now_global:
                    break
                for attempt in range(self.max_attempts):
                    # Within a component, only allow overlap on the **final local attempt** of the **final global try**
                    allow_now_local = bool(always_allow or (allow_now_global and (attempt + 1 == self.max_attempts)))

                    # Make entries for arbiter; optional substring pruning
                    comp_indices = [i for comp in comp_batch for i in comp]
                    comp_spans = [spans[i] for i in comp_indices]
                    if PRUNE_SUBSTRING_OVERLAPS and (global_try + 1 == self.max_attempts) and (attempt + 1 == self.max_attempts):
                        comp_spans = _prune_substring_overlaps(comp_spans)

                    batch_entries = [{
                        SPAN_FIELD: s["extracted_span"],
                        "entity_name": s["entity_name"],
                        "offset": [s["start"], s["end"]],
                        "preferred_term": s["concept"]["preferred_term"],
                        "type": s["concept"]["top_type"],
                    } for s in comp_spans]

                    prompt = arbiter_tpl.replace("#CRITERION#", criterion).replace(
                        "#STRINGS#", json.dumps(batch_entries, indent=2, ensure_ascii=False)
                    )
                    pfx = pathlib.Path(
                        "mbench/entity_mbench/entity_logs",
                        f"{trial_id}_{side}_req{idx:03d}_stage3b{b_no:02d}_try{attempt+1}",
                    )
                    verdicts = self._call_and_log(prompt, True, pfx)
                    yes_here = {
                        _make_key(v[SPAN_FIELD], *v["offset"])
                        for v in verdicts if v.get("keep") == "YES"
                    }
                    keep_key.update(yes_here)

                    kept_spans = [sp for sp in spans
                                  if _make_key(sp["extracted_span"], sp["start"], sp["end"]) in keep_key]

                    # ── Guard overlap check so retries can proceed without crashing
                    try:
                        has_overlap = _abort_if_overlapping(kept_spans, allow=allow_now_local)
                    except ValueError:
                        has_overlap = True

                    if has_overlap and not allow_now_local:
                        # roll back this mini-batch and retry / or defer to next global try
                        keep_key.difference_update(yes_here)
                        if attempt + 1 == self.max_attempts:
                            # Local exhausted on a strict global try → defer to next global try
                            needs_global_retry = True
                            _log("[Stage 3] Overlap in component after strict local attempts; "
                                 "deferring to next global try.")
                            break  # stop local attempts for this component batch
                        continue
                    break  # local batch passes (either no overlap or overlap tolerated)

            if needs_global_retry and not allow_now_global:
                # Defer to the next global try where overlap may be tolerated
                _log(f"[Stage 3] Deferring to global retry {global_try+2}/{self.max_attempts}")
                continue

            # Global overlap check across all kept spans
            kept_all = [sp for sp in spans
                        if _make_key(sp["extracted_span"], sp["start"], sp["end"]) in keep_key]

            # ── Guard overlap check so global retry can proceed
            try:
                has_overlap_global = _abort_if_overlapping(kept_all, allow=allow_now_global)
            except ValueError:
                has_overlap_global = True

            if has_overlap_global and not allow_now_global:
                if global_try + 1 == self.max_attempts:
                    # final global try exhausted and still not allowed => raise
                    raise ValueError("Overlapping kept spans after final strict attempt.")
                _log(f"[Stage 3] Overlap after arbitration, retry {global_try+1}/{self.max_attempts}")
                continue
            break  # success (either no overlap or overlap allowed on last try / forced)

        if not keep_key:
            ctx.setdefault("valid_entities_by_req", {})[idx] = {}
            return ctx

        # ───────── Assemble result dict ─────────
        final: Dict[str, Any] = {}
        for i, sp in enumerate(spans, 1):
            key = _make_key(sp["extracted_span"], sp["start"], sp["end"])
            if key not in keep_key:
                continue
            c = sp["concept"]
            final[f"entity_{i}_{sp['start']}"] = {
                SPAN_FIELD: sp["extracted_span"],
                "entity_name": sp["entity_name"],
                "preferred_term": c["preferred_term"],
                "fully_specified_name": c["fully_specified_name"],
                "type": c["top_type"],
                "conceptId": c["conceptId"],
                "start": sp["start"],
                "end": sp["end"],
            }

        ctx.setdefault("valid_entities_by_req", {})[idx] = final

        # optional consolidated log
        overlap_mode = (
            "always"
            if always_allow else
            ("final_only" if allow_last_try else "strict")
        )
        _write_json(
            pathlib.Path(
                "mbench/entity_mbench/entity_logs",
                f"{trial_id}_{side}_req{idx:03d}_detail.json",
            ),
            {
                "generated": dt.datetime.now().isoformat(timespec="seconds"),
                "requirement": criterion,
                "spans": spans,
                "entity_name": sp["entity_name"],
                "kept": sorted(list(keep_key)),
                "final_entities": final,
                "overlap_mode": overlap_mode,
            },
        )
        return ctx
