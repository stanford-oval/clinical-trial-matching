#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, re, time, pathlib
from typing import Any, Callable, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

# ───────────────────────── helpers ─────────────────────────

def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._:-]+', '_', str(s or ""))

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
def _strip_fence(s: str) -> str:
    return _JSON_FENCE.sub("", s or "")

def _extract_json_any(text: str) -> Optional[Any]:
    if text is None:
        return None
    s = str(text).strip()
    # Prefer object (batched)
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(s[first : last + 1])
        except Exception:
            pass
    # Fallback: array (legacy single)
    first, last = s.find("["), s.rfind("]")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(s[first : last + 1])
        except Exception:
            pass
    return None

def _norm_str(x: Any) -> str:
    return str(x or "").strip()

def _is_substring(a: str, b: str) -> bool:
    return a != b and a in b

def _keep_most_specific_strings(cands: List[str]) -> List[str]:
    """
    Keep strings that are not strict substrings of another kept candidate.
    Stable: preserves original order after greedy long-first pass.
    """
    uniq = list(dict.fromkeys(_norm_str(c) for c in cands if _norm_str(c)))
    uniq_sorted = sorted(uniq, key=lambda s: (-len(s), s))
    kept: List[str] = []
    for s in uniq_sorted:
        if any(_is_substring(s, t) for t in kept):
            continue
        kept.append(s)
    kept_set = set(kept)
    return [c for c in uniq if c in kept_set]

def _gather_blocked_texts_from_ctx(req_text: str, blocked_offsets: List[Tuple[int,int]]) -> List[str]:
    """
    Convert blocked (start,end) offsets → exact substrings for string-based blocking.
    """
    out: List[str] = []
    L = len(req_text)
    for (st, ed) in blocked_offsets or []:
        try:
            st_i, ed_i = int(st), int(ed)
            if 0 <= st_i <= ed_i <= L:
                out.append(req_text[st_i:ed_i])
        except Exception:
            continue
    return list(dict.fromkeys(out))

def _gather_blocked_from_ctx(ctx: Dict[str, Any]) -> Dict[int, List[Tuple[int,int]]]:
    """
    Best-effort collector of already captured spans (offsets) by requirement id.
    Uses:
      - ctx["final_qualifiers_by_requirement"][rid]["qualifiers"][*]["qualifier_start"/"qualifier_end"]
      - ctx["valid_entities_by_req"][rid][*]["start"/"end"]
    """
    out: Dict[int, List[Tuple[int,int]]] = {}
    fqbr = ctx.get("final_qualifiers_by_requirement", {}) or {}
    vebr = ctx.get("valid_entities_by_req", {}) or {}

    for rid_str, block in fqbr.items():
        try: rid = int(rid_str)
        except Exception: continue
        for q in block.get("qualifiers", []) or []:
            st, ed = q.get("qualifier_start"), q.get("qualifier_end")
            if st is None or ed is None:
                continue
            out.setdefault(rid, []).append((int(st), int(ed)))

    for rid_str, edict in vebr.items():
        try: rid = int(rid_str)
        except Exception: continue
        for ev in (edict or {}).values():
            st, ed = ev.get("start"), ev.get("end")
            if st is None or ed is None:
                continue
            out.setdefault(rid, []).append((int(st), int(ed)))
    return out

def _rid_tag(rid: int | str) -> str:
    try:
        return f"req{int(rid):03d}"
    except Exception:
        rid_str = str(rid)
        safe = re.sub(r'[^A-Za-z0-9._:-]+', '_', rid_str)
        return f"req_{safe}"

def _raw_req_text(req: Any) -> str:
    if isinstance(req, str):
        return req
    if isinstance(req, dict):
        for k in ("text", "requirement", "requirement_text", "sentence"):
            if k in req and req[k]:
                return str(req[k])
    return json.dumps(req, ensure_ascii=False)

def _write_txt(p: pathlib.Path, txt: str, *, enable: bool) -> None:
    if not enable:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def _write_json(p: pathlib.Path, obj: Any, *, enable: bool) -> None:
    if not enable:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _escape_for_regex(s: str) -> str:
    return re.escape(s)

def _overlap(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]

def _merge_overlaps(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not intervals: return []
    ints = sorted({(int(s), int(e)) for (s,e) in intervals})
    merged = [ints[0]]
    for s,e in ints[1:]:
        ps,pe = merged[-1]
        if s <= pe:  # touch/overlap -> merge
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s,e))
    return merged

def _resolve_conflicts_keep_longest(hits: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """
    hits: list of (start,end,text) possibly overlapping across different entity strings.
    Keep a maximal set of NON-overlapping intervals, preferring longer spans, then earlier start.
    """
    hits_sorted = sorted(hits, key=lambda x: (-(x[1]-x[0]), x[0], x[1], x[2]))
    kept: List[Tuple[int,int,str]] = []
    for h in hits_sorted:
        if any(_overlap((h[0],h[1]), (k[0],k[1])) for k in kept):
            continue
        kept.append(h)
    return sorted(kept, key=lambda x: (x[0], x[1]))

def _call_engine(engine, prompt: str, *, temperature: float = 0.0) -> str:
    out = engine(prompt, temperature=temperature) if callable(engine) else engine(prompt)
    return out[0] if isinstance(out, (list, tuple)) else str(out)

def _explode_entity_strings_to_offsets(
    criterion: str,
    entity_strings: List[str],
    *,
    blocked_offsets: List[Tuple[int,int]] | None = None,
    allow_overlaps_between_entities: bool = False,
    case_sensitive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Given the requirement `criterion` and a list of entity strings,
    return [{"text": str, "offset": [start,end]}, ...].

    - Explodes to *all* occurrences for each string.
    - Drops any occurrence that overlaps a blocked (already-captured) offset.
    - If `allow_overlaps_between_entities=False`, resolves inter-entity overlaps by keeping the longest spans.
    """
    blocked = _merge_overlaps(blocked_offsets or [])
    all_hits: List[Tuple[int,int,str]] = []
    _pat_cache: Dict[Tuple[str,bool], re.Pattern] = {}

    for s in entity_strings:
        if not s:
            continue
        key = (s, case_sensitive)
        pat = _pat_cache.get(key)
        if pat is None:
            flags = 0 if case_sensitive else re.IGNORECASE
            pat = re.compile(_escape_for_regex(s), flags)
            _pat_cache[key] = pat
        for m in pat.finditer(criterion):
            st, ed = m.start(), m.end()
            if any(_overlap((st,ed), b) for b in blocked):
                continue
            all_hits.append((st, ed, s))

    if not allow_overlaps_between_entities:
        all_hits = _resolve_conflicts_keep_longest(all_hits)

    # stable, dedup by (start,end)
    seen = set()
    out: List[Dict[str, Any]] = []
    for st, ed, s in all_hits:
        key = (st, ed)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": s, "offset": [st, ed]})
    return out

# ───────────────────────── main module ─────────────────────────

class SMTProgrammerFreeEntityExtractor(dspy.Module):
    """
    Batched LLM extractor for *additional* medically relevant entities as TEXT SPANS ONLY,
    with optional **offset explosion** in-module.

    • True batch prompting: one LLM call per batch (size = `batch_size`).
    • Prompt INPUT: JSON array of items:
         { "id": <int requirement index>, "criterion": str, "already_captured": [str, ...] }
      (The "id" is explicitly the requirement index / position.)
    • Prompt OUTPUT: JSON object mapping ids to arrays of strings:
         { "0": ["entity1","entity2",...], "1": ["..."] }  ← model may emit str or int keys; parser handles both.
    • Offsets are computed later OR inside this module (config).
    • Blocks by exact string matches against already-captured (derived from offsets if available).
    • Applies MOST-SPECIFIC string filter (drops strict substrings of other candidates).

    Writes:
      - ctx["free_additional_spans_by_req"]              = { int rid -> [ {"text": str, "offset":[st,ed]}, ... ] }
      - ctx["free_additional_spans_offsets_by_req"]      = same as above (alias)
      - ctx["free_additional_entity_strings_by_req"]     = { int rid -> [str, ...] }  # raw strings
    """

    def __init__(
        self,
        engine: Optional[Callable[[str], List[str] | str]] = None,
        batch_size: int = 10,
        max_retries: int = 2,
        temperature: float = 0.0,
        verbose: bool = True,
        microbench_dir: Optional[str | pathlib.Path] = None,
        # offset explosion config
        compute_offsets: bool = True,
        allow_overlaps_between_entities: bool = False,
        case_sensitive_offsets: bool = True,
        # optimization knobs
        default_concurrency: int = 4,
        debug_io: bool = False,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.temperature = temperature
        self.verbose = verbose
        # Logging roots
        self.mbench_dir = pathlib.Path(microbench_dir).expanduser() if microbench_dir else None
        # offsets
        self.compute_offsets = bool(compute_offsets)
        self.allow_overlaps_between_entities = bool(allow_overlaps_between_entities)
        self.case_sensitive_offsets = bool(case_sensitive_offsets)
        # optimization
        self.default_concurrency = int(default_concurrency)
        self.debug_io = bool(debug_io)

    def _make_batch_payload(
        self,
        start_idx: int,
        slice_reqs: List[Any],
        blocked_by_req: Dict[int, List[Tuple[int,int]]]
    ) -> List[Dict[str, Any]]:
        """
        Build the LLM payload. The 'id' field is the requirement index (integer).
        """
        payload: List[Dict[str, Any]] = []
        for local_i, req_obj in enumerate(slice_reqs):
            rid = start_idx + local_i  # integer requirement index
            crit = _raw_req_text(req_obj)
            blocked_offsets = blocked_by_req.get(rid, []) or []
            already_strings = _gather_blocked_texts_from_ctx(crit, blocked_offsets)
            payload.append({
                "id": rid,                    # ← INT id (explicit requirement index)
                "criterion": crit,
                "already_captured": already_strings,
            })
        return payload

    def _parse_batched_output(self, raw: str) -> Optional[Dict[str, List[str]]]:
        """
        Parse model output. Robust to string or integer keys.
        Returns a dict keyed by STRING form of the id, but fills both "k" and "str(int(k))" when possible.
        """
        payload = _extract_json_any(_strip_fence(raw))
        if payload is None:
            return None

        def _to_str_key(k: Any) -> str:
            # normalize keys to str; if numeric, use its int-string form
            try:
                return str(int(k))
            except Exception:
                return str(k)

        if isinstance(payload, dict):
            out: Dict[str, List[str]] = {}
            for k, v in payload.items():
                if isinstance(v, list):
                    vals = [str(x) for x in v if isinstance(x, (str, int, float))]
                    ks = _to_str_key(k)
                    out[ks] = vals
                    # if key was a non-canonical string that can also be int, ensure both aliases map
                    try:
                        k_int = int(k)
                        out[str(k_int)] = vals
                    except Exception:
                        pass
            return out

        elif isinstance(payload, list):
            # Legacy: single-item array → map to "0"
            return {"0": [str(x) for x in payload if isinstance(x, (str, int, float))]}

        return None

    def _postfilter_one(
        self,
        criterion: str,
        already_strings: List[str],
        candidates: List[str]
    ) -> List[str]:
        # Keep only verbatim, non-empty substrings of the criterion
        crit_str = str(criterion)
        filtered = []
        for s in candidates:
            s_norm = _norm_str(s)
            if s_norm and s_norm in crit_str:
                filtered.append(s_norm)
        # Remove already captured (exact string match)
        if already_strings:
            blocked = set(already_strings)
            filtered = [s for s in filtered if s not in blocked]
        # Most-specific strings only
        return _keep_most_specific_strings(filtered)

    def _run_one_batch_llm(
        self,
        start_idx: int,
        slice_reqs: List[Any],
        blocked_by_req: Dict[int, List[Tuple[int,int]]],
        ctx: Dict[str, Any],
        trial_id: str,
        side: str,
        batch_id: int,
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[Dict[str, Any]]]]:
        assert self.engine is not None, "No engine provided to SMTProgrammerFreeEntityExtractor"

        # Build payload and prompt
        payload = self._make_batch_payload(start_idx, slice_reqs, blocked_by_req)
        items_json = json.dumps(payload, ensure_ascii=False, indent=2)
        prompt = ctx["SMTProgrammerFreeEntityExtractor_prompt"].replace("{ITEMS_JSON}", items_json)

        # Microbench
        mb_dir: Optional[pathlib.Path] = (self.mbench_dir / _safe_id(trial_id) / _safe_id(side)) if self.mbench_dir else None
        enable_io = bool(self.debug_io and mb_dir is not None)
        if mb_dir is not None:
            mb_dir.mkdir(parents=True, exist_ok=True)
            if enable_io:
                for obj in payload:
                    tag = _rid_tag(obj["id"])
                    _write_json(mb_dir / f"{tag}_input.json", {
                        "requirement": obj["criterion"],
                        "already_captured": obj["already_captured"],
                    }, enable=True)
                ts = time.strftime("%Y%m%d-%H%M%S")
                _write_txt(mb_dir / f"gfree_batch{batch_id:03d}_{ts}_prompt.txt", prompt, enable=True)

        # Call LLM with retries
        parsed_map: Optional[Dict[str, List[str]]] = None
        last_raw = ""
        for attempt in range(1, self.max_retries + 1):
            llm_raw = _call_engine(self.engine, prompt, temperature=self.temperature)
            last_raw = llm_raw
            if enable_io:
                _write_txt(mb_dir / f"gfree_batch{batch_id:03d}_raw_attempt{attempt}.txt", llm_raw, enable=True)

            parsed_map = self._parse_batched_output(llm_raw)
            if parsed_map is not None:
                break

        if parsed_map is None:
            # one-shot repair
            repair = (
                "Return ONLY a JSON object mapping ids (requirement indices) to arrays of strings.\n"
                'Example: {"0":["entity a","entity b"], "1":["entity c"]}\n'
                "No prose. If you cannot recover, return {}.\n\n=== CONTENT ===\n" + last_raw
            )
            llm_raw2 = _call_engine(self.engine, repair, temperature=self.temperature)
            if enable_io:
                _write_txt(mb_dir / f"gfree_batch{batch_id:03d}_raw_repair.txt", llm_raw2, enable=True)
            parsed_map = self._parse_batched_output(llm_raw2) or {}

        # Post-filter per item → strings & (optional) offsets
        strings_by_req: Dict[int, List[str]] = {}
        offsets_by_req: Dict[int, List[Dict[str, Any]]] = {}

        # Maps for this batch
        crit_map: Dict[str, str] = {str(obj["id"]): obj["criterion"] for obj in payload}
        already_map: Dict[str, List[str]] = {str(obj["id"]): obj["already_captured"] for obj in payload}

        for obj in payload:
            rid_int = int(obj["id"])
            rid_key = str(rid_int)
            candidates = parsed_map.get(rid_key, [])
            # also try raw as a fallback if model echoed non-canonical id
            if not candidates and str(obj["id"]) != rid_key:
                candidates = parsed_map.get(str(obj["id"]), [])

            kept_strings = self._postfilter_one(
                criterion=crit_map[rid_key],
                already_strings=already_map[rid_key],
                candidates=candidates,
            )
            strings_by_req[rid_int] = kept_strings

            # Offsets explosion (optional)
            if self.compute_offsets:
                blocked_offsets = blocked_by_req.get(rid_int, []) or []
                exploded = _explode_entity_strings_to_offsets(
                    criterion=crit_map[rid_key],
                    entity_strings=kept_strings,
                    blocked_offsets=blocked_offsets,
                    allow_overlaps_between_entities=self.allow_overlaps_between_entities,
                    case_sensitive=self.case_sensitive_offsets,
                )
                offsets_by_req[rid_int] = exploded

                if enable_io:
                    tag = _rid_tag(rid_int)
                    _write_json(mb_dir / f"{tag}_spans.json", exploded, enable=True)
            else:
                if enable_io:
                    tag = _rid_tag(rid_int)
                    _write_json(mb_dir / f"{tag}_spans.json", kept_strings, enable=True)

        return strings_by_req, offsets_by_req

    # ───────────────────────────── forward ─────────────────────────────
    def forward(
        self,
        ctx: Dict[str, Any],
        *,
        blocked_by_req: Optional[Dict[int, List[Tuple[int,int]]]] = None,
    ) -> Dict[str, Any]:  # type: ignore[override]
        reqs = ctx.get("requirements", [])
        if not reqs:
            ctx["free_additional_spans_by_req"] = {}
            ctx["free_additional_spans_offsets_by_req"] = {}
            ctx["free_additional_entity_strings_by_req"] = {}
            return ctx

        engine = self.engine or ctx.get("engine")
        if engine is None:
            raise RuntimeError("No chat engine available for SMTProgrammerFreeEntityExtractor.")

        trial_id = str(ctx.get("trial_id", "trial"))
        side = str(ctx.get("inc_exc", "side"))

        blocked_offsets_by_req = blocked_by_req if blocked_by_req is not None else _gather_blocked_from_ctx(ctx)

        batch_size = max(1, int(ctx.get("free_entity_batch_size", self.batch_size)))
        per_req_strings: Dict[int, List[str]] = {}
        per_req_offsets: Dict[int, List[Dict[str, Any]]] = {}

        batches = [(b_id, start, reqs[start:start+batch_size])
                   for b_id, start in enumerate(range(0, len(reqs), batch_size))]

        max_workers = max(1, int(ctx.get("free_entity_concurrency", self.default_concurrency)))
        if max_workers <= 1 or len(batches) <= 1:
            # serial
            for b_id, start, slice_reqs in batches:
                s_map, o_map = self._run_one_batch_llm(
                    start_idx=start,
                    slice_reqs=slice_reqs,
                    blocked_by_req=blocked_offsets_by_req,
                    ctx=ctx,
                    trial_id=trial_id,
                    side=side,
                    batch_id=b_id,
                )
                per_req_strings.update(s_map); per_req_offsets.update(o_map)
        else:
            # parallel
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        self._run_one_batch_llm,
                        start, slice_reqs, blocked_by_req=blocked_offsets_by_req,
                        ctx=ctx, trial_id=trial_id, side=side, batch_id=b_id
                    ): (b_id, start)
                    for (b_id, start, slice_reqs) in batches
                }
                for fut in as_completed(futs):
                    s_map, o_map = fut.result()
                    per_req_strings.update(s_map); per_req_offsets.update(o_map)

        # normalize keys
        for i in range(len(reqs)):
            per_req_strings.setdefault(i, [])
            per_req_offsets.setdefault(i, [])

        # expose results
        ctx["free_additional_spans_by_req"] = per_req_offsets                  # offsets
        ctx["free_additional_spans_offsets_by_req"] = per_req_offsets          # alias
        ctx["free_additional_entity_strings_by_req"] = per_req_strings         # strings

        if self.verbose:
            total_str = sum(len(v) for v in per_req_strings.values())
            total_off = sum(len(v) for v in per_req_offsets.values())
            print(f"[FreeEntityExtractor:{_safe_id(trial_id)}/{_safe_id(side)}] "
                  f"{total_str} entity strings, {total_off} offset spans "
                  f"across {len(reqs)} requirements in {len(batches)} batch(es), "
                  f"concurrency={max_workers}, batch_size={batch_size}.")

        return ctx
