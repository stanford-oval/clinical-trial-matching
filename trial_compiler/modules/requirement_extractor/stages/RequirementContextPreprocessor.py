# requirement_context_preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────
# Splits raw eligibility text into per-cohort chunks before the extractor.
# Exposes both a normalized list of enrollment_cohorts and per-cohort cloned contexts.

from __future__ import annotations

import json, pathlib, re, copy
from typing import Dict, List, Any, Optional, Tuple
import dspy
import string

def _alpha_suffix(n: int) -> str:
    """0 -> 'a', 1 -> 'b', ... 25 -> 'z', 26 -> 'aa', 27 -> 'ab', ..."""
    letters = string.ascii_lowercase
    s = ""
    n0 = n
    while True:
        s = letters[n % 26] + s
        n //= 26
        if n == 0:
            break
        n -= 1  # Excel-like base-26 progression
    return s

def _with_cohort_trial_id(base_tid: str, k: int, total: int) -> str:
    """Return base_tid or base_tid + letter suffix when total>1."""
    if total <= 1:
        return base_tid
    return f"{base_tid}{_alpha_suffix(k)}"


# Optional: reuse the same header stripping idea to keep outputs clean
_HEADER_PATTERNS = (
    re.compile(r'^\s*inclusion criteria', re.I),
    re.compile(r'^\s*exclusion criteria', re.I),
    re.compile(r'^\s*criteria[:\s]',      re.I),
)

def _strip_headers(txt: str) -> str:
    lines = (txt or "").splitlines()
    if lines and any(pat.match(lines[0]) for pat in _HEADER_PATTERNS):
        lines = lines[1:]
    return "\n".join(lines).strip()

def _squash_space(s: Optional[str]) -> str:
    return re.sub(r'\s+', ' ', (s or '')).strip()

def _safe_json_find(s: str) -> Optional[Any]:
    """
    Try to parse the first JSON object/array inside the string `s`.
    Accepts either a bare JSON or text + JSON blob.
    """
    s = s.strip()
    if not s:
        return None
    # Quick path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: find first {...} or [...]
    first_obj = s.find('{')
    first_arr = s.find('[')
    start = min([p for p in [first_obj, first_arr] if p != -1], default=-1)
    if start == -1:
        return None
    for end in range(len(s), start, -1):
        frag = s[start:end]
        try:
            return json.loads(frag)
        except Exception:
            continue
    return None


def _normalize_cohorts_from_spec(
    parsed: Any,
    *,
    inc_exc: str,
    default_req_text: str,
    default_ctx_text: str
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse the REQUIRED output spec (new schema):

    {
      "shared_context": "<shared_context>",
      "enrollment_cohorts": [
        {
          "cohort_name": "<label>",
          "inclusion_criteria": "<inclusion>",
          "exclusion_criteria": "<exclusion>",
          "context": "<cleaned context for this cohort>"
        }, ...
      ]
    }

    Backward-compat:
      - If "enrollment_cohorts" absent, accept legacy "substudies" with the same fields
        (optionally "substudy_name" instead of "cohort_name").

    Returns:
      (cohorts, shared_context)

    Each cohort in the returned list has:
      {
        "id": "C1",                  # synthesized if absent
        "label": "<best available name>",
        "inclusion_criteria": "...", # passthrough
        "exclusion_criteria": "...", # passthrough
        "context": "...",            # per-cohort context (from spec)
        "requirement_text": "...",   # SELECTED by inc_exc (inclusion/exclusion)
        "contextual_text": "..."     # MERGED: shared_context + '\n\n' + cohort.context (when present)
      }
    """
    def _single_default():
        return ([{
            "id": "default",
            "label": "Overall",
            "inclusion_criteria": default_req_text if inc_exc == "inclusion" else "",
            "exclusion_criteria": default_req_text if inc_exc == "exclusion" else "",
            "context": default_ctx_text,
            "requirement_text": default_req_text,
            "contextual_text": default_ctx_text,
        }], "")

    if not isinstance(parsed, dict):
        return _single_default()

    shared = str(parsed.get("shared_context") or "").strip()

    # Prefer new key; fall back to legacy key if needed
    items = parsed.get("enrollment_cohorts")
    if not isinstance(items, list) or not items:
        items = parsed.get("substudies")

    if not isinstance(items, list) or not items:
        # No cohorts array → single default using the existing text
        return _single_default()

    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        # Name resolution with backward-compat
        name = it.get("cohort_name")
        if not name:
            # legacy alias
            name = it.get("set_name") or it.get("substudy_name") or it.get("label") or f"Cohort {i+1}"
        name = str(name).strip()

        inc = str(it.get("inclusion_criteria") or "").strip()
        exc = str(it.get("exclusion_criteria") or "").strip()
        ctx = str(it.get("context") or "").strip()

        # Choose requirement text by side
        req_txt = inc if inc_exc == "inclusion" else exc
        if not req_txt:
            # Fallback to default side text if model left it empty
            req_txt = default_req_text

        merged_ctx = (shared + ("\n\n" + ctx if ctx else "")).strip() or default_ctx_text

        out.append({
            "id": f"C{i+1}",
            "label": name or f"Cohort {i+1}",
            "inclusion_criteria": inc,
            "exclusion_criteria": exc,
            "context": ctx,
            "requirement_text": _strip_headers(req_txt),
            "contextual_text": merged_ctx,
        })

    return (out or _single_default()[0], shared)


def _clone_context_for_cohort(base_ctx: Dict[str, Any], cohort: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a per-cohort context dict so downstream code can run the pipeline per item.
    Only override analog fields (requirement_text, contextual_text); preserve everything else.
    """
    ctx = copy.deepcopy(base_ctx)
    ctx["requirement_text"] = cohort.get("requirement_text", "")
    ctx["contextual_text"]  = cohort.get("contextual_text", base_ctx.get("contextual_text", ""))

    # New canonical keys
    ctx["cohort_id"]        = cohort.get("id")
    ctx["cohort_label"]     = cohort.get("label")

    # Legacy aliases for backward compatibility
    ctx["substudy_id"]      = cohort.get("id")
    ctx["substudy_label"]   = cohort.get("label")

    # Helpful flag to avoid re-running the preprocessor on exploded contexts
    ctx["__preprocessed__"] = True
    return ctx


class RequirementContextPreprocessor(dspy.Module):
    """
    Pre-extractor preprocessor that splits multi-cohort / parallel-arm criteria
    into separate units and annotates the context accordingly.

    Usage:
        pre = RequirementContextPreprocessor(engine, log_dir="mbench/req_mbench/preproc_logs")
        context = pre.forward(context, use_full_context=True)

    Expects in `context`:
        - inc_exc: "inclusion" | "exclusion"
        - contextual_text: str
        - requirement_text: str
        - RequirementContextPreprocessor_prompt: str  (contains placeholders)
            Placeholders supported:
              #INC_EXC#, #CONTEXTUAL_TEXT#, #REQUIREMENT_TEXT#
    Produces in `context`:
        - enrollment_cohorts: List[Dict] with keys: id, label, requirement_text, contextual_text, ...
        - has_cohorts: bool
        - num_cohorts: int
        - __cohort_contexts__: List[Dict] – cloned contexts per cohort
        - preprocessor_raw_output: str (raw model text)
        - preprocessor_normalized: Dict (for debugging/export)
      Backward-compatibility fields (populated as mirrors):
        - substudies, has_substudies, num_substudies, __substudy_contexts__
    """

    def __init__(
        self,
        engine,
        *,
        # write preprocessor logs under mbench like other artifacts
        log_dir: str | pathlib.Path | None = "mbench/req_mbench/preproc_logs",
        explode: bool = False,   # if True, also sets requirement_text/contextual_text to the FIRST cohort
        debug: bool = True,
    ):
        super().__init__()
        self.engine = engine
        self.debug = debug
        self.explode = explode
        self.log_dir = pathlib.Path(log_dir).expanduser() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        inc_exc = str(ctx.get("inc_exc", "")).strip()
        ctx_txt = str(ctx.get("contextual_text", "") or "")
        req_txt = str(ctx.get("requirement_text", "") or "")
        prompt_t = ctx.get("RequirementContextPreprocessor_prompt", "")
        prompt = (
            prompt_t
            .replace("#INC_EXC#", inc_exc)
            .replace("#CONTEXTUAL_TEXT#", ctx_txt)
            .replace("#REQUIREMENT_TEXT#", req_txt)
        )
        return prompt

    def forward(self, context: Dict, *, use_full_context: bool) -> Dict:
        if context.get("__preprocessed__"):
            return context

        inc_exc = str(context.get("inc_exc", "")).strip()  # "inclusion" | "exclusion"
        base_tid = str(context.get("trial_id", "unknown"))
        ctx_txt  = str(context.get("contextual_text", "") or "")
        req_txt  = str(context.get("requirement_text", "") or "")

        # 1) Prompt & LLM call
        prompt = self._build_prompt(context)
        print(f"prompt is {prompt}")
        raw_output = self.engine(prompt)[0]
        print(f"raw_output is {raw_output}")

        # 2) Parse & normalize according to spec (new key + legacy fallback)
        parsed = _safe_json_find(raw_output)
        if self.debug:
            print("[debug] parsed:", parsed)
        cohorts, shared_context = _normalize_cohorts_from_spec(
            parsed,
            inc_exc=inc_exc,
            default_req_text=_strip_headers(req_txt),
            default_ctx_text=ctx_txt.strip()
        )

        # 3) Assign suffixed effective trial_ids and clone contexts
        total = len(cohorts)
        cohort_contexts = []
        for i, c in enumerate(cohorts):
            eff_tid = _with_cohort_trial_id(base_tid, i, total)
            c["trial_id_effective"] = eff_tid

            subctx = _clone_context_for_cohort(context, c)
            subctx["trial_id_parent"] = base_tid
            subctx["trial_id"]        = eff_tid        # <- downstream logs will use suffixed IDs
            subctx["cohort_index"]    = i
            subctx["cohort_count"]    = total
            # Preserve both raw per-cohort context and merged context
            subctx["cohort_context_raw"] = c.get("context", "")
            subctx["shared_context"]     = shared_context

            # Legacy mirrors for downstream code not yet migrated
            subctx["substudy_index"]  = i
            subctx["substudy_count"]  = total
            subctx["substudy_context_raw"] = c.get("context", "")

            cohort_contexts.append(subctx)

        # 4) Logging (note: keys updated; keep legacy mirror)
        if self.log_dir:
            try:
                side = inc_exc or "unknown"
                (self.log_dir / f"{base_tid}_{side}.pre.raw.txt").write_text(str(raw_output), encoding="utf-8")
                norm_payload = {
                    "trial_id": base_tid,
                    "side": side,
                    "input": {
                        "inc_exc": inc_exc,
                        "contextual_text": ctx_txt,
                        "requirement_text": req_txt,
                    },
                    "shared_context": shared_context,
                    "enrollment_cohorts": cohorts,
                    "effective_trial_ids": [c["trial_id_effective"] for c in cohorts],
                }
                (self.log_dir / f"{base_tid}_{side}.pre.normalized.json").write_text(
                    json.dumps(norm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                print(f"[Preprocessor] log write failed: {exc}")

        # 5) Annotate parent context (new canonical keys + legacy mirrors)
        context["shared_context"]            = shared_context
        context["enrollment_cohorts"]        = cohorts
        context["has_cohorts"]               = len(cohorts) > 1
        context["num_cohorts"]               = len(cohorts)
        context["__cohort_contexts__"]       = cohort_contexts
        context["effective_trial_ids"]       = [c["trial_id_effective"] for c in cohorts]
        context["parent_trial_id"]           = base_tid
        context["preprocessor_raw_output"]   = str(raw_output)
        context["preprocessor_normalized"]   = {"shared_context": shared_context, "enrollment_cohorts": cohorts}
        context["__preprocessed__"]          = True

        # Legacy mirrors
        context["substudies"]              = cohorts
        context["has_substudies"]          = len(cohorts) > 1
        context["num_substudies"]          = len(cohorts)
        context["__substudy_contexts__"]   = cohort_contexts

        # Optional: explode to first cohort for single-pass flows
        if self.explode and cohorts:
            first = cohorts[0]
            context["requirement_text"] = first.get("requirement_text", context.get("requirement_text", ""))
            context["contextual_text"]  = first.get("contextual_text",  context.get("contextual_text", ""))

        return context
