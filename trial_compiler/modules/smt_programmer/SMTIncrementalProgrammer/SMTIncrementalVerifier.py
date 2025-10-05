from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Tuple, List

import dspy

# ╔════════════════════════════════════════════════════════════════╗
# Fallback logger
# ╚════════════════════════════════════════════════════════════════╝
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [verifier] %(message)s",
        stream=sys.stderr,
    )

    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)


# ────────────────────────────────────────────────────────────────────
# Side/attempt helpers (parity with Translator)
# ────────────────────────────────────────────────────────────────────
def _bucket(side: str) -> str:
    s = (side or "").strip().lower()
    if s in {"inc", "inclusion", "include", "in"}:
        return "inclusion"
    if s in {"exc", "exclusion", "exclude", "ex"}:
        return "exclusion"
    return s or "unknown"


def _attempt_tag(ctx: Dict[str, Any]) -> str:
    """
    Outer (orchestrator) attempt tag, e.g., 'draft01'.
    Falls back to 'draft01' if not present.
    """
    try:
        return str(ctx.get("draft_attempt_tag") or f"draft{int(ctx.get('draft_attempt_index', 1)):02d}")
    except Exception:
        return "draft01"


# ────────────────────────────────────────────────────────────────────
# Declaration normalization helpers (old ↔ new tolerant)
# ────────────────────────────────────────────────────────────────────
def _norm_var_name(d: dict) -> str | None:
    # new compact schema uses "variable_name"; old canonical used "entity_variable_name"
    return d.get("variable_name") or d.get("entity_variable_name")


def _norm_canonical_form(d: dict) -> str | None:
    # try new-style "canonical_form" first, then the Namer keys
    return (
        d.get("canonical_form")
        or d.get(_CANON_KEY_NEW)
        or d.get("entity_canonical_form")
        or d.get("entity_canonical_form_used")
        or d.get("preferred_term")
    )


def _norm_timeframe(d: dict) -> str | None:
    return d.get("timeframe") or d.get("time_frame") or d.get("temporal_window")


def _norm_template(d: dict) -> str | None:
    return d.get("template") or d.get("stem_template") or d.get("pattern")


def _norm_qualifier_predicates(d: dict) -> List[str]:
    """
    Prefer explicit qualifier_predicates if present.
    As a fallback, derive a readable list from qualifier_variables (if available).
    """
    qps = d.get("qualifier_predicates")
    if isinstance(qps, list) and all(isinstance(x, str) for x in qps):
        return qps

    qvars = d.get("qualifier_variables") or []
    out: List[str] = []
    if isinstance(qvars, list):
        for q in qvars:
            if not isinstance(q, dict):
                continue
            # prefer composed name; otherwise keep the name
            cname = (
                q.get("qualifier_variable_composed_name")
                or q.get("composed_name")
                or q.get("qualifier_variable_name")
            )
            if isinstance(cname, str) and cname:
                out.append(cname)
    return out


# Canonical key (align with Namer)
_CANON_KEY_NEW = "entity_canonical_form_from_entity_canonical_forms_block"


# ╔════════════════════════════════════════════════════════════════╗
# Parser helpers (mirrors Translator)
# ╚════════════════════════════════════════════════════════════════╝
_FRAGMENT_RE = re.compile(r"<smtfragment>\s*(.*?)\s*</smtfragment>", re.S | re.I)


def _parse_fragment(text: str) -> str | None:
    """Return content inside <smtfragment>…</smtfragment> or None."""
    m = _FRAGMENT_RE.search(text or "")
    return m.group(1).strip() if m else None


def _strip_headers(block: str) -> List[str]:
    return [ln for ln in (block or "").splitlines() if not ln.strip().startswith(";; ---")]


# ╔════════════════════════════════════════════════════════════════╗
# Program-scan helpers (ported from Translator for #SMT_VARIABLES_BY_FAR#)
# ╚════════════════════════════════════════════════════════════════╝
_DECL_OR_DEF_START_RE = re.compile(
    r"^\s*\((?:declare-fun|declare-const|define-fun|define-const)\b", re.I
)

def _extract_decl_and_def_blocks_from_program(lines: List[str]) -> List[str]:
    """
    从已有 program 行中过滤并截取所有 (declare-*) / (define-*) 的完整 S 表达式块。
    与 Translator 的实现保持一致，用于填充 #SMT_VARIABLES_BY_FAR#。
    """
    out: List[str] = []
    i = 0
    n = len(lines or [])
    while i < n:
        ln = lines[i]
        if not isinstance(ln, str):
            i += 1
            continue
        if _DECL_OR_DEF_START_RE.match(ln):
            depth = 0
            block: List[str] = []
            j = i
            while j < n:
                raw = lines[j]
                if not isinstance(raw, str):
                    j += 1
                    continue
                code_part = raw.split(";;", 1)[0]
                block.append(raw)
                depth += code_part.count("(") - code_part.count(")")
                j += 1
                if depth <= 0:
                    break
            out.append("\n".join(block).strip())
            i = j
        else:
            i += 1
    return [b for b in out if b]


# ╔════════════════════════════════════════════════════════════════╗
# Canonical plumbing helpers
# ╚════════════════════════════════════════════════════════════════╝

def _requirement_json(req_entry: Any) -> str:
    """Mirror Translator's requirement_blob exactly: requirement, source, components[{text,constraint}]."""
    if isinstance(req_entry, dict):
        comps_out: List[Dict[str, str]] = []
        for c in (req_entry.get("components") or []):
            if isinstance(c, dict):
                comps_out.append({
                    "text": c.get("text", ""),
                    "constraint": c.get("constraint", ""),
                })
            else:
                comps_out.append({"text": str(c), "constraint": ""})
        obj = {
            "requirement": req_entry.get("requirement", ""),
            "source": req_entry.get("source", ""),
            "components": comps_out,
        }
    else:
        obj = {"requirement": str(req_entry), "source": "", "components": []}
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _rebuild_allowed_canonical_forms_if_missing(ctx: Dict[str, Any]) -> List[str]:
    """Fallback: derive allowed canon forms from valid_entities_by_req if namer didn’t persist them."""
    idx = ctx.get("current_requirement_index", 0)
    ver_all: dict = ctx.get("valid_entities_by_req") or {}
    ver_for_req = ver_all.get(str(idx)) or {}
    out: List[str] = []
    for e in ver_for_req.values():
        cf = (
            e.get(_CANON_KEY_NEW)
            or e.get("entity_canonical_form_used")
            or e.get("entity_canonical_form")
            or e.get("preferred_term")
            or ""
        )
        if cf:
            out.append(cf)
    # stable-ish ordering by appearance
    return out


def _canonicalizable_payload(ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Build the JSON block injected into #CANONICALIZABLE_VARIABLE_LIST#.

    Structure:
    {
      "declared_or_to_declare": [
        {"variable_name": "...", "canonical_form": "...", "timeframe": "...", "template": "...", "qualifier_predicates":[...]}
      ],
      "reusable": ["name1", "name2", ...],
      "canonical_forms_allowed": ["formA", "formB", ...]
    }
    """
    # New declarations proposed this turn (from Namer / Planner)
    decls_raw: List[dict] = ctx.get("new_canonical_variable_declarations", []) or []
    declared_min: List[dict] = []
    for d in decls_raw:
        if not isinstance(d, dict):
            continue
        vname = _norm_var_name(d)
        if not vname:
            continue
        declared_min.append(
            {
                "variable_name": vname,
                "canonical_form": _norm_canonical_form(d),
                "timeframe": _norm_timeframe(d),
                "template": _norm_template(d),
                "qualifier_predicates": _norm_qualifier_predicates(d),
            }
        )

    # Already-declared names that can be reused
    reusable_raw = ctx.get("reusable_variables") or []
    reusable_set = set()
    if isinstance(reusable_raw, list):
        for rv in reusable_raw:
            if isinstance(rv, str) and rv:
                reusable_set.add(rv)
            elif isinstance(rv, dict):
                nm = rv.get("variable_name") or rv.get("name")
                if isinstance(nm, str) and nm:
                    reusable_set.add(nm)
    reusable = sorted(reusable_set)

    # Canonical forms the translation should try to realize
    allowed = ctx.get("canonical_forms_allowed")
    if not allowed:
        allowed = _rebuild_allowed_canonical_forms_if_missing(ctx)
    if isinstance(allowed, set):
        allowed = sorted(allowed)
    elif not isinstance(allowed, list):
        allowed = list(allowed or [])

    payload_obj = {
        "declared_or_to_declare": declared_min,
        "reusable": reusable,
        "canonical_forms_allowed": allowed,
    }
    return json.dumps(payload_obj, indent=2, ensure_ascii=False), {
        "declared_count": len(declared_min),
        "reusable_count": len(reusable),
        "allowed_count": len(allowed),
    }


# ╔════════════════════════════════════════════════════════════════╗
# Verifier
# ╚════════════════════════════════════════════════════════════════╝
class SMTIncrementalVerifier(dspy.Module):
    """Cross-checks one SMT fragment against the checklist.

    **CHANGELOG**
    2025-07-29 • v2  Added fallback to `new_smt_lines`.
    2025-07-29 • v3  If still empty, parse `<smtfragment>` from `translator_raw_reply`.
    2025-08-24 • v4  Inject canonicalizable variable list; switch to ctx["SMTIncrementalVerifier{Inclusion|Exclusion}_prompt"].
    2025-09-06 • v5  Attempt-scoped logging parity with Translator: logs under
                     {log_root}/{trial}/{inclusion|exclusion}/reqNNN/{draftXX}/
    2025-09-07 • v6  Insert the exact *requirement_blob* (JSON) just like Translator, instead of plain text.
    2025-10-15 • v7  Switch #PARTIAL_PROGRAM# → #SMT_VARIABLES_BY_FAR# and fill with declare/define blocks only.
    """

    # Find a JSON object anywhere in the reply
    _JSON_RE = re.compile(r"\{.*\}", re.S)

    def __init__(self, engine, *, log_dir: str | None = "./verifier_logs"):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./verifier_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    # ── IO ---------------------------------------------------------
    def _write_logs(
        self,
        out_dir: str,
        prompt: str,
        raw: str,
        report: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Write verifier artifacts under the attempt-scoped req folder."""
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as fh:
            fh.write(prompt)
        with open(os.path.join(out_dir, "raw.txt"), "w", encoding="utf-8") as fh:
            fh.write(raw)
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
            fh.write(report)
        if extra:
            try:
                with open(os.path.join(out_dir, "canon_payload.json"), "w", encoding="utf-8") as fh:
                    json.dump(extra, fh, indent=2, ensure_ascii=False)
            except Exception:
                pass

    # ── reply parsing ---------------------------------------------
    def _parse_reply(self, raw: str) -> Tuple[bool, str]:
        m = self._JSON_RE.search(raw or "")
        if not m:
            return False, f"[parse-error] no JSON object\nLLM reply:\n{raw}"

        try:
            obj = json.loads(m.group(0))
        except Exception as e:  # pylint: disable=broad-except
            return False, f"[parse-error] {e}\nLLM reply:\n{raw}"

        # Required keys (match the prompt spec exactly, typos included)
        required = {
            "MEANING_FAITHFULNESS",
            "ENTITY_AND_QUALIFIER_COMPLETENESSS",  # note triple 'S' as specified
            "NO_VARIABLE_DUPLICATION",
            "POLARITY_CORRECTNESS",
            "VARIABLE_NAME_VERBATIMNESS",
            "USE_CANONICALIZABLE_LIST_AS_MUCH_AS_POSSIBLE",
            "QUALIFIER_ENTITY_RELATIONSHIPS_CAPTURED",
            "OVEARLL_OK",  # intentional spelling as per spec
            "explanations",
        }
        missing = required - set(obj.keys())
        if missing:
            return False, f"[key-error] missing {sorted(missing)}\nLLM reply:\n{raw}"

        try:
            ok = bool(obj["OVEARLL_OK"])
        except Exception:
            ok = False

        return ok, json.dumps(obj, indent=2, ensure_ascii=False)

    # ── main -------------------------------------------------------
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        idx = context["current_requirement_index"]
        req_entry = context["requirements"][idx]

        # Build the exact requirement_blob JSON (parity with Translator)
        requirement_blob = _requirement_json(req_entry)

        # ------------------------------------------------------------
        # 1) Primary source: committed slice via req_blocks
        # ------------------------------------------------------------
        smt_fragment = ""
        if idx in context.get("req_blocks", {}):
            start, end = context["req_blocks"][idx]
            smt_fragment = "\n".join(context["smt_program_lines"][start:end]).strip()

        # ------------------------------------------------------------
        # 2) Fallback: fresh slice in new_smt_lines
        # ------------------------------------------------------------
        if not smt_fragment:
            smt_fragment = "\n".join(context.get("new_smt_lines", [])).strip()

        # ------------------------------------------------------------
        # 3) Ultimate fallback: raw LLM reply from translator (if stored)
        # ------------------------------------------------------------
        if not smt_fragment and (raw_trans := context.get("translator_raw_reply")):
            frag = _parse_fragment(raw_trans)
            if frag:
                smt_fragment = "\n".join(_strip_headers(frag))

        # ---- variables/declarations so far (declare/define blocks only) ----
        _program_lines = context.get("smt_program_lines", [])
        vars_blocks = _extract_decl_and_def_blocks_from_program(_program_lines)
        smt_vars_so_far = "\n".join(vars_blocks)

        # ---- canonicalizable variable list payload -----------------
        canonicalizable_json, canon_counts = _canonicalizable_payload(context)

        # ---- choose prompt template (ctx-first, side-aware) --------
        side = context.get("inc_exc", "unknown")
        if side == "inclusion":
            tmpl = context.get("SMTIncrementalVerifierInclusion_prompt")
        else:
            tmpl = context.get("SMTIncrementalVerifierExclusion_prompt")

        # Compose prompt — IMPORTANT: insert the *exact* requirement_blob JSON
        prompt = (
            tmpl.replace("#REQUIREMENT#", requirement_blob)
            .replace("#SMT_VARIABLES_BY_FAR#", smt_vars_so_far)   # ← 新占位符
            .replace("#CANDIDATE_SMT#", smt_fragment)
            .replace("#CANONICALIZABLE_VARIABLE_LIST#", canonicalizable_json)
        )

        # (Expose an inner attempt index if you later add verifier retries)
        context["verifier_engine_attempt_index"] = 1

        # ---- call LLM ---------------------------------------------
        raw = self.engine(prompt)[0].strip()
        ok, report = self._parse_reply(raw)

        # ---------- attempt-scoped logging (parity with Translator) ----------
        trial = context.get("trial_id", "unknown_trial")
        stage_dir = os.path.join(
            self.log_dir,
            str(trial),
            _bucket(side),
            f"req{idx:03d}",
            _attempt_tag(context),  # e.g., 'draft01'
        )
        self._write_logs(
            stage_dir,
            prompt,
            raw,
            report,
            extra={
                "counts": canon_counts,
                "canonicalizable_variable_list": json.loads(canonicalizable_json),
                "timestamp": datetime.now().isoformat(),
            },
        )

        # ---------- update context ----
        context["verifier_ok"] = ok
        context["verifier_report"] = report

        tag = _attempt_tag(context)
        _log("verifier ✓" if ok else "verifier ✗", idx, f"(outer {tag})")
        return context


__all__ = ["SMTIncrementalVerifier"]
