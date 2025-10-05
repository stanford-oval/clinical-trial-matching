"""
 s m t _ i n c r e m e n t a l _ t r a n s l a t o r . p y
────────────────────────────────────────────────────────
Incrementally translate ONE atomic requirement into SMT-LIB and
append the fragment (with required :named tags) to context["new_smt_lines"].

Robustness adds:
- Multiple retry strategies (progressively stricter prompts)
- Exponential backoff with jitter between attempts
- Clear failure classification and logging
- Inclusion/Exclusion-aware :named-tag validation
- Tag validator is side-effect–free until a whole fragment passes
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import random
from typing import List, Dict, Any, Tuple, Optional

import dspy  # type: ignore

# ────────────────────────────────────────────────────────────────────
# Fallback logger
# ────────────────────────────────────────────────────────────────────
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [translator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    def _log(stage: str, idx: int, msg: str = "") -> None:  # pylint: disable=unused-argument
        logging.info("%s %s", stage, msg)


# ╔════════════════════════════════════════════════════════════════╗
# Config (env-overridable)
# ╚════════════════════════════════════════════════════════════════╝
MAX_ATTEMPTS = int(os.getenv("SMT_TRANSLATOR_MAX_ATTEMPTS", "5"))
BACKOFF_MS   = int(os.getenv("SMT_TRANSLATOR_BACKOFF_MS", "350"))   # base backoff (ms)
JITTER_MS    = int(os.getenv("SMT_TRANSLATOR_JITTER_MS", "150"))    # additional jitter (ms)


# ╔════════════════════════════════════════════════════════════════╗
# Regex helpers
# ╚════════════════════════════════════════════════════════════════╝
_FRAGMENT_RE = re.compile(r"<smtfragment>\s*(.*?)\s*</smtfragment>", re.IGNORECASE | re.DOTALL)
_FENCE_RE    = re.compile(r"```(?:smt2?|smt|lisp|cl)?\s*([\s\S]*?)```", re.I)
# Capture any token after :named; validation happens in _validate_tags
_TAG_RE      = re.compile(r":named\s+([A-Za-z0-9_]+)")

# ╔════════════════════════════════════════════════════════════════╗
# Allowed tag tokens (Inclusion & Exclusion)
# ╚════════════════════════════════════════════════════════════════╝
ALLOWED_CONSTRAINTS_INCLUSION = {
    "PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE",
    "OTHER_REQUIREMENTS",
    "NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION",
}
ALLOWED_CONSTRAINTS_EXCLUSION = {
    "OTHER_REQUIREMENTS",
    "OTHER_REQUIREMENTS",  # spec spelling
    "CAN_ALWAYS_GO_FROM_SATISFIED_TO_NOTSATISFIED",
}
# Robustness aliases for the common "POSSIBLE" vs "POSSBILE" typo
ALIASES_INCLUSION = {
    "POSSIBLE_TO_GO_FROM_NOTSATISFIED_TO_SATISFIED_CONDITIONAL_ON_PATIENT_STATE":
    "POSSBILE_TO_GO_FROM_NOTSATISFIED_TO_SATISFIED_CONDITIONAL_ON_PATIENT_STATE"
}
ALIASES_EXCLUSION = {
    "POSSIBLE_TO_GO_FROM_SATISFIED_TO_NOTSATISFIED_CONDITIONAL_ON_PATIENT_STATE":
    "OTHER_REQUIREMENTS"
}


def _listify_reusables(lst) -> List[Dict[str, str]]:
    """
    Normalize reusables to [{"variable_name": "..."}] and dedupe.
    Accepts list[str] or list[dict].
    """
    names = set()
    if isinstance(lst, list):
        for it in lst:
            if isinstance(it, str):
                s = it.strip()
                if s: names.add(s)
            elif isinstance(it, dict):
                nm = it.get("variable_name") or it.get("name")
                if isinstance(nm, str) and nm.strip():
                    names.add(nm.strip())
    return [{"variable_name": n} for n in sorted(names)]


def _filter_into_reusables(
    merged_newdecls: List[Dict[str, Any]],
    declared_ids: set[str],
    existing_reusables: List[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[str]]:
    """
    Partition merged_newdecls into:
      - newdecls_for_prompt: items NOT already declared and NOT already reusable
      - reusables_for_prompt: existing reusables UNION names moved from merged_newdecls
      - moved_names: names that were moved from merged_newdecls → reusables
    """
    # Current reusable names
    reusable_names = _names_from_mixed_list(existing_reusables)

    newdecls: List[Dict[str, Any]] = []
    moved: List[str] = []

    for obj in merged_newdecls or []:
        if not isinstance(obj, dict):
            continue
        name = obj.get("variable_name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        # If already declared in partial SMT or already listed as reusable → move
        if (name in declared_ids) or (name in reusable_names):
            if name not in reusable_names:
                moved.append(name)
            continue  # do not include in #NEW_VARIABLE_DECLARATIONS#
        newdecls.append(obj)

    # Build updated reusable list
    updated_reusable_names = reusable_names | set(moved)
    reusables_for_prompt = [{"variable_name": n} for n in sorted(updated_reusable_names)]

    # Dedup kept new decls by variable_name (keep last for stability)
    newdecls = _unique_by_var_name(newdecls)

    return newdecls, reusables_for_prompt, moved


# ────────────────────────────────────────────────────────────────────
# Program-scan helpers: extract decl/def blocks and declared symbols
# ────────────────────────────────────────────────────────────────────
_DECL_OR_DEF_START_RE = re.compile(
    r"^\s*\((?:declare-fun|declare-const|define-fun|define-const)\b", re.I
)
_SYMBOL_FROM_DECL_OR_DEF_RE = re.compile(
    r"^\s*\((?:declare-fun|declare-const|define-fun|define-const)\s+([^\s()]+)", re.I
)

def _extract_decl_and_def_blocks_from_program(lines: List[str]) -> List[str]:
    """
    从已有 program 行中过滤并截取所有 (declare-*) / (define-*) 的完整 S 表达式块。
    支持多行 define-fun；会保留块内的原始缩进与可能的行内注释。
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
                # 计数时忽略行内 ';;' 之后的注释
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

def _declared_symbols_from_program(lines: List[str]) -> set[str]:
    """
    提取所有已声明/定义的符号名，用于去重（declare-fun/const, define-fun/const）。
    仅需解析起始行即可拿到符号。
    """
    syms: set[str] = set()
    for ln in (lines or []):
        if not isinstance(ln, str):
            continue
        m = _SYMBOL_FROM_DECL_OR_DEF_RE.match(ln)
        if m:
            syms.add(m.group(1))
    return syms

# ────────────────────────────────────────────────────────────────────
# Persistence helpers for canon + demographics declarations
# ────────────────────────────────────────────────────────────────────

def _unique_by_var_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate a list of normalized decls by variable_name, keep last occurrence."""
    seen: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        vn = it.get("variable_name")
        if isinstance(vn, str) and vn:
            seen[vn] = it
    return list(seen.values())

def _names_from_mixed_list(lst) -> set[str]:
    """
    Accepts a list of strings or dicts. Extracts names from:
      - variable_name
      - name (legacy tolerance)
    """
    out: set[str] = set()
    if not isinstance(lst, list):
        return out
    for it in lst:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.add(s)
        elif isinstance(it, dict):
            nm = it.get("variable_name") or it.get("name")
            if isinstance(nm, str):
                s = nm.strip()
                if s:
                    out.add(s)
    return out

def _persist_canon_and_demographics(context: Dict[str, Any], base_dir: str | None = None) -> None:
    """
    Persist normalized canonical and demographics declarations in the context:
      - context["persisted_canonical_variables"]        : List[decl]
      - context["persisted_demographics_variables"]     : List[decl]
    Optionally write *_persisted_{canon,demographics}.json under base_dir.
    """
    # Source blocks from planner/namer (may be absent)
    canon_raw = context.get("new_canonical_variable_declarations") or []
    demo_raw  = context.get("new_age_sex_pregnancystatus_declarations") or []

    canon_norm = [_normalize_decl(x) for x in canon_raw if isinstance(x, dict)]
    demo_norm  = [_normalize_decl(x) for x in demo_raw  if isinstance(x, dict)]

    canon_norm = [x for x in canon_norm if x.get("variable_name")]
    demo_norm  = [x for x in demo_norm  if x.get("variable_name")]

    # Merge with persisted (dedup by variable_name)
    prev_canon = context.get("persisted_canonical_variables") or []
    prev_demo  = context.get("persisted_demographics_variables") or []

    if not isinstance(prev_canon, list): prev_canon = []
    if not isinstance(prev_demo, list):  prev_demo  = []

    merged_canon = _unique_by_var_name([*prev_canon, *canon_norm])
    merged_demo  = _unique_by_var_name([*prev_demo,  *demo_norm])

    context["persisted_canonical_variables"]    = merged_canon
    context["persisted_demographics_variables"] = merged_demo

    # Optional file logs for debugging/inspection
    if base_dir:
        try:
            with open(os.path.join(base_dir, "_persisted_canonical_variables.json"), "w", encoding="utf-8") as fh:
                json.dump(merged_canon, fh, indent=2, ensure_ascii=False)
            with open(os.path.join(base_dir, "_persisted_demographics_variables.json"), "w", encoding="utf-8") as fh:
                json.dump(merged_demo, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            _log("translator ⚠", context.get("current_requirement_index", -1), f"failed to write persisted lists: {e}")

# ────────────────────────────────────────────────────────────────────
# Text utilities
# ────────────────────────────────────────────────────────────────────

def _sanitize_fragment_text(text: str) -> str:
    """Drop pretty headers, markdown fences, and loose <smtfragment> tags."""
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith(";; ---"):
            continue
        if s.startswith("```") or s.endswith("```"):
            continue
        if s.lower() in {"<smtfragment>", "</smtfragment>"}:
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _parse_fragment(text: str) -> tuple[str | None, str]:
    """
    Try: XML tags → code fences → heuristic raw SMT.
    Returns (fragment_text_or_None, how).
    """
    if not isinstance(text, str):
        return None, "notstr"

    m = _FRAGMENT_RE.search(text)
    if m:
        return _sanitize_fragment_text(m.group(1)), "xmltag"

    m = _FENCE_RE.search(text)
    if m:
        return _sanitize_fragment_text(m.group(1)), "fence"

    for needle in ("(declare-const", "(declare-fun", "(assert"):
        pos = text.find(needle)
        if pos != -1:
            cand = text[pos:]
            return _sanitize_fragment_text(cand), "heuristic"

    return None, "none"


def _attempt_tag(ctx: dict) -> str:
    """Outer (verifier) attempt tag, e.g., 'draft01'."""
    try:
        return str(ctx.get("draft_attempt_tag") or f"draft{int(ctx.get('draft_attempt_index', 1)):02d}")
    except Exception:
        return "draft01"


def _strip_headers(block: str) -> List[str]:
    lines = []
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith(";; ---"):
            continue
        if s.lower() in {"<smtfragment>", "</smtfragment>"}:
            continue
        if s.startswith("```") or s.endswith("```"):
            continue
        lines.append(ln)
    return lines


# ────────────────────────────────────────────────────────────────────
# Tag Validation (side‑effect free until success)
# ────────────────────────────────────────────────────────────────────

def _validate_tags(fragment: str, req_idx: int, registry: set[str], side: str | None = None) -> None:
    """
    Ensure every :named tag conforms to the required scheme for this requirement index,
    with no duplicates. This function does **not** mutate `registry` unless validation
    succeeds for the whole fragment.

      Constraint (component) assertions (ACCEPT EITHER FORM):
        REQ{idx}_COMPONENT{comp_idx}_{CONSTRAINT}
        R{idx}_A{comp_idx}_{CONSTRAINT}

        INCLUSION {CONSTRAINT} ∈ {
          NOT_POSSIBLE_TO_GO_FROM_NOTSATISFIED_TO_SATISFIED,
          POSSBILE_TO_GO_FROM_NOTSATISFIED_TO_SATISFIED_CONDITIONAL_ON_PATIENT_STATE,
          NOT_REQUIREMENT_OR_ALWAYS_SATISFIABLE_WITH_ACTION
        }
        EXCLUSION {CONSTRAINT} ∈ {
          OTHER_REQUIREMENTS,
          OTHER_REQUIREMENTS,
          CAN_ALWAYS_GO_FROM_SATISFIED_TO_NOTSATISFIED
        }

      Auxiliary assertions (ACCEPT EITHER FORM):
        REQ{idx}_AUXILIARY{def_idx}
        R{idx}_D{def_idx}

    Note: also accepts the correctly spelled POSSIBLE_* as an alias to the spec’s POSSBILE_*.
    """
    matches = list(_TAG_RE.finditer(fragment))
    if not matches:
        raise ValueError("SMT fragment contains no :named tags")

    # Accept both the long REQ* and short R* tag schemes
    def_pat_a  = re.compile(rf"^REQ{req_idx}_AUXILIARY(\d+)$")
    def_pat_b  = re.compile(rf"^R{req_idx}_D(\d+)$")
    comp_pat_a = re.compile(rf"^REQ{req_idx}_COMPONENT(\d+)_([A-Z_]+)$")
    comp_pat_b = re.compile(rf"^R{req_idx}_A(\d+)_([A-Z_]+)$")

    if side == "inclusion":
        allowed_constraints = set(ALLOWED_CONSTRAINTS_INCLUSION)
        aliases = dict(ALIASES_INCLUSION)
    elif side == "exclusion":
        allowed_constraints = set(ALLOWED_CONSTRAINTS_EXCLUSION)
        aliases = dict(ALIASES_EXCLUSION)
    else:
        # Fallback: accept either (useful if caller didn't pass side)
        allowed_constraints = set(ALLOWED_CONSTRAINTS_INCLUSION) | set(ALLOWED_CONSTRAINTS_EXCLUSION)
        aliases = dict(ALIASES_INCLUSION)
        aliases.update(ALIASES_EXCLUSION)

    local_seen: set[str] = set()
    new_tags:   list[str] = []

    # First pass: validate structure & within-fragment uniqueness
    for m in matches:
        tag = m.group(1).strip()

        # within-fragment dup
        if tag in local_seen:
            raise ValueError(f"duplicate :named tag '{tag}'")
        local_seen.add(tag)

        # Auxiliary?
        if def_pat_a.match(tag) or def_pat_b.match(tag):
            new_tags.append(tag)
            continue

        # Component?
        m_comp = comp_pat_a.match(tag) or comp_pat_b.match(tag)
        if m_comp:
            constraint = m_comp.group(2)
            constraint = aliases.get(constraint, constraint)
            if constraint not in allowed_constraints:
                hint = "inclusion" if "NOTSATISFIED_TO_SATISFIED" in constraint else "exclusion"
                raise ValueError(
                    f"unknown constraint token '{constraint}' in :named tag '{tag}' "
                    f"(expected {hint} constraint keys for requirement {req_idx})"
                )
            new_tags.append(tag)
            continue

        raise ValueError(f":named tag '{tag}' does not match expected patterns for requirement {req_idx}")

    # Second pass: cross-fragment duplication against global registry
    for t in new_tags:
        if t in registry:
            raise ValueError(f"duplicate :named tag '{t}'")

    # success → now mutate
    registry.update(new_tags)


# ╔════════════════════════════════════════════════════════════════╗
# Retry strategy helpers
# ╚════════════════════════════════════════════════════════════════╝
RETRY_HINTS_GENERIC = [
    "",
    # Attempt 2: enforce container + tagging
    "\n\nReturn ONLY one block enclosed by <smtfragment>…</smtfragment>. "
    "Do not include markdown fences or prose. "
    "Every assertion MUST include a :named tag using the required scheme. "
    "Reuse provided variables; declare any missing ones with on-line JSON annotations.",
    # Attempt 3: restate inclusion/exclusion rules explicitly
    "\n\nIMPORTANT: If inc_exc=='exclusion', wrap each top-level exclusion condition in (not …). "
    "SAT means ELIGIBLE. Tag constraints exactly as REQ{REQ_IDX}_COMPONENT{n}_{CONSTRAINT} "
    "or R{REQ_IDX}_A{n}_{CONSTRAINT}. "
    "Use REQ{REQ_IDX}_AUXILIARY{n} or R{REQ_IDX}_D{n} for Auxiliaries.",
    # Attempt 4: hard constraints list
    "\n\nAllowed constraint keys (inclusion): "
    "NOT_POSSIBLE_TO_GO_FROM_NOTSATISFIED_TO_SATISFIED | "
    "POSSBILE_TO_GO_FROM_NOTSATISFIED_TO_SATISFIED_CONDITIONAL_ON_PATIENT_STATE | "
    "NOT_REQUIREMENT_OR_ALWAYS_SATISFIABLE_WITH_ACTION. "
    "Allowed constraint keys (exclusion): "
    "OTHER_REQUIREMENTS | "
    "OTHER_REQUIREMENTS | "
    "CAN_ALWAYS_GO_FROM_SATISFIED_TO_NOTSATISFIED.",
    # Attempt 5+: formatting hammer
    "\n\nOutput ONLY SMT-LIB lines inside <smtfragment>…</smtfragment>. "
    "No commentary. No backticks. Ensure at least one :named tag present and correctly prefixed.",
]

FAILURE_SPECIFIC_HINTS = {
    "parse_fail":
        "\n\nYour reply did not contain a parsable SMT fragment. "
        "Wrap the entire answer in <smtfragment>…</smtfragment> and include (declare-const …) and (assert …) lines only.",
    "no_named_tags":
        "\n\nYour fragment is missing :named tags. "
        "Add :named tags to EVERY assertion using REQ{REQ_IDX}_COMPONENT{n}_{CONSTRAINT} or REQ{REQ_IDX}_AUXILIARY{n} "
        "or R{REQ_IDX}_A{n}_{CONSTRAINT} / R{REQ_IDX}_D{n}.",
    "tag_mismatch":
        "\n\nOne or more :named tags do not match the required pattern. "
        "Fix tags to REQ{REQ_IDX}_COMPONENT{n}_{CONSTRAINT} or REQ{REQ_IDX}_AUXILIARY{n} "
        "or R{REQ_IDX}_A{n}_{CONSTRAINT} / R{REQ_IDX}_D{n} for requirement index {REQ_IDX}.",
    "unknown_constraint":
        "\n\nYou used an invalid constraint token. "
        "Use only the allowed tokens listed in the instructions. Spelling MUST match (POSSBILE_… is intentional).",
    "duplicate_tag":
        "\n\nYou emitted duplicate :named tags. Make each tag unique by incrementing the index numbers correctly.",
}


def _compose_retry_prompt(base_prompt: str, req_idx: int, side: str, attempt: int, failure_reason: str | None) -> str:
    """Append increasingly strict hints and failure-specific nudges."""
    parts = [base_prompt]
    if failure_reason and failure_reason in FAILURE_SPECIFIC_HINTS:
        parts.append(FAILURE_SPECIFIC_HINTS[failure_reason].replace("{REQ_IDX}", str(req_idx)))
    hint_idx = min(attempt - 1, len(RETRY_HINTS_GENERIC) - 1)
    parts.append(RETRY_HINTS_GENERIC[hint_idx].replace("{REQ_IDX}", str(req_idx)))
    return "".join(parts)


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with jitter."""
    base = BACKOFF_MS * (2 ** (attempt - 1))
    jitter = random.randint(0, JITTER_MS)
    time.sleep((base + jitter) / 1000.0)


def _classify_failure(exc: Exception | None, parse_how: str | None = None, validate_msg: str | None = None) -> str:
    if exc is not None:
        return "engine_error"
    if parse_how in {"none", "notstr"}:
        return "parse_fail"
    if validate_msg:
        msg = validate_msg.lower()
        if "no :named" in msg or "contains no :named" in msg:
            return "no_named_tags"
        if "duplicate :named" in msg:
            return "duplicate_tag"
        if "unknown constraint token" in msg:
            return "unknown_constraint"
        if "does not match expected patterns" in msg:
            return "tag_mismatch"
    return "unknown"


# ────────────────────────────────────────────────────────────────────
# Prompt prep helpers
# ────────────────────────────────────────────────────────────────────

def _slim_requirements_list(reqs: Any) -> List[Dict[str, Any]]:
    if isinstance(reqs, dict) and "requirements" in reqs:
        reqs = reqs.get("requirements", [])

    out: List[Dict[str, Any]] = []
    if not isinstance(reqs, list):
        return out

    for r in reqs:
        if isinstance(r, dict):
            raw_components = r.get("components") or []
            comps: List[Dict[str, Any]] = []
            for c in raw_components:
                if isinstance(c, dict):
                    comps.append(dict(c))
                else:
                    comps.append({"text": str(c)})
            out.append({
                "requirement": r.get("requirement", ""),
                "source": r.get("source", ""),
                "components": comps,
            })
        else:
            out.append({"requirement": str(r), "source": "", "components": []})
    return out


def _first_component_text(req_entry: Any) -> str:
    if isinstance(req_entry, dict):
        comps = req_entry.get("components") or []
        if comps:
            first = comps[0]
            if isinstance(first, dict):
                return first.get("text", "") or ""
            return str(first)
    return ""


def _components_json(req_entry: Any) -> str:
    comps = (req_entry.get("components") or []) if isinstance(req_entry, dict) else []
    return json.dumps(comps, ensure_ascii=False, indent=2)


def _join_component_texts(req_entry: Any) -> str:
    if isinstance(req_entry, dict):
        texts = []
        for c in (req_entry.get("components") or []):
            if isinstance(c, dict):
                texts.append(c.get("text", "") or "")
            else:
                texts.append(str(c))
        return " || ".join([t for t in texts if t])
    return ""


def _requirement_json(req_entry: Any) -> str:
    if isinstance(req_entry, dict):
        comps_out = []
        for c in (req_entry.get("components") or []):
            if isinstance(c, dict):
                comps_out.append({"text": c.get("text", ""), "constraint": c.get("constraint", "")})
            else:
                comps_out.append({"text": str(c), "constraint": ""})
        return json.dumps({
            "requirement": req_entry.get("requirement", ""),
            "source": req_entry.get("source", ""),
            "components": comps_out,
        }, ensure_ascii=False, indent=2)
    return json.dumps({"requirement": str(req_entry), "source": "", "components": []},
                      ensure_ascii=False, indent=2)


def _merged_new_declarations(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Assemble variables for #NEW_VARIABLE_DECLARATIONS# from canonical + demographics only.
    - Ignores new_variable_declarations and new_variable_declarations_min.
    - Synthesizes composed qualifier variables from canonical qualifiers.
    - Drops canonical **stems** that are listed in reusable_variables (to avoid redecl).
    - Optionally (env-guarded) merges remaining/noncanonical vars that aren't duplicates.
    """

    def _norm_list(lst):
        return [_normalize_decl(o) for o in (lst or []) if isinstance(o, dict)]

    out: List[Dict[str, Any]] = []

    # 1) Demographics (age/sex/pregnancy) — keep even if reusable
    out += _norm_list(context.get("new_age_sex_pregnancystatus_declarations"))

    # 2) Canonical stems
    canon: List[Dict[str, Any]] = context.get("new_canonical_variable_declarations") or []
    out += _norm_list(canon)

    # Track canonical stems explicitly (no '@@' = stem)
    canonical_stems: set[str] = set()
    for c in canon:
        if isinstance(c, dict):
            nm = c.get("entity_variable_name")
            if isinstance(nm, str) and nm.strip():
                canonical_stems.add(nm.strip())

    # 2b) Synthesize composed qualifier variables from canonical (always keep)
    for c in canon:
        if not isinstance(c, dict):
            continue
        stem = c.get("entity_variable_name")
        if not stem:
            continue

        # collect (form, meaning, qdecl)
        q_items: List[Tuple[str, str, Optional[str]]] = []

        raw_det = c.get("qualifier_predicates_detailed")
        if isinstance(raw_det, list) and raw_det and isinstance(raw_det[0], dict):
            for q in raw_det:
                form = (q.get("qualifier_variable_snake_case_form") or "").strip()
                if not form:
                    continue
                meaning = (q.get("qualifier_variable_meaning") or q.get("qualifier_meaning") or "").strip()
                qdecl = (
                    q.get("qualifier_variable_declaration")
                    or q.get("variable_declaration")
                    or q.get("declaration")
                )
                q_items.append((form, meaning, qdecl if isinstance(qdecl, str) and qdecl.strip() else None))
        else:
            for s in (c.get("qualifier_predicates") or []):
                if isinstance(s, str) and s.strip():
                    q_items.append((s.strip(), "", None))

        for form, meaning, qdecl in q_items:
            suffix = form if form.startswith("@@") else f"@@{form}"
            composed = f"{stem}{suffix}"
            item = {
                "span": "",
                "variable_name": composed,
                "variable_meaning": meaning,
            }
            if qdecl:
                # 按你的要求：独立限定词变量使用 qualifier_variable_declaration
                item["qualifier_variable_declaration"] = qdecl
            out.append(item)


    # 3) Drop reusable **canonical stems** (keep qualifiers & demographics)
    reusable_names = _names_from_mixed_list(context.get("reusable_variables"))
    reusable_canon_stems = canonical_stems & reusable_names
    if reusable_canon_stems:
        out = [
            o for o in out
            if not (isinstance(o, dict)
                    and isinstance(o.get("variable_name"), str)
                    and ("@@" not in o["variable_name"])            # stem (no qualifier)
                    and (o["variable_name"] in reusable_canon_stems) # listed reusable
                   )
        ]

    # 4) Optionally include remaining/noncanonical (on by default)
    include_remaining = True
    if include_remaining:
        # 4.1 读取并归一化 remaining/noncanonical
        remaining_raw: List[Dict[str, Any]] = []
        for key in ("new_remaining_variable_declarations", "new_noncanonical_variable_declarations"):
            remaining_raw += _norm_list(context.get(key))

        # 4.2 先加入 stem（避免与前面重复）
        present = {o.get("variable_name") for o in out if isinstance(o, dict)}
        remaining_stems = [
            r for r in remaining_raw
            if r.get("variable_name") and r["variable_name"] not in present
        ]
        out += remaining_stems
        present.update(r["variable_name"] for r in remaining_stems)

        # 4.3 ☆ 关键改动：从 remaining 的 qualifier_variables 合成独立限定词变量
        #     即把 @@suffix 与 stem 组合后的 composed 名字作为独立条目写入
        for r in remaining_raw:  # 用 raw：即便 stem 被去重，限定词仍可落地
            qvars = r.get("qualifier_variables") or []
            if not isinstance(qvars, list):
                continue
            for q in qvars:
                if not isinstance(q, dict):
                    continue
                cname = (q.get("qualifier_variable_composed_name") or "").strip()
                if not cname or cname in present:
                    continue
                item = {
                    "span": "",
                    "variable_name": cname,  # e.g., has_finding_of_impaired_cognition_now@@based_on_...
                    "variable_meaning": (q.get("qualifier_variable_meaning") or "").strip(),
                }
                qvd = q.get("qualifier_variable_declaration")
                if isinstance(qvd, str) and qvd.strip():
                    item["qualifier_variable_declaration"] = qvd.strip()
                out.append(item)
                present.add(cname)


    # 5) Dedup by variable_name (keep last) and sort for determinism
    out = _unique_by_var_name(out)
    out.sort(key=lambda d: d.get("variable_name", ""))

    # 6) Return only well-formed records
    return [o for o in out if o.get("variable_name")]


def _normalize_qualifier(q: dict) -> dict:
    if not isinstance(q, dict):
        return {}
    out = {
        "qualifier_variable_composed_name": (
            q.get("qualifier_variable_composed_name")
            or q.get("composed_name")
            or q.get("qualifier_variable_name")
        ),
        "qualifier_variable_meaning": (
            q.get("qualifier_variable_meaning")
            or q.get("meaning")
            or q.get("qualifier_meaning")
            or q.get("qualifier_variable_description")
        ),
    }
    qvd = (
        q.get("qualifier_variable_declaration")
        or q.get("variable_declaration")
        or q.get("declaration")
    )
    if isinstance(qvd, str) and qvd.strip():
        out["qualifier_variable_declaration"] = qvd.strip()
    return out

def _normalize_decl(obj: dict) -> dict:
    """
    Convert any legacy declaration (canonical or free) into the compact form:
      {
        "span": "...",
        "variable_name": "...",
        "variable_meaning": "...",
        "variable_declaration": "<stringified JSON, optional>",
        "qualifier_variable_declaration": "<stringified JSON, optional>",
        "qualifier_variables": [
          {
            "qualifier_variable_composed_name": "...",
            "qualifier_variable_meaning": "...",
            "qualifier_variable_declaration": "<stringified JSON, optional>"
          }, ...
        ]
      }
    """
    if not isinstance(obj, dict):
        return {}

    span = obj.get("span") or obj.get("entity_span") or obj.get("surface_string")

    variable_name = obj.get("variable_name") or obj.get("entity_variable_name")
    if isinstance(variable_name, str):
        variable_name = variable_name.strip()
    else:
        variable_name = None

    variable_meaning = (
        obj.get("variable_meaning")
        or obj.get("entity_variable_meaning")
        or obj.get("variable_description")
        or obj.get("entity_meaning")
    )

    variable_declaration = (
        obj.get("variable_declaration")
        or obj.get("declaration")
        or obj.get("variable_json_comment")
    )

    # 顶层限定词声明（若存在）
    top_q_decl = (
        obj.get("qualifier_variable_declaration")
        or obj.get("qualifier_declaration")
    )

    # ===== 关键：组合名构造 =====
    stem = (variable_name or obj.get("entity_variable_name") or "").strip()

    def _compose_name(form: str) -> str:
        """
        将传入 form 统一成 composed 名：
        - 若包含 '@@' 且以 '@@' 开头：在前面补上 stem
        - 若包含 '@@' 但不以 '@@' 开头：视为已组合好的 composed 名，原样返回
        - 若不含 '@@'：当作 suffix，补上 '@@'；若有 stem，则前缀 stem
        """
        if not isinstance(form, str):
            return ""
        form = form.strip()
        if not form:
            return ""
        if "@@" in form:
            # '@@suffix' → 需要前置 stem；'stem@@suffix' → 认为已组合，原样返回
            if form.startswith("@@") and stem:
                return f"{stem}{form}"
            return form
        # 无 '@@'，当作 suffix
        suffix = form if form.startswith("@@") else f"@@{form}"
        return f"{stem}{suffix}" if stem else suffix

    # ===== 先保留已有的 qualifier_variables，并归一化为 composed 名 =====
    qvars: list[dict] = []
    qvars_raw = obj.get("qualifier_variables") or []
    if isinstance(qvars_raw, list):
        tmp = [_normalize_qualifier(q) for q in qvars_raw if isinstance(q, dict)]
        for nq in tmp:
            cname = (nq.get("qualifier_variable_composed_name") or
                     nq.get("composed_name") or
                     nq.get("qualifier_variable_name") or "").strip()
            if cname:
                nq["qualifier_variable_composed_name"] = _compose_name(cname)
                qvars.append(nq)

    # ===== 从 canonical 字段补齐（含 declaration）=====
    for q in (obj.get("qualifier_predicates_detailed") or []):
        if not isinstance(q, dict):
            continue
        form = (q.get("qualifier_variable_snake_case_form") or "").strip()
        if not form:
            continue
        entry = {
            "qualifier_variable_composed_name": _compose_name(form),
            "qualifier_variable_meaning": (q.get("qualifier_variable_meaning")
                                           or q.get("qualifier_meaning") or "").strip(),
        }
        qvd = (
            q.get("qualifier_variable_declaration")
            or q.get("variable_declaration")
            or q.get("declaration")
        )
        if isinstance(qvd, str) and qvd.strip():
            entry["qualifier_variable_declaration"] = qvd.strip()
        qvars.append(entry)

    for s in (obj.get("qualifier_predicates_for_semantics_not_already_captured_with_stem") or []):
        if isinstance(s, str) and s.strip():
            qvars.append({
                "qualifier_variable_composed_name": _compose_name(s.strip()),
                "qualifier_variable_meaning": "",
            })

    for s in (obj.get("qualifier_predicates") or []):
        if isinstance(s, str) and s.strip():
            qvars.append({
                "qualifier_variable_composed_name": _compose_name(s.strip()),
                "qualifier_variable_meaning": "",
            })

    # ===== 去重：优先保留带 meaning / declaration 的项 =====
    dedup: dict[str, dict] = {}
    for q in qvars:
        cname = q.get("qualifier_variable_composed_name")
        if not isinstance(cname, str) or not cname:
            continue
        if cname not in dedup:
            dedup[cname] = q
        else:
            # merge 优先级：meaning / declaration
            if (not dedup[cname].get("qualifier_variable_meaning")
                and q.get("qualifier_variable_meaning")):
                dedup[cname]["qualifier_variable_meaning"] = q["qualifier_variable_meaning"]
            if (not dedup[cname].get("qualifier_variable_declaration")
                and q.get("qualifier_variable_declaration")):
                dedup[cname]["qualifier_variable_declaration"] = q["qualifier_variable_declaration"]

    # ===== 输出 =====
    out = {
        "span": span,
        "variable_name": variable_name,
        "variable_meaning": variable_meaning,
    }
    if isinstance(variable_declaration, str) and variable_declaration.strip():
        out["variable_declaration"] = variable_declaration.strip()
    if isinstance(top_q_decl, str) and top_q_decl.strip():
        out["qualifier_variable_declaration"] = top_q_decl.strip()
    if dedup:
        out["qualifier_variables"] = list(dedup.values())

    return out




# --- keep all your existing imports, constants, helpers above this line ---

class SMTIncrementalTranslator(dspy.Module):
    """LLM-driven, retry-aware incremental translator with solver-aware nudges."""
    MAX_ATTEMPTS = MAX_ATTEMPTS  # allow env override

    def __init__(self, engine, *, log_dir: str | None = None):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./translator_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def forward(self, context: dict) -> dict:  # type: ignore[override]
        if "requirements" in context:
            context["requirements"] = _slim_requirements_list(context["requirements"])

        requirements: list = context.get("requirements", [])
        idx: int = context.get("current_requirement_index", 0)

        if not requirements:
            return context
        if idx < 0 or idx >= len(requirements):
            _log("translator ✗", idx, "current_requirement_index out of range")
            return context

        req_entry = requirements[idx]
        requirement_txt = (req_entry.get("requirement") if isinstance(req_entry, dict) else str(req_entry))
        component_text = _first_component_text(req_entry)
        components_blob = _components_json(req_entry)
        requirement_blob = _requirement_json(req_entry)

        side = context.get("inc_exc", "unknown")
        print(f"In SMT IncrementalTranslator inc_exc is {side}")
        if side == "inclusion":
            smt_prompt_tpl: str = context.get("SMTIncrementalTranslatorInclusion_prompt", "")
        else:
            smt_prompt_tpl: str = context.get("SMTIncrementalTranslatorExclusion_prompt", "")

        if not smt_prompt_tpl:
            _log("translator ✗", idx, "prompt template missing")
            return context

        # Ensure all new declarations are present (merged) for the prompt
        merged_newdecls = _merged_new_declarations(context)

        # Gather already-declared SMT symbols from the partial program (declare/define)
        _program_lines = context.get("smt_program_lines") or []
        print("program_lines:", _program_lines)
        declared_ids = _declared_symbols_from_program(_program_lines)
        print("declared_ids: ", declared_ids)

        # Move anything already-declared (or already reusable) into the prompt's reusable list
        existing_reusables = context.get("reusable_variables", [])
        newdecls_for_prompt, reusables_for_prompt, moved = _filter_into_reusables(
            merged_newdecls, declared_ids, existing_reusables
        )
        context["reusable_variables"] = reusables_for_prompt

        reusable_json = json.dumps(reusables_for_prompt, indent=2, ensure_ascii=False)
        newdecl_json  = json.dumps(newdecls_for_prompt,  indent=2, ensure_ascii=False)

        if moved:
            _log("translator", idx, f"moved {len(moved)} vars to reusables: {', '.join(sorted(moved))}")

        smt_vars_blocks = _extract_decl_and_def_blocks_from_program(_program_lines)
        smt_vars_so_far = "\n".join(smt_vars_blocks)
        print("vars_blocks: ", smt_vars_blocks)
        print("smt_vars_so_far: ", smt_vars_so_far)

        # === NEW: hints based on previous solver failure ===
        extra_hint = ""
        h = context.get("translator_retry_hint_from_solver")
        if isinstance(h, dict):
            st = (h.get("status") or "").lower()
            if st == "error":
                extra_hint = (
                    "\n\nPrevious solver run hit a parse error. "
                    "Strictly emit parsable SMT-LIB only (no prose or markdown). "
                    "Declare any 0-arity symbols you introduce; ensure balanced parentheses. "
                    "Every assertion MUST include a :named tag with the required scheme."
                )
            elif st == "unsat":
                extra_hint = (
                    "\n\nPrevious solver run was UNSAT. Double-check polarity for EXCLUSION vs INCLUSION, "
                    "avoid over-constraining auxiliaries, and keep boolean structure minimal."
                )
            elif st == "unknown":
                extra_hint = (
                    "\n\nPrevious solver run was UNKNOWN. Prefer simpler boolean structure; avoid division / "
                    "non-linear arithmetic; unfold macros if any."
                )

        base_prompt = (
            smt_prompt_tpl
            .replace("#SMT_VARIABLES_BY_FAR#", smt_vars_so_far)
            .replace("#REQUIREMENT#", requirement_blob)
            .replace("#REQ_IDX#", str(idx))
            .replace("#NEW_VARIABLE_DECLARATIONS#", newdecl_json)
            .replace("#REUSABLE_VARIABLES#", reusable_json)
            + extra_hint
        )

        trial_id = context.get("trial_id", "unknown_trial")

        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc", "inclusion", "include", "in"}:  return "inclusion"
            if s in {"exc", "exclusion", "exclude", "ex"}:  return "exclusion"
            return s or "unknown"

        stage_dir = os.path.join(
            self.log_dir,
            str(trial_id),
            _bucket(side),
            f"req{idx:03d}",
            _attempt_tag(context),
        )
        os.makedirs(stage_dir, exist_ok=True)

        p_log  = os.path.join(stage_dir, "prompt.txt")
        r_log  = os.path.join(stage_dir, "raw.txt")
        f_log  = os.path.join(stage_dir, "fragment_final.smt2")
        c_log  = os.path.join(stage_dir, "components.json")
        q_log  = os.path.join(stage_dir, "requirement.json")
        err_log = os.path.join(stage_dir, "errors.log")

        try:
            with open(p_log, "w", encoding="utf-8") as fh:
                fh.write(base_prompt)
            with open(c_log, "w", encoding="utf-8") as fh:
                fh.write(components_blob)
            with open(q_log, "w", encoding="utf-8") as fh:
                fh.write(requirement_blob)
        except Exception as e:
            _log("translator ⚠", idx, f"failed to write prompt/json logs: {e}")

        registry: set[str] = context.setdefault("global_named_tags", set())
        failure_notes: List[str] = []

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            context["translator_llm_attempt_index"] = attempt
            failure_reason: str | None = None
            attempt_prompt = _compose_retry_prompt(base_prompt, idx, side, attempt, None)

            llm_exc: Exception | None = None
            try:
                llm_out = self.engine(attempt_prompt)[0]
            except Exception as e:
                llm_exc = e
                llm_out = ""

            try:
                with open(r_log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")
            except Exception as e:
                _log("translator ⚠", idx, f"failed to write raw log: {e}")

            fragment, how = _parse_fragment(llm_out)
            validate_msg: str | None = None

            if fragment:
                attempt_path = os.path.join(stage_dir, f"llm_attempt{attempt:02d}.smt2")
                try:
                    with open(attempt_path, "w", encoding="utf-8") as fh:
                        fh.write(fragment)
                except Exception as e:
                    _log("translator ⚠", idx, f"failed to write attempt fragment: {e}")

                try:
                    _validate_tags(fragment, idx, registry, side=side)
                    try:
                        with open(f_log, "w", encoding="utf-8") as fh:
                            fh.write(fragment)
                    except Exception as e:
                        _log("translator ⚠", idx, f"failed to write fragment file: {e}")

                    _persist_canon_and_demographics(context, stage_dir)

                    context["new_smt_lines"] = _strip_headers(fragment)
                    _log("translate ✓", idx, f"accepted on llm_attempt {attempt} (outer {_attempt_tag(context)})")
                    return context

                except ValueError as exc:
                    validate_msg = str(exc)

            failure_reason = _classify_failure(llm_exc, None if fragment else how, validate_msg)
            failure_notes.append(f"attempt {attempt}: {failure_reason} ({validate_msg or how or 'engine error'})")

            if attempt < self.MAX_ATTEMPTS:
                next_prompt = _compose_retry_prompt(base_prompt, idx, side, attempt + 1, failure_reason)
                base_prompt = next_prompt
                _sleep_backoff(attempt)

        final_msg = " | ".join(failure_notes) if failure_notes else "unknown failure"
        try:
            with open(err_log, "w", encoding="utf-8") as fh:
                fh.write(final_msg + "\n")
        except Exception:
            pass
        _log("translator ✗", idx, f"exhausted retries: {final_msg}")
        context["translator_error"] = final_msg
        return context
