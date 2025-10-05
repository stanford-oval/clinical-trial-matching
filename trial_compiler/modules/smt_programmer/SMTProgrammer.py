# file: smt_programmer.py
# ===============================================================
# Incremental driver that orchestrates:
#   free-prepass → reuser → canonical/DEM namer → free/noncanonical namer
#   → translator → solver → (optional) naive refiner → verifier
# With optional early-stop after submodules and a fail-open verifier policy.
# ===============================================================

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

import z3
import dspy

# ── pipeline pieces (as per your aggregator) ─────────────────────
from .SMTIncrementalProgrammer import (
    SMTIncrementalSemanticChecker,
    SMTIncrementalSolverBasedValidator,
    SMTIncrementalTranslator,
    SMTIncrementalSolverBasedNaiveRefiner,
    SMTIncrementalVerifier,
    SMTIncrementalReusableVariableIdentifier,
    SMTIncrementalDemographicsVariableNamer,
    SMTIncrementalCanonicalVariableNamer,
    SMTIncrementalFreeVariableNamer,
    SMTIncrementalNewVariableFilter
)

from .SMTProgrammerPreprocessor import (
    SMTProgrammerFreeEntityExtractor,
    SMTProgrammerFreeEntityQualifierIdentifier,
    SMTProgrammerTopLevelEntityFilter,
    SMTProgrammerEntityEnricher,
)

from .SMTDeclarationParser import SMTDeclarationParser
from smt_core.utils.z3_helpers import _log, _print_variable_index
from .integrated_reporter import IntegratedRequirementReporter
from .SMTIncrementalProgrammer.namer_checks import _extract_declared_symbols  # lazy import to avoid cycle


# ─────────────────────────────────────────────────────────────────
# Robust declaration de-duplication (form-aware, safe)
# ─────────────────────────────────────────────────────────────────
import re as _re
from typing import Tuple as _Tuple, Optional as _Optional

_DECL_START_RE = _re.compile(r'^\s*\(\s*declare-(?:const|fun)\b', _re.IGNORECASE | _re.ASCII)
_DECL_CONST_FORM_RE = _re.compile(
    r'^\s*\(\s*declare-const\s+([^\s()]+)\s+(.+?)\s*\)\s*$', _re.IGNORECASE | _re.DOTALL
)
_DECL_FUN_FORM_RE = _re.compile(
    r'^\s*\(\s*declare-fun\s+([^\s()]+)\s*\(\s*([^\)]*)\s*\)\s+(.+?)\s*\)\s*$',
    _re.IGNORECASE | _re.DOTALL
)

def _strip_line_comment_for_balance(line: str) -> str:
    out: List[str] = []
    in_str = False
    esc = False
    for ch in line:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == ';':
            break
        if ch == '"':
            in_str = True
        out.append(ch)
    return ''.join(out)

def _gather_complete_form(lines: List[str], i: int) -> _Tuple[str, int]:
    buf: List[str] = []
    depth = 0
    started = False
    j = i
    while j < len(lines):
        ln = lines[j]
        piece = ln
        bal_src = _strip_line_comment_for_balance(ln)
        for ch in bal_src:
            if ch == '(':
                depth += 1
                started = True
            elif ch == ')':
                depth -= 1
        buf.append(piece)
        j += 1
        if started and depth <= 0:
            break
    return ('\n'.join(buf), j)

def _normalize_ws(s: str) -> str:
    return _re.sub(r'\s+', ' ', (s or '').strip())

def _decl_signature_from_form(form_text: str) -> _Optional[_Tuple[str, str, str, str]]:
    t = _strip_line_comment_for_balance(form_text).strip()
    m = _DECL_CONST_FORM_RE.match(t)
    if m:
        name, sort = m.group(1), _normalize_ws(m.group(2))
        return ('const', name, '', sort)
    m = _DECL_FUN_FORM_RE.match(t)
    if m:
        name, args_raw, ret = m.group(1), _normalize_ws(m.group(2)), _normalize_ws(m.group(3))
        if args_raw in ('', '()'):
            return ('const', name, '', ret)
        return ('fun', name, args_raw, ret)
    return None

_DECL_LINE_SIG_RE = _re.compile(
    r'^\s*\(\s*declare-(const|fun)\s+([^\s()]+)'
    r'(?:\s*\(\s*([^\)]*)\s*\))?\s+(.+?)\s*\)\s*(?:;.*)?$',
    _re.IGNORECASE
)

def _decl_sig_from_line(ln: str):
    m = _DECL_LINE_SIG_RE.match(_strip_line_comment_for_balance(ln))
    if not m:
        return None
    kind = 'const' if m.group(1).lower() == 'const' or not m.group(3) else 'fun'
    name = m.group(2)
    args = _normalize_ws(m.group(3) or '')
    ret  = _normalize_ws(m.group(4))
    if args in ('', '()'):
        kind, args = 'const', ''
    return (kind, name, args, ret)

def _existing_decl_sigs(lines: List[str]) -> Set[_Tuple[str, str, str, str]]:
    seen = set()
    for ln in lines:
        sig = _decl_sig_from_line(ln)
        if sig:
            seen.add(sig)
    return seen


def _filter_new_decl_duplicates(existing_lines: List[str], new_lines: List[str]) -> List[str]:
    out: List[str] = []
    seen = _existing_decl_sigs(existing_lines)
    for ln in new_lines:
        sig = _decl_sig_from_line(ln)
        if sig:
            if sig in seen:
                continue
            seen.add(sig)
        out.append(ln)
    return out



def _dedup_program_lines_full(lines: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[_Tuple[str, str, str, str]] = set()
    for ln in lines:
        sig = _decl_sig_from_line(ln)
        if sig:
            if sig in seen:
                continue
            seen.add(sig)
        out.append(ln)
    return out

# ─────────────────────────────────────────────────────────────────
# Local helpers for program state and grouping
# ─────────────────────────────────────────────────────────────────
def _commit_slice(ctx: dict, req_idx: int) -> None:
    """Move ctx['new_smt_lines'] into the master program exactly once (dedup declarations)."""
    slice_lines: List[str] = ctx.pop("new_smt_lines", [])
    if not slice_lines:
        return
    prog = ctx.setdefault("smt_program_lines", [])
    start = len(prog)
    uniq = _filter_new_decl_duplicates(prog, slice_lines)
    prog.extend(uniq)
    prog[:] = _dedup_program_lines_full(prog)
    prog.append("")  # spacer
    end = len(prog)
    ctx.setdefault("req_blocks", {})[req_idx] = (start, end)

def _assert_all_symbols_declared(ctx):
    text = "\n".join(ctx.get("smt_program_lines", []))
    used = set(_re.findall(r'[^\s()]+', text))  # crude; refine if you want
    declared = {n for _, n, _, _ in _existing_decl_sigs(ctx.get("smt_program_lines", []))}
    missing = [u for u in used if u.startswith("patient_") and u not in declared and u not in {"true","false"}]
    if missing:
        raise AssertionError(f"Missing declarations: {sorted(missing)[:5]} ...")


def snapshot_program_state(ctx: dict) -> dict:
    return {
        "smt_program_lines": list(ctx.get("smt_program_lines", [])),
        "req_blocks": dict(ctx.get("req_blocks", {})),
        "global_named_tags": set(ctx.get("global_named_tags", set())),
    }

def _stage_req_block(ctx: dict, req_idx: int) -> None:
    prog = ctx.setdefault("smt_program_lines", [])
    draft = ctx.get("new_smt_lines", [])
    start = len(prog)
    end = start + len(draft)
    ctx.setdefault("req_blocks", {})[req_idx] = (start, end)

def _find_top_level_block_for_req(ctx: Dict[str, Any], idx: int) -> Optional[dict]:
    arr = ctx.get("requirements_entities_attributes_top_level") or []
    if not isinstance(arr, list):
        return None
    for b in arr:
        rid = b.get("requirement_id")
        if str(rid) == str(idx):
            return b
    try:
        if 0 <= idx < len(arr):
            return arr[idx]
    except Exception:
        pass
    return None

def _collect_qualifiers_from_entrec(entrec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for q in (entrec.get("qualifiers") or []):
        q = str(q or "").strip()
        if q:
            out.append(q)
    for a in (entrec.get("attributes") or []):
        oq = str((a or {}).get("original_qualifier") or "").strip()
        if oq:
            out.append(oq)
    seen = set()
    uniq: List[str] = []
    for q in out:
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq

def _seed_entity_qualifier_pairs(context: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    tlb = _find_top_level_block_for_req(context, idx)
    if not tlb:
        return pairs

    bucket: Dict[tuple, Dict[str, Any]] = {}

    def _key(ent: dict) -> tuple:
        cid = (ent or {}).get("conceptId")
        if cid not in (None, ""):
            return ("cid", str(cid))
        return (
            "span",
            ((ent or {}).get("surface_string") or "").strip().lower(),
            (ent or {}).get("start"),
            (ent or {}).get("end"),
        )

    for entrec in (tlb.get("entities") or []):
        ent = (entrec or {}).get("entity") or {}
        if ent.get("is_canonical_entity", True) is False:
            k = _key(ent)
            item = bucket.setdefault(
                k,
                {
                    "entity": {
                        "surface_string": ent.get("surface_string"),
                        "type": ent.get("type"),
                        "start": ent.get("start"),
                        "end": ent.get("end"),
                        "preferred_term": ent.get("preferred_term"),
                        "conceptId": ent.get("conceptId"),
                    },
                    "qualifiers": [],
                },
            )
            for f in ("surface_string", "type", "start", "end", "preferred_term", "conceptId"):
                if (item["entity"].get(f) in (None, "")) and (ent.get(f) not in (None, "")):
                    item["entity"][f] = ent.get(f)

            q_list = _collect_qualifiers_from_entrec(entrec)
            if q_list:
                seen = set(item["qualifiers"])
                for q in q_list:
                    if isinstance(q, str):
                        qn = q.strip()
                        if qn and qn not in seen:
                            item["qualifiers"].append(qn)
                            seen.add(qn)

    return list(bucket.values())

def _debug_find_duplicate_decls(lines: List[str]):
    first, dupes = {}, []
    i = 0
    while i < len(lines):
        if _DECL_START_RE.match(lines[i]):
            form, j = _gather_complete_form(lines, i)
            sig = _decl_signature_from_form(form)
            if sig:
                if sig in first:
                    dupes.append((sig, first[sig], i))
                else:
                    first[sig] = i
            i = j + 1
        else:
            i += 1
    return dupes

def _collect_already_declared_names(context: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()

    names |= set(_extract_declared_symbols(context.get("smt_program_lines", [])))

    for rv in (context.get("reusable_variables") or []):
        if isinstance(rv, dict):
            n = rv.get("variable_name")
            if n:
                names.add(str(n))

    for blk_key in ("new_age_sex_pregnancystatus_declarations", "new_canonical_variable_declarations"):
        for obj in (context.get(blk_key) or []):
            if isinstance(obj, dict):
                n = obj.get("entity_variable_name")
                if n:
                    names.add(str(n))

    adv = context.get("already_declared_variables")
    if isinstance(adv, list):
        for x in adv:
            if isinstance(x, str):
                names.add(x)
            elif isinstance(x, dict):
                n = x.get("variable_name") or x.get("name")
                if n:
                    names.add(str(n))

    return names

def _dump_already_declared_for_prompt(context: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"variable_name": n} for n in sorted(_collect_already_declared_names(context))]

def _canon_qualifiers_with_composed(decl: Dict[str, Any]) -> List[Dict[str, str]]:
    stem = str(decl.get("entity_variable_name") or "").strip()
    out: List[Dict[str, str]] = []
    if not stem:
        return out

    detailed = []
    qpd = decl.get("qualifier_predicates_detailed")
    if isinstance(qpd, list) and qpd:
        for d in qpd:
            if not isinstance(d, dict):
                continue
            form = str(d.get("qualifier_variable_snake_case_form") or "").strip()
            mean = str(d.get("qualifier_meaning") or "").strip()
            if form.startswith("@@"):
                detailed.append((form, mean))
    if not detailed:
        qps = decl.get("qualifier_predicates_for_semantics_not_already_captured_with_stem")
        if isinstance(qps, list):
            for t in qps:
                if isinstance(t, str) and t.strip().startswith("@@"):
                    detailed.append((t.strip(), ""))

    seen = set()
    for form, mean in detailed:
        composed = stem + form
        if composed in seen:
            continue
        seen.add(composed)
        out.append({
            "qualifier_variable_name": form,
            "qualifier_variable_meaning": mean,
            "qualifier_variable_composed_name": composed,
        })
    return out

def _free_qualifiers_with_composed(decl: Dict[str, Any]) -> List[Dict[str, str]]:
    stem = str(decl.get("variable_name") or "").strip()
    out: List[Dict[str, str]] = []
    if not stem:
        return out

    qvars = decl.get("qualifier_variables") or []
    seen = set()
    for q in qvars:
        if not isinstance(q, dict):
            continue
        name = str(q.get("qualifier_variable_name") or "").strip()
        mean = str(q.get("qualifier_variable_meaning") or "").strip()
        if not name.startswith("@@"):
            name = "@@" + name.lstrip("@")
        composed = q.get("qualifier_variable_composed_name") or (stem + name)
        if composed in seen:
            continue
        seen.add(composed)
        out.append({
            "qualifier_variable_name": name,
            "qualifier_variable_meaning": mean,
            "qualifier_variable_composed_name": composed,
        })
    return out

def _build_entity_groups_for_current(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups_by_stem: Dict[str, Dict[str, Any]] = {}

    for d in context.get("new_canonical_variable_declarations", []) or []:
        if not isinstance(d, dict):
            continue
        stem = str(d.get("entity_variable_name") or "").strip()
        if not stem:
            continue
        quals = _canon_qualifiers_with_composed(d)
        grp = groups_by_stem.get(stem, {
            "entity_variable_name": stem,
            "variable_meaning": str(d.get("variable_meaning") or ""),
            "entity_canonical_form": d.get("entity_canonical_form_from_entity_canonical_forms_block"),
            "span": d.get("span") or "",
            "entity_type": d.get("entity_type") or "",
            "qualifier_variables": [],
            "all_composed_variable_names": [stem],
            "source": "canonical",
        })
        if not grp.get("variable_meaning") and d.get("variable_meaning"):
            grp["variable_meaning"] = d.get("variable_meaning")
        existing = {q.get("qualifier_variable_composed_name") for q in grp["qualifier_variables"]}
        for q in quals:
            if q.get("qualifier_variable_composed_name") not in existing:
                grp["qualifier_variables"].append(q)
                grp["all_composed_variable_names"].append(q["qualifier_variable_composed_name"])
        groups_by_stem[stem] = grp

    for d in context.get("new_remaining_variable_declarations", []) or []:
        if not isinstance(d, dict):
            continue
        stem = str(d.get("variable_name") or "").strip()
        if not stem:
            continue
        quals = _free_qualifiers_with_composed(d)
        grp = groups_by_stem.get(stem, {
            "entity_variable_name": stem,
            "variable_meaning": str(d.get("variable_meaning") or ""),
            "entity_canonical_form": None,
            "span": "",
            "entity_type": d.get("entity_type") or "",
            "qualifier_variables": [],
            "all_composed_variable_names": [stem],
            "source": "free",
        })
        if not grp.get("variable_meaning") and d.get("variable_meaning"):
            grp["variable_meaning"] = d.get("variable_meaning")
        existing = {q.get("qualifier_variable_composed_name") for q in grp["qualifier_variables"]}
        for q in quals:
            if q.get("qualifier_variable_composed_name") not in existing:
                grp["qualifier_variables"].append(q)
                grp["all_composed_variable_names"].append(q["qualifier_variable_composed_name"])
        if stem not in groups_by_stem:
            groups_by_stem[stem] = grp
        else:
            base = groups_by_stem[stem]
            base["qualifier_variables"] = grp["qualifier_variables"]
            base["all_composed_variable_names"] = list({*base["all_composed_variable_names"], *grp["all_composed_variable_names"]})
            if not base.get("variable_meaning") and grp.get("variable_meaning"):
                base["variable_meaning"] = grp["variable_meaning"]

    def _sort_key(item: Dict[str, Any]):
        return (0 if item.get("source") == "canonical" else 1, item.get("entity_variable_name", ""))

    groups = sorted(groups_by_stem.values(), key=_sort_key)
    return groups

def _groups_to_min_for_translator(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in groups or []:
        stem = g.get("entity_variable_name") or ""
        span = g.get("span") or ""
        meaning = g.get("variable_meaning") or ""
        qvars_in = g.get("qualifier_variables") or []
        qvars_min: List[Dict[str, str]] = []
        for q in qvars_in:
            if not isinstance(q, dict):
                continue
            comp = q.get("qualifier_variable_composed_name") or ""
            mean = q.get("qualifier_variable_meaning") or ""
            if comp:
                qvars_min.append({
                    "qualifier_variable_composed_name": comp,
                    "qualifier_variable_meaning": mean,
                })
        out.append({
            "span": span,
            "variable_name": stem,
            "variable_meaning": meaning,
            "qualifier_variables": qvars_min,
        })
    return out

def _collect_all_linked_variables_for_persist(ctx: dict) -> List[Any]:
    arr = ctx.get("all_linked_variables")
    if not isinstance(arr, list):
        arr = ctx.get("all_linked_variales")
    if not isinstance(arr, list):
        return []
    out: List[Any] = []
    seen: set = set()
    for x in arr:
        if isinstance(x, dict):
            name = x.get("entity_variable_name")
            if isinstance(name, str) and name.strip():
                key = ("var", name.strip())
            else:
                cf = x.get("canonical_form")
                key = ("ent", str(cf).strip()) if cf else ("raw", json.dumps(x, sort_keys=True, ensure_ascii=False))
        else:
            key = ("raw", str(x))
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


# ─────────────────────────────────────────────────────────────────
# --- keep all your existing imports and helpers above this line ---

class SMTProgrammer(dspy.Module):
    """
    High-level orchestrator (incremental only).

    Early stopping via `stop_after`:
      - "reuse", "canonical", "free", "translator", "solver", "verifier", "free_prepass"
    Verifier fail-open policy: "never" | "eventual" | "always"
    """

    _STOP_ALIASES = {
        "program": "program", "full": "program", "all": "program", "end": "program",
        "reuse": "reuse", "reuser": "reuse", "reusable": "reuse",
        "canonical": "canonical", "canon": "canonical", "canonical_namer": "canonical",
        "free": "free", "noncanon": "free", "free_namer": "free",
        "translator": "translator", "translate": "translator",
        "solver": "solver", "validate": "solver", "validator": "solver",
        "verifier": "verifier", "verify": "verifier",
        "free_prepass": "free_prepass", "prepass": "free_prepass", "freepass": "free_prepass",
        "demographics": "demographics", "demog": "demographics", "age_sex_preg": "demographics",
    }

    def __init__(
        self,
        engine,
        *,
        max_reqs: Optional[int] = 25,
        max_verifier_retries: int = 8,
        max_global_passes: int = 8,
        max_stale_passes: int = 8,
        free_entity_out_dir: Path | str | None = None,
        free_qualifier_out_dir: Path | str | None = None,
        entity_filter_out_dir: Path | str | None = None,
        entity_enricher_out_dir: Path | str | None = None,
        smt_out_dir: Path | str | None = None,
        var_out_dir: Path | str | None = None,
        canon_out_dir: Path | str | None = None,
        ent_out_dir: Path | str | None = None,
        requirements_out_dir: Path | str | None = None,
        namer_log_dir: str | None = None,
        translator_log_dir: str | None = None,
        validator_log_dir: str | None = None,
        verifier_log_dir: str | None = None,
        report_dir: Path | str | None = None,
        profile_out_dir: Path | str | None = None,
        stop_after: Optional[str] = None,
        persist_on_stop: bool = False,
        verifier_fail_open_policy: str = "eventual",
        verifier_fail_open_after_attempts: int | None = None,
        enable_free_prepass: bool = True,
        qualifiers_only_prepass: bool = False,
        enable_naive_refiner: bool = False,
        translator_top_p_schedule: Optional[List[float]] = None,
    ):
        super().__init__()
        self.engine = engine
        self.max_reqs = max_reqs
        self.max_verifier_retries = max_verifier_retries
        self.max_global_passes = max_global_passes
        self.max_stale_passes = max_stale_passes

        self.smt_out_dir = Path(smt_out_dir).expanduser().resolve() if smt_out_dir else None
        self.var_out_dir = Path(var_out_dir).expanduser().resolve() if var_out_dir else None
        self.canon_out_dir = Path(canon_out_dir).expanduser().resolve() if canon_out_dir else None
        self.ent_out_dir = Path(ent_out_dir).expanduser().resolve() if ent_out_dir else None
        self.requirements_out_dir = Path(requirements_out_dir).expanduser().resolve() if requirements_out_dir else None
        self.report_dir = Path(report_dir).expanduser().resolve() if report_dir else Path("./req_reports")

        self.profile_out_dir = (
            Path(profile_out_dir).expanduser().resolve() if profile_out_dir
            else (self.report_dir / "a_profiles")
        )
        self._prof_events: List[Dict[str, Any]] = []

        self.stop_after = self._canon_stop(stop_after)
        self.persist_on_stop = persist_on_stop

        self.verifier_fail_open_policy = (verifier_fail_open_policy or "never").lower()
        self.verifier_fail_open_after_attempts = (
            int(verifier_fail_open_after_attempts)
            if verifier_fail_open_after_attempts is not None
            else self.max_verifier_retries
        )

        self.enable_free_prepass = bool(enable_free_prepass)
        self.qualifiers_only_prepass = bool(qualifiers_only_prepass)
        self.enable_naive_refiner = bool(enable_naive_refiner)

        self.translator_top_p_schedule = (
            list(translator_top_p_schedule) if translator_top_p_schedule
            else [0.001, 0.15, 0.35, 0.35, 0.35]
        )

        # ── wire sub-modules ───────────────────────────────────────
        self.reuser = SMTIncrementalReusableVariableIdentifier(engine, log_dir=namer_log_dir)
        self.demographics_namer = SMTIncrementalDemographicsVariableNamer(engine, log_dir=namer_log_dir)
        self.canonical_namer = SMTIncrementalCanonicalVariableNamer(engine, log_dir=namer_log_dir)
        self.free_namer = SMTIncrementalFreeVariableNamer(engine, log_dir=namer_log_dir)
        self.new_variable_filter = SMTIncrementalNewVariableFilter(log_dir=namer_log_dir)

        self.translator = SMTIncrementalTranslator(engine, log_dir=translator_log_dir)
        self.solver_validator = SMTIncrementalSolverBasedValidator(
            engine,
            validator_log_dir=validator_log_dir,
            raise_on_unresolved=False,
            dump_on_error=True,          # ← ensure failing programs are logged
            static_fix_enabled=True,
            static_fix_max_rounds=2,
        )
        self.naive_refiner = SMTIncrementalSolverBasedNaiveRefiner(engine) if self.enable_naive_refiner else None
        self.verifier = SMTIncrementalVerifier(engine, log_dir=verifier_log_dir)

        self.semantic_checker = SMTIncrementalSemanticChecker(engine)
        self.declaration_parser = SMTDeclarationParser()

        self.free_extractor = SMTProgrammerFreeEntityExtractor(engine=engine, microbench_dir=free_entity_out_dir)
        self.free_qid = SMTProgrammerFreeEntityQualifierIdentifier(engine=engine)
        self.toplevel_filter = SMTProgrammerTopLevelEntityFilter(engine=engine, microbench_dir=entity_filter_out_dir)
        self.enricher = SMTProgrammerEntityEnricher(persist_path=entity_enricher_out_dir)

        self.stop_after_preproc = True
    # ← helpers ----------------------------------------------------
    def _canon_stop(self, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        key = str(val).strip().lower()
        return self._STOP_ALIASES.get(key, key)

    def _maybe_early_return(self, stage: str, context: dict, reporter=None, attempt: Optional[int] = None) -> bool:
        if self.stop_after in (None, "program"):
            return False
        if self.stop_after != stage:
            return False

        context["early_stop_stage"] = stage
        rid = context.get("current_requirement_index", -1)
        try:
            _log("EARLY-STOP", rid, f"at stage: {stage}")
        except Exception:
            pass

        if reporter is not None:
            try:
                reporter.finalize(success=False, attempt=attempt or 1)
                reporter.write()
            except Exception:
                pass

        if self.persist_on_stop:
            try:
                context = self.declaration_parser(context)
            except Exception:
                pass
            try:
                self._persist(context)
            except Exception:
                pass

        return True

    def _should_bypass_verifier(self, attempt: int) -> bool:
        pol = self.verifier_fail_open_policy
        if pol == "always":
            return True
        if pol in {"eventual", "after"}:
            threshold = self.verifier_fail_open_after_attempts or self.max_verifier_retries
            return attempt >= threshold
        return False

    def _append_bypass_header(self, context: dict, attempt: int) -> None:
        hdr = (
            f";; --- verifier-bypassed (attempt {attempt}/{self.max_verifier_retries}) "
            f"{datetime.datetime.now().isoformat()} policy={self.verifier_fail_open_policy}"
        )
        context.setdefault("new_smt_lines", [])
        context["new_smt_lines"] = [hdr] + context["new_smt_lines"]

    def _run_free_prepass(self, context: dict) -> dict:
        if not context.get("requirements"):
            context["free_prepass_done"] = True
            return context

        if context.get("enable_free_prepass") is not None:
            self.enable_free_prepass = bool(context["enable_free_prepass"])

        context = self.free_extractor.forward(context)
        context = self.free_qid.forward(context)
        context = self.toplevel_filter.forward(context)

        context["free_prepass_done"] = True
        return context

    def _translator_top_p_for_attempt(self, attempt: int) -> float:
        idx = min(max(attempt, 1) - 1, len(self.translator_top_p_schedule) - 1)
        return float(self.translator_top_p_schedule[idx])

    # ── profiling helpers ────────────────────────────────────────
    def _prof_mark(
        self,
        *,
        stage: str,
        t0: float,
        req_idx: Optional[int] = None,
        attempt: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        try:
            ev = {
                "ts": datetime.datetime.now().isoformat(),
                "stage": str(stage),
                "elapsed_s": float(time.perf_counter() - t0),
                "req_idx": None if req_idx is None else int(req_idx),
                "attempt": None if attempt is None else int(attempt),
            }
            if extra:
                ev.update(extra)
            self._prof_events.append(ev)
        except Exception:
            pass

    def _persist_profile(self, context: dict) -> None:
        try:
            self.profile_out_dir.mkdir(parents=True, exist_ok=True)
            trial_id = context.get("trial_id", "unknown_trial")
            inc_exc = context.get("inc_exc", "unknown_side")
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            base = f"{trial_id}_{inc_exc}_profile"
            json_path_latest = self.profile_out_dir / f"{base}.json"
            json_path_stamped = self.profile_out_dir / f"{base}_{stamp}.json"
            payload = {
                "trial_id": trial_id,
                "side": inc_exc,
                "generated_at": datetime.datetime.now().isoformat(),
                "events": self._prof_events,
            }
            txt = json.dumps(payload, indent=2, ensure_ascii=False)
            json_path_latest.write_text(txt, encoding="utf-8")
            json_path_stamped.write_text(txt, encoding="utf-8")

            csv_path_latest = self.profile_out_dir / f"{base}.csv"
            csv_path_stamped = self.profile_out_dir / f"{base}_{stamp}.csv"
            header = ["ts", "stage", "elapsed_s", "req_idx", "attempt"]
            def _row(e: Dict[str, Any]) -> str:
                return ",".join([
                    str(e.get("ts","")),
                    str(e.get("stage","")).replace(",", " "),
                    f"{float(e.get('elapsed_s', 0.0)):.6f}",
                    "" if e.get("req_idx") is None else str(e.get("req_idx")),
                    "" if e.get("attempt") is None else str(e.get("attempt")),
                ])
            body = "\n".join([",".join(header)] + [_row(e) for e in self._prof_events])
            csv_path_latest.write_text(body, encoding="utf-8")
            csv_path_stamped.write_text(body, encoding="utf-8")

            context["profile_events"] = list(self._prof_events)
        except Exception as e:
            try:
                _log("profile persist ⚠", context.get("current_requirement_index", -1), str(e))
            except Exception:
                pass
    def _persist(self, context: dict) -> None:
        trial_id = context.get("trial_id", "unknown_trial")
        inc_exc = context["inc_exc"]

        if self.smt_out_dir:
            self.smt_out_dir.mkdir(parents=True, exist_ok=True)
            smt_path = self.smt_out_dir / f"{trial_id}_{inc_exc}_program.smt2"
            program_lines = _dedup_program_lines_full(context.get("smt_program_lines", []))
            smt_path.write_text("\n".join(program_lines), encoding="utf-8")

        if self.var_out_dir:
            self.var_out_dir.mkdir(parents=True, exist_ok=True)
            var_idx = context.get("variable_index", {})
            var_path = self.var_out_dir / f"{trial_id}_{inc_exc}_variable_index.json"
            var_path.write_text(json.dumps(var_idx, indent=2, ensure_ascii=False), encoding="utf-8")

        if self.canon_out_dir:
            self.canon_out_dir.mkdir(parents=True, exist_ok=True)
            linked = _collect_all_linked_variables_for_persist(context)
            canvars_path = self.canon_out_dir / f"{trial_id}_{inc_exc}_canonical_variables.json"
            canvars_payload = {"canonical_variables": linked}
            canvars_path.write_text(json.dumps(canvars_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        if self.ent_out_dir and "requirement_bundles" in context:
            self.ent_out_dir.mkdir(parents=True, exist_ok=True)
            ent_path = self.ent_out_dir / f"{trial_id}_{inc_exc}_entities.json"
            ent_path.write_text(
                json.dumps(context["requirement_bundles"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if self.requirements_out_dir and "requirements" in context:
            self.requirements_out_dir.mkdir(parents=True, exist_ok=True)
            reqs_path = self.requirements_out_dir / f"{trial_id}_{inc_exc}_requirements.json"
            reqs_path.write_text(
                json.dumps(context.get("requirements", []), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def forward(self, context: dict) -> dict:  # type: ignore[override]
        self._prof_events = []
        t0_run = time.perf_counter()

        if self.max_reqs and "requirements" in context:
            context["requirements"] = context["requirements"][: self.max_reqs]

        if context.get("enable_free_prepass") is not None:
            self.enable_free_prepass = bool(context["enable_free_prepass"])
        if context.get("qualifiers_only_prepass") is not None:
            self.qualifiers_only_prepass = bool(context["qualifiers_only_prepass"])

        if self.enable_free_prepass and not context.get("free_prepass_done"):
            t0_free = time.perf_counter()
            context = self._run_free_prepass(context)
            self._prof_mark(stage="free_prepass", t0=t0_free, req_idx=None, attempt=None)

            t0_enrich = time.perf_counter()
            context = self.enricher.enrich_ctx(
                context,
                src_key="requirements_entities_attributes_top_level",
                valid_key="valid_entities_by_req",
                mapping_key="preferred_term_canonical_form_map",
            )
            self._prof_mark(stage="entity_enricher", t0=t0_enrich, req_idx=None, attempt=None, extra={"substep": "enrich_ctx"})

            if self.qualifiers_only_prepass or self._maybe_early_return("free_prepass", context):
                if self.persist_on_stop:
                    try: self._persist(context)
                    except Exception: pass
                self._persist_profile(context)
                return context

        # =========================================================
        # INCREMENTAL MODE
        # =========================================================
        for i, _ in enumerate(context["requirements"]):
            context["current_requirement_index"] = i
            t0_req = time.perf_counter()

            # reset per-req scratch keys
            for k in (
                "new_variable_declarations",
                "new_age_sex_pregnancystatus_declarations",
                "new_canonical_variable_declarations",
                "all_other_variable_declarations",
                "new_noncanonical_variable_declarations",
                "new_remaining_variable_declarations",
                "entity_qualifier_pairs",
                "already_declared_variables",
                "new_smt_lines",
                "translator_error",
                "verifier_ok",
                "verifier_report",
                "reuse_errors",
                "noncanonical_errors",
                "errors",
                "namer_fallback",
                "grouped_variable_declarations_by_entity",
                "new_variable_declarations_min",
                "age_sex_preg_errors",
                "age_sex_preg_stage",
                "reusable_variables",
                "translator_retry_hint_from_solver",
            ):
                context.pop(k, None)

            reporter = IntegratedRequirementReporter(context, i, self.report_dir)

            # ===== attempt loop with catch-all =====
            for attempt in range(1, self.max_verifier_retries + 1):
                context["draft_attempt_index"] = attempt
                context["draft_attempt_tag"] = f"draft{attempt:02d}"

                rollback = snapshot_program_state(context)
                context.setdefault("smt_program_lines", [])

                try:
                    # 0a) REUSE
                    t0 = time.perf_counter()
                    context = self.reuser.forward(context)
                    self._prof_mark(stage="reuser", t0=t0, req_idx=i, attempt=attempt)
                    _log("reuse ✓", i, f"(draft {attempt})")
                    try: reporter.capture_reuse()
                    except Exception: pass
                    if self._maybe_early_return("reuse", context, reporter, attempt):
                        continue

                    # 0b) DEMOG
                    t0 = time.perf_counter()
                    context = self.demographics_namer.forward(context)
                    self._prof_mark(stage="demographics_namer", t0=t0, req_idx=i, attempt=attempt)
                    try: reporter.capture_demographics()
                    except Exception: pass
                    if self._maybe_early_return("demographics", context, reporter, attempt):
                        continue

                    # 0c) CANON
                    t0 = time.perf_counter()
                    context = self.canonical_namer.forward(context)
                    self._prof_mark(stage="canonical_namer", t0=t0, req_idx=i, attempt=attempt)
                    try: reporter.capture_canonical()
                    except Exception: pass
                    if self._maybe_early_return("canonical", context, reporter, attempt):
                        continue

                    # 0d) FREE / NON-CAN
                    context["entity_qualifier_pairs"] = _seed_entity_qualifier_pairs(context, i)
                    context["already_declared_variables"] = _dump_already_declared_for_prompt(context)
                    t0 = time.perf_counter()
                    context = self.free_namer.forward(context)
                    self._prof_mark(stage="free_namer", t0=t0, req_idx=i, attempt=attempt)
                    _log("free namer ✓", i, f"(draft {attempt})")
                    try: reporter.capture_free()
                    except Exception: pass
                    if self._maybe_early_return("free", context, reporter, attempt):
                        continue

                    # build grouped → minimal declarations for translator
                    try:
                        context["grouped_variable_declarations_by_entity"] = _build_entity_groups_for_current(context)
                        _log("grouped", i, f"entities={len(context['grouped_variable_declarations_by_entity'])}")
                    except Exception as e:
                        _log("grouped ⚠", i, f"failed to build groups: {e}")

                    try:
                        minimal_for_translator = _groups_to_min_for_translator(
                            context.get("grouped_variable_declarations_by_entity", [])
                        )
                        context["new_variable_declarations_min"] = minimal_for_translator
                        context["new_variable_declarations"] = minimal_for_translator
                        _log("translator-input", i, f"min_items={len(minimal_for_translator)}")
                    except Exception as e:
                        _log("translator-input ⚠", i, f"failed to build minimal declarations: {e}")

                    # 1) TRANSLATOR (with per-attempt top_p)
                    _orig_top_p = None
                    _tp = None
                    try:
                        if hasattr(self.engine, "kwargs"):
                            _orig_top_p = self.engine.kwargs.get("top_p", None)
                            _tp = self._translator_top_p_for_attempt(attempt)
                            self.engine.kwargs["top_p"] = _tp
                            _log("translator top_p", i, f"attempt {attempt} → top_p={_tp}")

                        t0 = time.perf_counter()
                        context = self.translator.forward(context)
                    finally:
                        if hasattr(self.engine, "kwargs") and _orig_top_p is not None:
                            self.engine.kwargs["top_p"] = _orig_top_p
                    self._prof_mark(stage="translator", t0=t0, req_idx=i, attempt=attempt, extra={"top_p": _tp})
                    _log("translate ✓", i, f"(draft {attempt})")
                    try: reporter.capture_translator()
                    except Exception: pass
                    if self._maybe_early_return("translator", context, reporter, attempt):
                        continue

                    # 2) SOLVER
                    t0 = time.perf_counter()
                    context = self.solver_validator.forward(context)
                    self._prof_mark(stage="solver_validator", t0=t0, req_idx=i, attempt=attempt)
                    sat_status = context["solver_check"]["status"]
                    _log("solver check ✓", i, f"→ {sat_status} (draft {attempt})")
                    try: reporter.capture_solver()
                    except Exception: pass
                    if self._maybe_early_return("solver", context, reporter, attempt):
                        continue

                    if sat_status != "sat":
                        # Hint translator for next attempt
                        context["translator_retry_hint_from_solver"] = {
                            "status": sat_status,
                            "message": context["solver_check"].get("message", ""),
                            "static_fixes": context["solver_check"].get("static_fixes", []),
                        }

                        # rollback + reset solver
                        print(f"[solver ❌ draft {attempt}] rolling back slice")
                        context.update(rollback)
                        if hasattr(self.solver_validator, "solver"):
                            new_solver = z3.Solver()
                            new_solver.set(unsat_core=True)
                            self.solver_validator.solver = new_solver
                            if context.get("smt_program_lines"):
                                context["smt_program_lines"] = _dedup_program_lines_full(context["smt_program_lines"])
                                try:
                                    new_solver.from_string("\n".join(context["smt_program_lines"]))
                                except z3.Z3Exception as e:
                                    _log("solver reset ⚠", i, f"ignored parse error during reload: {e}")

                        self._prof_mark(stage="rollback_after_solver_fail", t0=time.perf_counter(), req_idx=i, attempt=attempt)
                        time.sleep(min(0.05 * (2 ** (attempt - 1)), 0.5))  # small backoff
                        if attempt == self.max_verifier_retries:
                            reporter.finalize(success=False, attempt=attempt); reporter.write()
                            # persist artifacts even on failure
                            try: self._persist(context)
                            except Exception: pass
                            self._persist_profile(context)
                            raise RuntimeError(
                                f"Requirement #{i} failed after {self.max_verifier_retries} attempts "
                                f"(last solver status: {sat_status})"
                            )
                        continue  # next attempt (re-translate)

                    # 3) VERIFIER
                    _stage_req_block(context, req_idx=i)
                    t0 = time.perf_counter()
                    context = self.verifier.forward(context)
                    self._prof_mark(stage="verifier", t0=t0, req_idx=i, attempt=attempt)

                    if context.get("verifier_ok"):
                        _log("verifier ✓", i, f"(draft {attempt})")
                        _commit_slice(context, req_idx=i)
                        _assert_all_symbols_declared(context)
                        try:
                            reporter.finalize(success=True, attempt=attempt)
                            reporter.write()
                        except Exception:
                            pass
                        if self._maybe_early_return("verifier", context, reporter, attempt):
                            return context
                        break  # next requirement

                    # Fail-open eventual?
                    if self._should_bypass_verifier(attempt):
                        self._append_bypass_header(context, attempt)
                        _commit_slice(context, req_idx=i)
                        _assert_all_symbols_declared(context)
                        context.setdefault("verifier_bypassed_reqs", []).append(i)
                        context["verifier_bypassed"] = True
                        _log("verifier ⚠ bypass", i, f"(fallback, draft {attempt})")
                        try:
                            reporter.finalize(success=True, attempt=attempt)
                            reporter.write()
                        except Exception:
                            pass
                        if self._maybe_early_return("verifier", context, reporter, attempt):
                            return context
                        break

                    # verifier failed → rollback & retry
                    print(f"[verifier ❌ draft {attempt}] rolling back slice")
                    context.update(rollback)
                    if hasattr(self.solver_validator, "solver"):
                        self.solver_validator.solver = z3.Solver()
                        self.solver_validator.solver.set(unsat_core=True)

                    self._prof_mark(stage="rollback_after_verifier_fail", t0=time.perf_counter(), req_idx=i, attempt=attempt)
                    if attempt == self.max_verifier_retries:
                        try:
                            reporter.finalize(success=False, attempt=attempt)
                            reporter.write()
                        except Exception:
                            pass
                        # persist artifacts even on failure
                        try: self._persist(context)
                        except Exception: pass
                        self._persist_profile(context)
                        raise RuntimeError(
                            f"Requirement #{i} failed verifier after "
                            f"{self.max_verifier_retries} attempts:\n"
                            f"{context.get('verifier_report', '<no-report>')}"
                        )
                    time.sleep(min(0.05 * (2 ** (attempt - 1)), 0.5))
                    continue  # next attempt

                except Exception as e:
                    # CATCH-ALL: log, stash, rollback, reset, retry
                    try:
                        _log("attempt ⚠", i, f"unexpected error on draft {attempt}: {type(e).__name__}: {e}")
                    except Exception:
                        pass
                    context.setdefault("errors", []).append({
                        "stage": "attempt",
                        "attempt": attempt,
                        "type": type(e).__name__,
                        "message": str(e),
                    })
                    context.update(rollback)
                    if hasattr(self.solver_validator, "solver"):
                        self.solver_validator.solver = z3.Solver()
                        self.solver_validator.solver.set(unsat_core=True)
                        if context.get("smt_program_lines"):
                            context["smt_program_lines"] = _dedup_program_lines_full(context["smt_program_lines"])
                            try:
                                self.solver_validator.solver.from_string("\n".join(context["smt_program_lines"]))
                            except z3.Z3Exception:
                                pass
                    time.sleep(min(0.05 * (2 ** (attempt - 1)), 0.5))
                    if attempt == self.max_verifier_retries:
                        # persist artifacts before surfacing
                        try: self._persist(context)
                        except Exception: pass
                        self._persist_profile(context)
                        raise

            # end attempt loop

        # ── common post-processing ────────────────────────────────
        t0 = time.perf_counter()
        context = self.declaration_parser(context)
        self._prof_mark(stage="declaration_parser", t0=t0, req_idx=None, attempt=None)
        _print_variable_index(context["variable_index"])
        t0 = time.perf_counter()
        self._persist(context)
        self._prof_mark(stage="persist_artifacts", t0=t0, req_idx=None, attempt=None)
        self._prof_mark(stage="run_total", t0=t0_run, req_idx=None, attempt=None)
        self._persist_profile(context)
        return context