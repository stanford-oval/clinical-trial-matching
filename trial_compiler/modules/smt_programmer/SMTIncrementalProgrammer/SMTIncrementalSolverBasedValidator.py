# smt_validator_force_true.py
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple, Iterable

# ── minimal dspy shim so this remains drop-in even if dspy isn't importable ──
try:
    import dspy  # type: ignore
except Exception:
    class _BareModule:
        def __init__(self, *args, **kwargs): ...
        def forward(self, *args, **kwargs): raise NotImplementedError
    dspy = type("dspy", (), {"Module": _BareModule})()  # type: ignore

import z3
from z3 import Z3Exception


# ╔════════════════════════════════════════════════════════════════╗
# Fallback logger
# ╚════════════════════════════════════════════════════════════════╝
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [validator] %(message)s",
        stream=sys.stderr,
    )
    def _log(stage: str, idx: int | str = "", msg: str = "") -> None:  # type: ignore
        logging.info("%s %s %s", stage, idx, msg)


# ╔════════════════════════════════════════════════════════════════╗
# Regex helpers
# ╚════════════════════════════════════════════════════════════════╝
_TAG_RE = re.compile(r":named\s+([^) \t\n]+)")
_REQNO_FROM_TAG = re.compile(r"^[Rr]([0-9]+)_[A-Z]")  # R7_A0_..., REQ7_COMPONENT0_...
_ASSERT_NAMED_RE_TMPL = r"\(\s*assert\b[\s\S]*?:named\s+{TAG}\s*\)\s*\)"


def _collect_named_tags(program: str) -> Dict[str, int]:
    """Map :named tag → requirement-index for UNSAT-core attribution."""
    out: Dict[str, int] = {}
    for m in _TAG_RE.finditer(program):
        tag = m.group(1)
        m2 = _REQNO_FROM_TAG.match(tag)
        if m2:
            try:
                out[tag] = int(m2.group(1))
            except Exception:
                pass
    return out


def _tags_for_req(program_text: str, req_idx: int) -> List[str]:
    tags: List[str] = []
    for m in _TAG_RE.finditer(program_text):
        tag = m.group(1)
        m2 = _REQNO_FROM_TAG.match(tag)
        if m2 and int(m2.group(1)) == req_idx:
            tags.append(tag)
    return tags


class SMTIncrementalSolverBasedValidator(dspy.Module):  # type: ignore
    """
    Incrementally validates SMT-LIB snippets; supports:
      • Static parse fixes (dedup declarations, balance parens, opt. auto-declare unknowns).
      • UNSAT-core capture.
      • Fail-open pruning (optional).
      • NEW: Per-requirement "force TRUE" — rewrite asserts to `(assert (! true :named TAG))`
              for specified requirement indices to preserve tags but vacuously satisfy.

    Key flags:
      - force_true_on_unresolved_reqs: set[int]
          If the final result after retries is not SAT, and the current requirement
          index is in this set, the validator will rewrite ALL named assertions for that
          requirement to TRUE, re-check, and persist the change.
      - exclude_on_unresolved: bool
          If still unresolved (or for reqs not in the above set), prune minimal culprits
          (unsat-core or unknown-symbol carriers), falling back to dropping the slice.

    Both behaviors can co-exist; the force-TRUE step runs first for configured req indices.
    """

    def __init__(
        self,
        engine=None,
        *,
        produce_model: bool = False,
        produce_unsat_core: bool = True,
        max_registry_attempts: int = 0,
        refiner_mode: str = "naive",
        validator_log_dir: str | None = "./validator_logs",
        static_fix_enabled: bool = True,
        static_fix_auto_declare_unknowns: bool = False,  # default OFF
        static_fix_max_rounds: int = 2,
        dump_on_error: bool = False,
        dump_initial_program: bool = True,
        raise_on_unresolved: bool = False,
        # NEW behavior controls
        force_true_on_unresolved_reqs: Iterable[int] | None = None,
        exclude_on_unresolved: bool = True,
        max_exclusion_rounds: int = 3,
    ):
        super().__init__()
        self.engine = engine
        self.produce_model = produce_model
        self.produce_unsat_core = produce_unsat_core
        self.max_registry_attempts = max_registry_attempts
        self.refiner_mode = refiner_mode

        self.solver = z3.Solver()
        if produce_unsat_core:
            self.solver.set(unsat_core=True)

        self._prov: Dict[str, int] = {}
        self.log_dir = validator_log_dir or "./validator_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.dump_on_error = dump_on_error
        self.dump_initial_program = dump_initial_program
        self.raise_on_unresolved = raise_on_unresolved

        self.static_fix_enabled = static_fix_enabled
        self.static_fix_auto_declare_unknowns = static_fix_auto_declare_unknowns
        self.static_fix_max_rounds = static_fix_max_rounds

        # new
        self.force_true_on_unresolved_reqs = set(force_true_on_unresolved_reqs or [])
        self.exclude_on_unresolved = exclude_on_unresolved
        self.max_exclusion_rounds = max_exclusion_rounds

    # ── logging helper ─────────────────────────────────────────────
    def _dump_run(self, out_dir: str, stem: str, program_text: str, result: Dict[str, Any]) -> None:
        os.makedirs(out_dir, exist_ok=True)
        prog_path = os.path.join(out_dir, f"{'program' if stem=='program' else stem}_program.smt2")
        res_path  = os.path.join(out_dir, f"{'result' if stem=='program' else stem+'_result'}.json")
        with open(prog_path, "w", encoding="utf-8") as fh:
            fh.write(program_text)
        with open(res_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    # ── static-fix helpers (parse-only) ────────────────────────────
    _DECL_RE = re.compile(r"\(\s*declare-(?:const|fun)\s+([^\s()]+)")
    _UNKNOWN_RE = re.compile(r"unknown (?:constant|function)\s+'([^']+)'", re.IGNORECASE)

    def _existing_declared_symbols(self, text: str) -> set:
        return set(self._DECL_RE.findall(text))

    def _dedup_declarations(self, text: str) -> Tuple[str, List[str]]:
        seen = set()
        removed: List[str] = []
        out_lines: List[str] = []
        for ln in text.splitlines():
            m = self._DECL_RE.search(ln)
            if m:
                sym = m.group(1)
                if sym in seen:
                    removed.append(ln)
                    continue
                seen.add(sym)
            out_lines.append(ln)
        return "\n".join(out_lines), removed

    def _balance_parens(self, text: str) -> Tuple[str, int]:
        opens = text.count("(")
        closes = text.count(")")
        if opens > closes:
            missing = opens - closes
            return text + ("\n" + (")" * missing)), missing
        return text, 0

    def _find_unknowns_from_msg(self, msg: str) -> List[str]:
        return list(dict.fromkeys(self._UNKNOWN_RE.findall(msg)))

    def _guess_sort(self, name: str, text: str) -> str:
        usage_lines = [ln for ln in text.splitlines() if name in ln]
        joined = "\n".join(usage_lines)
        if re.search(rf"\(\s*assert\s+{re.escape(name)}\s*\)", joined): return "Bool"
        if re.search(rf"\(\s*not\s+{re.escape(name)}\s*\)", joined): return "Bool"
        if re.search(rf"\(\s*(?:and|or|=>)\s+[^)]*\b{re.escape(name)}\b", joined): return "Bool"
        if re.search(rf"\(\s*=\s+{re.escape(name)}\s+(?:true|false)\s*\)", joined, re.IGNORECASE): return "Bool"
        if re.search(rf"[()\s](?:[+\-*/<>]=?|=)\s+{re.escape(name)}[()\s]", joined) or \
           re.search(rf"{re.escape(name)}\s+(?:[+\-*/<>]=?|=)", joined) or \
           re.search(rf"\(\s*(?:<|<=|>|>=|=)\s+{re.escape(name)}\s+[^)]+\)", joined):
            if re.search(r"\d+\.\d", joined) or "/" in joined: return "Real"
            if any(tok in name for tok in ["count","number","grade","score","age","years"]): return "Int"
            if any(tok in name for tok in ["value","ratio","temperature","pressure","percent","percentage"]): return "Real"
            return "Int"
        if re.match(r"(has_|is_|can_|was_|does_|eligible_|pregnant|male_|female_)", name): return "Bool"
        if any(suf in name for suf in ["_now","_ever","_inthepast","_present","_absent"]): return "Bool"
        if any(tok in name for tok in ["value","ratio","temperature","pressure","percent","percentage"]): return "Real"
        if any(tok in name for tok in ["count","number","grade","score","age","years"]): return "Int"
        return "Bool"

    def _build_decl(self, name: str, sort: str) -> str:
        return f"(declare-const {name} {sort})"

    def _apply_static_fixes_once(self, program_text: str, err_msg: str) -> Tuple[str, Dict[str, Any]]:
        fixes: Dict[str, Any] = {
            "removed_duplicate_decls": [],
            "added_decls": [],
            "added_closing_parens": 0,
            "unknowns_targeted": [],
        }
        deduped, removed = self._dedup_declarations(program_text)
        if removed:
            fixes["removed_duplicate_decls"] = removed
        program_text = deduped

        if self.static_fix_auto_declare_unknowns:
            unknowns = [u for u in self._find_unknowns_from_msg(err_msg) if u.lower() != "null"]
            if unknowns:
                fixes["unknowns_targeted"] = unknowns
                declared = self._existing_declared_symbols(program_text)
                decls: List[str] = []
                for sym in unknowns:
                    if sym in declared or sym in {"true","false"}:
                        continue
                    if "(" in sym or ")" in sym:
                        continue
                    sort = self._guess_sort(sym, program_text)
                    decls.append(self._build_decl(sym, sort))
                if decls:
                    fixes["added_decls"] = decls
                    program_text = "\n".join(decls) + "\n" + program_text

        balanced, added = self._balance_parens(program_text)
        if added:
            fixes["added_closing_parens"] = added
        program_text = balanced
        return program_text, fixes

    # ── solver core ────────────────────────────────────────────────
    def _run_solver(self, program_text: str) -> Tuple[Dict[str, Any], z3.Solver | None]:
        def _attempt(text: str) -> Tuple[Dict[str, Any], z3.Solver | None, str, Dict[str, Any]]:
            ctx = z3.Context()
            s = z3.Solver(ctx=ctx)
            try:
                s.from_string(text)
            except Z3Exception as e:
                return (
                    {"status":"error","message":str(e),"model":None,"unsat_core":None,"stats":{}},
                    None, text, {}
                )
            t0 = time.time()
            res = s.check()
            elapsed = time.time() - t0
            out = {
                "status": {1:"sat",-1:"unsat",0:"unknown"}[res.r],
                "message": "" if res.r != 0 else s.reason_unknown(),
                "model": str(s.model()) if res.r == 1 and self.produce_model else None,
                "unsat_core": None,
                "stats": {k: s.statistics().get_key_value(k) for k in s.statistics().keys()},
            }
            out["stats"]["elapsed_sec"] = round(elapsed, 4)
            if res.r == -1 and self.produce_unsat_core:
                out["unsat_core"] = list(map(str, s.unsat_core()))
            return out, s, text, {}

        result, solver, current_text, _ = _attempt(program_text)
        all_fixes: List[Dict[str, Any]] = []

        rounds = 0
        while self.static_fix_enabled and result["status"] == "error" and rounds < self.static_fix_max_rounds:
            rounds += 1
            fixed_text, fixes = self._apply_static_fixes_once(current_text, result.get("message",""))
            if fixed_text == current_text:
                break
            all_fixes.append(fixes)
            result, solver, current_text, _ = _attempt(fixed_text)

        if all_fixes:
            result["static_fixes"] = all_fixes
        result["program_text"] = current_text
        return result, solver

    # ── build + check whole program ────────────────────────────────
    def _check_program(self, committed: List[str], fresh: List[str], req_idx: int) -> Dict[str, Any]:
        program_text = "\n".join(committed + fresh)
        self._prov.update(_collect_named_tags(program_text))
        result, _ = self._run_solver(program_text)
        result.update(tagged_slice=fresh)
        return result

    # ── persist any auto-added decls ───────────────────────────────
    def _persist_added_decls(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        fixes = result.get("static_fixes") or []
        if not fixes:
            return
        added: List[str] = []
        for round_fix in fixes:
            added.extend(round_fix.get("added_decls", []))
        if not added:
            return
        committed = context.setdefault("smt_program_lines", [])
        existing_text = "\n".join(committed)
        to_add = [d for d in added if d not in existing_text]
        if to_add:
            context["smt_program_lines"] = to_add + committed  # prepend so uses follow

    # ── text transforms: remove or force-TRUE ──────────────────────
    def _remove_named_assertions(self, text: str, tags: List[str]) -> Tuple[str, List[str]]:
        removed = []
        for tag in tags:
            pat = _ASSERT_NAMED_RE_TMPL.format(TAG=re.escape(tag))
            new_text, n = re.subn(pat, "", text, flags=re.DOTALL)
            if n > 0:
                removed.append(tag)
                text = new_text
        return text, removed

    def _rewrite_named_assertions_to_true(self, text: str, tags: List[str]) -> Tuple[str, List[str]]:
        """
        Replace the whole assert block by `(assert (! true :named TAG))`
        for each provided tag. Returns (new_text, actually_rewritten_tags).
        """
        rewritten: List[str] = []
        for tag in tags:
            pat = _ASSERT_NAMED_RE_TMPL.format(TAG=re.escape(tag))
            def _repl(_m):
                return f"(assert (! true :named {tag}))"
            new_text, n = re.subn(pat, _repl, text, flags=re.DOTALL)
            if n > 0:
                text = new_text
                rewritten.append(tag)
        return text, rewritten

    def _guess_error_symbols(self, message: str) -> List[str]:
        names = self._find_unknowns_from_msg(message)
        m = re.search(r"unknown .*?'([^']+)'", message, re.IGNORECASE)
        if m:
            names.append(m.group(1))
        out, seen = [], set()
        for n in names:
            if n not in seen:
                out.append(n); seen.add(n)
        return out

    def _remove_asserts_referencing_symbols(self, text: str, symbols: List[str]) -> Tuple[str, List[str]]:
        tags = re.findall(r":named\s+([^) \t\n]+)", text)
        to_drop = []
        for tag in tags:
            pat = _ASSERT_NAMED_RE_TMPL.format(TAG=re.escape(tag))
            m = re.search(pat, text, flags=re.DOTALL)
            if not m:
                continue
            block = m.group(0)
            if any(sym in block for sym in symbols):
                to_drop.append(tag)
        return self._remove_named_assertions(text, to_drop)

    # ── main entry ─────────────────────────────────────────────────
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        fresh_slice = context.get("new_smt_lines", [])
        req_idx     = context.get("current_requirement_index")
        committed   = context.setdefault("smt_program_lines", [])
        attempts    = context.setdefault("solver_check_attempts", [])

        if fresh_slice is None or req_idx is None:
            raise ValueError("Validator expects new_smt_lines and current_requirement_index")

        trial = context.get("trial_id", "unknown_trial")
        side  = context.get("inc_exc", "unknown")

        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc","inclusion","include","in"}:  return "inclusion"
            if s in {"exc","exclusion","exclude","ex"}:  return "exclusion"
            return s or "unknown"

        req_dir = os.path.join(self.log_dir, str(trial), _bucket(side), f"req{req_idx:03d}")

        # ---------- initial run (quiet parse-error retries) ----------
        attempt = 0
        result = self._check_program(committed, fresh_slice, req_idx)
        self._persist_added_decls(context, result)

        if (result["status"] != "error" and self.dump_initial_program) or (result["status"] == "error" and self.dump_on_error):
            self._dump_run(req_dir, "program", result["program_text"], result)

        _log("solver", req_idx, f"attempt {attempt} → {result['status']}")
        attempts.append({"phase":"registry","attempt":attempt, **{k:v for k,v in result.items() if k!="program_text"}})

        # ---------- refinement loop (disabled by default) ----------
        while result["status"] != "sat" and attempt < self.max_registry_attempts:
            attempt += 1
            committed = context["smt_program_lines"]
            fresh_slice = [ln for ln in committed if f"REQ{req_idx}_" in ln]
            result = self._check_program(committed, [], req_idx)
            self._persist_added_decls(context, result)
            if (result["status"] != "error") or self.dump_on_error:
                self._dump_run(req_dir, f"retry{attempt:02d}", result["program_text"], result)
            _log("solver", req_idx, f"attempt {attempt} → {result['status']}")
            attempts.append({"phase":"registry","attempt":attempt, **{k:v for k,v in result.items() if k!="program_text"}})

        # ---------- bookkeeping ----------
        context.update(solver_check=result, new_smt_lines=fresh_slice)
        if result["status"] == "sat":
            context["solver_ok"] = True
            return context

        # ---------- FORCE-TRUE path (per requirement idx) ----------
        program_text = "\n".join(context.get("smt_program_lines", []))
        context["solver_ok"] = False

        forced_true_tags_total: List[str] = context.setdefault("forced_true_tags", [])
        if req_idx in self.force_true_on_unresolved_reqs:
            # Rewrite ALL named asserts for this requirement to TRUE (preserve :named)
            req_tags = _tags_for_req(program_text, req_idx)
            if req_tags:
                program_text, rewritten = self._rewrite_named_assertions_to_true(program_text, req_tags)
                if rewritten:
                    context["smt_program_lines"] = [ln for ln in program_text.splitlines() if ln.strip()]
                    forced_true_tags_total.extend(rewritten)

                    fr_result, _ = self._run_solver(program_text)
                    context["solver_check"] = fr_result
                    if (fr_result["status"] != "error") or self.dump_on_error:
                        self._dump_run(req_dir, "forced_true", fr_result["program_text"], fr_result)

                    if fr_result["status"] == "sat":
                        context["solver_ok"] = True
                        return context
                    # else: fall through to fail-open pruning if enabled

        # ---------- FAIL-OPEN pruning (optional) ----------
        if not self.exclude_on_unresolved:
            if self.raise_on_unresolved:
                core = context["solver_check"].get("unsat_core") or []
                raise RuntimeError(
                    f"[SMTValidator] requirement {req_idx} unresolved.\n"
                    f"status: {context['solver_check']['status']} message: {context['solver_check'].get('message','')}\n"
                    f"core: {core}"
                )
            return context

        excluded: List[str] = context.setdefault("excluded_named_tags", [])
        attempts_info: List[Dict[str, Any]] = []

        for round_idx in range(self.max_exclusion_rounds):
            status = context["solver_check"]["status"]
            msg    = context["solver_check"].get("message", "")
            core   = context["solver_check"].get("unsat_core") or []

            tags_to_drop: List[str] = []

            if status == "unsat" and core:
                # Minimal conflicting set: drop exactly these tags
                tags_to_drop = [t for t in core if _REQNO_FROM_TAG.match(t)]
            elif status == "error" and msg:
                # Parse errors: drop named asserts that reference unknown symbols
                unknowns = self._guess_error_symbols(msg)
                if unknowns:
                    program_text, dropped = self._remove_asserts_referencing_symbols(program_text, unknowns)
                    tags_to_drop = dropped

            # Fallback: if nothing specific, drop the current requirement slice's named tags
            if not tags_to_drop:
                tags_to_drop = _tags_for_req(program_text, req_idx)

            if not tags_to_drop:
                attempts_info.append({"round": round_idx, "dropped": [], "result": status, "note": "no-identifiable-tags"})
                break

            program_text, actually_removed = self._remove_named_assertions(program_text, tags_to_drop)
            if not actually_removed:
                attempts_info.append({"round": round_idx, "dropped": [], "result": status, "note": "pattern-miss"})
                break

            excluded.extend(actually_removed)
            context["smt_program_lines"] = [ln for ln in program_text.splitlines() if ln.strip()]

            pruned_result, _ = self._run_solver(program_text)
            attempts_info.append({"round": round_idx, "dropped": actually_removed, "result": pruned_result["status"]})
            context["solver_check"] = pruned_result

            if (pruned_result["status"] != "error") or self.dump_on_error:
                self._dump_run(req_dir, f"pruned_round{round_idx:02d}", pruned_result["program_text"], pruned_result)

            if pruned_result["status"] == "sat":
                context["solver_ok"] = True
                break

        context["exclusion_attempts"] = attempts_info

        if context["solver_ok"] is not True and self.raise_on_unresolved:
            core = context["solver_check"].get("unsat_core") or []
            raise RuntimeError(
                f"[SMTValidator] requirement {req_idx} unresolved after force-true/pruning\n"
                f"status: {context['solver_check']['status']} message: {context['solver_check'].get('message','')}\n"
                f"core: {core}\n"
                f"forced_true_tags: {forced_true_tags_total}\n"
                f"excluded: {excluded}"
            )

        return context


# ─────────────────────────────────────────────────────────────────────
# Example usage (optional)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo: REQ 8 will be forced to TRUE if unresolved.
    validator = SMTIncrementalSolverBasedValidator(
        engine=None,
        produce_model=False,
        produce_unsat_core=True,
        static_fix_enabled=True,
        static_fix_auto_declare_unknowns=False,  # keep OFF to showcase force-true path
        dump_on_error=True,
        force_true_on_unresolved_reqs={8},      # <— configure which req idx get force-TRUE
        exclude_on_unresolved=True,             # keep pruning as a fallback
        max_exclusion_rounds=3,
    )

    committed = [
        '(declare-const patient_is_caregiver_of_veteran Bool)',
        '(declare-const patient_is_english_speaking Bool)',
    ]
    fresh = [
        '(assert (! patient_is_caregiver_of_veteran :named REQ8_COMPONENT0_PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE))',
        '(assert (! patient_is_accessible_by_telephone_for_scheduling :named REQ8_COMPONENT1_NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION))',
    ]
    ctx = {
        "trial_id": "demo_trial",
        "inc_exc": "inc",
        "current_requirement_index": 8,
        "smt_program_lines": committed.copy(),
        "new_smt_lines": fresh,
    }
    out = validator.forward(ctx)
    print(json.dumps({
        "status": out["solver_check"]["status"],
        "solver_ok": out.get("solver_ok"),
        "forced_true_tags": out.get("forced_true_tags", []),
        "excluded_named_tags": out.get("excluded_named_tags", []),
        "exclusion_attempts": out.get("exclusion_attempts", []),
    }, indent=2))