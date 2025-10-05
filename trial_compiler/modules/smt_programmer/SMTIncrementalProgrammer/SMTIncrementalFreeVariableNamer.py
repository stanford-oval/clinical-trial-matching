# modules/SMTIncrementalFreeVariableNamer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re, json, warnings

import dspy

from .namer_checks import _schema_log

# ───────────────────────── fallback logger ─────────────────────────
try:
    from smt_core.utils.z3_helpers import _log  # type: ignore
except Exception:  # pragma: no cover
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    def _log(stage: str, idx: int, msg: str = "") -> None:  # type: ignore
        logging.info("%s %s", stage, msg)

# ───────────────────────── parsing helpers ─────────────────────────

_NEW_BLOCK_RE = re.compile(
    r"<new_variable_declarations>\s*(\[[\s\S]*?\])\s*</new_variable_declarations>",
    re.IGNORECASE
)

def _parse_json_array_relaxed(s: str) -> List[Any]:
    """Trim code fences/backticks and json.loads (no normalization)."""
    s = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.S)
    return json.loads(s)

# ───────────────────────── already-declared (PROMPT ONLY) ─────────────────────────

def _names_from_mixed_list(lst) -> set[str]:
    """
    Accepts a list of strings or dicts. Extracts names from:
      - variable_name
      - entity_variable_name
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
            nm = it.get("variable_name") or it.get("entity_variable_name") or it.get("name")
            if isinstance(nm, str):
                s = nm.strip()
                if s:
                    out.add(s)
    return out

def _collect_already_declared_names(context: Dict[str, Any]) -> set[str]:
    """
    Union for PROMPT's <already_declared_variables>:
      - already_declared_variables
      - reusable_variables
      - new_age_sex_pregnancystatus_declarations (entity_variable_name preferred)
      - new_canonical_variable_declarations     (entity_variable_name preferred)
      - PLUS: composed canonical qualifiers from qualifier_predicates_detailed:
              <entity_variable_name><qualifier_variable_snake_case_form>
    NOTE: intentionally does NOT read smt_program_lines.
    """
    names: set[str] = set()

    # explicit & persisted
    # names |= _names_from_mixed_list(context.get("already_declared_variables"))
    names |= _names_from_mixed_list(context.get("reusable_variables"))

    # demographics
    for it in (context.get("new_age_sex_pregnancystatus_declarations") or []):
        if isinstance(it, dict):
            nm = it.get("entity_variable_name") or it.get("variable_name")
            if isinstance(nm, str) and nm.strip():
                names.add(nm.strip())

    # canonical stems + composed qualifiers (from detailed only)
    for obj in (context.get("new_canonical_variable_declarations") or []):
        if not isinstance(obj, dict):
            continue
        stem = obj.get("entity_variable_name")
        if isinstance(stem, str) and stem.strip():
            stem = stem.strip()
            names.add(stem)  # include the stem itself

            for q in (obj.get("qualifier_predicates_detailed") or []):
                if not isinstance(q, dict):
                    continue
                tail = (q.get("qualifier_variable_snake_case_form") or "").strip()
                if not tail:
                    continue
                # ensure "@@" prefix; do NOT normalize beyond what's provided
                tail = tail if tail.startswith("@@") else f"@@{tail.lstrip('@')}"
                composed = f"{stem}{tail}"
                names.add(composed)

    return names

# ───────── 新增：从混合列表提取 {name -> meaning} ─────────
def _items_from_mixed_list(lst) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    if not isinstance(lst, list):
        return out
    for it in lst:
        if isinstance(it, str):
            nm = it.strip()
            if nm:
                out.setdefault(nm, None)
        elif isinstance(it, dict):
            nm = it.get("variable_name") or it.get("entity_variable_name") or it.get("name")
            if not isinstance(nm, str):
                continue
            nm = nm.strip()
            if not nm:
                continue
            meaning = (
                it.get("variable_meaning")
                or it.get("entity_variable_meaning")
                or it.get("meaning")
                or it.get("semantics")
            )
            if isinstance(meaning, str):
                meaning = meaning.strip()
            # 若已有为空、而新值非空，则覆盖
            if nm not in out or not out[nm]:
                out[nm] = meaning
    return out


def _collect_already_declared_rich(context: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    仅聚合以下三类来源，构造 {name -> {"meaning","vdecl","qdecl"}}：
      - reusable_variables
      - new_age_sex_pregnancystatus_declarations
      - new_canonical_variable_declarations（含其 qualifier_predicates_detailed 合成的 composed 变量）
    不读取 already_declared_variables，不扫描 smt_program_lines。
    """
    out: Dict[str, Dict[str, str]] = {}

    def _name_of(d: dict) -> str:
        return (d.get("variable_name")
                or d.get("entity_variable_name")
                or d.get("name")
                or "").strip()

    def _meaning_of(d: dict) -> Optional[str]:
        m = (d.get("variable_meaning")
             or d.get("entity_variable_meaning")
             or d.get("meaning")
             or d.get("semantics"))
        return m.strip() if isinstance(m, str) else None

    def _qdecl_of(d: dict) -> Optional[str]:
        """
        仅取 qualifier 的声明文本。
        ⚠️ 不再回退到 variable_declaration（按新规：有 qualifier 只用 qualifier declaration）
        """
        if not isinstance(d, dict):
            return None
        for k in ("qualifier_variable_declaration", "qualifier_declaration"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _vdecl_of(d: dict) -> Optional[str]:
        v = d.get("variable_declaration")
        return v.strip() if isinstance(v, str) else None

    # 1) reusable_variables 作为基础（字符串或 dict 都支持）
    for it in (context.get("reusable_variables") or []):
        if isinstance(it, dict):
            _merge_decl_info(out, _name_of(it), _meaning_of(it), _vdecl_of(it), _qdecl_of(it))
        elif isinstance(it, str) and it.strip():
            _merge_decl_info(out, it.strip())

    # 2) demographics（age/sex/pregnancy），用来补全空白字段
    for it in (context.get("new_age_sex_pregnancystatus_declarations") or []):
        if isinstance(it, dict):
            _merge_decl_info(out, _name_of(it), _meaning_of(it), _vdecl_of(it), _qdecl_of(it))

    # 3) canonical stems + 组合 qualifiers（只看 detailed，吃上游 qdecl）
    for obj in (context.get("new_canonical_variable_declarations") or []):
        if not isinstance(obj, dict):
            continue
        stem = _name_of(obj)
        if not stem:
            continue
        stem_meaning = _meaning_of(obj)
        stem_vdecl   = _vdecl_of(obj)
        # stem 本身
        _merge_decl_info(out, stem, stem_meaning, stem_vdecl, None)

        # 合成 qualifier 变量
        for q in (obj.get("qualifier_predicates_detailed") or []):
            if not isinstance(q, dict):
                continue
            tail = (q.get("qualifier_variable_snake_case_form") or "").strip()
            if not tail:
                continue
            tail = tail if tail.startswith("@@") else f"@@{tail.lstrip('@')}"
            composed = f"{stem}{tail}"

            q_gloss = (
                q.get("qualifier_variable_meaning")
                or q.get("qualifier_meaning")
                or q.get("qualifier_human_readable")
                or q.get("human_readable")
                or q.get("gloss")
                or ""
            )
            q_gloss = q_gloss.strip() if isinstance(q_gloss, str) else ""

            if stem_meaning and q_gloss:
                composed_meaning = f"{stem_meaning} WITH {q_gloss}"
            elif stem_meaning:
                composed_meaning = stem_meaning
            else:
                composed_meaning = q_gloss or None

            qdecl = _qdecl_of(q)          # qualifier 的声明（优先 qualifier_variable_declaration）
            vdecl_q = _vdecl_of(q)        # 有些上游把 q 的声明放到了 variable_declaration，也收

            _merge_decl_info(out, composed, composed_meaning, vdecl_q, qdecl)

    return out



def _merge_decl_info(dst: Dict[str, Dict[str, str]],
                     name: str,
                     meaning: Optional[str] = None,
                     vdecl: Optional[str] = None,
                     qdecl: Optional[str] = None) -> None:
    """
    合并同名变量的语义/声明文本：先到先得，后来源仅在目标为空时补全。
    """
    if not isinstance(name, str) or not name.strip():
        return
    name = name.strip()
    rec = dst.setdefault(name, {"meaning": "", "vdecl": "", "qdecl": ""})

    if isinstance(meaning, str):
        meaning = meaning.strip()
        if meaning and not rec["meaning"]:
            rec["meaning"] = meaning

    if isinstance(vdecl, str):
        vdecl = vdecl.strip()
        if vdecl and not rec["vdecl"]:
            rec["vdecl"] = vdecl

    if isinstance(qdecl, str):
        qdecl = qdecl.strip()
        if qdecl and not rec["qdecl"]:
            rec["qdecl"] = qdecl


# ───────── 修改：PROMPT 中实际注入 JSON 的函数 ─────────
def _dump_already_declared_for_prompt(context: Dict[str, Any]) -> str:
    """
    仅导出 reusable + canonical(+其 qualifiers) + demographics。
    规则：
      - 变量名含 '@@' 视为 qualifier 变量 → 只给 'qualifier_variable_declaration'（若有）
      - 否则视为普通变量 → 只给 'variable_declaration'（若有）
      - 始终包含 variable_name, variable_meaning
    """
    rich = _collect_already_declared_rich(context)  # {name: {"meaning","vdecl","qdecl"}}
    arr = []
    for nm in sorted(rich.keys()):
        rec = rich[nm]
        meaning = (rec.get("meaning") or "").strip()
        vdecl   = (rec.get("vdecl")   or "").strip()
        qdecl   = (rec.get("qdecl")   or "").strip()

        is_qual = "@@" in nm
        obj = {
            "variable_name": nm,
            "variable_meaning": meaning,
        }

        if is_qual:
            if qdecl:
                obj["qualifier_variable_declaration"] = qdecl
        else:
            if vdecl:
                obj["variable_declaration"] = vdecl

        arr.append(obj)

    return json.dumps(arr, ensure_ascii=False, indent=2)






# ───────────────────────── module ─────────────────────────

class SMTIncrementalFreeVariableNamer(dspy.Module):
    """
    Free-variable pass (pass-through):
    - Reads ONLY the LLM-emitted <new_variable_declarations> JSON array.
    - Persists EXACTLY that array under context["new_remaining_variable_declarations"].
    - Canonical expansion is applied ONLY to the PROMPT's already-declared list,
      not to outputs or errors.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, engine, *, log_dir: Optional[str] = None, allow_mixed_case: bool = False):
        super().__init__()
        self.engine = engine
        self.log_dir = log_dir or "./namer_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self._allow_mixed_case = allow_mixed_case  # kept for API symmetry; not used

    def _build_prompt(self, context: Dict[str, Any], idx: int) -> str:
        tpl = context.get("SMTIncrementalFreeVariableNamer_prompt", "")
        if not tpl:
            return ""
        req = context["requirements"][idx]
        requirement_txt = req.get("requirement") if isinstance(req, dict) else str(req)

        eqp = json.dumps(context.get("entity_qualifier_pairs", []), ensure_ascii=False, indent=2)
        # IMPORTANT: expansion happens here (PROMPT ONLY)
        already_declared = _dump_already_declared_for_prompt(context)

        return (
            tpl.replace("#REQUIREMENT#", requirement_txt)
               .replace("#ENTITY_QUALIFIER_PAIRS#", eqp)
               .replace("#ALREADY_DECLARED_VARIABLES#", already_declared)
        )

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        # print(f">>>>>>>>>>>>> In SMTIncrementalFreeVariableNamer")
        if not context.get("requirements"):
            return context

        idx: int = int(context["current_requirement_index"])
        trial_id = context.get("trial_id", "unknown_trial")
        side = context.get("inc_exc", "unknown")

        prompt = self._build_prompt(context, idx)
        if not prompt:
            _log("noncanon ✗", idx, "prompt template missing (SMTIncrementalFreeVariableNamer_prompt)")
            context["new_remaining_variable_declarations"] = []
            context["noncanonical_errors"] = [{"code": "PROMPT_MISSING"}]
            context["noncanonical_stage"] = "remaining_noncanonical"
            return context

        # Layout: <log_root>/<trial_id>/<inclusion|exclusion>/reqNNN/
        def _bucket(s: str) -> str:
            s = (s or "").strip().lower()
            if s in {"inc", "inclusion", "include", "in"}:  return "inclusion"
            if s in {"exc", "exclusion", "exclude", "ex"}:  return "exclusion"
            return s or "unknown"

        stage_dir = os.path.join(self.log_dir, str(trial_id), _bucket(side), f"req{idx:03d}")
        os.makedirs(stage_dir, exist_ok=True)
        p_log    = os.path.join(stage_dir, "3noncanon_prompt.txt")
        r_log    = os.path.join(stage_dir, "3noncanon_raw.txt")
        plan_log = os.path.join(stage_dir, "3noncanon_plan.json")

        # Write prompt for debugging
        try:
            with open(p_log, "w", encoding="utf-8") as fh:
                fh.write(prompt)
        except Exception as e:
            _log("noncanon ⚠", idx, f"failed to write prompt: {e}")

        errors: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            # Call engine
            llm_out: str = ""
            try:
                llm_out = self.engine(prompt)[0]
            except Exception as e:
                last_error = e
                _schema_log(errors, idx=idx, block="engine", item_index=None,
                            code="ENGINE_ERROR", message="LLM engine raised.", detail=str(e))
                continue

            # Log raw
            try:
                with open(r_log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n--- attempt {attempt} ---\n{llm_out}\n")
            except Exception as e:
                _log("noncanon ⚠", idx, f"failed to write raw log: {e}")

            # Parse LLM block
            m = _NEW_BLOCK_RE.search(llm_out)
            if not m:
                _schema_log(errors, idx=idx, block="top_level", item_index=None,
                            code="MISSING_BLOCK",
                            message="Could not find <new_variable_declarations> JSON block.")
                last_error = RuntimeError("Missing <new_variable_declarations> block")
                continue

            try:
                parsed_list = _parse_json_array_relaxed(m.group(1))
                if not isinstance(parsed_list, list):
                    raise ValueError("Parsed content is not a JSON array")
            except Exception as e:
                last_error = e
                _schema_log(errors, idx=idx, block="new_variable_declarations", item_index=None,
                            code="JSON_PARSE", message="Failed to parse JSON array.", detail=str(e))
                continue

            # PASS-THROUGH PERSISTENCE (no canonical expansion here)
            plan = {
                "new_remaining_variable_declarations": parsed_list,
                "stage": "remaining_noncanonical",
                "errors": errors or [],
            }
            try:
                with open(plan_log, "w", encoding="utf-8") as fh:
                    json.dump(plan, fh, indent=2, ensure_ascii=False)
            except Exception as e:
                _log("noncanon ⚠", idx, f"failed to write plan file: {e}")

            context["new_remaining_variable_declarations"] = parsed_list
            context["new_noncanonical_variable_declarations"] = parsed_list  # optional alias
            context["noncanonical_errors"] = errors or []
            context["noncanonical_stage"] = "remaining_noncanonical"
            _log("noncanon ✓", idx, f"{len(parsed_list)} remaining variables declared (pass-through)")
            return context

        # Exhausted attempts → fallback empty
        warnings.warn(f"SMT non-canonical pass fallback engaged (req#{idx}): {last_error}", RuntimeWarning)
        plan = {
            "new_remaining_variable_declarations": [],
            "stage": "remaining_noncanonical",
            "errors": errors or [{"code": "FALLBACK_EMPTY", "detail": str(last_error) if last_error else ""}],
        }
        try:
            with open(plan_log, "w", encoding="utf-8") as fh:
                json.dump(plan, fh, indent=2, ensure_ascii=False)
        except Exception:
            pass

        context["new_remaining_variable_declarations"] = []
        context["new_noncanonical_variable_declarations"] = []
        context["noncanonical_errors"] = plan["errors"]
        context["noncanonical_stage"] = "remaining_noncanonical"
        _log("noncanon ⚠", idx, "fallback produced 0 items")
        return context


__all__ = ["SMTIncrementalFreeVariableNamer"]
