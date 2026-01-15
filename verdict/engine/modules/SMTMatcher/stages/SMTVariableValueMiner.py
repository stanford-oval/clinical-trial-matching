from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional, Tuple

import dspy
from ....parse_functions import parse_smt_output

from ...utils.mbench import get_mbench, mbench_enabled


_PAT_BLOCK = re.compile(
    r"<patient_variable_values>(.*?)</patient_variable_values>",
    re.DOTALL | re.IGNORECASE,
)

_NUMERIC_TYPES = {
    "int", "integer", "long",
    "float", "double", "decimal",
    "real", "number", "numeric",
}
_BOOL_TYPES = {"bool", "boolean"}

_MISSINGNESS_RE = re.compile(
    r"\b("
    r"no mention|not mentioned|not documented|not provided|not available|unknown|unclear|"
    r"not reported|not stated|not specified|"
    r"not performed|not done|not obtained|not assessed|"
    r"pending|needs evaluation|requires evaluation"
    r")\b",
    re.IGNORECASE,
)

_THRESH_HELPER_PREFIX = "__THRESH__::"


def _fill_common_prompt_fields(prompt_tpl: str, context: Dict[str, Any]) -> str:
    out = prompt_tpl
    out = out.replace("{{SIDE}}", str(context.get("encoding_side", "")))
    out = out.replace("{{TRIAL_TITLE}}", str(context.get("trial_title", "")))
    out = out.replace("{{INCLUSION_CRITERIA}}", str(context.get("trial_inclusion_criteria", "")))
    out = out.replace("{{EXCLUSION_CRITERIA}}", str(context.get("trial_exclusion_criteria", "")))
    out = out.replace("{{PATIENT_NOTES}}", "\n---\n".join(context.get("patient_notes", []) or []))
    return out


def _split_stem_qual(var_name: str) -> Tuple[str, Optional[str]]:
    s = str(var_name)
    if "@@" not in s:
        return s, None
    stem, qual = s.split("@@", 1)
    stem = stem.strip()
    qual = qual.strip()
    return stem, qual or None


def _is_qualifier(var_name: str) -> bool:
    return "@@" in str(var_name)


def _is_threshold_helper(var_name: str) -> bool:
    return str(var_name).startswith(_THRESH_HELPER_PREFIX)


def _parse_threshold_helper(var_name: str) -> Optional[Tuple[str, str, float | int]]:
    s = str(var_name)
    if not s.startswith(_THRESH_HELPER_PREFIX):
        return None
    parts = s.split("::", 4)
    if len(parts) != 5:
        return None
    _, _, parent_var, op_token, raw_value = parts
    op = {
        "ge": ">=",
        "gt": ">",
        "le": "<=",
        "lt": "<",
        "eq": "=",
    }.get(op_token)
    if op is None:
        return None

    try:
        if "." in raw_value:
            f = float(raw_value)
            return parent_var, op, f
        i = int(raw_value)
        return parent_var, op, i
    except Exception:
        try:
            f = float(raw_value)
            return parent_var, op, f
        except Exception:
            return None


def _invert_comparison(op: str) -> Optional[str]:
    return {
        ">=": "<",
        ">": "<=",
        "<=": ">",
        "<": ">=",
        "=": None,
    }.get(op)


def _truthy_flag(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    s = str(x).strip().lower()
    return s in {"yes", "y", "true", "t", "1"}


def _is_bool_type(meta: Dict[str, Any]) -> bool:
    ty = str(meta.get("type", "")).strip().lower()
    return ty in _BOOL_TYPES


def _is_numeric_type(meta: Dict[str, Any]) -> bool:
    ty = str(meta.get("type", "")).strip().lower()
    if ty in _NUMERIC_TYPES:
        return True
    for base in _NUMERIC_TYPES:
        if ty.startswith(base):
            return True
    return False


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int_if_integral(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-12:
            return int(round(x))
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        if "." in s:
            f = float(s)
            if abs(f - round(f)) < 1e-12:
                return int(round(f))
            return None
        return int(s)
    except Exception:
        return None


def _sanitize_range_obj(r: Any) -> Optional[Dict[str, Any]]:
    if r is None:
        return None
    if not isinstance(r, dict):
        return None
    if "min" not in r and "max" not in r:
        return None

    mn = _safe_float(r.get("min"))
    mx = _safe_float(r.get("max"))
    return {
        "min": mn,
        "max": mx,
        "min_strict": bool(r.get("min_strict", False)),
        "max_strict": bool(r.get("max_strict", False)),
    }


def _coerce_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        e = dict(entry)

        if "evidence" not in e:
            if "explanation" in e:
                e["evidence"] = e.pop("explanation")
            elif "evidence or explanation" in e:
                e["evidence"] = e.pop("evidence or explanation")

        out: Dict[str, Any] = {
            "assessment": "" if e.get("assessment", "") is None else str(e.get("assessment", "")),
            "value": e.get("value", None),
            "evidence": "" if e.get("evidence", "") is None else str(e.get("evidence", "")),
        }
        out["range"] = e.get("range") if "range" in e else None
        return out

    return {"assessment": "", "value": entry, "evidence": "", "range": None}


def parse_patient_var_values(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}

    m = _PAT_BLOCK.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()

    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(data, dict):
        return {}

    normalized: Dict[str, Any] = {}
    for var, entry in data.items():
        norm = _coerce_entry(entry)
        norm.setdefault("assessment", "")
        norm.setdefault("value", None)
        norm.setdefault("evidence", "")
        if "range" not in norm:
            norm["range"] = None
        normalized[str(var)] = norm

    return normalized


def _is_action_correctable(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    for k in (
        "always_satisfiable_with_action",
        "satisfiable_with_action",
        "correctable_by_action",
        "one_off_action",
        "action_correctable",
    ):
        if _truthy_flag(meta.get(k)):
            return True
    cat = str(meta.get("category", "") or meta.get("tag", "") or "").upper()
    if "ALWAYS_SATISFIABLE_WITH_ACTION" in cat:
        return True
    if "SATISFIABLE_WITH_ACTION" in cat:
        return True
    if "WITH_ACTION" in cat:
        return True
    return False


def _evidence_looks_like_missingness(ev: str) -> bool:
    if not ev:
        return True
    return bool(_MISSINGNESS_RE.search(ev))


class SMTVariableValueMiner(dspy.Module):
    """
    Alias-only miner view.

    Important behavior:
    - Prompt shows ONLY rewritten alias + rewritten meaning.
    - Miner outputs are expected to be keyed by rewritten alias.
    - Alias remapping happens in a later stage.
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @staticmethod
    def _group_by_stem_from_alias(
        alias_leaf_detail: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, List[str]]]:
        groups: Dict[str, List[str]] = {}

        for alias_name, meta in alias_leaf_detail.items():
            if meta.get("threshold_helper"):
                parent = str(meta.get("threshold_parent_alias") or meta.get("threshold_parent") or alias_name).strip()
                groups.setdefault(parent, []).append(alias_name)
                continue

            stem, _q = _split_stem_qual(alias_name)
            groups.setdefault(stem, []).append(alias_name)

        out: List[Tuple[str, List[str]]] = []
        for stem in sorted(groups.keys()):
            vars_in_group = groups[stem]
            stem_var = stem if stem in vars_in_group else None
            quals = sorted([x for x in vars_in_group if x != stem_var and not alias_leaf_detail.get(x, {}).get("threshold_helper")])
            helpers = sorted([x for x in vars_in_group if alias_leaf_detail.get(x, {}).get("threshold_helper")])
            ordered = ([stem_var] if stem_var else []) + quals + helpers
            out.append((stem, ordered))
        return out

    @staticmethod
    def _projection_payload(meta: Dict[str, Any]) -> Dict[str, str]:
        p = meta.get("projection")
        if isinstance(p, dict):
            return {
                "alias": str(p.get("rewritten_variable_name") or "").strip(),
                "meaning": str(p.get("meaning") or "").strip(),
                "projection_summary": str(p.get("projection_summary") or "").strip(),
                "projection_status": str(p.get("projection_status") or "").strip(),
            }
        return {
            "alias": "",
            "meaning": "",
            "projection_summary": "",
            "projection_status": "",
        }

    @classmethod
    def _build_alias_leaf_detail(
        cls,
        leaf_detail: Dict[str, Dict[str, Any]],
        projection_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        alias_leaf: Dict[str, Dict[str, Any]] = {}

        for original_var, meta in (leaf_detail or {}).items():
            proj = (projection_map or {}).get(original_var) or cls._projection_payload(meta)
            alias = str(proj.get("rewritten_variable_name") or "").strip() or original_var
            meaning = str(proj.get("meaning") or "").strip() or str(meta.get("description") or "").strip()

            new_meta = dict(meta)
            new_meta["original_variable_name"] = original_var
            new_meta["alias_name"] = alias
            new_meta["description"] = meaning
            new_meta["projection"] = {
                "rewritten_variable_name": alias,
                "meaning": meaning,
                "projection_summary": str(proj.get("projection_summary") or "").strip(),
                "projection_status": str(proj.get("projection_status") or "").strip(),
            }

            if meta.get("threshold_helper"):
                parent_orig = str(meta.get("threshold_parent") or "").strip()
                parent_proj = (projection_map or {}).get(parent_orig) or {}
                new_meta["threshold_parent_alias"] = str(parent_proj.get("rewritten_variable_name") or parent_orig).strip()

            alias_leaf[alias] = new_meta

        return alias_leaf

    @staticmethod
    def _format_definition(meta: Dict[str, Any]) -> str:
        p = meta.get("projection")
        if isinstance(p, dict):
            meaning = str(p.get("meaning") or "").strip()
            if meaning:
                return meaning

        d = meta.get("definition")
        if isinstance(d, dict):
            meaning = str(d.get("meaning") or "").strip()
            if meaning:
                return meaning

        return str(meta.get("description") or "").strip()

    @staticmethod
    def _format_threshold_helper(alias_name: str, meta: Dict[str, Any]) -> List[str]:
        parent = str(meta.get("threshold_parent_alias") or meta.get("threshold_parent") or "").strip()
        op = str(meta.get("threshold_op") or "").strip()
        value = meta.get("threshold_value")
        expr = str(meta.get("threshold_expr") or "").strip()
        desc = str(meta.get("description") or "").strip()

        lines: List[str] = []
        lines.append(f"    - name: {alias_name}")
        lines.append("      type: Bool")
        if desc:
            lines.append(f"      meaning: {desc}")
        if parent:
            lines.append(f"      compares_variable: {parent}")
        if op:
            lines.append(f"      comparison_operator: {op}")
        lines.append(f"      comparison_value: {value}")
        if expr:
            lines.append(f"      derived_from_smt: {expr}")
        return lines

    @classmethod
    def _format_var_list(
        cls,
        alias_leaf_detail: Dict[str, Dict[str, Any]],
        variable_scope_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        groups = cls._group_by_stem_from_alias(alias_leaf_detail)

        lines: List[str] = []
        lines.append(
            "IMPORTANT:\n"
            "- Each variable below is identified ONLY by its rewritten alias name.\n"
            "- Use the alias name and meaning as the extraction target.\n"
            "- Return JSON keyed by the alias names exactly as shown below.\n"
            "- Do not invent new keys.\n"
            "- Do not use any hidden/original SMT names.\n"
        )

        def _projected_scope_lines(scope_info: Dict[str, str], indent: str) -> List[str]:
            scope = str(scope_info.get("scope") or "").strip()
            reason = str(scope_info.get("reason") or "").strip()
            if scope.lower() != "projected_away":
                return []
            out: List[str] = [f"{indent}scope_classification: {scope}"]
            if reason:
                out.append(f"{indent}scope_reason: {reason}")
            return out

        for stem_alias, vars_in_group in groups:
            if stem_alias in alias_leaf_detail:
                meta = alias_leaf_detail[stem_alias]
                desc = cls._format_definition(meta)
                ty = meta.get("type", "<?>")
                lines.append(f"- name: {stem_alias}")
                lines.append(f"  type: {ty}")
                lines.append(f"  meaning: {desc or '(missing)'}")

                proj = meta.get("projection") or {}
                proj_status = str(proj.get("projection_status") or "").strip()
                proj_summary = str(proj.get("projection_summary") or "").strip()
                if proj_status:
                    lines.append(f"  rewrite_status: {proj_status}")
                if proj_summary:
                    lines.append(f"  rewrite_note: {proj_summary}")

                scope_info = (variable_scope_map or {}).get(meta.get("original_variable_name", ""), {})
                lines.extend(_projected_scope_lines(scope_info, "  "))
            else:
                lines.append(f"- name: {stem_alias}")
                lines.append("  type: <?>")
                lines.append("  meaning: (stem not present; qualifiers and/or helper decisions only)")

            quals = [
                v for v in vars_in_group
                if v != stem_alias and not alias_leaf_detail.get(v, {}).get("threshold_helper")
            ]
            if quals:
                lines.append("  qualifiers:")
                for q_alias in quals:
                    qmeta = alias_leaf_detail.get(q_alias, {})
                    qdesc = cls._format_definition(qmeta)
                    qty = qmeta.get("type", "<?>")
                    lines.append(f"    - name: {q_alias}")
                    lines.append(f"      type: {qty}")
                    lines.append(f"      meaning: {qdesc or '(missing)'}")

                    proj = qmeta.get("projection") or {}
                    proj_status = str(proj.get("projection_status") or "").strip()
                    proj_summary = str(proj.get("projection_summary") or "").strip()
                    if proj_status:
                        lines.append(f"      rewrite_status: {proj_status}")
                    if proj_summary:
                        lines.append(f"      rewrite_note: {proj_summary}")

            helpers = [v for v in vars_in_group if alias_leaf_detail.get(v, {}).get("threshold_helper")]
            if helpers:
                lines.append("  threshold_helper_decisions:")
                for hv in helpers:
                    hmeta = alias_leaf_detail.get(hv, {})
                    lines.extend(cls._format_threshold_helper(hv, hmeta))

            lines.append("")

        return "\n".join(lines).rstrip()

    @staticmethod
    def _notes_for_one(pid: str, db: List[dict]) -> List[str]:
        return [rec["text"] for rec in db if rec.get("_id") == pid]

    def _resolve_patient_notes(self, ctx: Dict[str, Any]) -> Dict[str, List[str]]:
        if "matched_patient_map" in ctx:
            return ctx["matched_patient_map"]

        db = ctx.get("patient_notes_db", [])
        if "patient_notes_map" in ctx:
            return ctx["patient_notes_map"]
        if "patient_notes" in ctx:
            return {"<unspecified>": ctx["patient_notes"]}
        if "patient_id" in ctx:
            return {ctx["patient_id"]: self._notes_for_one(ctx["patient_id"], db)}
        if "patient_ids" in ctx:
            return {pid: self._notes_for_one(pid, db) for pid in ctx["patient_ids"]}

        if "matched_patient_ids" in ctx:
            return {pid: self._notes_for_one(pid, db) for pid in ctx["matched_patient_ids"]}
        return {}

    def _build_prompt(
        self,
        prompt_tpl: str,
        alias_leaf_detail: Dict[str, Dict[str, Any]],
        notes: List[str],
        context: Dict[str, Any],
        variable_scope_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        tmp_ctx = dict(context)
        tmp_ctx["patient_notes"] = notes or []
        out = _fill_common_prompt_fields(prompt_tpl, tmp_ctx)
        out = out.replace(
            "{{VARIABLE_LIST}}",
            self._format_var_list(alias_leaf_detail, variable_scope_map=variable_scope_map),
        )
        return out.strip()

    def _run_llm(
        self,
        prompt_text: str,
        *,
        pid: str,
        chunk_idx: int,
        mb,
        log: bool,
    ) -> str:
        if log:
            mb.log_text(
                "SMTVariableValueMiner",
                f"{pid}__chunk{chunk_idx:03d}__prompt.txt",
                prompt_text,
            )
        resp = self.engine(prompt_text)
        raw = resp[0]
        if log:
            mb.log_text(
                "SMTVariableValueMiner",
                f"{pid}__chunk{chunk_idx:03d}__raw.txt",
                raw,
            )
        return raw

    @staticmethod
    def _validate_enums(var_map_rich: Dict[str, Any], alias_leaf_detail: Dict[str, Dict]):
        for alias_name, meta in alias_leaf_detail.items():
            allowed = meta.get("enum_values")
            if not allowed:
                continue

            entry = var_map_rich.get(alias_name)
            if not isinstance(entry, dict):
                continue

            actual = entry.get("value")
            if actual is None:
                continue

            if str(actual).strip() not in allowed:
                entry["value"] = None

    @staticmethod
    def _sanitize_by_type(
        var_map_rich: Dict[str, Any],
        alias_leaf_detail: Dict[str, Dict[str, Any]],
    ) -> None:
        for alias_name, meta in (alias_leaf_detail or {}).items():
            entry = var_map_rich.get(alias_name)
            if not isinstance(entry, dict):
                continue

            is_num = _is_numeric_type(meta)

            if not is_num:
                entry["range"] = None
            else:
                entry["range"] = _sanitize_range_obj(entry.get("range"))

            if is_num:
                val = entry.get("value")
                if val is None:
                    continue

                ty = str(meta.get("type", "")).strip().lower()
                if "int" in ty or "integer" in ty or ty.startswith("long"):
                    i = _safe_int_if_integral(val)
                    if i is not None:
                        entry["value"] = i
                    else:
                        f = _safe_float(val)
                        entry["value"] = f if f is not None else None
                else:
                    f = _safe_float(val)
                    entry["value"] = f if f is not None else None

    @staticmethod
    def _enforce_qualifier_semantics(
        var_map_rich: Dict[str, Any],
        alias_leaf_detail: Dict[str, Dict[str, Any]],
    ) -> None:
        for alias_name, meta in (alias_leaf_detail or {}).items():
            entry = var_map_rich.get(alias_name)
            if not isinstance(entry, dict):
                continue

            val = entry.get("value")
            ev = str(entry.get("evidence") or "")

            if val is False and _is_action_correctable(meta) and _evidence_looks_like_missingness(ev):
                entry["value"] = None
                continue

            if _is_threshold_helper(alias_name):
                continue

            if not _is_qualifier(alias_name):
                continue

            stem_alias, _q = _split_stem_qual(alias_name)
            stem_meta = alias_leaf_detail.get(stem_alias, {})
            if not (_is_bool_type(meta) and _is_bool_type(stem_meta)):
                continue

            if val is True:
                stem_entry = var_map_rich.get(stem_alias)
                stem_val = stem_entry.get("value") if isinstance(stem_entry, dict) else None
                if stem_val is not True:
                    entry["value"] = None

    @staticmethod
    def _ensure_rich_defaults_for_vars(var_names: List[str]) -> Dict[str, Any]:
        return {v: {"assessment": "", "value": None, "evidence": "", "range": None} for v in var_names}

    @staticmethod
    def _flatten_values(var_map_rich: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for v, entry in (var_map_rich or {}).items():
            if isinstance(entry, dict):
                out[v] = entry.get("value", None)
            else:
                out[v] = entry
        return out

    @staticmethod
    def _extract_evidence(var_map_rich: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for v, entry in (var_map_rich or {}).items():
            if isinstance(entry, dict):
                ev = entry.get("evidence", "")
                out[v] = "" if ev is None else str(ev)
            else:
                out[v] = ""
        return out

    @staticmethod
    def _extract_ranges(var_map_rich: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for v, entry in (var_map_rich or {}).items():
            if isinstance(entry, dict):
                out[v] = entry.get("range", None)
            else:
                out[v] = None
        return out

    @staticmethod
    def _ordered_by_stem(var_map_rich: Dict[str, Any], alias_leaf_detail: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        keys = list((var_map_rich or {}).keys())

        def _k(k: str) -> Tuple[str, int, str]:
            meta = alias_leaf_detail.get(k, {})
            if meta.get("threshold_helper"):
                stem = str(meta.get("threshold_parent_alias") or meta.get("threshold_parent") or k)
                return (stem, 2, k)
            stem, qual = _split_stem_qual(k)
            is_qual = 1 if qual is not None else 0
            qual_s = qual or ""
            return (stem, is_qual, qual_s)

        out: Dict[str, Any] = {}
        for k in sorted(keys, key=_k):
            out[k] = var_map_rich[k]
        return out

    def _chunk_groups(self, alias_leaf_detail: Dict[str, Dict[str, Any]], max_vars: int) -> List[List[str]]:
        groups = self._group_by_stem_from_alias(alias_leaf_detail)
        chunks: List[List[str]] = []
        cur: List[str] = []
        cur_n = 0
        for _stem, vars_in_group in groups:
            g_n = len(vars_in_group)
            if cur and (cur_n + g_n) > max_vars:
                chunks.append(cur)
                cur = []
                cur_n = 0
            cur.extend(vars_in_group)
            cur_n += g_n
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _merge_one_bound(existing: Optional[Dict[str, Any]], *, lower: bool, value: float, strict: bool) -> Dict[str, Any]:
        r = dict(existing or {"min": None, "max": None, "min_strict": False, "max_strict": False})
        r.setdefault("min", None)
        r.setdefault("max", None)
        r.setdefault("min_strict", False)
        r.setdefault("max_strict", False)

        if lower:
            cur = r.get("min")
            cur_strict = bool(r.get("min_strict", False))
            if cur is None or value > cur:
                r["min"] = value
                r["min_strict"] = strict
            elif cur is not None and abs(float(value) - float(cur)) < 1e-12:
                r["min_strict"] = cur_strict or strict
        else:
            cur = r.get("max")
            cur_strict = bool(r.get("max_strict", False))
            if cur is None or value < cur:
                r["max"] = value
                r["max_strict"] = strict
            elif cur is not None and abs(float(value) - float(cur)) < 1e-12:
                r["max_strict"] = cur_strict or strict

        return r

    @staticmethod
    def _comparison_to_bound(op: str) -> Tuple[bool, bool]:
        if op == ">=":
            return True, False
        if op == ">":
            return True, True
        if op == "<=":
            return False, False
        if op == "<":
            return False, True
        raise ValueError(f"Unsupported bound op: {op}")

    @classmethod
    def _merge_threshold_helpers_into_ranges(
        cls,
        var_map_rich: Dict[str, Any],
        alias_leaf_detail: Dict[str, Dict[str, Any]],
    ) -> None:
        for helper_alias, meta in (alias_leaf_detail or {}).items():
            if not meta.get("threshold_helper"):
                continue

            helper_entry = var_map_rich.get(helper_alias)
            if not isinstance(helper_entry, dict):
                continue

            helper_val = helper_entry.get("value")
            if helper_val is None:
                continue
            if not isinstance(helper_val, bool):
                continue

            parent_alias = str(meta.get("threshold_parent_alias") or meta.get("threshold_parent") or "").strip()
            op = str(meta.get("threshold_op") or "").strip()
            threshold_value = meta.get("threshold_value")
            parent_meta = alias_leaf_detail.get(parent_alias, {})
            parent_sort = str(parent_meta.get("type") or meta.get("threshold_parent_type") or "").strip().lower()

            if not parent_alias:
                continue

            effective_op = op if helper_val is True else _invert_comparison(op)
            if effective_op is None:
                continue

            parent_entry = var_map_rich.setdefault(
                parent_alias,
                {"assessment": "", "value": None, "evidence": "", "range": None},
            )
            if not isinstance(parent_entry, dict):
                parent_entry = {"assessment": "", "value": None, "evidence": "", "range": None}
                var_map_rich[parent_alias] = parent_entry

            if parent_entry.get("value") is not None:
                continue

            if threshold_value is None:
                continue

            if "int" in parent_sort or "integer" in parent_sort or parent_sort.startswith("long"):
                tv = _safe_float(threshold_value)
                if tv is None:
                    continue
                if abs(tv - round(tv)) < 1e-12:
                    tv_int = int(round(tv))
                else:
                    tv_int = tv

                if effective_op == ">":
                    if isinstance(tv_int, int):
                        effective_op = ">="
                        threshold_value = tv_int + 1
                    else:
                        threshold_value = tv
                elif effective_op == "<":
                    if isinstance(tv_int, int):
                        effective_op = "<="
                        threshold_value = tv_int - 1
                    else:
                        threshold_value = tv
                else:
                    threshold_value = tv_int

            if effective_op == "=":
                parent_entry["value"] = threshold_value
                continue

            try:
                is_lower, is_strict = cls._comparison_to_bound(effective_op)
            except Exception:
                continue

            existing_range = _sanitize_range_obj(parent_entry.get("range"))
            merged = cls._merge_one_bound(
                existing_range,
                lower=is_lower,
                value=float(threshold_value),
                strict=is_strict,
            )

            if "int" in parent_sort or "integer" in parent_sort or parent_sort.startswith("long"):
                if merged.get("min") is not None and abs(float(merged["min"]) - round(float(merged["min"]))) < 1e-12:
                    merged["min"] = int(round(float(merged["min"])))
                if merged.get("max") is not None and abs(float(merged["max"]) - round(float(merged["max"]))) < 1e-12:
                    merged["max"] = int(round(float(merged["max"])))

            parent_entry["range"] = merged

            if not parent_entry.get("evidence"):
                parent_entry["evidence"] = str(helper_entry.get("evidence") or "")
            if not parent_entry.get("assessment"):
                parent_entry["assessment"] = "Derived from threshold-helper decisions"

    @staticmethod
    def _partition_by_scope(
        alias_leaf_detail: Dict[str, Dict[str, Any]],
        variable_scope_map: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        active: Dict[str, Dict[str, Any]] = {}
        projected_away: Dict[str, Dict[str, Any]] = {}

        for alias_name, meta in (alias_leaf_detail or {}).items():
            original_var = str(meta.get("original_variable_name") or alias_name)
            scope = str((variable_scope_map.get(original_var) or {}).get("scope") or "").strip().lower()
            if scope == "projected_away":
                projected_away[alias_name] = meta
            else:
                active[alias_name] = meta

        return active, projected_away

    @staticmethod
    def _make_projected_away_entry(reason: str = "") -> Dict[str, Any]:
        reason = str(reason or "").strip() or "projected-away / don't-care dimension"
        return {
            "assessment": f"auto-null by scope filter: {reason}",
            "value": None,
            "evidence": "",
            "range": None,
        }

    @classmethod
    def _make_projected_away_defaults(
        cls,
        projected_away_alias_leaf_detail: Dict[str, Dict[str, Any]],
        variable_scope_map: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for alias_name, meta in (projected_away_alias_leaf_detail or {}).items():
            original_var = str(meta.get("original_variable_name") or alias_name)
            sc = variable_scope_map.get(original_var, {})
            reason = str(sc.get("reason") or "projected-away / don't-care dimension")
            out[alias_name] = cls._make_projected_away_entry(reason)
        return out

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("► VariableValueMiner: extracting patient-specific values")

        mb = get_mbench(context)
        log_enabled = mbench_enabled(context)

        leaf_detail = context.get("leaf_detail", {}) or {}
        projection_map = context.get("variable_projection_map", {}) or {}
        variable_scope_map = context.get("variable_scope_map", {}) or {}

        if context["inc_exc"] == "inclusion":
            prompt_tpl = (context.get("SMTVariableValueMinerInclusion_prompt", "") or "").strip()
        else:
            prompt_tpl = (context.get("SMTVariableValueMinerExclusion_prompt", "") or "").strip()

        batch_size = int(context.get("VV_BATCH_SIZE", 10) or 10)

        if not leaf_detail or not prompt_tpl:
            print("   !!!  missing leaf_detail or prompt – skipping.")
            context["patient_var_values_rich"] = {}
            context["patient_var_values"] = {}
            return context

        alias_leaf_detail = self._build_alias_leaf_detail(leaf_detail, projection_map)

        note_map = self._resolve_patient_notes(context)
        if not note_map:
            rich = self._ensure_rich_defaults_for_vars(list(alias_leaf_detail))
            for alias_name, meta in alias_leaf_detail.items():
                original_var = str(meta.get("original_variable_name") or alias_name)
                sc = variable_scope_map.get(original_var, {})
                if str(sc.get("scope") or "").strip().lower() == "projected_away":
                    rich[alias_name] = self._make_projected_away_entry(
                        str(sc.get("reason") or "projected-away / don't-care dimension")
                    )
            context["patient_var_values_rich"] = rich
            context["patient_var_values"] = self._flatten_values(rich)
            return context

        active_alias_leaf_detail, projected_away_alias_leaf_detail = self._partition_by_scope(
            alias_leaf_detail, variable_scope_map
        )

        results_rich: Dict[str, Any] = {}

        for pid, notes in note_map.items():
            if log_enabled:
                mb.log_text(
                    "SMTVariableValueMiner",
                    f"{pid}__notes.txt",
                    "\n---\n".join(notes or []),
                )
                mb.log_json(
                    "SMTVariableValueMiner",
                    f"{pid}__alias_leaf_detail.json",
                    alias_leaf_detail,
                )
                mb.log_json(
                    "SMTVariableValueMiner",
                    f"{pid}__scope_partition.json",
                    {
                        "active_count": len(active_alias_leaf_detail),
                        "projected_away_count": len(projected_away_alias_leaf_detail),
                        "active_aliases_sample": sorted(list(active_alias_leaf_detail.keys()))[:200],
                        "projected_away_aliases": sorted(list(projected_away_alias_leaf_detail.keys())),
                    },
                )

            full_var_map_rich: Dict[str, Any] = self._make_projected_away_defaults(
                projected_away_alias_leaf_detail,
                variable_scope_map,
            )

            if not notes:
                for alias_name in active_alias_leaf_detail:
                    full_var_map_rich.setdefault(
                        alias_name, {"assessment": "", "value": None, "evidence": "", "range": None}
                    )

                self._sanitize_by_type(full_var_map_rich, alias_leaf_detail)
                self._validate_enums(full_var_map_rich, alias_leaf_detail)
                self._enforce_qualifier_semantics(full_var_map_rich, alias_leaf_detail)
                self._merge_threshold_helpers_into_ranges(full_var_map_rich, alias_leaf_detail)
                self._sanitize_by_type(full_var_map_rich, alias_leaf_detail)

                for alias_name in alias_leaf_detail:
                    full_var_map_rich.setdefault(
                        alias_name, {"assessment": "", "value": None, "evidence": "", "range": None}
                    )

                results_rich[pid] = full_var_map_rich

                if log_enabled:
                    ordered = self._ordered_by_stem(full_var_map_rich, alias_leaf_detail)
                    mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_rich", ordered)
                    mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_values", self._flatten_values(ordered))
                    mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_evidence", self._extract_evidence(ordered))
                    mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_ranges", self._extract_ranges(ordered))
                continue

            if active_alias_leaf_detail:
                chunks = self._chunk_groups(active_alias_leaf_detail, batch_size)

                for ci, alias_names in enumerate(chunks, start=0):
                    sub_alias_leaf = {a: active_alias_leaf_detail[a] for a in alias_names if a in active_alias_leaf_detail}

                    prompt_text = self._build_prompt(
                        prompt_tpl,
                        sub_alias_leaf,
                        notes,
                        context,
                        variable_scope_map=variable_scope_map,
                    )

                    raw = self._run_llm(
                        prompt_text,
                        pid=pid,
                        chunk_idx=ci,
                        mb=mb,
                        log=log_enabled,
                    )

                    sub_map_rich = parse_patient_var_values(raw)
                    if not sub_map_rich:
                        legacy = parse_smt_output(raw) or {}
                        if not isinstance(legacy, dict):
                            legacy = {}
                        sub_map_rich = {str(k): _coerce_entry(v) for k, v in legacy.items()}

                    for alias_name in sub_alias_leaf:
                        sub_map_rich.setdefault(
                            alias_name, {"assessment": "", "value": None, "evidence": "", "range": None}
                        )
                        if isinstance(sub_map_rich[alias_name], dict):
                            sub_map_rich[alias_name].setdefault("assessment", "")
                            sub_map_rich[alias_name].setdefault("value", None)
                            sub_map_rich[alias_name].setdefault("evidence", "")
                            sub_map_rich[alias_name].setdefault("range", None)

                    full_var_map_rich.update(sub_map_rich)

            for alias_name in active_alias_leaf_detail:
                full_var_map_rich.setdefault(
                    alias_name, {"assessment": "", "value": None, "evidence": "", "range": None}
                )

            self._sanitize_by_type(full_var_map_rich, alias_leaf_detail)
            self._validate_enums(full_var_map_rich, alias_leaf_detail)
            self._enforce_qualifier_semantics(full_var_map_rich, alias_leaf_detail)
            self._merge_threshold_helpers_into_ranges(full_var_map_rich, alias_leaf_detail)
            self._sanitize_by_type(full_var_map_rich, alias_leaf_detail)

            for alias_name in alias_leaf_detail:
                full_var_map_rich.setdefault(
                    alias_name, {"assessment": "", "value": None, "evidence": "", "range": None}
                )

            results_rich[pid] = full_var_map_rich

            if log_enabled:
                ordered = self._ordered_by_stem(full_var_map_rich, alias_leaf_detail)
                mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_rich", ordered)
                mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_values", self._flatten_values(ordered))
                mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_evidence", self._extract_evidence(ordered))
                mb.log_json("SMTVariableValueMiner", f"{pid}__extraction_ranges", self._extract_ranges(ordered))

        rich_out = next(iter(results_rich.values())) if len(results_rich) == 1 else results_rich
        context["patient_var_values_rich"] = rich_out

        if isinstance(rich_out, dict) and rich_out and all(isinstance(v, dict) for v in rich_out.values()):
            context["patient_var_values"] = self._flatten_values(rich_out)
        else:
            flat_by_pid: Dict[str, Any] = {}
            for pid, mp in (results_rich or {}).items():
                flat_by_pid[pid] = self._flatten_values(mp)
            context["patient_var_values"] = flat_by_pid

        return context