from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Optional

import dspy
import z3

from ...utils.utils import _whole_program, _collect_leaf_vars
from ...utils.mbench import get_mbench, mbench_enabled, shorten


_ENUM_RE = re.compile(
    r"""\(declare-datatypes\s*\(\)\s*\(\s*
        \(?
        (?P<ty>[A-Za-z_][\w-]*)\s+
        (?P<body>(?:\s*[A-Za-z_][\w-]*\s*)+)\)\)
    """,
    re.VERBOSE,
)

_ENUM_SIMPLE_RE = re.compile(
    r"""\(declare-datatype\s+
        (?P<ty>[A-Za-z_][\w-]*)\s+
        \(\s*(?P<body>(?:[A-Za-z_][\w-]*\s*)+)\)
        \)""",
    re.VERBOSE,
)

_TAG_RE = re.compile(r"^[Rr]\d+_\d+$")

_DECL_CONST_RE = re.compile(r"\(declare-const\s+([^\s()]+)\s+([^\s()]+)\)")
_DECL_FUN0_RE = re.compile(r"\(declare-fun\s+([^\s()]+)\s+\(\s*\)\s+([^\s()]+)\)")


_NUMERIC_SORTS = {"Int", "Real"}


def _extract_enum_defs(smt: str) -> dict[str, list[str]]:
    enums: dict[str, list[str]] = {}
    for regex in (_ENUM_RE, _ENUM_SIMPLE_RE):
        for m in regex.finditer(smt):
            ty = m.group("ty")
            body = m.group("body")
            enums[ty] = body.split()
    return enums


def _extract_decl_sorts(smt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _DECL_CONST_RE.finditer(smt or ""):
        out[m.group(1)] = m.group(2)
    for m in _DECL_FUN0_RE.finditer(smt or ""):
        out[m.group(1)] = m.group(2)
    return out


def _split_stem_qual(var_name: str) -> tuple[str, str | None]:
    s = str(var_name)
    if "@@" not in s:
        return s, None
    stem, qual = s.split("@@", 1)
    stem = stem.strip()
    qual = qual.strip()
    return stem, qual or None


def _is_numeric_sort(sort_name: str) -> bool:
    return str(sort_name or "").strip() in _NUMERIC_SORTS


def _normalize_numeric_literal_text(x: int | float) -> str:
    # Canonicalize integer-valued reals to plain integers for stable helper names.
    if isinstance(x, int):
        return str(x)
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return f"{xf}"


def _op_token(op: str) -> str:
    return {
        ">=": "ge",
        ">": "gt",
        "<=": "le",
        "<": "lt",
        "=": "eq",
    }[op]


def _make_threshold_helper_name(var_name: str, op: str, value: int | float) -> str:
    return f"__THRESH__::{var_name}::{_op_token(op)}::{_normalize_numeric_literal_text(value)}"


def _helper_meaning(parent_name: str, parent_meta: Dict[str, Any], op: str, value: int | float) -> str:
    desc = str(parent_meta.get("description") or parent_name).strip()
    lit = _normalize_numeric_literal_text(value)
    return f"Whether {desc} satisfies {parent_name} {op} {lit}"


def _z3_num_value(expr: z3.ExprRef) -> Optional[int | float]:
    """
    Extract numeric literals from Z3 ASTs.

    Handles:
      - Int literals: 10
      - Rational/Real literals: 3/2, 1.5
      - Coercions: (to_real 10)
      - Unary minus: (- 10), (- (/ 3 2))
    """
    try:
        if z3.is_int_value(expr):
            return int(expr.as_long())

        if z3.is_rational_value(expr):
            num = expr.numerator_as_long()
            den = expr.denominator_as_long()
            frac = Fraction(num, den)
            if frac.denominator == 1:
                return int(frac.numerator)
            return float(frac)

        k = expr.decl().kind()

        # Handle (to_real k)
        if k == z3.Z3_OP_TO_REAL and expr.num_args() == 1:
            return _z3_num_value(expr.arg(0))

        # Handle unary minus on literals
        if k == z3.Z3_OP_UMINUS and expr.num_args() == 1:
            inner = _z3_num_value(expr.arg(0))
            return None if inner is None else -inner

    except Exception:
        return None

    return None


def _z3_symbol_name(expr: z3.ExprRef) -> Optional[str]:
    try:
        if z3.is_const(expr) and expr.num_args() == 0:
            d = expr.decl()
            if d.kind() == z3.Z3_OP_UNINTERPRETED:
                return d.name()
    except Exception:
        return None
    return None


def _extract_numeric_thresholds(smt: str, decl_sorts: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract simple numeric comparison atoms from the final SMT program, e.g.
      (>= age 18), (< bmi 35), (<= creatinine 1.5)
    and also flipped forms like
      (< 18 age)  => age > 18

    Returns:
      {
        "age": [
          {"op": ">=", "value": 18, "sort": "Int", "expr": "(>= age 18)"},
          ...
        ]
      }
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    try:
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        s.from_string(smt)

        seen = set()

        def add_one(var_name: str, op: str, value: int | float, expr_text: str) -> None:
            sort_name = decl_sorts.get(var_name, "")
            if not _is_numeric_sort(sort_name):
                return
            key = (var_name, op, float(value) if isinstance(value, float) else value)
            if key in seen:
                return
            seen.add(key)
            out.setdefault(var_name, []).append(
                {
                    "op": op,
                    "value": value,
                    "sort": sort_name,
                    "expr": expr_text,
                }
            )

        def walk(e: z3.ExprRef) -> None:
            try:
                k = e.decl().kind()
            except Exception:
                return

            if k in {
                z3.Z3_OP_LE,
                z3.Z3_OP_GE,
                z3.Z3_OP_LT,
                z3.Z3_OP_GT,
                z3.Z3_OP_EQ,
            } and e.num_args() == 2:
                a = e.arg(0)
                b = e.arg(1)

                a_sym = _z3_symbol_name(a)
                b_sym = _z3_symbol_name(b)
                a_num = _z3_num_value(a)
                b_num = _z3_num_value(b)

                op = {
                    z3.Z3_OP_LE: "<=",
                    z3.Z3_OP_GE: ">=",
                    z3.Z3_OP_LT: "<",
                    z3.Z3_OP_GT: ">",
                    z3.Z3_OP_EQ: "=",
                }[k]

                if a_sym is not None and b_num is not None:
                    add_one(a_sym, op, b_num, e.sexpr())
                elif a_num is not None and b_sym is not None:
                    flipped = {
                        "<=": ">=",
                        ">=": "<=",
                        "<": ">",
                        ">": "<",
                        "=": "=",
                    }[op]
                    add_one(b_sym, flipped, a_num, e.sexpr())

            for c in e.children():
                walk(c)

        for a in s.assertions():
            walk(a)

    except Exception:
        # Soft failure: threshold helpers are optional prompt enrichment.
        return {}

    for var_name in list(out.keys()):
        out[var_name].sort(
            key=lambda d: (
                str(d["op"]),
                float(d["value"]) if isinstance(d["value"], float) else d["value"],
            )
        )

    return out


class SMTLeafCollector(dspy.Module):
    """
    Adds to context:
        • leaf_variables    : set[str]
        • leaf_detail       : {var → {type, description, enum_values?, definition?}}
        • enum_definitions  : {EnumType → [values…]}
        • declared_sorts    : {Symbol → Sort}

    NEW:
    - If context["var_definitions"] is present, attach full definition blob under
      leaf_detail[var]["definition"] and set leaf_detail[var]["description"] to
      definition["meaning"] when available.
    - Stem backfill: if a STEM variable lacks a definition but one or more
      QUALIFIERS have one, backfill from a qualifier.
    - Numeric threshold helpers:
        For numeric variables, extract threshold comparisons from the final SMT
        program and add synthetic Bool helper extraction targets of the form:
            __THRESH__::<var>::<ge|gt|le|lt|eq>::<value>
        These are NOT declared SMT vars; they are prompt-time helper vars only.
        The backend later maps them back into numeric ranges.
    """

    def forward(self, context: dict) -> dict:
        mb = get_mbench(context)

        import sys

        def _dbg(*args):
            if context.get("VERBOSE"):
                print(*args, file=sys.stderr)

        with mb.timeit("SMTLeafCollector", "forward"):
            smt_full = _whole_program(context) or ""

            if mbench_enabled(context):
                mb.log_json(
                    "SMTLeafCollector",
                    "inputs",
                    {
                        "smt_len": len(smt_full),
                        "variable_index_size": len(context.get("variable_index", {}) or {}),
                        "var_definitions_loaded": len(context.get("var_definitions") or {}),
                        "var_definitions_source": context.get("var_definitions_source"),
                    },
                )
                mb.log_text(
                    "SMTLeafCollector",
                    "program_head_tail.smt2",
                    shorten(smt_full, 4000),
                )

            _dbg(f"[SMTLeafCollector] smt_len={len(smt_full)}")

            raw_vars = _collect_leaf_vars(smt_full)
            leaf_vars = {v for v in raw_vars if not _TAG_RE.match(v)}
            context["leaf_variables"] = leaf_vars

            enum_defs = _extract_enum_defs(smt_full)
            context["enum_definitions"] = enum_defs

            decl_sorts = _extract_decl_sorts(smt_full)
            context["declared_sorts"] = decl_sorts

            defs = context.get("var_definitions") or {}

            # stem -> [qual_vars...]
            stem2quals: dict[str, list[str]] = {}
            for v in leaf_vars:
                stem, qual = _split_stem_qual(v)
                if qual is not None:
                    stem2quals.setdefault(stem, []).append(v)

            detail: dict[str, dict] = {}
            covered_direct = 0

            for v in leaf_vars:
                ty = decl_sorts.get(v, "<?>")
                meta: dict[str, object] = {"type": ty, "description": ""}

                if ty in enum_defs:
                    meta["enum_values"] = enum_defs[ty]

                d = defs.get(v)
                if isinstance(d, dict):
                    meta["definition"] = d
                    meaning = str(d.get("meaning") or "").strip()
                    if meaning:
                        meta["description"] = meaning
                    covered_direct += 1

                detail[v] = meta

            # Backfill stem defs from qualifier defs if stem missing/blank
            covered_backfill = 0
            for stem, quals in stem2quals.items():
                if stem not in leaf_vars:
                    continue
                stem_meta = detail.get(stem, {})
                stem_desc = str(stem_meta.get("description") or "").strip()
                stem_def = stem_meta.get("definition")

                if stem_def or stem_desc:
                    continue

                picked = None
                for qv in sorted(quals):
                    qmeta = detail.get(qv, {})
                    qdef = qmeta.get("definition")
                    if isinstance(qdef, dict):
                        picked = qdef
                        break

                if isinstance(picked, dict):
                    stem_meta["definition"] = dict(picked)
                    stem_meta["definition_inherited_from_qualifier"] = True
                    meaning = str(picked.get("meaning") or "").strip()
                    if meaning:
                        stem_meta["description"] = meaning
                    detail[stem] = stem_meta
                    covered_backfill += 1

            # Extract numeric thresholds and create synthetic helper vars
            numeric_thresholds = _extract_numeric_thresholds(smt_full, decl_sorts)
            _dbg(f"[SMTLeafCollector] numeric_threshold_vars={sorted(numeric_thresholds.keys())}")
            _dbg(
                "[SMTLeafCollector] cigs_per_day_thresholds="
                f"{numeric_thresholds.get('patient_cigarette_smoking_tobacco_value_recorded_inthepast1months_withunit_cigarettes_per_day')}"
            )

            helper_count = 0
            threshold_parent_backfill_count = 0

            for parent_var, ths in numeric_thresholds.items():
                if parent_var not in detail:
                    ty = decl_sorts.get(parent_var, "<?>")
                    parent_meta: dict[str, object] = {"type": ty, "description": ""}

                    if ty in enum_defs:
                        parent_meta["enum_values"] = enum_defs[ty]

                    d = defs.get(parent_var)
                    if isinstance(d, dict):
                        parent_meta["definition"] = d
                        meaning = str(d.get("meaning") or "").strip()
                        if meaning:
                            parent_meta["description"] = meaning

                    detail[parent_var] = parent_meta
                    threshold_parent_backfill_count += 1

                parent_meta = detail[parent_var]
                parent_meta["numeric_thresholds"] = ths

                for th in ths:
                    helper_name = _make_threshold_helper_name(parent_var, th["op"], th["value"])
                    if helper_name in detail:
                        continue

                    helper_meta = {
                        "type": "Bool",
                        "description": _helper_meaning(parent_var, parent_meta, th["op"], th["value"]),
                        "threshold_helper": True,
                        "threshold_parent": parent_var,
                        "threshold_parent_type": parent_meta.get("type"),
                        "threshold_op": th["op"],
                        "threshold_value": th["value"],
                        "threshold_expr": th.get("expr", ""),
                    }
                    detail[helper_name] = helper_meta
                    helper_count += 1

            context["leaf_detail"] = detail

            if mbench_enabled(context):
                missing_sample = sorted([v for v in leaf_vars if v not in defs])[:50]
                mb.log_json(
                    "SMTLeafCollector",
                    "outputs",
                    {
                        "leaf_variables_count": len(leaf_vars),
                        "enum_types": sorted(list(enum_defs.keys())),
                        "declared_sorts_count": len(decl_sorts),
                        "leaf_detail_count": len(detail),
                        "var_definitions_covered_direct": covered_direct,
                        "var_definitions_covered_backfill": covered_backfill,
                        "numeric_threshold_parent_count": len(numeric_thresholds),
                        "numeric_threshold_parent_backfill_count": threshold_parent_backfill_count,
                        "numeric_threshold_helper_count": helper_count,
                        "var_definitions_missing_sample": missing_sample,
                    },
                )

        return context