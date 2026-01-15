# modules/SMTLeafCollector.py
from __future__ import annotations

import re, dspy

from smt_core.utils.z3_helpers import _whole_program, _collect_leaf_vars
from ...utils.mbench import get_mbench, mbench_enabled, shorten


_ENUM_RE = re.compile(
    r"""\(declare-datatypes\s*\(\)\s*\(\s*           # (declare-datatypes () (
        \(?                                          #   allow optional extra ‘(’
        (?P<ty>[A-Za-z_][\w-]*)\s+                   #   EnumType
        (?P<body>(?:\s*[A-Za-z_][\w-]*\s*)+)\)\)     #   val1 val2 ... ) )
    """,
    re.VERBOSE,
)

_ENUM_SIMPLE_RE = re.compile(
    r"""\(declare-datatype\s+
        (?P<ty>[A-Za-z_][\w-]*)\s+                   # EnumType
        \(\s*(?P<body>(?:[A-Za-z_][\w-]*\s*)+)\)     # (val1 val2 …)
        \)""",
    re.VERBOSE,
)

_TAG_RE = re.compile(r"^[Rr]\d+_\d+$")


def _extract_enum_defs(smt: str) -> dict[str, list[str]]:
    """Return {EnumType → [val1, val2, …]} for all enum declarations."""
    enums: dict[str, list[str]] = {}
    for regex in (_ENUM_RE, _ENUM_SIMPLE_RE):
        for m in regex.finditer(smt):
            ty   = m.group("ty")
            body = m.group("body")
            enums[ty] = body.split()
    return enums


class SMTLeafCollector(dspy.Module):
    """
    Adds to context:
        • leaf_variables    : set[str]
        • leaf_detail       : {var → {type, description, enum_values?}}
        • enum_definitions  : {EnumType → [values…]}

    Microbench (if enabled via context["MBENCH_ENABLED"] = True, default True):
        mbench/<run_id>/SMTLeafCollector/
            - inputs.json
            - program_head_tail.smt2
            - outputs.json
            - timings.jsonl
    """

    def forward(self, context: dict) -> dict:
        mb = get_mbench(context)

        with mb.timeit("SMTLeafCollector", "forward"):
            smt_full = _whole_program(context) or ""
            if mbench_enabled(context):
                mb.log_json(
                    "SMTLeafCollector",
                    "inputs",
                    {
                        "variable_index_size": len(context.get("variable_index", {})),
                        "smt_len": len(smt_full),
                    },
                )
                # store a truncated head/tail view to avoid giant files
                mb.log_text(
                    "SMTLeafCollector",
                    "program_head_tail.smt2",
                    shorten(smt_full, 4000),
                )

            print(f" ----- In SMTLeafCollector the full SMT program is {smt_full}")
            print()

            # ---------- 1) collect leaf vars ----------------------------
            raw_vars = _collect_leaf_vars(smt_full)
            context["leaf_variables"] = {v for v in raw_vars if not _TAG_RE.match(v)}

            # ---------- 2) parse enum declarations ----------------------
            enum_defs = _extract_enum_defs(smt_full)
            context["enum_definitions"] = enum_defs  # keep for later stages

            # ---------- 3) enrich leaf_detail ---------------------------
            var_idx = context.get("variable_index", {})
            detail: dict[str, dict] = {}

            for v in context["leaf_variables"]:
                meta = var_idx.get(v, {"type": "<?>", "description": ""}).copy()
                ty   = meta.get("type")
                if ty in enum_defs:
                    meta["enum_values"] = enum_defs[ty]
                detail[v] = meta

            context["leaf_detail"] = detail

            if mbench_enabled(context):
                context_vars = sorted(list(context["leaf_variables"])) if context.get("leaf_variables") else []
                mb.log_json(
                    "SMTLeafCollector",
                    "outputs",
                    {
                        "leaf_variables_count": len(context_vars),
                        "leaf_variables_sample": context_vars[:50],
                        "enum_types": sorted(list(enum_defs.keys())),
                        "leaf_detail_count": len(detail),
                    },
                )

        return context
