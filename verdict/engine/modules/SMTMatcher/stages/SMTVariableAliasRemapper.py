from __future__ import annotations

from typing import Dict, Any

import dspy

from ...utils.mbench import get_mbench, mbench_enabled


class SMTVariableAliasRemapper(dspy.Module):
    """
    Remap alias-keyed patient_var_values_rich back to original SMT variable names.

    Input:
      - context["patient_var_values_rich"] keyed by alias
      - context["projection_alias_to_original"]

    Output:
      - context["patient_var_values_rich"] keyed by original SMT var
      - context["patient_var_values"] flattened by original SMT var
    """

    @staticmethod
    def _default_entry() -> Dict[str, Any]:
        return {"assessment": "", "value": None, "evidence": "", "range": None}

    @classmethod
    def _flatten_values(cls, rich_map: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (rich_map or {}).items():
            if isinstance(v, dict):
                out[k] = v.get("value", None)
            else:
                out[k] = v
        return out

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        alias_to_original = context.get("projection_alias_to_original", {}) or {}
        rich = context.get("patient_var_values_rich", {}) or {}
        leaf_detail = context.get("leaf_detail", {}) or {}

        if not isinstance(rich, dict):
            return context

        mb = get_mbench(context)
        log_enabled = mbench_enabled(context)

        remapped: Dict[str, Any] = {}

        # single-patient shape only here
        for alias, entry in rich.items():
            original = alias_to_original.get(alias)
            if original is None:
                # ignore unknown alias keys conservatively
                continue
            remapped[original] = entry if isinstance(entry, dict) else {"assessment": "", "value": entry, "evidence": "", "range": None}

        for original_var in leaf_detail:
            remapped.setdefault(original_var, self._default_entry())

        context["patient_var_values_rich"] = remapped
        context["patient_var_values"] = self._flatten_values(remapped)

        if log_enabled:
            mb.log_json("SMTVariableAliasRemapper", "alias_to_original.json", alias_to_original)
            mb.log_json("SMTVariableAliasRemapper", "remapped_patient_var_values_rich.json", remapped)
            mb.log_json("SMTVariableAliasRemapper", "remapped_patient_var_values.json", context["patient_var_values"])

        return context