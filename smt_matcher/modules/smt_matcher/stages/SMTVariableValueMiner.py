# modules/stages/SMTVariableValueMiner.py
from __future__ import annotations

import json, re
from typing import List, Dict, Any
import dspy
from smt_core.parse_functions import parse_smt_output

from ...utils.mbench import get_mbench, mbench_enabled


_PAT_BLOCK = re.compile(
    r"<patient_variable_values>(.*?)</patient_variable_values>",
    re.DOTALL | re.IGNORECASE,
)


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

    for var, entry in list(data.items()):
        if not isinstance(entry, dict):
            entry = {"value": entry}

        if "evidence" not in entry:
            if "explanation" in entry:
                entry["evidence"] = entry.pop("explanation")
            elif "evidence or explanation" in entry:
                entry["evidence"] = entry.pop("evidence or explanation")

        entry.setdefault("value", None)
        entry.setdefault("evidence", "")
        data[var] = entry

    return data


class SMTVariableValueMiner(dspy.Module):
    """
    LLM-based extraction of { var → {value, evidence} }.

    Microbench logging:
      • For each patient:
          - `<pid>__notes.txt`                   ← input notes we fed in
          - `<pid>__chunkNNN__prompt.txt`        ← LLM input prompt (per chunk)
          - `<pid>__chunkNNN__raw.txt`           ← raw LLM output (per chunk)
          - `<pid>__extraction.json`             ← final extracted var→{value,evidence}
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @staticmethod
    def _format_var_list(leaf_detail: Dict[str, Dict[str, str]]) -> str:
        lines = []
        for name, meta in sorted(leaf_detail.items()):
            line = f"- {name} ({meta.get('type','<?>')}) – {meta.get('description','')}"
            if "enum_values" in meta:
                line += " | Allowed values: " + ", ".join(meta["enum_values"])
            lines.append(line)
        return "\n".join(lines)

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

    def _build_prompt(self, prompt_tpl: str, leaf: Dict[str, Dict[str, str]], notes: List[str]) -> str:
        return (
            prompt_tpl.replace("{{VARIABLE_LIST}}", self._format_var_list(leaf))
            .replace("{{PATIENT_NOTES}}", "\n---\n".join(notes))
            .strip()
        )

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
    def _validate_enums(var_map: Dict[str, Any], leaf_detail: Dict[str, Dict]):
        for v, meta in leaf_detail.items():
            allowed = meta.get("enum_values")
            if not allowed:
                continue
            val = var_map.get(v)
            if val is None:
                continue
            if str(val).strip() not in allowed:
                var_map[v] = None

    def _chunks(self, seq, n):
        """Yield n-sized chunks from seq."""
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("► VariableValueMiner: extracting patient-specific values")

        mb = get_mbench(context)
        log_enabled = mbench_enabled(context)

        leaf_detail = context.get("leaf_detail", {}) or {}
        if context["inc_exc"] == "inclusion":
            prompt_tpl = (context.get("SMTVariableValueMinerInclusion_prompt", "") or "").strip()
        else:
            prompt_tpl = (context.get("SMTVariableValueMinerExclusion_prompt", "") or "").strip()
        batch_size = int(context.get("VV_BATCH_SIZE", 10) or 10)

        if not leaf_detail or not prompt_tpl:
            print("   !!!  missing leaf_detail or prompt – skipping.")
            context["patient_var_values"] = {}
            return context

        note_map = self._resolve_patient_notes(context)
        if not note_map:
            context["patient_var_values"] = {v: None for v in leaf_detail}
            return context

        results: Dict[str, Any] = {}

        # ── loop over patients (usually 1, but supports many) ────────────────
        for pid, notes in note_map.items():
            # Log ONLY the input notes for this patient
            if log_enabled:
                mb.log_text(
                    "SMTVariableValueMiner",
                    f"{pid}__notes.txt",
                    "\n---\n".join(notes or []),
                )

            if not notes:
                results[pid] = {v: None for v in leaf_detail}
                if log_enabled:
                    mb.log_json("SMTVariableValueMiner", f"{pid}__extraction", results[pid])
                continue

            full_var_map: Dict[str, Any] = {}

            for ci, var_names in enumerate(self._chunks(list(leaf_detail), batch_size), start=0):
                sub_leaf = {v: leaf_detail[v] for v in var_names}

                # Build and log the exact prompt for this chunk
                prompt_text = self._build_prompt(prompt_tpl, sub_leaf, notes)

                # Run LLM and log the raw output for this chunk
                raw = self._run_llm(
                    prompt_text,
                    pid=pid,
                    chunk_idx=ci,
                    mb=mb,
                    log=log_enabled,
                )

                # Parse extraction for this chunk
                sub_map = parse_patient_var_values(raw) or parse_smt_output(raw) or {}

                # ensure every var in this chunk has some entry
                for v in sub_leaf:
                    sub_map.setdefault(v, None)

                # keep track of everything we have collected so far
                full_var_map.update(sub_map)

            # validate enum values only once the full map is assembled
            self._validate_enums(full_var_map, leaf_detail)

            # guarantee *all* variables appear (even if missing in every chunk)
            for v in leaf_detail:
                full_var_map.setdefault(v, None)

            results[pid] = full_var_map

            # Log ONLY the final extraction JSON (output) for this patient
            if log_enabled:
                mb.log_json("SMTVariableValueMiner", f"{pid}__extraction", full_var_map)

        # if there’s only one patient id, store the inner dict directly
        context["patient_var_values"] = (
            next(iter(results.values())) if len(results) == 1 else results
        )
        return context
