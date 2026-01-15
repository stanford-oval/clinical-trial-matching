from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import dspy

from ...utils.mbench import get_mbench, mbench_enabled


_SCOPE_BLOCK_RE = re.compile(
    r"<variable_scope_values>(.*?)</variable_scope_values>",
    re.DOTALL | re.IGNORECASE,
)

_STAGE_CACHE_SCHEMA_VERSION = "2026-03-17-stage-cache-v1"


def _split_stem_qual(var_name: str) -> Tuple[str, Optional[str]]:
    s = str(var_name)
    if "@@" not in s:
        return s, None
    stem, qual = s.split("@@", 1)
    stem = stem.strip()
    qual = qual.strip()
    return stem, qual or None


def _coerce_scope_entry(entry: Any) -> Dict[str, str]:
    if isinstance(entry, dict):
        scope = str(entry.get("scope", "") or "").strip().lower()
        reason = str(entry.get("reason", "") or "").strip()
    else:
        scope = str(entry or "").strip().lower()
        reason = ""

    if scope not in {"in_scope", "projected_away"}:
        scope = "projected_away"

    return {"scope": scope, "reason": reason}


def parse_variable_scope_values(raw: str | None) -> Dict[str, Dict[str, str]]:
    if not raw:
        return {}

    m = _SCOPE_BLOCK_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()

    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()

    try:
        data = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for var, entry in data.items():
        out[str(var)] = _coerce_scope_entry(entry)
    return out


def _json_default(o):
    import datetime as _dt
    import re as _re
    from pathlib import Path as _Path

    if isinstance(o, set):
        try:
            return sorted(o)
        except Exception:
            return list(o)
    if isinstance(o, _Path):
        return str(o)
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if isinstance(o, _re.Pattern):
        return o.pattern
    return str(o)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_json_default)


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(obj).encode("utf-8")).hexdigest()


def _stage_cache_root(context: Dict[str, Any]) -> Path:
    root = context.get("PROMPT_CACHE_ROOT") or "./_prompt_cache"
    return Path(root)


def _stage_cache_file(context: Dict[str, Any], stage: str, fingerprint: str) -> Path:
    return _stage_cache_root(context) / stage / f"{fingerprint}.json"


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_write_json_atomic(path: Path, obj: Any) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        tmp.replace(path)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _cache_payload_matches(payload: Optional[Dict[str, Any]], fingerprint: str) -> bool:
    if not isinstance(payload, dict):
        return False
    cache = payload.get("cache")
    if not isinstance(cache, dict):
        return False
    return (
        cache.get("schema_version") == _STAGE_CACHE_SCHEMA_VERSION
        and cache.get("fingerprint") == fingerprint
    )


def _scope_miner_fingerprint(
    *,
    leaf_detail: Dict[str, Dict[str, Any]],
    prompt_text: str,
    model_name: str,
) -> str:
    payload = {
        "schema_version": _STAGE_CACHE_SCHEMA_VERSION,
        "stage": "scope_miner",
        "model_name": model_name,
        "prompt_text": prompt_text,
        "leaf_detail": leaf_detail,
    }
    return _stable_hash(payload)


class SMTVariableScopeMiner(dspy.Module):
    """
    LLM-based variable scope filter.

    Output:
      context["variable_scope_map"] = {
        var_name: {
          "scope": "in_scope" | "projected_away",
          "reason": "..."
        }
      }

    Important behavior:
      - projected_away variables are automatically removed from context["leaf_detail"]
      - this module does NOT write meta["scope"] / meta["scope_reason"] into leaf_detail
      - downstream prompts therefore only see in-scope variables unless they explicitly
        inspect context["variable_scope_map"]

    NEW:
      - stage-level cache
      - freeze support
      - default freeze is controlled upstream by SMTMatcher
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @staticmethod
    def _group_by_stem(leaf_detail: Dict[str, Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
        groups: Dict[str, List[str]] = {}
        for v in leaf_detail:
            stem, _q = _split_stem_qual(v)
            groups.setdefault(stem, []).append(v)

        out: List[Tuple[str, List[str]]] = []
        for stem in sorted(groups.keys()):
            vars_in_group = groups[stem]
            stem_var = stem if stem in vars_in_group else None
            quals = sorted([x for x in vars_in_group if x != stem_var])
            ordered = ([stem_var] if stem_var else []) + quals
            out.append((stem, ordered))
        return out

    @staticmethod
    def _format_definition(meta: Dict[str, Any]) -> str:
        d = meta.get("definition")
        if isinstance(d, dict):
            meaning = str(d.get("meaning") or "").strip()
            if meaning:
                return meaning
        return str(meta.get("description") or "").strip()

    @classmethod
    def _format_var_list(cls, leaf_detail: Dict[str, Dict[str, Any]]) -> str:
        groups = cls._group_by_stem(leaf_detail)

        lines: List[str] = []
        lines.append(
            "IMPORTANT:\n"
            "- Classify each variable as either 'in_scope' or 'projected_away'.\n"
            "- 'in_scope' means the variable constrains the patient's current or past clinical state in a meaningful way.\n"
            "- 'projected_away' means the variable primarily concerns a dimension we do not care about for this task.\n"
            "- Dimensions that are usually projected away include: documentation status, confirmation strength, source procedure, discovery pathway, assessment pathway, logistics, willingness, scheduling, geography, site, clinic membership, care setting, hospitalization, and existing clinical-trial enrollment.\n"
            "- Apply this to BOTH stem variables and qualifier variables.\n"
            "- For qualifiers, judge the qualifier's own dimension. A stem can be in_scope while a qualifier is projected_away.\n"
            "- Use the provided meaning rather than guessing from the variable name string.\n"
            "- If unsure, prefer 'projected_away' unless the variable clearly constrains substantive patient state."
        )

        for stem, vars_in_group in groups:
            if stem in leaf_detail:
                meta = leaf_detail[stem]
                desc = cls._format_definition(meta)
                ty = meta.get("type", "<?>")
                lines.append(f"- name: {stem}")
                lines.append(f"  type: {ty}")
                lines.append(f"  meaning: {desc or '(missing)'}")
            else:
                lines.append(f"- name: {stem}")
                lines.append("  type: <?>")
                lines.append("  meaning: (stem not present; only qualifiers present)")

            quals = [v for v in vars_in_group if v != stem]
            if quals:
                lines.append("  qualifiers:")
                for qv in quals:
                    qmeta = leaf_detail.get(qv, {})
                    qdesc = cls._format_definition(qmeta)
                    qty = qmeta.get("type", "<?>")
                    lines.append(f"    - name: {qv}")
                    lines.append(f"      type: {qty}")
                    lines.append(f"      meaning: {qdesc or '(missing)'}")

            lines.append("")

        return "\n".join(lines).rstrip()

    def _build_prompt(self, prompt_tpl: str, leaf_detail: Dict[str, Dict[str, Any]]) -> str:
        return prompt_tpl.replace("{{VARIABLE_LIST}}", self._format_var_list(leaf_detail)).strip()

    @staticmethod
    def _filter_in_scope(
        leaf_detail: Dict[str, Dict[str, Any]],
        scope_map: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        return {
            v: meta
            for v, meta in leaf_detail.items()
            if scope_map.get(v, {}).get("scope") == "in_scope"
        }

    @staticmethod
    def _all_in_scope_map(
        leaf_detail: Dict[str, Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Dict[str, str]]:
        return {
            v: {"scope": "in_scope", "reason": reason}
            for v in leaf_detail
        }

    @staticmethod
    def _all_projected_away_map(
        leaf_detail: Dict[str, Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Dict[str, str]]:
        return {
            v: {"scope": "projected_away", "reason": reason}
            for v in leaf_detail
        }

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        leaf_detail = context.get("leaf_detail", {}) or {}
        prompt_tpl = (context.get("SMTVariableScopeMiner_prompt", "") or "").strip()

        if not leaf_detail:
            context["variable_scope_map"] = {}
            context["leaf_detail"] = {}
            return context

        if not prompt_tpl:
            # Conservative fallback when prompt is missing.
            scope_map = self._all_projected_away_map(
                leaf_detail,
                "scope miner prompt missing",
            )
            context["variable_scope_map"] = scope_map
            context["leaf_detail"] = {}
            return context

        mb = get_mbench(context)
        log_enabled = mbench_enabled(context)

        prompt_text = self._build_prompt(prompt_tpl, leaf_detail)

        freeze = bool(context.get("FREEZE_SCOPE_MINER", False))
        use_cache = bool(context.get("CACHE_SCOPE_MINER", True))
        miss_policy = str(context.get("FROZEN_SCOPE_MISS_POLICY", "all_in_scope") or "all_in_scope").strip().lower()
        model_name = str(
            context.get("MODEL_NAME")
            or context.get("model_name")
            or context.get("OPENAI_MODEL")
            or "unknown"
        )

        fingerprint = _scope_miner_fingerprint(
            leaf_detail=leaf_detail,
            prompt_text=prompt_text,
            model_name=model_name,
        )
        cache_path = _stage_cache_file(context, "scope_miner", fingerprint)

        # Cache replay
        if use_cache:
            cached = _safe_read_json(cache_path)
            if _cache_payload_matches(cached, fingerprint):
                payload = cached.get("payload") or {}
                scope_map = payload.get("scope_map") or {}
                if isinstance(scope_map, dict):
                    for v in leaf_detail:
                        scope_map.setdefault(
                            v,
                            {"scope": "projected_away", "reason": "missing from cached scope output"},
                        )
                    filtered_leaf_detail = self._filter_in_scope(leaf_detail, scope_map)
                    context["variable_scope_map"] = scope_map
                    context["leaf_detail"] = filtered_leaf_detail
                    context.setdefault("stage_cache_meta", {})["scope_miner"] = {
                        "used": True,
                        "fingerprint": fingerprint,
                        "path": str(cache_path),
                        "frozen": freeze,
                    }
                    if log_enabled:
                        mb.log_json("SMTVariableScopeMiner", "scope_map.json", scope_map)
                        mb.log_json(
                            "SMTVariableScopeMiner",
                            "filtered_leaf_vars.json",
                            {"kept": sorted(filtered_leaf_detail.keys())},
                        )
                    return context

        # Frozen mode: do not call model
        if freeze:
            if miss_policy == "all_projected_away":
                scope_map = self._all_projected_away_map(
                    leaf_detail,
                    "scope miner frozen and cache miss",
                )
            else:
                scope_map = self._all_in_scope_map(
                    leaf_detail,
                    "scope miner frozen and cache miss",
                )

            filtered_leaf_detail = self._filter_in_scope(leaf_detail, scope_map)
            context["variable_scope_map"] = scope_map
            context["leaf_detail"] = filtered_leaf_detail
            context.setdefault("stage_cache_meta", {})["scope_miner"] = {
                "used": False,
                "frozen_cache_miss": True,
                "fingerprint": fingerprint,
                "path": str(cache_path),
                "miss_policy": miss_policy,
            }

            if log_enabled:
                mb.log_json("SMTVariableScopeMiner", "scope_map.json", scope_map)
                mb.log_json(
                    "SMTVariableScopeMiner",
                    "filtered_leaf_vars.json",
                    {"kept": sorted(filtered_leaf_detail.keys())},
                )
            return context

        # Live model call
        if log_enabled:
            mb.log_text("SMTVariableScopeMiner", "prompt.txt", prompt_text)

        resp = self.engine(prompt_text)
        raw = resp[0] if resp else ""

        if log_enabled:
            mb.log_text("SMTVariableScopeMiner", "raw.txt", raw)

        scope_map = parse_variable_scope_values(raw)

        # Default any missing variable to projected_away.
        for v in leaf_detail:
            scope_map.setdefault(
                v,
                {"scope": "projected_away", "reason": "missing from scope miner output"}
            )

        filtered_leaf_detail = self._filter_in_scope(leaf_detail, scope_map)

        context["variable_scope_map"] = scope_map
        context["leaf_detail"] = filtered_leaf_detail
        context.setdefault("stage_cache_meta", {})["scope_miner"] = {
            "used": False,
            "fingerprint": fingerprint,
            "path": str(cache_path),
            "frozen": False,
        }

        if use_cache:
            _safe_write_json_atomic(
                cache_path,
                {
                    "cache": {
                        "schema_version": _STAGE_CACHE_SCHEMA_VERSION,
                        "fingerprint": fingerprint,
                        "stage": "scope_miner",
                        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                    "payload": {
                        "scope_map": scope_map,
                    },
                },
            )

        if log_enabled:
            mb.log_json("SMTVariableScopeMiner", "scope_map.json", scope_map)
            mb.log_json(
                "SMTVariableScopeMiner",
                "filtered_leaf_vars.json",
                {"kept": sorted(filtered_leaf_detail.keys())},
            )

        return context