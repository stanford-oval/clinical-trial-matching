from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import dspy

from ...utils.mbench import get_mbench, mbench_enabled


_PROJ_BLOCK_RE = re.compile(
    r"<variable_projection_rewrites>(.*?)</variable_projection_rewrites>",
    re.DOTALL | re.IGNORECASE,
)

_SNAKE_RE = re.compile(r"[^a-z0-9_]+")

_STAGE_CACHE_SCHEMA_VERSION = "2026-03-17-stage-cache-v1"


def _split_stem_qual(var_name: str) -> Tuple[str, Optional[str]]:
    s = str(var_name)
    if "@@" not in s:
        return s, None
    stem, qual = s.split("@@", 1)
    stem = stem.strip()
    qual = qual.strip()
    return stem, qual or None


def _normalize_snake_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(" ", "_")
    s = _SNAKE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_json_object(raw: str) -> Optional[str]:
    if not raw:
        return None

    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()

    m = _PROJ_BLOCK_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
    return None


def parse_variable_projection_rewrites(raw: str | None) -> Dict[str, Dict[str, str]]:
    if not raw:
        return {}

    js = _extract_json_object(raw)
    if js is None:
        return {}

    try:
        obj = json.loads(js)
    except Exception:
        return {}

    if not isinstance(obj, dict):
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for original_var, payload in obj.items():
        if not isinstance(payload, dict):
            continue

        out[str(original_var)] = {
            "rewritten_variable_name": str(payload.get("rewritten_variable_name") or "").strip(),
            "meaning": str(payload.get("meaning") or "").strip(),
            "projection_summary": str(payload.get("projection_summary") or "").strip(),
            "projection_status": str(payload.get("projection_status") or "").strip().lower(),
        }
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


def _projection_rewriter_fingerprint(
    *,
    leaf_detail: Dict[str, Dict[str, Any]],
    prompt_text: str,
    trial_context: str,
    model_name: str,
) -> str:
    payload = {
        "schema_version": _STAGE_CACHE_SCHEMA_VERSION,
        "stage": "projection_rewriter",
        "model_name": model_name,
        "prompt_text": prompt_text,
        "trial_context": trial_context,
        "leaf_detail": leaf_detail,
    }
    return _stable_hash(payload)


class SMTVariableProjectionRewriter(dspy.Module):
    """
    Post-scope projection rewrite stage.

    Input:
      - context["leaf_detail"] : scope-filtered active vars only
      - context["SMTVariableProjectionRewriter_prompt"]

    Output:
      - context["variable_projection_map"] : original_var -> rewrite payload
      - context["projection_alias_to_original"] : alias -> original_var
      - context["projection_original_to_alias"] : original_var -> alias
      - context["leaf_detail"][var]["projection"] populated
      - context["projection_alias_collisions"] for debugging

    Key invariants:
      - aliases are what downstream miner sees
      - aliases must be unique
      - qualifier aliases must inherit their rewritten stem alias

    NEW:
      - stage-level cache
      - freeze support
      - default freeze is controlled upstream by SMTMatcher
    """

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @staticmethod
    def _format_definition(meta: Dict[str, Any]) -> str:
        d = meta.get("definition")
        if isinstance(d, dict):
            meaning = str(d.get("meaning") or "").strip()
            if meaning:
                return meaning
        return str(meta.get("description") or "").strip()

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

    @classmethod
    def _build_variable_groups_json(cls, leaf_detail: Dict[str, Dict[str, Any]]) -> str:
        groups = cls._group_by_stem(leaf_detail)

        payload: List[Dict[str, Any]] = []
        for stem, vars_in_group in groups:
            group_obj: Dict[str, Any] = {"stem": stem, "variables": []}
            for v in vars_in_group:
                meta = leaf_detail.get(v, {})
                var_obj = {
                    "original_variable_name": v,
                    "type": meta.get("type"),
                    "meaning": cls._format_definition(meta),
                    "enum_values": meta.get("enum_values"),
                    "threshold_helper": bool(meta.get("threshold_helper")),
                    "threshold_parent": meta.get("threshold_parent"),
                    "threshold_parent_type": meta.get("threshold_parent_type"),
                    "threshold_op": meta.get("threshold_op"),
                    "threshold_value": meta.get("threshold_value"),
                    "threshold_expr": meta.get("threshold_expr"),
                }
                group_obj["variables"].append(var_obj)
            payload.append(group_obj)

        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_trial_context(context: Dict[str, Any]) -> str:
        chunks: List[str] = []

        trial_id = str(context.get("trial_id") or "").strip()
        inc_exc = str(context.get("inc_exc") or "").strip()
        if trial_id:
            chunks.append(f"trial_id: {trial_id}")
        if inc_exc:
            chunks.append(f"side: {inc_exc}")

        bundles = context.get("requirement_bundles")
        if bundles:
            try:
                chunks.append("requirement_bundles:\n" + json.dumps(bundles, ensure_ascii=False, indent=2))
            except Exception:
                chunks.append("requirement_bundles:\n" + str(bundles))

        smt_lines = context.get("smt_program_lines") or []
        if smt_lines:
            smt_text = "\n".join(str(x) for x in smt_lines)
            chunks.append("smt_program:\n" + smt_text)

        return "\n\n".join(chunks).strip()

    @classmethod
    def _identity_projection_map(cls, leaf_detail: Dict[str, Dict[str, Any]], reason: str = "") -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for v, meta in leaf_detail.items():
            out[v] = {
                "rewritten_variable_name": v,
                "meaning": cls._format_definition(meta),
                "projection_summary": reason,
                "projection_status": "unchanged",
            }
        return out

    @staticmethod
    def _normalize_status(status: str, *, original_var: str, alias: str, original_meaning: str, rewritten_meaning: str) -> str:
        s = str(status or "").strip().lower()
        if s in {"unchanged", "partially_projected"}:
            return s
        return "unchanged" if (alias == original_var and rewritten_meaning == original_meaning) else "partially_projected"

    @staticmethod
    def _extract_qual_suffix_from_alias(alias: str, original_var: str) -> str:
        """
        For qualifier vars, we only trust the qualifier suffix semantics.
        If the model returned a full alias with @@, keep only the qualifier suffix.
        If not, treat the whole alias as the qualifier suffix.
        """
        _orig_stem, orig_qual = _split_stem_qual(original_var)
        if orig_qual is None:
            return _normalize_snake_name(alias)

        alias = (alias or "").strip()
        if "@@" in alias:
            _a_stem, a_qual = _split_stem_qual(alias)
            return _normalize_snake_name(a_qual or orig_qual)
        return _normalize_snake_name(alias or orig_qual)

    @classmethod
    def _enforce_group_consistency(
        cls,
        leaf_detail: Dict[str, Dict[str, Any]],
        parsed: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, str]]:
        """
        Enforce:
          - stem alias chosen once per stem group
          - each qualifier alias = rewritten_stem_alias + "@@" + rewritten_qual_suffix
        """
        original_map = cls._identity_projection_map(leaf_detail)
        original_meanings = {v: original_map[v]["meaning"] for v in original_map}

        final_map: Dict[str, Dict[str, str]] = {}

        for stem, vars_in_group in cls._group_by_stem(leaf_detail):
            stem_var = stem if stem in vars_in_group else None

            # 1) choose rewritten stem alias
            if stem_var is not None:
                stem_payload = parsed.get(stem_var) or {}
                proposed_stem_alias = _normalize_snake_name(
                    stem_payload.get("rewritten_variable_name") or stem_var
                ) or stem_var
                stem_meaning = str(stem_payload.get("meaning") or "").strip() or original_meanings[stem_var]
                stem_summary = str(stem_payload.get("projection_summary") or "").strip()
                stem_status = cls._normalize_status(
                    str(stem_payload.get("projection_status") or ""),
                    original_var=stem_var,
                    alias=proposed_stem_alias,
                    original_meaning=original_meanings[stem_var],
                    rewritten_meaning=stem_meaning,
                )
            else:
                proposed_stem_alias = stem
                stem_meaning = ""
                stem_summary = ""
                stem_status = "unchanged"

            if stem_var is not None:
                final_map[stem_var] = {
                    "rewritten_variable_name": proposed_stem_alias,
                    "meaning": stem_meaning,
                    "projection_summary": stem_summary,
                    "projection_status": stem_status,
                }

            # 2) qualifiers inherit rewritten stem alias
            for v in vars_in_group:
                if v == stem_var:
                    continue

                payload = parsed.get(v) or {}
                proposed_alias = str(payload.get("rewritten_variable_name") or "").strip()
                qual_suffix = cls._extract_qual_suffix_from_alias(proposed_alias, v)

                if not qual_suffix:
                    _orig_stem, orig_qual = _split_stem_qual(v)
                    qual_suffix = _normalize_snake_name(orig_qual or "qualifier") or "qualifier"

                full_alias = f"{proposed_stem_alias}@@{qual_suffix}"

                rewritten_meaning = str(payload.get("meaning") or "").strip() or original_meanings[v]
                rewritten_summary = str(payload.get("projection_summary") or "").strip()
                rewritten_status = cls._normalize_status(
                    str(payload.get("projection_status") or ""),
                    original_var=v,
                    alias=full_alias,
                    original_meaning=original_meanings[v],
                    rewritten_meaning=rewritten_meaning,
                )

                final_map[v] = {
                    "rewritten_variable_name": full_alias,
                    "meaning": rewritten_meaning,
                    "projection_summary": rewritten_summary,
                    "projection_status": rewritten_status,
                }

        # fill any strange omissions conservatively
        for v in leaf_detail:
            final_map.setdefault(
                v,
                {
                    "rewritten_variable_name": v,
                    "meaning": original_meanings.get(v, ""),
                    "projection_summary": "missing from projection rewriter output; kept unchanged",
                    "projection_status": "unchanged",
                },
            )

        return final_map

    @staticmethod
    def _dedupe_aliases(
        projection_map: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        """
        Aliases must be unique because downstream miner only sees aliases.

        If collisions occur, minimally disambiguate by appending __dup2, __dup3, ...
        """
        alias_to_originals: Dict[str, List[str]] = {}
        for orig, payload in projection_map.items():
            alias = str(payload.get("rewritten_variable_name") or "").strip()
            alias_to_originals.setdefault(alias, []).append(orig)

        collisions = {a: xs for a, xs in alias_to_originals.items() if len(xs) > 1}
        if not collisions:
            return projection_map, {}

        fixed = dict(projection_map)
        for alias, origs in collisions.items():
            for idx, orig in enumerate(sorted(origs), start=1):
                if idx == 1:
                    continue
                fixed_alias = f"{alias}__dup{idx}"
                fixed[orig] = dict(fixed[orig])
                fixed[orig]["rewritten_variable_name"] = fixed_alias
                prev = str(fixed[orig].get("projection_summary") or "").strip()
                note = f"Alias collision resolved by internal suffixing from {alias} to {fixed_alias}"
                fixed[orig]["projection_summary"] = f"{prev} | {note}" if prev else note

        return fixed, collisions

    @staticmethod
    def _build_alias_maps(projection_map: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
        alias_to_original: Dict[str, str] = {}
        original_to_alias: Dict[str, str] = {}
        for original_var, payload in projection_map.items():
            alias = str(payload.get("rewritten_variable_name") or "").strip()
            if not alias:
                alias = original_var
            alias_to_original[alias] = original_var
            original_to_alias[original_var] = alias
        return alias_to_original, original_to_alias

    def _apply_projection_outputs(
        self,
        context: Dict[str, Any],
        projection_map: Dict[str, Dict[str, str]],
        collisions: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        alias_to_original, original_to_alias = self._build_alias_maps(projection_map)

        context["variable_projection_map"] = projection_map
        context["projection_alias_to_original"] = alias_to_original
        context["projection_original_to_alias"] = original_to_alias
        context["projection_alias_collisions"] = collisions

        for v, payload in projection_map.items():
            context["leaf_detail"][v]["projection"] = payload

        return context

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        leaf_detail = context.get("leaf_detail", {}) or {}
        prompt_tpl = (context.get("SMTVariableProjectionRewriter_prompt", "") or "").strip()

        if not leaf_detail:
            context["variable_projection_map"] = {}
            context["projection_alias_to_original"] = {}
            context["projection_original_to_alias"] = {}
            context["projection_alias_collisions"] = {}
            return context

        mb = get_mbench(context)
        log_enabled = mbench_enabled(context)

        if not prompt_tpl:
            projection_map = self._identity_projection_map(
                leaf_detail,
                reason="projection rewriter prompt missing; kept unchanged",
            )
            context = self._apply_projection_outputs(context, projection_map, {})
            return context

        trial_context = self._build_trial_context(context)
        variable_groups_json = self._build_variable_groups_json(leaf_detail)

        prompt = (
            prompt_tpl
            .replace("{{TRIAL_CONTEXT}}", trial_context)
            .replace("{{VARIABLE_GROUPS_JSON}}", variable_groups_json)
            .strip()
        )

        freeze = bool(context.get("FREEZE_PROJECTION_REWRITER", False))
        use_cache = bool(context.get("CACHE_PROJECTION_REWRITER", True))
        miss_policy = str(context.get("FROZEN_PROJECTION_MISS_POLICY", "identity") or "identity").strip().lower()
        model_name = str(
            context.get("MODEL_NAME")
            or context.get("model_name")
            or context.get("OPENAI_MODEL")
            or "unknown"
        )

        fingerprint = _projection_rewriter_fingerprint(
            leaf_detail=leaf_detail,
            prompt_text=prompt,
            trial_context=trial_context,
            model_name=model_name,
        )
        cache_path = _stage_cache_file(context, "projection_rewriter", fingerprint)

        # Cache replay
        if use_cache:
            cached = _safe_read_json(cache_path)
            if _cache_payload_matches(cached, fingerprint):
                payload = cached.get("payload") or {}
                projection_map = payload.get("projection_map") or {}
                collisions = payload.get("collisions") or {}
                if isinstance(projection_map, dict):
                    context = self._apply_projection_outputs(context, projection_map, collisions if isinstance(collisions, dict) else {})
                    context.setdefault("stage_cache_meta", {})["projection_rewriter"] = {
                        "used": True,
                        "fingerprint": fingerprint,
                        "path": str(cache_path),
                        "frozen": freeze,
                    }
                    if log_enabled:
                        mb.log_json("SMTVariableProjectionRewriter", "projection_map.json", projection_map)
                        mb.log_json("SMTVariableProjectionRewriter", "alias_to_original.json", context.get("projection_alias_to_original") or {})
                        mb.log_json("SMTVariableProjectionRewriter", "alias_collisions.json", context.get("projection_alias_collisions") or {})
                    return context

        # Frozen mode: do not call model
        if freeze:
            if miss_policy == "identity":
                projection_map = self._identity_projection_map(
                    leaf_detail,
                    reason="projection rewriter frozen and cache miss; kept unchanged",
                )
            else:
                projection_map = self._identity_projection_map(
                    leaf_detail,
                    reason=f"projection rewriter frozen and cache miss (unsupported miss policy={miss_policy}); kept unchanged",
                )

            context = self._apply_projection_outputs(context, projection_map, {})
            context.setdefault("stage_cache_meta", {})["projection_rewriter"] = {
                "used": False,
                "frozen_cache_miss": True,
                "fingerprint": fingerprint,
                "path": str(cache_path),
                "miss_policy": miss_policy,
            }

            if log_enabled:
                mb.log_json("SMTVariableProjectionRewriter", "projection_map.json", projection_map)
                mb.log_json("SMTVariableProjectionRewriter", "alias_to_original.json", context.get("projection_alias_to_original") or {})
                mb.log_json("SMTVariableProjectionRewriter", "alias_collisions.json", {})
            return context

        # Live model call
        if log_enabled:
            mb.log_text("SMTVariableProjectionRewriter", "prompt.txt", prompt)

        resp = self.engine(prompt)
        raw = resp[0] if resp else ""

        if log_enabled:
            mb.log_text("SMTVariableProjectionRewriter", "raw.txt", raw)

        parsed = parse_variable_projection_rewrites(raw)
        projection_map = self._enforce_group_consistency(leaf_detail, parsed)
        projection_map, collisions = self._dedupe_aliases(projection_map)

        context = self._apply_projection_outputs(context, projection_map, collisions)
        context.setdefault("stage_cache_meta", {})["projection_rewriter"] = {
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
                        "stage": "projection_rewriter",
                        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                    "payload": {
                        "projection_map": projection_map,
                        "collisions": collisions,
                    },
                },
            )

        if log_enabled:
            mb.log_json("SMTVariableProjectionRewriter", "projection_map.json", projection_map)
            mb.log_json("SMTVariableProjectionRewriter", "alias_to_original.json", context.get("projection_alias_to_original") or {})
            mb.log_json("SMTVariableProjectionRewriter", "alias_collisions.json", collisions)

        return context