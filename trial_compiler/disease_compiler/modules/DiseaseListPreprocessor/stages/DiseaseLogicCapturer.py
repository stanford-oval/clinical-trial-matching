# stages/DiseaseLogicCapturer.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json, pathlib, logging
import dspy

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

@dataclass
class LogicConfig:
    report_dir: pathlib.Path = pathlib.Path("entity_reports")
    # Optional inputs / prompts:
    # If present, we try to capture logic from these texts using an LLM.
    # Fallback is a safe deterministic OR-of-all.
    enable_llm_logic: bool = False
    llm_logic_prompt_key: str = "DiseaseLogicCapturer_prompt"
    # Where to look for textual clues if you want LLM mode
    logic_text_keys: tuple = ("criterion_text", "requirement_text", "metadata.logic", "metadata.disease_logic")

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _safe_id(i: int) -> str:
    return f"D{i+1:03d}"

def _expr_from_groups(groups: List[Dict[str, Any]]) -> str:
    """Build a readable infix expression from grouped ops."""
    # groups: [{"op":"OR","members":["D001","D002"]}, {"op":"NOT","members":["D003"]}]
    parts = []
    for g in groups:
        op = g.get("op","OR").upper()
        mem = g.get("members",[])
        if not mem: continue
        if op == "NOT":
            parts.append(" AND ".join(f"NOT({m})" for m in mem))
        else:
            glue = f" {op} "
            parts.append("(" + glue.join(mem) + ")")
    return " AND ".join(p for p in parts if p)

class DiseaseLogicCapturer(dspy.Module):
    """
    Captures Boolean logic across ctx['target_disease'] (AND/OR/NOT groups).

    Inputs:
      ctx['target_disease'] = [{"disease": "..."}...]
      Optional: ctx[llm_logic_prompt_key] + any of logic_text_keys

    Outputs:
      ctx['disease_logic'] = {
         "nodes": [{"id":"D001","label":"..."}],
         "groups": [{"op":"OR","members":["D001","D002"],"source":"deterministic|llm"}],
         "expr": "(D001 OR D002) AND NOT(D003)",
         "id_to_label": {"D001":"..."}
      }
    """

    def __init__(self, cfg: Optional[LogicConfig] = None, *, engine=None, verbose: bool = False):
        super().__init__()
        self.cfg = cfg or LogicConfig()
        self.engine = engine  # optional LLM callable
        self.verbose = verbose

    def _deterministic_logic(self, labels: List[str]) -> Dict[str, Any]:
        ids = [_safe_id(i) for i in range(len(labels))]
        nodes = [{"id": i, "label": l} for i, l in zip(ids, labels)]
        # Default: single OR group of all diseases (common eligibility pattern)
        groups = [{"op": "OR", "members": ids, "source": "deterministic"}] if ids else []
        expr = _expr_from_groups(groups)
        return {
            "nodes": nodes,
            "groups": groups,
            "expr": expr,
            "id_to_label": {i: l for i, l in zip(ids, labels)},
        }

    def _collect_logic_text(self, ctx: Dict[str, Any]) -> str:
        # Pull possible sources in priority order
        for key in self.cfg.logic_text_keys:
            # allow dotted path e.g. metadata.logic
            cur = ctx
            hit = None
            for part in str(key).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                    hit = cur
                else:
                    hit = None; break
            if isinstance(hit, str) and hit.strip():
                return hit
        # Fallback to contextual_text (entire trial pretty string)
        raw = ctx.get("contextual_text")
        return str(raw) if isinstance(raw, str) else ""

    def _llm_logic(self, labels: List[str], ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.engine: return None
        prompt_tpl = ctx.get(self.cfg.llm_logic_prompt_key, "")
        if not isinstance(prompt_tpl, str) or not prompt_tpl.strip():
            return None
        logic_text = self._collect_logic_text(ctx)
        payload = {
            "diseases": labels,
            "text": logic_text,
            "instruction": "Extract Boolean logic groups over the given disease list. "
                           "Return pure JSON with fields: groups=[{op:'OR|AND|NOT', members:[indexes], comment?}], "
                           "where indexes are 0-based into 'diseases'. No explanations outside JSON."
        }
        prompt = prompt_tpl.replace("#PAYLOAD#", json.dumps(payload, ensure_ascii=False, indent=2))
        raw = self.engine(prompt)
        out = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        if not isinstance(out, str): return None

        try:
            first, last = out.find("{"), out.rfind("}")
            if first != -1 and last != -1:
                out = out[first:last+1]
            data = json.loads(out)
            groups_spec = data.get("groups", [])
            if not isinstance(groups_spec, list): return None
            # Translate to id-based groups
            ids = [_safe_id(i) for i in range(len(labels))]
            groups = []
            for g in groups_spec:
                op = str(g.get("op","OR")).upper()
                idxs = g.get("members", [])
                if not isinstance(idxs, list): continue
                members = [ids[i] for i in idxs if isinstance(i, int) and 0 <= i < len(ids)]
                if not members: continue
                groups.append({"op": op, "members": members, "source": "llm", "comment": g.get("comment","")})
            expr = _expr_from_groups(groups)
            return {
                "nodes": [{"id": i, "label": l} for i, l in zip(ids, labels)],
                "groups": groups,
                "expr": expr,
                "id_to_label": {i: l for i, l in zip(ids, labels)},
            }
        except Exception:
            return None

    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        trial_id = ctx.get("trial_id") or "trial"
        td = ctx.get("target_disease") or []
        labels = [str(e.get("disease") or "").strip() for e in td if isinstance(e, dict)]
        labels = [x for x in labels if x]

        logic = None
        if self.cfg.enable_llm_logic:
            logic = self._llm_logic(labels, ctx)

        if logic is None:
            logic = self._deterministic_logic(labels)

        ctx["disease_logic"] = logic

        report = {
            "trial_id": trial_id,
            "counts": {
                "nodes": len(logic.get("nodes", [])),
                "groups": len(logic.get("groups", [])),
            },
            "expr": logic.get("expr",""),
            "groups": logic.get("groups", []),
        }
        ctx["logic_report"] = report
        try:
            _write_json(self.cfg.report_dir / f"{trial_id}_disease_logic.json", report)
        except Exception as e:
            _LOG.warning("Logic report write failed: %s", e)

        if self.verbose:
            _LOG.info("[Logic] %s", logic.get("expr",""))
        return ctx
