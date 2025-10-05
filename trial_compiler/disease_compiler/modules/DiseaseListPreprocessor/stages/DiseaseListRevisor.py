# stages/DiseaseListRevisor.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Iterable, Set
from dataclasses import dataclass
import json, pathlib, logging, re, unicodedata
import dspy

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------- config ----------
@dataclass
class RevisorConfig:
    alias_map_path: Optional[pathlib.Path] = None      # JSON: {"breast ca": "breast cancer", ...}
    synonym_keep_map_path: Optional[pathlib.Path] = None  # JSON: {"copd": "chronic obstructive pulmonary disease"}
    whitelist_path: Optional[pathlib.Path] = None      # newline terms (case-insensitive)
    blacklist_path: Optional[pathlib.Path] = None      # newline terms (case-insensitive)
    min_chars: int = 2
    drop_if_numeric_only: bool = True
    dedupe_casefold: bool = True
    keep_order: bool = True
    report_dir: pathlib.Path = pathlib.Path("entity_reports")
    # Optional LLM prompt keys in ctx (if you want to use LLM cleanup)
    llm_revision_prompt_key: str = "DiseaseListRevisor_prompt"
    enable_llm_revision: bool = False      # default off

# ---------- helpers ----------
def _read_json_map(p: Optional[pathlib.Path]):
    if not p: return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as e:
        _LOG.warning("Revisor: cannot read %s (%s)", p, e); return {}

def _read_list(p: Optional[pathlib.Path]) -> Set[str]:
    if not p: return set()
    try:
        return {ln.strip().casefold() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except Exception as e:
        _LOG.warning("Revisor: cannot read %s (%s)", p, e); return set()

def _unicode_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([/-])\s*", r"\1", s)
    return s

_NUM_RE = re.compile(r"^\d+([./-]\d+)*$")

def _basic_keep(s: str, min_chars: int, drop_numeric: bool) -> bool:
    if len(s) < min_chars: return False
    if drop_numeric and _NUM_RE.match(s): return False
    return True

def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        k = x.casefold()
        if k in seen: continue
        seen.add(k); out.append(x)
    return out

def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- module ----------
class DiseaseListRevisor(dspy.Module):
    """
    Revises ctx['target_disease'] items (normalize → alias/synonym → filters → dedupe).
    Optionally applies an LLM rewrite on top of deterministic rules.

    Inputs:
      ctx['target_disease'] = [{"disease": str, ...}, ...]
      ctx[config.llm_revision_prompt_key] (optional prompt text)

    Outputs:
      ctx['revisor_diff'] = [{"old": "...", "new": "..."}, ...]
      ctx['target_disease'] (updated; provenance += '|revised')
      ctx['diseases'] (single-quoted list string)
      ctx['revisor_report'] (counts, samples)
    """

    def __init__(self, cfg: Optional[RevisorConfig] = None, *, engine=None, verbose: bool = False):
        super().__init__()
        self.cfg = cfg or RevisorConfig()
        self.engine = engine  # optional LLM callable: prompt -> [str] or str
        self.verbose = verbose

        self.alias_map = {k.casefold(): v for k, v in _read_json_map(self.cfg.alias_map_path).items()}
        self.syn_keep = {k.casefold(): v for k, v in _read_json_map(self.cfg.synonym_keep_map_path).items()}
        self.whitelist = _read_list(self.cfg.whitelist_path)
        self.blacklist = _read_list(self.cfg.blacklist_path)

    # ---- deterministic pass ----
    def _revise_one(self, s: str) -> str:
        raw = s
        s = _unicode_norm(s)
        cf = s.casefold()
        # alias first
        if cf in self.alias_map:
            s = self.alias_map[cf]
            cf = s.casefold()
        # synonym preference (map to preferred term)
        if cf in self.syn_keep:
            s = self.syn_keep[cf]
        return s

    def _filters(self, s: str) -> bool:
        cf = s.casefold()
        if self.blacklist and cf in self.blacklist: return False
        if self.whitelist and cf not in self.whitelist: return False
        return _basic_keep(s, self.cfg.min_chars, self.cfg.drop_if_numeric_only)

    # ---- optional LLM pass ----
    def _llm_rewrite_batch(self, items: List[str], prompt_tpl: str) -> List[str]:
        if not items: return []
        # Expect the prompt to contain a placeholder #ITEMS# -> JSON list
        prompt = prompt_tpl.replace("#ITEMS#", json.dumps(items, ensure_ascii=False, indent=2))
        out = None
        if callable(self.engine):
            raw = self.engine(prompt)
            out = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        if not isinstance(out, str):
            return items
        # Try to parse a JSON list back
        try:
            first, last = out.find("["), out.rfind("]")
            if first != -1 and last != -1:
                out = out[first:last+1]
            parsed = json.loads(out)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [x.strip() for x in parsed]
        except Exception:
            pass
        return items

    # ---- forward ----
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        trial_id = ctx.get("trial_id") or "trial"
        td = ctx.get("target_disease") or []
        items = [str(e.get("disease") or "").strip() for e in td if isinstance(e, dict)]
        items = [x for x in items if x]

        stage_a = [_unicode_norm(x) for x in items]
        stage_b = [self._revise_one(x) for x in stage_a]
        stage_c = [x for x in stage_b if self._filters(x)]
        stage_d = _dedupe_preserve_order(stage_c) if self.cfg.dedupe_casefold else list(dict.fromkeys(stage_c))

        # Optional LLM revision
        final_items = stage_d
        if self.cfg.enable_llm_revision and self.engine:
            prompt_key = self.cfg.llm_revision_prompt_key
            tpl = ctx.get(prompt_key, "")
            if isinstance(tpl, str) and tpl.strip():
                final_items = self._llm_rewrite_batch(stage_d, tpl)
                # re-run minimal clean/dedupe after LLM
                final_items = [_unicode_norm(x) for x in final_items if self._filters(x)]
                final_items = _dedupe_preserve_order(final_items)

        # diff & writeback
        diffs: List[Dict[str, str]] = []
        old_map = {i: o for i, o in enumerate(items)}
        for i, newv in enumerate(final_items):
            oldv = old_map.get(i)
            if oldv and oldv != newv:
                diffs.append({"old": oldv, "new": newv})

        ctx["revisor_diff"] = diffs
        ctx["target_disease"] = [{"disease": s, "provenance": (e.get("provenance") if i < len(td) and isinstance((e:=td[i]), dict) else "preproc") + "|revised"} for i, s in enumerate(final_items)]
        ctx["diseases"] = "[" + ", ".join(f"'{x}'" for x in final_items) + "]"

        report = {
            "trial_id": trial_id,
            "counts": {
                "input": len(items),
                "stage_a_norm": len(stage_a),
                "stage_b_revised": len(stage_b),
                "stage_c_after_filters": len(stage_c),
                "final": len(final_items),
                "diffs": len(diffs),
            },
            "samples": {
                "input_head": items[:20],
                "final": final_items,
                "diffs": diffs[:20],
            },
            "config": {
                "min_chars": self.cfg.min_chars,
                "dedupe_casefold": self.cfg.dedupe_casefold,
                "has_whitelist": bool(self.whitelist),
                "has_blacklist": bool(self.blacklist),
                "enable_llm_revision": self.cfg.enable_llm_revision,
            },
        }
        ctx["revisor_report"] = report
        try:
            _write_json(self.cfg.report_dir / f"{trial_id}_disease_revisor.json", report)
        except Exception as e:
            _LOG.warning("Revisor report write failed: %s", e)

        if self.verbose:
            _LOG.info("[Revisor] final: %d", len(final_items))
        return ctx
