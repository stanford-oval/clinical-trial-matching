# modules/DiseaseListPreprocessor.py
from __future__ import annotations
from typing import Dict, Any, Iterable, List, Optional, Tuple, Set
from dataclasses import dataclass
import json, pathlib, logging, unicodedata, re
import dspy

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ───────────────────── I/O helpers ─────────────────────
def _write_text(p: pathlib.Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")


def _write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ───────────────────── Config ─────────────────────
@dataclass
class DiseasePreprocConfig:
    # resource files (all optional)
    synonym_map_path: Optional[pathlib.Path] = None     # JSON: {"copd": ["chronic obstructive pulmonary disease"], ...}
    alias_map_path: Optional[pathlib.Path] = None       # JSON: {"breast ca": "breast cancer", ...}
    whitelist_path: Optional[pathlib.Path] = None       # newline list, case-insensitive
    blacklist_path: Optional[pathlib.Path] = None       # newline list, case-insensitive

    # normalization/filter knobs
    min_chars: int = 2
    drop_if_numeric_only: bool = True
    dedupe_casefold: bool = True
    keep_order: bool = True

    # IMPORTANT:
    # For curated disease lists (like processed_disease_list), entries should be atomic.
    # Do NOT split on commas by default, because strings like:
    #   "Pulmonary Disease, Chronic Obstructive"
    #   "Kidney Failure, Chronic"
    # are single disease names, not multiple diseases.
    split_curated_entries: bool = False

    # If splitting is ever enabled, only use semicolon or slash-like separators.
    # We intentionally DO NOT split on commas.
    split_pattern: str = r"[;/|]"

    # report output
    report_dir: pathlib.Path = pathlib.Path("entity_reports")


# ───────────────────── Main module ─────────────────────
class DiseaseListPreprocessor(dspy.Module):
    """
    Disease list preprocessor.

    Input context (flexible):
        ctx["processed_disease_list"]  — curated list[str] from upstream extractor/eliminator

    Output (added/updated):
        ctx["raw_disease_candidates"]  — flat list[str] (pre-normalized tokens; atomic by default)
        ctx["target_disease"]          — list[{"disease": str, "provenance": "preproc"}]
        ctx["diseases"]                — display string: "['a', 'b', ...]"
        ctx["preproc_report"]          — dict with steps, counts, and samples

    Notes:
      - This module now STRICTLY uses ctx["processed_disease_list"].
      - It does NOT re-extract diseases from trial text/context.
      - It does NOT split comma-containing disease names by default.
    """

    _RE_NUM = re.compile(r"^\d+([./-]\d+)*$")

    def __init__(self, cfg: Optional[DiseasePreprocConfig] = None, *, verbose: bool = False):
        super().__init__()
        self.cfg = cfg or DiseasePreprocConfig()
        self.verbose = verbose

        self._alias_map: Dict[str, str] = self._load_json_map(self.cfg.alias_map_path)
        self._syn_map: Dict[str, List[str]] = self._load_json_map(self.cfg.synonym_map_path, list_map=True)
        self._whitelist: Set[str] = self._load_list(self.cfg.whitelist_path)
        self._blacklist: Set[str] = self._load_list(self.cfg.blacklist_path)

        self._re_split = re.compile(self.cfg.split_pattern)

    # ─────────────── utilities: load resources ───────────────
    @staticmethod
    def _load_json_map(p: Optional[pathlib.Path], list_map: bool = False):
        if not p:
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            if not list_map:
                return {
                    str(k).strip().casefold(): str(v).strip()
                    for k, v in raw.items()
                    if v is not None and str(v).strip()
                }
            out: Dict[str, List[str]] = {}
            for k, v in raw.items():
                if isinstance(v, list):
                    out[str(k).strip().casefold()] = [
                        str(s).strip() for s in v if str(s).strip()
                    ]
            return out
        except Exception as e:
            _LOG.warning("Failed to read JSON map %s: %s", p, e)
            return {}

    @staticmethod
    def _load_list(p: Optional[pathlib.Path]) -> Set[str]:
        if not p:
            return set()
        try:
            items = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
            return {it.casefold() for it in items if it}
        except Exception as e:
            _LOG.warning("Failed to read list %s: %s", p, e)
            return set()

    # ─────────────── normalization helpers ───────────────
    @staticmethod
    def _normalize_any_to_list(x: Any) -> List[str]:
        """
        Conservative coercion to list[str].

        IMPORTANT:
        - For plain strings that are not serialized JSON/Python lists, keep the
          string as a SINGLE atomic item.
        - Do NOT soft-split on commas, because curated disease terms often use
          comma modifiers, e.g. "Pulmonary Disease, Chronic Obstructive".
        """
        import json as _json, ast as _ast

        if x is None:
            return []

        if isinstance(x, (list, tuple, set)):
            out: List[str] = []
            for v in x:
                if isinstance(v, (str, int, float)):
                    sv = str(v).strip()
                    if sv:
                        out.append(sv)
            return out

        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []

            # Try JSON
            for attempt in (lambda z: z, lambda z: z.replace("'", '"')):
                try:
                    j = _json.loads(attempt(s))
                    return DiseaseListPreprocessor._normalize_any_to_list(j)
                except Exception:
                    pass

            # Try Python literal
            try:
                j = _ast.literal_eval(s)
                return DiseaseListPreprocessor._normalize_any_to_list(j)
            except Exception:
                pass

            # If list-like but parsing failed, try quoted extraction only
            if s.startswith("[") and s.endswith("]"):
                items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
                if items:
                    flat = [a or b for (a, b) in items]
                    return [t.strip() for t in flat if t.strip()]
                inner = s[1:-1].strip()
                return [inner] if inner else []

            # Atomic fallback: keep as one item
            return [s]

        if isinstance(x, (int, float)):
            return [str(x)]

        return []

    @staticmethod
    def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
        seen, out = set(), []
        for it in items:
            key = it.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    @staticmethod
    def _unicode_norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace("–", "-").replace("—", "-")
        return s

    # ─────────────── alias/synonym expansion ───────────────
    def _apply_alias(self, s: str) -> str:
        cf = s.casefold()
        return self._alias_map.get(cf, s)

    def _expand_synonyms(self, s: str) -> List[str]:
        cf = s.casefold()
        syns = self._syn_map.get(cf, [])
        return [s] + [t for t in syns if t and t != s]

    # ─────────────── whitelist/blacklist ───────────────
    def _passes_filters(self, s: str) -> bool:
        cf = s.casefold()
        if self.cfg.blacklist_path and cf in self._blacklist:
            return False
        if self.cfg.whitelist_path:
            return cf in self._whitelist
        return True

    def _basic_keep_rules(self, s: str) -> bool:
        if len(s) < self.cfg.min_chars:
            return False
        if self.cfg.drop_if_numeric_only and self._RE_NUM.match(s):
            return False
        return True

    # ─────────────── tokenization policy ───────────────
    def _tokenize_curated_entry(self, s: str) -> List[str]:
        """
        Tokenize a single curated disease entry.

        Default behavior:
          keep the entry atomic.

        Optional behavior if split_curated_entries=True:
          split only on conservative separators like semicolon/slash/pipe,
          but never on commas.
        """
        s = self._unicode_norm(s)
        if not s:
            return []

        if not self.cfg.split_curated_entries:
            return [s]

        parts = [p.strip() for p in self._re_split.split(s) if p.strip()]
        return parts or [s]

    # ─────────────── forward() ───────────────
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        """
        Modified version:
        - STRICTLY use ctx["processed_disease_list"].
        - If missing or empty, behave as empty input.
        - Treat each processed_disease_list entry as atomic by default.
        """
        trial_id = ctx.get("trial_id") or "trial"
        report_path = self.cfg.report_dir / f"{trial_id}_disease_preproc.json"

        # 1) Get only our own curated disease list
        seeds = ctx.get("processed_disease_list") or []
        if not isinstance(seeds, list):
            seeds = [str(seeds)]

        raw_items = [str(x).strip() for x in seeds if str(x).strip()]

        src_debug = {
            "source": "processed_disease_list",
            "from_processed": True,
            "from_raw": False,
            "from_contextual": False,
            "from_text": False,
            "split_curated_entries": self.cfg.split_curated_entries,
            "split_pattern": self.cfg.split_pattern if self.cfg.split_curated_entries else None,
        }

        if self.verbose:
            _LOG.info("[DLP] Using processed_disease_list: %d items", len(raw_items))

        # 2) Stage A: normalize & tokenize conservatively
        stage_a_norm = [self._unicode_norm(x) for x in raw_items]
        stage_a_flat: List[str] = []
        for s in stage_a_norm:
            stage_a_flat.extend(self._tokenize_curated_entry(s))

        # 3) Stage B: alias & synonym expansion
        stage_b_alias = [self._apply_alias(s) for s in stage_a_flat]
        stage_b_expand: List[str] = []
        for s in stage_b_alias:
            stage_b_expand.extend(self._expand_synonyms(s))

        # 4) Stage C: basic keep rules + whitelist/blacklist filters
        stage_c_keep = [s for s in stage_b_expand if self._basic_keep_rules(s)]
        stage_c_filt = [s for s in stage_c_keep if self._passes_filters(s)]

        # 5) Stage D: canonicalization
        stage_d_canon = [self._canon_string(s) for s in stage_c_filt]

        # 6) Stage E: dedupe
        if self.cfg.dedupe_casefold:
            final_list = self._dedupe_preserve_order(stage_d_canon)
        else:
            final_list = list(dict.fromkeys(stage_d_canon))

        # 7) Store results in ctx
        ctx["raw_disease_candidates"] = stage_a_flat
        ctx["target_disease"] = [
            {"disease": s, "provenance": "preproc"} for s in final_list
        ]
        ctx["diseases"] = "[" + ", ".join(f"'{x}'" for x in final_list) + "]"

        report = {
            "trial_id": trial_id,
            "sources": src_debug,
            "counts": {
                "raw": len(raw_items),
                "stage_a_norm": len(stage_a_norm),
                "stage_a_flat": len(stage_a_flat),
                "stage_b_expand": len(stage_b_expand),
                "stage_c_after_rules": len(stage_c_keep),
                "stage_c_after_filters": len(stage_c_filt),
                "final": len(final_list),
            },
            "samples": {
                "raw_head": raw_items[:20],
                "stage_a_flat": stage_a_flat[:20],
                "final": final_list,
            },
            "config": {
                "min_chars": self.cfg.min_chars,
                "drop_if_numeric_only": self.cfg.drop_if_numeric_only,
                "dedupe_casefold": self.cfg.dedupe_casefold,
                "keep_order": self.cfg.keep_order,
                "split_curated_entries": self.cfg.split_curated_entries,
                "split_pattern": self.cfg.split_pattern,
                "has_whitelist": bool(self._whitelist),
                "has_blacklist": bool(self._blacklist),
            },
        }
        ctx["preproc_report"] = report

        try:
            _write_json(report_path, report)
        except Exception as e:
            _LOG.warning("[DLP] report write failed: %s", e)

        if self.verbose:
            _LOG.info("[DLP] final unique diseases: %d", len(final_list))
            for s in final_list:
                _LOG.info("  • %s", s)

        return ctx

    # ─────────────── formatting helpers ───────────────
    @staticmethod
    def _canon_string(s: str) -> str:
        # Light canonicalization only. Preserve commas because they may be part of the disease name.
        s = s.strip()
        s = re.sub(r"\s*([/-])\s*", r"\1", s)
        s = re.sub(r"\s*,\s*", ", ", s)   # normalize comma spacing, do not split
        s = re.sub(r"\s{2,}", " ", s)
        return s


# ───────────────────── CLI (optional) ─────────────────────
if __name__ == "__main__":
    import argparse, sys, pathlib as _p

    p = argparse.ArgumentParser(description="Run DiseaseListPreprocessor on a single trial JSON or JSONL")
    p.add_argument("input", help="Path to JSON (single object) or JSONL (use --id to select).")
    p.add_argument("--id", help="If input is JSONL, select by _id or trial_id")
    p.add_argument("--alias-map")
    p.add_argument("--synonym-map")
    p.add_argument("--whitelist")
    p.add_argument("--blacklist")
    p.add_argument("--report-dir", default="entity_reports")
    p.add_argument("--verbose", action="store_true")

    # New flags
    p.add_argument(
        "--split-curated-entries",
        action="store_true",
        help="Split curated disease entries on conservative delimiters (never commas). Default: off",
    )
    p.add_argument(
        "--split-pattern",
        default=r"[;/|]",
        help="Regex used only when --split-curated-entries is enabled. Default: [;/|]",
    )
    p.add_argument(
        "--processed-disease-list",
        help='Optional JSON list string, e.g. \'["Pulmonary Disease, Chronic Obstructive"]\'',
    )

    args = p.parse_args()

    cfg = DiseasePreprocConfig(
        alias_map_path=_p.Path(args.alias_map) if args.alias_map else None,
        synonym_map_path=_p.Path(args.synonym_map) if args.synonym_map else None,
        whitelist_path=_p.Path(args.whitelist) if args.whitelist else None,
        blacklist_path=_p.Path(args.blacklist) if args.blacklist else None,
        report_dir=_p.Path(args.report_dir),
        split_curated_entries=bool(args.split_curated_entries),
        split_pattern=args.split_pattern,
    )

    # load trial
    ip = _p.Path(args.input)
    trial: Dict[str, Any]
    if ip.suffix.lower() == ".jsonl":
        if not args.id:
            print("For JSONL input you must supply --id", file=sys.stderr)
            sys.exit(2)
        with ip.open(encoding="utf-8") as fh:
            trial = next(
                json.loads(l)
                for l in fh
                if f'"_id": "{args.id}"' in l or f'"trial_id": "{args.id}"' in l
            )
    else:
        trial = json.loads(ip.read_text(encoding="utf-8"))

    # prepare ctx consistent with your pipeline
    if args.processed_disease_list:
        processed_disease_list = DiseaseListPreprocessor._normalize_any_to_list(
            args.processed_disease_list
        )
    else:
        # fallback: try to use trial["processed_disease_list"], else trial["diseases"], else empty
        processed_disease_list = DiseaseListPreprocessor._normalize_any_to_list(
            trial.get("processed_disease_list")
        )
        if not processed_disease_list:
            processed_disease_list = DiseaseListPreprocessor._normalize_any_to_list(
                trial.get("diseases")
            )

    ctx = {
        "trial_id": trial.get("_id") or trial.get("trial_id") or "UNKNOWN",
        "contextual": trial,
        "contextual_text": json.dumps(trial, ensure_ascii=False, indent=2),
        "processed_disease_list": processed_disease_list,
    }

    mod = DiseaseListPreprocessor(cfg, verbose=args.verbose)
    ctx = mod(ctx)

    print(json.dumps({
        "trial_id": ctx["trial_id"],
        "processed_disease_list": ctx.get("processed_disease_list"),
        "target_disease": ctx.get("target_disease"),
        "preproc_counts": ctx.get("preproc_report", {}).get("counts"),
        "stage_a_flat": ctx.get("raw_disease_candidates"),
    }, ensure_ascii=False, indent=2))