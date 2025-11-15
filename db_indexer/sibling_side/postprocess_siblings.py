#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-process sibling_logs aggregate.json to normalize YES examples.

For each base variable, we:
  - infer timeframe from the base variable name (_now / _past / _ever)
  - compute a canonical form for the base preferred term and each candidate concept
    using the same logic as PatientCanonicalEntityEnricher
  - reconstruct a sibling variable name for each YES example that:
        * uses the same stem/schema as the base variable
        * uses the same timeframe suffix as the base variable

We then write updated yes_examples entries like:

  {
    "base_concept": "...",
    "candidate_concept": "...",
    "conceptId": "123",
    "reason": "...",
    ...,
    "candidate_entity_canonical_form": "lithium_containing_product",
    "candidate_variable_name": "patient_is_taking_lithium_containing_product_now",
    "candidate_timeframe": "now"
  }

Usage (per-trial aggregate.json files):

  python postprocess_sibling_yes_examples.py \
      --log-root siblingsrc/sibling_logs \
      --out-root build/siblings

Usage (single global aggregate file):

  python postprocess_sibling_yes_examples.py \
      --global-aggregate siblingsrc/sibling_logs/all_trials_aggregate.json \
      --out-root build/siblings
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional
from pathlib import Path
import json, re, hashlib
# Reuse canonicalization helpers from PatientCanonicalEntityEnricher
# from modules.PatientCanonicalEntityEnricher import (  # type: ignore
#     _CanonicalFormRegistry,
#     _strip_tag,
# )

_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")
def _strip_tag(term: Optional[str]) -> str:
    return _TAG_RE.sub("", term or "").strip()



def _norm(x: Any) -> str:
    return str(x or "").strip()

def _to_int(x: Any) -> Optional[int]:
    if x is None: return None
    try: return int(x)
    except Exception:
        try: return int(str(x).strip())
        except Exception: return None

def _strip_tag(term: Optional[str]) -> str:
    return _TAG_RE.sub("", term or "").strip()

def _to_var(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def _sha12(s: str) -> str:
    return hashlib.sha256((_norm(s)).encode("utf-8")).hexdigest()[:12]

def _maybe_set(dst: Dict[str, Any], key: str, val: Any) -> None:
    if val in (None, "", [], {}): return
    cur = dst.get(key)
    if cur in (None, "", [], {}):
        dst[key] = val



class _CanonicalFormRegistry:
    """PreferredTerm(no tag) → snake_case; reuse across calls; persistable."""
    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self._pt2form: Dict[str, str] = dict(mapping or {})
        self._form2pts: Dict[str, set] = {}
        for pt, form in self._pt2form.items():
            self._form2pts.setdefault(form, set()).add(pt)

    def to_dict(self) -> Dict[str, str]:
        return dict(self._pt2form)

    @staticmethod
    def _base(pt: str) -> str:
        return _to_var(_strip_tag(pt))

    def get_or_assign(self, preferred_term: str, concept_id: Optional[str] = None) -> str:
        pt_key = _strip_tag(preferred_term or "")
        base = self._base(pt_key)
        if not pt_key:
            pt_key = preferred_term or ""
        if pt_key in self._pt2form:
            return self._pt2form[pt_key]

        if base not in self._form2pts:
            self._pt2form[pt_key] = base
            self._form2pts.setdefault(base, set()).add(pt_key)
            return base

        cand = _to_var(f"{base}_{concept_id}") if concept_id else _to_var(f"{base}_{_sha12(pt_key)[:6]}")
        if cand in self._form2pts and pt_key not in self._form2pts[cand]:
            k = 2
            while True:
                c2 = f"{cand}_{k}"
                if c2 not in self._form2pts:
                    cand = c2
                    break
                k += 1
        self._pt2form[pt_key] = cand
        self._form2pts.setdefault(cand, set()).add(pt_key)
        return cand

    @classmethod
    def load(cls, path: Optional[Path]) -> "_CanonicalFormRegistry":
        if path is None or not path.exists():
            return cls({})
        try:
            mapping = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                mapping = {}
        except Exception:
            mapping = {}
        return cls(mapping)

    def save(self, path: Optional[Path]) -> None:
        if path is None: return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._pt2form, ensure_ascii=False, indent=2), encoding="utf-8")

# ───────────────────────────── timeframe helpers ─────────────────────────────

def _infer_timeframe(v: str, default: str = "now") -> str:
    """Infer timeframe suffix from variable name."""
    v = (v or "").strip().lower()
    if v.endswith("_now"):
        return "now"
    if v.endswith("_past"):
        return "past"
    if v.endswith("_ever"):
        return "ever"
    return default


# ───────────────────────────── canonical helpers ─────────────────────────────

_CANON_REGISTRY = _CanonicalFormRegistry({})


def _canonical_form(preferred_term: str, concept_id: Optional[str] = None) -> str:
    """
    Turn a Preferred Term + conceptId into a stable snake_case canonical form,
    reusing the same logic as PatientCanonicalEntityEnricher.
    """
    pt = _strip_tag(preferred_term or "")
    cid = str(concept_id) if concept_id not in (None, "") else None
    return _CANON_REGISTRY.get_or_assign(pt, cid)


def _normalize_yes_examples(
    yes_examples: List[Dict[str, Any]],
    *,
    base_variable_name: str,
    base_preferred_term: str,
    base_concept_id: str,
) -> List[Dict[str, Any]]:
    """
    For a given base variable, attach a schema- and timeframe-aligned sibling
    variable name + canonical form to each YES example.

    We try to view the base variable as:

        base_variable_name = <prefix> + base_form + <suffix> + timeframe_suffix

    and then replace base_form with the candidate's canonical form.
    """
    # 1) timeframe from base variable name
    tf = _infer_timeframe(base_variable_name, default="now")
    tf_suffix = f"_{tf}" if base_variable_name.endswith(f"_{tf}") else ""
    base_core = (
        base_variable_name[: -len(tf_suffix)] if tf_suffix else base_variable_name
    )

    # 2) canonical form for the base preferred term
    base_form = _canonical_form(base_preferred_term or "", base_concept_id or None)

    idx = base_core.find(base_form)
    if idx == -1:
        # Fallback: we couldn't locate the canonical form string in the variable.
        # In that rare case, we will simply append the candidate form at the end,
        # keeping the entire base_core as prefix.
        logging.warning(
            "[WARN] could not locate base canonical form %r in base variable %r; "
            "using append-style schema for siblings.",
            base_form,
            base_variable_name,
        )
        prefix, suffix = base_core + "_", ""
    else:
        prefix = base_core[:idx]
        suffix = base_core[idx + len(base_form) :]

    normalized: List[Dict[str, Any]] = []

    for ex in yes_examples or []:
        cand_obj = ex.get("candidate_concept")

        if isinstance(cand_obj, dict):
            # Your JSON looks like: {"term": "...", "conceptId": "..."}
            cand_term = (
                cand_obj.get("preferred_term")
                or cand_obj.get("term")
                or cand_obj.get("concept_name")
                or cand_obj.get("label")
                or ""
            )
            cand_cid = (
                cand_obj.get("conceptId")
                or cand_obj.get("concept_id")
                or ex.get("conceptId")
            )
        else:
            # Old behavior: candidate_concept is already a string, or absent
            cand_term = (
                cand_obj
                or ex.get("preferred_term")
                or ex.get("concept_name")
                or ""
            )
            cand_cid = ex.get("conceptId")

        cand_form = _canonical_form(cand_term, cand_cid)
        cand_core = f"{prefix}{cand_form}{suffix}"
        cand_var = f"{cand_core}{tf_suffix}"

        ex_copy = dict(ex)
        ex_copy["candidate_entity_canonical_form"] = cand_form
        ex_copy["candidate_variable_name"] = cand_var
        ex_copy["candidate_timeframe"] = tf
        normalized.append(ex_copy)


    return normalized


# ───────────────────────────── file utilities ────────────────────────────────

def _load_json(path: pathlib.Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error("[ERROR] Failed to load JSON %s: %s", path, e)
        return None


def _write_json(path: pathlib.Path, obj: Any, *, backup: bool) -> None:
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        logging.info("  - backup written to %s", bak)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ───────────────────────────── core processing ───────────────────────────────

def _process_items(items: List[Dict[str, Any]]) -> int:
    """
    Normalize yes_examples for an array of aggregate items.
    Returns number of items modified.
    """
    modified = 0
    for item in items or []:
        vname = item.get("variable_name")
        cid = item.get("concept_id")
        pt = item.get("preferred_term")
        yes = item.get("yes_examples")

        if not (vname and cid and pt and isinstance(yes, list) and yes):
            continue

        yes_norm = _normalize_yes_examples(
            yes,
            base_variable_name=vname,
            base_preferred_term=pt,
            base_concept_id=str(cid),
        )
        item["yes_examples"] = yes_norm
        modified += 1
    return modified


def process_per_trial_aggregates(
    log_root: pathlib.Path, *, backup: bool, out_root: Optional[pathlib.Path]
) -> None:
    """
    Walk log_root/NCT*/aggregate.json and normalize each.

    If out_root is provided, write results to:

        out_root/<trial_id>/aggregate.json

    Otherwise overwrite the original aggregate.json in-place.
    """
    if not log_root.exists() or not log_root.is_dir():
        logging.error("[ERROR] log-root directory not found: %s", log_root)
        return

    total_files = 0
    total_items = 0

    for trial_dir in sorted(log_root.iterdir()):
        if not trial_dir.is_dir():
            continue
        agg_path = trial_dir / "aggregate.json"
        if not agg_path.exists():
            continue

        logging.info("Processing %s", agg_path)
        agg = _load_json(agg_path)
        if not isinstance(agg, dict):
            logging.warning("  - skipped (not an object)")
            continue

        items = agg.get("items")
        if not isinstance(items, list):
            logging.warning("  - skipped (no 'items' list)")
            continue

        modified = _process_items(items)
        if modified:
            logging.info("  - normalized %d items", modified)

            if out_root is not None:
                # mirror trial dir name under out_root
                dest_dir = out_root / trial_dir.name
                dest_path = dest_dir / "aggregate.json"
            else:
                dest_path = agg_path

            _write_json(dest_path, agg, backup=backup)
            logging.info("  - wrote updated aggregate to %s", dest_path)
            total_files += 1
            total_items += modified
        else:
            logging.info("  - no items needed normalization")

    logging.info(
        "Done. Updated %d aggregate.json files; %d items normalized.",
        total_files,
        total_items,
    )


def process_global_aggregate(
    global_path: pathlib.Path, *, backup: bool, out_root: Optional[pathlib.Path]
) -> None:
    """
    Normalize a single global aggregate JSON:
      { "trials": [ { "trial_id": "...", "items": [...] }, ... ] }

    If out_root is provided, write to:

        out_root/<basename_of_global_path>

    Otherwise overwrite the original file in-place.
    """
    if not global_path.exists():
        logging.error("[ERROR] global-aggregate file not found: %s", global_path)
        return

    logging.info("Processing global aggregate %s", global_path)
    obj = _load_json(global_path)
    if not isinstance(obj, dict):
        logging.error("[ERROR] global aggregate is not a JSON object")
        return

    trials = obj.get("trials")
    if not isinstance(trials, list):
        logging.error("[ERROR] global aggregate missing 'trials' list")
        return

    total_items = 0
    for t in trials:
        if not isinstance(t, dict):
            continue
        items = t.get("items")
        if not isinstance(items, list):
            continue
        modified = _process_items(items)
        trial_id = t.get("trial_id") or "UNKNOWN"
        logging.info("  - trial %s: normalized %d items", trial_id, modified)
        total_items += modified

    if total_items:
        if out_root is not None:
            out_path = out_root / global_path.name
        else:
            out_path = global_path

        _write_json(out_path, obj, backup=backup)
        logging.info(
            "Done. Total %d items normalized in global aggregate → %s.",
            total_items,
            out_path,
        )
    else:
        logging.info("Done. No items needed normalization in global aggregate.")


# ──────────────────────────────── main ───────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize sibling_logs aggregate.json files so that YES examples "
            "include schema-/timeframe-aligned candidate sibling variables."
        )
    )
    ap.add_argument(
        "--log-root",
        default="../../siblingsrc/sibling_logs/",
        help=(
            "root directory containing per-trial subdirs with aggregate.json "
            "(e.g., siblingsrc/sibling_logs)"
        ),
    )
    ap.add_argument(
        "--global-aggregate",
        default=None,
        help="path to a single global aggregate JSON (from SiblingOverlapPipeline --aggregate-out)",
    )
    ap.add_argument(
        "--out-root",
        default="../../build/siblings",
        help=(
            "optional output root dir. If set, results are written under this dir, "
            "leaving source JSON untouched. Example: build/siblings"
        ),
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="do NOT write .bak backups before overwriting JSON files",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    backup = not args.no_backup
    out_root = pathlib.Path(args.out_root).resolve() if args.out_root else None
    if out_root is not None:
        # When writing to a separate out-root, we generally don't need backups
        backup = False
        out_root.mkdir(parents=True, exist_ok=True)
        logging.info("Output root: %s", out_root)

    if args.global_aggregate:
        process_global_aggregate(
            pathlib.Path(args.global_aggregate),
            backup=backup,
            out_root=out_root,
        )
    elif args.log_root:
        process_per_trial_aggregates(
            pathlib.Path(args.log_root),
            backup=backup,
            out_root=out_root,
        )
    else:
        logging.error(
            "You must specify either --log-root (for per-trial aggregates) "
            "or --global-aggregate (for a single aggregate file)."
        )


if __name__ == "__main__":
    main()
