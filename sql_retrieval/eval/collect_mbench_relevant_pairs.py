#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_mbench_relevant_pairs.py

Collect pairs from mbench_root:
- relevant AND eligible
- relevant AND ineligible

Directory layout assumed:
  <mbench_root>/<mode>/<patient_id>/<parent_trial_id>/<subcohort_id>/
    relevance.txt
    eligibility.txt

Supports eligibility prompt outputs like:
  <subcohort_eligibility_decisions>
  [
    {"eligibility_decision": "ineligible", ...}
  ]
  </subcohort_eligibility_decisions>

Writes:
  <out_dir>/relevant_and_eligible.jsonl
  <out_dir>/relevant_and_ineligible.jsonl
  <out_dir>/summary.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Robust parsing helpers
# -------------------------

def _extract_tagged_payload(text: str, tag: str) -> Optional[str]:
    """
    Extract inner content between <tag> ... </tag>, tolerant to missing final '>' on closing tag.
    """
    s = text.strip()
    m = re.search(
        rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*(?:>|$)",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()


def _json_parse_best_effort(payload: str) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return ast.literal_eval(payload)


def _json_leading_value(text: str) -> Tuple[Optional[Any], Optional[int]]:
    s = text.lstrip()
    if not s:
        return None, None
    dec = json.JSONDecoder()
    try:
        obj, idx = dec.raw_decode(s)
        return obj, idx
    except Exception:
        return None, None


def _json_value_after_marker(text: str, marker_regex: str) -> Optional[Any]:
    m = re.search(marker_regex, text, flags=re.IGNORECASE)
    if not m:
        return None
    tail = text[m.end():].lstrip()
    if not tail:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _idx = dec.raw_decode(tail)
        return obj
    except Exception:
        return None


def _json_last_list_anywhere(text: str) -> Optional[list]:
    positions = [m.start() for m in re.finditer(r"\[", text)]
    dec = json.JSONDecoder()
    for pos in reversed(positions[-200:]):
        cand = text[pos:].strip()
        if not cand:
            continue
        if cand.startswith("[]"):
            return []
        try:
            obj, _idx = dec.raw_decode(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            return obj
    return None


def _coerce_boolish_str(s: str) -> Optional[bool]:
    sl = s.strip().lower()
    if sl in ("true", "yes", "y", "relevant", "eligible"):
        return True
    if sl in ("false", "no", "n", "irrelevant", "not relevant", "ineligible", "not eligible"):
        return False
    return None


def _extract_bool_from_obj(obj: Any, keys: List[str]) -> Optional[bool]:
    """
    For dict/object payloads only.
    IMPORTANT: does NOT treat list length as boolean; list semantics handled separately.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return _coerce_boolish_str(obj)

    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                v = obj[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    b = _coerce_boolish_str(v)
                    if b is not None:
                        return b
                if isinstance(v, list):
                    # nested best-effort
                    return len(v) > 0
        for v in obj.values():
            b = _extract_bool_from_obj(v, keys)
            if b is not None:
                return b
        return None

    return None


def _list_any_true_semantics(
    obj_list: list,
    *,
    decision_key_candidates: List[str],
    true_tokens: List[str],
) -> Optional[bool]:
    """
    List semantics:
      - [] => False
      - list[str] => non-empty => True (assumed positive list)
      - list[dict] => True iff ANY dict indicates true via decision keys
    """
    if not isinstance(obj_list, list):
        return None
    if len(obj_list) == 0:
        return False

    if all(isinstance(x, str) for x in obj_list):
        return True

    if all(isinstance(x, dict) for x in obj_list):
        saw_any = False
        saw_true = False
        for d in obj_list:
            for k in decision_key_candidates:
                if k not in d:
                    continue
                v = d[k]
                saw_any = True
                if isinstance(v, bool):
                    if v:
                        saw_true = True
                elif isinstance(v, str):
                    if v.strip().lower() in true_tokens:
                        saw_true = True
        if saw_any:
            return True if saw_true else False
        return None

    return None


def parse_relevance_bool(text: str) -> Optional[bool]:
    """
    Best-effort relevance parse.
    """
    s = text.strip()

    payload = _extract_tagged_payload(s, "relevant_subcohorts")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
                    true_tokens=["relevant", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    obj0, _ = _json_leading_value(s)
    if isinstance(obj0, list):
        b = _list_any_true_semantics(
            obj0,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj1 = _json_value_after_marker(s, r"\boutput\s*:\s*")
    if isinstance(obj1, list):
        b = _list_any_true_semantics(
            obj1,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj2 = _json_last_list_anywhere(s)
    if isinstance(obj2, list):
        b = _list_any_true_semantics(
            obj2,
            decision_key_candidates=["relevant", "is_relevant", "decision", "relevance_decision"],
            true_tokens=["relevant", "true", "yes", "y"],
        )
        if b is not None:
            return b

    try:
        obj = json.loads(s)
        b = _extract_bool_from_obj(obj, ["relevant", "is_relevant", "relevance", "decision", "relevance_decision"])
        if b is not None:
            return b
    except Exception:
        pass

    tl = s.lower()
    if "not relevant" in tl or "irrelevant" in tl:
        return False
    if re.search(r"(^|\W)relevant(\W|$)", tl) and not re.search(r"\bnot\s+relevant\b", tl):
        return True

    return None


def parse_eligibility_bool(text: str) -> Optional[bool]:
    """
    Best-effort eligibility parse (supports your new prompt format).
    """
    s = text.strip()

    # NEW tag: subcohort_eligibility_decisions
    payload = _extract_tagged_payload(s, "subcohort_eligibility_decisions")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
                    true_tokens=["eligible", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    # legacy tag: eligible_subcohorts
    payload = _extract_tagged_payload(s, "eligible_subcohorts")
    if payload is not None:
        try:
            obj = _json_parse_best_effort(payload)
            if isinstance(obj, list):
                b = _list_any_true_semantics(
                    obj,
                    decision_key_candidates=["eligible", "is_eligible", "decision", "eligibility_decision"],
                    true_tokens=["eligible", "true", "yes", "y"],
                )
                if b is not None:
                    return b
        except Exception:
            pass

    obj0, _ = _json_leading_value(s)
    if isinstance(obj0, list):
        b = _list_any_true_semantics(
            obj0,
            decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
            true_tokens=["eligible", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj1 = _json_value_after_marker(s, r"\boutput\s*:\s*")
    if isinstance(obj1, list):
        b = _list_any_true_semantics(
            obj1,
            decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
            true_tokens=["eligible", "true", "yes", "y"],
        )
        if b is not None:
            return b

    obj2 = _json_last_list_anywhere(s)
    if isinstance(obj2, list):
        b = _list_any_true_semantics(
            obj2,
            decision_key_candidates=["eligibility_decision", "decision", "eligible", "is_eligible"],
            true_tokens=["eligible", "true", "yes", "y"],
        )
        if b is not None:
            return b

    try:
        obj = json.loads(s)
        b = _extract_bool_from_obj(obj, ["eligible", "is_eligible", "eligibility", "decision", "eligibility_decision"])
        if b is not None:
            return b
    except Exception:
        pass

    tl = s.lower()
    if re.search(r"\bno\s+relevant\s+subcohorts?\b", tl):
        return False
    if re.search(r"\bno\s+eligibility\s+decisions?\s+are\s+required\b", tl):
        return False
    if re.search(r"\bthere\s+are\s+no\s+subcohorts?\s+to\s+evaluate\b", tl):
        return False

    if "ineligible" in tl or "not eligible" in tl:
        return False
    if re.search(r"(^|\W)eligible(\W|$)", tl) and "ineligible" not in tl and "not eligible" not in tl:
        return True

    return None


# -------------------------
# Mbench scanning
# -------------------------

@dataclass(frozen=True)
class Pair:
    mode: str
    patient_id: str
    parent_trial_id: str
    subcohort_id: str
    relevance_path: Path
    eligibility_path: Path


def iter_pairs(mbench_root: Path, modes: List[str]) -> List[Pair]:
    pairs: List[Pair] = []
    for mode in modes:
        mode_dir = mbench_root / mode
        if not mode_dir.exists():
            continue
        for patient_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir() and p.name != "pair_cache"]):
            for parent_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
                for sub_dir in sorted([p for p in parent_dir.iterdir() if p.is_dir()]):
                    rel = sub_dir / "relevance.txt"
                    elig = sub_dir / "eligibility.txt"
                    if not rel.exists() or not elig.exists():
                        continue
                    pairs.append(
                        Pair(
                            mode=mode,
                            patient_id=patient_dir.name,
                            parent_trial_id=parent_dir.name,
                            subcohort_id=sub_dir.name,
                            relevance_path=rel,
                            eligibility_path=elig,
                        )
                    )
    return pairs


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbench-root", type=str, required=True)
    ap.add_argument("--modes", type=str, default="ccr,all")
    ap.add_argument("--out-dir", type=str, default="./mbench_collections")
    ap.add_argument("--include-unknown", action="store_true",
                    help="Also write relevant_and_unknown_eligibility.jsonl (eligibility parse None).")
    args = ap.parse_args()

    mbench_root = Path(args.mbench_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    pairs = iter_pairs(mbench_root, modes)

    rel_elig: List[Dict[str, Any]] = []
    rel_inelig: List[Dict[str, Any]] = []
    rel_unknown: List[Dict[str, Any]] = []

    for p in pairs:
        rel_txt = _read_text(p.relevance_path)
        elig_txt = _read_text(p.eligibility_path)

        rel_b = parse_relevance_bool(rel_txt)
        elig_b = parse_eligibility_bool(elig_txt)

        row = {
            "mode": p.mode,
            "patient_id": p.patient_id,
            "parent_trial_id": p.parent_trial_id,
            "subcohort_id": p.subcohort_id,
            "relevant": rel_b,
            "eligible": elig_b,
            "relevance_path": str(p.relevance_path),
            "eligibility_path": str(p.eligibility_path),
        }

        if rel_b is True and elig_b is True:
            rel_elig.append(row)
        elif rel_b is True and elig_b is False:
            rel_inelig.append(row)
        elif rel_b is True and elig_b is None:
            rel_unknown.append(row)

    write_jsonl(out_dir / "relevant_and_eligible.jsonl", rel_elig)
    write_jsonl(out_dir / "relevant_and_ineligible.jsonl", rel_inelig)
    if args.include_unknown:
        write_jsonl(out_dir / "relevant_and_unknown_eligibility.jsonl", rel_unknown)

    summary = {
        "mbench_root": str(mbench_root),
        "modes": modes,
        "total_pairs_with_artifacts": len(pairs),
        "relevant_and_eligible": len(rel_elig),
        "relevant_and_ineligible": len(rel_inelig),
        "relevant_and_unknown_eligibility": len(rel_unknown),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] wrote:")
    print("  ", out_dir / "relevant_and_eligible.jsonl")
    print("  ", out_dir / "relevant_and_ineligible.jsonl")
    if args.include_unknown:
        print("  ", out_dir / "relevant_and_unknown_eligibility.jsonl")
    print("  ", out_dir / "summary.json")


if __name__ == "__main__":
    main()