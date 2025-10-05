#!/usr/bin/env python3
"""
analyze_mbench_fixes.py

Reads stage mbench folders to compute:
  1) # trial files fixed under each step across step 0,1,2,3
  2) # files fixed under multiple steps across step 0,1,2,3
  3) # trial files fixed under each step across step 1,2,3
  4) # files fixed under multiple steps across step 1,2,3

"trial file" is keyed by (effective_trial_id, side) like: (NCT123..., inclusion|exclusion)

It works purely from mbench:
  - pairs *_prompt.txt + *_raw.txt
  - parses raw JSON and extracts "repaired_smt"
  - extracts original SMT from prompt using tag/heuristic
  - marks FIXED if repaired_smt != "NO MODIFICATION REQUIRED" AND differs from original SMT

Usage:
  python3 analyze_mbench_fixes.py --work-dir /path/to/ir_orchestrated_YYYYMMDDTHHMMSS

Optional:
  --stage0-mbench <path> --stage1-mbench <path> --stage2-mbench <path> --stage3-mbench <path>
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, List

# -----------------------------
# Parsing helpers
# -----------------------------
NCT_SIDE_RE = re.compile(r"^(NCT[0-9]+[A-Za-z]?)[_](inclusion|exclusion)_cohort-")
SENTINELS = {"NO MODIFICATION REQUIRED", "NO_MODIFICATION_REQUIRED"}

TAG_BLOCKS = [
    "current_smt_program",
    "smt_program",
    "SMT_PROGRAM",
]

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]  # drop ```...
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        t = "\n".join(lines).strip()
    return t

def parse_json_object(raw_text: str) -> Optional[dict]:
    t = _strip_fences(raw_text)
    if not t:
        return None
    # try full parse
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # try first {...} span
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            obj = json.loads(t[l:r+1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

def extract_original_smt_from_prompt(prompt: str) -> Optional[str]:
    """
    Best-effort extraction:
      1) try XML-like blocks: <current_smt_program> ... </current_smt_program>, <smt_program> ... </smt_program>
      2) otherwise, find "CURRENT SMT" region and take from first '(' after it
      3) otherwise, take from first occurrence of "(set-logic" or "(declare-const" after last "SMT" mention
    """
    t = prompt or ""
    # 1) tag blocks
    for tag in TAG_BLOCKS:
        pat = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
        m = pat.search(t)
        if m:
            s = m.group(1).strip()
            return s if s else None

    # 2) header-based
    hdr = re.search(r"CURRENT\s+SMT", t, re.IGNORECASE)
    if hdr:
        tail = t[hdr.end():]
        i = tail.find("(")
        if i != -1:
            s = tail[i:].strip()
            return s if s else None

    # 3) heuristic: find last "SMT" mention, then first "(set-logic" or "(declare-const" after that
    last_smt = t.lower().rfind("smt")
    tail = t[last_smt:] if last_smt != -1 else t

    # prefer set-logic if present
    for kw in ["(set-logic", "(declare-const", "(assert"]:
        i = tail.find(kw)
        if i != -1:
            # back up to first '(' at/after i
            j = tail.rfind("\n", 0, i)
            s = tail[i:].strip()
            return s if s else None

    return None

def normalize_smt(s: str) -> str:
    # keep it simple: strip trailing whitespace, normalize CRLF
    return "\n".join(line.rstrip() for line in (s or "").replace("\r\n", "\n").replace("\r", "\n").strip().splitlines()).strip()

# -----------------------------
# Core reading
# -----------------------------
@dataclass(frozen=True)
class FixRecord:
    eff_tid: str
    side: str
    fixed: bool

def iter_mbench_records(mbench_dir: Path) -> Iterable[FixRecord]:
    """
    For each base:
      <eff>_<side>_cohort-..._prompt.txt
      <eff>_<side>_cohort-..._raw.txt
    decide fixed.
    """
    if not mbench_dir.exists():
        return

    prompts: Dict[str, Path] = {}
    raws: Dict[str, Path] = {}

    for p in mbench_dir.glob("*_prompt.txt"):
        base = p.name[:-len("_prompt.txt")]
        prompts[base] = p
    for p in mbench_dir.glob("*_raw.txt"):
        base = p.name[:-len("_raw.txt")]
        raws[base] = p

    for base, pp in prompts.items():
        rp = raws.get(base)
        if rp is None:
            continue

        m = NCT_SIDE_RE.match(base)
        if not m:
            continue
        eff, side = m.group(1), m.group(2)

        prompt_txt = _read_text(pp)
        raw_txt = _read_text(rp)

        obj = parse_json_object(raw_txt)
        if not obj:
            continue

        repaired = obj.get("repaired_smt") or obj.get("meaning_enriched_smt") or obj.get("fixed_smt")
        # In your code, key is consistently "repaired_smt" for polarity/logic,
        # and "repaired_smt" for meaning module too.
        if not isinstance(repaired, str):
            continue

        repaired_str = repaired.strip()
        if repaired_str.upper() in SENTINELS:
            yield FixRecord(eff, side, False)
            continue

        orig = extract_original_smt_from_prompt(prompt_txt)
        if not orig:
            # If we can't extract original SMT, fall back to "treated as changed"
            # because repaired_smt is a full SMT program (in your modules).
            yield FixRecord(eff, side, True)
            continue

        fixed = normalize_smt(repaired_str) != normalize_smt(orig)
        yield FixRecord(eff, side, fixed)

def aggregate_fixed_by_pair(mbench_dir: Path) -> Set[Tuple[str, str]]:
    fixed_pairs: Set[Tuple[str, str]] = set()
    seen_pairs: Set[Tuple[str, str]] = set()

    for rec in iter_mbench_records(mbench_dir):
        pair = (rec.eff_tid, rec.side)
        seen_pairs.add(pair)
        if rec.fixed:
            fixed_pairs.add(pair)

    return fixed_pairs

# -----------------------------
# Reporting
# -----------------------------
def overlap_stats(stage_to_fixed: Dict[str, Set[Tuple[str, str]]], stages: List[str]) -> Tuple[int, Dict[int, int]]:
    """
    Returns:
      - num_pairs_fixed_in_2plus
      - histogram: k -> count (fixed in exactly k of selected stages)
    """
    all_pairs: Set[Tuple[str, str]] = set()
    for s in stages:
        all_pairs |= stage_to_fixed.get(s, set())

    hist: Dict[int, int] = {}
    multi = 0
    for pair in all_pairs:
        k = sum(1 for s in stages if pair in stage_to_fixed.get(s, set()))
        hist[k] = hist.get(k, 0) + 1
        if k >= 2:
            multi += 1
    return multi, dict(sorted(hist.items()))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, help="Path to ir_orchestrated_... directory")
    ap.add_argument("--stage0-mbench", default=None)
    ap.add_argument("--stage1-mbench", default=None)
    ap.add_argument("--stage2-mbench", default=None)
    ap.add_argument("--stage3-mbench", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()

    stage_dirs = {
        "stage0": Path(args.stage0_mbench).resolve() if args.stage0_mbench else work / "stage0_meaning" / "mbench",
        "stage1": Path(args.stage1_mbench).resolve() if args.stage1_mbench else work / "stage1_polarity" / "mbench",
        "stage2": Path(args.stage2_mbench).resolve() if args.stage2_mbench else work / "stage2_repair" / "mbench",
        "stage3": Path(args.stage3_mbench).resolve() if args.stage3_mbench else work / "stage3_logic" / "mbench",
    }

    stage_to_fixed: Dict[str, Set[Tuple[str, str]]] = {}
    for st, mb in stage_dirs.items():
        fixed = aggregate_fixed_by_pair(mb)
        stage_to_fixed[st] = fixed

    # (1) fixed per step 0-3
    print("\n# (1) Fixed trial files per step (0,1,2,3)")
    for st in ["stage0", "stage1", "stage2", "stage3"]:
        print(f"  {st}: {len(stage_to_fixed[st])}")

    # (2) fixed in multiple steps 0-3
    multi_0_3, hist_0_3 = overlap_stats(stage_to_fixed, ["stage0", "stage1", "stage2", "stage3"])
    print("\n# (2) Fixed in multiple steps (>=2) across steps 0..3")
    print(f"  multi_step_count: {multi_0_3}")
    print(f"  overlap_histogram_exact_k_steps: {hist_0_3}")

    # (3) fixed per step 1-3
    print("\n# (3) Fixed trial files per step (1,2,3)")
    for st in ["stage1", "stage2", "stage3"]:
        print(f"  {st}: {len(stage_to_fixed[st])}")

    # (4) fixed in multiple steps 1-3
    multi_1_3, hist_1_3 = overlap_stats(stage_to_fixed, ["stage1", "stage2", "stage3"])
    print("\n# (4) Fixed in multiple steps (>=2) across steps 1..3")
    print(f"  multi_step_count: {multi_1_3}")
    print(f"  overlap_histogram_exact_k_steps: {hist_1_3}")

if __name__ == "__main__":
    main()
