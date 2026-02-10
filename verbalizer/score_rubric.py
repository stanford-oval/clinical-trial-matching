#!/usr/bin/env python3
"""Heuristic rubric scorer for freeform rationales.

Six axes, each scored 0–1; system mean reported per axis + composite.
Designed for fast iteration on verbalizer prompts WITHOUT an LLM rater
(no API cost, deterministic, reproducible across runs).

Axes
----
1. length_band      : 1.0 if 60–120 words, partial credit outside.
2. decisive_lead    : 1.0 if first sentence names a load-bearing finding
                      (chart fact + criterion verb), else 0.5/0.0.
3. chart_specificity: fraction of sentences citing concrete chart facts
                      (numbers, units, quoted phrases, ages, lab values).
4. no_system_leak   : 1.0 if no "matcher determined/noted/concluded/found"
                      phrasing.
5. verdict_consistency: 1.0 if prose contains a verdict-aligned cue
                        (eligible→satisfies/meets; ineligible→triggers/excluded/blocker).
6. hedge_discipline : penalize vague hedging ("may", "possibly", "could",
                      "appears to") past a small budget.

Composite = mean of the six axes.

Run
---
    python verbalizer/score_rubric.py \
        --files aegis=/tmp/v6_rationales/aegis_v6.jsonl \
                v5=/tmp/v6_rationales/v5_v6.jsonl \
                shah=/tmp/v6_rationales/shah_v6.jsonl
"""
from __future__ import annotations
import argparse, json, re, statistics, sys
from pathlib import Path
from collections import defaultdict

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
MATCHER_LEAK = re.compile(
    r"\bmatcher (?:determined|noted|concluded|found|reviewed|assessment|review)\b"
    r"|\baccording to the matcher\b|\bthe matcher\b",
    re.I,
)
CRITERION_VERB = re.compile(
    r"\b(satisf(?:y|ies|ied)|meet(?:s|ing)?|trigger(?:s|ed)?|fail(?:s|ed)? to meet|"
    r"violat(?:e|es|ed)|exclud(?:e|es|ed)|preclud(?:e|es|ed)|qualif(?:y|ies|ied)|"
    r"disqualif(?:y|ies|ied)|render(?:s|ed)? .* ineligible|block(?:s|ed|er)?)\b",
    re.I,
)
SPECIFIC_FACT = re.compile(
    r"\b\d+\s*(?:y(?:ear)?s?(?:[- ]?old)?|mg|ml|kg|mmHg|bpm|/(?:hr|min)|"
    r"%|cm|mm|IU|µg|mcg|ng|U|mEq)\b"  # numeric + unit
    r"|\bBP\s*\d+/\d+\b|\bHR\s*\d+\b|\bage\s*\d+\b"
    r"|\d{1,3}-year-old\b"
    r"|\"[^\"]{4,80}\"|“[^”]{4,80}”"  # quoted phrase
    r"|\b[A-Z][A-Z0-9]{2,}\b",  # acronym (NSCLC, HER2, etc.)
)
ELIG_CUES = re.compile(r"\b(satisf|meet|qualif|inclusion .* met|no exclusion)", re.I)
INELIG_CUES = re.compile(
    r"\b(trigger|exclud|preclud|disqualif|blocker|ineligibl|violat|fail.*meet)",
    re.I,
)
HEDGES = re.compile(
    r"\b(?:may|might|could|possibly|perhaps|appears to|seems to|suggests that)\b",
    re.I,
)


def split_sents(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]


def score_length(text: str) -> float:
    w = len(text.split())
    if 60 <= w <= 120:
        return 1.0
    # linear falloff: 0 at <=30 or >=180
    if w < 60:
        return max(0.0, (w - 30) / 30)
    return max(0.0, (180 - w) / 60)


def score_decisive_lead(text: str) -> float:
    sents = split_sents(text)
    if not sents:
        return 0.0
    first = sents[0]
    has_verb = bool(CRITERION_VERB.search(first))
    has_fact = bool(SPECIFIC_FACT.search(first))
    if has_verb and has_fact:
        return 1.0
    if has_verb or has_fact:
        return 0.5
    return 0.0


def score_chart_specificity(text: str) -> float:
    sents = split_sents(text)
    if not sents:
        return 0.0
    hits = sum(1 for s in sents if SPECIFIC_FACT.search(s))
    return min(1.0, hits / max(1, len(sents)))


def score_no_leak(text: str) -> float:
    return 0.0 if MATCHER_LEAK.search(text) else 1.0


def score_verdict_consistency(text: str, verdict: str) -> float:
    v = (verdict or "").strip().lower()
    if v == "eligible":
        return 1.0 if ELIG_CUES.search(text) and not INELIG_CUES.search(text) \
            else (0.5 if ELIG_CUES.search(text) else 0.0)
    if v in ("ineligible", "not eligible"):
        return 1.0 if INELIG_CUES.search(text) else 0.0
    # unknown / excluded / other → can't score, give neutral 0.5
    return 0.5


def score_hedge_discipline(text: str) -> float:
    n = len(HEDGES.findall(text))
    # budget: up to 1 hedge is fine; degrade linearly
    if n <= 1:
        return 1.0
    if n >= 5:
        return 0.0
    return max(0.0, 1.0 - (n - 1) * 0.25)


AXES = [
    ("length_band", score_length),
    ("decisive_lead", score_decisive_lead),
    ("chart_specificity", score_chart_specificity),
    ("no_system_leak", score_no_leak),
    ("hedge_discipline", score_hedge_discipline),
]


def score_one(text: str, verdict: str) -> dict:
    out = {}
    for name, fn in AXES:
        out[name] = fn(text)
    out["verdict_consistency"] = score_verdict_consistency(text, verdict)
    out["composite"] = statistics.mean(out.values())
    out["_words"] = len(text.split())
    return out


def load(path: Path) -> list[dict]:
    rows = []
    for ln in path.open():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("rationale"):
            rows.append(o)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="system=path pairs, e.g. aegis=/tmp/.../aegis_v6.jsonl")
    ap.add_argument("--worst", type=int, default=3,
                    help="print N worst rationales per system (default 3)")
    args = ap.parse_args()

    per_sys = {}
    for spec in args.files:
        name, path = spec.split("=", 1)
        rows = load(Path(path))
        scored = [
            (r["pair"], score_one(r["rationale"], r.get("eligibility", "")), r["rationale"])
            for r in rows
        ]
        per_sys[name] = scored
        print(f"  loaded {name}: {len(scored)} rationales from {path}")

    axes_all = [a for a, _ in AXES] + ["verdict_consistency", "composite"]
    print()
    print(f"{'axis':<22} " + " ".join(f"{s:>10}" for s in per_sys))
    print("-" * (22 + 11 * len(per_sys)))
    for axis in axes_all:
        row = [axis]
        for s, scored in per_sys.items():
            vals = [sc[axis] for _, sc, _ in scored]
            row.append(f"{statistics.mean(vals):>10.3f}")
        print(f"{row[0]:<22} " + " ".join(row[1:]))

    # word-count summary
    print()
    print(f"{'words':<22} " + " ".join(f"{s:>10}" for s in per_sys))
    print("-" * (22 + 11 * len(per_sys)))
    for stat_name, fn in [
        ("median", statistics.median), ("mean", statistics.mean),
        ("min", min), ("max", max),
    ]:
        row = [stat_name]
        for s, scored in per_sys.items():
            vals = [sc["_words"] for _, sc, _ in scored]
            row.append(f"{fn(vals):>10.1f}")
        print(f"{row[0]:<22} " + " ".join(row[1:]))

    # worst offenders per system on composite
    if args.worst:
        for s, scored in per_sys.items():
            print(f"\n=== {s}: worst {args.worst} by composite ===")
            for pair, sc, text in sorted(scored, key=lambda x: x[1]["composite"])[: args.worst]:
                bad_axes = [a for a in axes_all if a != "composite" and sc[a] < 0.5]
                print(f"  [{sc['composite']:.2f}] {pair}  weak: {','.join(bad_axes) or '-'}")
                print(f"    {text[:220]}{'...' if len(text) > 220 else ''}")


if __name__ == "__main__":
    main()
