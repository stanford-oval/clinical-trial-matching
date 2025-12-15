#!/usr/bin/env python3
"""Why is ZSPM's PIVOTALFLIPRATE low? Decompose its rejection grounds.

ZSPM (Shah/Koopman binary) evaluates each criterion independently and, when
the chart does not address a criterion, defaults it to is_met=false. Our
koopman_binary.prompt asks the matcher to self-report this via
`evidence_status` ∈ {deterministic_met, deterministic_not_met,
chart_silent_defaulted_not_met}.

A criterion is a FAILING ground iff is_met is False -- for both inclusion and
exclusion sides (exclusion is_met=True means the exclusion is *satisfied*,
i.e. the patient does not have the excluding condition). This matches
filter_shah_blockers() in
experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/run_full_ablation.py

Reports, over ZSPM's ineligible decisions:
  1. silent-defaulted vs evidence-backed failing criteria per rejection
  2. share of rejections with NO evidence-backed ground at all
  3. self-consistency: same (trial, criterion), chart silent for >=2
     patients -- is the default always the same?

Usage:
    python scripts/tables/zspm_silence_breakdown.py [--cell cell3_5m_v3v_filt]
"""
from __future__ import annotations
import argparse, json, pathlib, statistics
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
MBENCH = ROOT / "experiments/counterfactual/05_self_faithfulness/mbench_3cell"


def is_failing(criterion: str, is_met) -> bool:
    """A ground is failing iff is_met is False, on either side."""
    return is_met is False


def is_silent(evidence_status: str) -> bool:
    e = str(evidence_status or "")
    return "silent" in e or "defaulted" in e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="cell3_5m_v3v_filt")
    ap.add_argument("--system", default="shah_koopman_binary")
    args = ap.parse_args()

    base = MBENCH / args.cell / args.system
    if not base.exists():
        raise SystemExit(f"not found: {base}")

    n_silent, n_evid = [], []
    zero_evidence = total = 0
    treat = defaultdict(list)   # (nct, criterion) -> [is_met, ...] where silent

    for bucket in ("flipped", "not_flipped"):
        bd = base / bucket
        if not bd.exists():
            continue
        for pd in sorted(bd.iterdir()):
            rej = pd / "05_rejudged" / "response.json"
            if not rej.exists():
                continue
            try:
                o = json.loads(rej.read_text())
            except Exception:
                continue
            if o.get("eligibility") != "ineligible":
                continue
            total += 1
            nct = pd.name.split("__")[1] if "__" in pd.name else "?"
            s = d = 0
            for a in o.get("assessments", []) or []:
                crit, is_met = str(a.get("criterion", "")), a.get("is_met")
                silent = is_silent(a.get("evidence_status"))
                if silent:
                    treat[(nct, crit)].append(is_met)
                if is_failing(crit, is_met):
                    if silent: s += 1
                    else:      d += 1
            n_silent.append(s); n_evid.append(d)
            if d == 0 and s > 0:
                zero_evidence += 1

    if total == 0:
        raise SystemExit("no ineligible ZSPM decisions found")

    print(f"cell={args.cell}  system={args.system}")
    print(f"ZSPM ineligible decisions: {total}\n")
    print("Failing criteria per rejection (is_met=False):")
    print(f"  silent-defaulted : mean {statistics.mean(n_silent):5.1f}  "
          f"median {statistics.median(n_silent):.0f}  max {max(n_silent)}")
    print(f"  evidence-backed  : mean {statistics.mean(n_evid):5.1f}  "
          f"median {statistics.median(n_evid):.0f}  max {max(n_evid)}")

    ts, td = sum(n_silent), sum(n_evid)
    print(f"\nRejection grounds overall: {ts} silence-driven vs {td} evidence-backed"
          f"  -> {100*ts/max(1,ts+td):.0f}% of grounds are chart-silence")
    print(f"Rejections with ZERO evidence-backed ground: "
          f"{zero_evidence}/{total} = {100*zero_evidence/total:.1f}%")

    incons = same = 0
    examples = []
    for (nct, crit), vals in treat.items():
        if len(vals) >= 2:
            if len(set(vals)) > 1:
                incons += 1
                if len(examples) < 5:
                    examples.append((nct, crit, dict(Counter(vals))))
            else:
                same += 1
    denom = max(1, same + incons)
    print(f"\nSelf-consistency (same trial+criterion, chart silent for >=2 patients):")
    print(f"  consistent default : {same}")
    print(f"  inconsistent       : {incons}  ({100*incons/denom:.1f}%)")
    for nct, crit, c in examples:
        print(f"    {nct} {crit}: {c}")
    print("\nNote: ZSPM is mechanically self-consistent (one blanket default).")
    print("What it cannot do is vary the default by criterion type, which is")
    print("what the explicit policy Pi requires -- see Table 5.")


if __name__ == "__main__":
    main()
