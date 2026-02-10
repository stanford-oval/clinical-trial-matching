#!/usr/bin/env python3
"""Run any of the seven variants on a single pair and print the audit trail.

Examples:
    # Compare what each variant does on a known LM-failure case:
    python -m matchers.inspect_one sigir-20141__NCT02001545

    # Just one variant:
    python -m matchers.inspect_one sigir-20141__NCT02001545 --variant smt_lm_evidence_arbiter

    # All variants on all 122 disagreement cases:
    python -m matchers.inspect_one --all-disagreements --summary
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from matchers import variants as V
from matchers.data import list_disagreement_pairs, load_judge_verdict


def run_one(pair_id, variant_name=None):
    if variant_name and variant_name not in V.VARIANTS:
        print(f"Unknown variant: {variant_name}\nAvailable: {list(V.VARIANTS)}")
        return
    variants = [variant_name] if variant_name else list(V.VARIANTS.keys())
    gold = load_judge_verdict(pair_id)
    print(f"\n========== PAIR: {pair_id}  (gold={gold}) ==========\n")
    for v in variants:
        d = V.VARIANTS[v](pair_id)
        marker = "✓" if d.decision == gold else "✗" if gold else "?"
        audit_marker = "AUDITABLE" if d.is_auditable() else "PARTIAL-AUDIT"
        print(f"--- [{marker}] {v}  decision={d.decision}  ({audit_marker}) ---")
        print(f"    {d.reasoning}")
        for step in d.audit_trail:
            ev = ""
            if step.evidence:
                # Show only first few keys
                ks = list(step.evidence.keys())[:3]
                ev = " " + ", ".join(f"{k}={str(step.evidence[k])[:40]}" for k in ks)
            print(f"    [{step.stage}] {step.decision or ''}{ev}")
        print()


def run_all_disagreements(summary=True):
    pairs = list_disagreement_pairs()
    print(f"Running all variants on {len(pairs)} disagreement pairs...")
    rows = []
    for pair in pairs:
        gold = load_judge_verdict(pair)
        row = {"pair": pair, "gold": gold}
        for v in V.VARIANTS:
            d = V.VARIANTS[v](pair)
            row[v] = d.decision
            row[v + "_auditable"] = d.is_auditable()
        rows.append(row)
    if summary:
        # Aggregate accuracy per variant
        for v in V.VARIANTS:
            n_correct = sum(1 for r in rows if r[v] == r["gold"])
            n = len(rows)
            print(f"  {v:<28s} {n_correct}/{n} = {100*n_correct/n:.1f}% accuracy")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair_id", nargs="?", help="e.g. sigir-20141__NCT02001545")
    ap.add_argument("--variant", help="Only this variant (default: all)")
    ap.add_argument("--all-disagreements", action="store_true",
                    help="Run all variants on all 122 disagreement pairs")
    ap.add_argument("--summary", action="store_true",
                    help="With --all-disagreements: print accuracy summary")
    args = ap.parse_args()

    if args.all_disagreements:
        run_all_disagreements(summary=args.summary)
        return
    if not args.pair_id:
        print("Need a pair_id (e.g. sigir-20141__NCT02001545) or --all-disagreements")
        return
    run_one(args.pair_id, args.variant)


if __name__ == "__main__":
    main()
