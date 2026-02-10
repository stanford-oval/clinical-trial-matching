"""Example: reproduce the paper's accuracy numbers on the 122-pair disagreement subset."""
import pathlib as _pathlib
import sys as _sys

# Runnable from a clone without installing.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from matchers import VARIANTS
from matchers.data import list_disagreement_pairs, load_judge_verdict

pairs = list_disagreement_pairs()

def main() -> None:
    print(f"Running {len(VARIANTS)} variants on {len(pairs)} disagreement pairs...\n")

    print(f"{'variant':<28s} {'correct':>8s}  {'accuracy':>10s}")
    print("-" * 50)
    for name, fn in VARIANTS.items():
        n_correct = 0
        for pair in pairs:
            gold = load_judge_verdict(pair)
            d = fn(pair)
            if d.decision == gold:
                n_correct += 1
        print(f"{name:<28s} {n_correct:>5d}/{len(pairs):<3d}  {100*n_correct/len(pairs):>8.1f}%")


if __name__ == "__main__":
    main()
