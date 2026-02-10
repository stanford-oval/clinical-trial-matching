"""Example: deeply inspect one variant's audit trail."""
import pathlib as _pathlib
import sys as _sys

# Runnable from a clone without installing.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from matchers import variants

PAIR = "sigir-201414__NCT02192320"  # both-reject case — atoms-only arbiter rescues

d = variants.smt_lm_evidence_arbiter(PAIR)


def main() -> None:
    print(f"Pair: {d.pair_id}")
    print(f"Variant: {d.variant}")
    print(f"Decision: {d.decision}")
    print(f"Auditable: {d.is_auditable()}")
    print(f"Reasoning: {d.reasoning}\n")

    print("Audit trail:")
    for i, step in enumerate(d.audit_trail, start=1):
        print(f"  [{i}] {step.stage}: {step.decision or '—'}")
        if step.rationale:
            print(f"      rationale: {step.rationale[:200]}")
        for k, v in step.evidence.items():
            vs = str(v)[:80]
            print(f"      {k}: {vs}")


if __name__ == "__main__":
    main()
