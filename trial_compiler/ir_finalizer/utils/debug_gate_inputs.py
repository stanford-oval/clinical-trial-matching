#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# Import from your orchestrator file
from trial_compiler.ir_finalizer.orchestrators.orchestrate import (
    load_snapshot_for_program,
    extract_subcohort_for_eff_tid,
    build_corpus_lookup,
    base_trial_id,
)

# Import gate to build the exact prompt (no LLM call)
from trial_compiler.ir_finalizer.stages.smt_criteria_gate_module import SMTCriteriaGate


def _as_printable(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, indent=2)


def show(subctx, k):
    v = _as_printable(subctx.get(k))
    print(f"\n=== {k} (len={len(v)}) ===")
    print(v if v.strip() else "<EMPTY>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--corpus-jsonl", default=None)
    ap.add_argument("--eff-tid", required=True)
    ap.add_argument("--side", required=True, choices=["inclusion", "exclusion"])
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()

    parent_ctx = load_snapshot_for_program(snapshot_dir, args.eff_tid, args.side)
    if parent_ctx is None:
        raise SystemExit(f"missing snapshot for eff_tid={args.eff_tid} side={args.side}")

    corpus_lookup = {}
    if args.corpus_jsonl:
        corpus_lookup = build_corpus_lookup(
            Path(args.corpus_jsonl).resolve(),
            {base_trial_id(args.eff_tid)},
        )

    subctx = extract_subcohort_for_eff_tid(parent_ctx, args.eff_tid, args.side, corpus_lookup)

    print("\n########## SUBCTX SUMMARY ##########")
    for k in [
        "trial_id_parent",
        "trial_id_effective",
        "cohort_id",
        "cohort_label",
        "cohort_match_method",
        "cohort_match_index",
    ]:
        show(subctx, k)

    print("\n########## KEY FIELDS YOU CARE ABOUT ##########")
    for k in ["shared_context", "subcohort_context", "side_criteria"]:
        show(subctx, k)

    print("\n########## ALSO SHOW RAW INC/EXC FIELDS (for sanity) ##########")
    for k in ["inclusion_criteria", "exclusion_criteria"]:
        show(subctx, k)

    # Build the exact prompt the gate would send (no LLM call)
    gate = SMTCriteriaGate(call_llm=lambda _: "", prompt_template=None, model_name="debug")

    # Using the module’s internal extraction/prompt builder is fine for debugging.
    gi = gate._extract_inputs(subctx, args.side)  # internal, but OK for local debug
    prompt = gate._build_prompt(gi)

    print("\n########## GATE PROMPT (first 6000 chars) ##########\n")
    print(prompt[:6000])
    if len(prompt) > 6000:
        print("\n...(truncated)...\n")


if __name__ == "__main__":
    main()
