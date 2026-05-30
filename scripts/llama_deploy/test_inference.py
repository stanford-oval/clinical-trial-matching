"""Single-prompt smoke test for LocalHFEngine on a GPU node.

Run AFTER setup_env.sh + huggingface-cli login + clone of the repo:

    cd /scr/$USER/TrialGPT-SMT-Refactored
    python scripts/llama_deploy/test_inference.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/llama_deploy/test_inference.py --model meta-llama/Llama-3.1-70B-Instruct --4bit
"""
from __future__ import annotations
import argparse, sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from smt_core.local_hf_engine import LocalHFEngine


PROMPTS = [
    {"vocab": "L0_met",
     "text": "Output one word: MET, NOT_MET, or UNCLEAR.\n"
             "CRITERION: Exclude if patient has diabetes.\n"
             "PATIENT: Patient has type 2 diabetes mellitus, A1c 8.4.\n"
             "Output:"},
    {"vocab": "L1_excluded",
     "text": "Output one word: NOT_EXCLUDED, EXCLUDED, or UNCLEAR.\n"
             "CRITERION: Exclude if patient has diabetes.\n"
             "PATIENT: Patient has type 2 diabetes mellitus, A1c 8.4.\n"
             "Output:"},
    {"vocab": "L2_eligible",
     "text": "Output one word: ELIGIBLE, INELIGIBLE, or UNCLEAR.\n"
             "CRITERION: Exclude if patient has diabetes.\n"
             "PATIENT: Patient has type 2 diabetes mellitus, A1c 8.4.\n"
             "Output:"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--4bit", action="store_true", dest="four_bit",
                    help="load_in_4bit (recommended for 70B on a single 80GB GPU)")
    args = ap.parse_args()

    print(f"[smoke] model = {args.model}  4bit = {args.four_bit}")
    t0 = time.time()
    eng = LocalHFEngine(model_name=args.model, load_in_4bit=args.four_bit,
                        verbose=True, max_tokens=8)
    print(f"[smoke] load took {time.time() - t0:.1f}s\n")

    for p in PROMPTS:
        t = time.time()
        out = eng(p["text"])[0]
        print(f"  vocab={p['vocab']:<14}  "
              f"output={out!r:30}  "
              f"time={time.time() - t:.2f}s")


if __name__ == "__main__":
    main()
