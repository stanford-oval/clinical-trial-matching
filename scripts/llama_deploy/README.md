# Llama deploy + mech-interp on Stanford SC cluster

Step-by-step. **Do not run heavy work on `scdt` or `sc` login nodes** (per cluster
MOTD — they're investigating resource contention with Claude / Cursor right now).
All actual compute should run on a GPU node via the scheduler.

## 0. SSH config (your laptop)

Add to `~/.ssh/config`:

```
Host sc-cluster
    HostName sc.stanford.edu
    User zikai
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
    ProxyJump nlp-cluster
```

`ProxyJump` routes through `scdt` only briefly (file transfer / SSH handshake).

## 1. Get the repo onto the cluster

From your laptop:

```bash
# Push current branch (commit first if needed)
cd <path-to>/clinical-trial-matching
git push origin main          # or your branch

# On the cluster (scdt is the data-transfer node, OK for cloning + rsync only)
ssh sc-cluster
mkdir -p /scr/$USER
cd /scr/$USER
git clone <repo-url> TrialGPT-SMT-Refactored
cd TrialGPT-SMT-Refactored
git checkout <branch>
```

## 2. Probe the scheduler

```bash
ssh sc-cluster
sinfo -o "%P %N %G"     # partitions + nodes + GPU types
squeue -u $USER          # your queue (should be empty)
sinfo -p <partition> -o "%n %G %m %f"  # detailed per-node
```

Note which partition gives you A6000 / A100-40G (for 8B) or A100-80G / H100
(for 70B).  Update the partition name in the two `slurm_*.sbatch` files.

## 3. One-time env setup on a GPU node

Get an interactive session:

```bash
srun --partition=<your_partition> --gres=gpu:1 --mem=64G \
     --cpus-per-task=8 --time=1:00:00 --pty bash
```

On the GPU node:

```bash
cd /scr/$USER/TrialGPT-SMT-Refactored
bash scripts/llama_deploy/setup_env.sh
huggingface-cli login        # paste your HF token (one-time)
```

`setup_env.sh` creates conda env `polar` with PyTorch, transformers,
transformer_lens, bitsandbytes, sae_lens, nnsight.

## 4. Smoke test inference

In the same interactive session:

```bash
conda activate polar
python scripts/llama_deploy/test_inference.py --model meta-llama/Llama-3.1-8B-Instruct
```

Expected output: three lines printing `EXCLUDED` / `INELIGIBLE` / `NOT_MET` (or
`UNCLEAR` if the model hedges — that itself is the phenomenon we're studying).

For 70B: `python scripts/llama_deploy/test_inference.py --model meta-llama/Llama-3.1-70B-Instruct --4bit`
(needs A100-80G or H100 to fit, even at 4-bit).

## 5. Full toy on Llama 8B (batch)

```bash
sbatch scripts/llama_deploy/slurm_toy_8b.sbatch
```

This runs the synthetic toy under all 6 vocabs × 6 surface forms × 580 cells =
~21k inferences.  Expected wall-clock: 1–3 hours on a single A6000.

Result file: `experiments/12_policy_invariance/datasets/synthetic_toy/out/toy_judge_llama8b.jsonl`

Analyse with:

```bash
python experiments/12_policy_invariance/datasets/synthetic_toy/analyze_toy.py \
    --in experiments/12_policy_invariance/datasets/synthetic_toy/out/toy_judge_llama8b.jsonl \
    --out experiments/12_policy_invariance/datasets/synthetic_toy/out/toy_summary_llama8b.md
```

## 6. Mech-interp Step 1 — logit-lens trajectories

Once the 8B inference replicates the phenomenon, run logit lens on the UNSAT
cases where L0 fails and L1 succeeds:

```bash
srun --partition=<your_partition> --gres=gpu:a100:1 --mem=64G \
     --cpus-per-task=8 --time=2:00:00 --pty bash

conda activate polar
cd /scr/$USER/TrialGPT-SMT-Refactored
python experiments/12_policy_invariance/mech_interp/logit_lens_llama.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vocabs L0_met,L1_excluded \
    --surface v0_natural \
    --gt-filter UNSAT \
    --max-rows 50
```

Output: `experiments/12_policy_invariance/mech_interp/out/logit_lens.jsonl`,
one row per (rule, evidence_case, vocab, layer) with per-layer probabilities of
positive / negative / UNCLEAR label tokens.

The headline paper plot is `mean P(negative_label) by layer × vocab`:
- L1 (EXCLUDED): rises by layer ~15, stays high to the end.
- L0 (NOT_MET): rises in middle layers but the final layers smear it back to UNCLEAR.

If the trajectories show this pattern, that's the **mechanistic localization**:
RLHF-induced hedging happens in the late layers.

## 7. Next mech-interp steps (not yet scripted)

- **Activation patching** — replace L0's late-layer residual stream with L1's,
  see if the model commits.  Library: TransformerLens `run_with_hooks` +
  `patching` module.
- **Base vs Instruct** — re-run the toy + logit lens on Llama-3.1-8B-Base.  If
  Base is symmetric and Instruct is hedged, that's RLHF causal evidence.
- **SAE feature analysis** — load Goodfire's SAE for Llama-3-8B, find a feature
  that fires preferentially under L0 hedging, ablate, measure gap closure.

Each is roughly 2–3 days of work on top of step 6.
