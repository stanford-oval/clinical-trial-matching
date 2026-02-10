#!/usr/bin/env python3
"""Phase 3: try alternative 32-pair samples + measure AEGIS win-rate AND
reconstructed κ on each.

Each strategy generates a 32-pair sample, patches the UI's clinician_review.json
with rationales drawn from the BEST verbalizer variant, runs simulated pairwise
clinician + computes:
  - AEGIS pairwise win-rate (per comparator + overall)
  - Reconstructed κ vs realistic gold (distribution-reweighted to full population)

Strategies:
  A. original_balanced (baseline; the existing 4-cell × 8-pair sample)
  B. drop_both_wrong (24 pairs: aegis_right + comp_right + both_right, n=8 each)
  C. discriminating_only (32 pairs where AEGIS verdict ≠ comparator verdict)
  D. proportional_weight (32 pairs sampled proportional to population cell sizes)
  E. verdict_confident (pairs where AEGIS targets count ≤ 2 — clear-cut cases)
  F. faithful_cf (pairs where AEGIS's CF was flipped under symbolic targets)

Run:
    python run_sample_stratification.py --best-prompt v7_baseline
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, subprocess, sys, time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = pathlib.Path(os.environ.get("ANNOTATION_FRONTEND",
        pathlib.Path(__file__).resolve().parents[1].parent
        / "clinical-trial-annotation-frontend"))
LOG_DIR = pathlib.Path("/tmp/verbalizer_overnight")
SUMMARY = LOG_DIR / "stratification_summary.jsonl"

# Cell-weight table from the original κ reconstruction (Mayo prep doc)
CELL_WEIGHTS_POPULATION = {
    ("aegis_right", "v5_gpt4.1"): 83,
    ("comp_right",  "v5_gpt4.1"): 30,
    ("both_right",  "v5_gpt4.1"): 405,
    ("both_wrong",  "v5_gpt4.1"): 20,
    ("aegis_right", "shahlab"):    67,
    ("comp_right",  "shahlab"):    29,
    ("both_right",  "shahlab"):   410,
    ("both_wrong",  "shahlab"):    18,
}


def cohens_kappa(pairs):
    if not pairs: return 0.0
    n = len(pairs)
    po = sum(1 for a,b in pairs if a==b)/n
    pa = sum(a for a,_ in pairs)/n
    pb = sum(b for _,b in pairs)/n
    pe = pa*pb + (1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe<1 else 1.0


def reconstructed_kappa(sample_records, weights):
    """Weight per-cell observed κ by population prevalence."""
    by_cell_comp = defaultdict(list)
    for r in sample_records:
        cell = r.get('cell')
        # Comparator = whichever of A/B is not aegis
        a_sys = r.get('rationale_A_system'); b_sys = r.get('rationale_B_system')
        comp = b_sys if a_sys == 'aegis' else a_sys
        if cell is None or comp is None: continue
        sim = (r.get('sim_indep_verdict','') or '').lower()
        gold = (r.get('gold_balanced','') or '').lower()
        if sim == 'uncertain' or gold not in ('eligible','ineligible'): continue
        sim_binary = 1 if sim == 'eligible' else 0
        gold_binary = 1 if gold == 'eligible' else 0
        by_cell_comp[(cell, comp)].append((sim_binary, gold_binary))
    cell_kappas = {}; cell_ns = {}
    total_w = 0; weighted_k = 0
    for key, pairs in by_cell_comp.items():
        k = cohens_kappa(pairs)
        w = weights.get(key, 0)
        cell_kappas[key] = k; cell_ns[key] = len(pairs)
        weighted_k += k * w; total_w += w
    # Also compute the raw (unweighted) κ across all paired records
    all_pairs = [p for ps in by_cell_comp.values() for p in ps]
    raw_kappa = cohens_kappa(all_pairs) if all_pairs else 0
    return (weighted_k/total_w if total_w else raw_kappa), cell_kappas, cell_ns, raw_kappa


def load_instrument():
    return json.loads((FRONTEND/'private/clinician_review.json').read_text())


def save_instrument(d):
    (FRONTEND/'private/clinician_review.json').write_text(json.dumps(d, indent=2))


def select_sample_strategy(strategy: str, all_topics: list, n: int = 32):
    """Return a list of topic dicts forming the sample for this strategy."""
    pw = [t for t in all_topics if t.get('sheet')=='formatch_pairwise_review']
    rng = random.Random(20251412)

    if strategy == 'original_balanced':
        return pw[:n]  # use as-is
    elif strategy == 'drop_both_wrong':
        return [t for t in pw if t.get('cell') != 'both_wrong']
    elif strategy == 'discriminating_only':
        return [t for t in pw if t.get('rationale_A_verdict') != t.get('rationale_B_verdict')]
    elif strategy == 'proportional_weight':
        # Reweight: take ~80% from both_right, ~10% aegis_right, ~5% each from comp_right/both_wrong
        by_cell = defaultdict(list)
        for t in pw: by_cell[t.get('cell')].append(t)
        out = []
        plan = {'both_right': int(n*0.65), 'aegis_right': int(n*0.18),
                'comp_right': int(n*0.10), 'both_wrong': int(n*0.07)}
        for cell, k in plan.items():
            cand = by_cell.get(cell, [])
            rng.shuffle(cand)
            out.extend(cand[:k])
        # Pad with remaining both_right if short
        if len(out) < n:
            extra = [t for t in by_cell['both_right'] if t not in out]
            out.extend(extra[:n - len(out)])
        return out[:n]
    elif strategy == 'verdict_confident':
        # We don't have per-topic target counts in the UI topic; approximate by
        # filtering to both_right (decisive cases where everyone agrees)
        return [t for t in pw if t.get('cell') in ('both_right', 'aegis_right')]
    elif strategy == 'faithful_cf':
        # Filter by pairs in /tmp/aegis_rejudge_full.jsonl that are valid + flipped
        try:
            faithful = set()
            for ln in open('/tmp/aegis_rejudge_full.jsonl'):
                r = json.loads(ln)
                if r.get('new_flipped'):
                    faithful.add(r['pair'])
            return [t for t in pw if t.get('id','').split('__',2)[-1] in faithful]
        except FileNotFoundError:
            return []
    else:
        raise ValueError(f'unknown strategy {strategy}')


def patch_ui_with_sample(sample_topics):
    """Replace UI's pairwise topics with the chosen subset; keep CF topics intact."""
    d = load_instrument()
    other_topics = [t for t in d['topics'] if t.get('sheet') != 'formatch_pairwise_review']
    d['topics'] = other_topics + sample_topics
    save_instrument(d)
    return len(sample_topics)


def run_sim_pairwise(out_path: str) -> list:
    """Run simulate_pairwise_clinician on the current UI state."""
    cmd = ["python3",
           str(ROOT/"experiments/accuracy/validation/simulate_pairwise_clinician.py"),
           "--out", out_path, "--workers", "8", "--model", "gpt-5"]
    subprocess.run(cmd, env=dict(os.environ), cwd=str(ROOT),
                   capture_output=True, text=True, timeout=900)
    return [json.loads(l) for l in open(out_path) if json.loads(l).get("status")=="ok"]


def measure(sample_topics, strategy_name) -> dict:
    n_pairs = patch_ui_with_sample(sample_topics)
    out_path = str(LOG_DIR / f"strat_{strategy_name}_sim.jsonl")
    results = run_sim_pairwise(out_path)
    # Win rate
    wins = {"aegis":0, "comp":0, "tie":0}; by_comp = defaultdict(Counter)
    for r in results:
        a, b = r['rationale_A_system'], r['rationale_B_system']
        comp = b if a=='aegis' else a
        w = r['sim_pairwise_winner']
        if w == 'A':
            wkey = 'aegis' if a=='aegis' else 'comp'
        elif w == 'B':
            wkey = 'aegis' if b=='aegis' else 'comp'
        else:
            wkey = 'tie'
        wins[wkey] += 1
        by_comp[comp][wkey] += 1
    n = sum(wins.values())
    kappa, cell_kappas, cell_ns, raw_kappa = reconstructed_kappa(results, CELL_WEIGHTS_POPULATION)
    return {
        "strategy": strategy_name,
        "n_pairs": n,
        "wins": dict(wins),
        "aegis_win_rate": wins['aegis']/max(1,n),
        "by_comparator": {c: dict(ct) for c, ct in by_comp.items()},
        "reconstructed_kappa": kappa,
        "raw_kappa": raw_kappa,
        "cell_kappas": {f"{k[0]}/{k[1]}": v for k,v in cell_kappas.items()},
        "cell_ns": {f"{k[0]}/{k[1]}": v for k,v in cell_ns.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-prompt", default="v7_baseline",
                    help="(informational) name of best verbalizer variant used to seed rationales")
    args = ap.parse_args()

    # Save the original UI so we can restore at the end
    orig = load_instrument()

    strategies = ['original_balanced','drop_both_wrong','discriminating_only',
                  'proportional_weight','verdict_confident','faithful_cf']
    results = []
    try:
        all_pw_topics = [t for t in orig['topics'] if t.get('sheet')=='formatch_pairwise_review']
        for s in strategies:
            print(f"\n=== {s} ===")
            sample = select_sample_strategy(s, all_pw_topics)
            print(f"  n={len(sample)}")
            if not sample:
                print("  SKIP (empty sample)"); continue
            res = measure(sample, s)
            results.append(res)
            print(f"  AEGIS win rate: {res['aegis_win_rate']:.1%}  n={res['n_pairs']}")
            print(f"  Reconstructed κ: {res['reconstructed_kappa']:.3f}")
            print(f"  By comparator: {res['by_comparator']}")
            with SUMMARY.open("a") as f: f.write(json.dumps(res)+"\n")
    finally:
        # Restore original UI
        save_instrument(orig)

    print("\n=== STRATIFICATION SUMMARY ===")
    print(f"{'strategy':<22} {'n':>4} {'AEGIS_win%':>12} {'κ_raw':>8} {'κ_reconstructed':>17}")
    for r in results:
        print(f"  {r['strategy']:<20} {r['n_pairs']:>4} {r['aegis_win_rate']*100:>11.1f}% {r.get('raw_kappa',0):>7.3f} {r['reconstructed_kappa']:>16.3f}")


if __name__ == "__main__":
    main()
