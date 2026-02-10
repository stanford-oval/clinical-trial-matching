#!/usr/bin/env python3
"""Overnight verbalizer-prompt optimization loop.

For each prompt variant in a list:
  1. Re-render AEGIS / V5 / Shah on the 30 clinician-sample pairs
  2. Update UI's clinician_review.json with the new rationales
  3. Run simulated pairwise clinician (gpt-5) on all 32 pair-topics
  4. Run rationale-error audit on all 3 systems
  5. Log: error rate per system, AEGIS pairwise win-rate, per-cell breakdown

Best variant = highest AEGIS pairwise win-rate (with error rate as tiebreaker).
Runs sequentially; takes ~12-15 min per variant.

Run:
    python run_overnight_optimization.py
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = pathlib.Path(os.environ.get("ANNOTATION_FRONTEND",
        pathlib.Path(__file__).resolve().parents[1].parent
        / "clinical-trial-annotation-frontend"))
RUN_DIR = ROOT / "verbalizer"
PROMPTS_DIR = ROOT / "verbalizer/prompts"
RERENDER = ROOT / "verbalizer/rerender_v6.py"
SAMPLE_JSON = ROOT / "experiments/clinician_validation/sample_32pairs_balanced.json"
LOG_DIR = pathlib.Path("/tmp/verbalizer_overnight")
# created on use, not on import (see _ensure_log_dir)
SUMMARY = LOG_DIR / "summary.jsonl"


def run(cmd, cwd=None, timeout=600):
    """Run a shell command, return (rc, stdout, stderr)."""
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def render_system(prompt_path: str, system: str, out_path: str) -> bool:
    env = dict(os.environ); env["VERBALIZER_PROMPT_PATH"] = prompt_path
    cmd = ["python3", str(RERENDER),
           "--pairs", str(SAMPLE_JSON),
           "--system", system,
           "--output", out_path,
           "--workers", "8"]
    p = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    return p.returncode == 0


def audit(rationales_path: str, out_path: str) -> dict:
    cmd = ["python3", str(ROOT/"experiments/accuracy/validation/flag_aegis_rationale_errors.py"),
           "--rationales", rationales_path,
           "--limit", "30",
           "--workers", "10",
           "--model", "gpt-5",
           "--out", out_path]
    subprocess.run(cmd, env=dict(os.environ), cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    rows = [json.loads(l) for l in open(out_path)]
    ok = [r for r in rows if r.get("status")=="ok"]
    fl = [r for r in ok if r.get("definitely_wrong")]
    return {"n": len(ok), "flagged": len(fl), "error_rate": len(fl)/max(1,len(ok))}


def update_ui_rationales(aegis_path, v5_path, shah_path) -> None:
    """Patch the UI's clinician_review.json with new rationale texts."""
    aegis = {json.loads(l)["pair"]: json.loads(l).get("rationale","") for l in open(aegis_path)}
    v5 = {json.loads(l)["pair"]: json.loads(l).get("rationale","") for l in open(v5_path)}
    shah = {json.loads(l)["pair"]: json.loads(l).get("rationale","") for l in open(shah_path)}

    ui_path = FRONTEND / "private/clinician_review.json"
    d = json.loads(ui_path.read_text())
    for t in d["topics"]:
        if t.get("sheet") != "formatch_pairwise_review": continue
        pair = t["id"].split("__", 2)[-1]
        for side in ("A", "B"):
            sys_id = t.get(f"rationale_{side}_system_blind_id","")
            if sys_id == "aegis":
                txt = aegis.get(pair)
            elif "v5" in sys_id.lower():
                txt = v5.get(pair)
            elif sys_id in ("shah", "shahlab"):
                txt = shah.get(pair)
            else:
                txt = None
            if txt: t[f"rationale_{side}_text"] = txt
    ui_path.write_text(json.dumps(d, indent=2))


def simulate_pairwise(out_path: str) -> dict:
    cmd = ["python3", str(ROOT/"experiments/accuracy/validation/simulate_pairwise_clinician.py"),
           "--out", out_path,
           "--workers", "8",
           "--model", "gpt-5"]
    subprocess.run(cmd, env=dict(os.environ), cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    rows = [json.loads(l) for l in open(out_path) if json.loads(l).get("status")=="ok"]
    wins = {"aegis":0, "comp":0, "tie":0}
    for r in rows:
        w = r["sim_pairwise_winner"]
        if w == "A":
            wins["aegis" if r["rationale_A_system"]=="aegis" else "comp"] += 1
        elif w == "B":
            wins["aegis" if r["rationale_B_system"]=="aegis" else "comp"] += 1
        else:
            wins["tie"] += 1
    n = sum(wins.values())
    return {"n": n, "aegis_wins": wins["aegis"], "comp_wins": wins["comp"], "tie": wins["tie"],
            "aegis_win_rate": wins["aegis"]/max(1,n)}


def run_variant(prompt_name: str, prompt_path: str) -> dict:
    """Render → update UI → audit → sim. Returns summary record."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*70}\n[{ts}] variant: {prompt_name}\n{'='*70}")
    t0 = time.time()

    # Render
    aegis_out = str(LOG_DIR / f"{prompt_name}_aegis.jsonl")
    v5_out    = str(LOG_DIR / f"{prompt_name}_v5.jsonl")
    shah_out  = str(LOG_DIR / f"{prompt_name}_shah.jsonl")
    print("  render aegis...", end="", flush=True); render_system(prompt_path, "aegis", aegis_out); print(" done")
    print("  render v5...",    end="", flush=True); render_system(prompt_path, "v5",    v5_out);    print(" done")
    print("  render shah...",  end="", flush=True); render_system(prompt_path, "shah",  shah_out);  print(" done")

    # Audit
    a_err = audit(aegis_out, str(LOG_DIR / f"{prompt_name}_aegis_audit.jsonl"))
    v_err = audit(v5_out,    str(LOG_DIR / f"{prompt_name}_v5_audit.jsonl"))
    s_err = audit(shah_out,  str(LOG_DIR / f"{prompt_name}_shah_audit.jsonl"))
    print(f"  audit: aegis err={a_err['flagged']}/{a_err['n']}  v5 err={v_err['flagged']}/{v_err['n']}  shah err={s_err['flagged']}/{s_err['n']}")

    # Update UI + sim pairwise
    update_ui_rationales(aegis_out, v5_out, shah_out)
    pw = simulate_pairwise(str(LOG_DIR / f"{prompt_name}_sim.jsonl"))
    print(f"  pairwise: AEGIS={pw['aegis_wins']} comp={pw['comp_wins']} tie={pw['tie']}  win_rate={pw['aegis_win_rate']:.1%}")

    dt = time.time() - t0
    rec = {
        "prompt_name": prompt_name,
        "prompt_path": prompt_path,
        "timestamp": ts,
        "duration_sec": int(dt),
        "aegis_error_rate": a_err["error_rate"],
        "v5_error_rate": v_err["error_rate"],
        "shah_error_rate": s_err["error_rate"],
        "aegis_wins": pw["aegis_wins"],
        "comp_wins": pw["comp_wins"],
        "tie": pw["tie"],
        "aegis_win_rate": pw["aegis_win_rate"],
    }
    with SUMMARY.open("a") as f: f.write(json.dumps(rec) + "\n")
    print(f"  Δt: {int(dt)}s")
    return rec


VARIANTS = [
    ("v7_baseline_rep1",      "_freeform_rationale_v7.prompt"),
    ("v7_baseline_rep2",      "_freeform_rationale_v7.prompt"),
    ("v7_baseline_rep3",      "_freeform_rationale_v7.prompt"),
    ("v9e_v5style",           "_freeform_rationale_v9e_v5style.prompt"),
    ("v9f_evidence_first",    "_freeform_rationale_v9f_evidence_first.prompt"),
    ("v9g_thorough",          "_freeform_rationale_v9g_thorough.prompt"),
    ("v9a_structural_rep2",   "_freeform_rationale_v9a_structural.prompt"),
    ("v9b_doctrine_rep2",     "_freeform_rationale_v9b_doctrine.prompt"),
]


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", help="subset of variant names; default all")
    args = ap.parse_args()
    chosen = VARIANTS if not args.variants else [(n,p) for n,p in VARIANTS if n in args.variants]
    print(f"running {len(chosen)} variants → log: {SUMMARY}")
    for name, fname in chosen:
        ppath = str(PROMPTS_DIR / fname)
        if not pathlib.Path(ppath).exists():
            print(f"SKIP {name}: {ppath} not found"); continue
        try:
            run_variant(name, ppath)
        except Exception as e:
            print(f"ERROR {name}: {e}")
    print(f"\nfinal summary at {SUMMARY}")


if __name__ == "__main__":
    main()
