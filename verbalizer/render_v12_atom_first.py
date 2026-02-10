#!/usr/bin/env python3
"""Render AEGIS rationales from the structured atom artifact (v12 pipeline).

Inputs:
  --pairs       JSON list of {pair: ...} or pair strings (the 30-pair sample)
  --atoms       atom_artifacts.jsonl from build_atom_artifact.py
  --output      output jsonl (pair, eligibility, rationale, rerender_status)
  --workers     thread pool size

The verbalizer is fed the COMPLETE atom list, not a prose summary. Every
atom gets exactly one bullet, grouped by status.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_PATH = pathlib.Path(os.environ.get(
    "VERBALIZER_PROMPT_PATH",
    str(ROOT / "verbalizer/prompts/_freeform_rationale_v12_atom_first.prompt"),
))
ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "")
KEY = os.environ.get("OPENAI_API_KEY", "")


def llm(prompt: str, max_tokens: int = 3000) -> str:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=json.dumps(body).encode(),
        headers={"api-key": KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def _val_str(v):
    if v is True: return "true"
    if v is False: return "false"
    return "null"


def format_atom_list(inc_atoms: list[dict], exc_atoms: list[dict]) -> str:
    lines = []
    for a in inc_atoms:
        desc = (a.get("description") or "").replace("\n", " ").strip()
        lines.append(f"  inclusion | {desc} | {_val_str(a.get('value'))}")
    for a in exc_atoms:
        desc = (a.get("description") or "").replace("\n", " ").strip()
        lines.append(f"  exclusion | {desc} | {_val_str(a.get('value'))}")
    return "\n".join(lines)


def format_cohorts(cohorts: list[dict]) -> str:
    """Multi-cohort atom listing: each cohort's atoms tagged by variant."""
    blocks = []
    for c in cohorts:
        vid = c.get("variant_id", "?")
        status = "QUALIFYING (SAT)" if c.get("qualifying") else "FAILING (UNSAT)"
        decided = " — deciding cohort" if c.get("is_deciding") else ""
        header = f"  ## Cohort {vid} — {status}{decided}"
        mfs = c.get("min_flip_set") or []
        if mfs and not c.get("qualifying"):
            header += f"\n  ## min-flip-set (decisive blockers for this cohort): {', '.join(mfs)}"
        atoms = format_atom_list(c.get("inclusion_atoms", []), c.get("exclusion_atoms", []))
        blocks.append(header + "\n" + atoms)
    return "\n\n".join(blocks)


def render_one(pair: str, charts: dict, trials: dict, atom_rec: dict,
               prompt_template: str) -> dict:
    pid, parent = pair.split("__", 1)
    chart = charts.get(pid, "")
    trial_text = trials.get(parent, "")
    lo = trial_text.lower()
    i = lo.find("inclusion criteria"); e = lo.find("exclusion criteria")
    inc = trial_text[i:e] if i >= 0 and e > i else (trial_text[i:] if i >= 0 else "")
    exc = trial_text[e:] if e >= 0 else ""

    verdict = atom_rec.get("verdict", "")
    cohorts = atom_rec.get("cohorts") or []
    n_cohorts = atom_rec.get("n_cohorts") or len(cohorts) or 1

    if n_cohorts > 1 and cohorts:
        # Multi-cohort: hand the verbalizer the per-cohort atom list
        atom_list_text = format_cohorts(cohorts)
        # Aggregate min-flip across all cohorts for ineligible
        all_mfs = []
        for c in cohorts:
            for a in (c.get("min_flip_set") or []):
                all_mfs.append(f"[cohort {c.get('variant_id','?')}] {a}")
        min_flip_text = "\n".join(f"  - {a}" for a in all_mfs) if all_mfs else "(none — eligible verdict)"
    else:
        atom_list_text = format_atom_list(
            atom_rec.get("inclusion_atoms", []),
            atom_rec.get("exclusion_atoms", []),
        )
        min_flip = atom_rec.get("min_flip_set") or []
        min_flip_text = "\n".join(f"  - {a}" for a in min_flip) if min_flip else "(none — eligible verdict)"

    cohort_note = (
        f"This trial has {n_cohorts} cohort arms. "
        + ("Patient is eligible because at least one cohort admits them (the qualifying arm is the one marked SAT below); list only that cohort's atoms in the bullets." if verdict == "eligible"
           else "Patient is ineligible across ALL cohorts; list blockers grouped per cohort so the clinician can see why each arm fails.")
    ) if n_cohorts > 1 else "Single-cohort trial."

    prompt = (prompt_template
              .replace("{{CHART}}", chart[:5000])
              .replace("{{INCLUSION}}", inc[:3500])
              .replace("{{EXCLUSION}}", exc[:2500])
              .replace("{{VERDICT}}", verdict)
              .replace("{{COHORT_NOTE}}", cohort_note)
              .replace("{{ATOM_LIST}}", atom_list_text)
              .replace("{{MIN_FLIP_SET}}", min_flip_text))

    # Dynamic token budget: scales with total atoms across cohorts
    if cohorts:
        n_atoms = sum(len(c.get("inclusion_atoms",[])) + len(c.get("exclusion_atoms",[])) for c in cohorts)
    else:
        n_atoms = len(atom_rec.get("inclusion_atoms", [])) + len(atom_rec.get("exclusion_atoms", []))
    mt = max(2000, min(8000, n_atoms * 70 + 1000))

    try:
        raw = llm(prompt, mt)
        obj = json.loads(raw) if raw else {}
        rat = (obj.get("rationale") or "").strip()
        if not rat:
            return {"pair": pair, "eligibility": verdict, "rationale": "", "rerender_status": "empty"}
        return {"pair": pair, "eligibility": verdict, "rationale": rat, "rerender_status": "ok",
                "n_atoms_input": n_atoms}
    except Exception as e:
        return {"pair": pair, "eligibility": verdict, "rationale": "", "rerender_status": f"err:{str(e)[:120]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--atoms", required=True, help="atom_artifacts.jsonl from build_atom_artifact.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if not (ENDPOINT and KEY):
        sys.exit("Need OPENAI_ENDPOINT and OPENAI_API_KEY")

    prompt_template = PROMPT_PATH.read_text()

    # Load corpus
    charts, trials = {}, {}
    for ln in (ROOT/"dataset/clinical_trial/sigir/queries.jsonl").open():
        o = json.loads(ln); charts[o["_id"]] = o.get("text", "")
    for ln in (ROOT/"dataset/clinical_trial/sigir/corpus.jsonl").open():
        try: o = json.loads(ln)
        except: continue
        if o.get("_id"): trials[o["_id"]] = o.get("text", "")

    # Load atom artifacts
    atoms_by_pair = {}
    for ln in pathlib.Path(args.atoms).open():
        try: o = json.loads(ln)
        except: continue
        if o.get("pair"): atoms_by_pair[o["pair"]] = o

    # Pair list
    sample = json.load(open(args.pairs))
    pair_list = []
    if isinstance(sample, list) and sample and isinstance(sample[0], dict):
        pair_list = [r["pair"] for r in sample if r.get("pair")]
    elif isinstance(sample, list):
        pair_list = list(sample)
    pair_list = list(dict.fromkeys(pair_list))
    pair_list = [p for p in pair_list if p in atoms_by_pair]
    print(f"Rendering {len(pair_list)} pairs through v12 atom-first pipeline")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(render_one, p, charts, trials, atoms_by_pair[p], prompt_template)
                for p in pair_list]
        n = 0
        for fut in as_completed(futs):
            try: rec = fut.result()
            except Exception as e:
                print(f"  worker err: {e}"); continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 5 == 0 or n == len(pair_list):
                print(f"  [{n}/{len(pair_list)}]")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
