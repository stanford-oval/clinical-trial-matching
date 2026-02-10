#!/usr/bin/env python3
"""Re-render rationales using the v6 freeform prompt for fluency-fair clinician evaluation.

Reads existing system outputs (verdict + rationale or structured per-criterion text),
applies v6 prompt, writes uniform-style prose. Verdicts are fixed; only the rationale
text changes.

Run:
    python verbalizer/rerender_v6.py \
        --pairs /path/to/sample_32pairs_balanced.json \
        --system aegis|v5|shah \
        --output /tmp/{system}_v6_rationales.jsonl
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_PATH = pathlib.Path(os.environ.get(
    "VERBALIZER_PROMPT_PATH",
    str(ROOT / "verbalizer/prompts/_freeform_rationale_v6.prompt"),
))

ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "")
KEY = os.environ.get("OPENAI_API_KEY", "")


def llm(prompt: str, max_tokens: int = 600) -> str:
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
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def load_jsonl(p, key="pair"):
    out = {}
    if not p.exists(): return out
    for ln in p.open():
        try: o = json.loads(ln)
        except: continue
        if o.get(key): out[o[key]] = o
    return out


def render_one(pair: str, system: str, charts: dict, trials: dict,
               src_records: dict, prompt_template: str) -> dict:
    pid, parent = pair.split("__", 1)
    chart = charts.get(pid, "")
    trial_text = trials.get(parent, "")
    lo = trial_text.lower()
    i = lo.find("inclusion criteria"); e = lo.find("exclusion criteria")
    inc = trial_text[i:e] if i >= 0 and e > i else (trial_text[i:] if i >= 0 else "")
    exc = trial_text[e:] if e >= 0 else ""

    src = src_records.get(pair, {})
    verdict = (src.get("eligibility") or "").strip().lower()
    original = src.get("rationale") or src.get("explanation") or ""

    if not verdict or not original:
        return {"pair": pair, "eligibility": verdict, "rationale": "(no source)"}

    prompt = (prompt_template
              .replace("{{CHART}}", chart[:5000])
              .replace("{{INCLUSION}}", inc[:3500])
              .replace("{{EXCLUSION}}", exc[:2000])
              .replace("{{VERDICT}}", verdict)
              .replace("{{ARTIFACTS}}", original[:3500]))
    try:
        raw = llm(prompt)
        obj = json.loads(raw) if raw else {}
        new_rat = (obj.get("rationale") or "").strip()
        if not new_rat:
            return {"pair": pair, "eligibility": verdict, "rationale": original, "rerender_status": "empty"}
        return {"pair": pair, "eligibility": verdict, "rationale": new_rat, "rerender_status": "ok"}
    except Exception as e:
        return {"pair": pair, "eligibility": verdict, "rationale": original, "rerender_status": f"err:{str(e)[:120]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="32-pair sample JSON or any list of {pair} records")
    ap.add_argument("--system", required=True, choices=["aegis", "v5", "shah"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not (ENDPOINT and KEY):
        sys.exit("Need OPENAI_ENDPOINT and OPENAI_API_KEY")

    prompt = PROMPT_PATH.read_text()

    # Load corpus
    charts = {}
    for ln in (ROOT/"dataset/clinical_trial/sigir/queries.jsonl").open():
        o = json.loads(ln); charts[o["_id"]] = o.get("text", "")
    trials = {}
    for ln in (ROOT/"dataset/clinical_trial/sigir/corpus.jsonl").open():
        try: o = json.loads(ln)
        except: continue
        if o.get("_id"): trials[o["_id"]] = o.get("text", "")

    # Source records (verdict + original rationale) per system
    if args.system == "aegis":
        src = load_jsonl(pathlib.Path("/tmp/aegis_v4_freeform.jsonl"))
    elif args.system == "v5":
        src = load_jsonl(ROOT/"matchers/systems/single_shot_llm/v5_freeform.jsonl")
    else:  # shah — prefer canonical; fall back to freeform for any missing pairs
        src = load_jsonl(ROOT/"matchers/systems/shahlab/rationales.jsonl")
        ff = load_jsonl(ROOT/"backup/overnight/shahlab_freeform.jsonl")
        for p, rec in ff.items():
            if p not in src:
                src[p] = rec

    # Read pairs to process
    pair_list = []
    sample = json.load(open(args.pairs))
    for r in (sample if isinstance(sample, list) else []):
        if isinstance(r, dict) and r.get("pair"):
            pair_list.append(r["pair"])
    pair_list = list(dict.fromkeys(pair_list))  # dedup, preserve order
    print(f"Processing {len(pair_list)} pairs for {args.system}")

    out_path = pathlib.Path(args.output)
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(render_one, p, args.system, charts, trials, src, prompt) for p in pair_list]
        n = 0
        for fut in as_completed(futs):
            try: rec = fut.result()
            except Exception as e:
                print(f"  worker err: {e}"); continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 5 == 0 or n == len(pair_list):
                print(f"  [{n}/{len(pair_list)}]")
    print(f"done → {out_path}")


if __name__ == "__main__":
    main()
