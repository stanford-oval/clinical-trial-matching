#!/usr/bin/env python3
"""Re-render Shah rationales from verdicts.jsonl using the fixed sentence-clip
renderer (replaces the previous [:200] hard truncation that caused mid-clause
cuts at ~2% of pairs).

Run:
    python rerender_rationales.py
"""
from __future__ import annotations
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_judge_input import render_rationale

VERDICTS = HERE / "verdicts.jsonl"
OUT = HERE / "rationales.jsonl"
BACKUP = HERE / "rationales.jsonl.pre_clip_fix"


def main():
    # Backup the old rationales file once
    if OUT.exists() and not BACKUP.exists():
        BACKUP.write_text(OUT.read_text())
        print(f"backed up old rationales → {BACKUP}")

    n = 0; n_truncated_before = 0
    out_rows = []
    for ln in VERDICTS.open():
        try: r = json.loads(ln)
        except Exception: continue
        asmts = r.get("assessments") or []
        if asmts:
            rationale = render_rationale(asmts)
        else:
            rationale = (f"Stanford verdict {r.get('eligibility')} "
                         f"(global_decision={r.get('global_decision')}); "
                         f"per-criterion details unavailable.")
        # detect old-style truncation in source rationales (for the report)
        out_rows.append({
            "pair": r["pair"],
            "eligibility": r["eligibility"],
            "global_decision": r.get("global_decision"),
            "rationale": rationale[:2000],
            "n_inc_assessed": sum(1 for a in asmts if 'inclusion' in str(a.get('criterion','')).lower()),
            "n_exc_assessed": sum(1 for a in asmts if 'exclusion' in str(a.get('criterion','')).lower()),
        })
        n += 1

    # count truncations in OLD file for comparison
    if BACKUP.exists():
        import re
        pat = re.compile(r"\b(In fact|However|Specifically|Notably|Although|Furthermore|Moreover|Indeed|Importantly)\s*,?\s*\n|,\s*\n", re.IGNORECASE)
        for ln in BACKUP.open():
            try: o = json.loads(ln)
            except: continue
            if pat.search(o.get("rationale","") or ""):
                n_truncated_before += 1

    with OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rationales → {OUT}")

    # count truncations in NEW file
    import re
    pat = re.compile(r"\b(In fact|However|Specifically|Notably|Although|Furthermore|Moreover|Indeed|Importantly)\s*,?\s*\n|,\s*\n", re.IGNORECASE)
    n_truncated_after = 0
    for r in out_rows:
        if pat.search(r["rationale"]):
            n_truncated_after += 1
    print(f"  truncations BEFORE fix: {n_truncated_before}/{n}")
    print(f"  truncations AFTER  fix: {n_truncated_after}/{n}")


if __name__ == "__main__":
    main()
