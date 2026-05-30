# Data provenance and redistribution

This file records where each input comes from, what this repository actually
contains, and what has **not** been cleared for redistribution. Read it before
making the repository public.

## Third-party inputs

| Source | Used for | In this repo? | Redistribution |
|---|---|---|---|
| SIGIR 2016 clinical-trial matching benchmark (Koopman & Zuccon) | 552-pair headline evaluation; synthetic patient vignettes | `dataset/clinical_trial/sigir/` (~16 MB, currently tracked) | **NOT VERIFIED — see below** |
| TREC 2021 Clinical Trials track (Soboroff et al.) | 363-pair independent evaluation; patient topics + qrels | not tracked; obtained from the track | Governed by the track's participant terms |
| ClinicalTrials.gov | trial eligibility criteria text | derived form only | Public domain (US federal) |

### Open item: the SIGIR corpus

`dataset/clinical_trial/sigir/corpus.jsonl` is tracked in git. Publishing the
repository as-is therefore **redistributes the benchmark corpus**, which is a
separate question from citing it. Confirm the benchmark's terms permit
redistribution before going public. If they do not, the fix is to untrack the
corpus and ship a download script instead:

```bash
git rm --cached dataset/clinical_trial/sigir/corpus.jsonl
echo 'dataset/clinical_trial/sigir/corpus.jsonl' >> .gitignore
```

Note that the patient vignettes in both benchmarks are **synthetic** — they are
authored case descriptions, not derived from real patient records — so this is a
licensing question, not a privacy one. See the paper's Data Consent appendix for
the identifier and offensive-content screens we ran over both corpora.

## Data we generated

Derived annotations produced by this project (eligibility labels, counterfactual
edits, judge outputs, clinician audit responses) are ours to release. They are
labels *over* third-party pairs, so they are only meaningful alongside the source
corpora above.

## What lives outside this repository

Not all results are reproducible from this checkout alone.

| Artifact | Location | Needed for |
|---|---|---|
| the matcher itself | **vendored** at `verdict/engine/`; nothing to fetch | `verdict run`, `scripts/reproduce_all.sh` stage 1 |
| TREC 2021 eval (`svpo-rl`) | separate repo; set `$SVPO_RL` | Table 2 — see `REPRODUCE_TABLES.md` |
| SatIR compilation pipeline | `github.com/zikai-zhou/SatIR` | upstream SMT program compilation |

## Secrets

API credentials are read from `.env`, which is gitignored. `.env.example` is
committed and contains placeholders only — no keys, and no internal hostnames.
Never commit `.env`.
