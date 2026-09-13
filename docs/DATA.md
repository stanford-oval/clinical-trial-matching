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

## SatIR clause index (published)

| | |
|---|---|
| asset | [`trec_trial.db.gz`](https://github.com/stanford-oval/clinical-trial-matching/releases/tag/satir-index-trec2022) |
| size | 31 MB compressed, 172 MB unpacked |
| sha256 (gz) | `de9a18f2e669e79422a6e91a3bfbfab1980da644fc08f3090af23873d73b8016` |
| sha256 (db) | `c9b6215889287a93c9f1299d595f08e59e6d06a4ad5a130ab33672cd0c85df48` |
| contents | 3,963 trials, 50 TREC-2022 patients, 37 tables |

Unpack to `$VERDICT_BUILD/trial.db`. See
[REPRODUCE_SATIR.md](REPRODUCE_SATIR.md).

**There is more than one `trial.db` in a development checkout and they are not
interchangeable.** The published one is the TREC index. A SIGIR index also
exists (2.3 GB), and at least one older build lacks the tables retrieval
queries — pointing at it produces no error, just wrong or empty results. Check
the sha256 above rather than the filename.
