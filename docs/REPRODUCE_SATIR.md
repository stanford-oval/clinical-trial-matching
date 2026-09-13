# Reproducing SatIR's retrieval numbers

SatIR's published recall reproduces **exactly** on TREC 2022. TREC is the right
check because its relevance judgements (qrels) are freely available, unlike the
SIGIR collection.

## Reference

| mode | K | qrels | SatIR | BM25 | TrialGPT-Retrieve |
|---|---|---|---|---|---|
| treat-chief | 25 | label==2 | **0.4563** | 0.1806 | 0.1596 |
| treat-chief | 25 | label>=1 | **0.3702** | 0.2256 | 0.1888 |
| treat-any | 57 | label==2 | **0.5024** | 0.2629 | 0.2842 |
| treat-any | 57 | label>=1 | **0.4067** | 0.2967 | 0.3386 |
| relevant-to-any | 65 | label==2 | **0.5665** | 0.2763 | 0.2897 |
| relevant-to-any | 65 | label>=1 | **0.4590** | 0.3134 | 0.3510 |

`label==2` is strict eligibility (48 patients have at least one); `label>=1` is
any topical relevance (50 patients). K is not chosen — it is SatIR's mean
retrieval depth for that mode, so each baseline is given the same budget.

## What you need

The clause index is published as a release asset:

```bash
curl -L -O https://github.com/stanford-oval/clinical-trial-matching/releases/download/satir-index-trec2022/trec_trial.db.gz
curl -L -O https://github.com/stanford-oval/clinical-trial-matching/releases/download/satir-index-trec2022/trec_trial.db.gz.sha256
shasum -a 256 -c trec_trial.db.gz.sha256
mkdir -p build && gunzip -c trec_trial.db.gz > build/trial.db
export VERDICT_BUILD="$PWD/build"
```

31 MB compressed, 172 MB unpacked; 3,963 trials, 50 TREC-2022 patients.
Retrieval over this exact artifact was checked against the table above: all 18
rows, maximum absolute difference 0.

You also need:
- TREC 2022 qrels, padded subset: `dataset/clinical_trial/trec_2022_subset_padded/qrels/test.tsv`
- the BM25 and TrialGPT-Retrieve baseline retrievals

No LLM endpoint and no services: retrieval here is pure SQL over the index.

## The three modes

Each display name is a specific flag combination. Getting these wrong is the
easiest way to fail to reproduce — the author of this file scored 0.052
instead of 0.456 by using `--important-mode chief --alt-mode act` with
prevention off.

| display name | flags |
|---|---|
| treat-chief | `--important-mode ccr --alt-mode act --enable-prevention-hits` |
| treat-any | `--important-mode all --alt-mode act --enable-prevention-hits` |
| relevant-to-any | `--important-mode all --alt-mode nonact --enable-prevention-hits` |

## Run it

```bash
# for each mode, for each of the 50 patients
python -m sql_retrieval.ops.constraint_retrieval \
    --db "$VERDICT_BUILD/trial.db" --patient trec-20221 \
    --out out/ccr_prevent_act --scope any \
    --important-mode ccr --alt-mode act --enable-prevention-hits \
    --parallel 4 --quiet
```

Then score. Scoring reads `retrieved_mappings__<mode>__prevent__<alt>/labeled_clean`
and counts only lines tagged `all_satisfied` — the survivors — against the
qrels. `sql_retrieval/ops/eval_pr_rec_at_k.py` is a **different** evaluation
with different inputs; it will not give these numbers.

## Two traps

**The scoring set is `labeled_clean`, not `clean`.** `clean` holds every
considered trial; the recall figure counts only the `all_satisfied` subset,
which only `labeled_clean` distinguishes.

**The qrels are the padded subset**, not `trec_2022/qrels/test.tsv`. The
unpadded file covers a different trial universe and will silently score low
rather than error.
