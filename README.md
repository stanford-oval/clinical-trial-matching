<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/icons/umbrella-mono-white.svg">
  <img src="assets/icons/umbrella-display.svg" alt="" width="72" align="left" hspace="12" vspace="4">
</picture>

# Stanford OVAL Clinical Trial Matcher

Software that finds clinical trials a patient might qualify for, decides
whether they actually meet the criteria, and shows its reasoning so a
clinician can check it.

**Status.** Actively developed. The reproduction path, both CLIs, both Python
APIs and the accountability artifacts are tested and covered by CI. Some areas
are known to be incomplete — see [Known gaps](#known-gaps). **Issues and
questions are welcome**, especially reproduction failures, unclear
documentation, and any place where the code and the paper disagree. Single
maintainer on an academic timeline, so replies may be slow.

It comes in two parts, which run in order:

| | what it does | command |
|---|---|---|
| **SatIR** | Searches a large trial corpus and narrows it to the handful worth a closer look. To do that it first turns each trial's eligibility criteria, and each patient note, into structured facts a computer can compare. | `satir` |
| **VERDICT** | Takes one patient and one trial and answers *does this patient qualify?* — along with which criterion decided it, what it had to assume because the chart was silent, and what would change the answer. | `verdict` |

The two share a common core (`smt_core`) that reads clinical language and
normalizes it: mapping "kidney failure" and "renal insufficiency" to the same
concept, pulling out numbers and units, and so on.

Why bother with the second part? A language model asked "is this patient
eligible?" will answer, but you cannot tell whether its explanation is the
real reason for its answer. VERDICT separates *reading* the chart from
*deciding* the verdict: a solver makes the decision from the extracted facts,
so the explanation is generated from the same thing that produced the answer,
not written afterwards.

```bash
pip install -e .
satir --help             # find candidate trials
verdict systems          # decide a patient-trial pair
```


> **Evaluation corpora are obtained separately.** The SIGIR and TREC clinical-
> trial collections are third-party and carry their own terms, so they are not
> redistributed here; see [docs/DATA.md](docs/DATA.md) for how to obtain them
> and point the tool at them. Everything needed to compile a trial and decide a
> patient is in this repository.

## Papers

| System | Description | Paper |
| --- | --- | --- |
| <img src="assets/icons/satir-display.svg" width="32" alt=""> SatIR | Scalable, high-recall constraint-satisfaction-based retrieval for clinical trial matching | [arXiv:2604.08849](https://arxiv.org/abs/2604.08849) |
| <img src="assets/icons/verdict-display.svg" width="32" alt=""> VERDICT | Accountable eligibility decisions with grounded, faithful, consistent, actionable rationales | [arXiv:2609.03366](https://arxiv.org/abs/2609.03366) |

## Citation

If you use this work, please cite:

<details>
<summary>SatIR</summary>

```bibtex
@article{zhou2026satir,
  title={SatIR: Scalable High-Recall Constraint-Satisfaction-Based Information
         Retrieval for Clinical Trials Matching},
  author={Zhou, Zikai and Jin, Yufei and Xu, Yilin and Wang, Yu-Chiang and
          Chao, Chieh-Ju and Lam, Monica S},
  journal={arXiv preprint arXiv:2604.08849},
  year={2026}
}
```
</details>

<details>
<summary>VERDICT</summary>

```bibtex
@article{zhou2026accountable,
  title={Accountable AI with Grounded, Faithful, Consistent, Actionable
         Rationales: A Case Study in Clinical Trial Matching with VERDICT},
  author={Zhou, Zikai and Jin, Yufei and Xu, Yilin and Wang, Yu-Chiang and
          Chao, Chieh-Ju and Lam, Monica S},
  journal={arXiv preprint arXiv:2609.03366},
  year={2026}
}
```
</details>

## Quickstart

```bash
pip install -e .
cp .env.example .env      # fill in your API credentials
source .env
```

### SatIR — retrieval and compilation

```bash
satir setup               # validate environment and prerequisites
satir info
satir compile-trial ...   # trial text  -> constraint program
satir compile-patient ... # patient note -> coded facts
satir index               # build the clause database
satir retrieve ...        # candidate trials at corpus scale
```

Run `satir <subcommand> --help` for arguments. Services (SNOMED Snowstorm,
Elasticsearch) are described in `docs/QUICKSTART.md`; build artifacts in
`docs/BUILD_ARTIFACTS.md`.

### VERDICT — eligibility matching

Decide a patient--trial pair:

```bash
verdict list --limit 5
verdict match   sigir-20141__NCT00337116
verdict explain sigir-20141__NCT00337116     # with the audit trail
verdict systems                              # the matcher variants
```

```
pair    : sigir-20141__NCT00337116
system  : verdict
decision: ELIGIBLE
why     : SMT solver accepted; no arbitration needed.

audit trail (2 steps)
1. atom_mining
     evidence: {'n_atoms': 40}
2. smt_solve
     decision: eligible
```

Or from Python:

```python
from matchers import variants
d = variants.smt_lm_evidence_arbiter("sigir-20141__NCT00337116")
print(d.decision, d.reasoning)
for step in d.audit_trail:
    print(step)
```

## End to end: screen a patient

The two systems in sequence — retrieve candidates, then decide each one.

```bash
verdict screen sigir-20141 --db path/to/trial.db --limit 10
```

```
patient  : sigir-20141
retrieved: 10 candidate trial(s)
decided  : 3   eligible: 2

 rank  trial           verdict         why
    1  NCT02357212     ELIGIBLE        SMT solver accepted
    2  NCT00006055     not evaluated   no stage-1 data for this pair
```

```python
import pipeline

for r in pipeline.screen("sigir-20141", db="build/trial.db", limit=10):
    print(r.rank, r.nct_id, r.decision or "undecided", r.pivotal)
```

**Undecided is not ineligible.** A candidate VERDICT could not evaluate comes
back with `decision=None` and `eligible=None` — never `False`. For a trial
matcher those must not be confused: one means the patient does not qualify,
the other means nobody looked.

Retrieval needs a clause database and is pure SQL — no LLM, no services. The
decision step needs per-pair stage-1 artifacts under `$VERDICT_PAIR_DATA`.
See [docs/DATA.md](docs/DATA.md) for both.

## Python API

Both systems are importable, not just command-line tools.

### SatIR

```python
import satir

satir.config()                              # resolved SatIRConfig
satir.compile_trial("NCT00337116")          # trial text   -> SMT constraints
satir.compile_patient("sigir-20141")        # patient note -> coded facts
satir.index()                               # constraints  -> clause database
satir.retrieve()                            # SQL constraint-satisfaction retrieval
satir.match("sigir-20141", "NCT00337116")   # SMT eligibility check
```

The compiler and indexer stages are argv-driven underneath, so extra options
pass straight through rather than being re-declared in a parallel schema that
could drift from the real parser:

```python
satir.compile_trial("NCT00337116", "--side", "inclusion", "--stop-after", "ir")
```

For programmatic matching, the genuinely reusable pieces are re-exported:
`satir.load_patient`, `satir.build_ctx_from_persisted`, `satir.run_match_for_side`.

Imports are lazy — `import satir` pulls in no heavy backend, so reading config
costs nothing. Touching a function that needs one raises with the extra to
install (`pip install -e '.[llm]'`).

### VERDICT

```python
import verdict

verdict.systems()                           # {name: description}
d = verdict.match("sigir-20141__NCT00337116")
print(d.decision, d.reasoning)              # 'eligible', 'SMT solver accepted; ...'
for step in d.audit_trail:
    print(step.stage, step.decision)

print(verdict.explain(pair_id))             # rendered audit trail
verdict.match(pair_id, system="hybrid")     # a different variant
```

**The API defaults to `strict=True`**, so a pair that cannot be loaded raises
`MissingPairData` instead of returning a confident `ineligible`. Pass
`strict=False` for the paper-compatible sentinel:

```python
verdict.match("nope__NCT0")                  # raises MissingPairData
verdict.match("nope__NCT0", strict=False)    # Decision(reasoning='no data')
```

## What the system tells you about a decision

Every verdict comes with three things beyond yes/no. Using the paper's running
example — a trial that needs `egfr` true and creatinine clearance of at least
60, and a chart that records a failing eGFR but says nothing about clearance:

```python
from smt_core.maxsmt import Condition, solve, OBSERVED, UNRESOLVED

phi = ["(declare-const |egfr| Bool)", "(declare-const |crcl| Real)",
       "(assert |egfr|)", "(assert (>= |crcl| 60))"]
a = solve(phi, [Condition("egfr", False, OBSERVED),      # chart says: failing
                Condition("crcl", None, UNRESOLVED)])    # chart is silent
```

| what you get | in the example | what it means |
|---|---|---|
| `a.decision` | `'ineligible'` | the answer |
| `a.trace` | why | how the solver got there |
| `a.assumptions` | `{'crcl': ...}` | **what it had to assume.** The chart never mentions clearance, so the system filled it in to reach an answer. Every one of these is something a clinician should check. |
| `a.pivotal` | `['egfr']` | **what would change the answer.** The eGFR result is the one thing standing between this patient and the trial. |

`a.delta_e` and `a.delta_i` split that last one: what would have to change to
make the patient *eligible*, and what would make them *ineligible*. Whichever
is relevant becomes `pivotal`.

Three properties hold by construction, and the tests check them: a patient is
eligible exactly when nothing needs to change to make them eligible; ineligible
exactly when nothing needs to change to make them ineligible; and there is
always something that would flip the answer.

### Where the artifacts come from

`verdict.explain()` appends them automatically when the pair has a stored SMT
program and a solver is installed:

```
audit trail (2 steps)
1. atom_mining
2. smt_solve -> eligible

accountability artifacts (inclusion, maxsmt)
  decision by solver : eligible
  assumed (7 conditions the chart did not settle):
    ischemic_symptoms_duration...: assumed to meet >= 5.0
    patient_has_been_admitted_to_intensive_cardiac_care_unit_now: assumed True
  would change the answer:
    patient_age_value_recorded_now_in_years
```

Pass `artifacts=False` to suppress it. If no program is stored or no solver is
installed, the audit trail is printed exactly as before.

**The two inputs mean different things and are read from different places**
(`verdict/artifacts.py`):

- **`phi_t`** — the trial's requirements, from `smt_program_lines`. Hard
  constraints; the patient never appears in them.
- **conditions** — every variable `phi_t` declares, each with a status. A value
  in `patient_var_values` means the chart settled it (`OBSERVED`); `None`
  means the chart was silent (`UNRESOLVED`), and the solver's value for it
  becomes an assumption.

Every declared variable must appear as a condition. One left out is *free*, so
the solver can satisfy `not phi` through it while keeping every patient
constraint — which silently empties "what would change the answer". That is a
real bug this code hit on real data, and a test now guards it.

### An assumed value is not a measurement

When the chart is silent about a number, the solver picks *some* value that
satisfies the trial. For "clearance of at least 60" it might pick 60, 72, or
500 — all equally valid, none a fact about this patient.

So the explanation must state **the requirement, not the number the solver
picked**. Saying "creatinine clearance 72" would invent a lab result:

```python
a.for_verbalizer(phi)["assumptions"]["crcl"]
# {'requirement': '>= 60', 'witness': 60.0}
#   report the requirement; the witness is just the solver's pick
```

`verbalizer/prompts/_freeform_rationale_v13_maxsmt.prompt` carries this rule.
It is a new prompt rather than an edit to the old one, so rationales published
with the earlier version can still be reproduced.

### Two versions of the algorithm

The paper changed between the submitted and updated versions, so both are
implemented. The difference is real, not a rename.

```python
from smt_core.maxsmt import solve, RESIDUAL, MAXSMT

solve(phi, conds, version=RESIDUAL)   # submitted paper
solve(phi, conds, version=MAXSMT)     # updated paper (default)
```

In the submitted version, the unanswered conditions were reported as
*requirements still outstanding* (`crcl >= 60`). In the update they are
reported as *assumptions the system made* (`crcl = 60`) — a requirement and a
value satisfying it are different claims. The update also computes both
directions of "what would change the answer" separately, where the submitted
version computed one.

| | `RESIDUAL` (submitted) | `MAXSMT` (update) |
|---|---|---|
| unanswered conditions | requirements outstanding: `{crcl: '>= 60'}` | assumptions made: `{crcl: 60.0}` |
| what would flip it | one calculation | both directions, separately |
| chart-silent value supplied by policy | called `ASSUMED` | called `IMPUTED` |

`ASSUMED` is still accepted, so records written under either version load
unchanged.

### Tying the output back to the paper

If you are reading alongside the paper, `to_paper()` labels every field with
its symbol and the step that produces it, so the code and the paper cannot
drift apart:

```python
print(a.describe())
# VERDICT artifacts  [maxsmt = update paper]
#   d        (Step 2 ) ineligible
#   rho      (Step 4 ) {'crcl': 60.0}
#   delta    (Step 6 ) ['egfr']
#   delta_E  (Step 3 ) ['egfr']
#   delta_I  (Step 5 ) []
```

Fields that a version does not compute are left out of the export rather than
returned empty, so "we did not calculate this" cannot be misread as "we
calculated it and found nothing".

### Examples

```bash
python examples/02_read_the_artifacts.py     # needs only z3-solver
python examples/03_add_your_own_matcher.py   # needs nothing
python examples/01_decide_a_pair.py          # needs pair data
```

See [examples/README.md](examples/README.md). CI runs the first two, so they
cannot fall out of date.

### Adding your own matcher

A matcher is any callable taking a pair id and returning a `Decision`.
Register it and it is available everywhere the built-ins are, including the
`verdict` command's `--system` flag:

```python
import verdict
from matchers.schema import Decision

@verdict.register("my-matcher", description="my approach")
def my_matcher(pair_id):
    return Decision(pair_id, "my-matcher", "eligible", "because ...", [])
```

Replacing a built-in requires `override=True` — shadowing `verdict` silently
would make two people's results incomparable.

### Tests and checks

```bash
pytest                                    # fast unit suite (~0.3s, no services)
python scripts/check_invariants.py        # ~1s, 26 checks
```

Run this after every change. It covers the CLI, the Python API, the
reproduction numbers (asserted against the values printed in the paper, not
merely "exits 0"), dependency/import health, and secret and machine-path
hygiene. Checks whose local-only data is absent SKIP with the reason rather
than failing, so it is also meaningful in a fresh clone. CI runs it on 3.12
and 3.13.

### Missing data must not look like a "no"

If the system cannot find the data for a pair, the underlying matcher returns
`ineligible` with the note `"no data"`. That is how the paper's numbers were
produced, so it stays the default — but it is a trap for anything built on
top: a missing file then looks exactly like a patient who genuinely does not
qualify. For a trial matcher that is the dangerous direction, because the
patient quietly never gets surfaced.

So either check for it, or ask to be told loudly:

```python
from matchers import variants
from matchers.schema import MissingPairData

d = variants.smt_lm_evidence_arbiter(pair_id)
if d.is_missing_data:
    ...                      # this is not a verdict

variants.strict(True)        # or: export VERDICT_STRICT=1
try:
    d = variants.smt_lm_evidence_arbiter(pair_id)
except MissingPairData:
    ...                      # raises instead of guessing
```

The `verdict` command and the `verdict` Python package both do this for you:
they refuse an unknown pair rather than returning an answer.

### What this does not do yet

`verdict` works on pairs that have already been processed — it reads
per-pair files from `$VERDICT_PAIR_DATA` (default `experiments/53_v2_full`).

**The end-to-end path exists but has never been completed by us.** Compile a
trial once, then match any number of patients against it:

    verdict compile NCT02509286
    verdict run     NCT02509286 P001 --patient-file note.json

`compile` runs four stages -- compile, normalize, slice, link -- writing to one
build tree (`$VERDICT_BUILD`, default `<repo>/build`). Run `--stages` to do
part of it. It needs an LLM endpoint and a
compiled trial program under `$VERDICT_BUILD`, and it reports clearly which
one is missing. What we have verified is that it installs, imports, resolves
its prompts, and refuses cleanly without credentials — **not** that it returns
a correct decision, because we had no endpoint or working solver on hand to
try it. Treat that path as untested and please report what you find.

Both commands need an LLM endpoint. What has been verified is that the chain
is wired to one build tree, that every stage imports and resolves its paths
from any directory, and that the failure modes report themselves -- not that a
compiled program or a decision is correct.

To reproduce the numbers in the paper, see
[docs/REPRODUCE_TABLES.md](docs/REPRODUCE_TABLES.md). For where the data comes
from and what may be redistributed, see [docs/DATA.md](docs/DATA.md).

## Which matcher reproduces the paper

`verdict headline` runs the published system — v6 mined atoms plus
silence-null, compiled patches, and the population gate. `verdict match` runs
a **reimplementation** that agrees with the paper on 88.7% of pairs and is for
inspection and comparison, not reproduction.

The distinction matters and is easy to get wrong: see
[docs/MATCHERS.md](docs/MATCHERS.md), which also records a known discrepancy —
the headline pipeline currently prints F1 0.873 where the paper reports 0.863.

## Known gaps

Documented so you do not have to rediscover them.

| | |
|---|---|
| **`verdict run` is untested** | The matcher ships here, but we have never completed a live match: no LLM endpoint and no working solver on this machine. Import, prompt resolution and the failure paths are tested; a correct decision is not. See [What this does not do yet](#what-this-does-not-do-yet). |
| **No TrialGPT baseline** | TrialGPT-derived code was removed; our artifacts gave 13.2% against the paper's 44.4% and could not be cited honestly. Run TrialGPT from its own repository. See [docs/MATCHERS.md](docs/MATCHERS.md). |
| **Claude Haiku, ZSPM and Xu rows of the TREC table** | Separate API runs whose outputs were not retained. GPT-5-mini and Qwen rows do reproduce exactly. |
| **Batch drivers under `matchers/systems/`** | Several depend on internal modules that were not released. They raise a clear error saying so. The supported entry points are the `verdict` command and the `verdict` package. |
| **Test coverage is uneven** | `smt_core.maxsmt` is at 92% and the APIs are covered; `trial_compiler`, `patient_compiler`, `db_indexer` and `sql_retrieval` have no unit tests yet. |
| **The evaluation corpora are not redistributed** | Licensing is unresolved for the SIGIR corpus. See [docs/DATA.md](docs/DATA.md). |
| **`experiments/`** | Research scratch — exploratory scripts and dead ends kept for the record. Not held to library standards. The scripts that produced published numbers are listed in [docs/REPRODUCE_TABLES.md](docs/REPRODUCE_TABLES.md). |

## License

**Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Free for any use, academic or commercial, including modification and
redistribution. Apache-2.0 also grants an explicit patent licence for the
method, and terminates that grant for anyone who brings a patent suit over
it — protection that MIT and BSD do not provide.

The code licence is separate from the data. The evaluation corpora are
third-party and carry their own terms; see [docs/DATA.md](docs/DATA.md).

## Architecture

One parser, two consumers.

```
                    trial criteria (text)     patient note (text)
                              |                       |
                    +---------v-----------------------v---------+
                    |            SEMANTIC PARSER                |
                    |  smt_core/          concepts, attributes, |
                    |                     units, inference      |
                    |  trial_compiler/    criteria -> constraints|
                    |  patient_compiler/  note -> coded facts   |
                    +---------+-----------------------+---------+
                              |                       |
                 constraints  |                       | coded facts
                              |                       |
              +---------------v-------+   +-----------v--------------+
              |        SatIR          |   |        VERDICT           |
              |  db_indexer/          |   |  matchers/               |
              |  sql_retrieval/       |   |  smt_matcher/            |
              |  matching_batch/      |   |  verbalizer/             |
              |                       |   |  smt_core/maxsmt.py      |
              |  which trials are     |   |  does THIS patient meet  |
              |  worth looking at?    |   |  THIS trial, and why?    |
              +-----------------------+   +--------------------------+
                        satir                       verdict
```

The parser turns text into structured constraints. SatIR indexes those
constraints and retrieves candidates at corpus scale; VERDICT decides a single
pair over them and shows its reasoning.

**The two never import each other** — verified, not aspirational. SatIR makes
42 imports from the parser, VERDICT 11, and the parser imports neither. The
`architecture:layering-holds` check fails the build on any cross-import, so you
can run retrieval without the matcher, or the matcher without a database.

## The four matchers

Each `matchers/systems/<name>/` is a self-contained bundle:

```
<system>/
├── prompts/         — *.prompt files (edit to tune behavior)
├── run.py           — generator (chart+criteria → verdicts.jsonl)
├── verdicts.jsonl   — per-pair eligibility predictions
├── rationales.jsonl — per-pair NL reasoning (used by judges)
└── README.md        — system-specific notes
```

Tune any system by editing its `prompts/<name>.prompt` and re-running `run.py`.

| System | Model | F1 (5-system gold) |
|---|---|---|
| **VERDICT** | gpt-4.1 atom mining + Z3 solve | 0.873 |
| **single_shot_llm** (LLM-only) | gpt-4.1 two-step | **0.904** |
| **trialgpt** | gpt-4.1 per-criterion | 0.804 |
| **shahlab** | gpt-4.1 Koopman prompt | 0.836 |

| Hybrid | F1 |
|---|---|
| **VERDICT+LLM-only+TG MAJ** | **0.906** |
| VERDICT+LLM-only OR | 0.896 (R=0.975) |
| VERDICT+LLM-only AND | 0.879 (P=0.943) |
| Meta-fusion balanced | 0.878 (P=0.958) |

## The two experiments

**[`experiments/accuracy/`](experiments/accuracy/REPRODUCE.md)** — accuracy + F1 + inspection
- 5-system gold (n=532), bootstrap CIs, Pareto controllability, sharpness ratings, fliprate
- `inspection/mbench/` — 539-pair drillable mbench with per-atom mining outcomes
- `inspection/v5_beats_aegis/` — 52 cases where LLM-only beats VERDICT on the gold
  (directory names keep the historical spelling; see [docs/MATCHERS.md](docs/MATCHERS.md))

**`experiments/typed_policy/`** — policy alignment
- 3-policy benchmark: VERDICT 100% compliance vs LLM 52-72%
- 4-axis stacked policies for fine-grained control
- 22 typed-atom dispatch (lab_chemistry, functional_score, etc.)

## Running

```bash
# Set up env
export OPENAI_ENDPOINT=...
export OPENAI_API_KEY=...
export OPENAI_ENDPOINT_GPT5=...   # for gold derivation only

# Re-mine + re-solve VERDICT (its code lives under the historical `aegis/` path)
python matchers/systems/aegis/run.py
python matchers/systems/aegis/run_verbalize.py

# Run other systems
python matchers/systems/trialgpt/run.py
python matchers/systems/single_shot_llm/run.py --variant V5_TWO_STEP
python matchers/systems/shahlab/run.py

# Build gold + accuracy F1
python experiments/accuracy/scripts/run_5judges_5sys.py
python experiments/accuracy/scripts/build_gold_from_5sys.py
python experiments/accuracy/scripts/bootstrap_ci.py

# Build inspection mbench
python experiments/accuracy/scripts/build_aegis_v5_mbench.py
```

## Backup

`backup/` contains all historical artifacts — earlier experiment runs, exploratory scripts, deprecated tools, and pre-fix versions of data files. Preserved for reproducibility but not part of the active codebase.
