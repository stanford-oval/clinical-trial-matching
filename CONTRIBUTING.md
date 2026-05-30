# Contributing

## Setup

```bash
pip install -e '.[dev]'          # + [llm] / [local] / [viz] as needed
cp .env.example .env             # credentials; never commit this
```

## Before you push

```bash
pytest                            # fast, hermetic unit suite (~0.3s)
python scripts/check_invariants.py
```

Both must pass. Between them they cover the CLIs, both public APIs, the
MaxSMT artifacts, the reproduction numbers, packaging, and secret/path
hygiene.

## Two things this repo is strict about

**Reproduction numbers are asserted, not smoke-tested.** `check_invariants.py`
compares against the values printed in the paper (e.g. TREC F1 0.828 / 0.738 /
0.663). If a change moves one of those, that is a finding, not a test to
update — say so in the PR.

**Published behaviour is versioned, never edited in place.** The matcher
returns `reasoning="no data"` for an unloadable pair because the paper's
numbers were produced that way; the safer behaviour is opt-in
(`strict=True`). Likewise the verbalizer prompt gained a `v13` file rather
than an edit to `v12`. If you need different semantics, add a version.

## Test layout

- `tests/unit/` — fast, no services, no API keys, no corpus. Run by default.
- `tests/*.py` — service-level scripts needing Elasticsearch, Snowstorm or a
  built `trial.db`. Run explicitly: `python tests/test_end_to_end.py`.
  They are excluded from the default `pytest` run on purpose.

## Style

Match the surrounding code. Public functions carry a docstring saying what
the caller gets, and anything mapping to the paper names its symbol and step
(see `smt_core/maxsmt.py::SYMBOLS`).
