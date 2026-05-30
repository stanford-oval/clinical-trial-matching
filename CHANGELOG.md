# Changelog

Notable changes. Dates are the release date.

## [0.2.0] — 2026-09-08

First public release. Two systems in one repository.

### Added
- **`satir`** — trial and patient compilation, clause-database indexing, and
  constraint-satisfaction retrieval, with a CLI and a Python API.
- **`verdict`** — eligibility matching with a per-criterion audit trail, six
  matcher variants, a CLI and a Python API.
- **`smt_core.maxsmt`** — the accountability artifacts (decision, trace,
  assumptions, pivotal conditions). Implements both published formulations:
  `MAXSMT` (updated paper, default) and `RESIDUAL` (submitted paper).
- **Matcher registry** — `verdict.register` adds a matcher without editing
  the package, so a new approach can be compared against the built-ins.
- **`examples/`** — three runnable scripts, exercised in CI.
- **`scripts/check_invariants.py`** — 28 checks covering the CLIs, both APIs,
  the reproduction numbers, packaging, and secret/path hygiene.
- Unit test suite (41 tests, no services required), CI on Python 3.12/3.13.
- Apache-2.0 licence, `NOTICE`, `CITATION.cff`, `CONTRIBUTING.md`.
- `docs/DATA.md` — data provenance and what may be redistributed.

### Notes for users
- `verdict.match` defaults to `strict=True`: a pair that cannot be loaded
  raises rather than returning `ineligible`. Pass `strict=False` for the
  behaviour the paper's numbers were produced under.
- Rationales report the requirement a trial imposes, not the value the solver
  assigned, since that value is arbitrary within the satisfying region.
- `verdict` decides pairs that stage-1 mining has already processed. Running
  end to end from a raw chart still requires the separate `cmsrc` codebase.
