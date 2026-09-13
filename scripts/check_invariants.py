#!/usr/bin/env python3
"""Invariant checks — run after EVERY change.

Asserts the things that must not break: the CLI works, the Python API works,
the reproduction scripts still emit the paper's numbers, and no secret or
machine-specific path has leaked back into shippable code.

    python scripts/check_invariants.py          # all
    python scripts/check_invariants.py -k table # only matching checks
    python scripts/check_invariants.py -v       # show output of failures

Exit code 0 = all green.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
CHECKS = []
VERBOSE = False


class Skip(Exception):
    """Raised when a check's local-only data is absent (fresh clone)."""


def check(name, needs=None):
    """needs: repo-relative path that must exist, else the check SKIPs."""
    def deco(fn):
        def wrapped():
            if needs and not (ROOT / needs).exists():
                raise Skip(needs)
            return fn()
        CHECKS.append((name, wrapped))
        return wrapped
    return deco


PAIR_DATA = 'experiments/53_v2_full/cmsrc_out'
#: evaluation corpus; absent from the tool-only build
GOLD = 'data/gold/gold_5sys_freeform_balanced.json'
HEADLINE_DATA = 'data/headline/v6'
VERBALIZER_PROMPT = 'verbalizer/prompts/_freeform_rationale_v13_maxsmt.prompt'


def sh(*args, timeout=600):
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def want(hay: str, *needles):
    missing = [n for n in needles if n not in hay]
    if missing:
        raise AssertionError(f'missing from output: {missing}')


# ---------------------------------------------------------------- CLI
@check('cli:systems')
def _():
    rc, out = sh('verdict_cli.py', 'systems')
    assert rc == 0, out
    want(out, 'smt_lm_evidence_arbiter', 'hybrid_strict')


@check('cli:list', needs=PAIR_DATA)
def _():
    rc, out = sh('verdict_cli.py', 'list', '--limit', '3')
    assert rc == 0, out
    want(out, 'sigir-', 'pair(s)')
    n = int(re.search(r'(\d+) pair\(s\)', out).group(1))
    assert n > 800, f'expected >800 pairs, got {n}'


@check('cli:match', needs=PAIR_DATA)
def _():
    rc, out = sh('verdict_cli.py', 'match', 'sigir-20141__NCT00337116')
    assert rc == 0, out
    want(out, 'decision: ELIGIBLE')


@check('cli:explain-audit-trail', needs=PAIR_DATA)
def _():
    rc, out = sh('verdict_cli.py', 'explain', 'sigir-20141__NCT00337116')
    assert rc == 0, out
    want(out, 'audit trail', 'atom_mining', 'smt_solve')


@check('cli:bad-pair-errors-cleanly', needs=PAIR_DATA)
def _():
    rc, out = sh('verdict_cli.py', 'match', 'nope__NCT0')
    assert rc != 0, 'expected nonzero exit for unknown pair'
    want(out, 'unknown pair')


@check('cli:ineligible-path', needs=PAIR_DATA)
def _():
    rc, out = sh('verdict_cli.py', 'match', 'sigir-20141__NCT00990262')
    assert rc == 0, out
    want(out, 'decision: INELIGIBLE')


# ---------------------------------------------------------------- API
@check('api:python-import', needs=PAIR_DATA)
def _():
    rc, out = sh('-c', 'from matchers import variants;'
                       'd=variants.smt_lm_evidence_arbiter("sigir-20141__NCT00337116");'
                       'print(d.decision, len(d.audit_trail), d.reasoning)')
    assert rc == 0, out
    decision = out.split()[0]
    assert decision == 'eligible', f'expected exactly "eligible", got {decision!r}'
    assert 'no data' not in out, ('API silently returned a verdict with no pair '
                                  'data loaded -- see api:no-data-is-not-a-verdict')


@check('api:no-data-is-not-a-verdict')
def _():
    """A missing pair must be distinguishable from a real INELIGIBLE.

    matchers.variants returns decision='ineligible', reasoning='no data' when
    the pair cannot be loaded. That is the dangerous error direction for a
    trial matcher, so callers MUST check reasoning. verdict_cli guards this by
    validating the pair id up front; this check pins both halves.
    """
    rc, out = sh('-c', 'from matchers import variants;'
                       'd=variants.smt_lm_evidence_arbiter("TOTALLY__FAKE999");'
                       'print(repr(d.decision), repr(d.reasoning))')
    assert rc == 0, out
    assert 'no data' in out, ('sentinel changed: a missing pair no longer '
                              'reports "no data" -- update verdict_cli and docs/DATA.md')
    rc, out = sh('verdict_cli.py', 'match', 'TOTALLY__FAKE999')
    assert rc != 0, 'CLI must refuse an unknown pair rather than emit a verdict'
    assert ('unknown pair' in out) or ('no pair data' in out), \
        f'CLI refused but with an unhelpful message: {out[:200]}'
    assert 'decision' not in out.lower(), \
        'CLI emitted a verdict for a nonexistent pair'


@check('api:strict-mode-raises', needs=PAIR_DATA)
def _():
    """Strict mode must raise on a missing pair, and not change real verdicts."""
    rc, out = sh('-c',
        'from matchers import variants;'
        'from matchers.schema import MissingPairData;'
        'variants.strict(True);'
        'ok=0\n'
        'try:\n'
        '    variants.smt_lm_evidence_arbiter("FAKE__NCT0")\n'
        'except MissingPairData: ok=1\n'
        'print("raised", ok);'
        'print("real", variants.smt_lm_evidence_arbiter("sigir-20141__NCT00337116").decision)')
    assert rc == 0, out
    want(out, 'raised 1', 'real eligible')


@check('api:default-mode-preserved', needs=PAIR_DATA)
def _():
    """Default (non-strict) behaviour must stay exactly as the paper ran it."""
    rc, out = sh('-c',
        'from matchers import variants;'
        'd=variants.smt_lm_evidence_arbiter("FAKE__NCT0");'
        'print(d.decision, "|", d.reasoning, "|", d.is_missing_data)')
    assert rc == 0, out
    want(out, 'ineligible | no data | True')


@check('api:public-packages-import-cleanly')
def _():
    """`import satir` / `import verdict` must work with no optional deps."""
    rc, out = sh('-c', 'import satir, verdict;'
                       'print(len(satir.__all__), len(verdict.__all__));'
                       'print(sorted(verdict.systems())[:2]);'
                       'print(type(satir.config()).__name__)')
    assert rc == 0, out
    want(out, 'SatIRConfig')


@check('api:verdict-api-matches-cli', needs=PAIR_DATA)
def _():
    """The verdict API and CLI must agree, and the API must default to strict."""
    rc, out = sh('-c',
        'import verdict;'
        'd=verdict.match("sigir-20141__NCT00337116");'
        'print("decision", d.decision);'
        'print("npairs", len(verdict.pairs()));'
        'ok=0\n'
        'try:\n'
        '    verdict.match("FAKE__NCT0")\n'
        'except verdict.MissingPairData: ok=1\n'
        'print("strict", ok);'
        'print("lenient", verdict.match("FAKE__NCT0", strict=False).reasoning)')
    assert rc == 0, out
    want(out, 'decision eligible', 'strict 1', 'lenient no data')
    rc2, cli = sh('verdict_cli.py', 'match', 'sigir-20141__NCT00337116')
    assert rc2 == 0 and 'ELIGIBLE' in cli, 'CLI and API disagree'


@check('api:satir-lazy-imports')
def _():
    """satir must not import heavy backends at module import time."""
    rc, out = sh('-c', 'import sys, satir;'
                       'heavy=[m for m in ("torch","elasticsearch","matplotlib",'
                       '"dspy","azure") if m in sys.modules];'
                       'print("eager:", heavy)')
    assert rc == 0, out
    want(out, 'eager: []')


@check('maxsmt:paper-invariants-hold')
def _():
    """Both published formulations, their invariants, and the paper export.

        delta_E = {} iff d = ELIGIBLE
        delta_I = {} iff d = INELIGIBLE
        delta  != {}

    and rho must hold the solver's WITNESS while the verbalizer view reports
    the REQUIREMENT (a witness is arbitrary within the satisfying region).

    Runs in a subprocess with cwd=ROOT so `smt_core` is importable.
    """
    rc, out = sh('-c', """
import sys
try:
    import z3
    assert all(hasattr(z3, a) for a in ('parse_smt2_string', 'Optimize'))
except Exception:
    print('NO_Z3'); sys.exit(0)
from smt_core.maxsmt import (Condition, solve, OBSERVED, IMPUTED,
                             UNRESOLVED, ELIGIBLE, INELIGIBLE)
phi = ['(declare-const |egfr| Bool)', '(declare-const |crcl| Real)',
       '(assert |egfr|)', '(assert (>= |crcl| 60))']
cases = [
    [Condition('egfr', False, OBSERVED), Condition('crcl', None, UNRESOLVED)],
    [Condition('egfr', True, OBSERVED),  Condition('crcl', None, UNRESOLVED)],
    [Condition('egfr', True, OBSERVED),  Condition('crcl', 80, OBSERVED)],
    [Condition('egfr', True, OBSERVED),  Condition('crcl', 40, OBSERVED)],
    [Condition('egfr', True, IMPUTED, 'assume-normal'),
     Condition('crcl', None, UNRESOLVED)],
]
for conds in cases:
    a = solve(phi, conds)
    assert (a.delta_e == []) == (a.decision == ELIGIBLE), ('dE', a)
    assert (a.delta_i == []) == (a.decision == INELIGIBLE), ('dI', a)
    assert a.pivotal != [], ('delta empty', a)
a = solve(phi, cases[0])
assert a.decision == INELIGIBLE and a.pivotal == ['egfr'], a
assert 'crcl' in a.assumptions
v = a.for_verbalizer(phi)
assert v['assumptions']['crcl']['requirement'] == '>= 60', v
assert 'witness' in v['assumptions']['crcl'], v

# --- both published versions must be runnable and must differ ----------
from smt_core.maxsmt import RESIDUAL, MAXSMT, PAPER_OF, SYMBOLS
r = solve(phi, cases[0], version=RESIDUAL)
u = solve(phi, cases[0], version=MAXSMT)
assert r.version == RESIDUAL and u.version == MAXSMT
assert r.decision == u.decision == INELIGIBLE
# rho differs in KIND: submitted holds a requirement, update holds a witness
assert r.assumptions['crcl'] == '>= 60', r.assumptions
assert isinstance(u.assumptions['crcl'], float), u.assumptions
# delta_E / delta_I exist only in the updated formulation
assert r.delta_e == [] and r.delta_i == []
assert u.delta_e == ['egfr']

# --- paper export ------------------------------------------------------
er, eu = r.to_paper(), u.to_paper()
assert er['paper'] == 'submitted' and eu['paper'] == 'update'
for k in ('d', 'gamma', 'rho', 'delta'):
    assert k in er and k in eu, k
# omitted, not emitted empty, under the submitted formulation
assert 'delta_E' not in er and 'delta_I' not in er, er.keys()
assert 'delta_E' in eu and 'delta_I' in eu, eu.keys()
assert eu['rho']['step'] == 'Step 4' and er['rho']['step'] == 'Step 3'
assert 'witness' in eu['rho']['meaning']
assert 'residual constraints' in er['rho']['meaning']
assert set(SYMBOLS) >= {'d', 'gamma', 'rho', 'delta', 'delta_E', 'delta_I'}
assert r.describe() and u.describe()
print('MAXSMT_OK')
""")
    assert rc == 0, out
    if 'NO_Z3' in out:
        raise Skip('z3-solver (pip install "z3-solver>=4.12")')
    want(out, 'MAXSMT_OK')


@check('maxsmt:verbalizer-prompt-forbids-witnesses', needs=VERBALIZER_PROMPT)
def _():
    """The v13 prompt must carry the requirement-not-witness rule."""
    p = ROOT / 'verbalizer/prompts/_freeform_rationale_v13_maxsmt.prompt'
    assert p.exists(), 'v13 MaxSMT verbalizer prompt missing'
    txt = p.read_text()
    for needle in ('assumptions', 'witness', 'requirement the trial imposes',
                   'pivotal'):
        assert needle.lower() in txt.lower(), f'prompt missing rule: {needle}'


# ------------------------------------------------- reproduction numbers
@check('table2:trec-f1-matches-paper')
def _():
    svpo = pathlib.Path(os.environ.get('SVPO_RL', ROOT.parent / 'svpo-rl'))
    if not (svpo / 'data/test_clean_tagged.jsonl').exists():
        raise Skip('svpo-rl checkout (set $SVPO_RL)')
    rc, out = sh('scripts/tables/table2_trec2021_f1.py')
    assert rc == 0, out
    want(out, '363 pairs', '190 eligible / 173 ineligible')
    for label, f1 in (('GPT-5-mini  VERDICT', '0.828'),
                      ('Qwen2.5-7B  VERDICT', '0.738'),
                      ('Qwen2.5-7B  CoT', '0.663')):
        row = next((l for l in out.splitlines() if l.startswith(label)), None)
        assert row, f'row missing: {label}'
        assert f1 in row, f'{label}: expected F1 {f1}, got: {row.strip()}'


@check('table9:sigir-552-pairs', needs=GOLD)
def _():
    rc, out = sh('scripts/tables/table9_dataset_stats.py')
    assert rc == 0, out
    want(out, '552')


@check('table10:verdict-balance-matches-paper', needs=GOLD)
def _():
    rc, out = sh('scripts/tables/table10_verdict_balance.py')
    assert rc == 0, out
    want(out, '552-pair comparison set')
    # the three systems whose canonical source is confirmed exact
    for label, elig, med in (('VERDICT', '55.4', '1021'),
                             ('ourLLM',  '35.0', '1195'),
                             ('CoT LLM', '36.1', '592')):
        row = next((l for l in out.splitlines() if l.startswith(label)), None)
        assert row, f'row missing: {label}'
        assert row.count(elig) >= 2, f'{label}: elig% != paper: {row.strip()}'
        assert row.count(med) >= 2, f'{label}: median != paper: {row.strip()}'
        assert row.rstrip().endswith('ok'), f'{label} no longer exact: {row.strip()}'
    # The TrialGPT row was removed with the rest of the TrialGPT-derived
    # material (docs/MATCHERS.md). Assert it stays gone, so it cannot creep
    # back in without the licensing and reproducibility questions being
    # answered again.
    assert 'TrialGPT' not in out, 'TrialGPT row is back in table 10'


@check('zspm:92pct-silence',
       needs='experiments/counterfactual/05_self_faithfulness/mbench_3cell')
def _():
    rc, out = sh('scripts/tables/zspm_silence_breakdown.py')
    assert rc == 0, out
    want(out, '92% of grounds are chart-silence')


@check('gold:278-eligible-274-ineligible', needs=GOLD)
def _():
    fp = ROOT / 'data/gold/gold_5sys_freeform_balanced.json'
    g = json.loads(fp.read_text())['gold']
    e = sum(1 for v in g.values() if v is True)
    assert (len(g), e, len(g) - e) == (552, 278, 274), (len(g), e)


@check('engine:vendored-data-is-tracked')
def _():
    """Every file the vendored matcher loads must be committed.

    .gitignore carries a broad `*_out/` rule for run artifacts, which also
    matches the matcher's prompt_out/ INPUT directory. That silently produced
    a clone whose matcher had no prompts -- green locally, broken in CI.
    """
    import subprocess as _sp
    # Everything the shipped tool READS at runtime, not just the prompts.
    # .gitignore carries a broad `*_out/` rule for run artifacts, which has now
    # twice matched input directories -- prompts/prompt_out/ and the demo's
    # cmsrc_out/ -- producing a clone whose tool had no data. Green locally,
    # broken for everyone else.
    watched = []
    pkg = ROOT / 'verdict' / 'engine'
    watched += [f for f in pkg.rglob('*.prompt')]
    demo = ROOT / 'data' / 'demo'
    if demo.exists():
        watched += [f for f in demo.rglob('*') if f.is_file()]
    assert watched, 'no runtime data on disk to check'
    on_disk = {str(f.relative_to(ROOT)) for f in watched}
    out = _sp.run(['git', 'ls-files', 'verdict/engine', 'data/demo'], cwd=ROOT,
                  capture_output=True, text=True)
    tracked = set(out.stdout.split())
    missing = sorted(on_disk - tracked)
    assert not missing, 'runtime data not tracked by git: ' + ', '.join(missing[:6])


@check('artifacts:no-raw-witness-surfaced')
def _():
    """No shipped code may hand a caller rho's raw solver witnesses.

    Under MAXSMT a witness is an arbitrary point in the satisfying region.
    Surfacing `crcl = 60.0` as if it were a finding would invent a lab
    result, so consumers must go through assumptions_report() /
    for_verbalizer(), which report the requirement instead.
    """
    import ast as _ast
    SHIPPED = ['verdict', 'satir', 'pipeline.py', 'verbalizer',
               'verdict_cli.py', 'examples']
    OK = {'smt_core/maxsmt.py'}
    bad = []
    for rel in SHIPPED:
        base = ROOT / rel
        if not base.exists():
            continue
        files = [base] if base.is_file() else list(base.rglob('*.py'))
        for f in files:
            r = str(f.relative_to(ROOT))
            if r in OK:
                continue
            try:
                tree = _ast.parse(f.read_text(errors='ignore'))
            except SyntaxError:
                continue
            for n in _ast.walk(tree):
                # `<something>.assumptions` read straight off an Artifacts
                if (isinstance(n, _ast.Attribute) and n.attr == 'assumptions'
                        and isinstance(n.value, _ast.Name)
                        and n.value.id in ('a', 'art', 'artifacts')):
                    bad.append(f'{r}:{n.lineno}')
    assert not bad, ('raw rho surfaced without the requirement view: '
                     + ', '.join(bad[:6]))


@check('gold:no-stale-paths')
def _():
    """No file may still point at the gold set's pre-move location.

    The gold set moved to data/gold/ so the tool no longer reaches into
    experiments/. A leftover reference is a silently broken script.
    """
    stale = []
    for f in ROOT.rglob('*'):
        if not f.is_file() or '.git' in f.parts: continue
        if f.resolve() == pathlib.Path(__file__).resolve(): continue  # this check names the path
        if f.suffix not in ('.py', '.md', '.sh', '.yaml', '.yml'): continue
        try: body = f.read_text()
        except (UnicodeDecodeError, OSError): continue
        if 'accuracy/data/gold_5sys_freeform_balanced' in body:
            stale.append(str(f.relative_to(ROOT)))
    assert not stale, 'stale gold path in: ' + ', '.join(stale)


@check('deps:declared-match-imports')
def _():
    """Every third-party import in shipped code must be a declared dependency.

    Guards against the usual open-source failure: code grows an import, the
    install spec never learns about it, and a fresh clone dies on ImportError.
    """
    import ast as _ast
    try:
        import tomllib
    except ModuleNotFoundError:                       # pragma: no cover
        raise Skip('tomllib (py<3.11)')

    meta = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    declared = set(meta.get('dependencies', []))
    for extra in meta.get('optional-dependencies', {}).values():
        declared |= set(extra)
    # "z3-solver>=4.12" -> "z3-solver"
    dist = {re.split(r'[<>=!\[]', d)[0].strip().lower() for d in declared}

    # import name -> distribution name, where they differ
    ALIAS = {'z3': 'z3-solver', 'dspy': 'dspy-ai', 'sentence_transformers':
             'sentence-transformers', 'azure': 'azure-identity',
             'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}

    stdlib = set(sys.stdlib_module_names)
    local = {'matchers', 'smt_core', 'verbalizer', 'rationale_generators',
             'counterfactual_modifier', 'scripts', 'verdict_cli',
             # SatIR packages
             'trial_compiler', 'patient_compiler', 'db_indexer',
             'sql_retrieval', 'smt_matcher', 'matching_batch', 'evaluation',
             'audit', 'satir_cli'}
    undeclared = {}
    for d in ('matchers', 'smt_core', 'verbalizer', 'counterfactual_modifier',
              'trial_compiler', 'patient_compiler', 'db_indexer',
              'sql_retrieval', 'smt_matcher', 'evaluation'):
        base = ROOT / d
        if not base.exists():
            continue
        for f in base.rglob('*.py'):
            if '.bak' in f.name:
                continue
            src = f.read_text(errors='ignore')
            try:
                tree = _ast.parse(src)
            except SyntaxError:
                continue
            # imports inside try/except ImportError are optional by construction
            guarded = set()
            for n in _ast.walk(tree):
                if isinstance(n, _ast.Try) and any(
                        isinstance(h.type, _ast.Name) and
                        h.type.id in ('ImportError', 'ModuleNotFoundError')
                        for h in n.handlers):
                    for sub in _ast.walk(n):
                        if isinstance(sub, _ast.Import):
                            guarded |= {a.name.split('.')[0] for a in sub.names}
                        elif isinstance(sub, _ast.ImportFrom) and sub.module:
                            guarded.add(sub.module.split('.')[0])
            # modules made importable by an explicit sys.path.insert in this file
            path_local = 'sys.path.insert' in src or 'sys.path.append' in src
            for n in _ast.walk(tree):
                mods = []
                if isinstance(n, _ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, _ast.ImportFrom) and n.level == 0 and n.module:
                    mods = [n.module]
                for m in mods:
                    top = m.split('.')[0]
                    if not top or top in stdlib or top in local:
                        continue
                    if top in guarded or path_local:
                        continue   # optional, or resolved via sys.path (see
                                   # imports:shipped-modules-are-importable)
                    # sibling modules imported by filename, not packages
                    if (f.parent / f'{top}.py').exists():
                        continue
                    if ALIAS.get(top, top).lower() not in dist:
                        undeclared.setdefault(top, str(f.relative_to(ROOT)))
    assert not undeclared, ('imported but not declared in pyproject: '
                            + ', '.join(f'{k} ({v})' for k, v in
                                        sorted(undeclared.items())[:8]))


@check('imports:shipped-modules-are-importable')
def _():
    """Every module in the release payload must import without ImportError.

    Catches sys.path hacks reaching into directories excluded from the release,
    and imports of modules that no longer exist anywhere.
    """
    import ast as _ast
    SHIP = ('matchers', 'verbalizer', 'counterfactual_modifier', 'smt_core',
            'trial_compiler', 'patient_compiler', 'db_indexer', 'sql_retrieval',
            'smt_matcher', 'matching_batch', 'evaluation', 'audit')
    # Module names that are known not to resolve, with the reason. These are
    # tracked debt, not silent ignores -- shrink this set, never grow it.
    # Verified 2026-09-03 by a full scan; file lists are exact.
    KNOWN_BROKEN = {
        'trialgpt_judge':
            'TrialGPT-derived source was removed from the repository (see '
            'docs/MATCHERS.md) -- breaks the five evaluation/explainability '
            'scripts that imported it. Run TrialGPT from its own repo.',
        'overnight':
            'no such module anywhere -- breaks aegis/apply_soft_aux_solve.py, '
            'aegis/apply_strict_inc_to_4o.py',
        'run_better_nl_full':
            'no such module anywhere -- breaks aegis/run_verbalize.py, '
            'single_shot_llm/run.py, '
            'verbalizer/run_aegis_verbalize.py',
        'repro_headline':
            'lives in experiments/accuracy/scripts (excluded from the release) '
            '-- breaks aegis/aegis_arbiter.py, aegis/aegis_arbiter_eval.py, '
            'verbalizer/run_aegis_v9_arbiter_verbalize.py',
        'cf_maxsat':          'lives in experiments/counterfactual/utils, excluded',
        'cf_dataset':         'lives in experiments/, excluded from the release',
        'cf_blockers':        'lives in experiments/, excluded from the release',
        'build_judge_input':  'sibling script, not importable as a module',
    }
    # 9 shipped files total. The `verdict` CLI and matchers.variants do NOT
    # depend on any of these, which is why the CLI works; the affected files
    # are batch entry points. Shrink this set, never grow it.
    # A declared dependency that merely is not installed here is NOT breakage.
    try:
        import tomllib
        meta = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
        declared = set(meta.get('dependencies', []))
        for extra in meta.get('optional-dependencies', {}).values():
            declared |= set(extra)
        dist = {re.split(r'[<>=!\[]', d)[0].strip().lower() for d in declared}
    except Exception:
        dist = set()
    ALIAS = {'z3': 'z3-solver', 'dspy': 'dspy-ai',
             'sentence_transformers': 'sentence-transformers',
             'azure': 'azure-identity', 'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}

    # Precompute local module names ONCE (scanning per-import walks the whole
    # tree and takes minutes).
    localmods = set(SHIP)
    for d in SHIP:
        base = ROOT / d
        if not base.exists():
            continue
        for f in base.rglob('*.py'):
            if '__pycache__' in str(f):
                continue
            localmods.add(f.stem)
            localmods.add(f.parent.name)

    resolvable, unresolvable = set(), {}
    for d in SHIP:
        base = ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob('*.py')):
            if '.bak' in f.name or '__pycache__' in str(f):
                continue
            rel = str(f.relative_to(ROOT))
            try:
                tree = _ast.parse(f.read_text(errors='ignore'))
            except SyntaxError:
                continue
            for n in tree.body:          # module level only; nested = lazy
                mods = []
                if isinstance(n, _ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, _ast.ImportFrom) and n.level == 0 and n.module:
                    mods = [n.module]
                for m in mods:
                    top = m.split('.')[0]
                    if (top in sys.stdlib_module_names or top in localmods
                            or top in resolvable or top in KNOWN_BROKEN
                            or ALIAS.get(top, top).lower() in dist):
                        continue
                    if top in unresolvable.values():
                        unresolvable.setdefault(rel, top)
                        continue
                    try:
                        __import__(top)
                        resolvable.add(top)
                    except Exception:
                        unresolvable.setdefault(rel, top)

    assert not unresolvable, (
        'modules that will not import (not declared, not local, not known debt):'
        '\n  ' + '\n  '.join(f'{k}: {v}' for k, v in unresolvable.items()))


@check('packaging:declared-targets-exist')
def _():
    """Every package pattern and console script in pyproject must resolve.

    Before the SatIR merge this repo declared five packages that did not exist
    and two console scripts (satir, compile-trial) that installed as commands
    which crashed on invocation -- pip does not verify either.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        raise Skip('tomllib')
    cfg = tomllib.loads((ROOT / 'pyproject.toml').read_text())

    missing_pkg = []
    for pat in (cfg.get('tool', {}).get('setuptools', {})
                   .get('packages', {}).get('find', {}).get('include', [])):
        name = pat.rstrip('*')
        if not name:
            continue
        if not (ROOT / name).is_dir():
            missing_pkg.append(pat)
    assert not missing_pkg, f'declared packages that do not exist: {missing_pkg}'

    for mod in cfg.get('tool', {}).get('setuptools', {}).get('py-modules', []):
        assert (ROOT / f'{mod}.py').exists(), f'py-module missing: {mod}.py'

    bad_scripts = []
    for script, target in cfg.get('project', {}).get('scripts', {}).items():
        mod = target.split(':')[0]
        path = ROOT / (mod.replace('.', '/') + '.py')
        pkg_init = ROOT / mod.replace('.', '/') / '__init__.py'
        if not path.exists() and not pkg_init.exists():
            bad_scripts.append(f'{script} -> {target}')
    assert not bad_scripts, f'console scripts pointing at nothing: {bad_scripts}'


@check('licensing:license-present-and-declared')
def _():
    """LICENSE must exist and pyproject must name the same licence.

    Without a licence file the default is all-rights-reserved, so nobody may
    use the code even from a public repo.
    """
    lic = ROOT / 'LICENSE'
    assert lic.exists(), 'LICENSE missing -- default is all rights reserved'
    txt = lic.read_text()
    for needle in ('Apache License', 'Version 2.0, January 2004',
                   'Grant of Patent License', 'Redistribution'):
        assert needle in txt, f'LICENSE missing section: {needle}'
    assert (ROOT / 'NOTICE').exists(), 'Apache-2.0 ships a NOTICE file'
    try:
        import tomllib
    except ModuleNotFoundError:
        raise Skip('tomllib')
    cfg = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    declared = str(cfg['project'].get('license', ''))
    assert 'Apache-2.0' in declared, \
        f'pyproject licence does not match LICENSE: {declared!r}'


@check('examples:run-without-data')
def _():
    """The examples that need no corpus must actually run.

    Documentation rots silently; an executed example cannot.
    """
    import subprocess
    ex = ROOT / 'examples'
    assert ex.is_dir(), 'examples/ missing'
    rc, out = sh(str(ex / '03_add_your_own_matcher.py'))
    assert rc == 0, out
    want(out, 'always-refer', 'ELIGIBLE')

    # 02 needs a working solver; skip rather than fail without one
    probe, _o = sh('-c', "import z3; print(hasattr(z3,'Optimize'))")
    if 'True' in _o:
        rc, out = sh(str(ex / '02_read_the_artifacts.py'))
        assert rc == 0, out
        want(out, 'Decision: INELIGIBLE', 'What it had to assume', '>= 60')


@check('imports:no-work-at-import-time')
def _():
    """Importing a module must not run analyses, print, or touch the disk.

    matchers.variants used to read four jsonl caches at import, and the example
    scripts ran a full sweep. A library that works when merely imported cannot
    be embedded, tested, or introspected safely.

    Pure config (env-var branching) and sys.path setup are allowed.
    """
    import ast as _ast
    ALLOWED_PREFIX = ("sys.path.", "_sys.path.", "warnings.", "logging.",
                      "csv.field_size_limit", "session.mount")
    SAFE = (_ast.Import, _ast.ImportFrom, _ast.FunctionDef, _ast.AsyncFunctionDef,
            _ast.ClassDef, _ast.Assign, _ast.AnnAssign, _ast.AugAssign,
            _ast.Try, _ast.Delete, _ast.Pass)

    def is_main_guard(n):
        return (isinstance(n, _ast.If) and isinstance(n.test, _ast.Compare)
                and isinstance(n.test.left, _ast.Name)
                and n.test.left.id == "__name__")

    # env-var config branching: an If whose body only assigns
    def is_config_if(n):
        return (isinstance(n, _ast.If)
                and all(isinstance(b, (_ast.Assign, _ast.AnnAssign, _ast.Pass))
                        for b in list(n.body) + list(n.orelse)))

    # `if TYPE_CHECKING:` never executes at runtime, so an import inside it
    # does no work at import time -- it is the standard way to keep a heavy
    # optional dependency out of the module-load path.
    def is_type_checking_if(n):
        return (isinstance(n, _ast.If) and isinstance(n.test, _ast.Name)
                and n.test.id == "TYPE_CHECKING"
                and all(isinstance(b, (_ast.Import, _ast.ImportFrom, _ast.Pass))
                        for b in list(n.body) + list(n.orelse)))

    offenders = []
    for d in ("matchers", "verbalizer", "smt_core", "counterfactual_modifier",
              "rationale_generators", "satir", "verdict"):
        base = ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            if ".bak" in f.name or "__pycache__" in str(f):
                continue
            try:
                tree = _ast.parse(f.read_text(errors="ignore"))
            except SyntaxError:
                continue
            for n in tree.body:
                if (isinstance(n, SAFE) or is_main_guard(n) or is_config_if(n)
                        or is_type_checking_if(n)):
                    continue
                if isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Constant):
                    continue                      # docstring
                src = _ast.unparse(n).split("\n")[0]
                if src.startswith(ALLOWED_PREFIX):
                    continue
                offenders.append(f"{f.relative_to(ROOT)}:{n.lineno} {src[:60]}")
    assert not offenders, ("modules doing work at import:\n  "
                           + "\n  ".join(offenders[:10]))


@check('syntax:every-shipped-python-parses')
def _():
    """Every tracked .py must parse on the declared Python version.

    sql_retrieval/ops/eval_pr_rec_at_k.py had a stray character that made it
    unparseable from the day it was written; nothing imported it, so nothing
    noticed. requires-python is >=3.12, so files using 3.12+ syntax are fine --
    this runs on the interpreter executing the checks.
    """
    import ast as _ast
    import subprocess
    if sys.version_info < (3, 12):
        raise Skip('needs the declared floor, python>=3.12')
    files = subprocess.run(['git', 'ls-files', '*.py'], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    broken = []
    for rel in files:
        f = ROOT / rel
        if not f.exists():
            continue
        try:
            _ast.parse(f.read_text(errors='ignore'))
        except SyntaxError as e:
            broken.append(f'{rel}:{e.lineno} {e.msg}')
    assert not broken, ('files that do not parse:\n  ' + '\n  '.join(broken[:10]))


@check('hygiene:no-personal-paths-or-internal-hosts')
def _():
    """No home directory, cluster path, or internal endpoint may ship.

    Wider than the ship-path check: this covers EVERY tracked text file, not
    just the library directories, because docs, logs and experiment scripts
    leak identity and infrastructure just as effectively.
    """
    import subprocess
    pats = [r'/Users/[A-Za-z]', r'/home/[a-z]', r'/nlp/scr/[a-z]',
            r'/juice[0-9]+/', r'/sailhome/', r'scdt\.stanford',
            r'[a-z0-9-]+\.openai\.azure\.com']
    ALLOW = {'scripts/check_invariants.py'}          # this file states the patterns
    out = subprocess.run(['git', 'grep', '-lIE', '|'.join(pats)],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    hits = [f for f in out if f not in ALLOW]
    # the placeholder in .env.example is intentional
    hits = [f for f in hits if f != '.env.example']
    assert not hits, ('personal paths or internal hosts in: ' + ', '.join(hits[:8]))


#: The layering. The semantic parser turns text into structured constraints;
#: SatIR and VERDICT are two independent consumers of that output.
LAYERS = {
    'parser':  ['smt_core', 'trial_compiler', 'patient_compiler'],
    'satir':   ['db_indexer', 'sql_retrieval', 'matching_batch'],
    'verdict': ['matchers', 'smt_matcher', 'verbalizer',
                'counterfactual_modifier', 'rationale_generators', 'verdict'],
    'app':     ['pipeline'],
}
#: who may import whom. `pipeline` is the app layer: it is the ONLY place
#: allowed to touch both systems, which is what keeps them independent.
ALLOWED = {('satir', 'parser'), ('verdict', 'parser'),
           ('app', 'parser'), ('app', 'satir'), ('app', 'verdict')}


@check('architecture:layering-holds')
def _():
    """parser <- {satir, verdict}, and satir must not know about verdict.

    The parser is the base layer: it may not import its consumers. SatIR and
    VERDICT are siblings -- coupling them would mean you could not run
    retrieval without the matcher, or the matcher without a database.
    """
    import ast as _ast
    owner = {pkg: layer for layer, pkgs in LAYERS.items() for pkg in pkgs}
    violations = []
    for layer, pkgs in LAYERS.items():
        for pkg in pkgs:
            base = ROOT / pkg
            if not base.exists():
                continue
            for f in base.rglob('*.py'):
                if '.bak' in f.name or '__pycache__' in str(f):
                    continue
                try:
                    tree = _ast.parse(f.read_text(errors='ignore'))
                except SyntaxError:
                    continue
                for n in _ast.walk(tree):
                    mods = ([a.name for a in n.names] if isinstance(n, _ast.Import)
                            else [n.module] if isinstance(n, _ast.ImportFrom)
                            and n.module else [])
                    for m in mods:
                        tgt = owner.get(m.split('.')[0])
                        if tgt and tgt != layer and (layer, tgt) not in ALLOWED:
                            violations.append(
                                f'{f.relative_to(ROOT)}: {layer} -> {tgt} ({m})')
    assert not violations, ('layering violated:\n  ' + '\n  '.join(violations[:10]))


@check('headline:paper-pipeline-reproduces')
def _():
    """The vendored paper pipeline must still produce its table.

    verdict/headline.py is the actual system behind the paper: v6 mined atoms
    + silence-null on both sides + 58 compiled patches + the population gate.
    matchers/variants.py is a reimplementation and does NOT reproduce it.
    """
    if not (ROOT / 'data/headline/v6').is_dir():
        raise Skip('data/headline/v6')
    probe, out = sh('-c', "import z3; print(hasattr(z3,'Optimize'))")
    if 'True' not in out:
        raise Skip('z3-solver')
    rc, out = sh('verdict/headline.py')
    assert rc == 0, out[-800:]
    want(out, 'VERDICT default (compiled)', 'VERDICT opt-in', 'LLM-only')
    # counts recorded in the paper's own HEADLINE_VERIFIED.json
    want(out, '856', '553', '539')
    assert 'not written' in out, 'headline.py must not write unless asked'


@check('naming:no-legacy-names-user-facing')
def _():
    """AEGIS and V5 are pre-publication names; the paper says VERDICT and
    LLM-only. They must not appear in anything a reader sees.

    On-disk paths (matchers/systems/aegis/) and the contents of data artifacts
    keep their historical names on purpose -- renaming those would break
    provenance and every stored reference. See docs/MATCHERS.md.
    """
    import re as _re
    import subprocess
    surfaces = ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'pipeline.py',
                'verdict_cli.py', 'satir_cli.py']
    surfaces += [f'docs/{p.name}' for p in (ROOT / 'docs').glob('*.md')]
    surfaces += [str(p.relative_to(ROOT)) for p in (ROOT / 'verdict').glob('*.py')]
    surfaces += [str(p.relative_to(ROOT)) for p in (ROOT / 'satir').glob('*.py')]
    surfaces += [str(p.relative_to(ROOT)) for p in (ROOT / 'examples').glob('*')]
    # docs/MATCHERS.md is the mapping document: it must name both the paper
    # name and the legacy one, side by side. It is the single exemption.
    EXEMPT = {'docs/MATCHERS.md'}
    bad = []
    for rel in surfaces:
        if rel in EXEMPT:
            continue
        f = ROOT / rel
        if not f.is_file():
            continue
        for i, line in enumerate(f.read_text(errors='ignore').splitlines(), 1):
            # the mapping note in MATCHERS.md is allowed to name them
            if 'historical name' in line or 'legacy name' in line:
                continue
            # display names only. Paths and filenames keep the historical
            # spelling on purpose (matchers/systems/aegis/, v5_beats_aegis/),
            # so ignore any occurrence adjacent to a path or identifier char.
            if _re.search(r'(?<![\w/])(AEGIS|V5)(?![\w/])', line):
                bad.append(f'{rel}:{i}')
    assert not bad, ('legacy names in user-facing text: ' + ', '.join(bad[:8]))


@check('selfcontained:tool-does-not-need-experiments')
def _():
    """The installable tool must not read anything under experiments/.

    experiments/ is research code kept for the record; the tool has to install
    and run without it. Only two documented fallbacks may name it, so an old
    checkout keeps working, and both prefer data/ when it exists.

    SCOPE: this is a STATIC check over source text. It proves no tool file
    names experiments/; it does NOT prove the tool functions without that
    tree. The pair-data fallback legitimately resolves into experiments/ and
    is exercised by no test when pair artifacts are absent (all pair tests
    skip). Do not read a pass here as "the tool runs standalone" -- for that,
    see headline:paper-pipeline-reproduces plus the assertion below.
    """
    TOOL = ['verdict', 'satir', 'pipeline.py', 'verdict_cli.py', 'satir_cli.py',
            'examples']
    ALLOWED_FALLBACK = {'verdict/data.py', 'matchers/data.py'}
    needle_a = chr(34) + 'experiments/'
    needle_b = chr(39) + 'experiments/'
    bad = []
    for rel in TOOL:
        base = ROOT / rel
        files = [base] if base.is_file() else [f for f in base.rglob('*.py')]
        for f in files:
            r = str(f.relative_to(ROOT))
            if r in ALLOWED_FALLBACK:
                continue
            for i, line in enumerate(f.read_text(errors='ignore').splitlines(), 1):
                if line.lstrip().startswith('#'):
                    continue
                if needle_a in line or needle_b in line:
                    bad.append(r + ':' + str(i))
    assert not bad, ('tool code reading experiments/: ' + ', '.join(bad[:8]))

    # Behavioural half: the headline reproduction is the one tool path we can
    # actually run here, so assert it never reaches for pair data (whose
    # fallback does live under experiments/).
    hl = (ROOT / 'verdict' / 'headline.py').read_text(errors='ignore')
    for sym in ('pair_root', 'iter_pairs', 'VERDICT_PAIR_DATA'):
        assert sym not in hl, 'headline.py must not depend on pair data: ' + sym


# ---------------------------------------------------------------- hygiene
# Both systems live here: VERDICT (the matcher) and SatIR (retrieval/compilation).
SHIPPED = ['scripts', 'smt_core', 'verbalizer', 'rationale_generators',
           'counterfactual_modifier', 'verdict_cli.py',
           # matchers was missing from this list, so 12 files under it kept
           # hardcoded home paths that the check never saw
           'matchers', 'satir', 'verdict',
           # SatIR
           'trial_compiler', 'patient_compiler', 'db_indexer', 'sql_retrieval',
           'smt_matcher', 'matching_batch', 'evaluation', 'audit',
           'satir_cli.py']


@check('hygiene:no-secrets-in-env-example')
def _():
    t = (ROOT / '.env.example').read_text()
    assert not re.search(r'sk-[A-Za-z0-9]{20,}', t), 'API key in .env.example'
    for host in re.findall(r'https?://([^/"\s]+)', t):
        assert host.startswith(('YOUR-', 'localhost', '127.0.0.1')), \
            f'real hostname in .env.example: {host}'


@check('hygiene:env-is-gitignored')
def _():
    gi = (ROOT / '.gitignore').read_text().splitlines()
    assert '.env' in gi and '!.env.example' in gi


@check('hygiene:no-hardcoded-paths-on-ship-path')
def _():
    bad = []
    for rel in SHIPPED:
        p = ROOT / rel
        files = [p] if p.is_file() else [f for f in p.rglob('*')
                                         if f.suffix in ('.py', '.sh')]
        for f in files:
            if '.bak' in f.name:
                continue
            try:
                txt = f.read_text(errors='ignore')
            except Exception:
                continue
            for m in re.findall(r'/Users/[A-Za-z0-9_.-]+', txt):
                bad.append(f'{f.relative_to(ROOT)}: {m}')
    assert not bad, 'hardcoded home paths:\n  ' + '\n  '.join(bad[:10])


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', help='only checks whose name contains this')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    VERBOSE = a.verbose

    checks = [(n, f) for n, f in CHECKS if not a.k or a.k in n]
    print(f'running {len(checks)} invariant checks\n')
    failed, skipped = [], []
    for name, fn in checks:
        try:
            fn()
            print(f'  PASS  {name}')
        except Skip as s:
            print(f'  SKIP  {name}  (needs {s})')
            skipped.append(name)
        except Exception as e:
            print(f'  FAIL  {name}')
            failed.append((name, e))
            if VERBOSE:
                print(f'        {e}')
    print()
    if skipped:
        print(f'{len(skipped)} skipped -- local-only data absent '
              f'(expected in a fresh clone; see docs/DATA.md)')
    if failed:
        print(f'{len(failed)}/{len(checks)} FAILED')
        for n, e in failed:
            print(f'\n--- {n} ---\n{str(e)[:900]}')
        return 1
    print(f'all {len(checks) - len(skipped)} runnable invariants hold')
    return 0


if __name__ == '__main__':
    sys.exit(main())
