#!/usr/bin/env python3
# scripts/find_positive_canon_literals.py

"""
Extract canonical boolean literals per SMT2 file.

Defaults to --direction unsat_on_true:
  Collect symbols that appear exactly as (not <symbol>) inside kept constraints
  (asserts named with 'COMPONENT'). Setting such a symbol to TRUE tends to make
  the program UNSAT (i.e., patient excluded).

Use --direction sat_on_true:
  Collect positive bare-symbol literals (by polarity accounting for not/=>/and/or)
  inside kept constraints only.

Enhancements in this version:
  - Learns qualifier→stem relationships directly from logic via an implication graph.
  - Matches by stem so @@-qualified variables are recognized even if canon contains stems.
  - Works even if canonical list is empty; falls back to declared-bools minus demographics.
  - Outputs stems only by default (to remove @@-qualified names from final results).
  - Per-file JSON also includes raw hits and representatives for traceability when requested.
  - Inlines boolean AUX definitions of the form (= X φ) into COMPONENT formulas
    before scanning, so positivity can flow into φ.
  - Implication graph also learns shallow links from (= X φ).
  - NEW (domain semantics): In sat_on_true mode, diagnoses (and other canon vars)
    that appear in the antecedent of AUXILIARY implications of the form
      (=> ANTECEDENT CONSEQUENT)
    are treated as positive literals if CONSEQUENT is already a positive hit
    in a COMPONENT.
  - NEW (reverse domain semantics): In sat_on_true mode, canonical CONSEQUENT
    variables are also treated as positive literals when their bare-symbol
    ANTECEDENT is known positive (either already in hits or positive in a
    COMPONENT). This covers patterns like:
        COMPONENT:  patient_has_active_graves_hyperthyroidism_now
        AUX:        (=> patient_has_active_graves_hyperthyroidism_now
                         patient_has_finding_of_thyrotoxicosis_due_to_graves_disease_now)
  - NEW (numeric antecedent semantics): In sat_on_true mode, canonical CONSEQUENT
    variables of AUXILIARY implications whose (possibly numeric) ANTECEDENT is
    itself enforced positively in a COMPONENT (e.g., a threshold like
        (>= patient_has_finding_of_pigmented_lesion_now_count_in_integer 1)
    that appears as a positive conjunct in a COMPONENT) are also treated as
    positive literals.
"""

import json, csv, re, argparse
from pathlib import Path
from typing import List, Union, Tuple, Set, Dict, Iterable
from collections import defaultdict, deque

# ==================== Tiny S-expression parser ====================
Token = str
Sexpr = Union[str, List["Sexpr"]]

def tokenize(s: str) -> List[Token]:
    tokens, cur, in_str = [], [], False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if in_str:
            if c == '"':
                cur.append(c); tokens.append("".join(cur)); cur=[]; in_str=False
            else:
                cur.append(c)
            i += 1; continue
        if c.isspace():
            if cur: tokens.append("".join(cur)); cur=[]
            i += 1; continue
        if c == ';':  # comment to EOL
            if cur: tokens.append("".join(cur)); cur=[]
            while i < n and s[i] != '\n': i += 1
            continue
        if c in '()':
            if cur: tokens.append("".join(cur)); cur=[]
            tokens.append(c); i += 1; continue
        if c == '"':
            if cur: tokens.append("".join(cur)); cur=[]
            cur.append(c); in_str=True; i += 1; continue
        cur.append(c); i += 1
    if cur: tokens.append("".join(cur))
    return tokens

def parse(tokens: List[Token], pos: int = 0) -> Tuple[Sexpr, int]:
    if pos >= len(tokens): raise ValueError("Unexpected end of tokens")
    t = tokens[pos]
    if t == '(':
        lst: List[Sexpr] = []; pos += 1
        while pos < len(tokens) and tokens[pos] != ')':
            node, pos = parse(tokens, pos)
            lst.append(node)
        if pos >= len(tokens) or tokens[pos] != ')': raise ValueError("Unmatched '('")
        return lst, pos + 1
    elif t == ')':
        raise ValueError("Unmatched ')'")
    else:
        return t, pos + 1

def parse_all(s: str) -> List[Sexpr]:
    toks = tokenize(s)
    out, pos = [], 0
    while pos < len(toks):
        node, pos = parse(toks, pos)
        out.append(node)
    return out

# ==================== Demographic stem filter ====================
UNITS = r"(?:minutes|hours|days|weeks|months|years)"
TIMEFRAME = (
    r"(?:(?:now|inthehistory|inthefuture)|"
    r"(?:inthepast\d+" + UNITS + r")|"
    r"(?:inthefuture\d+" + UNITS + r")|"
    r"(?:foradurationof\d+" + UNITS + r"))"
)
SEX = r"(?:male|female|other)"

DEMOGRAPHIC_PATTERNS = [
    re.compile(rf"^patient_age_value_recorded_{TIMEFRAME}_in_years$"),
    re.compile(rf"^patient_age_value_recorded_{TIMEFRAME}_in_months$"),
    re.compile(rf"^patient_age_value_recorded_{TIMEFRAME}_in_days$"),
    re.compile(rf"^patient_sex_is_{SEX}_{TIMEFRAME}$"),
    re.compile(rf"^patient_is_pregnant_{TIMEFRAME}$"),
    re.compile(rf"^patient_is_able_to_be_pregnant_{TIMEFRAME}$"),
    re.compile(rf"^patient_has_childbearing_potential_{TIMEFRAME}$"),
    re.compile(rf"^patient_is_breastfeeding_{TIMEFRAME}$"),
    re.compile(rf"^patient_is_lactating_{TIMEFRAME}$"),
]

def is_demographic_stem(var: str) -> bool:
    return any(p.fullmatch(var) for p in DEMOGRAPHIC_PATTERNS)

# ==================== Canonical loader (flat strings only) ====================
# Allow optional subcohort letter after the 8 digits (e.g., NCT02417740b)
SIR_PATTERN = re.compile(
    r"^(NCT\d{8}[A-Za-z]?)_(inclusion|exclusion)_program(?:\.assumed)?\.smt2$"
)
MCANON_FMT = "{trial}_{arm}_canonical_variables_entity_variable_names.json"

def match_trial_and_arm(p: Path):
    m = SIR_PATTERN.match(p.name)
    if not m: return None, None
    return m.group(1), m.group(2)

STRING_KEYS = {"name", "variable", "variable_name", "entity_variable_name"}

def _flatten_strings(x) -> Iterable[str]:
    if x is None:
        return
    if isinstance(x, str):
        yield x; return
    if isinstance(x, list):
        for y in x:
            yield from _flatten_strings(y)
        return
    if isinstance(x, dict):
        for k in ("entity_variable_names","canonical_variables","variables","names","items"):
            if k in x:
                yield from _flatten_strings(x[k])
        for k in STRING_KEYS:
            v = x.get(k)
            if isinstance(v, str):
                yield v
        return
    # ignore other types

def load_canonical_list(minified_canon_dir: Path, trial_id: str, arm: str) -> List[str]:
    """Load canonical list for the exact arm; empty is acceptable."""
    path = minified_canon_dir / MCANON_FMT.format(trial=trial_id, arm=arm)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    seen, out = set(), []
    for s in _flatten_strings(data):
        if isinstance(s, str) and s not in seen:
            seen.add(s); out.append(s)
    return out

# ==================== Declared Bool names ====================
DECL_CONST_BOOL = re.compile(r"\(declare-const\s+([^\s\)]+)\s+Bool\)")
DECL_FUN_BOOL0  = re.compile(r"\(declare-fun\s+([^\s\)]+)\s+\(\s*\)\s+Bool\)")

def get_declared_bools(smt_text: str) -> set[str]:
    names = set(m.group(1) for m in DECL_CONST_BOOL.finditer(smt_text))
    names.update(m.group(1) for m in DECL_FUN_BOOL0.finditer(smt_text))
    return names

# ==================== Helpers: stems and assertion unwrap ====================
def stem_of(var: str) -> str:
    """Lexical stem: split at '@@' and take the left part."""
    return var.split('@@', 1)[0]

def unwrap_assert_inner(form: Sexpr) -> Sexpr:
    """Unwrap (! INNER :named TAG ...) shells."""
    if isinstance(form, list) and len(form) >= 2 and form[0] == "!":
        return form[1]
    return form

def unwrap_named(form: Sexpr):
    """Return (inner, name_or_None). Accepts bare INNER or (! INNER :named TAG ...)."""
    if isinstance(form, list) and len(form) >= 2 and form[0] == "!":
        inner = form[1]
        name = None
        attrs = form[2:]
        for i in range(len(attrs) - 1):
            if attrs[i] == ":named" and isinstance(attrs[i+1], str):
                name = attrs[i+1]
                break
        return inner, name
    return form, None

# ==================== Boolean definition extraction & inlining ====================
def is_assert(node) -> bool:
    return isinstance(node, list) and len(node) >= 2 and node[0] == "assert"

def extract_boolean_defs(sexprs: List[Sexpr]) -> dict[str, Sexpr]:
    """
    Collect boolean definitional equalities from asserts:
       (assert (! (= X φ) :named REQ*_AUXILIARY*))
    Keep when X is an atomic symbol (str).
    """
    defs: dict[str, Sexpr] = {}
    for node in sexprs:
        if not is_assert(node):
            continue
        inner, _ = unwrap_named(node[1])
        if isinstance(inner, list) and len(inner) == 3 and inner[0] == "=":
            lhs, rhs = inner[1], inner[2]
            if isinstance(lhs, str):
                defs[lhs] = rhs
    return defs

def subst_defs(expr: Sexpr, env: dict[str, Sexpr], seen: set[str] | None = None, depth: int = 0, max_depth: int = 8) -> Sexpr:
    """
    Recursively inline X -> env[X] inside expr.
    Cycle-safe via 'seen'; depth-limited for robustness.
    """
    if depth > max_depth:
        return expr
    if seen is None:
        seen = set()
    if isinstance(expr, str):
        if expr in env and expr not in seen:
            seen2 = set(seen); seen2.add(expr)
            return subst_defs(env[expr], env, seen2, depth + 1, max_depth)
        return expr
    if not isinstance(expr, list):
        return expr
    return [expr[0]] + [subst_defs(e, env, seen, depth + 1, max_depth) for e in expr[1:]]

# ==================== Implication graph (=>) + shallow (=) edges ====================
def build_implication_graph(sexprs: List[Sexpr]) -> Dict[str, Set[str]]:
    """
    Build directed edges a -> b for:
      - (=> a b) where a,b are atomic strings
      - shallow edges from (= X φ): add X -> atom for atoms found shallowly in φ
    """
    G: Dict[str, Set[str]] = defaultdict(set)
    for node in sexprs:
        if not is_assert(node):
            continue
        inner = unwrap_assert_inner(node[1])

        # Handle (=> a b)
        if isinstance(inner, list) and len(inner) == 3 and inner[0] == "=>":
            a, b = inner[1], inner[2]
            if isinstance(a, str) and isinstance(b, str):
                G[a].add(b)
            continue

        # Handle (= X φ) shallowly
        if isinstance(inner, list) and len(inner) == 3 and inner[0] == "=":
            lhs, rhs = inner[1], inner[2]
            if isinstance(lhs, str):
                def add_edges_from_rhs(e):
                    if isinstance(e, str):
                        G[lhs].add(e)
                    elif isinstance(e, list):
                        for sub in e[1:]:
                            add_edges_from_rhs(sub)
                add_edges_from_rhs(rhs)
    return G

def closure_ancestors(G: Dict[str, Set[str]], v: str) -> Set[str]:
    """Return all nodes reachable forward from v (i.e., its implied ancestors)."""
    seen: Set[str] = set()
    q: deque[str] = deque([v])
    while q:
        x = q.popleft()
        for y in G.get(x, ()):
            if y not in seen:
                seen.add(y); q.append(y)
    return seen

def choose_rep(v: str, declared_bools: Set[str], G: Dict[str, Set[str]]) -> str:
    """
    Pick a representative ("stem") for variable v using implication graph and lexical stem hints.
    Preference order:
      1) lexical stem if declared or in ancestor closure,
      2) an ancestor without '@@',
      3) otherwise the shortest (then alphabetically) among {ancestors ∪ {v}}.
    """
    anc = closure_ancestors(G, v) | {v}
    lex = stem_of(v)
    if lex in anc or lex in declared_bools:
        return lex
    candidates = sorted(anc, key=lambda s: (('@@' in s), len(s), s))
    return candidates[0] if candidates else v

# ==================== Polarity-aware scan (for sat_on_true) ====================
LOGICAL_CONNECTIVES = {"and", "or"}
NEGATION = "not"
IMPLIES = "=>"

def collect_positive_constraint_literals(expr: Sexpr, canon: Set[str], polarity: bool, out: Set[str]) -> None:
    """Collect bare-symbol positives under given polarity."""
    if isinstance(expr, str):
        if polarity and expr in canon:
            out.add(expr)
        return
    if not isinstance(expr, list) or not expr:
        return
    head = expr[0] if isinstance(expr[0], str) else None
    if head == NEGATION and len(expr) == 2:
        collect_positive_constraint_literals(expr[1], canon, not polarity, out)
        return
    if head == IMPLIES and len(expr) == 3:
        collect_positive_constraint_literals(expr[1], canon, not polarity, out)  # antecedent
        collect_positive_constraint_literals(expr[2], canon, polarity, out)      # consequent
        return
    if head in LOGICAL_CONNECTIVES:
        for sub in expr[1:]:
            collect_positive_constraint_literals(sub, canon, polarity, out)
        return
    for sub in expr[1:]:
        collect_positive_constraint_literals(sub, canon, polarity, out)

# ==================== Component scanners (two directions) ====================
def collect_from_components(smt_text: str, canon: Set[str], direction: str) -> Set[str]:
    """
    direction:
      - 'unsat_on_true' => symbols appearing exactly as (not <sym>) in kept COMPONENT asserts
      - 'sat_on_true'   => positive bare symbols (by polarity) within kept COMPONENT asserts
    """
    sexprs = parse_all(smt_text)
    res: Set[str] = set()

    # harvest boolean (= X φ) defs once
    def_map = extract_boolean_defs(sexprs)

    for node in sexprs:
        if not (isinstance(node, list) and len(node) >= 2 and node[0] == "assert"):
            continue

        form = node[1]
        name = None
        inner = form

        # unwrap attributes like: (assert (! INNER :named TAG ...))
        if isinstance(form, list) and len(form) >= 2 and form[0] == "!":
            attrs = form[2:]
            for i in range(len(attrs) - 1):
                if attrs[i] == ":named" and isinstance(attrs[i+1], str):
                    name = attrs[i+1]
                    break
            inner = form[1]

        # only kept constraints (COMPONENT). If unnamed or non-COMPONENT, skip.
        if not (isinstance(name, str) and "COMPONENT" in name):
            continue

        # NEW: in sat_on_true mode, first scan the *unexpanded* inner so bare
        # canonical booleans like patient_is_postpartum_now are recorded
        # before we inline (= X φ) away.
        if direction == "sat_on_true":
            collect_positive_constraint_literals(inner, canon, True, res)

        # inline defs so references to X expand to φ before we scan internals
        inner_expanded = subst_defs(inner, def_map)

        if direction == "unsat_on_true":
            # exactly (not <sym>)
            if (
                isinstance(inner_expanded, list)
                and len(inner_expanded) == 2
                and inner_expanded[0] == "not"
                and isinstance(inner_expanded[1], str)
            ):
                sym = inner_expanded[1]
                if sym in canon:
                    res.add(sym)
        else:  # sat_on_true
            collect_positive_constraint_literals(inner_expanded, canon, True, res)

    return res

# ==================== AUX implication expansion (sat_on_true) ====================
def _occurs_positive(expr: Sexpr, target: str, polarity: bool) -> bool:
    """Return True iff `target` occurs as a bare symbol under positive polarity in expr."""
    if isinstance(expr, str):
        return polarity and expr == target

    if not isinstance(expr, list) or not expr:
        return False

    head = expr[0] if isinstance(expr[0], str) else None

    if head == NEGATION and len(expr) == 2:
        return _occurs_positive(expr[1], target, not polarity)

    if head == IMPLIES and len(expr) == 3:
        # antecedent under flipped polarity, consequent under same polarity
        return (
            _occurs_positive(expr[1], target, not polarity)
            or _occurs_positive(expr[2], target, polarity)
        )

    if head in LOGICAL_CONNECTIVES:
        return any(_occurs_positive(sub, target, polarity) for sub in expr[1:])

    return any(_occurs_positive(sub, target, polarity) for sub in expr[1:])

def _consequent_is_positive_in_component(sexprs: List[Sexpr], var: str) -> bool:
    """
    Check whether `var` appears as a positive literal inside any COMPONENT assert,
    regardless of whether `var` is in the canonical set.
    """
    for node in sexprs:
        if not is_assert(node):
            continue
        form = node[1]
        inner, name = unwrap_named(form)
        if not (isinstance(name, str) and "COMPONENT" in name):
            continue
        if _occurs_positive(inner, var, True):
            return True
    return False

def _expr_occurs_positive(expr: Sexpr, target: Sexpr, polarity: bool) -> bool:
    """
    Structural analogue of _occurs_positive: return True iff `target` (which may be
    a compound expression, such as a numeric threshold) occurs under positive
    polarity inside `expr`.
    """
    # Direct structural match under positive polarity
    if polarity and expr == target:
        return True

    if not isinstance(expr, list) or not expr:
        return False

    head = expr[0] if isinstance(expr[0], str) else None

    if head == NEGATION and len(expr) == 2:
        return _expr_occurs_positive(expr[1], target, not polarity)

    if head == IMPLIES and len(expr) == 3:
        # antecedent under flipped polarity, consequent under same polarity
        return (
            _expr_occurs_positive(expr[1], target, not polarity)
            or _expr_occurs_positive(expr[2], target, polarity)
        )

    if head in LOGICAL_CONNECTIVES:
        return any(_expr_occurs_positive(sub, target, polarity) for sub in expr[1:])

    return any(_expr_occurs_positive(sub, target, polarity) for sub in expr[1:])

def _expr_occurs_positively_in_component(sexprs: List[Sexpr], target: Sexpr) -> bool:
    """
    Check whether `target` (which may be a compound expression, e.g. a numeric
    comparison) occurs as a positive sub-expression in any COMPONENT assert.

    This is more general than just "top-level conjunct": it will traverse
    conjunctions and other connectives, while respecting polarity and =>.
    """
    for node in sexprs:
        if not is_assert(node):
            continue
        form = node[1]
        inner, name = unwrap_named(form)
        if not (isinstance(name, str) and "COMPONENT" in name):
            continue
        if _expr_occurs_positive(inner, target, True):
            return True
    return False

def expand_hits_via_aux_implications(hits: Set[str], sexprs: List[Sexpr], canon: Set[str]) -> Set[str]:
    """
    Domain-specific expansion used in sat_on_true mode:

    For AUXILIARY asserts of the form
        (assert (! (=> ANTECEDENT CONSEQUENT) :named REQ*_AUXILIARY*))

    1) Existing direction:
       If CONSEQUENT is either
         - already in 'hits', or
         - appears as a positive literal in a COMPONENT,
       then any canonical variables appearing in ANTECEDENT are also treated as
       positive literals and added to 'hits'.

    2) Reverse direction:
       If ANTECEDENT is a bare symbol that is
         - already in 'hits', or
         - appears as a positive literal in a COMPONENT,
       and CONSEQUENT is canonical, then CONSEQUENT is also added to 'hits'.

       This allows us to treat canonical diagnosis variables that sit on the
       CONSEQUENT side of an AUX implication as positive when their driver
       flag (often a more "active" or composite variable) is positive in a
       COMPONENT.

    3) Numeric/general antecedent direction (NEW):
       If ANTECEDENT is a (possibly non-boolean-typed) expression — e.g.,
       a numeric threshold such as (>= count 1) — and CONSEQUENT is
       canonical, and that ANTECEDENT occurs under positive polarity in
       some COMPONENT constraint, then CONSEQUENT is added to 'hits'.

       This captures patterns like:
         COMPONENT:  (>= patient_has_finding_of_pigmented_lesion_now_count_in_integer 1)
         AUX:        (=> (>= patient_has_finding_of_pigmented_lesion_now_count_in_integer 1)
                          patient_has_finding_of_pigmented_skin_lesion_now)
    """
    if not hits and not canon:
        # Without any hits and without a canonical filter, there's nothing useful
        # to expand. Returning early keeps behavior predictable.
        return hits

    expanded_hits = set(hits)

    def collect_atoms(expr: Sexpr, acc: Set[str]) -> None:
        # For the "forward" rule we only want canonical atoms.
        if isinstance(expr, str):
            if expr in canon:
                acc.add(expr)
            return
        if isinstance(expr, list):
            for sub in expr[1:] if expr else []:
                collect_atoms(sub, acc)

    for node in sexprs:
        if not is_assert(node):
            continue
        form = node[1]
        inner, name = unwrap_named(form)

        # Only AUXILIARY links carry the "driver" semantics we’re after.
        if not (isinstance(name, str) and "AUXILIARY" in name):
            continue

        # Look for (=> ANTECEDENT CONSEQUENT)
        if isinstance(inner, list) and len(inner) == 3 and inner[0] == "=>":
            antecedent, consequent = inner[1], inner[2]

            # ---------- Rule 1: CONSEQUENT-driven expansion ----------
            # Treat CONSEQUENT as a driver if:
            #   - it is already in hits (canonical positive), OR
            #   - it appears positively in some COMPONENT (even if non-canonical)
            if isinstance(consequent, str) and (
                consequent in expanded_hits
                or _consequent_is_positive_in_component(sexprs, consequent)
            ):
                diags: Set[str] = set()
                collect_atoms(antecedent, diags)
                expanded_hits.update(diags)

            # ---------- Rule 2: ANTECEDENT-driven expansion ----------
            # If the ANTECEDENT is a *bare symbol* that we know is positive
            # (either already in expanded_hits or positive in a COMPONENT),
            # and CONSEQUENT is canonical, then we add CONSEQUENT.
            if (
                isinstance(antecedent, str)
                and isinstance(consequent, str)
                and consequent in canon
            ):
                antecedent_positive = (
                    antecedent in expanded_hits
                    or _consequent_is_positive_in_component(sexprs, antecedent)
                )
                if antecedent_positive:
                    expanded_hits.add(consequent)

            # ---------- Rule 3: Numeric / general antecedent-driven expansion (NEW) ----------
            # If the ANTECEDENT is a compound expression (e.g., a numeric comparison),
            # and that exact expression occurs under positive polarity in some
            # COMPONENT, and CONSEQUENT is canonical, then CONSEQUENT is also treated
            # as positive.
            if (
                isinstance(antecedent, list)     # we only need this rule for compound exprs
                and isinstance(consequent, str)
                and consequent in canon
            ):
                if _expr_occurs_positively_in_component(sexprs, antecedent):
                    expanded_hits.add(consequent)

    return expanded_hits

# ==================== Main driver ====================
def run(build_root: Path, warn_empty: bool, direction: str, output_mode: str):
    """
    output_mode:
      - 'stems' (default): emit representative stems only (no @@ in final hits)
      - 'raw': emit raw literal hits only
      - 'both': include stems in 'hits' and add 'raw_hits' alongside (CSV uses stems)
    """
    slice_ir_dir = build_root / "slice_ir_linked"
    minified_canon_dir = build_root / "minified_canon"
    out_root = build_root / "positive_constraint_literals"
    out_per_file = out_root / "per_file"
    out_root.mkdir(parents=True, exist_ok=True)
    out_per_file.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    processed = 0
    empty_canon = 0

    for smt_file in sorted(slice_ir_dir.glob("*.smt2")):
        trial_id, arm = match_trial_and_arm(smt_file)
        if not trial_id:
            continue

        try:
            smt_text = smt_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[ERROR] Could not read {smt_file.name}: {e}")
            continue

        # Parse once for graph learning and AUX expansion
        sexprs = parse_all(smt_text)

        # Declared Bool universe
        declared_bools = get_declared_bools(smt_text)

        # Try to load canonical list (optional)
        try:
            canon_list = load_canonical_list(minified_canon_dir, trial_id, arm)
        except Exception as e:
            print(f"[ERROR] Canonical load failed for {smt_file.name}: {e}")
            canon_list = []

        if not canon_list and warn_empty:
            empty_canon += 1
            print(f"[WARN] Empty canonical list for {smt_file.name}")

        # Build a stem-aware canon set for filtering collection:
        # - If canonical list exists: accept any declared Bool whose *stem* is in canonical stems.
        # - Else (logic-only fallback): accept all declared Bools excluding demographics by stem.
        if canon_list:
            canonical_stems: Set[str] = {stem_of(s) for s in canon_list if isinstance(s, str)}
            canon: Set[str] = {
                v for v in declared_bools
                if stem_of(v) in canonical_stems and not is_demographic_stem(stem_of(v))
            }
        else:
            canon = {
                v for v in declared_bools
                if not is_demographic_stem(stem_of(v))
            }

        # Collect hits by direction (with definitional inlining already handled)
        try:
            hits_from_components = collect_from_components(smt_text, canon, direction)
        except Exception as e:
            print(f"[ERROR] {smt_file.name}: {e}")
            continue

        hits = set(hits_from_components)

        # In sat_on_true mode, expand hits via AUX implications
        if direction == "sat_on_true":
            hits = expand_hits_via_aux_implications(hits, sexprs, canon)

        # -------- DEBUG: per-file summary --------
        print(f"[DEBUG] File {smt_file.name}: trial_id={trial_id}, arm={arm}")
        print(f"        declared_bools={sorted(declared_bools)}")
        print(f"        canon={sorted(canon)}")
        print(f"        hits_from_components={sorted(hits_from_components)}")
        print(f"        hits_after_aux={sorted(hits)}")

        # Uncomment if you want deep debug for a particular trial:
        # if trial_id == "NCT02417740b":
        #     print("        *** SPECIAL DEBUG FOR NCT02417740b ***")
        #     for node in sexprs:
        #         if is_assert(node):
        #             inner, name = unwrap_named(node[1])
        #             print(f"          assert name={name}")
        #             print(f"                 inner={inner}")

        # -------- end DEBUG --------

        # Learn qualifier→stem from logic and expand to representatives
        G = build_implication_graph(sexprs)

        # Add cheap lexical edges v -> stem(v) when both declared (helps when AUX edges are absent)
        for v in declared_bools:
            st = stem_of(v)
            if st != v and st in declared_bools:
                G[v].add(st)

        reps_map: Dict[str, str] = {}
        stems_set: Set[str] = set()
        for h in sorted(hits):
            rep = choose_rep(h, declared_bools, G)
            if not is_demographic_stem(stem_of(rep)):
                reps_map[h] = rep
                stems_set.add(rep)

        # Decide what to emit
        raw_hits_sorted = sorted(hits)
        stems_sorted = sorted(stems_set)
        if output_mode == "raw":
            out_hits = raw_hits_sorted
        else:
            out_hits = stems_sorted  # 'stems' and 'both' use stems as primary

        # Persist per-file JSON
        per_file_json = {
            "trial_id": trial_id,
            "arm": arm,
            "smt2_file": smt_file.name,
            "num_hits": len(out_hits),
            "hits": out_hits,
            "direction": direction,
            "representatives": [{"literal": h, "rep": reps_map.get(h, h)} for h in sorted(hits)],
        }
        if output_mode in ("both", "raw"):
            per_file_json["raw_hits"] = raw_hits_sorted
        if output_mode == "both":
            per_file_json["hits_plus_representatives"] = sorted(set(raw_hits_sorted) | stems_set)

        (out_per_file / f"{smt_file.name}.json").write_text(
            json.dumps(per_file_json, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        rows.append({
            "trial_id": trial_id,
            "arm": arm,
            "smt2_file": smt_file.name,
            "num_hits": str(len(out_hits)),
            "hits": ";".join(out_hits),
            "direction": direction,
        })
        processed += 1

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id","arm","smt2_file","num_hits","hits","direction"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Processed {processed} SMT2 files.")
    print(f"[OK] Wrote {len(rows)} rows to {out_root/'summary.csv'}")
    print(f"[OK] Per-file JSONs at {out_per_file}")
    if empty_canon and warn_empty:
        print(f"[INFO] {empty_canon} file(s) had empty canonical lists")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract canonical boolean literals from kept constraints (demographics excluded; Bool-only)."
    )
    parser.add_argument("build_path", nargs="?", default=None,
                        help="Path to build/ (default: ../../build relative to this script)")
    parser.add_argument("--warn-empty", action="store_true",
                        help="Log warnings for files with empty canonical lists")
    parser.add_argument("--direction", choices=["unsat_on_true", "sat_on_true"],
                        default="sat_on_true",
                        help="unsat_on_true: (not <sym>) under COMPONENT; sat_on_true: positive literals under COMPONENT")
    parser.add_argument("--output-mode", choices=["stems","raw","both"], default="stems",
                        help="stems: emit representative stems only (default); raw: emit raw literals; both: stems as 'hits' plus 'raw_hits' for traceability")
    args = parser.parse_args()

    if args.build_path:
        base = Path(args.build_path).resolve()
    else:
        here = Path(__file__).resolve()
        base = (here.parent / ".." / ".." / "build").resolve()

    run(base, warn_empty=args.warn_empty, direction=args.direction, output_mode=args.output_mode)
