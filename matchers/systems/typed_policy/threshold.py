"""Threshold-currying utilities (AEGIS-style).

Each numeric comparison in a trial's SMT program becomes a separate
Boolean atom of the form `__THRESH__::<var>::<op>::<N>`. This lets the
classifier and miner reason about each threshold independently, route
by clinical semantic of the comparison, and benefit from the typed
missingness policy on a per-threshold basis instead of collapsing all
numerics to one defer cell.

Functions:
    extract_thresh_atoms(smt_text)    Walk an SMT program; return list of
                                       (var, op_short, value) tuples.
    make_thresh_atom_name(var, op, n) Format as __THRESH__::var::op::N.
    thresh_binding_lines(atom, prog)  Emit SMT-LIB lines binding the
                                       Boolean atom to the comparison.
    quote_smt(name)                   Quote a name for SMT-LIB if needed.
    smt_number(v)                     Format a number for SMT-LIB.
"""
import re
import z3

_OP_TOKEN = {'>=': 'ge', '>': 'gt', '<=': 'le', '<': 'lt', '=': 'eq', '!=': 'ne'}
_OP_TOKEN_INV = {v: k for k, v in _OP_TOKEN.items()}
# When binding, default to boundary-inclusive (AEGIS convention): gt -> >=, lt -> <=.
_OP_BIND = {'ge': '>=', 'le': '<=', 'gt': '>=', 'lt': '<=', 'eq': '=', 'ne': '!='}

_THRESH_RE = re.compile(r'^__THRESH__::(.+)::(ge|le|gt|lt|eq|ne)::(-?\d+(?:\.\d+)?)$')


def quote_smt(name: str) -> str:
    return name if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) else f'|{name}|'


def smt_number(v) -> str:
    """Format a number for SMT-LIB without scientific notation."""
    f = float(v)
    if f == 0.0: return '0.0'
    s = f'{f:.20f}'.rstrip('0')
    if s.endswith('.'): s = s + '0'
    if s.startswith('-.'): s = '-0' + s[1:]
    if s.startswith('.'):  s = '0' + s
    return s


def _normalize_value(v) -> str:
    """Match cmsrc's literal-text normalization."""
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f.is_integer():
        return str(int(f))
    return f"{f}"


def make_thresh_atom_name(var: str, op_short: str, value) -> str:
    return f'__THRESH__::{var}::{op_short}::{_normalize_value(value)}'


def parse_thresh_atom(atom_name: str):
    """Parse a __THRESH__ atom name into (var, op_short, value)."""
    m = _THRESH_RE.match(atom_name)
    if not m: return None
    var, op_short, val = m.groups()
    val = float(val)
    if val.is_integer(): val = int(val)
    return var, op_short, val


def _z3_num_value(expr):
    try:
        if z3.is_int_value(expr):
            return expr.as_long()
        if z3.is_rational_value(expr):
            return float(expr.as_decimal(20).rstrip('?'))
        if z3.is_to_real(expr):
            return _z3_num_value(expr.children()[0])
        if z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_UMINUS:
            v = _z3_num_value(expr.children()[0])
            return -v if v is not None else None
    except Exception:
        return None
    return None


_OP_KIND_TOKEN = {
    z3.Z3_OP_GE: 'ge', z3.Z3_OP_GT: 'gt',
    z3.Z3_OP_LE: 'le', z3.Z3_OP_LT: 'lt',
    z3.Z3_OP_EQ: 'eq',
}


def _walk(expr, out: set):
    """Recurse over a Z3 AST; collect (var, op_short, value) triples for
    every numeric comparison whose RHS is a literal (or LHS is, with op
    flipped)."""
    if not z3.is_app(expr):
        return
    kind = expr.decl().kind()
    op = _OP_KIND_TOKEN.get(kind)
    if op is not None and len(expr.children()) == 2:
        lhs, rhs = expr.children()
        lhs_num = _z3_num_value(lhs)
        rhs_num = _z3_num_value(rhs)
        # var OP literal  (most common)
        if rhs_num is not None and lhs_num is None:
            try:
                lhs_str = str(lhs).strip('|')
                # Skip if LHS isn't a simple variable
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', lhs_str) or '_' in lhs_str:
                    out.add((lhs_str, op, rhs_num))
            except Exception:
                pass
        # literal OP var  (flip)
        elif lhs_num is not None and rhs_num is None:
            try:
                rhs_str = str(rhs).strip('|')
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', rhs_str) or '_' in rhs_str:
                    flip = {'ge': 'le', 'le': 'ge', 'gt': 'lt', 'lt': 'gt', 'eq': 'eq'}
                    out.add((rhs_str, flip[op], lhs_num))
            except Exception:
                pass
    for c in expr.children():
        _walk(c, out)


def extract_thresh_atoms(smt_text: str) -> list[tuple]:
    """Parse an SMT-LIB program; return sorted list of (var, op_short, value)
    triples for every numeric comparison against a literal."""
    if not smt_text or not smt_text.strip():
        return []
    triples = set()
    try:
        ast = z3.parse_smt2_string(smt_text)
    except z3.Z3Exception:
        return []
    if z3.is_ast(ast):
        try:
            _walk(ast, triples)
        except Exception:
            pass
    else:
        for a in ast:
            try:
                _walk(a, triples)
            except Exception:
                pass
    return sorted(triples, key=lambda t: (t[0], t[1], t[2]))


def thresh_binding_lines(atom_name: str, prog_lines: list = None) -> list:
    """SMT-LIB lines binding a Bool atom to its numeric comparison.

    Defaults to boundary-inclusive interpretation (gt -> >=, lt -> <=),
    matching AEGIS convention for prescreen clinical intent.
    """
    parsed = parse_thresh_atom(atom_name)
    if not parsed: return []
    var, op_short, val = parsed
    qvar = quote_smt(var); qatom = quote_smt(atom_name)
    op = _OP_BIND[op_short]
    if op == '!=':
        binding = f'(not (= {qvar} {smt_number(val)}))'
    else:
        binding = f'({op} {qvar} {smt_number(val)})'
    out = [f'(declare-const {qatom} Bool)']
    if prog_lines is not None:
        already = any(
            re.search(rf'\(declare-const\s+\|?{re.escape(var)}\|?\s+', ln) or
            re.search(rf'\(declare-fun\s+\|?{re.escape(var)}\|?\s+', ln)
            for ln in prog_lines
        )
        if not already:
            out.append(f'(declare-const {qvar} Real)')
    out.append(f'(assert (= {qatom} {binding}))')
    return out
