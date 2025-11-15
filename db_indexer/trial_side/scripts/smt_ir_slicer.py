# smt_ir_slicer.py
# Slice SMT-LIB for IR with a param-only API:
#   • Inclusion-style: keep asserts whose :named tag contains STRICT_TOKEN
#   • Exclusion-style: keep asserts whose :named tag DO NOT contain any excluded substrings
# Always keep relevant AUXILIARY (transitively) and only the needed declarations/definitions.
# NEW: Preserves define-fun and their transitive dependencies (closure), including 0-arity helpers.

from __future__ import annotations
from typing import Any, List, Tuple, Dict, Set, Optional
import re
import json
import pathlib

STRICT_TOKEN_DEFAULT = "PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE"

# ---------------------- S-expression parser ----------------------

Token = str
S = Any  # Atom = str, List[S] = list

def _tokenize(s: str) -> List[Token]:
    s = re.sub(r";[^\n]*", "", s)  # strip line comments
    s = re.sub(r'("([^"\\]|\\.)*")', r' \1 ', s)  # keep quoted strings atomic
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]

def _parse(tokens: List[Token], i: int = 0) -> Tuple[S, int]:
    if i >= len(tokens):
        raise ValueError("unexpected EOF")
    t = tokens[i]
    if t == "(":
        i += 1
        out: List[S] = []
        while i < len(tokens) and tokens[i] != ")":
            node, i = _parse(tokens, i)
            out.append(node)
        if i >= len(tokens) or tokens[i] != ")":
            raise ValueError("missing ')'")
        return out, i + 1
    if t == ")":
        raise ValueError("unexpected ')'")
    return t, i + 1

def parse_sexpr(src: str) -> List[S]:
    toks = _tokenize(src)
    i = 0
    out = []
    while i < len(toks):
        node, i = _parse(toks, i)
        out.append(node)
    return out

# ---------------------- AST helpers ----------------------

def is_list(x: S) -> bool:
    return isinstance(x, list)

def sym(x: S) -> Optional[str]:
    return x if isinstance(x, str) else None

def asrt_tag_and_body(assert_form: S) -> Tuple[Optional[str], S]:
    """
    ['assert', term] where term may be ['!', body, ':named', TAG, ...].
    Returns (TAG|None, body_term).
    """
    if not (is_list(assert_form) and len(assert_form) >= 2 and sym(assert_form[0]) == "assert"):
        raise ValueError("not an assert")
    term = assert_form[1]
    if is_list(term) and term and sym(term[0]) == "!":
        body = term[1]
        tag = None
        j = 2
        while j + 1 < len(term):
            if sym(term[j]) == ":named":
                tag = sym(term[j + 1])
                break
            j += 2
        return tag, body
    return None, term

def to_smt2(x: S) -> str:
    if isinstance(x, str):
        return x
    return "(" + " ".join(to_smt2(y) for y in x) + ")"

# Core/arith/logical tokens and builtins to ignore during symbol collection
NON_SYMBOLS: Set[str] = {
    "assert","declare-const","declare-fun","define-fun","let","ite","and","or","not","=>","=",
    "distinct","<","<=",">",">=","+","-","*","/","mod","div","true","false","!","as","_",
    "forall","exists", ":named", ":pattern", ":weight"
}
SORTS = {"Int","Real","Bool"}

def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r'[-+]?\d+(\.\d+)?', tok))

def _is_kw(tok: str) -> bool:
    return tok.startswith(":")

def _is_builtin(tok: str) -> bool:
    return tok in NON_SYMBOLS or tok in SORTS or _is_kw(tok) or _is_number(tok) or tok.startswith('"')

def free_symbols(t: S, bound: Set[str] | None = None) -> Set[str]:
    """Collect free symbols, respecting let-bound vars and counting user heads."""
    if bound is None: bound = set()
    out: Set[str] = set()
    if isinstance(t, str):
        if not _is_builtin(t) and t not in bound:
            out.add(t)
        return out
    if not t:
        return out

    head = sym(t[0])

    # (let ((x e) (y f) ...) body)
    if head == "let" and len(t) >= 3 and is_list(t[1]):
        new_bound = set(bound)
        for bind in t[1]:
            if is_list(bind) and bind:
                v = sym(bind[0])
                if v:
                    new_bound.add(v)
                if len(bind) >= 2:
                    out |= free_symbols(bind[1], bound)
        for body in t[2:]:
            out |= free_symbols(body, new_bound)
        return out

    # Count function/predicate head if user symbol
    if head and not _is_builtin(head) and head not in bound:
        out.add(head)

    for child in t[1:]:
        out |= free_symbols(child, bound)
    return out

def defun_signature(form: S) -> Tuple[str, List[str], S]:
    """
    Parse (define-fun name ((a1 S1) (a2 S2) ...) Sort body)
    Returns (name, [arg_names], body)
    """
    if not (is_list(form) and len(form) >= 5 and sym(form[0]) == "define-fun"):
        raise ValueError("not a define-fun")
    name = sym(form[1]) or ""
    params = form[2] if is_list(form[2]) else []
    arg_names: List[str] = []
    for p in params:
        if is_list(p) and p:
            v = sym(p[0])
            if v:
                arg_names.append(v)
    body = form[4]  # (define-fun n (args) Sort body)
    return name, arg_names, body

# ---------------------- Core slicer (string in -> string out) ----------------------

def slice_for_ir_ast(
    src: str,
    *,
    # Inclusion-style default:
    strict_token: str = STRICT_TOKEN_DEFAULT,
    include_auxiliary: bool = True,
    # Flexible tag policy (optional):
    include_tag_substrings: Optional[List[str]] = None,
    exclude_tag_substrings: Optional[List[str]] = None,
    require_named: bool = True,
    auxiliary_tag_substring: str = "AUXILIARY",
) -> Tuple[str, Dict[str, Any]]:
    """
    Slice SMT-LIB text for IR.

    Returns (sliced_smt2, manifest_dict).

    Tag policy (in order of precedence):
      1) If include_tag_substrings is provided: keep an assert iff its tag contains ANY of them.
      2) Else if exclude_tag_substrings is provided: keep an assert iff its tag contains NONE of them.
      3) Else: keep asserts whose tag contains `strict_token`.
         If require_named=False, untagged asserts are also kept by the active policy.

    Also:
      • Keep AUXILIARY assertions that reference any kept symbol (transitively).
      • Include declarations/definitions for symbols used by the kept slice.
      • NEW: If a used symbol is defined via define-fun, include that definition and
             add its body’s free symbols (respecting bound args) to the dependency closure.
    """
    ast = parse_sexpr(src)

    def _tag_keep(tag: Optional[str]) -> bool:
        if tag is None:
            return not require_named
        if include_tag_substrings:
            return any(s in tag for s in include_tag_substrings)
        if exclude_tag_substrings:
            return not any(s in tag for s in exclude_tag_substrings)
        if strict_token:
            return strict_token in tag
        return True

    # Index top-level forms
    decls: Dict[str, S] = {}
    defines: Dict[str, S] = {}
    asserts: List[Tuple[S, Optional[str], S]] = []

    for form in ast:
        if is_list(form) and form:
            h = sym(form[0])
            if h == "declare-const" and len(form) >= 3:
                name = sym(form[1])
                if name: decls[name] = form
            elif h == "declare-fun" and len(form) >= 4:
                name = sym(form[1])
                if name: decls[name] = form
            elif h == "define-fun" and len(form) >= 5:
                name = sym(form[1])
                if name: defines[name] = form
            elif h == "assert":
                tag, body = asrt_tag_and_body(form)
                asserts.append((form, tag, body))

    # Seed kept asserts by policy
    kept_asserts: List[S] = []
    kept_tags: List[str] = []
    used: Set[str] = set()
    for full, tag, body in asserts:
        if _tag_keep(tag):
            if full not in kept_asserts:
                kept_asserts.append(full)
                if tag is not None:
                    kept_tags.append(tag)
                used |= free_symbols(body)

    # AUXILIARY closure
    if include_auxiliary:
        changed = True
        while changed:
            changed = False
            for full, tag, body in asserts:
                if full in kept_asserts:
                    continue
                if tag and auxiliary_tag_substring in tag:
                    syms = free_symbols(body)
                    if used & syms:
                        kept_asserts.append(full)
                        before = len(used)
                        used |= syms
                        changed = changed or (len(used) > before)

    # --- NEW: include define-fun dependency closure ---
    kept_defines: List[S] = []
    changed = True
    while changed:
        changed = False
        for name, def_form in defines.items():
            if def_form in kept_defines:
                continue
            if name in used:
                # Include this definition
                kept_defines.append(def_form)
                # Add its body's free symbols, respecting bound params
                _, arg_names, body = defun_signature(def_form)
                before = len(used)
                used |= free_symbols(body, bound=set(arg_names))
                changed = changed or (len(used) > before)

    # Declarations for used symbols (excluding any that are defined by kept_defines)
    kept_decls: List[S] = []
    seen_decl_or_def: Set[str] = set(s for s, f in defines.items() if f in kept_defines)

    missing_decls: Set[str] = set()
    for sname in sorted(used):  # deterministic manifest
        if sname in seen_decl_or_def:
            continue  # provided by define-fun already included
        if sname in decls and sname not in seen_decl_or_def:
            kept_decls.append(decls[sname])
            seen_decl_or_def.add(sname)
        elif (sname not in SORTS and sname not in NON_SYMBOLS and
              sname not in defines):  # not a built-in, not defined, not declared
            missing_decls.add(sname)

    # Emit in original order
    def in_list(lst: List[S], f: S) -> bool:
        try:
            lst.index(f)
            return True
        except ValueError:
            return False

    decl_text: List[str] = []
    def_text: List[str] = []
    aux_text: List[str] = []
    kept_text: List[str] = []

    for f in ast:
        if in_list(kept_decls, f):
            decl_text.append(to_smt2(f))
        elif in_list(kept_defines, f):
            def_text.append(to_smt2(f))
        elif in_list(kept_asserts, f):
            # retrieve tag to decide AUX bucket
            tag_here = None
            for full, tag, _ in asserts:
                if full == f:
                    tag_here = tag
                    break
            if tag_here and auxiliary_tag_substring in tag_here:
                aux_text.append(to_smt2(f))
            else:
                kept_text.append(to_smt2(f))

    # Header explains active policy
    if include_tag_substrings:
        policy_line = f"Tag policy: INCLUDE any of {include_tag_substrings}"
    elif exclude_tag_substrings:
        policy_line = f"Tag policy: EXCLUDE any of {exclude_tag_substrings}"
    else:
        policy_line = f"STRICT token: {strict_token}"

    out_lines: List[str] = []
    out_lines.append(";; ===================== IR SLICE (AST) =====================")
    out_lines.append(f";; {policy_line}")
    out_lines.append(";; ===========================================================")
    if decl_text or def_text:
        out_lines.append("\n;; Declarations/definitions needed by IR slice")
        out_lines.extend(decl_text)
        out_lines.extend(def_text)
    if aux_text:
        out_lines.append("\n;; Relevant auxiliary (linking) assertions")
        out_lines.extend(aux_text)
    if kept_text:
        out_lines.append("\n;; Kept constraints per tag policy")
        out_lines.extend(kept_text)
    out_lines.append("")  # newline

    manifest: Dict[str, Any] = {
        "tag_policy": {
            "include": include_tag_substrings or [],
            "exclude": exclude_tag_substrings or [],
            "strict_token": strict_token if (not include_tag_substrings and not exclude_tag_substrings) else None,
            "require_named": require_named,
            "auxiliary_tag_substring": auxiliary_tag_substring,
        },
        "kept_assert_count": len(kept_text),
        "kept_aux_assert_count": len(aux_text),
        "kept_decl_count": len(decl_text),
        "kept_define_count": len(def_text),
        "used_symbol_count": len(used),
        "used_symbols": sorted(used),
        "missing_declarations": sorted(missing_decls),
        "kept_tags_first_10": kept_tags[:10],
    }
    return "\n".join(out_lines), manifest

# ---------------------- Param-only convenience wrappers ----------------------

def slice_ir_text(
    smt_text: str,
    *,
    strict_token: str = STRICT_TOKEN_DEFAULT,
    include_auxiliary: bool = True,
    include_tag_substrings: Optional[List[str]] = None,
    exclude_tag_substrings: Optional[List[str]] = None,
    require_named: bool = True,
    auxiliary_tag_substring: str = "AUXILIARY",
) -> Tuple[str, Dict[str, Any]]:
    """String -> (sliced_string, manifest) with flexible tag policy."""
    return slice_for_ir_ast(
        smt_text,
        strict_token=strict_token,
        include_auxiliary=include_auxiliary,
        include_tag_substrings=include_tag_substrings,
        exclude_tag_substrings=exclude_tag_substrings,
        require_named=require_named,
        auxiliary_tag_substring=auxiliary_tag_substring,
    )

def slice_ir_paths(
    *,
    in_path: str,
    out_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    strict_token: str = STRICT_TOKEN_DEFAULT,
    include_auxiliary: bool = True,
    include_tag_substrings: Optional[List[str]] = None,
    exclude_tag_substrings: Optional[List[str]] = None,
    require_named: bool = True,
    auxiliary_tag_substring: str = "AUXILIARY",
    encoding: str = "utf-8",
) -> Tuple[str, Dict[str, Any]]:
    """
    File-path based API (no CLI).
    """
    src = pathlib.Path(in_path).read_text(encoding=encoding)
    sliced, manifest = slice_for_ir_ast(
        src,
        strict_token=strict_token,
        include_auxiliary=include_auxiliary,
        include_tag_substrings=include_tag_substrings,
        exclude_tag_substrings=exclude_tag_substrings,
        require_named=require_named,
        auxiliary_tag_substring=auxiliary_tag_substring,
    )
    if out_path:
        p = pathlib.Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(sliced, encoding=encoding)
    if manifest_path:
        p = pathlib.Path(manifest_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding=encoding)
    return sliced, manifest

# Optional convenience wrappers for common policies:

def slice_ir_text_inclusion(smt_text: str, *, include_auxiliary: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Keep STRICT-tagged constraints (default token), plus relevant AUX."""
    return slice_ir_text(smt_text, include_auxiliary=include_auxiliary)

def slice_ir_text_exclusion(smt_text: str, *, include_auxiliary: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Keep everything EXCEPT 'NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION' (plus relevant AUX)."""
    return slice_ir_text(
        smt_text,
        include_auxiliary=include_auxiliary,
        include_tag_substrings=None,
        exclude_tag_substrings=["NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION"],
    )

def slice_ir_paths_exclusion(
    *, in_path: str, out_path: Optional[str] = None, manifest_path: Optional[str] = None,
    include_auxiliary: bool = True, encoding: str = "utf-8"
) -> Tuple[str, Dict[str, Any]]:
    """File-based exclusion policy wrapper."""
    return slice_ir_paths(
        in_path=in_path,
        out_path=out_path,
        manifest_path=manifest_path,
        include_auxiliary=include_auxiliary,
        include_tag_substrings=None,
        exclude_tag_substrings=["NOT_REQUIREMNET_OR_ALWAYS_SATISFIABLE_WITH_ACTION"],
        encoding=encoding,
    )
