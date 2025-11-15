# smt_qualifier_linker.py
# Post-process sliced SMT: ensure qualifiers imply stems; never the reverse.
# Now preserves top-level define-fun (and uses their result sorts for type checks).

from __future__ import annotations
from typing import Any, List, Tuple, Dict, Set, Optional
import pathlib
import json
import re

# ---------------------- S-expression parser (minimal, no comments kept) ----------------------

Token = str
S = Any  # Atom = str, List[S] = list

def _tokenize(s: str) -> List[Token]:
    s = re.sub(r";[^\n]*", "", s)  # strip line comments
    s = re.sub(r'("([^"\\]|\\.)*")', r' \1 ', s)  # keep quoted strings atomic
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]

def _parse(tokens: List[Token], i: int = 0) -> Tuple[S, int]:
    if i >= len(tokens): raise ValueError("unexpected EOF")
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

def is_list(x: S) -> bool: return isinstance(x, list)
def sym(x: S) -> Optional[str]: return x if isinstance(x, str) else None
def to_smt2(x: S) -> str: return x if isinstance(x, str) else "(" + " ".join(to_smt2(y) for y in x) + ")"

# ---------------------- Utilities ----------------------

def asrt_tag_and_body(assert_form: S) -> Tuple[Optional[str], S]:
    """Return (tag|None, body_term) for an (assert ...) potentially with (! ... :named TAG)."""
    if not (is_list(assert_form) and len(assert_form) >= 2 and sym(assert_form[0]) == "assert"):
        raise ValueError("not an assert")
    term = assert_form[1]
    if is_list(term) and term and sym(term[0]) == "!":
        body = term[1]
        tag = None
        j = 2
        while j + 1 < len(term):
            if sym(term[j]) == ":named":
                tag = sym(term[j + 1]); break
            j += 2
        return tag, body
    return None, term

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
    if not t: return out
    head = sym(t[0])
    # (let ((x e) ...) body)
    if head == "let" and len(t) >= 3 and is_list(t[1]):
        new_bound = set(bound)
        for bind in t[1]:
            if is_list(bind) and bind:
                v = sym(bind[0])
                if v: new_bound.add(v)
                if len(bind) >= 2:
                    out |= free_symbols(bind[1], bound)
        for body in t[2:]:
            out |= free_symbols(body, new_bound)
        return out
    # count predicate head
    if head and not _is_builtin(head) and head not in bound:
        out.add(head)
    # args
    for child in t[1:]:
        out |= free_symbols(child, bound)
    return out

def _split_stem_qual(name: str) -> Optional[Tuple[str, str]]:
    """Return (stem, qual_tail) if name contains '@@', else None."""
    if "@@" not in name: return None
    stem, qual = name.split("@@", 1)
    if stem and qual:
        return stem, qual
    return None

def _decl_sort_and_kind(form: S) -> Tuple[str, str]:
    """Return (kind, sort) where kind in {'const','fun'} and sort is a stringified sort."""
    if is_list(form) and form:
        h = sym(form[0])
        if h == "declare-const" and len(form) >= 3:
            return "const", to_smt2(form[2])
        if h == "declare-fun" and len(form) >= 4:
            return "fun", to_smt2(form[3])
    raise ValueError("not a declare-const/fun")

def _define_result_sort(form: S) -> str:
    """Return result sort of (define-fun name (args) Sort body)."""
    # form = ["define-fun", name, (args), Sort, body]
    return to_smt2(form[3]) if (is_list(form) and sym(form[0]) == "define-fun" and len(form) >= 5) else ""

# ---------------------- Linker core ----------------------

def ensure_qualifier_implies_stem(
    smt_text: str,
    *,
    add_missing_stem_decls: bool = False,          # safer default
    remove_reverse_implications: bool = True,
    drop_ill_typed_implications: bool = True,      # drop (=> ...) if args not Bool
    aux_tag_prefix: str = "AUTO_AUXILIARY_QUAL_IMPLIES_STEM_",
) -> Tuple[str, Dict[str, Any]]:
    ast = parse_sexpr(smt_text)

    # Index declarations, definitions, and asserts (and remember original order for emit)
    decls: Dict[str, S] = {}
    decl_kind_sort: Dict[str, Tuple[str, str]] = {}
    defines: Dict[str, S] = {}
    define_result_sort: Dict[str, str] = {}
    asserts: List[Tuple[S, Optional[str], S]] = []

    for form in ast:
        if not (is_list(form) and form):
            continue
        h = sym(form[0])
        if h == "declare-const" and len(form) >= 3:
            name = sym(form[1])
            if name:
                decls[name] = form
                decl_kind_sort[name] = ("const", to_smt2(form[2]))
        elif h == "declare-fun" and len(form) >= 4:
            name = sym(form[1])
            if name:
                decls[name] = form
                decl_kind_sort[name] = ("fun", to_smt2(form[3]))
        elif h == "define-fun" and len(form) >= 5:
            name = sym(form[1])
            if name:
                defines[name] = form
                define_result_sort[name] = _define_result_sort(form)
        elif h == "assert":
            tag, body = asrt_tag_and_body(form)
            asserts.append((form, tag, body))

    def sort_of(name: Optional[str]) -> Optional[str]:
        if not name: return None
        if name in decl_kind_sort:
            return decl_kind_sort[name][1]
        if name in define_result_sort:
            return define_result_sort[name]
        return None

    # Collect qualified pairs Q (=stem@@qual) -> stem
    symnames: Set[str] = set(decls.keys()) | set(defines.keys())
    for _full, _tag, body in asserts:
        symnames |= free_symbols(body)

    qual_pairs: Set[Tuple[str, str]] = set()
    for name in symnames:
        sp = _split_stem_qual(name)
        if sp:
            stem, _qual_tail = sp
            qual_pairs.add((name, stem))

    bool_bool_pairs: Set[Tuple[str, str]] = set()
    skipped_nonbool_pairs: Set[Tuple[str, str]] = set()
    for Q, Sname in qual_pairs:
        qs, ss = sort_of(Q), sort_of(Sname)
        if qs == "Bool" and ss == "Bool":
            bool_bool_pairs.add((Q, Sname))
        else:
            # If stem missing AND allowed AND Q is Bool → add Bool stem (declare-const)
            if add_missing_stem_decls and ss is None and qs == "Bool" and Sname not in defines:
                new_decl: S = ["declare-const", Sname, "Bool"]
                # update maps
                decls[Sname] = new_decl
                decl_kind_sort[Sname] = ("const", "Bool")
                bool_bool_pairs.add((Q, Sname))
            else:
                skipped_nonbool_pairs.add((Q, Sname))

    # Sweep existing assertions: keep, remove reverse links, drop ill-typed '=>'
    existing_q_impl_s: Set[Tuple[str, str]] = set()
    kept_asserts: List[S] = []
    removed_reverse: List[S] = []
    removed_illtyped: List[S] = []

    for full, tag, body in asserts:
        if is_list(body) and body and sym(body[0]) == "=>" and len(body) == 3:
            lhs = body[1]; rhs = body[2]
            lhs_name = sym(lhs); rhs_name = sym(rhs)

            # --- FIXED: only drop "ill-typed" when BOTH sides are simple symbols
            # and we actually know their sorts. This avoids deleting things like
            # (=> (or diseases...) risk), where lhs is a complex Bool expression.
            if drop_ill_typed_implications and lhs_name is not None and rhs_name is not None:
                lhs_sort = sort_of(lhs_name)
                rhs_sort = sort_of(rhs_name)
                if lhs_sort is not None and rhs_sort is not None:
                    if lhs_sort != "Bool" or rhs_sort != "Bool":
                        removed_illtyped.append(full)
                        continue

            # Record Q -> S that already exist (only for Bool-Bool symbol pairs)
            if lhs_name and rhs_name and (lhs_name, rhs_name) in bool_bool_pairs:
                existing_q_impl_s.add((lhs_name, rhs_name))
                kept_asserts.append(full)
                continue

            # Remove S -> Q (stem ⇒ qualifier), regardless of type
            sp_rhs = _split_stem_qual(rhs_name) if rhs_name else None
            if remove_reverse_implications and lhs_name and sp_rhs and sp_rhs[0] == lhs_name:
                removed_reverse.append(full)
                continue

        # keep everything else
        kept_asserts.append(full)

    # Add missing Q -> S links (only Bool-Bool)
    added_aux: List[S] = []
    counter = 0
    for Q, Sname in sorted(bool_bool_pairs):
        if (Q, Sname) not in existing_q_impl_s:
            body = ["=>", Q, Sname]
            tag = f"{aux_tag_prefix}{counter}"
            counter += 1
            annotated = ["!", body, ":named", tag]
            added_aux.append(["assert", annotated])

    # Rebuild top-level: preserve original order for declare-*, define-*, assert
    out_decls_defs: List[S] = []
    seen_decldef_text: Set[str] = set()

    # 1) Original declares/defines in original order
    for f in ast:
        if not (is_list(f) and f):
            continue
        h = sym(f[0])
        if h in {"declare-const", "declare-fun", "define-fun"}:
            txt = to_smt2(f)
            if txt not in seen_decldef_text:
                out_decls_defs.append(f); seen_decldef_text.add(txt)

    # 2) Any newly-added declare-const for stems not previously declared/defined
    for name, (k, srt) in decl_kind_sort.items():
        if name in decls:
            txt = to_smt2(decls[name])
            if txt not in seen_decldef_text and name not in defines:
                out_decls_defs.append(decls[name]); seen_decldef_text.add(txt)

    # Classify kept asserts back into aux vs other, then append auto-added aux
    kept_aux: List[S] = []
    kept_other: List[S] = []
    for form, tag, _ in asserts:
        if form in kept_asserts:
            if tag and "AUXILIARY" in tag:
                kept_aux.append(form)
            else:
                kept_other.append(form)

    # Emit
    lines: List[str] = []
    lines.append(";; ===================== QUALIFIER LINKER =====================")
    lines.append(";; Rule: add (qualifier => stem) only when BOTH are Bool; remove (stem => qualifier); drop ill-typed implications.")
    lines.append(f";; Added_aux={len(added_aux)} Removed_reverse={len(removed_reverse)} Removed_illtyped={len(removed_illtyped)} Skipped_nonbool_pairs={len(skipped_nonbool_pairs)}")
    lines.append(";; =============================================================")
    if out_decls_defs:
        lines.append("\n;; Declarations/definitions (original + any auto-added Bool stems)")
        lines.extend(to_smt2(f) for f in out_decls_defs)
    if kept_aux:
        lines.append("\n;; Existing auxiliary assertions")
        lines.extend(to_smt2(f) for f in kept_aux)
    if added_aux:
        lines.append("\n;; AUTO auxiliary (qualifier => stem) links [Bool→Bool]")
        lines.extend(to_smt2(f) for f in added_aux)
    if kept_other:
        lines.append("\n;; Other kept assertions")
        lines.extend(to_smt2(f) for f in kept_other)
    lines.append("")

    report = {
        "bool_bool_pairs": sorted(list(bool_bool_pairs)),
        "skipped_nonbool_pairs": sorted(list(skipped_nonbool_pairs)),
        "removed_reverse_count": len(removed_reverse),
        "removed_illtyped_count": len(removed_illtyped),
        "added_aux_count": len(added_aux),
    }
    return "\n".join(lines), report

# ---------------------- Param-only file helpers ----------------------

def link_qualifiers_paths(
    *,
    in_path: str,
    out_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    add_missing_stem_decls: bool = True,
    remove_reverse_implications: bool = True,
    drop_ill_typed_implications: bool = True,
    encoding: str = "utf-8",
) -> Tuple[str, Dict[str, Any]]:
    src = pathlib.Path(in_path).read_text(encoding=encoding)
    new_text, report = ensure_qualifier_implies_stem(
        src,
        add_missing_stem_decls=add_missing_stem_decls,
        remove_reverse_implications=remove_reverse_implications,
        drop_ill_typed_implications=drop_ill_typed_implications,
    )
    if out_path:
        p = pathlib.Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new_text, encoding=encoding)
    if manifest_path:
        p = pathlib.Path(manifest_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding=encoding)
    return new_text, report

def link_qualifiers_dir(
    *,
    in_dir: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    glob: str = "*.smt2",
    write_manifests: bool = True,
    encoding: str = "utf-8",
    skip_if_fresh: bool = True,
) -> Dict[str, Dict[str, Any]]:
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_dir / "_manifests"
    if write_manifests:
        manifest_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for smt in sorted(in_dir.glob(glob)):
        out_path = out_dir / smt.name
        man_path = manifest_dir / (smt.stem + ".json") if write_manifests else None
        if skip_if_fresh and out_path.exists() and out_path.stat().st_mtime >= smt.stat().st_mtime:
            # still copy manifest if missing
            if write_manifests and not man_path.exists():
                _, rep = link_qualifiers_paths(
                    in_path=str(smt), out_path=str(out_path), manifest_path=str(man_path),
                    encoding=encoding
                )
                results[smt.name] = rep
            else:
                # approximate report
                results[smt.name] = {"skipped": True}
            continue

        _, rep = link_qualifiers_paths(
            in_path=str(smt), out_path=str(out_path), manifest_path=str(man_path),
            encoding=encoding
        )
        print(f"[ok] {smt.name} -> {out_path.name}  (added_links={rep.get('added_aux_count',0)}, removed_reverse={rep.get('removed_reverse_count',0)})")
        results[smt.name] = rep

    # optional index
    (out_dir / "_index.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding=encoding)
    return results
