"""
registry_decoder.py – comment-aware edition
-------------------------------------------
Re-creates SMT-LIB v2 text from the Registries object produced by
registry_builder.RegistryBuilder.scan().
"""

from __future__ import annotations
from typing import List
import textwrap

# single source of data models
from .RegistryBuilder import SortMeta, Registries


# ───────────────────────────── registry → SMT ────────────────────────────
class RegistryDecoder:
    # .....................................................................
    @staticmethod
    def to_smt(regs: Registries,
               *,
               header: str | None = None,
               pretty_assert: bool = True,
               include_comments: bool = True) -> str:
        """
        Emit a *complete* SMT-LIB program from `regs`.

        Order:
        0. banner                 (optional)
        1. stand-alone comments   (original order)
        2. sorts      (alphabetical)
        3. const/fun   "
        4. assertions (tag order)

        Inline comments captured by the builder are re-appended to their
        declaration lines automatically.
        """
        lines: List[str] = []

        # 0) optional banner
        if header:
            lines += [f";; {header}", ""]

        # 1) stand-alone comments
        if include_comments:
            for c in sorted(regs.comments, key=lambda x: x.line_no):
                lines.append(f"; {c.text}")

            if regs.comments:
                lines.append("")  # blank line after comment block

        # 2) sorts -----------------------------------------------------
        for name, meta in sorted(regs.sorts.items()):
            if meta.kind == "enum":
                lits = " ".join(sorted(meta.literals))
                decl = f"(declare-datatypes () (({name} {lits})))"
            else:
                decl = f"(declare-sort {name} 0)"

            if include_comments and meta.comment:
                decl += f" ; {meta.comment}"
            lines.append(decl)

        # 3) consts & functions ---------------------------------------
        for sym, meta in sorted(regs.consts.items()):
            if meta.arity == 0:
                decl = f"(declare-const {sym} {meta.sort})"
            else:
                arg_sorts = " ".join([meta.sort] * meta.arity)
                decl = f"(declare-fun {sym} ({arg_sorts}) {meta.sort})"

            if include_comments and meta.comment:
                decl += f" ; {meta.comment}"
            lines.append(decl)

        # 4) assertions -----------------------------------------------
        for tag, a in sorted(regs.assertions.items()):
            if pretty_assert:
                lines.append(textwrap.indent(a.text, "  ").lstrip())
            else:
                lines.append(a.text)

        return "(set-logic ALL)\n(set-option :produce-unsat-cores true)\n" +"\n".join(lines) + "\n"

    # ───────────────── incremental diff / slice ─────────────────────────
    @staticmethod
    def diff(old: Registries, new: Registries, **kwargs) -> str:
        """
        Return an SMT slice containing only declarations / assertions that are
        present in `new` but not in `old`.  Comments are *not* diffed.
        Additional kwargs are forwarded to `to_smt` (e.g., pretty_assert=False).
        """
        slice_regs = Registries()

        # sorts: fresh sorts or enum-literal extensions
        for s, meta in new.sorts.items():
            if s not in old.sorts:
                slice_regs.sorts[s] = meta
            elif meta.kind == "enum":
                delta = meta.literals - old.sorts[s].literals
                if delta:
                    merged = SortMeta(kind="enum",
                                      literals=old.sorts[s].literals | delta,
                                      comment=meta.comment)
                    slice_regs.sorts[s] = merged

        # new consts / funs
        for c, meta in new.consts.items():
            if c not in old.consts:
                slice_regs.consts[c] = meta

        # new assertions (by tag)
        for tag, a in new.assertions.items():
            if tag not in old.assertions:
                slice_regs.assertions[tag] = a

        return RegistryDecoder.to_smt(slice_regs,
                                      header="incremental slice",
                                      include_comments=False,
                                      **kwargs)
