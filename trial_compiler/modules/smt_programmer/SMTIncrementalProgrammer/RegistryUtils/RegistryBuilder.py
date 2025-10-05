# registry_builder.py  – comment-aware edition
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Set, List


# ────────────────────────────── data models ──────────────────────────────
@dataclass
class SortMeta:
    kind: str                        # "enum" or "uninterpreted"
    literals: Set[str] = field(default_factory=set)
    comment: str | None = None


@dataclass
class ConstMeta:
    sort: str                        # return sort
    arity: int                       # 0 = constant, >0 = function
    comment: str | None = None

@dataclass
class AssertionMeta:
    tag: str
    text: str                        # includes any internal/trailing comments

@dataclass
class CommentMeta:
    line_no: int
    text: str

@dataclass
class Registries:
    sorts: Dict[str, SortMeta]            = field(default_factory=dict)
    consts: Dict[str, ConstMeta]          = field(default_factory=dict)
    assertions: Dict[str, AssertionMeta]  = field(default_factory=dict)
    comments: List[CommentMeta]           = field(default_factory=list)


# ───────────────────────────── registry builder ──────────────────────────
class RegistryBuilder:
    """
    Regex-based scanner that keeps every semicolon comment it encounters.
    """

    # ........................ declarations .................................
    _ENUM_DECL_RE = re.compile(
        r"\(declare-datatypes\s*\(\)\s*\(\(\s*"
        r"(?P<sort>\w+)\s+(?P<lits>[^)]+?)\s*\)\)\)\s*(?:;(?P<cmt>[^\n]*))?",
        re.DOTALL)

    _SORT_DECL_RE = re.compile(
        r"\(declare-sort\s+(?P<sort>\w+)\s+0\)\s*(?:;(?P<cmt>[^\n]*))?")

    _CONST_DECL_RE = re.compile(
        r"\(declare-const\s+(?P<sym>\w+)\s+(?P<sort>\w+)\)\s*(?:;(?P<cmt>[^\n]*))?")

    _FUN_DECL_RE = re.compile(
        r"\(declare-fun\s+(?P<sym>\w+)\s*\(\s*(?P<args>[^\)]*)\)\s+"
        r"(?P<ret>\w+)\s*\)\s*(?:;(?P<cmt>[^\n]*))?")

    # ........................ assertions ...................................
    _ASSERT_START_RE = re.compile(r"\(assert\b")
    _TAG_RE = re.compile(r":named\s+(\w+)\b")

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def scan(cls, smt: str) -> Registries:
        regs = Registries()

        # first pass – pull out standalone comments so they don't confuse regexes
        code_lines: List[str] = []
        for idx, raw in enumerate(smt.splitlines(), start=1):
            if raw.lstrip().startswith(";"):                 # pure comment
                regs.comments.append(CommentMeta(idx, raw.lstrip()[1:].strip()))
                code_lines.append("")                         # keep line count
            else:
                code_lines.append(raw)

        code = "\n".join(code_lines)   # code w/ inline comments intact

        # 1) sorts ----------------------------------------------------
        for m in cls._ENUM_DECL_RE.finditer(code):
            meta = regs.sorts.setdefault(m.group("sort"),
                                          SortMeta(kind="enum"))
            meta.literals.update(m.group("lits").split())
            meta.comment = (m.group("cmt") or "").strip()

        for m in cls._SORT_DECL_RE.finditer(code):
            regs.sorts.setdefault(
                m.group("sort"),
                SortMeta(
                    kind="uninterpreted",
                    comment=(m.group("cmt") or "").strip()
                )
            )

        # 2) consts & funs -------------------------------------------
        for m in cls._CONST_DECL_RE.finditer(code):
            regs.consts[m.group("sym")] = ConstMeta(
                sort=m.group("sort"),
                arity=0,
                comment=(m.group("cmt") or "").strip()
            )

        for m in cls._FUN_DECL_RE.finditer(code):
            arity = 0 if not m.group("args").strip() else len(m.group("args").split())
            regs.consts[m.group("sym")] = ConstMeta(
                sort=m.group("ret"),
                arity=arity,
                comment=(m.group("cmt") or "").strip()
            )

        # 3) assertions ----------------------------------------------
        for text in cls._extract_assertions(code):
            tag_match = cls._TAG_RE.search(text)
            tag = tag_match.group(1) if tag_match else f"U{len(regs.assertions)+1}"
            regs.assertions[tag] = AssertionMeta(tag=tag, text=text)

        # optional handy index (1-based)
        regs.assertions_by_idx = {i: a for i, a in
                                  enumerate(regs.assertions.values(), start=1)}
        return regs

    # ───────────────────── helpers ───────────────────────────────────
    @classmethod
    def _extract_assertions(cls, code: str) -> List[str]:
        """
        Returns a list of full '(assert …)' strings, **including** any
        comments that appear inside or after the top-level form.
        """
        out, pos = [], 0
        while True:
            m = cls._ASSERT_START_RE.search(code, pos)
            if not m:
                break
            start = m.start()
            depth, i = 0, start
            while i < len(code):
                if code[i] == '(':
                    depth += 1
                elif code[i] == ')':
                    depth -= 1
                    if depth == 0:
                        # grab through end-of-line so trailing comment stays
                        line_end = code.find("\n", i)
                        if line_end == -1:
                            line_end = len(code)
                        out.append(code[start:line_end].rstrip())
                        pos = line_end
                        break
                i += 1
            else:
                break
        return out
