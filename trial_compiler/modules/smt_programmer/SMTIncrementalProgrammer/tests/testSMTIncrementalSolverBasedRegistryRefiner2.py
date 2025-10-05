# tests/test_registry_refiner_comment_invariance.py
# -------------------------------------------------
"""
Guarantees that the refiner never changes or deletes comments that were
present in the incoming SMT code (validated + new slice), unless an LLM edit
explicitly rewrites those very comment lines.

We compare the set of *comment tokens* (`; …`) before and after the refiner
runs.  Any original comment missing or modified causes the test to fail.
"""

import json, re, difflib, importlib, textwrap, pytest

# ---------------------------------------------------------------------------
# module under test
# ---------------------------------------------------------------------------
mod = importlib.import_module(
    "modules.SMTProgrammer.SMTIncrementalProgrammer."
    "SMTIncrementalSolverBasedRegistryRefiner"
)

# ---------------------------------------------------------------------------
# tiny stubs / helpers
# ---------------------------------------------------------------------------
class DummyLLM:
    """Returns fixed text exactly like the real engine (a one-element list)."""
    def __init__(self, reply): self.reply = reply
    def __call__(self, *_):    return [self.reply]

UNSAT = {"status": "unsat", "message": "", "unsat_core": ["R_bad"]}

_COMMENT_RE = re.compile(r"\s*;.*")     # matches “; …” including leading space


def _comment_tokens(text: str) -> set[str]:
    """Return every comment substring (inline OR stand-alone) in `text`."""
    toks = set()
    for line in text.splitlines():
        if ";" in line:
            toks.add(line[line.index(";"):].rstrip())
    return toks


def _split_code_comment(line: str):
    """Return (code_part, comment_part) where comment_part starts with ';'
    or '' if no comment.  Leading whitespace on the code_part is kept so
    comparison is stable."""
    if ";" not in line:
        return line.rstrip("\n"), ""
    pos = line.index(";")
    return line[:pos].rstrip(), line[pos:].rstrip()


def assert_comments_unchanged(before: str, after: str) -> None:
    """Fail if any comment was modified *or* a stand-alone comment vanished."""
    before_lines = before.splitlines()
    after_lines  = after.splitlines()

    # Build quick lookup maps for after-state
    after_set             = set(after_lines)
    after_code_to_comment = {}
    for ln in after_lines:
        code, com = _split_code_comment(ln)
        if code:
            after_code_to_comment[code] = com    # last one wins – fine

    violations = []

    for ln in before_lines:
        code, com = _split_code_comment(ln)

        # ── stand-alone comment line ──────────────────────────────────────
        if not code:               # entire line is comment
            if ln not in after_set:
                violations.append(ln)
            continue

        # ── code line with (possibly empty) trailing comment ─────────────
        if not com:                # no comment on that line → nothing to verify
            continue

        # If the code part survived, its comment must be verbatim identical.
        if code in after_code_to_comment and after_code_to_comment[code] != com:
            violations.append(f"{com}  (on line: {code})")

    assert not violations, (
        "Comment(s) lost or modified:\n" +
        "\n".join(violations) +
        "\n\nUnified diff for context:\n" +
        "\n".join(difflib.unified_diff(
            before_lines, after_lines,
            fromfile="before", tofile="after", lineterm=""))
    )

# ---------------------------------------------------------------------------
# parameterised scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    # ──────────────────────────────────────────────────────────────────
    # 1) Remove a failing assertion; comments must remain identical.
    # ──────────────────────────────────────────────────────────────────
    dict(
        validated=textwrap.dedent("""\
            (set-logic ALL)
            ; banner
            (declare-const x Int) ; variable x
            (assert (! (= x 0) :named R_ok))
        """),
        new_slice="(assert (! (> x 0) :named R_bad)) ; failing",
        llm_edit=json.dumps(
            [{"edits":[
                {"op":"remove",
                 "target":{"kind":"assertion","assert_tag":"R_bad"}}
            ]}]
        )
    ),
    # ──────────────────────────────────────────────────────────────────
    # 2) Replace an enum but preserve inline and stand-alone comments.
    # ──────────────────────────────────────────────────────────────────
    dict(
        validated="(declare-datatypes () ((flavour sweet salty))) ; flavours\n",
        new_slice="(assert (! (= taste spicy) :named R_bad)) ; needs enum\n",
        llm_edit=json.dumps(
            [{"edits":[
                {"op":"replace",
                 "target":{"kind":"enum","name":"flavour"},
                 "text":"(declare-datatypes () "
                         "((flavour sweet salty spicy))) ; flavours"}
            ]}]
        )
    ),
]



# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", SCENARIOS, ids=["remove_assertion", "replace_enum"])
def test_comments_preserved(case):
    ctx = {
        "smt_program_lines": case["validated"].splitlines(),
        "new_smt_lines":     [case["new_slice"]],
        "current_requirement_index": 123,
    }
    refiner = mod.SMTIncrementalSolverBasedRegistryRefiner(
        DummyLLM(case["llm_edit"])
    )
    out_ctx = refiner.forward(ctx, UNSAT)
    rebuilt = "\n".join(out_ctx["smt_program_lines"])
    # Ensure every original comment token survived *unchanged*.
    assert_comments_unchanged(case["validated"] + case["new_slice"], rebuilt)