# tests/test_registry_refiner.py
# ------------------------------
"""
Unit-tests for modules.SMTIncrementalSolverBasedRegistryRefiner
Run with:  pytest -q
"""

from textwrap import dedent
import json
import pytest
import importlib

# module under test
mod = importlib.import_module(
    "modules.SMTProgrammer.SMTIncrementalProgrammer."
    "SMTIncrementalSolverBasedRegistryRefiner"
)

# ────────────────────────── helpers / fixtures ──────────────────────────

SIMPLE_PROGRAM = dedent("""\
    (set-logic ALL)
    (declare-datatypes () ((color red blue)))
    (declare-const sky color)
    (assert (! (= sky blue) :named R1_0))
""")

NEW_SLICE = dedent("""\
    (declare-const grass color)
    (assert (! (= grass red) :named R2_0))
""")

@pytest.fixture
def regs():
    """Fresh Registries object from SIMPLE_PROGRAM + NEW_SLICE."""
    return mod.RegistryBuilder.scan(SIMPLE_PROGRAM + "\n" + NEW_SLICE)


class FakeEngine:
    """Mimics the LLM API: returns a fixed first-element list."""
    def __init__(self, reply: str):
        self.reply = reply
    def __call__(self, *_):
        return [self.reply]


def fake_context():
    return {
        "smt_program_lines": SIMPLE_PROGRAM.splitlines(),
        "new_smt_lines":     NEW_SLICE.splitlines(),
        "current_requirement_index": 42,
    }


FAKE_SOLVER_RESULT = {
    "status":     "unsat",
    "message":    "",
    "unsat_core": ["R2_0"],
}

# ─────────────────────── _extract_json tests ───────────────────────────

def test_extract_json_single_object():
    txt = '{"edits":[]}'
    res = mod._extract_json(txt)
    assert isinstance(res, list)
    assert len(res) in (0, 1)
    if res:
        assert res[0] == json.loads(txt)

def test_extract_json_list():
    txt = '[{"edits":[]},{"edits":[]}]'
    assert mod._extract_json(txt) == json.loads(txt)

def test_extract_json_invalid():
    assert mod._extract_json("no-json-here") is None


# ────────────────── edit fixtures (reflect new semantics) ──────────────
# NB: extending an existing enum is now done with **replace**, not add.

ENUM_EXTEND = {                       # formerly ENUM_ADD
    "op": "replace",
    "target": {"kind": "enum", "name": "color"},
    "text": "(declare-datatypes () ((color red blue green)))",
}
ENUM_REPLACE = {
    "op": "replace",
    "target": {"kind": "enum", "name": "color"},
    "text": "(declare-datatypes () ((color red green)))",
}
ENUM_REMOVE = {"op": "remove", "target": {"kind": "enum", "name": "color"}}

CONST_ADD = {
    "op": "add",
    "target": {"kind": "const", "name": "sea"},
    "text": "(declare-const sea color)",
}

ASSERT_ADD = {
    "op": "add",
    "target": {"kind": "assertion", "assert_tag": "R3_0"},
    "text":  '(assert (! (= sea red) :named R3_0))',
}
ASSERT_REPLACE = {
    "op": "replace",
    "target": {"kind": "assertion", "assert_tag": "R1_0"},
    "text": '(assert (! (= sky red) :named R1_0))',
}
ASSERT_REMOVE = {
    "op": "remove",
    "target": {"kind": "assertion", "assert_tag": "R2_0"},
}


# ────────────────── tests for _apply_single_edit ───────────────────────

@pytest.mark.parametrize(
    "edit, predicate",
    [
        (ENUM_EXTEND,  lambda r: "green" in r.sorts["color"].literals),
        (ENUM_REPLACE, lambda r: "blue" not in r.sorts["color"].literals),
        (ENUM_REMOVE,  lambda r: "color" not in r.sorts),
        (CONST_ADD,    lambda r: "sea"   in r.consts),
        (ASSERT_ADD,   lambda r: "R3_0"  in r.assertions),
        (ASSERT_REPLACE, lambda r: "sky red" in r.assertions["R1_0"].text),
        (ASSERT_REMOVE,  lambda r: "R2_0" not in r.assertions),
    ],
)
def test_apply_single_edit_success(regs, edit, predicate):
    mod._apply_single_edit(regs, edit)
    assert predicate(regs)


def test_apply_single_edit_duplicate_assert_tag_raises(regs):
    dup_add = {
        "op":     "add",
        "target": {"kind": "assertion", "assert_tag": "R1_0"},
        "text":   '(assert (! (= sky red) :named R1_0))',
    }
    with pytest.raises(ValueError, match="already exists"):
        mod._apply_single_edit(regs, dup_add)


def test_apply_single_edit_missing_symbol_raises(regs):
    bad_enum = {
        "op": "replace",
        "target": {"kind": "enum", "name": "color"},
        "text": "(declare-datatypes () ((shape circle square)))",
    }
    with pytest.raises(ValueError, match="color"):
        mod._apply_single_edit(regs, bad_enum)


# ───────────────────────── forward() integration tests ─────────────────

def test_forward_success_updates_context():
    engine_json = json.dumps(
        [{
            "edits": [ENUM_EXTEND, ASSERT_REMOVE],
            "comment": "added green; removed bad assertion",
        }]
    )
    refiner = mod.SMTIncrementalSolverBasedRegistryRefiner(
        engine=FakeEngine(engine_json)
    )
    ctx  = fake_context()
    out  = refiner.forward(ctx, FAKE_SOLVER_RESULT)

    assert "R2_0" not in out["smt_program_lines"]        # assertion removed
    assert any("green" in ln for ln in out["smt_program_lines"])
    assert out["registry_refiner_comment"] == "added green; removed bad assertion"
    assert "refiner_failures" not in out


def test_forward_json_parse_failure_records_error():
    refiner = mod.SMTIncrementalSolverBasedRegistryRefiner(
        engine=FakeEngine("this is not json")
    )
    ctx = fake_context()
    out = refiner.forward(ctx, FAKE_SOLVER_RESULT)
    assert out["refiner_failures"][-1]["reason"] == "json_parse_fail"


def test_forward_apply_edit_error_records_failure():
    bogus_edit = json.dumps([{
        "edits": [{"op": "explode", "target": {"kind": "enum", "name": "color"}}]
    }])
    refiner = mod.SMTIncrementalSolverBasedRegistryRefiner(
        engine=FakeEngine(bogus_edit)
    )
    ctx = fake_context()
    out = refiner.forward(ctx, FAKE_SOLVER_RESULT)
    failure = out["refiner_failures"][-1]
    assert failure["reason"].startswith("apply_edit_error")
