# test_registry_codec.py
# ───────────────────────────────────────────────────────────────
import textwrap, random, string

from RegistryBuilder import RegistryBuilder      # local modules
from RegistryDecoder import RegistryDecoder


# ── helpers ─────────────────────────────────────────────────────
def _d(src: str) -> str:
    """Dedent a triple-quoted snippet and ensure exactly one trailing NL."""
    return textwrap.dedent(src).strip() + "\n"


def _eq_no_comments(r1, r2):
    """
    Compare two Registries *ignoring* the `.comments` list.  This lets us
    verify logical round-trips while tolerating benign comment re-ordering.
    """
    return (
        r1.sorts, r1.consts, r1.assertions
    ) == (
        r2.sorts, r2.consts, r2.assertions
    )


# ── 1 · basic smoke ─────────────────────────────────────────────
def test_round_trip_basic():
    prog = _d("""
        (declare-datatypes () ((diagnosis adenocarcinoma_bladder)))
        (declare-const age Int)
        (assert (! (> age 18) :named R1_1))
    """)
    regs = RegistryBuilder.scan(prog)
    rebuilt = RegistryDecoder.to_smt(regs)
    assert _eq_no_comments(RegistryBuilder.scan(rebuilt), regs)


# ── 2 · enum extension & diff  ─────────────────────────────────
def test_enum_extension_slice():
    slice1 = _d("""
        (declare-datatypes () ((color red blue)))
        (declare-const x color)
        (assert (! (= x red) :named R1_1))
    """)
    slice2 = _d("""
        (assert (! (= x green) :named R2_1))
        (declare-datatypes () ((color red blue green)))
    """)
    regs1 = RegistryBuilder.scan(slice1)
    regs2 = RegistryBuilder.scan(slice1 + slice2)

    delta = RegistryDecoder.diff(regs1, regs2)
    assert "(declare-datatypes" in delta and "green" in delta
    assert "(assert (! (= x green)" in delta


# ── 3 · function declaration parse ─────────────────────────────
def test_function_declaration():
    prog = _d("""
        (declare-sort Foo 0)
        (declare-fun score (Int Foo) Real) ; demo fun
        (assert (! (> (score 3 a) 0.5) :named R1_1))
    """)
    fn = RegistryBuilder.scan(prog).consts["score"]
    assert fn.arity == 2 and fn.sort == "Real"


# ── 4 · multiline enum parse ───────────────────────────────────
def test_multiline_enum_parse():
    prog = _d("""
        (declare-datatypes ()
          ((shape
              circle
              square
              triangle))) ; shapes
    """)
    regs = RegistryBuilder.scan(prog)
    assert regs.sorts["shape"].literals == {"circle", "square", "triangle"}
    assert regs.sorts["shape"].comment == "shapes"


# ── 5 · nested-paren assertion extractor ───────────────────────
def test_balanced_paren_assert_extractor():
    prog = _d("""
        (assert (! (and (> age 18)
                        (or (= sex male) (= sex female)))
                 :named R1_1)) ; demo
    """)
    a_text = next(iter(RegistryBuilder.scan(prog).assertions.values())).text
    body = a_text.split(";", 1)[0].rstrip()      # drop trailing comment
    assert a_text.count("(or") == 1 and body.endswith("))")


# ── 6 · diff only new items ────────────────────────────────────
def test_diff_only_new_items():
    base = _d("""
        (declare-const a Int)
        (assert (! (> a 0) :named R1_1))
    """)
    extended = _d("""
        (declare-const a Int)
        (assert (! (> a 0) :named R1_1))
        (declare-const b Int) ; fresh
        (assert (! (< b 0) :named R2_1))
    """)
    delta = RegistryDecoder.diff(RegistryBuilder.scan(base),
                                 RegistryBuilder.scan(extended))
    assert "declare-const b" in delta
    assert "(assert (! (< b 0)" in delta
    assert "declare-const a" not in delta


# ── 7 · idempotent re-encode ───────────────────────────────────
def test_reencode_program_is_stable():
    prog = _d("""
        ;; demo banner
        (declare-datatypes () ((d t1 t2 t3)))
        (declare-const c d)
        (declare-fun f (d Int) Bool)
        (assert (! (not (f c 5)) :named R1_1))
    """)
    s1 = RegistryDecoder.to_smt(RegistryBuilder.scan(prog))
    s2 = RegistryDecoder.to_smt(RegistryBuilder.scan(s1))
    assert s1 == s2


# ═══════════════════════════════════════════════════════════════
# 1 · Large mixed program round-trip
# ═══════════════════════════════════════════════════════════════
def test_round_trip_large_program():
    prog = _d("""
        (set-logic ALL)

        ;; big enum
        (declare-datatypes () ((stage I II III IV V)))

        ;; consts & funs
        (declare-const current_stage stage)
        (declare-const  psa          Real)
        (declare-fun   risk_score    (Real) Int)

        ;; assertions (3 requirements, multiple tags per req)
        (assert (! (= current_stage IV)         :named R1_1))
        (assert (! (> psa 10.0)                 :named R1_2))
        (assert (! (= (risk_score psa) 3)       :named R2_1))
        (assert (! (>= psa 5.0)                 :named R3_1))
        (assert (! (<= psa 30.0)                :named R3_2))
    """)
    regs = RegistryBuilder.scan(prog)
    rebuilt = RegistryDecoder.to_smt(regs)
    assert _eq_no_comments(RegistryBuilder.scan(rebuilt), regs)


# ═══════════════════════════════════════════════════════════════
# 2 · Enum grows twice; diff detects each growth exactly once
# ═══════════════════════════════════════════════════════════════
def test_enum_double_extension_diff():
    base = _d("""
        (declare-datatypes () ((color red blue)))
        (declare-const choice color)
        (assert (! (= choice red) :named R1_1))
    """)
    regs0 = RegistryBuilder.scan(base)

    slice1 = _d("""
        (declare-datatypes () ((color red blue green)))
        (assert (! (= choice green) :named R2_1))
    """)
    regs1 = RegistryBuilder.scan(base + slice1)
    delta1 = RegistryDecoder.diff(regs0, regs1)
    assert delta1.count("green") >= 2    # enum + assert

    slice2 = _d("""
        (declare-datatypes () ((color red blue green yellow)))
        (assert (! (= choice yellow) :named R3_1))
    """)
    regs2 = RegistryBuilder.scan(base + slice1 + slice2)
    delta2 = RegistryDecoder.diff(regs1, regs2)
    assert "yellow" in delta2 and delta2.count("(= choice yellow)") == 1
    assert delta2.count("(= choice green)") == 0


# ═══════════════════════════════════════════════════════════════
# 3 · 100 auto-generated tagged assertions survive round-trip
# ═══════════════════════════════════════════════════════════════
def _rand_sym(n=6):
    return "".join(random.choices(string.ascii_lowercase, k=n))

def test_many_assertions_round_trip():
    lines = ["(set-logic ALL)"]
    for req in range(1, 11):
        for k in range(1, 11):
            sym = _rand_sym()
            lines.append(f"(declare-const {sym} Bool)")
            lines.append(f"(assert (! {sym} :named R{req}_{k}))")
    prog = "\n".join(lines)

    regs = RegistryBuilder.scan(prog)
    rebuilt = RegistryDecoder.to_smt(regs)
    assert RegistryBuilder.scan(rebuilt).assertions.keys() == regs.assertions.keys()


# ═══════════════════════════════════════════════════════════════
# 4 · Mix of tagged and UNTAGGED assertions
# ═══════════════════════════════════════════════════════════════
def test_tag_fallback_and_lookup():
    prog = _d("""
        (declare-const a Bool)
        (declare-const b Bool)
        (assert (! a :named R1_1))
        (assert b)                      ;; untagged
    """)
    regs = RegistryBuilder.scan(prog)
    tagged = set(regs.assertions)
    assert "R1_1" in tagged
    untags = [t for t in tagged if t.startswith("U")]
    assert len(untags) == 1
    smt = RegistryDecoder.to_smt(regs)
    assert "(assert b)" in smt


# ═══════════════════════════════════════════════════════════════
# 5 · Comment preservation round-trip
# ═══════════════════════════════════════════════════════════════
def test_comment_round_trip():
    prog = _d("""
        ; header line
        (declare-const x Int) ; var
        (assert (! (> x 0) :named R1_1)) ; positive
    """)
    regs = RegistryBuilder.scan(prog)
    assert len(regs.comments) == 1                       # stand-alone only
    assert regs.consts["x"].comment == "var"
    rebuilt = RegistryDecoder.to_smt(regs)
    assert "; header line" in rebuilt
    assert "; var" in rebuilt and "; positive" in rebuilt
    assert _eq_no_comments(RegistryBuilder.scan(rebuilt), regs)
