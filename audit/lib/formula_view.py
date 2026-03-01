"""Formula + facts + conflict view.

For each REQ, show:
  - the trial's logical formula (pretty-printed SMT-LIB)
  - the patient facts bound to the variables in that formula
  - a conflict marker when the REQ is unsat

This is the most direct rendering of AEGIS's reasoning: no prose
verbalization, just the rule, the facts, and where they disagree.
"""
from __future__ import annotations
import html, re
from typing import Any


# --- 1. Parse REQ blocks + leading comment from smt_program_lines ----------

def _extract_req_blocks(smt_lines: list[str]) -> list[dict]:
    """Return list of {name, body, comment, body_start} for each (assert ... :named REQ...)."""
    src = "\n".join(smt_lines)
    blocks = []
    i = 0
    while i < len(src):
        j = src.find("(assert", i)
        if j < 0:
            break
        depth, k = 0, j
        while k < len(src):
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        block = src[j : k + 1]
        m = re.search(r":named\s+(REQ\w+)", block)
        if m:
            # Scan backwards from j for the most recent contiguous `;;` comment run
            pre = src[:j].rstrip()
            comment_lines = []
            lines = pre.split("\n")
            for ln in reversed(lines):
                s = ln.strip()
                if s.startswith(";;"):
                    txt = s.lstrip(";").strip()
                    # skip separator banners
                    if not txt or txt.startswith("=="):
                        if comment_lines:
                            break
                        continue
                    comment_lines.insert(0, txt)
                elif not s:
                    if comment_lines:
                        break
                else:
                    break
            blocks.append(
                {
                    "name": m.group(1),
                    "body": block,
                    "comment": " ".join(comment_lines),
                }
            )
        i = k + 1
    return blocks


# --- 2. Extract the formula core (strip assert/! wrapper and :named tag) ---

def _strip_assert(body: str) -> str:
    # `(assert (! <core> :named FOO))`  ->  `<core>`
    m = re.search(r"\(assert\s+\(!\s*(.*)\s+:named\s+\w+\s*\)\s*\)\s*$", body, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: just strip outer (assert ...)
    return body.strip()


# --- 3. Pretty-print formula with compact operators ------------------------

_OP_SUB = [
    (r"\band\b", "∧"),
    (r"\bor\b", "∨"),
    (r"\bnot\b", "¬"),
    (r"\bimplies\b", "⇒"),
]


def _compact_formula(core: str) -> str:
    """Render SMT-LIB s-expr in a slightly more math-y form, with pretty-printing
    so that sub-clauses of top-level (and ...) / (or ...) land on their own lines.
    """
    s = core
    s = re.sub(r"\(\s*>=\s", "(≥ ", s)
    s = re.sub(r"\(\s*<=\s", "(≤ ", s)
    s = re.sub(r"\(\s*=\s", "(= ", s)
    for pat, sym in _OP_SUB:
        s = re.sub(r"\(\s*" + pat[2:-2] + r"\s", "(" + sym + " ", s)
    s = re.sub(r"(\d)\.0\b", r"\1", s)
    s = _simplify_common(s)
    s = _pretty_sexp(s)
    s = _infix_comparisons(s)
    # Strip top-level parens for single-predicate formulas — reads cleaner.
    if s.startswith("(") and s.endswith(")"):
        # Only strip if these parens are truly the outermost pair
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
    return s


def _infix_comparisons(s: str) -> str:
    """Rewrite `(> X N)` / `(≥ X N)` / `(<= X N)` / `(= X N)` to `X > N` etc.
    Only when X is a single bare variable (no nested parens)."""
    ops = ["≥", "≤", ">", "<", "="]
    # Iterate to fixpoint
    prev = None
    while prev != s:
        prev = s
        for op in ops:
            pat = re.compile(
                r"\(\s*" + re.escape(op) + r"\s+([^()\s]+(?:\s+\([^()]*\))?)\s+([^()\s]+)\s*\)"
            )
            s = pat.sub(lambda m: f"{m.group(1)} {op} {m.group(2)}", s)
    return s


def _simplify_common(s: str) -> str:
    """Collapse common idioms to more natural math notation:
      (¬ (< X N))  →  (≥ X N)
      (¬ (> X N))  →  (≤ X N)
      (¬ (≤ X N))  →  (> X N)
      (¬ (≥ X N))  →  (< X N)
      (∧ (≥ X a) (≤ X b))  →  (∈ X [a, b])
    Pure textual — safe because these operator tokens are unambiguous at this stage.
    """
    # negation-of-comparison
    pairs = [("<", "≥"), (">", "≤"), ("≤", ">"), ("≥", "<")]
    for op, inv in pairs:
        pat = re.compile(r"\(¬\s+\(" + re.escape(op) + r"\s+([^()]+?)\)\s*\)")
        s = pat.sub(lambda m: f"({inv} {m.group(1)})", s)
    # interval: (∧ (≥ X a) (≤ X b))  — same variable
    iv = re.compile(
        r"\(∧\s+\(≥\s+([^()]+?)\s+([0-9.\-]+)\)\s+\(≤\s+\1\s+([0-9.\-]+)\)\s*\)"
    )
    s = iv.sub(lambda m: f"({m.group(1)} ∈ [{m.group(2)}, {m.group(3)}])", s)
    return s


def _split_top_children(s: str) -> tuple[str | None, list[str]]:
    """`(op a b c)` → ('op', ['a','b','c'])  |  leaf → (None, [])"""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None, []
    depth, buf, parts = 0, "", []
    head = None
    for ch in s[1:]:
        if head is None:
            if ch == " ":
                head = buf
                buf = ""
                continue
            buf += ch
            continue
        if ch == "(":
            depth += 1
            buf += ch
        elif ch == ")":
            if depth == 0:
                if buf.strip():
                    parts.append(buf.strip())
                break
            depth -= 1
            buf += ch
        elif ch == " " and depth == 0:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    return head, parts


def _pretty_sexp(s: str, indent: int = 2, inline_threshold: int = 85) -> str:
    """Render an s-expression readably:
      - leaf or non-connective: return as-is
      - (∧ a b) / (∨ a b) with ≤3 simple children AND total inline width small:
            render as `a  AND  b  AND  c` (no surrounding parens)
      - else: break onto lines with word-ish connectives
    """
    s = s.strip()
    head, parts = _split_top_children(s)
    if head not in ("∧", "∨") or len(parts) <= 1:
        return s
    sentinel = "@@AND@@" if head == "∧" else "@@OR@@"
    word = "and" if head == "∧" else "or"
    rendered_parts = [_pretty_sexp(p, indent + 2, inline_threshold) for p in parts]
    inline_plain = f"  {word}  ".join(rendered_parts)
    if "\n" not in inline_plain and len(inline_plain) <= inline_threshold:
        return f"  {sentinel}  ".join(rendered_parts)
    pad = " " * indent
    joined = f"\n{pad}{sentinel} ".join(
        p.replace("\n", "\n" + pad) for p in rendered_parts
    )
    return joined


# --- 4. Shorten variable names for display ---------------------------------

def _shorten_var(v: str) -> str:
    """Human-readable variable name for display."""
    s = v
    s = re.sub(r"^patient_", "", s)
    # unit handling: `_withunit_mm_hg` -> ` (mmHg)`
    s = re.sub(r"_withunit_mm_hg\b", " (mmHg)", s)
    s = re.sub(r"_withunit_percent\b", " (%)", s)
    s = re.sub(r"_withunit_ml_per_min\b", " (mL/min)", s)
    s = re.sub(r"_withunit_mmhg\b", " (mmHg)", s)
    s = re.sub(r"_withunit_years\b", " (years)", s)
    s = re.sub(r"_withunit_kg\b", " (kg)", s)
    s = re.sub(r"_withunit_cm\b", " (cm)", s)
    s = re.sub(r"_withunit_months\b", " (months)", s)
    s = re.sub(r"_withunit_days\b", " (days)", s)
    # age wrappers
    s = re.sub(r"_value_recorded_now_in_years\b", " (years)", s)
    s = re.sub(r"_value_recorded_now\b", "", s)
    s = re.sub(r"_value\b", "", s)
    s = re.sub(r"_recorded_now\b", "", s)
    s = re.sub(r"_now$", "", s)
    s = re.sub(r"_inthehistory\b", " (history)", s)
    s = re.sub(r"^has_diagnosis_of_", "has diagnosis of ", s)
    s = re.sub(r"^has_finding_of_", "has finding of ", s)
    s = re.sub(r"^has_history_of_", "has history of ", s)
    s = re.sub(r"^has_", "has ", s)
    s = re.sub(r"^is_", "is ", s)
    s = s.replace("@@", " — ")
    return s.replace("_", " ")


# --- 5. Collect variables referenced in the formula core -------------------

_VAR_RE = re.compile(r"\bpatient_[a-zA-Z0-9_@]+\b")


def _vars_in(core: str) -> list[str]:
    seen = []
    for m in _VAR_RE.finditer(core):
        v = m.group(0)
        if v not in seen:
            seen.append(v)
    return seen


# --- 6. Render one REQ as an HTML row --------------------------------------

_STATUS_STYLE = {
    "sat": ("SATISFIED", "#16a34a", "#ecfdf5"),
    "unsat": ("VIOLATED", "#dc2626", "#fef2f2"),
    "unknown": ("UNKNOWN", "#f59e0b", "#fffbeb"),
}


def _evidence_html(fact: dict | None) -> str:
    if not fact:
        return ""
    evidence = (fact.get("evidence") or "").strip()
    if evidence and fact.get("variable_mentioned_in_patient_note", "").lower().startswith("y"):
        return (
            f'<div style="color:#6b7280;font-size:0.82em;margin-top:4px;padding:4px 8px;'
            f'background:#f9fafb;border-left:2px solid #e5e7eb;border-radius:2px">'
            f'chart: “{html.escape(evidence[:200])}”</div>'
        )
    return ""


def _atom_status(var: str, fact: dict | None) -> str:
    """Return 'sat', 'unsat', 'unknown', or 'na' for a bool-atom assertion.
    Used when an atom appears alone (not inside a comparison) in a conjunction.
    """
    if fact is None:
        return "na"
    v = fact.get("value")
    if v is None:
        return "unknown"
    if v is True:
        return "sat"
    if v is False:
        return "unsat"
    return "na"


def _inline_patient_values(formula_html: str, patient_vars: dict) -> str:
    """After rendering the formula with _hl (which wraps vars in <code>…</code>),
    append the patient's value next to each variable occurrence.
    The <code> spans now contain the shortened name; map back via patient_vars keys.
    """
    # Build shortened-name -> (original var, fact) map
    short_to_orig = {}
    for orig, fact in (patient_vars or {}).items():
        short_to_orig[_shorten_var(orig)] = (orig, fact)

    def _pill(orig, val, label_hint="click"):
        dv = html.escape(orig)
        if val is None:
            txt, color, bg = "unknown", "#6b7280", "#f3f4f6"
        elif val is True:
            txt, color, bg = "yes", "#065f46", "#d1fae5"
        elif val is False:
            txt, color, bg = "no", "#7f1d1d", "#fee2e2"
        else:
            txt, color, bg = html.escape(str(val)), "#1e3a8a", "#dbeafe"
        return (
            f'<span class="aegis-val" data-var="{dv}" data-format="pill" '
            f'title="Click to change" '
            f'style="display:inline-block;margin-left:4px;padding:1px 8px;border-radius:10px;'
            f'background:{bg};color:{color};font-weight:600;font-size:0.82em;cursor:pointer;'
            f'border:1px solid transparent">{txt}</span>'
        )

    def _repl(m):
        content = m.group(1)
        if content not in short_to_orig:
            return m.group(0)
        orig, fact = short_to_orig[content]
        if not fact:
            return m.group(0)
        return m.group(0) + _pill(orig, fact.get("value"))

    # Target the <code>…</code> spans produced by _hl
    return re.sub(
        r'<code style="color:#374151;background:#f3f4f6;padding:0 3px;border-radius:2px">([^<]+)</code>',
        _repl,
        formula_html,
    )


_STATUS_ICON = {"sat": "✓", "unsat": "✗", "unknown": "?"}


def _synthesize_title(core: str, side: str | None, patient_vars: dict) -> str:
    """Synthesize a human-readable title from a raw SMT-LIB s-expression.
    The result goes into the card header when no `;;` comment is present.
    """
    f = core.strip()
    head, parts = _split_top_children(f)
    def _phrase(v: str) -> str:
        """Turn a shortened var into a noun phrase: strip leading 'has ' / 'is '."""
        s = v
        s = re.sub(r"^has\s+", "", s)
        s = re.sub(r"^is\s+", "", s)
        return s

    # Leaf boolean variable: "Patient must have X" / "must not" for exclusion
    if head is None:
        var = _shorten_var(f)
        if side == "exclusion":
            return "Must not have " + html.escape(_phrase(var))
        return "Must have " + html.escape(_phrase(var))
    if head == "not" and len(parts) == 1:
        # exclusion-style: `(not P)` → "Must not have P"
        inner_head, inner_parts = _split_top_children(parts[0])
        if inner_head is None:
            return "Must not have " + html.escape(_phrase(_shorten_var(parts[0])))
        if inner_head in (">=", "<=", ">", "<") and len(inner_parts) == 2:
            op_inv = {">=": "&lt;", "<=": "&gt;", ">": "≤", "<": "≥"}
            return (
                f"{html.escape(_shorten_var(inner_parts[0]))} "
                f"{op_inv[inner_head]} {html.escape(inner_parts[1])}"
            )
        return "Must not: " + html.escape(parts[0])
    if head in (">=", "<=", ">", "<", "=") and len(parts) == 2:
        op_map = {">=": "≥", "<=": "≤", ">": "&gt;", "<": "&lt;", "=": "="}
        return f"{html.escape(_shorten_var(parts[0]))} {op_map[head]} {html.escape(parts[1])}"
    if head == "and":
        # pattern: (and (>= X a) (<= X b)) → "X between a and b"
        if len(parts) == 2:
            h1, p1 = _split_top_children(parts[0])
            h2, p2 = _split_top_children(parts[1])
            if h1 == ">=" and h2 == "<=" and len(p1) == 2 and len(p2) == 2 and p1[0] == p2[0]:
                return f"{html.escape(_shorten_var(p1[0]))} between {html.escape(p1[1])} and {html.escape(p2[1])}"
        leaves = [p.strip("()") for p in parts[:3]]
        extra = "" if len(parts) <= 3 else f" (+{len(parts)-3} more)"
        return "All of: " + ", ".join(html.escape(_shorten_var(l)) for l in leaves) + extra
    if head == "or":
        leaves = [p.strip("()") for p in parts[:3]]
        extra = "" if len(parts) <= 3 else f" (+{len(parts)-3} more)"
        return "At least one of: " + ", ".join(html.escape(_shorten_var(l)) for l in leaves) + extra
    # Unknown shape → truncated raw
    return html.escape(f[:120])


def render_req(
    req: dict, status: str, patient_vars: dict, side: str | None = None
) -> str:
    """Render one REQ as a compact one-row card."""
    core = _strip_assert(req["body"])
    formula = _compact_formula(core)
    vars_ = _vars_in(core)
    _, color, bg = _STATUS_STYLE.get(status, ("?", "#6b7280", "#f9fafb"))
    icon = _STATUS_ICON.get(status, "?")
    comment = req.get("comment") or ""

    # Wrap the formula nicely: break at top-level `(and`/`(or` if possible.
    # Start plain — just pre-wrap it.
    formula_html = html.escape(formula)
    # bold the operator glyphs
    for sym in ("∧", "∨", "¬", "≥", "≤", "⇒"):
        formula_html = formula_html.replace(
            sym, f'<b style="color:#7c3aed">{sym}</b>'
        )
    # highlight variable names
    def _hl(m):
        v = m.group(0)
        return f'<code style="color:#374151;background:#f3f4f6;padding:0 3px;border-radius:2px">{html.escape(_shorten_var(v))}</code>'

    formula_html = _VAR_RE.sub(_hl, html.escape(formula))
    formula_html = _inline_patient_values(formula_html, patient_vars)
    # replace sentinels with styled words
    formula_html = formula_html.replace(
        "@@AND@@", '<b style="color:#9ca3af;font-weight:500">AND</b>'
    ).replace(
        "@@OR@@", '<b style="color:#9ca3af;font-weight:500">OR</b>'
    )
    # Style remaining operator glyphs (if any stayed in s-expr form)
    formula_html = formula_html.replace(
        "¬ ", '<b style="color:#9ca3af;font-weight:500">NOT</b> '
    )
    # Clinician-friendly operator phrases (≥/≤ → 'at least' / 'at most')
    op_style = 'style="color:#9ca3af;font-weight:500"'
    formula_html = formula_html.replace(
        " ≥ ", f' <b {op_style}>at least</b> '
    ).replace(
        " ≤ ", f' <b {op_style}>at most</b> '
    ).replace(
        " &gt; ", f' <b {op_style}>more than</b> '
    ).replace(
        " &lt; ", f' <b {op_style}>less than</b> '
    ).replace(
        " > ", f' <b {op_style}>more than</b> '
    ).replace(
        " < ", f' <b {op_style}>less than</b> '
    ).replace(" ⇒ ", f' <b {op_style}>implies</b> ')
    # Interval operator
    formula_html = formula_html.replace(" ∈ ", f' <b {op_style}>in range</b> ')

    # Evidence lines, one per variable that has chart support
    evidence_html = "".join(_evidence_html(patient_vars.get(v)) for v in vars_)

    req_attr = f'data-req="{html.escape(req["name"])}"'
    side_tag = ""
    if side:
        side_tag = (
            f'<span style="display:inline-block;background:#f3f4f6;color:#6b7280;'
            f'padding:1px 7px;border-radius:3px;font-size:0.68em;font-weight:600;'
            f'letter-spacing:0.05em;margin-right:8px;vertical-align:1px">{side.upper()}</span>'
        )
    # Prefer the leading comment; else synthesize a title from the RAW core
    if comment:
        c = comment.rstrip(".").strip()
        # Strip internal "Component N:" / "Component N -" prefix the compiler inserts
        c = re.sub(r"^\s*Component\s+\d+\s*[:\-–]\s*", "", c)
        # Collapse wordy openings to a single "Must …"
        c = re.sub(
            r"^(?:To be included,?\s*)?(?:the\s+)?patient\s+must\s+",
            "Must ", c, count=1, flags=re.IGNORECASE,
        )
        c = re.sub(r"^Patient must\s+", "Must ", c, count=1, flags=re.IGNORECASE)
        c = re.sub(r"^The patient must\s+", "Must ", c, count=1, flags=re.IGNORECASE)
        # Lighter clinical-symbol swap inside comments: ≥/≤ to plain words
        c = c.replace("≥", "at least").replace("≤", "at most")
        title_text = html.escape(c)
    else:
        title_text = _synthesize_title(core, side, patient_vars)
    title = side_tag + title_text

    # One-row compact card (no redundant status badge — section + icon + border color carry it)
    row = f"""
      <div style="display:flex;gap:10px;align-items:flex-start">
        <div class="aegis-icon" style="color:{color};font-size:1.05em;font-weight:700;
                    flex-shrink:0;width:1.2em;text-align:center;line-height:1.5em">{icon}</div>
        <div style="flex:1;min-width:0">
          <div class="aegis-req-title" style="color:#111827;font-size:0.95em;line-height:1.4">{title}</div>
          <div style="margin:4px 0 0 0;padding:2px 0;font-size:0.88em;color:#6b7280;
                      line-height:1.55;white-space:pre-wrap;word-break:break-word">{formula_html}</div>
          {evidence_html}
        </div>
      </div>"""

    show_formula = True
    if status == "sat":
        summary = (
            f'<summary style="cursor:pointer;list-style:none;display:flex;gap:10px;align-items:center">'
            f'<span class="aegis-icon" style="color:{color};font-weight:700;width:1.2em;'
            f'text-align:center">{icon}</span>'
            f'<span style="flex:1;color:#6b7280;font-size:0.9em">{title}</span>'
            f'<span style="color:#9ca3af;font-size:0.8em">▸</span>'
            f'</summary>'
        )
        detail_body = (
            f'<div style="margin:0;padding:2px 0;font-size:0.88em;color:#6b7280;'
            f'line-height:1.55;white-space:pre-wrap;word-break:break-word">{formula_html}</div>'
            f'{evidence_html}' if show_formula else evidence_html
        )
        return f"""
        <details {req_attr} class="aegis-card"
            style="border:1px solid #e5e7eb;border-left:3px solid {color};
                   border-radius:4px;padding:8px 12px;margin:4px 0;background:white">
          {summary}
          <div style="margin-top:8px;padding-left:calc(1.3em + 10px)">
            {detail_body}
          </div>
        </details>"""
    return f"""
    <div {req_attr} class="aegis-card"
         style="border:1px solid #e5e7eb;border-left:3px solid {color};
                border-radius:4px;padding:10px 14px;margin:6px 0;background:white">
      {row}
    </div>"""


# --- 7. Top-level: render a side (inclusion or exclusion) ------------------

def _collect(raw: dict) -> tuple[list, dict, dict]:
    """Return (blocks, status_of, patient_vars) for a side."""
    smt_lines = raw.get("smt_program_lines") or []
    pvv = raw.get("patient_var_values") or {}
    er = raw.get("eval_result") or {}
    lbls = er.get("label_status") or {}
    status_of = {}
    for bucket in ("sat", "unsat", "unknown"):
        for lname in lbls.get(bucket) or []:
            status_of[lname] = bucket
    blocks = [
        b for b in _extract_req_blocks(smt_lines)
        if "_AUXILIARY" not in b["name"]
        and "NOT_REQUIREMNET" not in b["name"]
        and "NOT_REQUIREMENT" not in b["name"]
    ]
    return blocks, status_of, pvv


def render_formula_view(smt_decision: dict) -> str:
    """Primary view: three sections (violations / unknown / satisfied) across both sides.
    Side labels (inclusion / exclusion) are shown as small tags on each card."""
    inc_raw = (smt_decision.get("inclusion") or {}).get("raw") or {}
    exc_raw = (smt_decision.get("exclusion") or {}).get("raw") or {}

    inc_blocks, inc_status, inc_pvv = _collect(inc_raw)
    exc_blocks, exc_status, exc_pvv = _collect(exc_raw)
    all_pvv = {**inc_pvv, **exc_pvv}

    # Build one unified list (block, side, status)
    items = []
    for b in inc_blocks:
        items.append((b, "inclusion", inc_status.get(b["name"], "unknown")))
    for b in exc_blocks:
        items.append((b, "exclusion", exc_status.get(b["name"], "unknown")))

    def render_with_side_tag(b, side, status):
        return render_req(b, status, all_pvv, side=side)

    def sort_key(item):
        b, side, status = item
        m = re.match(r"REQ(\d+)", b["name"])
        n = int(m.group(1)) if m else 999
        return (0 if side == "inclusion" else 1, n)

    items_sorted = sorted(items, key=sort_key)

    by_status = {"unsat": [], "unknown": [], "sat": []}
    for it in items_sorted:
        by_status[it[2]].append(it)

    def section(label, icon, color, bucket, default_open=True):
        xs = by_status[bucket]
        if not xs:
            return ""
        summary_txt = f"{icon} {label} <span style='color:#9ca3af;font-weight:400'>({len(xs)})</span>"
        body = "\n".join(render_with_side_tag(b, s, st) for (b, s, st) in xs)
        return f"""
        <details id="section-{bucket}" {"open" if default_open else ""} style="margin:14px 0;scroll-margin-top:80px">
          <summary style="cursor:pointer;font-weight:600;color:{color};font-size:1.05em;
                          padding:6px 0;border-bottom:1px solid #e5e7eb;margin-bottom:10px;
                          list-style:none">
            <span style="display:inline-block;transition:transform 0.15s">▾</span>
            {summary_txt}
          </summary>
          <div>{body}</div>
        </details>"""

    return f"""
    <section style="margin:20px 0">
      {section("Blocking eligibility", "✗", "#dc2626", "unsat", True)}
      {section("Insufficient evidence", "?", "#f59e0b", "unknown", False)}
      {section("Satisfied", "✓", "#16a34a", "sat", False)}
    </section>"""
