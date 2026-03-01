"""What-if editor: make atom values editable and re-evaluate eligibility live.

Ship the SMT requirements as a compact JSON AST + initial atom values, plus
a small 3-valued-logic JS evaluator. A clinician clicks any atom badge
("=T"/"=F"/"=?"/numeric) to change it, and every REQ card's status + the
top-level eligibility verdict recompute instantly in the browser.
"""
from __future__ import annotations
import html, json, re


_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def _parse_tokens(tokens: list[str]):
    t = tokens.pop(0)
    if t == "(":
        out = []
        while tokens and tokens[0] != ")":
            out.append(_parse_tokens(tokens))
        if tokens:
            tokens.pop(0)
        return out
    return t


def parse_sexp(s: str):
    """Parse a full SMT-LIB s-expression into nested lists / atom strings."""
    return _parse_tokens(_tokenize(s))


def _strip_assert(body: str) -> str:
    m = re.search(r"\(assert\s+\(!\s*(.*)\s+:named\s+\w+\s*\)\s*\)\s*$", body, re.DOTALL)
    if m:
        return m.group(1).strip()
    return body.strip()


def build_req_payload(smt_lines: list[str]) -> list[dict]:
    """Return [{name, ast, vars}] for each real REQ (non-AUX, non-placeholder)."""
    from audit.lib.formula_view import _extract_req_blocks, _VAR_RE

    blocks = _extract_req_blocks(smt_lines)
    payload = []
    for b in blocks:
        n = b["name"]
        if "_AUXILIARY" in n or "NOT_REQUIREMNET" in n or "NOT_REQUIREMENT" in n:
            continue
        core = _strip_assert(b["body"])
        try:
            ast = parse_sexp(core)
        except Exception:
            continue
        vars_ = sorted(set(m.group(0) for m in _VAR_RE.finditer(core)))
        payload.append({"name": n, "ast": ast, "vars": vars_})
    return payload


def build_initial_values(patient_vars: dict) -> dict:
    """Map var_name -> {value, type}.  type ∈ 'bool' | 'num' | 'unknown'."""
    out = {}
    for k, v in (patient_vars or {}).items():
        val = v.get("value") if isinstance(v, dict) else v
        if val is True or val is False:
            out[k] = {"value": val, "type": "bool"}
        elif isinstance(val, (int, float)):
            out[k] = {"value": float(val), "type": "num"}
        else:
            out[k] = {"value": None, "type": "unknown"}
    return out


def _precompute_decision(smt_decision: dict) -> dict:
    """Server-side echo of the JS decision rule, for first-paint rendering."""
    eligible = smt_decision.get("eligible")
    inc_raw = (smt_decision.get("inclusion") or {}).get("raw") or {}
    exc_raw = (smt_decision.get("exclusion") or {}).get("raw") or {}

    def counts(raw):
        er = raw.get("eval_result") or {}
        lbls = er.get("label_status") or {}
        def c(b): return sum(1 for l in (lbls.get(b) or []) if "_AUXILIARY" not in l and "NOT_REQUIREMNET" not in l)
        return c("sat"), c("unsat"), c("unknown")

    inc_s, inc_u, inc_k = counts(inc_raw)
    exc_s, exc_u, exc_k = counts(exc_raw)

    if eligible is True:
        return {"icon": "✓", "text": f"Eligible · all {inc_s + exc_s} requirements satisfied",
                "color": "#16a34a", "bg": "#ecfdf5"}
    if eligible is False:
        bits = []
        if inc_u: bits.append(f"{inc_u} inclusion violation{'s' if inc_u > 1 else ''}")
        if exc_u: bits.append(f"{exc_u} exclusion violation{'s' if exc_u > 1 else ''}")
        return {"icon": "✗", "text": "Ineligible · " + ", ".join(bits),
                "color": "#dc2626", "bg": "#fef2f2"}
    unk = inc_k + exc_k
    return {"icon": "?", "text": f"Defer · {unk} requirement{'s' if unk != 1 else ''} with unknown values",
            "color": "#f59e0b", "bg": "#fffbeb"}


def build_payload(smt_decision: dict) -> dict:
    inc_raw = (smt_decision.get("inclusion") or {}).get("raw") or {}
    exc_raw = (smt_decision.get("exclusion") or {}).get("raw") or {}
    inc_pvv = inc_raw.get("patient_var_values") or {}
    exc_pvv = exc_raw.get("patient_var_values") or {}
    all_vars = {**inc_pvv, **exc_pvv}

    return {
        "inclusion": build_req_payload(inc_raw.get("smt_program_lines") or []),
        "exclusion": build_req_payload(exc_raw.get("smt_program_lines") or []),
        "values": build_initial_values(all_vars),
    }


# --- JS: 3-valued eval + UI wiring ----------------------------------------

# kept compact.  3VL: true=1, false=0, unknown=null.
INTERACTIVE_JS = r"""
<script>
(function () {
  const P = window.__AEGIS_PAYLOAD__;
  if (!P) return;
  const V = P.values;  // var -> {value, type}

  // --- 3-valued logic evaluator ---
  function ev(e) {
    if (typeof e === "string") {
      if (e in V) {
        const x = V[e].value;
        return x === null ? null : x;
      }
      // literal number
      const n = parseFloat(e);
      if (!isNaN(n)) return n;
      if (e === "true") return true;
      if (e === "false") return false;
      return null;
    }
    if (!Array.isArray(e) || e.length === 0) return null;
    const op = e[0];
    const args = e.slice(1).map(ev);
    switch (op) {
      case "and": {
        if (args.includes(false)) return false;
        if (args.includes(null)) return null;
        return true;
      }
      case "or": {
        if (args.includes(true)) return true;
        if (args.includes(null)) return null;
        return false;
      }
      case "not": {
        const a = args[0];
        return a === null ? null : !a;
      }
      case "=>":
      case "implies": {
        const [a, b] = args;
        if (a === false) return true;
        if (a === null) return null;
        return b;
      }
      case "=": {
        const [a, b] = args;
        if (a === null || b === null) return null;
        return a === b;
      }
      case ">=": { const [a,b]=args; return (a===null||b===null)?null:(a>=b); }
      case "<=": { const [a,b]=args; return (a===null||b===null)?null:(a<=b); }
      case ">":  { const [a,b]=args; return (a===null||b===null)?null:(a>b); }
      case "<":  { const [a,b]=args; return (a===null||b===null)?null:(a<b); }
      case "distinct": {
        if (args.some(x=>x===null)) return null;
        const s = new Set(args.map(JSON.stringify));
        return s.size === args.length;
      }
      default: return null;
    }
  }

  function status(v) {
    if (v === true)  return "sat";
    if (v === false) return "unsat";
    return "unknown";
  }

  // For a violated REQ, compute minimal patient-value edits that would satisfy it.
  // Returns [{var, newVal, desc}] suggestions; first one is the quickest fix.
  function suggestFixes(ast) {
    const fixes = [];
    function suggestForVar(v, target, desc) {
      if (!(v in V)) return;
      fixes.push({ var: v, newVal: target, desc });
    }
    function atomName(v) { return v.replace(/^patient_/, "").replace(/_/g, " ").replace(/@@/g, " / "); }
    function walk(e, polarity) {
      // polarity: +1 if we need this subexpr true, -1 if false (for solving)
      if (typeof e === "string") {
        if (!(e in V)) return;
        if (polarity > 0) suggestForVar(e, true,  `assume ${atomName(e)}`);
        else              suggestForVar(e, false, `assume ${atomName(e)} is not the case`);
        return;
      }
      if (!Array.isArray(e)) return;
      const op = e[0];
      if (op === "not") { walk(e[1], -polarity); return; }
      if ((op === "and" && polarity > 0) || (op === "or" && polarity < 0)) {
        // need all children to have this polarity
        for (const c of e.slice(1)) {
          if (status(ev(c)) !== (polarity > 0 ? "sat" : "unsat")) walk(c, polarity);
        }
        return;
      }
      if ((op === "or" && polarity > 0) || (op === "and" && polarity < 0)) {
        // Need just one child to flip; pick the cheapest option (atom that is null/false)
        for (const c of e.slice(1)) {
          walk(c, polarity);
          // One suggestion per or-branch is enough
        }
        return;
      }
      // Comparisons: figure out what value would satisfy
      if ([">=","<=",">","<","="].includes(op) && polarity > 0 && typeof e[1] === "string") {
        const v = e[1]; const nRaw = e[2]; const n = parseFloat(nRaw);
        if (isNaN(n) || !(v in V)) return;
        const cur = V[v].value;
        const nm = atomName(v);
        if (op === ">=") suggestForVar(v, cur !== null && cur >= n ? cur : n,
                                        `set ${nm} to ${n} (≥ ${n})`);
        else if (op === "<=") suggestForVar(v, cur !== null && cur <= n ? cur : n,
                                        `set ${nm} to ${n} (≤ ${n})`);
        else if (op === ">")  suggestForVar(v, cur !== null && cur > n ? cur : n + 1,
                                        `set ${nm} to ${n+1} (> ${n})`);
        else if (op === "<")  suggestForVar(v, cur !== null && cur < n ? cur : n - 1,
                                        `set ${nm} to ${n-1} (< ${n})`);
        else if (op === "=")  suggestForVar(v, n, `set ${nm} to ${n}`);
      }
    }
    walk(ast, 1);
    // Dedupe by var; keep first
    const seen = new Set(); const out = [];
    for (const f of fixes) {
      if (seen.has(f.var)) continue;
      seen.add(f.var); out.push(f);
    }
    return out;
  }

  function badgeStyle(st) {
    if (st === "sat")     return {txt:"SATISFIED",color:"#16a34a",bg:"#ecfdf5"};
    if (st === "unsat")   return {txt:"VIOLATED", color:"#dc2626",bg:"#fef2f2"};
    return                       {txt:"UNKNOWN", color:"#f59e0b",bg:"#fffbeb"};
  }

  // Recompute: mark each req card with new status; recompute top decision
  function recompute() {
    let anyUnsatInc = false, anyUnsatExc = false;
    let anyUnkInc = false, anyUnkExc = false;
    let totSat = 0, totAll = 0;

    function updateSide(reqs, sideKey) {
      for (const r of reqs) {
        const v = ev(r.ast);
        const st = status(v);
        totAll++;
        if (st === "sat") totSat++;
        if (sideKey === "inclusion") {
          if (st === "unsat") anyUnsatInc = true;
          if (st === "unknown") anyUnkInc = true;
        } else {
          if (st === "unsat") anyUnsatExc = true;
          if (st === "unknown") anyUnkExc = true;
        }
        const card = document.querySelector(`[data-req="${r.name}"]`);
        if (card) {
          const b = badgeStyle(st);
          card.style.borderLeftColor = b.color;
          const badge = card.querySelector(".aegis-badge");
          if (badge) {
            badge.textContent = b.txt;
            badge.style.color = b.color;
            badge.style.background = b.bg;
          }
          const conflict = card.querySelector(".aegis-conflict");
          if (conflict) conflict.style.display = (st === "unsat") ? "" : "none";
          // Quick-fix strip: shown only when unsat
          let fix = card.querySelector(".aegis-fixes");
          if (!fix) {
            fix = document.createElement("div");
            fix.className = "aegis-fixes";
            fix.style.cssText = "margin-top:8px;padding-left:calc(1.2em + 10px);"
              + "font-size:0.85em;display:none";
            const body = card.querySelector(".aegis-req-title")?.parentElement;
            if (body) body.appendChild(fix);
          }
          if (st === "unsat") {
            const sugs = suggestFixes(r.ast).slice(0, 4);
            if (sugs.length) {
              fix.style.display = "";
              fix.innerHTML = '<span style="color:#9ca3af;margin-right:6px">What would fix this →</span>'
                + sugs.map((s, i) =>
                    `<button data-fvar="${s.var}" data-fval='${JSON.stringify(s.newVal)}'
                       style="margin:0 4px 4px 0;padding:2px 10px;border:1px solid #fca5a5;
                              background:white;color:#dc2626;border-radius:12px;cursor:pointer;
                              font-size:0.85em">${s.desc}</button>`
                  ).join("");
            } else {
              fix.style.display = "none";
            }
          } else {
            fix.style.display = "none";
          }
        }
      }
    }
    updateSide(P.inclusion, "inclusion");
    updateSide(P.exclusion, "exclusion");

    // Overall decision (mirror AEGIS rule)
    let decision, color, bg, icon, text;
    const unsatInc = P.inclusion.filter(r => status(ev(r.ast)) === "unsat").length;
    const unsatExc = P.exclusion.filter(r => status(ev(r.ast)) === "unsat").length;
    const unkInc  = P.inclusion.filter(r => status(ev(r.ast)) === "unknown").length;
    const unkExc  = P.exclusion.filter(r => status(ev(r.ast)) === "unknown").length;
    if (!anyUnsatInc && !anyUnsatExc && !anyUnkInc && !anyUnkExc) {
      decision="eligible"; color="#16a34a"; bg="#ecfdf5"; icon="✓";
      text="Eligible · all "+totAll+" requirements satisfied";
    } else if (anyUnsatInc || anyUnsatExc) {
      decision="ineligible"; color="#dc2626"; bg="#fef2f2"; icon="✗";
      const bits = [];
      if (unsatInc) bits.push(unsatInc+" inclusion violation"+(unsatInc>1?"s":""));
      if (unsatExc) bits.push(unsatExc+" exclusion violation"+(unsatExc>1?"s":""));
      text="Ineligible · " + bits.join(", ");
    } else {
      decision="defer"; color="#f59e0b"; bg="#fffbeb"; icon="?";
      text="Defer · " + (unkInc+unkExc) + " requirement"+((unkInc+unkExc)>1?"s":"") + " with unknown values";
    }
    const totUnsat = unsatInc + unsatExc;
    const totUnk = unkInc + unkExc;
    const countsEl = document.getElementById("aegis-counts");
    if (countsEl) {
      countsEl.innerHTML =
        `<a href="#section-unsat" title="Jump to violations" style="color:#dc2626;text-decoration:none;cursor:pointer">${totUnsat} ✗</a> · ` +
        `<a href="#section-unknown" title="Jump to unknown" style="color:#f59e0b;text-decoration:none;cursor:pointer">${totUnk} ?</a> · ` +
        `<a href="#section-sat" title="Jump to satisfied" style="color:#16a34a;text-decoration:none;cursor:pointer">${totSat} ✓</a>`;
    }
    const banner = document.getElementById("aegis-live-decision");
    const dtext = document.getElementById("aegis-decision-text");
    if (banner) {
      banner.style.background = bg;
      banner.style.borderColor = color;
    }
    if (dtext) {
      dtext.innerHTML = `<span style="color:${color};font-size:1.3em;margin-right:6px">${icon}</span>`
        + `<span style="color:${color}">AEGIS: ${text}</span>`;
    }
    // Update all .aegis-val spans whose variable changed
    document.querySelectorAll(".aegis-val").forEach(el => {
      const vr = el.dataset.var;
      if (!(vr in V)) return;
      const x = V[vr].value;
      const fmt = el.dataset.format || "pill";
      let txt, color, bg;
      if (x === null)       { txt="unknown"; color="#6b7280"; bg="#f3f4f6"; }
      else if (x === true)  { txt="yes";     color="#065f46"; bg="#d1fae5"; }
      else if (x === false) { txt="no";      color="#7f1d1d"; bg="#fee2e2"; }
      else                  { txt=String(x); color="#1e3a8a"; bg="#dbeafe"; }
      if (fmt === "pill") {
        el.textContent = txt;
        el.style.color = color;
        el.style.background = bg;
      } else {
        // fact format: larger text
        el.innerHTML = `<span style="color:${color};font-weight:600">${txt}</span>`;
      }
    });
  }

  // Bind click handlers: click .aegis-val to cycle bool; numeric opens prompt
  // Fix-button handler
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-fvar]");
    if (!btn) return;
    const v = btn.dataset.fvar;
    const val = JSON.parse(btn.dataset.fval);
    if (!(v in V)) return;
    V[v].value = val;
    if (val === true || val === false) V[v].type = "bool";
    else if (typeof val === "number") V[v].type = "num";
    recompute();
    // brief highlight on banner
    const banner = document.getElementById("aegis-live-decision");
    if (banner) {
      banner.style.transition = "box-shadow 0.3s";
      banner.style.boxShadow = "0 0 0 3px rgba(124,58,237,0.3)";
      setTimeout(() => banner.style.boxShadow = "0 1px 3px rgba(0,0,0,0.05)", 400);
    }
  });

  document.addEventListener("click", (e) => {
    const el = e.target.closest(".aegis-val");
    if (!el) return;
    const vr = el.dataset.var;
    if (!(vr in V)) return;
    const meta = V[vr];
    if (meta.type === "bool" || meta.type === "unknown") {
      // cycle: null -> true -> false -> null
      const cur = meta.value;
      meta.value = (cur === null) ? true : (cur === true ? false : null);
      meta.type = "bool";
    } else if (meta.type === "num") {
      const s = prompt(`Set ${vr} (current = ${meta.value}). Blank = unknown.`,
                       meta.value === null ? "" : meta.value);
      if (s === null) return;  // cancelled
      const f = parseFloat(s);
      meta.value = (s === "" || isNaN(f)) ? null : f;
    }
    recompute();
  });

  // Clicking a count anchor should auto-open that collapsed section
  function openHashSection() {
    const h = window.location.hash;
    if (!h) return;
    const el = document.querySelector(h);
    if (el && el.tagName === "DETAILS") el.open = true;
  }
  window.addEventListener("hashchange", openHashSection);
  openHashSection();

  // Snapshot initial values for edit-tracking & reset
  const initial = JSON.parse(JSON.stringify(V));

  function fmtVal(x) {
    if (x === null) return "unknown";
    if (x === true) return "yes";
    if (x === false) return "no";
    return String(x);
  }
  function atomName(v) {
    return v.replace(/^patient_/, "").replace(/_/g, " ").replace(/@@/g, " / ");
  }
  function editStatus() {
    const edits = [];
    for (const k of Object.keys(V)) {
      if (JSON.stringify(V[k].value) !== JSON.stringify(initial[k].value)) edits.push(k);
    }
    const n = edits.length;
    const el = document.getElementById("aegis-edit-status");
    const reset = document.getElementById("aegis-reset");
    const fixAll = document.getElementById("aegis-fix-all");
    const diff = document.getElementById("aegis-edit-diff");
    if (el) {
      el.innerHTML = n === 0
        ? '<span style="color:#9ca3af">unchanged from chart</span>'
        : `<b style="color:#7c3aed">what-if · ${n} edit${n>1?"s":""}</b>`;
    }
    if (reset) reset.style.display = n > 0 ? "" : "none";
    // Fix-all visible whenever there is at least one violation
    const anyViolation = [...P.inclusion, ...P.exclusion]
      .some(r => status(ev(r.ast)) === "unsat");
    if (fixAll) fixAll.style.display = anyViolation ? "" : "none";
    // Edit-diff summary
    if (diff) {
      if (n === 0) {
        diff.style.display = "none";
      } else {
        diff.style.display = "";
        const rows = edits.slice(0, 8).map(k => {
          const before = fmtVal(initial[k].value);
          const after  = fmtVal(V[k].value);
          return `<li style="margin:2px 0"><code style="color:#374151;font-size:0.95em">${atomName(k)}</code>: <span style="color:#9ca3af">${before}</span> → <b>${after}</b></li>`;
        }).join("");
        const more = n > 8 ? `<li style="color:#9ca3af">(+${n-8} more)</li>` : "";
        diff.innerHTML = '<b>Assumptions you\'ve applied:</b><ul style="margin:4px 0;padding-left:20px">'
          + rows + more + '</ul>';
      }
    }
  }

  // Run edit-status after every recompute
  const _origRecompute = recompute;
  recompute = function() { _origRecompute(); editStatus(); };
  recompute();

  const resetBtn = document.getElementById("aegis-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      for (const k of Object.keys(V)) {
        V[k].value = initial[k].value;
        V[k].type  = initial[k].type;
      }
      recompute();
    });
  }

  const fixAllBtn = document.getElementById("aegis-fix-all");
  if (fixAllBtn) {
    fixAllBtn.addEventListener("click", () => {
      // Iterate up to a few passes — applying one REQ's fix may satisfy others
      for (let pass = 0; pass < 4; pass++) {
        let changed = false;
        for (const r of [...P.inclusion, ...P.exclusion]) {
          if (status(ev(r.ast)) !== "unsat") continue;
          const sugs = suggestFixes(r.ast);
          if (!sugs.length) continue;
          const s = sugs[0];
          if (JSON.stringify(V[s.var].value) === JSON.stringify(s.newVal)) continue;
          V[s.var].value = s.newVal;
          if (s.newVal === true || s.newVal === false) V[s.var].type = "bool";
          else if (typeof s.newVal === "number") V[s.var].type = "num";
          changed = true;
        }
        if (!changed) break;
      }
      recompute();
    });
  }
})();
</script>
"""


def render_interactive_block(smt_decision: dict) -> str:
    """Primary decision banner + inline what-if reset. This IS the AEGIS-decision box."""
    payload = build_payload(smt_decision)
    # Pre-compute the baked-in decision so the banner has real content on first paint
    pre_decision = _precompute_decision(smt_decision)
    return f"""
    <div id="aegis-live-decision" title="Click any value in a rule below to explore what-if scenarios."
         style="padding:12px 18px;border-radius:8px;margin:16px 0 6px 0;
                border:1px solid {pre_decision['color']};background:{pre_decision['bg']};
                position:sticky;top:0;z-index:10;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);
                display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
      <div id="aegis-decision-text" style="font-size:1.15em;font-weight:600;color:{pre_decision['color']}">
        <span style="font-size:1.2em;margin-right:6px">{pre_decision['icon']}</span>
        AEGIS: {pre_decision['text']}
      </div>
      <div style="display:flex;gap:10px;align-items:center;font-size:0.85em;color:#6b7280;flex-wrap:wrap">
        <span id="aegis-counts" style="font-family:'SF Mono',Menlo,monospace;font-size:0.88em;
              background:white;padding:3px 8px;border-radius:4px;border:1px solid #e5e7eb">
          <span style="color:#dc2626">0 ✗</span> ·
          <span style="color:#f59e0b">0 ?</span> ·
          <span style="color:#16a34a">0 ✓</span>
        </span>
        <span id="aegis-edit-status"></span>
        <button id="aegis-fix-all" title="Apply one quick-fix per remaining violation"
                style="display:none;padding:3px 10px;border:1px solid #7c3aed;
                       background:#ede9fe;color:#5b21b6;border-radius:4px;cursor:pointer;
                       font-size:0.85em;font-weight:600">
          Fix all remaining
        </button>
        <button id="aegis-reset"
                style="display:none;padding:3px 10px;border:1px solid #d1d5db;
                       background:white;border-radius:4px;cursor:pointer;font-size:0.85em">
          Reset to chart
        </button>
      </div>
    </div>
    <div id="aegis-edit-diff" style="display:none;margin:0 0 14px 0;padding:8px 12px;
         background:#faf5ff;border-left:3px solid #7c3aed;border-radius:4px;
         font-size:0.86em;color:#5b21b6"></div>
    <script id="aegis-payload" type="application/json">{html.escape(json.dumps(payload))}</script>
    <script>window.__AEGIS_PAYLOAD__ = JSON.parse(document.getElementById('aegis-payload').textContent);</script>
    {INTERACTIVE_JS}
    """
