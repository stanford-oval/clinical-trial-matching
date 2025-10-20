#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_ir_units_ucum_report.py
─────────────────────────────────
UCUM + Pint–backed unit normalization for SMT-LIB IR **with a detailed report**.

What you get
  • Renamed numeric variables to canonical, unit-bearing stems (snake_case preserved)
  • Numeric thresholds rescaled in simple comparisons via UCUM conversions
  • Age canonicalized to years
  • A comprehensive normalization report:
      - totals (files/vars changed, scaled comparators by op, etc.)
      - per-file details
      - per-variable rename map with UCUM from→to + factor
      - all scaled comparator events (old/new values, factors)
      - unknown unit tokens encountered
      - usage counts (how often each old var appeared)
      - roll-ups by unit token (old→canonical)

Usage (no CLI)
  import normalize_ir_units_ucum_report as norm

  # 1) Whole directory → mirrored output + report dict
  report = norm.normalize_directory_with_report(
      in_dir="<SATIR_ROOT>/build/ir",
      out_dir="<SATIR_ROOT>/build/ir_normalized",
      dry_run=False,
  )

  # 2) Save JSON/TSV artifacts if you want:
  norm.save_report_json(report, "<SATIR_ROOT>/build/ir_normalized/normalization_report.json")
  norm.save_report_tsv(report,  "<SATIR_ROOT>/build/ir_normalized/normalization_report.tsv")

  # 3) Single-text normalization (returns new_text + mapping + mini-report of events)
  out_text, result = norm.normalize_smt_text_with_report(in_text)

Requirements
  pip install ucumvert pint
"""

from __future__ import annotations
import math, re, json, csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# UCUM + Pint
# ────────────────────────────────────────────────────────────────────────────────
try:
    from ucumvert import PintUcumRegistry  # UCUM-aware UnitRegistry
except Exception as e:
    raise RuntimeError("Please install dependencies:  pip install ucumvert pint") from e

_ureg = PintUcumRegistry()

def _ucum_factor(old_ucum: str, target_ucum: str) -> float:
    """
    Return multiplicative factor to convert a numeric threshold
    from old_ucum → target_ucum.

    We ask ucumvert for pint objects for both units and divide them.
    Whatever `from_ucum` returns (Quantity, Unit, or float), we try to
    reduce to a plain scaling factor.
    """
    q_old = _ureg.from_ucum(old_ucum)
    q_tgt = _ureg.from_ucum(target_ucum)

    # Normal case: both are pint quantities or units; division should work.
    try:
        ratio = q_old / q_tgt
    except TypeError:
        # Extremely defensive: if something really odd comes back, coerce.
        ratio = float(q_old) / float(q_tgt)

    # If this is a pint Quantity, it has .magnitude; otherwise just treat
    # the result itself as a scalar.
    mag = getattr(ratio, "magnitude", ratio)
    return float(mag)


# ────────────────────────────────────────────────────────────────────────────────
# S-expression parser / printer (comment-tolerant)
# ────────────────────────────────────────────────────────────────────────────────

Token = str
S = Any  # Atom=str, List[S]=list

def _tokenize(s: str) -> List[Token]:
    s = re.sub(r";[^\n]*", "", s)                       # strip ';' comments
    s = re.sub(r'("([^"\\]|\\.)*")', r' \1 ', s)        # keep strings atomic
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
    i, out = 0, []
    while i < len(toks):
        node, i = _parse(toks, i)
        out.append(node)
    return out

def sym(x: S) -> Optional[str]:
    return x if isinstance(x, str) else None

def sexpr_to_str(x: S) -> str:
    if isinstance(x, list):
        return "(" + " ".join(sexpr_to_str(t) for t in x) + ")"
    return str(x)

def _is_number(tok: str) -> bool:
    try:
        float(tok); return True
    except Exception:
        return False

def _fmt_num(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.9g}"

# ────────────────────────────────────────────────────────────────────────────────
# Variable finders & UCUM token mapping
# ────────────────────────────────────────────────────────────────────────────────

# Age variables: age_value_recorded_in_(years|months|days)_{timeframe}
AGE_VAR_RE = re.compile(r"\bage_value_recorded_in_(years|months|days)_([a-z0-9]+)\b")

# Generic numeric observable stems:
#   <entity>_value_recorded_<timeframe>_(withunit|in)_<unit_token>
NUMERIC_VAR_RE = re.compile(
    r"\b([a-z0-9_]+)_value_recorded_([a-z0-9]+)_(?:withunit|in)_([a-z0-9_]+)\b"
)

@dataclass(frozen=True)
class UnitResolution:
    canon_token: str      # canonical snake_case unit token for the variable name
    old_ucum: str         # UCUM code parsed from the original token
    target_ucum: str      # canonical UCUM code we normalize to

# Known tokens you already use (extend as needed)
UNIT_TOKEN_RESOLVE: Dict[str, UnitResolution] = {
    # pressure
    "mm_hg": UnitResolution("mm_hg", "mm[Hg]", "mm[Hg]"),
    "mmhg":  UnitResolution("mm_hg", "mm[Hg]", "mm[Hg]"),
    "kpa":   UnitResolution("mm_hg", "kPa",    "mm[Hg]"),  # Pint understands kPa; UCUM "kPa"
    # flow / rates
    "ml_per_min": UnitResolution("ml_per_min", "mL/min", "mL/min"),
    "bpm":        UnitResolution("bpm",        "1/min",  "1/min"),
    # labs / density
    "g_per_dl":   UnitResolution("g_per_dl",   "g/dL",   "g/dL"),
    "g_dl":       UnitResolution("g_per_dl",   "g/dL",   "g/dL"),
    "mg_per_dl":  UnitResolution("g_per_dl",   "mg/dL",  "g/dL"),
    "g_per_l":    UnitResolution("g_per_dl",   "g/L",    "g/dL"),
    "kg_per_m2":  UnitResolution("kg_per_m2",  "kg/m2",  "kg/m2"),
    # counts
    "per_microliter": UnitResolution("per_microliter", "1/uL",  "1/uL"),
    "per_ul":         UnitResolution("per_microliter", "1/uL",  "1/uL"),
    "per_u_l":        UnitResolution("per_microliter", "1/uL",  "1/uL"),
    "per_mm_cubed":   UnitResolution("per_microliter", "1/mm3", "1/uL"),
    "per_mm3":        UnitResolution("per_microliter", "1/mm3", "1/uL"),
    "10e9_per_l":     UnitResolution("per_microliter", "10*9/L","1/uL"),  # ×1000
    "per_l":          UnitResolution("per_l",          "1/L",    "1/L"),
    # weights
    "kg":         UnitResolution("kilograms", "kg",     "kg"),
    "kilograms":  UnitResolution("kilograms", "kg",     "kg"),
    "lb":         UnitResolution("kilograms", "[lb_av]","kg"),
    "lbs":        UnitResolution("kilograms", "[lb_av]","kg"),
    "pounds":     UnitResolution("kilograms", "[lb_av]","kg"),
}

def _resolve_unit_token(token: str) -> UnitResolution | None:
    token = token.lower()
    if token in UNIT_TOKEN_RESOLVE:
        return UNIT_TOKEN_RESOLVE[token]
    # generic "x_per_y" → "x/y" (best-effort)
    m = re.match(r"^([a-z0-9_]+)_per_([a-z0-9_]+)$", token)
    if m:
        num = m.group(1).replace("_", "")
        den = m.group(2).replace("_", "")
        uc = f"{num}/{den}"
        try:
            _ureg.from_ucum(uc)  # validate
            return UnitResolution(token, uc, uc)
        except Exception:
            return None
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Reporting structures
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class VarRename:
    old_var: str
    new_var: str
    factor: float
    category: str               # "age" | "unit"
    unit_token_old: str | None  # e.g., "kpa"
    unit_token_new: str | None  # e.g., "mm_hg"
    ucum_from: str | None       # e.g., "kPa"
    ucum_to: str | None         # e.g., "mm[Hg]"
    occurrences_replaced: int   # how many times we substituted this var in the AST

@dataclass
class ScaledComparator:
    op: str                     # one of >=, <=, >, <, =
    orientation: str            # "var_const" or "const_var"
    var_old: str
    var_new: str
    value_old: float
    value_new: float
    factor: float

@dataclass
class FileReport:
    path: str
    changed: bool
    var_renames: List[VarRename]
    scaled_comparators: List[ScaledComparator]
    skipped_scalings: List[str]         # free-form reasons
    unknown_unit_tokens: List[str]
    counters: Dict[str, int]            # small stats per file

@dataclass
class NormalizationReport:
    in_dir: str
    out_dir: str
    files_processed: int
    files_changed: int
    totals: Dict[str, int]              # rolled-up counts
    unit_token_rollup: Dict[str, Dict[str, int]]  # {old_token: {new_token: count}}
    by_file: List[FileReport]

# ────────────────────────────────────────────────────────────────────────────────
# Build rename/scale map with metadata + recorder
# ────────────────────────────────────────────────────────────────────────────────

def _build_var_map_for_text_with_meta(text: str):
    """
    Returns:
      var_map: { old_var -> (new_var, factor) }
      renames_meta: { old_var -> VarRename (occurrences_replaced initialized to 0) }
      unknown_tokens: set[str]
    """
    var_map: Dict[str, Tuple[str, float]] = {}
    renames_meta: Dict[str, VarRename] = {}
    unknown_tokens: set[str] = set()

    # 1) Age → years
    for m in AGE_VAR_RE.finditer(text):
        unit, tf = m.group(1), m.group(2)
        old = m.group(0)
        if unit == "years":
            new, k = old, 1.0
        else:
            uc_from = {"months": "mo", "days": "d"}[unit]
            new = f"age_value_recorded_in_years_{tf}"
            k = _ucum_factor(uc_from, "a")  # to years
        var_map[old] = (new, k)
        renames_meta[old] = VarRename(
            old_var=old, new_var=new, factor=k, category="age",
            unit_token_old=unit, unit_token_new="years",
            ucum_from={"months":"mo","days":"d","years":"a"}[unit] if unit in {"months","days","years"} else None,
            ucum_to="a", occurrences_replaced=0
        )

    # 2) Numeric observables with units
    for m in NUMERIC_VAR_RE.finditer(text):
        ent, tf, utok = m.group(1), m.group(2), m.group(3)
        old = m.group(0)
        res = _resolve_unit_token(utok)
        if not res:
            unknown_tokens.add(utok)
            var_map.setdefault(old, (old, 1.0))
            continue
        k = _ucum_factor(res.old_ucum, res.target_ucum)
        new = f"{ent}_value_recorded_{tf}_withunit_{res.canon_token}"
        var_map[old] = (new, k)
        renames_meta[old] = VarRename(
            old_var=old, new_var=new, factor=k, category="unit",
            unit_token_old=utok, unit_token_new=res.canon_token,
            ucum_from=res.old_ucum, ucum_to=res.target_ucum,
            occurrences_replaced=0
        )
    return var_map, renames_meta, unknown_tokens

# ────────────────────────────────────────────────────────────────────────────────
# AST transform: rename + rescale (with event recording)
# ────────────────────────────────────────────────────────────────────────────────

NUM_CMP_OPS = {">=", "<=", ">", "<", "="}

def _scale_if_simple_cmp(form: List[S],
                         var_map: Dict[str, Tuple[str, float]],
                         events: List[ScaledComparator]) -> Optional[List[S]]:
    """If form is (op var num) or (op num var), rescale constant and record event."""
    if not (isinstance(form, list) and len(form) == 3 and sym(form[0]) in NUM_CMP_OPS):
        return None
    op, a, b = sym(form[0]), form[1], form[2]

    # (op var num)
    if isinstance(a, str) and isinstance(b, str) and _is_number(b):
        if a in var_map:
            new_var, k = var_map[a]
            old_val = float(b)
            new_val = float(_fmt_num(old_val * k))
            events.append(ScaledComparator(op=op, orientation="var_const",
                                           var_old=a, var_new=new_var,
                                           value_old=old_val, value_new=new_val, factor=k))
            return [op, new_var, _fmt_num(new_val)]
        return None

    # (op num var)
    if isinstance(a, str) and _is_number(a) and isinstance(b, str):
        if b in var_map:
            new_var, k = var_map[b]
            old_val = float(a)
            new_val = float(_fmt_num(old_val * k))
            events.append(ScaledComparator(op=op, orientation="const_var",
                                           var_old=b, var_new=new_var,
                                           value_old=old_val, value_new=new_val, factor=k))
            return [op, _fmt_num(new_val), new_var]
        return None

    return None

def _transform_expr(x: S,
                    var_map: Dict[str, Tuple[str, float]],
                    rename_meta: Dict[str, VarRename],
                    scaled_events: List[ScaledComparator],
                    skipped_scalings: List[str]) -> S:
    # atom
    if isinstance(x, str):
        if x in var_map and var_map[x][0] != x:
            rename_meta[x].occurrences_replaced += 1
            return var_map[x][0]
        return x

    if not isinstance(x, list) or not x:
        return x

    head = sym(x[0])

    # Preserve annotations: (! term :named tag)
    if head == "!":
        if len(x) >= 2:
            transformed = _transform_expr(x[1], var_map, rename_meta, scaled_events, skipped_scalings)
            return ["!", transformed, *x[2:]]
        return x

    # Transform children first
    children = [_transform_expr(t, var_map, rename_meta, scaled_events, skipped_scalings) for t in x]

    # After renames, try to rescale simple comparator
    if head in NUM_CMP_OPS:
        maybe = _scale_if_simple_cmp(children, var_map, scaled_events)
        if maybe is not None:
            return maybe
        else:
            # Optional: record skip when comparator isn't simple
            skipped_scalings.append(f"Skipped complex comparator: {sexpr_to_str(children)}")

    return children

# ────────────────────────────────────────────────────────────────────────────────
# Core normalize (single text) with report
# ────────────────────────────────────────────────────────────────────────────────

def normalize_smt_text_with_report(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize a single SMT-LIB string and return:
      out_text, {
        'var_renames': [VarRename...],
        'scaled_comparators': [ScaledComparator...],
        'skipped_scalings': [str...],
        'unknown_unit_tokens': [str...],
        'counters': {...}
      }
    """
    var_map, renames_meta, unknown_tokens = _build_var_map_for_text_with_meta(text)

    # Quick no-op return if nothing mapped
    if all(old == new for old, (new, _) in var_map.items()):
        header = [
            ";; =====================================================================",
            ";; UNIT-NORMALIZED (UCUM-backed) — no changes detected",
            ";; =====================================================================",
            "",
        ]
        return ("\n".join(header) + text, {
            "var_renames": [],
            "scaled_comparators": [],
            "skipped_scalings": [],
            "unknown_unit_tokens": sorted(list(unknown_tokens)),
            "counters": {"declarations_renamed": 0, "atoms_replaced": 0, "comparators_scaled": 0}
        })

    ast = parse_sexpr(text)
    scaled_events: List[ScaledComparator] = []
    skipped_scalings: List[str] = []
    decls_renamed = 0

    new_ast: List[S] = []
    for form in ast:
        # Also handle (declare-const var T) renames
        if isinstance(form, list) and len(form) >= 3 and sym(form[0]) == "declare-const" and isinstance(form[1], str):
            old = form[1]
            new = var_map.get(old, (old, 1.0))[0]
            if new != old:
                new_ast.append(["declare-const", new, form[2]])
                if old in renames_meta:
                    renames_meta[old].occurrences_replaced += 1
                decls_renamed += 1
                continue
        new_ast.append(_transform_expr(form, var_map, renames_meta, scaled_events, skipped_scalings))

    header = [
        ";; =====================================================================",
        ";; UNIT-NORMALIZED (UCUM-backed) by normalize_ir_units_ucum_report.py",
        ";;  - Variables renamed to canonical unit-bearing stems (snake_case preserved)",
        ";;  - Numeric thresholds scaled via UCUM conversions",
        ";;  - Age canonicalized to years",
        ";; =====================================================================",
        "",
    ]
    out_text = "\n".join(header) + "\n".join(sexpr_to_str(f) for f in new_ast) + "\n"

    # build payload
    renames_list = [asdict(v) for v in renames_meta.values()]
    payload = {
        "var_renames": renames_list,
        "scaled_comparators": [asdict(e) for e in scaled_events],
        "skipped_scalings": skipped_scalings,
        "unknown_unit_tokens": sorted(list(unknown_tokens)),
        "counters": {
            "declarations_renamed": decls_renamed,
            "atoms_replaced": sum(v["occurrences_replaced"] for v in renames_list),
            "comparators_scaled": len(scaled_events),
        },
    }
    return out_text, payload

# ────────────────────────────────────────────────────────────────────────────────
# File & directory helpers with full report
# ────────────────────────────────────────────────────────────────────────────────

def normalize_file_with_report(src: Path, dst: Path) -> FileReport:
    text = src.read_text(encoding="utf-8")
    out_text, payload = normalize_smt_text_with_report(text)

    dst.parent.mkdir(parents=True, exist_ok=True)
    changed = (out_text != text)
    # always mirror so the output tree is complete
    dst.write_text(out_text if changed else text, encoding="utf-8")

    # Convert payload to dataclasses for FileReport
    var_renames = [VarRename(**d) for d in payload["var_renames"]]
    scaled = [ScaledComparator(**d) for d in payload["scaled_comparators"]]
    unknown = payload["unknown_unit_tokens"]
    counters = payload["counters"]
    return FileReport(
        path=str(src), changed=changed, var_renames=var_renames,
        scaled_comparators=scaled, skipped_scalings=payload["skipped_scalings"],
        unknown_unit_tokens=unknown, counters=counters
    )

def normalize_directory_with_report(in_dir: Path | str,
                                    out_dir: Path | str,
                                    dry_run: bool = False) -> NormalizationReport:
    in_dir = Path(in_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_file: List[FileReport] = []
    patterns = (".smt2", ".smt", ".smtlib")
    for src in sorted(in_dir.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in patterns:
            continue
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        if dry_run:
            # Compute but do not write
            text = src.read_text(encoding="utf-8")
            _, payload = normalize_smt_text_with_report(text)
            fr = FileReport(
                path=str(src), changed=bool(payload["var_renames"] or payload["scaled_comparators"]),
                var_renames=[VarRename(**d) for d in payload["var_renames"]],
                scaled_comparators=[ScaledComparator(**d) for d in payload["scaled_comparators"]],
                skipped_scalings=payload["skipped_scalings"],
                unknown_unit_tokens=payload["unknown_unit_tokens"],
                counters=payload["counters"]
            )
            by_file.append(fr)
        else:
            by_file.append(normalize_file_with_report(src, dst))

    files_processed = len(by_file)
    files_changed = sum(1 for f in by_file if f.changed)

    # roll-ups
    totals = {
        "vars_renamed": sum(len(f.var_renames) for f in by_file),
        "comparators_scaled": sum(len(f.scaled_comparators) for f in by_file),
        "declarations_renamed": sum(f.counters.get("declarations_renamed", 0) for f in by_file),
        "atoms_replaced": sum(f.counters.get("atoms_replaced", 0) for f in by_file),
        "skipped_scalings": sum(len(f.skipped_scalings) for f in by_file),
        "unknown_unit_tokens": len({t for f in by_file for t in f.unknown_unit_tokens}),
    }

    # unit token rollup: {old_token: {new_token: count}}
    roll: Dict[str, Dict[str, int]] = {}
    for f in by_file:
        for vr in f.var_renames:
            if vr.category != "unit": continue
            old_tok = vr.unit_token_old or "unknown"
            new_tok = vr.unit_token_new or "unknown"
            d = roll.setdefault(old_tok, {})
            d[new_tok] = d.get(new_tok, 0) + 1

    return NormalizationReport(
        in_dir=str(in_dir), out_dir=str(out_dir),
        files_processed=files_processed, files_changed=files_changed,
        totals=totals, unit_token_rollup=roll, by_file=by_file
    )

# ────────────────────────────────────────────────────────────────────────────────
# Serialization helpers (optional)
# ────────────────────────────────────────────────────────────────────────────────

def _dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj

def save_report_json(report: NormalizationReport, path: str | Path) -> None:
    """Save the full report as JSON."""
    data = _dataclass_to_dict(report)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_report_tsv(report: NormalizationReport, path: str | Path) -> None:
    """
    Save a compact TSV with one row per variable rename and one per scaled comparator
    (two sections separated by a blank line).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        # section 1: var renames
        w.writerow(["SECTION", "file", "old_var", "new_var", "factor", "category",
                    "unit_token_old", "unit_token_new", "ucum_from", "ucum_to", "occurrences"])
        for fr in report.by_file:
            for vr in fr.var_renames:
                w.writerow(["VAR_RENAME", fr.path, vr.old_var, vr.new_var, vr.factor, vr.category,
                            vr.unit_token_old or "", vr.unit_token_new or "",
                            vr.ucum_from or "", vr.ucum_to or "", vr.occurrences_replaced])
        w.writerow([])

        # section 2: scaled comparators
        w.writerow(["SECTION", "file", "op", "orientation", "var_old", "var_new", "value_old", "value_new", "factor"])
        for fr in report.by_file:
            for ev in fr.scaled_comparators:
                w.writerow(["SCALED_CMP", fr.path, ev.op, ev.orientation, ev.var_old, ev.var_new,
                            ev.value_old, ev.value_new, ev.factor])

# ────────────────────────────────────────────────────────────────────────────────
# Summary: unit-bearing variable identification & rewrites
# ────────────────────────────────────────────────────────────────────────────────

def summarize_unit_variables(report: NormalizationReport) -> Dict[str, Any]:
    """
    Compute, across all input files, how many unit-bearing variables were:
      - seen (match our naming patterns),
      - identified (unit token resolved or age recognized),
      - rewritten (canonical var name differs).
    Returns counts both for UNIQUE variable names and for total OCCURRENCES.
    """
    totals_unique = {
        "all_seen": 0, "all_identified": 0, "all_rewritten": 0,
        "numeric_seen": 0, "numeric_identified": 0, "numeric_rewritten": 0,
        "age_seen": 0, "age_identified": 0, "age_rewritten": 0,
    }
    totals_occ = {
        "all_seen": 0, "all_identified": 0, "all_rewritten": 0,
        "numeric_seen": 0, "numeric_identified": 0, "numeric_rewritten": 0,
        "age_seen": 0, "age_identified": 0, "age_rewritten": 0,
    }
    per_file: List[Dict[str, Any]] = []

    for fr in report.by_file:
        p = Path(fr.path)
        txt = p.read_text(encoding="utf-8")

        # ---- Numeric observables with units ----
        num_matches = list(NUMERIC_VAR_RE.finditer(txt))
        # unique by full var name
        num_unique_names = {}
        for m in num_matches:
            old_name = m.group(0)
            if old_name in num_unique_names:
                continue
            ent, tf, utok = m.group(1), m.group(2), m.group(3)
            res = _resolve_unit_token(utok)
            identified = res is not None
            if identified:
                new_name = f"{ent}_value_recorded_{tf}_withunit_{res.canon_token}"
                rewritten = (new_name != old_name)
            else:
                new_name = old_name
                rewritten = False
            num_unique_names[old_name] = (identified, rewritten)

        # occurrence-level stats
        num_occ_seen = len(num_matches)
        num_occ_identified = 0
        num_occ_rewritten  = 0
        for m in num_matches:
            ent, tf, utok = m.group(1), m.group(2), m.group(3)
            old_name = m.group(0)
            res = _resolve_unit_token(utok)
            if res:
                num_occ_identified += 1
                new_name = f"{ent}_value_recorded_{tf}_withunit_{res.canon_token}"
                if new_name != old_name:
                    num_occ_rewritten += 1

        # ---- Age variables (treated as unit-bearing; target is years) ----
        age_matches = list(AGE_VAR_RE.finditer(txt))
        # unique by full var name
        age_unique_names = {}
        for m in age_matches:
            old_name = m.group(0)
            if old_name in age_unique_names:
                continue
            unit, tf = m.group(1), m.group(2)
            identified = True  # age pattern always recognized
            rewritten = (unit != "years")
            age_unique_names[old_name] = (identified, rewritten)

        age_occ_seen = len(age_matches)
        age_occ_identified = age_occ_seen  # all matched → identified
        age_occ_rewritten  = sum(1 for m in age_matches if m.group(1) != "years")

        # ---- Per-file rollup (unique) ----
        file_num_seen_u = len(num_unique_names)
        file_num_ident_u = sum(1 for v in num_unique_names.values() if v[0])
        file_num_rewr_u  = sum(1 for v in num_unique_names.values() if v[1])

        file_age_seen_u  = len(age_unique_names)
        file_age_ident_u = sum(1 for v in age_unique_names.values() if v[0])
        file_age_rewr_u  = sum(1 for v in age_unique_names.values() if v[1])

        # ---- Update totals (unique) ----
        totals_unique["numeric_seen"]       += file_num_seen_u
        totals_unique["numeric_identified"] += file_num_ident_u
        totals_unique["numeric_rewritten"]  += file_num_rewr_u

        totals_unique["age_seen"]           += file_age_seen_u
        totals_unique["age_identified"]     += file_age_ident_u
        totals_unique["age_rewritten"]      += file_age_rewr_u

        # ---- Update totals (occurrences) ----
        totals_occ["numeric_seen"]       += num_occ_seen
        totals_occ["numeric_identified"] += num_occ_identified
        totals_occ["numeric_rewritten"]  += num_occ_rewritten

        totals_occ["age_seen"]           += age_occ_seen
        totals_occ["age_identified"]     += age_occ_identified
        totals_occ["age_rewritten"]      += age_occ_rewritten

        per_file.append({
            "file": str(p),
            "unique": {
                "numeric_seen": file_num_seen_u,
                "numeric_identified": file_num_ident_u,
                "numeric_rewritten": file_num_rewr_u,
                "age_seen": file_age_seen_u,
                "age_identified": file_age_ident_u,
                "age_rewritten": file_age_rewr_u,
                "all_seen": file_num_seen_u + file_age_seen_u,
                "all_identified": file_num_ident_u + file_age_ident_u,
                "all_rewritten": file_num_rewr_u + file_age_rewr_u,
            },
            "occurrences": {
                "numeric_seen": num_occ_seen,
                "numeric_identified": num_occ_identified,
                "numeric_rewritten": num_occ_rewritten,
                "age_seen": age_occ_seen,
                "age_identified": age_occ_identified,
                "age_rewritten": age_occ_rewritten,
                "all_seen": num_occ_seen + age_occ_seen,
                "all_identified": num_occ_identified + age_occ_identified,
                "all_rewritten": num_occ_rewritten + age_occ_rewritten,
            }
        })

    # finalize totals
    totals_unique["all_seen"]       = totals_unique["numeric_seen"]       + totals_unique["age_seen"]
    totals_unique["all_identified"] = totals_unique["numeric_identified"] + totals_unique["age_identified"]
    totals_unique["all_rewritten"]  = totals_unique["numeric_rewritten"]  + totals_unique["age_rewritten"]

    totals_occ["all_seen"]       = totals_occ["numeric_seen"]       + totals_occ["age_seen"]
    totals_occ["all_identified"] = totals_occ["numeric_identified"] + totals_occ["age_identified"]
    totals_occ["all_rewritten"]  = totals_occ["numeric_rewritten"]  + totals_occ["age_rewritten"]

    return {
        "totals_unique": totals_unique,
        "totals_occurrences": totals_occ,
        "by_file": per_file,
    }

# ────────────────────────────────────────────────────────────────────────────────
# Convenience single-call runner (no CLI)
# ────────────────────────────────────────────────────────────────────────────────

def run(
    in_dir: Path | str = "../../build/ir", 
    out_dir: Path | str = "../../build/ir_normalized",
    dry_run: bool = False,
) -> NormalizationReport:
    """
    Normalize the directory and return a NormalizationReport (also prints a brief summary).
    """
    report = normalize_directory_with_report(in_dir, out_dir, dry_run=dry_run)
    print(f"[normalize_ir_units_ucum_report] files: {report.files_processed}  changed: {report.files_changed}  "
          f"vars_renamed: {report.totals['vars_renamed']}  comparators_scaled: {report.totals['comparators_scaled']}")
    if report.totals["unknown_unit_tokens"]:
        unk = ", ".join(sorted({t for f in report.by_file for t in f.unknown_unit_tokens}))
        print(f"[warn] unknown unit tokens encountered: {unk}")

    save_report_json(report, Path(out_dir) / "normalization_report.json")
    save_report_tsv(report,  Path(out_dir) / "normalization_report.tsv")

    summary = summarize_unit_variables(report)

    print("UNIQUE variable names:")
    print("  all_seen:",       summary["totals_unique"]["all_seen"])
    print("  all_identified:", summary["totals_unique"]["all_identified"])
    print("  all_rewritten:",  summary["totals_unique"]["all_rewritten"])

    print("\nOCCURRENCES:")
    print("  all_seen:",       summary["totals_occurrences"]["all_seen"])
    print("  all_identified:", summary["totals_occurrences"]["all_identified"])
    print("  all_rewritten:",  summary["totals_occurrences"]["all_rewritten"])

    return report

# Optional: allow running as a module (still no CLI parsing)
if __name__ == "__main__":
    run()


