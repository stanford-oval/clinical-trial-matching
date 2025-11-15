#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_schema_implications.py
Lightweight per-patient implication runner with explicit source→target provenance.

Exports:
  - RULES_PATH (str): path to rules JSON; driver may overwrite
  - ROOT_IN, ROOT_OUT (str): input/output roots; driver may overwrite
  - load_rules(path) -> dict
  - process_patient(pin, pout, rules, tf_cfg) -> (rows, schema_new_rc, timeframe_new_rc)

Input (per patient dir pin):
  - canonical.enriched.jsonl  (preferred)
    or canonical.jsonl        (fallback)

Outputs (per patient dir pout):
  - canonical.enriched.jsonl                              (passthrough copy from pin)
  - canonical.enriched.implications.structured.json       (structured implications with provenance)
"""

from __future__ import annotations
import os, re, json
from typing import Dict, Any, List, Tuple, Optional

# ---- Defaults (driver will overwrite these) -----------------------------------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 规则路径仍然允许：
#   1) 显式 --rules 覆盖
#   2) 环境变量 SMT_RULES_PATH
#   3) 否则退回 module 相对路径
RULES_PATH: str = os.environ.get(
    "SMT_RULES_PATH",
    os.path.join(MODULE_DIR, "../rules/schema_implications.json")  # relative fallback
)

# IN/OUT ROOT：只从环境变量读，用作“单独跑脚本”的 fallback。
# 正常 pipeline 中应通过命令行 --in-root / --out-root 显式传入。
ROOT_IN: str  = os.environ.get("IMP_INPUT_ROOT", "")
ROOT_OUT: str = os.environ.get("IMP_OUTPUT_ROOT", "")

# ---- Timeframe token regex (aligned with driver) ------------------------------
_TIMEFRAME_UNITS = r"(?:minutes|hours|days|weeks|months|years)"
_TIMEFRAME_RE = (
    r"(?:now|inthehistory|inthefuture|"
    r"inthepast\d+" + _TIMEFRAME_UNITS + r"|"
    r"inthefuture\d+" + _TIMEFRAME_UNITS + r"|"
    r"foradurationof\d+" + _TIMEFRAME_UNITS + r")"
)
_TF_RX = re.compile(_TIMEFRAME_RE)

# ---- Utilities ----------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.isfile(path): return out
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                out.append(json.loads(s))
            except Exception:
                # best-effort reader; ignore malformed lines
                pass
    return out

def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _copy_passthrough(pin: str, pout: str) -> int:
    """Ensure pout/canonical.enriched.jsonl exists by copying from pin. Returns row count."""
    os.makedirs(pout, exist_ok=True)
    src = None
    for cand in ("canonical.enriched.jsonl", "canonical.jsonl"):
        p = os.path.join(pin, cand)
        if os.path.isfile(p):
            src = p; break
    rows = 0
    dst = os.path.join(pout, "canonical.enriched.jsonl")
    if src:
        with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for ln in fin:
                if ln.strip():
                    rows += 1
                fout.write(ln)
    else:
        open(dst, "w", encoding="utf-8").close()
    return rows

def load_rules(path: Optional[str] = None) -> Dict[str, Any]:
    """Load rules JSON. Returns dict with at least {'rules': [...]}."""
    p = path or RULES_PATH
    if p and os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, dict) and "rules" in doc:
            return doc
    # empty fallback
    return {"rules": []}

# ---- Time window helpers (NEW) -----------------------------------------------
TIME_KEYS = (
    "start_time_in_hours",
    "end_time_in_hours",
    "start_time_inclusive",
    "end_time_inclusive",
)

def _copy_time_fields(src: Dict[str, Any], dst: Dict[str, Any]) -> None:
    for k in TIME_KEYS:
        if k in src:
            dst[k] = src[k]


# ---- Template engine ----------------------------------------------------------
class _Template:
    """
    Convert a rule 'match_template' into a regex.
    Placeholders:
      {t}  timeframe token (OPTIONAL; if present in varname we'll match it,否则不需要)
      {e}  greedy entity segment (.+?) across underscores
    All other characters are treated literally.
    """
    def __init__(self, tmpl: str):
        self.tmpl = tmpl
        self.rx = self._compile(tmpl)

    @staticmethod
    def _escape_lit(s: str) -> str:
        return re.escape(s)

    def _compile(self, tmpl: str) -> re.Pattern:
        parts: List[str] = []
        i = 0
        while i < len(tmpl):
            if tmpl.startswith("{e}", i):
                parts.append(r"(?P<e>.+?)")
                i += 3
                continue
            if tmpl.startswith("{t}", i):
                # 默认把 {t} 变成“可选”。若前一个字面量是 '_'，把它合并进可选组里。
                optional = f"(?:(?P<t>{_TIMEFRAME_RE}))?"
                if parts and parts[-1] == "_":
                    parts.pop()
                    optional = f"(?:(?:_(?P<t>{_TIMEFRAME_RE})))?"
                parts.append(optional)
                i += 3
                continue
            # 其他字面量字符
            ch = tmpl[i]
            parts.append(self._escape_lit(ch))
            i += 1
        pat = "".join(parts) + r"$"
        return re.compile(pat, re.I)

    def match(self, varname: str) -> Optional[re.Match]:
        return self.rx.match(varname)


# ---- Helpers to parse rows ----------------------------------------------------
def _bool_value(row: Dict[str, Any]) -> Optional[bool]:
    """
    Return True/False if the row explicitly encodes a boolean, else None.
    Accepts 'value' or 'extracted_value', and string 'true'/'false'.
    """
    for k in ("extracted_value", "value"):
        if k in row:
            v = row[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                s = v.strip().lower()
                if s == "true": return True
                if s == "false": return False
    return None

def _bool_true(row: Dict[str, Any]) -> bool:
    """Legacy: True iff explicitly true; used by any legacy code paths."""
    bv = _bool_value(row)
    return bv is True

def _varname(row: Dict[str, Any]) -> str:
    v = row.get("entity_variable_name") or row.get("name") or row.get("new_variable_name") or ""
    return str(v)

def _substitute(template: str, e: Optional[str], t: Optional[str]) -> str:
    s = template
    if "{e}" in s:
        s = s.replace("{e}", e or "")
    # 不再在变量名里保留 timeframe；去掉下划线 + {t} 或裸 {t}
    s = re.sub(r"_\{t\}", "", s)
    s = s.replace("{t}", "")
    # 折叠可能出现的重复下划线与收尾下划线
    s = re.sub(r"__+", "_", s).strip("_")
    return s


def _rule_label(rule: Dict[str, Any]) -> str:
    # Prefer explicit id/label; fallback to match_template string.
    return (rule.get("id")
            or rule.get("label")
            or rule.get("match_template")
            or "implication_rule")

# ---- Core per-patient runner --------------------------------------------------
def process_patient(pin: str, pout: str, rules: List[Dict[str, Any]], tf_cfg: Dict[str, Any]) -> Tuple[int,int,int]:
    """
    pin:  patient input dir (expects canonical.enriched.jsonl or canonical.jsonl)
    pout: patient output dir for implications
    rules: list of rule dicts:
      {
        "id": "...",
        "match_template": "patient_sex_is_female_{t}",
        "require_bool": true/false  # exact gating: True→only when src==True; False→only when src==False; omitted→no gating
        "produce": [
          { "template": "patient_sex_is_male_{t}", "type": "Bool", "value": false, "preserve_qualifiers": true },
          ...
        ]
      }
    tf_cfg: timeframe behavior (unused by this minimal engine; reserved)

    Returns: (rows_input, schema_new_rc, timeframe_new_rc)
             schema_new_rc/timeframe_new_rc count produced-variable occurrences per group.
    """
    os.makedirs(pout, exist_ok=True)

    # 1) Ensure passthrough exists for downstream
    in_rows = _copy_passthrough(pin, pout)

    # 2) Load candidate variables (we match by name + optional boolean requirement)
    src_path = os.path.join(pin, "canonical.enriched.jsonl")
    if not os.path.isfile(src_path):
        src_path = os.path.join(pin, "canonical.jsonl")
    rows = _read_jsonl(src_path)

    # 3) Precompile rule templates
    compiled: List[Tuple[Dict[str, Any], _Template]] = []
    for r in rules or []:
        mt = r.get("match_template")
        if isinstance(mt, str) and mt:
            compiled.append((r, _Template(mt)))

    # 4) Structured containers (back-compat + provenance)
    groups_schema: Dict[str, Dict[str, Any]] = {}    # key = source var → {"name","extracted_value","implied":[...]}
    groups_timeframe: Dict[str, Dict[str, Any]] = {}
    pairs: List[Dict[str, Any]] = []

    schema_new_rc = 0
    timeframe_new_rc = 0

    # 5) Apply rules
    for row in rows:
        source_name = _varname(row)
        if not source_name:
            continue

        # timeframe extraction for defaulting when {t} appears but varname lacks a token
        m_tf = _TF_RX.search(source_name)
        default_t = m_tf.group(0) if m_tf else "now"
        src_val = _bool_value(row)  # True / False / None

        for r, mt in compiled:
            m = mt.match(source_name)
            if not m:
                continue

            if r.get("id") == "not_has_undergone_implies_no_positive_outcome_same_time":
                # 源变量已带 outcome → 跳过
                if re.search(r"_outcome_is_(?:positive|negative|abnormal|normal)$", source_name, re.I):
                    continue
            # NEW: exact-value gating semantics for require_bool
            if "require_bool" in r:
                req = r["require_bool"]
                if req is True and src_val is not True:
                    continue
                if req is False and src_val is not False:
                    continue
            # else: no gating, fire regardless of src_val

            t_tok = m.groupdict().get("t") or default_t
            e_tok = m.groupdict().get("e")

            produces = r.get("produce") or []
            for prod in produces:
                tmpl = (prod or {}).get("template")
                if not isinstance(tmpl, str) or not tmpl:
                    continue

                implied_name = _substitute(tmpl, e_tok, t_tok)
                val = bool(prod.get("value", True))
                rule_str = _rule_label(r)

                # --- Build the implied edge (explicit provenance) ---
                implied_item = {
                    "name": implied_name,
                    "value": val,
                    "derivation_stage": "imp",
                    "derivation_rule": rule_str,
                    "source_variable_name": source_name,
                    "target_variable_name": implied_name,
                }
                _copy_time_fields(row, implied_item)  # NEW: 带上 4 个 time 字段


                # --- Back-compat 'groups' structure with explicit 'implied' under each source ---
                container = groups_timeframe if "{t}" in tmpl else groups_schema
                g = container.get(source_name)
                if g is None:
                    g = {"name": source_name}
                    # Record actual boolean if known; useful for audits.
                    if src_val is not None:
                        g["extracted_value"] = src_val
                    container[source_name] = g
                # ensure list exists even if coming from older code paths
                g.setdefault("implied", []).append(implied_item)

                # --- Pairs list (flat, explicit) ---
                pair = {
                    "source_variable_name": source_name,
                    "target_variable_name": implied_name,
                    "value": val,
                    "derivation_stage": "imp",
                    "derivation_rule": rule_str,
                }
                _copy_time_fields(row, pair)  # NEW
                pairs.append(pair)


                # --- Counts for return tuple ---
                if "{t}" in tmpl:
                    timeframe_new_rc += 1
                else:
                    schema_new_rc += 1

    # 6) Write structured file (stable ordering for reproducibility)
    def _sorted_group_values(d: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for k in sorted(d.keys()):
            g = d[k]
            # sort implied by target name for determinism
            g2 = dict(g)
            g2["implied"] = sorted(
                g.get("implied", []),
                key=lambda x: (str(x.get("target_variable_name") or x.get("name") or ""))
            )
            out.append(g2)
        return out

    structured = {
        "groups": {
            "schema": _sorted_group_values(groups_schema),
            "timeframe": _sorted_group_values(groups_timeframe),
        },
        # New explicit edge list (auditing convenience)
        "pairs": pairs,
        "meta": {
            "counts": {
                "schema_implied": schema_new_rc,
                "timeframe_implied": timeframe_new_rc,
                "total_pairs": len(pairs),
            }
        }
    }

    _write_json(os.path.join(pout, "canonical.enriched.implications.structured.json"), structured)


    # --- NEW: write annotated jsonl for downstream (adds `implications` per row) ---
    # build source_name -> implied lists (names + time fields)
    def _group_to_name_items(grps):
        out = {}
        for g in grps:
            src = g.get("name") or ""
            items = []
            for it in g.get("implied", []) or []:
                items.append({
                    "name": it.get("target_variable_name") or it.get("name"),
                    "start_time_in_hours": it.get("start_time_in_hours"),
                    "end_time_in_hours": it.get("end_time_in_hours"),
                    "start_time_inclusive": it.get("start_time_inclusive"),
                    "end_time_inclusive": it.get("end_time_inclusive"),
                })
            # de-dup by name (keep first with times)
            seen = set(); dedup = []
            for x in items:
                n = x.get("name")
                if not n or n in seen: continue
                seen.add(n); dedup.append(x)
            out[src] = dedup
        return out

    schema_map    = _group_to_name_items(_sorted_group_values(groups_schema))
    timeframe_map = _group_to_name_items(_sorted_group_values(groups_timeframe))

    # read from pout's ensured jsonl, write annotated jsonl next to it
    src_jsonl = os.path.join(pout, "canonical.enriched.jsonl")
    ann_jsonl = os.path.join(pout, "canonical.enriched.annotated.jsonl")
    with open(src_jsonl, "r", encoding="utf-8") as fin, open(ann_jsonl, "w", encoding="utf-8") as fout:
        for ln in fin:
            s = ln.strip()
            if not s: 
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            vname = (_varname(row) or "").strip()
            imp = {
                "schema_new_only":    schema_map.get(vname, []),
                "timeframe_new_only": timeframe_map.get(vname, []),
            }
            row["implications"] = imp
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


    return in_rows, schema_new_rc, timeframe_new_rc

# Optional: simple dir-wide runner (not used by the driver, but handy)
def process_dir(in_root: str, out_root: str, rules: List[Dict[str, Any]], tf_cfg: Dict[str, Any]) -> int:
    n = 0
    if not os.path.isdir(in_root): return 0
    for pid in sorted(d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))):
        pin = os.path.join(in_root, pid)
        pout = os.path.join(out_root, pid)
        try:
            rows, _, _ = process_patient(pin, pout, rules, tf_cfg)
            if rows > 0:
                n += 1
        except Exception:
            # keep going; per-patient isolation
            pass
    return n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Apply schema implications per patient or over a directory."
    )
    ap.add_argument(
        "--in-root",
        dest="in_root",
        required=False,
        default=None,
        help=(
            "Input root that contains per-patient dirs. "
            "Required unless IMP_INPUT_ROOT environment variable is set."
        ),
    )
    ap.add_argument(
        "--out-root",
        dest="out_root",
        required=False,
        default=None,
        help=(
            "Output root to write results. "
            "Required unless IMP_OUTPUT_ROOT environment variable is set."
        ),
    )
    ap.add_argument(
        "--rules",
        dest="rules_path",
        required=False,
        default=RULES_PATH,
        help=(
            "Path to schema_implications.json "
            "(default: env SMT_RULES_PATH or module default)."
        ),
    )
    ap.add_argument(
        "--patient",
        dest="patient",
        required=False,
        default=None,
        help="Process only this patient id (dir name under in-root).",
    )
    args = ap.parse_args()

    # ---------- 解析 in_root / out_root：优先命令行，其次环境变量，缺失则报错 ----------
    if args.in_root:
        in_root = args.in_root
    elif ROOT_IN:
        in_root = ROOT_IN
    else:
        print(
            "[apply_schema_implications] --in-root is required "
            "unless IMP_INPUT_ROOT is set.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.out_root:
        out_root = args.out_root
    elif ROOT_OUT:
        out_root = ROOT_OUT
    else:
        print(
            "[apply_schema_implications] --out-root is required "
            "unless IMP_OUTPUT_ROOT is set.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    in_root = os.path.abspath(in_root)
    out_root = os.path.abspath(out_root)

    # load rules
    doc = load_rules(args.rules_path)
    rules = doc.get("rules", [])
    tf_cfg = doc.get("timeframe_implication", {"collapse_timeframes": True})

    if args.patient:
        pin = os.path.join(in_root, args.patient)
        pout = os.path.join(out_root, args.patient)
        os.makedirs(pout, exist_ok=True)
        rows, s_rc, t_rc = process_patient(pin, pout, rules, tf_cfg)
        print(
            f"[apply_schema_implications] patient={args.patient} "
            f"rows={rows} schema_imp={s_rc} time_imp={t_rc}"
        )
    else:
        if not os.path.isdir(in_root):
            print(
                f"[apply_schema_implications] in-root is not a directory: {in_root}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        os.makedirs(out_root, exist_ok=True)
        touched = process_dir(in_root, out_root, rules, tf_cfg)
        print(f"[apply_schema_implications] processed {touched} patient(s).")
