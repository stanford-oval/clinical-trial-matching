#!/usr/bin/env python3
"""Build a per-pair structured atom artifact for the bullet verbalizer.

The v11 hybrid verbalizer was previously fed the LLM-prose-summary from
/tmp/aegis_v4_freeform.jsonl, which selectively mentions only a few atoms
on long-criteria trials. This script reads the SMT pipeline's per-atom
output (cmsrc_out_REMINE_v9_full/<patient>/<trial>__full.json) and emits
a structured artifact the verbalizer can enumerate from directly.

Output schema (one record per pair):
{
  "pair": "sigir-XXXXX__NCTNNNNNNN",
  "verdict": "eligible" | "ineligible",
  "inclusion_atoms": [
    {"atom": "...", "description": "...", "value": true|false|null, "side": "inclusion"},
    ...
  ],
  "exclusion_atoms": [
    {"atom": "...", "description": "...", "value": true|false|null, "side": "exclusion"},
    ...
  ],
  "min_flip_set": [...]  // only for ineligible: which atoms drive the verdict
}

Run:
    python verbalizer/build_atom_artifact.py \
        --pairs experiments/clinician_validation/sample_32pairs_balanced.json \
        --out /tmp/verbalizer_overnight/atom_artifacts.jsonl
"""
from __future__ import annotations
import argparse, json, pathlib, sys
from collections import OrderedDict

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMSRC_OUT = ROOT / "experiments/53_v2_full/cmsrc_out_REMINE_v9_full"
AEGIS_CANONICAL = ROOT / "matchers/systems/aegis/rationales_v9_arbiter.jsonl"


def _load_deciding_variants() -> dict[str, str]:
    """Map pair -> canonical AEGIS deciding variant (e.g., 'NCT00757120b')."""
    out = {}
    if not AEGIS_CANONICAL.exists():
        return out
    for ln in AEGIS_CANONICAL.open():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        pair = o.get("pair")
        dv = o.get("deciding_variant")
        if pair and dv:
            out[pair] = dv
    return out


_DECIDING_VARIANTS: dict[str, str] = {}


def first_existing(path_candidates):
    for p in path_candidates:
        if p.exists():
            return p
    return None


def load_pair_artifact(pid: str, trial: str, pair_id: str | None = None) -> dict | None:
    """Load the SMT full.json for the deciding cohort variant.

    For multi-cohort trials, the canonical AEGIS rationale records the
    deciding variant (e.g., 'NCT00757120b') in its `deciding_variant`
    field. We prefer that variant. Fallback to alphabetic first only if
    no deciding variant is recorded.
    """
    pair_dir = CMSRC_OUT / pid
    if not pair_dir.exists():
        return None
    candidates = sorted(pair_dir.glob(f"{trial}*__full.json"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return json.loads(candidates[0].read_text())
    # Multi-cohort trial — look up deciding variant
    deciding = _DECIDING_VARIANTS.get(pair_id or f"{pid}__{trial}")
    if deciding:
        for c in candidates:
            if c.name.startswith(f"{deciding}__"):
                return json.loads(c.read_text())
    # No deciding variant info — fall back to alphabetic first (note this)
    return json.loads(candidates[0].read_text())


def clean_description(text: str) -> str:
    """Strip the meta-circular bits the program-relax stage adds.

    Stage 1 generates atoms whose description is sometimes verbose meta:
      "Whether <inner description> satisfies <smt_expression>"
      "Indicates whether <criterion>..."
      "Boolean indicating whether <criterion>..."
      "Numeric value (in years) for ... used to assess the inclusion
       requirement of ..."
    Compress these to the criterion-flavored core.
    """
    import re
    if not text: return ""
    s = text.strip()
    # Drop "Whether X satisfies <SMT expression>" → keep just X
    m = re.match(r"^Whether\s+(.+?)\s+satisfies\s+[a-zA-Z_].*$", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    # Strip "Indicates whether " / "Boolean indicating whether " / "Numeric value indicating "
    s = re.sub(r"^(Indicates whether|Boolean indicating whether|Numeric value indicating(?: whether)?|A boolean indicating whether)\s+",
               "", s, flags=re.I)
    # Drop trailing meta-clauses like " used to assess the inclusion requirement of X"
    s = re.sub(r";?\s*used to assess (?:the )?(?:inclusion|exclusion)? ?requirement\s+of\s+[^.]+\.?\s*$",
               ".", s, flags=re.I)
    # Drop trailing "as estimated/documented by a clinician" / "at the time of evaluation/screening"
    s = re.sub(r"\s+(?:as\s+(?:estimated|documented)[^.]*|at\s+the\s+time\s+of[^.]*)\.?\s*$", ".", s, flags=re.I)
    # Final tidy
    s = re.sub(r"\s+", " ", s).strip()
    # Capitalize first letter for readability
    if s and s[0].islower(): s = s[0].upper() + s[1:]
    return s


def resolve_threshold(atom_name: str, value_lookup: dict) -> bool | None:
    """For `__THRESH__::<var>::<op>::<thresh>` atoms, compute the comparison
    from the underlying value atom if it's known. Returns True/False if
    computable, None otherwise.
    """
    import re
    m = re.match(r"^__THRESH__::([^:]+(?:::[^:]+)*?)::(eq|ne|lt|le|gt|ge)::([\-\d.]+)$", atom_name)
    if not m:
        return None
    var, op, thresh = m.group(1), m.group(2), m.group(3)
    try:
        thresh_v = float(thresh)
    except ValueError:
        return None
    raw_val = value_lookup.get(var)
    if raw_val is None:
        return None
    try:
        v = float(raw_val)
    except (TypeError, ValueError):
        return None
    cmp = {"eq": v == thresh_v, "ne": v != thresh_v,
           "lt": v < thresh_v,  "le": v <= thresh_v,
           "gt": v > thresh_v,  "ge": v >= thresh_v}.get(op)
    return cmp


def extract_side(side_data: dict, side_label: str) -> list[dict]:
    """Pull atoms from inclusion/exclusion subtree of the full.json.

    Threshold atoms (`__THRESH__::<var>::<op>::<thresh>`) are resolved
    when the underlying numeric atom has a value. This handles the
    common case where Stage 1 splits "age > 16" into a value atom
    (mined: 65) and a threshold atom (null until comparison is run).
    """
    raw = side_data.get("raw", {})
    leaf_detail = raw.get("leaf_detail", {}) or {}
    values = dict(raw.get("patient_var_values", {}) or {})
    # First pass: resolve thresholds against the value lookup.
    for atom_name in list(leaf_detail.keys()):
        if atom_name.startswith("__THRESH__::") and values.get(atom_name) is None:
            resolved = resolve_threshold(atom_name, values)
            if resolved is not None:
                values[atom_name] = resolved
    out = []
    for atom_name, detail in leaf_detail.items():
        description = ""
        if isinstance(detail, dict):
            description = (detail.get("description") or "").strip()
        value = values.get(atom_name)
        # Friendlier description for resolved threshold atoms
        if atom_name.startswith("__THRESH__::"):
            import re
            m = re.match(r"^__THRESH__::[^:]+(?:::[^:]+)*?::(eq|ne|lt|le|gt|ge)::([\-\d.]+)$", atom_name)
            if m:
                op_label = {"eq":"=", "ne":"≠", "lt":"<", "le":"≤", "gt":">", "ge":"≥"}[m.group(1)]
                description = f"{description.rstrip('.')}; threshold check {op_label} {m.group(2)}"
        out.append({
            "atom": atom_name,
            "description": clean_description(description)[:240],
            "value": value,
            "side": side_label,
        })
    return out


def extract_min_flip_set(side_data: dict) -> list[str]:
    """Return atom names from the SMT unsat-core for unsat sides.

    The cmsrc pipeline stores the unsat-core under `eval_result.unsat_core`.
    The encoder prefixes user-facing atom names with `patient_` for
    namespacing; entries beginning with `REQ`/`AUXILIARY` are constraint
    labels, not atoms. We strip the prefix and filter to true atoms only.
    """
    raw = side_data.get("raw", {})
    eval_result = raw.get("eval_result") or {}
    # Try several plausible field names
    core = (eval_result.get("unsat_core")
            or eval_result.get("min_flip_set")
            or eval_result.get("blocking_atoms")
            or [])
    if not isinstance(core, list):
        return []
    leaf_keys = set((raw.get("leaf_detail") or {}).keys())
    out = []
    for entry in core:
        s = str(entry)
        # Skip constraint labels
        if s.startswith(("REQ", "AUXILIARY", "COMPONENT")):
            continue
        # Strip leading "patient_" if present and the unprefixed form is a known atom
        cand = s
        if cand.startswith("patient_") and cand[len("patient_"):] in leaf_keys:
            cand = cand[len("patient_"):]
        elif cand not in leaf_keys:
            # Last-ditch: match any leaf that ends with this name's body
            matches = [k for k in leaf_keys if cand.endswith(k) or k.endswith(cand)]
            if matches:
                cand = matches[0]
        out.append(cand)
    # Dedupe while preserving order
    seen = set(); dedup = []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup


def load_all_cohorts(pid: str, trial: str) -> list[tuple[str, dict]]:
    """Return [(variant_id, full_json_dict), ...] for every cohort variant
    of this trial, sorted by variant id."""
    pair_dir = CMSRC_OUT / pid
    if not pair_dir.exists():
        return []
    out = []
    for path in sorted(pair_dir.glob(f"{trial}*__full.json")):
        # variant_id like "NCT01161849b"
        variant_id = path.name.replace("__full.json", "")
        try:
            out.append((variant_id, json.loads(path.read_text())))
        except Exception:
            continue
    return out


def count_trial_criteria(trial_text: str) -> tuple[int, int]:
    """Count inclusion + exclusion criteria in the trial's human-readable
    criteria text.

    Counts items separated by blank lines within the Inclusion/Exclusion
    sections (ClinicalTrials.gov format) or numbered/bulleted lines.
    Returns (n_inclusion, n_exclusion).
    """
    import re
    if not trial_text:
        return 0, 0
    lo = trial_text.lower()
    i = lo.find("inclusion criteria")
    e = lo.find("exclusion criteria")

    def count_items(blob: str) -> int:
        if not blob: return 0
        # Strip the leading "Inclusion criteria:" / "Exclusion criteria:" header
        blob = re.sub(r"^[^\n]*(inclusion|exclusion)\s*criteria\s*:[^\n]*\n?", "", blob,
                      count=1, flags=re.I)
        # Split on blank lines and count non-empty items
        items = [seg.strip() for seg in re.split(r"\n\s*\n", blob) if seg.strip()]
        # Each item is one criterion. Strip stray colons.
        items = [it for it in items if len(it) > 3]
        return len(items)

    inc_blob = trial_text[i:e] if i >= 0 and e > i else (trial_text[i:] if i >= 0 else "")
    exc_blob = trial_text[e:] if e >= 0 else ""
    return count_items(inc_blob), count_items(exc_blob)


def build_for_pair(pair: str) -> dict | None:
    """Build a cohort-aware atom artifact.

    Pair-level verdict = disjunction over cohorts:
      eligible = any cohort SAT
    For eligible verdicts we mark the SAT cohort(s) as "qualifying"; for
    ineligible verdicts every cohort is failing, and we keep the
    per-cohort min-flip set so the verbalizer can enumerate per-arm
    blockers."""
    try:
        pid, trial = pair.split("__", 1)
    except ValueError:
        return None

    cohorts_raw = load_all_cohorts(pid, trial)
    if not cohorts_raw:
        return None

    deciding = _DECIDING_VARIANTS.get(pair)
    cohorts = []
    any_eligible = False
    for variant_id, data in cohorts_raw:
        inc_sat = data.get("inclusion", {}).get("sat_like")
        exc_sat = data.get("exclusion", {}).get("sat_like")
        v_eligible = bool(data.get("eligible") or data.get("eligible_strict") or
                          (inc_sat is True and exc_sat is True))
        any_eligible = any_eligible or v_eligible
        cohorts.append({
            "variant_id": variant_id,
            "qualifying": v_eligible,
            "is_deciding": (variant_id == deciding),
            "inclusion_sat": inc_sat,
            "exclusion_sat": exc_sat,
            "inclusion_atoms": extract_side(data.get("inclusion", {}), "inclusion"),
            "exclusion_atoms": extract_side(data.get("exclusion", {}), "exclusion"),
            "min_flip_set": (extract_min_flip_set(data.get("inclusion", {})) +
                             extract_min_flip_set(data.get("exclusion", {}))) if not v_eligible else [],
        })

    pair_verdict = "eligible" if any_eligible else "ineligible"

    # Top-level union for backward-compat consumers: pick the deciding
    # cohort (or the first qualifying one if eligible) and re-expose its
    # atoms at the top level. New consumers should use `cohorts` directly.
    primary = next((c for c in cohorts if c["is_deciding"]), None)
    if primary is None:
        primary = next((c for c in cohorts if c["qualifying"]), cohorts[0])

    return {
        "pair": pair,
        "verdict": pair_verdict,
        "n_cohorts": len(cohorts),
        "cohorts": cohorts,
        # backward-compat fields
        "inclusion_sat": primary.get("inclusion_sat"),
        "exclusion_sat": primary.get("exclusion_sat"),
        "inclusion_atoms": primary.get("inclusion_atoms", []),
        "exclusion_atoms": primary.get("exclusion_atoms", []),
        "min_flip_set": primary.get("min_flip_set", []),
        "primary_variant_id": primary.get("variant_id"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="Path to a JSON list of {pair: ...} entries OR a JSON list of pair strings.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load the deciding-variant lookup once, before processing pairs
    global _DECIDING_VARIANTS
    _DECIDING_VARIANTS = _load_deciding_variants()
    print(f"Loaded deciding-variant info for {len(_DECIDING_VARIANTS)} canonical pairs")

    sample = json.load(open(args.pairs))
    if isinstance(sample, list) and sample and isinstance(sample[0], dict):
        pair_list = [r["pair"] for r in sample if r.get("pair")]
    elif isinstance(sample, list):
        pair_list = list(sample)
    else:
        sys.exit("--pairs must be a list of pair strings or {pair: ...} dicts")

    pair_list = list(OrderedDict.fromkeys(pair_list))
    print(f"Building atom artifacts for {len(pair_list)} pairs")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = []
    with out_path.open("w") as f:
        for p in pair_list:
            rec = build_for_pair(p)
            if rec is None:
                missing.append(p)
                continue
            f.write(json.dumps(rec) + "\n")
            written += 1

    print(f"wrote {written} -> {out_path}")
    if missing:
        print(f"missing artifacts for {len(missing)} pairs:")
        for p in missing[:5]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
