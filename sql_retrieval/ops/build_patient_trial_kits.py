#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, logging, os, shutil, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

"""
make_sigir_out_kits.py — Build OG-style kits under a dedicated subfolder per patient,
and ALSO write a survivors-only OG bundle (both preserve original ranks).

Bundles written (defaults):
  - O_original_default/     : all original items (our order)
  - O_original_survivors/   : only survivors from the original list

Key points:
- Rank folder names use the original `_best_rank_from_subs` where available.
  If missing, we fall back to enumeration order.
- Structure under each rank mirrors your existing make_sigir_out_kits pipeline.

Examples:
  python make_sigir_out_kits.py \
    --project-root /path/to/TrialGPT-SM T \
    --retrieved-root retrieved_mappings \
    --out /path/to/out_sigir_kits_threeway \
    --bundle-dir O_original_default \
    --bundle-dir-survivors O_original_survivors \
    --patient sigir-20141 \
    --corpus /path/to/dataset/clinical_trial/sigir/corpus.jsonl \
    --queries /path/to/dataset/clinical_trial/sigir/queries.jsonl
"""

LOG = logging.getLogger("make_sigir_out_kits")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------------------------
# FS helpers
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _rel_symlink_or_copy(src: Path, dst: Path) -> None:
    """Create relative symlink; fallback to copy. Skip if src missing."""
    if not src.exists():
        LOG.warning("[skip] missing: %s", src)
        return
    _ensure_dir(dst.parent)
    try:
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(rel)
    except Exception as e:
        LOG.warning("[fallback->copy] %s → %s  (%s)", src, dst, e)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

# ----------------------------
# NCT helpers
# ----------------------------

def _is_subnct(nct: str) -> bool:
    import re
    return bool(re.match(r"^NCT\d{8}[a-d]$", nct))

def _canon8(nct: str) -> str:
    import re
    m = re.match(r"^(NCT\d{8})", nct)
    return m.group(1) if m else nct

# ----------------------------
# Loaders
# ----------------------------

def build_rank_units_from_detailed_with_ranks(retrieved_root: Path, patient: str) -> List[Dict[str, Any]]:
    p = retrieved_root / "json" / f"{patient}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing detailed json for patient: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = data.get("trials", []) or []

    for i, r in enumerate(rows):
        r["_seq"] = i

    groups: Dict[str, Dict[str, Any]] = {}

    def _status_rank(s: str) -> int:
        s = (s or "").lower()
        return {"survivor": 0, "mixed": 1, "eliminated": 2}.get(s, 3)

    def _to_float(x, default=float("inf")):
        try:
            return float(x)
        except Exception:
            return default

    for r in rows:
        nct_id = str(r.get("nct_id", "")).strip()
        if not nct_id:
            continue
        canon = _canon8(nct_id)
        g = groups.setdefault(canon, {
            "canon": canon,
            "children": [],
            "status_candidates": [],
            "disease_vars": set(),
            "literal_vars": set(),
        })
        g["children"].append(r)
        g["status_candidates"].append(r.get("status", ""))
        for v in (r.get("disease_vars") or []):
            g["disease_vars"].add(v)
        for v in (r.get("literal_vars") or []):
            g["literal_vars"].add(v)

    units: List[Dict[str, Any]] = []
    for canon, g in groups.items():
        kids: List[Dict[str, Any]] = g["children"]

        def rep_key(r: Dict[str, Any]):
            return (
                int(r.get("rank", 10**9)),
                _to_float(r.get("frac_unsat_any")),
                _to_float(r.get("unsat_any")),
                int(r.get("_seq", 10**9)),
            )

        rep = min(kids, key=rep_key)
        status = sorted(g["status_candidates"], key=_status_rank)[0] if g["status_candidates"] else ""

        sorted_kids = sorted(kids, key=lambda r: (int(r.get("rank", 10**9)), int(r.get("_seq", 10**9))))
        sub_nct_ids = [r["nct_id"] for r in sorted_kids if _is_subnct(r.get("nct_id", ""))]
        trial_ids   = [int(r["trial_id"]) for r in sorted_kids if str(r.get("trial_id", "")).isdigit()]

        unit = {
            "canonical_nct_id": canon,
            "status": status,
            "rep_trial_id": rep.get("trial_id"),
            "rep_status": rep.get("status"),
            "total_clauses": rep.get("total_clauses"),
            "unsat_any": rep.get("unsat_any"),
            "unsat_explicit": rep.get("unsat_explicit"),
            "frac_unsat_any": rep.get("frac_unsat_any"),
            "pct_unsat_any": rep.get("pct_unsat_any"),
            "pct_unsat_explicit": rep.get("pct_unsat_explicit"),
            "sub_nct_ids": sub_nct_ids,
            "trial_ids": trial_ids,
            "disease_vars": sorted(g["disease_vars"]),
            "literal_vars": sorted(g["literal_vars"]),
            "_best_rank_from_subs": int(rep.get("rank", 10**9)),
        }
        units.append(unit)

    # Keep our natural order by original best-sub rank, tie on NCT
    units.sort(key=lambda u: (u["_best_rank_from_subs"], u["canonical_nct_id"]))
    return units

def load_merged_rows(retrieved_root: Path, patient: str) -> List[Dict[str, Any]]:
    p = retrieved_root / "merged_json" / f"{patient}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing merged_json for patient: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("trials", [])

def load_detailed_rows(retrieved_root: Path, patient: str) -> Dict[int, Dict[str, Any]]:
    p = retrieved_root / "json" / f"{patient}.json"
    if not p.exists():
        LOG.warning("[warn] missing detailed json for patient: %s", p)
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("trials", [])
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        try:
            out[int(r["trial_id"])] = r
        except Exception:
            pass
    return out

def load_corpus_index(corpus_path: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not corpus_path or not corpus_path.exists():
        LOG.warning("[warn] corpus not found: %s", corpus_path)
        return idx
    n_total = 0
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                nct_canon = _canon8(str(obj.get("_id", "")).strip())
                if nct_canon:
                    idx[nct_canon] = obj
                    n_total += 1
            except Exception:
                continue
    LOG.info("[corpus] loaded %d docs (unique=%d) from %s", n_total, len(idx), corpus_path)
    return idx

def load_queries_index(queries_path: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not queries_path or not queries_path.exists():
        LOG.warning("[warn] queries not found: %s", queries_path)
        return idx
    n_total = 0
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = str(obj.get("_id", "")).strip()
                if pid:
                    idx[pid] = obj
                    n_total += 1
            except Exception:
                continue
    LOG.info("[queries] loaded %d notes (unique=%d) from %s", n_total, len(idx), queries_path)
    return idx

# ----------------------------
# Path planners per NCT
# ----------------------------

def per_nct_sources(project_root: Path, nct: str) -> Dict[str, List[Path]]:
    IR  = project_root / "build" / "ir"
    MINC = project_root / "build" / "minified_canon"
    RED  = project_root / "build" / "canon_projection" / "_projected_smt"
    DIS  = project_root / "build" / "disease"

    nct_canon = _canon8(nct)

    return {
        "smt_results": [
            IR / f"{nct}_inclusion_program.smt2",
            IR / f"{nct}_exclusion_program.smt2",
        ],
        "minified_canon": [
            MINC / f"{nct}_inclusion_canonical_variables_entity_variable_names.json",
            MINC / f"{nct}_exclusion_canonical_variables_entity_variable_names.json",
        ],
        "reduced_smt": [
            RED / f"{nct}_inclusion_program_canon_projected.smt2",
            RED / f"{nct}_inclusion_program.assumed_canon_projected.smt2",
            RED / f"{nct}_exclusion_program_canon_projected.smt2",
        ],
        "disease_list": [
            DIS / f"{nct_canon}_disease_link_filter_summary.json",
        ],
    }

# ----------------------------
# Writers
# ----------------------------

def write_relevance_files(dst_dir: Path, disease_vars: List[str], literal_vars: List[str]) -> None:
    rel_dir = dst_dir / "relevance_stage"
    _ensure_dir(rel_dir)
    (rel_dir / "disease_hits").write_text(
        "\n".join(sorted(set(disease_vars))) + ("\n" if disease_vars else ""),
        encoding="utf-8"
    )
    (rel_dir / "literal_hits").write_text(
        "\n".join(sorted(set(literal_vars))) + ("\n" if literal_vars else ""),
        encoding="utf-8"
    )

def link_patient_inputs(dst_dir: Path, project_root: Path, patient: str) -> None:
    facts_export_dir = project_root / "patient_build_inclusion" / "patient_facts_export" / patient / "inclusion"
    coded_results_dir = project_root / "patient_build_inclusion" / "patient_coded_results" / patient

    _rel_symlink_or_copy(facts_export_dir / "canonical.jsonl", dst_dir / "0patient_coded_results" / "canonical.jsonl")
    _rel_symlink_or_copy(coded_results_dir / "demographics.jsonl", dst_dir / "0patient_coded_results" / "demographics.jsonl")

def _write_metainfojson(dst_dir: Path, project_root: Path, nct: str) -> None:
    nct_canon = _canon8(nct)
    disease_json = project_root / "build" / "disease" / f"{nct_canon}_disease_link_filter_summary.json"
    proj_dir = project_root / "build" / "canon_projection" / "_projected_smt"
    projected_incl = proj_dir / f"{nct}_inclusion_program_canon_projected.smt2"
    projected_excl = proj_dir / f"{nct}_exclusion_program_canon_projected.smt2"

    payload = {"trial_id": nct_canon, "contextual": {}}

    if disease_json.exists():
        try:
            data = json.loads(disease_json.read_text(encoding="utf-8"))
            obj = data[0] if isinstance(data, list) and data else data
            payload["trial_id"] = obj.get("trial_id", nct_canon)
            payload["contextual"] = obj.get("contextual", {}) | {}
        except Exception as e:
            LOG.info("[metainfojson] parse failed: %s (%s); writing minimal.", disease_json, e)
            payload["contextual"] = {
                "note": "disease_link_filter_summary.json parse failed; minimal metainfo written.",
                "has_projected_inclusion_smt": projected_incl.exists(),
                "has_projected_exclusion_smt": projected_excl.exists(),
            }
    else:
        payload["contextual"] = {
            "note": "disease_link_filter_summary.json not found; minimal metainfo written.",
            "has_projected_inclusion_smt": projected_incl.exists(),
            "has_projected_exclusion_smt": projected_excl.exists(),
        }

    out_dir = dst_dir / "1trial" / "metainfojson"
    _ensure_dir(out_dir)
    (out_dir / "metainfo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def _extract_str(meta: Dict[str, Any], key: str) -> str:
    val = meta.get(key)
    return val.strip() if isinstance(val, str) else ""

def write_corpus_bundle(dst_dir: Path, nct: str, corpus_idx: Dict[str, Dict[str, Any]]) -> None:
    if not corpus_idx:
        return
    nct_canon = _canon8(nct)
    doc = corpus_idx.get(nct_canon)
    if not doc:
        return

    out_dir = dst_dir / "1trial" / "corpus"
    _ensure_dir(out_dir)

    (out_dir / "corpus.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    title = str(doc.get("title") or doc.get("metadata", {}).get("brief_title") or nct_canon)
    text = str(doc.get("text") or "").strip()
    meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
    brief_summary = _extract_str(meta, "brief_summary")
    inclusion = _extract_str(meta, "inclusion_criteria")
    exclusion = _extract_str(meta, "exclusion_criteria")

    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"**NCT**: `{nct_canon}`\n")
    if brief_summary:
        lines.append("## Brief Summary\n")
        lines.append(brief_summary + "\n")
    if inclusion:
        lines.append("## Inclusion Criteria\n")
        lines.append(inclusion + "\n")
    if exclusion:
        lines.append("## Exclusion Criteria\n")
        lines.append(exclusion + "\n")
    if text and (not brief_summary or ("Summary" in text[:50] or "Inclusion" in text[:50] or "Exclusion" in text[:50])):
        lines.append("## Full Text\n")
        lines.append(text + "\n")

    (out_dir / "corpus.md").write_text("".join(lines), encoding="utf-8")

def write_patient_note(dst_dir: Path, patient_id: str, queries_idx: Dict[str, Dict[str, Any]]) -> None:
    if not queries_idx:
        return
    doc = queries_idx.get(patient_id)
    if not doc:
        return

    out_dir = dst_dir / "0patient_note"
    _ensure_dir(out_dir)

    (out_dir / "patient_note.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    text = str(doc.get("text", "")).strip()
    lines = [
        f"# Patient Note for `{patient_id}`\n\n",
        text + ("\n" if text else ""),
    ]
    (out_dir / "patient_note.md").write_text("".join(lines), encoding="utf-8")

def materialize_trial_block(dst_dir: Path, project_root: Path, nct: str,
                            corpus_idx: Dict[str, Dict[str, Any]],
                            patient_id: str,
                            queries_idx: Dict[str, Dict[str, Any]]) -> None:
    srcs = per_nct_sources(project_root, nct)
    for p in srcs["smt_results"]:
        _rel_symlink_or_copy(p, dst_dir / "1trial" / "smt_results" / p.name)
    for p in srcs["minified_canon"]:
        _rel_symlink_or_copy(p, dst_dir / "1trial" / "minified_canon" / p.name)
    for p in srcs["reduced_smt"]:
        _rel_symlink_or_copy(p, dst_dir / "1trial" / "reduced_smt" / p.name)
    for p in srcs["disease_list"]:
        _rel_symlink_or_copy(p, dst_dir / "1trial" / "disease_list" / p.name)
    _write_metainfojson(dst_dir, project_root, nct)
    write_corpus_bundle(dst_dir, nct, corpus_idx)
    write_patient_note(dst_dir, patient_id, queries_idx)

def write_rank_metrics(dst_dir: Path, merged_row: Dict[str, Any]) -> None:
    payload = {
        "canonical_nct_id": merged_row.get("canonical_nct_id", ""),
        "status": merged_row.get("status", ""),
        "rep_trial_id": merged_row.get("rep_trial_id"),
        "rep_status": merged_row.get("rep_status"),
        "total_clauses": merged_row.get("total_clauses"),
        "unsat_any": merged_row.get("unsat_any"),
        "unsat_explicit": merged_row.get("unsat_explicit"),
        "frac_unsat_any": merged_row.get("frac_unsat_any"),
        "pct_unsat_any": merged_row.get("pct_unsat_any"),
        "pct_unsat_explicit": merged_row.get("pct_unsat_explicit"),
    }
    if "_best_rank_from_subs" in merged_row:
        payload["best_sub_rank"] = merged_row["_best_rank_from_subs"]

    (dst_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if str(merged_row.get("status", "")).lower() == "eliminated":
        reason = (
            "This canonical NCT is labeled 'eliminated' because ALL of its subcohorts "
            "were explicitly contradicted in the eliminate() stage (explicit contradiction). "
            "Granular clause-level reasons are not recorded by the current pipeline."
        )
        (dst_dir / "eliminate_reason.txt").write_text(reason + "\n", encoding="utf-8")

def write_subnct_metrics(sub_dir: Path, detailed_row: Optional[Dict[str, Any]]) -> None:
    if not detailed_row:
        return
    payload = {
        "trial_id": detailed_row.get("trial_id"),
        "nct_id": detailed_row.get("nct_id", ""),
        "status": detailed_row.get("status", ""),
        "total_clauses": detailed_row.get("total_clauses"),
        "unsat_any": detailed_row.get("unsat_any"),
        "unsat_explicit": detailed_row.get("unsat_explicit"),
        "frac_unsat_any": detailed_row.get("frac_unsat_any"),
        "pct_unsat_any": detailed_row.get("pct_unsat_any"),
        "pct_unsat_explicit": detailed_row.get("pct_unsat_explicit"),
    }
    (sub_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if str(detailed_row.get("status", "")).lower() == "eliminated":
        reason = (
            "This subcohort (trial) is labeled 'eliminated' due to an explicit contradiction "
            "detected by eliminate(). Clause-level details are not captured in this build."
        )
        (sub_dir / "eliminate_reason.txt").write_text(reason + "\n", encoding="utf-8")

# ----------------------------
# Build one rank
# ----------------------------

def build_one_rank(
    out_root: Path,
    project_root: Path,
    patient: str,
    rank_idx: int,
    merged_row: Dict[str, Any],
    detailed_map: Dict[int, Dict[str, Any]],
    corpus_idx: Dict[str, Dict[str, Any]],
    queries_idx: Dict[str, Dict[str, Any]],
    bundle_dir: Optional[str] = None,
) -> None:
    canonical_nct = str(merged_row.get("canonical_nct_id") or "").strip()
    sub_ncts = [s for s in merged_row.get("sub_nct_ids", []) if _is_subnct(s)]
    disease_vars = list(merged_row.get("disease_vars", []))
    literal_vars = list(merged_row.get("literal_vars", []))

    base_dir = out_root / patient
    if bundle_dir:
        base_dir = base_dir / bundle_dir

    rank_dir = base_dir / f"rank{rank_idx}_{_canon8(canonical_nct)}"
    _ensure_dir(rank_dir)

    link_patient_inputs(rank_dir, project_root, patient)
    write_rank_metrics(rank_dir, merged_row)

    if sub_ncts:
        trial_ids = [int(t) for t in merged_row.get("trial_ids", []) if str(t).isdigit()]
        for i, sub in enumerate(sub_ncts):
            subdir = rank_dir / sub
            _ensure_dir(subdir)
            materialize_trial_block(subdir, project_root, sub, corpus_idx, patient, queries_idx)
            write_relevance_files(subdir, disease_vars, literal_vars)
            dr = detailed_map.get(trial_ids[i]) if i < len(trial_ids) else None
            write_subnct_metrics(subdir, dr)
    else:
        materialize_trial_block(rank_dir, project_root, _canon8(canonical_nct), corpus_idx, patient, queries_idx)
        write_relevance_files(rank_dir, disease_vars, literal_vars)

# ----------------------------
# Patient discovery
# ----------------------------

def _discover_patients(retrieved_root: Path) -> List[str]:
    """Return unique patient IDs from retrieved_mappings/{json,merged_json}/*.json (stems)."""
    found: set[str] = set()
    for sub in ("json", "merged_json"):
        pdir = retrieved_root / sub
        if not pdir.exists():
            continue
        for p in pdir.glob("*.json"):
            if p.is_file():
                found.add(p.stem)
    return sorted(found)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="OG kit builder with survivors-only bundle.")
    ap.add_argument("--patient", action="append", default=[],
                    help="Patient ID to process (e.g., sigir-20141). Repeatable. "
                         "If omitted, ALL discovered patients are processed by default.")
    ap.add_argument("--project-root", type=Path, default=Path("../../"),
                    help="Path to TrialGPT-SMT project root (absolute)")
    ap.add_argument("--retrieved-root", type=Path, default=Path("retrieved_mappings"),
                    help="Path to retrieved_mappings root (default: ./retrieved_mappings)")
    ap.add_argument("--out", type=Path, default=Path("out_sigir_kits_threeway"),
                    help="Output directory (relative or absolute)")
    ap.add_argument("--bundle-dir", type=str, default="O_original_default",
                    help="Write OG output under this subfolder of each patient (default: O_original_default)")
    ap.add_argument("--bundle-dir-survivors", type=str, default="O_original_survivors",
                    help="Write OG survivors-only output under this subfolder (default: O_original_survivors)")
    ap.add_argument("--corpus", type=Path,
                    default=Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus.jsonl"),
                    help="Path to corpus.jsonl (JSON lines)")
    ap.add_argument("--queries", type=Path,
                    default=Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/queries.jsonl"),
                    help="Path to queries.jsonl (patient notes)")

    args = ap.parse_args()

    project_root: Path = args.project_root.resolve()
    retrieved_root: Path = args.retrieved_root
    out_root: Path = args.out
    corpus_path: Path = args.corpus
    queries_path: Path = args.queries

    bundle_dir_default: str = args.bundle_dir
    bundle_dir_surv: str = args.bundle_dir_survivors

    # Determine patients: default = ALL discovered
    patients: List[str] = list(dict.fromkeys(args.patient))  # dedupe, keep order if provided
    if not patients:
        patients = _discover_patients(retrieved_root)
        if not patients:
            LOG.info("No patients discovered under %s/{json,merged_json}. Nothing to do.", retrieved_root)
            return
        LOG.info("[discover] %d patients found: %s", len(patients), ", ".join(patients))

    # Heavy, shared loads done once
    corpus_idx  = load_corpus_index(corpus_path)
    queries_idx = load_queries_index(queries_path)

    for patient in patients:
        LOG.info("\n=== Building OG kits for patient: %s ===", patient)
        try:
            # Prefer detailed-based rank-units
            rank_units = build_rank_units_from_detailed_with_ranks(retrieved_root, patient)
            if not rank_units:
                LOG.info("No ranks built from detailed json for patient=%s", patient)
                continue

            detailed_map = load_detailed_rows(retrieved_root, patient)

            # ---- O_original_default (ALL) ----
            for enum_idx, mr in enumerate(rank_units, start=1):
                rank_idx = int(mr.get("_best_rank_from_subs", enum_idx))
                build_one_rank(out_root, project_root, patient, rank_idx, mr, detailed_map, corpus_idx, queries_idx, bundle_dir=bundle_dir_default)

            # ---- O_original_survivors (ONLY survivors) ----
            survivors = [u for u in rank_units if str(u.get("status", "")).lower() == "survivor"]
            for enum_idx, mr in enumerate(survivors, start=1):
                rank_idx = int(mr.get("_best_rank_from_subs", enum_idx))
                build_one_rank(out_root, project_root, patient, rank_idx, mr, detailed_map, corpus_idx, queries_idx, bundle_dir=bundle_dir_surv)

            LOG.info("[done] wrote OG kits under: %s", (out_root / patient / bundle_dir_default).resolve())
            LOG.info("[done] wrote OG survivors under: %s", (out_root / patient / bundle_dir_surv).resolve())
        except FileNotFoundError as e:
            LOG.warning("[skip] %s", e)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            LOG.exception("[error] building patient %s failed: %s", patient, e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
