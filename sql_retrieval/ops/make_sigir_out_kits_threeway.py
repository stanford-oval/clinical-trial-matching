#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sigir_out_kits_threeway.py — Build five bundles per patient:
  A) baseline's fetched (ranked by BASELINE original order)
  B) ours NOT baseline (ranked by OUR original order)
  C) baseline NOT ours (ranked by BASELINE original order)
  D) subset of B: ONLY survivors (ranked by OUR original order)
  E) ours eliminated ∩ baseline (ranked by BASELINE original order)

Key behaviors:
- Preserves original rank indices from the source lists.
  • A, C, E use the baseline's original positions.
  • B, D use our original positions (_best_rank_from_subs from detailed json).
- Subcohort collapsing for baseline: normalize to base NCT######## and keep
  the first occurrence (i.e., highest-ranked among variants).
- Materializes the same per-rank kit structure reused from make_sigir_out_kits.py.
  If a trial only exists in baseline (not in our merged/detailed rows),
  it still gets a minimal kit with a small metrics marker.

Supports baseline input either as:
  --baseline-json  /path/to/trialgpt_retrieve.json   (preferred: {patient_id: [trial_ids...]})
  --baseline-clean-dir retrieved_mappings/clean      (legacy: patient.txt, one NCT per line)

Additional mode:
- With --cap-to-our-survivors, we also compute capped variants B_cap and C_cap, where both
  our list and the baseline list are effectively truncated to K = # of OUR survivors
  (ours: by our original order; baseline: by baseline original order). Original (uncapped)
  folders are still produced in all cases.
"""

from __future__ import annotations
import argparse, json, logging, sys, re
from pathlib import Path
from typing import Dict, List, Set, Optional

# Import helpers from your existing script
from build_patient_trial_kits import (
    build_rank_units_from_detailed_with_ranks,
    load_detailed_rows,
    load_corpus_index,
    load_queries_index,
    build_one_rank,
    _ensure_dir,  # type: ignore
)

LOG = logging.getLogger("make_sigir_out_kits_threeway")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------------------------
# NCT helpers
# ----------------------------

def _canon8(nct: str) -> str:
    m = re.match(r"^(NCT\d{8})", str(nct))
    return m.group(1) if m else str(nct)

def _collapse_to_base(seq: List[str]) -> List[str]:
    """Normalize tokens to base NCT######## and keep first occurrence per base."""
    ranked: List[str] = []
    seen: Set[str] = set()
    for tok in seq:
        s = str(tok).strip().upper()
        if not s:
            continue
        m = re.search(r"(NCT\d{8})", s)
        if not m:
            continue
        base = m.group(1)
        if base in seen:
            continue
        seen.add(base)
        ranked.append(base)
    return ranked

# ----------------------------
# Baseline loaders
# ----------------------------

def load_baseline_index_json(baseline_json: Path) -> Dict[str, List[str]]:
    """Load JSON mapping {patient_id: [trial_ids...]} and collapse each list."""
    if not baseline_json or not baseline_json.exists():
        LOG.warning("[baseline] missing json: %s", baseline_json)
        return {}
    try:
        data = json.loads(baseline_json.read_text(encoding="utf-8"))
    except Exception as e:
        LOG.error("[baseline] failed to parse %s: %s", baseline_json, e)
        return {}
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for pid, arr in data.items():
            if isinstance(arr, list):
                out[str(pid)] = _collapse_to_base([str(x) for x in arr])
    else:
        LOG.error("[baseline] JSON must be an object mapping patient_id -> list[str]")
    return out

def load_baseline_ranked_from_dir(baseline_clean_dir: Path, patient: str) -> List[str]:
    """Legacy: read clean/{patient}.txt and collapse to base NCTs."""
    p = baseline_clean_dir / f"{patient}.txt"
    if not p.exists():
        LOG.warning("[baseline] missing clean list: %s", p)
        return []
    with p.open("r", encoding="utf-8") as f:
        return _collapse_to_base([line.strip() for line in f if line.strip()])

# ----------------------------
# Minimal materializer for trials lacking our merged row
# ----------------------------

def materialize_minimal(
    out_root: Path,
    project_root: Path,
    patient: str,
    rank_idx: int,
    base_nct: str,
    corpus_idx: Dict[str, dict],
    queries_idx: Dict[str, dict],
) -> None:
    """Construct a rank dir with patient inputs + trial block (no detailed metrics)."""
    from build_patient_trial_kits import (
        link_patient_inputs,
        materialize_trial_block,
    )

    rank_dir = out_root / patient / f"rank{rank_idx}_{_canon8(base_nct)}"
    _ensure_dir(rank_dir)

    # Patient inputs
    link_patient_inputs(rank_dir, project_root, patient)

    # Trial block
    materialize_trial_block(
        rank_dir,
        project_root,
        _canon8(base_nct),
        corpus_idx,
        patient,
        queries_idx,
    )

    # Minimal marker
    (rank_dir / "metrics.json").write_text(
        json.dumps({
            "canonical_nct_id": _canon8(base_nct),
            "note": "baseline-only (no merged/detailed row found in our build)",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ----------------------------
# Five-way builder (+ capped extras)
# ----------------------------

def build_fiveway_for_patient(
    patient: str,
    project_root: Path,
    retrieved_root: Path,
    baseline_index: Optional[Dict[str, List[str]]],
    baseline_clean_dir: Optional[Path],
    out_root: Path,
    corpus_idx: Dict[str, dict],
    queries_idx: Dict[str, dict],
    cap_to_our_survivors: bool = False,
) -> None:
    LOG.info("\n=== Five-way kits for patient: %s ===", patient)

    # 1) Load baseline (collapsed) list
    if baseline_index is not None and patient in baseline_index:
        baseline_ranked: List[str] = list(baseline_index.get(patient, []))
    elif baseline_clean_dir is not None:
        baseline_ranked = load_baseline_ranked_from_dir(baseline_clean_dir, patient)
    else:
        baseline_ranked = []
        LOG.warning("[baseline] no source provided for patient=%s", patient)

    # Build baseline position map (1-based, preserving original positions)
    baseline_pos: Dict[str, int] = {nct: idx for idx, nct in enumerate(baseline_ranked, start=1)}

    # 2) Load OUR units (already sorted by _best_rank_from_subs in helper)
    ours_units = build_rank_units_from_detailed_with_ranks(retrieved_root, patient)
    ours_by_canon: Dict[str, dict] = {u["canonical_nct_id"]: u for u in ours_units}

    # Build our position map using the original best-sub rank (1-based)
    ours_pos: Dict[str, int] = {}
    for u in ours_units:
        cid = u["canonical_nct_id"]
        pos = int(u.get("_best_rank_from_subs", 10**9))
        ours_pos[cid] = pos

    # Helpers (reused for capped computation without mutating originals)
    def _is_survivor(u: dict) -> bool:
        return str(u.get("status", "")).lower() == "survivor"
    def _is_eliminated(u: dict) -> bool:
        return str(u.get("status", "")).lower() == "eliminated"
    def _sorted_by_our_pos(units: List[dict]) -> List[dict]:
        return sorted(units, key=lambda x: ours_pos.get(x["canonical_nct_id"], 10**9))

    # Sets for partitioning (uncapped base)
    BSET: Set[str] = set(baseline_ranked)
    OSET: Set[str] = set(ours_by_canon.keys())

    # A: All baseline ranked (baseline order); ranks = baseline_pos[nct]
    A_seq: List[str] = list(baseline_ranked)

    # B (UNCAPPED): ours NOT baseline (OUR order); ranks = ours_pos[cid]
    B_seq: List[dict] = [u for u in _sorted_by_our_pos(ours_units)
                         if u["canonical_nct_id"] not in BSET]

    # C (UNCAPPED): baseline NOT ours (baseline order); ranks = baseline_pos[nct]
    C_seq: List[str] = [n for n in baseline_ranked if n not in OSET]

    # D: subset of B — only survivors (our order)
    D_seq: List[dict] = [u for u in B_seq if _is_survivor(u)]

    # E: in baseline AND our status=eliminated (baseline order)
    E_seq: List[str] = [base for base in A_seq if base in OSET and _is_eliminated(ours_by_canon.get(base, {}))]

    # Shared loads
    detailed_map = load_detailed_rows(retrieved_root, patient)

    # -------- Build originals (always) --------

    # Build A (use baseline's original positions)
    outA = out_root / patient / "A_baseline_ranked"
    _ensure_dir(outA)
    for base in A_seq:
        rank_idx = baseline_pos[base]
        u = ours_by_canon.get(base)
        if u is not None:
            build_one_rank(outA, project_root, patient, rank_idx, u, detailed_map, corpus_idx, queries_idx)
        else:
            materialize_minimal(outA, project_root, patient, rank_idx, base, corpus_idx, queries_idx)

    # Build B (UNCAPPED; use our original positions)
    outB = out_root / patient / "B_ours_not_baseline"
    _ensure_dir(outB)
    for u in B_seq:
        rank_idx = ours_pos[u["canonical_nct_id"]]
        build_one_rank(outB, project_root, patient, rank_idx, u, detailed_map, corpus_idx, queries_idx)

    # Build C (UNCAPPED; use baseline's original positions)
    outC = out_root / patient / "C_baseline_not_ours"
    _ensure_dir(outC)
    for base in C_seq:
        rank_idx = baseline_pos[base]
        materialize_minimal(outC, project_root, patient, rank_idx, base, corpus_idx, queries_idx)

    # Build D (only survivors from B), using our original positions
    outD = out_root / patient / "D_ours_survivors_not_baseline"
    _ensure_dir(outD)
    for u in D_seq:
        rank_idx = ours_pos[u["canonical_nct_id"]]
        build_one_rank(outD, project_root, patient, rank_idx, u, detailed_map, corpus_idx, queries_idx)

    # Build E (ours eliminated but present in baseline), baseline positions
    outE = out_root / patient / "E_ours_eliminated_in_baseline"
    _ensure_dir(outE)
    for base in E_seq:
        rank_idx = baseline_pos[base]
        u = ours_by_canon.get(base)
        if u:
            build_one_rank(outE, project_root, patient, rank_idx, u, detailed_map, corpus_idx, queries_idx)

    # -------- Extra capped outputs (optional) --------
    if cap_to_our_survivors:
        all_survivors = [u for u in ours_units if _is_survivor(u)]
        capN = len(all_survivors)
        LOG.info("[cap] additional capped outputs with our survivor count: %d", capN)
        if capN > 0:
            ours_sorted = _sorted_by_our_pos(ours_units)
            ours_units_cap = ours_sorted[:capN]
            ours_by_canon_cap = {u["canonical_nct_id"]: u for u in ours_units_cap}
            baseline_ranked_cap = baseline_ranked[:capN]
            baseline_pos_cap: Dict[str, int] = {nct: idx for idx, nct in enumerate(baseline_ranked_cap, start=1)}
            BSET_cap: Set[str] = set(baseline_ranked_cap)
            OSET_cap: Set[str] = set(ours_by_canon_cap.keys())

            # B_cap: ours NOT baseline within the cap (OUR order)
            B_cap_seq: List[dict] = [u for u in ours_units_cap if u["canonical_nct_id"] not in BSET_cap]
            outB_cap = out_root / patient / "B_cap"
            _ensure_dir(outB_cap)
            for u in B_cap_seq:
                rank_idx = ours_pos[u["canonical_nct_id"]]
                build_one_rank(outB_cap, project_root, patient, rank_idx, u, detailed_map, corpus_idx, queries_idx)

            # C_cap: baseline NOT ours within the cap (BASELINE order)
            C_cap_seq: List[str] = [n for n in baseline_ranked_cap if n not in OSET_cap]
            outC_cap = out_root / patient / "C_cap"
            _ensure_dir(outC_cap)
            for base in C_cap_seq:
                rank_idx = baseline_pos_cap[base]
                materialize_minimal(outC_cap, project_root, patient, rank_idx, base, corpus_idx, queries_idx)
        else:
            LOG.info("[cap] zero survivors → skipping B_cap/C_cap for patient=%s", patient)

    LOG.info(
        "[done] A=%d, B=%d, C=%d, D=%d, E=%d%s → %s",
        len(A_seq), len(B_seq), len(C_seq), len(D_seq), len(E_seq),
        ("" if not cap_to_our_survivors else " (plus capped B_cap/C_cap)"),
        (out_root / patient).resolve()
    )

# ----------------------------
# Patient discovery
# ----------------------------

def _discover_patients(retrieved_root: Path) -> List[str]:
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
    ap = argparse.ArgumentParser(description="Five-way SIGIR out kits (baseline vs ours)")
    ap.add_argument("--patient", action="append", default=[],
                    help="Patient ID to process (e.g., sigir-20141). Repeatable. If omitted, ALL discovered patients are processed.")
    ap.add_argument("--project-root", type=Path, default=Path("../../"),
                    help="Absolute path to TrialGPT-SMT project root")
    ap.add_argument("--retrieved-root", type=Path, default=Path("retrieved_mappings"),
                    help="Path to retrieved_mappings root")
    ap.add_argument("--baseline-json", type=Path, default=Path("trialgptref/trialgpt_retrieve.json"),
                    help="Path to baseline JSON mapping {patient_id: [trial_ids...]} (preferred)")
    ap.add_argument("--baseline-clean-dir", type=Path, default=None,
                    help="(Legacy) Directory containing baseline clean ranked *.txt files")
    ap.add_argument("--out", type=Path, default=Path("out_sigir_kits_threeway"),
                    help="Output directory root")
    ap.add_argument("--corpus", type=Path,
                    default=Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/corpus.jsonl"),
                    help="Path to corpus.jsonl")
    ap.add_argument("--queries", type=Path,
                    default=Path("<SATIR_ROOT>/dataset/clinical_trial/sigir/queries.jsonl"),
                    help="Path to queries.jsonl (patient notes)")
    ap.add_argument("--cap-to-our-survivors", action="store_true", default=True,
                    help="Also emit B_cap and C_cap, computed under a cap K equal to the number of OUR survivors (ours capped by our order; baseline capped by baseline order). Originals remain unchanged.")

    args = ap.parse_args()

    project_root: Path = args.project_root.resolve()
    retrieved_root: Path = args.retrieved_root
    out_root: Path = args.out

    corpus_idx  = load_corpus_index(args.corpus)
    queries_idx = load_queries_index(args.queries)

    # Load baseline mapping if JSON provided
    baseline_index: Optional[Dict[str, List[str]]] = None
    if args.baseline_json is not None:
        baseline_index = load_baseline_index_json(args.baseline_json)

    patients: List[str] = list(dict.fromkeys(args.patient))
    if not patients:
        patients = _discover_patients(retrieved_root)
        LOG.info("[discover] %d patients found: %s", len(patients), ", ".join(patients))
        if not patients:
            LOG.info("No patients discovered; nothing to do.")
            return

    for patient in patients:
        try:
            build_fiveway_for_patient(
                patient=patient,
                project_root=project_root,
                retrieved_root=retrieved_root,
                baseline_index=baseline_index,
                baseline_clean_dir=args.baseline_clean_dir,
                out_root=out_root,
                corpus_idx=corpus_idx,
                queries_idx=queries_idx,
                cap_to_our_survivors=args.cap_to_our_survivors,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            LOG.exception("[error] patient %s failed: %s", patient, e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
