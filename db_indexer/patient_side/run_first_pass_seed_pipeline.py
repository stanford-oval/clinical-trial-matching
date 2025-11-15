#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_first_pass_seed_pipeline.py — Pass-n driver (stage-specific seeds, per-side inputs & outputs, single side)

设计：
- 所有路径由命令行传入或由 --build-root 推导，不再使用全局路径常量。
- 下游子脚本全部显式使用 --in-root / --out-root / --side / --patient。
- 架构约定（相对 build-root）：
    <build-root>/
      patient_coded_results/
      patient_coded_results_isa/
      patient_coded_results_isa_imp/
      patient_coded_results_finding_to_procedure_enrichment/
      patient_coded_results_finding_to_observable_entity_enrichment/
      patient_coded_results_finding_to_procedure_enrichment_other/
      patient_coded_results_procedure_to_finding_enrichment/
      patient_coded_results_observable_entity_to_finding_enrichment/
"""

from __future__ import annotations
import os, sys, json, shutil, argparse, subprocess
from typing import Dict, Any, List, Optional, Tuple

STAGES = ["isa", "imp", "f2p", "f2oe", "f2p_rel", "p2f", "oe2f"]


# ---------------- 工具函数 ----------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("[cmd]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")


def _copy(src: str, dst: str) -> None:
    _ensure_dir(os.path.dirname(dst))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)


def _clean_outputs_for_patient(patient: str, roots: List[str]) -> None:
    """
    删除本 patient 在各阶段输出根目录下的子目录：
      <root>/<patient>
    """
    print("[reset] cleaning previous outputs for patient:", patient)
    for root in roots:
        if not root:
            continue
        pdir = os.path.join(root, patient)
        if os.path.isdir(pdir):
            print("        rm -rf", pdir)
            shutil.rmtree(pdir, ignore_errors=True)
    print("[reset] done.")


def _pick_inroot_from_seed_path(seed_path: str, patient: str) -> Optional[str]:
    """
    若 seed_path 形如 .../<side>/<patient>/canonical.jsonl，
    返回上两级目录 .../<side> 以用作 ISA / IMP 的 --in-root；
    否则返回 None。
    """
    try:
        if os.path.basename(seed_path) != "canonical.jsonl":
            return None
        parent = os.path.dirname(seed_path)         # .../<patient>
        if os.path.basename(parent) != patient:
            return None
        return os.path.dirname(parent)              # .../<side>
    except Exception:
        return None


# Seed 行构造（统一 10 键）
SEED_KEYS = [
    "entity_variable_name", "type", "fact_id", "template", "extracted_value",
    "start_time_in_hours", "end_time_in_hours", "start_time_inclusive", "end_time_inclusive", "conceptId",
]


def _mk_seed_row(
    name: str,
    base: Dict[str, Any],
    *,
    template_override: Optional[str] = None,
    time_override: Optional[Dict[str, Any]] = None,
    extracted_value_override: Optional[bool] = None,
    concept_id_override: Optional[str] = None,
) -> Dict[str, Any]:
    tctx = {
        "start_time_in_hours": base.get("start_time_in_hours"),
        "end_time_in_hours": base.get("end_time_in_hours"),
        "start_time_inclusive": base.get("start_time_inclusive"),
        "end_time_inclusive": base.get("end_time_inclusive"),
    }
    if time_override:
        tctx.update({k: time_override.get(k, tctx[k]) for k in tctx.keys()})

    row = {
        "entity_variable_name": name,
        "type": base.get("type", "Bool"),
        "fact_id": base.get("fact_id"),
        "template": template_override
        if template_override is not None
        else (base.get("template") or base.get("template_used")),
        "extracted_value": extracted_value_override
        if extracted_value_override is not None
        else base.get("extracted_value"),
        **tctx,
        "conceptId": concept_id_override if concept_id_override is not None else base.get("conceptId"),
    }
    return {k: row.get(k) for k in SEED_KEYS}


def _dedupe_key(seed_row: Dict[str, Any]) -> Tuple:
    return (
        seed_row.get("entity_variable_name"),
        seed_row.get("start_time_in_hours"),
        seed_row.get("end_time_in_hours"),
        bool(seed_row.get("start_time_inclusive")),
        bool(seed_row.get("end_time_inclusive")),
    )


def _load_seen_keys_from_prior_seeds(
    seeds_root: str,
    patient: str,
    side: str,
    upto_pass_inclusive: int,
) -> set[Tuple]:
    """
    聚合 1..upto_pass_inclusive 各轮生成的 per-side canonical.jsonl，返回已出现过的
    去重键集合（与 _dedupe_key 一致：entity_variable_name + 四个时间键）。
    """
    seen: set[Tuple] = set()
    base = os.path.abspath(seeds_root)
    for p in range(1, max(1, int(upto_pass_inclusive) + 1)):
        path = os.path.join(base, f"pass{p}", side, patient, "canonical.jsonl")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = (
                        obj.get("entity_variable_name"),
                        obj.get("start_time_in_hours"),
                        obj.get("end_time_in_hours"),
                        bool(obj.get("start_time_inclusive")),
                        bool(obj.get("end_time_inclusive")),
                    )
                    seen.add(key)
        except Exception as e:
            print(f"[warn] cannot read prior seed {path}: {e}")
            continue
    return seen


# ---------------- 各阶段：从 annotated 中提取“该阶段特定的新 variable”→ seed 行 ----------------
def extract_stage_seed_records(stage: str, annotated_file: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not (annotated_file and os.path.isfile(annotated_file)):
        return out

    with open(annotated_file, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue

            # ISA：从 isa_enriched.flat.annotated.jsonl 取 new_variable_name + derived_conceptId
            if stage == "isa":
                name = row.get("new_variable_name")
                if not name:
                    continue

                cls = (row.get("class") or "").strip().lower()
                if cls in ("parents", "ancestors"):
                    ev = True
                elif cls in ("descendants", "descendents", "children"):
                    ev = False
                else:
                    mode = (row.get("enrichment_mode") or "").strip().lower()
                    if mode in ("parents", "ancestors"):
                        ev = True
                    elif mode in ("descendants", "descendents", "children"):
                        ev = False
                    else:
                        ev = row.get("extracted_value")

                concept_id = row.get("derived_conceptId") or row.get("derived_concept_id")
                out.append(
                    _mk_seed_row(
                        name,
                        row,
                        template_override=row.get("template_used"),
                        extracted_value_override=ev,
                        concept_id_override=str(concept_id) if concept_id else None,
                    )
                )
                continue

            # IMP：用原行的 conceptId
            if stage == "imp":
                imp = row.get("implications") or {}
                base_cid = row.get("conceptId")
                for bk in ("schema_new_only", "timeframe_new_only"):
                    for item in (imp.get(bk) or []):
                        name = (item or {}).get("name")
                        if not name:
                            continue
                        tctx = {
                            "start_time_in_hours": item.get("start_time_in_hours"),
                            "end_time_in_hours": item.get("end_time_in_hours"),
                            "start_time_inclusive": item.get("start_time_inclusive"),
                            "end_time_inclusive": item.get("end_time_inclusive"),
                        }
                        if all(v is None for v in tctx.values()):
                            tctx = None
                        out.append(
                            _mk_seed_row(
                                name,
                                row,
                                time_override=tctx,
                                concept_id_override=str(base_cid) if base_cid else None,
                            )
                        )
                continue

            # 其余阶段：从 produced[] 里取 name + 对应 id（由阶段决定）
            key_map = {
                "f2p": ("finding_to_procedure_enrichment", "procedureId"),
                "f2p_rel": ("finding_to_procedure_via_relations", "procedureId"),
                "f2oe": ("finding_to_observable_entity_enrichment", "observableEntityId"),
                "p2f": ("procedure_to_finding_enrichment", "findingId"),
                "oe2f": ("observable_entity_to_finding_enrichment", "findingId"),
            }
            blk_name, id_key = key_map.get(stage, (None, None))
            blk = row.get(blk_name) if blk_name else None
            produced = (blk or {}).get("produced") or []
            if not isinstance(produced, list):
                continue

            for item in produced:
                if isinstance(item, dict):
                    name = item.get("name")
                    if not name:
                        continue
                    tctx = {
                        "start_time_in_hours": item.get("start_time_in_hours"),
                        "end_time_in_hours": item.get("end_time_in_hours"),
                        "start_time_inclusive": item.get("start_time_inclusive"),
                        "end_time_inclusive": item.get("end_time_inclusive"),
                    }
                    if all(v is None for v in tctx.values()):
                        tctx = None
                    concept_id = item.get(id_key) if id_key else None
                    out.append(
                        _mk_seed_row(
                            name,
                            row,
                            time_override=tctx,
                            concept_id_override=str(concept_id) if concept_id else None,
                        )
                    )
                elif isinstance(item, str):
                    out.append(_mk_seed_row(item, row))
    return out


# ---------------- 阶段执行（调用脚本 + 快照产物） ----------------
def run_stage(
    stage: str,
    side: str,
    pass_side_dir: str,
    patient: str,
    side_seed_snapshot: str,
    side_seed_source_path: str,
    scripts: Dict[str, str],
    roots: Dict[str, str],
) -> Dict[str, Any]:
    """
    scripts: {"isa": path, "imp": path, "f2p": path, ...}
    roots:   {
        "canonical_root": ...,
        "isa_root": ...,
        "imp_root": ...,
        "f2p_root": ...,
        "f2oe_root": ...,
        "f2p_rel_root": ...,
        "p2f_root": ...,
        "oe2f_root": ...,
    }
    """
    stage_dir = os.path.join(pass_side_dir, stage)
    _ensure_dir(stage_dir)
    _copy(side_seed_snapshot, os.path.join(stage_dir, "seed.jsonl"))

    annotated_path = None
    note = "ok"

    if stage == "isa":
        isa_script = scripts["isa"]
        isa_out_root = roots["isa_root"]
        # per-side seed dir as in-root 优先；否则回退 canonical_root
        isa_inroot = _pick_inroot_from_seed_path(side_seed_source_path, patient) or roots["canonical_root"]

        print(f"[isa] in-root={isa_inroot} out-root={isa_out_root}")
        _run(
            [
                sys.executable,
                isa_script,
                "--patient",
                patient,
                "--in-root",
                isa_inroot,
                "--out-root",
                isa_out_root,
            ],
            cwd=os.path.dirname(isa_script),
        )

        isa_patient_dir = os.path.join(isa_out_root, patient)
        canon_enriched_jsonl = os.path.join(isa_patient_dir, "canonical.enriched.jsonl")
        if os.path.isfile(canon_enriched_jsonl):
            shutil.copyfile(
                canon_enriched_jsonl,
                os.path.join(stage_dir, "canonical.enriched.jsonl"),
            )

        isa_flat_json = os.path.join(isa_patient_dir, "isa_enriched.flat.json")
        if os.path.isfile(isa_flat_json):
            isa_annot_jsonl = os.path.join(
                stage_dir, "isa_enriched.flat.annotated.jsonl"
            )
            with open(isa_flat_json, "r", encoding="utf-8") as fin, open(
                isa_annot_jsonl, "w", encoding="utf-8"
            ) as fout:
                try:
                    arr = json.load(fin)
                except Exception:
                    arr = []
                for obj in arr or []:
                    if isinstance(obj, dict) and "new_variable_name" in obj:
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            annotated_path = isa_annot_jsonl
            note = "ran enrich_with_isa.py and expanded isa_enriched.flat.json"
        else:
            note = "ran enrich_with_isa.py (no isa_enriched.flat.json)"

    elif stage == "imp":
        imp_script = scripts["imp"]
        imp_out_root = roots["imp_root"]
        imp_inroot = _pick_inroot_from_seed_path(side_seed_source_path, patient) or roots[
            "canonical_root"
        ]
        rules_path = os.path.join(
            os.path.dirname(imp_script), "../rules/schema_implications.json"
        )

        print(f"[imp] in-root={imp_inroot} out-root={imp_out_root}")
        _run(
            [
                sys.executable,
                imp_script,
                "--in-root",
                imp_inroot,
                "--out-root",
                imp_out_root,
                "--rules",
                rules_path,
                "--patient",
                patient,
            ],
            cwd=os.path.dirname(imp_script),
        )
        annotated_path = os.path.join(
            imp_out_root, patient, "canonical.enriched.annotated.jsonl"
        )
        note = "ran apply_schema_implications.py"

    elif stage == "f2p":
        f2p_script = scripts["f2p"]
        f2p_in_root = roots["imp_root"]
        f2p_out_root = roots["f2p_root"]
        print(f"[f2p] in-root={f2p_in_root} out-root={f2p_out_root}")
        _run(
            [
                sys.executable,
                f2p_script,
                "--side",
                side,
                "--patient",
                patient,
                "--input-filename",
                "canonical.enriched.annotated.jsonl",
                "--in-root",
                f2p_in_root,
                "--out-root",
                f2p_out_root,
            ],
            cwd=os.path.dirname(f2p_script),
        )
        annotated_path = os.path.join(
            f2p_out_root, patient, side, "finding_to_procedure_enriched.annotated.jsonl"
        )
        note = "ran enrich_findings_to_procedure.py"

    elif stage == "f2oe":
        f2oe_script = scripts["f2oe"]
        f2oe_in_root = roots["f2p_root"]
        f2oe_out_root = roots["f2oe_root"]
        print(f"[f2oe] in-root={f2oe_in_root} out-root={f2oe_out_root}")
        _run(
            [
                sys.executable,
                f2oe_script,
                "--side",
                side,
                "--patient",
                patient,
                "--input-filename",
                "finding_to_procedure_enriched.annotated.jsonl",
                "--in-root",
                f2oe_in_root,
                "--out-root",
                f2oe_out_root,
            ],
            cwd=os.path.dirname(f2oe_script),
        )
        annotated_path = os.path.join(
            f2oe_out_root,
            patient,
            side,
            "finding_to_observable_entity_enriched.annotated.jsonl",
        )
        note = "ran enrich_findings_to_observable_entity.py"

    elif stage == "f2p_rel":
        f2p_rel_script = scripts["f2p_rel"]
        f2p_rel_in_root = roots["imp_root"]
        f2p_rel_out_root = roots["f2p_rel_root"]
        print(f"[f2p_rel] in-root={f2p_rel_in_root} out-root={f2p_rel_out_root}")
        _run(
            [
                sys.executable,
                f2p_rel_script,
                "--side",
                side,
                "--patient",
                patient,
                "--input-filename",
                "canonical.enriched.annotated.jsonl",
                "--in-root",
                f2p_rel_in_root,
                "--out-root",
                f2p_rel_out_root,
            ],
            cwd=os.path.dirname(f2p_rel_script),
        )
        annotated_path = os.path.join(
            f2p_rel_out_root,
            patient,
            side,
            "finding_to_procedure_via_relations.annotated.jsonl",
        )
        note = "ran enrich_findings_to_procedure_via_other_relations.py"

    elif stage == "p2f":
        p2f_script = scripts["p2f"]
        p2f_in_root = roots["f2p_root"]
        p2f_out_root = roots["p2f_root"]
        print(f"[p2f] in-root={p2f_in_root} out-root={p2f_out_root}")
        _run(
            [
                sys.executable,
                p2f_script,
                "--side",
                side,
                "--patient",
                patient,
                "--in-root",
                p2f_in_root,
                "--out-root",
                p2f_out_root,
            ],
            cwd=os.path.dirname(p2f_script),
        )
        annotated_path = os.path.join(
            p2f_out_root, patient, side, "procedure_to_finding_enriched.annotated.jsonl"
        )
        note = "ran enrich_procedure_to_finding.py"

    elif stage == "oe2f":
        oe2f_script = scripts["oe2f"]
        oe2f_in_root = roots["f2oe_root"]
        oe2f_out_root = roots["oe2f_root"]
        print(f"[oe2f] in-root={oe2f_in_root} out-root={oe2f_out_root}")
        _run(
            [
                sys.executable,
                oe2f_script,
                "--side",
                side,
                "--patient",
                patient,
                "--in-root",
                oe2f_in_root,
                "--out-root",
                oe2f_out_root,
            ],
            cwd=os.path.dirname(oe2f_script),
        )
        annotated_path = os.path.join(
            oe2f_out_root,
            patient,
            side,
            "observable_entity_to_finding_enriched.annotated.jsonl",
        )
        note = "ran enrich_observable_entity_to_finding.py"

    else:
        placeholder = os.path.join(stage_dir, f"{stage}_enriched.annotated.jsonl")
        open(placeholder, "w", encoding="utf-8").close()
        annotated_path = placeholder
        note = "placeholder (stage not recognized)"

    # 把 annotated 快照复制到当前 pass/side/stage 目录
    if annotated_path and os.path.isfile(annotated_path):
        snapshot_ann = os.path.join(stage_dir, f"{stage}_enriched.annotated.jsonl")
        if os.path.abspath(annotated_path) != os.path.abspath(snapshot_ann):
            shutil.copyfile(annotated_path, snapshot_ann)
        annotated_path = snapshot_ann
    else:
        annotated_path = None

    return {
        "stage": stage,
        "side": side,
        "patient": patient,
        "annotated_snapshot": annotated_path,
        "note": note,
    }


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Pass-n driver (single-side entailment pipeline)"
    )
    ap.add_argument("--patient", required=True)
    ap.add_argument("--seed", required=True, help="Seed jsonl for this side.")
    ap.add_argument(
        "--side",
        default="inclusion",
        help="Logical side name to pass to downstream scripts (default: inclusion).",
    )
    ap.add_argument("--work-root", required=True, help="Root for per-pass work snapshots (passes/).")
    ap.add_argument(
        "--build-root",
        required=True,
        help="Root that contains patient_coded_results* directories for this pipeline run.",
    )
    ap.add_argument(
        "--seeds-root",
        required=True,
        help="Root dir to store per-pass seeds (e.g., <build-root>/seeds).",
    )
    ap.add_argument(
        "--pass-no",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--reset-outputs",
        action="store_true",
        help="Clear previous stage outputs for this patient (keeps passes snapshots and seeds-root).",
    )
    ap.add_argument(
        "--scripts-dir",
        default=None,
        help="Directory containing stage scripts (default: alongside this driver).",
    )
    args = ap.parse_args()

    patient = args.patient
    side = args.side
    seed_path = os.path.abspath(args.seed)
    work_root = os.path.abspath(args.work_root)
    build_root = os.path.abspath(args.build_root)
    seeds_root = os.path.abspath(args.seeds_root)

    pass_dir = os.path.join(work_root, patient, f"pass{args.pass_no:02d}")
    _ensure_dir(pass_dir)

    # 推导各阶段根目录
    canonical_root = os.path.join(build_root, "patient_coded_results")
    isa_root = os.path.join(build_root, "patient_coded_results_isa")
    imp_root = os.path.join(build_root, "patient_coded_results_isa_imp")
    f2p_root = os.path.join(
        build_root, "patient_coded_results_finding_to_procedure_enrichment"
    )
    f2oe_root = os.path.join(
        build_root, "patient_coded_results_finding_to_observable_entity_enrichment"
    )
    f2p_rel_root = os.path.join(
        build_root,
        "patient_coded_results_finding_to_procedure_enrichment_other",
    )
    p2f_root = os.path.join(
        build_root, "patient_coded_results_procedure_to_finding_enrichment"
    )
    oe2f_root = os.path.join(
        build_root,
        "patient_coded_results_observable_entity_to_finding_enrichment",
    )

    roots = {
        "canonical_root": canonical_root,
        "isa_root": isa_root,
        "imp_root": imp_root,
        "f2p_root": f2p_root,
        "f2oe_root": f2oe_root,
        "f2p_rel_root": f2p_rel_root,
        "p2f_root": p2f_root,
        "oe2f_root": oe2f_root,
    }

    # 所有需要 reset 的阶段输出根
    output_roots = [
        isa_root,
        imp_root,
        f2p_root,
        f2oe_root,
        f2p_rel_root,
        p2f_root,
        oe2f_root,
    ]

    # 子脚本路径
    if args.scripts_dir:
        scripts_dir = os.path.abspath(args.scripts_dir)
    else:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))

    scripts = {
        "isa": os.path.join(scripts_dir, "enrich_with_isa.py"),
        "imp": os.path.join(scripts_dir, "apply_schema_implications.py"),
        "f2p": os.path.join(scripts_dir, "enrich_findings_to_procedure.py"),
        "f2oe": os.path.join(scripts_dir, "enrich_findings_to_observable_entity.py"),
        "f2p_rel": os.path.join(
            scripts_dir, "enrich_findings_to_procedure_via_other_relations.py"
        ),
        "p2f": os.path.join(scripts_dir, "enrich_procedure_to_finding.py"),
        "oe2f": os.path.join(
            scripts_dir, "enrich_observable_entity_to_finding.py"
        ),
    }

    if args.reset_outputs:
        _clean_outputs_for_patient(patient, output_roots)

    summary: Dict[str, Any] = {
        "patient": patient,
        "pass": args.pass_no,
        "side": side,
        "seed": seed_path,
        "stages": STAGES,
        "side_detail": {},
    }

    all_seed_rows: List[Dict[str, Any]] = []

    # 当前 side 的工作目录 + seed 快照
    side_dir = os.path.join(pass_dir, side)
    _ensure_dir(side_dir)
    side_seed_snapshot = os.path.join(side_dir, "seed.jsonl")
    _copy(seed_path, side_seed_snapshot)

    side_rec = {"stages": [], "seed_snapshot": side_seed_snapshot}
    summary["side_detail"] = side_rec

    # 逐阶段运行
    for stage in STAGES:
        meta = run_stage(
            stage,
            side,
            side_dir,
            patient,
            side_seed_snapshot,
            seed_path,
            scripts,
            roots,
        )
        side_rec["stages"].append(meta)

        ann = meta.get("annotated_snapshot")
        if ann and os.path.isfile(ann):
            seed_rows = extract_stage_seed_records(stage, ann)

            stage_seed_jsonl = os.path.join(
                os.path.dirname(ann), f"{stage}.produced.seed.jsonl"
            )
            with open(stage_seed_jsonl, "w", encoding="utf-8") as fp:
                for r in seed_rows:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")

            all_seed_rows.extend(seed_rows)

    # 按 (变量 + 时间窗口) 去重 —— 只对这一 side
    def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for r in rows:
            k = _dedupe_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    dedup_rows = _dedupe_rows(all_seed_rows)

    # 载入 1..N 轮历史 seeds 中已出现过的 key（同一个 side）
    seen_keys = _load_seen_keys_from_prior_seeds(
        seeds_root=seeds_root,
        patient=patient,
        side=side,
        upto_pass_inclusive=args.pass_no,
    )

    def _only_new(
        rows: List[Dict[str, Any]], already: set[Tuple]
    ) -> List[Dict[str, Any]]:
        kept = []
        for r in rows:
            k = _dedupe_key(r)
            if k in already:
                continue
            kept.append(r)
        return kept

    new_rows = _only_new(dedup_rows, seen_keys)

    # 写出下一轮 seed（passes 目录下的快照）
    next_seed_path = os.path.join(pass_dir, f"next_pass_seed.{side}.jsonl")
    _ensure_dir(os.path.dirname(next_seed_path))
    with open(next_seed_path, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 同步到 seeds-root/pass{N+1}/{side}/{patient}/canonical.jsonl
    seeds_pass_dir = os.path.join(seeds_root, f"pass{args.pass_no+1}")
    side_patient_dir = os.path.join(seeds_pass_dir, side, patient)
    _ensure_dir(side_patient_dir)

    seeds_side_path = os.path.join(side_patient_dir, "canonical.jsonl")
    with open(seeds_side_path, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary["next_pass_seed"] = {
        "passes": next_seed_path,
        "seeds_root": seeds_side_path,
    }
    summary["produced_variable_count"] = {
        "produced_this_pass": len(dedup_rows),
        "new_vs_all_prev_passes": len(new_rows),
    }

    with open(os.path.join(pass_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] pass{args.pass_no:02d}  patient={patient}")
    print(f"       side: {side}")
    print(f"       next seeds (passes): {next_seed_path}  (unique: {len(dedup_rows)})")
    print(f"       seeds-root snapshot: {seeds_side_path}")


if __name__ == "__main__":
    main()
