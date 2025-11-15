#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_first_pass_seed_pipeline.py — Pass-n driver (stage-specific seeds, per-side inputs & outputs)

新增
• 每个 side 可用不同的输入 seed：--seed-inclusion / --seed-exclusion（缺省回退到 --seed）
• 每个 pass 结束：分别写 passes/passNN/next_pass_seed.inclusion.jsonl 与 .exclusion.jsonl
• 同步把每个 side 的种子写到 <seeds-root>/pass{N}/{side}/seed.jsonl
• --reset-outputs 清空阶段产物根（不动 passes 快照与 seeds-root）

其他
• 各阶段只统计“该阶段专属的新 variable”，展开为 9 键 seed 行
• ISA 的 extracted_value：parents/ancestors→True；descendants/descendents→False；否则回退 enrichment_mode
"""

from __future__ import annotations
import os, sys, json, shutil, argparse, subprocess
from typing import Dict, Any, List, Optional, Tuple

# ---------------- 项目与脚本路径 ----------------
PROJ_ROOT = "../../"

# enrich_with_isa.py
ISA_IN_ROOT   = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results")
ISA_OUT_ROOT  = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_isa")

# apply_schema_implications.py
IMP_IN_ROOT   = ISA_IN_ROOT
IMP_OUT_ROOT  = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_isa_imp")

# enrich_findings_to_procedure.py
F2P_IN_ROOT   = IMP_OUT_ROOT
F2P_OUT_ROOT  = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_finding_to_procedure_enrichment")

# enrich_findings_to_observable_entity.py
F2OE_IN_ROOT  = F2P_OUT_ROOT
F2OE_OUT_ROOT = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_finding_to_observable_entity_enrichment")

# via-relations / p2f / oe2f
F2P_REL_OUT_ROOT  = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_finding_to_procedure_enrichment_other")
P2F_OUT_ROOT      = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_procedure_to_finding_enrichment")
OE2F_OUT_ROOT     = os.path.join(PROJ_ROOT, "patient_build/patient_coded_results_observable_entity_to_finding_enrichment")

# 阶段脚本路径
SCRIPTS_DIR     = os.path.join(PROJ_ROOT, "irsrc/patient_side")
ISA_SCRIPT      = os.path.join(SCRIPTS_DIR, "enrich_with_isa.py")
IMP_SCRIPT      = os.path.join(SCRIPTS_DIR, "apply_schema_implications.py")
F2P_SCRIPT      = os.path.join(SCRIPTS_DIR, "enrich_findings_to_procedure.py")
F2OE_SCRIPT     = os.path.join(SCRIPTS_DIR, "enrich_findings_to_observable_entity.py")
F2P_REL_SCRIPT  = os.path.join(SCRIPTS_DIR, "enrich_findings_to_procedure_via_other_relations.py")
P2F_SCRIPT2     = os.path.join(SCRIPTS_DIR, "enrich_procedure_to_finding.py")
OE2F_SCRIPT2    = os.path.join(SCRIPTS_DIR, "enrich_observable_entity_to_finding.py")

STAGES = ["isa", "imp", "f2p", "f2oe", "f2p_rel", "p2f", "oe2f"]
SIDES  = ["inclusion", "exclusion"]

# 哪些根会在 --reset-outputs 时被清空（仅删除 patient 子目录）
_OUTPUT_CLEAN_ROOTS = [
    ISA_OUT_ROOT, IMP_OUT_ROOT, F2P_OUT_ROOT, F2OE_OUT_ROOT,
    F2P_REL_OUT_ROOT, P2F_OUT_ROOT, OE2F_OUT_ROOT,
]

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

def _clean_outputs_for_patient(patient: str) -> None:
    print("[reset] cleaning previous outputs for patient:", patient)
    for root in _OUTPUT_CLEAN_ROOTS:
        pdir = os.path.join(root, patient)
        if os.path.isdir(pdir):
            print("        rm -rf", pdir)
            shutil.rmtree(pdir, ignore_errors=True)
    print("[reset] done.")

def _pick_inroot_from_seed_path(seed_path: str, patient: str) -> Optional[str]:
    """
    若 seed_path 形如 .../<side>/<patient>/canonical.jsonl，
    返回上两级目录 .../<side> 以用作 ISA 的 --in-root；
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

# Seed 行构造（统一 9 键）
SEED_KEYS = [
    "entity_variable_name", "type", "fact_id", "template", "extracted_value",
    "start_time_in_hours", "end_time_in_hours", "start_time_inclusive", "end_time_inclusive", "conceptId",
]

def _mk_seed_row(name: str, base: Dict[str, Any], *,
                 template_override: Optional[str] = None,
                 time_override: Optional[Dict[str, Any]] = None,
                 extracted_value_override: Optional[bool] = None,
                 concept_id_override: Optional[str] = None) -> Dict[str, Any]:
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
        "template": template_override if template_override is not None else (base.get("template") or base.get("template_used")),
        "extracted_value": extracted_value_override if extracted_value_override is not None else base.get("extracted_value"),
        **tctx,
        "conceptId": concept_id_override if concept_id_override is not None else base.get("conceptId"),  # ← NEW
    }
    # 只保留 10 键
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
    for p in range(1, max(1, int(upto_this := int(upto_pass_inclusive)) + 1)):
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
                    # 构造与本轮相同的去重 key：name + 4个时间键
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
                elif cls in ("descendants", "descendents", "children"):  # ← 新增 children 也按 False
                    ev = False
                else:
                    mode = (row.get("enrichment_mode") or "").strip().lower()
                    if mode in ("parents", "ancestors"):
                        ev = True
                    elif mode in ("descendants", "descendents", "children"):  # ← 兜底也识别 children
                        ev = False
                    else:
                        ev = row.get("extracted_value")

                concept_id = row.get("derived_conceptId") or row.get("derived_concept_id")
                out.append(_mk_seed_row(
                    name,
                    row,
                    template_override=row.get("template_used"),
                    extracted_value_override=ev,
                    concept_id_override=str(concept_id) if concept_id else None
                ))
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
                        out.append(_mk_seed_row(
                            name, row,
                            time_override=tctx,
                            concept_id_override=str(base_cid) if base_cid else None
                        ))
                continue

            # 其余阶段：从 produced[] 里取 name + 对应 id（由阶段决定）
            key_map = {
                "f2p":    ("finding_to_procedure_enrichment",            "procedureId"),
                "f2p_rel": ("finding_to_procedure_via_relations",         "procedureId"),
                "f2oe":   ("finding_to_observable_entity_enrichment",   "observableEntityId"),
                "p2f":    ("procedure_to_finding_enrichment",           "findingId"),
                "oe2f":   ("observable_entity_to_finding_enrichment",   "findingId"),
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
                    out.append(_mk_seed_row(
                        name, row,
                        time_override=tctx,
                        concept_id_override=str(concept_id) if concept_id else None
                    ))
                elif isinstance(item, str):
                    # produced 是纯字符串时，无法从 item 拿到 id；退回 row.conceptId（多数情况下无意义）
                    out.append(_mk_seed_row(item, row))
    return out


# ---------------- 阶段执行（调用脚本 + 快照产物） ----------------
def run_stage(stage: str, side: str, pass_side_dir: str, patient: str,
              side_seed_snapshot: str, side_seed_source_path: str) -> Dict[str, Any]:
    stage_dir = os.path.join(pass_side_dir, stage); _ensure_dir(stage_dir)
    _copy(side_seed_snapshot, os.path.join(stage_dir, "seed.jsonl"))

    annotated_path = None
    note = "ok"

    if stage == "isa":
        # 优先：若本 side 的 seed 源路径已经是 .../<side>/<patient>/canonical.jsonl，
        # 直接用其上两级目录作为 ISA 的 --in-root
        isa_inroot = _pick_inroot_from_seed_path(side_seed_source_path, patient)

        if isa_inroot:
            print(f"[isa] using per-side seed dir as --in-root: {isa_inroot}")
            _run([
                sys.executable, ISA_SCRIPT,
                "--patient", patient,
                "--in-root", isa_inroot,
            ], cwd=os.path.dirname(ISA_SCRIPT))
        else:
            # 否则不做注入，保持你当前默认行为
            print("[isa] seed path not matched as .../<side>/<patient>/canonical.jsonl; "
                "running ISA with its default IN_ROOT")
            _run([sys.executable, ISA_SCRIPT, "--patient", patient],
                cwd=os.path.dirname(ISA_SCRIPT))

        isa_patient_dir = os.path.join(ISA_OUT_ROOT, patient)
        canon_enriched_jsonl = os.path.join(isa_patient_dir, "canonical.enriched.jsonl")
        if os.path.isfile(canon_enriched_jsonl):
            shutil.copyfile(canon_enriched_jsonl, os.path.join(stage_dir, "canonical.enriched.jsonl"))

        isa_flat_json = os.path.join(isa_patient_dir, "isa_enriched.flat.json")
        if os.path.isfile(isa_flat_json):
            isa_annot_jsonl = os.path.join(stage_dir, "isa_enriched.flat.annotated.jsonl")
            with open(isa_flat_json, "r", encoding="utf-8") as fin, open(isa_annot_jsonl, "w", encoding="utf-8") as fout:
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
        imp_inroot = _pick_inroot_from_seed_path(side_seed_source_path, patient)
        if imp_inroot:
            print(f"[imp] using per-side seed dir as --in-root: {imp_inroot}")
        else:
            # 回退到固定根（与你原来一致）
            imp_inroot = ISA_IN_ROOT
            print(f"[imp] seed path not matched; fallback --in-root: {imp_inroot}")

        _run([sys.executable, IMP_SCRIPT, "--in-root",  imp_inroot,
              "--out-root", IMP_OUT_ROOT,
              "--rules", os.path.join(os.path.dirname(IMP_SCRIPT), "../rules/schema_implications.json"),
              "--patient",  patient], cwd=os.path.dirname(IMP_SCRIPT))
        annotated_path = os.path.join(IMP_OUT_ROOT, patient, "canonical.enriched.annotated.jsonl")
        note = "ran apply_schema_implications.py"

    elif stage == "f2p":
        _run([sys.executable, F2P_SCRIPT, "--side", side, "--patient", patient,
              "--input-filename", "canonical.enriched.annotated.jsonl"], cwd=os.path.dirname(F2P_SCRIPT))
        annotated_path = os.path.join(F2P_OUT_ROOT, patient, side, "finding_to_procedure_enriched.annotated.jsonl")
        note = "ran enrich_findings_to_procedure.py"

    elif stage == "f2oe":
        _run([sys.executable, F2OE_SCRIPT, "--side", side, "--patient", patient,
              "--input-filename", "finding_to_procedure_enriched.annotated.jsonl"], cwd=os.path.dirname(F2OE_SCRIPT))
        annotated_path = os.path.join(F2OE_OUT_ROOT, patient, side, "finding_to_observable_entity_enriched.annotated.jsonl")
        note = "ran enrich_findings_to_observable_entity.py"

    elif stage == "f2p_rel":
        _run([sys.executable, F2P_REL_SCRIPT, "--side", side, "--patient", patient,
              "--input-filename", "canonical.enriched.annotated.jsonl"], cwd=os.path.dirname(F2P_REL_SCRIPT))
        annotated_path = os.path.join(F2P_REL_OUT_ROOT, patient, side, "finding_to_procedure_via_relations.annotated.jsonl")
        note = "ran enrich_findings_to_procedure_via_other_relations.py"

    elif stage == "p2f":
        _run([sys.executable, P2F_SCRIPT2, "--side", side, "--patient", patient], cwd=os.path.dirname(P2F_SCRIPT2))
        annotated_path = os.path.join(P2F_OUT_ROOT, patient, side, "procedure_to_finding_enriched.annotated.jsonl")
        note = "ran enrich_procedure_to_finding.py"

    elif stage == "oe2f":
        _run([sys.executable, OE2F_SCRIPT2, "--side", side, "--patient", patient], cwd=os.path.dirname(OE2F_SCRIPT2))
        annotated_path = os.path.join(OE2F_OUT_ROOT, patient, side, "observable_entity_to_finding_enriched.annotated.jsonl")
        note = "ran enrich_observable_entity_to_finding.py"

    else:
        placeholder = os.path.join(stage_dir, f"{stage}_enriched.annotated.jsonl")
        open(placeholder, "w", encoding="utf-8").close()
        annotated_path = placeholder
        note = "placeholder (stage not recognized)"

    if annotated_path and os.path.isfile(annotated_path):
        snapshot_ann = os.path.join(stage_dir, f"{stage}_enriched.annotated.jsonl")
        if os.path.abspath(annotated_path) != os.path.abspath(snapshot_ann):
            shutil.copyfile(annotated_path, snapshot_ann)
        annotated_path = snapshot_ann
    else:
        annotated_path = None

    return {"stage": stage, "side": side, "patient": patient,
            "annotated_snapshot": annotated_path, "note": note}

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Pass-n driver (stage-specific seeds, per-side inputs & outputs)")
    ap.add_argument("--patient", required=True)
    # ap.add_argument("--seed", required=True, help="Default seed for both sides (jsonl).")
    ap.add_argument("--seed-inclusion", default=None, help="Override seed for inclusion side.")
    ap.add_argument("--seed-exclusion", default=None, help="Override seed for exclusion side.")
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--pass-no", type=int, default=1)
    ap.add_argument("--seeds-root", default=os.path.join(PROJ_ROOT, "patient_build", "seeds"),
                    help="Root dir to store per-pass seeds (default: <PROJECT>/patient_build/seeds)")
    ap.add_argument("--reset-outputs", action="store_true",
                    help="Clear previous stage outputs for this patient (keeps passes snapshots and seeds-root)")
    args = ap.parse_args()

    patient = args.patient
    # default_seed = os.path.abspath(args.seed)
    
    seed_for_side = {
        "inclusion": os.path.abspath(args.seed_inclusion) ,
        "exclusion": os.path.abspath(args.seed_exclusion) ,
    }

    work_root = os.path.abspath(args.work_root)
    pass_dir  = os.path.join(work_root, patient, f"pass{args.pass_no:02d}")

    _ensure_dir(pass_dir)

    if args.reset_outputs:
        _clean_outputs_for_patient(patient)

    summary: Dict[str, Any] = {
        "patient": patient, "pass": args.pass_no,
        "seed": {
                 "inclusion": seed_for_side["inclusion"],
                 "exclusion": seed_for_side["exclusion"]},
        "sides": {}, "stages": STAGES
    }
    all_seed_rows_by_side: Dict[str, List[Dict[str, Any]]] = {"inclusion": [], "exclusion": []}

    # 给每个 side 各自的 seed 做快照（pass 目录内）
    for side in SIDES:
        side_dir = os.path.join(pass_dir, side); _ensure_dir(side_dir)
        side_seed_snapshot = os.path.join(side_dir, "seed.jsonl")
        _copy(seed_for_side[side], side_seed_snapshot)

        side_rec = summary["sides"].setdefault(side, {"stages": [], "seed_snapshot": side_seed_snapshot})

        # 逐阶段运行
        for stage in STAGES:
            meta = run_stage(stage, side, side_dir, patient, side_seed_snapshot, seed_for_side[side])
            side_rec["stages"].append(meta)

            ann = meta.get("annotated_snapshot")
            if ann and os.path.isfile(ann):
                seed_rows = extract_stage_seed_records(stage, ann)

                stage_seed_jsonl = os.path.join(os.path.dirname(ann), f"{stage}.produced.seed.jsonl")
                with open(stage_seed_jsonl, "w", encoding="utf-8") as fp:
                    for r in seed_rows:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

                all_seed_rows_by_side[side].extend(seed_rows)

    # 按 side 去重（变量名 + 四个时间键）
    def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set(); out: List[Dict[str, Any]] = []
        for r in rows:
            k = _dedupe_key(r)
            if k in seen: continue
            seen.add(k); out.append(r)
        return out

    dedup_incl = _dedupe_rows(all_seed_rows_by_side["inclusion"])
    dedup_excl = _dedupe_rows(all_seed_rows_by_side["exclusion"])

    # === 新增：加载既往各轮（1..N）的已见 key，并从本轮去重结果里做差集，得到“新增变量” ===
    seen_incl = _load_seen_keys_from_prior_seeds(  # 注意：这里用 N（含本轮输入所在 pass）
        seeds_root=args.seeds_root, patient=patient, side="inclusion", upto_pass_inclusive=args.pass_no
    )
    seen_excl = _load_seen_keys_from_prior_seeds(
        seeds_root=args.seeds_root, patient=patient, side="exclusion", upto_pass_inclusive=args.pass_no
    )

    def _only_new(rows: List[Dict[str, Any]], already: set[Tuple]) -> List[Dict[str, Any]]:
        kept = []
        for r in rows:
            k = _dedupe_key(r)
            if k in already:
                continue
            kept.append(r)
        return kept

    new_incl = _only_new(dedup_incl, seen_incl)
    new_excl = _only_new(dedup_excl, seen_excl)


    # 写出两份下一轮 seed（到 passes）
    next_seed_incl = os.path.join(pass_dir, "next_pass_seed.inclusion.jsonl")
    next_seed_excl = os.path.join(pass_dir, "next_pass_seed.exclusion.jsonl")
    _ensure_dir(os.path.dirname(next_seed_incl)); _ensure_dir(os.path.dirname(next_seed_excl))
    with open(next_seed_incl, "w", encoding="utf-8") as f:
        for r in new_incl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(next_seed_excl, "w", encoding="utf-8") as f:
        for r in new_excl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 同步到 seeds-root/pass{N+1}/{side}/{patient}/canonical.jsonl
    seeds_pass_dir = os.path.join(os.path.abspath(args.seeds_root), f"pass{args.pass_no+1}")
    incl_patient_dir = os.path.join(seeds_pass_dir, "inclusion", patient)
    excl_patient_dir = os.path.join(seeds_pass_dir, "exclusion", patient)
    _ensure_dir(incl_patient_dir); _ensure_dir(excl_patient_dir)

    seeds_incl_path = os.path.join(incl_patient_dir, "canonical.jsonl")
    seeds_excl_path = os.path.join(excl_patient_dir, "canonical.jsonl")
    with open(seeds_incl_path, "w", encoding="utf-8") as f:
        for r in new_incl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(seeds_excl_path, "w", encoding="utf-8") as f:
        for r in new_excl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


    # 更新 summary 的记录路径（变量名不变，值已是新的路径）
    summary["next_pass_seed"] = {
        "passes_inclusion": next_seed_incl,
        "passes_exclusion": next_seed_excl,
        "seeds_root_inclusion": seeds_incl_path,
        "seeds_root_exclusion": seeds_excl_path,
    }
    summary["produced_variable_count"] = {
        "inclusion_produced_this_pass": len(dedup_incl),
        "exclusion_produced_this_pass": len(dedup_excl),
        "inclusion_new_vs_all_prev_passes": len(new_incl),
        "exclusion_new_vs_all_prev_passes": len(new_excl),
        "total_new": len(new_incl) + len(new_excl),
    }


    with open(os.path.join(pass_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] pass{args.pass_no:02d}  patient={patient}")
    print(f"       next seeds (passes):")
    print(f"         inclusion: {next_seed_incl}  (unique: {len(dedup_incl)})")
    print(f"         exclusion: {next_seed_excl}  (unique: {len(dedup_excl)})")
    print(f"       seeds-root snapshots:")
    print(f"         inclusion: {seeds_incl_path}")
    print(f"         exclusion: {seeds_excl_path}")

if __name__ == "__main__":
    main()
