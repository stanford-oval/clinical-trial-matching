#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, subprocess, sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

CAN_EXC_KEY = "can_be_used_for_exclusion"


def _boolish(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def _is_diag(name: str) -> bool:
    return (name or "").strip().lower().startswith("patient_has_diagnosis_of_")


def _dedupe_key(obj: Dict[str, Any]) -> Tuple:
    return (
        obj.get("entity_variable_name"),
        obj.get("start_time_in_hours"),
        obj.get("end_time_in_hours"),
        bool(obj.get("start_time_inclusive")),
        bool(obj.get("end_time_inclusive")),
    )


def _normalize_row_inplace(obj: Dict[str, Any]) -> None:
    # extracted_value: null -> True
    if obj.get("extracted_value") is None:
        obj["extracted_value"] = True

    # can_be_used_for_exclusion: diagnosis-only; normalize to bool/None; drop for non-diagnosis
    name = obj.get("entity_variable_name") or ""
    if _is_diag(str(name)):
        obj[CAN_EXC_KEY] = _boolish(obj.get(CAN_EXC_KEY))
    else:
        if CAN_EXC_KEY in obj:
            obj.pop(CAN_EXC_KEY, None)


def _merge_can_exc(dst: Dict[str, Any], src: Dict[str, Any]) -> bool:
    """
    OR-merge can_be_used_for_exclusion for diagnosis rows.
    Returns True if dst was upgraded from False/None -> True.
    """
    if not _is_diag(str(dst.get("entity_variable_name") or "")):
        return False

    dv = _boolish(dst.get(CAN_EXC_KEY))
    sv = _boolish(src.get(CAN_EXC_KEY))

    if dv is True or sv is True:
        upgraded = (dv is not True)
        dst[CAN_EXC_KEY] = True
        return upgraded

    if dv is False or sv is False:
        dst[CAN_EXC_KEY] = False
        return False

    dst[CAN_EXC_KEY] = None
    return False


def discover_patients(patient_root: Path, specified: str | None) -> List[str]:
    if specified:
        return [specified]
    if not patient_root.exists():
        return []
    return sorted(p.name for p in patient_root.iterdir() if p.is_dir())


def seed_path(seeds_root: Path, pass_no: int, side: str, patient: str) -> Path:
    return seeds_root / f"pass{pass_no}" / side / patient / "canonical.jsonl"


def write_pass1_seed_from_diagnosis(dia: Path, seed1: Path, side: str) -> int:
    """
    只用 diagnosis.jsonl 生成 pass1 seeds：
      - extracted_value None -> True
      - diagnosis can_be_used_for_exclusion 做 OR 合并（True 优先）
      - 以 (entity_variable_name + 4 time keys) 去重
    """
    if not dia.exists():
        print(f"[STEP1] not_certain_caonical.jsonl 不存在，跳过：{dia}")
        return 0

    rows: List[Dict[str, Any]] = []
    idx: Dict[Tuple, int] = {}

    read_n = 0
    upgraded = 0

    with dia.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            read_n += 1
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            _normalize_row_inplace(obj)
            k = _dedupe_key(obj)

            if k not in idx:
                idx[k] = len(rows)
                rows.append(obj)
            else:
                dst = rows[idx[k]]
                if _merge_can_exc(dst, obj):
                    upgraded += 1
                # best-effort backfill missing fields
                for fld in ("conceptId", "template", "type", "fact_id", "extracted_value"):
                    if dst.get(fld) in (None, "") and obj.get(fld) not in (None, ""):
                        dst[fld] = obj.get(fld)

    seed1.parent.mkdir(parents=True, exist_ok=True)
    with seed1.open("w", encoding="utf-8") as fw:
        for r in rows:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    msg = f"[STEP1] wrote pass1 seed from diagnosis ({side}): {len(rows)} unique rows"
    if upgraded:
        msg += f", upgraded {upgraded} rows {CAN_EXC_KEY}->True"
    msg += f" → {seed1}"
    print(msg)
    return len(rows)


def export_aggregated_facts(
    seeds_root: Path,
    facts_export_root: Path,
    patient: str,
    total_passes: int,
    side: str,
) -> None:
    """
    把 seeds/pass*/<side>/<patient>/canonical.jsonl 全部累积去重后导出（只导出一个 side）。
    注意：重复 key 不是简单跳过，而是对 diagnosis.can_be_used_for_exclusion 做 OR 合并（True 优先）。
    """
    agg: List[Dict[str, Any]] = []
    idx: Dict[Tuple, int] = {}

    # +2：第 N 轮 pipeline 会写 pass(N+1) 的 seeds
    for p in range(1, total_passes + 2):
        f = seeds_root / f"pass{p}" / side / patient / "canonical.jsonl"
        if not f.exists():
            continue

        with f.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue

                _normalize_row_inplace(obj)
                k = _dedupe_key(obj)

                if k not in idx:
                    idx[k] = len(agg)
                    agg.append(obj)
                else:
                    dst = agg[idx[k]]
                    _merge_can_exc(dst, obj)
                    for fld in ("conceptId", "template", "type", "fact_id", "extracted_value"):
                        if dst.get(fld) in (None, "") and obj.get(fld) not in (None, ""):
                            dst[fld] = obj.get(fld)

    out_dir = facts_export_root / patient / side
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canonical.jsonl"
    with out_path.open("w", encoding="utf-8") as fw:
        for row in agg:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[export] {side}: wrote {len(agg)} unique facts → {out_path}")


def run_one_pass(
    pipeline_cmd: list[str],
    build_root: Path,
    work_root: Path,
    seeds_root: Path,
    patient: str,
    pass_no: int,
    side: str,
) -> None:
    seed = seed_path(seeds_root, pass_no, side, patient)
    cmd = pipeline_cmd + [
        "--patient", patient,
        "--seed", str(seed),
        "--side", side,
        "--build-root", str(build_root),
        "--work-root", str(work_root),
        "--seeds-root", str(seeds_root),
        "--pass-no", str(pass_no),
        "--reset-outputs",
    ]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def process_patient(
    pipeline_cmd: list[str],
    patient_root: Path,
    seeds_root: Path,
    work_root: Path,
    facts_export_root: Path,
    build_root: Path,
    patient: str,
    total_passes: int,
    side: str,
) -> None:
    print(f"\n===== Patient: {patient} (side={side}) =====")

    dia = patient_root / patient / "not_certain_canonical.jsonl"
    seed1 = seed_path(seeds_root, 1, side, patient)

    # --- STEP 1：只用 diagnosis.jsonl 生成 pass1 seed（不合并，不用 canonical/embedding） ---
    n = write_pass1_seed_from_diagnosis(dia, seed1, side=side)
    if n == 0:
        print(f"[warn] {patient}: pass1 seed is empty; skipping passes.")
        return

    # --- STEP 2：按照 passes 次数跑流水线（单 side） ---
    for p in range(1, total_passes + 1):
        run_one_pass(pipeline_cmd, build_root, work_root, seeds_root, patient, p, side)

    # --- STEP 3：导出聚合事实（只导出当前 side） ---
    export_aggregated_facts(seeds_root, facts_export_root, patient, total_passes, side)


def main():
    ap = argparse.ArgumentParser(
        "Run multi-pass pipeline from diagnosis.jsonl only (single-side, diagnose-classified seeds)"
    )
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--patient", type=str)
    ap.add_argument(
        "--side",
        choices=["inclusion", "exclusion"],
        default="inclusion",
        help="Which side to run (default: inclusion)",
    )
    ap.add_argument(
        "--build-root",
        required=True,
        help=(
            "Build root directory path or name under project root.\n"
            "Examples: 'patient_build', 'patient_build_sigir_exclusion'."
        ),
    )
    ap.add_argument(
        "--pipeline-script",
        default="run_first_pass_seed_pipeline_diagnose_classified.py",
        help="Path to run_first_pass_seed_pipeline_diagnose_classified.py (default: same dir as this script).",
    )
    args = ap.parse_args()

    if args.passes < 1:
        raise SystemExit("--passes 必须 ≥ 1")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]  # .../TrialGPT-SMT/

    build_root = Path(args.build_root)
    if not build_root.is_absolute():
        build_root = (project_root / build_root).resolve()

    patient_root = build_root / "patient_coded_results"
    seeds_root = build_root / "seeds"
    work_root = build_root / "passes"
    facts_export_root = build_root / "patient_facts_export"

    pipeline_script = Path(args.pipeline_script)
    if not pipeline_script.is_absolute():
        pipeline_script = script_dir / pipeline_script
    pipeline_cmd = [sys.executable, str(pipeline_script)]

    for patient in discover_patients(patient_root, args.patient):
        try:
            process_patient(
                pipeline_cmd=pipeline_cmd,
                patient_root=patient_root,
                seeds_root=seeds_root,
                work_root=work_root,
                facts_export_root=facts_export_root,
                build_root=build_root,
                patient=patient,
                total_passes=args.passes,
                side=args.side,
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {patient}: pass 运行失败 – {e}")


if __name__ == "__main__":
    main()
