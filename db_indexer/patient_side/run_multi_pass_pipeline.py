#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil, subprocess, sys
from pathlib import Path
from collections import defaultdict
from typing import List


# ---------- ⬇️ 通用追加函数：把 src 的行按去重/值修正规则并入 canonical -----------------
def append_into_canonical(src: Path, can: Path, label: str) -> int:
    """
    把 src(jsonl) 的行写进 canonical.jsonl：
      1) 若 'extracted_value' 为 null → 置为 True
      2) 以 (entity_variable_name, start_time_in_hours, end_time_in_hours,
             start_time_inclusive, end_time_inclusive) 作为去重键；
         已存在则跳过
    """
    if not src.exists():
        print(f"[STEP1] 无{label}文件，跳过追加：{src}")
        return 0

    can.parent.mkdir(parents=True, exist_ok=True)
    can.touch(exist_ok=True)

    # ① 收集 canonical 里已有 key
    seen = set()
    with can.open("r", encoding="utf-8") as f:
        for line in f:
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

    # ② 逐行处理 src
    new_rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # 修正 extracted_value
            if obj.get("extracted_value") is None:
                obj["extracted_value"] = True

            key = (
                obj.get("entity_variable_name"),
                obj.get("start_time_in_hours"),
                obj.get("end_time_in_hours"),
                bool(obj.get("start_time_inclusive")),
                bool(obj.get("end_time_inclusive")),
            )
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(obj)

    if not new_rows:
        print(f"[STEP1] {label}行均已存在，无追加")
        return 0

    # ③ 追加写入，保证文件末尾有换行
    with can.open("ab+") as f:
        f.seek(0, 2)
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")

    print(f"[STEP1] 追加 {len(new_rows)} 行（{label}）→ {can}")
    return len(new_rows)
# -----------------------------------------------------------------


def copy_file(src: Path, dst: Path, tag: str) -> None:
    if not src.exists():
        print(f"[STEP2] 源不存在，跳过复制（{tag}）：{src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[STEP2] 复制 {src} → {dst}（{tag}）")


def discover_patients(patient_root: Path, specified: str | None) -> List[str]:
    if specified:
        return [specified]
    if not patient_root.exists():
        return []
    return sorted(p.name for p in patient_root.iterdir() if p.is_dir())


def seed_path(seeds_root: Path, pass_no: int, side: str, patient: str) -> Path:
    return seeds_root / f"pass{pass_no}" / side / patient / "canonical.jsonl"


# >>> 导出聚合事实（单 side） --------------------------------------------------------------------
def export_aggregated_facts(
    seeds_root: Path,
    facts_export_root: Path,
    patient: str,
    total_passes: int,
    side: str,
) -> None:
    """
    把 seeds/pass*/<side>/<patient>/canonical.jsonl 全部累积去重后导出
    （只导出一个 side，例如 inclusion 或 exclusion）
    """
    agg: list[dict] = []
    seen: set[tuple] = set()

    # +2 因为：第 N 轮的 pipeline 会写 pass(N+1) 的 seeds
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
                key = (
                    obj.get("entity_variable_name"),
                    obj.get("start_time_in_hours"),
                    obj.get("end_time_in_hours"),
                    bool(obj.get("start_time_inclusive")),
                    bool(obj.get("end_time_inclusive")),
                )
                if key in seen:
                    continue
                seen.add(key)
                agg.append(obj)

    out_dir = facts_export_root / patient / side
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canonical.jsonl"
    with out_path.open("w", encoding="utf-8") as fw:
        for row in agg:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[export] {side}: wrote {len(agg)} unique facts → {out_path}")
# ---------------------------------------------------------------------------


def run_one_pass(
    pipeline_cmd: list[str],
    build_root: Path,
    work_root: Path,
    seeds_root: Path,
    patient: str,
    pass_no: int,
    side: str,
) -> None:
    """
    调用单-side 的 run_first_pass_seed_pipeline.py：
      - seed 从 seeds/pass{pass_no}/{side}/{patient}/canonical.jsonl 读取
      - 显式传入 --side / --build-root / --work-root / --seeds-root
    """
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
    can = patient_root / patient / "canonical.jsonl"
    dia = patient_root / patient / "diagnosis.jsonl"
    emb = patient_root / patient / "embedding_search_other_candidate_canonical.jsonl"

    # # --- STEP 1：先并入 diagnosis，再并入 embedding 候选 ---
    # append_into_canonical(dia, can, "诊断")
    # append_into_canonical(emb, can, "embedding候选")

    # --- STEP 2：把合并后的 canonical 复制到 pass1 seeds（仅当前 side） ---
    copy_file(can, seed_path(seeds_root, 1, side, patient), f"pass1 {side}")

    # --- STEP 3：按照 passes 次数跑流水线（单 side） ---
    for p in range(1, total_passes + 1):
        run_one_pass(pipeline_cmd, build_root, work_root, seeds_root, patient, p, side)

    # --- 导出聚合事实（只导出当前 side） ---
    export_aggregated_facts(seeds_root, facts_export_root, patient, total_passes, side)


def main():
    ap = argparse.ArgumentParser(
        "Run multi-pass pipeline with diagnosis + embedding merge (single-side)"
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
        default="run_first_pass_seed_pipeline.py",
        help="Path to run_first_pass_seed_pipeline.py (default: same dir as this script).",
    )
    args = ap.parse_args()

    if args.passes < 1:
        raise SystemExit("--passes 必须 ≥ 1")

    # 计算 project_root（原来的 ROOT='../../'）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]  # .../TrialGPT-SMT/

    # 解析 build_root：绝对路径直接用，否则认为是相对 project_root
    build_root = Path(args.build_root)
    if not build_root.is_absolute():
        build_root = (project_root / build_root).resolve()

    patient_root = build_root / "patient_coded_results"
    seeds_root = build_root / "seeds"
    work_root = build_root / "passes"
    facts_export_root = build_root / "patient_facts_export"

    # 解析 pipeline 脚本路径
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
