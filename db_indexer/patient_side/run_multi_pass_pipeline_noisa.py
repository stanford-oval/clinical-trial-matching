#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil, subprocess
from pathlib import Path
from collections import defaultdict   # >>> NEW


ROOT = Path("../../")
PATIENT_ROOT = ROOT / "patient_build" / "patient_coded_results"
SEEDS_ROOT   = ROOT / "patient_build" / "seeds"
WORK_ROOT    = ROOT / "patient_build" / "passes"
PIPELINE = ["python", "run_first_pass_noisa.py"]  # 如需绝对路径可替换

FACTS_EXPORT_ROOT = ROOT / "patient_build" / "patient_facts_export"   # >>> NEW


# ---------- ⬇️ 改动 1：追加诊断时做去重+值修正 -----------------
def append_diagnosis(dia: Path, can: Path):
    """把 diagnosis.jsonl 的行写进 canonical.jsonl：
       1) 'extracted_value' 为 null → True
       2) canonical 中已存在同变量(含时间键)则跳过
    """
    if not dia.exists():
        print(f"[STEP1] 无诊断文件，跳过追加：{dia}")
        return

    can.parent.mkdir(parents=True, exist_ok=True)
    can.touch(exist_ok=True)

    # ① 先收集 canonical 里已有的 key
    seen = set()
    with can.open("r", encoding="utf-8") as f:
        for line in f:
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

    # ② 逐行处理 diagnosis
    new_rows = []
    with dia.open("r", encoding="utf-8") as f:
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
                continue  # 已有 → 跳过
            seen.add(key)
            new_rows.append(obj)

    if not new_rows:
        print("[STEP1] 诊断行均已存在，无追加")
        return

    # ③ 追加写入，保证文件末尾有换行
    with can.open("ab+") as f:
        f.seek(0, 2)
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")

    print(f"[STEP1] 追加 {len(new_rows)} 行诊断 → {can}")

# -----------------------------------------------------------------

def copy_file(src: Path, dst: Path, tag: str):
    if not src.exists():
        print(f"[STEP2] 源不存在，跳过复制（{tag}）：{src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[STEP2] 复制 {src} → {dst}（{tag}）")

def discover_patients(specified):
    if specified:
        return [specified]
    return sorted(p.name for p in PATIENT_ROOT.iterdir() if p.is_dir())

def seed_path(pass_no, side, patient):
    return SEEDS_ROOT / f"pass{pass_no}" / side / patient / "canonical.jsonl"

# >>> NEW --------------------------------------------------------------------
def export_aggregated_facts(patient: str, total_passes: int):
    """把 seeds/pass*/<side>/<patient>/canonical.jsonl 全部累积去重后导出"""
    agg: dict[str, list[dict]] = defaultdict(list)  # side -> rows
    seen: dict[str, set] = {"inclusion": set(), "exclusion": set()}

    for p in range(1, total_passes + 2):            # +2 因为上个 pass 已写下一轮目录
        for side in ("inclusion", "exclusion"):
            f = SEEDS_ROOT / f"pass{p}" / side / patient / "canonical.jsonl"
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
                    if key in seen[side]:
                        continue
                    seen[side].add(key)
                    agg[side].append(obj)

    for side in ("inclusion", "exclusion"):
        out_dir = FACTS_EXPORT_ROOT / patient / side
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "canonical.jsonl"
        with out_path.open("w", encoding="utf-8") as fw:
            for row in agg[side]:
                fw.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[export] {side}: wrote {len(agg[side])} unique facts → {out_path}")
# ---------------------------------------------------------------------------


def run_one_pass(patient, pass_no):
    inc, exc = seed_path(pass_no, "inclusion", patient), seed_path(pass_no, "exclusion", patient)
    cmd = PIPELINE + [
        "--patient", patient,
        "--seed-inclusion", str(inc),
        "--seed-exclusion", str(exc),
        "--work-root", str(WORK_ROOT),
        "--pass-no", str(pass_no),
        "--seeds-root", str(SEEDS_ROOT),
        "--reset-outputs",
    ]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def process_patient(patient, total_passes):
    print(f"\n===== Patient: {patient} =====")
    can = PATIENT_ROOT / patient / "canonical.jsonl"
    dia = PATIENT_ROOT / patient / "diagnosis.jsonl"

    # --- STEP 1 ---
    append_diagnosis(dia, can)

    # --- STEP 2 ---
    copy_file(can, seed_path(1, "inclusion", patient), "pass1 inclusion")
    copy_file(can, seed_path(1, "exclusion",  patient), "pass1 exclusion")

    # --- STEP 3 ---
    for p in range(1, total_passes + 1):
        run_one_pass(patient, p)
    
    export_aggregated_facts(patient, total_passes)

def main():
    ap = argparse.ArgumentParser("Run multi-pass pipeline with diagnosis merge")
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--patient", type=str)
    args = ap.parse_args()

    if args.passes < 1:
        raise SystemExit("--passes 必须 ≥ 1")

    for patient in discover_patients(args.patient):
        try:
            process_patient(patient, args.passes)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {patient}: pass 运行失败 – {e}")

if __name__ == "__main__":
    main()
