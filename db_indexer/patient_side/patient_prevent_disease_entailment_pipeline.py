#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# ---------- 读写工具 ----------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def _read_isa_flat_json(path: Path) -> List[Dict[str, Any]]:
    """
    isa_prevent_enriched.flat.json 是 JSON array；这里逐个元素读出。
    """
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            return []
        return [obj for obj in arr if isinstance(obj, dict)]
    except Exception:
        return []


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- 去重 key ----------

TIME_KEYS = (
    "start_time_in_hours",
    "end_time_in_hours",
    "start_time_inclusive",
    "end_time_inclusive",
)

def _dedupe_key(row: Dict[str, Any]) -> Tuple[Any, Any, Any, bool, bool]:
    """
    使用统一 key 去重：
      (variable_name, start_time_in_hours, end_time_in_hours,
       start_time_inclusive, end_time_inclusive)

    variable_name = entity_variable_name 或 new_variable_name。
    """
    name = row.get("entity_variable_name") or row.get("new_variable_name") or ""
    s = row.get("start_time_in_hours")
    e = row.get("end_time_in_hours")
    si = bool(row.get("start_time_inclusive"))
    ei = bool(row.get("end_time_inclusive"))
    return (name, s, e, si, ei)


def _normalize_row_for_merge(row: Dict[str, Any], source_tag: str) -> Dict[str, Any]:
    """
    轻微规范化：
    - 如果没有 entity_variable_name，用 new_variable_name 填。
    - 如果 extracted_value 缺失，默认补 True。
    - 时间字段如果缺失就补 None / False，避免 KeyError。
    - 标记 prevent_source，方便后面排查。
    """
    out = dict(row)

    if not out.get("entity_variable_name") and out.get("new_variable_name"):
        out["entity_variable_name"] = out["new_variable_name"]

    if out.get("extracted_value") is None:
        out["extracted_value"] = True

    for k in TIME_KEYS:
        if k not in out:
            # hours 用 None，inclusive 用 False 作为默认
            if "hours" in k:
                out[k] = None
            else:
                out[k] = False

    # 标记来源（原始 / isa_prevent）
    if "prevent_source" not in out:
        out["prevent_source"] = source_tag

    return out


# ---------- 合并逻辑 ----------

def merge_disease_prevention_for_patient(
    build_root: Path,
    patient: str,
    coded_root: Path,
    isa_root: Path,
    facts_export_root: Path,
    dry_run: bool = False,
) -> None:
    """
    对单个病人：
      - 读 disease_prevention.jsonl
      - 读 isa_prevent_enriched.flat.json
      - 去重合并，写 disease_prevention.jsonl 到 patient_fact_export 下。
      即使两边都没有行，也会在 facts_export 里留一个空文件。
    """
    orig_path = coded_root / patient / "disease_prevention.jsonl"
    isa_flat_path = isa_root / patient / "isa_prevent_enriched.flat.json"

    orig_rows = _read_jsonl(orig_path)
    isa_rows_raw = _read_isa_flat_json(isa_flat_path)

    print(
        f"[merge] {patient}: original={len(orig_rows)} ISA-derived={len(isa_rows_raw)}"
    )

    merged: List[Dict[str, Any]] = []
    seen_keys: set = set()

    def add_row(r: Dict[str, Any], src_tag: str) -> None:
        nr = _normalize_row_for_merge(r, src_tag)
        k = _dedupe_key(nr)
        if k in seen_keys:
            return
        seen_keys.add(k)
        merged.append(nr)

    for r in orig_rows:
        add_row(r, "original")

    for r in isa_rows_raw:
        add_row(r, "isa_prevent")

    out_dir = facts_export_root / patient
    out_path = out_dir / "disease_prevention.jsonl"

    if dry_run:
        print(f"[merge] DRY-RUN: would write {len(merged)} rows → {out_path}")
        return

    _ensure_dir(out_dir)
    with out_path.open("w", encoding="utf-8") as fw:
        for row in merged:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 这里 merged 可以是 0 条，也会留一个空文件（0 行）
    print(
        f"[merge] {patient}: wrote {len(merged)} merged rows → {out_path}"
    )



# ---------- 调用 enrich_with_isa_prevent_disease.py ----------

def run_enrich_script(
    enrich_script: Path,
    in_root: Path,
    out_root: Path,
    patient: Optional[str],
) -> None:
    cmd = [
        sys.executable,
        str(enrich_script),
        "--in-root",
        str(in_root),
        "--out-root",
        str(out_root),
    ]
    if patient:
        cmd.extend(["--patient", patient])
    print("[ENRICH]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(enrich_script.parent))


# ---------- 清理 build_root（保留 patient_coded_results） ----------

def reset_build_root(build_root: Path, keep_dirs: List[str] = None) -> None:
    """
    删除 build_root 下除 keep_dirs 以外的所有文件/目录。
    默认 keep_dirs = ["patient_coded_results"]。
    """
    if keep_dirs is None:
        keep_dirs = ["patient_coded_results"]

    if not build_root.is_dir():
        return

    keep_set = set(keep_dirs)
    print(f"[reset] cleaning build_root: {build_root}")
    for child in build_root.iterdir():
        name = child.name
        if name in keep_set:
            print(f"[reset] keep: {child}")
            continue
        # 删除其余目录/文件
        if child.is_dir():
            print(f"[reset] rm -rf {child}")
            shutil.rmtree(child, ignore_errors=True)
        else:
            print(f"[reset] rm {child}")
            try:
                child.unlink()
            except Exception:
                pass
    print("[reset] done.\n")


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(
        description="Run ISA-prevent enrichment and merge with original disease_prevention.jsonl."
    )
    ap.add_argument(
        "--src-build-root",
        type=str,
        required=True,
        help=(
            "Build root that contains patient_disease_prevention data, "
            "e.g. <SATIR_ROOT>/patient_disease_prevention_build"
        ),
    )
    ap.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Optional: only process this patient ID (e.g., sigir-20144).",
    )
    ap.add_argument(
        "--enrich-script",
        type=str,
        default="enrich_with_isa_prevent_disease.py",
        help=(
            "Path to enrich_with_isa_prevent_disease.py "
            "(default: same directory as this script)."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be run / written, do not actually merge or write files.",
    )
    ap.add_argument(
        "--reset-build",
        action="store_true",
        help=(
            "Before running, delete everything under src-build-root "
            "EXCEPT the 'patient_coded_results' directory."
        ),
    )
    args = ap.parse_args()

    build_root = Path(args.src_build_root).expanduser().resolve()
    coded_root = build_root / "patient_coded_results"
    isa_root = build_root / "patient_coded_results_isa_prevent"
    facts_export_root = build_root / "patient_facts_export"

    if not coded_root.is_dir():
        raise SystemExit(f"[err] patient_coded_results not found at: {coded_root}")

    # enrich 脚本路径
    script_dir = Path(__file__).resolve().parent
    enrich_script = Path(args.enrich_script)
    if not enrich_script.is_absolute():
        enrich_script = (script_dir / enrich_script).resolve()
    if not enrich_script.is_file():
        raise SystemExit(f"[err] enrich script not found: {enrich_script}")


    reset_build_root(build_root, keep_dirs=["patient_coded_results"])

    # 1) 跑 enrich_with_isa_prevent_disease.py
    run_enrich_script(
        enrich_script=enrich_script,
        in_root=coded_root,
        out_root=isa_root,
        patient=args.patient,
    )

    # 2) 合并 disease_prevention.jsonl + isa_prevent_enriched.flat.json
    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted(p.name for p in coded_root.iterdir() if p.is_dir())

    for pid in patients:
        merge_disease_prevention_for_patient(
            build_root=build_root,
            patient=pid,
            coded_root=coded_root,
            isa_root=isa_root,
            facts_export_root=facts_export_root,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
