#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def prepare_sigir_build(src_build_root: Path, side: str) -> Path:
    """
    根据 side（inclusion / exclusion）构建一份新的 build 目录：

      <src_build_root>_{side}/patient_coded_results/<patient>_{side}

    只复制对应 side 的病人目录。
    例如：
      src_build_root = /.../patient_build_sigir
      side = exclusion
      → dest_build_root = /.../patient_build_sigir_exclusion
    """
    dest_build_root = Path(str(src_build_root) + f"_{side}").resolve()
    if dest_build_root.exists():
        print(f"[prep] removing existing build root: {dest_build_root}")
        shutil.rmtree(dest_build_root)

    # 源病人目录
    src_patient_root = src_build_root / "patient_coded_results"
    if not src_patient_root.is_dir():
        raise SystemExit(f"[prep] source patient root not found: {src_patient_root}")

    # 目标病人目录
    dest_patient_root = dest_build_root / "patient_coded_results"
    dest_patient_root.mkdir(parents=True, exist_ok=True)

    for entry in src_patient_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        # 只复制 *_side 的病人目录，例如 sigir-20141_exclusion
        if not name.endswith(f"_{side}"):
            continue
        dst = dest_patient_root / name
        # print(f"[prep] copy {entry} -> {dst}")
        shutil.copytree(entry, dst)

    return dest_build_root


def main():
    ap = argparse.ArgumentParser(
        description="Prepare SIGIR per-side build and run multi-pass pipeline"
    )
    ap.add_argument("--side", choices=["inclusion", "exclusion"], required=True)
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument(
        "--patient",
        type=str,
        help="Optional: run only one patient (e.g. sigir-20141_exclusion or sigir-20141)",
    )
    ap.add_argument(
        "--src-build-root",
        type=str,
        required=True,
        help=(
            "Source build root that contains combined SIGIR data "
            "(e.g. <data-root>/.../patient_build_sigir). "
            "This script will create <src-build-root>_{side} as the per-side build root."
        ),
    )
    ap.add_argument(
        "--multi-script",
        type=str,
        default=None,
        help=(
            "Path to run_multi_pass_pipeline.py. "
            "Default: <this_script_dir>/run_multi_pass_pipeline.py"
        ),
    )
    args = ap.parse_args()

    # 0) 解析源 build root
    src_build_root = Path(args.src_build_root).expanduser().resolve()
    if not src_build_root.is_dir():
        raise SystemExit(f"[prep] --src-build-root not found or not a directory: {src_build_root}")

    # 1) 根据 side 准备 per-side build_root = <src_build_root>_{side}
    dest_build_root = prepare_sigir_build(src_build_root, args.side)
    seeds_root = dest_build_root / "seeds"
    work_root = dest_build_root / "passes"

    # 2) multi-pass 脚本路径
    script_dir = Path(__file__).resolve().parent
    if args.multi_script:
        run_multi = Path(args.multi_script)
        if not run_multi.is_absolute():
            run_multi = (script_dir / run_multi).resolve()
    else:
        run_multi = script_dir / "run_multi_pass_pipeline.py"

    if not run_multi.is_file():
        raise SystemExit(f"[run] run_multi_pass_pipeline.py not found at: {run_multi}")

    # 3) 组装调用 run_multi_pass_pipeline.py 的命令
    cmd = [
        sys.executable,
        str(run_multi),
        "--passes",
        str(args.passes),
        "--side",
        args.side,
        "--build-root",
        str(dest_build_root),
    ]

    if args.patient:
        patient_name = args.patient
        # 如果你只写了 sigir-20141，就自动补上 _side
        if not patient_name.endswith(f"_{args.side}"):
            patient_name = f"{patient_name}_{args.side}"
        cmd.extend(["--patient", patient_name])

    print("[RUN]", " ".join(cmd))
    # cwd 设置成 multi-pass 脚本所在目录（方便相对路径解析）
    subprocess.run(cmd, check=True, cwd=str(run_multi.parent))


if __name__ == "__main__":
    main()


