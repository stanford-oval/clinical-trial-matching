# irsrc/tgpt_paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

DEFAULT_BUILD_NAME = "build"

@dataclass(frozen=True)
class Paths:
    root: Path
    build: Path

def _git_toplevel(start: Path) -> Path | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=start, text=True).strip()
        return Path(out)
    except Exception:
        return None

def resolve_paths(
    root: str | Path | None = None,
    build: str | Path | None = None,
    *,
    env_root_var: str = "TRIALGPT_ROOT",
    env_build_var: str = "TRIALGPT_BUILD",
) -> Paths:
    # 1) CLI args → 2) env vars → 3) git toplevel → 4) current file’s parent → 5) cwd
    root_candidate = (
        Path(root).expanduser().resolve() if root
        else Path(os.environ.get(env_root_var, "")).expanduser().resolve() if os.environ.get(env_root_var)
        else _git_toplevel(Path.cwd())
        or Path(__file__).resolve().parents[2]  # repo root heuristic: irsrc/tgpt_paths.py -> repo/
        or Path.cwd()
    )

    build_candidate = (
        Path(build).expanduser().resolve() if build
        else Path(os.environ.get(env_build_var, "")).expanduser().resolve() if os.environ.get(env_build_var)
        else root_candidate / DEFAULT_BUILD_NAME
    )

    return Paths(root=root_candidate, build=build_candidate)

def ensure_dirs(p: Paths) -> None:
    p.build.mkdir(parents=True, exist_ok=True)

def export_env(p: Paths, env_root_var: str = "TRIALGPT_ROOT", env_build_var: str = "TRIALGPT_BUILD") -> None:
    os.environ[env_root_var] = str(p.root)
    os.environ[env_build_var] = str(p.build)
