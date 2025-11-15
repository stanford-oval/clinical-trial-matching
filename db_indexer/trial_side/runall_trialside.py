#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, subprocess, sys, shlex, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "scripts"
CLAUSE_SCRIPTS_DIR = SCRIPTS_DIR  # same as trial_side scripts

REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE
sys.path.insert(0, str(REPO_ROOT / "trial_side"))
sys.path.insert(0, str(REPO_ROOT / "trial_side" / "lib"))
sys.path.insert(0, str(REPO_ROOT / "irsrc"))

try:
    from path_utils import resolve_paths, ensure_dirs, export_env  # type: ignore
except Exception:
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class _P: root: Path; build: Path
    def resolve_paths(root=None, build=None):  # type: ignore
        rr = Path(root).resolve() if root else REPO_ROOT
        bb = Path(build).resolve() if build else Path(os.environ.get("SATIR_BUILD", rr / "build")).resolve()
        return _P(root=rr, build=bb)
    def ensure_dirs(p): p.build.mkdir(parents=True, exist_ok=True)  # type: ignore
    def export_env(p):  # type: ignore
        os.environ["SATIR_ROOT"]  = str(p.root)
        os.environ["SATIR_BUILD"] = str(p.build)
        # backward compat
        os.environ["TRIALGPT_ROOT"]  = str(p.root)
        os.environ["TRIALGPT_BUILD"] = str(p.build)

def run(cmd: list[str], env: dict[str, str]) -> int:
    print("▶", " ".join(cmd))
    return subprocess.call(cmd, env=env)

def _db_table_count(db_path: Path, table: str) -> int:
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not row or row[0] == 0:
            conn.close()
            return 0
        (cnt,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        conn.close()
        return int(cnt or 0)
    except Exception:
        return 0

def main() -> int:
    ap = argparse.ArgumentParser(
        description="TrialGPT-SMT controller (find positives → ingest positives → decide → lift)."
    )
    ap.add_argument("--root", type=Path, help="Repo root (overrides TRIALGPT_ROOT).")
    ap.add_argument("--build", type=Path, help="Build dir (overrides TRIALGPT_BUILD).")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    ap.add_argument("--dry-run", action="store_true", help="Print steps, then exit.")

    step_choices = [
        "normalize","slice","link","minify","project","db",
        "find_poslit","poslit","decide","lift",
        "decide_clause","lift_clause",
        "all"
    ]
    ap.add_argument("--steps", nargs="*", choices=step_choices, default=["all"])

    ap.add_argument("--db", type=str, default=None, help="DB path (default: <BUILD>/trial.db)")
    ap.add_argument("--finder-args", type=str, default="", help="Extra args for find_positive_canon_literals.py")
    ap.add_argument("--poslit-args", type=str, default="", help="Extra args for ingest_positive_constraint_literals.py")
    ap.add_argument("--lift-args", type=str, default="", help="Extra args for ontology_lifter.py")
    ap.add_argument("--decider-args", type=str, default="", help="Extra args for hop_policy_decider_lineage_mgsr.py")

    ap.add_argument("--lift-clause-args", type=str, default="", help="Extra args for ontology_lifter_clause.py")
    ap.add_argument("--decider-clause-args", type=str, default="", help="Extra args for hop_policy_decider_lineage_mgsr_clause.py")

    args = ap.parse_args()

    paths = resolve_paths(args.root, args.build)
    ensure_dirs(paths)
    export_env(paths)

    base_db = Path(args.db) if args.db else (paths.build / "trial.db")

    # categorized positives dir
    pos_per_file_dir = paths.build / "positive_constraint_literals_categorized" / "per_file"

    canon_dir = paths.build / "canon"
    minified_dir = paths.build / "minified_canon"

    print(f"""
Config:
  ROOT   = {paths.root}
  BUILD  = {paths.build}
  DB     = {base_db}
  POSLIT = {pos_per_file_dir}
""")

    name_to_script = {
        "normalize": SCRIPTS_DIR / "normalize_units.py",
        "slice":     SCRIPTS_DIR / "batch_slice_ir.py",
        "link":      SCRIPTS_DIR / "batch_link_qualifiers.py",
        "minify":    SCRIPTS_DIR / "batch_minify_canon.py",
        "project":   SCRIPTS_DIR / "batch_project_canon.py",
        "db":        SCRIPTS_DIR / "smt_clause_db.py",
        "find_poslit": SCRIPTS_DIR / "find_positive_canon_literals.py",
        "poslit":      SCRIPTS_DIR / "ingest_positive_constraint_literals.py",
        "decide":      SCRIPTS_DIR / "hop_policy_decider_lineage_mgsr_new.py",
        "lift":        SCRIPTS_DIR / "ontology_lifter_new.py",
        "decide_clause": CLAUSE_SCRIPTS_DIR / "hop_policy_decider_lineage_mgsr_clause.py",
        "lift_clause":   CLAUSE_SCRIPTS_DIR / "ontology_lifter_clause.py",
    }

    steps = list(name_to_script.keys()) if "all" in args.steps else args.steps

    if args.dry_run:
        print("Would run:")
        for s in steps:
            print(f"  - {s} -> {name_to_script[s]}")
        return 0

    env = os.environ.copy()

    # ── PREP: rebuild predicate_to_concept using a dedicated script (recommended)
    # This avoids calling the NEW lift-only ontology_lifter with rebuild flags.
    need_predicate_to_concept = any(s in steps for s in ("decide","lift","decide_clause","lift_clause"))
    if need_predicate_to_concept:
        rebuild_script = (SCRIPTS_DIR / "rebuild_predicate_to_concept.py").resolve()
        if rebuild_script.exists():
            prep_cmd = [
                args.python, str(rebuild_script),
                "--db", str(base_db),
                "--canon-dir", str(canon_dir),
                "--minified-canon-dir", str(minified_dir),
                "--overwrite",
            ]
            print("\n[prep] rebuilding predicate_to_concept via rebuild_predicate_to_concept.py")
            if run(prep_cmd, env) != 0:
                return 2
        else:
            print("\n[prep] WARNING: scripts/rebuild_predicate_to_concept.py not found; skipping predicate_to_concept rebuild.")
            print("      If predicate_to_concept is missing, decide/lift will fail. Add rebuild_predicate_to_concept.py or restore rebuild flags.")

    # ── Auto find + ingest if missing/empty
    if "decide" in steps or "lift" in steps:
        if _db_table_count(base_db, "positive_constraint_literals") == 0:
            print("[prep] positive_constraint_literals empty → running finder + ingest")

            find_cmd = [
                args.python,
                str((SCRIPTS_DIR / "find_positive_canon_literals.py").resolve()),
                str(paths.build),
            ] + (shlex.split(args.finder_args) if args.finder_args else [])
            if run(find_cmd, env) != 0:
                return 2

            ingest_cmd = [
                args.python,
                str((SCRIPTS_DIR / "ingest_positive_constraint_literals.py").resolve()),
                "--db", str(base_db),
                "--per-file-dir", str(pos_per_file_dir),
            ] + (shlex.split(args.poslit_args) if args.poslit_args else [])
            if run(ingest_cmd, env) != 0:
                return 2

    # ── MAIN STEPS
    for step in steps:
        script = name_to_script[step].resolve()
        if not script.exists():
            print(f"❌ Missing script: {script}")
            return 2

        cmd = [args.python, str(script)]

        if step == "find_poslit":
            cmd += (shlex.split(args.finder_args) if args.finder_args else [])
            cmd += [str(paths.build)]

        elif step == "poslit":
            cmd += ["--db", str(base_db), "--per-file-dir", str(pos_per_file_dir)]
            cmd += (shlex.split(args.poslit_args) if args.poslit_args else [])

        elif step == "decide":
            cmd += ["--db", str(base_db)]
            cmd += (shlex.split(args.decider_args) if args.decider_args else [])

        elif step == "lift":
            cmd += ["--db", str(base_db)]
            cmd += (shlex.split(args.lift_args) if args.lift_args else [])

        elif step in ("decide_clause","lift_clause"):
            cmd += ["--db", str(base_db)]

        elif step == "db":
            cmd += ["--db", str(base_db)]

        print(f"\n[run] {step}")
        if run(cmd, env) != 0:
            print(f"❌ step failed: {step}")
            return 2

    print("\n✅ Pipeline complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
