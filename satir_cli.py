#!/usr/bin/env python3
"""
SatIR — Unified command-line interface.

Usage:
    satir setup                     Validate environment and prerequisites
    satir compile-trial <NCT_ID>    Compile trial eligibility to SMT constraints
    satir compile-patient <ID>      Compile patient notes to constraint variables
    satir index trial|patient       Index constraints into database
    satir retrieve [--patient ID]   SQL constraint satisfaction retrieval
    satir match --trial T --patient P   SMT-based eligibility checking
    satir benchmark                 Benchmark retrieval performance
    satir info                      Show configuration and system status
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_setup(args):
    """Validate environment: API keys, services, data paths, dependencies."""
    from smt_core.config import get_config
    cfg = get_config()
    checks = []

    def check(name, ok, detail=""):
        status = "OK" if ok else "FAIL"
        checks.append((name, ok))
        mark = "\u2713" if ok else "\u2717"
        print(f"  {mark} {name}" + (f" ({detail})" if detail else ""))

    print("SatIR Setup Validation\n")

    # API
    print("[API]")
    check("OPENAI_ENDPOINT", bool(cfg.api.endpoint), cfg.api.endpoint[:50] + "..." if cfg.api.endpoint else "not set")
    check("OPENAI_API_KEY", bool(cfg.api.api_key), "set" if cfg.api.api_key else "not set")
    check("Model detected", bool(cfg.api.model), cfg.api.model)

    if cfg.api.endpoint and cfg.api.api_key:
        try:
            from smt_core.engine_factory import create_engine
            engine = create_engine()
            resp = engine("Say OK")[0]
            check("LLM test call", bool(resp), f"response: {resp[:30]}...")
        except Exception as e:
            check("LLM test call", False, str(e)[:80])
    else:
        check("LLM test call", False, "skipped (no credentials)")

    # Services
    print("\n[Services]")
    import urllib.request
    for name, url in [("Snowstorm", cfg.services.snowstorm_url), ("Elasticsearch", cfg.services.elasticsearch_url)]:
        try:
            urllib.request.urlopen(url, timeout=3)
            check(name, True, url)
        except Exception:
            check(name, False, f"{url} not reachable")

    # Paths
    print("\n[Paths]")
    for name, path in [("Build dir", cfg.paths.build_dir), ("Data dir", cfg.paths.data_dir)]:
        p = Path(path)
        check(name, p.exists(), str(p.resolve()))

    db = Path(cfg.paths.build_dir) / "trial.db"
    check("trial.db", db.exists(), str(db))

    # Dependencies
    print("\n[Dependencies]")
    for mod in ["z3", "dspy", "elasticsearch", "azure.ai.inference"]:
        try:
            __import__(mod)
            check(mod, True)
        except ImportError:
            check(mod, False, "not installed")

    # Summary
    n_pass = sum(1 for _, ok in checks if ok)
    n_fail = sum(1 for _, ok in checks if not ok)
    print(f"\n{'='*50}")
    print(f"  {n_pass} passed, {n_fail} failed")
    if n_fail == 0:
        print("  Ready to run!")
    else:
        print("  Fix the above issues before running the pipeline.")
    print(f"{'='*50}")
    return 0 if n_fail == 0 else 1


def cmd_info(args):
    """Show current configuration."""
    from smt_core.config import get_config
    cfg = get_config()

    print("SatIR Configuration\n")
    print("[API]")
    print(f"  endpoint:      {cfg.api.endpoint}")
    print(f"  endpoint_gpt5: {cfg.api.endpoint_gpt5}")
    print(f"  model:         {cfg.api.model}")
    print(f"  api_key:       {'***set***' if cfg.api.api_key else '(not set)'}")

    print("\n[Services]")
    print(f"  snowstorm:     {cfg.services.snowstorm_url}")
    print(f"  elasticsearch: {cfg.services.elasticsearch_url}")

    print("\n[Paths]")
    print(f"  build_dir:     {cfg.paths.build_dir}")
    print(f"  data_dir:      {cfg.paths.data_dir}")
    print(f"  patient_dir:   {cfg.paths.patient_data_dir}")

    print("\n[Retrieval]")
    print(f"  scope:         {cfg.retrieval.scope}")
    print(f"  important:     {cfg.retrieval.important_mode}")
    print(f"  alt_mode:      {cfg.retrieval.alt_mode}")
    print(f"  parallel:      {cfg.retrieval.parallel}")
    print(f"  prevention:    {cfg.retrieval.enable_prevention}")

    print(f"\n[Config file]")
    from smt_core.config import _find_config_file
    print(f"  {_find_config_file()}")


def cmd_compile_trial(args):
    """Compile trial eligibility criteria to SMT constraints."""
    sys.argv = ["compile-trial"] + args.extra
    from trial_compiler.cli import main
    main()


def cmd_compile_patient(args):
    """Compile patient notes to canonical constraint variables."""
    sys.argv = ["compile-patient"] + args.extra
    from patient_compiler.cli import main
    main()


def cmd_index(args):
    """Index constraints into database."""
    sys.argv = ["index-db"] + args.extra
    from db_indexer.cli import main
    main()


def cmd_retrieve(args):
    """Run SQL constraint satisfaction retrieval."""
    from smt_core.config import get_config
    cfg = get_config()

    cmd = [
        sys.executable, "-m", "sql_retrieval.ops.constraint_retrieval",
        "--db", str(Path(cfg.paths.build_dir) / "trial.db"),
        "--scope", cfg.retrieval.scope,
        "--important-mode", cfg.retrieval.important_mode,
        "--alt-mode", cfg.retrieval.alt_mode,
        "--parallel", str(cfg.retrieval.parallel),
    ]
    if cfg.retrieval.enable_prevention:
        cmd.append("--enable-prevention-hits")
    cmd += args.extra

    import subprocess
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(ROOT)})
    return result.returncode


def cmd_match(args):
    """SMT-based eligibility checking."""
    sys.argv = ["match-trial"] + args.extra
    from smt_matcher.cli import main
    main()


def cmd_services(args):
    """Manage Elasticsearch + Snowstorm services."""
    import subprocess
    script = ROOT / "scripts" / "services.sh"
    cmd = [str(script)] + args.extra
    return subprocess.run(cmd).returncode


def cmd_benchmark(args):
    """Benchmark retrieval performance."""
    sys.argv = ["benchmark"] + args.extra
    exec(open(ROOT / "tests" / "benchmark_retrieval.py").read())


def main():
    ap = argparse.ArgumentParser(
        prog="satir",
        description="SatIR: Constraint-Satisfaction-Based Clinical Trial Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  satir setup                                  # Validate environment
  satir info                                   # Show configuration
  satir compile-trial NCT03362970 --side both  # Compile a trial
  satir compile-patient sigir-20141            # Compile a patient
  satir index trial                            # Index trial constraints
  satir retrieve --patient sigir-20141         # Retrieve matching trials
  satir match --trial NCT03362970 --patient sigir-20141
  satir services start                           # Start ES + Snowstorm
  satir services status                          # Check service health
  satir benchmark --warmup 2 --runs 3            # Profile performance

Configuration:
  Edit satir.toml or set environment variables (env overrides toml).
  Run 'satir info' to see current config.
  Run 'satir setup' to validate prerequisites.

Website: https://satir.genie.stanford.edu/
        """,
    )

    sub = ap.add_subparsers(dest="command")

    sub.add_parser("setup", help="Validate environment and prerequisites")
    sub.add_parser("info", help="Show current configuration")

    p = sub.add_parser("compile-trial", help="Compile trial eligibility to SMT constraints")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    p = sub.add_parser("compile-patient", help="Compile patient notes to constraint variables")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    p = sub.add_parser("index", help="Index constraints into database")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    p = sub.add_parser("retrieve", help="SQL constraint satisfaction retrieval")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    p = sub.add_parser("match", help="SMT-based eligibility checking")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    p = sub.add_parser("services", help="Manage Elasticsearch + Snowstorm (start/stop/status/install/import/check)")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    p = sub.add_parser("benchmark", help="Benchmark retrieval performance")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    args = ap.parse_args()

    if not args.command:
        ap.print_help()
        return 1

    dispatch = {
        "setup": cmd_setup,
        "info": cmd_info,
        "compile-trial": cmd_compile_trial,
        "compile-patient": cmd_compile_patient,
        "index": cmd_index,
        "retrieve": cmd_retrieve,
        "match": cmd_match,
        "services": cmd_services,
        "benchmark": cmd_benchmark,
    }

    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
