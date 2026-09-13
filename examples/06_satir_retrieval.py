"""SatIR: narrowing a trial corpus by constraint satisfaction.

Pure SQL over the clause index -- no LLM, no Elasticsearch, no Snowstorm.

    export VERDICT_BUILD=/path/containing/trial.db
    python examples/06_satir_retrieval.py [patient_id]

The index is not in this repository (see docs/DATA.md). Without it this
script says so and exits.
"""
import collections
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smt_core.buildroot import build_root, describe  # noqa: E402

PATIENT = sys.argv[1] if len(sys.argv) > 1 else "trec-20224"
DB = build_root() / "trial.db"

if not DB.exists():
    raise SystemExit(
        f"no clause index at {DB}\n"
        f"(build root resolved from: {describe()})\n"
        "SatIR retrieves over an index built by `satir index`; it is not\n"
        "distributed with the repository. See docs/DATA.md.")

import sqlite3  # noqa: E402
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
n_trials = con.execute("select count(*) from trials").fetchone()[0]
known = {r[0] for r in con.execute(
    "select distinct patient_id from patient_inclusion_constraints")}
if PATIENT not in known:
    raise SystemExit(f"{PATIENT} is not in this index. Available: "
                     + ", ".join(sorted(known)[:5]) + " ...")

out = pathlib.Path(tempfile.mkdtemp())
print(f"index   : {DB.name}  ({n_trials:,} trials)")
print(f"patient : {PATIENT}")
print("retrieving (treat-chief: ccr + prevention + active) ...")

rc = subprocess.call([
    sys.executable, "-m", "sql_retrieval.ops.constraint_retrieval",
    "--db", str(DB), "--patient", PATIENT, "--out", str(out),
    "--scope", "any", "--important-mode", "ccr", "--alt-mode", "act",
    "--enable-prevention-hits", "--parallel", "4", "--quiet"],
    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if rc != 0:
    raise SystemExit(f"retrieval failed (exit {rc})")

f = next(out.rglob(f"{PATIENT}__*.json"), None)
lm = next((p for p in out.rglob(f"{PATIENT}__*.json") if "list_to_match" in str(p)), None)
if lm is None:
    raise SystemExit("retrieval produced no candidate list")

trials = json.loads(lm.read_text())["canonical_trials"]
by = collections.Counter(t["status"] for t in trials)
print()
print(f"  {n_trials:>6,}  trials in the index")
print(f"  {len(trials):>6,}  considered")
for status, n in by.most_common():
    print(f"  {n:>6,}  {status}")
print()
print("Every elimination carries a reason -- `explicit_contradiction` means a")
print("trial constraint is refuted by the chart, not merely unsupported by it.")
print("Survivors are what a matcher then has to decide, one at a time.")
print()
print("Recall of this filter is checkable: docs/REPRODUCE_SATIR.md")
