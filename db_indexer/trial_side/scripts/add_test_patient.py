#!/usr/bin/env python3
# add_test_patient_age_gender.py
from __future__ import annotations
import argparse, sqlite3, time
from pathlib import Path

def add_test_patient_age_gender(
    db_path: str | Path,
    patient_id: str = "pat_age_gender_01",
    age_years: float = 64.0,
    sex: str = "male",              # "male" or "female"
    observed_at: float | None = None,
) -> None:
    if observed_at is None:
        observed_at = time.time()

    sex = sex.lower().strip()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    # Base var names align with your timeframe-aware schema (no timeframe suffix here)
    base_age   = "age_value_recorded_in_years"
    base_male  = "patient_sex_is_male"
    base_female= "patient_sex_is_female"

    male_val   = 1 if sex == "male" else 0
    female_val = 1 - male_val

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Ensure tables exist (no-op if already created by your builder)
        cur.execute("""CREATE TABLE IF NOT EXISTS patient_facts_ts (
            patient_id  TEXT NOT NULL,
            base_var    TEXT NOT NULL,
            value       INTEGER NOT NULL CHECK (value IN (0,1)),
            observed_at REAL NOT NULL,
            PRIMARY KEY (patient_id, base_var, observed_at)
        )""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pf_ts_var ON patient_facts_ts(base_var, observed_at)")
        cur.execute("""CREATE TABLE IF NOT EXISTS patient_num_facts_ts (
            patient_id  TEXT NOT NULL,
            base_var    TEXT NOT NULL,
            value       REAL NOT NULL,
            observed_at REAL NOT NULL,
            PRIMARY KEY (patient_id, base_var, observed_at)
        )""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pnf_ts_var ON patient_num_facts_ts(base_var, observed_at)")

        # Insert boolean gender facts (both keys, one true, one false)
        cur.executemany(
            "INSERT OR REPLACE INTO patient_facts_ts (patient_id, base_var, value, observed_at) VALUES (?,?,?,?)",
            [
                (patient_id, base_male,   male_val,   observed_at),
                (patient_id, base_female, female_val, observed_at),
            ],
        )
        # Insert numeric age
        cur.execute(
            "INSERT OR REPLACE INTO patient_num_facts_ts (patient_id, base_var, value, observed_at) VALUES (?,?,?,?)",
            (patient_id, base_age, float(age_years), observed_at),
        )
        conn.commit()
        print(f"[ok] inserted test patient {patient_id}: age={age_years}, sex={sex}, ts={observed_at}")
    finally:
        conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to your expr_index.db (or trial.db)")
    ap.add_argument("--patient-id", default="pat_age_gender_01")
    ap.add_argument("--age", type=float, default=64.0)
    ap.add_argument("--sex", choices=["male","female"], default="male")
    ap.add_argument("--observed-at", type=float, default=None, help="Epoch seconds; default now()")
    args = ap.parse_args()
    add_test_patient_age_gender(db_path=args.db, patient_id=args.patient_id, age_years=args.age, sex=args.sex, observed_at=args.observed_at)

"""
python add_test_patient_age_gender.py --db ../../build/trial.db \
  --patient-id pat_age_gender_01 --age 64 --sex male
"""
