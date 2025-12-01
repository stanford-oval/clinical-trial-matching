#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import sys
from pathlib import Path

db = Path(sys.argv[1]).resolve()
print("DB:", db)

conn = sqlite3.connect(str(db))
cur = conn.cursor()

print("\nTables:")
for name, in cur.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
      AND name LIKE '%nonact%'
    ORDER BY name
"""):
    print(" ", name)

targets = [
    "disease_constraint_alternatives_nonact",
    "positive_constraint_alternatives_expanded_nonact",
    "disease_constraint_alternatives_prevent_nonact",
    "positive_constraint_alternatives_expanded_prevention_nonact",
    "positive_constraint_literals",
    "disease_constraint_atoms",
]

for t in targets:
    exists = cur.execute("""
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type='table' AND name=?
    """, (t,)).fetchone()[0]
    print(f"\n[{t}] exists={exists}")
    if not exists:
        continue

    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
    print("  cols:", cols)

    try:
        if "trial_id" in cols:
            c = cur.execute(f"SELECT COUNT(*) FROM {t} WHERE trial_id=?", ("NCT02107001",)).fetchone()[0]
            print("  rows where trial_id='NCT02107001':", c)
        if "nct_id" in cols:
            c = cur.execute(f"SELECT COUNT(*) FROM {t} WHERE nct_id=?", ("NCT02107001",)).fetchone()[0]
            print("  rows where nct_id='NCT02107001':", c)
    except Exception as e:
        print("  query error:", e)

conn.close()