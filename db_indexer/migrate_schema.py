#!/usr/bin/env python3
"""
Migrate trial.db to constraint-aligned naming.

Renames all tables to use the paper's constraint terminology:
  - facts_* → patient_*_constraints
  - clause_literals → constraint_clause_atoms
  - clauses → constraint_clauses
  - var_* → predicate_*
  - etc.

Usage:
    python -m db_indexer.migrate_schema build/trial.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import Dict, List, Tuple

# ── Table rename map: old_name → new_name ────────────────────────────────

TABLE_RENAMES: Dict[str, str] = {
    # Patient constraint atoms (PC(p))
    "facts_inclusion": "patient_inclusion_constraints",
    "facts_exclusion": "patient_exclusion_constraints",
    "facts_inclusion_important": "patient_inclusion_constraints_important",
    "facts_inclusion_important_all": "patient_inclusion_constraints_important_all",
    "facts_inclusion_important_ccr": "patient_inclusion_constraints_important_ccr",
    "facts_inclusion_important_chief": "patient_inclusion_constraints_important_chief",
    "facts_disease_prevention": "patient_prevention_constraints",

    # Clause structure (CNFD, DA)
    "clause_literals": "constraint_clause_atoms",
    "clauses": "constraint_clauses",
    "numerical_clauses": "numerical_constraint_clauses",
    "numerical_predicates": "numerical_constraint_predicates",
    "clause_numeric_range": "constraint_clause_numeric_range",

    # Trial-side constraint mapping (ECNF)
    "trial_side_clauses": "trial_constraint_clauses",
    "trial_sides": "trial_constraint_sides",
    "trial_clauses": "trial_merged_constraint_clauses",
    "filtered_trial_clauses": "filtered_trial_constraint_clauses",
    "filtered_merged_trial_clauses": "filtered_merged_trial_constraint_clauses",

    # Disease constraint atoms
    "disease_list_items": "disease_constraint_atoms",
    "disease_list_items_nonact": "disease_constraint_atoms_nonact",
    "disease_list_items_prevent_nonact": "disease_constraint_atoms_prevent_nonact",
    "disease_list_items_prevent_other": "disease_constraint_atoms_prevent_other",
    "disease_accepted_alternatives": "disease_constraint_alternatives",
    "disease_accepted_alternatives_nonact": "disease_constraint_alternatives_nonact",
    "disease_accepted_alternatives_prevent": "disease_constraint_alternatives_prevent",
    "disease_accepted_alternatives_prevent_nonact": "disease_constraint_alternatives_prevent_nonact",
    "disease_var2concept": "disease_predicate_concepts",
    "disease_var2concept_nonact": "disease_predicate_concepts_nonact",
    "disease_var2concept_prevent": "disease_predicate_concepts_prevent",
    "disease_var2concept_prevent_nonact": "disease_predicate_concepts_prevent_nonact",
    "disease_lifted_members": "disease_constraint_lifted_atoms",
    "disease_lifted_members_nonact": "disease_constraint_lifted_atoms_nonact",
    "disease_lifted_members_prevent": "disease_constraint_lifted_atoms_prevent",
    "disease_lifted_members_prevent_nonact": "disease_constraint_lifted_atoms_prevent_nonact",
    "disease_concept_accepted_alternatives": "disease_concept_constraint_alternatives",

    # Positive literal constraint atoms
    "positive_literals": "positive_constraint_literals",
    "positive_literals_nonact": "positive_constraint_literals_nonact",
    "positive_literals_prevention": "positive_constraint_literals_prevention",
    "positive_literals_prevention_nonact": "positive_constraint_literals_prevention_nonact",
    "positive_literal_accepted_alternatives": "positive_constraint_alternatives",
    "positive_literal_accepted_alternatives_expanded": "positive_constraint_alternatives_expanded",
    "positive_literal_accepted_alternatives_expanded_nonact": "positive_constraint_alternatives_expanded_nonact",
    "positive_literal_accepted_alternatives_expanded_prevention": "positive_constraint_alternatives_expanded_prevention",
    "positive_literal_accepted_alternatives_expanded_prevention_nonact": "positive_constraint_alternatives_expanded_prevention_nonact",

    # Lifted / alternative atoms
    "literal_accepted_alternatives": "constraint_literal_alternatives",
    "literal_accepted_alternatives_prevention": "constraint_literal_alternatives_prevention",
    "lifted_literal_members": "constraint_lifted_atoms",
    "lifted_literal_members_prevention": "constraint_lifted_atoms_prevention",

    # Variable/predicate catalog
    "var_catalog": "predicate_catalog",
    "var2concept": "predicate_to_concept",
    "concept_accepted_alternatives": "concept_constraint_alternatives",
    "concept_accepted_alternatives_expanded": "concept_constraint_alternatives_expanded",
    "concept_accepted_alternatives_nonact": "concept_constraint_alternatives_nonact",
    "concept_accepted_alternatives_prevent": "concept_constraint_alternatives_prevent",
    "concept_accepted_alternatives_prevent_nonact": "concept_constraint_alternatives_prevent_nonact",
    "concept_accepted_alternatives_prevention": "concept_constraint_alternatives_prevention",
    "concept_accepted_alternatives_prevention_nonact": "concept_constraint_alternatives_prevention_nonact",

    # Patient demographics
    "patient_demographics": "patient_demographic_constraints",

    # Default values
    "default_var_values": "default_predicate_values",

    # Usage analytics
    "clause_usage": "constraint_clause_usage",
    "clause_usage_stem": "constraint_clause_usage_stem",
    "clause_usage_stem_direction": "constraint_clause_usage_stem_direction",
    "var_usage_directional": "predicate_usage_directional",
    "direction_usage": "constraint_direction_usage",
    "top_common_clauses": "top_common_constraint_clauses",
    "top_common_clause_patterns": "top_common_constraint_clause_patterns",
}

# ── View rename map (views that reference tables) ────────────────────────

VIEW_RENAMES: Dict[str, str] = {
    "vw_fact_timeframes": "vw_constraint_timeframes",
    "vw_assumed_relevance_lifts": "vw_assumed_relevance_constraint_lifts",
    "vw_clause_reuse": "vw_constraint_clause_reuse",
    "vw_clause_reuse_stem": "vw_constraint_clause_reuse_stem",
    "vw_clause_reuse_stem_direction": "vw_constraint_clause_reuse_stem_direction",
    "vw_direction_usage": "vw_constraint_direction_usage",
    "vw_member_timeframes": "vw_constraint_member_timeframes",
    "vw_var_usage_directional": "vw_predicate_usage_directional",
}


def get_existing_tables(conn: sqlite3.Connection) -> set:
    """Get all table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def get_existing_views(conn: sqlite3.Connection) -> set:
    """Get all view names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()
    return {r[0] for r in rows}


def migrate(db_path: str, dry_run: bool = False) -> Dict[str, str]:
    """
    Rename tables and views in trial.db to constraint-aligned names.

    Returns dict of {old_name: new_name} for tables actually renamed.
    """
    conn = sqlite3.connect(db_path)
    existing_tables = get_existing_tables(conn)
    existing_views = get_existing_views(conn)

    renamed = {}

    # Rename tables
    for old, new in TABLE_RENAMES.items():
        if old in existing_tables and new not in existing_tables:
            sql = f'ALTER TABLE "{old}" RENAME TO "{new}"'
            if dry_run:
                print(f"  [dry-run] {sql}")
            else:
                conn.execute(sql)
            renamed[old] = new
        elif old in existing_tables and new in existing_tables:
            print(f"  [skip] {old} → {new} (target already exists)")

    # Drop old views, recreate with new names would require knowing the SQL
    # For now, just drop them — they'll be recreated on next index run
    for old, new in VIEW_RENAMES.items():
        if old in existing_views:
            if dry_run:
                print(f"  [dry-run] DROP VIEW {old}")
            else:
                conn.execute(f'DROP VIEW IF EXISTS "{old}"')

    if not dry_run:
        conn.commit()
        # Recompute query planner statistics after renames
        conn.execute("ANALYZE")
        conn.commit()

    conn.close()
    return renamed


def main():
    ap = argparse.ArgumentParser(description="Migrate trial.db to constraint-aligned naming")
    ap.add_argument("db", help="Path to trial.db")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    args = ap.parse_args()

    print(f"Migrating {args.db}...")
    renamed = migrate(args.db, dry_run=args.dry_run)
    print(f"\nRenamed {len(renamed)} tables.")

    if renamed:
        for old, new in sorted(renamed.items()):
            print(f"  {old} → {new}")


if __name__ == "__main__":
    main()
