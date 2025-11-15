VIEW inclusion_trial_clauses            -- (merged_trial_id, clause_id)
  = SELECT trials.id AS merged_trial_id, trial_side_clauses.clause_id
    FROM trials
    JOIN trial_side_clauses
      ON trial_side_clauses.trial_id = trials.inclusion_trial_side_id;

VIEW exclusion_trial_clauses            -- (merged_trial_id, clause_id)
  = SELECT trials.id AS merged_trial_id, trial_side_clauses.clause_id
    FROM trials
    LEFT JOIN trial_side_clauses
      ON trial_side_clauses.trial_id = trials.exclusion_trial_side_id;

-- Per inclusion clause: known/satisfied member counts (Boolean, Numeric)
VIEW inclusion_known_boolean_members    -- (merged_trial_id, clause_id, known_boolean_member_count)
  = JOIN clause_literals WITH patient_facts_tok
    ON base_var AND tf_token = COALESCE(timeframe,'inthehistory');

VIEW inclusion_satisfied_boolean_members -- (..., satisfied_boolean_member_count)
  = same join + satisfaction: (is_neg=0 AND value=1) OR (is_neg=1 AND value=0);

VIEW inclusion_known_numeric_members     -- (merged_trial_id, clause_id, known_numeric_member_count)
  = JOIN clause_numeric_range WITH patient_num_facts_tok
    ON base_var AND tf_token = COALESCE(timeframe,'inthehistory');

VIEW inclusion_satisfied_numeric_members -- (..., satisfied_numeric_member_count)
  = same join + range pass:
    (lb IS NULL OR val > lb OR (val = lb AND lb_inc=1))
 AND (ub IS NULL OR val < ub OR (val = ub AND ub_inc=1));

VIEW inclusion_known_vs_satisfied        -- (merged_trial_id, clause_id,
                                         --  number_of_clause_members,
                                         --  total_known_member_count,
                                         --  total_satisfied_member_count)
  = JOIN inclusion_trial_clauses WITH clauses
  + LEFT JOIN the four inclusion_* views
  + sum boolean+numeric counts.

VIEW inclusion_contradicted_trials       -- (merged_trial_id)
  = FROM inclusion_known_vs_satisfied
    WHERE number_of_clause_members > 0
      AND total_known_member_count = number_of_clause_members
      AND total_satisfied_member_count = 0;

-- Exclusion side: same four “known/satisfied” views + combine:
VIEW exclusion_known_boolean_members
VIEW exclusion_satisfied_boolean_members
VIEW exclusion_known_numeric_members
VIEW exclusion_satisfied_numeric_members

VIEW exclusion_known_vs_satisfied        -- (... same columns as inclusion)
VIEW exclusion_contradicted_trials       -- (merged_trial_id) with the same rule

-- Final outputs (for slides):
VIEW trial_elimination_tok               -- (merged_trial_id, nct_id,
                                         --  eliminated_by_exclusion_flag,
                                         --  eliminated_by_contradicted_inclusion_flag,
                                         --  eliminated_flag)
  = SELECT from trials
    + LEFT JOIN exclusion_contradicted_trials
    + LEFT JOIN inclusion_contradicted_trials
    + compute the three flags.

-- For ranking survivors:
VIEW inclusion_satisfied_clauses         -- (merged_trial_id, clause_id)
  = DISTINCT UNION of
    inclusion_satisfied_boolean_members and inclusion_satisfied_numeric_members.

VIEW inclusion_total_clause_counts       -- (merged_trial_id, inclusion_total_clauses)
  = COUNT DISTINCT clause_id FROM inclusion_trial_clauses;

VIEW inclusion_certainly_satisfied_clause_counts
                                         -- (merged_trial_id, inclusion_certainly_satisfied_clauses)
  = COUNT DISTINCT clause_id FROM inclusion_satisfied_clauses;

VIEW trial_ranking_tok                   -- (merged_trial_id, nct_id,
                                         --  inclusion_total_clauses,
                                         --  inclusion_certainly_satisfied_clauses,
                                         --  inclusion_certainly_satisfied_fraction)
  = SELECT from trials
    + LEFT JOIN inclusion_total_clause_counts
    + LEFT JOIN inclusion_certainly_satisfied_clause_counts
    WHERE trials.id NOT IN inclusion_contradicted_trials
      AND trials.id NOT IN exclusion_contradicted_trials
    + fraction = satisfied / total (0 if total=0).
