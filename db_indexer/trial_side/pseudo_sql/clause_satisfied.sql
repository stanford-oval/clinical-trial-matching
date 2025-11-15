/* For this patient, across all merged trials:
   For each (merged_trial_id, clause_id), does the clause pass?
   OR logic: if any piece passes, the clause is satisfied. */

WITH
all_trial_clause_pairs AS (  -- all trial–clause pairs that exist
  SELECT trial_clauses.merged_trial_id,
         trial_clauses.clause_id
  FROM trial_clauses
),

satisfied_boolean AS (  -- Boolean pieces that pass
  SELECT clause_literals.clause_id,
         COUNT(DISTINCT clause_literals.literal_index) AS satisfied_boolean_member_count
  FROM clause_literals
  JOIN patient_boolean_facts
    ON patient_boolean_facts.patient_id = :patient_id
   AND patient_boolean_facts.base_var   = clause_literals.base_var
   AND patient_boolean_facts.tf_token   = clause_literals.timeframe
  WHERE (clause_literals.is_neg = 0 AND patient_boolean_facts.value = 1)  -- positive wants TRUE
     OR (clause_literals.is_neg = 1 AND patient_boolean_facts.value = 0)  -- negated wants FALSE
  GROUP BY clause_literals.clause_id
),

satisfied_numeric AS (  -- Numeric pieces that pass (inside the allowed range)
  SELECT patient_numerical_facts_intervalized_table.clause_id,
         COUNT(DISTINCT patient_numerical_facts_intervalized_table.member_index) AS satisfied_numeric_member_count
  FROM patient_numerical_facts_intervalized_table
  JOIN patient_num_facts_tok
    ON patient_num_facts_tok.patient_id = :patient_id
   AND patient_num_facts_tok.base_var   = patient_numerical_facts_intervalized_table.base_var
   AND patient_num_facts_tok.tf_token   = patient_numerical_facts_intervalized_table.timeframe
  WHERE (patient_numerical_facts_intervalized_table.lb IS NULL
         OR patient_num_facts_tok.value > patient_numerical_facts_intervalized_table.lb
         OR (patient_num_facts_tok.value = patient_numerical_facts_intervalized_table.lb AND patient_numerical_facts_intervalized_table.lb_inc = 1))
    AND (patient_numerical_facts_intervalized_table.ub IS NULL
         OR patient_num_facts_tok.value < patient_numerical_facts_intervalized_table.ub
         OR (patient_num_facts_tok.value = patient_numerical_facts_intervalized_table.ub AND patient_numerical_facts_intervalized_table.ub_inc = 1))
  GROUP BY patient_numerical_facts_intervalized_table.clause_id
)

SELECT
  all_trial_clause_pairs.merged_trial_id,
  all_trial_clause_pairs.clause_id,
  satisfied_boolean.satisfied_boolean_member_count
+ satisfied_numeric.satisfied_numeric_member_count
    AS satisfied_member_count,
  CASE
    WHEN satisfied_boolean.satisfied_boolean_member_count
       + satisfied_numeric.satisfied_numeric_member_count > 0
    THEN 1 ELSE 0
  END AS clause_satisfied_flag
FROM all_trial_clause_pairs
LEFT JOIN satisfied_boolean ON satisfied_boolean.clause_id = all_trial_clause_pairs.clause_id
LEFT JOIN satisfied_numeric  ON satisfied_numeric.clause_id  = all_trial_clause_pairs.clause_id
ORDER BY all_trial_clause_pairs.merged_trial_id, all_trial_clause_pairs.clause_id;








