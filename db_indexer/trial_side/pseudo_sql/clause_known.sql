/* For this patient, across all merged trials:
   For each (merged_trial_id, clause_id), how many pieces do we know?
   A piece is “known” if the patient has a matching fact (0 or 1) or a number recorded. */

WITH
all_trial_clause_pairs AS (  -- all trial–clause pairs that exist
  SELECT trial_clauses.merged_trial_id,
         trial_clauses.clause_id,
         clauses.number_of_clause_members
  FROM trial_clauses
  JOIN clauses ON clauses.id = trial_clauses.clause_id
),

known_boolean AS (  -- Boolean pieces we know (patient has a row, value 0 or 1)
  SELECT clause_literals.clause_id,
         COUNT(DISTINCT clause_literals.literal_index) AS known_boolean_member_count
  FROM clause_literals
  JOIN patient_boolean_facts
    ON patient_boolean_facts.patient_id = :patient_id
   AND patient_boolean_facts.base_var   = clause_literals.base_var
   AND patient_boolean_facts.tf_token   = clause_literals.timeframe
  GROUP BY clause_literals.clause_id
),

known_numeric AS (  -- Numeric pieces we know (patient has a value)
  SELECT patient_numerical_facts_intervalized_table.clause_id,
         COUNT(DISTINCT patient_numerical_facts_intervalized_table.member_index) AS known_numeric_member_count
  FROM patient_numerical_facts_intervalized_table
  JOIN patient_numerical_facts
    ON patient_numerical_facts.patient_id = :patient_id
   AND patient_numerical_facts.base_var   = patient_numerical_facts_intervalized_table.base_var
   AND patient_numerical_facts.tf_token   = patient_numerical_facts_intervalized_table.timeframe
  GROUP BY patient_numerical_facts_intervalized_table.clause_id
)

SELECT
  all_trial_clause_pairs.merged_trial_id,
  all_trial_clause_pairs.clause_id,
  all_trial_clause_pairs.number_of_clause_members,
  known_boolean.known_boolean_member_count
+ known_numeric.known_numeric_member_count
    AS total_known_member_count,
  CASE
    WHEN known_boolean.known_boolean_member_count
       + known_numeric.known_numeric_member_count
         = all_trial_clause_pairs.number_of_clause_members
    THEN 1 ELSE 0
  END AS all_members_known_flag
FROM all_trial_clause_pairs
LEFT JOIN known_boolean ON known_boolean.clause_id = all_trial_clause_pairs.clause_id
LEFT JOIN known_numeric ON known_numeric.clause_id = all_trial_clause_pairs.clause_id
ORDER BY all_trial_clause_pairs.merged_trial_id, all_trial_clause_pairs.clause_id;





