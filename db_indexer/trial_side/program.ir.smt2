;; ===================== IR SLICE (AST) =====================
;; STRICT token: NEED_TO_BE_STRICTLY_ENFORCED_IN_INFORMATION_RETRIEVAL
;; ===========================================================

;; Declarations needed by IR slice
(declare-const has_finding_of_hypertensive_disorder_now Bool)
(declare-const has_finding_of_diabetes_mellitus_now Bool)
(declare-const has_finding_of_tobacco_smoking_behavior_finding_now Bool)
(declare-const has_finding_of_disorder_of_lipid_metabolism_now Bool)
(declare-const has_finding_of_ischemic_heart_disease_now Bool)
(declare-const has_finding_of_cerebrovascular_disease_now Bool)
(declare-const body_mass_index_value_recorded_now_in_kg_per_m2 Real)
(declare-const has_finding_of_obesity_now Bool)
(declare-const has_finding_of_chronic_heart_failure_now Bool)
(declare-const has_finding_of_chronic_heart_failure_now@@chronic_heart_failure_nyha_class_ii_iii Bool)
(declare-const has_finding_of_electrocardiogram_abnormality_now Bool)
(declare-const has_finding_of_electrocardiogram_abnormality_now@@ecg_abnormality_left_ventricular_hypertrophy Bool)

;; Relevant auxiliary (linking) assertions
(assert (! (= has_finding_of_obesity_now (> body_mass_index_value_recorded_now_in_kg_per_m2 25.0)) :named REQ1_AUXILIARY0))
(assert (! (=> has_finding_of_chronic_heart_failure_now@@chronic_heart_failure_nyha_class_ii_iii has_finding_of_chronic_heart_failure_now) :named REQ1_AUXILIARY1))
(assert (! (=> has_finding_of_electrocardiogram_abnormality_now@@ecg_abnormality_left_ventricular_hypertrophy has_finding_of_electrocardiogram_abnormality_now) :named REQ1_AUXILIARY2))

;; STRICT constraints (for information retrieval)
(assert (! has_finding_of_hypertensive_disorder_now :named REQ0_COMPONENT0_NEED_TO_BE_STRICTLY_ENFORCED_IN_INFORMATION_RETRIEVAL))
(assert (! (or has_finding_of_diabetes_mellitus_now has_finding_of_tobacco_smoking_behavior_finding_now has_finding_of_disorder_of_lipid_metabolism_now has_finding_of_ischemic_heart_disease_now has_finding_of_cerebrovascular_disease_now has_finding_of_obesity_now has_finding_of_chronic_heart_failure_now@@chronic_heart_failure_nyha_class_ii_iii has_finding_of_electrocardiogram_abnormality_now@@ecg_abnormality_left_ventricular_hypertrophy) :named REQ1_COMPONENT0_NEED_TO_BE_STRICTLY_ENFORCED_IN_INFORMATION_RETRIEVAL))
