

DEMOGRAPHIC_TEMPLATE = [
    "patient_age_value_recorded_{timeframe}_in_years",
    "patient_age_value_recorded_{timeframe}_in_months",
    "patient_age_value_recorded_{timeframe}_in_days",
    "patient_sex_is_{sex}_{timeframe}",
    "patient_is_pregnant_{timeframe}",
    "patient_is_able_to_be_pregnant_{timeframe}",
    "patient_has_childbearing_potential_{timeframe}",
    "patient_is_breastfeeding_{timeframe}",
    "patient_is_lactating_{timeframe}",
    "patient_is_postmenopausal_{timeframe}",
    "patient_is_in_transition_to_menopausal_{timeframe}",
    "patient_is_infertile_{timeframe}",
    "patient_is_inpatient_{timeframe}",
    "patient_is_outpatient_{timeframe}",
    "patient_has_been_inpatient_{timeframe}",
    "patient_has_been_outpatient_{timeframe}",
    "patient_is_child_{timeframe}",
    "patient_is_adolescent_{timeframe}",
    "patient_is_adult_{timeframe}",
    "patient_is_middle_aged_{timeframe}",
    "patient_is_older_adult_{timeframe}",
    "patient_is_neonate_{timeframe}",
    "patient_is_toddler_{timeframe}",
    "patient_is_preschooler_{timeframe}",
    "patient_is_school_aged_{timeframe}",
    "patient_is_premenopausal_{timeframe}",
    "patient_is_perimenopausal_{timeframe}",
    "patient_is_postpartum_{timeframe}",
    "patient_is_postabortion_{timeframe}",
    "patient_is_emergency_department_patient_{timeframe}",
    "patient_is_long_term_care_resident_{timeframe}",
    "patient_is_nursing_home_resident_{timeframe}",
    "patient_is_assisted_living_resident_{timeframe}",
]


def preprocess_data(data):
    """
    Preprocess a list of dictionaries by filtering out entries based on conditions.

    Args:
        data (list[dict]): Input list of dictionaries.
        demographic_templates (list[str]): List of template names to exclude.

    Returns:
        list[dict]: Filtered list of dictionaries.
    """
    output = []

    for item in data:
        # Skip if "entity_variable_name" contains "withunit"
        if "entity_variable_name" in item and "withunit" in item["entity_variable_name"]:
            continue

        # Skip if "template" is in demographic list
        if "template" in item and item["template"] in DEMOGRAPHIC_TEMPLATE:
            continue

        # Keep item otherwise
        output.append(item)

    return output


