#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import List, Dict

#ALLOWED_CATEGORIES = {"A", "B", "C", "D", "E"}


class ParsingError(Exception):
    """Raised when we cannot parse the LLM output into valid JSON."""
    pass


class VerificationError(Exception):
    """Raised when the parsed JSON does not satisfy schema/constraints."""
    pass


def _extract_json_block(text: str) -> str:
    """
    Extract the first top-level JSON array from the text.
    This is robust to models that wrap the JSON in prose.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ParsingError("Could not find a JSON array in the LLM output.")
    return text[start : end + 1]


def _parse_raw_output(raw_output: str):
    """
    Parse raw model output into a Python object (list of dicts).
    """
    json_str = _extract_json_block(raw_output)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParsingError(f"JSON decoding failed: {e}") from e

    if not isinstance(parsed, list):
        raise ParsingError("Top-level JSON structure must be a list.")
    return parsed


def _verify_categories_for_positive_constraint_literals(
    parsed: List[Dict],
    expected_count: int,
    template_kind: str,
) -> Dict[str, List[str]]:
    """
    Verify that:
      1) Number of items matches expected_count.
      2) Each item has 'entity_variable_idx' and 'target_category'.
      3) For each item at list position i, entity_variable_idx == str(i).
      4) target_category is a non-empty list of allowed category letters.
      5) target_category contains each letter at most once (no duplicates).
      6) Non-relevant category is exclusive:
           - findings/procedures: 'D' must appear alone if present
           - other templates:     'E' must appear alone if present
      7) Allowed categories depend on template:
           - findings/procedures: A/B/C/D
           - others:              A/B/C/D/E

    Returns:
      Dict[str, List[str]] mapping entity_variable_idx -> list of category letters.
    """
    if len(parsed) != expected_count:
        raise VerificationError(
            f"Expected {expected_count} items, got {len(parsed)}."
        )

    idx_to_category: Dict[str, List[str]] = {}

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise VerificationError(f"Item at index {i} is not an object/dict.")

        if "entity_variable_idx" not in item or "target_category" not in item:
            raise VerificationError(
                f"Item at index {i} must contain 'entity_variable_idx' and 'target_category'."
            )

        idx_str = item["entity_variable_idx"]
        categories = item["target_category"]

        if not isinstance(idx_str, str):
            raise VerificationError(
                f"'entity_variable_idx' at index {i} must be a string."
            )

        # Validate index
        try:
            idx_int = int(idx_str)
        except ValueError as e:
            raise VerificationError(
                f"'entity_variable_idx' at item index {i} = {idx_str!r} "
                f"is not an integer string."
            ) from e

        if idx_int != i:
            raise VerificationError(
                f"'entity_variable_idx' at item index {i} = {idx_int}, "
                f"but expected {i} based on list position."
            )

        # target_category must be a non-empty list
        if not isinstance(categories, list) or len(categories) == 0:
            raise VerificationError(
                f"'target_category' at item index {i} must be a non-empty list."
            )

        # Determine template-kind bucket from item["template"]
        tmpl = template_kind
        if not isinstance(tmpl, str):
            tmpl = "other"

        is_finding_or_procedure = tmpl in ("findings", "procedures")

        # Allowed categories depend on template-kind
        allowed_categories = {"A", "B", "C", "D"} if is_finding_or_procedure else {"A", "B", "C", "D", "E"}

        cleaned_categories: List[str] = []
        seen_raw_after_norm = set()

        for cat in categories:
            if not isinstance(cat, str):
                raise VerificationError(
                    f"Each category in 'target_category' at index {i} must be a string."
                )
            cat_norm = cat.strip().upper()

            if cat_norm not in allowed_categories:
                raise VerificationError(
                    f"Invalid category {cat_norm!r} at index {i} for template={tmpl!r}. "
                    f"Allowed: {sorted(allowed_categories)}."
                )

            # Enforce "appear exactly once": duplicates are an error
            if cat_norm in seen_raw_after_norm:
                raise VerificationError(
                    f"Duplicate category {cat_norm!r} in 'target_category' at item index {i}. "
                    f"Each category letter must appear exactly once."
                )
            seen_raw_after_norm.add(cat_norm)
            cleaned_categories.append(cat_norm)

        # Non-relevant exclusivity:
        # - findings/procedures: D must appear alone if present
        # - others:             E must appear alone if present
        if is_finding_or_procedure:
            if "D" in cleaned_categories and len(cleaned_categories) > 1:
                raise VerificationError(
                    f"'target_category' at item index {i} contains 'D' together with other categories "
                    f"{cleaned_categories!r}. If 'D' is present (findings/procedures), it must be the only category."
                )
        else:
            if "E" in cleaned_categories and len(cleaned_categories) > 1:
                raise VerificationError(
                    f"'target_category' at item index {i} contains 'E' together with other categories "
                    f"{cleaned_categories!r}. If 'E' is present (other templates), it must be the only category."
                )

        idx_to_category[idx_str] = cleaned_categories

    return idx_to_category


def parse_and_verify_positive_constraint_literals(
    raw_output: str,
    expected_count: int,
    template_kind: str
) -> Dict[str, List[str]]:
    """
    High-level helper:
      - parse raw output
      - verify constraints for positive literals

    Returns:
      dict[entity_variable_idx_str] = category_letter ('A'..'E')
    """
    parsed = _parse_raw_output(raw_output)
    return _verify_categories_for_positive_constraint_literals(parsed, expected_count, template_kind)
