#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
categorization_parser.py (modified)

Adds "adopt whatever was returned" support:

- parse_and_verify(raw_output, expected_diseases): STRICT (original behavior)
- parse_partial(raw_output, expected_diseases): BEST-EFFORT
    * parses the same JSON array-of-objects format
    * returns a dict[disease_idx_str] -> category_letter for all VALID items it can salvage
    * does NOT require full coverage / exact length
    * silently drops invalid/mismatched/duplicate/out-of-range items

This lets your DiseaseCategorizer do:
  try strict retries
  if still fails => partial = parse_partial(...) and fill missing with '?' (or leave uncategorized).
"""

import json
from typing import List, Dict, Any, Optional

ALLOWED_CATEGORIES = {"A", "B", "C", "D"}


class ParsingError(Exception):
    """Raised when we cannot parse the LLM output into valid JSON."""
    pass


class VerificationError(Exception):
    """Raised when the parsed JSON does not satisfy schema/constraints."""
    pass


def _extract_json_block(text: str) -> str:
    """
    Extract the first top-level JSON array from the text.
    Robust to models that wrap the JSON in prose.

    NOTE: This is *intentionally* simple and matches your original behavior.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ParsingError("Could not find a JSON array in the LLM output.")
    return text[start: end + 1]


def _parse_raw_output(raw_output: str) -> List[Dict[str, Any]]:
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


def _verify_categories(
    parsed: List[Dict[str, Any]],
    expected_diseases: List[str],
) -> Dict[str, List[str]]:
    """
    Verify that:
      1) Number of items matches number of expected diseases.
      2) Each item has 'disease', 'disease_idx', and 'trial_effect_category'.
      3) 'disease_idx' is an integer-like string in [0, len(expected_diseases)-1].
      4) The 'disease' field matches expected_diseases[disease_idx].
      5) trial_effect_category is a non-empty list of allowed category letters.
      6) Category D (not of clinical interest) doesn't present with other categories
      7) Each disease_idx is used exactly once.

    Returns:
      A dict mapping disease_idx (as string) -> List[str] of category letters.
    """
    n_expected = len(expected_diseases)

    if len(parsed) != n_expected:
        raise VerificationError(f"Expected {n_expected} items, got {len(parsed)}.")

    seen_indices = set()
    idx_to_category: Dict[str, List[str]] = {}

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise VerificationError(f"Item at index {i} is not an object/dict.")

        # Required keys
        if "disease" not in item or "disease_idx" not in item or "trial_effect_category" not in item:
            raise VerificationError(
                f"Item at index {i} must contain 'disease', 'disease_idx', "
                f"and 'trial_effect_category'."
            )

        disease = item["disease"]
        disease_idx = item["disease_idx"]
        category = item["trial_effect_category"]

        if not isinstance(disease, str):
            raise VerificationError(f"'disease' at index {i} must be a string.")
        if not isinstance(disease_idx, str):
            raise VerificationError(f"'disease_idx' at index {i} must be a string.")

        try:
            idx_int = int(disease_idx)
        except ValueError as e:
            raise VerificationError(
                f"'disease_idx' at item index {i} = {disease_idx!r} is not an integer string."
            ) from e

        if not (0 <= idx_int < n_expected):
            raise VerificationError(
                f"'disease_idx' at item index {i} = {idx_int} out of range [0, {n_expected - 1}]."
            )

        expected_name = expected_diseases[idx_int]
        if disease != expected_name:
            raise VerificationError(
                f"'disease' at item index {i} = {disease!r} does not match "
                f"expected disease for index {idx_int}: {expected_name!r}."
            )

        # ---- NEW: category must be a non-empty list of allowed letters ----
        if not isinstance(category, list) or len(category) == 0:
            raise VerificationError(
                f"'trial_effect_category' at item index {i} must be a non-empty list."
            )

        cleaned_letters: List[str] = []
        for L in category:
            if not isinstance(L, str):
                raise VerificationError(
                    f"Each entry in 'trial_effect_category' at index {i} must be a string."
                )
            L = L.strip().upper()
            if L not in ALLOWED_CATEGORIES:
                raise VerificationError(
                    f"'trial_effect_category' contains invalid letter {L!r} at item index {i}. "
                    f"Allowed: {sorted(ALLOWED_CATEGORIES)}."
                )
            if L not in cleaned_letters:
                cleaned_letters.append(L)

        # If D ("not of clinical interest") is selected, it must be exclusive
        if "D" in cleaned_letters and len(cleaned_letters) > 1:
            raise VerificationError(
                f"'trial_effect_category' at item index {i} contains 'D' together with "
                f"other categories {cleaned_letters!r}. If 'D' is present, it must be the only category."
            )
        
        # Ensure each index appears once
        if idx_int in seen_indices:
            raise VerificationError(f"Duplicate 'disease_idx': {idx_int}.")
        seen_indices.add(idx_int)
        idx_to_category[disease_idx] = cleaned_letters

    if len(seen_indices) != n_expected:
        missing = sorted(set(range(n_expected)) - seen_indices)
        raise VerificationError(f"Missing indices in output: {missing}")

    return idx_to_category


def parse_and_verify(
    raw_output: str,
    expected_diseases: List[str],
) -> Dict[str, List[str]]:
    """
    High-level helper:
      - parse raw output
      - verify constraints

    Returns:
      dict[disease_idx_str] = List[category_letters]
    """
    parsed = _parse_raw_output(raw_output)
    return _verify_categories(parsed, expected_diseases)


# ─────────────────────────────────────────────────────────────
# NEW: Partial parsing fallback
# ─────────────────────────────────────────────────────────────

def parse_partial(raw_output: str, expected_diseases: List[str]) -> Dict[str, str]:
    """
    BEST-EFFORT parse:
      - returns whatever valid (idx -> category) items we can salvage
      - DOES NOT require len(parsed) == n_expected
      - DOES NOT require full index coverage
      - DOES NOT raise VerificationError for missing items
      - Drops invalid objects silently (wrong types, out-of-range indices, mismatched disease names, invalid categories, duplicates)

    Raises:
      ParsingError only when we cannot even parse a JSON array out of the text.
    """
    parsed = _parse_raw_output(raw_output)  # may raise ParsingError
    n_expected = len(expected_diseases)

    seen_indices = set()
    idx_to_category: Dict[str, str] = {}

    for item in parsed:
        if not isinstance(item, dict):
            continue

        disease = item.get("disease")
        disease_idx = item.get("disease_idx")
        category = item.get("target_category")

        if not isinstance(disease, str) or not isinstance(disease_idx, str) or not isinstance(category, str):
            continue

        # parse idx
        try:
            idx_int = int(disease_idx)
        except Exception:
            continue
        if not (0 <= idx_int < n_expected):
            continue

        # disease name must match the expected disease at that index (keep semantics aligned)
        if disease != expected_diseases[idx_int]:
            continue

        category_u = category.strip().upper()
        if category_u not in ALLOWED_CATEGORIES:
            continue

        # keep first occurrence; drop duplicates
        if idx_int in seen_indices:
            continue

        seen_indices.add(idx_int)
        idx_to_category[str(idx_int)] = category_u

    return idx_to_category
