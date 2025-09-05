import json


def dict_to_readable_string(data_dict):
    """
    Convert a Python dictionary into a nicely formatted string.

    Args:
        data_dict (dict): The dictionary to convert.

    Returns:
        str: A readable JSON-formatted string.
    """
    return json.dumps(data_dict, indent=2)


def _load_patient_notes(path: str) -> list[dict]:
    """
    Read a JSON-Lines file where every line is:
        {"_id": "...", "text": "...", "metadata": {...}}
    Returns a list of dicts *in the same order* they appear in the file.
    """
    notes = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                notes.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {ln} of {path}: {e}") from None
    return notes