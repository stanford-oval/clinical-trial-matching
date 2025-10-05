from __future__ import annotations
from typing import Dict, List
import json
import os
import logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s - %(message)s",
)
log = logging.getLogger("deterministic_extractor")

# ────────────────────────────────────────────────────────────────
# 1.  Robust field resolver
# ────────────────────────────────────────────────────────────────
def get_raw_criteria(info: Dict, inc_exc: str) -> str:
    """
    Return the raw criteria string for 'inclusion' or 'exclusion',
    trying several common field names / nestings.

    Order of precedence:
      1) top-level "<inc_exc>_criteria"
      2) metadata["<inc_exc>_criteria"]
      3) metadata["<inc_exc>"]
      4) top-level "<inc_exc>"
    """
    key_flat = f"{inc_exc}_criteria"

    # a. flat key at root
    if key_flat in info:
        return info[key_flat]

    # b. under metadata dict
    meta = info.get("metadata", {})
    if isinstance(meta, dict):
        if key_flat in meta:
            return meta[key_flat]
        if inc_exc in meta:
            return meta[inc_exc]

    # c. simple short key at root
    if inc_exc in info:
        return info[inc_exc]

    raise KeyError(
        f"Could not locate raw '{inc_exc}' criteria field. "
        "Checked root and metadata for keys "
        f"'{key_flat}' / '{inc_exc}_criteria' / '{inc_exc}'."
    )

# ────────────────────────────────────────────────────────────────
# 2.  Line-based criteria splitter (handles bullets, numbers, dashes)
# ────────────────────────────────────────────────────────────────
def parse_criteria(criteria: str) -> str:
    """
    Deterministically split eligibility text into individual constraint_clauses and
    return a *numbered* block, one clause per line.

    Bullet styles handled:  •   -   1)   1.
    Heading lines containing 'Inclusion criteria' / 'Exclusion criteria'
    are ignored.
    """
    numbered: List[str] = []
    idx = 0

    for raw in criteria.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if "inclusion criteria" in lower or "exclusion criteria" in lower:
            continue

        # strip common bullet prefixes
        line = line.lstrip("•-").lstrip("0123456789.) ").strip()
        if len(line) < 5:
            continue

        numbered.append(f"{idx}. {line}")
        idx += 1

    return "\n".join(numbered)

# (optional) pretty-printer for debugging
def print_trial(trial_info: Dict, inc_exc: str = "inclusion") -> str:
    txt  = f"Title: {trial_info.get('brief_title', trial_info.get('title', ''))}\n"
    txt += f"Target diseases: {', '.join(trial_info.get('diseases_list', []))}\n"
    txt += f"Interventions: {', '.join(trial_info.get('drugs_list', []))}\n"
    txt += f"Summary: {trial_info.get('brief_summary', trial_info.get('text', '')[:300])}\n\n"
    raw = get_raw_criteria(trial_info if 'inclusion_criteria' in trial_info else trial_info.get('metadata', {}), inc_exc)
    txt += f"{inc_exc.capitalize()} criteria:\n{parse_criteria(raw)}"
    return txt

# ────────────────────────────────────────────────────────────────
# 3.  dspy Module wrapper (callable directly too)
# ────────────────────────────────────────────────────────────────
try:
    import dspy
    from signatures import ExtractStatementsSignature  # optional
except ImportError:
    class dspy:          # tiny stub so the file runs without dspy installed
        class Module:
            def __init__(self, *_, **__): pass
            def forward(self, ctx, **kw): return ctx
    ExtractStatementsSignature = None


class RequirementDeterministicExtractor(dspy.Module):
    """
    Deterministic extractor (no LLM).  Adds context["requirements"] = List[str].
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, context: Dict, **_) -> Dict:
        info: Dict = context["trial_info"]
        inc_exc: str = context.get("inc_exc", "inclusion")

        raw = get_raw_criteria(info, inc_exc)
        numbered_block = parse_criteria(raw)

        context["requirements"] = [
            ln.split(". ", 1)[1] if ". " in ln else ln
            for ln in numbered_block.split("\n") if ln.strip()
        ]
        return context

# ────────────────────────────────────────────────────────────────
# 4.  CLI smoke-test
# ────────────────────────────────────────────────────────────────
def load_example(path: str | os.PathLike) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    JSON_FILE = "long_example.json"  # override as needed  # adjust to your path
    trial_dict = load_example(JSON_FILE)["longest_inclusion"]

    extractor = RequirementDeterministicExtractor()
    result = extractor({"trial_info": trial_dict, "inc_exc": "inclusion"})

    print("Requirements extracted:", len(result["requirements"]))
    for i, req in enumerate(result["requirements"], 1):
        print(f"{i}. {req}")
