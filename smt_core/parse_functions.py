from typing import List, Union, Optional, Dict
import json
import re


EXPECTED_KEYS = {"requirement", "text_span"}

def _is_valid_entry(entry: dict) -> bool:
    """Tiny schema check — add/relax fields as needed."""
    return isinstance(entry, dict) and EXPECTED_KEYS.issubset(entry)

def _safe_json_loads(s: str):
    """Return parsed JSON or None (never raises)."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def parse_extract_requirements_output(llm_output: str) -> Union[List[dict], bool]:
    """
    Grab everything between <requirements> … </requirements> and return
    a *list of dictionaries* [{requirement: …, text_span: …}, …].

    Returns False only when the tags are missing or nothing valid is found.
    """
    m = re.search(r"<requirements>(.*?)</requirements>", llm_output, re.S)
    if not m:
        return False

    body = m.group(1).strip()
    entries: List[dict] = []

    # ── 1) most‑likely: one big JSON array ───────────────────────────
    parsed = _safe_json_loads(body)
    if isinstance(parsed, list) and all(_is_valid_entry(e) for e in parsed):
        return parsed

    # ── 2) fallback: many one‑line JSON arrays ───────────────────────
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _safe_json_loads(line)
        if (
            isinstance(parsed, list)
            and len(parsed) == 1
            and _is_valid_entry(parsed[0])
        ):
            entries.append(parsed[0])
            continue
        # could also be a bare dict
        if _is_valid_entry(parsed):
            entries.append(parsed)
            continue

    # ── 3) last‑ditch: treat each non‑empty line as plain text ───────
    if not entries:
        entries = [{"requirement": ln, "text_span": ""}
                   for ln in body.splitlines() if ln.strip()]

    return entries if entries else False




def parse_improve_atomicity_output(llm_output: str) -> Union[List[str], bool]:
    """
    Extract everything between <atomic_requirements> and </atomic_requirements>.
    Return a list of the extracted lines, or False if the tags are not both found.
    """
    start_tag = "<atomic_requirements>"
    end_tag = "</atomic_requirements>"
    if start_tag not in llm_output or end_tag not in llm_output:
        return False

    content = llm_output.split(start_tag)[-1].split(end_tag)[0]
    requirements = [s.strip() for s in content.splitlines() if s.strip()]
    return requirements


def parse_improve_self_containedness_output(llm_output: str) -> Union[str, bool]:
    """
    Extract everything between <rewritten_requirement> and </rewritten_requirement>.
    Return a single string with the extracted content, or False if the tags are not both found.
    """
    start_tag = "<trimmed_requirement>"
    end_tag = "</trimmed_requirement>"
    if start_tag not in llm_output or end_tag not in llm_output:
        return False

    requirement = llm_output.split(start_tag)[-1].split(end_tag)[0].strip()
    return requirement


def parse_remove_duplicates_output(llm_output: str) -> Union[List[int], bool]:
    """
    Extract everything between <duplicate_indices> and </duplicate_indices>.
    Return a list of zero-based duplicate indices, or False if the tags are not both found.
    """
    start_tag = "<duplicate_indices>"
    end_tag = "</duplicate_indices>"
    if start_tag not in llm_output or end_tag not in llm_output:
        return False

    content = llm_output.split(start_tag)[-1].split(end_tag)[0].strip()
    if not content:
        return []

    str_indices = [x.strip() for x in content.split(",")]
    return [int(idx) - 1 for idx in str_indices if idx.isdigit()]


import re
from typing import Union

_TAG_RE = re.compile(r"<smtfragment>\s*(.*?)\s*</smtfragment>",
                     re.DOTALL | re.IGNORECASE)

# NEW – recognise ``` … ```  or  ```smt … ``` fences
_FENCE_RE = re.compile(r"```(?:smt)?\s*(.*?)```",
                       re.DOTALL | re.IGNORECASE)


def parse_smt_output(text: str) -> Union[str, bool]:
    """Return SMT‑LIB inside <smtfragment>, code‑fence, or bare snippet."""
    # 1) preferred wrapper
    m = _TAG_RE.search(text)
    if m:
        body = m.group(1).strip()
        return body if body else False

    # 2) fenced block
    m = _FENCE_RE.search(text)
    if m:
        body = m.group(1).strip()
        return body if body else False

    # 3) already‐bare SMT (starts with ‘(’)
    stripped = text.strip()
    if stripped.startswith("("):
        return stripped

    return False



def extract_tag_block(text: str, tag: str) -> str:
    """
    Helper that returns the text between <tag> and </tag>.
    If either tag is missing, returns an empty string.
    """
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        return ""

    return text[start_idx + len(start_tag) : end_idx]





# ------------------------- helpers -----------------------------------
_CORR_RE = re.compile(
    r"<corrected_program>(.*?)</corrected_program>",
    re.DOTALL | re.IGNORECASE,
)

_VARS_RE  = re.compile(
    r"<variableaddition>(.*?)</variableaddition>",
    re.DOTALL | re.IGNORECASE,
)

_ASSERT_RE = re.compile(
    r"<outsmtexpression>(.*?)</outsmtexpression>",
    re.DOTALL | re.IGNORECASE,
)


def parse_corrected_program(llm_out: str) -> List[str] | None:
    """
    Return the corrected slice as list[str] **including** the
    <variableaddition> and <outsmtexpression> tags (so the caller can
    splice it back verbatim), or None if tags missing.
    """
    block = _CORR_RE.search(llm_out)
    if not block:
        return None

    var_block = _VARS_RE.search(block.group(1))
    out_block = _ASSERT_RE.search(block.group(1))

    if not (var_block and out_block):
        return None   # malformed output

    # Normalise whitespace inside each block
    var_lines = [ln.rstrip()
                 for ln in var_block.group(0).strip().splitlines()]
    out_lines = [ln.rstrip()
                 for ln in out_block.group(0).strip().splitlines()]

    # Keep the same order (vars first, asserts second)
    return var_lines + [""] + out_lines


# def parse_corrected_program(llm_output: str) -> List[str]:
#     """
#     Extract everything between <atomic_requirements> and </atomic_requirements>.
#     Return a list of the extracted lines, or False if the tags are not both found.
#     """
#     start_tag = "<atomic_requirements>"
#     end_tag = "</atomic_requirements>"
#     if start_tag not in llm_output or end_tag not in llm_output:
#         return False

#     content = llm_output.split(start_tag)[-1].split(end_tag)[0]
#     requirements = [s.strip() for s in content.splitlines() if s.strip()]
#     return requirements


def parse_corrected_program(output: str):
    """
    Extracts the contents of the <corrected_program_slice> tag from the LLM output.
    Returns a list of non-empty lines representing the corrected SMT slice.
    """
    match = re.search(r"<corrected_program_slice>\s*(.*?)\s*</corrected_program_slice>", output, re.DOTALL)
    if not match:
        return None  # parsing failed

    slice_text = match.group(1).strip()

    # Split into lines and remove empty/comment-only lines if needed
    return [line for line in slice_text.splitlines() if line.strip()]


_STRAT_RE = re.compile(
    r"<strategy_description>(.*?)</strategy_description>",
    re.S | re.I,
)
_CORR_RE = re.compile(
    r"<corrected_whole_program>(.*?)</corrected_whole_program>",
    re.S | re.I,
)


def _strip_block(match: Optional[re.Match]) -> str:
    """Return inner text with trailing whitespace trimmed line-by-line."""
    if not match:
        return ""
    return "\n".join(ln.rstrip() for ln in match.group(1).strip().splitlines())

def parse_naive_refiner_output(raw: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse the SMT-debugger LLM reply.

    Returns
    -------
    {
        "strategy_description": str,      # may be empty
        "corrected_whole_program": List[str]  # SMT-LIB lines; empty ⇒ bad output
    }
    """
    strategy = _strip_block(_STRAT_RE.search(raw))
    corr_txt = _strip_block(_CORR_RE.search(raw))

    return {
        "strategy_description": strategy,
        "corrected_whole_program": corr_txt.splitlines() if corr_txt else [],
    }


def parse_rewrite_output(text: str, expect_n: int):
    """
    Very lenient parser: finds lines like “[02] …” or “[02] …”
    Returns list[str] or False if count mismatches.
    """
    import re
    lines = re.findall(r"\[\d{2,}\]\s*(.+)", text)
    return lines if len(lines) == expect_n else False
