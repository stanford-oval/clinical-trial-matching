# tagger.py
"""
Post‑process an <smtfragment> so every top‑level (assert …) becomes

  (! <inner‑assertion> :named R{req}_{idx}_{constraint})

•   All three items (req‑index, assert‑index‑within‑req, label) are
    concatenated with underscores and stored ONLY inside :named.
•   A global registry guarantees uniqueness; duplicates raise TaggingError.
"""

from __future__ import annotations
import re


_ASSERT_RE = re.compile(r"^\s*\(assert\b", re.IGNORECASE)
_WRAP_PREFIX, _WRAP_SUFFIX = "(! ", ")"

class TaggingError(RuntimeError):
    """Raised when the generated :named tag is not globally unique."""

# tagger.py


def add_tags(smt_fragment: str, *, req_idx: int,
             constraint: str, tag_registry: set[str]) -> str:
    out, local_idx = [], 1
    for line in smt_fragment.splitlines():
        if _ASSERT_RE.match(line):
            core, *comment = line.split(";")                 # keep trailing comment
            term = core.strip()[len("(assert"):].rstrip(") ").strip()
            # skip if already wrapped
            if term.startswith("(!") and ":named" in term:
                out.append(line)            # leave as‑is
                continue

            tag = f"R{req_idx}_{local_idx}_{constraint}"
            if tag in tag_registry:
                raise TaggingError(f"duplicate :named {tag}")
            tag_registry.add(tag)

            wrapped = (f"(assert (! {term}"
                       f" :named {tag}"
                       f" :req_idx {req_idx}"
                       f" :assert_idx {local_idx}"
                       f" :constraint {constraint}))")
            if comment:
                wrapped += " ;" + ";".join(comment)
            out.append(wrapped)
            local_idx += 1
        else:
            out.append(line)
    return "\n".join(out)