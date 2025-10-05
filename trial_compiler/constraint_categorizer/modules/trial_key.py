# modules/trial_key.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Optional

# Accept NCT######## and optional single-letter suffix a/b/c/...
SUBCOHORT_RE = re.compile(r"^(NCT\d{8})([a-z])?$", re.IGNORECASE)


@dataclass(frozen=True)
class TrialKey:
    parent_trial_id: str
    subcohort_index: Optional[str]  # 'a'/'b'/... or None
    subcohort_id: str               # parent + suffix if any


def parse_trial_key(trial_id: str) -> TrialKey:
    """
    Accepts:
      - NCT00000369
      - NCT00000369a
      - NCT00000369b
    Returns canonical parent + subcohort info (uppercased IDs, lowercase suffix).
    """
    tid = (trial_id or "").strip()
    m = SUBCOHORT_RE.match(tid)
    if not m:
        # fallback: treat whole string as parent
        up = tid.upper()
        return TrialKey(parent_trial_id=up, subcohort_index=None, subcohort_id=up)

    parent = m.group(1).upper()
    idx = m.group(2).lower() if m.group(2) else None
    sub_id = parent + idx if idx else parent
    return TrialKey(parent_trial_id=parent, subcohort_index=idx, subcohort_id=sub_id)


def subcohort_ordinal(idx: Optional[str]) -> Optional[int]:
    if idx is None:
        return None
    return ord(idx) - ord("a")
