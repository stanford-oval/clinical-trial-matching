"""
snomed_utils.py  ·  Lightweight helper around the public Snowstorm API

✔  search(term[, limit])      → list[{id, fsn, pt}]
✔  ancestors(concept_id)      → list[{id, fsn, pt}]
✔  canonical(concept_id)      → dict with term, defStatus, ancestors

Designed for quick prototyping; heavy/batch use still needs your own
Snowstorm instance because the public server enforces rate limits.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any, List, Dict

import requests
from requests.adapters import HTTPAdapter, Retry

# ---------------------------------------------------------------------
BASE    = "https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct"
BRANCH  = "MAIN"                # change to e.g. "MAIN/SNOMEDCT-US" if needed
HEADERS = {"Accept": "application/json"}

# ---------------------------------------------------------------------
# 1) persistent HTTP session with automatic retry / back-off
# ---------------------------------------------------------------------
session       = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,                  # <= 5 overall attempts
            backoff_factor=1,         # 1 s → 2 s → 4 s …
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
    ),
)


def _url(*parts: str) -> str:
    """Join path segments onto the API base."""
    return "/".join([BASE] + list(parts))


@lru_cache(maxsize=2048)
def _get(url: str, **params) -> dict[str, Any]:
    """GET with retry/back-off + simple LRU cache."""
    attempt = 0
    while True:
        r = session.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()

        # 429 Too Many Requests → honour server header or back-off
        retry_after = int(r.headers.get("Retry-After", 0))
        attempt += 1
        if attempt > 5:          # gave it our best shot
            r.raise_for_status()
        wait = max(retry_after, math.pow(2, attempt))     # seconds
        time.sleep(wait)


# ---------------------------------------------------------------------
# 2) PUBLIC HELPERS
# ---------------------------------------------------------------------
def search(term: str, limit: int = 20) -> List[Dict[str, str]]:
    """
    Free-text search → minimal concept info list.
    """
    data = _get(
        _url("browser", BRANCH, "concepts"),
        term=term,
        activeFilter="true",
        limit=str(limit),
    )
    return [
        {"id": c["id"], "fsn": c["fsn"]["term"], "pt": c["pt"]["term"]}
        for c in data["items"]
    ]


def ancestors(concept_id: str) -> List[Dict[str, str]]:
    """
    All (recursive) ISA ancestors of a concept (no duplicates).
    """
    data = _get(
        _url("browser", BRANCH, "concepts", concept_id, "ancestors")
    )
    return [
        {"id": c["id"], "fsn": c["fsn"]["term"], "pt": c["pt"]["term"]}
        for c in data
    ]


def canonical(concept_id: str) -> Dict[str, Any]:
    """
    Canonical record with preferred term, definition status,
    effective time and full ancestor list.
    """
    concept = _get(
        _url("browser", BRANCH, "concepts", concept_id)
    )
    return {
        "id": concept["id"],
        "term": concept["pt"]["term"],
        "definitionStatus": concept["definitionStatus"],
        "effectiveTime": concept["effectiveTime"],
        "ancestors": ancestors(concept_id),
    }


# ---------------------------------------------------------------------
# 3) quick demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint

    pprint(search("pneumonia", 5))
    pprint(canonical("233604007"))   # Pneumonia
