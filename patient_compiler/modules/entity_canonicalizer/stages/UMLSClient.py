from __future__ import annotations

from typing import Dict, List
import os, logging, time                # NEW: os
import requests                                   # NEW: used twice


# ──────────────────────────── UMLS helpers ───────────────────────────────
_SERVICE_URL = "http://umlsks.nlm.nih.gov"

class UMLSClient:
    """Lightweight ticket-based UMLS REST helper."""

    def __init__(self, api_key: str | None, verbose: bool = False):
        self.api_key  = api_key or os.getenv("UMLS_API_KEY", "")
        self.verbose  = verbose
        self._tgt     = ""
        self._tgt_ts  = 0.0     # when we fetched the TGT
        self.http     = requests.Session()
        self.http.headers.update({"Accept": "application/json"})
        if not self.api_key and self.verbose:
            logging.warning("No UMLS API key provided – definition lookup disabled")

        # tiny caches
        self._cui_cache: dict[str, str]                       = {}  # sctid → CUI
        self._def_cache: dict[str, List[Dict[str, str]]]      = {}  # cui  → defs

    # ── ticket handling ──────────────────────────────────────────────
    def _ensure_tgt(self) -> None:
        """(Re-)fetch TGT if we have none or it’s > 7 min old."""
        if not self.api_key:
            raise RuntimeError("UMLS definitions requested but no API key configured")
        if self._tgt and (time.time() - self._tgt_ts) < 7 * 60:
            return
        r = self.http.post(
            "https://utslogin.nlm.nih.gov/cas/v1/api-key",
            data={"apikey": self.api_key},
            timeout=10,
        )
        r.raise_for_status()
        self._tgt    = r.headers["location"]
        self._tgt_ts = time.time()

    def _service_ticket(self) -> str:
        self._ensure_tgt()
        r = self.http.post(self._tgt, data={"service": _SERVICE_URL}, timeout=10)
        r.raise_for_status()
        return r.text

    # ── public helpers ───────────────────────────────────────────────
    def snomed_to_cui(self, sctid: str, sab: str = "SNOMEDCT_US",
                      ver: str = "current") -> str | None:
        if sctid in self._cui_cache:
            return self._cui_cache[sctid]
        if not self.api_key:
            return None
        r = self.http.get(
            f"https://uts-ws.nlm.nih.gov/rest/search/{ver}",
            params={
                "string":     sctid,
                "inputType":  "sourceUi",
                "searchType": "exact",
                "sabs":       sab,
                "ticket":     self._service_ticket(),
            },
            timeout=10,
        )
        r.raise_for_status()
        hits = r.json().get("result", {}).get("results", [])
        for hit in hits:
            ui = hit.get("ui", "")
            if ui.startswith("C"):
                self._cui_cache[sctid] = ui
                return ui
        return None

    def definitions(self, cui: str, ver: str = "current") -> List[Dict[str, str]]:
        if cui in self._def_cache:
            return self._def_cache[cui]
        if not self.api_key:
            return []
        r = self.http.get(
            f"https://uts-ws.nlm.nih.gov/rest/content/{ver}/CUI/{cui}/definitions",
            params={"ticket": self._service_ticket()},
            timeout=10,
        )
        r.raise_for_status()
        defs = r.json().get("result", [])
        self._def_cache[cui] = defs
        return defs