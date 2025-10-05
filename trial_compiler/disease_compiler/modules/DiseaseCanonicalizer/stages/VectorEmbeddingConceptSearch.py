#!/usr/bin/env python3
"""
vector_embedding_concept_search.py
────────────────────────────────────────────────────────────────────────────
Vector‑similarity search that groups hits by conceptId and produces
canonical PT / FSN plus exhaustive synonym info, then prints a full roll‑up.

This version retrieves the **Preferred Term (PT)** and **Fully‑Specified
Name (FSN)** directly from a Snowstorm terminology‑server API whenever
possible, falling back to the Elasticsearch‑only heuristic if the API is
unavailable.  The rest of the behaviour is unchanged.

Each candidate row contains
    preferred_term         – SNOMED PT
    fully_specified_name   – SNOMED FSN
    best_match_term        – highest‑scoring synonym from the vector search
    match_score            – similarity score of that synonym
    matched_alternative_synonyms[]              – every vector hit ≥ score_cut
    all_terms[]            – every description the index holds for the ID
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import re, os, logging
import requests
import time  

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from .UMLSClient import UMLSClient
import dspy

import getpass
from functools import lru_cache


__all__ = ["VectorEmbeddingConceptSearch"]



# ───────────────────────── semantic‑tag helpers ──────────────────────────
_TAG_RE = re.compile(r"\(([^()]+)\)$", re.I)
_TAG2TYPE = {
    "finding": "Clinical finding",
    "disorder": "Clinical finding",
    "situation": "Situation with explicit context",
    "procedure": "Procedure",
    "event": "Event",
    "regime/therapy": "Regime / therapy",
    "observable entity": "Observable entity",
    "specimen": "Specimen",
    "body structure": "Body structure",
    "morphologic abnormality": "Morphologic abnormality",
    "pharmaceutical / biologic product": "Pharmaceutical / biologic product",
    "substance": "Substance",
    "organism": "Organism",
    "physical object": "Physical object",
    "physical force": "Physical force",
    "social context": "Social context",
    "environment": "Environment / geographical location",
    "qualifier value": "Qualifier value",
    "record artifact": "Record artifact",
    "staging and scales": "Staging and scales",
    "special concept": "Special concept",
}


def _concept_type(fsn_or_pt: str | None) -> str:
    if not fsn_or_pt:
        return ""
    m = _TAG_RE.search(fsn_or_pt.strip())
    tag = m.group(1).lower() if m else ""
    return _TAG2TYPE.get(tag, tag.title() if tag else "")


# ──────────────────────────── main module ────────────────────────────────
class VectorEmbeddingConceptSearch(dspy.Module):
    """Find SNOMED concepts via dense-vector similarity and enrich the hits.

    Added: Automatic UMLS definition lookup (see ``definition`` field)."""

    def __init__(
        self,
        *,
        es_url: str = "http://localhost:9200",
        index: str = "snomed_vectors",
        model_name: str = "cambridgeltl/sapbert-from-pubmedbert-fulltext",
        snowstorm_url: str | None = "http://localhost:8080",
        branch: str = "MAIN",
        umls_api_key: str | None = None,
        top_k: int = 5,
        score_cut: float = 0.30,
        overshoot: int = 4,
        verbose: bool = True,
        abort_on_missing_def: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

        # ── ES + embedding model ─────────────────────────────────────
        self.es = Elasticsearch(es_url, request_timeout=20)
        if not self.es.indices.exists(index=index):
            raise ValueError(f"Elasticsearch index {index!r} not found at {es_url}")
        self.index       = index
        self.top_k       = top_k
        self.cut         = score_cut
        self.overshoot   = max(1, overshoot)
        self.model       = SentenceTransformer(model_name)
        self.es_major    = int(self.es.info()["version"]["number"].split(".")[0])

        # ── Snowstorm (optional) ─────────────────────────────────────
        self.snowstorm_url = snowstorm_url.rstrip("/") if snowstorm_url else None
        self.branch        = branch
        self.http          = requests.Session()
        self.http.headers.update({"Accept": "application/json"})

        # ── UMLS client (optional) ───────────────────────────────────
        self.umls = UMLSClient(umls_api_key, verbose=self.verbose)

        # ── misc caches / config ─────────────────────────────────────
        self.allowed_root_ids = {
            "363787002",  # Observable entity
            "71388002",   # Procedure
            "404684003",  # Clinical finding
            "373873005",  # Pharmaceutical / biologic product
            "105590001",  # Substance
        }
        self._ancestor_cache: dict[str, set[str]] = {}
        self._init_umls(umls_api_key) 
        self.abort_on_missing_def = abort_on_missing_def


    # ──────────────────────────── UMLS helpers ─────────────────────────────
    def _init_umls(self, api_key: str | None) -> None:
        """
        Initialise UMLS bits: fetch an API key, create an HTTP session,
        and fetch one Ticket-Granting Ticket (TGT) that will be reused
        until it expires (~8 min TTL by NLM).
        """
        self._umls_key = api_key or os.getenv("UMLS_API_KEY")                              \
                        or getpass.getpass("UTS API key (press Enter to skip definitions): ")
        self._umls_ht  = requests.Session()
        self._umls_ht.headers.update({"Accept": "application/json"})
        self._umls_tgt, self._umls_tgt_ts, self._umls_tgt_ttl = "", 0.0, 7 * 60
        if not self._umls_key and self.verbose:
            logging.warning("[UMLS] No API key – definitions disabled")

    def _service_ticket(self) -> str:
        """Return a fresh one-shot service ticket (fetch TGT lazily)."""
        if not self._umls_key:
            raise RuntimeError("UMLS look-up requested but no API key configured")

        if not self._umls_tgt or (time.time() - self._umls_tgt_ts) > self._umls_tgt_ttl:
            r = self._umls_ht.post(
                    "https://utslogin.nlm.nih.gov/cas/v1/api-key",
                    data={"apikey": self._umls_key}, timeout=10)
            r.raise_for_status()
            self._umls_tgt, self._umls_tgt_ts = r.headers["location"], time.time()
            if self.verbose:
                logging.info("[UMLS] Fetched new TGT at %s", time.ctime(self._umls_tgt_ts))

        r = self._umls_ht.post(self._umls_tgt,
                            data={"service": "http://umlsks.nlm.nih.gov"}, timeout=10)
        r.raise_for_status()
        return r.text.strip()

    @lru_cache(maxsize=4096)
    def _snomed_to_cui(self, sctid: str, sab: str = "SNOMEDCT_US",
                    ver: str = "current") -> str | None:
        """Straight port of your working snippet, with TTL-aware tickets."""
        if not self._umls_key:
            return None

        resp = self._umls_ht.get(
            f"https://uts-ws.nlm.nih.gov/rest/search/{ver}",
            params={"string":     sctid,
                    "inputType":  "sourceUi",
                    "searchType": "exact",
                    "sabs":       sab,
                    "ticket":     self._service_ticket()},
            timeout=10)
        hits = resp.json().get("result", {}).get("results", [])
        if self.verbose:
            logging.info("[UMLS] search %s → %d hit(s)", sctid, len(hits))
        for h in hits:
            ui = h.get("ui", "")
            if ui.startswith("C"):
                return ui
        return None            # let caller decide what to do

    @lru_cache(maxsize=4096)
    def _definitions(self, cui: str, ver: str = "current") -> list[dict]:
        if not self._umls_key:
            return []
        url = f"https://uts-ws.nlm.nih.gov/rest/content/{ver}/CUI/{cui}/definitions"
        resp = self._umls_ht.get(url,
                                params={"ticket": self._service_ticket()},
                                timeout=10)
        return resp.json().get("result", [])
    # ───────────────────── ES query builders ────────────────────────────
    def _knn_query(self, vec: list[float], k: int) -> dict[str, Any]:
        return {
            "size": k,
            "query": {
                "knn": {
                    "field": "vector",
                    "query_vector": vec,
                    "k": k,
                    "num_candidates": max(1_000, k * 10),
                }
            },
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }

    def _script_query(self, vec: list[float], k: int) -> dict[str, Any]:
        return {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.qv, 'vector') + 1.0",
                        "params": {"qv": vec},
                    },
                }
            },
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }

    def _make_query(self, vec: list[float], k: int) -> dict[str, Any]:
        return self._knn_query(vec, k) if self.es_major >= 8 else self._script_query(vec, k)

    # ─────────── helpers: PT/FSN and synonym list ────────────
    def _fetch_pt_fsn(self, cid: str) -> Tuple[str, str]:
        """Return *(PT, FSN)* for *cid* (Snowstorm → ES fallback)."""
        pt: str = ""
        fsn: str = ""

        # 1) preferred: Snowstorm
        if self.snowstorm_url:
            url = f"{self.snowstorm_url}/browser/{self.branch}/concepts/{cid}?expand=fsn(),pt()"
            try:
                resp = self.http.get(url, timeout=8)
                resp.raise_for_status()
                data = resp.json()
                pt  = (data.get("pt")  or {}).get("term", "")
                fsn = (data.get("fsn") or {}).get("term", "")
                if pt and fsn:
                    return pt, fsn
            except Exception as exc:
                if self.verbose:
                    logging.warning("Snowstorm lookup failed for %s: %s", cid, exc)

        # 2) fallback: derive from ES descriptions
        term_fields = ["term", "is_pt", "is_fsn", "typeId", "description_type"]
        should = [
            {"term": {"concept_id.keyword": cid}},
            {"term": {"sctid.keyword": cid}},
            {"term": {"concept_id": cid}},
            {"term": {"sctid": cid}},
        ]
        try:
            cid_num = int(cid)
            should += [
                {"term": {"concept_id": cid_num}},
                {"term": {"sctid": cid_num}},
            ]
        except ValueError:
            pass

        q = {"size": 1000, "query": {"bool": {"should": should}}, "_source": term_fields}
        hits = self.es.search(index=self.index, body=q)["hits"]["hits"]

        tagged_fsn_row = None
        for h in hits:
            s = h["_source"]
            term = s["term"]

            is_pt  = s.get("is_pt") or s.get("description_type") == "pt" \
                     or s.get("typeId") == "900000000000013009"
            is_fsn = s.get("is_fsn") or s.get("description_type") == "fsn" \
                     or s.get("typeId") == "900000000000003001"

            if is_pt and not pt:
                pt = term
            if is_fsn and not fsn:
                fsn = term
            if not tagged_fsn_row and _TAG_RE.search(term):
                tagged_fsn_row = term
            if pt and fsn:
                break

        if not fsn and tagged_fsn_row:
            fsn = tagged_fsn_row
        if not pt and hits:
            pt = hits[0]["_source"]["term"]
        if not fsn and pt:
            fsn = pt  # worst case

        return pt, fsn

    # ─────────────────── helpers: UMLS definition (simplified) ─────────────
    _DEF_PREF_ORDER = ("MSH", "NCI", "HPO")    # keep your taste

    def _fetch_definition(self, cid: str) -> str:
        """Return a plain-text definition or raise RuntimeError if absent."""
        try:
            cui = self._snomed_to_cui(cid)          # may return None
            if not cui:
                raise RuntimeError(f"no CUI found for SNOMED {cid}")

            defs = self._definitions(cui)           # may be []
            if not defs:
                raise RuntimeError(f"CUI {cui} has no definitions")

            for pref in self._DEF_PREF_ORDER:
                hit = next((d for d in defs if d["rootSource"] == pref), None)
                if hit:
                    return re.sub(r"\s+", " ", hit["value"].strip())

            # fall back to first definition
            return re.sub(r"\s+", " ", defs[0]["value"].strip())

        except Exception as exc:
            # log once, then abort
            if self.verbose:
                logging.error("[UMLS] definition lookup failed for %s: %s", cid, exc)
            if self.abort_on_missing_def:
                raise

    # ─────────────────── helpers: IS-A category test ─────────────────────
    def _fetch_ancestors(self, cid: str) -> set[str]:
        if cid in self._ancestor_cache:
            return self._ancestor_cache[cid]

        if not self.snowstorm_url:
            return set()

        url = f"{self.snowstorm_url}/browser/{self.branch}/concepts/{cid}/ancestors?form=inferred"
        try:
            r = self.http.get(url, timeout=6)
            r.raise_for_status()
            anc = {a["conceptId"] for a in r.json()} | {cid}
        except Exception as exc:
            if self.verbose:
                logging.warning("Ancestor fetch failed for %s: %s", cid, exc)
            anc = set()

        self._ancestor_cache[cid] = anc
        return anc

    def _passes_category_filter(self, cid: str) -> bool:
        if not self.allowed_root_ids:
            return True
        anc = self._fetch_ancestors(cid)
        return bool(anc & self.allowed_root_ids)

    def _fetch_all_terms(self, cid: str) -> list[str]:
        should: List[Dict[str, Any]] = [
            {"term": {"concept_id.keyword": cid}},
            {"term": {"sctid.keyword": cid}},
            {"term": {"concept_id": cid}},
            {"term": {"sctid": cid}},
        ]
        try:
            cid_num = int(cid)
            should.extend([
                {"term": {"concept_id": cid_num}},
                {"term": {"sctid": cid_num}},
            ])
        except ValueError:
            pass

        q = {
            "size": 10_000,
            "query": {"bool": {"should": should}},
            "_source": ["term"],
        }
        hits = self.es.search(index=self.index, body=q)["hits"]["hits"]
        return sorted({h["_source"]["term"] for h in hits})

    # ─────────────────────────── forward() ──────────────────────────────
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        ents = ctx.get("target_disease", [])

        if not ents:
            return ctx

        vecs  = self.model.encode([e["disease"] for e in ents], normalize_embeddings=True)
        raw_k = self.top_k * self.overshoot

        collected: Dict[str, Dict[str, Any]] = {}

        for ent, vec in zip(ents, vecs):
            hits = self.es.search(index=self.index,
                                  body=self._make_query(vec.tolist(), raw_k))["hits"]["hits"]

            by_cid: Dict[str, Dict[str, Any]] = {}
            for h in hits:
                if h["_score"] < self.cut:
                    continue
                src = h["_source"]
                cid = src.get("concept_id") or src.get("sctid")
                if not cid:
                    continue

                score = h["_score"]
                term  = src.get("term", ent["disease"])
                rec   = by_cid.setdefault(
                    cid,
                    dict(
                        conceptId      = cid,
                        preferred_term = "",
                        fully_specified_name = "",
                        type          = "",
                        definition    = "",
                        best_match_term   = term,
                        match_score       = score,
                        matched_alternative_synonyms = [],
                    ),
                )
                rec["matched_alternative_synonyms"].append(
                    {"term": term, "score": round(score, 3)}
                )
                if score > rec["match_score"]:
                    rec["match_score"]     = score
                    rec["best_match_term"] = term

            by_cid = {c: r for c, r in by_cid.items()
                      if self._passes_category_filter(c)}
            if not by_cid:
                continue

            for rec in by_cid.values():
                pt, fsn = self._fetch_pt_fsn(rec["conceptId"])
                rec["preferred_term"]       = pt
                rec["fully_specified_name"] = fsn or rec["best_match_term"]
                rec["type"]                 = _concept_type(rec["fully_specified_name"] or pt)
                rec["all_terms"]            = self._fetch_all_terms(rec["conceptId"])
                # NEW: definition
                if not rec["definition"]:
                    rec["definition"] = self._fetch_definition(rec["conceptId"])

                col = collected.setdefault(rec["conceptId"], dict(pt=pt, all_terms=set()))
                col["all_terms"].update(rec["all_terms"])

            ent["candidates"] = sorted(
                by_cid.values(),
                key=lambda r: r["match_score"],
                reverse=True
            )[: self.top_k]
            ent["provenance"] = ent.get("provenance", "llm_vec") + "|vec"

        # ——————— VERBOSE LOG —————————————————————————————
        if self.verbose:
            for ent in ents:
                cands = ent.get("candidates", [])
                if not cands:
                    logging.info('  • "%s" → 0 hits ≥ %.2f', ent["disease"], self.cut)
                    continue

                logging.info('  • "%s" → %d concept(s):', ent["disease"], len(cands))
                for rec in cands:
                    logging.info("      ↳ %s (%s)  score=%.3f",
                                 rec["preferred_term"], rec["conceptId"], rec["match_score"])
                    logging.info("        PT      : %s", rec["preferred_term"])
                    logging.info("        FSN     : %s", rec["fully_specified_name"])
                    logging.info("        Type    : %s", rec["type"])
                    if rec["definition"]:
                        logging.info("        Defn    : %s", rec["definition"])
                    logging.info("        Best hit: \"%s\"", rec["best_match_term"])

                    if rec["matched_alternative_synonyms"]:
                        m = ", ".join(f"{m['term']} ({m['score']:.3f})"
                                      for m in rec["matched_alternative_synonyms"])
                        logging.info("        Vector hits [%d]: %s",
                                     len(rec["matched_alternative_synonyms"]), m)
                    if rec["all_terms"]:
                        logging.info("        All synonyms [%d]:", len(rec["all_terms"]))
                        for t in rec["all_terms"]:
                            logging.info("            • %s", t)
                logging.info("")

            if collected:
                logging.info("  ——— Synonym summary across all concepts ———")
                for cid, data in collected.items():
                    terms = sorted(data["all_terms"])
                    pt    = data["pt"] or "(no PT)"
                    logging.info("    %s (%s)  [%d]:", pt, cid, len(terms))
                    for t in terms:
                        logging.info("        • %s", t)
                logging.info("")
        # ————————————————————————————————————————————————
        return ctx