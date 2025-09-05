#!/usr/bin/env python3
"""
vector_embedding_concept_search.py
────────────────────────────────────────────────────────────────────────────
Fine-grained profiling spans:
  • CANON/VEC/encode
  • CANON/VEC/search
  • CANON/VEC/enrich

Safe speedups:
  • Batched kNN via _msearch (falls back if unsupported)
  • Adaptive beam (widen once if unique conceptIds < top_k)
  • NEW: Persistent SQLite cache for query→hits across runs

Existing improvements kept: GPU batching, in-run memo, lru_cache metadata,
optional lazy definitions.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Any, List, Tuple, Optional
import re, logging, time, json, os, sqlite3
import requests
from functools import lru_cache

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import dspy

from .UMLSClient import UMLSClient

__all__ = ["VectorEmbeddingConceptSearch"]

# ───────────────────────── semantic-tag helpers ──────────────────────────
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


class VectorEmbeddingConceptSearch(dspy.Module):
    """Dense vector search + concept enrichment with profiling, msearch, and SQLite cache."""

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
        # perf knobs
        encode_batch: int = 32,
        query_cache_size: int = 4096,   # in-run memo
        es_workers: int = 8,
        fetch_definitions: bool = False,
        defs_top_n: int = 1,
        # profiler wiring
        profiler: Optional[Any] = None,
        profiler_run_id: str = "run",
        # safe speed toggles
        use_msearch: bool = True,
        adaptive_widen: bool = True,
        widen_factor: int = 2,          # single widen pass ×2
        widen_floor: int = 800,         # min num_candidates on widen
        # NEW: persistent cache
        persistent_cache: bool = True,
        cache_path: str = "run_logs/query_cache.sqlite",
        cache_ttl_days: int = 60,
        cache_require_k: bool = True,   # only cache when uniq ≥ top_k
        cache_version: int = 1,         # bump to invalidate
    ):
        super().__init__()
        self.verbose = bool(verbose)
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

        self.profiler = profiler
        self.run_id   = profiler_run_id

        # ES + model
        self.es = Elasticsearch(es_url, request_timeout=40)
        if not self.es.indices.exists(index=index):
            raise ValueError(f"Elasticsearch index {index!r} not found at {es_url}")
        self.index       = index
        self.top_k       = int(top_k)
        self.cut         = float(score_cut)
        self.overshoot   = max(1, int(overshoot))
        self.model       = SentenceTransformer(model_name)
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass
        self.es_major    = int(self.es.info()["version"]["number"].split(".")[0])

        # Snowstorm (optional) — keep-alive pool a bit larger
        from requests.adapters import HTTPAdapter
        self.snowstorm_url = snowstorm_url.rstrip("/") if snowstorm_url else None
        self.branch        = branch
        self.http          = requests.Session()
        self.http.headers.update({"Accept": "application/json"})
        try:
            self.http.mount("http://", HTTPAdapter(pool_maxsize=32))
            self.http.mount("https://", HTTPAdapter(pool_maxsize=32))
        except Exception:
            pass

        # UMLS (optional)
        self.umls = UMLSClient(umls_api_key, verbose=self.verbose)

        # caches / config
        self.allowed_root_ids = {"363787002","71388002","404684003","373873005","105590001"}
        self._ancestor_cache: dict[str, set[str]] = {}

        self.encode_batch = int(encode_batch)
        self._query_cache_size = int(query_cache_size)   # in-run memo size
        self.es_workers   = max(1, int(es_workers))
        self.fetch_definitions = bool(fetch_definitions)
        self.defs_top_n   = int(defs_top_n)

        # in-run memo (python dict)
        self._query_hits_cache: dict[str, list[dict]] = {}
        self._query_hits_order: list[str] = []

        # speed toggles
        self.use_msearch     = bool(use_msearch)
        self.adaptive_widen  = bool(adaptive_widen)
        self.widen_factor    = max(2, int(widen_factor))
        self.widen_floor     = max(400, int(widen_floor))

        # persistent SQLite cache
        self.persistent_cache = bool(persistent_cache)
        self.cache_path = cache_path
        self.cache_ttl_days = int(cache_ttl_days)
        self.cache_require_k = bool(cache_require_k)
        self.cache_version = int(cache_version)
        self._pc = None  # type: Optional[sqlite3.Connection]
        if self.persistent_cache:
            self._pc_init()

    # ---------- profiler helper ----------
    def _span(self, stage: str, ctx: Dict[str, Any], **extra):
        if not self.profiler:
            return contextlib.nullcontext()
        return self.profiler.span(
            run_id=self.run_id,
            trial_id=str(ctx.get("trial_id", "?")),
            side=str(ctx.get("inc_exc", "?")),
            stage=stage,
            cohort_id=str(ctx.get("trial_id", "?")),
            **extra
        )

    # ---------- in-run memo ----------
    def _memo_hits(self, q: str, hits: list[dict]):
        self._query_hits_cache[q] = hits
        self._query_hits_order.append(q)
        if len(self._query_hits_order) > self._query_cache_size:
            old = self._query_hits_order.pop(0)
            self._query_hits_cache.pop(old, None)

    # ---------- persistent cache (SQLite) ----------
    def _pc_init(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        self._pc = sqlite3.connect(self.cache_path, isolation_level=None, check_same_thread=False)
        cur = self._pc.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS qcache (
                q TEXT NOT NULL,
                k INTEGER NOT NULL,
                ver INTEGER NOT NULL,
                uniq_count INTEGER NOT NULL,
                ts REAL NOT NULL,
                hits TEXT NOT NULL,
                PRIMARY KEY (q, k, ver)
            )
        """)
        # Cheap pragmas for speed
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    def _pc_get(self, q: str, k: int) -> Optional[list[dict]]:
        if not self._pc:
            return None
        try:
            cur = self._pc.cursor()
            row = cur.execute(
                "SELECT uniq_count, ts, hits FROM qcache WHERE q=? AND k=? AND ver=?",
                (q, k, self.cache_version)
            ).fetchone()
            cur.close()
            if not row:
                return None
            uniq_count, ts, hits_json = row
            # TTL check
            if self.cache_ttl_days > 0 and (time.time() - ts) > (self.cache_ttl_days * 86400):
                return None
            # If we require at least k unique concepts for reuse, enforce it
            if self.cache_require_k and uniq_count < k:
                return None
            return json.loads(hits_json)
        except Exception:
            return None

    def _pc_put(self, q: str, k: int, hits: list[dict]):
        if not self._pc:
            return
        try:
            uniq = self._dedup_by_concept(hits)
            if self.cache_require_k and len(uniq) < k:
                return  # don't cache starved results
            cur = self._pc.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO qcache (q,k,ver,uniq_count,ts,hits) VALUES (?,?,?,?,?,?)",
                (q, k, self.cache_version, len(uniq), time.time(), json.dumps(hits))
            )
            cur.close()
        except Exception:
            pass  # cache should be best-effort only

    # ---------- UMLS helpers (cached) ----------
    @lru_cache(maxsize=4096)
    def _snomed_to_cui(self, sctid: str, sab: str = "SNOMEDCT_US",
                       ver: str = "current") -> str | None:
        if not self.umls.api_key:
            return None
        return self.umls.snomed_to_cui(sctid, sab=sab, ver=ver)

    @lru_cache(maxsize=4096)
    def _definitions(self, cui: str, ver: str = "current") -> list[dict]:
        if not self.umls.api_key:
            return []
        return self.umls.definitions(cui, ver=ver)

    # ---------- ES query builders ----------
    def _knn_query(self, vec: List[float], k: int, num_candidates: int | None = None) -> dict[str, Any]:
        nc = num_candidates if num_candidates is not None else max(k * 40, 400)
        return {
            "size": k,
            "query": {
                "knn": {
                    "field": "vector",
                    "query_vector": vec,
                    "k": k,
                    "num_candidates": nc,
                }
            },
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }

    def _script_query(self, vec: List[float], k: int, num_candidates: int | None = None) -> dict[str, Any]:
        # Fallback path for ES<8
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

    def _make_query(self, vec: List[float], k: int, num_candidates: int | None = None) -> dict[str, Any]:
        if self.es_major >= 8:
            return self._knn_query(vec, k, num_candidates)
        return self._script_query(vec, k, num_candidates)

    # ---------- batched msearch ----------
    def _msearch_hits(self, vecs: List[List[float]], k: int, num_candidates: int) -> List[List[dict]]:
        body: List[dict] = []
        for v in vecs:
            body.append({"index": self.index})
            body.append(self._make_query(v, k, num_candidates))
        try:
            resp = self.es.msearch(body=body)
            return [r.get("hits", {}).get("hits", []) for r in resp.get("responses", [])]
        except Exception as e:
            if self.verbose:
                logging.warning("[VecEmbed] msearch fallback due to: %s", e)
            out: List[List[dict]] = []
            for v in vecs:
                out.append(self.es.search(index=self.index, body=self._make_query(v, k, num_candidates))["hits"]["hits"])
            return out

    # ---------- dedup & adaptive widen ----------
    @staticmethod
    def _dedup_by_concept(hits: List[dict]) -> List[dict]:
        seen: set[str] = set()
        uniq: List[dict] = []
        for h in hits:
            src = h.get("_source") or {}
            cid = src.get("concept_id") or src.get("sctid")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            uniq.append(h)
        return uniq

    def _search_one_adaptive(self, vec: List[float], raw_k: int) -> List[dict]:
        hits = self.es.search(index=self.index, body=self._make_query(vec, raw_k))["hits"]["hits"]
        uniq = self._dedup_by_concept(hits)
        if (not self.adaptive_widen) or (len(uniq) >= self.top_k):
            return hits
        widened_nc = max(raw_k * self.widen_factor, self.widen_floor)
        hits2 = self.es.search(index=self.index, body=self._make_query(vec, widened_nc))["hits"]["hits"]
        return hits + hits2

    # ---------- concept metadata (cached) ----------
    @lru_cache(maxsize=8192)
    def _fetch_pt_fsn(self, cid: str) -> Tuple[str, str]:
        pt: str = ""
        fsn: str = ""

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

        term_fields = ["term", "is_pt", "is_fsn", "typeId", "description_type"]
        should = [
            {"term": {"concept_id.keyword": cid}},
            {"term": {"sctid.keyword": cid}},
            {"term": {"concept_id": cid}},
            {"term": {"sctid": cid}},
        ]
        try:
            cid_num = int(cid)
            should += [{"term": {"concept_id": cid_num}}, {"term": {"sctid": cid_num}}]
        except ValueError:
            pass

        q = {"size": 1000, "query": {"bool": {"should": should}}, "_source": term_fields}
        hits = self.es.search(index=self.index, body=q)["hits"]["hits"]

        tagged_fsn_row = None
        for h in hits:
            s = h["_source"]
            term = s["term"]
            is_pt  = s.get("is_pt") or s.get("description_type") == "pt" or s.get("typeId") == "900000000000013009"
            is_fsn = s.get("is_fsn") or s.get("description_type") == "fsn" or s.get("typeId") == "900000000000003001"
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
            fsn = pt
        return pt, fsn

    @lru_cache(maxsize=8192)
    def _fetch_all_terms(self, cid: str) -> List[str]:
        should: List[Dict[str, Any]] = [
            {"term": {"concept_id.keyword": cid}},
            {"term": {"sctid.keyword": cid}},
            {"term": {"concept_id": cid}},
            {"term": {"sctid": cid}},
        ]
        try:
            cid_num = int(cid)
            should.extend([{"term": {"concept_id": cid_num}}, {"term": {"sctid": cid_num}}])
        except ValueError:
            pass

        q = {"size": 10_000, "query": {"bool": {"should": should}}, "_source": ["term"]}
        hits = self.es.search(index=self.index, body=q)["hits"]["hits"]
        return sorted({h["_source"]["term"] for h in hits})

    _DEF_PREF_ORDER = ("MSH", "NCI", "HPO")

    def _fetch_definition(self, cid: str) -> str:
        try:
            cui = self._snomed_to_cui(cid)
            if not cui:
                return ""
            defs = self._definitions(cui)
            if not defs:
                return ""
            for pref in self._DEF_PREF_ORDER:
                hit = next((d for d in defs if d["rootSource"] == pref), None)
                if hit:
                    return re.sub(r"\s+", " ", hit["value"].strip())
            return re.sub(r"\s+", " ", defs[0]["value"].strip())
        except Exception as exc:
            if self.verbose:
                logging.error("[UMLS] definition lookup failed for %s: %s", cid, exc)
            return ""

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

    # ─────────────────────────── forward() ──────────────────────────────
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        idx  = ctx["current_requirement_index"]
        ents = ctx.get("llm_surface_entities_by_req", {}).get(idx, [])
        if not ents:
            return ctx

        # queries from query_term > entity_name > text
        queries: List[str] = []
        ent_positions: List[int] = []
        for i, ent in enumerate(ents):
            q = (ent.get("query_term") or ent.get("entity_name") or ent.get("text") or "").strip()
            if not q:
                continue
            ent["query_term"] = q
            queries.append(q)
            ent_positions.append(i)
        if not queries:
            return ctx

        # dedupe (per requirement)
        uniq: List[str] = []
        back: Dict[str, int] = {}
        for q in queries:
            if q not in back:
                back[q] = len(uniq)
                uniq.append(q)

        # ENCODE
        with self._span("CANON/VEC/encode", ctx, requirement_index=idx, uniq_queries=len(uniq)):
            vecs = self.model.encode(
                uniq,
                batch_size=self.encode_batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device if hasattr(self.model, "device") else None,
                show_progress_bar=False,
            )
        vec_for_ent = [vecs[back[q]] for q in queries]

        raw_k = self.top_k * self.overshoot

        # SEARCH (aggregate span)
        with self._span("CANON/VEC/search", ctx, requirement_index=idx, num_entities=len(vec_for_ent)):
            # Cache-aware lookup
            def _lookup(q_vec, q_str) -> List[dict]:
                # 1) in-run memo
                cached = self._query_hits_cache.get(q_str)
                if cached is not None:
                    return cached
                # 2) persistent cache
                pc_hits = self._pc_get(q_str, self.top_k) if self.persistent_cache else None
                if pc_hits is not None:
                    self._memo_hits(q_str, pc_hits)
                    return pc_hits
                # 3) fresh search (adaptive)
                hits = self._search_one_adaptive(q_vec.tolist(), raw_k)
                self._memo_hits(q_str, hits)
                # write-through to persistent cache (best-effort)
                if self.persistent_cache:
                    self._pc_put(q_str, self.top_k, hits)
                return hits

            # Prefer msearch for batch, then optional widen per-entity
            if self.use_msearch and len(vec_for_ent) > 1 and self.es_major >= 8:
                try:
                    all_hits = self._msearch_hits([v.tolist() for v in vec_for_ent], k=raw_k, num_candidates=raw_k)
                    if self.adaptive_widen:
                        for i, (hits, q) in enumerate(zip(all_hits, queries)):
                            uniq_hits = self._dedup_by_concept(hits)
                            if len(uniq_hits) < self.top_k:
                                widened_nc = max(raw_k * self.widen_factor, self.widen_floor)
                                extra = self.es.search(index=self.index, body=self._make_query(vec_for_ent[i].tolist(), widened_nc))["hits"]["hits"]
                                hits = hits + extra
                                all_hits[i] = hits
                            # memo + persist
                            self._memo_hits(q, hits)
                            if self.persistent_cache:
                                self._pc_put(q, self.top_k, hits)
                except Exception as e:
                    if self.verbose:
                        logging.warning("[VecEmbed] msearch batch failed, reverting to per-query: %s", e)
                    all_hits = [_lookup(v, q) for v, q in zip(vec_for_ent, queries)]
            else:
                all_hits = [_lookup(v, q) for v, q in zip(vec_for_ent, queries)]

        collected: Dict[str, Dict[str, Any]] = {}

        # ENRICH
        with self._span("CANON/VEC/enrich", ctx, requirement_index=idx):
            for ent_idx, hits in zip(ent_positions, all_hits):
                ent = ents[ent_idx]
                by_cid: Dict[str, Dict[str, Any]] = {}
                for h in hits:
                    if h["_score"] < self.cut:
                        continue
                    src = h.get("_source") or {}
                    cid = src.get("concept_id") or src.get("sctid")
                    if not cid:
                        continue

                    score = h["_score"]
                    term  = src.get("term", ent["query_term"])
                    rec   = by_cid.setdefault(
                        cid,
                        dict(
                            conceptId=cid,
                            preferred_term="",
                            fully_specified_name="",
                            type="",
                            definition="",
                            best_match_term=term,
                            match_score=score,
                            matched_alternative_synonyms=[],
                        ),
                    )
                    rec["matched_alternative_synonyms"].append({"term": term, "score": round(score, 3)})
                    if score > rec["match_score"]:
                        rec["match_score"]     = score
                        rec["best_match_term"] = term

                by_cid = {c: r for c, r in by_cid.items() if self._passes_category_filter(c)}
                if not by_cid:
                    continue

                ranked = sorted(by_cid.values(), key=lambda r: r["match_score"], reverse=True)
                topN   = ranked[: self.top_k]
                for j, rec in enumerate(topN):
                    pt, fsn = self._fetch_pt_fsn(rec["conceptId"])
                    rec["preferred_term"]       = pt
                    rec["fully_specified_name"] = fsn or rec["best_match_term"]
                    rec["type"]                 = _concept_type(rec["fully_specified_name"] or pt)
                    rec["all_terms"]            = self._fetch_all_terms(rec["conceptId"])
                    if self.fetch_definitions and j < self.defs_top_n and not rec["definition"]:
                        rec["definition"] = self._fetch_definition(rec["conceptId"])

                    col = collected.setdefault(rec["conceptId"], dict(pt=pt, all_terms=set()))
                    col["all_terms"].update(rec["all_terms"])

                ent["candidates"] = topN
                ent["provenance"] = ent.get("provenance", "llm_vec") + "|vec"

        # VERBOSE LOG
        if self.verbose:
            logging.info("\n[VecEmbed] requirement #%s:", idx)
            for ent in ents:
                cands = ent.get("candidates", [])
                if not cands:
                    logging.info('  • "%s" → 0 hits ≥ %.2f', ent.get("text", ent.get("query_term","")), self.cut)
                    continue
                logging.info('  • "%s" → %d concept(s):', ent.get("text", ent.get("query_term","")), len(cands))
                for rec in cands:
                    logging.info("      ↳ %s (%s)  score=%.3f",
                                 rec["preferred_term"], rec["conceptId"], rec["match_score"])
                    logging.info("        PT      : %s", rec["preferred_term"])
                    logging.info("        FSN     : %s", rec["fully_specified_name"])
                    logging.info("        Type    : %s", rec["type"])
                    if rec["definition"]:
                        logging.info("        Defn    : %s", rec["definition"])
                    logging.info('        Best hit: "%s"', rec["best_match_term"])

        return ctx
