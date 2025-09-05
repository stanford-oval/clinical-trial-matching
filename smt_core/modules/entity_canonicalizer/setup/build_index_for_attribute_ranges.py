#!/usr/bin/env python
"""
Build (or rebuild) a dense-vector index of **selected** SNOMED CT concepts
(English active descriptions only).  
提供 --allowed-file 或 --allowed-ids，就会过滤到这些 concept；否则默认全量。

依赖：
  pip install pandas sentence-transformers elasticsearch tqdm torch
"""

from pathlib import Path
from typing import Iterator, Dict, Any, Set
import logging, time, argparse, sys

import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers, ConnectionError

# ─────────────── USER DEFAULTS ───────────────
CONCEPT_PATH = Path(
    "SnomedCT_InternationalRF2_PRODUCTION_20250501T120000Z/"
    "Full/Terminology/sct2_Concept_Full_INT_20250501.txt"
)
DESC_PATH = Path(
    "SnomedCT_InternationalRF2_PRODUCTION_20250501T120000Z/"
    "Full/Terminology/sct2_Description_Full-en_INT_20250501.txt"
)

ES_URL      = "http://localhost:9200"
INDEX_NAME  = "snomed_vectors_test"   # 可被 --index-name 覆盖
MODEL_NAME  = "cambridgeltl/sapbert-from-pubmedbert-fulltext"

VECTOR_DIMS = 768
BATCH_SIZE  = 2_048    # encode batch
BULK_CHUNK  = 512      # ES bulk chunk
# ──────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("snomed-rebuild")


# ───────────── Elasticsearch helpers ─────────────
def recreate_index(es: Elasticsearch, index_name: str) -> None:
    """Delete if exists, then create fresh HNSW dense-vector index."""
    if es.indices.exists(index=index_name):
        log.info("Index %s exists – deleting …", index_name)
        es.indices.close(index=index_name, ignore_unavailable=True)
        es.indices.delete(index=index_name, ignore_unavailable=True)
        log.info("Index %s deleted.", index_name)

    mapping = {
        "properties": {
            "sctid":  {"type": "keyword"},
            "term":   {"type": "text", "analyzer": "english"},
            "vector": {
                "type": "dense_vector",
                "dims": VECTOR_DIMS,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
        }
    }
    es.indices.create(
        index=index_name,
        settings={"number_of_shards": 1},
        mappings=mapping,
    )
    log.info("Fresh index %s created.", index_name)


# ──────────────── Data loading ────────────────
def load_rf2(allowed: Set[str]) -> pd.DataFrame:
    """Load Concept + Description tables, filter to allowed conceptIds."""
    log.info("Reading Concept table …")
    concepts = (
        pd.read_csv(CONCEPT_PATH, sep="\t", dtype=str)
          .sort_values("effectiveTime")
          .drop_duplicates("id", keep="last")
          .query("active == '1'")[["id"]]
    )
    log.info("Active concepts: %d", len(concepts))

    log.info("Reading Description table …")
    descs = (
        pd.read_csv(
            DESC_PATH,
            sep="\t",
            dtype=str,
            usecols=["effectiveTime", "active", "conceptId", "term", "languageCode"],
        )
        .query("active == '1' and languageCode == 'en'")
        .sort_values("effectiveTime")
        .drop_duplicates(["conceptId", "term"], keep="last")
    )
    log.info("Active English descriptions: %d", len(descs))

    terms = descs.merge(concepts, left_on="conceptId", right_on="id")[["conceptId", "term"]]

    if allowed:
        before = len(terms)
        terms = terms.query("conceptId in @allowed")
        log.info("Filtered to allowed concepts: %d → %d", before, len(terms))
    else:
        log.info("No allowed list supplied – using all %d concepts.", len(terms))

    log.info("Joined pairs: %d", len(terms))
    return terms


# ─────────────── Doc generator ────────────────
def doc_stream(df: pd.DataFrame,
               model: SentenceTransformer,
               index_name: str) -> Iterator[Dict[str, Any]]:
    """Encode terms in batches, yield actions for helpers.streaming_bulk."""
    for start in tqdm(range(0, len(df), BATCH_SIZE),
                      desc="encoding", unit_scale=BATCH_SIZE):
        batch = df.iloc[start: start + BATCH_SIZE]
        vectors = model.encode(batch["term"].tolist(), normalize_embeddings=True)
        for (sctid, term), vec in zip(batch.values, vectors):
            yield {
                "_index": index_name,
                "_id": f"{sctid}-{hash(term)}",
                "_source": {"sctid": sctid, "term": term, "vector": vec.tolist()},
            }


# ───────────────────── Main ────────────────────
def main() -> None:
    # ---------- CLI ----------
    parser = argparse.ArgumentParser(
        description="Rebuild SNOMED dense-vector index (optionally subset).")
    parser.add_argument("--allowed-file",
                        help="Text file: one conceptId per line")
    parser.add_argument("--allowed-ids",
                        help="Comma-separated conceptId list")
    parser.add_argument("--index-name",
                        help=f"Name of index to create (default {INDEX_NAME})")
    args = parser.parse_args()

    index_name = args.index_name or INDEX_NAME

    # ---------- Build allowed set ----------
    allowed: Set[str] = set()
    if args.allowed_file:
        try:
            allowed |= {ln.strip()
                        for ln in Path(args.allowed_file).read_text().splitlines()
                        if ln.strip()}
        except FileNotFoundError:
            log.error("allowed-file %s not found", args.allowed_file)
            sys.exit(1)
    if args.allowed_ids:
        allowed |= {x.strip() for x in args.allowed_ids.split(",") if x.strip()}

    # ---------- ES connection ----------
    es = Elasticsearch(ES_URL, request_timeout=120)
    try:
        es.info()
    except ConnectionError as e:
        log.error("Elasticsearch not reachable @ %s – %s", ES_URL, e)
        return

    recreate_index(es, index_name)

    # ---------- Data load ----------
    terms = load_rf2(allowed)

    # ---------- Embedding model ----------
    log.info("Loading SapBERT model (%s)…", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("Model loaded.")

    # ---------- Bulk ingest ----------
    t0 = time.time()
    log.info("Bulk indexing (chunk %d) …", BULK_CHUNK)
    ok, fail = 0, 0
    for success, _ in helpers.streaming_bulk(
        es,
        doc_stream(terms, model, index_name),
        chunk_size=BULK_CHUNK,
        request_timeout=300,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        ok += success
        fail += 1 - success
    es.indices.refresh(index=index_name)
    log.info("Done.  OK: %d  Fail: %d  Runtime: %.1f min",
             ok, fail, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
