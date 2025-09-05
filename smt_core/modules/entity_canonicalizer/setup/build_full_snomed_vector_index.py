#!/usr/bin/env python
"""
Build (or rebuild) a dense-vector index of every active English SNOMED CT
description.  Existing index is deleted first ― you always get a fresh copy.

Requires:
  pip install pandas sentence-transformers elasticsearch tqdm torch
"""

from pathlib import Path
from typing import Iterator, Dict, Any
import logging, time

import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers, ConnectionError

# ─────────────── USER SETTINGS ───────────────
CONCEPT_PATH = Path(
    "SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z/"
    "Full/Terminology/sct2_Concept_Full_INT_20251101.txt"
)
DESC_PATH = Path(
    "SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z/"
    "Full/Terminology/sct2_Description_Full-en_INT_20251101.txt"
)

ES_URL        = "http://localhost:9200"
INDEX_NAME    = "snomed_vectors"
MODEL_NAME    = "cambridgeltl/sapbert-from-pubmedbert-fulltext"

VECTOR_DIMS   = 768
BATCH_SIZE    = 2_048    # encode batch
BULK_CHUNK    = 512      # ES bulk chunk
# ──────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("snomed-rebuild")

# --------- Elasticsearch helpers ----------
def recreate_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=INDEX_NAME):
        log.info("Index %s exists – deleting …", INDEX_NAME)
        es.indices.close(index=INDEX_NAME, ignore_unavailable=True)
        es.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
        log.info("Index %s deleted.", INDEX_NAME)

    mapping = {
        "properties": {
            "sctid": {"type": "keyword"},
            "term":  {"type": "text", "analyzer": "english"},
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
        index=INDEX_NAME,
        settings={"number_of_shards": 1},
        mappings=mapping,
    )
    log.info("Fresh index %s created.", INDEX_NAME)


def load_rf2() -> pd.DataFrame:
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
            usecols=["effectiveTime","active","conceptId","term","languageCode"],
        )
        .query("active == '1' and languageCode == 'en'")
        .sort_values("effectiveTime")
        .drop_duplicates(["conceptId","term"], keep="last")
    )
    log.info("Active English descriptions: %d", len(descs))

    terms = descs.merge(concepts, left_on="conceptId", right_on="id")[
        ["conceptId", "term"]
    ]
    log.info("Joined pairs: %d", len(terms))
    return terms


def doc_stream(df: pd.DataFrame, model: SentenceTransformer) -> Iterator[Dict[str, Any]]:
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="encoding", unit_scale=BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        vectors = model.encode(batch["term"].tolist(), normalize_embeddings=True)
        for (sctid, term), vec in zip(batch.values, vectors):
            yield {
                "_index": INDEX_NAME,
                "_id": f"{sctid}-{hash(term)}",
                "_source": {"sctid": sctid, "term": term, "vector": vec.tolist()},
            }


def main() -> None:
    t0 = time.time()
    es = Elasticsearch(ES_URL, request_timeout=120)

    try:
        es.info()
    except ConnectionError as e:
        log.error("Elasticsearch not reachable @ %s – %s", ES_URL, e)
        return

    recreate_index(es)

    terms = load_rf2()

    log.info("Loading SapBERT (%s)…", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("Model loaded.")

    log.info("Bulk indexing (chunk %d) …", BULK_CHUNK)
    ok, fail = 0, 0
    for success, _ in helpers.streaming_bulk(
        es,
        doc_stream(terms, model),
        chunk_size=BULK_CHUNK,
        request_timeout=300,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        ok += success
        fail += 1 - success
    es.indices.refresh(index=INDEX_NAME)
    log.info("Done.  OK: %d  Fail: %d  Runtime: %.1f min", ok, fail, (time.time()-t0)/60)


if __name__ == "__main__":
    main()
