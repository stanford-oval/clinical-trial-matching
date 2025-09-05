#!/usr/bin/env python
"""
Smoke-test SNOMED vector index on ES 7.17 *or* 8.x.

• Checks mapping dims.
• Encodes probe terms with SapBERT.
• Runs k-NN (ES 8) or cosine script_score (ES 7).
• Asserts probe term appears in top-10.
"""

import sys, logging
from typing import Dict, Any

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_URL       = "http://localhost:9200"
INDEX        = "snomed_vectors"
MODEL        = "cambridgeltl/sapbert-from-pubmedbert-fulltext"
DIMS         = 768
TOPK         = 10
PROBES       = [
    "coronary artery bypass graft",
    "metformin",
    "stage iii",
    "left ventricular ejection fraction",
    "CABG",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("vector-test")


def es_major(es: Elasticsearch) -> int:
    return int(es.info()["version"]["number"].split(".")[0])


def knn_query(vec, k) -> Dict[str, Any]:
    return {
        "size": k,
        "query": {
            "knn": {
                "field": "vector",
                "query_vector": vec,
                "k": k,
                "num_candidates": 1000,
            }
        },
        "_source": ["term"],
    }


def script_query(vec, k) -> Dict[str, Any]:
    return {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": vec},
                },
            }
        },
        "_source": ["term"],
    }


def main() -> None:
    es = Elasticsearch(ES_URL, request_timeout=30)
    assert es.indices.exists(index=INDEX), "Index not found; run builder first."

    # mapping check
    dims = es.indices.get_mapping(index=INDEX)[INDEX]["mappings"]["properties"]["vector"][
        "dims"
    ]
    assert int(dims) == DIMS, f"Mapping dims {dims} ≠ expected {DIMS}"
    log.info("Mapping OK (%s dims).", dims)

    major = es_major(es)
    log.info("Elasticsearch %d.x detected – using %s query.", major, "knn" if major >= 8 else "script_score")

    model = SentenceTransformer(MODEL)
    vecs = model.encode(PROBES, normalize_embeddings=True)

    failures = 0
    for term, vec in zip(PROBES, vecs):
        body = knn_query(vec.tolist(), TOPK) if major >= 8 else script_query(vec.tolist(), TOPK)
        hits = es.search(index=INDEX, body=body)["hits"]["hits"]
        if not hits:
            log.error("No hits for probe %r", term)
            failures += 1
            continue
        hit_terms = [h["_source"]["term"].lower() for h in hits]
        if term.lower() in hit_terms:
            log.info("✓ Probe %r found (score %.3f)", term, hits[0]["_score"])
        else:
            log.warning("✗ Probe %r NOT in top-%d  (best hit %s)",
                        term, TOPK, hits[0]['_source']['term'])
            failures += 1

    if failures:
        log.error("%d / %d probes failed", failures, len(PROBES))
        sys.exit(1)
    else:
        log.info("All probes passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
