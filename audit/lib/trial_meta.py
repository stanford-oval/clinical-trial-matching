"""Load trial metadata (title, summary, official criteria text) from the dataset."""
from __future__ import annotations
import json, pathlib, functools

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOTS = [
    pathlib.Path("/tmp/satir_full_dataset"),
    ROOT / "dataset" / "clinical_trial",
]


@functools.lru_cache(maxsize=1)
def _corpus_cache() -> dict:
    for data_root in DATA_ROOTS:
        p = data_root / "sigir" / "corpus.jsonl"
        if not p.exists(): continue
        out = {}
        with open(p) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    out[obj.get("_id","")] = obj
                except Exception: pass
        if out: return out
    return {}


def load_trial_meta(tid: str) -> dict:
    corpus = _corpus_cache()
    if tid not in corpus: return {}
    obj = corpus[tid]
    md = obj.get("metadata") or {}
    return {
        "nct_id": obj.get("_id") or tid,
        "brief_title": md.get("brief_title", ""),
        "official_title": md.get("official_title", ""),
        "brief_summary": md.get("brief_summary", ""),
        "detailed_description": md.get("detailed_description", ""),
        "phase": md.get("phase", ""),
        "study_type": md.get("study_type", ""),
        "inclusion_criteria": md.get("inclusion_criteria", ""),
        "exclusion_criteria": md.get("exclusion_criteria", ""),
    }
