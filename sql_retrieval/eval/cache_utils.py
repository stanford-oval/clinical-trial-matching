from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tf:
        tf.write(text)
        tmp = tf.name
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


@dataclass(frozen=True)
class CacheKey:
    mode: str  # "cc" | "ccr" | "all"
    patient_id: str
    trial_id: str

    def dir(self, cache_root: Path) -> Path:
        return cache_root / self.mode / self.patient_id / self.trial_id


class PairDiskCache:
    """
    Cache layout:
      <output_root>/.cache/<mode>/<patient_id>/<trial_id>/
        meta.json
        relevance.txt
        eligibility.txt
    """

    def __init__(self, output_root: Path):
        self.cache_root = output_root / ".cache"

    def _meta_path(self, key: CacheKey) -> Path:
        return key.dir(self.cache_root) / "meta.json"

    def _relevance_path(self, key: CacheKey) -> Path:
        return key.dir(self.cache_root) / "relevance.txt"

    def _eligibility_path(self, key: CacheKey) -> Path:
        return key.dir(self.cache_root) / "eligibility.txt"

    def compute_meta(
        self,
        *,
        patient_text: str,
        trial_text: str,
        relevance_prompt_path: Path,
        eligibility_prompt_path: Path,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        # Strict meta: any change -> cache miss
        meta: Dict[str, Any] = {
            "patient_text_sha256": sha256_text(patient_text),
            "trial_text_sha256": sha256_text(trial_text),
            "relevance_prompt_path": str(relevance_prompt_path),
            "relevance_prompt_sha256": sha256_file(relevance_prompt_path),
            "eligibility_prompt_path": str(eligibility_prompt_path),
            "eligibility_prompt_sha256": sha256_file(eligibility_prompt_path),
            "model_name": model_name,
            "temperature": float(temperature),
            "max_tokens": None if max_tokens is None else int(max_tokens),
        }
        return meta

    def load_if_fresh(self, key: CacheKey, expected_meta: Dict[str, Any]) -> Optional[Dict[str, str]]:
        mp = self._meta_path(key)
        rp = self._relevance_path(key)
        ep = self._eligibility_path(key)

        if not (mp.exists() and rp.exists() and ep.exists()):
            return None

        try:
            cur = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None

        if cur != expected_meta:
            return None

        return {
            "relevance": rp.read_text(encoding="utf-8"),
            "eligibility": ep.read_text(encoding="utf-8"),
        }

    def save(self, key: CacheKey, meta: Dict[str, Any], relevance: str, eligibility: str) -> None:
        atomic_write_text(self._relevance_path(key), relevance)
        atomic_write_text(self._eligibility_path(key), eligibility)
        atomic_write_json(self._meta_path(key), meta)