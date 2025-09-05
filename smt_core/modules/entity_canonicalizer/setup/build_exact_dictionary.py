#!/usr/bin/env python
"""
build_exact_dictionary.py   –   SNOMED → 19-root dictionary
----------------------------------------------------------------
Creates lines like:
   {"term": "tracheal stenosis following tracheostomy",
    "class": "clinical finding"}
Only concepts whose FSN tag maps to one of the 19 SNOMED hierarchies
are kept; everything else is dropped.
"""

from __future__ import annotations
import argparse, csv, gzip, json, re, sys
from pathlib import Path
from typing import Iterable, List, Tuple

csv.field_size_limit(sys.maxsize)          # allow huge RF2 rows

# ────────────────── normaliser ──────────────────
_WS_RE = re.compile(r"\s+")

def norm_text(txt: str) -> str:
    return _WS_RE.sub(" ", txt.casefold()).strip()

_BRACKET_PREFIX_RE = re.compile(r"^\[[^\]]+]\s*")   # strip “[M] …”

FSN_TYPE_ID = "900000000000003001"

# ────────────── FSN tag → 19-root map ───────────
TAG2TOP = {
    "clinical finding": "clinical finding",
    "finding":          "clinical finding",
    "disorder":         "clinical finding",
    "procedure":        "procedure",
    "observable entity":"observable entity",
    "organism":         "organism",
    "substance":        "substance",
    "pharmaceutical / biologic product":
                       "pharmaceutical / biologic product",
    "morphologic abnormality": "morphologic abnormality",
    "body structure":          "body structure",
    "specimen":                "specimen",
    "event":                   "event",
    "situation":               "situation with explicit context",
    "situation with explicit context":
                       "situation with explicit context",
    "physical object":         "physical object",
    "physical force":          "physical force",
    "environment":             "environment or geographical location",
    "geographic location":     "environment or geographical location",
    "qualifier value":         "qualifier value",
    "record artifact":         "record artifact",
    "social context":          "social context",
}

# ────────────────── iterators ───────────────────
def iter_rf2(path: Path) -> Iterable[Tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        idx = {c: header.index(c)
               for c in ("active", "languageCode", "typeId", "term")}
        for row in reader:
            if row[idx["active"]] != "1" or row[idx["languageCode"]] != "en":
                continue
            if row[idx["typeId"]] != FSN_TYPE_ID:
                continue
            fsn = row[idx["term"]]
            if "(" not in fsn:
                continue
            tag_raw = fsn.rsplit("(", 1)[-1].rstrip(")").casefold()
            top = TAG2TOP.get(tag_raw)
            if top is None:
                continue            # skip tags we don’t map
            surface = fsn.rsplit("(", 1)[0].strip()
            surface = _BRACKET_PREFIX_RE.sub("", surface)
            yield norm_text(surface), top

def iter_curated(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            term_raw, cls_raw = line.rstrip("\n").split("\t")
            cls = TAG2TOP.get(cls_raw.casefold())
            if cls:                                  # only keep mappable
                yield norm_text(term_raw), cls

# ─────────────────── main ───────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf2", required=True, help="Description Snapshot file")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--curated", help="Optional TSV term\tclass")
    args = ap.parse_args()

    pairs: List[Tuple[str, str]] = list(iter_rf2(Path(args.rf2)))
    if args.curated:
        pairs.extend(iter_curated(Path(args.curated)))

    seen: set[str] = set()
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as fh:
        for term, cls in pairs:
            if term in seen:
                continue
            seen.add(term)
            json.dump({"term": term, "class": cls}, fh, ensure_ascii=False)
            fh.write("\n")

    print(f"Wrote {len(seen):,} unique terms → {out_p}")

if __name__ == "__main__":
    sys.exit(main())
