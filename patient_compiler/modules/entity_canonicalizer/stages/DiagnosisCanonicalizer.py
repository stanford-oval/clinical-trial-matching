# modules/DiagnosisCanonicalizer.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json, re, logging, requests
import dspy

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# ───────────────────────── SNOMED type helpers ─────────────────────────
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
    m = _TAG_RE.search((fsn_or_pt or "").strip())
    tag = m.group(1).lower() if m else ""
    return _TAG2TYPE.get(tag, tag.title() if tag else "")

# ───────────────────────── Config / prompts ─────────────────────────
@dataclass
class _ESCfg:
    es_url: str = "http://localhost:9200"
    index: str = "snomed_vectors"
    model_name: str = "cambridgeltl/sapbert-from-pubmedbert-fulltext"
    top_k: int = 5
    score_cut: float = 0.30
    overshoot: int = 4

# Built-in fallback prompts (used only if ctx doesn’t provide local files)
_FALLBACK_LINK_PROMPT = """\
# ROLE
You are a SNOMED concept linker for DIAGNOSIS STRINGS.

# PATIENT NOTE (context)
{NOTE}

# TASK
For EACH diagnosis, choose ONE best SNOMED concept from its candidate list.
If NO candidate correctly represents the diagnosis (as meant in this note), set "choice": null.

# STRICT JSON
Output strict JSON. Use lowercase true/false, no comments, no trailing commas.

# INPUT
<diagnoses_with_candidates>
{DIAG_BLOCK}
</diagnoses_with_candidates>

# OUTPUT (JSON array)
[
  {{
    "diagnosis": "<original diagnosis string>",
    "choice": {{
      "conceptId": "<SNOMED ID>",
      "preferred_term": "<PT>",
      "fully_specified_name": "<FSN>",
      "type": "<top semantic type>",
      "score": <float or omit>
    }}  // or null
  }}
]
"""

# ✅ Updated fallback verify prompt to 3-boolean schema
_FALLBACK_VERIFY_PROMPT = """\
# === ROLE ===
You are a SNOMED link auditor for DIAGNOSIS STRINGS.

# === GUIDELINES ===
1. Set "SNOMED_CONCEPT_AND_DIAGNOSIS_SEMANTICALLY_EQUIVALENT" to true if the chosen concept’s meaning is semantically equivalent to the diagnosis
as intended in the patient note context; else false.
2. Set "SNOMED_CONCEPT_IS_ANCESTOR_OF_DIAGNOSIS" to true if the chosen SNOMED concept is an ancestor of the diagnosis; else false.
3. Set "SNOMED_CONCEPT_AND_DIAGNOSIS_ARE_RELATED" to true if the SNOMED concept and the diagnosis are related; else false.

# STRICT JSON
Output strict JSON. Use lowercase true/false, no comments, no trailing commas.

# === PATIENT NOTE (context) ===
{NOTE}

# === PAIR ===
{PAIR}

# === OUTPUT (JSON object) ===
{{  "SNOMED_CONCEPT_AND_DIAGNOSIS_SEMANTICALLY_EQUIVALENT": true | false,
    "SNOMED_CONCEPT_IS_ANCESTOR_OF_DIAGNOSIS": true | false,
    "SNOMED_CONCEPT_AND_DIAGNOSIS_ARE_RELATED": true | false,
    "why": "<≤20 words if REJECT else empty>" }}
"""

# Keys expected in ctx (populated by ensure_prompt_templates)
LINK_PROMPT_KEY = "DiagnosisFilterLinker_prompt"
VERIFY_PROMPT_KEY = "DiagnosisFilterVerifier_prompt"

# ───────────────────────── Key normalizer ─────────────────────────
def _norm_diag_key(s: str) -> str:
    """
    Normalize diagnosis strings used as dict keys:
    - trim spaces
    - normalize curly quotes to straight
    - casefold for stable dictionary behavior
    """
    s = (s or "").strip()
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return s.casefold()

# ───────────────────────── JSON cleaning helpers ─────────────────────────
_CODEFENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.S)

def _strip_code_fences(s: str) -> str:
    return _CODEFENCE_RE.sub("", (s or "").strip())

def _largest_json_bracket_block(s: str, open_ch: str, close_ch: str) -> str:
    l, r = s.find(open_ch), s.rfind(close_ch)
    return s[l:r+1] if (l != -1 and r != -1 and r > l) else s

def _clean_jsonish_object(s: str) -> str:
    """Make common LLM 'JSON-like' outputs parseable as JSON objects."""
    s = _strip_code_fences(s)
    s = _largest_json_bracket_block(s, "{", "}")
    # normalize Python tokens to JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # conservative single-quote to double-quote fix if no double quotes present
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    return s

def _clean_jsonish_array(s: str) -> str:
    """Make common LLM 'JSON-like' outputs parseable as JSON arrays."""
    s = _strip_code_fences(s)
    s = _largest_json_bracket_block(s, "[", "]")
    # normalize Python tokens to JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # don't blanket replace quotes for arrays (may contain objects already valid)
    return s

# ───────────────────────── Canonicalizer ─────────────────────────
class DiagnosisCanonicalizer(dspy.Module):
    """
    Link + Filter with LLM:
      1) LLM links each diagnosis to 0/1 SNOMED concept from ES candidates.
      2) LLM verifies that single choice.

    Verification rule:
      • KEEP mapping if ANY of the three booleans is True:
          - SNOMED_CONCEPT_AND_DIAGNOSIS_SEMANTICALLY_EQUIVALENT
          - SNOMED_CONCEPT_IS_ANCESTOR_OF_DIAGNOSIS
          - SNOMED_CONCEPT_AND_DIAGNOSIS_ARE_RELATED
        Otherwise REJECT (mapping = {}).
      • Backward compatible with legacy {"decision":"KEEP"|"REJECT"} outputs.

    Inputs per diagnosis (from Diagnoser):
      diagnosis, confidence, status ("almost_certain"|"likely"|"possible"),
      supporting_evidence (or legacy 'support'), rationale, timeframe, duration

    Writes ctx['diagnosis_canonical'] with chosen 'mapping' (or {} if none verified).

    Logging (if `log_dir` provided, treated as a DIRECTORY):
      • link_raw.txt
      • verify_<diagnosis_snippet>.txt
      • ddx_cand.json
      • ddx_final.json
    """

    def __init__(
        self,
        engine: Callable[[str], List[str]] | dspy.Module,  # LLM for link+verify
        *,
        es_url: str = "http://localhost:9200",
        index: str = "snomed_vectors",
        model_name: str = "cambridgeltl/sapbert-from-pubmedbert-fulltext",
        snowstorm_url: str | None = "http://localhost:8080",
        branch: str = "MAIN",
        top_k: int = 5,
        score_cut: float = 0.30,
        overshoot: int = 4,
        allowed_types: tuple[str, ...] = ("Clinical finding",),
        strict_only_findings: bool = True,
        min_es_score_for_verify: float = 0.0,
        verbose: bool = False,
        log_dir: str | Path | None = None,
    ):
        super().__init__()
        self.cfg = _ESCfg(es_url, index, model_name, top_k, score_cut, overshoot)
        self.verbose = verbose
        self.engine = engine

        self.es = Elasticsearch(self.cfg.es_url, request_timeout=20)
        if not self.es.indices.exists(index=self.cfg.index):
            raise ValueError(f"Elasticsearch index {self.cfg.index!r} not found at {self.cfg.es_url}")

        self.model = SentenceTransformer(self.cfg.model_name)
        self.es_major = int(self.es.info()["version"]["number"].split(".")[0])

        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json"})
        self.snowstorm_url = snowstorm_url.rstrip("/") if snowstorm_url else None
        self.branch = branch

        self.allowed_types = set(allowed_types)
        self.strict_only_findings = strict_only_findings
        self.min_es_score_for_verify = float(min_es_score_for_verify)

        # Treat log_dir as a folder (create lazily on first write)
        self.log_dir = Path(log_dir) if log_dir else None

        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # ───────────────────────── internals ─────────────────────────
    def _render(self, template: str, **vars: Any) -> str:
        """Simple safe formatter for prompts."""
        return template.format(**vars)

    def _get_link_prompt(self, ctx: Dict[str, Any], note_text: str, diag_block: str) -> str:
        tmpl = ctx.get(LINK_PROMPT_KEY) or _FALLBACK_LINK_PROMPT
        return self._render(tmpl, NOTE=note_text, DIAG_BLOCK=diag_block)

    def _get_verify_prompt(self, ctx: Dict[str, Any], note_text: str, pair_json: str) -> str:
        tmpl = ctx.get(VERIFY_PROMPT_KEY) or _FALLBACK_VERIFY_PROMPT
        return self._render(tmpl, NOTE=note_text, PAIR=pair_json)

    # logging helpers (all inside self.log_dir)
    def _log_txt(self, name: str, content: str) -> None:
        if not self.log_dir:
            return
        p = Path(self.log_dir) / f"{name}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    # LLM calls
    def _call_llm(self, prompt: str) -> str:
        # engine returns List[str] or a dspy.Module behaving similarly
        out = self.engine(prompt)
        return out[0] if isinstance(out, list) else str(out)

    # ES queries
    def _knn_query(self, vec: list[float], k: int) -> dict:
        return {
            "size": k,
            "query": {"knn": {"field": "vector", "query_vector": vec, "k": k, "num_candidates": max(1000, k*10)}},
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }
    def _script_query(self, vec: list[float], k: int) -> dict:
        return {
            "size": k,
            "query": {"script_score": {"query": {"match_all": {}},
            "script": {"source": "cosineSimilarity(params.qv, 'vector') + 1.0", "params": {"qv": vec}}}},
            "_source": ["concept_id", "sctid", "term", "fsn", "pt", "is_pt"],
        }
    def _make_query(self, vec: list[float], k: int) -> dict:
        return self._knn_query(vec, k) if self.es_major >= 8 else self._script_query(vec, k)

    # PT/FSN via Snowstorm, fallback to ES
    def _fetch_pt_fsn(self, cid: str) -> Tuple[str, str]:
        pt, fsn = "", ""
        if self.snowstorm_url:
            try:
                r = self.http.get(f"{self.snowstorm_url}/browser/{self.branch}/concepts/{cid}?expand=fsn(),pt()", timeout=6)
                r.raise_for_status()
                j = r.json()
                pt, fsn = (j.get("pt") or {}).get("term", ""), (j.get("fsn") or {}).get("term", "")
                if pt or fsn:
                    return pt, fsn
            except Exception:
                pass
        # fallback: minimal fetch from ES
        q = {"size": 1, "query": {"bool": {"should": [{"term": {"concept_id.keyword": cid}}, {"term": {"sctid.keyword": cid}}]}}}
        try:
            hit = self.es.search(index=self.cfg.index, body=q)["hits"]["hits"][0]["_source"]
            return hit.get("pt") or hit.get("term", ""), hit.get("fsn") or hit.get("term", "")
        except Exception:
            return "", ""

    # Link with LLM (batch)
    def _llm_link(self, ctx: Dict[str, Any], note_text: str, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any] | None]:
        diag_block = json.dumps([
            {
                "diagnosis": it["diagnosis"],
                "candidates": [
                    {k: c.get(k) for k in ("conceptId","preferred_term","fully_specified_name","type","score")}
                    for c in (it.get("candidates") or [])
                ]
            }
            for it in items
        ], ensure_ascii=False, indent=2)
        prompt = self._get_link_prompt(ctx, note_text, diag_block)
        raw = self._call_llm(prompt)
        self._log_txt("link_raw", raw)

        # tolerate code fences / stray text
        clean = _clean_jsonish_array(raw)
        try:
            arr = json.loads(clean)
            if not isinstance(arr, list):
                raise ValueError
        except Exception:
            if self.verbose:
                logging.warning("[DDX-LINK] JSON parse failed; raw=%r clean=%r", raw, clean)
            arr = []

        # Prefer first non-null choice per normalized diagnosis key
        out_map: Dict[str, Dict[str, Any] | None] = {}
        for r in arr:
            if not isinstance(r, dict):
                continue
            k = _norm_diag_key(r.get("diagnosis", ""))
            ch = r.get("choice") or None
            if k not in out_map or (out_map[k] is None and ch is not None):
                out_map[k] = ch
        return out_map

    # ✅ Verify 1 diagnosis-choice pair (KEEP if any of the three booleans is True; supports legacy decision)
    def _llm_verify(self, ctx: Dict[str, Any], note_text: str, diagnosis: str, choice: Dict[str, Any]) -> bool:
        """
        Accepts either the new 3-boolean audit JSON or the legacy {"decision": "KEEP"|"REJECT"}.

        KEEP if ANY of:
          - SNOMED_CONCEPT_AND_DIAGNOSIS_SEMANTICALLY_EQUIVALENT
          - SNOMED_CONCEPT_IS_ANCESTOR_OF_DIAGNOSIS
          - SNOMED_CONCEPT_AND_DIAGNOSIS_ARE_RELATED
        Otherwise REJECT.
        """
        pair = {"diagnosis": diagnosis, "concept": choice}
        v_prompt = self._get_verify_prompt(ctx, note_text, json.dumps(pair, ensure_ascii=False, indent=2))
        raw = self._call_llm(v_prompt)

        safe_name = diagnosis.replace(" ", "_")[:40]
        self._log_txt(f"verify_{safe_name}", raw)

        # Robust JSON extraction & normalization
        raw_clean = _clean_jsonish_object(raw)
        try:
            obj = json.loads(raw_clean)
        except Exception:
            if self.verbose:
                logging.warning("[DDX-VERIFY] JSON parse failed; raw=%r clean=%r", raw, raw_clean)
            return False

        # Back-compat: support legacy {"decision": "..."} outputs
        decision = str(obj.get("decision", "")).strip().upper()
        if decision in {"KEEP", "REJECT"}:
            return decision == "KEEP"

        # New 3-boolean schema
        def _as_bool(x: Any) -> bool:
            if isinstance(x, bool):
                return x
            if isinstance(x, str):
                return x.strip().lower() in {"true", "yes", "y", "1"}
            if isinstance(x, (int, float)):
                return bool(x)
            return False

        eq  = _as_bool(obj.get("SNOMED_CONCEPT_AND_DIAGNOSIS_SEMANTICALLY_EQUIVALENT", False))
        anc = _as_bool(obj.get("SNOMED_CONCEPT_IS_ANCESTOR_OF_DIAGNOSIS", False))
        rel = _as_bool(obj.get("SNOMED_CONCEPT_AND_DIAGNOSIS_ARE_RELATED", False))

        # KEEP if any is True
        return bool(eq or anc or rel)

    # Main
    def forward(self, ctx: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        diags: List[Dict[str, Any]] = ctx.get("diagnosis_candidates", []) or []
        if not diags:
            ctx["diagnosis_canonical"] = []
            return ctx

        texts = [str(d.get("diagnosis", "")).strip() for d in diags]
        vecs  = self.model.encode(texts, normalize_embeddings=True)

        # Build ES candidates (Clinical finding only), top-k trimmed
        items_for_llm: List[Dict[str, Any]] = []
        all_cand_dump: Dict[str, List[Dict[str, Any]]] = {}

        for d, v in zip(diags, vecs):
            name = str(d.get("diagnosis", "")).strip()
            key  = _norm_diag_key(name)
            if not name:
                items_for_llm.append({"diagnosis": name, "candidates": []})
                all_cand_dump[key] = []
                continue

            hits = self.es.search(
                index=self.cfg.index,
                body=self._make_query(v.tolist(), self.cfg.top_k * self.cfg.overshoot)
            )["hits"]["hits"]

            cands: List[Dict[str, Any]] = []
            for h in hits:
                score = float(h["_score"])
                if score < self.cfg.score_cut:
                    continue
                src = h["_source"]
                cid = str(src.get("concept_id") or src.get("sctid") or "")
                if not cid:
                    continue
                pt, fsn = src.get("pt") or "", src.get("fsn") or ""
                if not (pt or fsn):
                    pt, fsn = self._fetch_pt_fsn(cid)
                ctype = _concept_type(fsn or pt)
                rec = {
                    "conceptId": cid,
                    "preferred_term": pt or src.get("term", name),
                    "fully_specified_name": fsn or pt or src.get("term", name),
                    "type": ctype or "",
                    "score": round(score, 3),
                }
                if (not self.allowed_types) or (rec["type"] in self.allowed_types):
                    cands.append(rec)

            cands = sorted(cands, key=lambda r: r["score"], reverse=True)[: self.cfg.top_k]
            all_cand_dump[key] = cands
            items_for_llm.append({"diagnosis": name, "candidates": cands})

        # LLM LINK (choose 0/1 per diagnosis)
        note_text = str(ctx.get("patient_note") or ctx.get("requirement_text") or "")
        linked_map = self._llm_link(ctx, note_text, items_for_llm)

        # LLM VERIFY (no fallback beyond the rules above)
        out_rows: List[Dict[str, Any]] = []
        for d in diags:
            name = str(d.get("diagnosis", "")).strip()
            key  = _norm_diag_key(name)
            choice = linked_map.get(key)
            mapping: Dict[str, Any] = {}

            # optional numeric floor before verify
            if choice and self.min_es_score_for_verify > 0:
                # Prefer score provided by the linker; fallback to candidate lookup
                score = choice.get("score") if isinstance(choice, dict) else None
                if score is None:
                    score = next(
                        (c.get("score") for c in all_cand_dump.get(key, [])
                         if c.get("conceptId") == (choice or {}).get("conceptId")),
                        0.0
                    )
                if float(score or 0.0) < self.min_es_score_for_verify:
                    if self.verbose:
                        logging.info("[DDX-GATE] %s → dropped by min score (%.3f < %.3f)",
                                     name, float(score or 0.0), self.min_es_score_for_verify)
                    choice = None

            # backfill score on choice for consistent logging/gating downstream
            if choice is not None and "score" not in choice:
                choice["score"] = next(
                    (c.get("score") for c in all_cand_dump.get(key, [])
                     if c.get("conceptId") == choice.get("conceptId")),
                    None
                )

            keep = False
            if choice:
                keep = self._llm_verify(ctx, note_text, name, choice)
                if self.verbose:
                    score_log = choice.get("score")
                    if score_log is None:
                        score_log = next((c.get("score") for c in all_cand_dump.get(key, [])
                                          if c.get("conceptId") == choice.get("conceptId")), 0.0)
                    logging.info("[DDX-VERIFY] %s → %s (cid=%s, score=%s)",
                                 name, "KEEP" if keep else "REJECT",
                                 choice.get("conceptId"), "NA" if score_log is None else f"{float(score_log):.3f}")

            if keep:
                mapping = {
                    "conceptId": choice.get("conceptId",""),
                    "preferred_term": choice.get("preferred_term",""),
                    "fully_specified_name": choice.get("fully_specified_name",""),
                    "type": choice.get("type",""),
                }

            evidence = d.get("supporting_evidence", d.get("support", []))
            out_rows.append({
                "diagnosis": name,
                "confidence": d.get("confidence", 0.5),
                "status": d.get("status", "possible"),
                "supporting_evidence": evidence,
                "rationale": d.get("rationale", ""),
                "timeframe": d.get("timeframe", ""),
                "duration": d.get("duration", None),
                # "confirmable_latest_start_time": d.get("confirmable_latest_start_time", None),
                # "confirmable_earliest_end_time": d.get("confirmable_earliest_end_time", None),
                "start_time_in_hours": d.get("start_time_in_hours", None),
                "end_time_in_hours": d.get("end_time_in_hours", None),
                "start_time_inclusive": d.get("start_time_inclusive", True),
                "end_time_inclusive": d.get("end_time_inclusive", True),
                "mapping": mapping,                           # {} if rejected or no choice
                "all_candidates": all_cand_dump.get(key, []),
            })

        ctx["diagnosis_canonical"] = out_rows

        # sidecar logs (all under the same folder)
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.log_dir) / "ddx_cand.json").write_text(
                json.dumps(all_cand_dump, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (Path(self.log_dir) / "ddx_final.json").write_text(
                json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if self.verbose:
            mapped = sum(1 for r in out_rows if r.get("mapping"))
            logging.info("[DDX-CANON] LLM linked+verified %d/%d diagnoses (findings-only=%s)",
                         mapped, len(out_rows), self.strict_only_findings)

        return ctx
