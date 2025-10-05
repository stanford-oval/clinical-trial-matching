# modules/joint_specificity_filter.py
# -*- coding: utf-8 -*-

import os
import json
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------
# Parser / verifier
# ----------------------------

class ParsingError(Exception):
    pass

class VerificationError(Exception):
    pass


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction: find first '{' ... last '}' and parse as JSON.
    """
    if not isinstance(text, str):
        raise ParsingError("LLM output is not a string")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ParsingError("Could not find JSON object in output")

    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except Exception as e:
        raise ParsingError(f"JSON parse failed: {e}")


def parse_and_verify_joint_filter(
    raw_text: str,
    *,
    disease_count: int,
    poslit_count: int,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    obj = _extract_json_object(raw_text)

    if not isinstance(obj, dict):
        raise VerificationError("Top-level JSON must be an object")

    diseases = obj.get("diseases")
    poslits = obj.get("positive_constraint_literals")

    if not isinstance(diseases, dict):
        raise VerificationError("Missing/invalid 'diseases' object")
    if not isinstance(poslits, dict):
        raise VerificationError("Missing/invalid 'positive_constraint_literals' object")

    def _validate_bucket(bucket: Dict[str, Any], n: int, label: str):
        decisions: Dict[int, str] = {}
        meta: Dict[int, Dict[str, Any]] = {}

        for k, v in bucket.items():
            try:
                idx = int(k)
            except Exception:
                raise VerificationError(f"{label}: key {k!r} is not an int index string")

            if not (0 <= idx < n):
                raise VerificationError(f"{label}: index {idx} out of range [0, {n})")

            if not isinstance(v, dict):
                raise VerificationError(f"{label}[{idx}]: value must be an object")

            decision = v.get("decision")
            if not isinstance(decision, str):
                raise VerificationError(f"{label}[{idx}]: missing decision")
            decision = decision.strip().upper()
            if decision not in ("KEEP", "DROP"):
                raise VerificationError(f"{label}[{idx}]: decision must be KEEP/DROP")

            decisions[idx] = decision
            meta[idx] = v

        return decisions, meta

    disease_decisions, disease_meta = _validate_bucket(diseases, disease_count, "diseases")
    poslit_decisions, poslit_meta = _validate_bucket(poslits, poslit_count, "positive_constraint_literals")

    return disease_decisions, poslit_decisions, disease_meta, poslit_meta


# ----------------------------
# Filter module
# ----------------------------

@dataclass
class JointFilterResult:
    trial_id: str
    disease_in: Optional[str]
    poslit_in: Optional[str]
    disease_out: Optional[str]
    poslit_out: Optional[str]
    error: Optional[str]
    raw_output: Optional[str]


class JointSpecificityFilter:
    def __init__(
        self,
        engine,
        *,
        build_root: Path,
        disease_subdir: str = "disease",
        positive_literal_subdir: str = "positive_constraint_literals/per_file",
        canon_expanded_subdir: str = "canon",
        context_folder: Path = Path("../../canonical_subcohort_results"),
        corpus_jsonl: Path = Path("../../dataset/clinical_trial/sigir/corpus.jsonl"),
        prompt_path: Path = Path("prompts/JointSpecificityFilter.prompt"),
        default_temp: float = 0.0,
        default_top_p: float = 1.0,
        max_retries: int = 3,
        log_dir: Path = Path("mbench/joint_filter"),
    ) -> None:
        self.engine = engine
        self.build_root = build_root

        self.disease_dir = build_root / disease_subdir
        self.poslit_dir = build_root / positive_literal_subdir
        self.canon_dir = build_root / canon_expanded_subdir

        self.disease_out_dir = build_root / f"{disease_subdir}_filtered"
        self.poslit_out_dir = build_root / "positive_constraint_literals_filtered" / "per_file"
        self.disease_out_dir.mkdir(parents=True, exist_ok=True)
        self.poslit_out_dir.mkdir(parents=True, exist_ok=True)

        self.context_folder = context_folder
        self.corpus_jsonl = corpus_jsonl

        self.prompt_path = prompt_path
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")

        self.default_temp = default_temp
        self.default_top_p = default_top_p
        self.max_retries = max_retries

        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.warnings: List[Dict[str, str]] = []

        self._corpus_text_by_id: Optional[Dict[str, str]] = None

    def _trial_dir(self, trial_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", trial_id)
        d = self.log_dir / "trials" / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _log_attempt(
        self,
        *,
        trial_id: str,
        attempt: int,
        prompt: str,
        raw: str,
        parsed_obj: Optional[Dict[str, Any]] = None,
        decisions: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
        failed: bool = False,
    ) -> None:
        tdir = self._trial_dir(trial_id)
        adir = tdir / (f"attempt_{attempt:02d}_FAIL" if failed else f"attempt_{attempt:02d}")
        adir.mkdir(parents=True, exist_ok=True)

        (adir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (adir / "raw.txt").write_text(raw, encoding="utf-8")

        if note:
            (adir / "note.txt").write_text(note, encoding="utf-8")

        if parsed_obj is not None:
            (adir / "parsed.json").write_text(
                json.dumps(parsed_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if decisions is not None:
            (adir / "decisions.json").write_text(
                json.dumps(decisions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _write_result_json(
        self,
        *,
        trial_id: str,
        disease_path: Optional[Path],
        poslit_path: Optional[Path],
        disease_out: Optional[str],
        poslit_out: Optional[str],
        error: Optional[str],
        raw_output: Optional[str] = None,
    ) -> None:
        tdir = self._trial_dir(trial_id)
        payload: Dict[str, Any] = {
            "trial_id": trial_id,
            "disease_in": str(disease_path) if disease_path else None,
            "poslit_in": str(poslit_path) if poslit_path else None,
            "disease_out": disease_out,
            "poslit_out": poslit_out,
            "error": error,
        }
        if error and raw_output is not None:
            payload["raw_output_preview"] = (raw_output[:2000] + "...") if len(raw_output) > 2000 else raw_output

        (tdir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log_trial_bundle(
        self,
        *,
        trial_id: str,
        trial_text: str,
        trial_provenance: Dict[str, Any],
        disease_names: List[str],
        poslit_hits: List[str],
        arm: str,
        canon_path: Optional[Path],
        poslit_items: List[Dict[str, Any]],
        disease_obj: Optional[Dict[str, Any]],
        poslit_obj: Optional[Dict[str, Any]],
    ) -> None:
        tdir = self._trial_dir(trial_id)
        info_dir = tdir / "trial_info"
        info_dir.mkdir(parents=True, exist_ok=True)

        (info_dir / "trial_text.txt").write_text(trial_text or "", encoding="utf-8")
        (info_dir / "trial_text_source.json").write_text(
            json.dumps(trial_provenance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (info_dir / "inputs.json").write_text(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "arm": arm,
                    "disease_names": disease_names,
                    "poslit_hits": poslit_hits,
                    "canon_path_used": str(canon_path) if canon_path else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (info_dir / "poslit_items.json").write_text(
            json.dumps(poslit_items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if disease_obj is not None:
            (info_dir / "disease_input.json").write_text(
                json.dumps(disease_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if poslit_obj is not None:
            (info_dir / "poslit_input.json").write_text(
                json.dumps(poslit_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        (info_dir / "trial_meta.json").write_text(
            json.dumps(
                {
                    "trial_text_chars": len(trial_text or ""),
                    "num_diseases": len(disease_names),
                    "num_poslit_hits": len(poslit_hits),
                    "num_poslit_items": len(poslit_items),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _complete_with_decoding(self, prompt: str, *, seed: int, temp: float, top_p: float) -> str:
        try:
            return self.engine(prompt, temperature=temp, top_p=top_p, seed=seed)[0]
        except TypeError:
            try:
                if hasattr(self.engine, "kwargs") and isinstance(self.engine.kwargs, dict):
                    self.engine.kwargs["temperature"] = temp
                    self.engine.kwargs["top_p"] = top_p
                return self.engine(prompt)[0]
            except Exception:
                return self.engine(prompt)[0]

    def _call_llm(self, prompt: str, *, seed: int) -> str:
        return self._complete_with_decoding(prompt, seed=seed, temp=self.default_temp, top_p=self.default_top_p)

    def _ensure_corpus_index(self) -> None:
        if self._corpus_text_by_id is not None:
            return
        idx: Dict[str, str] = {}
        p = self.corpus_jsonl
        if not p.exists():
            self._corpus_text_by_id = {}
            return
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _id = obj.get("_id")
                txt = obj.get("text")
                if isinstance(_id, str) and _id:
                    idx[_id] = txt if isinstance(txt, str) else ""
        self._corpus_text_by_id = idx

    def _load_trial_text_with_provenance(self, trial_id: str) -> Tuple[str, Dict[str, Any]]:
        base_id = re.sub(r"[a-zA-Z]$", "", trial_id)
        context_path = self.context_folder / f"{base_id}.json"

        prov: Dict[str, Any] = {
            "trial_id": trial_id,
            "base_id": base_id,
            "used_source": None,
            "context_path": str(context_path),
            "corpus_jsonl": str(self.corpus_jsonl),
            "subcohort": (trial_id != base_id),
            "details": {},
        }

        if context_path.exists():
            try:
                data = json.loads(context_path.read_text(encoding="utf-8"))
                extracted = data["extracted"]["preprocessor_normalized"]

                if trial_id == base_id:
                    shared = extracted.get("shared_context", "")
                    if isinstance(shared, str) and shared.strip():
                        prov["used_source"] = "canonical_subcohort_results"
                        prov["details"] = {"mode": "shared_context"}
                        return shared, prov
                else:
                    cohorts = extracted.get("enrollment_cohorts", [])
                    if isinstance(cohorts, list):
                        for c in cohorts:
                            if isinstance(c, dict) and c.get("trial_id_effective") == trial_id:
                                label = c.get("label", "") if isinstance(c.get("label"), str) else ""
                                ctxt = c.get("contextual_text", "") if isinstance(c.get("contextual_text"), str) else ""
                                inc = c.get("inclusion_criteria", "") if isinstance(c.get("inclusion_criteria"), str) else ""
                                exc = c.get("exclusion_criteria", "") if isinstance(c.get("exclusion_criteria"), str) else ""
                                cohort_context = f"{label}\n{ctxt}\ninclusion criteria:\n{inc}\nexclusion criteria:\n{exc}".strip()
                                if cohort_context:
                                    prov["used_source"] = "canonical_subcohort_results"
                                    prov["details"] = {"mode": "enrollment_cohort", "label": label}
                                    return cohort_context, prov

                shared = extracted.get("shared_context", "")
                if isinstance(shared, str) and shared.strip():
                    prov["used_source"] = "canonical_subcohort_results"
                    prov["details"] = {"mode": "shared_context_fallback"}
                    return shared, prov

            except Exception as e:
                prov["details"] = {"canonical_parse_error": repr(e)}

        self._ensure_corpus_index()
        assert self._corpus_text_by_id is not None
        for key in (trial_id, base_id):
            if key in self._corpus_text_by_id:
                txt = self._corpus_text_by_id.get(key, "") or ""
                if txt.strip():
                    prov["used_source"] = "corpus"
                    prov["details"] = {"lookup_key": key}
                    return txt, prov

        prov["used_source"] = "none"
        prov["details"] = {"reason": "not_found_in_context_or_corpus"}
        return "", prov

    def _find_canon_file(self, trial_id: str, arm: str) -> Optional[Path]:
        pattern = f"{trial_id}_{arm}*.json"
        candidates = list(self.canon_dir.glob(pattern))
        if candidates:
            return sorted(candidates)[0]
        fallback = list(self.canon_dir.glob(f"{trial_id}*.json"))
        if fallback:
            return sorted(fallback)[0]
        return None

    @staticmethod
    def _load_canonical_variables(canon_path: Path) -> List[Dict[str, Any]]:
        data = json.loads(canon_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "canonical_variables" in data:
            cv = data["canonical_variables"]
        else:
            cv = data
        if not isinstance(cv, list):
            raise ValueError("canon file doesn't contain list")
        return cv

    @staticmethod
    def _strip_qualifiers(cv_item: Dict[str, Any]) -> Dict[str, Any]:
        drop = {
            "qualifier_predicates",
            "qualifier_predicates_detailed",
            "qualifier_predicates_for_semantics_not_already_captured_with_stem",
            "index_in_canonical_forms",
            "inc_exc",
            "__reuse_existing_symbol",
            "__do_not_update_variable_meaning",
            "requirement_id",
        }
        return {k: v for k, v in cv_item.items() if k not in drop}

    def _build_poslit_items_from_hits(self, hits: List[str], canon_path: Path) -> List[Dict[str, Any]]:
        cvs = self._load_canonical_variables(canon_path)
        index_exact: Dict[str, Dict[str, Any]] = {}
        index_stem: Dict[str, Dict[str, Any]] = {}

        for it in cvs:
            name = it.get("entity_variable_name")
            if isinstance(name, str) and name:
                index_exact.setdefault(name, it)
                stem = name.split("@@", 1)[0]
                index_stem.setdefault(stem, it)

        out: List[Dict[str, Any]] = []
        for h in hits:
            if not isinstance(h, str) or not h:
                continue
            chosen = index_exact.get(h)
            if chosen is None:
                stem = h.split("@@", 1)[0]
                chosen = index_stem.get(stem)
            if chosen is None:
                out.append({"entity_variable_name": h, "template": "other", "_missing_in_canon": True})
                continue
            out.append(self._strip_qualifiers(chosen))
        return out

    def _build_prompt(
        self,
        *,
        trial_text: str,
        disease_names: List[str],
        poslit_items: List[Dict[str, Any]],
    ) -> str:
        diseases_dict = {str(i): d for i, d in enumerate(disease_names)}
        poslit_dict = {
            str(i): {
                "name": it.get("entity_variable_name", ""),
                "template": it.get("template", "other"),
                "entity_text": it.get("entity_text", it.get("entity", "")),
            }
            for i, it in enumerate(poslit_items)
        }

        prompt = self._prompt_template
        prompt = prompt.replace("#CLINICAL_TRIAL_DESCRIPTION#", trial_text or "")
        prompt = prompt.replace("#DISEASES_JSON#", json.dumps(diseases_dict, ensure_ascii=False, indent=2))
        prompt = prompt.replace("#POSLITS_JSON#", json.dumps(poslit_dict, ensure_ascii=False, indent=2))
        return prompt

    @staticmethod
    def _trial_id_from_poslit_file(obj: Dict[str, Any], default: str) -> str:
        tid = obj.get("trial_id")
        return tid if isinstance(tid, str) and tid else default

    @staticmethod
    def _trial_id_from_disease_file(obj: Dict[str, Any], default: str) -> str:
        tid = obj.get("trial_id")
        return tid if isinstance(tid, str) and tid else default

    def filter_one_trial(
        self,
        *,
        trial_id: str,
        disease_path: Optional[Path],
        poslit_path: Optional[Path],
        seed: int,
    ) -> JointFilterResult:
        disease_obj: Optional[Dict[str, Any]] = None
        poslit_obj: Optional[Dict[str, Any]] = None

        disease_names: List[str] = []
        poslit_hits: List[str] = []
        arm = "inclusion"

        if disease_path and disease_path.exists():
            disease_obj = json.loads(disease_path.read_text(encoding="utf-8"))
            fscbd = disease_obj.get("final_selected_concept_by_disease", {})
            if isinstance(fscbd, dict):
                disease_names = list(fscbd.keys())

        if poslit_path and poslit_path.exists():
            poslit_obj = json.loads(poslit_path.read_text(encoding="utf-8"))
            arm = (poslit_obj.get("arm", "") or "").lower() or "inclusion"
            poslit_hits = poslit_obj.get("hits", [])
            if not isinstance(poslit_hits, list):
                poslit_hits = []

        if not disease_names and not poslit_hits:
            trial_text, prov = self._load_trial_text_with_provenance(trial_id)

            self._log_trial_bundle(
                trial_id=trial_id,
                trial_text=trial_text,
                trial_provenance=prov,
                disease_names=disease_names,
                poslit_hits=poslit_hits,
                arm=arm,
                canon_path=None,
                poslit_items=[],
                disease_obj=disease_obj,
                poslit_obj=poslit_obj,
            )

            disease_out = None
            poslit_out = None
            if disease_obj is not None and disease_path is not None:
                outp = self.disease_out_dir / disease_path.name
                outp.write_text(json.dumps(disease_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                disease_out = str(outp)
            if poslit_obj is not None and poslit_path is not None:
                outp = self.poslit_out_dir / poslit_path.name
                outp.write_text(json.dumps(poslit_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                poslit_out = str(outp)

            self._write_result_json(
                trial_id=trial_id,
                disease_path=disease_path,
                poslit_path=poslit_path,
                disease_out=disease_out,
                poslit_out=poslit_out,
                error=None,
            )

            return JointFilterResult(
                trial_id,
                str(disease_path) if disease_path else None,
                str(poslit_path) if poslit_path else None,
                disease_out,
                poslit_out,
                None,
                None,
            )

        poslit_items: List[Dict[str, Any]] = []
        canon_used: Optional[Path] = None

        if poslit_hits:
            canon_used = self._find_canon_file(trial_id, arm)
            if canon_used is None:
                poslit_items = [{"entity_variable_name": h, "template": "other"} for h in poslit_hits if isinstance(h, str)]
            else:
                poslit_items = self._build_poslit_items_from_hits(poslit_hits, canon_used)

        trial_text, prov = self._load_trial_text_with_provenance(trial_id)

        self._log_trial_bundle(
            trial_id=trial_id,
            trial_text=trial_text,
            trial_provenance=prov,
            disease_names=disease_names,
            poslit_hits=poslit_hits,
            arm=arm,
            canon_path=canon_used,
            poslit_items=poslit_items,
            disease_obj=disease_obj,
            poslit_obj=poslit_obj,
        )

        prompt = self._build_prompt(trial_text=trial_text, disease_names=disease_names, poslit_items=poslit_items)

        last_err: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            raw = self._call_llm(prompt, seed=seed + attempt - 1)
            last_raw = raw

            try:
                parsed_obj = _extract_json_object(raw)

                d_dec, p_dec, d_meta, p_meta = parse_and_verify_joint_filter(
                    raw,
                    disease_count=len(disease_names),
                    poslit_count=len(poslit_items),
                )

                self._log_attempt(
                    trial_id=trial_id,
                    attempt=attempt,
                    prompt=prompt,
                    raw=raw,
                    parsed_obj=parsed_obj,
                    decisions={
                        "disease_decisions": d_dec,
                        "poslit_decisions": p_dec,
                        "disease_meta": d_meta,
                        "poslit_meta": p_meta,
                    },
                    failed=False,
                )

                keep_disease = []
                for i, name in enumerate(disease_names):
                    if d_dec.get(i, "KEEP") == "KEEP":
                        keep_disease.append(name)

                keep_poslit_names = []
                for i, it in enumerate(poslit_items):
                    nm = it.get("entity_variable_name", "")
                    if not isinstance(nm, str) or not nm:
                        continue
                    if p_dec.get(i, "KEEP") == "KEEP":
                        keep_poslit_names.append(nm)

                disease_out = None
                poslit_out = None

                if disease_obj is not None and disease_path is not None:
                    fscbd = disease_obj.get("final_selected_concept_by_disease", {})
                    if isinstance(fscbd, dict):
                        for k in list(fscbd.keys()):
                            if k not in set(keep_disease):
                                fscbd.pop(k, None)
                    outp = self.disease_out_dir / disease_path.name
                    outp.write_text(json.dumps(disease_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                    disease_out = str(outp)

                if poslit_obj is not None and poslit_path is not None:
                    hits2 = [h for h in poslit_obj.get("hits", []) if isinstance(h, str)]
                    keep_set = set(keep_poslit_names)

                    def _stem(x: str) -> str:
                        return x.split("@@", 1)[0]

                    keep_stems = set(_stem(x) for x in keep_set)

                    new_hits = []
                    for h in hits2:
                        if h in keep_set or _stem(h) in keep_stems:
                            new_hits.append(h)

                    poslit_obj["hits"] = new_hits

                    outp = self.poslit_out_dir / poslit_path.name
                    outp.write_text(json.dumps(poslit_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                    poslit_out = str(outp)

                tdir = self._trial_dir(trial_id)
                final_dir = tdir / "final"
                final_dir.mkdir(parents=True, exist_ok=True)

                if disease_obj is not None:
                    (final_dir / "disease_filtered.json").write_text(
                        json.dumps(disease_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                if poslit_obj is not None:
                    (final_dir / "poslit_filtered.json").write_text(
                        json.dumps(poslit_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                self._write_result_json(
                    trial_id=trial_id,
                    disease_path=disease_path,
                    poslit_path=poslit_path,
                    disease_out=disease_out,
                    poslit_out=poslit_out,
                    error=None,
                )

                return JointFilterResult(
                    trial_id=trial_id,
                    disease_in=str(disease_path) if disease_path else None,
                    poslit_in=str(poslit_path) if poslit_path else None,
                    disease_out=disease_out,
                    poslit_out=poslit_out,
                    error=None,
                    raw_output=None,
                )

            except (ParsingError, VerificationError) as e:
                last_err = f"{type(e).__name__}: {e}"
                self._log_attempt(
                    trial_id=trial_id,
                    attempt=attempt,
                    prompt=prompt,
                    raw=raw,
                    note=last_err,
                    failed=True,
                )
                self.warnings.append({
                    "trial_id": trial_id,
                    "attempt": str(attempt),
                    "error": last_err,
                })

        self._write_result_json(
            trial_id=trial_id,
            disease_path=disease_path,
            poslit_path=poslit_path,
            disease_out=None,
            poslit_out=None,
            error=last_err,
            raw_output=last_raw,
        )

        return JointFilterResult(
            trial_id=trial_id,
            disease_in=str(disease_path) if disease_path else None,
            poslit_in=str(poslit_path) if poslit_path else None,
            disease_out=None,
            poslit_out=None,
            error=last_err,
            raw_output=last_raw,
        )

    def run_all(self, *, seed: int = 0, jobs: int = 1) -> List[JointFilterResult]:
        disease_files = sorted(self.disease_dir.glob("*.json")) if self.disease_dir.exists() else []
        poslit_files = sorted(self.poslit_dir.glob("*.json")) if self.poslit_dir.exists() else []

        disease_map: Dict[str, Path] = {}
        for p in disease_files:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tid = self._trial_id_from_disease_file(obj, p.stem)
            disease_map[tid] = p

        poslit_map: Dict[str, Path] = {}
        for p in poslit_files:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tid = self._trial_id_from_poslit_file(obj, p.stem)
            poslit_map[tid] = p

        all_trial_ids = sorted(set(disease_map.keys()) | set(poslit_map.keys()))
        if not all_trial_ids:
            return []

        results: List[JointFilterResult] = []

        if jobs <= 1:
            for i, tid in enumerate(all_trial_ids):
                try:
                    res = self.filter_one_trial(
                        trial_id=tid,
                        disease_path=disease_map.get(tid),
                        poslit_path=poslit_map.get(tid),
                        seed=seed + i * 1000,
                    )
                except Exception as e:
                    res = JointFilterResult(
                        trial_id=tid,
                        disease_in=str(disease_map.get(tid)) if disease_map.get(tid) else None,
                        poslit_in=str(poslit_map.get(tid)) if poslit_map.get(tid) else None,
                        disease_out=None,
                        poslit_out=None,
                        error=f"WorkerException: {repr(e)}",
                        raw_output=None,
                    )
                results.append(res)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = {}
                for i, tid in enumerate(all_trial_ids):
                    fut = ex.submit(
                        self.filter_one_trial,
                        trial_id=tid,
                        disease_path=disease_map.get(tid),
                        poslit_path=poslit_map.get(tid),
                        seed=seed + i * 1000,
                    )
                    futs[fut] = tid

                for fut in as_completed(futs):
                    tid = futs[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        results.append(
                            JointFilterResult(
                                trial_id=tid,
                                disease_in=str(disease_map.get(tid)) if disease_map.get(tid) else None,
                                poslit_in=str(poslit_map.get(tid)) if poslit_map.get(tid) else None,
                                disease_out=None,
                                poslit_out=None,
                                error=f"WorkerException: {repr(e)}",
                                raw_output=None,
                            )
                        )

        if self.warnings:
            out_csv = self.log_dir / "failed_attempts.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["trial_id", "attempt", "error"])
                w.writeheader()
                w.writerows(self.warnings)

        return results