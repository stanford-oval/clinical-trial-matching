#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv

from .parser.positive_literal_categorization_parser import (
    parse_and_verify_positive_constraint_literals,
    ParsingError,
    VerificationError,
)

# Category *names* depend on variable template-kind.
# The LLM always outputs 'A'..'E'; we only change the human meaning per template.
CATEGORY_NAME_MAP: Dict[str, Dict[str, str]] = {
    # template == "procedures"
    "procedures": {
        "A": "Improves effectiveness of (procedure)",
        "B": "Reduces procedure-related adverse effect (procedure)",
        "C": "Other (procedure)",
        "D": "Not of clinical interest",
    },

    # template == "findings"
    "findings": {
        "A": "Clinically address (finding)",
        "B": "Prevention (finding)",
        "C": "Other (finding)",
        "D": "Not of clinical interest",
    },

    # all other templates (meds, observables, etc.)
    "other": {
        "A": "Reduce exposure/use (substance/product)",
        "B": "Mitigates harms of exposure/use (substance/product)",
        "C": "Enhances benefits of exposure/use (substance/product)",
        "D": "Other (substance/product)",
        "E": "Not of clinical interest",
    },
}


class PositiveLiteralCategorizer:
    def __init__(
        self,
        engine,
        *,
        build_root: Path = Path("../../build"),
        positive_literal_subdir: str = "positive_constraint_literals/per_file",
        canon_expanded_subdir: str = "canon",
        disease_subdir: str = "disease",
        context_folder: Path = Path("../../canonical_subcohort_results"),
        corpus_jsonl: Path = Path("../../dataset/clinical_trial/sigir/corpus.jsonl"),
        # default prompt for non-finding / non-procedure templates
        substance_prompt_path: Optional[Path] = None,
        findings_prompt_path: Optional[Path] = None,
        procedures_prompt_path: Optional[Path] = None,
        default_temp: float = 0.0,
        default_top_p: float = 1.0,
        max_retries: int = 3,
        log_dir: Path = Path("mbench"),
    ) -> None:
        """
        engine: initialized AzureInferenceEngine (or compatible)
        build_root: root build dir, e.g. ../../build
        positive_literal_subdir: relative to build_root (for per_file positives)
        canon_expanded_subdir: relative to build_root
        disease_subdir: relative to build_root, holds <trial_id>_disease_link_filter_summary.json

        prompt_path: default / "other" prompt (non-finding, non-procedure)
        findings_prompt_path: prompt for template == "findings"
        procedures_prompt_path: prompt for template == "procedures"
        """

        self.engine = engine
        self.build_root = build_root

        self.positive_dir = build_root / positive_literal_subdir
        self.output_dir = build_root / "positive_constraint_literals_categorized" / "per_file"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.canon_dir = build_root / canon_expanded_subdir
        self.disease_dir = build_root / disease_subdir

        self.prompt_path_default = substance_prompt_path or Path(
            "prompts/PositiveLiteralSubstanceCategorization.prompt"
        )
        self.prompt_path_findings = findings_prompt_path or Path(
            "prompts/PositiveLiteralFindingCategorization.prompt"
        )
        self.prompt_path_procedures = procedures_prompt_path or Path(
            "prompts/PositiveLiteralProcedureCategorization.prompt"
        )
        self.context_folder = context_folder
        self.corpus_jsonl = corpus_jsonl
        self.default_temp = default_temp
        self.default_top_p = default_top_p
        self.max_retries = max_retries

        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.warnings: List[Dict[str, str]] = []

        # Root for positive-literal categorization logs
        self.log_root = self.log_dir / "positive_literal_categorized"
        self.log_root.mkdir(parents=True, exist_ok=True)

        # Ensure prompts exist
        for p in [
            self.prompt_path_default,
            self.prompt_path_findings,
            self.prompt_path_procedures,
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Prompt file not found: {p}")

        # Load prompt templates once
        self._prompt_template_default = self._load_prompt_template(self.prompt_path_default)
        self._prompt_template_findings = self._load_prompt_template(self.prompt_path_findings)
        self._prompt_template_procedures = self._load_prompt_template(self.prompt_path_procedures)

    # ───────────────────────────────────────
    # Prompt handling
    # ───────────────────────────────────────
    @staticmethod
    def _load_prompt_template(path: Path) -> str:
        with path.open("r", encoding="utf-8") as f:
            return f.read()

    def _ensure_corpus_index(self) -> None:
        """Build a dict {_id: text} from corpus.jsonl once."""
        if self._corpus_text_by_id is not None:
            return

        idx: Dict[str, str] = {}
        path = self.corpus_jsonl
        if not path.exists():
            print(f"[WARN] corpus.jsonl not found: {path}. Corpus fallback disabled.")
            self._corpus_text_by_id = {}
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                for ln, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        print(f"[WARN] corpus.jsonl invalid JSON at line {ln}; skipping.")
                        continue
                    _id = obj.get("_id")
                    text = obj.get("text")
                    if isinstance(_id, str) and _id:
                        if isinstance(text, str):
                            idx[_id] = text
                        else:
                            # Keep empty string if missing; still register id
                            idx.setdefault(_id, "")
        except Exception as e:
            print(f"[WARN] Failed reading corpus.jsonl {path}: {e}")
            idx = {}

        self._corpus_text_by_id = idx


    def _load_trial_text_from_corpus(self, trial_id: str) -> str:
        """Fallback: look up trial_id or base_id in corpus.jsonl and return its 'text'."""
        if not trial_id:
            return ""

        self._ensure_corpus_index()
        assert self._corpus_text_by_id is not None

        base_id = re.sub(r"[a-zA-Z]$", "", trial_id)

        # Try exact id first, then base_id
        for key in (trial_id, base_id):
            if key in self._corpus_text_by_id:
                txt = self._corpus_text_by_id.get(key, "")
                if txt:
                    return txt
                # If present but empty, still return empty (caller can decide)
                return txt

        print(
            f"[WARN] corpus fallback: no entry found for trial_id={trial_id!r} "
            f"(also tried base_id={base_id!r}) in {self.corpus_jsonl}"
        )
        return ""

    def _load_trial_text(self, trial_id: str) -> str:
        """
        Load trial context from context_folder/base_id.json.

        If trial_id has no subcohort suffix (trial_id == base_id):
            load from corpus

        Else:
            try to find cohort_dict in extracted["enrollment_cohorts"] with
            cohort_dict["trial_id_effective"] == trial_id, then build cohort-specific context.

        If ANY error occurs for subcohort handling, fall back to shared_context.
        """
        if not trial_id:
            print("[WARN] Empty trial_id passed to _load_trial_text; returning empty text.")
            return ""

        base_id = re.sub(r"[a-zA-Z]$", "", trial_id)
        context_path = self.context_folder / f"{base_id}.json"

        def _corpus_fallback(reason: str) -> str:
            print(
                f"[WARN] Context load failed for trial_id={trial_id!r} (base_id={base_id!r}): "
                f"{reason}. Falling back to corpus text."
            )
            return self._load_trial_text_from_corpus(trial_id)

        if not context_path.exists():
            return _corpus_fallback(f"context file not found: {context_path}")

        try:
            with context_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return _corpus_fallback(f"failed to load context file: {e}")

        try:
            extracted = data["extracted"]["preprocessor_normalized"]
        except Exception as e:
            return _corpus_fallback(f"missing extracted.preprocessor_normalized: {e}")

        # Helper: shared_context (with corpus fallback if missing/empty)
        def _shared_or_corpus(reason: str) -> str:
            shared = extracted.get("shared_context", "")
            if isinstance(shared, str) and shared.strip():
                print(f"[WARN] Using shared_context fallback for {trial_id!r}: {reason}")
                return shared
            return _corpus_fallback(f"{reason}; shared_context missing/empty")

        # No subcohort
        if base_id == trial_id:
            shared = extracted.get("shared_context", "")
            if isinstance(shared, str) and shared.strip():
                return shared
            return _corpus_fallback("no-cohort trial: shared_context missing/empty")

        # Subcohort path
        try:
            cohorts = extracted.get("enrollment_cohorts", [])
            if not isinstance(cohorts, list):
                return _shared_or_corpus("enrollment_cohorts is not a list")

            cohort_dict = None
            for c in cohorts:
                if isinstance(c, dict) and c.get("trial_id_effective") == trial_id:
                    cohort_dict = c
                    break

            if cohort_dict is None:
                avail = [
                    c.get("trial_id_effective")
                    for c in cohorts
                    if isinstance(c, dict) and "trial_id_effective" in c
                ]
                return _shared_or_corpus(f"no matching cohort for trial_id_effective; available={avail}")

            def _get_str(key: str) -> str:
                v = cohort_dict.get(key, "")
                return v if isinstance(v, str) else ""

            label = _get_str("label")
            contextual_text = _get_str("contextual_text")
            inc = _get_str("inclusion_criteria")
            exc = _get_str("exclusion_criteria")

            # If cohort-specific context is basically empty, fall back too
            cohort_context = (
                f"{label}\n"
                f"{contextual_text}\n"
                f"inclusion criteria:\n{inc}\n"
                f"exclusion criteria:\n{exc}"
            ).strip()

            if cohort_context:
                return cohort_context

            return _shared_or_corpus("cohort_dict found but cohort context empty")

        except Exception as e:
            return _shared_or_corpus(f"exception in subcohort handling: {e}")


    def _select_prompt_template(self, template_kind: str) -> str:
        """
        Choose which prompt template string to use based on template_kind:
          - 'findings'   -> findings prompt
          - 'procedures' -> procedures prompt
          - otherwise    -> default prompt
        """
        if template_kind == "findings":
            return self._prompt_template_findings
        elif template_kind == "procedures":
            return self._prompt_template_procedures
        else:
            return self._prompt_template_default

    def _build_prompt(
        self,
        trial_text: str,
        pos_literal_items: List[Dict[str, Any]],
        *,
        template_kind: str,
    ) -> str:
        """
        Insert the clinical trial description and canonicalized positive literal list
        into the appropriate prompt (based on template_kind).
        """
        tmpl = self._select_prompt_template(template_kind)
        block = json.dumps(pos_literal_items, ensure_ascii=False, indent=2)
        prompt = tmpl.replace("#POSITIVE_LITERAL_LIST#", block)
        prompt = prompt.replace("#CLINICAL_TRIAL_DESCRIPTION#", trial_text or "")
        return prompt

    def _log_prompt(
        self,
        json_path: Path,
        prompt: str,
        *,
        suffix: str = "",
        template_kind: Optional[str] = None,
    ) -> None:
        subdir = self._get_log_subdir(template_kind)
        out_name = f"{json_path.name}{suffix}_prompt.txt"
        out_path = subdir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            f.write(prompt)

    def _log_raw_output(
        self,
        json_path: Path,
        raw_text: str,
        *,
        suffix: str = "",
        template_kind: Optional[str] = None,
    ) -> None:
        subdir = self._get_log_subdir(template_kind)
        out_name = f"{json_path.name}{suffix}_raw.txt"
        out_path = subdir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            f.write(raw_text)


    def _get_log_subdir(self, template_kind: Optional[str]) -> Path:
        """
        Return a subdirectory under mbench/positive_literal_categorized
        based on template_kind:

          findings   -> mbench/positive_literal_categorized/findings
          procedures -> mbench/positive_literal_categorized/procedures
          everything else -> mbench/positive_literal_categorized/others
        """
        if template_kind == "findings":
            bucket = "findings"
        elif template_kind == "procedures":
            bucket = "procedures"
        else:
            bucket = "others"

        subdir = self.log_root / bucket
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir


    # ───────────────────────────────────────
    # Engine call
    # ───────────────────────────────────────
    def _complete_with_decoding(self, prompt: str, *, seed: int, temp: float, top_p: float) -> str:
        try:
            return self.engine(
                prompt,
                temperature=temp,
                top_p=top_p,
                seed=seed,
            )[0]
        except TypeError:
            try:
                if hasattr(self.engine, "kwargs") and isinstance(self.engine.kwargs, dict):
                    self.engine.kwargs["temperature"] = temp
                    self.engine.kwargs["top_p"] = top_p
                return self.engine(prompt)[0]
            except Exception:
                return self.engine(prompt)[0]

    def _call_llm(self, prompt: str, *, seed: int = 0) -> str:
        return self._complete_with_decoding(
            prompt,
            seed=seed,
            temp=self.default_temp,
            top_p=self.default_top_p,
        )

    # ───────────────────────────────────────
    # Canonical variable extraction
    # ───────────────────────────────────────
    def _find_canon_file(self, trial_id: str, arm: str) -> Optional[Path]:
        pattern = f"{trial_id}_{arm}*.json"
        candidates = list(self.canon_dir.glob(pattern))
        if not candidates:
            fallback = list(self.canon_dir.glob(f"{trial_id}*.json"))
            if not fallback:
                return None
            if len(fallback) > 1:
                print(f"[WARN] Multiple canon files for trial {trial_id}, using first: {fallback[0]}")
            return fallback[0]
        if len(candidates) > 1:
            print(f"[WARN] Multiple canon files for trial {trial_id}, arm {arm}, using first: {candidates[0]}")
        return candidates[0]

    @staticmethod
    def _load_canonical_variables(canon_path: Path) -> List[Dict[str, Any]]:
        with canon_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "canonical_variables" in data:
            cv = data["canonical_variables"]
        else:
            cv = data

        if not isinstance(cv, list):
            raise ValueError(f"Canonical file {canon_path} does not contain a list of canonical variables.")
        return cv

    @staticmethod
    def _strip_qualifiers(cv_item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in cv_item.items()
            if k not in (
                "qualifier_predicates",
                "qualifier_predicates_detailed",
                "qualifier_predicates_for_semantics_not_already_captured_with_stem",
                "index_in_canonical_forms",
                "inc_exc",
                "__reuse_existing_symbol",
                "__do_not_update_variable_meaning",
                "requirement_id",
            )
        }

    def _build_positive_literal_items(
    self,
        hits: List[str],
        canon_path: Path,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        canonical_vars = self._load_canonical_variables(canon_path)

        index_exact: Dict[str, List[Dict[str, Any]]] = {}
        index_stem: Dict[str, List[Dict[str, Any]]] = {}

        for item in canonical_vars:
            name = item.get("entity_variable_name")
            if not isinstance(name, str) or not name:
                continue
            index_exact.setdefault(name, []).append(item)
            stem = name.split("@@", 1)[0]
            index_stem.setdefault(stem, []).append(item)

        collected: List[Dict[str, Any]] = []
        missing: List[str] = []

        for idx, h in enumerate(hits):
            if not isinstance(h, str) or not h:
                # treat invalid hit as missing and skip
                msg = f"[WARN] Invalid hit at idx={idx} not found in {self.canon_dir}; defaulting to 'not of clinical interest'"
                print(msg)
                missing.append(str(h))
                continue

            chosen: Optional[Dict[str, Any]] = None

            # 1) exact
            cands = index_exact.get(h)
            if cands:
                chosen = cands[0]
            else:
                # 2) stem match
                h_stem = h.split("@@", 1)[0]
                stem_cands = index_stem.get(h_stem)
                if stem_cands:
                    chosen = stem_cands[0]
                    if len(stem_cands) > 1:
                        print(
                            f"[WARN] Hit stem {h_stem!r} matched {len(stem_cands)} canonical vars in {canon_path.name}; "
                            f"using first: {stem_cands[0].get('entity_variable_name')!r}"
                        )

            if chosen is None:
                print(
                    f"[WARN] variable {h!r} not found in {self.canon_dir} (canon file {canon_path.name}); "
                    f"set to 'not of clinical interest'"
                )
                missing.append(h)
                continue

            cv_item = self._strip_qualifiers(chosen)

            # preserve your index field naming (use the one your prompt/parser expect)
            new_item: Dict[str, Any] = {}
            inserted = False
            for k, v in cv_item.items():
                new_item[k] = v
                if k == "entity_variable_name":
                    new_item["entity_variable_idx"] = str(idx)
                    inserted = True
            if not inserted:
                new_item["entity_variable_idx"] = str(idx)

            collected.append(new_item)

        return collected, missing

    # ───────────────────────────────────────
    # Apply categories into representatives
    # ───────────────────────────────────────
    @staticmethod
    def _apply_categories_to_representatives(
        data: Dict[str, Any],
        entity_to_category: Dict[str, List[str]],
        entity_to_template_kind: Dict[str, str],
    ) -> None:
        """
        For each representative, write:
        - category: List[str], human-readable names corresponding to *all* category letters
            returned by the LLM for this representative.

        Category is determined by the representative's 'rep' field.
        Multiple representatives may share the same 'rep' and therefore the same categories.
        """
        reps = data.get("representatives", [])
        if not isinstance(reps, list):
            return

        for item in reps:
            if not isinstance(item, dict):
                continue
            rep_name = item.get("rep")
            if not isinstance(rep_name, str):
                continue

            # Does this rep correspond to a categorized entity?
            if rep_name not in entity_to_category:
                continue

            letter = entity_to_category[rep_name]  # NOW: List[str]
            template_kind = entity_to_template_kind.get(rep_name, "other")

            # Special case: observable_entities_numeric are always "not relevant (numeric)"
            if template_kind == "observable_entities_numeric":
                item["category"] = ["not relevant (numeric)"]
                continue

            # Special case: demographics
            if template_kind == "demographic":
                item["category"] = ["not relevant (demographic)"]
                continue

            # Normal case: map each letter -> per-template text
            per_template = CATEGORY_NAME_MAP.get(template_kind, CATEGORY_NAME_MAP["other"])

            # Defensive: allow legacy single-string letter without renaming variables
            if isinstance(letter, str):
                letters = [letter]
            else:
                letters = letter

            category_texts: List[str] = []
            for L in letters:
                if not isinstance(L, str):
                    continue
                L = L.strip().upper()
                category_texts.append(per_template.get(L, CATEGORY_NAME_MAP["other"]["E"]))

            # Ensure non-empty output for robustness
            if not category_texts:
                category_texts = [CATEGORY_NAME_MAP["other"]["E"]]

            item["category"] = category_texts


    def _log_failed_attempt(self, *, trial_id: str, json_path: Path,
                        template_kind: str, attempt: int, error: Exception):
        self.warnings.append({
            "trial_id": trial_id,
            "json_path": str(json_path),
            "template_kind": template_kind,
            "attempt": attempt,
            "error": str(error),
        })

    # ───────────────────────────────────────
    # Per-file processing with parse+verify
    # ───────────────────────────────────────
    def categorize_file(self, json_path: Path, *, seed: int = 0) -> Dict[str, Any]:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        trial_id = data.get("trial_id", "")
        arm = (data.get("arm", "") or "").lower()

        # Skip & copy exclusion files
        if arm == "exclusion":
            #print(f"[INFO] Skipping exclusion file (but copying output): {json_path.name}")
            out_path = self.output_dir / json_path.name
            with out_path.open("w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)
            return {
                "trial_id": trial_id,
                "json_path": str(json_path),
                "output_path": str(out_path),
                "error": "Skipped exclusion file",
                "raw_output": None,
            }

        hits = data.get("hits", [])

        if not hits:
            msg = f"No hits in {json_path.name}, copying input to output."
            print(f"[INFO] {msg}")
            out_path = self.output_dir / json_path.name
            with out_path.open("w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)
            return {
                "trial_id": trial_id,
                "json_path": str(json_path),
                "output_path": str(out_path),
                "error": msg,
                "raw_output": None,
            }

        canon_path = self._find_canon_file(trial_id, arm)
        if canon_path is None:
            msg = f"No canon-expansion file found for trial {trial_id}, arm {arm}."
            print(f"[WARN] {msg}")
            return {
                "trial_id": trial_id,
                "json_path": str(json_path),
                "error": msg,
                "raw_output": None,
            }

        #pos_literal_items = self._build_positive_literal_items(hits, canon_path)
        pos_literal_items, missing_hits = self._build_positive_literal_items(hits, canon_path)
        if not pos_literal_items:
            msg = f"No canonical items matched hits for {json_path.name}, copying input to output."
            print(f"[WARN] {msg}")
            out_path = self.output_dir / json_path.name
            with out_path.open("w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)
            return {
                "trial_id": trial_id,
                "json_path": str(json_path),
                "output_path": str(out_path),
                "error": msg,
                "raw_output": None,
            }

        trial_text = self._load_trial_text(trial_id)

        # Group by template and build maps:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        entity_to_template_kind: Dict[str, str] = {}
        entity_to_category: Dict[str, List[str]] = {}

        for item in pos_literal_items:
            tmpl = item.get("template", "other")
            if not isinstance(tmpl, str):
                tmpl = "other"

            evn = item.get("entity_variable_name")

            # Special case: observable_entities_numeric
            if tmpl == "observable_entities_numeric":
                if isinstance(evn, str):
                    entity_to_template_kind[evn] = "observable_entities_numeric"
                    entity_to_category[evn] = ["E"]  # keep as list
                continue

            # Special case: templates starting with "patient" are demographic variables
            if tmpl.startswith("patient"):
                template_kind = "demographic"
                if isinstance(evn, str):
                    entity_to_template_kind[evn] = template_kind
                    entity_to_category[evn] = ["E"]  # keep as list
                continue

            if tmpl == "findings":
                template_kind = "findings"
            elif tmpl == "procedures":
                template_kind = "procedures"
            else:
                template_kind = "other"

            groups.setdefault(template_kind, []).append(item)

            if isinstance(evn, str):
                entity_to_template_kind[evn] = template_kind

        # Default any missing hits to "not of clinical interest" (E)
        # These will be applied in _apply_categories_to_representatives by matching rep names.
        for h in missing_hits:
            if isinstance(h, str) and h:
                entity_to_template_kind[h] = "other"
                entity_to_category[h] = ["E"]

        last_error: Optional[str] = None
        raw_outputs: Dict[str, str] = {}

        # One LLM call per template-kind group
        for template_kind, items in groups.items():
            items_for_llm: List[Dict[str, Any]] = []
            expected_entities: List[str] = []

            for idx, it in enumerate(items):
                evn = it.get("entity_variable_name")
                if not isinstance(evn, str):
                    continue

                expected_entities.append(evn)

                it_llm = dict(it)
                it_llm["entity_variable_idx"] = str(idx)
                items_for_llm.append(it_llm)

            if not items_for_llm:
                continue

            prompt = self._build_prompt(trial_text, items_for_llm, template_kind=template_kind)
            suffix = f"_{template_kind}"
            self._log_prompt(
                json_path,
                prompt,
                suffix=suffix,
                template_kind=template_kind,
            )

            group_success = False
            raw_output: Optional[str] = None

            for attempt in range(1, self.max_retries + 1):
                raw_output = self._call_llm(prompt, seed=seed + attempt - 1)
                try:
                    # Parser returns idx_str -> List[str] (letters)
                    idx_to_letter = parse_and_verify_positive_constraint_literals(
                        raw_output,
                        expected_count=len(items_for_llm),
                        template_kind = template_kind,
                    )

                    # Map index -> entity_variable_name -> List[str]
                    group_mapping: Dict[str, List[str]] = {}
                    for idx_str, letter in idx_to_letter.items():
                        idx_int = int(idx_str)
                        entity_name = expected_entities[idx_int]

                        # Defensive: if parser returns a single string in older runs
                        if isinstance(letter, str):
                            group_mapping[entity_name] = [letter]
                        else:
                            group_mapping[entity_name] = letter

                    entity_to_category.update(group_mapping)
                    group_success = True
                    break

                except (ParsingError, VerificationError) as e:
                    last_error = f"{type(e).__name__} (template={template_kind}): {e}"
                    print(
                        f"[WARN] Parse/verify failed for {json_path.name} "
                        f"(template={template_kind}) on attempt "
                        f"{attempt}/{self.max_retries}: {e}"
                    )
                    self._log_failed_attempt(
                        trial_id=trial_id,
                        json_path=json_path,
                        template_kind=template_kind,
                        attempt=attempt,
                        error=e,
                    )

            if raw_output is not None:
                raw_outputs[template_kind] = raw_output
                self._log_raw_output(
                    json_path,
                    raw_output,
                    suffix=suffix,
                    template_kind=template_kind,
                )

            if not group_success:
                return {
                    "trial_id": trial_id,
                    "json_path": str(json_path),
                    "error": last_error,
                    "raw_output": "\n\n".join(
                        f"=== {k} ===\n{v}" for k, v in raw_outputs.items()
                    ) if raw_outputs else None,
                }

        if not entity_to_category:
            msg = "No categories produced for any template group."
            print(f"[WARN] {msg}")
            return {
                "trial_id": trial_id,
                "json_path": str(json_path),
                "error": msg,
                "raw_output": "\n\n".join(
                    f"=== {k} ===\n{v}" for k, v in raw_outputs.items()
                ) if raw_outputs else None,
            }

        # Apply categories using template-aware mapping
        self._apply_categories_to_representatives(
            data,
            entity_to_category=entity_to_category,
            entity_to_template_kind=entity_to_template_kind,
        )

        out_path = self.output_dir / json_path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        #print(f"[OK] {json_path.name} -> {out_path}")
        return {
            "trial_id": trial_id,
            "json_path": str(json_path),
            "output_path": str(out_path),
            "error": None,
            "raw_output": "\n\n".join(
                f"=== {k} ===\n{v}" for k, v in raw_outputs.items()
            ) if raw_outputs else None,
        }
    
    
    def run_all(self, *, seed: int = 0) -> List[Dict[str, Any]]:
        if not self.positive_dir.exists():
            raise FileNotFoundError(f"Positive literal dir not found: {self.positive_dir}")
        if not self.canon_dir.exists():
            raise FileNotFoundError(f"Canon-expanded dir not found: {self.canon_dir}")

        json_files = sorted(self.positive_dir.glob("*.json"))
        if not json_files:
            print(f"[INFO] No json files found in {self.positive_dir}")
            return []

        results: List[Dict[str, Any]] = []
        for i, json_path in enumerate(json_files, start=1):
            print(f"[INFO] ({i}/{len(json_files)}) Processing {json_path.name}")
            result = self.categorize_file(json_path, seed=seed)
            results.append(result)

        if self.warnings:
            summary_path = self.log_dir / "positive_literal_failed_attempts.csv"
            fieldnames = ["trial_id", "json_path", "template_kind", "attempt", "error"]
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.warnings)
            print(f"[INFO] Wrote failed-attempt summary to {summary_path}")

        return results