#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from .parser.categorization_parser import (
    parse_and_verify,
    ParsingError,
    VerificationError,
)

LETTER_TO_CATEGORY = {
    "A": "Clinically address",
    "B": "Prevent",
    "C": "Other",
    "D": "Not of clinical interest",
}


class DiseaseCategorizer:
    def __init__(
        self,
        engine,
        *,
        build_root: Path,
        input_folder_name: str = "disease",
        prompt_path: Path = Path("prompts/DiseaseCategorization.prompt"),
        default_temp: float = 0.0,
        default_top_p: float = 1.0,
        max_retries: int = 3,
        log_dir: Path = Path("mbench/disease_categorization"),
    ) -> None:
        """
        engine: an initialized AzureInferenceEngine (or compatible)
        build_root: root build dir, e.g. Path("../../build")
        input_folder_name: name of the input folder under build_root (e.g. "disease")
        """
        self.engine = engine
        self.build_root = build_root

        self.disease_dir = build_root / input_folder_name
        self.output_dir = build_root / f"{input_folder_name}_categorized"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_path = prompt_path
        self.default_temp = default_temp
        self.default_top_p = default_top_p
        self.max_retries = max_retries

        self._prompt_template = self._load_prompt_template(self.prompt_path)

    # ───────────────────────────────────────
    # Prompt handling
    # ───────────────────────────────────────
    @staticmethod
    def _load_prompt_template(path: Path) -> str:
        with path.open("r", encoding="utf-8") as f:
            return f.read()

    # @staticmethod
    # def _format_disease_list(disease_names: List[str]) -> str:
    #     """
    #     How diseases are inserted into #DISEASE#.
    #     Adjust format if your prompt expects JSON / bullets / etc.
    #     """
    #     return "\n".join(disease_names)

    @staticmethod
    def _format_disease_list(disease_names: List[str]) -> str:
        """
        Convert disease list into a dictionary with stringified indices:
        {
            "0": "Disease A",
            "1": "Disease B",
            ...
        }
        And serialize it into JSON for insertion into prompt.
        """
        disease_dict = {str(i): name for i, name in enumerate(disease_names)}
        return json.dumps(disease_dict, ensure_ascii=False, indent=2)


    def _build_prompt(self, trial_text: str, diseases: List[str]) -> str:
        disease_block = self._format_disease_list(diseases)
        prompt = self._prompt_template.replace("#CLINICAL_TRIAL_DESCRIPTION#", trial_text)
        prompt = prompt.replace("#DISEASE#", disease_block)
        return prompt

    # ───────────────────────────────────────
    # Engine call (decoding wrapper)
    # ───────────────────────────────────────
    def _complete_with_decoding(self, prompt: str, *, seed: int, temp: float, top_p: float) -> str:
        """
        Tries to pass per-call temperature/top_p and a seed.
        If the engine ignores 'seed', jitter still provides diversity.
        """
        try:
            return self.engine(
                prompt,
                temperature=temp,
                top_p=top_p,
                seed=seed,
            )[0]
        except TypeError:
            # Fallback: set defaults on the engine then call without kwargs
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
    # JSON extraction + categorization
    # ───────────────────────────────────────
    @staticmethod
    def _extract_trial_text_and_diseases(data: Dict[str, Any]) -> Tuple[str, List[str]]:
        trial_text = data["contextual"]["text"]

        fscbd = data.get("final_selected_concept_by_disease", {})
        if not isinstance(fscbd, dict) or not fscbd:
            raise ValueError("Missing or empty final_selected_concept_by_disease")

        disease_names = list(fscbd.keys())
        return trial_text, disease_names
    
    # NEW -------------------------------------------------------------
    def _log_raw_output(
        self,
        trial_id: str,
        contextual_text: str,
        diseases: List[str],
        raw_output: str
    ):
        """
        Save JSON log with:
        - trial_id
        - input text
        - input diseases list
        - raw LLM output
        """
        log_path = self.log_dir / f"{trial_id}.llm.json"

        log_obj = {
            "trial_id": trial_id,
            "input": {
                "contextual_text": contextual_text,
                "diseases": diseases,
            },
            "raw_output": raw_output,
        }

        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log_obj, f, ensure_ascii=False, indent=2)


    def _log_raw_output_txt(self, trial_id: str, content: str):
        """
        Save raw LLM text into log_dir/<trial_id>.raw.txt
        """
        log_path = self.log_dir / f"{trial_id}.raw.txt"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(content)
    # ----------------------------------------------------------------


    def _apply_categories_to_data(
        self,
        data: Dict[str, Any],
        disease_to_letter: Dict[str, List[str]],
    ) -> None:
        """
        Mutates `data` in-place, adding 'category' under each disease in
        final_selected_concept_by_disease, mapping letter(s) -> human label list.
        """
        fscbd = data.get("final_selected_concept_by_disease", {})
        for disease, letter in disease_to_letter.items():
            # Defensive: support legacy single-string letter without renaming variables
            if isinstance(letter, str):
                letters = [letter]
            else:
                letters = letter

            mapped_list: List[str] = []
            for L in letters:
                mapped = LETTER_TO_CATEGORY.get(L)
                if mapped is None:
                    mapped = "others"
                mapped_list.append(mapped)

            if disease in fscbd and isinstance(fscbd[disease], dict):
                fscbd[disease]["category"] = mapped_list

    def categorize_file(self, json_path: Path, *, seed: int = 0) -> Dict[str, Any]:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        trial_id = data.get("trial_id", json_path.stem)

        try:
            trial_text, disease_names = self._extract_trial_text_and_diseases(data)
        except Exception as e:
            print(
                f"[WARN] {json_path} ({e}); copying original file to {self.output_dir}"
            )
            out_path = self.output_dir / json_path.name
            with out_path.open("w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)

            return {
                "trial_id": trial_id,
                "json_path": str(json_path),
                "output_path": str(out_path),
                "error": str(e),
                "raw_output": None,
            }

        batch_size = 5
        disease_batches: List[List[str]] = [
            disease_names[i : i + batch_size]
            for i in range(0, len(disease_names), batch_size)
        ]

        all_disease_to_letter: Dict[str, List[str]] = {}
        last_error: Optional[str] = None
        last_raw_output: Optional[str] = None

        for batch_idx, disease_batch in enumerate(disease_batches):
            prompt = self._build_prompt(trial_text, disease_batch)

            batch_disease_to_letter: Optional[Dict[str, List[str]]] = None

            for attempt in range(1, self.max_retries + 1):
                llm_seed = seed + batch_idx * self.max_retries + (attempt - 1)

                raw_output = self._call_llm(prompt, seed=llm_seed)
                last_raw_output = raw_output

                log_trial_id = f"{trial_id}_b{batch_idx}_a{attempt}"
                self._log_raw_output(
                    trial_id=log_trial_id,
                    contextual_text=trial_text,
                    diseases=disease_batch,
                    raw_output=raw_output,
                )
                self._log_raw_output_txt(log_trial_id, raw_output)

                try:
                    idx_to_letter = parse_and_verify(raw_output, disease_batch)

                    batch_disease_to_letter = {}
                    for idx_str, letter in idx_to_letter.items():
                        idx_int = int(idx_str)
                        if not (0 <= idx_int < len(disease_batch)):
                            raise VerificationError(
                                f"Index {idx_int} out of range for disease_batch of size {len(disease_batch)}."
                            )

                        disease_name = disease_batch[idx_int]

                        # Defensive: if parser returns legacy single-string letter
                        if isinstance(letter, str):
                            batch_disease_to_letter[disease_name] = [letter]
                        else:
                            batch_disease_to_letter[disease_name] = letter

                    break

                except (ParsingError, VerificationError) as e:
                    last_error = f"{type(e).__name__}: {e}"
                    print(
                        f"[WARN] Parse/verify failed for {trial_id} "
                        f"(batch {batch_idx}, attempt {attempt}/{self.max_retries}): {e}"
                    )
                    batch_disease_to_letter = None

            if batch_disease_to_letter is None:
                return {
                    "trial_id": trial_id,
                    "json_path": str(json_path),
                    "error": last_error,
                    "raw_output": last_raw_output,
                }

            all_disease_to_letter.update(batch_disease_to_letter)

        self._apply_categories_to_data(data, all_disease_to_letter)

        out_path = self.output_dir / json_path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {
            "trial_id": trial_id,
            "json_path": str(json_path),
            "output_path": str(out_path),
            "error": None,
            "raw_output": last_raw_output,
        }



    def run_all(self, *, seed: int = 0) -> List[Dict[str, Any]]:
        """
        Iterate over all json files in self.disease_dir and categorize them.
        """
        if not self.disease_dir.exists():
            raise FileNotFoundError(f"Disease dir not found: {self.disease_dir}")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        json_files = sorted(self.disease_dir.glob("*.json"))
        if not json_files:
            print(f"[INFO] No json files found in {self.disease_dir}")
            return []

        results: List[Dict[str, Any]] = []
        for json_path in json_files:
            result = self.categorize_file(json_path, seed=seed)
            results.append(result)

            if result.get("error"):
                print(
                    f"[ERROR] {result['trial_id']} failed: {result['error']} "
                    f"(file: {result['json_path']})"
                )
            else:
                print(f"[OK] {result['trial_id']} -> {result['output_path']}")

        return results