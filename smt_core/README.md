# smt_core/ -- Shared Infrastructure

Shared infrastructure layer used by all SatIR components. Centralizes utilities that were previously duplicated across 6+ source directories.

## Modules

### Inference Engine
- **`inference_engine.py`** -- Robust Azure AI Inference chat completions wrapper with exponential full-jitter backoff, Retry-After honoring, timeout ramp-up per attempt, optional fallback endpoint, and profiling integration. Used for GPT-4o and GPT-4.1 models.
- **`inference_engine_5.py`** -- Variant for GPT-5 models.
- **`engine_factory.py`** -- Centralized engine selection based on `OPENAI_ENDPOINT` / `OPENAI_MODEL` env vars. Replaces the duplicated if/elif blocks that were present in every entry point script.

### Persistence & Parsing
- **`checkpoint_io.py`** -- Stage-level checkpoint save/load for resumable pipelines. Supports PREPROC, CANON, ATTR, PROGRAM, FINAL stages.
- **`parse_functions.py`** -- LLM output parsers: extract requirements from `<requirements>` tags, parse SMT-LIB from `<smtfragment>` blocks, parse refiner outputs, etc.
- **`profiling.py`** -- Wall-clock profiling to JSONL for per-stage and per-LLM-call timing.

### DSPy & Signatures
- **`signatures.py`** -- DSPy signature classes for LLM interfaces (IdentifyMissingInfoSignature, DraftSmTSignature, RefineSmTSignature, etc.).
- **`helpers_entities.py`** -- Entity annotation loader (loads pre-annotated entity JSONL for enrichment).

### SNOMED-CT & Ontology
- **`snomed/snowstorm.py`** -- Lightweight wrapper around the Snowstorm SNOMED-CT terminology server API. Functions: `search(term)`, `ancestors(concept_id)`, `canonical(concept_id)`.

### Utilities
- **`utils/z3_helpers.py`** -- Z3 SMT solver introspection: `_collect_leaf_vars()` (extract leaf variables from Z3 AST), `_whole_program()` (rebuild full SMT-LIB program from lines), `_log()` (timestamped stage logging), `_print_variable_index()`.
- **`utils/text_utils.py`** -- Text formatting: `dict_to_readable_string()` for rendering trial/patient metadata as human-readable text for LLM prompts.

### Shared Modules
- **`modules/entity_canonicalizer/`** -- Base EntityCanonicalizer used by both trial and patient compilers. Performs entity linking against the medical ontology (SNOMED-CT) via:
  1. `LLMBasedMedicalEntityRecognizer` -- NER for medical entities
  2. `VectorEmbeddingConceptSearch` -- Dense vector search against Elasticsearch SNOMED index
  3. `LLMBasedMedicalEntityFilter` -- LLM-based validation/filtering of candidates
  4. `UMLSClient` -- UMLS concept definition lookup

- **`modules/attribute_extractor/`** -- Shared attribute extraction stages (qualifier identification, attribute translation, canonical value search/filter/verification).

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |
| `OPENAI_API_KEY` | Yes | Azure API key |
| `OPENAI_MODEL` | Optional | Explicit model override (gpt-4o, gpt-4.1, gpt-5) |
| `SNOWSTORM_BASE` | Optional | Snowstorm URL (default: http://localhost:8080) |
| `SNOWSTORM_BRANCH` | Optional | SNOMED branch (default: MAIN) |
