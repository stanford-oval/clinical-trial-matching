## EntityCanonicalizer contains the prompt for entity canonicalization with the following workflow:

entity recognizer -> entity vector searcher -> entity linker -> entity checker -> entity arbiter


### Entity recognizer (LLMBasedMedicalEntityRecognizer.prompt): 

Raw text -> list of extracted entity strings

Detects and extract all potential relevant medical entities from ral trial eligibility text. It is optimized for high recall, capturing overlapping and nested mentions.
    

### Entity Vector Searcher: 

Entities -> list of candidate SNOMED concepts

Entity vector searcher does not rely on a prompt. Instead, it performs vector-based semantic search to match extracted entities against a medical knowledge base.

### Entity linker (LLMBasedMedicalEntityFilterLinker.prompt): 

Candidates -> one best concept per span

Entity linker connects each extracted entity span to a single, best-matching SNOMED CT concept. It ensures that all recognized surface strings are explicitly anchored to standardized terminology.

### Entity verifier (LLMBasedMedicalEntityFilterVerifier.prompt): 

Verified concepts -> remove redundancies, keep valid canonical forms

verifies each extracted span–concept pair produced by the Linker and decides whether the SNOMED candidate truly matches the intended meaning of the span in its criterion context

### Entity Arbiter (LLMBasedMedicalEntityFilterArbiter.prompt): 

spans with candidate terms -> final kept/discarded entities with canonical variable names

Entity arbiter checks each candidate entity against strict rules, generates a canonical variable name (for SMT program translation), and decides whether to keep or discard the entity
