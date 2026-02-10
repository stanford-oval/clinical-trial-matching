# Changelog

## 0.1.0 (initial release)

- Nine variants exposed: `trialgpt`, `lm_only`, `lm_only_prescreen`, `multiagent_nl`,
  `smt_raw`, `smt_atoms_arbiter`, `smt_lm_evidence_arbiter`, `hybrid_loose`,
  `hybrid_strict`.
- All variants share `Decision` / `AuditStep` schema.
- Cached LM outputs from sibling experiment directories ensure paper numbers reproduce exactly.
- CLI: `python -m matchers.inspect_one`.
- Examples in `examples/`.
