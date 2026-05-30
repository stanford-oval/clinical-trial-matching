# Counterfactual self-faithfulness architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   matchers/systems/         canonical eligibility matchers      │
│   ────────────────                                              │
│     aegis/             VERDICT: Z3 + atom mining                │
│     shahlab/           Koopman per-criterion ternary verdict    │
│     single_shot_llm/   LLM-only: single-prompt LM               │
│     trialgpt/          TrialGPT per-criterion                   │
│                                                                 │
│   matchers/configs/         (matcher × prompt × backbone) recipes│
│   ─────────────────                                             │
│     aegis.default.yaml            VERDICT, default              │
│     single_shot_llm.v5.yaml         V5_PROMPT,             4.1  │
│     single_shot_llm.v5_gpt5.yaml    V5_PROMPT,             5    │
│     single_shot_llm.v5_blockers.yaml  V5_TWO_STEP_BLOCKERS,4.1  │
│     shahlab.koopman.yaml                                        │
│     trialgpt.default.yaml                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │ matcher output
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│   rationale_generators/     extract BLOCKER-ONLY rationale      │
│   ─────────────────────                                         │
│     aegis_maxsat/      Z3 Optimize → minimum-flip atom targets  │
│     shahlab_blockers/  shah JSON → [not met] rows (both sides)  │
│     trialgpt_blockers/ TG JSON → [not included]+[excluded] rows │
│     v5_explanation/    pass-through explanation field (1-3 sent)│
│     v5_blockers_list/  V5_TWO_STEP_BLOCKERS → blockers[] array  │
│                                                                 │
│   Output: RationaleSpec = {                                     │
│     'kind':         'atom_targets' | 'rationale_text',          │
│     'atom_targets': [...]   (when kind='atom_targets')          │
│     'rationale':    '...'   (when kind='rationale_text')        │
│     'supports':     '...'   facts to preserve in the CF chart   │
│     'source':       '<system>.<config>' for traceability        │
│   }                                                             │
└─────────────────────────────────────────────────────────────────┘
                            │ RationaleSpec
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│   counterfactual_modifier/  rewrite chart to flip the verdict   │
│   ────────────────────────                                      │
│     atom_target_modifier.py   gpt-5 typed-atom editor (aegis)   │
│     rationale_modifier.py     gpt-5 free-text editor (others)   │
│     prompts/                                                    │
│                                                                 │
│   Output: {cf_chart, modifier, source_rationale, source}        │
└─────────────────────────────────────────────────────────────────┘
                            │ cf_chart
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│   matchers/systems/         RE-judge the cf_chart               │
│   ────────────────                                              │
│   (same matcher + same config that produced the original verdict)│
│                                                                 │
│   self-flip rate = (rejudge == eligible | cf_valid)             │
└─────────────────────────────────────────────────────────────────┘
```

Each stage has a uniform interface; the experiment driver chains them
agnostic of which system is running.

See `experiments/counterfactual/05_self_faithfulness/REPRODUCE.md` for the
end-to-end rerun commands and `TRACEABILITY_AUDIT.md` for the list of bugs
caught + fixed during development.
