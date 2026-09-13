# Demo pair

One synthetic patient and one invented trial, so the tool can be tried without
an evaluation corpus, an LLM endpoint or a compiled trial program.

**Nothing here is real.** `demo-patient-01` is not a person, `NCT00000000` is
not a trial, and neither is drawn from SIGIR, TREC or any other collection.
The numbers are chosen to make one point, not to represent clinical practice.

## The point it makes

The note reports age, performance status and histology, and shows no brain
metastases. It never mentions renal function. The trial requires eGFR >= 60.

- **`--system lm-only`** answers ELIGIBLE. Silence about a lab, in a short
  prescreen note, usually means it was unremarkable.
- **`--system smt-only`** answers INELIGIBLE. A required condition is not
  established, and the solver will not supply it.

Neither is simply wrong. What the tool adds is that the disagreement is
*locatable*: one unresolved condition, named, with the requirement it would
have to satisfy.

```bash
export VERDICT_PAIR_DATA=data/demo/pairs
verdict list
verdict match   demo-patient-01__NCT00000000 --system smt-only
verdict match   demo-patient-01__NCT00000000 --system lm-only
verdict explain demo-patient-01__NCT00000000 --system smt-only
python examples/05_the_demo_pair.py
```
