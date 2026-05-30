---
name: Reproduction problem
about: A number in the paper does not come out
labels: reproduction
---

**Which table or number**

**What you got instead**

**How you ran it**
Command, and where your data came from (docs/DATA.md).

**Checker output**
```
python scripts/check_invariants.py
```
The invariants assert the paper's published values, so a failure there points
at the same problem.

Some rows are known not to reproduce from this repository — see Known gaps in
the README before filing.
