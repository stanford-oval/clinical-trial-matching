---
name: Bug report
about: Something does not work, or does not match the paper
labels: bug
---

**What happened**
The command or code you ran, and what you got.

**What you expected**

**Does the checker pass?**
```
python scripts/check_invariants.py
```
Paste the last few lines. Checks that SKIP because local data is absent are
expected — see docs/DATA.md.

**Environment**
- Python version (`python -V`), OS
- Installed how (`pip install -e .`, extras used)
- `z3-solver` version if the solver is involved

**Already known?**
Please check the Known gaps table in the README first.
