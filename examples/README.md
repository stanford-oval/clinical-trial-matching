# Examples

Runnable scripts, smallest first. Each says up front what it needs.

| | needs | what it shows |
|---|---|---|
| [`01_decide_a_pair.py`](01_decide_a_pair.py) | pair data | decide one patient–trial pair and read the verdict |
| [`02_read_the_artifacts.py`](02_read_the_artifacts.py) | `z3-solver` only | what the system assumed and what would change the answer |
| [`03_add_your_own_matcher.py`](03_add_your_own_matcher.py) | nothing | register a new matcher and compare it against the built-ins |

```bash
python examples/02_read_the_artifacts.py     # start here; no data needed
```
