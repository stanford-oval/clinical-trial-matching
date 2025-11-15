# control group identifier


control_group_identifier.py labels whether each enrollment subcohort in a clinical trial functions as a control group.

- For trials with multiple subcohorts, the script uses an LLM to determine which cohorts serve as control groups.
- For trials with a single subcohort, the cohort is automatically labeled as not a control group.



```
python control_group_identifier.py \
```

| Argument        | Description                                                     | Default                                   |
|-----------------|-----------------------------------------------------------------|-------------------------------------------|
| `--in-dir`      | Directory containing subcohort result JSON files                | `../../subcohort_results`                 |
| `--out-dir`     | Directory to write control-labeled JSON files                   | `../../subcohort_results_control_labeled` |
| `--log-dir`     | Directory for prompt, raw LLM output, and parsed cohort logs     | `mbench/subcohort_control`                |
| `--prompt-path` | Prompt template file (must contain `#ENROLLMENT_COHORTS#`)      | `prompts/control_group_identifier.prompt` |
| `--max-retries` | Maximum LLM retries if parsing or verification fails             | `3`                                       |



## OUTPUT
- A mirrored directory structure containing the same JSON files
- Each subcohort (i.e. items in "enrollment_cohorts") will include an added field:
```
"is_control": true | false
```