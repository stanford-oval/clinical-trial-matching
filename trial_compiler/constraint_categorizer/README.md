Place build/ at project root directory

Required folders:
- project_root/build/canon_expanded
- project_root/build/disease
- project_root/build/positive_literal

# how to run
```
python categorize_disease_and_positive_literals.py --task positive_literal  # only categorize positive literal
python categorize_disease_and_positive_literals.py --task disease   # only categorize disease
python categorize_disease_and_positive_literals.py --task all   # categorize both positive literal and disease
```

# output
../build/positive_literals_categorized
../build/disease_categorzied

# Final mapped category names for positive literls
procedures:
- "Improves effectiveness of (procedure)",
- "Reduces procedure-related adverse effect (procedure)",
- "Other (procedure)"
- "Not of clinical interest"

findings:
- "Clinically address (finding)",
- "Prevention (finding)",
- "Other (finding)",
- "Not of clinical interest"

substance / product:
- "Reduce exposure/use (substance/product)",
- "Mitigates harms of exposure/use (substance/product)",
- "Enhances venefits of exposure/use (substance/product)",
- "Other (substance/product)",
- "Not of clinical interest"

other deterministically filtered literals: 
- "not relevant (numeric)",
- "not relevant (demographic)",

# Final mapped category names for disease list items:
- "Clinically Address",
- "Prevent",
- "Other",
- "Not of clinical interest"