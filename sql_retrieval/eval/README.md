# how to run
python run_all_judges.py \
    --pairs-file sampling_from_trialgpt/token_count_50_pairs.jsonl \    # run random 50 patient-trial pairs
    --num-workers 16 \
    --relevance-prompt ccr \    # choose from [cc (chief complaint only), ccr (chief complaint related), all (all factors)]
    --output-dir test_output