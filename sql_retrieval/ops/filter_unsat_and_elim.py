from pathlib import Path
import pandas as pd

INPUT_DIR = Path("eval_out/labeled_csv")
OUTPUT_DIR = Path("eval_out/filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for csv_path in INPUT_DIR.glob("sigir-*.csv"):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1) keep only 1s and 2s
    df = df[df["gt_label"].isin([1, 2])]

    # 2) keep only non-satisfied labels
    df = df[df["label"].isin(["unsatisfied_inclusion", "explicit_contradiction"])]

    stem = csv_path.stem  # e.g. "sigir-20145"

    # ---- unsatisfied_inclusion ----
    df_ui = df[df["label"] == "unsatisfied_inclusion"]
    df_ui_2 = df_ui[df_ui["gt_label"] == 2]
    df_ui_1 = df_ui[df_ui["gt_label"] == 1]

    if not df_ui_2.empty:
        df_ui_2.to_csv(OUTPUT_DIR / f"{stem}_unsat_inclusion_gt2.csv", index=False)
    if not df_ui_1.empty:
        df_ui_1.to_csv(OUTPUT_DIR / f"{stem}_unsat_inclusion_gt1.csv", index=False)

    # ---- explicit_contradiction ----
    df_ec = df[df["label"] == "explicit_contradiction"]
    df_ec_2 = df_ec[df_ec["gt_label"] == 2]
    df_ec_1 = df_ec[df_ec["gt_label"] == 1]

    if not df_ec_2.empty:
        df_ec_2.to_csv(OUTPUT_DIR / f"{stem}_explicit_contradiction_gt2.csv", index=False)
    if not df_ec_1.empty:
        df_ec_1.to_csv(OUTPUT_DIR / f"{stem}_explicit_contradiction_gt1.csv", index=False)
