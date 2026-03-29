from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "training_data_202601.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "splits" / "group_split_seed42.json"

SEED = 42
TRAIN_ID_COUNT = 392
VAL_ID_COUNT = 85
TEST_ID_COUNT = 85


def main():
    df = pd.read_csv(DATA_PATH)

    unique_ids = sorted(df["unique_id"].unique().tolist())
    expected_total = TRAIN_ID_COUNT + VAL_ID_COUNT + TEST_ID_COUNT
    if len(unique_ids) != expected_total:
        raise ValueError(f"Expected {expected_total} unique ids, got {len(unique_ids)}")

    rng = np.random.default_rng(SEED)
    shuffled_ids = unique_ids.copy()
    rng.shuffle(shuffled_ids)

    train_ids = sorted(shuffled_ids[:TRAIN_ID_COUNT])
    val_ids = sorted(shuffled_ids[TRAIN_ID_COUNT:TRAIN_ID_COUNT + VAL_ID_COUNT])
    test_ids = sorted(shuffled_ids[TRAIN_ID_COUNT + VAL_ID_COUNT:])

    split_artifact = {
        "seed": SEED,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "label_order": [
            "The Persistence of Memory",
            "The Starry Night",
            "The Water Lily Pond",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_artifact, f, indent=2)

    verify_split(df, train_ids, val_ids, test_ids)


def verify_split(df, train_ids, val_ids, test_ids):
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print("overlap train/val:", len(train_set & val_set))
    print("overlap train/test:", len(train_set & test_set))
    print("overlap val/test:", len(val_set & test_set))

    print("num train ids:", len(train_ids))
    print("num val ids:", len(val_ids))
    print("num test ids:", len(test_ids))

    train_df = df[df["unique_id"].isin(train_ids)].copy()
    val_df = df[df["unique_id"].isin(val_ids)].copy()
    test_df = df[df["unique_id"].isin(test_ids)].copy()

    print("train rows:", len(train_df))
    print("val rows:", len(val_df))
    print("test rows:", len(test_df))

    print("\ntrain label counts:")
    print(train_df["Painting"].value_counts().sort_index())

    print("\nval label counts:")
    print(val_df["Painting"].value_counts().sort_index())

    print("\ntest label counts:")
    print(test_df["Painting"].value_counts().sort_index())


if __name__ == "__main__":
    main()
