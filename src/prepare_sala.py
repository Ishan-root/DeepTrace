# src/prepare_sala.py
#
# Sorts Sala's dataset into:
# data/raw_sala_sorted/train/REAL
# data/raw_sala_sorted/train/FAKE
# data/raw_sala_sorted/test/REAL
# data/raw_sala_sorted/test/FAKE
#
# label 0 = REAL
# label 1 = FAKE

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


# ---- PATHS ----
SOURCE_DIR = "data/raw_sala_source/archive (2)"
DEST_DIR = "data/raw_sala_sorted"
# ----------------

TEST_FRACTION = 0.1

LABEL_TO_FOLDER = {
    0: "REAL",
    1: "FAKE"
}


def main():

    # Find Sala's CSV
    csv_path = os.path.join(SOURCE_DIR, "train.csv")

    # Read CSV
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from train.csv")
    print("\nLabel counts:")
    print(df["label"].value_counts())

    # Split 90% train / 10% test
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_FRACTION,
        stratify=df["label"],
        random_state=42
    )

    print(f"\nTrain: {len(train_df)}")
    print(f"Test:  {len(test_df)}")

    # Copy images into the correct folders
    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df)
    ]:

        print(f"\nProcessing {split_name} split...")

        copied = 0
        missing = 0

        for _, row in split_df.iterrows():

            label = int(row["label"])
            folder = LABEL_TO_FOLDER[label]

            # CSV already contains train_data/ in file_name
            src_path = os.path.join(
                SOURCE_DIR,
                row["file_name"]
            )

            # Destination folder
            dest_folder = os.path.join(
                DEST_DIR,
                split_name,
                folder
            )

            os.makedirs(dest_folder, exist_ok=True)

            # Keep original filename
            filename = os.path.basename(row["file_name"])

            dest_path = os.path.join(
                dest_folder,
                filename
            )

            # Check that source image exists
            if not os.path.exists(src_path):
                print(f"WARNING: missing file, skipping: {src_path}")
                missing += 1
                continue

            # Copy image
            shutil.copy2(src_path, dest_path)
            copied += 1

        print(f"Copied: {copied}")
        print(f"Missing: {missing}")

    print("\n====================================")
    print("DONE")
    print("Sorted dataset:", DEST_DIR)
    print("====================================")


if __name__ == "__main__":
    main()