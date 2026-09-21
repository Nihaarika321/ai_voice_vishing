import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "short_window_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Make project root importable
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_loader import load_audio
from features.mfcc import extract_mfcc, mfcc_statistics


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# For our first baseline experiment:
# use 1-second CLEAN speech.

WINDOW_SEC = 1.0
CONDITION = "clean"

RANDOM_STATE = 42
TEST_MODE = False
TEST_SAMPLES_PER_CLASS = 20


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def process_row(row):
    """
    Load one audio window and extract
    26-dimensional MFCC statistical features.
    """

    audio, sr = load_audio(
        file_name=row["file"],
        split=row["split"],
        condition=CONDITION,
        duration=float(row["window_sec"]),
        start_sec=float(row["start_sec"])
    )

    mfcc = extract_mfcc(
        audio,
        sr
    )

    features = mfcc_statistics(mfcc)

    return features


# ---------------------------------------------------------
# Build balanced dataset
# ---------------------------------------------------------

def build_dataset(
    metadata,
    split,
    output_name
):

    print()
    print("=" * 60)
    print(f"Building {split.upper()} dataset")
    print("=" * 60)

    # Only use 1-second windows
    subset = metadata[
        (metadata["split"] == split)
        & (metadata["window_sec"] == WINDOW_SEC)
    ].copy()

    print("Available 1-second windows:", len(subset))

    # Separate classes
    bonafide = subset[
        subset["label"] == "bonafide"
    ]

    spoof = subset[
        subset["label"] == "spoof"
    ]

    print("Bonafide:", len(bonafide))
    print("Spoof   :", len(spoof))

    # Balance the classes
    n_samples = min(
        len(bonafide),
        len(spoof)
    )

    if TEST_MODE:
        n_samples = min(
            n_samples,
            TEST_SAMPLES_PER_CLASS
    )

    print("Samples per class:", n_samples)

    bonafide = bonafide.sample(
        n=n_samples,
        random_state=RANDOM_STATE
    )

    spoof = spoof.sample(
        n=n_samples,
        random_state=RANDOM_STATE
    )

    balanced = pd.concat(
        [bonafide, spoof],
        ignore_index=True
    )

    # Shuffle
    balanced = balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("Final dataset size:", len(balanced))

    # -----------------------------------------------------
    # Extract features
    # -----------------------------------------------------

    feature_rows = []

    metadata_rows = []

    for _, row in tqdm(
        balanced.iterrows(),
        total=len(balanced),
        desc=f"Extracting {split} MFCC"
    ):

        try:

            features = process_row(row)

            feature_rows.append(features)

            metadata_rows.append({
                "file": row["file"],
                "speaker_id": row["speaker_id"],
                "label": row["label"],
                "split": row["split"],
                "window_sec": row["window_sec"],
                "start_sec": row["start_sec"]
            })

        except Exception as e:

            print()
            print("ERROR processing:", row["file"])
            print("Error:", e)

    # -----------------------------------------------------
    # Convert to arrays
    # -----------------------------------------------------

    X = np.asarray(
        feature_rows,
        dtype=np.float32
    )

    metadata_output = pd.DataFrame(
        metadata_rows
    )

    print()
    print("Feature matrix shape:", X.shape)

    # -----------------------------------------------------
    # Create column names
    # -----------------------------------------------------

    columns = []

    for i in range(1, 14):
        columns.append(
            f"mfcc_{i}_mean"
        )

    for i in range(1, 14):
        columns.append(
            f"mfcc_{i}_std"
        )

    feature_df = pd.DataFrame(
        X,
        columns=columns
    )

    # Add labels and metadata
    feature_df.insert(
        0,
        "label",
        metadata_output["label"].values
    )

    feature_df.insert(
        0,
        "split",
        metadata_output["split"].values
    )

    feature_df.insert(
        0,
        "file",
        metadata_output["file"].values
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = OUTPUT_DIR / output_name

    feature_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("Saved:")
    print(output_path)

    print()
    print("Class distribution:")
    print(feature_df["label"].value_counts())

    return feature_df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Loading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        "Total metadata rows:",
        len(metadata)
    )

    # Build training dataset
    train_df = build_dataset(
        metadata,
        "train",
        "mfcc_1s_clean_train.csv"
    )

    # Build development dataset
    dev_df = build_dataset(
        metadata,
        "dev",
        "mfcc_1s_clean_dev.csv"
    )

    print()
    print("=" * 60)
    print("MFCC DATASET BUILD COMPLETE")
    print("=" * 60)