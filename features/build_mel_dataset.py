"""
Build Mel Spectrogram Dataset
=============================

Project:
    AI Voice Vishing Detection

Purpose:
    Build a balanced 1-second clean-speech Mel-spectrogram
    dataset using the same metadata and sampling strategy
    as the existing MFCC dataset.

Input:
    data/metadata/short_window_metadata.csv

Output:
    data/features/mel_1s_clean_train.csv
    data/features/mel_1s_clean_dev.csv

Each audio sample produces:

    64 Mel bands x 101 time frames
    =
    6464 Mel features
"""


# ============================================================
# IMPORTS
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


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


# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from preprocessing.audio_loader import (
    load_audio
)

from features.mel_spectrogram import (
    extract_mel_from_audio
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SEC = 1.0

CONDITION = "clean"

RANDOM_STATE = 42


# ============================================================
# TEST MODE
# ============================================================
#
# FIRST RUN:
#
#     TEST_MODE = True
#
# This processes only:
#
#     20 bonafide
#     20 spoof
#
# for train and dev.
#
# AFTER THE TEST WORKS:
#
#     TEST_MODE = False
#
# Then the full dataset will be created.
# ============================================================

TEST_MODE = False

TEST_SAMPLES_PER_CLASS = 20


# ============================================================
# MEL CONFIGURATION
# ============================================================

N_MELS = 64

N_FFT = 512

HOP_LENGTH = 160

WIN_LENGTH = 400

EXPECTED_MEL_SHAPE = (
    64,
    101
)


# ============================================================
# PROCESS ONE AUDIO ROW
# ============================================================

def process_row(row):
    """
    Load one audio window and extract
    its 64 x 101 log-Mel spectrogram.

    The spectrogram is flattened into
    a 6464-dimensional feature vector.
    """

    # --------------------------------------------------------
    # Load audio using the existing project loader
    # --------------------------------------------------------

    audio, sr = load_audio(
        file_name=row["file"],
        split=row["split"],
        condition=CONDITION,
        duration=float(
            row["window_sec"]
        ),
        start_sec=float(
            row["start_sec"]
        )
    )

    # --------------------------------------------------------
    # Extract Mel spectrogram
    # --------------------------------------------------------

    mel = extract_mel_from_audio(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if mel.shape != EXPECTED_MEL_SHAPE:

        raise ValueError(
            f"Unexpected Mel shape: "
            f"{mel.shape}. "
            f"Expected: "
            f"{EXPECTED_MEL_SHAPE}"
        )

    # --------------------------------------------------------
    # Flatten Mel spectrogram
    # --------------------------------------------------------

    features = mel.flatten()

    # --------------------------------------------------------
    # Check feature length
    # --------------------------------------------------------

    if len(features) != 6464:

        raise ValueError(
            f"Unexpected feature length: "
            f"{len(features)}. "
            f"Expected: 6464"
        )

    return features


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    metadata,
    split,
    output_name
):
    """
    Build a balanced Mel dataset for one split.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Complete metadata.

    split : str
        train or dev.

    output_name : str
        Output CSV filename.
    """

    print()

    print("=" * 60)

    print(
        f"Building {split.upper()} MEL dataset"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Select only 1-second windows
    # --------------------------------------------------------

    subset = metadata[
        (metadata["split"] == split)
        &
        (metadata["window_sec"] == WINDOW_SEC)
    ].copy()

    print()

    print(
        "Available 1-second windows:",
        len(subset)
    )

    # --------------------------------------------------------
    # Separate classes
    # --------------------------------------------------------

    bonafide = subset[
        subset["label"] == "bonafide"
    ]

    spoof = subset[
        subset["label"] == "spoof"
    ]

    print(
        "Bonafide:",
        len(bonafide)
    )

    print(
        "Spoof   :",
        len(spoof)
    )

    # --------------------------------------------------------
    # Balance the classes
    # --------------------------------------------------------

    n_samples = min(
        len(bonafide),
        len(spoof)
    )

    if TEST_MODE:

        n_samples = min(
            n_samples,
            TEST_SAMPLES_PER_CLASS
        )

    print()

    print(
        "Samples per class:",
        n_samples
    )

    # --------------------------------------------------------
    # Randomly select bonafide samples
    # --------------------------------------------------------

    bonafide = bonafide.sample(
        n=n_samples,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Randomly select spoof samples
    # --------------------------------------------------------

    spoof = spoof.sample(
        n=n_samples,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Combine the two classes
    # --------------------------------------------------------

    balanced = pd.concat(
        [
            bonafide,
            spoof
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    balanced = balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    print(
        "Final dataset size:",
        len(balanced)
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature_rows = []

    metadata_rows = []

    successful = 0

    failed = 0

    print()

    for _, row in tqdm(
        balanced.iterrows(),
        total=len(balanced),
        desc=f"Extracting {split} Mel"
    ):

        try:

            features = process_row(
                row
            )

            feature_rows.append(
                features
            )

            metadata_rows.append(
                {
                    "file": row["file"],
                    "speaker_id":
                        row["speaker_id"],
                    "label":
                        row["label"],
                    "split":
                        row["split"],
                    "window_sec":
                        row["window_sec"],
                    "start_sec":
                        row["start_sec"]
                }
            )

            successful += 1

        except Exception as e:

            failed += 1

            print()

            print(
                "ERROR processing:",
                row["file"]
            )

            print(
                "Error:",
                e
            )

    # --------------------------------------------------------
    # Extraction summary
    # --------------------------------------------------------

    print()

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    # --------------------------------------------------------
    # Make sure something was extracted
    # --------------------------------------------------------

    if len(feature_rows) == 0:

        raise RuntimeError(
            "No Mel features were extracted."
        )

    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    X = np.asarray(
        feature_rows,
        dtype=np.float32
    )

    metadata_output = pd.DataFrame(
        metadata_rows
    )

    print()

    print(
        "Feature matrix shape:",
        X.shape
    )

    # --------------------------------------------------------
    # Create column names
    # --------------------------------------------------------

    columns = []

    for mel_index in range(
        N_MELS
    ):

        for frame_index in range(
            EXPECTED_MEL_SHAPE[1]
        ):

            columns.append(
                f"mel_{mel_index + 1}"
                f"_frame_{frame_index + 1}"
            )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        X,
        columns=columns
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    feature_df.insert(
        0,
        "label",
        metadata_output[
            "label"
        ].values
    )

    feature_df.insert(
        0,
        "split",
        metadata_output[
            "split"
        ].values
    )

    feature_df.insert(
        0,
        "file",
        metadata_output[
            "file"
        ].values
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        output_name
    )

    feature_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        "Saved:"
    )

    print(
        output_path
    )

    print()

    print(
        "Dataset shape:",
        feature_df.shape
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        feature_df[
            "label"
        ].value_counts()
    )

    return feature_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Loading metadata..."
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print()

    print(
        "Total metadata rows:",
        len(metadata)
    )

    # --------------------------------------------------------
    # Build TRAIN dataset
    # --------------------------------------------------------

    train_df = build_dataset(
        metadata,
        "train",
        "mel_1s_clean_train.csv"
    )

    # --------------------------------------------------------
    # Build DEV dataset
    # --------------------------------------------------------

    dev_df = build_dataset(
        metadata,
        "dev",
        "mel_1s_clean_dev.csv"
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "MEL DATASET BUILD COMPLETE"
    )

    print("=" * 60)