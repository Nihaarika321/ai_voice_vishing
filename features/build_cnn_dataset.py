from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "short_window_metadata.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_loader import load_audio
from features.mfcc import extract_mfcc


WINDOW_SEC = 1.0
CONDITION = "clean"
RANDOM_STATE = 42

# Same number of samples already used by your classical ML baseline
TRAIN_SAMPLES_PER_CLASS = 2580
DEV_SAMPLES_PER_CLASS = 2548


def select_balanced_samples(metadata, split, n_samples):
    """Select the same balanced population used by the baseline."""

    subset = metadata[
        (metadata["split"] == split)
        & (metadata["window_sec"] == WINDOW_SEC)
    ].copy()

    bonafide = subset[subset["label"] == "bonafide"]
    spoof = subset[subset["label"] == "spoof"]

    if len(bonafide) < n_samples or len(spoof) < n_samples:
        raise ValueError(
            f"Not enough samples for {split}: "
            f"bonafide={len(bonafide)}, spoof={len(spoof)}"
        )

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

    balanced = balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return balanced


def build_cnn_dataset(metadata, split, n_samples, output_name):

    print()
    print("=" * 60)
    print(f"Building CNN {split.upper()} dataset")
    print("=" * 60)

    selected = select_balanced_samples(
        metadata,
        split,
        n_samples
    )

    print("Selected samples:", len(selected))
    print(selected["label"].value_counts())

    X = []
    y = []
    metadata_rows = []

    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=f"Extracting CNN MFCC {split}"
    ):

        try:

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

            # Expected shape: (13, 101)
            if mfcc.shape != (13, 101):
                print(
                    f"Skipping {row['file']} "
                    f"because MFCC shape is {mfcc.shape}"
                )
                continue

            X.append(mfcc.astype(np.float32))

            y.append(
                1 if row["label"] == "spoof" else 0
            )

            metadata_rows.append({
                "file": row["file"],
                "speaker_id": row["speaker_id"],
                "label": row["label"],
                "split": row["split"]
            })

        except Exception as e:

            print()
            print("ERROR:", row["file"])
            print("Error:", e)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    metadata_output = pd.DataFrame(metadata_rows)

    print()
    print("CNN feature matrix:", X.shape)
    print("Labels:", y.shape)

    print()
    print("Class distribution:")
    print(metadata_output["label"].value_counts())

    output_path = OUTPUT_DIR / output_name

    np.savez_compressed(
        output_path,
        X=X,
        y=y
    )

    print()
    print("Saved:")
    print(output_path)

    return X, y


if __name__ == "__main__":

    print("Loading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        "Total metadata rows:",
        len(metadata)
    )

    # Same population sizes as the classical ML datasets
    build_cnn_dataset(
        metadata,
        "train",
        TRAIN_SAMPLES_PER_CLASS,
        "cnn_mfcc_1s_clean_train.npz"
    )

    build_cnn_dataset(
        metadata,
        "dev",
        DEV_SAMPLES_PER_CLASS,
        "cnn_mfcc_1s_clean_dev.npz"
    )

    print()
    print("=" * 60)
    print("CNN MFCC DATASET BUILD COMPLETE")
    print("=" * 60)