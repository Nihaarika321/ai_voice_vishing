import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project root available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_loader import load_audio
from features.mfcc import extract_mfcc, mfcc_statistics


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "short_window_metadata.csv"
)


def extract_features_from_row(row):
    """
    Load one audio window using Member 1's audio loader
    and extract MFCC statistical features.
    """

    audio, sr = load_audio(
        file_name=row["file"],
        split=row["split"],
        condition="clean",
        duration=float(row["window_sec"]),
        start_sec=float(row["start_sec"])
    )

    mfcc = extract_mfcc(
        audio,
        sr
    )

    features = mfcc_statistics(mfcc)

    return features


if __name__ == "__main__":

    print("Testing real ASVspoof → MFCC pipeline...")
    print()

    # Read Member 1's metadata
    metadata = pd.read_csv(METADATA_PATH)

    print("Metadata loaded.")
    print("Total metadata rows:", len(metadata))
    print()

    # Take ONLY ONE row for the first real test
    row = metadata.iloc[0]

    print("Testing row:")
    print(row)
    print()

    features = extract_features_from_row(row)

    print("Real audio loaded successfully.")
    print("Feature vector shape:", features.shape)
    print()
    print("First 5 features:")
    print(features[:5])
    print()
    print("Real ASVspoof MFCC test complete.")