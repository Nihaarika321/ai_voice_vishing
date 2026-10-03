from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(".")

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "mel_cnn_best.keras"
)

METADATA_PATH = (
    BASE_DIR /
    "data" /
    "metadata" /
    "phone_robustness_eval_metadata.csv"
)

TELEPHONE_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "phone_eval"
)

CODEC_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "phone_codec_eval"
)

RESULTS_DIR = (
    BASE_DIR /
    "data" /
    "results"
)

SUMMARY_PATH = (
    RESULTS_DIR /
    "phone_robustness_summary.csv"
)


# ============================================================
# MEL PARAMETERS
# ============================================================

TARGET_SR = 16000

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

# CNN requires 101 time frames
TARGET_FRAMES = 101


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading CNN:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nCNN model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("\nCNN loaded successfully.")

    print("\nCNN input shape:")
    print(model.input_shape)

    return model


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    print("\nLoading metadata...")

    print("\nMetadata path:")
    print(METADATA_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"\nMetadata file not found:\n{METADATA_PATH}"
        )

    df = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"\nMetadata rows: {len(df):,}"
    )

    print("\nMetadata columns:")
    print(list(df.columns))

    required = [
        "file",
        "label"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing metadata columns: {missing}"
        )

    print("\nLabels loaded:", len(df))

    print("\nMetadata label distribution:")
    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# FIND AUDIO FILE
# ============================================================

def find_audio_file(audio_dir, file_name):

    audio_dir = Path(audio_dir)

    # --------------------------------------------------------
    # Original metadata may contain:
    #
    # LA_T_1000137
    #
    # Generated audio:
    #
    # LA_T_1000137_phone.wav
    # --------------------------------------------------------

    stem = Path(
        str(file_name)
    ).stem

    candidates = [

        audio_dir / f"{stem}.wav",

        audio_dir / f"{stem}_phone.wav",

        audio_dir / f"{stem}_phone.flac",

        audio_dir / f"{stem}.flac"
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # --------------------------------------------------------
    # Fallback: search by stem
    # --------------------------------------------------------

    matches = list(
        audio_dir.glob(
            f"{stem}*.wav"
        )
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    return audio


# ============================================================
# CREATE MEL SPECTROGRAM
# ============================================================

def create_mel(audio):

    # --------------------------------------------------------
    # Ensure minimum length
    # --------------------------------------------------------

    required_samples = (
        WIN_LENGTH +
        HOP_LENGTH * (TARGET_FRAMES - 1)
    )

    if len(audio) < required_samples:

        audio = np.pad(
            audio,
            (
                0,
                required_samples - len(audio)
            ),
            mode="constant"
        )

    # --------------------------------------------------------
    # Create Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=TARGET_SR // 2,
        power=2.0
    )

    # --------------------------------------------------------
    # Convert to dB
    # --------------------------------------------------------

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Force exactly 101 frames
    # --------------------------------------------------------

    if mel.shape[1] < TARGET_FRAMES:

        mel = np.pad(
            mel,
            (
                (0, 0),
                (0, TARGET_FRAMES - mel.shape[1])
            ),
            mode="constant",
            constant_values=mel.min()
        )

    elif mel.shape[1] > TARGET_FRAMES:

        mel = mel[
            :,
            :TARGET_FRAMES
        ]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mel_min = mel.min()
    mel_max = mel.max()

    if mel_max > mel_min:

        mel = (
            mel - mel_min
        ) / (
            mel_max - mel_min
        )

    else:

        mel = np.zeros_like(
            mel
        )

    # --------------------------------------------------------
    # Add CNN channel dimension
    #
    # (64,101)
    #
    # becomes
    #
    # (64,101,1)
    # --------------------------------------------------------

    mel = mel.astype(
        np.float32
    )

    mel = np.expand_dims(
        mel,
        axis=-1
    )

    return mel


# ============================================================
# PROCESS DATASET
# ============================================================

def build_dataset(
    df,
    audio_dir,
    condition_name
):

    print("\n" + "=" * 60)
    print(
        f"EVALUATING: {condition_name}"
    )
    print("=" * 60)

    X = []
    y = []

    missing_files = 0
    failed_files = 0

    total = len(df)

    for i, row in df.iterrows():

        file_name = str(
            row["file"]
        )

        label = str(
            row["label"]
        ).strip().lower()

        # ----------------------------------------------------
        # Convert label
        # ----------------------------------------------------

        if label == "spoof":

            numeric_label = 1

        elif label == "bonafide":

            numeric_label = 0

        else:

            print(
                f"WARNING: Unknown label "
                f"{label} for {file_name}"
            )

            continue

        # ----------------------------------------------------
        # Find generated audio
        # ----------------------------------------------------

        audio_path = find_audio_file(
            audio_dir,
            file_name
        )

        if audio_path is None:

            missing_files += 1

            if missing_files <= 10:

                print(
                    f"WARNING: Audio not found: "
                    f"{file_name}"
                )

            continue

        try:

            # ------------------------------------------------
            # Load audio
            # ------------------------------------------------

            audio = load_audio(
                audio_path
            )

            # ------------------------------------------------
            # Generate Mel
            # ------------------------------------------------

            mel = create_mel(
                audio
            )

            # ------------------------------------------------
            # Verify shape
            # ------------------------------------------------

            if mel.shape != (
                N_MELS,
                TARGET_FRAMES,
                1
            ):

                print(
                    f"WARNING: Wrong Mel shape "
                    f"{mel.shape} for "
                    f"{file_name}"
                )

                failed_files += 1

                continue

            X.append(
                mel
            )

            y.append(
                numeric_label
            )

        except Exception as e:

            failed_files += 1

            if failed_files <= 10:

                print(
                    f"ERROR processing "
                    f"{file_name}: {e}"
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (i + 1) % 100 == 0
            or
            (i + 1) == total
        ):

            print(
                f"Processed "
                f"{i + 1}/{total}"
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    if len(X) == 0:

        print("\nNo valid samples.")

        return None, None

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    print(
        f"\nValid samples: {len(X):,}"
    )

    print(
        f"Missing files: {missing_files:,}"
    )

    print(
        f"Failed files: {failed_files:,}"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print("\nClass distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for cls, count in zip(
        unique,
        counts
    ):

        name = (
            "spoof"
            if cls == 1
            else "bonafide"
        )

        print(
            f"{name}: {count}"
        )

    return X, y


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    condition_name
):

    print("\nRunning CNN prediction...")

    probabilities = model.predict(
        X,
        batch_size=32,
        verbose=1
    )

    # --------------------------------------------------------
    # Handle output shape
    # --------------------------------------------------------

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    # --------------------------------------------------------
    # Probability is assumed to represent spoof
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        if len(np.unique(y)) == 2:

            roc_auc = roc_auc_score(
                y,
                probabilities
            )

        else:

            roc_auc = np.nan

    except Exception:

        roc_auc = np.nan

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(
        f"{condition_name} RESULTS"
    )
    print("-" * 60)

    print(
        f"Samples   : {len(y)}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    if np.isnan(roc_auc):

        print(
            "ROC-AUC   : N/A"
        )

    else:

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                "bonafide",
                "spoof"
            ],
            zero_division=0
        )
    )

    return {
        "condition": condition_name,
        "samples": len(y),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "PHONE / CODEC ROBUSTNESS EVALUATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    print("\nChecking paths...")

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )

    print(
        f"Telephone: {TELEPHONE_DIR}"
    )

    print(
        f"Codec: {CODEC_DIR}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Check CNN shape
    # --------------------------------------------------------

    expected_shape = (
        None,
        N_MELS,
        TARGET_FRAMES,
        1
    )

    actual_shape = model.input_shape

    if actual_shape != expected_shape:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected CNN input: "
            f"{expected_shape}"
        )

        print(
            f"Actual CNN input: "
            f"{actual_shape}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = load_metadata()

    # --------------------------------------------------------
    # Make results directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # ========================================================
    # TELEPHONE
    # ========================================================

    X_phone, y_phone = build_dataset(
        df,
        TELEPHONE_DIR,
        "Telephone"
    )

    if X_phone is not None:

        result = evaluate_model(
            model,
            X_phone,
            y_phone,
            "Telephone"
        )

        results.append(
            result
        )

    # ========================================================
    # TELEPHONE + G.711
    # ========================================================

    X_codec, y_codec = build_dataset(
        df,
        CODEC_DIR,
        "Telephone + G.711 A-law"
    )

    if X_codec is not None:

        result = evaluate_model(
            model,
            X_codec,
            y_codec,
            "Telephone + G.711 A-law"
        )

        results.append(
            result
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    if len(results) == 0:

        print(
            "\nNo results were generated."
        )

        print(
            "\nCheck that these directories contain "
            "the generated WAV files:"
        )

        print(
            TELEPHONE_DIR
        )

        print(
            CODEC_DIR
        )

        return

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        f"\nSummary saved:"
        f"\n{SUMMARY_PATH}"
    )

    print(
        f"\nDetailed results saved in:"
        f"\n{RESULTS_DIR}"
    )

    print("\n" + "=" * 60)
    print(
        "PHONE ROBUSTNESS EVALUATION COMPLETE"
    )
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()