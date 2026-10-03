"""
============================================================
ROBUST MEL-SPECTROGRAM CNN TRAINING
============================================================

AI Voice Vishing Detection

Purpose:
    Retrain the Mel CNN using:
        1. Clean speech
        2. Telephone-bandlimited speech
        3. Telephone + G.711 A-law speech

IMPORTANT:
    The robustness evaluation set is NOT used here.

Training data:
    Original ASVspoof2019 LA TRAIN split only.

Output:
    models/mel_cnn_robust_best.keras
    models/mel_cnn_robust_final.keras
"""

from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from scipy.signal import butter, sosfilt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.model_selection import train_test_split
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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "asvspoof2019_la_metadata.csv"
)

AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LA"
    / "LA"
    / "ASVspoof2019_LA_train"
    / "flac"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "mel_cnn_robust_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "mel_cnn_robust_final.keras"
)

DATA_CACHE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "robust_training_data.npz"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# AUDIO / MEL CONFIGURATION
# ============================================================

TARGET_SR = 16000

N_MELS = 64

N_FFT = 512

HOP_LENGTH = 160

N_FRAMES = 101

DURATION = 1.0

EXPECTED_SAMPLES = TARGET_SR


# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Number from EACH CLASS used for robustness augmentation.

SAMPLES_PER_CLASS = 1000

# Therefore:
#
# clean:
#       1000 bonafide
#       1000 spoof
#
# telephone:
#       1000 bonafide
#       1000 spoof
#
# telephone + G711:
#       1000 bonafide
#       1000 spoof
#
# Total = 6000 samples


# ============================================================
# TELEPHONE PARAMETERS
# ============================================================

PHONE_SR = 8000

LOW_FREQ = 300

HIGH_FREQ = 3400


# ============================================================
# G.711 A-LAW
# ============================================================

def alaw_encode(samples):

    samples = np.asarray(
        samples,
        dtype=np.int16
    )

    encoded = np.zeros(
        len(samples),
        dtype=np.uint8
    )

    for i, sample in enumerate(samples):

        sample = int(sample)

        sign = (
            0x80
            if sample < 0
            else 0x00
        )

        value = abs(sample)

        if value > 32635:
            value = 32635

        if value >= 256:

            exponent = 7
            mask = 0x4000

            while (
                exponent > 0
                and (value & mask) == 0
            ):

                exponent -= 1
                mask >>= 1

            mantissa = (
                value
                >> (exponent + 3)
            ) & 0x0F

            alaw = (
                sign
                | (exponent << 4)
                | mantissa
            )

        else:

            alaw = (
                sign
                | (value >> 4)
            )

        encoded[i] = alaw ^ 0x55

    return encoded


def alaw_decode(encoded):

    encoded = np.asarray(
        encoded,
        dtype=np.uint8
    )

    decoded = np.zeros(
        len(encoded),
        dtype=np.int16
    )

    for i, byte in enumerate(encoded):

        byte = int(byte)

        byte ^= 0x55

        sign = byte & 0x80

        exponent = (
            byte >> 4
        ) & 0x07

        mantissa = byte & 0x0F

        if exponent == 0:

            value = (
                mantissa << 4
            ) + 8

        else:

            value = (
                (mantissa << 4)
                + 0x108
            ) << (exponent - 1)

        if sign:
            value = -value

        decoded[i] = np.clip(
            value,
            -32768,
            32767
        )

    return decoded


# ============================================================
# TELEPHONE FILTER
# ============================================================

def telephone_filter(audio):

    sos = butter(
        6,
        [
            LOW_FREQ,
            HIGH_FREQ
        ],
        btype="bandpass",
        fs=TARGET_SR,
        output="sos"
    )

    return sosfilt(
        sos,
        audio
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_audio(audio):

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = (
            audio
            / peak
            * 0.95
        )

    return audio


# ============================================================
# FIX AUDIO LENGTH
# ============================================================

def fix_length(audio):

    if len(audio) < EXPECTED_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                EXPECTED_SAMPLES - len(audio)
            )
        )

    elif len(audio) > EXPECTED_SAMPLES:

        audio = audio[
            :EXPECTED_SAMPLES
        ]

    return audio.astype(
        np.float32
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, sr = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True
    )

    audio = fix_length(
        audio
    )

    return audio


# ============================================================
# TELEPHONE TRANSFORMATION
# ============================================================

def make_telephone(audio):

    # Telephone frequency limitation
    audio = telephone_filter(
        audio
    )

    # Downsample to telephone bandwidth
    audio = librosa.resample(
        audio,
        orig_sr=TARGET_SR,
        target_sr=PHONE_SR
    )

    # Restore to model sampling rate
    audio = librosa.resample(
        audio,
        orig_sr=PHONE_SR,
        target_sr=TARGET_SR
    )

    audio = fix_length(
        audio
    )

    audio = normalize_audio(
        audio
    )

    return audio


# ============================================================
# TELEPHONE + G.711
# ============================================================

def make_telephone_codec(audio):

    audio = make_telephone(
        audio
    )

    audio_int16 = np.int16(
        np.clip(
            audio,
            -1.0,
            1.0
        )
        * 32767
    )

    encoded = alaw_encode(
        audio_int16
    )

    decoded = alaw_decode(
        encoded
    )

    audio = (
        decoded.astype(
            np.float32
        )
        / 32767.0
    )

    audio = fix_length(
        audio
    )

    return audio


# ============================================================
# MEL EXTRACTION
# ============================================================

def extract_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Force exact dimensions
    if mel_db.shape[1] < N_FRAMES:

        mel_db = np.pad(
            mel_db,
            (
                (0, 0),
                (
                    0,
                    N_FRAMES
                    - mel_db.shape[1]
                )
            ),
            mode="constant",
            constant_values=-80
        )

    elif mel_db.shape[1] > N_FRAMES:

        mel_db = mel_db[
            :,
            :N_FRAMES
        ]

    # Same normalization as mel_cnn.py
    mel_db = (
        mel_db + 80.0
    ) / 80.0

    mel_db = np.clip(
        mel_db,
        0.0,
        1.0
    )

    return mel_db.astype(
        np.float32
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():

    print("=" * 60)
    print("BUILDING ROBUST TRAINING DATASET")
    print("=" * 60)

    print(
        f"\nMetadata:\n{METADATA_PATH}"
    )

    print(
        f"\nAudio directory:\n{AUDIO_DIR}"
    )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"\nMetadata not found:\n"
            f"{METADATA_PATH}"
        )

    if not AUDIO_DIR.exists():

        raise FileNotFoundError(
            f"\nTraining audio directory not found:\n"
            f"{AUDIO_DIR}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"\nTotal metadata rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # TRAIN ONLY
    # --------------------------------------------------------

    train_df = df[
        df["split"].str.lower()
        == "train"
    ].copy()

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    # --------------------------------------------------------
    # Balanced sampling
    # --------------------------------------------------------

    bonafide = train_df[
        train_df["label"].str.lower()
        == "bonafide"
    ]

    spoof = train_df[
        train_df["label"].str.lower()
        == "spoof"
    ]

    print(
        "\nAvailable:"
    )

    print(
        "Bonafide:",
        len(bonafide)
    )

    print(
        "Spoof:",
        len(spoof)
    )

    if (
        len(bonafide)
        < SAMPLES_PER_CLASS
        or len(spoof)
        < SAMPLES_PER_CLASS
    ):

        raise ValueError(
            "Not enough samples for requested "
            "balanced dataset."
        )

    bonafide = bonafide.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    spoof = spoof.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    selected = pd.concat(
        [
            bonafide,
            spoof
        ],
        ignore_index=True
    )

    selected = selected.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    print(
        "\nSelected:"
    )

    print(
        selected["label"].value_counts()
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    X = []

    y = []

    conditions = []

    failed = 0

    print(
        "\nGenerating Mel features..."
    )

    for index, row in selected.iterrows():

        file_id = str(
            row["file"]
        ).strip()

        label = (
            1
            if str(row["label"]).lower()
            == "spoof"
            else 0
        )

        audio_path = (
            AUDIO_DIR
            / f"{file_id}.flac"
        )

        try:

            if not audio_path.exists():

                print(
                    f"Missing: {audio_path}"
                )

                failed += 1
                continue

            clean_audio = load_audio(
                audio_path
            )

            phone_audio = make_telephone(
                clean_audio.copy()
            )

            codec_audio = make_telephone_codec(
                clean_audio.copy()
            )

            # ------------------------------------------------
            # Clean
            # ------------------------------------------------

            clean_mel = extract_mel(
                clean_audio
            )

            X.append(
                clean_mel
            )

            y.append(
                label
            )

            conditions.append(
                "clean"
            )

            # ------------------------------------------------
            # Telephone
            # ------------------------------------------------

            phone_mel = extract_mel(
                phone_audio
            )

            X.append(
                phone_mel
            )

            y.append(
                label
            )

            conditions.append(
                "telephone"
            )

            # ------------------------------------------------
            # Telephone + codec
            # ------------------------------------------------

            codec_mel = extract_mel(
                codec_audio
            )

            X.append(
                codec_mel
            )

            y.append(
                label
            )

            conditions.append(
                "telephone_codec"
            )

            processed = index + 1

            if processed % 100 == 0:

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(selected)}"
                )

        except Exception as e:

            print(
                f"\nERROR: {file_id}"
            )

            print(e)

            failed += 1

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    conditions = np.asarray(
        conditions
    )

    # CNN channel dimension
    X = X[..., np.newaxis]

    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET CREATED"
    )

    print(
        "=" * 60
    )

    print(
        "\nX shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Failed:",
        failed
    )

    print(
        "\nCondition distribution:"
    )

    print(
        pd.Series(
            conditions
        ).value_counts()
    )

    print(
        "\nClass distribution:"
    )

    print(
        pd.Series(
            y
        ).value_counts()
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    print(
        "\nSaving dataset:"
    )

    print(
        DATA_CACHE
    )

    np.savez_compressed(
        DATA_CACHE,
        X=X,
        y=y,
        conditions=conditions
    )

    print(
        "\nDataset cache saved."
    )

    return X, y, conditions


# ============================================================
# LOAD OR BUILD DATASET
# ============================================================

def get_dataset():

    if DATA_CACHE.exists():

        print(
            "\nExisting robustness training "
            "cache found."
        )

        print(
            DATA_CACHE
        )

        response = input(
            "\nUse existing cache? "
            "[Y/n]: "
        ).strip().lower()

        if response != "n":

            data = np.load(
                DATA_CACHE,
                allow_pickle=True
            )

            return (
                data["X"],
                data["y"],
                data["conditions"]
            )

    return build_dataset()


# ============================================================
# BUILD CNN
# ============================================================

def build_model():

    model = Sequential(
        [

            Input(
                shape=(
                    N_MELS,
                    N_FRAMES,
                    1
                )
            ),

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            BatchNormalization(),

            MaxPooling2D(
                (2, 2)
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            BatchNormalization(),

            MaxPooling2D(
                (2, 2)
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            BatchNormalization(),

            MaxPooling2D(
                (2, 2)
            ),

            Dropout(
                0.30
            ),

            Flatten(),

            Dense(
                128,
                activation="relu"
            ),

            Dropout(
                0.40
            ),

            Dense(
                1,
                activation="sigmoid"
            )
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print(
        "\n" + "=" * 60
    )

    print(
        "ROBUST MODEL VALIDATION"
    )

    print(
        "=" * 60
    )

    probabilities = (
        model.predict(
            X_test,
            batch_size=32,
            verbose=1
        )
        .ravel()
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        "\nAccuracy :",
        f"{accuracy:.4f}"
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall   :",
        f"{recall:.4f}"
    )

    print(
        "F1 Score :",
        f"{f1:.4f}"
    )

    print(
        "ROC-AUC  :",
        f"{auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "bonafide",
                "spoof"
            ],
            zero_division=0
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "ROBUST MEL CNN TRAINING"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    X, y, conditions = get_dataset()

    print(
        "\nFinal dataset:"
    )

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    # --------------------------------------------------------
    # Train / validation split
    #
    # Stratify by label + condition
    # --------------------------------------------------------

    stratify_labels = np.array(
        [
            f"{int(label)}_{condition}"
            for label, condition
            in zip(y, conditions)
        ]
    )

    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=stratify_labels
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "\nTraining labels:"
    )

    print(
        pd.Series(
            y_train
        ).value_counts()
    )

    print(
        "\nValidation labels:"
    )

    print(
        pd.Series(
            y_val
        ).value_counts()
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print(
        "\nBuilding robust CNN..."
    )

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING ROBUST TRAINING"
    )

    print(
        "=" * 60
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=30,
        batch_size=32,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        shuffle=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "ROBUST TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nFinal model:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print(
        "\nTraining dataset:"
    )

    print(
        DATA_CACHE
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Evaluate mel_cnn_robust_best.keras "
        "on the untouched phone robustness test set."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()