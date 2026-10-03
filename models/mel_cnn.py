"""
Mel-Spectrogram CNN
===================

Project:
    AI Voice Vishing Detection

Purpose:
    Train a CNN using 1-second log-Mel spectrograms
    to classify speech as:

        bonafide
        spoof

Input:
    data/features/mel_1s_clean_train.csv
    data/features/mel_1s_clean_dev.csv

Mel representation:
    64 Mel bands
    101 time frames

Input to CNN:
    64 x 101 x 1
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import LabelEncoder

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


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mel_1s_clean_train.csv"
)

DEV_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mel_1s_clean_dev.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "mel_cnn_best.keras"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# MEL DIMENSIONS
# ============================================================

N_MELS = 64

N_FRAMES = 101

N_FEATURES = (
    N_MELS * N_FRAMES
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 0.001


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    csv_path
):
    """
    Load flattened Mel features from CSV
    and reshape them into CNN input format.

    CSV:

        samples × 6464

    CNN:

        samples × 64 × 101 × 1
    """

    print()

    print(
        "Loading:"
    )

    print(
        csv_path
    )

    df = pd.read_csv(
        csv_path
    )

    print(
        "CSV shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("mel_")
    ]

    print(
        "Number of Mel features:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    if len(feature_columns) != N_FEATURES:

        raise ValueError(
            f"Expected {N_FEATURES} Mel features, "
            f"but found {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Extract X
    # --------------------------------------------------------

    X = df[
        feature_columns
    ].values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    y_text = df[
        "label"
    ].values

    # --------------------------------------------------------
    # Convert labels
    #
    # bonafide = 0
    # spoof    = 1
    # --------------------------------------------------------

    y = np.where(
        y_text == "spoof",
        1,
        0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Reshape for CNN
    # --------------------------------------------------------

    X = X.reshape(
        -1,
        N_MELS,
        N_FRAMES,
        1
    )

    # --------------------------------------------------------
    # Normalize Mel values
    #
    # Mel dB values are approximately:
    #
    #       -80 to 0
    #
    # Convert them to approximately:
    #
    #       0 to 1
    # --------------------------------------------------------

    X = (
        X + 80.0
    ) / 80.0

    X = np.clip(
        X,
        0.0,
        1.0
    )

    print(
        "CNN input shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )

    print()

    print(
        "Bonafide:",
        np.sum(y == 0)
    )

    print(
        "Spoof:",
        np.sum(y == 1)
    )

    return X, y


# ============================================================
# BUILD CNN
# ============================================================

def build_model():
    """
    Build the Mel-spectrogram CNN.
    """

    model = Sequential(
        [

            # ------------------------------------------------
            # Input
            # ------------------------------------------------

            Input(
                shape=(
                    N_MELS,
                    N_FRAMES,
                    1
                )
            ),

            # ------------------------------------------------
            # Convolution Block 1
            # ------------------------------------------------

            Conv2D(
                filters=32,
                kernel_size=(
                    3,
                    3
                ),
                activation="relu",
                padding="same"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),

            # ------------------------------------------------
            # Convolution Block 2
            # ------------------------------------------------

            Conv2D(
                filters=64,
                kernel_size=(
                    3,
                    3
                ),
                activation="relu",
                padding="same"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),

            # ------------------------------------------------
            # Convolution Block 3
            # ------------------------------------------------

            Conv2D(
                filters=128,
                kernel_size=(
                    3,
                    3
                ),
                activation="relu",
                padding="same"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),

            # ------------------------------------------------
            # Dropout
            # ------------------------------------------------

            Dropout(
                0.30
            ),

            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------

            Flatten(),

            # ------------------------------------------------
            # Dense layer
            # ------------------------------------------------

            Dense(
                128,
                activation="relu"
            ),

            Dropout(
                0.40
            ),

            # ------------------------------------------------
            # Binary output
            # ------------------------------------------------

            Dense(
                1,
                activation="sigmoid"
            )
        ]
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
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
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_dev,
    y_dev
):
    """
    Evaluate the trained CNN.
    """

    print()

    print("=" * 60)

    print(
        "EVALUATING MEL CNN"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    probabilities = (
        model.predict(
            X_dev,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        .ravel()
    )

    # --------------------------------------------------------
    # Binary predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(
        int
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_dev,
        predictions
    )

    precision = precision_score(
        y_dev,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_dev,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_dev,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_dev,
        probabilities
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()

    print(
        "Accuracy :",
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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_dev,
        predictions
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_dev,
            predictions,
            target_names=[
                "bonafide",
                "spoof"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "MEL-SPECTROGRAM CNN"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{TRAIN_PATH}"
        )

    if not DEV_PATH.exists():

        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{DEV_PATH}"
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train, y_train = load_dataset(
        TRAIN_PATH
    )

    # --------------------------------------------------------
    # Load development data
    # --------------------------------------------------------

    X_dev, y_dev = load_dataset(
        DEV_PATH
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()

    print(
        "Building CNN..."
    )

    model = build_model()

    print()

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
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "STARTING TRAINING"
    )

    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_dev,
            y_dev
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_model(
        model,
        X_dev,
        y_dev
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR
        / "mel_cnn_final.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "MEL CNN TRAINING COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Best model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Final model:"
    )

    print(
        final_model_path
    )

    print()

    print(
        "Final metrics:"
    )

    print(
        f"Accuracy : {results['accuracy']:.4f}"
    )

    print(
        f"Precision: {results['precision']:.4f}"
    )

    print(
        f"Recall   : {results['recall']:.4f}"
    )

    print(
        f"F1 Score : {results['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {results['roc_auc']:.4f}"
    )