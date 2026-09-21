from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "cnn_mfcc_1s_clean_train.npz"
)

DEV_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "cnn_mfcc_1s_clean_dev.npz"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = MODEL_DIR / "best_cnn.keras"


# =========================================================
# RANDOM SEEDS
# =========================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("Loading CNN training dataset...")

train_data = np.load(TRAIN_PATH)

X_train = train_data["X"]
y_train = train_data["y"]

print(
    "Training X shape:",
    X_train.shape
)

print(
    "Training y shape:",
    y_train.shape
)


# =========================================================
# LOAD DEVELOPMENT DATA
# =========================================================

print()
print("Loading CNN development dataset...")

dev_data = np.load(DEV_PATH)

X_dev = dev_data["X"]
y_dev = dev_data["y"]

print(
    "Development X shape:",
    X_dev.shape
)

print(
    "Development y shape:",
    y_dev.shape
)


# =========================================================
# CHECK DATA
# =========================================================

print()
print("Training class distribution:")

unique_train, counts_train = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(
    unique_train,
    counts_train
):
    print(
        f"Class {label}: {count}"
    )


print()
print("Development class distribution:")

unique_dev, counts_dev = np.unique(
    y_dev,
    return_counts=True
)

for label, count in zip(
    unique_dev,
    counts_dev
):
    print(
        f"Class {label}: {count}"
    )


# =========================================================
# ADD CHANNEL DIMENSION
# =========================================================

# Current shape:
#
# (samples, 13, 101)
#
# CNN requires:
#
# (samples, 13, 101, 1)

X_train = X_train[..., np.newaxis]

X_dev = X_dev[..., np.newaxis]

print()
print(
    "CNN training input shape:",
    X_train.shape
)

print(
    "CNN development input shape:",
    X_dev.shape
)


# =========================================================
# NORMALIZE MFCC INPUT
# =========================================================
#
# Calculate statistics ONLY from training data.
#
# This avoids using development information during
# preprocessing.
# =========================================================

train_mean = np.mean(
    X_train,
    axis=(0, 1, 2),
    keepdims=True
)

train_std = np.std(
    X_train,
    axis=(0, 1, 2),
    keepdims=True
)

train_std = np.maximum(
    train_std,
    1e-8
)

X_train = (
    X_train - train_mean
) / train_std

X_dev = (
    X_dev - train_mean
) / train_std


# =========================================================
# BUILD CNN
# =========================================================

print()
print("Building CNN model...")

model = Sequential(
    [

        # -------------------------------------------------
        # Convolution block 1
        # -------------------------------------------------

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=(
                X_train.shape[1],
                X_train.shape[2],
                X_train.shape[3]
            )
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),


        # -------------------------------------------------
        # Convolution block 2
        # -------------------------------------------------

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),


        # -------------------------------------------------
        # Classification layers
        # -------------------------------------------------

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.5
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ]
)


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =========================================================
# PRINT ARCHITECTURE
# =========================================================

print()
print("CNN architecture:")

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


model_checkpoint = ModelCheckpoint(
    filepath=str(BEST_MODEL_PATH),

    monitor="val_loss",

    save_best_only=True,

    save_weights_only=False,

    verbose=1
)


callbacks = [
    early_stopping,
    model_checkpoint
]


# =========================================================
# TRAIN CNN
# =========================================================

print()
print("=" * 60)
print("TRAINING CNN")
print("=" * 60)

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_dev,
        y_dev
    ),

    epochs=15,

    batch_size=32,

    callbacks=callbacks,

    verbose=1
)


# =========================================================
# BEST EPOCH
# =========================================================

best_epoch = (
    np.argmin(
        history.history["val_loss"]
    ) + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print()
print("=" * 60)
print("BEST CNN TRAINING RESULT")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best validation loss: {best_val_loss:.4f}"
)

print(
    f"Best validation accuracy: {best_val_accuracy:.4f}"
)


# =========================================================
# LOAD BEST MODEL
# =========================================================

print()
print("Loading best CNN model...")

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# =========================================================
# PREDICTIONS
# =========================================================

print()
print("Generating predictions...")

y_probability = (
    best_model.predict(
        X_dev,
        verbose=0
    ).ravel()
)

y_pred = (
    y_probability >= 0.5
).astype(int)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_dev,
    y_pred
)

precision = precision_score(
    y_dev,
    y_pred
)

recall = recall_score(
    y_dev,
    y_pred
)

f1 = f1_score(
    y_dev,
    y_pred
)

roc_auc = roc_auc_score(
    y_dev,
    y_probability
)

cm = confusion_matrix(
    y_dev,
    y_pred
)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("CNN BASELINE RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("Confusion Matrix:")

print(cm)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("Classification Report:")

print(
    classification_report(
        y_dev,
        y_pred,
        target_names=[
            "bonafide",
            "spoof"
        ]
    )
)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_path = (
    MODEL_DIR
    / "cnn_training_history.npz"
)

np.savez(
    history_path,
    loss=np.asarray(
        history.history["loss"]
    ),
    accuracy=np.asarray(
        history.history["accuracy"]
    ),
    val_loss=np.asarray(
        history.history["val_loss"]
    ),
    val_accuracy=np.asarray(
        history.history["val_accuracy"]
    )
)


# =========================================================
# FINAL INFORMATION
# =========================================================

print()
print("=" * 60)
print("CNN BASELINE TEST COMPLETE")
print("=" * 60)

print()
print("Best model saved to:")
print(BEST_MODEL_PATH)

print()
print("Training history saved to:")
print(history_path)