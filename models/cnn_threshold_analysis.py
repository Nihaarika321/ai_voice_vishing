from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "cnn_mfcc_1s_clean_dev.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_cnn.keras"
)


# ---------------------------------------------------------
# Load development dataset
# ---------------------------------------------------------

print("Loading development dataset...")

data = np.load(DEV_DATA_PATH)

X_dev = data["X"]
y_dev = data["y"]

print("X_dev shape:", X_dev.shape)
print("y_dev shape:", y_dev.shape)


# ---------------------------------------------------------
# Prepare CNN input
# ---------------------------------------------------------

X_dev = X_dev.astype(np.float32)

# CNN expects:
# (samples, MFCC, time, channel)

X_dev = X_dev[..., np.newaxis]

print("CNN input shape:", X_dev.shape)


# ---------------------------------------------------------
# Load best CNN
# ---------------------------------------------------------

print()
print("Loading best CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Generate probabilities
# ---------------------------------------------------------

print()
print("Generating spoof probabilities...")

probabilities = model.predict(
    X_dev,
    batch_size=32,
    verbose=1
).ravel()


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []


print()
print("=" * 100)
print("CNN THRESHOLD ANALYSIS")
print("=" * 100)

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'TN':>7} "
    f"{'FP':>7} "
    f"{'FN':>7} "
    f"{'TP':>7}"
)

print("-" * 100)


for threshold in thresholds:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_dev,
        y_pred
    )

    precision = precision_score(
        y_dev,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_dev,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_dev,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_dev,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

    print(
        f"{threshold:10.2f} "
        f"{accuracy:10.4f} "
        f"{precision:10.4f} "
        f"{recall:10.4f} "
        f"{f1:10.4f} "
        f"{tn:7d} "
        f"{fp:7d} "
        f"{fn:7d} "
        f"{tp:7d}"
    )


# ---------------------------------------------------------
# Find best thresholds
# ---------------------------------------------------------

print()
print("=" * 100)
print("BEST THRESHOLDS")
print("=" * 100)


best_accuracy = max(
    results,
    key=lambda x: x["accuracy"]
)

best_precision = max(
    results,
    key=lambda x: x["precision"]
)

best_recall = max(
    results,
    key=lambda x: x["recall"]
)

best_f1 = max(
    results,
    key=lambda x: x["f1"]
)


print()
print("Best Accuracy:")
print(best_accuracy)

print()
print("Best Precision:")
print(best_precision)

print()
print("Best Recall:")
print(best_recall)

print()
print("Best F1:")
print(best_f1)


# ---------------------------------------------------------
# Recommended operating point
# ---------------------------------------------------------

# For spoof/vishing detection, recall is important.
# Select the threshold with the best F1 while
# maintaining at least 0.90 spoof recall.

eligible = [
    r
    for r in results
    if r["recall"] >= 0.90
]

if eligible:

    recommended = max(
        eligible,
        key=lambda x: x["f1"]
    )

    print()
    print("=" * 100)
    print("RECOMMENDED OPERATING POINT")
    print("=" * 100)

    print(
        f"Threshold : {recommended['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {recommended['accuracy']:.4f}"
    )

    print(
        f"Precision : {recommended['precision']:.4f}"
    )

    print(
        f"Recall    : {recommended['recall']:.4f}"
    )

    print(
        f"F1 Score  : {recommended['f1']:.4f}"
    )

    print(
        f"TN        : {recommended['tn']}"
    )

    print(
        f"FP        : {recommended['fp']}"
    )

    print(
        f"FN        : {recommended['fn']}"
    )

    print(
        f"TP        : {recommended['tp']}"
    )

else:

    print()
    print(
        "No threshold achieved at least 0.90 spoof recall."
    )


print()
print("=" * 100)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 100)