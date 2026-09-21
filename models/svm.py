from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mfcc_1s_clean_train.csv"
)

DEV_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mfcc_1s_clean_dev.csv"
)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading training dataset...")

train_df = pd.read_csv(TRAIN_PATH)

print("Training shape:", train_df.shape)

print()

print("Loading development dataset...")

dev_df = pd.read_csv(DEV_PATH)

print("Development shape:", dev_df.shape)


# ---------------------------------------------------------
# Separate features and labels
# ---------------------------------------------------------

DROP_COLUMNS = [
    "file",
    "split",
    "label"
]

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

X_dev = dev_df.drop(
    columns=DROP_COLUMNS
)

y_train = (
    train_df["label"] == "spoof"
).astype(int)

y_dev = (
    dev_df["label"] == "spoof"
).astype(int)


print()
print("X_train shape:", X_train.shape)
print("X_dev shape  :", X_dev.shape)


# ---------------------------------------------------------
# Scale features
# ---------------------------------------------------------

print()
print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_dev_scaled = scaler.transform(
    X_dev
)


# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------

print()
print("Training SVM...")

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

print()
print("Generating predictions...")

y_pred = model.predict(
    X_dev_scaled
)

y_probability = model.predict_proba(
    X_dev_scaled
)[:, 1]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 60)
print("SVM RESULTS")
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

print()
print("Confusion Matrix:")
print(cm)

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

print()
print("SVM test complete.")