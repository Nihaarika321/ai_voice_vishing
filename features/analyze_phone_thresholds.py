from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = Path("data/results")

INPUT_FILE = RESULTS_DIR / "phone_robustness_detailed.csv"

OUTPUT_FILE = RESULTS_DIR / "phone_threshold_analysis.csv"


# ============================================================
# FIND COLUMNS
# ============================================================

def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHONE ROBUSTNESS THRESHOLD ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nDetailed results not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nLoaded:")
    print(INPUT_FILE)

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # FIND IMPORTANT COLUMNS
    # --------------------------------------------------------

    label_col = find_column(
        df,
        [
            "label",
            "true_label",
            "y_true",
            "target"
        ]
    )

    probability_col = find_column(
        df,
        [
            "spoof_probability",
            "probability",
            "prob",
            "score",
            "prediction_probability"
        ]
    )

    condition_col = find_column(
        df,
        [
            "condition"
        ]
    )

    if label_col is None:

        raise ValueError(
            "\nCould not find label column."
        )

    if probability_col is None:

        raise ValueError(
            "\nCould not find spoof probability column."
        )

    if condition_col is None:

        raise ValueError(
            "\nCould not find condition column."
        )

    print("\nDetected columns:")

    print("Label       :", label_col)
    print("Probability :", probability_col)
    print("Condition   :", condition_col)

    # --------------------------------------------------------
    # CONVERT LABELS
    # --------------------------------------------------------

    def convert_label(value):

        value = str(value).lower().strip()

        if value == "spoof":
            return 1

        if value == "bonafide":
            return 0

        if value in ["1", "1.0"]:
            return 1

        if value in ["0", "0.0"]:
            return 0

        return np.nan

    df["y_true"] = df[label_col].apply(convert_label)

    df["probability"] = pd.to_numeric(
        df[probability_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "y_true",
            "probability"
        ]
    )

    df["y_true"] = df["y_true"].astype(int)

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    print("\nValid rows:", len(df))

    print("\nClass distribution:")

    print(
        df["y_true"].value_counts()
    )

    print("\nProbability statistics:")

    print(
        df["probability"].describe()
    )

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.96,
        0.05
    )

    results = []

    for condition, group in df.groupby(condition_col):

        y_true = group["y_true"].values

        probabilities = group[
            "probability"
        ].values

        auc = roc_auc_score(
            y_true,
            probabilities
        )

        print("\n" + "=" * 60)
        print("CONDITION:", condition)
        print("=" * 60)

        print(
            f"ROC-AUC: {auc:.4f}"
        )

        print(
            "\nThreshold performance:"
        )

        for threshold in thresholds:

            y_pred = (
                probabilities >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            results.append({

                "condition": condition,

                "threshold": threshold,

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "roc_auc": auc

            })

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # BEST THRESHOLD BY F1
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BEST THRESHOLDS")
    print("=" * 60)

    for condition, group in results_df.groupby(
        "condition"
    ):

        best = group.loc[
            group["f1"].idxmax()
        ]

        print(
            f"\n{condition}"
        )

        print(
            f"Threshold : "
            f"{best['threshold']:.2f}"
        )

        print(
            f"Accuracy  : "
            f"{best['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{best['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{best['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{best['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{best['roc_auc']:.4f}"
        )

    print(
        "\nThreshold analysis saved:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()