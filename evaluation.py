import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.calibration import calibration_curve


def run_evaluation(summary_df, all_artifacts, splits):

    # ==========================================
    # 1. TABEL PERBANDINGAN MODEL
    # ==========================================

    print("\n========== HASIL EVALUASI MODEL ==========\n")

    evaluation_df = summary_df.sort_values(
        by="F1", ascending=False
    ).reset_index(drop=True)

    display(evaluation_df)

    # ==========================================
    # 2. MENENTUKAN MODEL TERBAIK
    # ==========================================

    best_model = evaluation_df.iloc[0]

    print("\n========== MODEL TERBAIK ==========")
    print("Run       :", best_model["Run"])
    print("Model     :", best_model["Model"])
    print("F1-Score  :", best_model["F1"])
    print("Precision :", best_model["Precision"])
    print("Recall    :", best_model["Recall"])
    print("PR-AUC    :", best_model["PR-AUC"])
    print("ROC-AUC   :", best_model["ROC-AUC"])

    # ==========================================
    # 3. CONFUSION MATRIX
    # ==========================================

    n_models = len(all_artifacts)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 4)
    )

    if n_models == 1:
        axes = [axes]

    for ax, (run_name, art) in zip(
        axes, all_artifacts.items()
    ):

        cm = art.get("confusion_matrix")

        if cm is None:
            ax.set_visible(False)
            continue

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Selesai", "Batal"]
        ).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        ax.set_title(run_name)
        ax.grid(False)

    plt.suptitle("Confusion Matrix Perbandingan Model")
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 4. PRECISION-RECALL CURVE
    # ==========================================

    y_test = splits["y_test"]

    plt.figure(figsize=(9, 6))

    for run_name, art in all_artifacts.items():

        y_probs = art.get("y_probs")

        if y_probs is None:
            continue

        precision, recall, _ = precision_recall_curve(
            y_test, y_probs
        )

        plt.plot(
            recall,
            precision,
            label=run_name
        )

    baseline = y_test.mean()

    plt.axhline(
        baseline,
        linestyle="--",
        label=f"Baseline ({baseline:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ==========================================
    # 5. ROC CURVE
    # ==========================================

    plt.figure(figsize=(9, 6))

    for run_name, art in all_artifacts.items():

        y_probs = art.get("y_probs")

        if y_probs is None:
            continue

        fpr, tpr, _ = roc_curve(
            y_test, y_probs
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{run_name} (AUC={roc_auc:.2f})"
        )

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ==========================================
    # 6. RELIABILITY CURVE
    # ==========================================

    plt.figure(figsize=(9, 6))

    for run_name, art in all_artifacts.items():

        y_probs = art.get("y_probs")

        if y_probs is None:
            continue

        prob_true, prob_pred = calibration_curve(
            y_test,
            y_probs,
            n_bins=10,
            strategy="quantile"
        )

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=run_name
        )

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()