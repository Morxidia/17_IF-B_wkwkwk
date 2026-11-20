import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def create_model(model_type, num_cols, cat_cols, **hyperparams):
    """
    Step 1: Builds a clean Pipeline that scales numbers, encodes categories,
    and attaches either a KNN or SVM classifier.
    """
    # 1. Prepare the column preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # 2. Pick the model (KNN or SVM)
    if model_type == "knn":
        model = KNeighborsClassifier(**hyperparams)
    elif model_type == "svm":
        # SVM needs probability=True so we can get cancellation probabilities
        hyperparams["probability"] = True
        model = SVC(**hyperparams)
    else:
        raise ValueError("Please choose either 'knn' or 'svm'")

    # 3. Combine preprocessor and model into one Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline


def find_best_threshold(y_true, y_probs, min_precision=0.30):
    """
    Step 2: Tests thresholds from 0.10 to 0.89 on the validation set
    to find the cutoff that gives the highest F1-score.
    """
    best_threshold = 0.50
    best_f1 = 0.0

    for thresh in np.arange(0.10, 0.90, 0.01):
        # Convert probabilities into 0 (Selesai) or 1 (Batal)
        predictions = (y_probs >= thresh).astype(int)

        prec = precision_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        if prec >= min_precision and f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(thresh), 2)

    return best_threshold


def run_model(model_type, splits, min_precision=0.30, **hyperparams):
    """
    Step 3: Trains the model, finds the best threshold on validation data,
    and evaluates performance on the test set.
    """
    start_time = time.time()

    # Build and train the pipeline
    pipeline = create_model(
        model_type=model_type,
        num_cols=splits["num_cols"],
        cat_cols=splits["cat_cols"],
        **hyperparams
    )
    pipeline.fit(splits["X_train"], splits["y_train"])
    train_time = round(time.time() - start_time, 2)

    # Tune probability threshold using validation data
    val_probs = pipeline.predict_proba(splits["X_val"])[:, 1]
    best_thresh = find_best_threshold(splits["y_val"], val_probs, min_precision)

    # Make final predictions on test data
    test_probs = pipeline.predict_proba(splits["X_test"])[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    y_test = splits["y_test"]

    # Store all scores in a simple dictionary
    results = {
        "Model": model_type.upper(),
        "Hyperparameters": str(hyperparams),
        "Best_Threshold": best_thresh,
        "F1_Score": round(float(f1_score(y_test, test_preds, zero_division=0)), 6),
        "PR_AUC": round(float(average_precision_score(y_test, test_probs)), 6),
        "ROC_AUC": round(float(roc_auc_score(y_test, test_probs)), 6),
        "Recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 6),
        "Precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 6),
        "Train_Time_Sec": train_time,
        "pipeline": pipeline,
        "y_probs": test_probs,
        "y_preds": test_preds,
        "confusion_matrix": confusion_matrix(y_test, test_preds)
    }
    return results


def compare_models(experiment_list, splits, min_precision=0.30):
    """
    Step 4: Runs a list of KNN and SVM experiments and returns
    a clean Pandas DataFrame comparing all of them.
    """
    all_rows = []
    saved_results = {}

    for i, (model_type, params) in enumerate(experiment_list, start=1):
        run_label = f"Run_{i}_{model_type.upper()}"
        print(f"Running {run_label}...")

        res = run_model(model_type, splits, min_precision=min_precision, **params)
        saved_results[run_label] = res

        # Keep only the metric columns for the summary table
        table_row = {
            "Run": run_label,
            "Model": res["Model"],
            "Hyperparameters": res["Hyperparameters"],
            "Threshold": res["Best_Threshold"],
            "F1": res["F1_Score"],
            "PR-AUC": res["PR_AUC"],
            "ROC-AUC": res["ROC_AUC"],
            "Recall": res["Recall"],
            "Precision": res["Precision"],
            "Time (s)": res["Train_Time_Sec"]
        }
        all_rows.append(table_row)

    summary_table = pd.DataFrame(all_rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return summary_table, saved_results