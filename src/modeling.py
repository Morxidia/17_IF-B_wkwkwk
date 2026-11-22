import time
import numpy as np
import pandas as pd
from typing import Any, Literal, overload
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, matthews_corrcoef, brier_score_loss,
    confusion_matrix, precision_recall_curve
)

# ==============================================================================
# DEFAULT HYPERPARAMETERS (#define-style constants)
# ==============================================================================
RANDOM_STATE      = 42
MIN_PRECISION     = 0.20
TARGET_RECALL     = 0.70

# KNN Defaults
KNN_N_NEIGHBORS   = 15
KNN_WEIGHTS       = "distance"     # "uniform" | "distance"
KNN_METRIC        = "manhattan"    # "manhattan" | "euclidean" | "cosine"

# SVM Defaults
SVM_C             = 0.5
SVM_KERNEL        = "rbf"          # "rbf" | "linear" | "poly"
SVM_GAMMA         = "scale"        # "scale" | "auto" | float
SVM_CLASS_WEIGHT  = "balanced"     # "balanced" | None

# Random Forest Defaults
RF_N_ESTIMATORS   = 300
RF_MAX_DEPTH      = 10             # int | None
RF_MIN_LEAF       = 5
RF_MIN_SPLIT      = 10
RF_MAX_FEATURES   = "sqrt"         # "sqrt" | "log2" | float
RF_CLASS_WEIGHT   = "balanced_subsample"
# ==============================================================================

def check_univariate_auc(splits: dict, threshold: float = 0.80) -> pd.DataFrame:
    """C.3: Univariate AUC guard to detect suspicious target leakage features."""
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_va, y_va = splits["X_val"], splits["y_val"]
    cat_cols = splits["cat_cols"]

    rows = []
    for col in splits["all_features"]:
        if col in cat_cols:
            rate = pd.concat([X_tr[col], y_tr], axis=1).groupby(col)["target_batal"].mean()
            score = X_va[col].map(rate).astype(float).fillna(float(y_tr.mean()))
        else:
            score = X_va[col].astype(float).fillna(X_va[col].median())
        auc = roc_auc_score(y_va, score)
        auc_sym = max(float(auc), float(1.0 - auc))
        rows.append({
            "Fitur": col,
            "AUC univariat (validasi)": round(auc_sym, 4),
            "Status": "CURIGA: periksa kebocoran" if auc_sym > threshold else "wajar"
        })

    return pd.DataFrame(rows).sort_values("AUC univariat (validasi)", ascending=False).reset_index(drop=True)


def create_model(model_type, num_cols, cat_cols, **hyperparams):
    """Builds a preprocessing + classifier Pipeline using top-level constants + overrides."""
    if model_type == "knn":
        params = {
            "n_neighbors": KNN_N_NEIGHBORS,
            "weights": KNN_WEIGHTS,
            "metric": KNN_METRIC,
            "n_jobs": -1,
            **hyperparams,
        }
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        clf = KNeighborsClassifier(**params)

    elif model_type == "svm":
        params = {
            "C": SVM_C,
            "kernel": SVM_KERNEL,
            "gamma": SVM_GAMMA,
            "class_weight": SVM_CLASS_WEIGHT,
            "random_state": RANDOM_STATE,
            **hyperparams,
            "probability": True,
        }
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        clf = SVC(**params)

    elif model_type == "random_forest":
        params = {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "min_samples_leaf": RF_MIN_LEAF,
            "min_samples_split": RF_MIN_SPLIT,
            "max_features": RF_MAX_FEATURES,
            "class_weight": RF_CLASS_WEIGHT,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            **hyperparams,
        }
        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), cat_cols),
        ])
        clf = RandomForestClassifier(**params)

    else:
        raise ValueError("Choose model_type from: 'knn', 'svm', or 'random_forest'")

    return Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

def tune_random_forest(splits: dict, n_iter: int = 20, random_state: int = 42):
    """C.4: Hyperparameter tuning with TimeSeriesSplit cross-validation."""
    base_pipe = create_model(
        "random_forest",
        num_cols=splits["num_cols"],
        cat_cols=splits["cat_cols"],
        random_state=random_state
    )

    param_dist = {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [6, 10, 14, 20, None],
        "clf__min_samples_leaf": [1, 3, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }

    search = RandomizedSearchCV(
        base_pipe,
        param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=4),
        random_state=random_state,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(splits["X_train"], splits["y_train"])

    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cols = ["rank_test_score", "mean_test_score", "std_test_score"] + [
        c for c in cv_df.columns if c.startswith("param_")
    ]
    return search.best_estimator_, search.best_score_, search.best_params_, cv_df[cols]


def find_validation_thresholds(y_val, p_val, target_recall: float = 0.70) -> tuple:
    """C.5: Finds max-F1 threshold and Recall >= target_recall threshold on validation data."""
    prec_v, rec_v, thr_v = precision_recall_curve(y_val, p_val)
    prec_v, rec_v = prec_v[:-1], rec_v[:-1]
    f1_v = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)

    thr_f1 = float(thr_v[np.argmax(f1_v)])

    ok = rec_v >= target_recall
    thr_rec = float(thr_v[ok][np.argmax(prec_v[ok])]) if ok.any() else None
    return thr_f1, thr_rec


def score_row(name: str, y_true, score, thr: float, with_brier: bool = True) -> dict:
    """Computes full classification & calibration metrics for a given threshold."""
    pred = (np.asarray(score) >= thr).astype(int)
    return {
        "Model": name,
        "Threshold": round(float(thr), 4),
        "F1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "F2": round(float(fbeta_score(y_true, pred, beta=2, zero_division=0)), 4),
        "Precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "PR-AUC": round(float(average_precision_score(y_true, score)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, score)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, pred)), 4),
        "Brier": round(float(brier_score_loss(y_true, score)), 4) if with_brier else np.nan,
    }


def evaluate_on_test(fitted_model: Pipeline, splits: dict, model_label: str = "Random Forest", random_state: int = 42) -> dict:
    """C.5: Evaluates model on test set with baselines, bootstrap 95% CI, and monthly stability."""
    X_va, y_va = splits["X_val"], splits["y_val"]
    X_te, y_te = splits["X_test"], splits["y_test"]

    p_va = fitted_model.predict_proba(X_va)[:, 1]
    p_te = fitted_model.predict_proba(X_te)[:, 1]

    thr_f1, thr_rec = find_validation_thresholds(y_va, p_va, target_recall=0.70)

    is_cod_te = splits["parts"]["Uji"]["is_cod"].values
    rows = [
        score_row("Baseline: selalu Selesai", y_te, np.zeros(len(y_te)), 0.5, with_brier=False),
        score_row("Baseline: COD = Batal", y_te, is_cod_te, 0.5, with_brier=False),
        score_row(f"{model_label} (threshold F1)", y_te, p_te, thr_f1),
    ]
    if thr_rec is not None:
        rows.append(score_row(f"{model_label} (threshold Recall>=0.70)", y_te, p_te, thr_rec))

    results_df = pd.DataFrame(rows).set_index("Model")

    # Bootstrap 95% CI on test set
    rng = np.random.default_rng(random_state)
    y_arr = y_te.values
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        yb, pb = y_arr[idx], p_te[idx]
        if yb.sum() == 0:
            continue
        boot.append((
            f1_score(yb, (pb >= thr_f1).astype(int), zero_division=0),
            average_precision_score(yb, pb)
        ))
    boot = np.array(boot)
    ci_f1 = np.percentile(boot[:, 0], [2.5, 97.5])
    ci_pr = np.percentile(boot[:, 1], [2.5, 97.5])

    # Monthly stability across test months
    per_month = []
    for m in splits["test_months"]:
        mask = (splits["parts"]["Uji"]["order_year_month"] == m).values
        ym, pm = y_arr[mask], p_te[mask]
        per_month.append({
            "Bulan": m,
            "Pesanan": int(mask.sum()),
            "Tingkat Batal (%)": round(float(ym.mean()) * 100, 2),
            "F1": round(float(f1_score(ym, (pm >= thr_f1).astype(int), zero_division=0)), 4),
            "PR-AUC": round(float(average_precision_score(ym, pm)), 4) if ym.sum() > 0 else np.nan,
        })

    return {
        "thr_f1": thr_f1,
        "thr_rec": thr_rec,
        "results_table": results_df,
        "ci_f1": (round(float(ci_f1[0]), 3), round(float(ci_f1[1]), 3)),
        "ci_pr": (round(float(ci_pr[0]), 3), round(float(ci_pr[1]), 3)),
        "monthly_stability": pd.DataFrame(per_month),
        "y_probs_val": p_va,
        "y_probs_test": p_te,
        "confusion_matrix": confusion_matrix(y_te, (p_te >= thr_f1).astype(int)),
    }

def find_best_threshold(y_true, y_probs, min_precision=0.30):
    """Finds the validation cutoff that maximizes F1 subject to min_precision."""
    prec, rec, thr = precision_recall_curve(y_true, y_probs)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    ok = prec >= min_precision
    if ok.any():
        return float(thr[np.argmax(np.where(ok, f1, -1))])
    print(f"[warn] no threshold reaches precision>={min_precision}; using best-F1 threshold instead")
    return float(thr[np.argmax(f1)])


def run_model(model_type, splits, min_precision=0.30, **hyperparams):
    """Trains KNN, SVM, or Random Forest, tunes threshold on Val, and evaluates on Test."""
    start_time = time.time()

    pipeline = create_model(
        model_type=model_type,
        num_cols=splits["num_cols"],
        cat_cols=splits["cat_cols"],
        **hyperparams
    )
    pipeline.fit(splits["X_train"], splits["y_train"])
    train_time = round(time.time() - start_time, 2)

    val_probs = pipeline.predict_proba(splits["X_val"])[:, 1]
    best_thresh = find_best_threshold(splits["y_val"], val_probs, min_precision)

    test_probs = pipeline.predict_proba(splits["X_test"])[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    y_test = splits["y_test"]

    return {
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


def compare_models(experiment_list, splits, min_precision=0.30):
    """Runs a list of (model_type, params) experiments and returns a comparison DataFrame."""
    all_rows = []
    saved_results = {}

    for i, (model_type, params) in enumerate(experiment_list, start=1):
        run_label = f"Run_{i}_{model_type.upper()}"
        print(f"Running {run_label}...")

        res = run_model(model_type, splits, min_precision=min_precision, **params)
        saved_results[run_label] = res

        all_rows.append({
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
        })

    summary_table = pd.DataFrame(all_rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return summary_table, saved_results