"""Logging prediksi, PSI sederhana, dan metrik setelah label aktual tersedia."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    brier_score_loss
)

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "monitoring" / "prediction_log.csv"

def log_predictions(raw, features, probabilities, threshold, predictions):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = raw.copy().reset_index(drop=True)
    out["probabilitas_batal"] = probabilities
    out["threshold"] = threshold
    out["prediksi_batal"] = predictions.astype(int)
    out["waktu_prediksi"] = pd.Timestamp.now().isoformat(timespec="seconds")
    out["actual_target"] = np.nan
    header = not LOG_PATH.exists()
    out.to_csv(LOG_PATH, mode="a", header=header, index=False)
    return out

def calculate_psi(expected, actual, bins=10):
    """PSI untuk satu fitur numerik; bin dibentuk dari kuantil data latih."""
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if len(expected) < 20 or len(actual) < 20:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def evaluate_labeled(log_df):
    if "actual_target" not in log_df or log_df["actual_target"].notna().sum() == 0:
        return None
    d = log_df.dropna(subset=["actual_target"]).copy()
    y = d["actual_target"].astype(int)
    pred = d["prediksi_batal"].astype(int)
    p = d["probabilitas_batal"].astype(float)
    result = {
        "Jumlah label": len(d),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "PR-AUC": average_precision_score(y, p) if y.nunique() > 1 else np.nan,
        "Brier": brier_score_loss(y, p),
    }
    return result
