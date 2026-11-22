"""
evaluation.py
Tahap Evaluasi: Klasifikasi Risiko Pembatalan Pesanan (target_batal: 1 = Batal, 0 = Selesai).

Fungsi dipanggil per bagian dari notebook supaya tiap hasil bisa dijelaskan:
    1. tabel_metrik_test        -> metrik akhir di DATA TEST
    2. plot_confusion_matrix    -> grafik pendukung (klasifikasi)
    3. plot_kurva               -> PR curve, ROC curve, reliability curve
    4. bandingkan_train_test    -> cek overfitting (train vs validasi vs test)
    5. bandingkan_baseline      -> model vs tebakan sederhana + cek kriteria berhasil
    6. pilih_model_terbaik      -> pilih model + alasan (bukan cuma F1 tertinggi)
    7. analisis_kesalahan       -> contoh kesalahan (FP & FN) + profilnya
    run_evaluation(...)         -> menjalankan 1-7 sekaligus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.metrics import (
    ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve

# Kriteria berhasil dari Business Understanding
TARGET_RECALL = 0.70
TARGET_PRECISION = 0.30


def _metrik(y, prob, thr):
    pred = (np.asarray(prob) >= thr).astype(int)
    return {
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "PR-AUC": average_precision_score(y, prob),
        "ROC-AUC": roc_auc_score(y, prob),
    }


# ==============================================================================
# 1. METRIK AKHIR DI DATA TEST
# ==============================================================================
def tabel_metrik_test(summary_df):
    """Tabel metrik semua model di data test, diurutkan dari F1 tertinggi."""
    cols = ["Run", "Model", "Threshold", "F1", "Precision", "Recall", "PR-AUC", "ROC-AUC"]  # semua dihitung di data TEST
    df = summary_df.sort_values("F1", ascending=False).reset_index(drop=True)[cols]
    print("========== METRIK AKHIR PADA DATA TEST ==========")
    display(df.round(4))
    return df


# ==============================================================================
# 2. CONFUSION MATRIX
# ==============================================================================
def plot_confusion_matrix(all_artifacts):
    rows = []
    for run, art in all_artifacts.items():
        tn, fp, fn, tp = art["confusion_matrix"].ravel()
        rows.append({"Run": run, "TP (batal terdeteksi)": int(tp), "FN (batal terlewat)": int(fn),
                     "FP (salah alarm)": int(fp), "TN (selesai benar)": int(tn)})
    display(pd.DataFrame(rows))

    n = len(all_artifacts)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (run, art) in zip(axes, all_artifacts.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=art["confusion_matrix"],
            display_labels=["Selesai", "Batal"],
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{run}\nthr={art['Best_Threshold']:.2f}, F1={art['F1_Score']:.3f}", fontsize=10)
        ax.grid(False)
    plt.suptitle("Confusion Matrix pada Data Test (baris = aktual, kolom = prediksi)")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 3. PR / ROC / RELIABILITY CURVE
# ==============================================================================
def plot_kurva(all_artifacts, splits):
    y_test = splits["y_test"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PR curve
    ax = axes[0]
    for run, art in all_artifacts.items():
        p, r, _ = precision_recall_curve(y_test, art["y_probs"])
        ax.plot(r, p, label=run)
    ax.axhline(y_test.mean(), ls="--", color="gray", label=f"Baseline ({y_test.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ROC curve
    ax = axes[1]
    for run, art in all_artifacts.items():
        fpr, tpr, _ = roc_curve(y_test, art["y_probs"])
        ax.plot(fpr, tpr, label=f"{run} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Tebak acak")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Reliability curve
    ax = axes[2]
    for run, art in all_artifacts.items():
        pt, pp = calibration_curve(y_test, art["y_probs"], n_bins=10, strategy="quantile")
        ax.plot(pp, pt, marker="o", label=run)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Kalibrasi sempurna")
    ax.set(xlabel="Rata-rata probabilitas prediksi", ylabel="Proporsi batal sebenarnya",
           title="Reliability Curve")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. TRAIN vs VALIDASI vs TEST  (cek overfitting)
# ==============================================================================
def bandingkan_train_test(all_artifacts, splits, plot=True):
    """Hitung metrik yang sama (threshold yang sama) di data latih, validasi, dan test."""
    rows = []
    for run, art in all_artifacts.items():
        pipe, thr = art["pipeline"], art["Best_Threshold"]
        row = {"Run": run}
        for nama, key in [("Train", "train"), ("Val", "val"), ("Test", "test")]:
            prob = pipe.predict_proba(splits[f"X_{key}"])[:, 1]
            m = _metrik(splits[f"y_{key}"], prob, thr)
            row[f"F1 {nama}"] = m["F1"]
            row[f"PR-AUC {nama}"] = m["PR-AUC"]
            row[f"Precision {nama}"] = m["Precision"]
        row["Selisih F1 (Train-Test)"] = row["F1 Train"] - row["F1 Test"]
        row["Selisih PR-AUC (Train-Test)"] = row["PR-AUC Train"] - row["PR-AUC Test"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Status"] = np.where(
        (df["Selisih PR-AUC (Train-Test)"] > 0.20) | (df["Selisih F1 (Train-Test)"] > 0.30),
        "OVERFITTING", "relatif stabil",
    )
    print("========== PERBANDINGAN DATA TRAIN vs VALIDASI vs TEST ==========")
    display(df.round(4))

    if plot:
        x = np.arange(len(df)); w = 0.27
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for ax, metric in zip(axes, ["F1", "PR-AUC"]):
            ax.bar(x - w, df[f"{metric} Train"], w, label="Train")
            ax.bar(x, df[f"{metric} Val"], w, label="Validasi")
            ax.bar(x + w, df[f"{metric} Test"], w, label="Test")
            ax.set_xticks(x); ax.set_xticklabels(df["Run"], rotation=20, ha="right", fontsize=8)
            ax.set_title(f"{metric}: Train vs Validasi vs Test"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.show()
    return df


# ==============================================================================
# 5. BASELINE + KRITERIA BERHASIL
# ==============================================================================
def bandingkan_baseline(summary_df, all_artifacts, splits):
    """Bandingkan model terbaik dengan tebakan sederhana, lalu cek kriteria berhasil."""
    y = splits["y_test"].values
    is_cod = splits["parts"]["Uji"]["is_cod"].values

    def _row(nama, pred):
        return {
            "Pembanding": nama,
            "F1": f1_score(y, pred, zero_division=0),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
        }

    best = summary_df.sort_values("F1", ascending=False).iloc[0]
    best_run = best["Run"]
    best_pred = all_artifacts[best_run]["y_preds"]

    df = pd.DataFrame([
        _row("Baseline 1: semua pesanan ditebak Batal", np.ones(len(y), dtype=int)),
        _row("Baseline 2: COD = Batal, selain itu Selesai", is_cod.astype(int)),
        _row(f"Model terbaik ({best_run})", best_pred),
    ])
    print("========== MODEL vs BASELINE (DATA TEST) ==========")
    display(df.round(4))

    f1_model = df.iloc[2]["F1"]
    f1_base = df.iloc[:2]["F1"].max()
    cek = pd.DataFrame([
        {"Kriteria (Business Understanding)": "F1 model > F1 baseline terbaik",
         "Target": f"> {f1_base:.3f}", "Hasil": round(f1_model, 3),
         "Tercapai?": "YA" if f1_model > f1_base else "BELUM"},
        {"Kriteria (Business Understanding)": "Recall >= 0,70",
         "Target": TARGET_RECALL, "Hasil": round(df.iloc[2]["Recall"], 3),
         "Tercapai?": "YA" if df.iloc[2]["Recall"] >= TARGET_RECALL else "BELUM"},
        {"Kriteria (Business Understanding)": "Precision >= 0,30",
         "Target": TARGET_PRECISION, "Hasil": round(df.iloc[2]["Precision"], 3),
         "Tercapai?": "YA" if df.iloc[2]["Precision"] >= TARGET_PRECISION else "BELUM"},
    ])
    print("\n========== CEK KRITERIA BERHASIL ==========")
    display(cek)
    return df, cek


# ==============================================================================
# 6. PEMILIHAN MODEL TERBAIK (dengan alasan)
# ==============================================================================
def pilih_model_terbaik(summary_df, train_test_df):
    """Pilih model: F1 test tertinggi DAN tidak overfitting. Cetak alasannya."""
    tabel = summary_df[["Run", "F1", "PR-AUC", "ROC-AUC", "Recall", "Precision", "Time (s)"]].merge(
        train_test_df[["Run", "Selisih F1 (Train-Test)", "Selisih PR-AUC (Train-Test)", "Status"]], on="Run")
    tabel = tabel.rename(columns={"F1": "F1 Test"}).sort_values("F1 Test", ascending=False).reset_index(drop=True)

    layak = tabel[tabel["Status"] != "OVERFITTING"]
    best = (layak if len(layak) else tabel).iloc[0]

    print("========== PEMILIHAN MODEL TERBAIK ==========")
    display(tabel.round(4))
    print(f"\nModel terpilih : {best['Run']}")
    print("Alasan:")
    print(f"  1. F1 test tertinggi di antara model yang tidak overfitting ({best['F1 Test']:.4f}).")
    print(f"  2. Selisih F1 train-test hanya {best['Selisih F1 (Train-Test)']:.3f} "
          f"(selisih PR-AUC {best['Selisih PR-AUC (Train-Test)']:.3f}) -> performanya konsisten.")
    ovf = tabel[tabel["Status"] == "OVERFITTING"]["Run"].tolist()
    if ovf:
        print(f"  3. Model yang overfitting dan tidak dipilih: {', '.join(ovf)}.")
    return best


# ==============================================================================
# 7. ANALISIS KESALAHAN
# ==============================================================================
def analisis_kesalahan(best_run, all_artifacts, splits, n_contoh=10):
    """Contoh salah alarm (FP) dan batal terlewat (FN), plus profil ciri-cirinya."""
    art = all_artifacts[best_run]
    te = splits["parts"]["Uji"].copy()
    te["prob"] = art["y_probs"]
    te["pred"] = art["y_preds"]
    te["aktual"] = splits["y_test"].values
    te["luar_jawa"] = 1 - te["is_java"]

    fp = te[(te["aktual"] == 0) & (te["pred"] == 1)]
    fn = te[(te["aktual"] == 1) & (te["pred"] == 0)]
    tp = te[(te["aktual"] == 1) & (te["pred"] == 1)]
    tn = te[(te["aktual"] == 0) & (te["pred"] == 0)]

    kolom = ["order_id", "Metode Pembayaran", "shipping_tier", "Provinsi", "total_qty", "Perkiraan Ongkos Kirim"]
    print(f"========== CONTOH KESALAHAN ({best_run}) ==========")
    print(f"\nFalse Positive / salah alarm (Selesai tapi diprediksi Batal): {len(fp)} pesanan")
    display(fp.sort_values("prob", ascending=False)[kolom + ["prob"]].head(n_contoh).round(3))
    print(f"\nFalse Negative / batal terlewat (Batal tapi diprediksi Selesai): {len(fn)} pesanan")
    display(fn.sort_values("prob")[kolom + ["prob"]].head(n_contoh).round(3))

    def profil(d):
        return {
            "Jumlah": len(d),
            "% COD": d["is_cod"].mean() * 100,
            "% luar Jawa": d["luar_jawa"].mean() * 100,
            "% kurir Hemat/Kargo": (d["shipping_tier"] == "Hemat / Kargo").mean() * 100,
            "Median ongkir (Rp)": d["Perkiraan Ongkos Kirim"].median(),
            "Median probabilitas batal": d["prob"].median(),
        }

    prof = pd.DataFrame({
        "Semua test": profil(te),
        "Batal terdeteksi (TP)": profil(tp),
        "Batal terlewat (FN)": profil(fn),
        "Salah alarm (FP)": profil(fp),
        "Selesai benar (TN)": profil(tn),
    }).T
    print("\n========== PROFIL CIRI-CIRI PER KELOMPOK ==========")
    display(prof.round(2))
    return fp, fn, prof


# ==============================================================================
# JALANKAN SEMUA
# ==============================================================================
def run_evaluation(summary_df, all_artifacts, splits):
    tabel_metrik_test(summary_df)
    plot_confusion_matrix(all_artifacts)
    plot_kurva(all_artifacts, splits)
    tt = bandingkan_train_test(all_artifacts, splits)
    bandingkan_baseline(summary_df, all_artifacts, splits)
    best = pilih_model_terbaik(summary_df, tt)
    analisis_kesalahan(best["Run"], all_artifacts, splits)