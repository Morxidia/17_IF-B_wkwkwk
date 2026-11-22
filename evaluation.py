"""
evaluation.py
Tahap Evaluasi: Klasifikasi Risiko Pembatalan Pesanan (target_batal: 1 = Batal, 0 = Selesai).

Prinsip: model DIPILIH dari data validasi, data test hanya untuk MELAPORKAN.

    1. tabel_metrik            -> metrik validasi (untuk memilih) dan test (untuk melapor)
    2. plot_confusion_matrix   -> grafik pendukung (klasifikasi)
    3. plot_kurva              -> PR curve, ROC curve, reliability curve
    4. bandingkan_train_test   -> cek stabilitas (train / validasi / test)
    5. pilih_model_terbaik     -> pilih model dari validasi + alasan
    6. bandingkan_baseline     -> model TERPILIH vs tebakan sederhana + cek kriteria berhasil
    7. analisis_kesalahan      -> contoh kesalahan (FP & FN) + profilnya
    8. bootstrap_ci            -> ketidakpastian (CI 95%) termasuk selisih vs baseline
    9. metrik_tambahan         -> F2, MCC, Brier, ECE
   10. stabilitas_bulanan      -> metrik per bulan di data test
   11. tabel_titik_operasi     -> apakah Recall>=0,70 dan Precision>=0,30 mungkin bersamaan
   12. prevalensi_per_split    -> prevalensi dan PR-AUC relatif terhadap baseline-nya
   13. importance_permutasi    -> permutation importance (di validasi)
   14. cv_waktu                -> cross-validation berbasis waktu (opsional, lambat)
    run_evaluation(...)        -> menjalankan semuanya berurutan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc,
    f1_score, fbeta_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    matthews_corrcoef, brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Kriteria berhasil dari Business Understanding
TARGET_RECALL = 0.70
TARGET_PRECISION = 0.30

# Ambang selisih validasi-test yang ditandai "TIDAK STABIL" (heuristik, bukan uji statistik)
AMBANG_STABIL = 0.05

_VAL_CACHE = {}


def _val_probs(run, art, splits):
    """Probabilitas model pada data validasi (di-cache supaya tidak dihitung berulang)."""
    key = (run, id(art["pipeline"]))
    if key not in _VAL_CACHE:
        _VAL_CACHE[key] = art["pipeline"].predict_proba(splits["X_val"])[:, 1]
    return _VAL_CACHE[key]


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
# 1. METRIK: VALIDASI (memilih) dan TEST (melapor)
# ==============================================================================
def tabel_metrik(summary_df, all_artifacts, splits):
    """Urutan tabel berdasarkan VALIDASI. Kolom test hanya untuk pelaporan."""
    rows = []
    for run, art in all_artifacts.items():
        m = _metrik(splits["y_val"], _val_probs(run, art, splits), art["Best_Threshold"])
        rows.append({"Run": run, "Val F1": m["F1"], "Val PR-AUC": m["PR-AUC"]})
    val = pd.DataFrame(rows)

    cols = ["Run", "Model", "Threshold", "F1", "Precision", "Recall", "PR-AUC", "ROC-AUC"]
    df = summary_df[cols].merge(val, on="Run").rename(columns={
        "F1": "F1 Test", "Precision": "Precision Test", "Recall": "Recall Test",
        "PR-AUC": "PR-AUC Test", "ROC-AUC": "ROC-AUC Test"})
    df = df[["Run", "Model", "Threshold", "Val F1", "Val PR-AUC",
             "F1 Test", "Precision Test", "Recall Test", "PR-AUC Test", "ROC-AUC Test"]]
    df = df.sort_values(["Val F1", "Val PR-AUC"], ascending=False).reset_index(drop=True)
    print("========== METRIK: VALIDASI (dasar memilih) dan TEST (untuk melapor) ==========")
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

    ax = axes[0]
    for run, art in all_artifacts.items():
        p, r, _ = precision_recall_curve(y_test, art["y_probs"])
        ax.plot(r, p, label=run)
    ax.axhline(y_test.mean(), ls="--", color="gray", label=f"Baseline ({y_test.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for run, art in all_artifacts.items():
        fpr, tpr, _ = roc_curve(y_test, art["y_probs"])
        ax.plot(fpr, tpr, label=f"{run} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Tebak acak")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

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
# 4. TRAIN vs VALIDASI vs TEST  (cek stabilitas)
# ==============================================================================
def bandingkan_train_test(all_artifacts, splits, plot=True):
    """Metrik yang sama (threshold yang sama) di train, validasi, dan test.
    Status memakai selisih VALIDASI vs TEST (stabilitas threshold dan drift waktu).
    Skor train hanya informasi: pada KNN, titik latih adalah tetangga dirinya sendiri,
    sehingga skor train hampir pasti tinggi dan bukan bukti overfitting."""
    rows = []
    for run, art in all_artifacts.items():
        pipe, thr = art["pipeline"], art["Best_Threshold"]
        row = {"Run": run}
        for nama, key in [("Train", "train"), ("Val", "val"), ("Test", "test")]:
            if key == "val":
                prob = _val_probs(run, art, splits)
            else:
                prob = pipe.predict_proba(splits[f"X_{key}"])[:, 1]
            m = _metrik(splits[f"y_{key}"], prob, thr)
            row[f"F1 {nama}"] = m["F1"]
            row[f"PR-AUC {nama}"] = m["PR-AUC"]
            row[f"Precision {nama}"] = m["Precision"]
        row["Selisih F1 (Val-Test)"] = row["F1 Val"] - row["F1 Test"]
        row["Selisih PR-AUC (Val-Test)"] = row["PR-AUC Val"] - row["PR-AUC Test"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Status"] = np.where(
        (df["Selisih F1 (Val-Test)"].abs() > AMBANG_STABIL) |
        (df["Selisih PR-AUC (Val-Test)"].abs() > AMBANG_STABIL),
        "TIDAK STABIL", "stabil",
    )
    print("========== PERBANDINGAN TRAIN vs VALIDASI vs TEST ==========")
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
# 5. PEMILIHAN MODEL TERBAIK (dari VALIDASI)
# ==============================================================================
def pilih_model_terbaik(summary_df, train_test_df):
    """Pilih model dari metrik VALIDASI (Val F1, lalu Val PR-AUC). Test tidak dipakai untuk memilih."""
    tt = train_test_df[["Run", "F1 Val", "PR-AUC Val", "F1 Test", "PR-AUC Test",
                        "Selisih F1 (Val-Test)", "Selisih PR-AUC (Val-Test)", "Status"]]
    sm = summary_df[["Run", "Recall", "Precision", "Time (s)"]].rename(
        columns={"Recall": "Recall Test", "Precision": "Precision Test"})
    tabel = (tt.merge(sm, on="Run")
               .sort_values(["F1 Val", "PR-AUC Val"], ascending=False)
               .reset_index(drop=True))
    best = tabel.iloc[0]

    print("========== PEMILIHAN MODEL TERBAIK (berdasarkan VALIDASI) ==========")
    display(tabel.round(4))
    print(f"\nModel terpilih : {best['Run']}")
    print("Alasan:")
    print(f"  1. F1 validasi tertinggi ({best['F1 Val']:.4f}); PR-AUC validasinya {best['PR-AUC Val']:.4f}.")
    print(f"  2. Selisih validasi-test: F1 {best['Selisih F1 (Val-Test)']:.3f}, "
          f"PR-AUC {best['Selisih PR-AUC (Val-Test)']:.3f} -> status: {best['Status']}.")
    tidak_stabil = tabel[tabel["Status"] == "TIDAK STABIL"]["Run"].tolist()
    if tidak_stabil:
        print(f"  3. Model yang ditandai tidak stabil antara validasi dan test: {', '.join(tidak_stabil)}.")
    print("  Catatan: F1 validasi dihitung pada threshold yang dituning di validasi sendiri sehingga "
          "sedikit optimis; PR-AUC validasi bebas threshold dan lebih bersih untuk membandingkan model.")
    return best


# ==============================================================================
# 6. BASELINE + KRITERIA BERHASIL (untuk model TERPILIH)
# ==============================================================================
def bandingkan_baseline(all_artifacts, splits, best_run):
    """Bandingkan model terpilih (best_run dari pilih_model_terbaik) dengan tebakan sederhana."""
    y = splits["y_test"].values
    is_cod = splits["parts"]["Uji"]["is_cod"].values
    best_pred = all_artifacts[best_run]["y_preds"]

    def _row(nama, pred):
        return {
            "Pembanding": nama,
            "F1": f1_score(y, pred, zero_division=0),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
        }

    df = pd.DataFrame([
        _row("Baseline 1: semua pesanan ditebak Batal", np.ones(len(y), dtype=int)),
        _row("Baseline 2: COD = Batal, selain itu Selesai", is_cod.astype(int)),
        _row(f"Model terpilih ({best_run})", best_pred),
    ])
    print("========== MODEL TERPILIH vs BASELINE (DATA TEST) ==========")
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
    print("\n========== CEK KRITERIA BERHASIL (estimasi titik; lihat bootstrap_ci untuk ketidakpastian) ==========")
    display(cek)
    return df, cek


# ==============================================================================
# 7. ANALISIS KESALAHAN
# ==============================================================================
def analisis_kesalahan(best_run, all_artifacts, splits, bagian="Uji", n_contoh=10):
    """bagian='Uji' untuk pelaporan akhir. Pakai bagian='Validasi' kalau hasilnya dipakai
    untuk menambah fitur atau menyetel model, supaya data test tidak ikut terpakai."""
    art = all_artifacts[best_run]
    te = splits["parts"][bagian].copy()
    if bagian == "Validasi":
        te["prob"] = _val_probs(best_run, art, splits)
        te["aktual"] = splits["y_val"].values
    else:
        te["prob"] = art["y_probs"]
        te["aktual"] = splits["y_test"].values
    te["pred"] = (te["prob"] >= art["Best_Threshold"]).astype(int)
    te["luar_jawa"] = 1 - te["is_java"]

    fp = te[(te["aktual"] == 0) & (te["pred"] == 1)]
    fn = te[(te["aktual"] == 1) & (te["pred"] == 0)]
    tp = te[(te["aktual"] == 1) & (te["pred"] == 1)]
    tn = te[(te["aktual"] == 0) & (te["pred"] == 0)]

    kolom = ["order_id", "Metode Pembayaran", "shipping_tier", "Provinsi", "total_qty", "Perkiraan Ongkos Kirim"]
    print(f"========== CONTOH KESALAHAN ({best_run}, data {bagian}) ==========")
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
        f"Semua {bagian}": profil(te),
        "Batal terdeteksi (TP)": profil(tp),
        "Batal terlewat (FN)": profil(fn),
        "Salah alarm (FP)": profil(fp),
        "Selesai benar (TN)": profil(tn),
    }).T
    print("\n========== PROFIL CIRI-CIRI PER KELOMPOK ==========")
    display(prof.round(2))
    return fp, fn, prof


# ==============================================================================
# 8. KETIDAKPASTIAN: BOOTSTRAP CI (termasuk selisih model vs baseline)
# ==============================================================================
def _stat(y, prob, pred, cod):
    f1_m = f1_score(y, pred, zero_division=0)
    pr = average_precision_score(y, prob)
    return {
        "F1": f1_m,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "PR-AUC": pr,
        "PR-AUC lift (x prevalensi)": pr / y.mean(),
        "Selisih F1 vs 'semua Batal'": f1_m - f1_score(y, np.ones_like(y), zero_division=0),
        "Selisih F1 vs 'COD = Batal'": f1_m - f1_score(y, cod, zero_division=0),
    }


def bootstrap_ci(best_run, all_artifacts, splits, n_boot=1000, seed=42):
    """Bootstrap berpasangan pada data test (resample yang sama untuk model dan baseline)."""
    art = all_artifacts[best_run]
    y = splits["y_test"].values
    prob = np.asarray(art["y_probs"])
    pred = np.asarray(art["y_preds"])
    cod = splits["parts"]["Uji"]["is_cod"].values.astype(int)

    titik = _stat(y, prob, pred, cod)
    rng = np.random.default_rng(seed)
    hasil = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        hasil.append(_stat(y[idx], prob[idx], pred[idx], cod[idx]))
    b = pd.DataFrame(hasil)

    tabel = pd.DataFrame({
        "Estimasi": pd.Series(titik),
        "CI 95% bawah": b.quantile(0.025),
        "CI 95% atas": b.quantile(0.975),
    })
    tabel["Kesimpulan"] = ""
    for k in tabel.index:
        if k.startswith("Selisih"):
            lo, hi = tabel.loc[k, "CI 95% bawah"], tabel.loc[k, "CI 95% atas"]
            tabel.loc[k, "Kesimpulan"] = ("model lebih baik (CI > 0)" if lo > 0 else
                                          "baseline lebih baik (CI < 0)" if hi < 0 else
                                          "tidak berbeda nyata (CI memuat 0)")
    print(f"========== BOOTSTRAP 95% CI ({best_run}, {n_boot}x, data test) ==========")
    display(tabel.round(4))
    return tabel


# ==============================================================================
# 9. METRIK TAMBAHAN: F2, MCC, Brier, ECE
# ==============================================================================
def _ece(y, p, n_bins=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], q=n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), y_rata=("y", "mean"), p_rata=("p", "mean"))
    return float((g["n"] / g["n"].sum() * (g["y_rata"] - g["p_rata"]).abs()).sum())


def metrik_tambahan(all_artifacts, splits):
    """F2, MCC, Brier, ECE semua model pada data test (threshold dari validasi)."""
    y = splits["y_test"].values
    brier_dasar = brier_score_loss(y, np.full(len(y), splits["y_train"].mean()))
    rows = []
    for run, art in all_artifacts.items():
        p, pred = np.asarray(art["y_probs"]), np.asarray(art["y_preds"])
        rows.append({
            "Run": run,
            "F2": fbeta_score(y, pred, beta=2, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
            "Brier": brier_score_loss(y, p),
            "Brier dasar (prevalensi latih)": brier_dasar,
            "ECE": _ece(y, p),
        })
    df = pd.DataFrame(rows)
    print("========== METRIK TAMBAHAN (DATA TEST) ==========")
    display(df.round(4))
    return df


# ==============================================================================
# 10. STABILITAS ANTAR BULAN (data test)
# ==============================================================================
def stabilitas_bulanan(best_run, all_artifacts, splits):
    art = all_artifacts[best_run]
    te = splits["parts"]["Uji"]
    y = splits["y_test"].values
    p, pred = np.asarray(art["y_probs"]), np.asarray(art["y_preds"])
    rows = []
    for m in splits["test_months"]:
        mask = (te["order_year_month"] == m).values
        ym = y[mask]
        ap = average_precision_score(ym, p[mask]) if ym.sum() > 0 else np.nan
        rows.append({
            "Bulan": m, "Pesanan": int(mask.sum()), "Batal": int(ym.sum()),
            "Prevalensi (%)": ym.mean() * 100,
            "Precision": precision_score(ym, pred[mask], zero_division=0),
            "Recall": recall_score(ym, pred[mask], zero_division=0),
            "F1": f1_score(ym, pred[mask], zero_division=0),
            "PR-AUC": ap,
            "PR-AUC lift": ap / ym.mean() if ym.sum() > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    print(f"========== STABILITAS PER BULAN ({best_run}) ==========")
    display(df.round(4))
    return df


# ==============================================================================
# 11. TITIK OPERASI PR CURVE
# ==============================================================================
def tabel_titik_operasi(best_run, all_artifacts, splits):
    """Apakah Recall >= 0,70 dan Precision >= 0,30 mungkin tercapai bersamaan di threshold mana pun?"""
    art = all_artifacts[best_run]
    rows = []
    for nama, y, p in [("Validasi", splits["y_val"].values, _val_probs(best_run, art, splits)),
                       ("Test", splits["y_test"].values, art["y_probs"])]:
        prec, rec, _ = precision_recall_curve(y, p)
        prec, rec = prec[:-1], rec[:-1]
        ok_r, ok_p = rec >= TARGET_RECALL, prec >= TARGET_PRECISION
        rows.append({
            "Data": nama,
            "Prevalensi": y.mean(),
            f"Precision maks saat Recall >= {TARGET_RECALL}": prec[ok_r].max() if ok_r.any() else np.nan,
            f"Recall maks saat Precision >= {TARGET_PRECISION}": rec[ok_p].max() if ok_p.any() else np.nan,
            "Kedua target tercapai bersamaan?": "YA" if (ok_r & ok_p).any() else "TIDAK",
        })
    df = pd.DataFrame(rows)
    print(f"========== TITIK OPERASI PR CURVE ({best_run}) ==========")
    display(df.round(4))
    return df


# ==============================================================================
# 12. PREVALENSI PER SPLIT + PR-AUC RELATIF TERHADAP BASELINE-NYA
# ==============================================================================
def prevalensi_per_split(best_run, all_artifacts, splits):
    art = all_artifacts[best_run]
    rows = []
    for nama in ["Latih", "Validasi", "Uji"]:
        d = splits["parts"][nama]
        rows.append({"Split": nama,
                     "Periode": f"{d['order_year_month'].min()} s.d. {d['order_year_month'].max()}",
                     "Pesanan": len(d), "Batal": int(d["target_batal"].sum()),
                     "Prevalensi (%)": d["target_batal"].mean() * 100})
    df = pd.DataFrame(rows)
    ap = {"Validasi": average_precision_score(splits["y_val"], _val_probs(best_run, art, splits)),
          "Uji": average_precision_score(splits["y_test"], art["y_probs"])}
    df["PR-AUC"] = df["Split"].map(ap)
    df["PR-AUC / prevalensi (lift)"] = df["PR-AUC"] / (df["Prevalensi (%)"] / 100)
    print("========== PREVALENSI PER SPLIT ==========")
    display(df.round(4))
    return df


# ==============================================================================
# 13. PERMUTATION IMPORTANCE (di VALIDASI, bukan test)
# ==============================================================================
def importance_permutasi(best_run, all_artifacts, splits, n_repeats=3, top=15, seed=42):
    pipe = all_artifacts[best_run]["pipeline"]
    r = permutation_importance(pipe, splits["X_val"], splits["y_val"],
                               scoring="average_precision", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    df = (pd.DataFrame({"Fitur": splits["X_val"].columns,
                        "Penurunan PR-AUC": r.importances_mean,
                        "Std": r.importances_std})
          .sort_values("Penurunan PR-AUC", ascending=False).reset_index(drop=True))
    d = df.head(top).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(d["Fitur"], d["Penurunan PR-AUC"], xerr=d["Std"])
    plt.xlabel("Penurunan PR-AUC validasi saat fitur diacak")
    plt.title(f"Permutation Importance ({best_run})")
    plt.tight_layout(); plt.show()
    display(df.head(top).round(4))
    return df


# ==============================================================================
# 14. CROSS-VALIDATION BERBASIS WAKTU (di data latih; lambat untuk SVM)
# ==============================================================================
def cv_waktu(all_artifacts, splits, n_splits=3, runs=None):
    rows = []
    for run, art in all_artifacts.items():
        if runs is not None and run not in runs:
            continue
        skor = cross_val_score(clone(art["pipeline"]), splits["X_train"], splits["y_train"],
                               scoring="average_precision", cv=TimeSeriesSplit(n_splits=n_splits), n_jobs=1)
        rows.append({"Run": run, "PR-AUC CV rata-rata": skor.mean(), "PR-AUC CV std": skor.std(),
                     **{f"Fold {i + 1}": s for i, s in enumerate(skor)}})
    df = pd.DataFrame(rows)
    print("========== CROSS-VALIDATION BERBASIS WAKTU (TimeSeriesSplit, PR-AUC) ==========")
    display(df.round(4))
    return df


# ==============================================================================
# JALANKAN SEMUA
# ==============================================================================
def run_evaluation(summary_df, all_artifacts, splits):
    tabel_metrik(summary_df, all_artifacts, splits)
    plot_confusion_matrix(all_artifacts)
    plot_kurva(all_artifacts, splits)
    tt = bandingkan_train_test(all_artifacts, splits)
    best = pilih_model_terbaik(summary_df, tt)
    bandingkan_baseline(all_artifacts, splits, best["Run"])
    bootstrap_ci(best["Run"], all_artifacts, splits)
    metrik_tambahan(all_artifacts, splits)
    stabilitas_bulanan(best["Run"], all_artifacts, splits)
    tabel_titik_operasi(best["Run"], all_artifacts, splits)
    prevalensi_per_split(best["Run"], all_artifacts, splits)
    importance_permutasi(best["Run"], all_artifacts, splits)
    analisis_kesalahan(best["Run"], all_artifacts, splits)