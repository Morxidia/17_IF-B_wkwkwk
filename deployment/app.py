import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import streamlit as st

from deployment.bundle import load_bundle
from deployment.features import build_features, ALL_FEATURES
from deployment.schema import RAW_REQUIRED, validate_raw_columns
from deployment.monitoring import log_predictions, calculate_psi, evaluate_labeled, LOG_PATH

st.set_page_config(page_title="Deteksi Risiko Pembatalan", page_icon="📦", layout="wide")
st.title("Dashboard Deteksi Risiko Pembatalan Pesanan")
st.caption("Model klasifikasi status pesanan — SVM RBF | Run_2_SVM")

@st.cache_resource
def get_model():
    return load_bundle()

try:
    model, metadata = get_model()
except Exception as exc:
    st.error(str(exc))
    st.info("Ikuti petunjuk ekspor model di DEPLOYMENT.md. Model training kelompok tidak dibuat ulang oleh aplikasi ini.")
    st.stop()

threshold = float(metadata.get("threshold", 0.28))
top_categories = metadata.get("top_categories", [])
tabs = st.tabs(["Prediksi satu pesanan", "Prediksi CSV", "Monitoring", "Informasi model"])

def show_results(raw, X):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    out = raw.copy().reset_index(drop=True)
    out["Probabilitas batal (%)"] = (prob * 100).round(2)
    out["Prediksi"] = np.where(pred == 1, "Batal (berisiko)", "Selesai (risiko lebih rendah)")
    out["Kategori risiko"] = np.where(
        prob >= threshold, "Perlu ditinjau", "Risiko lebih rendah"
    )
    log_predictions(raw, X, prob, threshold, pred)
    return out, prob, pred

with tabs[0]:
    st.subheader("Masukkan atribut pesanan saat checkout")
    st.caption("Isi data yang tersedia saat pesanan dibuat. Jangan masukkan status akhir atau informasi setelah pesanan diproses.")
    with st.form("single_order"):
        c1, c2 = st.columns(2)
        with c1:
            order_id = st.text_input("ID pesanan (opsional)", value="DEMO-001")
            payment = st.selectbox("Metode pembayaran", [
                "COD (Bayar di Tempat)", "Saldo ShopeePay", "Online Payment",
                "SPayLater", "SeaBank", "Kartu Kredit/Debit", "Lainnya"
            ])
            shipping = st.selectbox("Opsi pengiriman", [
                "Reguler", "Hemat", "Kargo", "Instant", "Same Day", "Next Day", "Lainnya"
            ])
            province = st.text_input("Provinsi tujuan", "JAWA BARAT")
            city = st.text_input("Kota/Kabupaten", "BANDUNG")
            timestamp = st.text_input("Waktu pesanan dibuat", "2025-11-22 10:30:00")
        with c2:
            discount = st.number_input("Total diskon (Rp)", min_value=0.0, value=0.0, step=1000.0)
            qty = st.number_input("Total jumlah barang", min_value=1, value=1, step=1)
            weight = st.number_input("Total berat (gram)", min_value=1.0, value=500.0, step=50.0)
            ncat = st.number_input("Jumlah kategori produk", min_value=1, value=1, step=1)
            category = st.text_input("Kategori produk", "Celengan")
            shipping_cost = st.number_input("Perkiraan ongkos kirim (Rp)", min_value=0.0, value=15000.0, step=1000.0)
            repeat_count = st.number_input("Pesanan sebelumnya pada tanggal/kota/kategori yang sama", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Prediksi risiko")
    if submitted:
        raw = pd.DataFrame([{
            "order_id": order_id, "Metode Pembayaran": payment,
            "Opsi Pengiriman": shipping, "Provinsi": province,
            "Total Diskon": discount, "total_qty": qty,
            "total_weight_gr": weight, "num_product_categories": ncat,
            "product_categories": category, "Perkiraan Ongkos Kirim": shipping_cost,
            "Waktu Pesanan Dibuat": timestamp, "Kota/Kabupaten": city,
            "same_day_repeat_count": repeat_count,
        }])
        try:
            X = build_features(raw, top_categories)
            result, prob, pred = show_results(raw, X)
            st.metric("Skor model untuk kelas Batal", f"{prob[0]*100:.2f}%")
            if pred[0] == 1:
                st.warning("Pesanan perlu ditinjau lebih lanjut. Ini bukan keputusan pembatalan otomatis.")
            else:
                st.success("Risiko lebih rendah menurut threshold model.")
            st.dataframe(result[["order_id", "Probabilitas batal (%)", "Prediksi", "Kategori risiko"]], use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memproses input: {e}")

with tabs[1]:
    st.subheader("Prediksi banyak pesanan melalui CSV")
    st.write("CSV harus berisi seluruh kolom wajib berikut. `order_id` dan `same_day_repeat_count` opsional.")
    st.code(", ".join(RAW_REQUIRED), language="text")
    uploaded = st.file_uploader("Unggah CSV pesanan", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            validate_raw_columns(raw.columns)
            X = build_features(raw, top_categories)
            result, _, _ = show_results(raw, X)
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Unduh hasil prediksi CSV",
                result.to_csv(index=False).encode("utf-8-sig"),
                file_name="hasil_prediksi_pesanan.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"CSV tidak dapat diproses: {e}")

with tabs[2]:
    st.subheader("Monitoring penggunaan model")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        a, b, c = st.columns(3)
        a.metric("Total prediksi tercatat", len(logs))
        b.metric("Proporsi ditandai berisiko", f"{logs['prediksi_batal'].mean()*100:.1f}%")
        c.metric("Rata-rata skor batal", f"{logs['probabilitas_batal'].mean()*100:.1f}%")
        st.dataframe(logs.tail(20), use_container_width=True)
        st.download_button("Unduh log monitoring", logs.to_csv(index=False).encode("utf-8-sig"),
                           "prediction_log.csv", "text/csv")
        st.markdown("**Evaluasi setelah label aktual tersedia**")
        st.caption("Isi kolom `actual_target` pada salinan log: 1 = Batal, 0 = Selesai. Unggah kembali untuk menghitung metrik.")
        labels_file = st.file_uploader("Unggah log dengan actual_target", type=["csv"], key="labels")
        if labels_file:
            labeled = pd.read_csv(labels_file)
            metrics = evaluate_labeled(labeled)
            if metrics:
                st.json({k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()})
            else:
                st.info("Belum ada label aktual yang bisa dievaluasi.")
    else:
        st.info("Belum ada prediksi. Gunakan tab prediksi untuk mulai membuat log.")

    st.markdown("**Rencana pemantauan drift**")
    st.write("Bandingkan distribusi fitur prediksi terbaru dengan distribusi data latih. PSI < 0,10: relatif stabil; 0,10–0,25: peringatan; > 0,25: investigasi.")
    reference_path = Path(__file__).resolve().parents[1] / "models" / "reference_numeric.json"
    if reference_path.exists() and LOG_PATH.exists():
        ref = json.loads(reference_path.read_text(encoding="utf-8"))
        rows = []
        # Log menyimpan kolom mentah; fitur turunan dihitung ulang dari baris log.
        try:
            from deployment.features import build_features
            raw_cols = [c for c in RAW_REQUIRED if c in logs.columns]
            feat = build_features(logs[raw_cols + (["same_day_repeat_count"] if "same_day_repeat_count" in logs else [])], top_categories)
            for col, values in ref.items():
                if col in feat:
                    psi = calculate_psi(values, feat[col])
                    rows.append({"Fitur": col, "PSI": psi,
                                 "Status": "Investigasi" if psi > .25 else ("Peringatan" if psi >= .10 else "Stabil")})
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("PSI", ascending=False), use_container_width=True)
        except Exception as e:
            st.caption(f"Drift belum dapat dihitung: {e}")
    else:
        st.caption("Referensi distribusi data latih akan tersedia setelah artefak diekspor dari notebook.")

with tabs[3]:
    st.subheader("Informasi model")
    st.write("Model terpilih: **Run_2_SVM** — Support Vector Machine dengan kernel RBF.")
    st.write(f"Threshold prediksi: **{threshold:.2f}**")
    st.write("Fitur model: " + ", ".join(metadata.get("features", ALL_FEATURES)))
    st.write("Top kategori produk dipelajari hanya dari bagian data latih; kategori lain dipetakan ke `Other_Category`.")
    st.warning("Skor probabilitas belum dikalibrasi. Gunakan sebagai indikator/ranking risiko untuk pemeriksaan manusia, bukan probabilitas pasti atau dasar pembatalan otomatis.")
    st.write("Catatan: `same_day_repeat_count` tidak bisa dihitung sempurna untuk pesanan tunggal tanpa riwayat pesanan sebelumnya. Nilai default adalah 0; untuk penggunaan operasional, hubungkan dengan basis data transaksi.")
