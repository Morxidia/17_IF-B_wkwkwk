# Deployment — Deteksi Risiko Pembatalan Pesanan

## 1. Tujuan dan batasan

Antarmuka menggunakan Streamlit untuk menjalankan model terpilih dari notebook, yaitu **Run_2_SVM** (SVM kernel RBF, `C=0.5`, `gamma=auto`) dengan threshold validasi **0,28**. Aplikasi tidak melatih ulang atau mengganti model. Artefak model harus diekspor dari objek pipeline asli di notebook.

Hasil test pada dokumentasi proyek: F1 0,14; Precision 0,32; Recall 0,09; PR-AUC 0,17; ROC-AUC 0,61. Model belum memenuhi target bisnis dan skor belum dikalibrasi. Karena itu aplikasi hanya memberi indikasi untuk pemeriksaan manusia, bukan membatalkan pesanan otomatis.

## 2. Struktur folder

```text
repo/
├── main.ipynb
├── src/
│   ├── preprocessing.py
│   └── modeling.py
├── deployment/
│   ├── app.py
│   ├── schema.py
│   ├── features.py
│   ├── bundle.py
│   └── monitoring.py
├── models/
│   ├── model_artifact.joblib
│   ├── metadata.json
│   └── reference_numeric.json
├── monitoring/
│   └── prediction_log.csv   # dibuat otomatis saat aplikasi dipakai
├── requirements-deploy.txt
└── DEPLOYMENT.md
```

Folder `src/`, notebook, dan file evaluasi yang sudah ada tidak perlu diubah.

## 3. Ekspor model asli dari notebook

Buka `main.ipynb`, jalankan cell sampai bagian evaluasi/pemilihan model selesai dan pastikan variabel berikut tersedia:
- `best_pipeline`: pipeline dari `Run_2_SVM`
- `best_thresh`: threshold `Run_2_SVM` (0,28)
- `splits`: hasil `prep.get_temporal_splits(df)`

Pada notebook yang disertakan, cell evaluasi mengambil `best_pipeline` dan `best_thresh` dari `best_art` pada model dengan peringkat teratas. Jalankan cell berikut setelah cell tersebut:

```python
from pathlib import Path
import json
import joblib
import numpy as np

# Pastikan objek yang diekspor benar-benar Run_2_SVM.
assert best_run == "Run_2_SVM", f"Model terpilih tidak sesuai: {best_run}"
assert best_pipeline is not None

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Simpan pipeline SVM yang sudah dilatih; tidak melakukan fit ulang.
joblib.dump(best_pipeline, model_dir / "model_artifact.joblib")

# Daftar kategori top-20 dihitung hanya dari data latih, sama seperti preprocessing.
top_categories = (
    splits["parts"]["Latih"]["product_categories"]
    .value_counts()
    .head(20)
    .index.astype(str)
    .tolist()
)

metadata = {
    "model_name": "Run_2_SVM",
    "algorithm": "SVC",
    "kernel": "rbf",
    "C": 0.5,
    "gamma": "auto",
    "threshold": float(best_thresh),
    "top_categories": top_categories,
    "features": list(splits["all_features"]),
    "test_metrics": {
        "F1": 0.14,
        "Precision": 0.32,
        "Recall": 0.09,
        "PR_AUC": 0.17,
        "ROC_AUC": 0.61
    },
    "probability_calibrated": False
}
(model_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

# Simpan referensi fitur numerik dari data latih untuk monitoring PSI.
reference = {}
for col in splits["num_cols"]:
    reference[col] = (
        splits["X_train"][col].dropna().astype(float).tolist()
    )
(model_dir / "reference_numeric.json").write_text(
    json.dumps(reference), encoding="utf-8"
)

print("Artefak Run_2_SVM berhasil diekspor.")
```

> Jangan menjalankan cell ini sebelum variabel `best_run`, `best_pipeline`, `best_thresh`, dan `splits` terbentuk. Ekspor ini hanya menyimpan model yang sudah ada; tidak menimpa notebook atau model teman.

## 4. Instalasi dan menjalankan aplikasi

Dari terminal pada folder utama repository:

```bash
pip install -r requirements-deploy.txt
streamlit run deployment/app.py
```

Streamlit akan membuka alamat lokal di browser. Jika perintah `streamlit` tidak dikenali, aktifkan virtual environment terlebih dahulu lalu ulangi instalasi.

## 5. Skema I/O

### Input mentah wajib

| Kolom | Keterangan |
|---|---|
| `Metode Pembayaran` | Metode pembayaran ketika checkout |
| `Opsi Pengiriman` | Layanan pengiriman yang dipilih |
| `Provinsi` | Provinsi tujuan |
| `Total Diskon` | Total diskon toko |
| `total_qty` | Total unit barang |
| `total_weight_gr` | Total berat barang dalam gram |
| `num_product_categories` | Jumlah kategori dalam pesanan |
| `product_categories` | Kategori produk |
| `Perkiraan Ongkos Kirim` | Estimasi ongkos kirim |
| `Waktu Pesanan Dibuat` | Timestamp pembuatan pesanan |
| `Kota/Kabupaten` | Kota/kabupaten tujuan |

Kolom opsional: `order_id` dan `same_day_repeat_count`. Untuk input tunggal, `same_day_repeat_count` default 0 jika tidak diberikan. Dalam sistem operasional, fitur ini sebaiknya dihitung dari riwayat pesanan yang tersedia saat checkout.

Kolom target atau pasca-kejadian seperti `Status Pesanan`, `Alasan Pembatalan`, `total_returned_qty`, `Total Pembayaran`, `Ongkos Kirim Dibayar`, dan `Estimasi Potongan Biaya Pengiriman` ditolak sebagai input untuk mencegah data leakage.

### Output

- `probabilitas_batal`: skor keluaran `predict_proba` untuk kelas Batal.
- `threshold`: ambang klasifikasi dari validasi (0,28).
- `prediksi`: Batal (berisiko) jika skor >= threshold; selain itu Selesai (risiko lebih rendah).
- `kategori_risiko`: label ringkas untuk tindak lanjut.

Skor belum dikalibrasi; angka persentase adalah skor model, bukan jaminan probabilitas aktual.

## 6. Monitoring dan deteksi drift

Aplikasi menyimpan prediksi ke `monitoring/prediction_log.csv`, termasuk waktu prediksi, skor, threshold, dan hasil klasifikasi. Tidak ada label aktual yang diasumsikan tersedia langsung.

| Frekuensi | Hal yang dipantau | Tindakan |
|---|---|---|
| Harian | Jumlah prediksi, error input, proporsi pesanan yang ditandai berisiko | Periksa bila input gagal atau volume berubah tidak wajar |
| Mingguan | Perubahan distribusi fitur numerik dengan Population Stability Index (PSI) | PSI < 0,10 stabil; 0,10–0,25 peringatan; > 0,25 investigasi |
| Bulanan | Precision, Recall, F1, PR-AUC, Brier setelah label aktual tersedia | Bandingkan dengan hasil evaluasi dan target bisnis |
| Berkala | Perubahan proses checkout, metode bayar, kurir, atau wilayah layanan | Tinjau ulang relevansi fitur dan kebutuhan pelatihan ulang |

Label aktual dapat ditambahkan pada kolom `actual_target` di salinan log (`1` = Batal, `0` = Selesai), lalu diunggah melalui tab Monitoring. Evaluasi kinerja hanya bermakna jika label aktual sudah tersedia dan jumlah sampel memadai. PSI adalah sinyal perubahan distribusi, bukan bukti bahwa performa model pasti turun.

## 7. Siklus retraining

1. Kumpulkan prediksi dan label aktual setelah status pesanan final tersedia.
2. Evaluasi metrik secara berkala pada data berlabel, terutama Recall, Precision, F1, PR-AUC, dan Brier.
3. Investigasi jika PSI > 0,25, performa menurun secara konsisten, atau proses bisnis berubah.
4. Siapkan dataset terbaru dengan aturan target dan pencegahan leakage yang sama.
5. Latih kandidat model menggunakan pemisahan temporal dan tuning hanya pada data latih/validasi.
6. Bandingkan kandidat dengan model aktif pada periode evaluasi yang sama; jangan memilih model hanya dari data test.
7. Lakukan review manusia dan dokumentasikan versi, parameter, fitur, threshold, serta metrik.
8. Promosikan model baru hanya setelah lolos evaluasi; simpan artefak lama untuk rollback.

Retraining tidak dijalankan otomatis oleh dashboard. Proses ini sengaja dipisahkan agar model aktif tidak berubah tanpa evaluasi dan persetujuan.

## 8. Keterbatasan implementasi

- Dataset dan model training tidak disertakan sebagai bagian dari paket deployment; pipeline harus diekspor dari notebook yang sudah dijalankan.
- `same_day_repeat_count` membutuhkan riwayat pesanan berdasarkan tanggal, kota/kabupaten, dan kategori. Nilai default 0 pada demo tidak sepenuhnya merepresentasikan riwayat toko.
- Input CSV harus mengikuti skema kolom mentah yang ditentukan.
- File log lokal cocok untuk demonstrasi satu lingkungan, bukan sistem multi-user/produksi.
- Model belum memenuhi target bisnis (Recall test 0,09) sehingga penggunaan dibatasi sebagai alat bantu analisis.
