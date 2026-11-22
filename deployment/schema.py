"""Skema input dan output untuk layanan prediksi risiko pembatalan."""
RAW_REQUIRED = [
    "Metode Pembayaran", "Opsi Pengiriman", "Provinsi",
    "Total Diskon", "total_qty", "total_weight_gr",
    "num_product_categories", "product_categories",
    "Perkiraan Ongkos Kirim", "Waktu Pesanan Dibuat",
    "Kota/Kabupaten",
]

RAW_OPTIONAL = ["order_id", "same_day_repeat_count"]

LEAKAGE_COLUMNS = [
    "Status Pesanan", "Alasan Pembatalan", "total_returned_qty",
    "Total Pembayaran", "Ongkos Kirim Dibayar",
    "Estimasi Potongan Biaya Pengiriman", "target_batal",
]

OUTPUT_COLUMNS = [
    "order_id", "probabilitas_batal", "threshold",
    "prediksi", "kategori_risiko",
]

def validate_raw_columns(columns):
    cols = set(columns)
    forbidden = sorted(cols.intersection(LEAKAGE_COLUMNS))
    missing = sorted(set(RAW_REQUIRED) - cols)
    if forbidden:
        raise ValueError(
            "Kolom yang berpotensi menyebabkan data leakage tidak boleh menjadi input: "
            + ", ".join(forbidden)
        )
    if missing:
        raise ValueError("Kolom input belum lengkap: " + ", ".join(missing))
    return True
