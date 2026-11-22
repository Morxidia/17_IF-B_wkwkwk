"""Feature engineering untuk serving; diselaraskan dengan src/preprocessing.py."""
import numpy as np
import pandas as pd

JABODETABEK_JABAR_BANTEN = {"BANTEN", "DKI JAKARTA", "JAWA BARAT"}
JATENG_DIY_JATIM = {"JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR"}
JAVA_PROVINCES = JABODETABEK_JABAR_BANTEN | JATENG_DIY_JATIM
SUMATERA_PROVINCES = {
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "KEPULAUAN RIAU",
    "JAMBI", "SUMATERA SELATAN", "KEPULAUAN BANGKA BELITUNG", "BENGKULU", "LAMPUNG"
}

NUM_FEATURES = [
    "log_total_qty", "log_total_weight_gr", "log_Total Diskon",
    "log_Perkiraan Ongkos Kirim", "log_weight_per_item", "log_shipping_per_kg",
    "num_product_categories", "order_hour", "order_dayofweek",
    "is_missing_timestamp", "is_late_night", "is_cod", "is_java",
    "has_discount", "is_multi_category", "same_day_repeat_count"
]
CAT_FEATURES = [
    "Metode Pembayaran", "shipping_tier", "destination_zone",
    "Provinsi", "top_product_cat"
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

def map_destination_zone(prov):
    p = str(prov).strip().upper()
    if p in JABODETABEK_JABAR_BANTEN:
        return "1. Jabodetabek, Jabar & Banten"
    if p in JATENG_DIY_JATIM:
        return "2. Jawa Tengah, DIY & Jawa Timur"
    if p in SUMATERA_PROVINCES:
        return "3. Pulau Sumatera"
    return "4. Kalimantan, Sulawesi, Bali-Nusra & Timur"

def map_shipping_tier(opsi):
    s = str(opsi).lower()
    if "hemat" in s or "kargo" in s:
        return "Hemat / Kargo"
    if "reguler" in s:
        return "Reguler"
    if "instant" in s or "same day" in s or "next day" in s:
        return "Instant / Same Day"
    return "Ekonomi / Lainnya"

def build_features(raw_df, top_categories):
    """Terima kolom mentah saat checkout dan hasilkan fitur sesuai urutan training."""
    from deployment.schema import validate_raw_columns
    validate_raw_columns(raw_df.columns)
    d = raw_df.copy()

    numeric = [
        "Total Diskon", "total_qty", "total_weight_gr",
        "num_product_categories", "Perkiraan Ongkos Kirim"
    ]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d[numeric] = d[numeric].fillna(0)

    dt = pd.to_datetime(d["Waktu Pesanan Dibuat"], errors="coerce")
    d["is_missing_timestamp"] = dt.isna().astype(int)
    d["order_hour"] = dt.dt.hour.fillna(-1).astype(int)
    d["order_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    d["is_late_night"] = dt.dt.hour.isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

    d["is_cod"] = d["Metode Pembayaran"].astype(str).str.contains(
        "COD", case=False, na=False
    ).astype(int)
    d["is_java"] = d["Provinsi"].astype(str).str.upper().isin(JAVA_PROVINCES).astype(int)
    d["destination_zone"] = d["Provinsi"].apply(map_destination_zone)
    d["shipping_tier"] = d["Opsi Pengiriman"].apply(map_shipping_tier)
    d["has_discount"] = (d["Total Diskon"] > 0).astype(int)
    d["is_multi_category"] = (d["num_product_categories"] > 1).astype(int)

    d["avg_item_weight_gr"] = d["total_weight_gr"] / d["total_qty"].clip(lower=1)
    d["shipping_per_kg"] = d["Perkiraan Ongkos Kirim"] / (
        d["total_weight_gr"] / 1000.0
    ).clip(lower=0.1)

    # Pada satu pesanan, nilai ini tidak dapat diketahui dari pesanan lain yang belum
    # tercatat. Input manual diperbolehkan; default 0 sama dengan asumsi tidak ada
    # pesanan terdahulu dalam grup tanggal-kota-kategori.
    if "same_day_repeat_count" not in d:
        d["same_day_repeat_count"] = 0
    d["same_day_repeat_count"] = pd.to_numeric(
        d["same_day_repeat_count"], errors="coerce"
    ).fillna(0)

    d["log_total_qty"] = np.log1p(d["total_qty"].clip(lower=0))
    d["log_total_weight_gr"] = np.log1p(d["total_weight_gr"].clip(lower=0))
    d["log_Total Diskon"] = np.log1p(d["Total Diskon"].clip(lower=0))
    d["log_Perkiraan Ongkos Kirim"] = np.log1p(
        d["Perkiraan Ongkos Kirim"].clip(lower=0)
    )
    d["log_weight_per_item"] = np.log1p(d["avg_item_weight_gr"].clip(lower=0))
    d["log_shipping_per_kg"] = np.log1p(d["shipping_per_kg"].clip(lower=0))

    known = set(top_categories or [])
    d["top_product_cat"] = d["product_categories"].astype(str).where(
        d["product_categories"].astype(str).isin(known), "Other_Category"
    )
    return d[ALL_FEATURES].copy()
