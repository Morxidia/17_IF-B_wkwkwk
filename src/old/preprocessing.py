import numpy as np
import pandas as pd

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

JABODETABEK_JABAR_BANTEN = {"BANTEN", "DKI JAKARTA", "JAWA BARAT"}
JATENG_DIY_JATIM = {"JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR"}
JAVA_PROVINCES = JABODETABEK_JABAR_BANTEN | JATENG_DIY_JATIM
SUMATERA_PROVINCES = {
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "KEPULAUAN RIAU",
    "JAMBI", "SUMATERA SELATAN", "KEPULAUAN BANGKA BELITUNG", "BENGKULU", "LAMPUNG"
}

FEATURE_GROUPS = {
    "Pembayaran": ["Metode Pembayaran", "Total Diskon", "log_Total Diskon", "is_cod", "has_discount"],
    "Pengiriman": [
        "shipping_tier", "destination_zone", "Provinsi", "is_java",
        "Perkiraan Ongkos Kirim", "log_Perkiraan Ongkos Kirim",
        "total_weight_gr", "log_total_weight_gr",
        "avg_item_weight_gr", "log_weight_per_item", "log_shipping_per_kg"
    ],
    "Keranjang": [
        "top_product_cat", "total_qty", "log_total_qty",
        "num_product_categories", "is_multi_category"
    ],
    "Waktu": [
        "order_hour", "order_dayofweek", "is_late_night",
        "is_missing_timestamp", "same_day_repeat_count"
    ],
}

CAT_FEATURES = ["Metode Pembayaran", "shipping_tier", "destination_zone", "Provinsi", "top_product_cat"]
ALL_FEATURES = [col for cols in FEATURE_GROUPS.values() for col in cols]
NUM_FEATURES = [col for col in ALL_FEATURES if col not in CAT_FEATURES]

FORBIDDEN_FEATURES = [
    "Status Pesanan", "Alasan Pembatalan", "total_returned_qty",
    "Total Pembayaran", "Ongkos Kirim Dibayar oleh Pembeli",
    "Estimasi Potongan Biaya Pengiriman", "order_id", "order_year_month", "target_batal"
]


def map_destination_zone(prov: str) -> str:
    p = str(prov).strip().upper()
    if p in JABODETABEK_JABAR_BANTEN:
        return "1. Jabodetabek, Jabar & Banten"
    elif p in JATENG_DIY_JATIM:
        return "2. Jawa Tengah, DIY & Jawa Timur"
    elif p in SUMATERA_PROVINCES:
        return "3. Pulau Sumatera"
    return "4. Kalimantan, Sulawesi, Bali-Nusra & Timur"


def map_shipping_tier(opsi: str) -> str:
    s = str(opsi).lower()
    if "hemat" in s or "kargo" in s:
        return "Hemat / Kargo"
    elif "reguler" in s:
        return "Reguler"
    elif "instant" in s or "same day" in s or "next day" in s:
        return "Instant / Same Day"
    return "Ekonomi / Lainnya"


def load_and_clean_data(filepath: str = "./dataset/all_months_clean.csv") -> pd.DataFrame:
    """Loads raw CSV, filters binary target, removes leakage, and engineers t0 features."""
    df = pd.read_csv(filepath)

    # 1. Binary target: Batal = 1, Selesai = 0
    df = df[df["Status Pesanan"].isin(["Selesai", "Batal"])].copy()
    df["target_batal"] = (df["Status Pesanan"] == "Batal").astype(int)

    # 2. Recover Year-Month from source_file & sort chronologically
    extracted = df["source_file"].astype(str).str.extract(r"([A-Za-z]+)Sales(\d{4})")
    df["order_year"] = extracted[1].fillna("2024").astype(int)
    df["order_month"] = extracted[0].map(MONTH_MAP).fillna(1).astype(int)
    df["order_year_month"] = df["order_year"].astype(str) + "-" + df["order_month"].astype(str).str.zfill(2)

    dt_raw = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    fallback_dt = pd.to_datetime(df["order_year_month"] + "-15")
    df["dt_sort"] = dt_raw.fillna(fallback_dt)
    df = df.sort_values(["order_year_month", "dt_sort", "order_id"], kind="stable").reset_index(drop=True)

    # 3. Temporal features + missing timestamp flag
    dt = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    df["is_missing_timestamp"] = dt.isna().astype(int)
    df["order_hour"] = dt.dt.hour.fillna(-1).astype(int)
    df["order_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["is_late_night"] = dt.dt.hour.isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

    # 4. Valid checkout (t0) domain features
    df["is_cod"] = df["Metode Pembayaran"].str.contains("COD", case=False, na=False).astype(int)
    df["is_java"] = df["Provinsi"].str.upper().isin(JAVA_PROVINCES).astype(int)
    df["destination_zone"] = df["Provinsi"].apply(map_destination_zone)
    df["shipping_tier"] = df["Opsi Pengiriman"].apply(map_shipping_tier)
    # Deleted, karena udah ada shipping_tier
    # df["courier_group"] = df["Opsi Pengiriman"].astype(str).str.split("-").str[0].str.strip()
    df["has_discount"] = (df["Total Diskon"] > 0).astype(int)
    df["is_multi_category"] = (df["num_product_categories"] > 1).astype(int)

    df["avg_item_weight_gr"] = df["total_weight_gr"] / df["total_qty"].clip(lower=1)
    df["shipping_per_kg"] = df["Perkiraan Ongkos Kirim"] / (df["total_weight_gr"] / 1000.0).clip(lower=0.1)

    top_20_cats = df["product_categories"].value_counts().head(20).index
    df["top_product_cat"] = np.where(
        df["product_categories"].isin(top_20_cats),
        df["product_categories"],
        "Other_Category"
    )

    df["order_date_str"] = dt.dt.date.astype(str)
    df["same_day_repeat_count"] = np.where(
        dt.isna(),
        0,
        df.groupby(["order_date_str", "Kota/Kabupaten", "product_categories"]).cumcount()
    )

    # 5. Log1p transform right-skewed columns
    log_map = {
        "total_qty": "log_total_qty",
        "total_weight_gr": "log_total_weight_gr",
        "Total Diskon": "log_Total Diskon",
        "Perkiraan Ongkos Kirim": "log_Perkiraan Ongkos Kirim",
        "avg_item_weight_gr": "log_weight_per_item",
        "shipping_per_kg": "log_shipping_per_kg",
    }
    for raw_col, log_col in log_map.items():
        df[log_col] = np.log1p(df[raw_col].clip(lower=0))

    assert not (set(ALL_FEATURES) & set(FORBIDDEN_FEATURES)), "Forbidden leakage feature detected!"
    return df


def get_temporal_splits(df: pd.DataFrame, val_months_count: int = 3, test_months_count: int = 3) -> dict:
    """
    Splits chronologically by month:
    last `test_months_count` months = Test,
    previous `val_months_count` months = Validation,
    remaining earlier months = Train.
    """
    data = df.sort_values(["order_year_month", "order_id"], kind="stable").reset_index(drop=True)
    months = sorted(data["order_year_month"].unique())

    train_months = months[:-(val_months_count + test_months_count)]
    val_months = months[-(val_months_count + test_months_count):-test_months_count]
    test_months = months[-test_months_count:]

    parts = {
        "Latih": data[data["order_year_month"].isin(train_months)],
        "Validasi": data[data["order_year_month"].isin(val_months)],
        "Uji": data[data["order_year_month"].isin(test_months)],
    }

    split_tbl = pd.DataFrame([{
        "Subset": name,
        "Periode": f"{p['order_year_month'].min()} s.d. {p['order_year_month'].max()}",
        "Pesanan": len(p),
        "Batal": int(p["target_batal"].sum()),
        "Tingkat Batal (%)": round(float(p["target_batal"].mean()) * 100, 2),
    } for name, p in parts.items()])

    return {
        "X_train": parts["Latih"][ALL_FEATURES],
        "y_train": parts["Latih"]["target_batal"],
        "X_val": parts["Validasi"][ALL_FEATURES],
        "y_val": parts["Validasi"]["target_batal"],
        "X_test": parts["Uji"][ALL_FEATURES],
        "y_test": parts["Uji"]["target_batal"],
        "parts": parts,
        "test_months": test_months,
        "split_summary": split_tbl,
        "num_cols": NUM_FEATURES,
        "cat_cols": CAT_FEATURES,
        "all_features": ALL_FEATURES,
    }