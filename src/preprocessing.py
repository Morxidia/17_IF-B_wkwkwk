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

NUMERIC_FEATURES = [
    "log_total_qty",
    "log_total_weight_gr",
    "log_Total Diskon",
    "num_product_categories",
    "log_Perkiraan Ongkos Kirim",
    "log_weight_per_item",
    "log_shipping_per_kg",
    "is_missing_waktu",
    "order_hour",
    "order_dayofweek",
    "is_late_night",
    "is_cod",
    "is_java",
    "has_discount",
    "is_multi_category",
    "same_day_repeat_count"
]

CATEGORICAL_FEATURES = [
    "Metode Pembayaran",
    "courier_group",
    "Provinsi",
    "destination_zone",
    "top_product_cat"
]


def map_destination_zone(prov: str) -> str:
    """Maps 34 Indonesian provinces into 4 BPS geographic zones."""
    p = str(prov).strip().upper()
    if p in JABODETABEK_JABAR_BANTEN:
        return "1. Jabodetabek, Jabar & Banten"
    elif p in JATENG_DIY_JATIM:
        return "2. Jawa Tengah, DIY & Jawa Timur"
    elif p in SUMATERA_PROVINCES:
        return "3. Pulau Sumatera"
    return "4. Kalimantan, Sulawesi, Bali-Nusra & Timur"


def load_and_clean_data(filepath: str = "./dataset/all_months_clean.csv") -> pd.DataFrame:
    """Loads raw CSV, filters binary target, removes leakage, and engineers t0 features."""
    df = pd.read_csv(filepath)

    # 1. Binary target: Batal = 1, Selesai = 0 (drop ongoing orders)
    df = df[df["Status Pesanan"].isin(["Selesai", "Batal"])].copy()
    df["target"] = (df["Status Pesanan"] == "Batal").astype(int)

    # 2. Recover Year-Month from source_file & sort chronologically FIRST
    extracted = df["source_file"].astype(str).str.extract(r"([A-Za-z]+)Sales(\d{4})")
    df["order_year"] = extracted[1].fillna("2024").astype(int)
    df["order_month"] = extracted[0].map(MONTH_MAP).fillna(1).astype(int)
    df["year_month_sort"] = df["order_year"] * 100 + df["order_month"]

    dt_raw = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    fallback_dt = pd.to_datetime(
        df["order_year"].astype(str) + "-" + df["order_month"].astype(str).str.zfill(2) + "-15"
    )
    df["dt_sort"] = dt_raw.fillna(fallback_dt)
    df = df.sort_values(["year_month_sort", "dt_sort", "order_id"]).reset_index(drop=True)

    # 3. Temporal features + missing timestamp flag
    dt = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    df["is_missing_waktu"] = dt.isna().astype(int)
    df["order_hour"] = dt.dt.hour.fillna(-1).astype(int)
    df["order_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["is_late_night"] = dt.dt.hour.isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

    # 4. Valid checkout (t0) domain features (zero post-cancellation leakage)
    df["is_cod"] = df["Metode Pembayaran"].str.contains("COD", case=False, na=False).astype(int)
    df["is_java"] = df["Provinsi"].str.upper().isin(JAVA_PROVINCES).astype(int)
    df["destination_zone"] = df["Provinsi"].apply(map_destination_zone)
    df["has_discount"] = (df["Total Diskon"] > 0).astype(int)
    df["is_multi_category"] = (df["num_product_categories"] > 1).astype(int)

    df["weight_per_item"] = df["total_weight_gr"] / df["total_qty"].clip(lower=1)
    df["shipping_per_kg"] = df["Perkiraan Ongkos Kirim"] / (df["total_weight_gr"] / 1000.0).clip(lower=0.1)

    df["courier_group"] = df["Opsi Pengiriman"].astype(str).str.split("-").str[0].str.strip()

    # Keep Top 20 product categories; group rare ones into 'Other_Category'
    top_20_cats = df["product_categories"].value_counts().head(20).index
    df["top_product_cat"] = np.where(
        df["product_categories"].isin(top_20_cats),
        df["product_categories"],
        "Other_Category"
    )

    # Causal same-day repeat checkout count (strictly prior orders on same day with same city + product)
    df["order_date_str"] = dt.dt.date.astype(str)
    df["same_day_repeat_count"] = np.where(
        dt.isna(),
        0,
        df.groupby(["order_date_str", "Kota/Kabupaten", "product_categories"]).cumcount()
    )

    # 5. Log-transform all right-skewed numerical features (crucial for KNN & SVM distances)
    skewed_cols = [
        "total_qty", "total_weight_gr", "Total Diskon",
        "Perkiraan Ongkos Kirim", "weight_per_item", "shipping_per_kg"
    ]
    for col in skewed_cols:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def get_temporal_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> dict:
    """Splits dataset chronologically into Train, Validation, and Test sets."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["target"]

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
        "num_cols": NUMERIC_FEATURES,
        "cat_cols": CATEGORICAL_FEATURES,
    }