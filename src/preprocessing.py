import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "./dataset/all_months_clean.csv"
TOP_N_CATEGORIES  = 20
VAL_MONTHS        = 3
TEST_MONTHS       = 3

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

JABODETABEK_JABAR_BANTEN = {"BANTEN", "DKI JAKARTA", "JAWA BARAT"}
JATENG_DIY_JATIM         = {"JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR"}
JAVA_PROVINCES           = JABODETABEK_JABAR_BANTEN | JATENG_DIY_JATIM
SUMATERA_PROVINCES       = {
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

CAT_FEATURES = ["Metode Pembayaran", "shipping_tier", "destination_zone", "Provinsi", "top_product_cat"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def map_destination_zone(prov: str) -> str:
    p = str(prov).strip().upper()
    if p in JABODETABEK_JABAR_BANTEN:
        return "1. Jabodetabek, Jabar & Banten"
    if p in JATENG_DIY_JATIM:
        return "2. Jawa Tengah, DIY & Jawa Timur"
    if p in SUMATERA_PROVINCES:
        return "3. Pulau Sumatera"
    return "4. Kalimantan, Sulawesi, Bali-Nusra & Timur"


def map_shipping_tier(opsi: str) -> str:
    s = str(opsi).lower()
    if "hemat" in s or "kargo" in s:
        return "Hemat / Kargo"
    if "reguler" in s:
        return "Reguler"
    if "instant" in s or "same day" in s or "next day" in s:
        return "Instant / Same Day"
    return "Ekonomi / Lainnya"


def load_and_clean_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Loads raw CSV, filters binary target, removes leakage, and engineers t0 features."""
    df = pd.read_csv(filepath)
    df = df[df["Status Pesanan"].isin(["Selesai", "Batal"])].copy()
    df["target_batal"] = (df["Status Pesanan"] == "Batal").astype(int)

    # Recover Year-Month & sort chronologically
    ext = df["source_file"].astype(str).str.extract(r"([A-Za-z]+)Sales(\d{4})")
    df["order_year_month"] = ext[1].fillna("2024") + "-" + ext[0].map(MONTH_MAP).fillna(1).astype(int).astype(str).str.zfill(2)
    dt = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    df["dt_sort"] = dt.fillna(pd.to_datetime(df["order_year_month"] + "-15"))
    df = df.sort_values(["order_year_month", "dt_sort", "order_id"], kind="stable").reset_index(drop=True)

    # Temporal & quality flags
    dt = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    df["is_missing_timestamp"] = dt.isna().astype(int)
    df["order_hour"] = dt.dt.hour.fillna(-1).astype(int)
    df["order_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["is_late_night"] = dt.dt.hour.isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

    # Checkout (t0) domain features
    df["is_cod"] = df["Metode Pembayaran"].str.contains("COD", case=False, na=False).astype(int)
    df["is_java"] = df["Provinsi"].str.upper().isin(JAVA_PROVINCES).astype(int)
    df["destination_zone"] = df["Provinsi"].apply(map_destination_zone)
    df["shipping_tier"] = df["Opsi Pengiriman"].apply(map_shipping_tier)
    df["has_discount"] = (df["Total Diskon"] > 0).astype(int)
    df["is_multi_category"] = (df["num_product_categories"] > 1).astype(int)
    df["avg_item_weight_gr"] = df["total_weight_gr"] / df["total_qty"].clip(lower=1)
    df["shipping_per_kg"] = df["Perkiraan Ongkos Kirim"] / (df["total_weight_gr"] / 1000.0).clip(lower=0.1)

    df["order_date_str"] = dt.dt.date.astype(str)
    df["same_day_repeat_count"] = np.where(
        dt.isna(), 0, df.groupby(["order_date_str", "Kota/Kabupaten", "product_categories"]).cumcount()
    )

    # Log-transform skewed features
    log_cols = {
        "total_qty": "log_total_qty",
        "total_weight_gr": "log_total_weight_gr",
        "Total Diskon": "log_Total Diskon",
        "Perkiraan Ongkos Kirim": "log_Perkiraan Ongkos Kirim",
        "avg_item_weight_gr": "log_weight_per_item",
        "shipping_per_kg": "log_shipping_per_kg",
    }
    for raw_col, log_col in log_cols.items():
        df[log_col] = np.log1p(df[raw_col].clip(lower=0))

    return df

# ==============================================================================
# USAGE: get_temporal_splits(df, val_months=3, test_months=3) -> dict
#
# Examples:
#   splits = get_temporal_splits(df)                              # default: last 3m Test, prior 3m Val
#   splits = get_temporal_splits(df, val_months=2, test_months=2) # custom:  last 2m Test, prior 2m Val
#
# How Month Slicing Works (Example with 20 sorted months: M1 ... M20):
#   - test_months=2 -> Last 2 months        [M19, M20]       -> "Uji" (Test Set)
#   - val_months=2  -> 2 months before Test [M17, M18]       -> "Validasi" (Validation Set)
#   - Remaining     -> All earlier months   [M1 ... M16]     -> "Latih" (Train Set)
#
# Leakage Protection:
#   - `top_product_cat` (Top 20 categories) is fitted ONLY on `Latih` rows, then mapped
#     to `Validasi` & `Uji` (unseen categories become "Other_Category").
#
# Returned Dictionary Keys:
#   - splits["X_train"], splits["y_train"] : Features & target for Training
#   - splits["X_val"],   splits["y_val"]   : Features & target for Validation / threshold tuning
#   - splits["X_test"],  splits["y_test"]  : Features & target for final Test evaluation
#   - splits["parts"]                      : {"Latih": df_tr, "Validasi": df_va, "Uji": df_te}
#   - splits["split_summary"]              : Summary DataFrame (period, row count, cancel rate %)
#   - splits["test_months"]                : List of YYYY-MM strings in Test set
#   - splits["num_cols"], splits["cat_cols"], splits["all_features"]
# ==============================================================================
def get_temporal_splits(df: pd.DataFrame, val_months: int = VAL_MONTHS, test_months: int = TEST_MONTHS) -> dict:
    """Splits chronologically by month and fits top_product_cat strictly on Train."""
    data = df.copy()
    months = sorted(data["order_year_month"].unique())
    train_m = months[:-(val_months + test_months)]
    val_m   = months[-(val_months + test_months):-test_months]
    test_m  = months[-test_months:]

    train_mask = data["order_year_month"].isin(train_m)
    top_cats = data.loc[train_mask, "product_categories"].value_counts().head(TOP_N_CATEGORIES).index
    data["top_product_cat"] = np.where(data["product_categories"].isin(top_cats), data["product_categories"], "Other_Category")

    parts = {
        "Latih": data[train_mask],
        "Validasi": data[data["order_year_month"].isin(val_m)],
        "Uji": data[data["order_year_month"].isin(test_m)],
    }

    split_summary = pd.DataFrame([{
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
        "test_months": test_m,
        "split_summary": split_summary,
        "num_cols": NUM_FEATURES,
        "cat_cols": CAT_FEATURES,
        "all_features": ALL_FEATURES,
    }