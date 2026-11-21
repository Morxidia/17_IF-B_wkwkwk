"""preprocessing_v2.py - drop-in replacement for preprocessing.py (same function names / return shapes)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MONTH_MAP = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
             "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

_TIER0 = {"BANTEN", "DKI JAKARTA", "JAWA BARAT"}
_TIER1 = {"JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR"}
_TIER2 = {"LAMPUNG", "SUMATERA SELATAN", "BENGKULU", "JAMBI", "KEPULAUAN BANGKA BELITUNG", "BANGKA BELITUNG"}


def _region_tier(p):
    p = str(p).upper()
    if p in _TIER0: return 0
    if p in _TIER1: return 1
    if p in _TIER2: return 2
    if "SUMATERA" in p or p in {"RIAU", "ACEH", "KEPULAUAN RIAU"} or p in {"BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR"}: return 3
    if "KALIMANTAN" in p: return 4
    if "SULAWESI" in p or "GORONTALO" in p: return 5
    return 6  # Maluku, Papua, unknown


NUMERIC_FEATURES = [
    "log_total_qty",
    "log_total_weight_gr",
    "log_Perkiraan Ongkos Kirim",
    "is_night",
    "region_tier",
]
CATEGORICAL_FEATURES = ["pay_group", "top_product_cat", "courier_clean"]


def load_and_clean_data(filepath: str = "./dataset/all_months_clean.csv", train_ratio: float = 0.70) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df["Status Pesanan"].isin(["Selesai", "Batal"])].copy()
    df["target"] = (df["Status Pesanan"] == "Batal").astype(int)

    extracted = df["source_file"].astype(str).str.extract(r"([A-Za-z]+)Sales(\d{4})")
    if extracted.isna().any().any():
        raise ValueError("source_file name did not match '<Month>Sales<Year>' for some rows")
    df["order_year"] = extracted[1].astype(int)
    df["order_month"] = extracted[0].map(MONTH_MAP).astype(int)
    df["year_month_sort"] = df["order_year"] * 100 + df["order_month"]

    dt = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    fallback = pd.to_datetime(df["order_year"].astype(str) + "-" + df["order_month"].astype(str).str.zfill(2) + "-15")
    df["dt_sort"] = dt.fillna(fallback)
    df = df.sort_values(["year_month_sort", "dt_sort", "order_id"]).reset_index(drop=True)
    dt = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")

    # --- time: NO sentinel -1, NO is_missing_waktu (whole months Dec-2024 / Jul-2025 lack timestamps:
    #     the flag is a month identifier, not customer behaviour). Missing -> neutral 0.
    df["is_night"] = dt.dt.hour.isin([22, 23, 0, 1, 2, 3]).astype(int)

    # --- t0-safe shipping / order features
    df["region_tier"] = df["Provinsi"].map(_region_tier)
    for col in ["total_qty", "total_weight_gr", "Perkiraan Ongkos Kirim"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    pm = df["Metode Pembayaran"].astype(str)
    df["pay_group"] = np.select(
        [pm.str.contains("COD"), pm.str.contains("Alfamart|Indomaret"), pm.str.contains("ShopeePay|SeaBank"),
         pm.str.contains("SPayLater|Cicilan"), pm.str.contains("Online Payment|Kartu|BCA")],
        ["COD", "Minimarket", "Wallet", "PayLater", "OnlineBank"], default="Other")

    # Courier: ONLY the part before '-'. The carrier AFTER '-' (SPX/JNE/J&T) is missing for ~650 orders
    # that are 100% cancelled (never assigned a carrier) -> using it would be label leakage.
    cg = df["Opsi Pengiriman"].astype(str).str.split("-").str[0].str.strip()
    df["courier_clean"] = np.select(
        [cg.str.contains("Hemat Kargo|SPX Hemat"), cg.str.contains("Reguler|Standard"),
         cg.str.contains("Instant|Same Day|Sameday|Next Day|GoSend|Grab"),
         cg.str.contains("Kargo|Cargo|Trucking|Economy|Agen")],
        ["hemat", "reguler", "instant", "cargo"], default="other")

    # Top-20 product categories learned from the TRAIN portion only
    train_n = int(len(df) * train_ratio)
    top20 = df["product_categories"].iloc[:train_n].value_counts().head(20).index
    df["top_product_cat"] = np.where(df["product_categories"].isin(top20), df["product_categories"], "Other_Category")
    return df


def get_temporal_splits(df, train_ratio=0.70, val_ratio=0.15, random_state=42):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = df[feature_cols], df["target"]
    test_ratio = 1 - train_ratio - val_ratio

    # 1) train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=random_state, stratify=y
    )
    # 2) split the remainder into val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state, stratify=y_tmp
    )

    print(X_train.shape, X_val.shape, X_test.shape)
    return {"X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "num_cols": NUMERIC_FEATURES, "cat_cols": CATEGORICAL_FEATURES}