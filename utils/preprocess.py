"""
Script: preprocess.py
Deskripsi: Modul pembersihan data (cleaning) dan rekayasa fitur (feature engineering)
           khusus untuk Model Klasifikasi Risiko Pembatalan Pesanan E-Commerce.

Input  : ./dataset/all_months_clean.csv (CSV standar pemisah koma ',', 20.848 baris)
Output : ./dataset/processed_ecommerce_data.csv (Dataset terfilter & siap latih model / EDA, 19.865 baris)
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd

DEFAULT_INPUT_CSV = "./dataset/all_months_clean.csv"
DEFAULT_OUTPUT_CSV = "./dataset/processed_ecommerce_data.csv"

def classify_shipping_tier(opt):
    s = str(opt).lower()
    if "hemat" in s or "kargo" in s or "trucking" in s or "gokil" in s:
        return "Hemat / Kargo"
    elif "instant" in s or "same day" in s or "sameday" in s:
        return "Instant / Same Day"
    elif "reguler" in s or "standard" in s:
        return "Reguler (Standard)"
    else:
        return "Ekonomi / Kurir Lainnya"

def classify_region(prov):
    if prov in ["JAWA BARAT", "BANTEN", "DKI JAKARTA"]:
        return "1. Jabodetabek, Jabar & Banten"
    elif prov in ["JAWA TENGAH", "JAWA TIMUR", "DI YOGYAKARTA"]:
        return "2. Jawa Tengah, DIY & Jawa Timur"
    elif prov in [
        "LAMPUNG", "SUMATERA SELATAN", "SUMATERA UTARA", "RIAU", "JAMBI",
        "SUMATERA BARAT", "KEPULAUAN BANGKA BELITUNG", "KEPULAUAN RIAU", "BENGKULU", "ACEH"
    ]:
        return "3. Pulau Sumatera"
    else:
        return "4. Kalimantan, Sulawesi, Bali-Nusra & Timur"

def parse_source_ym(fname):
    month_map = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }
    m = re.match(r"([A-Za-z]+)Sales(\d{4})\.xlsx", str(fname))
    if m:
        m_code = month_map.get(m.group(1), "01")
        return f"{m.group(2)}-{m_code}"
    return "2024-01"

def preprocess_data(input_csv=DEFAULT_INPUT_CSV, output_csv=DEFAULT_OUTPUT_CSV):
    """
    Melakukan data cleaning dan feature engineering:
    1. Memfilter pesanan ke status terminal (Selesai vs Batal).
    2. Mengeliminasi anomali sistem (Total Pembayaran <= Rp 0).
    3. Memulihkan informasi temporal (year-month) dari source_file.
    4. Melakukan rekayasa fitur rasio ongkir, kurir, dan zona pengiriman.
    5. Melakukan winsorizing p99 dan transformasi log1p pada fitur bernilai besar.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File input '{input_csv}' tidak ditemukan. Jalankan download_dataset.py terlebih dahulu.")

    print(f"[*] Membaca data untuk praproses dari: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
        if len(df.columns) == 1 and ';' in df.columns[0]:
            df = pd.read_csv(input_csv, sep=';')
    except Exception:
        df = pd.read_csv(input_csv, sep=None, engine='python')

    n_initial = len(df)
    print(f"-> Total baris awal: {n_initial:,} baris x {df.shape[1]} kolom")

    # 1. Filter status terminal (Selesai & Batal saja, eliminasi in-flight 250 baris)
    df = df[df["Status Pesanan"].isin(["Selesai", "Batal"])].copy()
    n_terminal = len(df)
    print(f"-> Setelah filter status terminal (Selesai/Batal): {n_terminal:,} baris (-{n_initial - n_terminal:,} in-flight)")

    # 2. Filter anomali pembayaran (Total Pembayaran > 0, eliminasi 733 baris Rp 0)
    df = df[df["Total Pembayaran"] > 0].copy()
    n_valid_pay = len(df)
    print(f"-> Setelah filter pembayaran valid (> Rp 0): {n_valid_pay:,} baris (-{n_terminal - n_valid_pay:,} anomali Rp 0)")

    # 3. Label target biner
    df["target_batal"] = (df["Status Pesanan"] == "Batal").astype(int)

    # 4. Pemulihan Informasi Temporal
    df["is_missing_timestamp"] = df["Waktu Pesanan Dibuat"].isna().astype(int)
    dt_parsed = pd.to_datetime(df["Waktu Pesanan Dibuat"], errors="coerce")
    df["order_year_month"] = dt_parsed.dt.strftime("%Y-%m").fillna(df["source_file"].apply(parse_source_ym))
    df["order_hour"] = dt_parsed.dt.hour.fillna(-1).astype(int)
    df["order_dayofweek"] = dt_parsed.dt.dayofweek.fillna(-1).astype(int)

    # 5. Rekayasa Fitur Rasio Pengiriman & Pembayaran
    df["shipping_burden_ratio"] = df["Ongkos Kirim Dibayar oleh Pembeli"] / (df["Total Pembayaran"] + 1.0)
    df["shipping_subsidy_coverage"] = np.clip(
        df["Estimasi Potongan Biaya Pengiriman"] / (df["Perkiraan Ongkos Kirim"] + 1.0), 0.0, 1.0
    )
    df["net_unsubsidized_shipping"] = np.maximum(
        0, df["Perkiraan Ongkos Kirim"] - df["Estimasi Potongan Biaya Pengiriman"]
    )
    df["is_free_shipping"] = (df["Ongkos Kirim Dibayar oleh Pembeli"] == 0).astype(int)
    df["is_cod"] = (df["Metode Pembayaran"] == "COD (Bayar di Tempat)").astype(int)
    df["has_discount"] = (df["Total Diskon"] > 0).astype(int)
    df["avg_item_weight_gr"] = df["total_weight_gr"] / np.maximum(1, df["total_qty"])
    df["shipping_tier"] = df["Opsi Pengiriman"].apply(classify_shipping_tier)
    df["destination_zone"] = df["Provinsi"].apply(classify_region)

    # 6. Winsorizing pada P99 & Transformasi Log1p
    skewed_cols = [
        "total_qty", "total_weight_gr", "Total Diskon",
        "Ongkos Kirim Dibayar oleh Pembeli", "Estimasi Potongan Biaya Pengiriman",
        "Perkiraan Ongkos Kirim", "Total Pembayaran"
    ]
    for col in skewed_cols:
        p99 = df[col].quantile(0.99)
        df[f"{col}_winsor"] = np.clip(df[col], 0, p99)
        df[f"{col}_log1p"] = np.log1p(df[f"{col}_winsor"])

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, sep=',')
    print(f"[OK] Sukses menyimpan data siap EDA/Model ke: {output_csv} ({df.shape[0]:,} baris x {df.shape[1]} kolom)")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing & Feature Engineering Dataset E-Commerce")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="File input CSV (default: ./dataset/all_months_clean.csv)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="File output CSV (default: ./dataset/processed_ecommerce_data.csv)")
    args = parser.parse_args()

    try:
        preprocess_data(input_csv=args.input, output_csv=args.output)
    except Exception as e:
        print(f"[ERROR] Preprocessing gagal: {e}")
        sys.exit(1)