"""
Script: 01_download_and_merge.py
Description: Automatically downloads the "Apartment Prices in Poland" dataset from Kaggle,
             extracts it, and merges the monthly sale offers into a single structured CSV.

[GenAI Declaration]
The following code structure and pandas merging logic were assisted by Gemini 3.1 Pro
on March 17, 2026. Prompt used: "Create a Python script using Kaggle API to download
mlenkin/apartment-prices-in-poland, unzip it, and concatenate all apartments_pl_YYYY_MM.csv files."
"""

import os
import glob
import re
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def main():

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    print("🔑 Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    dataset_name = "krzysztofjamroz/apartment-prices-in-poland"

    print(f"⬇️ Downloading dataset '{dataset_name}' to {raw_data_dir}...")

    api.dataset_download_files(dataset_name, path=raw_data_dir, unzip=True)
    print("✅ Download and extraction complete!\n")


    print("🔄 Merging monthly sale datasets...")
    # Explicitly match only sale files (apartments_pl_YYYY_MM.csv),
    # excluding rent files (apartments_rent_pl_YYYY_MM.csv).
    sale_file_pattern = re.compile(r"^apartments_pl_\d{4}_\d{2}\.csv$")
    all_csvs = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    csv_files = [f for f in all_csvs if sale_file_pattern.match(os.path.basename(f))]

    if not csv_files:
        print("❌ Error: No sale CSV files found. Check the download step.")
        return

    csv_files.sort()

    df_list = []
    for file in csv_files:

        filename = os.path.basename(file)
        # Extract YYYY_MM and convert to YYYY-MM for standard date compatibility
        year_month_raw = filename.replace("apartments_pl_", "").replace(".csv", "")
        year_month = year_month_raw.replace("_", "-")  # e.g. "2023-08"

        print(f"   - Processing {filename}...")
        df = pd.read_csv(file)

        df['snapshot_month'] = year_month
        df_list.append(df)


    master_df = pd.concat(df_list, ignore_index=True)

    print(f"\n📊 Merged dataset shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns.")


    output_file = processed_data_dir / "master_sales_dataset.csv"
    master_df.to_csv(output_file, index=False)
    print(f"🎉 Success! Merged dataset saved to: {output_file}")

if __name__ == "__main__":
    main()