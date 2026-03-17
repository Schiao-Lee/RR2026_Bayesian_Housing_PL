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
import zipfile
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
    search_pattern = os.path.join(raw_data_dir, "apartments_pl_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("❌ Error: No CSV files found. Check the download step.")
        return

    csv_files.sort()
    
    df_list = []
    for file in csv_files:
        
        filename = os.path.basename(file)
        year_month = filename.replace("apartments_pl_", "").replace(".csv", "")
        
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