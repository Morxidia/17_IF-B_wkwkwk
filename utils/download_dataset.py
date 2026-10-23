import os
import shutil
import re
import pandas as pd
import kagglehub

def download_and_process_kaggle_data(kaggle_link):
    raw_dir = 'dataset/raw/'
    out_dir = 'dataset/'
    
    # Create target directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    file_path = None
    
    # Extract dataset ID (username/dataset-name) and specific file name
    if "kaggle.com" in kaggle_link:
        match = re.search(r'kaggle\.com/datasets/([^/]+/[^/?]+)', kaggle_link)
        if match:
            dataset_id = match.group(1)
        else:
            raise ValueError("Could not extract valid dataset ID from URL.")
            
        # Detect if a specific file is selected via ?select= parameter
        select_match = re.search(r'\?select=([^&]+)', kaggle_link)
        if select_match:
            file_path = select_match.group(1)
    else:
        dataset_id = kaggle_link

    # Use kagglehub to download directly into raw_dir
    if file_path:
        print(f"Downloading file '{file_path}' from dataset: {dataset_id}")
        kagglehub.dataset_download(dataset_id, path=file_path, output_dir=raw_dir)
    else:
        print(f"Downloading full dataset: {dataset_id}")
        kagglehub.dataset_download(dataset_id, output_dir=raw_dir)
        
    print("Download complete.")

    # Process all files in the raw directory
    for filename in os.listdir(raw_dir):
        raw_path = os.path.join(raw_dir, filename)
        out_path = os.path.join(out_dir, filename)
        
        # Skip directories
        if not os.path.isfile(raw_path):
            continue
            
        # Delete hidden files like .complete
        if filename.startswith('.'):
            os.remove(raw_path)
            continue
            
        if filename.endswith('.csv'):
            # Read the first line to detect the delimiter
            try:
                with open(raw_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
            except UnicodeDecodeError:
                with open(raw_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline()
            
            # Check if semicolon is the primary separator
            if ';' in first_line and first_line.count(';') > first_line.count(','):
                print(f"Converting semicolon-separated CSV: {filename}")
                df = pd.read_csv(raw_path, sep=';', encoding='utf-8', on_bad_lines='skip')
                df.to_csv(out_path, index=False, sep=',')
            else:
                print(f"Copying standard CSV: {filename}")
                shutil.copy(raw_path, out_path)
        else:
            print(f"Copying non-CSV file: {filename}")
            shutil.copy(raw_path, out_path)
            
    print("Processing finished.")

if __name__ == "__main__":
    link = "https://www.kaggle.com/datasets/bakitacos/indonesia-e-commerce-sales-and-shipping-20232025/data?select=all_months_clean.csv"
    download_and_process_kaggle_data(link)