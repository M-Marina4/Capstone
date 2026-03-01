#!/usr/bin/env python3
"""
Extract features directly from zip files to avoid OneDrive filesystem overhead.
Reads images from the original camera zip files, computes RGB histogram features,
and saves them to numpy arrays for the analysis pipeline.
"""

import os
import sys
import time
import zipfile
import io
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

MARINA_DIR = r"C:\Users\Admin\OneDrive\Desktop\Work\AUA\Capstone Projects\Marina"
CAPSTONE_DIR = os.path.join(MARINA_DIR, "Capstone")
CSV_PATH = os.path.join(CAPSTONE_DIR, "data", "metadata", "streetcare-drift-dataset-2021-2025.csv")


def extract_histogram(image_bytes, bins=256):
    """Extract RGB histogram features from image bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        hist = np.concatenate([
            np.histogram(img_array[:,:,i], bins=bins, range=(0, 1))[0] 
            for i in range(3)
        ])
        hist = hist / (hist.sum() + 1e-8)
        return hist
    except Exception as e:
        return np.zeros(bins * 3)


def main():
    start = time.time()
    
    # --- Load metadata ---
    print("[1/3] Loading metadata...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    
    # Filter Q1/Q3 only
    mask = df['time_tag'].str.contains('Q1|Q3', na=False)
    df = df[mask].copy()
    
    # Label quarter and daynight
    df['quarter'] = df['time_tag'].str.extract(r'(Q[13])')[0]
    df['dn_label'] = df['daynight'].map({0.0: 'daytime', 1.0: 'nighttime'})
    df = df.dropna(subset=['image_name', 'serial', 'quarter', 'dn_label'])
    
    # Build lookup: serial -> {image_name: row_index}
    serial_images = {}
    for idx, row in df.iterrows():
        serial = row['serial']
        if serial not in serial_images:
            serial_images[serial] = {}
        serial_images[serial][row['image_name']] = idx
    
    serials = sorted(serial_images.keys())
    print(f"   Total images to process: {len(df)}")
    print(f"   Cameras: {len(serials)}")
    
    # --- Extract features from zip files ---
    print("\n[2/3] Extracting features from zip files...")
    features_dict = {}  # idx -> feature vector
    total_extracted = 0
    total_missing = 0
    
    for i, serial in enumerate(serials):
        zip_path = os.path.join(MARINA_DIR, f"{serial}.zip")
        needed = serial_images[serial]
        
        if not os.path.exists(zip_path):
            print(f"   [{i+1}/{len(serials)}] SKIP {serial} - zip not found")
            total_missing += len(needed)
            continue
        
        t0 = time.time()
        extracted = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in needed:
                    idx = needed[basename]
                    image_bytes = zf.read(name)
                    feat = extract_histogram(image_bytes)
                    features_dict[idx] = feat
                    extracted += 1
        
        elapsed = time.time() - t0
        not_found = len(needed) - extracted
        total_extracted += extracted
        total_missing += not_found
        print(f"   [{i+1}/{len(serials)}] {serial}: {extracted} features, {not_found} missing, {elapsed:.1f}s")
    
    print(f"\n   Total: {total_extracted} features, {total_missing} missing")
    
    # --- Build Q1/Q3 feature arrays ---
    print("\n[3/3] Building feature arrays...")
    
    # Get successfully extracted indices
    valid_indices = set(features_dict.keys())
    df_valid = df.loc[df.index.isin(valid_indices)].copy()
    
    q1_mask = df_valid['quarter'] == 'Q1'
    q3_mask = df_valid['quarter'] == 'Q3'
    
    q1_indices = df_valid[q1_mask].index.tolist()
    q3_indices = df_valid[q3_mask].index.tolist()
    
    q1_features = np.array([features_dict[i] for i in q1_indices])
    q3_features = np.array([features_dict[i] for i in q3_indices])
    
    print(f"   Q1: {q1_features.shape}")
    print(f"   Q3: {q3_features.shape}")
    
    # Save feature arrays
    os.makedirs(os.path.join(CAPSTONE_DIR, 'results'), exist_ok=True)
    np.save(os.path.join(CAPSTONE_DIR, 'results', 'q1_features_all.npy'), q1_features)
    np.save(os.path.join(CAPSTONE_DIR, 'results', 'q3_features_all.npy'), q3_features)
    
    # Save the valid metadata
    q1_meta = df_valid[q1_mask]
    q3_meta = df_valid[q3_mask]
    q1_meta.to_csv(os.path.join(CAPSTONE_DIR, 'results', 'q1_metadata.csv'), index=False)
    q3_meta.to_csv(os.path.join(CAPSTONE_DIR, 'results', 'q3_metadata.csv'), index=False)
    
    elapsed_total = time.time() - start
    print(f"\n   Done! Total time: {elapsed_total/60:.1f} minutes")
    print(f"   Saved: results/q1_features_all.npy ({q1_features.shape})")
    print(f"   Saved: results/q3_features_all.npy ({q3_features.shape})")
    print(f"   Saved: results/q1_metadata.csv, results/q3_metadata.csv")
    
    # Summary stats
    for label, meta in [('Q1', q1_meta), ('Q3', q3_meta)]:
        print(f"\n   {label} breakdown:")
        print(f"     Daytime:   {len(meta[meta['dn_label']=='daytime'])}")
        print(f"     Nighttime: {len(meta[meta['dn_label']=='nighttime'])}")
        print(f"     Years:     {sorted(meta['date'].dt.year.unique())}")


if __name__ == '__main__':
    main()
