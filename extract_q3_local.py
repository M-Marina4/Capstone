#!/usr/bin/env python3
"""
Extract Q3 features by first copying sampled images to local temp dir,
then extracting from there to bypass OneDrive I/O overhead.
"""
import os, sys, time, shutil, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image

CAPSTONE_DIR = r"C:\Users\Admin\OneDrive\Desktop\Work\AUA\Capstone Projects\Marina\Capstone"
CSV_PATH = os.path.join(CAPSTONE_DIR, "data", "metadata", "q1q3_all_extracted.csv")
IMG_BASE = os.path.join(CAPSTONE_DIR, "data", "organized_images")
RESULTS_DIR = os.path.join(CAPSTONE_DIR, "results")
TEMP_DIR = r"C:\Temp\q3_sample"  # Local, non-OneDrive directory

SAMPLE_PER_QUARTER = 20000


def extract_histogram(img_path, bins=256):
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        hist = np.concatenate([
            np.histogram(arr[:,:,c], bins=bins, range=(0,1))[0]
            for c in range(3)
        ])
        return hist / (hist.sum() + 1e-8)
    except Exception:
        return np.zeros(bins * 3)


def main():
    start = time.time()
    print("=" * 60, flush=True)
    print("Q3 Feature Extraction (via local temp copy)", flush=True)
    print("=" * 60, flush=True)

    # Load metadata
    print("\n[1/5] Loading metadata...", flush=True)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    df['quarter'] = df['time_tag'].str.extract(r'(Q[13])')[0]
    df['dn_label'] = df['daynight'].map({0.0: 'daytime', 1.0: 'nighttime'})
    df = df.dropna(subset=['image_name', 'quarter', 'dn_label'])
    df['year'] = df['date'].dt.year
    df['img_path'] = df.apply(
        lambda r: os.path.join(IMG_BASE, r['quarter'], r['dn_label'], str(r['image_name'])),
        axis=1
    )

    # Filter to existing Q3 files
    existing = set()
    for dn in ['daytime', 'nighttime']:
        d = os.path.join(IMG_BASE, 'Q3', dn)
        if os.path.isdir(d):
            existing.update(os.path.join(IMG_BASE, 'Q3', dn, f) for f in os.listdir(d))
    q3 = df[(df['quarter'] == 'Q3') & (df['img_path'].isin(existing))].copy()
    print(f"  Q3 available: {len(q3)}", flush=True)

    # Stratified sampling
    print(f"\n[2/5] Sampling {SAMPLE_PER_QUARTER} images...", flush=True)
    sampled = q3.groupby(['serial', 'year'], group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), max(1, int(SAMPLE_PER_QUARTER * len(x) / len(q3)))),
            random_state=42
        )
    )
    if len(sampled) > SAMPLE_PER_QUARTER:
        sampled = sampled.sample(n=SAMPLE_PER_QUARTER, random_state=42)
    elif len(sampled) < SAMPLE_PER_QUARTER:
        remaining = q3[~q3.index.isin(sampled.index)]
        extra = remaining.sample(n=min(SAMPLE_PER_QUARTER - len(sampled), len(remaining)),
                                random_state=42)
        sampled = pd.concat([sampled, extra])
    q3_df = sampled.copy()
    print(f"  Sampled: {len(q3_df)} ({q3_df['serial'].nunique()} cameras)", flush=True)

    # Copy to temp directory
    print(f"\n[3/5] Copying {len(q3_df)} images to {TEMP_DIR}...", flush=True)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    paths_onedrive = q3_df['img_path'].tolist()
    paths_local = []
    t0 = time.time()
    copy_errors = 0
    
    for i, src in enumerate(paths_onedrive):
        fname = os.path.basename(src)
        dst = os.path.join(TEMP_DIR, fname)
        try:
            shutil.copy2(src, dst)
            paths_local.append(dst)
        except Exception as e:
            paths_local.append(src)  # fallback to OneDrive path
            copy_errors += 1
        
        if (i+1) % 1000 == 0 or (i+1) == len(paths_onedrive):
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (len(paths_onedrive) - i - 1) / rate if rate > 0 else 0
            print(f"  Copy: {i+1}/{len(paths_onedrive)} | {rate:.0f}/s | ETA: {eta:.0f}s", flush=True)
    
    if copy_errors > 0:
        print(f"  Warning: {copy_errors} copy errors (using OneDrive fallback)", flush=True)
    print(f"  Copy done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Extract features from local copies
    print(f"\n[4/5] Extracting features from local copies...", flush=True)
    n = len(paths_local)
    features = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()
    
    for i, p in enumerate(paths_local):
        features[i] = extract_histogram(p)
        if (i+1) % 500 == 0 or (i+1) == n:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  Q3: {i+1}/{n} ({(i+1)/n*100:.0f}%) | {rate:.0f} img/s | ETA: {eta:.0f}s", flush=True)

    # Save
    print(f"\n[5/5] Saving...", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    npy_path = os.path.join(RESULTS_DIR, 'q3_features_all.npy')
    np.save(npy_path, features)
    if os.path.exists(npy_path):
        print(f"  Features: {npy_path} ({os.path.getsize(npy_path)/1024/1024:.1f} MB)", flush=True)
    
    meta_path = os.path.join(RESULTS_DIR, 'q3_metadata.csv')
    q3_df.to_csv(meta_path, index=False)
    print(f"  Metadata: {meta_path} ({len(q3_df)} rows)", flush=True)

    # Also ensure Q1 metadata exists
    q1_meta = os.path.join(RESULTS_DIR, 'q1_metadata.csv')
    if not os.path.exists(q1_meta):
        print("  Creating Q1 metadata...", flush=True)
        q1 = df[(df['quarter'] == 'Q1')].copy()
        existing_q1 = set()
        for dn in ['daytime', 'nighttime']:
            d = os.path.join(IMG_BASE, 'Q1', dn)
            if os.path.isdir(d):
                existing_q1.update(os.path.join(IMG_BASE, 'Q1', dn, f) for f in os.listdir(d))
        q1 = q1[q1['img_path'].isin(existing_q1)].copy()
        q1_sampled = q1.groupby(['serial', 'year'], group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(1, int(SAMPLE_PER_QUARTER * len(x) / len(q1)))),
                random_state=42
            )
        )
        if len(q1_sampled) > SAMPLE_PER_QUARTER:
            q1_sampled = q1_sampled.sample(n=SAMPLE_PER_QUARTER, random_state=42)
        elif len(q1_sampled) < SAMPLE_PER_QUARTER:
            remaining = q1[~q1.index.isin(q1_sampled.index)]
            extra = remaining.sample(n=min(SAMPLE_PER_QUARTER - len(q1_sampled), len(remaining)),
                                    random_state=42)
            q1_sampled = pd.concat([q1_sampled, extra])
        q1_sampled.to_csv(q1_meta, index=False)
        print(f"  Q1 metadata: {len(q1_sampled)} rows", flush=True)

    # Cleanup temp
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"  Temp cleaned up", flush=True)
    except:
        print(f"  Note: temp dir at {TEMP_DIR} (cleanup manually)", flush=True)

    elapsed = time.time() - start
    print(f"\n{'='*60}", flush=True)
    print(f"DONE in {elapsed/60:.1f} min", flush=True)
    print(f"Q3 features: {features.shape}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
