#!/usr/bin/env python3
"""Q3-only feature extraction - uses same sampling as v2 for consistency."""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from PIL import Image

CAPSTONE_DIR = r"C:\Users\Admin\OneDrive\Desktop\Work\AUA\Capstone Projects\Marina\Capstone"
CSV_PATH = os.path.join(CAPSTONE_DIR, "data", "metadata", "q1q3_all_extracted.csv")
IMG_BASE = os.path.join(CAPSTONE_DIR, "data", "organized_images")
RESULTS_DIR = os.path.join(CAPSTONE_DIR, "results")

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
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered
    print("Q3-only Feature Extraction", flush=True)

    # Load metadata
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
    print(f"Q3 available: {len(q3)}", flush=True)

    # Same stratified sampling as v2 (same seed for reproducibility)
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
    print(f"Sampled: {len(q3_df)} ({q3_df['serial'].nunique()} cams)", flush=True)

    # Extract
    paths = q3_df['img_path'].tolist()
    n = len(paths)
    features = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()

    for i, p in enumerate(paths):
        features[i] = extract_histogram(p)
        if (i+1) % 500 == 0 or (i+1) == n:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  Q3: {i+1}/{n} ({(i+1)/n*100:.0f}%) | {rate:.0f} img/s | ETA: {eta:.0f}s", flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'q3_features_all.npy')
    print(f"Saving {features.shape} to {save_path}...", flush=True)
    np.save(save_path, features)
    
    # Verify
    if os.path.exists(save_path):
        sz = os.path.getsize(save_path)
        print(f"Saved OK: {sz} bytes ({sz/1024/1024:.1f} MB)", flush=True)
    else:
        print("ERROR: File not created!", flush=True)

    # Save metadata
    meta_path = os.path.join(RESULTS_DIR, 'q3_metadata.csv')
    q3_df.to_csv(meta_path, index=False)
    print(f"Metadata saved: {len(q3_df)} rows", flush=True)

    # Also save Q1 metadata (same sampling) if not exists
    q1_meta = os.path.join(RESULTS_DIR, 'q1_metadata.csv')
    if not os.path.exists(q1_meta):
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
        print(f"Q1 metadata saved: {len(q1_sampled)} rows", flush=True)

    elapsed = time.time() - start
    print(f"DONE in {elapsed/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
