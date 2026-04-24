#!/usr/bin/env python3
"""
Fast feature extraction - single process, optimized.
Reads from already-extracted organized_images directory.
Supports stratified sampling for large datasets.
Saves incrementally to avoid losing progress.
"""
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
    """Extract RGB histogram from image. Returns 768-dim vector."""
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
    except Exception as e:
        return np.zeros(bins * 3)


def extract_batch(paths, label, save_path):
    """Extract features for a list of image paths with progress."""
    n = len(paths)
    features = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()
    
    for i, p in enumerate(paths):
        features[i] = extract_histogram(p)
        
        if (i+1) % 500 == 0 or (i+1) == n:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (n - i - 1) / rate
            print(f"   {label}: {i+1}/{n} ({(i+1)/n*100:.0f}%) | {rate:.0f} img/s | ETA: {eta:.0f}s",
                  flush=True)
    
    # Save immediately
    np.save(save_path, features)
    print(f"   {label}: saved {features.shape} to {os.path.basename(save_path)}", flush=True)
    return features


def main():
    start = time.time()
    print("=" * 60)
    print("Feature Extraction - 20K per quarter")
    print("=" * 60, flush=True)

    # --- Load metadata ---
    print("\n[1/4] Loading metadata...", flush=True)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    df['quarter'] = df['time_tag'].str.extract(r'(Q[13])')[0]
    df['dn_label'] = df['daynight'].map({0.0: 'daytime', 1.0: 'nighttime'})
    df = df.dropna(subset=['image_name', 'quarter', 'dn_label'])
    df['year'] = df['date'].dt.year

    # Build paths
    df['img_path'] = df.apply(
        lambda r: os.path.join(IMG_BASE, r['quarter'], r['dn_label'], str(r['image_name'])),
        axis=1
    )

    # Filter to files that exist
    print("   Building file index...", flush=True)
    existing = set()
    for q in ['Q1', 'Q3']:
        for dn in ['daytime', 'nighttime']:
            d = os.path.join(IMG_BASE, q, dn)
            if os.path.isdir(d):
                existing.update(os.path.join(IMG_BASE, q, dn, f) for f in os.listdir(d))
    df = df[df['img_path'].isin(existing)].copy()
    q1_total = len(df[df['quarter']=='Q1'])
    q3_total = len(df[df['quarter']=='Q3'])
    print(f"   Available: Q1={q1_total}, Q3={q3_total}", flush=True)

    # --- Stratified sampling ---
    print(f"\n[2/4] Sampling {SAMPLE_PER_QUARTER}/quarter (stratified)...", flush=True)
    sampled_dfs = []
    for q in ['Q1', 'Q3']:
        qdf = df[df['quarter'] == q]
        if len(qdf) <= SAMPLE_PER_QUARTER:
            sampled_dfs.append(qdf)
            print(f"   {q}: all {len(qdf)}", flush=True)
        else:
            sampled = qdf.groupby(['serial', 'year'], group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), max(1, int(SAMPLE_PER_QUARTER * len(x) / len(qdf)))),
                    random_state=42
                )
            )
            if len(sampled) > SAMPLE_PER_QUARTER:
                sampled = sampled.sample(n=SAMPLE_PER_QUARTER, random_state=42)
            elif len(sampled) < SAMPLE_PER_QUARTER:
                remaining = qdf[~qdf.index.isin(sampled.index)]
                extra = remaining.sample(n=min(SAMPLE_PER_QUARTER - len(sampled), len(remaining)),
                                        random_state=42)
                sampled = pd.concat([sampled, extra])
            sampled_dfs.append(sampled)
            print(f"   {q}: {len(sampled)}/{len(qdf)} ({sampled['serial'].nunique()} cams, "
                  f"years {sorted(sampled['year'].unique())})", flush=True)

    q1_df = sampled_dfs[0].copy()
    q3_df = sampled_dfs[1].copy()
    print(f"   Total: {len(q1_df) + len(q3_df)}", flush=True)

    # --- Extract features ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n[3/4] Extracting features...", flush=True)
    
    q1_paths = q1_df['img_path'].tolist()
    q3_paths = q3_df['img_path'].tolist()
    
    q1_feat = extract_batch(q1_paths, "Q1",
                           os.path.join(RESULTS_DIR, 'q1_features_all.npy'))
    q3_feat = extract_batch(q3_paths, "Q3",
                           os.path.join(RESULTS_DIR, 'q3_features_all.npy'))

    # --- Save metadata ---
    print(f"\n[4/4] Saving metadata...", flush=True)
    q1_df.to_csv(os.path.join(RESULTS_DIR, 'q1_metadata.csv'), index=False)
    q3_df.to_csv(os.path.join(RESULTS_DIR, 'q3_metadata.csv'), index=False)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Q1: {q1_feat.shape}  |  Q3: {q3_feat.shape}")
    for label, meta in [('Q1', q1_df), ('Q3', q3_df)]:
        print(f"\n{label} breakdown:")
        for dn in ['daytime', 'nighttime']:
            print(f"  {dn}: {len(meta[meta['dn_label']==dn])}")
        print(f"  Cameras: {meta['serial'].nunique()}")
        print(f"  Years: {sorted(meta['year'].unique())}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
