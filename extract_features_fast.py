#!/usr/bin/env python3
"""
Fast feature extraction using multiprocessing.
Reads from already-extracted organized_images directory.
Supports stratified sampling for large datasets.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

CAPSTONE_DIR = r"C:\Users\Admin\OneDrive\Desktop\Work\AUA\Capstone Projects\Marina\Capstone"
CSV_PATH = os.path.join(CAPSTONE_DIR, "data", "metadata", "q1q3_all_extracted.csv")
IMG_BASE = os.path.join(CAPSTONE_DIR, "data", "organized_images")

SAMPLE_PER_QUARTER = 20000  # 20K per quarter → 40K total


def extract_histogram(img_path, bins=256):
    """Extract RGB histogram features from a single image file."""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        hist = np.concatenate([
            np.histogram(img_array[:,:,i], bins=bins, range=(0, 1))[0]
            for i in range(3)
        ])
        hist = hist / (hist.sum() + 1e-8)
        return hist
    except Exception:
        return np.zeros(bins * 3)


def process_image(args):
    """Worker function for multiprocessing."""
    img_path, idx = args
    return idx, extract_histogram(img_path)


def main():
    start = time.time()
    n_workers = max(1, cpu_count() - 1)
    print(f"Workers: {n_workers} CPUs")

    # --- Load and filter metadata ---
    print("\n[1/4] Loading metadata...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    df['quarter'] = df['time_tag'].str.extract(r'(Q[13])')[0]
    df['dn_label'] = df['daynight'].map({0.0: 'daytime', 1.0: 'nighttime'})
    df = df.dropna(subset=['image_name', 'quarter', 'dn_label'])
    df['year'] = df['date'].dt.year

    # Build image paths
    df['img_path'] = df.apply(
        lambda r: os.path.join(IMG_BASE, r['quarter'], r['dn_label'], str(r['image_name'])),
        axis=1
    )

    # Filter to existing files (fast: check using pre-built index)
    print("   Building file index...")
    existing = set()
    for q in ['Q1', 'Q3']:
        for dn in ['daytime', 'nighttime']:
            d = os.path.join(IMG_BASE, q, dn)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    existing.add(os.path.join(IMG_BASE, q, dn, f))
    df = df[df['img_path'].isin(existing)].copy()
    print(f"   Total available: Q1={len(df[df['quarter']=='Q1'])}, Q3={len(df[df['quarter']=='Q3'])}")

    # --- Stratified sampling ---
    print(f"\n[2/4] Sampling {SAMPLE_PER_QUARTER} per quarter (stratified by serial+year)...")
    sampled_dfs = []
    for q in ['Q1', 'Q3']:
        qdf = df[df['quarter'] == q]
        if len(qdf) <= SAMPLE_PER_QUARTER:
            sampled_dfs.append(qdf)
            print(f"   {q}: using all {len(qdf)} (below sample target)")
        else:
            # Stratified sample by serial and year
            sampled = qdf.groupby(['serial', 'year'], group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), max(1, int(SAMPLE_PER_QUARTER * len(x) / len(qdf)))),
                    random_state=42
                )
            )
            # Adjust to exact target
            if len(sampled) > SAMPLE_PER_QUARTER:
                sampled = sampled.sample(n=SAMPLE_PER_QUARTER, random_state=42)
            elif len(sampled) < SAMPLE_PER_QUARTER:
                remaining = qdf[~qdf.index.isin(sampled.index)]
                extra = remaining.sample(n=SAMPLE_PER_QUARTER - len(sampled), random_state=42)
                sampled = pd.concat([sampled, extra])
            sampled_dfs.append(sampled)
            print(f"   {q}: sampled {len(sampled)} from {len(qdf)} "
                  f"({len(sampled)/len(qdf)*100:.0f}%, "
                  f"{sampled['serial'].nunique()} cameras, "
                  f"{sorted(sampled['year'].unique())} years)")

    q1_df = sampled_dfs[0].copy()
    q3_df = sampled_dfs[1].copy()
    total = len(q1_df) + len(q3_df)
    print(f"   Total samples: {total}")

    # --- Extract features with multiprocessing ---
    print(f"\n[3/4] Extracting features ({total} images, {n_workers} workers)...")

    # Prepare work items
    q1_work = [(row['img_path'], i) for i, (_, row) in enumerate(q1_df.iterrows())]
    q3_work = [(row['img_path'], i) for i, (_, row) in enumerate(q3_df.iterrows())]

    # Process Q1
    print(f"   Q1: {len(q1_work)} images...")
    t0 = time.time()
    q1_results = {}
    with Pool(n_workers) as pool:
        for idx, feat in pool.imap_unordered(process_image, q1_work, chunksize=100):
            q1_results[idx] = feat
            done = len(q1_results)
            if done % 2000 == 0 or done == len(q1_work):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(q1_work) - done) / rate if rate > 0 else 0
                print(f"      {done}/{len(q1_work)} ({done/len(q1_work)*100:.0f}%) "
                      f"| {rate:.0f} img/s | ETA: {remaining:.0f}s")
    q1_features = np.array([q1_results[i] for i in range(len(q1_results))])
    print(f"   Q1 done: {q1_features.shape} in {time.time()-t0:.1f}s")

    # Process Q3
    print(f"   Q3: {len(q3_work)} images...")
    t0 = time.time()
    q3_results = {}
    with Pool(n_workers) as pool:
        for idx, feat in pool.imap_unordered(process_image, q3_work, chunksize=100):
            q3_results[idx] = feat
            done = len(q3_results)
            if done % 2000 == 0 or done == len(q3_work):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(q3_work) - done) / rate if rate > 0 else 0
                print(f"      {done}/{len(q3_work)} ({done/len(q3_work)*100:.0f}%) "
                      f"| {rate:.0f} img/s | ETA: {remaining:.0f}s")
    q3_features = np.array([q3_results[i] for i in range(len(q3_results))])
    print(f"   Q3 done: {q3_features.shape} in {time.time()-t0:.1f}s")

    # --- Save everything ---
    print(f"\n[4/4] Saving results...")
    os.makedirs(os.path.join(CAPSTONE_DIR, 'results'), exist_ok=True)
    np.save(os.path.join(CAPSTONE_DIR, 'results', 'q1_features_all.npy'), q1_features)
    np.save(os.path.join(CAPSTONE_DIR, 'results', 'q3_features_all.npy'), q3_features)
    q1_df.to_csv(os.path.join(CAPSTONE_DIR, 'results', 'q1_metadata.csv'), index=False)
    q3_df.to_csv(os.path.join(CAPSTONE_DIR, 'results', 'q3_metadata.csv'), index=False)

    elapsed_total = time.time() - start
    print(f"\n   Done! Total time: {elapsed_total/60:.1f} minutes")
    print(f"   Q1: {q1_features.shape} | Q3: {q3_features.shape}")
    print(f"   Saved to results/q1_features_all.npy, results/q3_features_all.npy")

    for label, meta in [('Q1', q1_df), ('Q3', q3_df)]:
        print(f"\n   {label} breakdown:")
        for dn in ['daytime', 'nighttime']:
            print(f"     {dn}: {len(meta[meta['dn_label']==dn])}")
        print(f"     Years: {sorted(meta['year'].unique())}")
        print(f"     Cameras: {meta['serial'].nunique()}")


if __name__ == '__main__':
    main()
