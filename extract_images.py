#!/usr/bin/env python3
"""
Extract and organize Q1/Q3 images from camera zip files.
Uses the streetcare-drift-dataset CSV to determine which images belong
to Q1/Q3 and whether they are daytime/nighttime.

Output structure:
    data/organized_images/Q1/daytime/
    data/organized_images/Q1/nighttime/
    data/organized_images/Q3/daytime/
    data/organized_images/Q3/nighttime/
"""

import os
import sys
import zipfile
import pandas as pd
from collections import defaultdict
import time
import shutil

# Paths
MARINA_DIR = r"C:\Users\Admin\OneDrive\Desktop\Work\AUA\Capstone Projects\Marina"
CAPSTONE_DIR = os.path.join(MARINA_DIR, "Capstone")
CSV_PATH = os.path.join(CAPSTONE_DIR, "data", "metadata", "streetcare-drift-dataset-2021-2025.csv")
OUTPUT_DIR = os.path.join(CAPSTONE_DIR, "data", "organized_images")


def main():
    start = time.time()

    # --- Step 1: Load CSV and build lookup ---
    print("[1/4] Loading CSV metadata...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Filter to Q1 and Q3 only
    mask = df['time_tag'].str.contains('Q1|Q3', na=False)
    df_q = df[mask].copy()

    # Determine quarter and daynight category
    df_q['quarter'] = df_q['time_tag'].str.extract(r'(Q[13])')[0]
    df_q['dn_label'] = df_q['daynight'].map({0.0: 'daytime', 1.0: 'nighttime'})

    # Drop rows with missing required info
    df_q = df_q.dropna(subset=['image_name', 'serial', 'quarter', 'dn_label'])

    print(f"   Total images to extract: {len(df_q)}")
    for q in ['Q1', 'Q3']:
        for dn in ['daytime', 'nighttime']:
            count = len(df_q[(df_q['quarter'] == q) & (df_q['dn_label'] == dn)])
            print(f"   {q}/{dn}: {count}")

    # Build lookup: serial -> list of (image_name, quarter, dn_label)
    serial_images = defaultdict(list)
    for _, row in df_q.iterrows():
        serial_images[row['serial']].append((row['image_name'], row['quarter'], row['dn_label']))

    print(f"   Cameras to process: {len(serial_images)}")

    # --- Step 2: Create new output structure ---
    print("\n[2/4] Preparing output directories...")
    for q in ['Q1', 'Q3']:
        for dn in ['daytime', 'nighttime']:
            path = os.path.join(OUTPUT_DIR, q, dn)
            os.makedirs(path, exist_ok=True)
            print(f"   Created: {path}")

    # --- Step 3: Extract images from each zip ---
    print("\n[3/4] Extracting images from zip files...")
    total_extracted = 0
    total_missing = 0
    serials = sorted(serial_images.keys())

    for idx, serial in enumerate(serials):
        zip_path = os.path.join(MARINA_DIR, f"{serial}.zip")
        images_for_serial = serial_images[serial]

        if not os.path.exists(zip_path):
            print(f"   [{idx+1}/{len(serials)}] SKIP {serial} - zip not found")
            total_missing += len(images_for_serial)
            continue

        print(f"   [{idx+1}/{len(serials)}] {serial}: {len(images_for_serial)} images...", end=" ", flush=True)
        t0 = time.time()

        # Build a set of needed image names for fast lookup
        needed = {img: (q, dn) for img, q, dn in images_for_serial}

        extracted = 0
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in needed:
                    q, dn = needed[basename]
                    dest_dir = os.path.join(OUTPUT_DIR, q, dn)
                    dest_file = os.path.join(dest_dir, basename)

                    # Extract directly to destination
                    with zf.open(name) as src, open(dest_file, 'wb') as dst:
                        dst.write(src.read())

                    extracted += 1

        elapsed = time.time() - t0
        not_found = len(images_for_serial) - extracted
        total_extracted += extracted
        total_missing += not_found
        print(f"extracted {extracted}, missing {not_found}, {elapsed:.1f}s")

    # --- Step 4: Summary ---
    print(f"\n[4/4] Done!")
    print(f"   Total extracted: {total_extracted}")
    print(f"   Total missing:   {total_missing}")
    
    # Count actual files
    for q in ['Q1', 'Q3']:
        for dn in ['daytime', 'nighttime']:
            path = os.path.join(OUTPUT_DIR, q, dn)
            count = len(os.listdir(path))
            print(f"   {q}/{dn}: {count} files")

    elapsed_total = time.time() - start
    print(f"   Total time: {elapsed_total/60:.1f} minutes")

    # --- Step 5: Save updated metadata CSV ---
    print("\nSaving filtered metadata CSV...")
    df_q.to_csv(os.path.join(CAPSTONE_DIR, "data", "metadata", "q1q3_all_extracted.csv"), index=False)
    print("   Saved: data/metadata/q1q3_all_extracted.csv")


if __name__ == '__main__':
    main()
