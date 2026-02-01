#!/usr/bin/env python3
import os
import sys
sys.path.append('src')

from processing import create_dataset
from feature_extr import train_autoencoder, Autoencoder
from metrics import detect_drift_windows
from decomp import decompose_drift_scores
from validator import validate_drift
from utils import extract_latents
import pandas as pd
import numpy as np
import torch

def main(max_samples=2000):
    print("=== Urban IoT Drift Monitoring ===\n")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. LOAD DATASET
    print("1. Loading ZIP dataset...")
    dataset = create_dataset(max_samples=max_samples)
    
    # Get metadata for analysis
    metadata_df = dataset.metadata
    
    # 2. TRAIN MODEL
    print("\n2. Training autoencoder...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_autoencoder(dataset, device=device)
    
    # 3. EXTRACT FEATURES
    print("\n3. Extracting latents...")
    latents = extract_latents(model, dataset, device)
    
    # 4. DRIFT ANALYSIS
    print("\n4. Drift detection...")
    scores = detect_drift_windows(latents, metadata_df)
    scores = decompose_drift_scores(scores)
    
    print("\n5. Validation...")
    validation = validate_drift(scores, metadata_df)
    
    # 5. SAVE
    print("\n6. Saving results...")
    scores.to_csv('results/drift_scores.csv', index=False)
    validation.to_csv('results/validation_results.csv', index=False)
    metadata_df.head(1000).to_csv('results/metadata_sample.csv', index=False)
    
    print("\n🎉 SUCCESS!")
    print(f"📊 Windows: {len(scores)}")
    print(f"🚨 High drift: {len(validation)}")
    print(f"📁 Results in 'results/'")

if __name__ == "__main__":
    main()
