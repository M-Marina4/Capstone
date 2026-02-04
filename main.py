#!/usr/bin/env python3
"""
Capstone: Multi-ZIP Q1/Q3 Drift Analysis
Monitoring Feature Drift in IoT Sensor Networks for Reliable Continuous Data Streams

Marina Melkonyan | American University of Armenia | February 2026
Paper: https://arxiv.org/html/2512.12205v1

Methodology:
- Unsupervised autoencoder learning of latent representations
- MI-LHD: Metadata-Invariant Latent Histogram Divergence
- STKA: Spatio-Temporal Kernel Alignment
- Time-series decomposition for drift attribution
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.append('src')

from processing import create_q1q3_datasets, batch_extract_features, add_temporal_metadata
from autoencoder import train_autoencoder, extract_latent_representations, save_model
from metrics import analyze_drift
from validator import validate_drift, generate_validation_report
from decomp import compute_time_series_drift_scores, analyze_drift_sources

def main():
    print("="*90)
    print("FEATURE DRIFT MONITORING IN IoT SENSOR NETWORKS")
    print("Unsupervised Detection via Latent Autoencoder Representations")
    print("="*90)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n Device: {device}")
    
    # ============================================================================
    # PHASE 1: LOAD AND PREPARE DATA
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 1: DATA LOADING AND PREPROCESSING")
    print("="*90)
    
    print("\n🔍 Loading Q1/Q3 daytime imagery dataset...")
    q1_data, q3_data = create_q1q3_datasets()
    
    # Add temporal features for drift analysis
    q1_data = add_temporal_metadata(q1_data)
    q3_data = add_temporal_metadata(q3_data)
    
    print(f"\n Dataset Summary:")
    print(f"   Q1: {len(q1_data)} images | Years: {q1_data['year'].min()}-{q1_data['year'].max()}")
    print(f"   Q3: {len(q3_data)} images | Years: {q3_data['year'].min()}-{q3_data['year'].max()}")
    
    # ============================================================================
    # PHASE 2: EXTRACT FEATURES FROM IMAGES
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 2: FEATURE EXTRACTION")
    print("="*90)
    
    print("\n Extracting histogram features from images...")
    q1_features, q1_paths = batch_extract_features(q1_data, feature_type='histogram')
    q3_features, q3_paths = batch_extract_features(q3_data, feature_type='histogram')
    
    print(f"Feature extraction complete:")
    print(f"   Q1 features: {q1_features.shape}")
    print(f"   Q3 features: {q3_features.shape}")
    
    # ============================================================================
    # PHASE 3: TRAIN AUTOENCODER FOR LATENT REPRESENTATIONS
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 3: UNSUPERVISED AUTOENCODER LEARNING")
    print("="*90)
    
    model, history = train_autoencoder(
        q1_features, q3_features, 
        latent_dim=32, 
        epochs=50, 
        batch_size=32, 
        device=device
    )
    
    print("\nAutoencoder training metrics:")
    print(f"   Final reconstruction loss: {history['reconstruction_loss'][-1]:.6f}")
    print(f"   Final KL divergence: {history['kl_loss'][-1]:.6f}")
    
    # Save model
    save_model(model, 'models/autoencoder.pt')
    print(f"Model saved: models/autoencoder.pt")
    
    # ============================================================================
    # PHASE 4: DRIFT ANALYSIS (MI-LHD + STKA)
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 4: DRIFT DETECTION & ANALYSIS")
    print("="*90)
    
    drift_results = analyze_drift(q1_data, q3_data, model=model, device=device)
    
    print(f"\nDrift Metrics Computed:")
    print(f"   MI-LHD (Distributional Divergence): {drift_results['mi_lhd']:.6f}")
    print(f"   STKA (Kernel Alignment):             {drift_results['stka']:.6f}")
    print(f"   Euclidean Distance (Latent Space):  {drift_results['euclidean']:.6f}")
    print(f"   Overall Drift Magnitude:             {drift_results['drift_magnitude']:.2f}%")
    
    # ============================================================================
    # PHASE 5: METADATA VALIDATION & CROSS-DOMAIN ADAPTATION
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 5: METADATA VALIDATION & CROSS-DOMAIN ADAPTATION")
    print("="*90)
    
    validation = validate_drift(q1_data, q3_data, drift_results)
    
    print(f"\n Validation Metrics:")
    print(f"   GPS Spatial Distance:           {validation['gps_distance']:.2f} m")
    print(f"   RGB/Brightness Shift:           {validation['rgb_distance']:.4f}")
    print(f"   Fault Detection Rate Change:    {validation['fault_delta']:+.4f}")
    print(f"   Confidence Score Change:        {validation['confidence_delta']:+.4f}")
    print(f"   Metadata Consistency:           {validation['metadata_consistency']:.4f}")
    print(f"   Cross-Domain Adaptation Score:  {validation['cross_domain_adaptation_score']:.4f}")
    print(f"   Validation Status:              {validation['validation_status']}")
    print(f"   Confidence Level:               {validation['confidence_score']:.2%}")
    
    # ============================================================================
    # PHASE 6: TIME-SERIES DECOMPOSITION (DRIFT ATTRIBUTION)
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 6: TIME-SERIES DECOMPOSITION & DRIFT ATTRIBUTION")
    print("="*90)
    
    decomposition = compute_time_series_drift_scores(q1_data, q3_data, drift_results['mi_lhd'])
    drift_sources = analyze_drift_sources(q1_data, q3_data, validation)
    
    print(f"\n Decomposition Components:")
    print(f"   Trend Component:      {decomposition['trend_component']:.6f}")
    print(f"   Seasonal Component:   {decomposition['seasonal_component']:.6f}")
    print(f"   Residual Component:   {decomposition['residual_component']:.6f}")
    
    print(f"\n Drift Source Attribution:")
    for source, percentage in drift_sources.items():
        print(f"   {source}: {percentage:.2f}%")
    
    # ============================================================================
    # PHASE 7: GENERATE PAPER-READY RESULTS
    # ============================================================================
    print("\n" + "="*90)
    print("RESULTS: Q1 → Q3 FEATURE DRIFT ANALYSIS (PAPER FORMAT)")
    print("="*90)
    
    report = generate_validation_report(q1_data, q3_data, drift_results, validation)
    
    # Format results table
    print(f"\n{'='*90}")
    print(f"{'METRIC':<40} {'VALUE':<20}")
    print(f"{'='*90}")
    
    print(f"\n{'DRIFT DETECTION METRICS':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'MI-LHD [12]':<40} {drift_results['mi_lhd']:<20.6f}")
    print(f"{'STKA [13]':<40} {drift_results['stka']:<20.6f}")
    print(f"{'Euclidean Distance':<40} {drift_results['euclidean']:<20.6f}")
    print(f"{'Drift Magnitude (%)':<40} {drift_results['drift_magnitude']:<20.2f}")
    
    print(f"\n{'VALIDATION METRICS':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'GPS Distance (m)':<40} {validation['gps_distance']:<20.2f}")
    print(f"{'RGB Shift':<40} {validation['rgb_distance']:<20.4f}")
    print(f"{'Fault Detection Δ':<40} {validation['fault_delta']:+<20.4f}")
    print(f"{'Confidence Score Δ':<40} {validation['confidence_delta']:+<20.4f}")
    print(f"{'Metadata Consistency':<40} {validation['metadata_consistency']:<20.4f}")
    
    print(f"\n{'CROSS-DOMAIN ADAPTATION':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Adaptation Score':<40} {validation['cross_domain_adaptation_score']:<20.4f}")
    print(f"{'Validation Status':<40} {validation['validation_status']:<20}")
    print(f"{'Confidence Level':<40} {validation['confidence_score']:><20.2%}")
    
    print(f"\n{'TIME-SERIES DECOMPOSITION':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Trend Component':<40} {decomposition['trend_component']:<20.6f}")
    print(f"{'Seasonal Component':<40} {decomposition['seasonal_component']:<20.6f}")
    print(f"{'Residual Component':<40} {decomposition['residual_component']:<20.6f}")
    
    print(f"\n{'DRIFT SOURCE ATTRIBUTION':<40} {'PERCENTAGE':<20}")
    print(f"{'-'*90}")
    for source, percentage in drift_sources.items():
        label = source.replace('_', ' ').title()
        print(f"{label:<40} {percentage:>19.2f}%")
    
    # ============================================================================
    # PHASE 8: SAVE RESULTS
    # ============================================================================
    print(f"\n" + "="*90)
    print("SAVING RESULTS")
    print("="*90)
    
    # Save detailed results to CSV
    results_df = pd.DataFrame([report])
    results_df.to_csv('results/capstone_results.csv', index=False)
    print(f" Saved: results/capstone_results.csv")
    

if __name__ == "__main__":
    main()