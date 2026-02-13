#!/usr/bin/env python3
"""
Capstone: Monitoring Feature Drift in IoT Sensor Networks
Marina Melkonyan | American University of Armenia

Train autoencoder on Q1+Q3 streetlight imagery, detect drift via
MI-LHD, STKA, and reconstruction-error anomaly detection.

Usage:
    python main.py            # Full pipeline (train + analyze)
    python main.py --phase4   # Skip training, load saved model
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.append('src')

from processing import create_q1q3_datasets, batch_extract_features, add_temporal_metadata
from autoencoder import train_autoencoder, extract_latent_representations, save_model, load_model
from metrics import analyze_drift
from validator import validate_drift, generate_validation_report
from decomp import compute_time_series_drift_scores, analyze_drift_sources
from drift_detectors import DistributionShiftAnalyzer, DriftConfidenceEstimator
from drift_classifier import DriftTypeClassifier, MetadataAnomalyDetector
from anomaly_detector import AnomalyDriftEnsemble


def main():
    print('=' * 80)
    print('FEATURE DRIFT MONITORING IN IoT SENSOR NETWORKS')
    print('=' * 80)

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    skip_training = '--phase4' in sys.argv
    print(f'Device: {device} | Skip training: {skip_training}')

    # --- Phase 1: Load data ---
    print('\n[Phase 1] Loading data...')
    q1_data, q3_data = create_q1q3_datasets()
    q1_data = add_temporal_metadata(q1_data)
    q3_data = add_temporal_metadata(q3_data)
    print(f'  Q1: {len(q1_data)} samples | Q3: {len(q3_data)} samples')

    # --- Phase 2: Extract features ---
    print('\n[Phase 2] Extracting histogram features...')
    q1_features, _ = batch_extract_features(q1_data, feature_type='histogram')
    q3_features, _ = batch_extract_features(q3_data, feature_type='histogram')
    print(f'  Q1: {q1_features.shape} | Q3: {q3_features.shape}')

    # --- Phase 3: Train or load autoencoder ---
    if not skip_training:
        print('\n[Phase 3] Training VAE autoencoder...')
        model, history = train_autoencoder(
            q1_features, q3_features,
            latent_dim=64, epochs=100, batch_size=32, device=device
        )
        save_model(model, 'models/autoencoder.pt')
        print(f'  Final loss: {history["reconstruction_loss"][-1]:.6f}')
    else:
        print('\n[Phase 3] Loading pre-trained model...')
        model = load_model('models/autoencoder.pt')
        model.to(device)

    # Fit anomaly baseline on Q1
    q1_t = torch.from_numpy(q1_features).float().to(device)
    q1_recon, _, _, _ = model(q1_t)
    q1_errors = torch.mean((q1_t - q1_recon) ** 2, dim=1).detach().cpu().numpy()
    anomaly_ensemble = AnomalyDriftEnsemble()
    anomaly_ensemble.fit(q1_errors)

    # --- Phase 4: Drift metrics ---
    print('\n[Phase 4] Computing drift metrics...')
    drift = analyze_drift(q1_data, q3_data, model=model, device=device,
                          q1_features=q1_features, q3_features=q3_features)
    print(f'  MI-LHD:  {drift["mi_lhd"]:.6f}')
    print(f'  STKA:    {drift["stka"]:.6f}')
    print(f'  Euclid:  {drift["euclidean"]:.6f}')
    print(f'  Drift:   {drift["drift_magnitude"]:.2f}%')

    # Enhanced analysis
    latent_q1 = extract_latent_representations(model, q1_features, device)
    latent_q3 = extract_latent_representations(model, q3_features, device)
    enhanced = DistributionShiftAnalyzer().analyze(latent_q1, latent_q3)
    confidence = DriftConfidenceEstimator().estimate_confidence(drift, enhanced)
    drift.update({'enhanced_metrics': enhanced, 'confidence_analysis': confidence})

    # --- Phase 5: Validation ---
    print('\n[Phase 5] Metadata validation...')
    validation = validate_drift(q1_data, q3_data, drift)
    classification = DriftTypeClassifier().classify(drift, validation)
    anomalies_meta = MetadataAnomalyDetector().analyze_metadata_changes(validation)
    validation.update({'drift_classification': classification, 'metadata_anomalies': anomalies_meta})

    # Anomaly detection on Q3
    q3_t = torch.from_numpy(q3_features).float().to(device)
    q3_recon, _, _, _ = model(q3_t)
    q3_errors = torch.mean((q3_t - q3_recon) ** 2, dim=1).detach().cpu().numpy()
    anomaly_results = anomaly_ensemble.detect(q3_errors, validation)
    validation.update({'anomaly_analysis': anomaly_results})

    print(f'  Status: {validation["validation_status"]}')
    print(f'  Anomalous: {anomaly_results["anomalous_ratio"]:.2%}')
    print(f'  Type: {classification["drift_type"]}')

    # --- Phase 6: Decomposition ---
    print('\n[Phase 6] Time-series decomposition...')
    decomp = compute_time_series_drift_scores(q1_data, q3_data, drift['mi_lhd'])
    sources = analyze_drift_sources(q1_data, q3_data, validation)

    # --- Save results ---
    print('\n[Saving] results/capstone_results_v2.csv')
    report = generate_validation_report(q1_data, q3_data, drift, validation)
    pd.DataFrame([report]).to_csv('results/capstone_results_v2.csv', index=False)

    # --- Summary ---
    mag = drift['drift_magnitude']
    level = 'MINIMAL' if mag < 10 else 'LOW' if mag < 20 else 'MODERATE' if mag < 35 else 'HIGH'
    print(f'\n{"=" * 80}')
    print(f'SUMMARY: {mag:.2f}% drift ({level})')
    print(f'  Primary source: {max(sources, key=sources.get)}')
    print(f'  Confidence: {validation["confidence_score"]:.1%}')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
