#!/usr/bin/env python3
"""
Capstone: Monitoring Feature Drift in IoT Sensor Networks
Marina Melkonyan | American University of Armenia

Train autoencoder on Q1+Q3 streetlight imagery, detect drift via
MI-LHD, STKA, and reconstruction-error anomaly detection.

Dataset: ~240K images from 22 streetlight cameras (Bristol, UK)
  - Q1 (Jan-Mar): daytime + nighttime
  - Q3 (Jul-Sep): daytime + nighttime

Usage:
    python main.py                    # Full pipeline (train + analyze)
    python main.py --phase4            # Skip training, load saved model
    python main.py --daynight daytime  # Filter to daytime only
"""

import os
import sys
import time
import torch
import pandas as pd
import numpy as np

# Force unbuffered stdout for real-time output
import functools
print = functools.partial(print, flush=True)

sys.path.append('src')

from processing import create_q1q3_datasets, batch_extract_features, batch_extract_cnn_features, add_temporal_metadata
from autoencoder import train_autoencoder, extract_latent_representations, save_model, load_model
from metrics import analyze_drift, bootstrap_drift_confidence, per_camera_drift
from validator import validate_drift, generate_validation_report
from decomp import compute_time_series_drift_scores, analyze_drift_sources
from drift_detectors import DistributionShiftAnalyzer, DriftConfidenceEstimator
from drift_classifier import DriftTypeClassifier, MetadataAnomalyDetector
from anomaly_detector import AnomalyDriftEnsemble


def batched_reconstruction_errors(model, features, device, batch_size=512):
    """Compute reconstruction MSE errors in batches to avoid OOM."""
    model.eval()
    all_errors = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[i:i+batch_size]).float().to(device)
            recon, _, _, _ = model(batch)
            errors = torch.mean((batch - recon) ** 2, dim=1)
            all_errors.append(errors.cpu().numpy())
    return np.concatenate(all_errors)


def main():
    pipeline_start = time.time()
    print('=' * 80)
    print('FEATURE DRIFT MONITORING IN IoT SENSOR NETWORKS')
    print('Full dataset: ~240K images | 22 cameras | 2021-2025')
    print('=' * 80)

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    skip_training = '--phase4' in sys.argv
    
    # Parse daynight filter
    daynight_filter = None
    if '--daynight' in sys.argv:
        idx = sys.argv.index('--daynight')
        if idx + 1 < len(sys.argv):
            daynight_filter = sys.argv[idx + 1]
    
    print(f'Device: {device} | Skip training: {skip_training} | Daynight filter: {daynight_filter or "all"}')

    # --- Phase 1+2: Load data and features ---
    cache_suffix = f'_{daynight_filter}' if daynight_filter else '_all'
    feature_type = 'cnn'  # v6: use ResNet18 features (512-dim) instead of histograms (768-dim)
    q1_cache = f'results/q1_features_{feature_type}_all.npy'
    q3_cache = f'results/q3_features_{feature_type}_all.npy'
    q1_meta_cache = f'results/q1_metadata.csv'
    q3_meta_cache = f'results/q3_metadata.csv'
    
    # Check if we have pre-extracted features + matching metadata
    have_cache = (os.path.exists(q1_cache) and os.path.exists(q3_cache) and
                  os.path.exists(q1_meta_cache) and os.path.exists(q3_meta_cache))
    
    if have_cache:
        print('\n[Phase 1+2] Loading cached features and metadata...')
        q1_features = np.load(q1_cache)
        q3_features = np.load(q3_cache)
        q1_data = pd.read_csv(q1_meta_cache, low_memory=False)
        q3_data = pd.read_csv(q3_meta_cache, low_memory=False)
        q1_data['date'] = pd.to_datetime(q1_data['date'], errors='coerce')
        q3_data['date'] = pd.to_datetime(q3_data['date'], errors='coerce')
        q1_data = add_temporal_metadata(q1_data)
        q3_data = add_temporal_metadata(q3_data)
        # Ensure metadata rows match feature rows
        assert len(q1_data) == len(q1_features), f'Q1 mismatch: {len(q1_data)} meta vs {len(q1_features)} features'
        assert len(q3_data) == len(q3_features), f'Q3 mismatch: {len(q3_data)} meta vs {len(q3_features)} features'
        
        # Apply daynight filter AFTER loading (slice metadata + features together)
        if daynight_filter is not None:
            # Map daynight column to label
            q1_dn = q1_data['daynight'].map({0.0: 'daytime', 0: 'daytime', 1.0: 'nighttime', 1: 'nighttime'})
            q3_dn = q3_data['daynight'].map({0.0: 'daytime', 0: 'daytime', 1.0: 'nighttime', 1: 'nighttime'})
            q1_mask = (q1_dn == daynight_filter).values
            q3_mask = (q3_dn == daynight_filter).values
            q1_data = q1_data[q1_mask].reset_index(drop=True)
            q3_data = q3_data[q3_mask].reset_index(drop=True)
            q1_features = q1_features[q1_mask]
            q3_features = q3_features[q3_mask]
            print(f'  Filtered to {daynight_filter}: Q1={len(q1_data)}, Q3={len(q3_data)}')
        
        print(f'  Q1: {q1_features.shape} features, {len(q1_data)} metadata rows')
        print(f'  Q3: {q3_features.shape} features, {len(q3_data)} metadata rows')
    else:
        print('\n[Phase 1] Loading data...')
        q1_data, q3_data = create_q1q3_datasets(daynight_filter=daynight_filter)
        q1_data = add_temporal_metadata(q1_data)
        q3_data = add_temporal_metadata(q3_data)
        print(f'  Q1: {len(q1_data)} samples | Q3: {len(q3_data)} samples')

        print(f'\n[Phase 2] Extracting CNN features (ResNet18, 512-dim)...')
        t0 = time.time()
        print(f'  Extracting Q1 CNN features ({len(q1_data)} images)...')
        q1_features, _ = batch_extract_cnn_features(q1_data, device=device, batch_size=64)
        print(f'  Extracting Q3 CNN features ({len(q3_data)} images)...')
        q3_features, _ = batch_extract_cnn_features(q3_data, device=device, batch_size=64)
        elapsed = time.time() - t0
        print(f'  Q1: {q1_features.shape} | Q3: {q3_features.shape} | Time: {elapsed/60:.1f} min')
        
        # Cache features
        np.save(q1_cache, q1_features)
        np.save(q3_cache, q3_features)
        print(f'  Features cached to {q1_cache} and {q3_cache}')

    # --- Phase 3: Train or load autoencoder ---
    if not skip_training:
        print('\n[Phase 3] Training VAE autoencoder (v3: KL annealing + early stopping)...')
        model, history = train_autoencoder(
            q1_features, q3_features,
            latent_dim=128, epochs=80, batch_size=64, device=device
        )
        save_model(model, f'models/autoencoder_v3{cache_suffix}.pt')
        print(f'  Final recon loss: {history["reconstruction_loss"][-1]:.6f}')
        if history['val_loss']:
            print(f'  Best val loss: {min(history["val_loss"]):.6f}')
    else:
        print('\n[Phase 3] Loading pre-trained model...')
        model = load_model(f'models/autoencoder_v3{cache_suffix}.pt')
        model.to(device)

    # Fit anomaly baseline on Q1 (batched)
    print('  Fitting anomaly baseline on Q1...')
    q1_errors = batched_reconstruction_errors(model, q1_features, device)
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

    # --- Phase 4b: Bootstrap confidence intervals ---
    print('\n[Phase 4b] Bootstrap confidence intervals (20 iterations)...')
    bootstrap_ci = bootstrap_drift_confidence(latent_q1, latent_q3, n_bootstrap=20)
    print(f'  MI-LHD: {bootstrap_ci["mi_lhd"]["mean"]:.4f} ± {bootstrap_ci["mi_lhd"]["std"]:.4f} '
          f'  95% CI: [{bootstrap_ci["mi_lhd"]["ci_low"]:.4f}, {bootstrap_ci["mi_lhd"]["ci_high"]:.4f}]')
    print(f'  STKA:   {bootstrap_ci["stka"]["mean"]:.4f} ± {bootstrap_ci["stka"]["std"]:.4f} '
          f'  95% CI: [{bootstrap_ci["stka"]["ci_low"]:.4f}, {bootstrap_ci["stka"]["ci_high"]:.4f}]')
    drift['bootstrap_ci'] = bootstrap_ci

    # --- Phase 4c: Per-camera drift analysis ---
    print('\n[Phase 4c] Per-camera drift analysis...')
    # Identify camera column
    cam_col = None
    for c in ['hostname', 'camera_id', 'camera', 'cam_id', 'CameraName', 'cameraname']:
        if c in q1_data.columns:
            cam_col = c
            break
    if cam_col is None:
        # Try to extract from image_name (e.g., "bristolcam_001_..." pattern)
        cam_col = '_camera_extracted'
        q1_data[cam_col] = q1_data['image_name'].astype(str).str.extract(r'(cam\d+|camera\d+|[A-Za-z]+_\d+)', expand=False).fillna('unknown')
        q3_data[cam_col] = q3_data['image_name'].astype(str).str.extract(r'(cam\d+|camera\d+|[A-Za-z]+_\d+)', expand=False).fillna('unknown')
    
    camera_drift = per_camera_drift(latent_q1, latent_q3,
                                     q1_data[cam_col].values, q3_data[cam_col].values)
    valid_cams = [c for c in camera_drift if c.get('drift_magnitude') is not None]
    print(f'  Analyzed {len(valid_cams)} cameras')
    if valid_cams:
        top3 = valid_cams[:3]
        bot3 = valid_cams[-3:]
        top_str = ", ".join(f"{c['camera']}={c['drift_magnitude']:.1f}%" for c in top3)
        bot_str = ", ".join(f"{c['camera']}={c['drift_magnitude']:.1f}%" for c in bot3)
        print(f'  Highest drift: {top_str}')
        print(f'  Lowest drift:  {bot_str}')
    drift['per_camera'] = camera_drift

    # --- Phase 5: Validation ---
    print('\n[Phase 5] Metadata validation...')
    validation = validate_drift(q1_data, q3_data, drift)
    classification = DriftTypeClassifier().classify(drift, validation)
    anomalies_meta = MetadataAnomalyDetector().analyze_metadata_changes(validation)
    validation.update({'drift_classification': classification, 'metadata_anomalies': anomalies_meta})

    # Anomaly detection on Q3 (batched)
    q3_errors = batched_reconstruction_errors(model, q3_features, device)
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
    result_file = f'results/capstone_results_v6{cache_suffix}.csv'
    print(f'\n[Saving] {result_file}')
    report = generate_validation_report(q1_data, q3_data, drift, validation)
    pd.DataFrame([report]).to_csv(result_file, index=False)

    # --- Summary ---
    mag = drift['drift_magnitude']
    level = 'MINIMAL' if mag < 10 else 'LOW' if mag < 20 else 'MODERATE' if mag < 35 else 'HIGH'
    elapsed_total = time.time() - pipeline_start
    print(f'\n{"=" * 80}')
    print(f'SUMMARY: {mag:.2f}% drift ({level})')
    print(f'  Dataset: Q1={len(q1_data)} + Q3={len(q3_data)} = {len(q1_data)+len(q3_data)} images')
    print(f'  Features: ResNet18 (512-dim CNN)')
    print(f'  Primary source: {max(sources, key=sources.get)}')
    print(f'  Confidence: {validation["confidence_score"]:.1%}')
    print(f'  Bootstrap MI-LHD: {bootstrap_ci["mi_lhd"]["mean"]:.4f} [{bootstrap_ci["mi_lhd"]["ci_low"]:.4f}, {bootstrap_ci["mi_lhd"]["ci_high"]:.4f}]')
    print(f'  Bootstrap STKA:   {bootstrap_ci["stka"]["mean"]:.4f} [{bootstrap_ci["stka"]["ci_low"]:.4f}, {bootstrap_ci["stka"]["ci_high"]:.4f}]')
    if valid_cams:
        worst = valid_cams[0]
        best = valid_cams[-1]
        print(f'  Worst camera: {worst["camera"]} ({worst["drift_magnitude"]:.1f}%)')
        print(f'  Best camera:  {best["camera"]} ({best["drift_magnitude"]:.1f}%)')
    print(f'  Total pipeline time: {elapsed_total/60:.1f} minutes')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
