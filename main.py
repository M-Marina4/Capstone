#!/usr/bin/env python3
"""
Capstone:Monitoring Feature Drift in IoT Sensor Networks for Reliable Continuous Data Streams

Marina Melkonyan | American University of Armenia

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
import json

sys.path.append('src')

from processing import create_q1q3_datasets, batch_extract_features, add_temporal_metadata
from autoencoder import train_autoencoder, extract_latent_representations, save_model, load_model
from metrics import analyze_drift
from validator import validate_drift, generate_validation_report
from decomp import compute_time_series_drift_scores, analyze_drift_sources
from drift_detectors import DistributionShiftAnalyzer,  DriftConfidenceEstimator
from drift_classifier import DriftTypeClassifier, MetadataAnomalyDetector
from anomaly_detector import AnomalyDriftEnsemble

def run_notebook_pipeline():
    """Execute organize_images_by_quarter notebook by running cells directly"""
    notebook_path = "notebooks/organize_images_by_quarter.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"Notebook not found: {notebook_path}")
        return
    
    try:
        # Load notebook with UTF-8 encoding
        with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
            notebook = json.load(f)
        
        # Change to notebooks directory for relative paths
        original_dir = os.getcwd()
        os.chdir('notebooks')
        
        print("\n" + "="*90)
        print("EXECUTING: organize_images_by_quarter.ipynb")
        print("="*90)
        
        # Execute each cell
        namespace = {
            'pd': pd,
            'np': np,
            'os': os,
            'zipfile': __import__('zipfile'),
            'datetime': __import__('datetime'),
            'Path': __import__('pathlib').Path,
            're': __import__('re')
        }
        
        cell_count = 0
        for i, cell in enumerate(notebook.get('cells', [])):
            if cell['cell_type'] == 'code':
                code = ''.join(cell['source'])
                if code.strip():
                    try:
                        exec(code, namespace)
                        cell_count += 1
                    except Exception as e:
                        print(f"[Cell {i+1}] {type(e).__name__}: {str(e)[:150]}")
        
        os.chdir(original_dir)
        print(f"\n✓ Notebook pipeline completed ({cell_count} cells executed)")
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"Pipeline error: {e}")

def main():
    print("="*90)
    print("FEATURE DRIFT MONITORING IN IoT SENSOR NETWORKS")
    print("Unsupervised Detection via Latent Autoencoder Representations")
    print("="*90)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n Device: {device}")
    
    # Check if running from phase 4
    run_from_phase4 = '--phase4' in sys.argv
    
    if not run_from_phase4:
        # Run notebook pipeline first
        run_notebook_pipeline()
        
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
        print(f"   Total samples: {len(q1_data) + len(q3_data)}")
        
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
            latent_dim=64, 
            epochs=100, 
            batch_size=32, 
            device=device
        )
        
        print("\nAutoencoder training metrics:")
        print(f"   Final reconstruction loss: {history['reconstruction_loss'][-1]:.6f}")
        print(f"   Final KL divergence: {history['kl_loss'][-1]:.6f}")
        print(f"   Loss improvement: {(history['reconstruction_loss'][0] - history['reconstruction_loss'][-1])/history['reconstruction_loss'][0]*100:.2f}%")
        print(f"   Convergence status: {'✓ Converged' if history['reconstruction_loss'][-1] < 0.25 else '⚠ May need more epochs'}")
        
        # Save model
        save_model(model, 'models/autoencoder.pt')
        print(f"Model saved: models/autoencoder.pt")
        
        # Fit anomaly ensemble on Q1 reconstruction errors
        print("\nFitting anomaly detection ensemble on Q1 baseline...")
        q1_recon, _, _, _ = model(torch.from_numpy(q1_features).float().to(device))
        q1_recon_errors = torch.mean((torch.from_numpy(q1_features).float() - q1_recon) ** 2, dim=1).detach().cpu().numpy()
        
        anomaly_ensemble = AnomalyDriftEnsemble()
        anomaly_ensemble.fit(q1_recon_errors)
        print(f"Ensemble fitted on {len(q1_recon_errors)} Q1 baseline samples")
    else:
        # PHASE 4: LOAD PRECOMPUTED DATA AND MODEL
        print("\n" + "="*90)
        print("SKIPPING PHASES 1-3: LOADING PRECOMPUTED DATA AND MODEL")
        print("="*90)
        
        print("\n🔍 Loading Q1/Q3 daytime imagery dataset...")
        q1_data, q3_data = create_q1q3_datasets()
        
        # Add temporal features for drift analysis
        q1_data = add_temporal_metadata(q1_data)
        q3_data = add_temporal_metadata(q3_data)
        
        print(f"\n Dataset Summary:")
        print(f"   Q1: {len(q1_data)} images | Years: {q1_data['year'].min()}-{q1_data['year'].max()}")
        print(f"   Q3: {len(q3_data)} images | Years: {q3_data['year'].min()}-{q3_data['year'].max()}")
        print(f"   Total samples: {len(q1_data) + len(q3_data)}")
        
        print("\n Extracting histogram features from images...")
        q1_features, q1_paths = batch_extract_features(q1_data, feature_type='histogram')
        q3_features, q3_paths = batch_extract_features(q3_data, feature_type='histogram')
        
        print(f"Feature extraction complete:")
        print(f"   Q1 features: {q1_features.shape}")
        print(f"   Q3 features: {q3_features.shape}")
        
        print("\n Loading pre-trained model...")
        model = load_model('models/autoencoder.pt')
        model.to(device)
        print(f"Model loaded successfully")
        
        # Fit anomaly ensemble on Q1 reconstruction errors
        print("\nFitting anomaly detection ensemble on Q1 baseline...")
        q1_recon, _, _, _ = model(torch.from_numpy(q1_features).float().to(device))
        q1_recon_errors = torch.mean((torch.from_numpy(q1_features).float() - q1_recon) ** 2, dim=1).detach().cpu().numpy()
        
        anomaly_ensemble = AnomalyDriftEnsemble()
        anomaly_ensemble.fit(q1_recon_errors)
        print(f"Ensemble fitted on {len(q1_recon_errors)} Q1 baseline samples")
     
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
    
    # Interpret drift level
    drift_mag = drift_results['drift_magnitude']
    if drift_mag < 10:
        drift_level = "MINIMAL"
    elif drift_mag < 20:
        drift_level = "LOW"
    elif drift_mag < 35:
        drift_level = "MODERATE"
    else:
        drift_level = "HIGH"
    print(f"   Drift Level: {drift_level}")
    
    # ============================================================================
    # PHASE 4 EXTENSION: ENHANCED DRIFT ANALYSIS
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 4 EXTENSION: ENHANCED STATISTICAL ANALYSIS")
    print("="*90)
    
    # Extract latent representations for extended analysis
    latent_q1 = extract_latent_representations(model, q1_features, device)
    latent_q3 = extract_latent_representations(model, q3_features, device)
    
    # Compute additional statistical pathways
    dist_analyzer = DistributionShiftAnalyzer()
    enhanced_metrics = dist_analyzer.analyze(latent_q1, latent_q3)
    
    print(f"\nEnhanced Metrics (4 Pathways):")
    print(f"   Pathway 1 - Mean Shift:       {enhanced_metrics['mean_shift']:.6f}")
    print(f"   Pathway 2 - Covariance Shift: {enhanced_metrics['covariance_shift']:.6f}")
    print(f"   Pathway 3 - KS Rejection:     {enhanced_metrics['ks_rejection_rate']:.4f}")
    print(f"   Pathway 4 - Mahalanobis Dist: {enhanced_metrics['mahalanobis_distance']:.6f}")
    
    # Compute confidence in drift detection
    confidence_estimator = DriftConfidenceEstimator()
    confidence_results = confidence_estimator.estimate_confidence(drift_results, enhanced_metrics)
    
    print(f"\nDrift Confidence Analysis:")
    print(f"   Overall Confidence: {confidence_results['overall_confidence']:.2%}")
    print(f"   Pathway Consistency: {confidence_results['consistency_score']:.2%}")
    print(f"   Drift Confirmed: {confidence_results['drift_confirmed']}")
    
    # Merge enhanced metrics into drift results
    drift_results.update({
        'enhanced_metrics': enhanced_metrics,
        'confidence_analysis': confidence_results
    })
    
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
    # PHASE 5 EXTENSION: DRIFT TYPE CLASSIFICATION
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 5 EXTENSION: DRIFT TYPE CLASSIFICATION")
    print("="*90)
    
    drift_classifier = DriftTypeClassifier()
    drift_classification = drift_classifier.classify(drift_results, validation)
    
    print(f"\nDrift Type Analysis:")
    print(f"   Drift Type: {drift_classification['drift_type']}")
    print(f"   Confidence: {drift_classification['confidence']:.2%}")
    print(f"   Virtual Score: {drift_classification['virtual_score']:.2%}")
    print(f"   Real Score: {drift_classification['real_score']:.2%}")
    print(f"   Explanation: {drift_classification['explanation']}")
    
    # Metadata anomaly analysis
    metadata_analyzer = MetadataAnomalyDetector()
    metadata_anomalies = metadata_analyzer.analyze_metadata_changes(validation)
    
    print(f"\nMetadata Anomaly Analysis:")
    print(f"   Brightness Anomaly: {metadata_anomalies['brightness_anomaly']:.2%}")
    print(f"   Fault Detection Anomaly: {metadata_anomalies['fault_detection_anomaly']:.2%}")
    print(f"   Confidence Anomaly: {metadata_anomalies['confidence_anomaly']:.2%}")
    print(f"   GPS Anomaly: {metadata_anomalies['gps_anomaly']:.2%}")
    print(f"   Most Anomalous Field: {metadata_anomalies['most_anomalous']}")
    
    # Merge into validation results
    validation.update({
        'drift_classification': drift_classification,
        'metadata_anomalies': metadata_anomalies
    })
    
    # ============================================================================
    # PHASE 5 EXTENSION: ANOMALY-BASED DRIFT DETECTION
    # ============================================================================
    print("\n" + "="*90)
    print("PHASE 5 EXTENSION: ANOMALY-BASED DRIFT DETECTION")
    print("="*90)
    
    # Compute reconstruction errors for Q3
    q3_recon, _, _, _ = model(torch.from_numpy(q3_features).float().to(device))
    q3_recon_errors = torch.mean((torch.from_numpy(q3_features).float() - q3_recon) ** 2, dim=1).detach().cpu().numpy()
    
    # Detect anomalies using pre-fitted ensemble
    anomaly_results = anomaly_ensemble.detect(q3_recon_errors, validation)
    
    print(f"\nAnomaly Detection Ensemble:")
    print(f"   Anomalous Samples: {anomaly_results['anomalous_count']} / {len(q3_recon_errors)} ({anomaly_results['anomalous_ratio']:.2%})")
    print(f"   Ensemble Confidence: {anomaly_results['ensemble_confidence']:.2%}")
    print(f"   Primary Anomaly Type: {anomaly_results['metadata_pathway']['primary_anomaly']}")
    print(f"   Recommendation: {anomaly_results['recommendation']}")
    
    # Merge into validation results
    validation.update({
        'anomaly_analysis': anomaly_results
    })
    
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
    print("RESULTS: Q1 → Q3 FEATURE DRIFT ANALYSIS")
    print("="*90)
    
    report = generate_validation_report(q1_data, q3_data, drift_results, validation)
    
    # Format results table
    print(f"\n{'='*90}")
    print(f"{'METRIC':<40} {'VALUE':<20}")
    print(f"{'='*90}")
    
    print(f"\n{'DRIFT DETECTION METRICS':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'MI-LHD [12]':<40} {drift_results['mi_lhd']:>20.6f}")
    print(f"{'STKA [13]':<40} {drift_results['stka']:>20.6f}")
    print(f"{'Euclidean Distance':<40} {drift_results['euclidean']:>20.6f}")
    print(f"{'Drift Magnitude (%)':<40} {drift_results['drift_magnitude']:>20.2f}")
    
    print(f"\n{'VALIDATION METRICS':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'GPS Distance (m)':<40} {validation['gps_distance']:>20.2f}")
    print(f"{'RGB Shift':<40} {validation['rgb_distance']:>20.4f}")
    print(f"{'Fault Detection Δ':<40} {validation['fault_delta']:>+20.4f}")
    print(f"{'Confidence Score Δ':<40} {validation['confidence_delta']:>+20.4f}")
    print(f"{'Metadata Consistency':<40} {validation['metadata_consistency']:>20.4f}")
    
    print(f"\n{'CROSS-DOMAIN ADAPTATION':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Adaptation Score':<40} {validation['cross_domain_adaptation_score']:>20.4f}")
    print(f"{'Validation Status':<40} {validation['validation_status']:>20}")
    print(f"{'Confidence Level':<40} {validation['confidence_score']:>20.2%}")
    
    print(f"\n{'TIME-SERIES DECOMPOSITION':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Trend Component':<40} {decomposition['trend_component']:>20.6f}")
    print(f"{'Seasonal Component':<40} {decomposition['seasonal_component']:>20.6f}")
    print(f"{'Residual Component':<40} {decomposition['residual_component']:>20.6f}")
    
    print(f"\n{'DRIFT SOURCE ATTRIBUTION':<40} {'PERCENTAGE':<20}")
    print(f"{'-'*90}")
    for source, percentage in drift_sources.items():
        label = source.replace('_', ' ').title()
        print(f"{label:<40} {percentage:>19.2f}%")
    
    print(f"\n{'ENHANCED DRIFT ANALYSIS (Phase 4 Extension)':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Mean Shift (Pathway 1)':<40} {enhanced_metrics['mean_shift']:>20.6f}")
    print(f"{'Covariance Shift (Pathway 2)':<40} {enhanced_metrics['covariance_shift']:>20.6f}")
    print(f"{'KS Rejection Rate (Pathway 3)':<40} {enhanced_metrics['ks_rejection_rate']:>20.4f}")
    print(f"{'Mahalanobis Distance (Pathway 4)':<40} {enhanced_metrics['mahalanobis_distance']:>20.6f}")
    
    print(f"\n{'DRIFT CONFIDENCE ANALYSIS':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Overall Confidence':<40} {confidence_results['overall_confidence']:>20.2%}")
    print(f"{'Pathway Consistency':<40} {confidence_results['consistency_score']:>20.2%}")
    print(f"{'Drift Confirmed':<40} {str(confidence_results['drift_confirmed']):>20}")
    
    print(f"\n{'DRIFT TYPE CLASSIFICATION':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Drift Type':<40} {drift_classification['drift_type']:>20}")
    print(f"{'Classification Confidence':<40} {drift_classification['confidence']:>20.2%}")
    print(f"{'Virtual Score':<40} {drift_classification['virtual_score']:>20.2%}")
    print(f"{'Real Score':<40} {drift_classification['real_score']:>20.2%}")
    
    print(f"\n{'ANOMALY DETECTION ANALYSIS':<40} {'VALUE':<20}")
    print(f"{'-'*90}")
    print(f"{'Anomalous Samples (%)':<40} {anomaly_results['anomalous_ratio']:>20.2%}")
    print(f"{'Ensemble Confidence':<40} {anomaly_results['ensemble_confidence']:>20.2%}")
    print(f"{'Primary Anomaly Type':<40} {anomaly_results['metadata_pathway']['primary_anomaly']:>20}")
    
    # ============================================================================
    # PHASE 8: GENERATE VISUALIZATIONS
    # ============================================================================
    print(f"\n" + "="*90)
    print("PHASE 8: VISUALIZATIONS")
    print("="*90)
    print("\n📊 Visualizations can be generated using the interactive notebook:")
    print("   notebooks/visualizations.ipynb")
    print("\nTo generate plots:")
    print("   1. Open: notebooks/visualizations.ipynb")
    print("   2. Run all cells to generate visualizations")
    print("   3. Plots saved to: results/plots/")
    print("\n✓ Visualization notebook ready for execution")
    
    # ============================================================================
    # PHASE 9: SAVE RESULTS
    # ============================================================================
    print(f"\n" + "="*90)
    print("PHASE 9: SAVING RESULTS")
    print("="*90)
    
    # Save detailed results to CSV
    results_df = pd.DataFrame([report])
    results_df.to_csv('results/capstone_results.csv', index=False)
    print(f" Saved: results/capstone_results.csv")
    
    # ============================================================================
    # PHASE 10: EXECUTIVE SUMMARY
    # ============================================================================
    print(f"\n" + "="*90)
    print("EXECUTIVE SUMMARY")
    print("="*90)
    
    print(f"\n📊 Drift Detection Summary:")
    print(f"   • Detected drift magnitude: {drift_results['drift_magnitude']:.2f}% ({drift_level})")
    print(f"   • Primary drift source: {max(drift_sources, key=drift_sources.get)}")
    print(f"   • Model confidence: {validation['confidence_score']:.1%}")
    
    print(f"\n🔍 Key Findings:")
    print(f"   • Fault detection changes drive {drift_sources['fault_detection_drift']:.1f}% of drift")
    print(f"   • Environmental factors account for {drift_sources['environmental_drift']:.1f}%")
    print(f"   • Temporal patterns contribute {drift_sources['temporal_drift']:.1f}%")
    
    print(f"\n✓ Analysis complete. Results saved to results/capstone_results.csv")
    print("="*90)
    

if __name__ == "__main__":
    main()