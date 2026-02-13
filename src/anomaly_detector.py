"""
Anomaly-Based Drift Detection - Phase 5 Extension.
Uses reconstruction errors and metadata consistency to identify anomalous samples.
Integrates with Phase 5 validation results.

Papers: [5] Drift Detection Analytics for IoT Sensors
        [8] An encode-then-decompose approach to unsupervised time series anomaly detection
        [7] Intrusion detection in IoT data streams using concept drift localization
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler


class ReconstructionAnomalyDetector:
    """
    Detects anomalous samples using VAE reconstruction error threshold.
    Samples with high reconstruction error may indicate sensor malfunction or degradation.
    
    Works with autoencoder from Phase 3.
    """
    
    def __init__(self, threshold_percentile=95):
        """
        Parameters:
        -----------
        threshold_percentile : int
            Percentile of Q1 reconstruction errors to use as threshold for Q3 anomalies
        """
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.q1_errors = None
    
    def fit_on_q1(self, q1_reconstruction_errors):
        """
        Fit detector on Q1 reconstruction error distribution.
        Establishes baseline of normal behavior.
        
        Parameters:
        -----------
        q1_reconstruction_errors : ndarray
            Per-sample reconstruction errors from Q1 autoencoder forward pass
        """
        self.q1_errors = q1_reconstruction_errors
        self.threshold = np.percentile(q1_reconstruction_errors, self.threshold_percentile)
    
    def detect_anomalies(self, q3_reconstruction_errors):
        """
        Detect anomalies in Q3 based on Q1 threshold.
        
        Parameters:
        -----------
        q3_reconstruction_errors : ndarray
            Per-sample reconstruction errors from Q3
        
        Returns:
        --------
        dict
            Anomaly detection results
        """
        if self.threshold is None:
            raise ValueError("Detector not fitted. Call fit_on_q1() first.")
        
        anomaly_flags = q3_reconstruction_errors > self.threshold
        anomaly_scores = np.clip(q3_reconstruction_errors / self.threshold, 0, 1)
        
        return {
            'anomaly_flags': anomaly_flags,
            'anomaly_scores': anomaly_scores,
            'anomalous_count': int(np.sum(anomaly_flags)),
            'anomalous_ratio': float(np.mean(anomaly_flags)),
            'threshold': float(self.threshold),
            'q1_mean_error': float(np.mean(self.q1_errors)),
            'q3_mean_error': float(np.mean(q3_reconstruction_errors)),
            'error_increase': float(np.mean(q3_reconstruction_errors) / np.mean(self.q1_errors))
        }


class MetadataConsistencyAnomalyDetector:
    """
    Detects metadata inconsistencies indicating sensor problems.
    Uses Phase 5 validation metrics.
    
    E.g., sudden brightness changes, GPS shifts, confidence drops all suggest anomalies.
    """
    
    def __init__(self):
        """Initialize detector"""
        pass
    
    def compute_metadata_anomaly_score(self, validation_results):
        """
        Compute overall metadata anomaly score.
        Higher score = more anomalous metadata behavior.
        
        Parameters:
        -----------
        validation_results : dict
            Results from Phase 5 validator
        
        Returns:
        --------
        dict
            Metadata anomaly analysis
        """
        # Extract metadata deltas (changes Q1→Q3)
        rgb_delta = validation_results.get('rgb_distance', 0)
        fault_delta = abs(validation_results.get('fault_delta', 0))
        confidence_delta = abs(validation_results.get('confidence_delta', 0))
        gps_delta = validation_results.get('gps_distance', 0)
        consistency = validation_results.get('metadata_consistency', 0.5)
        
        # Normalize deltas to [0, 1]
        rgb_score = self._normalize_score(rgb_delta, max_val=0.3)
        fault_score = self._normalize_score(fault_delta, max_val=0.1)
        confidence_score = self._normalize_score(confidence_delta, max_val=0.2)
        gps_score = self._normalize_score(gps_delta, max_val=100.0)
        consistency_score = 1.0 - consistency  # Inverted: low consistency = anomalous
        
        # Combine scores
        overall_score = np.mean([rgb_score, fault_score, confidence_score, gps_score, consistency_score])
        
        # Create scores dictionary for easy lookup
        scores_dict = {
            'rgb': rgb_score,
            'fault_detection': fault_score,
            'confidence': confidence_score,
            'gps': gps_score,
            'consistency': consistency_score
        }
        
        return {
            'overall_anomaly_score': float(np.clip(overall_score, 0, 1)),
            'rgb_anomaly': rgb_score,
            'fault_detection_anomaly': fault_score,
            'confidence_anomaly': confidence_score,
            'gps_anomaly': gps_score,
            'consistency_anomaly': consistency_score,
            'primary_anomaly': max(scores_dict, key=scores_dict.get),
            'is_anomalous': overall_score > 0.6
        }
    
    def _normalize_score(self, value, max_val):
        """Normalize value to [0, 1]"""
        if max_val <= 0:
            return 0.0
        normalized = min(1.0, value / max_val)
        return float(normalized)


class AnomalyDriftEnsemble:
    """
    Combines multiple anomaly detection pathways into unified score.
    Integrates reconstruction errors + metadata consistency.
    
    Input: Phase 3 autoencoder + Phase 4 latent space + Phase 5 validation
    Output: Per-sample anomaly scores and drift summary
    """
    
    def __init__(self):
        """Initialize ensemble"""
        self.reconstruction_detector = ReconstructionAnomalyDetector(threshold_percentile=90)
        self.metadata_detector = MetadataConsistencyAnomalyDetector()
    
    def fit(self, q1_reconstruction_errors):
        """Fit on Q1 baseline"""
        self.reconstruction_detector.fit_on_q1(q1_reconstruction_errors)
    
    def detect(self, q3_reconstruction_errors, validation_results):
        """
        Detect anomalies using ensemble of pathways.
        
        Parameters:
        -----------
        q3_reconstruction_errors : ndarray
            Per-sample reconstruction errors from Q3
        validation_results : dict
            Results from Phase 5 validator
        
        Returns:
        --------
        dict
            Ensemble anomaly detection results
        """
        # Pathway 1: Reconstruction error
        recon_results = self.reconstruction_detector.detect_anomalies(q3_reconstruction_errors)
        recon_scores = recon_results['anomaly_scores']
        
        # Pathway 2: Metadata consistency
        metadata_results = self.metadata_detector.compute_metadata_anomaly_score(validation_results)
        metadata_score = metadata_results['overall_anomaly_score']
        
        # Ensemble: average the two pathways
        # (could use weighted average or voting if preferred)
        ensemble_scores = (recon_scores * 0.5) + (metadata_score * 0.5)
        ensemble_flags = ensemble_scores > 0.5
        
        return {
            'ensemble_anomaly_scores': ensemble_scores,
            'ensemble_anomaly_flags': ensemble_flags,
            'anomalous_count': int(np.sum(ensemble_flags)),
            'anomalous_ratio': float(np.mean(ensemble_flags)),
            'ensemble_confidence': float(np.mean(ensemble_scores)),
            'reconstruction_pathway': recon_results,
            'metadata_pathway': metadata_results,
            'recommendation': self._get_recommendation(metadata_results, np.mean(ensemble_scores))
        }
    
    def _get_recommendation(self, metadata_results, ensemble_mean):
        """Generate actionable recommendation based on anomalies"""
        if ensemble_mean < 0.3:
            return "No anomalies detected. Drift appears nominal."
        elif ensemble_mean < 0.6:
            recommendation = "Minor anomalies detected. Monitor closely."
        else:
            recommendation = "Significant anomalies detected. Recommend investigation."
        
        if metadata_results['is_anomalous']:
            recommendation += f" Primary anomaly: {metadata_results['primary_anomaly']}"
        
        return recommendation


