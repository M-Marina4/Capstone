"""
Enhanced Statistical Drift Analysis - Phase 4 Extension.
Computes additional drift confirmation metrics to strengthen Phase 4 results.

Papers: [2] Concept Drift Detection Methods Based on Different Weighting Strategies
        [10] One or two things we know about concept drift - detecting concept drift
"""

import numpy as np
from scipy import stats


class DistributionShiftAnalyzer:
    """
    Analyzes distribution shifts using multiple statistical tests.
    Complements MI-LHD metric from Phase 4.
    
    Input: Latent representations from Phase 3 (autoencoder)
    Output: Multiple pathway scores confirming drift direction and magnitude
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.pathways = {}
    
    def analyze(self, latent_q1, latent_q3):
        """
        Compute multiple drift metrics on latent representations.
        
        Parameters:
        -----------
        latent_q1 : ndarray, shape (n_samples_q1, latent_dim)
            Q1 latent representations from Phase 3
        latent_q3 : ndarray, shape (n_samples_q3, latent_dim)
            Q3 latent representations from Phase 3
        
        Returns:
        --------
        dict
            Multiple drift metrics for confirmation
        """
        results = {}
        
        # Pathway 1: Mean shift (first two moments)
        results['mean_shift'] = self._compute_mean_shift(latent_q1, latent_q3)
        
        # Pathway 2: Covariance shift (Frobenius norm)
        results['covariance_shift'] = self._compute_covariance_shift(latent_q1, latent_q3)
        
        # Pathway 3: Univariate KS tests per dimension
        results['ks_test_results'] = self._compute_ks_tests(latent_q1, latent_q3)
        results['ks_rejection_rate'] = np.mean([r['pvalue'] < 0.05 for r in results['ks_test_results']])
        
        # Pathway 4: Mahalanobis distance between centroids
        results['mahalanobis_distance'] = self._compute_mahalanobis_distance(latent_q1, latent_q3)
        
        return results
    
    def _compute_mean_shift(self, latent_q1, latent_q3):
        """Euclidean distance between mean vectors"""
        mean_q1 = np.mean(latent_q1, axis=0)
        mean_q3 = np.mean(latent_q3, axis=0)
        return float(np.linalg.norm(mean_q1 - mean_q3))
    
    def _compute_covariance_shift(self, latent_q1, latent_q3):
        """Frobenius norm of covariance matrix difference"""
        cov_q1 = np.cov(latent_q1.T)
        cov_q3 = np.cov(latent_q3.T)
        return float(np.linalg.norm(cov_q1 - cov_q3, ord='fro'))
    
    def _compute_ks_tests(self, latent_q1, latent_q3):
        """Kolmogorov-Smirnov test per latent dimension"""
        results = []
        for dim in range(latent_q1.shape[1]):
            statistic, pvalue = stats.ks_2samp(latent_q1[:, dim], latent_q3[:, dim])
            results.append({
                'dimension': dim,
                'statistic': float(statistic),
                'pvalue': float(pvalue)
            })
        return results
    
    def _compute_mahalanobis_distance(self, latent_q1, latent_q3):
        """Mahalanobis distance between centroid and Q3 points"""
        mean_q1 = np.mean(latent_q1, axis=0)
        cov_q1 = np.cov(latent_q1.T)
        
        # Regularize covariance
        cov_q1 += np.eye(cov_q1.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov_q1)
        
        mean_q3 = np.mean(latent_q3, axis=0)
        diff = mean_q3 - mean_q1
        
        distance = float(np.sqrt(diff @ cov_inv @ diff.T))
        return distance


class ReconstructionErrorAnalyzer:
    """
    Analyzes VAE reconstruction errors to identify anomalous samples.
    Works with autoencoder from Phase 3.
    
    Input: Original features and VAE model from Phase 3
    Output: Per-sample reconstruction error profiles
    """
    
    def __init__(self, model, device='cpu'):
        """
        Parameters:
        -----------
        model : autoencoder model from Phase 3
        device : str, 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_reconstruction_errors(self, features_q1, features_q3):
        """
        Compute per-sample reconstruction errors.
        
        Parameters:
        -----------
        features_q1, features_q3 : ndarray, shape (n_samples, n_features)
        
        Returns:
        --------
        dict
            Error statistics and thresholds
        """
        import torch
        
        # Convert to tensors
        q1_tensor = torch.from_numpy(features_q1).float().to(self.device)
        q3_tensor = torch.from_numpy(features_q3).float().to(self.device)
        
        # Compute reconstruction errors
        with torch.no_grad():
            # Q1 errors (baseline)
            q1_recon, _, _ = self.model(q1_tensor)
            q1_errors = torch.mean((q1_tensor - q1_recon) ** 2, dim=1).cpu().numpy()
            
            # Q3 errors (test)
            q3_recon, _, _ = self.model(q3_tensor)
            q3_errors = torch.mean((q3_tensor - q3_recon) ** 2, dim=1).cpu().numpy()
        
        return {
            'q1_errors': q1_errors,
            'q3_errors': q3_errors,
            'q1_mean_error': float(np.mean(q1_errors)),
            'q1_std_error': float(np.std(q1_errors)),
            'q3_mean_error': float(np.mean(q3_errors)),
            'q3_std_error': float(np.std(q3_errors)),
            'error_increase_ratio': float(np.mean(q3_errors) / np.mean(q1_errors)),
            'anomalous_sample_ratio': float(np.mean(q3_errors > np.mean(q1_errors) + 2*np.std(q1_errors)))
        }


class DriftConfidenceEstimator:
    """
    Estimates confidence in Phase 4 drift detection results.
    Combines multiple pathways into unified confidence score.
    
    Input: Phase 4 results + enhanced metrics from this module
    Output: Confidence scores and drift type indicators
    """
    
    def __init__(self):
        """Initialize estimator"""
        pass
    
    def estimate_confidence(self, phase4_results, enhanced_metrics, validation_results=None):
        """
        Estimate confidence in drift detection.
        
        Parameters:
        -----------
        phase4_results : dict
            Results from Phase 4 (MI-LHD, STKA, etc.)
        enhanced_metrics : dict
            Results from DistributionShiftAnalyzer
        validation_results : dict, optional
            Results from Phase 5 validator
        
        Returns:
        --------
        dict
            Confidence scores and summary
        """
        scores = {}
        
        # Confidence from different pathways
        scores['pathway_1_mean_shift'] = self._score_pathway(
            enhanced_metrics.get('mean_shift', 0), 
            baseline=0.1, 
            high=0.5
        )
        
        scores['pathway_2_covariance_shift'] = self._score_pathway(
            enhanced_metrics.get('covariance_shift', 0),
            baseline=1.0,
            high=5.0
        )
        
        scores['pathway_3_ks_rejection'] = float(enhanced_metrics.get('ks_rejection_rate', 0))
        
        scores['pathway_4_mahalanobis'] = self._score_pathway(
            enhanced_metrics.get('mahalanobis_distance', 0),
            baseline=0.5,
            high=2.0
        )
        
        # Overall confidence: average of normalized pathways
        pathway_scores = [scores['pathway_1_mean_shift'],
                         scores['pathway_2_covariance_shift'],
                         scores['pathway_3_ks_rejection'],
                         scores['pathway_4_mahalanobis']]
        
        overall_confidence = float(np.mean(pathway_scores))
        
        # Consistency: how well do multiple pathways agree?
        consistency = float(1.0 - np.std(pathway_scores))
        
        return {
            'pathway_scores': scores,
            'overall_confidence': np.clip(overall_confidence, 0, 1),
            'consistency_score': np.clip(consistency, 0, 1),
            'drift_confirmed': overall_confidence > 0.5 and consistency > 0.3,
            'mi_lhd_baseline': phase4_results.get('mi_lhd', 0)
        }
    
    def _score_pathway(self, value, baseline=0, high=1.0):
        """Convert pathway metric to [0,1] confidence score"""
        # Normalize: baseline=0.2, high=0.8
        if value <= baseline:
            return 0.2
        elif value >= high:
            return 0.8
        else:
            return 0.2 + (value - baseline) / (high - baseline) * 0.6
