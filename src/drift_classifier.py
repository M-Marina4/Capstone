"""
Drift Type Classification - Phase 4 Extension.
Classifies detected drift into VIRTUAL (seasonal/expected) vs REAL (sensor degradation).
Leverages metadata from Phase 5 validation.

Papers: [4] One or two things we know about concept drift - locating and explaining
        [10] One or two things we know about concept drift - detecting drift
"""

import numpy as np
from scipy import stats


class DriftTypeClassifier:
    """
    Classifies Q1→Q3 drift as VIRTUAL, REAL, or MIXED.
    
    VIRTUAL: Expected seasonal variation (normal Q1→Q3 lighting/temperature changes)
    REAL: Actual sensor degradation or infrastructure changes requiring attention
    MIXED: Both virtual and real drift occurring simultaneously
    
    Uses metadata shift patterns from Phase 5 to distinguish types.
    """
    
    def __init__(self):
        """Initialize classifier"""
        self.drift_type = None
        self.confidence = 0.0
    
    def classify(self, phase4_results, validation_results, decomposition_results=None):
        """
        Classify drift type based on Phase 4, Phase 5, and Phase 6 results.
        
        Parameters:
        -----------
        phase4_results : dict
            Results from Phase 4 (mi_lhd, stka, drift_magnitude, etc.)
        validation_results : dict
            Results from Phase 5 (metadata consistency, cross_domain_adaptation_score, etc.)
        decomposition_results : dict, optional
            Results from Phase 6 (trend, seasonal, residual components)
        
        Returns:
        --------
        dict
            Classification with drift type and confidence
        """
        classification = {}
        
        # Extract key indicators
        drift_magnitude = phase4_results.get('drift_magnitude', 0)
        mi_lhd = phase4_results.get('mi_lhd', 0)
        metadata_consistency = validation_results.get('metadata_consistency', 0.5)
        rgb_distance = validation_results.get('rgb_distance', 0)
        fault_delta = abs(validation_results.get('fault_delta', 0))
        gps_distance = validation_results.get('gps_distance', 0)
        
        # Virtual Drift Indicators
        virtual_score = self._compute_virtual_score(
            rgb_distance=rgb_distance,
            metadata_consistency=metadata_consistency,
            gps_distance=gps_distance,
            decomposition_results=decomposition_results
        )
        
        # Real Drift Indicators
        real_score = self._compute_real_score(
            drift_magnitude=drift_magnitude,
            mi_lhd=mi_lhd,
            fault_delta=fault_delta,
            metadata_consistency=metadata_consistency
        )
        
        # Classify based on score comparison
        if virtual_score > 0.55 and virtual_score > real_score + 0.15:
            classification['drift_type'] = 'VIRTUAL'
            classification['confidence'] = virtual_score
        elif real_score > 0.55 and real_score > virtual_score + 0.15:
            classification['drift_type'] = 'REAL'
            classification['confidence'] = real_score
        else:
            classification['drift_type'] = 'MIXED'
            classification['confidence'] = max(virtual_score, real_score)
        
        classification['virtual_score'] = virtual_score
        classification['real_score'] = real_score
        classification['explanation'] = self._get_explanation(
            classification['drift_type'], virtual_score, real_score
        )
        
        return classification
    
    def _compute_virtual_score(self, rgb_distance, metadata_consistency, gps_distance, 
                               decomposition_results=None):
        """
        Score likelihood of VIRTUAL drift (seasonal/expected changes).
        
        High RGB shifts + consistent metadata = likely seasonal lighting/weather change.
        """
        # RGB shift is expected for seasonal changes (higher in Q1, lower in Q3)
        rgb_indicator = min(1.0, rgb_distance / 0.2)
        
        # Metadata consistency: cameras haven't moved or changed
        consistency_indicator = metadata_consistency
        
        # GPS distance should be minimal (cameras didn't move)
        gps_indicator = max(0, 1.0 - (gps_distance / 100.0))
        
        # If decomposition available, check seasonal component dominance
        if decomposition_results and 'seasonal_component' in decomposition_results:
            seasonal = decomposition_results['seasonal_component']
            total = (decomposition_results.get('trend_component', 0) +
                    seasonal +
                    decomposition_results.get('residual_component', 0))
            if total > 0:
                seasonal_indicator = seasonal / total
            else:
                seasonal_indicator = 0.5
        else:
            seasonal_indicator = 0.5
        
        # Virtual drift combines these
        virtual_score = np.mean([rgb_indicator, consistency_indicator, gps_indicator, seasonal_indicator])
        return float(np.clip(virtual_score, 0, 1))
    
    def _compute_real_score(self, drift_magnitude, mi_lhd, fault_delta, metadata_consistency):
        """
        Score likelihood of REAL drift (sensor degradation/malfunction).
        
        Large drift + fault changes + low consistency = likely real issue.
        """
        # Large overall drift magnitude
        drift_indicator = min(1.0, drift_magnitude / 50.0)
        
        # MI-LHD: higher values indicate more latent distribution shift
        mi_lhd_indicator = min(1.0, mi_lhd / 0.5)
        
        # Fault detection rate changes: sign of sensor problems
        fault_indicator = min(1.0, fault_delta / 0.2)
        
        # Low metadata consistency: something is wrong
        consistency_bad_indicator = 1.0 - metadata_consistency
        
        # Real drift combines these
        real_score = np.mean([drift_indicator, mi_lhd_indicator, fault_indicator, consistency_bad_indicator])
        return float(np.clip(real_score, 0, 1))
    
    def _get_explanation(self, drift_type, virtual_score, real_score):
        """Generate human-readable explanation of classification"""
        if drift_type == 'VIRTUAL':
            return (f"Drift appears SEASONAL/EXPECTED (score: {virtual_score:.2f}). "
                   f"Likely due to natural Q1→Q3 lighting, temperature, or weather changes. "
                   f"No immediate action required.")
        elif drift_type == 'REAL':
            return (f"Drift appears ANOMALOUS/REAL (score: {real_score:.2f}). "
                   f"Likely due to sensor degradation or infrastructure changes. "
                   f"Recommend investigation and potential maintenance.")
        else:
            return (f"Drift appears MIXED (virtual: {virtual_score:.2f}, real: {real_score:.2f}). "
                   f"Both seasonal and real changes may be occurring. "
                   f"Recommend deeper analysis to isolate components.")


class MetadataAnomalyDetector:
    """
    Detects anomalies in metadata shifts (brightness, fault_detection_rate, confidence).
    Works with Phase 5 validation metrics.
    
    Identifies which metadata indicators are driving drift.
    """
    
    def __init__(self):
        """Initialize detector"""
        pass
    
    def analyze_metadata_changes(self, validation_results):
        """
        Analyze which metadata fields show anomalous changes.
        
        Parameters:
        -----------
        validation_results : dict
            Full results from Phase 5 validator
        
        Returns:
        --------
        dict
            Per-metadata anomaly scores
        """
        anomalies = {}
        
        # Brightness anomaly: large RGB shift
        rgb_distance = validation_results.get('rgb_distance', 0)
        anomalies['brightness_anomaly'] = self._score_anomaly(rgb_distance, threshold=0.15)
        
        # Fault detection anomaly: large change in fault rate
        fault_delta = abs(validation_results.get('fault_delta', 0))
        anomalies['fault_detection_anomaly'] = self._score_anomaly(fault_delta, threshold=0.15)
        
        # Confidence anomaly: large change in model confidence
        confidence_delta = abs(validation_results.get('confidence_delta', 0))
        anomalies['confidence_anomaly'] = self._score_anomaly(confidence_delta, threshold=0.1)
        
        # GPS anomaly: camera location shift
        gps_distance = validation_results.get('gps_distance', 0)
        anomalies['gps_anomaly'] = self._score_anomaly(gps_distance, threshold=50.0)
        
        # Identify most anomalous metadata
        anomalies['most_anomalous'] = max(anomalies, key=lambda k: anomalies[k] if k != 'most_anomalous' else 0)
        anomalies['anomaly_severity'] = float(np.mean([v for k,v in anomalies.items() if k != 'most_anomalous']))
        
        return anomalies
    
    def _score_anomaly(self, value, threshold):
        """Convert value to anomaly score [0,1] where 1=highly anomalous"""
        if value <= 0:
            return 0.0
        elif value >= threshold * 2:
            return 1.0
        else:
            return value / (threshold * 2)


# ============================================================================
# INTEGRATION INTO MAIN.PY
# ============================================================================
# Phase 4 Extension (after analyze_drift):
#
#  classifier = DriftTypeClassifier()
#  drift_classification = classifier.classify(drift_results, validation)
#  drift_results['drift_classification'] = drift_classification
#
# Phase 5 Extension (in validation section):
#
#  metadata_analyzer = MetadataAnomalyDetector()
#  metadata_anomalies = metadata_analyzer.analyze_metadata_changes(validation)
#  validation['metadata_anomalies'] = metadata_anomalies
#
# Then report in Phase 7 results section.
