import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel

def detect_drift_windows(latents, metadata_df, window_size=100, stride=50):
    """Drift detection using correct metadata columns"""
    scores = []
    ref_window = latents[:window_size]
    
    for i in range(window_size, len(latents), stride):
        test_window = latents[i:i+window_size]
        
        # Custom MI-LHD
        hist_ref, _ = np.histogram(ref_window.flatten(), bins=50, density=True)
        hist_test, _ = np.histogram(test_window.flatten(), bins=50, density=True)
        milhd_score = np.sqrt(np.sum((hist_ref - hist_test)**2))
        
        # Dataset baselines
        window_meta = metadata_df.iloc[i:i+window_size]
        baseline_centroid = window_meta['relative_centroid_drift'].mean()
        baseline_recon = window_meta['relative_recon_error'].mean()
        
        scores.append({
            'window_start': i,
            'window_end': i+window_size,
            'milhd': milhd_score,
            'baseline_centroid': baseline_centroid,
            'baseline_recon': baseline_recon,
            'combined_drift': milhd_score + 0.4*baseline_centroid + 0.3*baseline_recon
        })
    
    return pd.DataFrame(scores)
