import pandas as pd
import numpy as np

def validate_drift(scores_df, metadata_df):
    """Metadata-driven drift validation"""
    high_drift = scores_df[scores_df['combined_drift_trend'] > 
                          scores_df['combined_drift_trend'].quantile(0.9)]
    
    validated = []
    for _, drift in high_drift.iterrows():
        # Map window to metadata
        start_idx = int(drift['window_start'] / len(metadata_df) * len(metadata_df))
        end_idx = min(int(drift['window_end'] / len(metadata_df) * len(metadata_df)), len(metadata_df))
        window_meta = metadata_df.iloc[start_idx:end_idx]
        
        validated.append({
            'window_start': drift['window_start'],
            'combined_drift': drift['combined_drift'],
            'node_shifts': window_meta['serial'].nunique() > 3,
            'gps_variance': window_meta[['lat', 'lon']].var().mean() > 0.001,
            'fault_correlation': window_meta['fault_detected'].mean() > 0.1,
            'daynight_mix': window_meta['daynight'].nunique() > 1,
            'type_variation': window_meta['image_type'].nunique() if 'image_type' in window_meta else False
        })
    
    return pd.DataFrame(validated)
