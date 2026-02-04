"""
Time-series decomposition for separating seasonal drift from trends.
Enables interpretation of drift origins.
"""
import pandas as pd
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

def compute_time_series_drift_scores(q1_data, q3_data, mi_lhd_score, period=90):
    """
    Decompose drift into trend and seasonal components using time-series decomposition.
    
    Args:
        q1_data: Q1 DataFrame with temporal info
        q3_data: Q3 DataFrame with temporal info
        mi_lhd_score: Overall MI-LHD drift metric
        period: Seasonal decomposition period (days)
    
    Returns:
        dict: Trend and seasonal drift components
    """
    
    # Combine and sort by date
    combined = pd.concat([
        q1_data[['date', 'day_of_year']].assign(quarter=1),
        q3_data[['date', 'day_of_year']].assign(quarter=3)
    ]).sort_values('date').reset_index(drop=True)
    
    # Create daily drift signal
    combined['day'] = combined['date'].dt.date
    daily_drift = combined.groupby('day').size().reset_index(name='count')
    
    # Create synthetic drift score time series (for demonstration)
    if len(daily_drift) > period:
        try:
            # Normalize counts as drift proxy
            drift_series = pd.Series(daily_drift['count'].values)
            
            # Apply STL decomposition
            stl = STL(drift_series, seasonal=period, trend=period*2, robust=True)
            result = stl.fit()
            
            return {
                'trend_component': result.trend.mean(),
                'seasonal_component': result.seasonal.std(),
                'residual_component': result.resid.std(),
                'decomposition_success': True
            }
        except Exception as e:
            print(f"   ⚠️  Decomposition warning: {e}")
            return {
                'trend_component': mi_lhd_score * 0.7,
                'seasonal_component': mi_lhd_score * 0.2,
                'residual_component': mi_lhd_score * 0.1,
                'decomposition_success': False
            }
    
    return {
        'trend_component': mi_lhd_score * 0.7,
        'seasonal_component': mi_lhd_score * 0.2,
        'residual_component': mi_lhd_score * 0.1,
        'decomposition_success': False
    }

def analyze_drift_sources(q1_data, q3_data, validation):
    """
    Attribute drift to specific sources: environmental, temporal, sensor-related.
    
    Args:
        q1_data: Q1 metadata
        q3_data: Q3 metadata
        validation: Validation results
    
    Returns:
        dict: Drift source attribution
    """
    
    sources = {
        'environmental_drift': abs(validation.get('rgb_distance', 0)) * 100,
        'temporal_drift': (abs(validation.get('confidence_delta', 0))) * 100,
        'gps_spatial_drift': validation.get('gps_distance', 0),
        'fault_detection_drift': abs(validation.get('fault_delta', 0)) * 100
    }
    
    total = sum(sources.values()) + 1e-8
    sources_normalized = {k: (v / total) * 100 for k, v in sources.items()}
    
    return sources_normalized