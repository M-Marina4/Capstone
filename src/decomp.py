import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np

def decompose_drift_scores(scores_df, period=90):  # Quarterly-ish
    """Decompose drift scores into trend/seasonal"""
    scores_copy = scores_df.copy()
    
    for col in ['milhd', 'combined_drift']:
        if col in scores_copy.columns:
            try:
                stl = STL(scores_copy[col].fillna(method='ffill'), period=period, robust=True)
                result = stl.fit()
                scores_copy[f'{col}_trend'] = result.trend
                scores_copy[f'{col}_seasonal'] = result.seasonal
            except:
                scores_copy[f'{col}_trend'] = scores_copy[col]
    
    return scores_copy
