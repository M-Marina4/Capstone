"""
Data preprocessing module for IoT streetlight imagery dataset.
Handles loading, filtering, and feature extraction for Q1/Q3 analysis.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_metadata(csv_path="data/metadata/q1q3_daytime_extracted.csv"):
    """Load pre-filtered Q1/Q3 daytime metadata from extracted CSV.
    
    This metadata is already filtered for:
    - Q1 and Q3 quarters only
    - Daytime images (daynight == 0.0)
    - Successfully extracted images
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[df['date'].notna()].copy()

def create_q1q3_datasets(csv_path="data/metadata/q1q3_daytime_extracted.csv", 
                         img_base="data/organized_images"):
    """Create Q1 and Q3 datasets from pre-filtered metadata and organized images.
    
    Args:
        csv_path: Path to pre-filtered metadata (q1q3_daytime_extracted.csv)
        img_base: Base path for organized images directory
    
    Returns:
        q1_df: Q1 daytime images metadata
        q3_df: Q3 daytime images metadata
    """
    # Load pre-filtered metadata
    df = load_metadata(csv_path)
    
    # Parse quarter from time_tag (already filtered to Q1/Q3)
    def extract_quarter(time_tag):
        if pd.isna(time_tag):
            return None
        time_tag = str(time_tag).upper()
        if 'Q1' in time_tag:
            return 1
        elif 'Q3' in time_tag:
            return 3
        return None
    
    df['quarter'] = df['time_tag'].apply(extract_quarter)
    df['year'] = df['date'].dt.year
    
    # Split by quarter
    q1_df = df[df['quarter'] == 1].copy()
    q3_df = df[df['quarter'] == 3].copy()
    
    print(f"✅ Loaded Q1 (daytime): {len(q1_df)} images, {q1_df['year'].nunique()} years")
    print(f"✅ Loaded Q3 (daytime): {len(q3_df)} images, {q3_df['year'].nunique()} years")
    print(f"✅ Daynight verification: All values are {df['daynight'].unique()}")
    
    return q1_df, q3_df

def extract_histogram_features(image_path, bins=256):
    """Extract RGB histogram features from image."""
    try:
        if not os.path.exists(image_path):
            return np.zeros(bins * 3)
        
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Extract histograms for each channel
        hist = np.concatenate([
            np.histogram(img_array[:,:,i], bins=bins, range=(0, 1))[0] 
            for i in range(3)
        ])
        
        # Normalize
        hist = hist / (hist.sum() + 1e-8)
        return hist
    except Exception as e:
        print(f"⚠️  Error processing {image_path}: {e}")
        return np.zeros(bins * 3)

def extract_statistical_features(image_path):
    """Extract color statistics (mean, std, min, max per channel)."""
    try:
        if not os.path.exists(image_path):
            return np.zeros(12)
        
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        stats = []
        for i in range(3):
            channel = img_array[:,:,i]
            stats.extend([
                channel.mean(),
                channel.std(),
                channel.min(),
                channel.max()
            ])
        
        return np.array(stats)
    except Exception as e:
        print(f"⚠️  Error processing {image_path}: {e}")
        return np.zeros(12)

def batch_extract_features(df, img_base="data/organized_images", feature_type='histogram'):
    """
    Batch extract features from image paths.
    
    Args:
        df: DataFrame with 'image_name' and 'time_tag' columns
        img_base: Base path for organized images
        feature_type: 'histogram' or 'statistics'
    
    Returns:
        features: (n_samples × feature_dim) array
        paths: List of image paths
    """
    features = []
    paths = []
    
    for idx, row in df.iterrows():
        image_name = row['image_name']
        time_tag = str(row['time_tag']).upper()
        
        # Determine quarter from time_tag
        if 'Q1' in time_tag:
            quarter = 1
        elif 'Q3' in time_tag:
            quarter = 3
        else:
            print(f"⚠️  Skipping {image_name}: unable to determine quarter from {row['time_tag']}")
            continue
        
        # Construct path
        img_path = os.path.join(img_base, f"Q{quarter}", os.path.basename(image_name))
        
        if feature_type == 'histogram':
            feat = extract_histogram_features(img_path)
        else:
            feat = extract_statistical_features(img_path)
        
        features.append(feat)
        paths.append(img_path)
    
    return np.array(features), paths

def add_temporal_metadata(df):
    """Add temporal features for drift analysis."""
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    return df