"""
Data preprocessing module for IoT streetlight imagery dataset.
Handles loading, filtering, and feature extraction for Q1/Q3 analysis.
"""

import pandas as pd
import numpy as np
import os
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_metadata(csv_path="data/metadata/q1q3_all_extracted.csv"):
    """Load pre-filtered Q1/Q3 metadata from extracted CSV.
    
    This metadata is already filtered for:
    - Q1 and Q3 quarters only
    - Both daytime and nighttime images
    - Successfully extracted images
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[df['date'].notna()].copy()

def create_q1q3_datasets(csv_path="data/metadata/q1q3_all_extracted.csv", 
                         img_base="data/organized_images",
                         daynight_filter=None):
    """Create Q1 and Q3 datasets from pre-filtered metadata and organized images.
    
    Args:
        csv_path: Path to pre-filtered metadata (q1q3_all_extracted.csv)
        img_base: Base path for organized images directory
        daynight_filter: None (all), 'daytime', or 'nighttime'
    
    Returns:
        q1_df: Q1 images metadata
        q3_df: Q3 images metadata
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
    
    # Map daynight numeric to label
    df['dn_label'] = df['daynight'].map({0.0: 'daytime', 1.0: 'nighttime'})
    
    # Apply daynight filter if specified
    if daynight_filter is not None:
        df = df[df['dn_label'] == daynight_filter].copy()
    
    # Build file index for fast existence checking (avoids 250K os.path.exists calls)
    print(f" Building file index for {img_base}...")
    existing_files = set()
    for q in ['Q1', 'Q3']:
        for dn in ['daytime', 'nighttime']:
            dir_path = os.path.join(img_base, q, dn)
            if os.path.isdir(dir_path):
                for f in os.listdir(dir_path):
                    existing_files.add(f"{q}/{dn}/{f}")
    print(f" File index: {len(existing_files)} images on disk")
    
    # Filter to images that exist on disk using the pre-built index
    df = df.dropna(subset=['quarter', 'dn_label'])
    df['_img_key'] = 'Q' + df['quarter'].astype(int).astype(str) + '/' + df['dn_label'] + '/' + df['image_name']
    df = df[df['_img_key'].isin(existing_files)].copy()
    df.drop(columns=['_img_key'], inplace=True)
    
    # Split by quarter
    q1_df = df[df['quarter'] == 1].copy()
    q3_df = df[df['quarter'] == 3].copy()
    
    filter_label = daynight_filter if daynight_filter else 'all (daytime+nighttime)'
    print(f" Loaded Q1 ({filter_label}): {len(q1_df)} images, {q1_df['year'].nunique()} years")
    print(f" Loaded Q3 ({filter_label}): {len(q3_df)} images, {q3_df['year'].nunique()} years")
    print(f" Daynight distribution: {df['dn_label'].value_counts().to_dict()}")
    
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
        print(f"Error processing {image_path}: {e}")
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
        print(f"Error processing {image_path}: {e}")
        return np.zeros(12)

def batch_extract_features(df, img_base="data/organized_images", feature_type='histogram'):
    """
    Batch extract features from image paths.
    
    Args:
        df: DataFrame with 'image_name', 'time_tag', and 'dn_label' columns
        img_base: Base path for organized images
        feature_type: 'histogram' or 'statistics'
    
    Returns:
        features: (n_samples × feature_dim) array
        paths: List of image paths
    """
    features = []
    paths = []
    total = len(df)
    report_interval = max(1, total // 20)  # Report progress every 5%
    
    for i, (idx, row) in enumerate(df.iterrows()):
        image_name = row['image_name']
        time_tag = str(row['time_tag']).upper()
        
        # Determine quarter from time_tag
        if 'Q1' in time_tag:
            quarter = 1
        elif 'Q3' in time_tag:
            quarter = 3
        else:
            continue
        
        # Determine daynight subfolder
        dn_label = row.get('dn_label', 'daytime')
        if pd.isna(dn_label):
            dn_label = 'daytime'
        
        # Construct path: img_base/Q{quarter}/{dn_label}/{image_name}
        img_path = os.path.join(img_base, f"Q{quarter}", str(dn_label), os.path.basename(str(image_name)))
        
        if feature_type == 'histogram':
            feat = extract_histogram_features(img_path)
        else:
            feat = extract_statistical_features(img_path)
        
        features.append(feat)
        paths.append(img_path)
        
        # Progress reporting
        if (i + 1) % report_interval == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"    Progress: {i+1}/{total} ({pct:.0f}%)", end='\r')
    
    print()  # Newline after progress
    return np.array(features), paths

def build_cnn_extractor(device='cpu'):
    """Build a ResNet18 feature extractor (pretrained, no final FC layer).
    
    Returns a model that outputs 512-dim feature vectors per image.
    """
    import torch
    import torchvision.models as models
    
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    # Remove final FC layer — output is 512-dim avg-pooled features
    modules = list(resnet.children())[:-1]
    extractor = torch.nn.Sequential(*modules)
    extractor.eval()
    extractor.to(device)
    return extractor


def batch_extract_cnn_features(df, img_base="data/organized_images",
                                device='cpu', batch_size=32):
    """
    Extract 512-dim ResNet18 features from images in batches.
    
    Args:
        df: DataFrame with 'image_name', 'time_tag', 'dn_label' columns
        img_base: Base path for organized images
        device: 'cpu' or 'cuda'
        batch_size: Batch size for CNN inference
    
    Returns:
        features: (n_samples × 512) numpy array
        paths: List of successfully processed paths
    """
    import torch
    from torchvision import transforms
    
    extractor = build_cnn_extractor(device)
    
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    all_features = []
    all_paths = []
    batch_tensors = []
    batch_paths = []
    total = len(df)
    report_interval = max(1, total // 20)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        image_name = row['image_name']
        time_tag = str(row['time_tag']).upper()
        quarter = 1 if 'Q1' in time_tag else (3 if 'Q3' in time_tag else None)
        if quarter is None:
            continue
        dn_label = row.get('dn_label', 'daytime')
        if pd.isna(dn_label):
            dn_label = 'daytime'
        img_path = os.path.join(img_base, f"Q{quarter}", str(dn_label),
                                os.path.basename(str(image_name)))
        
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            batch_tensors.append(tensor)
            batch_paths.append(img_path)
        except Exception:
            # Zero-fill for missing/corrupt images
            batch_tensors.append(torch.zeros(3, 224, 224))
            batch_paths.append(img_path)
        
        # Process batch
        if len(batch_tensors) >= batch_size or (i + 1) == total:
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    feats = extractor(batch).squeeze(-1).squeeze(-1)
                all_features.append(feats.cpu().numpy())
                all_paths.extend(batch_paths)
                batch_tensors = []
                batch_paths = []
        
        if (i + 1) % report_interval == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"    CNN Progress: {i+1}/{total} ({pct:.0f}%)", end='\r', flush=True)
    
    print(flush=True)
    return np.vstack(all_features), all_paths


def add_temporal_metadata(df):
    """Add temporal features and derived columns for drift analysis."""
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Map column names for validator compatibility
    if 'lat' in df.columns and 'gps_lat' not in df.columns:
        df['gps_lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['gps_lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Compute brightness from RGB if not present
    if 'brightness' not in df.columns and 'red' in df.columns:
        r = pd.to_numeric(df['red'], errors='coerce').fillna(0)
        g = pd.to_numeric(df['green'], errors='coerce').fillna(0)
        b = pd.to_numeric(df['blue'], errors='coerce').fillna(0)
        # Perceived luminance (ITU-R BT.601)
        df['brightness'] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    
    return df