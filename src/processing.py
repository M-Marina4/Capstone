import pandas as pd
import os
import zipfile
import glob
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import io
import warnings

class ZIPImageDataset:
    def __init__(self, zip_paths, metadata_df, max_samples=None):
        self.zip_paths = zip_paths
        self.metadata = metadata_df
        if max_samples:
            self.metadata = self.metadata.head(max_samples)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.zip_cache = {}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['image_name'] + '.jpg'
        
        img_tensor = None
        for zip_path in self.zip_paths:
            try:
                if zip_path not in self.zip_cache:
                    self.zip_cache[zip_path] = zipfile.ZipFile(zip_path, 'r')
                
                zf = self.zip_cache[zip_path]
                if img_name in zf.namelist():
                    with zf.open(img_name) as img_file:
                        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
                        img_tensor = self.transform(img)
                        break
            except:
                continue
        
        row_dict = row.to_dict()
        row_dict['features'] = img_tensor if img_tensor is not None else torch.zeros(3,224,224)
        return row_dict

def load_metadata(metadata_path='data/metadata/streetcare-drift-dataset-2021-2025.csv'):
    """FIXED: Correct columns + dtypes for your dataset"""
    # Explicit dtypes to fix mixed type warnings
    dtype_dict = {
        'id': 'str',
        'serial': 'str',
        'date': 'str',  # Convert later
        'hostname': 'str',
        'lat': 'float',
        'lon': 'float',
        'image_name': 'str',
        'time_tag': 'str',
        'fault_detected': 'float',
        'confidence': 'float',
        'daynight': 'str',
        'red': 'float',
        'green': 'float', 
        'blue': 'float',
        'relative_centroid_drift': 'float',
        'relative_recon_error': 'float'
    }
    
    print("Loading metadata with explicit dtypes...")
    df = pd.read_csv(metadata_path, dtype=dtype_dict, low_memory=False)
    
    # Convert date column to datetime
    df['timestamp'] = pd.to_datetime(df['date'])
    
    # Sort chronologically by node and time
    df = df.sort_values(['serial', 'timestamp']).reset_index(drop=True)
    
    print(f"✅ Metadata loaded: {len(df)} images, {df['serial'].nunique()} nodes")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def create_dataset(zip_dir='data/raw', metadata_path='data/metadata/streetcare-drift-dataset-2021-2025.csv', max_samples=5000):
    zip_paths = glob.glob(os.path.join(zip_dir, "*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No ZIPs in {zip_dir}")
    
    print(f"✅ Found {len(zip_paths)} ZIPs")
    metadata = load_metadata(metadata_path)
    
    if max_samples:
        metadata = metadata.head(max_samples)
    
    dataset = ZIPImageDataset(zip_paths, metadata, max_samples)
    print(f"✅ Dataset ready: {len(dataset)} samples")
    return dataset
