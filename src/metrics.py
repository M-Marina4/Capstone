"""
Drift detection metrics: MI-LHD and STKA with latent representations.
"""

import numpy as np
from scipy.spatial.distance import euclidean, jensenshannon

def metadata_invariant_latent_histogram_divergence(latent_q1, latent_q3, bins=32):
    """
    MI-LHD: Metadata-Invariant Latent Histogram Divergence [12]
    
    Detects distributional changes in latent features using histogram-based
    divergence measures, enabling robust drift detection without labeled data.
    
    Args:
        latent_q1: Latent representations from Q1 (n_samples × latent_dim)
        latent_q3: Latent representations from Q3 (n_samples × latent_dim)
        bins: Number of histogram bins per dimension
    
    Returns:
        float: MI-LHD score (0-1, higher = more drift)
    """
    if len(latent_q1) == 0 or len(latent_q3) == 0:
        return 0.0
    
    # Normalize latent representations to zero-mean, unit variance
    q1_norm = (latent_q1 - latent_q1.mean(axis=0)) / (latent_q1.std(axis=0) + 1e-8)
    q3_norm = (latent_q3 - latent_q3.mean(axis=0)) / (latent_q3.std(axis=0) + 1e-8)
    
    # Compute histogram divergence per dimension
    divergences = []
    for d in range(latent_q1.shape[1]):
        # Create normalized histograms
        hist_q1, bin_edges = np.histogram(q1_norm[:, d], bins=bins, range=(-4, 4), density=False)
        hist_q3, _ = np.histogram(q3_norm[:, d], bins=bins, range=(-4, 4), density=False)
        
        # Convert to probability distributions
        hist_q1 = hist_q1 / (hist_q1.sum() + 1e-8)
        hist_q3 = hist_q3 / (hist_q3.sum() + 1e-8)
        
        # Jensen-Shannon divergence (symmetric, bounded 0-1)
        js_div = jensenshannon(hist_q1, hist_q3)
        divergences.append(js_div)
    
    # Return mean divergence across dimensions
    return float(np.mean(divergences))

def spatio_temporal_kernel_alignment(latent_q1, latent_q3, kernel='rbf', gamma='auto'):
    """
    STKA: Spatio-Temporal Kernel Alignment [13]
    
    Captures gradual and structured drift by aligning latent representations
    across spatial and temporal dimensions using kernel-based similarity.
    
    Args:
        latent_q1: Latent representations from Q1 (n_samples × latent_dim)
        latent_q3: Latent representations from Q3 (n_samples × latent_dim)
        kernel: 'rbf' or 'linear'
        gamma: RBF kernel bandwidth
    
    Returns:
        float: STKA alignment score (0-1, higher = less drift)
    """
    if len(latent_q1) < 2 or len(latent_q3) < 2:
        return 1.0
    
    # Subsample for computational efficiency (gram matrix is O(n^2))
    max_samples = 200
    min_samples = min(len(latent_q1), len(latent_q3))
    if min_samples > max_samples:
        rng = np.random.RandomState(42)
        idx_q1 = rng.choice(len(latent_q1), max_samples, replace=False)
        idx_q3 = rng.choice(len(latent_q3), max_samples, replace=False)
        latent_q1_aligned = latent_q1[idx_q1]
        latent_q3_aligned = latent_q3[idx_q3]
    else:
        latent_q1_aligned = latent_q1[:min_samples]
        latent_q3_aligned = latent_q3[:min_samples]
    
    def gram_matrix(X, kernel='rbf', gamma=1.0):
        """Compute normalized gram matrix (vectorized)."""
        if kernel == 'rbf':
            sq_norms = np.sum(X**2, axis=1)
            sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
            sq_dists = np.maximum(sq_dists, 0)  # Numerical stability
            G = np.exp(-gamma * sq_dists)
        else:  # linear
            G = X @ X.T
        
        # Normalize
        G_norm = np.linalg.norm(G, 'fro')
        return G / (G_norm + 1e-8) if G_norm > 0 else G
    
    # Adaptive gamma: median heuristic for RBF bandwidth selection
    if gamma == 'auto':
        from scipy.spatial.distance import pdist
        combined = np.vstack([latent_q1_aligned, latent_q3_aligned])
        if len(combined) > 500:
            rng = np.random.RandomState(42)
            subset_idx = rng.choice(len(combined), 500, replace=False)
            dists = pdist(combined[subset_idx], 'sqeuclidean')
        else:
            dists = pdist(combined, 'sqeuclidean')
        gamma = 1.0 / (np.median(dists) + 1e-8)

    # Compute gram matrices on aligned data
    K_q1 = gram_matrix(latent_q1_aligned, kernel, gamma)
    K_q3 = gram_matrix(latent_q3_aligned, kernel, gamma)
    
    # Alignment: measure similarity between kernel matrices
    numerator = np.trace(K_q1 @ K_q3.T)
    denominator = np.linalg.norm(K_q1, 'fro') * np.linalg.norm(K_q3, 'fro')
    
    alignment = numerator / (denominator + 1e-8)
    return float(max(0.0, min(1.0, alignment)))

def compute_euclidean_drift(latent_q1, latent_q3):
    """Compute Euclidean distance between latent distributions."""
    mean_q1 = latent_q1.mean(axis=0)
    mean_q3 = latent_q3.mean(axis=0)
    return float(euclidean(mean_q1, mean_q3))

def analyze_drift(q1_data, q3_data, model=None, device='cpu', q1_features=None, q3_features=None):
    """
    Analyze feature drift using MI-LHD and STKA metrics.
    
    Args:
        q1_data: Q1 DataFrame
        q3_data: Q3 DataFrame
        model: Trained autoencoder (optional)
        device: 'cpu' or 'cuda'
        q1_features: Pre-computed Q1 features (optional, avoids re-extraction)
        q3_features: Pre-computed Q3 features (optional, avoids re-extraction)
    
    Returns:
        dict: Drift metrics and latent representations
    """
    if q1_features is None or q3_features is None:
        from processing import batch_extract_features
        
        # Extract histogram features
        print("Extracting image features (Q1)...")
        q1_features, q1_paths = batch_extract_features(q1_data, feature_type='histogram')
        
        print("Extracting image features (Q3)...")
        q3_features, q3_paths = batch_extract_features(q3_data, feature_type='histogram')
    else:
        print("Using pre-computed features...")
    
    # Get latent representations
    if model is not None:
        print("Computing latent representations via autoencoder...")
        from autoencoder import extract_latent_representations
        latent_q1 = extract_latent_representations(model, q1_features, device)
        latent_q3 = extract_latent_representations(model, q3_features, device)
    else:
        # Fallback: use PCA for latent space
        print("Computing latent representations via PCA (fallback)...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=32)
        latent_q1 = pca.fit_transform(q1_features)
        latent_q3 = pca.transform(q3_features)
    
    # Compute drift metrics
    print("Computing MI-LHD...")
    mi_lhd = metadata_invariant_latent_histogram_divergence(latent_q1, latent_q3)
    
    print("Computing STKA...")
    stka = spatio_temporal_kernel_alignment(latent_q1, latent_q3)
    
    print("Computing Euclidean distance...")
    euclidean_dist = compute_euclidean_drift(latent_q1, latent_q3)
    
    return {
        'mi_lhd': mi_lhd,
        'stka': stka,
        'euclidean': euclidean_dist,
        'latent_q1': latent_q1,
        'latent_q3': latent_q3,
        'drift_magnitude': (0.5 * mi_lhd + 0.3 * (1 - stka) + 0.2 * min(1.0, euclidean_dist)) * 100
    }