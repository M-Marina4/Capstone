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

def spatio_temporal_kernel_alignment(latent_q1, latent_q3, kernel='rbf', gamma=1.0):
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
    
    # Handle mismatched sample sizes by using minimum count
    min_samples = min(len(latent_q1), len(latent_q3))
    latent_q1_aligned = latent_q1[:min_samples]
    latent_q3_aligned = latent_q3[:min_samples]
    
    def gram_matrix(X, kernel='rbf', gamma=1.0):
        """Compute normalized gram matrix."""
        n = X.shape[0]
        G = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if kernel == 'rbf':
                    dist_sq = np.sum((X[i] - X[j])**2)
                    G[i, j] = np.exp(-gamma * dist_sq)
                else:  # linear
                    G[i, j] = np.dot(X[i], X[j])
                
                if i != j:
                    G[j, i] = G[i, j]
        
        # Normalize
        G_norm = np.linalg.norm(G, 'fro')
        return G / (G_norm + 1e-8) if G_norm > 0 else G
    
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

def analyze_drift(q1_data, q3_data, model=None, device='cpu'):
    """
    Analyze feature drift using MI-LHD and STKA metrics.
    
    Args:
        q1_data: Q1 DataFrame
        q3_data: Q3 DataFrame
        model: Trained autoencoder (optional)
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Drift metrics and latent representations
    """
    from processing import batch_extract_features
    
    # Extract histogram features
    print("Extracting image features (Q1)...")
    q1_features, q1_paths = batch_extract_features(q1_data, feature_type='histogram')
    
    print("Extracting image features (Q3)...")
    q3_features, q3_paths = batch_extract_features(q3_data, feature_type='histogram')
    
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
        'drift_magnitude': (1 - stka) * 100  # Percentage drift
    }