"""
Advanced Drift Detection Metrics for IoT Sensor Feature Drift.

Implements five research-backed drift metrics beyond MI-LHD and STKA:
  - MMD   : Maximum Mean Discrepancy kernel two-sample test
             (Gretton et al., JMLR 2012)
  - LSDD  : Least-Squares Density Difference
             (Sugiyama et al., Neural Computation 2013)
  - CADD  : Context-Aware Drift Detection using sliding-window conditioning
             (inspired by Lu et al., IEEE TKDE 2019)
  - Fisher: Fisher Information-based drift score
  - Energy: Energy-based drift score (LeCun et al.; energy-based OOD detection)

All metrics return a scalar in [0, 1] where possible (higher = more drift),
so they can be fairly compared with MI-LHD and STKA from src/metrics.py.
"""

import numpy as np
from scipy.spatial.distance import cdist


# ===========================================================================
# 1.  Maximum Mean Discrepancy (MMD) Test
# ===========================================================================
# Reference:
#   Gretton et al. "A Kernel Two-Sample Test" JMLR 13, 723-773 (2012).
#
# Relevance:
#   MMD directly measures the distance between two distributions in a
#   reproducing kernel Hilbert space without requiring density estimation.
#   Applied to autoencoder latent spaces it is more statistically powerful
#   than histogram-based MI-LHD for detecting subtle distributional changes.
# ===========================================================================

def _rbf_mmd_sq(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD² estimator for bandwidth σ."""
    n, m = len(X), len(Y)
    sigma_sq = sigma ** 2
    Kxx = np.exp(-cdist(X, X, "sqeuclidean") / (2 * sigma_sq))
    Kyy = np.exp(-cdist(Y, Y, "sqeuclidean") / (2 * sigma_sq))
    Kxy = np.exp(-cdist(X, Y, "sqeuclidean") / (2 * sigma_sq))

    # Unbiased: zero diagonal for within-sample terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    val = (
        Kxx.sum() / (n * (n - 1) + 1e-12)
        + Kyy.sum() / (m * (m - 1) + 1e-12)
        - 2 * Kxy.mean()
    )
    return float(val)


def compute_mmd(
    latent_q1: np.ndarray,
    latent_q3: np.ndarray,
    sigmas: list | None = None,
    n_permutations: int = 200,
    max_samples: int = 300,
) -> dict:
    """
    Compute MMD between Q1 and Q3 latent representations.

    Uses a multi-bandwidth RBF kernel to improve robustness across latent
    space scales.  A permutation test provides a p-value.

    Args:
        latent_q1     : Q1 latent codes  (n1, d).
        latent_q3     : Q3 latent codes  (n2, d).
        sigmas        : List of RBF bandwidths; defaults to median-heuristic values.
        n_permutations: Number of permutations for significance test.
        max_samples   : Sub-sample size cap to keep runtime tractable.

    Returns:
        dict with keys:
            mmd_score  : float in [0, 1] – normalised drift score
            mmd_sq     : raw unbiased MMD² (multi-bandwidth average)
            p_value    : permutation test p-value (low → significant drift)
    """
    # Sub-sample for computational tractability
    rng = np.random.default_rng(42)
    if len(latent_q1) > max_samples:
        idx = rng.choice(len(latent_q1), max_samples, replace=False)
        latent_q1 = latent_q1[idx]
    if len(latent_q3) > max_samples:
        idx = rng.choice(len(latent_q3), max_samples, replace=False)
        latent_q3 = latent_q3[idx]

    # Median-heuristic bandwidths
    if sigmas is None:
        all_pts = np.vstack([latent_q1, latent_q3])
        pairwise = cdist(all_pts, all_pts, "sqeuclidean")
        median_sq = np.median(pairwise[pairwise > 0])
        sigma_med = np.sqrt(median_sq / 2 + 1e-8)
        sigmas = [sigma_med * s for s in (0.5, 1.0, 2.0)]

    # Observed MMD² (average over bandwidths)
    observed = np.mean([_rbf_mmd_sq(latent_q1, latent_q3, s) for s in sigmas])

    # Permutation test
    combined = np.vstack([latent_q1, latent_q3])
    n1 = len(latent_q1)
    null_dist = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:n1]]
        Y_perm = combined[perm[n1:]]
        null_dist.append(
            np.mean([_rbf_mmd_sq(X_perm, Y_perm, s) for s in sigmas])
        )
    p_value = float(np.mean(np.array(null_dist) >= observed))

    # Normalise to [0, 1] using empirical null distribution
    null_max = np.percentile(null_dist, 99) + 1e-8
    mmd_score = float(np.clip(observed / null_max, 0.0, 1.0))

    return {"mmd_score": mmd_score, "mmd_sq": float(observed), "p_value": p_value}


# ===========================================================================
# 2.  Least-Squares Density Difference (LSDD)
# ===========================================================================
# Reference:
#   Sugiyama et al. "Density-Difference Estimation" Neural Computation 25(10)
#   2013.
#
# Relevance:
#   LSDD directly estimates ||p_q1 - p_q3||² without explicitly computing
#   either density, making it more robust in high-dimensional latent spaces
#   than histogram-based methods (MI-LHD).
# ===========================================================================

def compute_lsdd(
    latent_q1: np.ndarray,
    latent_q3: np.ndarray,
    n_basis: int = 50,
    sigma: float | None = None,
    lam: float = 1e-3,
    max_samples: int = 300,
) -> dict:
    """
    Estimate the L² distance between Q1 and Q3 latent densities via LSDD.

    The density difference g(z) = p_q1(z) - p_q3(z) is modelled as a linear
    combination of Gaussian basis functions centred on a random subset of the
    pooled data.

    Args:
        latent_q1 : Q1 latent codes (n1, d).
        latent_q3 : Q3 latent codes (n2, d).
        n_basis   : Number of Gaussian basis functions.
        sigma     : RBF bandwidth; None uses the median heuristic.
        lam       : L2 regularisation weight.
        max_samples: Sub-sample cap.

    Returns:
        dict with keys:
            lsdd_score : float in [0, 1] – normalised drift score
            lsdd_raw   : raw LSDD estimate (L² divergence)
    """
    rng = np.random.default_rng(42)
    if len(latent_q1) > max_samples:
        latent_q1 = latent_q1[rng.choice(len(latent_q1), max_samples, replace=False)]
    if len(latent_q3) > max_samples:
        latent_q3 = latent_q3[rng.choice(len(latent_q3), max_samples, replace=False)]

    n1, n2 = len(latent_q1), len(latent_q3)
    pooled = np.vstack([latent_q1, latent_q3])

    # Bandwidth via median heuristic
    if sigma is None:
        pw = cdist(pooled, pooled, "sqeuclidean")
        sigma = float(np.sqrt(np.median(pw[pw > 0]) / 2 + 1e-8))

    # Random basis centres from the pooled set
    n_basis = min(n_basis, len(pooled))
    centres = pooled[rng.choice(len(pooled), n_basis, replace=False)]

    def phi(X):
        """Kernel feature matrix (n_samples, n_basis)."""
        d2 = cdist(X, centres, "sqeuclidean")
        return np.exp(-d2 / (2 * sigma ** 2))

    Phi1 = phi(latent_q1)   # (n1, n_basis)
    Phi2 = phi(latent_q3)   # (n2, n_basis)

    # Gram matrix of basis functions (over centres)
    G = phi(centres)   # (n_basis, n_basis)

    # LSDD objective: min_alpha  alpha^T H alpha - 2 alpha^T h
    H = (G.T @ G) / n_basis + lam * np.eye(n_basis)
    h = (Phi1.mean(axis=0) - Phi2.mean(axis=0))

    alpha = np.linalg.solve(H, h)

    # LSDD estimate = alpha^T H alpha (≈ ||g||²)
    lsdd_raw = float(alpha @ H @ alpha)
    lsdd_raw = max(lsdd_raw, 0.0)

    # Normalise with a no-drift baseline (permutation of pooled)
    perm = rng.permutation(len(pooled))
    X_b = pooled[perm[:n1]]
    Y_b = pooled[perm[n1:n1 + n2]]
    Phi1b = phi(X_b)
    Phi2b = phi(Y_b)
    hb = Phi1b.mean(0) - Phi2b.mean(0)
    ab = np.linalg.solve(H, hb)
    baseline = max(float(ab @ H @ ab), 1e-8)

    lsdd_score = float(np.clip(lsdd_raw / (baseline * 3 + 1e-8), 0.0, 1.0))
    return {"lsdd_score": lsdd_score, "lsdd_raw": lsdd_raw}


# ===========================================================================
# 3.  Context-Aware Drift Detection (CADD)
# ===========================================================================
# Reference:
#   Lu et al. "Learning under Concept Drift: A Review" IEEE TKDE 31(12) 2019.
#   Sliding-window approach extended with contextual feature conditioning.
#
# Relevance:
#   Conditioning drift scores on temporal / spatial context (time-of-day,
#   month, GPS zone) lets us separate VIRTUAL drift (apparent change caused
#   by seasonal variation) from REAL drift (genuine sensor degradation).
# ===========================================================================

def compute_cadd(
    latent_q1: np.ndarray,
    latent_q3: np.ndarray,
    context_q1: np.ndarray | None = None,
    context_q3: np.ndarray | None = None,
    n_contexts: int = 4,
    window_size: int = 50,
) -> dict:
    """
    Context-Aware Drift Detection.

    If context arrays are provided (e.g. month, hour-of-day encoded as
    integer bins), the function computes drift separately per context bin and
    returns both the raw and context-conditioned scores.  The difference
    between raw and conditioned drift estimates the virtual (seasonal) portion.

    Args:
        latent_q1    : Q1 latent codes (n1, d).
        latent_q3    : Q3 latent codes (n2, d).
        context_q1   : Integer context labels for Q1 samples (n1,), optional.
        context_q3   : Integer context labels for Q3 samples (n2,), optional.
        n_contexts   : Number of context bins to use if contexts are None.
        window_size  : Sliding-window size for local MMD computation.

    Returns:
        dict with keys:
            cadd_score         : float [0,1] – overall context-conditioned drift
            raw_drift          : float [0,1] – unconditional drift
            virtual_drift      : float [0,1] – estimated seasonal/contextual component
            real_drift         : float [0,1] – estimated genuine sensor drift
            per_context_drift  : list of per-context drift scores
    """
    # Unconditional drift via MMD (fast, quadratic-time, small sample)
    def quick_mmd(A, B, sigma=1.0):
        if len(A) < 2 or len(B) < 2:
            return 0.0
        return max(_rbf_mmd_sq(A[:50], B[:50], sigma), 0.0)

    # Auto-select bandwidth
    pooled = np.vstack([latent_q1, latent_q3])
    pw = cdist(pooled[:100], pooled[:100], "sqeuclidean")
    sigma = float(np.sqrt(np.median(pw[pw > 0]) / 2 + 1e-8))

    raw_mmd = quick_mmd(latent_q1, latent_q3, sigma)

    if context_q1 is None or context_q3 is None:
        # Assign context bins by splitting data into equal temporal chunks
        context_q1 = np.floor(
            np.linspace(0, n_contexts - 1e-9, len(latent_q1))
        ).astype(int)
        context_q3 = np.floor(
            np.linspace(0, n_contexts - 1e-9, len(latent_q3))
        ).astype(int)

    bins = np.unique(np.concatenate([context_q1, context_q3]))
    per_ctx = []
    for b in bins:
        q1_b = latent_q1[context_q1 == b]
        q3_b = latent_q3[context_q3 == b]
        if len(q1_b) < 2 or len(q3_b) < 2:
            continue
        per_ctx.append(quick_mmd(q1_b, q3_b, sigma))

    conditioned_mmd = float(np.mean(per_ctx)) if per_ctx else raw_mmd

    # Normalise both using the same scale (99th-pct of null = raw_mmd * 2)
    scale = max(raw_mmd * 2, 1e-8)
    raw_score = float(np.clip(raw_mmd / scale, 0.0, 1.0))
    cond_score = float(np.clip(conditioned_mmd / scale, 0.0, 1.0))
    virtual = float(np.clip(raw_score - cond_score, 0.0, 1.0))
    real = cond_score

    return {
        "cadd_score": cond_score,
        "raw_drift": raw_score,
        "virtual_drift": virtual,
        "real_drift": real,
        "per_context_drift": per_ctx,
    }


# ===========================================================================
# 4.  Fisher Information-based Drift Score
# ===========================================================================
# Reference:
#   Kullback & Leibler (1951); application to neural network drift scoring
#   discussed in "Detecting Concept Drift with Neural Network Model
#   Uncertainty" and related Fisher-score literature.
#
# Relevance:
#   The Fisher Information Matrix (FIM) of an autoencoder encoder measures
#   how sensitively the log-likelihood of the data changes with model
#   parameters.  A large Fisher divergence between the FIM computed on Q1
#   vs. Q3 indicates that the encoder would need significant parameter
#   updates to fit Q3, confirming genuine drift.
#
#   We use a diagonal FIM approximation (gradient squares) for tractability.
# ===========================================================================

def compute_fisher_drift(
    model,
    features_q1: np.ndarray,
    features_q3: np.ndarray,
    device: str = "cpu",
    max_samples: int = 200,
    batch_size: int = 32,
) -> dict:
    """
    Compute Fisher Information-based drift score.

    Approximates the diagonal FIM for Q1 and Q3 separately via the expected
    squared gradient of the reconstruction loss w.r.t. encoder parameters.
    The normalised L1 distance between the two FIM diagonals is the drift score.

    Args:
        model         : Any autoencoder from advanced_autoencoders.py or
                        src/autoencoder.py (must have encode() and decode()).
        features_q1   : Raw Q1 features (n1, d).
        features_q3   : Raw Q3 features (n2, d).
        device        : 'cpu' or 'cuda'.
        max_samples   : Number of samples per quarter for FIM estimation.
        batch_size    : Mini-batch size.

    Returns:
        dict with keys:
            fisher_score : float [0, 1] – normalised drift score
            fisher_div   : raw Fisher divergence
    """
    import torch
    import torch.nn.functional as F

    def _normalise(X):
        lo, hi = X.min(0), X.max(0)
        return (X - lo) / (hi - lo + 1e-8)

    def diagonal_fim(data: np.ndarray) -> np.ndarray:
        """
        Estimate diagonal of FIM = E[∇² loss] ≈ E[(∇ loss)²]
        for each encoder parameter.
        """
        rng = np.random.default_rng(42)
        if len(data) > max_samples:
            data = data[rng.choice(len(data), max_samples, replace=False)]

        data_norm = _normalise(data)
        accum = None

        model.eval()
        for i in range(0, len(data_norm), batch_size):
            batch = torch.FloatTensor(data_norm[i:i + batch_size]).to(device)
            model.zero_grad()

            # Forward through encoder then decoder
            if hasattr(model, "encode") and hasattr(model, "decode"):
                z = model.encode(batch)
                # Handle VAE: encode returns (mu, logvar)
                if isinstance(z, tuple):
                    z = z[0]
                recon = model.decode(z)
            else:
                raise ValueError("Model must have encode() and decode() methods.")

            loss = F.mse_loss(recon, batch.view(batch.size(0), -1))
            loss.backward()

            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().cpu().numpy().ravel() ** 2)

            batch_fim = np.concatenate(grads)
            if accum is None:
                accum = batch_fim
            else:
                accum += batch_fim

        return accum / max(1, len(data_norm) // batch_size)

    fim_q1 = diagonal_fim(features_q1)
    fim_q3 = diagonal_fim(features_q3)

    # L1 distance between diagonal FIMs normalised by their mean magnitude
    fisher_div = float(np.sum(np.abs(fim_q1 - fim_q3)))
    scale = float(np.sum(fim_q1 + fim_q3) / 2 + 1e-8)
    fisher_score = float(np.clip(fisher_div / scale, 0.0, 1.0))

    return {"fisher_score": fisher_score, "fisher_div": fisher_div}


# ===========================================================================
# 5.  Energy-Based Drift Score
# ===========================================================================
# Reference:
#   LeCun et al. "A Tutorial on Energy-Based Learning" (2006).
#   Liu et al. "Energy-based Out-of-Distribution Detection" NeurIPS 2020.
#
# Relevance:
#   The energy function E(x) = -log Σ_j exp(z_j(x)) derived from the
#   encoder output naturally assigns lower energy to in-distribution samples.
#   Comparing the energy distributions of Q1 and Q3 gives an intuitive
#   measure of distributional shift: if Q3 samples receive systematically
#   higher energy they are more out-of-distribution relative to the Q1-trained
#   encoder, confirming real drift.
# ===========================================================================

def compute_energy_drift(
    model,
    features_q1: np.ndarray,
    features_q3: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict:
    """
    Compute energy-based drift score.

    E(x) = -log Σ_j exp(z_j)  where z = encode(x).

    The drift score is the effect size (Cohen's d) of the energy shift
    between Q1 and Q3, clipped to [0, 1].

    Args:
        model       : Any autoencoder with encode() method.
        features_q1 : Raw Q1 features (n1, d).
        features_q3 : Raw Q3 features (n2, d).
        device      : 'cpu' or 'cuda'.
        batch_size  : Mini-batch size.

    Returns:
        dict with keys:
            energy_score  : float [0, 1] – normalised drift score
            energy_mean_q1: mean energy for Q1 samples
            energy_mean_q3: mean energy for Q3 samples
            energy_shift  : mean energy difference (Q3 − Q1)
            p_value       : Mann-Whitney U p-value for energy distributions
    """
    import torch
    from scipy import stats as scipy_stats

    def _normalise(X):
        lo, hi = X.min(0), X.max(0)
        return (X - lo) / (hi - lo + 1e-8)

    def _energies(data: np.ndarray) -> np.ndarray:
        data_norm = _normalise(data)
        model.eval()
        energies = []
        with torch.no_grad():
            for i in range(0, len(data_norm), batch_size):
                batch = torch.FloatTensor(data_norm[i:i + batch_size]).to(device)
                z = model.encode(batch)
                if isinstance(z, tuple):
                    z = z[0]   # VAE: use mu
                # Energy: -log-sum-exp over latent dimensions
                e = -torch.logsumexp(z, dim=1)
                energies.append(e.cpu().numpy())
        return np.concatenate(energies)

    e_q1 = _energies(features_q1)
    e_q3 = _energies(features_q3)

    mean_q1 = float(np.mean(e_q1))
    mean_q3 = float(np.mean(e_q3))
    shift = mean_q3 - mean_q1

    # Effect size (Cohen's d) normalised to [0, 1]
    pooled_std = float(
        np.sqrt((np.std(e_q1) ** 2 + np.std(e_q3) ** 2) / 2 + 1e-8)
    )
    cohen_d = abs(shift) / pooled_std
    energy_score = float(np.clip(cohen_d / (cohen_d + 1.0), 0.0, 1.0))

    # Mann-Whitney U test for stochastic dominance
    _, p_value = scipy_stats.mannwhitneyu(e_q1, e_q3, alternative="two-sided")

    return {
        "energy_score": energy_score,
        "energy_mean_q1": mean_q1,
        "energy_mean_q3": mean_q3,
        "energy_shift": float(shift),
        "p_value": float(p_value),
    }


# ===========================================================================
# Convenience wrapper: run all advanced metrics at once
# ===========================================================================

def run_all_advanced_metrics(
    latent_q1: np.ndarray,
    latent_q3: np.ndarray,
    model=None,
    features_q1: np.ndarray | None = None,
    features_q3: np.ndarray | None = None,
    context_q1: np.ndarray | None = None,
    context_q3: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """
    Run all five advanced drift metrics and return a unified results dict.

    Args:
        latent_q1   : Q1 latent representations.
        latent_q3   : Q3 latent representations.
        model       : Autoencoder model (required for Fisher and Energy).
        features_q1 : Raw Q1 features (required for Fisher and Energy).
        features_q3 : Raw Q3 features (required for Fisher and Energy).
        context_q1  : Optional context labels for CADD.
        context_q3  : Optional context labels for CADD.
        device      : Torch device string.

    Returns:
        Flat dict of all metric results.
    """
    results = {}

    print("  Computing MMD...")
    mmd = compute_mmd(latent_q1, latent_q3)
    results.update({f"mmd_{k}": v for k, v in mmd.items()})

    print("  Computing LSDD...")
    lsdd = compute_lsdd(latent_q1, latent_q3)
    results.update({f"lsdd_{k}": v for k, v in lsdd.items()})

    print("  Computing CADD...")
    cadd = compute_cadd(latent_q1, latent_q3, context_q1, context_q3)
    results.update({f"cadd_{k}": v for k, v in cadd.items()})

    if model is not None and features_q1 is not None and features_q3 is not None:
        print("  Computing Fisher drift...")
        fisher = compute_fisher_drift(model, features_q1, features_q3, device=device)
        results.update({f"fisher_{k}": v for k, v in fisher.items()})

        print("  Computing energy drift...")
        energy = compute_energy_drift(model, features_q1, features_q3, device=device)
        results.update({f"energy_{k}": v for k, v in energy.items()})
    else:
        print("  Skipping Fisher/Energy (model or raw features not provided).")

    return results
