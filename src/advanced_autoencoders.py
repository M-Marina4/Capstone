"""
Advanced Autoencoder Architectures for Feature Drift Detection in IoT Sensor Networks.

Implements four novel architectures beyond the baseline VAE in autoencoder.py:
  - TCAE : Temporal Convolutional Autoencoder (dilated causal convolutions)
  - CAE  : Contractive Autoencoder (Rifai et al., ICML 2011)
  - DAE  : Denoising Autoencoder (Vincent et al., 2008) with multiple corruption strategies
  - WAE  : Wasserstein Autoencoder / MMD-WAE (Tolstikhin et al., ICLR 2018)

All models expose the same interface as ImageAutoencoder in src/autoencoder.py:
  encode(x)  -> latent representation(s)
  decode(z)  -> reconstructed input
  forward(x) -> (reconstruction, *extras)
  get_latent(x) -> deterministic latent vector (no sampling)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helper: normalise a raw feature matrix to [0, 1] per column
# ---------------------------------------------------------------------------

def _normalise(X: np.ndarray) -> np.ndarray:
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    return (X - lo) / (hi - lo + 1e-8)


# ===========================================================================
# 1.  Temporal Convolutional Autoencoder (TCAE)
# ===========================================================================
# Reference:
#   Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent
#   Networks for Sequence Modelling" (2018).  Dilated causal convolutions are
#   used here in a 1-D setting to capture multi-scale temporal patterns in
#   sensor feature sequences.
#
# Relevance to capstone:
#   IoT sensor streams have strong local temporal correlations (time-of-day,
#   seasonal patterns).  Dilated convolutions let the model capture patterns
#   at multiple scales without expensive recurrence, producing a latent space
#   that preserves temporal structure and makes genuine drift more visible.
# ===========================================================================

class _CausalDilatedBlock(nn.Module):
    """Residual dilated causal convolution block."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        h = self.conv(x)
        h = h[:, :, :x.size(2)]   # trim to causal length
        h = h.permute(0, 2, 1)    # -> (batch, seq_len, channels)
        h = self.norm(h)
        h = h.permute(0, 2, 1)    # -> (batch, channels, seq_len)
        return self.act(h) + x    # residual


class TemporalConvAutoencoder(nn.Module):
    """
    Temporal Convolutional Autoencoder (TCAE).

    Treats the feature vector as a 1-D sequence of length `seq_len` with
    `channels` channels at each step, then applies dilated causal convolutions
    to encode temporal structure into a compact latent code.

    Args:
        input_dim  : Total feature dimensionality (seq_len × channels).
        latent_dim : Latent code size.
        seq_len    : Sequence length; input_dim must be divisible by seq_len.
        channels   : Number of channels per temporal step.
        dilations  : List of dilation rates for the encoder blocks.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        seq_len: int = 12,
        channels: int = 64,
        dilations: list | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.channels = channels

        if dilations is None:
            dilations = [1, 2, 4, 8]

        # Project flat features to (channels, seq_len)
        self.input_proj = nn.Linear(input_dim, channels * seq_len)

        # Encoder: stack of dilated causal blocks
        self.enc_blocks = nn.ModuleList([
            _CausalDilatedBlock(channels, kernel_size=3, dilation=d)
            for d in dilations
        ])

        # Bottleneck
        self.enc_fc = nn.Linear(channels * seq_len, latent_dim)

        # Decoder: FC back to (channels, seq_len)
        self.dec_fc = nn.Linear(latent_dim, channels * seq_len)

        # Mirror dilated blocks (non-causal for reconstruction)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.GELU()
            )
            for _ in dilations
        ])

        # Output projection
        self.output_proj = nn.Linear(channels * seq_len, input_dim)

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten -> projected -> (batch, channels, seq_len)."""
        h = self.input_proj(x.view(x.size(0), -1))
        return h.view(-1, self.channels, self.seq_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self._to_seq(x)
        for block in self.enc_blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)
        return self.enc_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, self.channels, self.seq_len)
        for block in self.dec_blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)
        return torch.sigmoid(self.output_proj(h))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def train_tcae(
    features_q1: np.ndarray,
    features_q3: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seq_len: int = 12,
) -> tuple:
    """
    Train a Temporal Convolutional Autoencoder on combined Q1/Q3 features.

    Returns:
        model  : Trained TCAE
        history: Loss history dict
    """
    X = _normalise(np.vstack([features_q1, features_q3]))
    X_t = torch.FloatTensor(X).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    channels = max(32, min(128, input_dim // seq_len))
    # Adjust seq_len so input_dim is divisible
    while input_dim % seq_len != 0 and seq_len > 1:
        seq_len -= 1

    model = TemporalConvAutoencoder(
        input_dim=input_dim, latent_dim=latent_dim,
        seq_len=seq_len, channels=channels
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": []}
    print(f"\nTraining TCAE (latent_dim={latent_dim}, seq_len={seq_len}, epochs={epochs})...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch.view(batch.size(0), -1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        history["loss"].append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg:.6f}")

    print("TCAE training complete.")
    return model, history


# ===========================================================================
# 2.  Contractive Autoencoder (CAE)
# ===========================================================================
# Reference:
#   Rifai et al. "Contractive Auto-Encoders: Explicit Invariance During
#   Feature Extraction" ICML 2011.
#
# Relevance to capstone:
#   The Jacobian penalty |∂h/∂x|² forces the encoder to be insensitive to
#   small, noisy input perturbations (sensor noise).  As a consequence the
#   latent space will show LARGER divergence only for genuine distribution
#   shifts (real drift), making MI-LHD and MMD scores more discriminative.
# ===========================================================================

class ContractiveAutoencoder(nn.Module):
    """
    Contractive Autoencoder (CAE) with Jacobian-norm regularisation.

    Args:
        input_dim  : Feature dimensionality.
        latent_dim : Latent code size.
        lambda_c   : Weight on the contractive (Jacobian) penalty.
    """

    def __init__(self, input_dim: int = 768, latent_dim: int = 32, lambda_c: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_c = lambda_c

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        z = self.encode(x_flat)
        recon = self.decode(z)
        return recon, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def contractive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Frobenius norm of the encoder Jacobian using autograd.
        J_F² = Σ_ij (∂h_j/∂x_i)²
        """
        x_flat = x.view(x.size(0), -1).requires_grad_(True)
        z = self.encoder(x_flat)

        jac_norm_sq = torch.zeros(x_flat.size(0), device=x.device)
        for j in range(z.size(1)):
            grad = torch.autograd.grad(
                z[:, j].sum(), x_flat,
                create_graph=True, retain_graph=True
            )[0]
            jac_norm_sq += (grad ** 2).sum(dim=1)

        return jac_norm_sq.mean()


def cae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    model: ContractiveAutoencoder,
) -> tuple:
    """Reconstruction + contractive penalty loss for CAE."""
    recon_loss = F.mse_loss(recon, x.view(x.size(0), -1))
    contr_loss = model.contractive_loss(x)
    total = recon_loss + model.lambda_c * contr_loss
    return total, recon_loss, contr_loss


def train_cae(
    features_q1: np.ndarray,
    features_q3: np.ndarray,
    latent_dim: int = 32,
    lambda_c: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple:
    """
    Train a Contractive Autoencoder on combined Q1/Q3 features.

    Returns:
        model  : Trained CAE
        history: Loss history dict
    """
    X = _normalise(np.vstack([features_q1, features_q3]))
    X_t = torch.FloatTensor(X).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    model = ContractiveAutoencoder(
        input_dim=X.shape[1], latent_dim=latent_dim, lambda_c=lambda_c
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "recon_loss": [], "contractive_loss": []}
    print(f"\nTraining CAE (latent_dim={latent_dim}, λ_c={lambda_c}, epochs={epochs})...")

    for epoch in range(epochs):
        tot, rec, con = 0.0, 0.0, 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss, rl, cl = cae_loss(recon, batch, model)
            loss.backward()
            optimizer.step()
            tot += loss.item(); rec += rl.item(); con += cl.item()
        n = len(loader)
        history["loss"].append(tot / n)
        history["recon_loss"].append(rec / n)
        history["contractive_loss"].append(con / n)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {tot/n:.6f}  (recon {rec/n:.6f}, contr {con/n:.6f})")

    print("CAE training complete.")
    return model, history


# ===========================================================================
# 3.  Denoising Autoencoder (DAE) with multiple corruption strategies
# ===========================================================================
# Reference:
#   Vincent et al. "Extracting and Composing Robust Features with Denoising
#   Autoencoders" ICML 2008.
#
# Relevance to capstone:
#   By training to reconstruct clean inputs from corrupted versions, the DAE
#   learns features that ignore sensor noise.  The reconstruction error profile
#   under different corruption types (Gaussian, masking, salt-and-pepper) can
#   distinguish genuine distribution shift (high error under all corruptions)
#   from ordinary sensor noise (high error only under targeted corruption).
# ===========================================================================

def _corrupt_gaussian(x: torch.Tensor, noise_std: float = 0.2) -> torch.Tensor:
    """Additive Gaussian noise corruption."""
    return torch.clamp(x + noise_std * torch.randn_like(x), 0.0, 1.0)


def _corrupt_masking(x: torch.Tensor, mask_fraction: float = 0.3) -> torch.Tensor:
    """Random feature masking (zeroing)."""
    mask = (torch.rand_like(x) > mask_fraction).float()
    return x * mask


def _corrupt_salt_pepper(x: torch.Tensor, sp_fraction: float = 0.1) -> torch.Tensor:
    """Salt-and-pepper noise."""
    noisy = x.clone()
    salt = torch.rand_like(x) < sp_fraction / 2
    pepper = torch.rand_like(x) < sp_fraction / 2
    noisy[salt] = 1.0
    noisy[pepper] = 0.0
    return noisy


CORRUPTION_STRATEGIES = {
    "gaussian": _corrupt_gaussian,
    "masking": _corrupt_masking,
    "salt_pepper": _corrupt_salt_pepper,
}


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder (DAE).

    Identical architecture to the baseline VAE encoder/decoder but trained to
    map corrupted → clean features instead of using the variational objective.

    Args:
        input_dim         : Feature dimensionality.
        latent_dim        : Latent code size.
        corruption        : One of 'gaussian', 'masking', 'salt_pepper'.
        corruption_level  : Noise level (std for Gaussian, fraction otherwise).
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        corruption: str = "gaussian",
        corruption_level: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.corruption = corruption
        self.corruption_level = corruption_level

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        fn = CORRUPTION_STRATEGIES[self.corruption]
        return fn(x, self.corruption_level)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        x_corrupted = self._add_noise(x_flat) if self.training else x_flat
        z = self.encode(x_corrupted)
        recon = self.decode(z)
        return recon, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.encode(x)


def train_dae(
    features_q1: np.ndarray,
    features_q3: np.ndarray,
    latent_dim: int = 32,
    corruption: str = "gaussian",
    corruption_level: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple:
    """
    Train a Denoising Autoencoder on combined Q1/Q3 features.

    Args:
        corruption      : 'gaussian', 'masking', or 'salt_pepper'
        corruption_level: Noise level (std / fraction)

    Returns:
        model  : Trained DAE
        history: Loss history dict
    """
    X = _normalise(np.vstack([features_q1, features_q3]))
    X_t = torch.FloatTensor(X).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    model = DenoisingAutoencoder(
        input_dim=X.shape[1], latent_dim=latent_dim,
        corruption=corruption, corruption_level=corruption_level
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": []}
    print(f"\nTraining DAE ({corruption} corruption={corruption_level}, epochs={epochs})...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch.view(batch.size(0), -1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        history["loss"].append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg:.6f}")

    print("DAE training complete.")
    return model, history


# ===========================================================================
# 4.  Wasserstein Autoencoder / MMD-WAE
# ===========================================================================
# Reference:
#   Tolstikhin et al. "Wasserstein Auto-Encoders" ICLR 2018.
#
# Relevance to capstone:
#   WAE replaces the KL divergence of a VAE with a Maximum Mean Discrepancy
#   (MMD) penalty between the aggregate posterior and a standard Gaussian
#   prior.  This yields smoother, more uniformly populated latent manifolds
#   compared to VAE, which is beneficial for drift measurement because the
#   latent space has a more consistent geometry across time.
# ===========================================================================

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian (RBF) kernel: k(x,y) = exp(-||x-y||²/(2σ²))."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
    dist_sq = (diff ** 2).sum(-1)            # (n, m)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def mmd_penalty(z: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Unbiased MMD² between z and samples from N(0, I).

    MMD²(P_z, P_prior) = E[k(z,z')] - 2 E[k(z,ξ)] + E[k(ξ,ξ')]
    where ξ ~ N(0, I).
    """
    z_prior = torch.randn_like(z)

    K_zz = _rbf_kernel(z, z, sigma)
    K_pp = _rbf_kernel(z_prior, z_prior, sigma)
    K_zp = _rbf_kernel(z, z_prior, sigma)

    n = z.size(0)
    # Unbiased: exclude diagonal for zz and pp
    mmd = (
        (K_zz.sum() - K_zz.trace()) / (n * (n - 1) + 1e-8)
        + (K_pp.sum() - K_pp.trace()) / (n * (n - 1) + 1e-8)
        - 2 * K_zp.mean()
    )
    return mmd.clamp(min=0.0)


class WassersteinAutoencoder(nn.Module):
    """
    Wasserstein Autoencoder with MMD regulariser (MMD-WAE).

    Architecture mirrors ImageAutoencoder but uses a deterministic encoder
    (no µ/σ split) and an MMD-based latent regulariser instead of KL.

    Args:
        input_dim  : Feature dimensionality.
        latent_dim : Latent code size.
        lambda_mmd : Weight on the MMD regulariser.
        sigma      : RBF kernel bandwidth for MMD.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        lambda_mmd: float = 10.0,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_mmd = lambda_mmd
        self.sigma = sigma

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encode(x)


def wae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    model: WassersteinAutoencoder,
) -> tuple:
    """Reconstruction + MMD regulariser loss for WAE."""
    recon_loss = F.mse_loss(recon, x.view(x.size(0), -1))
    mmd = mmd_penalty(z, model.sigma)
    total = recon_loss + model.lambda_mmd * mmd
    return total, recon_loss, mmd


def train_wae(
    features_q1: np.ndarray,
    features_q3: np.ndarray,
    latent_dim: int = 32,
    lambda_mmd: float = 10.0,
    sigma: float = 1.0,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple:
    """
    Train a Wasserstein Autoencoder on combined Q1/Q3 features.

    Returns:
        model  : Trained WAE
        history: Loss history dict
    """
    X = _normalise(np.vstack([features_q1, features_q3]))
    X_t = torch.FloatTensor(X).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    model = WassersteinAutoencoder(
        input_dim=X.shape[1], latent_dim=latent_dim,
        lambda_mmd=lambda_mmd, sigma=sigma
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "recon_loss": [], "mmd_loss": []}
    print(f"\nTraining WAE (latent_dim={latent_dim}, λ_mmd={lambda_mmd}, epochs={epochs})...")

    for epoch in range(epochs):
        tot, rec, mmd_ = 0.0, 0.0, 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, z = model(batch)
            loss, rl, ml = wae_loss(recon, batch, z, model)
            loss.backward()
            optimizer.step()
            tot += loss.item()
            rec += rl.item()
            mmd_ += ml.item()
        n = len(loader)
        history["loss"].append(tot / n)
        history["recon_loss"].append(rec / n)
        history["mmd_loss"].append(mmd_ / n)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {tot/n:.6f}  (recon {rec/n:.6f}, MMD {mmd_/n:.6f})")

    print("WAE training complete.")
    return model, history


# ===========================================================================
# Shared utility: extract latent representations from any model
# ===========================================================================

def extract_latent(
    model: nn.Module,
    features: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract deterministic latent vectors from any model in this module.

    All models implement `get_latent(x)` that returns a (batch, latent_dim)
    tensor without any stochastic sampling.

    Args:
        model    : Any TCAE / CAE / DAE / WAE instance.
        features : Raw feature array (n_samples, input_dim).
        device   : 'cpu' or 'cuda'.
        batch_size: Mini-batch size for inference.

    Returns:
        latents: (n_samples, latent_dim) NumPy array.
    """
    features_norm = _normalise(features)
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(features_norm), batch_size):
            batch = torch.FloatTensor(features_norm[i:i + batch_size]).to(device)
            z = model.get_latent(batch)
            latents.append(z.cpu().numpy())
    return np.vstack(latents)
