"""
Unsupervised Autoencoder for Learning Latent Representations
Enables drift detection in high-dimensional IoT imagery streams
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle

class ImageAutoencoder(nn.Module):
    """Variational Autoencoder for learning latent feature representations.
    
    Improvements v2:
      - BatchNorm after each linear layer for training stability
      - Wider encoder (768→512→256→128) with BN
      - LeakyReLU for better gradient flow
    """
    
    def __init__(self, input_dim=768, latent_dim=32):
        super(ImageAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: image features -> latent space (with BatchNorm)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Latent bottleneck
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder: latent space -> reconstructed features (with BatchNorm)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def get_latent(self, x):
        """Get latent representation (deterministic)."""
        mu, _ = self.encode(x.view(x.size(0), -1))
        return mu

def vae_loss(recon_x, x, mu, logvar, beta=0.4):
    """VAE loss: reconstruction + KL divergence."""
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view_as(recon_x), reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD

def train_autoencoder(features_q1, features_q3, latent_dim=32, epochs=50, batch_size=32, device='cpu'):
    """
    Train autoencoder on combined Q1/Q3 data for unsupervised latent learning.
    
    Improvements v3:
      - KL annealing: beta ramps from 0 to beta_max over warmup_epochs
      - Early stopping: saves best model by validation loss (10% holdout)
    
    Args:
        features_q1: Q1 image features (n_samples × feature_dim)
        features_q3: Q3 image features (n_samples × feature_dim)
        latent_dim: Dimensionality of latent space
        epochs: Training epochs
        batch_size: Batch size
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Trained autoencoder (best by val loss)
        history: Training loss history
    """
    
    # Combine and normalize data
    X_combined = np.vstack([features_q1, features_q3])
    X_combined = (X_combined - X_combined.min(axis=0)) / (X_combined.max(axis=0) - X_combined.min(axis=0) + 1e-8)
    
    # Train/val split (90/10) for early stopping
    n = len(X_combined)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_size = max(1, int(n * 0.1))
    train_idx, val_idx = perm[val_size:], perm[:val_size]
    
    X_train = torch.FloatTensor(X_combined[train_idx]).to(device)
    X_val = torch.FloatTensor(X_combined[val_idx]).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_combined.shape[1]
    model = ImageAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # KL annealing parameters
    beta_max = 0.4
    warmup_epochs = min(15, epochs // 4)  # ramp KL weight over first 25% of training
    
    # Early stopping state
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    history = {'total_loss': [], 'reconstruction_loss': [], 'kl_loss': [], 'val_loss': []}
    
    print(f"\n Training Autoencoder (latent_dim={latent_dim}, epochs={epochs}, "
          f"warmup={warmup_epochs}, patience={patience})...", flush=True)
    
    for epoch in range(epochs):
        # KL annealing: linearly ramp beta from 0 to beta_max
        if epoch < warmup_epochs:
            beta = beta_max * (epoch + 1) / warmup_epochs
        else:
            beta = beta_max
        
        # --- Training ---
        model.train()
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon, mu, logvar, z = model(x)
            loss, bce, kld = vae_loss(recon, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += bce.item()
            kl_loss += kld.item()
        
        avg_loss = total_loss / len(train_loader)
        history['total_loss'].append(avg_loss)
        history['reconstruction_loss'].append(recon_loss / len(train_loader))
        history['kl_loss'].append(kl_loss / len(train_loader))
        
        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                recon, mu, logvar, z = model(x)
                vloss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
                val_loss_sum += vloss.item()
        avg_val_loss = val_loss_sum / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            es_mark = ' *' if patience_counter == 0 else f' (es:{patience_counter}/{patience})'
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} "
                  f"Val: {avg_val_loss:.6f} Beta: {beta:.3f} LR: {lr_now:.6f}{es_mark}", flush=True)
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1} (best val: {best_val_loss:.6f})", flush=True)
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Restored best model (val_loss={best_val_loss:.6f})", flush=True)
    
    print(f"Autoencoder training complete!", flush=True)
    return model, history

def extract_latent_representations(model, features, device='cpu', batch_size=64):
    """Extract latent representations using trained autoencoder."""
    model.eval()
    features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
    
    latents = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.FloatTensor(features_norm[i:i+batch_size]).to(device)
            latent = model.get_latent(batch)
            latents.append(latent.cpu().numpy())
    
    return np.vstack(latents)

def save_model(model, path='models/autoencoder_model.pkl'):
    """Save trained model to disk in pickle format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path='models/autoencoder_model.pkl'):
    """Load trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model