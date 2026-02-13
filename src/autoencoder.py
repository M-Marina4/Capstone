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
    """Variational Autoencoder for learning latent feature representations."""
    
    def __init__(self, input_dim=768, latent_dim=32):
        super(ImageAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: image features -> latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Latent bottleneck
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder: latent space -> reconstructed features
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
    return BCE + beta * KLD

def train_autoencoder(features_q1, features_q3, latent_dim=32, epochs=50, batch_size=32, device='cpu'):
    """
    Train autoencoder on combined Q1/Q3 data for unsupervised latent learning.
    
    Args:
        features_q1: Q1 image features (n_samples × feature_dim)
        features_q3: Q3 image features (n_samples × feature_dim)
        latent_dim: Dimensionality of latent space
        epochs: Training epochs
        batch_size: Batch size
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Trained autoencoder
        history: Training loss history
    """
    
    # Combine and normalize data
    X_combined = np.vstack([features_q1, features_q3])
    X_combined = (X_combined - X_combined.min(axis=0)) / (X_combined.max(axis=0) - X_combined.min(axis=0) + 1e-8)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_combined).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_combined.shape[1]
    model = ImageAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'total_loss': [], 'reconstruction_loss': [], 'kl_loss': []}
    
    print(f"\n Training Autoencoder (latent_dim={latent_dim}, epochs={epochs})...")
    
    for epoch in range(epochs):
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon, mu, logvar, z = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += nn.functional.binary_cross_entropy(recon, x, reduction='mean').item()
            kl_loss += (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())).item()
        
        avg_loss = total_loss / len(dataloader)
        history['total_loss'].append(avg_loss)
        history['reconstruction_loss'].append(recon_loss / len(dataloader))
        history['kl_loss'].append(kl_loss / len(dataloader))
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    print(f"Autoencoder training complete!")
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