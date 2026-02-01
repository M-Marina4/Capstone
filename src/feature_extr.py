import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):  # Reduced for speed
        super().__init__()
        self.latent_dim = latent_dim
        input_dim = 3 * 224 * 224
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def train_autoencoder(dataset, epochs=3, batch_size=16, device='cpu'):
    """Train on ZIP dataset directly"""
    print(f"Training autoencoder (batch_size={batch_size})...")
    
    # Sample for training
    train_size = min(1000, len(dataset))
    train_data = [dataset[i] for i in range(train_size)]
    
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    features = torch.stack([d['features'].flatten() for d in train_data])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features), 
        batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            recon, _ = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}: Loss {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), 'models/autoencoder.pth')
    return model
