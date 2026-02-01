import torch
import torch.nn as nn
from src.config import LATENT_DIM

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, LATENT_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 32 * 32 * 32),
            nn.Unflatten(1, (32, 32, 32)),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
