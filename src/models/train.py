import torch
import torch.nn.functional as F

def train_autoencoder(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()

        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
