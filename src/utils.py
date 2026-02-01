import torch
import numpy as np
from tqdm import tqdm

def extract_latents(model, dataset, device, max_samples=2000):
    model.eval()
    latents = []
    
    sample_indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    
    print(f"Extracting {len(sample_indices)} latents...")
    with torch.no_grad():
        for idx in tqdm(sample_indices):  # ← NOW WORKS
            sample = dataset[idx]
            x = sample['features'].flatten().unsqueeze(0).to(device)
            _, latent = model(x)
            latents.append(latent.cpu().numpy())
    
    return np.vstack(latents)
