# Monitoring Feature Drift in IoT Sensor Networks

**Capstone Project** — Marina Melkonyan | American University of Armenia

Detecting and quantifying distributional drift in streetlight camera imagery (Bristol, UK) using Variational Autoencoders. The system compares winter (Q1: Jan–Mar) and summer (Q3: Jul–Sep) images captured across 22 streetlight cameras (2021–2025) to measure seasonal and temporal feature drift without labeled data.

---

## Key Results

The model was run on 40K streetlight images (Q1 winter vs Q3 summer, 22 cameras). Seasonal drift exists but is low and well-understood — not a sign of sensor failure.

| Version | Drift | Level | Confidence | Notes |
|---------|-------|-------|------------|-------|
| v4 (baseline) | 23.77% | MODERATE | 90% | — |
| **v5 all** | **22.07%** | **LOW** | **90%** | 33% better MI-LHD vs v4 |
| v5 daytime | 19.98% | LOW | 75% | Lowest drift overall |
| v5 nighttime | 18.93% | LOW | 90% | Not a lighting artefact |
| v6 (latest) | — | — | — | CNN features, results pending |

The jump from v4 to v5 is the headline win: drift dropped from MODERATE to LOW just by improving the VAE architecture (BatchNorm, larger latent space, better scheduler). Splitting day vs night confirmed the drift is real seasonal change, not simply caused by the camera switching between day and night modes.

---

## Approach

1. **Data Loading** — Load Q1 (winter) and Q3 (summer) streetlight images from 22 cameras (~240K total, 20K per quarter sampled)
2. **Feature Extraction** — Extract features using one of two methods:
   - *v4/v5*: 768-dimensional RGB histogram features (256 bins × 3 channels)
   - *v6*: 512-dimensional CNN features via pretrained ResNet18 (ImageNet weights)
3. **VAE Training** — Train a Variational Autoencoder on combined Q1+Q3 data with progressive improvements:
   - *v4*: Basic VAE (ReLU, Dropout 0.2, latent_dim=64, 40 epochs)
   - *v5*: BatchNorm, LeakyReLU, CosineAnnealingLR, latent_dim=128, 60 epochs
   - *v6*: KL annealing (warmup 15 epochs, β: 0→0.4), early stopping (patience=10), 80 epochs
4. **Latent Encoding** — Encode both Q1 and Q3 into 128-dimensional latent representations
5. **Drift Measurement** — Compute MI-LHD, STKA (1000-point subsample), and Euclidean centroid distance; derive composite drift score
6. **Bootstrap Confidence** *(v6)* — 20-iteration bootstrap with 70% subsampling for 95% confidence intervals on MI-LHD and STKA
7. **Per-Camera Analysis** *(v6)* — Compute drift metrics per camera to identify worst/best performing sensors
8. **Statistical Validation** — Permutation tests, anomaly detection via reconstruction error, metadata corroboration
9. **Daytime/Nighttime Split** *(v5+)* — Separate analysis by lighting condition to isolate true seasonal drift

---

## Project Structure

```
├── main.py                     # CLI pipeline (full run or skip-training mode)
├── notebooks/
│   ├── drift_detection.ipynb   # Full analysis notebook with visualizations
│   ├── visualizations.ipynb    # Additional exploration
│   └── organize_images_by_quarter.ipynb  # Image organization utility
├── src/
│   ├── processing.py           # Data loading, histogram & CNN feature extraction
│   ├── autoencoder.py          # VAE architecture with KL annealing & early stopping
│   ├── metrics.py              # MI-LHD, STKA, Euclidean, bootstrap CI, per-camera drift
│   ├── validator.py            # Metadata-based drift validation
│   ├── decomp.py               # Time-series drift decomposition
│   ├── drift_detectors.py      # Multi-pathway drift confidence estimation
│   ├── drift_classifier.py     # Virtual vs real drift classification
│   └── anomaly_detector.py     # Reconstruction-based anomaly ensemble
├── data/
│   ├── metadata/               # CSV metadata (timestamps, GPS, faults, day/night labels)
│   └── organized_images/       # Q1/ and Q3/ image directories (daytime + nighttime)
├── models/
│   └── autoencoder_v3*.pt      # Trained VAE checkpoints (v3 = latest architecture)
├── results/
│   ├── capstone_results_v4*.csv  # v4 baseline results
│   ├── capstone_results_v5*.csv  # v5 results (all, daytime, nighttime)
│   ├── capstone_results_v6*.csv  # v6 results (pending)
│   ├── q{1,3}_features_all.npy   # Cached histogram features (768-dim)
│   ├── q{1,3}_features_cnn_all.npy  # Cached CNN features (512-dim)
│   └── q{1,3}_metadata.csv      # Cached metadata with day/night labels
└── ARCHITECTURE_EXPLANATION.md # Detailed module architecture docs
```

---

## Model Architecture

**Variational Autoencoder (v3 — latest)**

```
Encoder: input_dim → 512 → 256 → 128 (mu + logvar)
Decoder: 128 → 256 → 512 → input_dim
```

- Input: 512-dim CNN features (v6) or 768-dim RGB histograms (v4/v5)
- Latent space: 128 dimensions
- Activation: LeakyReLU(0.1) + BatchNorm1d + Dropout(0.15)
- KL annealing: β ramps linearly from 0 → 0.4 over 15 warmup epochs
- Early stopping: patience=10 on validation loss (10% holdout)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5) + CosineAnnealingLR
- Training: up to 80 epochs on combined Q1+Q3 data

---

## Drift Metrics

- **MI-LHD** (Metadata-Invariant Latent Histogram Divergence): Jensen-Shannon divergence per latent dimension, averaged across all 128 dimensions
- **STKA** (Spatio-Temporal Kernel Alignment): RBF kernel alignment between Q1 and Q3 latent distributions (adaptive gamma, subsampled to 1000 points)
- **Euclidean**: L2 distance between Q1 and Q3 latent centroids
- **Composite**: `(0.5 × MI-LHD + 0.3 × (1 - STKA) + 0.2 × min(1, Euclidean)) × 100`

| Severity | Range |
|----------|-------|
| MINIMAL | < 10% |
| LOW | 10–20% |
| MODERATE | 20–35% |
| HIGH | > 35% |

---

## Usage

### Command Line

```bash
# Full pipeline — all data (train + analyze)
python main.py

# Daytime only
python main.py --daynight daytime

# Nighttime only
python main.py --daynight nighttime

# Skip training, load saved model
python main.py --phase4
```

### Notebook

Open `notebooks/drift_detection.ipynb` for the full analysis with inline visualizations.

---

## Requirements

- Python 3.10+
- PyTorch + torchvision
- NumPy, Pandas
- scikit-learn
- Pillow
- Matplotlib, Seaborn (for notebooks)

---

## Version History

| Version | Changes |
|---------|---------|
| **v6** (latest) | ResNet18 CNN features (512-dim), KL annealing, early stopping, bootstrap CI, per-camera drift |
| **v5** | BatchNorm + LeakyReLU VAE, CosineAnnealingLR, latent_dim=128, STKA 1000-subsample, daytime/nighttime split |
| **v4** | Scaled to 40K images (20K/quarter), 22 cameras, basic VAE |
| **v3** | Initial notebook-based analysis (~3.8K images, daytime only) |

---

## Conclusions

- Seasonal drift is real and statistically significant (permutation test p ≈ 0), but it's **LOW** — cameras are behaving as expected.
- Better VAE architecture alone pushed drift from MODERATE down to LOW, with no extra data.
- Day and night cameras drift at almost the same rate (~19–20%), ruling out lighting as the cause.
- Scaling from 3.8K to 40K images made results more stable and reliable.
- The whole pipeline runs unsupervised — no labels needed — making it practical for real deployments.