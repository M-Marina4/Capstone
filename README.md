# Monitoring Feature Drift in IoT Sensor Networks

**Capstone Project** — Marina Melkonyan | American University of Armenia

Detecting and quantifying distributional drift in streetlight camera imagery (Bristol, UK) using Variational Autoencoders. The system compares winter (Q1) and summer (Q3) daytime images captured between 2021–2025 to measure seasonal and temporal feature drift without labeled data.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Composite Drift Magnitude** | **27.6% (MODERATE)** |
| MI-LHD (histogram divergence) | 0.1813 |
| STKA (kernel alignment) | 0.6737 |
| Euclidean (centroid distance) | 0.4384 |
| Permutation Test p-value | 0.0000 (statistically significant) |
| Q3 Anomaly Rate | 7.62% (144 / 1889 samples) |
| Q1 Mean Reconstruction MSE | 0.0093 |
| Q3 Mean Reconstruction MSE | 0.0097 |
| Reconstruction Error Ratio (Q3/Q1) | 1.05 |

### Temporal Drift Trend (Q3 vs Q1 Baseline by Year)

| Year | Samples | Drift (%) |
|------|---------|-----------|
| 2021 | 554 | 29.5% |
| 2022 | 270 | 32.0% |
| 2023 | 601 | 31.4% |
| 2024 | 416 | 32.2% |
| 2025 | 48 | 38.6% |

Drift is increasing over time, with 2025 approaching the SEVERE threshold.

### Per-Channel Drift (KS Test)

| Channel | KS Statistic | Contribution |
|---------|-------------|-------------|
| Red | 0.043 | 39.7% |
| Green | 0.004 | 32.2% |
| Blue | 0.040 | 28.1% |

### Latent Dimension Sensitivity

| Latent Dim | MI-LHD | STKA | Euclidean | Drift (%) |
|-----------|--------|------|-----------|-----------|
| 8 | 0.290 | 0.596 | 0.000 | 26.6% |
| 16 | 0.230 | 0.575 | 0.001 | 24.3% |
| 32 | 0.134 | 0.548 | 0.046 | 21.2% |
| **64** | **0.181** | **0.674** | **0.438** | **27.6%** |
| 128 | 0.184 | 0.686 | 0.592 | 30.4% |

Drift is consistently detected across all latent dimensionalities (21–30%), confirming the robustness of the finding.

---

## Approach

1. **Data Loading** — Load Q1 (winter) and Q3 (summer) daytime streetlight images (1,919 Q1 + 1,889 Q3)
2. **Feature Extraction** — Extract 768-dimensional RGB histogram features (256 bins × 3 channels)
3. **VAE Training** — Train a beta-VAE (β=0.4) on **Q1 data only**, so Q3 serves as a pure out-of-distribution test set
4. **Latent Encoding** — Encode both Q1 and Q3 into 64-dimensional latent representations
5. **Drift Measurement** — Compute MI-LHD, STKA, and Euclidean centroid distance; derive composite drift score
6. **Statistical Validation** — Permutation test (500 iterations) to confirm drift is not due to chance
7. **Anomaly Detection** — Identify anomalous Q3 samples via reconstruction error (Q1 95th percentile threshold)
8. **Metadata Validation** — Corroborate drift with GPS, brightness, fault rate, and model confidence shifts

---

## Project Structure

```
├── main.py                     # CLI pipeline (full run or skip-training mode)
├── notebooks/
│   ├── drift_detection.ipynb   # Full analysis notebook with visualizations
│   └── visualizations.ipynb    # Additional exploration
├── src/
│   ├── processing.py           # Data loading, feature extraction, temporal metadata
│   ├── autoencoder.py          # VAE architecture (768→512→256→128→64) and training
│   ├── metrics.py              # MI-LHD, STKA, Euclidean drift metrics
│   ├── validator.py            # Metadata-based drift validation
│   ├── decomp.py               # Time-series drift decomposition
│   ├── drift_detectors.py      # Multi-pathway drift confidence estimation
│   ├── drift_classifier.py     # Virtual vs real drift classification
│   └── anomaly_detector.py     # Reconstruction-based anomaly ensemble
├── data/
│   ├── metadata/               # CSV metadata (timestamps, GPS, faults)
│   └── organized_images/       # Q1/ and Q3/ image directories
├── results/
│   └── capstone_results_v3.csv # Final results output
└── ARCHITECTURE_EXPLANATION.md # Detailed module architecture docs
```

---

## Model Architecture

**Variational Autoencoder (beta-VAE, β=0.4)**

```
Encoder: 768 → 512 → 256 → 128 → 64 (mu + logvar)
Decoder: 64 → 128 → 256 → 512 → 768
```

- Input: 768-dim RGB histogram features
- Latent space: 64 dimensions
- Training: Q1 only (100 epochs, Adam optimizer, lr=1e-3)
- Final reconstruction loss: 0.2157
- Final KL loss: 0.0243

---

## Drift Metrics

- **MI-LHD** (Metadata-Invariant Latent Histogram Divergence): Jensen-Shannon divergence per latent dimension, averaged across all 64 dimensions
- **STKA** (Spatio-Temporal Kernel Alignment): RBF kernel alignment between Q1 and Q3 latent distributions (adaptive gamma, subsampled to 200 points)
- **Euclidean**: L2 distance between Q1 and Q3 latent centroids
- **Composite**: `(0.5 × MI-LHD + 0.3 × (1 - STKA) + 0.2 × min(1, Euclidean)) × 100`

| Severity | Range |
|----------|-------|
| LOW | < 15% |
| MODERATE | 15–35% |
| SEVERE | > 35% |

---

## Usage

### Notebook (recommended)

Open and run `notebooks/drift_detection.ipynb` — contains the full pipeline with inline visualizations including PCA, t-SNE, per-channel decomposition, permutation tests, temporal tracking, reconstruction examples, and latent dimension sensitivity analysis.

### Command Line

```bash
# Full pipeline (train + analyze)
python main.py

# Skip training, load saved model
python main.py --phase4
```

---

## Requirements

- Python 3.12+
- PyTorch (CUDA recommended)
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- Pillow

---

## Conclusions

1. **Drift is statistically significant** — Permutation test (p=0.0000) confirms the observed MI-LHD is well above chance level
2. **Q1-only training improves anomaly detection** — With baseline learned purely from Q1, Q3 reconstruction errors become a direct drift signal
3. **Temporal trend** — Year-by-year analysis reveals drift is worsening over time (29.5% in 2021 → 38.6% in 2025)
4. **Channel decomposition** — Red channel contributes most to seasonal shift
5. **Latent dimension robustness** — Drift detection is consistent across bottleneck sizes {8, 16, 32, 64, 128}
6. **Metadata alignment** — Brightness, fault rates, and GPS shifts corroborate the detected drift
7. **Practical utility** — The VAE approach provides an unsupervised, continuous signal for monitoring IoT sensor drift without labeled data