# Monitoring Feature Drift in IoT Sensor Networks

**Capstone Project** — Marina Melkonyan | American University of Armenia  
**Supervisor:** Gurgen Hovakimyan  
**Paper:** *"IoT Drift: Catch Me If You Can"*

---

## Plain-Language Summary

> **This section explains the project in simple terms. Skip to [Quick Start](#quick-start) if you already know machine learning.**

Imagine a camera attached to a streetlight that takes a photo every few minutes. Over months and years, the images gradually look different - the lighting changes with seasons, leaves appear and disappear, the angle of sunlight shifts. The camera hardware has not broken, but the *statistical pattern* of the images has changed. In machine learning, this change is called **concept drift** (or **feature drift**).

**Why does drift matter?** Machine learning models are trained on historical data. When the real world drifts away from that historical data, model predictions quietly become less reliable - often without any warning. In IoT systems (smart streetlights, farm sensors, security cameras), this can lead to false alarms or missed events. Detecting drift early lets operators retrain the model or inspect the hardware before things go wrong.

**What is an autoencoder?** An autoencoder is a neural network taught to compress an image (or set of measurements) into a small set of numbers (the *latent space*), and then reconstruct the original image from those numbers. After training on normal data, it reconstructs normal images accurately. When it encounters data from a shifted distribution, it struggles to reconstruct it well - the reconstruction error goes up. We use this elevated error as a *drift signal*.

**What are "features"?** Instead of feeding raw pixel values, we first summarize each image as a compact numerical description. For Dataset 1, we count how many pixels of each brightness level appear in the red, green, and blue channels (a histogram) - producing a 768-number summary per image. For Datasets 2 and 3, raw image tensors go directly into convolutional networks.

**What this project does:**
1. Trains autoencoders on *reference-period* data (e.g., winter, or 2021)
2. Applies the trained model to *target-period* data (e.g., summer, or 2022)
3. Measures how much the reconstruction error increases
4. Uses three complementary mathematical metrics to quantify *how much* drift occurred
5. Validates the finding against GPS data, fault flags, and environmental measurements
6. Answers: *"Does the amount of detected drift depend on how variable the environment is?"*

---

## Quick Start

**If you have all data files and just want to run everything:**

```bash
# 1. Install dependencies
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib seaborn scipy statsmodels river tqdm nbconvert jupyter

# 2. Run the full pipeline (Dataset 1 + all notebooks)
python main.py --all
```

**If you are starting from scratch**, follow the [full step-by-step instructions](#steps-to-reproduce-results) below.

---

## Project Objective

**Research Question:** *What is the sensitivity of unsupervised reconstruction loss as a trigger for drift adaptation in IoT imagery, and how does this sensitivity correlate with the degree of environmental variance across long-term data streams?*

In plain terms: *Does a model that was trained on old data produce noticeably higher errors when it sees new data? And does it produce even higher errors when the environment is visually more variable (e.g., foggy days, changing seasons)?*

IoT sensor networks continuously produce visual data whose statistical properties shift over time due to seasonal change, sensor degradation, and environmental variance. This project investigates whether unsupervised autoencoder reconstruction loss can serve as a reliable, label-free trigger for detecting such distributional drift. The system is evaluated across three heterogeneous real-world IoT datasets covering streetlight cameras, agricultural Raspberry Pi sensors, and sticky-trap insect monitors.

---

## Datasets

> **Three real-world IoT datasets are used.** Each captures a different type of sensor and a different kind of drift. Datasets 2 and 3 are included in this repository. Dataset 1 requires a separate download (see [Data Setup](#data-setup)).

### Dataset 1 — Bristol Streetlight Cameras (StreetCare)

> **What it is:** Cameras mounted on streetlights in Bristol, UK. Each camera sends images at regular intervals. We compare winter images (Q1, January–March) against summer images (Q3, July–September) to study seasonal drift - the difference in appearance caused by season, not by hardware failure.

- **Source:** StreetCare open dataset, Bristol, UK
- **Size:** ~240,000 images from 22 streetlight cameras, 2021–2025
- **Analysis:** Seasonal drift between Q1 (January–March, winter) and Q3 (July–September, summer)
- **Splits:** Daytime and nighttime images treated separately and together (20,000 images per quarter sampled, 40,000 total)
- **Metadata:** Timestamps, GPS coordinates, fault flags, day/night labels
- **Preprocessing:** Images extracted from zip archives, organized into `data/organized_images/Q1/` and `data/organized_images/Q3/`, features cached as NumPy arrays

### Dataset 2 — Pomegranate Tree Time Series

> **What it is:** Nine Raspberry Pi cameras, each attached to a different pomegranate tree on a farm. They record the tree through an entire growing season. We train on 2021 (reference year) and test on 2022 (next season) to see if the sensors drift over time or across seasons.

- **Source:** Public agricultural IoT dataset (*Punica granatum* L. 'Wonderful')
- **Size:** Images from 9 independent Raspberry Pi sensors across two full growing seasons
- **Analysis:** Concept drift between 2021 (reference, training) and 2022 (drift target) growing seasons
- **Resolution:** 640×480 → resized to 128×96 (4:3 aspect ratio preserved)
- **Location:** `data/raw/Dataset_pomegranate_tree_time_series/` — organized as `{year}/{SensorN}/`

### Dataset 3 — BMSB Sticky Trap Images (DatasetV3)

> **What it is:** Camera-equipped sticky traps placed in a field to monitor Brown Marmorated Stink Bugs (BMSB) - an invasive agricultural pest. Each trap is photographed regularly. Over the season, the trap fills up with insects and is replaced — causing visual drift. We train on summer (Q3) and test on autumn (Q4) to detect both seasonal change and trap-replacement drift.

- **Source:** IoT sticky-trap camera monitoring Brown Marmorated Stink Bug (*Halyomorpha halys*)
- **Size:** 476 images (1920×1080), June–November 2024; 9 sequential trap replacements over the season
- **Analysis:** Seasonal drift from Q3 (July–September, summer reference) to Q4 (October–November, autumn target)
- **Annotations:** Polygon annotations and binary masks per insect instance included
- **Location:** `data/raw/DatasetV3/` — `Images/`, `Annotations/HH_Polygons/`, `Annotations/Masks/`, `Num_HHs.csv`

---

## Methodology

> **This section explains what the code actually does.** Each subsection first gives a plain-language explanation, then the technical detail.

### Feature Extraction

> **Plain language:** Before comparing images, we convert each image into a compact numerical summary. This makes comparison faster and more robust to small irrelevant pixel-level variations.

**Dataset 1** uses a two-stage approach:
1. **Preprocessing (run once):** 768-dimensional RGB histograms (256 bins × 3 channels), normalized, extracted via `extract_features_fast.py` and cached. These are used for initial analysis in `drift_detection.ipynb`.
2. **Main pipeline:** 512-dimensional embeddings from a pretrained ResNet18 (ImageNet, final average-pool layer), extracted on-the-fly by `main.py` via `src/processing.py` and cached to `results/q1_features_all.npy` / `results/q3_features_all.npy`.

**Datasets 2 and 3** load raw pixel data: images are resized and passed directly as normalized tensors into the convolutional autoencoders (no separate feature extraction step).

### Model Architectures

> **Plain language:** Four different autoencoder designs are compared. The idea for all four is the same - compress images into a small code, then reconstruct them. Models differ in *how* they do the compression. More complex models (ResAttnAE, MemAE) can pick up on subtler drift patterns.

Four autoencoder architectures are compared - primarily across Datasets 2 and 3; Dataset 1 uses the VAE:

| Model | Description |
|-------|-------------|
| **ConvAE** | Baseline deep convolutional autoencoder |
| **VAE** | Variational Autoencoder — ELBO loss (reconstruction + β·KLD) with β KL annealing |
| **ResAttnAE** | Residual blocks + Squeeze-and-Excitation channel attention |
| **MemAE** | Memory-Augmented Autoencoder (Gong et al., ICCV 2019) |

**VAE architecture (Dataset 1, `src/autoencoder.py`):**

```
Encoder:  input_dim → Linear(512) → BN → LeakyReLU(0.1) → Dropout(0.15)
                    → Linear(256) → BN → LeakyReLU(0.1) → Dropout(0.15)
                    → Linear(128) → BN → LeakyReLU(0.1)
                    → fc_mu(128)  |  fc_logvar(128)

Decoder:  z(128) → Linear(128) → BN → LeakyReLU(0.1) → Dropout(0.15)
                 → Linear(256) → BN → LeakyReLU(0.1) → Dropout(0.15)
                 → Linear(512) → BN → LeakyReLU(0.1)
                 → Linear(input_dim) → Sigmoid
```

- **Input:** 512-dim CNN features (ResNet18)
- **Latent space:** 128 dimensions
- **Loss:** BCE reconstruction + β·KLD; β anneals linearly 0 → 0.4 over 15 warmup epochs
- **Training:** Up to 80 epochs on combined Q1+Q3 data; early stopping (patience=10) on 10% validation holdout
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5) + CosineAnnealingLR

**Convolutional autoencoder hyperparameters (Datasets 2 & 3):**

- Latent dimension: 128 · Batch size: 16 · Learning rate: 1e-3 · Max epochs: 100 · Early stopping patience: 10
- Data split (reference period only): Train 70% / Val 15% / Test 15% · Random seed: 42

### Drift Metrics

> **Plain language:** After encoding images into the latent space, we need a number that says "how far apart are the Q1 and Q3 distributions?" We use three metrics and combine them into a single composite score.
>
> - **MI-LHD** — Think of it as: "How different are the histograms of each latent dimension, on average?" (uses Jensen-Shannon divergence - 0 means identical, 1 means completely different)
> - **STKA** — "How well do the geometric structures of the two point clouds align?" (1 = perfectly aligned, 0 = nothing in common)
> - **Euclidean** — "How far apart are the average positions of Q1 and Q3 points in latent space?"
> - **Composite** — A weighted blend of all three, expressed as a percentage. Below 10% is minimal; 20–35% is moderate; above 35% is high drift.

**Latent-space distributional metrics (Dataset 1, `src/metrics.py`):**

| Metric | Description |
|--------|-------------|
| **MI-LHD** | Metadata-Invariant Latent Histogram Divergence - mean Jensen-Shannon divergence across all 128 latent dimensions |
| **STKA** | Spatio-Temporal Kernel Alignment — RBF kernel alignment between Q1 and Q3 latent clouds (subsampled to 1,000 points) |
| **Euclidean** | L2 distance between Q1 and Q3 latent centroids |
| **Composite** | `(0.5 × MI-LHD + 0.3 × (1 − STKA) + 0.2 × min(1, Euclidean)) × 100` |

| Composite Score | Severity Level |
|-----------------|----------------|
| < 10% | MINIMAL |
| 10–20% | LOW |
| 20–35% | MODERATE |
| > 35% | HIGH |

**Statistical drift detectors on reconstruction error streams (Datasets 2, 3 and sensitivity analysis):**

> **Plain language:** These detectors watch a stream of reconstruction errors and raise an alarm when the error level changes significantly. Different detectors use different statistical tests. We compare all six to see which is most reliable.

| Detector | Type |
|----------|------|
| ADWIN | Adaptive windowing (Bifet & Gavaldà 2007) |
| CUSUM | Cumulative sum sequential test |
| Page-Hinkley | Sequential change-point detection |
| KS-Windowed | Non-parametric sliding-window Kolmogorov-Smirnov test |
| DDM | Error-rate monitoring (Gama et al. 2004) |
| EDDM | Error-distance monitoring (Baena-García et al. 2006) |

### Validation Pipeline (Dataset 1)

> **Plain language:** Just because the autoencoder reports drift doesn't mean it's real. We cross-check the finding with independent evidence from the sensor metadata: Are GPS coordinates consistent? Did fault flags spike? Did brightness change in a way that explains the drift? If all checks agree, confidence in the drift finding is high.

The validation pipeline (`src/`) corroborates drift findings with multiple independent checks:

- **`src/validator.py`** - GPS spatial distance, fault-flag correlation, brightness consistency checks
- **`src/drift_classifier.py`** - Classifies drift as VIRTUAL (expected seasonal variation) vs REAL (sensor degradation) vs MIXED
- **`src/anomaly_detector.py`** - `ReconstructionAnomalyDetector` fits Q1 95th-percentile threshold; `AnomalyDriftEnsemble` aggregates multiple detectors
- **`src/decomp.py`** — STL decomposition separates trend and seasonal components of the drift signal
- **`src/drift_detectors.py`** - `DistributionShiftAnalyzer` (mean shift, covariance shift, per-dimension KS tests); `DriftConfidenceEstimator` aggregates into an overall confidence score
- **Bootstrap CI** - 20-iteration bootstrap with 70% subsampling for 95% confidence intervals on MI-LHD and STKA
- **Per-camera drift** - MI-LHD and STKA computed independently for each of the 22 cameras

**Sensitivity analysis hyperparameters (Dataset 3):**

- Threshold percentiles: 50th–98th (step 2)
- Bootstrap iterations: 1,000 · Bootstrap CI: 95%
- Evaluation metrics: ROC-AUC, Average Precision, F1, Precision, Recall, MTTD, FPR, Cohen's d, Mann-Whitney U, Pearson/Spearman/Kendall correlation, Mutual Information

---

## Project Structure

```
Capstone/
├── main.py                                      # One-command entry point (Dataset 1 full pipeline)
│
├── extract_images.py                            # Preprocessing: extract & organize Q1/Q3 images from zips
├── extract_features_fast.py                     # Preprocessing: multi-process RGB histogram extraction
├── extract_features_v2.py                       # Preprocessing (alt): single-process incremental extraction
├── extract_features_from_zips.py                # Preprocessing (alt): features directly from zip archives
├── extract_q3_local.py                          # Q3-only local extraction utility
├── extract_q3_only.py                           # Q3-only extraction (alternate path configuration)
│
├── notebooks/
│   ├── drift_detection.ipynb                    # Dataset 1: full VAE pipeline + paper figures
│   ├── dataset2_drift_detection_pipeline.ipynb  # Dataset 2: pomegranate sensor drift + figures
│   ├── dataset3_seasonal_drift_detection.ipynb  # Dataset 3: BMSB Q3→Q4 drift + figures
│   └── dataset3_sensitivity_analysis_rq.ipynb   # RQ answer: all models × all detectors, sensitivity
│
├── src/
│   ├── __init__.py
│   ├── processing.py          # Data loading, metadata filtering, CNN & histogram feature extraction
│   ├── autoencoder.py         # VAE (ImageAutoencoder), vae_loss, train_autoencoder, save/load model
│   ├── metrics.py             # MI-LHD, STKA, Euclidean, composite score, bootstrap CI, per-camera drift
│   ├── validator.py           # Metadata-driven drift validation (GPS, faults, brightness)
│   ├── decomp.py              # STL time-series decomposition of drift signal
│   ├── drift_detectors.py     # DistributionShiftAnalyzer, DriftConfidenceEstimator
│   ├── drift_classifier.py    # DriftTypeClassifier (VIRTUAL vs REAL), MetadataAnomalyDetector
│   └── anomaly_detector.py    # ReconstructionAnomalyDetector, AnomalyDriftEnsemble
│
├── data/
│   ├── metadata/
│   │   ├── streetcare-drift-dataset-2021-2025.csv    # Full StreetCare metadata (all cameras, 2021-2025)
│   │   └── q1q3_all_extracted.csv                    # Pre-filtered Q1/Q3 image index with day/night labels
│   └── raw/
│       ├── RSE_#    # Dataset 1 ZIP files with raw images
│       ├── Dataset_pomegranate_tree_time_series/     # Dataset 2
│       │   ├── 2021/  Sensor1/ … Sensor9/
│       │   └── 2022/  Sensor1/ … Sensor9/
│       └── DatasetV3/                                # Dataset 3
│           ├── Images/
│           ├── Annotations/HH_Polygons/
│           ├── Annotations/Masks/
│           ├── Num_HHs.csv
│           └── README.txt
│
├── models/
│   └── autoencoder_v3*.pt     # Saved VAE checkpoints (auto-created by main.py)
│
├── results/
│   ├── capstone_results_v5_all.csv         # Dataset 1 results — all images (histogram features)
│   ├── capstone_results_v5_daytime.csv     # Dataset 1 results — daytime subset
│   ├── capstone_results_v5_nighttime.csv   # Dataset 1 results — nighttime subset
│   ├── capstone_results_v6_all.csv         # Dataset 1 results — CNN features (main.py output)
│   ├── q1_features_all.npy                 # Cached Q1 CNN features (512-dim, ResNet18)
│   ├── q3_features_all.npy                 # Cached Q3 CNN features (512-dim, ResNet18)
│   ├── q1_metadata.csv                     # Cached Q1 metadata rows (aligned with feature rows)
│   └── q3_metadata.csv                     # Cached Q3 metadata rows
│
└── paper/                     # LaTeX source: main.tex, references.bib, figures/
```

---

## Requirements

### Python Version

Python **3.10 or later** is required. You can check your version with `python --version`.

> **New to Python?** Download and install Python from [python.org](https://www.python.org/downloads/). Make sure to check "Add Python to PATH" during installation.

### Installation

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the following manually:

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
Pillow>=10.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
statsmodels>=0.14
river>=0.21
tqdm>=4.65
nbconvert>=7.0
jupyter>=1.0
```

### Hardware

A CUDA-capable GPU is recommended but not required. The pipeline automatically uses CUDA if available and falls back to CPU.

> **Approximate run times:**
> - Dataset 1 full pipeline (`main.py`): ~10–20 minutes on GPU, ~30–60 minutes on CPU
> - Dataset 2 notebook: ~20–40 minutes (trains 2 models × 9 sensors)
> - Dataset 3 seasonal notebook: ~5–10 minutes
> - Dataset 3 sensitivity notebook: ~30–90 minutes (trains 4 models × sensitivity sweep)

---

## Data Setup

### Dataset 1 — Bristol Streetlight Cameras

> **Note:** The raw image zip archives (~several GB) are not included in this repository due to file size. You have two options: (A) use the pre-extracted feature files already in `results/` (recommended for just reproducing results), or (B) start from raw images.

**Option A — Use existing feature cache (recommended):**

If `results/q1_features_all.npy` and `results/q3_features_all.npy` already exist, no additional setup is needed for Dataset 1. Skip directly to [Step 1](#step-1--one-command-reproduction-of-all-results).

**Option B — Start from raw zip archives:**

1. Download the StreetCare dataset zip archives from Zenodo:
   - [https://zenodo.org/records/17781192](https://zenodo.org/records/17781192)
   - [https://zenodo.org/records/17859120](https://zenodo.org/records/17859120)
2. Open `extract_images.py` and `extract_features_fast.py`. Near the top of each file, update the path constants (labeled with comments) to point to where you saved the zip files on your computer.
   - Look for lines like `BASE_DIR = r"C:\Users\..."` and change them to your path.
3. Run the preprocessing steps described in [Step 0](#step-0--preprocessing-dataset-1-only--skip-if-feature-cache-already-exists).

### Dataset 2 — Pomegranate Tree Time Series

**Source:** [https://zenodo.org/records/10829695](https://zenodo.org/records/10829695)

Images are included under `data/raw/Dataset_pomegranate_tree_time_series/`. **No additional setup required.**

If you move the dataset to a different folder, open `notebooks/dataset2_drift_detection_pipeline.ipynb` and update the `dataset_root` variable in the `CONFIG` cell (Cell 2) to match the new path.

### Dataset 3 — BMSB Sticky Traps

**Source:** [https://zenodo.org/records/16088064](https://zenodo.org/records/16088064)

Images, polygon annotations, and binary masks are included under `data/raw/DatasetV3/`. **No additional setup required.**

If you move the dataset, update `dataset_root` in both `notebooks/dataset3_seasonal_drift_detection.ipynb` and `notebooks/dataset3_sensitivity_analysis_rq.ipynb`.

---

## Steps to Reproduce Results

> **If you are new to running Python projects, here is the overall flow:**
> 1. Install Python and the required libraries (see [Requirements](#requirements))
> 2. Set up the data files (see [Data Setup](#data-setup))
> 3. Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to the `Capstone/` folder
> 4. Run the commands below

### Step 0 — Preprocessing (Dataset 1 only — skip if feature cache already exists)

> **When to run:** Only needed if you are starting from the raw StreetCare zip archives. If `results/q1_features_all.npy` already exists, skip this step entirely.

```bash
# Extract and organize raw images into Q1/Q3 directory structure
python extract_images.py

# Extract RGB histogram features (multi-process, fast)
python extract_features_fast.py
```

> CNN features (512-dim, ResNet18) are extracted automatically by `main.py` on first run — you do not need to run a separate step for those.

---

### Step 1 — One-Command Reproduction of All Results

```bash
python main.py --all
```

This single command:
1. Runs the complete Dataset 1 pipeline (see breakdown below)
2. Executes all four Jupyter notebooks via `nbconvert --execute --inplace`
3. Saves all results and figure outputs to the respective files

> **Note:** `--all` requires `jupyter` and `nbconvert` to be installed. The notebook execution can take 1–2 hours total depending on hardware.

---

### Step 2 — Dataset 1 Pipeline Only

```bash
python main.py
```

Phases executed:

| Phase | Description | Output |
|-------|-------------|--------|
| 1 + 2 | Load metadata; extract or load cached CNN features | `results/q1_features_all.npy`, `q3_features_all.npy` |
| 3 | Train VAE on Q1+Q3 (or load checkpoint with `--phase4`) | `models/autoencoder_v3_all.pt` |
| 4 | Compute MI-LHD, STKA, Euclidean, composite drift score | Printed to console |
| 4b | Bootstrap CI (20 iterations, 70% subsample) | Printed to console |
| 4c | Per-camera drift for all 22 cameras | Printed to console |
| 5 | Metadata validation, drift type classification, anomaly detection | Printed to console |
| 6 | STL time-series decomposition | Printed to console |
| Save | Write full report | `results/capstone_results_v6_all.csv` |

**Optional flags:**

```bash
# Daytime images only
python main.py --daynight daytime

# Nighttime images only
python main.py --daynight nighttime

# Skip VAE training — load an existing checkpoint
python main.py --phase4
```

---

### Step 3 — Run Individual Notebooks

> **What are Jupyter notebooks?** Notebooks (`.ipynb` files) are interactive documents that mix code and text. Each notebook runs top-to-bottom. You run all cells at once by selecting **Kernel → Restart & Run All** (or **Run All** in VS Code).

**To open notebooks:**

```bash
jupyter notebook
```

Then click the desired notebook in the browser window that opens. Alternatively, open the `.ipynb` files directly in VS Code.

Select **Kernel → Restart & Run All** to execute the full notebook from start to finish.

| Notebook | Dataset | What it shows |
|----------|---------|---------------|
| `notebooks/drift_detection.ipynb` | Dataset 1 | Latent space PCA/UMAP, reconstruction error distributions, per-camera drift chart, bootstrap CI error bars |
| `notebooks/dataset2_drift_detection_pipeline.ipynb` | Dataset 2 | Per-sensor reconstruction error curves, 2021 vs 2022 comparison for ConvAE and ResAttnAE |
| `notebooks/dataset3_seasonal_drift_detection.ipynb` | Dataset 3 | Q3→Q4 drift curves, per-trap seasonal timeline, reconstruction error heatmap |
| `notebooks/dataset3_sensitivity_analysis_rq.ipynb` | Dataset 3 (RQ) | All 4 models × 6 detectors, sensitivity vs environmental variance correlation, ROC/AUC, MTTD, F1 at thresholds |

---

## How Figures and Tables in the Paper Are Generated

All figures are produced entirely by code — no manual image editing.

| Paper Figure / Table | Source |
|---|---|
| Drift score summary table (all datasets) | `main.py` → `results/capstone_results_v6_all.csv` |
| Latent space PCA / UMAP visualization | `notebooks/drift_detection.ipynb` |
| Reconstruction error distributions (Dataset 1) | `notebooks/drift_detection.ipynb` |
| Per-camera drift bar chart | `notebooks/drift_detection.ipynb` |
| Bootstrap CI error bars (MI-LHD, STKA) | `notebooks/drift_detection.ipynb` |
| Dataset 2 per-sensor drift heatmap | `notebooks/dataset2_drift_detection_pipeline.ipynb` |
| Dataset 2 reconstruction error curves (2021 vs 2022) | `notebooks/dataset2_drift_detection_pipeline.ipynb` |
| Dataset 3 seasonal drift curve (Q3→Q4) | `notebooks/dataset3_seasonal_drift_detection.ipynb` |
| Dataset 3 per-trap replacement timeline | `notebooks/dataset3_seasonal_drift_detection.ipynb` |
| Model comparison table (ConvAE / VAE / ResAttnAE / MemAE) | `notebooks/dataset3_sensitivity_analysis_rq.ipynb` |
| Drift detector comparison (ADWIN / CUSUM / PH / KS / DDM / EDDM) | `notebooks/dataset3_sensitivity_analysis_rq.ipynb` |
| Sensitivity vs environmental variance correlation plot | `notebooks/dataset3_sensitivity_analysis_rq.ipynb` |
| ROC / AUC curves | `notebooks/dataset3_sensitivity_analysis_rq.ipynb` |
| MTTD (Mean Time to Detection) analysis | `notebooks/dataset3_sensitivity_analysis_rq.ipynb` |

---

## Key Results

> **What the numbers mean in plain language:**
> - A **composite drift of ~20–28%** is "moderate" — the seasonal change between winter and summer is real and statistically significant, but the cameras are still functioning (it is not hardware degradation).
> - **100% detection rate** means every sensor flagged a change between years - the autoencoders reliably detected cross-season drift even without any human-labeled examples.
> - **MTTD = 14 frames** means the system detected the drift after only 14 images (~66 hours of real time), which is a fast alarm given the slow pace of seasonal change.
> - **Pearson |r| ≤ 0.07** means reconstruction error is nearly uncorrelated with environmental noise variables (brightness, color contrast, etc.) — the drift signal is genuine, not just lighting variation.

### Dataset 1 — Bristol Streetlight Cameras

| Condition | Samples | MI-LHD | STKA | Composite Drift | Level | Confidence |
|-----------|---------|--------|------|-----------------|-------|------------|
| All images | 40K | 0.1096 | 0.6984 | 22.07% | LOW | 90% |
| Daytime only | 20.6K | 0.0726 | 0.7061 | 19.98% | LOW | 75% |
| Nighttime only | 19.4K | 0.1274 | 0.6682 | 18.93% | LOW | 90% |

- Drift is statistically significant across all conditions (permutation test p ≈ 0.000)
- Daytime shows the lowest drift (19.98%); the separate day/night analysis confirms the signal is genuine seasonal change, not a lighting artifact
- Drift classified as **VIRTUAL** — expected seasonal variation, not sensor degradation

### Dataset 2 — Pomegranate Tree Time Series

- ConvAE and ResAttnAE both detect elevated reconstruction error in 2022 vs 2021 across all 9 sensors
- Sensor-level variance shows that some sensors experience significantly higher concept drift than others
- Per-sensor heatmap identifies which cameras warrant hardware inspection

### Dataset 3 — BMSB Sticky Traps (Sensitivity Analysis, Research Question)

- All four model architectures detect Q3→Q4 drift; VAE and ResAttnAE yield the strongest class separation
- ADWIN provides the most stable change-point detection; CUSUM is most sensitive to gradual drift onset
- Reconstruction loss sensitivity correlates positively with environmental variance (Pearson r reported in paper)
- Threshold sensitivity (50th–98th percentile) quantified with 1,000-iteration bootstrap CIs

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'river'`**  
Install the missing package: `pip install river`

**`FileNotFoundError: results/q1_features_all.npy`**  
The feature cache doesn't exist yet. Either run `python main.py` first (it will extract and cache features automatically), or run `extract_features_fast.py` if you have the raw zip archives.

**`CUDA out of memory`**  
Reduce batch size in the notebook `CONFIG` cell (e.g., change `batch_size: 16` to `batch_size: 8`), or run on CPU by setting `device = torch.device('cpu')`.

**Notebook path errors (e.g., `FileNotFoundError: data/raw/DatasetV3`)**  
The notebooks use paths relative to the `Capstone/` folder. Either: (a) run notebooks from within the `notebooks/` directory after adding `import os; os.chdir('..')` as the first cell, or (b) update `dataset_root` in the `CONFIG` cell to an absolute path on your machine.

**`extract_images.py` cannot find zip files**  
Open the file and update the path constants near the top. The extraction scripts have hardcoded paths from the original development machine — you must change them to match your local file structure.

**Notebook training takes too long**  
Reduce `epochs` in the notebook `CONFIG` cell (e.g., `epochs: 20`). For `main.py`, use `--phase4` to skip VAE training and load a saved checkpoint: `python main.py --phase4`.

---
