# Seasonal Concept Drift Detection in IoT Sticky-Trap Imagery

**Author:** Marina Melkonyan
**Institution:** American University of Armenia
**Species:** *Halyomorpha halys* (Brown Marmorated Stink Bug, BMSB)
**Dataset:** 476 images (1920×1080) from IoT sticky-trap camera, Jun–Nov 2024
**Seasonal Split:** Q3 (Jul–Sep) reference vs Q4 (Oct–Nov) drift target

---

## Project Overview

This capstone project detects **seasonal concept drift** in IoT pest-monitoring imagery using unsupervised deep learning. Autoencoders are trained on Q3 (summer) sticky-trap images to learn the reference distribution; reconstruction error on Q4 (autumn) images reveals distributional shift caused by seasonal environmental changes and trap degradation.

Two notebooks address the research questions. The **sensitivity analysis notebook is the main notebook** producing the core results.

---

## Research Questions

### RQ: Reconstruction Loss Sensitivity & Environmental Variance *(main notebook)*
> What is the sensitivity of unsupervised reconstruction loss as a trigger for drift adaptation in IoT imagery, and how does this sensitivity correlate with environmental variance across long-term data streams?

- Compare four autoencoder architectures (ConvAE, VAE, ResAttnAE, MemAE)
- Evaluate six drift detectors (ADWIN, Page-Hinkley, CUSUM, KS-Windowed, DDM, EDDM)
- Sweep detection thresholds (50th–98th percentile) for sensitivity curves
- Correlate reconstruction error with environmental metrics (brightness, contrast, saturation, edge density, entropy, color divergence)
- Bootstrap confidence intervals for statistical robustness
- Comprehensive evaluation: ROC-AUC, F1, MTTD, Cohen's d, Mann-Whitney U

**Notebook:** [`notebooks/sensitivity_analysis_rq.ipynb`](notebooks/sensitivity_analysis_rq.ipynb) *(main)*
**Results:** `results/sensitivity_comprehensive_eval.csv`, `results/sensitivity_correlation_analysis.csv`, `results/sensitivity_detector_results.csv`, `results/sensitivity_bootstrap_ci.csv`

### Supporting: Seasonal Drift Detection & MTTD
> How quickly can an autoencoder-based system detect seasonal concept drift when monitoring transitions from summer (Q3) to autumn (Q4)?

- Train a single deep convolutional autoencoder on Q3 images
- Measure reconstruction error increase on Q4
- Compute Mean Time To Detect (MTTD) using consecutive-detection criterion
- Analyze drift by trap replacement cycle

**Notebook:** [`notebooks/seasonal_drift_detection.ipynb`](notebooks/seasonal_drift_detection.ipynb) *(supporting)*
**Results:** `results/seasonal_drift_results.csv`, `results/seasonal_mttd_analysis.csv`

---

## Dataset

**Source:** Kargar, A., Zorbas, D., Gaffney, M., O'Flynn, B., & Tedesco, S. (2024). *Image Dataset of Brown Marmorated Stink Bug (BMSB) or Halyomorpha Halys (HH)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7887045

| Property | Value |
|----------|-------|
| Total images | 476 |
| Resolution | 1920×1080 px |
| Capture device | IoT camera (sticky trap) |
| Location | Orchard in Italy |
| Period | June–November 2024 |
| Trap replacements | 9 sequential over the season |
| Annotations | Polygon JSON + binary masks per HH instance |
| Metadata CSV | Filename, HH count, trap ID |

### Seasonal Split

| Quarter | Months | Role | Note |
|---------|--------|------|------|
| Q3 | Jul–Sep 2024 | Reference (70/15/15 train/val/test) | Summer peak season |
| Q4 | Oct–Nov 2024 | Drift target (100% test) | Autumn late season |

> Q2 (June) was excluded due to insufficient sample size (7 images).

---

## Project Structure

```
Capstone/
├── notebooks/
│   ├── sensitivity_analysis_rq.ipynb    # Main — multi-model sensitivity & correlation
│   └── seasonal_drift_detection.ipynb   # Supporting — single-model drift & MTTD
│
├── data/
│   └── raw/
│       └── DatasetV3/
│           ├── Images/                  # 476 JPEG sticky-trap images
│           ├── Annotations/
│           │   ├── HH_Polygons/         # Per-image JSON polygon annotations
│           │   └── Masks/               # Binary masks of HH instances
│           ├── Num_HHs.csv              # HH count + trap ID per image
│           └── README.txt               # Original dataset documentation
│
├── results/                             # Generated CSV outputs
│   ├── sensitivity_comprehensive_eval.csv   # Main: full eval table
│   ├── sensitivity_detector_results.csv     # Main: detector comparison
│   ├── sensitivity_correlation_analysis.csv # Main: env ↔ error correlations
│   ├── sensitivity_bootstrap_ci.csv         # Main: bootstrap CIs
│   ├── sensitivity_env_metrics_errors.csv   # Main: per-image env metrics
│   ├── sensitivity_sweep_*.csv              # Main: threshold sweep per model/quarter
│   ├── seasonal_drift_results.csv           # Supporting: drift detection statistics
│   ├── seasonal_mttd_analysis.csv           # Supporting: MTTD per quarter
│   └── seasonal_per_image_errors.csv        # Supporting: per-image reconstruction errors
│
└── README.md
```

---

## Models

| Model | Architecture | Loss | Reference |
|-------|-------------|------|-----------|
| **ConvAE** | 3-block conv encoder-decoder, FC bottleneck | MSE | Baseline |
| **VAE** | Same conv backbone + reparameterization trick | ELBO (MSE + β·KL) | Kingma & Welling 2014 |
| **ResAttnAE** | Residual blocks + Squeeze-and-Excitation attention | MSE | He+2016, Hu+2018 |
| **MemAE** | Memory-augmented autoencoder with prototype bank | MSE + entropy reg. | Gong et al., ICCV 2019 |

All models resize input to 128×72 (16:9), use latent dimension 128, and are trained with Adam (lr=1e-3) and early stopping (patience=10).

## Drift Detectors

| Detector | Type | Reference |
|----------|------|-----------|
| ADWIN | Adaptive windowing | Bifet & Gavaldà 2007 |
| Page-Hinkley | Sequential cumulative deviation | Page 1954 |
| CUSUM | Cumulative sum control chart | Page 1954 |
| KS-Windowed | Sliding-window Kolmogorov-Smirnov | Non-parametric |
| DDM | Error-rate monitoring | Gama et al. 2004 |
| EDDM | Error-distance monitoring | Baena-García et al. 2006 |

---

## Technologies

- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Statistical Analysis:** SciPy (Mann-Whitney U, KS tests), Scikit-Learn (ROC-AUC, bootstrap)
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** PIL

---

## Running the Analysis

### Prerequisites

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn pillow scipy
```

### Execution

Run the main notebook (top to bottom). The supporting notebook can be run independently.

```bash
cd notebooks
jupyter notebook sensitivity_analysis_rq.ipynb        # Main analysis
jupyter notebook seasonal_drift_detection.ipynb        # Supporting (optional)
```

Runtime depends on GPU availability; expect 15–30 minutes per notebook on CPU.

### Outputs

After execution, CSV results are written to `results/` and all visualizations are embedded in the notebooks.

---

## Key Output Files

| File | Contents |
|------|----------|
| `seasonal_drift_results.csv` | Per-quarter drift %, p-value, detection flag |
| `seasonal_mttd_analysis.csv` | MTTD in frames and hours, detection rate |
| `seasonal_per_image_errors.csv` | Per-image MSE with quarter, trap, HH count |
| `sensitivity_comprehensive_eval.csv` | ROC-AUC, F1, precision, recall, Cohen's d per model × quarter |
| `sensitivity_detector_results.csv` | Detection counts per model × quarter × detector |
| `sensitivity_correlation_analysis.csv` | Pearson/Spearman/Kendall/MI: env metrics ↔ recon error |
| `sensitivity_bootstrap_ci.csv` | 95% bootstrap CIs for TPR, FPR, F1, AUC |
| `sensitivity_sweep_*.csv` | Threshold sweep metrics (50th–98th percentile) per model |

---

## References

- Kargar, A. et al. (2024). Image Dataset of BMSB. Zenodo. https://doi.org/10.5281/zenodo.7887045
- Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
- Gong, D. et al. (2019). Memorizing Normality to Detect Anomaly. ICCV.
- He, K. et al. (2016). Deep Residual Learning. CVPR.
- Hu, J. et al. (2018). Squeeze-and-Excitation Networks. CVPR.
- Bifet, A. & Gavaldà, R. (2007). Learning from Time-Changing Data with Adaptive Windowing. SDM.
- Gama, J. et al. (2004). Learning with Drift Detection. SBIA.
- Baena-García, M. et al. (2006). Early Drift Detection Method. ECML PKDD Workshop.
