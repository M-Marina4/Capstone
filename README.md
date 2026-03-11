# Concept Drift Detection in IoT Sensor Networks
## Capstone Project: Monitoring Feature Drift for Reliable Continuous Data Streams

**Author:** Marina Melkonyan  
**Institution:** American University of Armenia  
**Focus Species:** *Punica granatum* L. 'Wonderful' (Pomegranate)  
**Study Period:** 2021-2025 Growing Seasons  
**Deployment:** 9 Independent Raspberry Pi IoT Sensors

---

## 📋 Project Overview

This capstone project addresses **concept drift detection** in long-term IoT sensor networks through unsupervised deep learning. The core challenge: production models degrade over time as data distributions naturally shift due to seasonal changes, environmental drift, and sensor aging. This project develops and validates a comprehensive pipeline to detect these shifts automatically.

### Key Innovation
Rather than relying on single-metric detection (vulnerable to bias), the pipeline employs a **3-layer architecture** combining:
- **Layer 1:** Core autoencoder-based reconstruction error analysis
- **Layer 2:** Multiple independent drift detection pathways with confidence scoring
- **Layer 3:** Comprehensive validation and anomaly attribution

---

## 🎯 Research Questions

This analysis addresses two primary research questions:

### **RQ1: Mean Time To Detection (MTTD)**
> How does the mean time to detection (MTTD) of the validator change when applied to cross-domain data streams compared to its baseline performance in the source domain?

**Key Metrics:**
- Detection latency (hours from start of target domain)
- Detection rate (% of drift samples flagged above threshold)
- Cross-domain penalty factors (domain shift impact quantification)

**Results Location:** [`results/mttd_analysis.csv`](results/mttd_analysis.csv)

### **RQ2: Reconstruction Loss Sensitivity & Environmental Variance**
> What is the sensitivity of unsupervised reconstruction loss as a trigger for drift adaptation in IoT imagery, and how does this sensitivity correlate with environmental variance across long-term data streams?

**Key Metrics:**
- ROC-AUC scores (reconstruction loss discriminative power)
- Optimal F1-achieving thresholds
- Pearson correlation (sensitivity ↔ environmental variance)
- Environmental variance components (brightness, contrast, saturation)

**Results Location:** [`results/sensitivity_analysis.csv`](results/sensitivity_analysis.csv)

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Data Discovery & Preprocessing                   │
│  • Load 2021 (baseline) & 2022 (target) image time series   │
│  • Q1/Q3 temporal windowing (16-hour quarters)              │
│  • Per-sensor train/validation/test splits                  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2-3: Autoencoder Training                            │
│  • Three model architectures (ConvAE, ResAttnAE, TransAE)   │
│  • Per-sensor training with early stopping                  │
│  • Latent representation extraction (bottleneck)            │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Drift Detection (Multiple Pathways)               │
│  • Path 1: Mean Shift (L2 distance)                         │
│  • Path 2: Covariance Shift (Frobenius norm)                │
│  • Path 3: Kolmogorov-Smirnov Tests (per-dimension)         │
│  • Path 4: Mahalanobis Distance (robust metrics)            │
│  → Consensus voting for drift confidence                    │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Drift Classification & Analysis                  │
│  • Virtual/Real/Mixed drift categorization                  │
│  • Environmental metadata correlation                       │
│  • Time-series decomposition for drift attribution          │
│  • MTTD and sensitivity analysis                            │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 6: Validation & Final Reporting                      │
│  • Statistical significance testing                         │
│  • ROC curve analysis                                       │
│  • Environmental variance correlation                       │
│  • Comprehensive CSV & visualization export                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Models & Technologies

### Deep Learning Models

Three autoencoder architectures for robustness and comparison:

| Model | Architecture | Key Feature | Best For |
|-------|--------------|-------------|----------|
| **ConvAE** | Convolutional Encoder-Decoder | Simple, fast inference | Baseline comparison |
| **ResAttnAE** | Residual Blocks + Attention | Attention mechanisms for important features | Feature importance |
| **ConvTransAE** | Transposed Convolution Decoder | Progressive upsampling | Fine-grained reconstruction |

### Key Technologies

- **Deep Learning:** PyTorch 2.0+
- **Data Processing:** Pandas, NumPy, Scikit-Image
- **Statistical Analysis:** SciPy (Mann-Whitney U, KS tests), Scikit-Learn (ROC-AUC)
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** PIL, OpenCV

---

## 📁 Project Structure

```
Capstone/
├── notebooks/                          # Jupyter analysis notebooks
│   ├── drift_detection_pipeline.ipynb  # Main analysis (45 cells)
│   ├── organize_images_by_quarter.ipynb
│   ├── visualizations.ipynb
│   └── research_question_analysis_pipeline.ipynb
│
├── src/                                # Core Python modules
│   ├── autoencoder.py                  # Model architectures & training
│   ├── processing.py                   # Data loading & preprocessing
│   ├── metrics.py                      # Drift detection metrics
│   ├── validator.py                    # Statistical validation
│   ├── decomp.py                       # Time-series decomposition
│   ├── drift_detectors.py              # Multi-pathway drift detection
│   ├── drift_classifier.py             # Drift type classification
│   ├── anomaly_detector.py             # Ensemble anomaly detection
│   └── __init__.py
│
├── data/
│   ├── raw/
│   │   └── Dataset_pomegranate_tree_time_series/
│   │       ├── 2021/
│   │       │   └── Sensor1-9/         # Q1, Q2, Q3 images
│   │       └── 2022/
│   │           └── Sensor1-9/         # Q1, Q2, Q3 images
│   └── metadata/
│       ├── q1q3_daytime_extracted.csv # Temporal metadata
│       └── streetcare-drift-dataset-2021-2025.csv
│
├── results/                            # Output CSV files & analysis
│   ├── drift_detection_results_comparison.csv    # Model comparison
│   ├── drift_detection_results.csv                # Individual model results
│   ├── capstone_results.csv                       # Aggregated findings
│   ├── mttd_analysis.csv                          # RQ1 results
│   └── sensitivity_analysis.csv                   # RQ2 results
│
├── main.py                             # CLI entry point
├── README.md                           # This file
```

---

## 🔍 Dataset Details

### Sensors & Coverage
- **9 Independent Sensors** deployed across pomegranate plantings
- **2 Years** of continuous monitoring (2021-2022)
- **Quarterly Quarters (Q1-Q3)** image collections per year
- **High-resolution JPEG** images from Raspberry Pi cameras
- **Timestamped Metadata** with environmental context

### Data Characteristics
- **2021 (Source Domain):** Baseline distribution, ~53-70 training images per sensor
- **2022 (Target Domain):** Natural seasonal/environmental drift, ~84-105 test images per sensor
- **Environmental Variables:** Brightness, contrast, saturation changes tracked per image

---

## 📈 Key Results Summary

### Research Question 1: MTTD Analysis

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Avg MTTD** | ~12-24 hours | Detection typically within 1 day |
| **MTTD Range** | 0-48 hours | Fast to moderate detection across sensors |
| **Avg Detection Rate** | >95% | High consistency in flagging drift |
| **Cross-Domain Penalty** | 1.8-2.0x | Domain shift increases detection latency |

**Finding:** Models trained on 2021 baseline reliably detect significant drift in 2022 data within 1-2 days of seasonal transition.

### Research Question 2: Sensitivity Analysis

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Avg ROC-AUC** | 0.95-1.0 | Reconstruction loss highly discriminative |
| **Optimal Threshold** | 90-95th percentile | Conservative thresholds minimize false alarms |
| **Sensitivity-Variance Corr.** | -0.15 to +0.35 | Weak to moderate relationship |
| **Best F1 Score** | 0.99+ | Excellent balanced precision/recall |

**Finding:** Reconstruction loss alone is ROBUST across environmental variance—external factors minimally affect detection sensitivity.

---

## 🚀 Running the Pipeline

### Prerequisites
```bash
# Install dependencies
pip install torch pandas numpy matplotlib seaborn scikit-learn scikit-image pillow scipy
```

### Option 1: Jupyter Notebook (Recommended)
```bash
# Navigate to notebooks folder
cd notebooks

# Open and run the main pipeline
jupyter notebook drift_detection_pipeline.ipynb

# Run cells sequentially from top to bottom
# Expected runtime: 15-30 minutes (depends on GPU availability)
```

### Expected Output Files
After successful execution:
- ✅ `drift_detection_results_comparison.csv` (Model metrics comparison)
- ✅ `mttd_analysis.csv` (RQ1 metrics per sensor)
- ✅ `sensitivity_analysis.csv` (RQ2 metrics per sensor)
- ✅ Matplotlib figures (visualizations embedded in notebook)

---

## 📊 Output Files Description

### `drift_detection_results_comparison.csv`
**Purpose:** Compare three model architectures on drift detection

**Columns:**
- `Sensor_ID`: Which sensor (Sensor1-9)
- `N_Train/Val/Test_*`: Dataset sizes per split
- `Conv_Mean_2021/2022`, `Res_Mean_*`, `Trans_Mean_*`: Reconstruction errors per model/year
- `Conv_Drift_%`, `Res_Drift_%`, `Trans_Drift_%`: % increase 2021→2022
- `*_P_Value`: Statistical significance (Mann-Whitney U test)
- `*_Drift_Detected`: Boolean drift detection result

### `mttd_analysis.csv`
**Purpose:** Quantify detection latency (RQ1)

**Key Columns:**
- `MTTD_Hours`: Hours until sustained drift detection
- `Detection_Rate_%`: % of 2022 images flagged above threshold
- `Cross_Domain_Penalty`: How much slower detection vs baseline
- `Baseline_Max_Error`: Threshold from 2021 data
- `Drift_Mean_Error`: Average 2022 reconstruction error

### `sensitivity_analysis.csv`
**Purpose:** Analyze threshold sensitivity (RQ2)

**Key Columns:**
- `ROC_AUC`: Receiver Operating Characteristic AUC
- `Best_F1_Score`: Optimal F1 score achieved
- `Best_F1_Percentile`: Which threshold percentile achieves it
- `Brightness_Change`, `Contrast_Change`, `Saturation_Change`: Environmental variance
- `Env_Variance_Score`: Composite environmental change metric

---

## 📝 Methodology Notes

### Why Reconstruction Loss?
- **Interpretable:** Directly reflects model uncertainty on new data
- **Unsupervised:** No labels required for 2022 data
- **Sensitive:** Captures subtle distribution shifts
- **Robust:** Multiple model architectures provide consensus

### Why Multiple Models?
- **Robustness:** Reduces false alarms from single-model bias
- **Coverage:** Different architectures excel at different drift types
- **Confidence:** Agreement across models strengthens conclusions
- **Generalization:** Prevents overfitting to one approach

### Statistical Validation
- **Mann-Whitney U Test:** Non-parametric test for difference in error distributions
- **ROC Curves:** Threshold-dependent performance across sensitivity/specificity trade-off
- **Pearson Correlation:** Linear relationship between metrics
- **Early Stopping:** Prevents model overfitting during training

---

## 🔧 Configuration

Key configuration parameters (see `drift_detection_pipeline.ipynb` cells 1-3):

```python
CONFIG = {
    "data_root": r"C:\Users\melko\Capstone\data\raw",
    "output_dir": r"C:\Users\melko\Capstone\results",
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "early_stopping_patience": 10,
}
```

---

## 📌 Notebook Cell Structure

### Main Pipeline Notebook (45 cells)

| Cells | Purpose |
|-------|---------|
| 1-3 | Configuration, libraries, paths |
| 4-7 | Data discovery and exploration |
| 8-15 | Autoencoder architectures (3 models) |
| 16-25 | Training loops per sensor |
| 26-35 | Reconstruction error computation |
| 36-37 | Basic drift statistics |
| 38-42 | **MTTD Analysis (RQ1)** |
| 40-42 | **Sensitivity Analysis (RQ2)** |
| 43 | Comprehensive visualizations |
| 44 | Research findings summary |
| 45 | Methodology appendix |

---

## 🎓 Research Contributions

This project advances concept drift detection through:

1. **Multi-Model Validation:** Three independent architectures provide robust consensus
2. **Environmental Correlation:** Quantifies relationship between domain shift severity and detection sensitivity
3. **MTTD Quantification:** Measures realistic detection latency in cross-domain deployment
4. **Unsupervised Approach:** No labels required for 2022 data—critical for real IoT systems
5. **Comprehensive Attribution:** Distinguishes between virtual drift (label distribution), real drift (feature distribution), and measurement drift

---

## 📚 References

- **Autoencoders for Anomaly Detection:** Goodfellow et al. (2016)
- **Concept Drift in Data Streams:** Gama et al. (2014)
- **Statistical Divergence Tests:** Kolmogorov-Smirnov, Mann-Whitney U
- **ROC Analysis:** Fawcett (2006)

---

## 📧 Contact & Questions

For questions about methodology, results, or implementation, refer to:
- Notebook markdown cells with research context

---

## 📄 License

This capstone project is part of coursework at American University of Armenia.

---

**Last Updated:** March 2026  
**Status:** ✅ Analysis Complete | Results Validated | Ready for Production Review