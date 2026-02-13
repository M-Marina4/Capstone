# Architecture: How the Modules Work Together

## The Problem Before

Your original code had:
- **Phase 4**: MI-LHD + STKA metrics
- **Problem**: Only ONE way to measure drift; vulnerable to bias or error
- **Result**: High confidence but low robustness

## The Solution: 3-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Core Pipeline (Phases 1-6)                   │
│  Q1/Q3 data → Features → Autoencoder → Metrics → Results│
└──────────────────────┬──────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       v               v               v
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  LAYER 2A    │  │  LAYER 2B    │  │   LAYER 2C       │
│ Enhances     │  │ Classifies   │  │  Detects         │
│ Confidence   │  │ Drift Type   │  │  Anomalies       │
│              │  │              │  │                  │
│ Phase 4 Ext. │  │ Phase 5 Ext. │  │  Phase 5 Ext.    │
│              │  │              │  │                  │
│ 4 pathways   │  │ Virtual/Real/│  │  Reconstruction  │
│ + voting     │  │ Mixed        │  │  + Metadata      │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                   │
       └─────────────────┼───────────────────┘
                         │
                         v
        ┌────────────────────────────────┐
        │  Layer 3: Final Integration    │
        │  Merge all results             │
        │  Generate comprehensive report │
        └────────────────────────────────┘
```

## Layer 2A: Enhanced Confidence (drift_detectors.py)

**Goal**: Strengthen Phase 4 drift detection with multiple confirmations

**Components**:
```
Input: latent_q1, latent_q3 from Phase 3 autoencoder
                    ↓
    ┌───────────────┼───────────────┐
    │               │               │
    v               v               v
Pathway 1       Pathway 2        Pathway 3      Pathway 4
Mean Shift    Covariance Shift   KS Tests      Mahalanobis
(L2 distance) (Frobenius norm)   (per-dim)     (robust distance)
    │               │               │              │
    └───────────────┼───────────────┴──────────────┘
                    │
                    v
        Normalize to [0,1] scores
                    │
                    v
            Average pathway scores
                    │
                    v
        Overall Confidence = mean(pathways)
        Consistency = 1 - std(pathways)
                    │
                    v
            Output: Confidence in Phase 4 result
```

**Why this works**:
- If all 4 pathways agree → High confidence, low consistency variance
- If pathways disagree → Lower confidence, triggers further investigation
- MI-LHD already detected drift (Phase 4); this confirms it independently

**Example output**:
```
Pathway 1 (Mean Shift):        0.75  ← Large mean difference
Pathway 2 (Covariance Shift):  0.62  ← Moderate covariance change
Pathway 3 (KS Rejection):      0.85  ← Most dimensions differ
Pathway 4 (Mahalanobis):       0.70  ← Robust distance confirms

Overall Confidence: 73% ← Average of 4 pathways
Consistency: 85%   ← Pathways mostly agree
Status: DRIFT CONFIRMED ← High consistency + high confidence
```

---

## Layer 2B: Drift Classification (drift_classifier.py)

**Goal**: Distinguish VIRTUAL (expected seasonal) from REAL (sensor problems)

**Components**:
```
Input: drift_results (Phase 4), validation (Phase 5)
                    ↓
        ┌───────────────┬───────────────┐
        │               │               │
        v               v               v
    VIRTUAL Indicators REAL Indicators Mixed Check
    
    High RGB shift      High drift mag   Both significant
    + GPS stable        + Low consistency
    + RGB pattern        + Fault changes
    └───────────────┬───────────────┘
                    │
            Score each pathway [0,1]:
            VIRTUAL_SCORE = 0.75
            REAL_SCORE = 0.45
                    │
                    v
        Compare and classify:
        IF VIRTUAL > 0.6 → "VIRTUAL"
        ELIF REAL > 0.6 → "REAL"  
        ELSE → "MIXED"
                    │
                    v
        Output: Drift type + confidence
```

**Why this matters**:
- **VIRTUAL drift**: Light changes Q1→Q3 (normal, no action)
- **REAL drift**: Sensor degradation (investigate, maintain)
- **MIXED**: Both happening (complex situation, detailed analysis needed)

**Also analyzes**:
- Which metadata field is most anomalous (brightness? fault rate? GPS?)
- Per-field anomaly scores
- Actionable recommendations

**Example output**:
```
Drift Type: VIRTUAL (75% confidence)
Explanation: Drift appears seasonal due to high RGB shift, 
             stable GPS, consistent metadata. No immediate 
             action required.

Most Anomalous Field: brightness
- Brightness anomaly: 0.82 (significant change)
- Fault detection anomaly: 0.15 (stable)
- GPS anomaly: 0.02 (no movement)
```

---

## Layer 2C: Anomaly Detection (anomaly_detector.py)

**Goal**: Identify problematic samples and sensor issues

**Components**:
```
Input: q1_features, q3_features, reconstruction errors, validation
                    ↓
        ┌───────────────┬───────────────┐
        │               │               │
        v               v               v
    Reconstruction   Metadata      Combine into
    Error Pathway    Pathway       Ensemble Vote
    
    Q1 baseline      Brightness    Check if:
    Q3 vs baseline   Fault delta   - Reconstruction
    per-sample       Confidence    - Metadata
    error shifts     GPS shift     Both anomalous?
                     Consistency   
    
    Anomaly if > threshold
    
    └───────────────┬───────────────┘
                    │
            Score per-sample:
            anomaly_score = 
              0.5 * recon_anomaly +
              0.5 * metadata_anomaly
                    │
                    v
        Flag if > 0.5 as anomalous
                    │
                    v
        Output: Anomaly flags + confidence
```

**Why this matters**:
- Identifies individual **problematic samples**
- Distinguishes reconstruction errors (sensor noise) from metadata issues (configuration changes)
- Ensemble voting more robust than single detector

**Also analyzes**:
- % of Q3 samples anomalous (e.g., 12.5% indicate widespread issue)
- Primary anomaly type (reconstruction vs metadata)
- Per-camera or temporal patterns

**Example output**:
```
Anomalous Samples: 142 / 1136 (12.5%)
Ensemble Confidence: 81%

Reconstruction Pathway:
- Q1 mean error: 0.024
- Q3 mean error: 0.031 (29% increase)
- Threshold exceeded: 142 samples

Metadata Pathway:  
- Primary anomaly: brightness (shift +0.18)
- Secondary: fault_detection (shift +0.05)

Recommendation: Significant anomalies detected. 
Recommend investigation into brightness calibration 
and potential fault rate increases.
```

---

## How They Work Together

### Scenario: Large drift detected in Phase 4

**Phase 4**: MI-LHD = 0.45 (high drift detected)

**Layer 2A** (Confidence):
- Runs 4 independent statistical tests
- All show agreement: 70-80% all pathways positive
- **Result**: "High confidence this drift is REAL"

**Layer 2B** (Classification):
- Analyzes metadata patterns
- RGB shift is very high (+0.25) but GPS stable
- Fault detection rate increased
- **Result**: "This is REAL drift (sensor degradation), not VIRTUAL"

**Layer 2C** (Anomalies):
- Finds 15% of Q3 samples have high reconstruction errors
- Brightness metadata also anomalous
- **Result**: "Specific samples affected; likely sensor malfunction or lighting issue"

**Final Report**:
```
✓ Drift detected: MI-LHD=0.45 (Phase 4)
✓ Confidence: 73% (multiple pathways agree)
✓ Type: REAL (not seasonal) 
✓ Primary issue: brightness + reconstruction errors (12% samples)
→ Recommendation: Check sensor calibration and brightness settings
```


## With Integration (The Solution)

- Modules initialize **automatically** in the right phase
- Results **flow naturally** through pipeline
- Single `python main.py` runs everything
- Output is **unified** in final report
- Easy to understand the complete analysis

---

## Data Flow

```
main.py PHASE 1-3
│
├─ Load Q1/Q3 data
├─ Extract features  
└─ Train autoencoder → latent_q1, latent_q3
     │                   q1_features, q3_features
     │
     v
main.py PHASE 4
├─ analyze_drift() → drift_results (MI-LHD, STKA, etc.)
│
└─ [NEW] drift_detectors.py
   ├─ DistributionShiftAnalyzer(latent_q1, latent_q3)
   │  → enhanced_metrics
   ├─ DriftConfidenceEstimator(drift_results, enhanced_metrics)
   │  → confidence_results
   └─ MERGE: drift_results.update(enhanced_metrics + confidence)
     │
     v
main.py PHASE 5  
├─ validate_drift() → validation (metadata analysis)
│
├─ [NEW] drift_classifier.py
│  ├─ DriftTypeClassifier(drift_results, validation)
│  │  → drift_classification (VIRTUAL/REAL/MIXED)
│  └─ MetadataAnomalyDetector(validation)
│     → metadata_anomalies
│
├─ [NEW] anomaly_detector.py
│  ├─ AnomalyDriftEnsemble()
│  ├─ .fit(q1_recon_errors)
│  └─ .detect(q3_recon_errors, validation)
│     → anomaly_results
│
└─ MERGE: validation.update(classification + anomalies)
     │
     v
main.py PHASE 6-7
├─ decomposition = compute_time_series_drift_scores()
├─ report = generate_validation_report()
└─ PRINT: Final results table with all analyses
```

---

## Summary

**Layer 2A (Detectors)**: "Is the drift real?" → Confidence score
**Layer 2B (Classifier)**: "What kind of drift?" → Type (VIRTUAL/REAL/MIXED)  
**Layer 2C (Anomalies)**: "Which samples/sensors are affected?" → Anomaly flags

Together they provide **robust, comprehensive, actionable drift analysis**.
