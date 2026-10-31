# MROP: Investigating Neural Correlates of Consciousness Through Computational EEG Analysis

**Master's Research Oriented Project | Computational Neuroscience | EEG Analysis**

## Overview

This project explores neural correlates of consciousness (NCC) through computational analysis of electroencephalography (EEG) data, motivated by research on split-brain epilepsy patients.

The original research objective was to investigate neural correlates of consciousness in patients with drug-resistant epilepsy who had undergone corpus callosotomy, a surgical procedure that disconnects the cerebral hemispheres by severing the corpus callosum.

## Research Scope and Dataset Limitation

A major limitation was the unavailability of a suitable publicly accessible EEG dataset containing recordings from split-brain patients.

To address this limitation, the research approach was adapted using publicly available seizure- and spike-related EEG recordings from the Harvard EEG Brain Data Science Platform (BDSP).

Instead of directly studying patients with surgically disconnected hemispheres, the project focused on computational characterisation of EEG activity, including:

- Left-right hemispheric activity and asymmetry.
- Functional connectivity between EEG channels.
- Brain network organisation using graph-theoretical measures.
- Time-frequency characteristics of EEG signals.
- Machine learning-based analysis of EEG patterns.

**Research limitation:** Hemispheric asymmetry in conventional EEG recordings cannot establish functional disconnection between cerebral hemispheres. Consequently, these analyses do not directly demonstrate neural correlates of consciousness in split-brain patients.

## Research Objectives

- Explore hemispheric differences in EEG activity.
- Investigate functional connectivity and neural network organisation.
- Characterise EEG signals through time-frequency analysis.
- Apply machine learning to EEG data.
- Explore pretrained EEG foundation models for downstream analysis.

## Computational Workflow

### 1. EEG Data Inspection
Loading recordings, examining channels, sampling characteristics and annotations.

### 2. Signal Preprocessing
Filtering and preparing EEG recordings for subsequent analysis.

### 3. Window Extraction
Segmenting continuous EEG recordings into fixed-duration windows.

### 4. Feature Extraction and Comparative Analysis
Extracting EEG characteristics and comparing signal patterns.

### 5. REVE Foundation Model
Integrating the REVE EEG foundation model into the downstream analysis workflow.

### 6. Machine Learning
Training models and exploring single-subject and multi-subject approaches.

### 7. Hemispheric Asymmetry
Investigating left-right hemispheric differences in EEG activity.

### 8. Time-Frequency Analysis
Examining frequency-dependent characteristics of EEG signals.

### 9. Functional Connectivity
Investigating relationships between EEG channels.

### 10. Graph-Theoretical Analysis
Representing connectivity patterns as networks and investigating their organisation.

### 11. Topographical Visualisation
Generating EEG topographical maps to visualise spatial patterns.

### 12. Model Interpretation
Using SHAP-based analysis to investigate feature contributions to model predictions.

## Repository Structure

```text
MROP_NCC_Consciousness/
│
├── analysis_pipeline/
│   ├── preprocessing and inspection
│   ├── feature extraction
│   ├── model training
│   ├── hemispheric asymmetry
│   ├── time-frequency analysis
│   ├── connectivity analysis
│   ├── graph theory
│   └── SHAP analysis
│
├── REVE/
│   └── reve_eeg/
│       ├── src/
│       ├── preprocessing/
│       ├── preprocessing_harvard_seizure.py
│       ├── pyproject.toml
│       ├── README.md
│       ├── CITATION.cff
│       └── LICENSE
│
├── Preliminary pipeline scripts
├── config.py
├── .gitignore
└── README.md
```

## Technologies

- Python
- MNE-Python
- NumPy and SciPy
- PyTorch
- Scikit-learn
- REVE EEG Foundation Model
- SHAP
- Matplotlib

## Data Availability

Raw EEG recordings and restricted patient-related data are not included in this repository.

Dataset access and usage are subject to the respective providers' permissions and licensing requirements.

Pretrained model checkpoints and generated intermediate outputs are also excluded.

## Attribution

The REVE EEG foundation model is an independently developed research framework incorporated into this project.

Its original source code, authorship, citation metadata and licensing information are retained under `REVE/reve_eeg/`.

Please refer to its original README and CITATION.cff when using or citing REVE.

## Limitations

- No suitable split-brain patient EEG dataset was available for direct analysis.
- The available epilepsy-related recordings do not represent corpus callosotomy patients.
- Hemispheric asymmetry does not establish interhemispheric disconnection.
- Computational findings require further validation before biological or clinical interpretation.

## Disclaimer

This repository documents research conducted as part of a Master's Research Oriented Project (MROP). The analyses are intended for academic and research purposes and are not clinical diagnostic tools.
