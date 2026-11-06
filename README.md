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

## Results and Key Findings

### 1. Classical Machine Learning: LOSO Cross-Validation

Classical machine learning models were evaluated using leave-one-subject-out (LOSO) cross-validation.

| Model | ROC-AUC | Accuracy | Baseline Recall | Spike Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.658 | 0.859 | 0.104 | 0.983 |
| Random Forest | 0.623 | 0.859 | 0.005 | 0.999 |

Evaluation used three reliable folds across four selected subjects, covering 52,278 windows.

Although both models achieved approximately 86% accuracy, their extremely low baseline recall revealed substantial class imbalance and a tendency to predict the spike class. ROC-AUC was therefore considered a more informative performance metric.

### 2. REVE Foundation Model: Linear Probing

The pretrained REVE encoder was evaluated using linear probing, with a frozen backbone and a trainable classification head.

| Split | Accuracy | Balanced Accuracy | F1-score | AUROC | AUC-PR |
|---|---:|---:|---:|---:|---:|
| Training | 0.610 | 0.611 | 0.611 | 0.640 | 0.578 |
| Validation | 0.592 | 0.594 | 0.593 | 0.625 | 0.565 |
| Held-out Test | 0.532 | 0.524 | 0.508 | 0.487 | 0.473 |

The held-out test AUROC of 0.487 indicated limited generalisation to unseen subjects. The progressive decline from training to testing suggested subject-level overfitting, potentially reflecting the limited size and diversity of the available cohort.

### 3. Spectral Band-Power Analysis

Spike and baseline EEG windows were compared across five canonical frequency bands.

In an example session (sub-031/ses-1), spike windows exhibited:

- Increased delta power.
- Increased theta power.
- Increased alpha power.
- Increased beta power.
- Reduced gamma power.

Theta and alpha power emerged among the strongest statistically discriminating features, with effect sizes approximately 0.46–0.50.

These observations suggest a shift towards lower-frequency spectral activity during epileptiform events.

### 4. Hemispheric Power Asymmetry

Hemispheric asymmetry was evaluated using eight homologous left-right electrode pairs across five frequency bands.

The most consistent effects were observed in frontal electrode pairs:

| Band-Pair | Spike Asymmetry | Baseline Asymmetry | Subject Agreement |
|---|---:|---:|---:|
| Alpha F3–F4 | -0.259 | -0.132 | 4/4 |
| Theta F3–F4 | -0.271 | -0.149 | 4/4 |
| Beta F3–F4 | -0.255 | -0.153 | 4/4 |
| Gamma F3–F4 | -0.203 | -0.095 | 4/4 |
| Theta T5–T6 | 0.040 | -0.063 | 2/4 |
| Delta T5–T6 | 0.037 | -0.083 | 2/4 |

Frontal channel pairs demonstrated consistent right-lateralised asymmetry during spike activity across all four subjects evaluated.

Temporal and parietal effects showed weaker subject-level consistency and require further investigation.

### 5. Functional Connectivity Analysis

Session-paired connectivity analysis across 14 sessions identified six edges that remained significant after FDR correction.

All six demonstrated weakened or reversed connectivity during spike activity.

| Electrode Pair | Spike Mean | Baseline Mean | Session Agreement |
|---|---:|---:|---:|
| Fz–Pz | -0.304 | -0.032 | 14/14 |
| F7–P3 | 0.170 | 0.306 | 14/14 |
| Fz–P4 | -0.185 | 0.041 | 13/14 |
| F3–P4 | -0.186 | 0.036 | 12/14 |
| F7–P4 | 0.022 | 0.171 | 12/14 |
| Cz–Fz | -0.159 | 0.098 | 12/14 |

The findings highlighted changes in anterior-posterior functional connectivity, particularly involving frontal, central and parietal electrodes.

### 6. Graph-Theoretical Analysis

Graph-theoretical metrics were calculated from functional connectivity networks thresholded at |r| >= 0.30.

Key findings:

- Clustering coefficient showed a significant difference between spike and baseline conditions (FDR-adjusted p = 0.012).
- Network density did not show a significant difference.
- Global efficiency did not show a significant difference.
- Characteristic path length did not show a significant difference.
- No individual electrode demonstrated significant changes in strength, betweenness or eigenvector centrality after FDR correction.

These findings suggest that epileptiform activity was associated with changes in network clustering and specific connectivity relationships rather than significant changes across all global network measures.

### 7. SHAP-Based Model Interpretability

SHAP (SHapley Additive exPlanations) was applied to the multi-subject Random Forest classifier using TreeExplainer.

The analysis highlighted:

- Theta and alpha power at posterior and temporal electrodes as important features contributing to spike predictions.
- Higher gamma power at frontal electrode F3 contributing towards baseline predictions.
- Reduced gamma power contributing towards spike predictions.

These patterns were consistent with the independent spectral band-power analysis, providing complementary evidence for the identified frequency-dependent EEG characteristics.

### 8. Overall Interpretation

Across the different computational analyses, several findings converged:

- Classical machine learning models outperformed REVE linear probing on the evaluated held-out subjects.
- Epileptiform activity was associated with increased lower-frequency power and reduced gamma power in the example session.
- Frontal hemispheric asymmetry shifted towards right-sided dominance consistently across the four subjects examined.
- Functional connectivity analysis identified reproducible changes in anterior-posterior electrode relationships.
- Graph-theoretical analysis identified a significant change in clustering coefficient.

Together, these findings provide an exploratory characterisation of electrophysiological and network-level changes associated with epileptiform activity in the available epilepsy EEG cohort.

**Interpretation boundary:** These results do not establish neural correlates of consciousness or demonstrate functional disconnection between cerebral hemispheres. The dataset contained no documented split-brain patients, and the analyses should be interpreted as exploratory findings in a general clinical epilepsy population.

### 9. Future Directions

- Extend preprocessing and evaluation to the remaining subjects.
- Improve subject-level generalisation through larger and more balanced cohorts.
- Investigate full fine-tuning of the REVE foundation model.
- Explore behavioural and clinical annotations to relate EEG patterns to arousal and awareness states.
- Investigate ZUNA1.1 for reconstruction of missing EEG channels.
- Develop a separate validated pipeline for intracranial EEG data from OpenNeuro dataset ds003029.
- Perform matched evaluation across classical and foundation-model approaches.

## Limitations

- No suitable split-brain patient EEG dataset was available for direct analysis.
- The available epilepsy-related recordings do not represent corpus callosotomy patients.
- Hemispheric asymmetry does not establish interhemispheric disconnection.
- Computational findings require further validation before biological or clinical interpretation.

## Disclaimer

This repository documents research conducted as part of a Master's Research Oriented Project (MROP). The analyses are intended for academic and research purposes and are not clinical diagnostic tools.
