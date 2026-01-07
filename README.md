# TCGA Pan-Cancer Survival Modeling
# TCGA Pan-Cancer Survival Modeling Toolkit

## Problem

Multi-cancer transcriptomic modeling for survival prediction using TCGA data. This project integrates clinical and gene expression data from six cancer types to build both traditional and deep learning–based survival models. The goal is to evaluate cross-cancer risk prediction performance and demonstrate scalable modeling pipelines suitable for large public cohorts.

---

## Dataset

- **Source**: [UCSC Xena Toil Recomputed TPMs](https://xenabrowser.net/datapages/) + Xena clinical data
- **Cancer types**:
  - Glioblastoma (GBM)
  - Lower Grade Glioma (LGG)
  - Lung Adenocarcinoma (LUAD)
  - Breast Cancer (BRCA)
  - Colon Adenocarcinoma (COAD)
  - Skin Cutaneous Melanoma (SKCM)
- **Modalities**:
  - Gene-level RNA-seq (TPM)
  - Clinical survival metadata (OS time/event, age, cancer type)

---

## Methods (high level)

1. **Data acquisition**:
   - Download pan-cancer expression and clinical data from UCSC Xena.
   - Subset to primary tumor samples.

2. **Preprocessing**:
   - Select top 2000 most variable genes.
   - Z-score scaling of expression features.
   - Harmonize clinical survival metadata (event, time, age).

3. **Exploratory analysis**:
   - PCA and UMAP visualization of expression matrix across cancers.
   - Kaplan–Meier survival curves by cancer type.

4. **Modeling**:
   - **ElasticNet Cox (lifelines)**:
     - Cross-validated penalized Cox model using top gene features.
     - Kaplan–Meier stratification using model-predicted risk.
   - **PyTorch MLP Cox** (custom):
     - Custom deep neural net with Cox partial likelihood loss.
     - Dropout, batch norm, and ReLU activations.
     - Custom stratification using predicted neural risk scores.

5. **Evaluation**:
   - Concordance index (C-index) for train/test splits.
   - KM plots for high vs low risk stratification (median cutoff).

---

## Results

### ElasticNet Cox (lifelines)

- Mean CV C-index: **0.676**
- Test C-index: **0.687**
- Output: `results/tables/elasticnet_summary.csv`
- KM Plot: `figures/km_elasticnet_high_low_test.png`

Selected features include genes with known cancer associations and prognostic roles. ElasticNet provides a transparent, interpretable sparse model.

---

### PyTorch Cox MLP (custom)

- Architecture: 128 → 64 → 1 (2-layer MLP)
- Activation: ReLU, Dropout(0.2), BatchNorm
- Loss: Cox negative partial log-likelihood
- Train C-index: **0.680**
- Test C-index: **0.687**
- Output: `results/tables/pytorch_survival_summary.tsv`
- KM Plot: `figures/km_pytorch_risk_strat_test.png`

The PyTorch model replicates ElasticNet performance while being fully customizable for future extensions (multi-omics, attention, or transfer learning).

---

## Why PyTorch?

This project includes a from-scratch PyTorch survival model to demonstrate deep learning skills without relying on external libraries like `pycox`. The model uses a custom loss function and supports modular experimentation with architecture and optimization.

---

## File structure
.
├── data/
│ ├── raw/ # Do not commit raw data here
│ └── processed/ # Processed expression tables (e.g. top 2000 genes)
├── results/
│ ├── tables/ # Model summaries, C-index tables
│ └── figures/ # KM plots, UMAPs, PCA
├── scripts/
│ ├── 01_make_expression_subset.py
│ ├── 02_pca_umap.py
│ ├── 05_fetch_clinical_from_xena.py
│ ├── 06_survival_baselines.py
│ ├── 07_expression_survival_model.py
│ └── 08_survival_pytorch_custom.py
├── README.md
├── requirements.txt
├── .gitignore

Requirements
pandas
numpy
scikit-learn
lifelines
matplotlib
torch


Install with:

pip install -r requirements.txt
