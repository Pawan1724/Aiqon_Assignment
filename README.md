# 🚢 Logistics Shipment ETA & Delay Risk Prediction

**Aiqon AI Take-Home Assignment: End-to-End ML Pipeline, Delay Propagation Engine & Interactive Route Planner**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.1+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9+-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)](tests/)

---

## 📌 Project Overview

This repository implements a production-grade Machine Learning pipeline and simulation engine designed to:
1. **Predict Final Shipment ETA & Delay Magnitude (Regression):** Benchmarks Baseline Mean, Ridge Regression, and Random Forest Regressors using cross-validation.
2. **Estimate Delay Risk & Likelihood (Classification):** Classifies shipments into on-time vs. delayed states with class-imbalance-resilient metrics (F1-score & ROC-AUC).
3. **Simulate Cascading Stage Disruption (Delay Propagation):** Dynamically computes cumulative downstream delays across shipment stages (*Departure*, *Transit*, *Customs*, and *Last-Mile Delivery*) under adverse weather and carrier volatility.
4. **Explain Model Decisions (Explainable AI):** Quantifies top contributing delay factors using SHAP and permutation importance.
5. **Interactive Operations Dashboard (Flask Web App):** Provides an intermodal corridor route planner, real-time prediction interface, and stage-by-stage simulation playground.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/Pawan1724/Aiqon_Assignment.git
cd Aiqon_Assignment
```

### 2. Install Dependencies
Ensure Python 3.10+ is installed, then run:
```powershell
pip install -r requirements.txt
```

*(Note: The raw dataset `logistics-shipment-delay-prediction.xlsx` is pre-configured in `data/raw/`).*

---

## 🚀 Execution Guide

You can run the full model training workflow either via **Jupyter Notebook** or via **Terminal Scripts**.

### Option A: Run via Jupyter Notebook (Recommended for Evaluators)

1. Open `notebooks/train_models.ipynb` in VS Code or Jupyter Lab:
   ```powershell
   jupyter notebook notebooks/train_models.ipynb
   ```
2. Click **Run All** (or execute cells sequentially).
3. The notebook will:
   - Clean raw shipment data and build leakage-free pre-departure features.
   - Train and cross-validate regression and classification models.
   - Serialize the best models into `models/eta_model.joblib` and `models/delay_classifier.joblib`.
   - Run a live prediction verification check using `PredictionService`.

---

### Option B: Run via Command Line Scripts

**1. Dataset Audit & Quality Inspection**
```powershell
python scripts/audit_dataset.py
```

**2. Train & Save Models**
```powershell
python scripts/train_models.py
```

**3. Evaluate Generalization & Generate Visual Reports**
```powershell
python scripts/evaluate_models.py
```
*(Saved figures and metric logs are exported to `reports/figures/` and `reports/metrics.json`).*

---

### 🌐 Launch the Interactive Web Application

Once models are trained and saved in `models/`:
```powershell
python app.py
```
Open **`http://localhost:5000`** in your browser to access:
- **Corridor Route Planner:** Multi-modal transport corridor recommendations (Highway, Rail, Ocean, Air).
- **Live ETA Predictor:** Enter shipment parameters and receive ETA predictions, delay risk level, and SHAP contributing factors.
- **Stage Delay Propagation Simulator:** What-if analysis engine testing weather impacts, stage bottlenecks, and cascading delays.
- **Shipment Telemetry Explorer:** Filterable record tables and historical reliability metrics.

---

## 🧪 Running the Test Suite

Run unit and integration tests across data validation, model persistence, route planning, and the propagation engine:
```powershell
python -m pytest
```
*All 32 test cases pass deterministically.*

---

## 📂 Repository Structure

```plaintext
├── config/
│   └── settings.yaml            # Central configuration (paths, hyperparameters, stages)
├── data/
│   ├── raw/                     # Original Excel dataset
│   └── processed/               # Cleaned CSV data (never alters raw dataset)
├── models/
│   ├── eta_model.joblib         # Serialized best ETA regression pipeline
│   ├── eta_model_meta.json      # ETA model metadata & CV performance
│   ├── delay_classifier.joblib  # Serialized best Delay classifier pipeline
│   ├── delay_classifier_meta.json
│   ├── split_info.json          # Stratified train/test split index records
│   └── train_models.ipynb       # Backup training notebook copy
├── notebooks/
│   └── train_models.ipynb       # Main interactive training & verification notebook
├── reports/
│   ├── metrics.json             # Test set evaluation metrics
│   └── figures/                 # Evaluation plots (residuals, ROC, feature importances)
├── scripts/
│   ├── audit_dataset.py         # Data profiling & anomaly check script
│   ├── train_models.py          # End-to-end model training script
│   └── evaluate_models.py       # Model evaluation & figure generation script
├── src/
│   ├── __init__.py              # Central package init with Windows DLL protection
│   ├── data_loader.py           # Excel loader with path resolution
│   ├── preprocessing.py         # Timestamp parsing & discrepancy tracking
│   ├── feature_engineering.py   # Leakage-safe pre-departure features
│   ├── eta_model.py             # ETA regression training & persistence
│   ├── delay_classifier.py      # Delay classification training & persistence
│   ├── propagation_engine.py    # Multi-stage delay propagation simulation engine
│   ├── route_planner.py         # Multi-modal corridor route planning engine
│   └── prediction_service.py    # Unified prediction API with SHAP explainability
├── static/                      # Web assets (CSS stylesheets, JS modules, icons)
├── templates/
│   └── index.html               # Web UI template
├── tests/                       # Complete pytest test suite (32 tests)
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation
```

---

## 📋 Assignment Requirements Checklist

| Requirement | Implementation Details | Status |
| :--- | :--- | :---: |
| **Predict Final ETA & Delay Magnitude** | Benchmarked Baseline Mean, Ridge Regression, and Random Forest Regressors with cross-validated MAE. | ✅ Complete |
| **Estimate Likelihood of Delay** | Logistic Regression & Random Forest Classifiers with stratified splits and F1 / ROC-AUC scoring. | ✅ Complete |
| **Explain Contributing Factors** | SHAP feature attribution and permutation importance identifying top delay drivers. | ✅ Complete |
| **Represent Stage Delay Propagation** | Rule-based cascade engine modeling stage multipliers, weather severity, and disruption propagation. | ✅ Complete |
| **Leakage-Safe Preprocessing** | Strictly excludes post-departure and target-derived columns from training feature space. | ✅ Complete |
| **Zero-Error Interactive Notebook** | `notebooks/train_models.ipynb` runs end-to-end without warnings/errors on both Windows and Linux. | ✅ Complete |
| **Interactive Demo** | Full-featured Flask operations dashboard with route corridor planner, live simulation, and telemetry. | ✅ Complete |
