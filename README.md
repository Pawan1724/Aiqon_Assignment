# Shipment ETA & Delay Prediction

Aiqon AI Internship Take-Home Assignment

## Project Overview

This project implements a complete Machine Learning pipeline and simulation engine to predict shipment ETA, classify delay risk, and model how delays propagate across shipment stages. 

The system uses pre-departure features to predict final ETA and utilizes a rule-based engine to simulate stage-by-stage delay propagation, as intermediate timestamps were not present in the dataset.

## Setup & Installation (Windows PowerShell)

```powershell
# 1. Ensure Python 3.10+ is installed.
# 2. Clone/download this repository.
# 3. Install dependencies:
pip install -r requirements.txt
```

*(Note: Place `logistics-shipment-delay-prediction.xlsx` in the `data/raw/` directory if not already there).*

## Execution Instructions

Run these commands from the project root directory:

**1. Data Audit**
```powershell
python scripts/audit_dataset.py
```

**2. Train Models**
```powershell
python scripts/train_models.py
```

**3. Evaluate Models (Generates Plots)**
```powershell
python scripts/evaluate_models.py
```

**4. Launch Web Application**
```powershell
python app.py
```

## Running Tests

```powershell
pytest tests/
```

## Architecture & Features

- **Strict Leakage Prevention:** Only features available before departure are used in the ETA model.
- **Unified Prediction Service:** Combines ML predictions with rule-based simulations.
- **Delay Propagation Engine:** Simulates downstream cascading delays caused by disruptions.
- **Explainable AI:** Uses SHAP and permutation importance to explain model decisions.

## Requirements Checklist

- [x] Predict ETA and delay magnitude.
- [x] Estimate likelihood of delay (Classification).
- [x] Explain contributing factors (SHAP).
- [x] Represent relationship between stages and delay propagation (Simulated Engine).
- [x] Handle missing data and evaluate performance.
- [x] Interactive Demo (Flask Web App).

