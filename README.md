# 🏥 Intelligent Hospital Queue & Appointment Optimization System

[![Build & Test Status](https://img.shields.io/badge/tests-34%20passed-success)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.95-blue)](app/main.py)
[![React](https://img.shields.io/badge/React-Vite-purple)](frontend/)
[![Docker Compose](https://img.shields.io/badge/Docker-Ready-2496ED)](docker-compose.yml)

## 📌 Project Overview
An end-to-end AI-powered hospital queue and appointment optimization platform built with Python SimPy discrete-event simulation, domain feature engineering, empirical prediction intervals, non-causal explainability, production monitoring, FastAPI backend, and React glassmorphism dashboard.

> ⚠️ **Data Disclaimer**: This project uses 24,002 synthetic/simulated patient encounter records generated via Discrete-Event Simulation (DES). It contains **no real patient medical data (PHI/PII)**.  
> ⚠️ **Clinical Disclaimer**: Predictions are operational estimates intended solely for queue-management decision support and are **not medical diagnoses or clinical guarantees**.

---

## 🏗️ Architecture Pipeline

```text
DES Simulator (SimPy)
        ↓
24,002 Synthetic Patient Encounters
        ↓
Chronological Data Loader (70/15/15 Split)
        ↓
Domain Feature Engineering (Load Ratios, Traffic Momentum)
        ↓
Model Benchmarking & Staged Hyperparameter Tuning
        ↓
Tuned Random Forest Model (Test MAE: 10.71 min, RMSE: 45.98 min)
        ↓
Empirical Prediction Intervals (Validation Quantile Calibration)
        ↓
Model Explainability & Non-Causal Drivers
        ↓
Production PSI Statistical Drift Monitoring
        ↓
FastAPI Hardened REST API
        ↓
React + Vite Glassmorphism Dashboard
```

---

## 📊 Model Performance & Metrics

Evaluating the final **Tuned Random Forest Regressor** on the untouched chronological Test set (3,601 records):

| Metric | Test Metric Value | Operational Context |
| :--- | :--- | :--- |
| **Test MAE** | **10.7066 min** | Mean absolute error across patient encounters. |
| **Test RMSE** | **45.9751 min** | **-10.2% error reduction** in peak heavy-tail surge spikes. |
| **Test $R^2$** | **0.2280** | Proved strong explanatory variance gain over baseline. |
| **Median Absolute Error** | **4.7770 min** | **50% of predictions are off by less than 4.8 minutes**. |
| **90th Percentile Error** | **18.4290 min** | 90% of predictions are off by less than 18.4 minutes. |

---

## 🛡️ Empirical Prediction Intervals (Validation Calibrated)

To prevent data leakage, residual quantiles ($e = |y_{\text{val}} - \hat{y}_{\text{val}}|$) were calibrated strictly on the Validation set:

| Target Confidence Level | Actual Test Coverage (%) | Coverage Error (%) | Mean Interval Width |
| :--- | :--- | :--- | :--- |
| **80% Target** | **79.48%** | **0.52%** | 19.82 min |
| **90% Target** | **89.31%** | **0.69%** | 28.49 min |
| **95% Target** | **94.42%** | **0.58%** | 38.00 min |

### Department Coverage Breakdown (at 90% Target)
- **General Medicine**: **96.38%**
- **Pediatrics**: **90.96%**
- **Orthopedics**: **85.50%**
- **Cardiology**: **63.92%** *(Explicitly flagged due to high consult variability $\sigma=18.0$ min & emergency preemption spikes)*

---

## 🔍 Explainability & Production Drift Monitoring

1. **Global Feature Importance**:
   - `hour_of_day` (0.2562)
   - `avg_waiting_time_last_30min` (0.1820)
   - `recent_throughput_past_hour` (0.0769)
   - `day_of_week` (0.0705)
   - `traffic_momentum` (0.0593)
2. **Production Monitoring**: Evaluates Population Stability Index (PSI) to detect feature distribution shifts (`WARNING` if $\text{PSI} > 0.10$, `CRITICAL` if $\text{PSI} > 0.25$).

---

## 🚀 Running Locally

### 1. Backend Server Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run automated test suite (34 test cases)
python -m pytest -v -s tests/

# Run latency benchmark
python -m tests.benchmark_prediction

# Start FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit API docs at: `http://localhost:8000/docs`

### 2. Frontend Dashboard Setup
```bash
cd frontend
npm install
npm run dev
```

Visit Web UI at: `http://localhost:5173`

---

## 🐳 Docker Deployment

To launch the complete application using Docker Compose:

```bash
docker compose up --build
```

- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:5173`

---

## 🔌 Core API Endpoints

- `GET /health`: Returns system status, loaded model name, and version.
- `POST /predict`: Accepts patient attributes, returns point estimate, likely range, confidence badge, and human-readable explanation.
- `GET /queue-status`: Returns department crowd status and active doctor capacity.

---

## 🧪 Verification & Testing
```powershell
python -m pytest -v -s tests/
# ====================== 34 passed in 4.74s ======================
```
