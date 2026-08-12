# 📊 MLOps Monitoring & Statistical Drift Guide

## Population Stability Index (PSI) Thresholds

Production feature and target distributions are evaluated against baseline training distributions using PSI:

- $\text{PSI} \le 0.10$: **NORMAL** (Healthy distribution match)
- $0.10 < \text{PSI} \le 0.25$: **WARNING** (Moderate feature drift)
- $\text{PSI} > 0.25$: **CRITICAL** (Significant drift requiring re-calibration)

Monitoring module: `models/monitoring.py`
