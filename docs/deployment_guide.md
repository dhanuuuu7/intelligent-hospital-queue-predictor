# 🏥 Local & Production Deployment Guide

## 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run pytest test suite (37 test cases)
python -m pytest -v -s tests/

# Start API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 2. Docker Container Deployment
```bash
# Build multi-container backend & frontend
docker compose up --build
```

- Backend API: `http://localhost:8000`
- Frontend Dashboard: `http://localhost:5173`
- Healthcheck Endpoint: `http://localhost:8000/health`
- Readiness Endpoint: `http://localhost:8000/ready`
