# 🔄 Model & System Rollback Strategy

## Automated Quality Gate Enforcement
Before any candidate model image is tagged for production deployment, `python -m models.model_quality_gate` must pass.

## Rollback Procedure
If production monitoring detects critical distribution drift ($\text{PSI} > 0.25$) or performance degradation:

1. Update `MODEL_PATH` in `.env` back to `models/artifacts/final_waiting_time_model.joblib` (Version 1.0.0).
2. Restart backend service via `docker compose restart backend`.
3. Verify readiness via `GET /ready`.
