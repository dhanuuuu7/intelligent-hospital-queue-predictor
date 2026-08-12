# ☁️ Cloud Container Deployment Guide (GCP Cloud Run / AWS ECS)

## Architecture
The application is containerized and ready for serverless container execution on Google Cloud Run or AWS ECS/Fargate.

## Required Secrets for Automated Cloud Deployment
To enable automated deployment via `.github/workflows/deployment.yml`, configure the following GitHub Repository Secrets:

- `GCP_SA_KEY`: Base64 encoded Google Cloud Service Account key JSON.
- `GCP_PROJECT_ID`: Target GCP project identifier.

## Deployment Status Notice
> **Status**: Cloud deployment configuration and Docker workflows are complete. Live deployment execution is **PENDING CLOUD CREDENTIALS**.
