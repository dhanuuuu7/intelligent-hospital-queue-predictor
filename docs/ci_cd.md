# ⚙️ GitHub Actions CI/CD Architecture

## Workflows Overview

1. **`ci.yml`**: Python backend unit, integration, and E2E test execution on `push` and `pull_request`.
2. **`frontend-ci.yml`**: Node 20 dependency installation and Vite production build validation.
3. **`docker.yml`**: Multi-container Docker Buildx validation and GitHub Container Registry (GHCR) image publishing (`ghcr.io/<owner>/backend`, `ghcr.io/<owner>/frontend`).
4. **`deployment.yml`**: Cloud container deployment pipeline with automated secret checks.
