FROM python:3.13-slim

WORKDIR /app

# Prevent python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source application code and artifacts
COPY simulator/ simulator/
COPY models/ models/
COPY app/ app/
COPY data/ data/

# Expose backend port
EXPOSE 8000

# Production non-reload execution
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
