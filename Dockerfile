# Reproducible environment for the GenAI / critical-thinking analysis pipeline.
# Base image is pinned to an exact Python version.
FROM python:3.12.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only what the pipeline needs
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/sample/ ./data/sample/

# Results are written here; mount a host folder to keep them
RUN mkdir -p /app/results
VOLUME ["/app/results"]

# Default: run the pipeline on the synthetic sample
CMD ["python", "src/main.py", "--config", "configs/config.yaml"]
