# ==============================================================================
# DesertFox: Open-Source Facial OSINT & Identity Exposure Engine
# Multi-stage Docker build for production deployment
# ==============================================================================

FROM python:3.12-slim AS base

# System dependencies for OpenCV headless and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for vector store and model weights
RUN mkdir -p /app/data /app/desertfox/models/weights

# Download ONNX models (optional — comment out to use fallback heuristics)
# RUN python3 -m desertfox.models.download --model all

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/stats || exit 1

# Default: launch the Cyber-OSINT HUD web application
CMD ["python3", "-m", "desertfox.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
