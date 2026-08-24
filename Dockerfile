# Use python:3.10-slim as base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directories and set permissions for HF/Torch/Numba
RUN mkdir -p /tmp/cache /tmp/voices && chmod -R 777 /tmp
ENV HF_HOME=/tmp/cache/huggingface
ENV TORCH_HOME=/tmp/cache/torch
ENV NUMBA_CACHE_DIR=/tmp/cache/numba

# Copy application files
COPY . .
RUN chmod -R 777 /app

# Expose port
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV HOST=0.0.0.0

# Run Flask app
CMD ["python", "app.py"]
