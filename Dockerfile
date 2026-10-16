FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/cache /tmp/voices && chmod -R 777 /tmp
ENV HF_HOME=/tmp/cache/huggingface
ENV TORCH_HOME=/tmp/cache/torch
ENV NUMBA_CACHE_DIR=/tmp/cache/numba

COPY . .
RUN chmod -R 777 /app

EXPOSE 7860

ENV PORT=7860
ENV HOST=0.0.0.0

CMD ["python", "app.py"]
