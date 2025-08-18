# ------------ Stage 1: Builder with dependencies ------------
    FROM --platform=linux/amd64 python:3.9-slim-bullseye AS builder

    # Set environment variables
    ENV DEBIAN_FRONTEND=noninteractive \
        PIP_NO_CACHE_DIR=off \
        PIP_DISABLE_PIP_VERSION_CHECK=on \
        PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # Install build dependencies for PyMuPDF, spaCy
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libharfbuzz-dev \
        libfreetype6-dev \
        libfontconfig1-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy and install Python dependencies
    COPY requirements.txt .
    RUN pip install --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt
    
    # Pre-download models (to avoid redownload during runtime)
    RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('msmarco-MiniLM-L-6-v3')"
    RUN python -m spacy download en_core_web_sm
    
    # ------------ Stage 2: Lightweight runtime image ------------
    FROM --platform=linux/amd64 python:3.9-slim-bullseye
    
    ENV PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # Install runtime system dependencies only
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libharfbuzz0b \
        libfreetype6 \
        libfontconfig1 \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy dependencies and models from builder
    COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
    COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
    COPY --from=builder /usr/local/lib/python3.9/site-packages/en_core_web_sm /usr/local/lib/python3.9/site-packages/en_core_web_sm

    # Copy app code
    COPY . /app/
    
    # Ensure input/output directories exist
    RUN mkdir -p /app/input /app/output
    
    # Run your main script by default
    CMD ["python", "main.py"]
    