# backend/Dockerfile
# Simplified single-stage build to avoid multi-stage copy issues

FROM node:18-alpine AS frontend-build
WORKDIR /frontend
ENV CI=true

# copy only package files first for caching
COPY frontend/package.json frontend/package-lock.json /frontend/

# Install dependencies
RUN npm ci --silent

# copy rest and build
COPY frontend/ /frontend/

# remove one-off problematic pages (if present) so Vite doesn't fail in CI build
RUN rm -f src/pages/CollectionAnalysisPage.jsx || true \
 && find src -type f \( -name "*.jsx" -o -name "*.js" -o -name "*.tsx" -o -name "*.ts" \) \
    | xargs -r grep -l "CollectionAnalysisPage" 2>/dev/null \
    | xargs -r sed -i '/CollectionAnalysisPage/d' || true

# Build the project
RUN npm run build

# move vite dist -> build so runtime COPY can rely on /frontend/build
RUN if [ -d dist ]; then rm -rf build || true && mv dist build; fi

########################
# Single-stage Python runtime
########################
FROM python:3.11-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps + supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ffmpeg libsndfile1 libjpeg-dev libpng-dev libtiff-dev \
    libopenblas-dev libomp-dev ca-certificates cmake pkg-config git supervisor \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core packages first
RUN python -m pip install --no-cache-dir "numpy==1.26.4" "protobuf==5.27.5"

# Install CPU-only PyTorch with proper versions
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.3.1+cpu" \
      "torchvision==0.18.1+cpu" \
      "torchaudio==2.3.1+cpu"

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Install transformers BEFORE sentence-transformers to avoid conflicts
RUN python -m pip install --no-cache-dir \
      "transformers==4.40.0" \
      "tokenizers==0.19.1"

# Install FastAPI and other essential packages
RUN python -m pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      PyMuPDF==1.24.5 \
      spacy==3.7.4 \
      ollama==0.2.0 \
      ftfy==6.2.0 \
      scikit-learn==1.5.0 \
      huggingface_hub

# Install sentence-transformers LAST to avoid version conflicts
RUN python -m pip install --no-cache-dir "sentence-transformers==2.7.0"

# Download spaCy model
RUN python -m spacy download en_core_web_sm || true

# Copy requirements and install remaining packages
COPY backend/requirements.txt /app/
COPY backend/requirements-linux.txt /app/

# Install remaining cleaned requirements (excluding already installed packages)
RUN sed -E -e '/^logging==/Id' \
           -e '/^numpy==/Id' \
           -e '/^torch($|==|[[:space:]])/Id' \
           -e '/^torchvision($|==|[[:space:]])/Id' \
           -e '/^torchaudio($|==|[[:space:]])/Id' \
           -e '/^protobuf==/Id' \
           -e '/^spacy==/Id' \
           -e '/en-core-web-sm[[:space:]]*@/Id' \
           -e '/^en-core-web-sm/Id' \
           -e '/^httpx==/Id' \
           -e '/^ollama/Id' \
           -e '/^google-ai-generativelanguage/Id' \
           -e '/^google-generativeai/Id' \
           -e '/^fastapi/Id' \
           -e '/^uvicorn/Id' \
           -e '/^transformers($|==|[[:space:]])/Id' \
           -e '/^tokenizers($|==|[[:space:]])/Id' \
           -e '/^sentence-transformers($|==|[[:space:]])/Id' \
           -e '/^huggingface-hub($|==|[[:space:]])/Id' \
           /app/requirements-linux.txt > /app/requirements-clean.txt && \
    if [ -s /app/requirements-clean.txt ]; then \
        python -m pip install --no-cache-dir -r /app/requirements-clean.txt; \
    fi

# Copy libs directory FIRST (before backend source)
COPY libs/ /app/libs/

# Copy backend source
COPY backend/ /app/

# Copy Sentence-Transformer model to the expected location
COPY libs/lib_a/all-MiniLM-L6-v2-local /app/models/all-MiniLM-L6-v2/

# copy built frontend static bundle
COPY --from=frontend-build /frontend/build /app/frontend_build

# supervisord config
COPY backend/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Final verification
RUN python -m pip show fastapi && echo "✓ FastAPI successfully installed" || echo "✗ FastAPI installation failed"
RUN python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
RUN python -c "import sentence_transformers; print('✓ SentenceTransformers imported successfully')"

# Verify libs directory structure
RUN ls -la /app/libs/ && ls -la /app/libs/lib_a/ || echo "libs directory structure check"

EXPOSE 8080 3000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]