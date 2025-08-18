# Dockerfile for PDF Heading Extractor (Building for AMD64, to run on ARM64 with emulation)
# Use a slim Python base image for a smaller footprint, specifying AMD64 platform
FROM --platform=linux/amd64 python:3.9-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by PyMuPDF (fitz) and potentially PyTorch.
# libgl1 is for potential image processing/rendering.
# build-essential, pkg-config, libfontconfig1, libharfbuzz-gobject0, libfreetype6-dev, gcc
# are common build tools and font-related libraries often required by PDF processing libs.
# libgomp1 is often a runtime dependency for PyTorch on ARM64/slim images (good to keep for robustness).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    build-essential \
    pkg-config \
    libfontconfig1 \
    libharfbuzz-gobject0 \
    libfreetype6-dev \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Create a directory for the locally stored models
RUN mkdir -p /app/models

# Copy the manually downloaded all-MiniLM-L6-v2 model into the container.
# Make sure your 'all-MiniLM-L6-v2-local' local path matches where you saved the model.
COPY ./all-MiniLM-L6-v2-local /app/models/all-MiniLM-L6-v2

# Copy only the main.py script into the container (no more run_processor.sh)
COPY main.py .

# Install Python dependencies including those for semantic filtering.
# We include torch and transformers as they are dependencies for sentence-transformers
# when using PyTorch models like all-MiniLM-L6-v2.
# Note: Installing torch can significantly increase the image size.
RUN pip install --no-cache-dir --default-timeout=600 \
    PyMuPDF \
    packaging \
    scikit-learn \
    numpy \
    # Upgrade these to recent versions. Removing specific versions might pull latest compatible.
    sentence-transformers \
    torch \
    transformers \
    huggingface_hub

# Create input and output directories as expected by the application
RUN mkdir -p /app/input /app/output

# No need to chmod run_processor.sh anymore as it's not being executed directly

# ********************************************************************************
# CRITICAL CHANGE: Directly execute the Python script.
# This bypasses any shell script emulation issues.
CMD ["python", "main.py"]
# ********************************************************************************
