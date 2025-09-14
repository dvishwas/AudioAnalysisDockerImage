FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (single PyTorch installation)
RUN pip install --only-binary=all --ignore-installed \
    numpy==1.26.4 \
    pyannote.audio pyannote.metrics \
    speechbrain==0.5.15 \
    fastapi uvicorn[standard] python-multipart \
    torchvision soundfile

# Copy download script
COPY download_models.py .

# Copy app code
COPY app /app

# Download models during build (requires HF_TOKEN_BUILD build arg)
ARG HF_TOKEN_BUILD
RUN if [ -n "$HF_TOKEN_BUILD" ]; then \
    export HF_TOKEN_BUILD="$HF_TOKEN_BUILD" && \
    python download_models.py; \
    fi

# Clean up download script
RUN rm -f download_models.py

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
