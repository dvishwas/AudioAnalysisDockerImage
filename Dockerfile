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

# Download models during build (requires HF_TOKEN_BUILD build arg)
ARG HF_TOKEN_BUILD
RUN if [ -z "$HF_TOKEN_BUILD" ]; then \
        echo "ERROR: HF_TOKEN_BUILD build argument is required"; \
        echo "Build with: docker build --build-arg HF_TOKEN_BUILD=your_token ."; \
        exit 1; \
    fi

# Download and cache models
RUN export HF_TOKEN_BUILD="$HF_TOKEN_BUILD" && \
    python download_models.py && \
    echo "Models cached successfully"

# Verify models are cached
RUN python -c "import os; print('Cache contents:'); os.system('find /root/.cache -name \"*.bin\" -o -name \"*.pt\" -o -name \"*.pth\" | head -10')"

# Copy app code
COPY app /app

# Clean up download script and token
RUN rm -f download_models.py
ENV HF_TOKEN_BUILD=""

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
