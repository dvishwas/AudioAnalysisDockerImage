#!/bin/bash

# Build script for audio analysis Docker image
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Audio Analysis Docker Image${NC}"
echo "========================================"

# Check if HF_TOKEN is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: HuggingFace token required${NC}"
    echo "Usage: $0 <HF_TOKEN>"
    echo ""
    echo "Steps to get token:"
    echo "1. Visit https://hf.co/settings/tokens"
    echo "2. Create a token with read access"
    echo "3. Accept terms at https://hf.co/pyannote/speaker-diarization-3.1"
    echo ""
    echo "Example: $0 hf_xxxxxxxxxxxxxxxxxxxx"
    exit 1
fi

HF_TOKEN="$1"
IMAGE_NAME="audio-analysis-template"
TAG="latest"

echo -e "${YELLOW}Building with HuggingFace authentication...${NC}"
echo "This will download and cache models during build time."
echo ""

# Build the Docker image
docker build \
    --build-arg HF_TOKEN_BUILD="$HF_TOKEN" \
    --platform linux/amd64 \
    -t "$IMAGE_NAME:$TAG" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo ""
    echo "To run the container:"
    echo "  docker run --gpus all -p 8000:8000 $IMAGE_NAME:$TAG"
    echo ""
    echo "To test the API:"
    echo "  curl http://localhost:8000/health"
    echo ""
    echo "Models are now cached and no runtime authentication is needed!"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
