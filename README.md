# Audio Analysis Template - Combined Pyannote + SpeechBrain

Docker image combining pyannote speaker diarization and SpeechBrain speaker recognition in a unified FastAPI application.

## 🚀 Quick Start

### Docker Hub
```bash
docker pull dvishwas/audio-analysis-template:latest
docker run -p 8000:8000 dvishwas/audio-analysis-template:latest
```

### RunPod Deployment
**Container Image**: `dvishwas/audio-analysis-template:latest`  
**Container Disk**: 20GB minimum  
**Expose HTTP Ports**: `8000`  
**Docker Command**: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for both models |
| `/diarize` | POST | Speaker diarization (who spoke when) |
| `/embedding` | POST | Extract speaker embedding vector |
| `/compare` | POST | Compare two speakers for similarity |

## 🔧 Usage Examples

```bash
# Health check
curl https://your-pod-id-8000.proxy.runpod.net/health

# Speaker diarization
curl -X POST -F "audio=@audio.wav" \
  https://your-pod-id-8000.proxy.runpod.net/diarize

# Extract speaker embedding
curl -X POST -F "audio=@speaker.wav" \
  https://your-pod-id-8000.proxy.runpod.net/embedding

# Compare two speakers
curl -X POST -F "audio1=@speaker1.wav" -F "audio2=@speaker2.wav" \
  https://your-pod-id-8000.proxy.runpod.net/compare
```

## ✨ Features

- ✅ **No Authentication Required** - Models pre-loaded during build
- ✅ **GPU Acceleration Ready** - CUDA 12.1.1 support
- ✅ **Multiple Audio Formats** - WAV, MP3, FLAC support
- ✅ **Automatic Resampling** - Handles different sample rates
- ✅ **Combined Analysis** - Diarization + recognition in one container

## 🏗️ Build Locally

```bash
git clone <this-repo>
cd audio-analysis-template
docker build --build-arg HF_TOKEN_BUILD=your_hf_token -t audio-analysis .
```

## 📦 Models Included

- **Pyannote**: `pyannote/speaker-diarization-3.1`
- **SpeechBrain**: `speechbrain/spkrec-ecapa-voxceleb`

## 🔧 Technical Details

- **Base**: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
- **Python**: 3.10
- **PyTorch**: 2.2.0
- **NumPy**: 1.26.4 (pinned for compatibility)

## 📄 License

MIT License - see LICENSE file for details.
