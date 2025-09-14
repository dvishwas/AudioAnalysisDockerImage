# Audio Analysis Template for RunPod

Combined speaker diarization and recognition using pyannote + SpeechBrain.

## Docker Image
```
dvishwas/audio-analysis-template:latest
```

## RunPod Configuration

**Container Image**: `dvishwas/audio-analysis-template:latest`
**Container Disk**: 20GB minimum
**Expose HTTP Ports**: `8000`
**Docker Command**: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`

## API Endpoints

- `GET /health` - Health check
- `POST /diarize` - Speaker diarization (who spoke when)
- `POST /embedding` - Extract speaker embedding
- `POST /compare` - Compare two speakers

## Usage Examples

```bash
# Health check
curl https://your-pod-id-8000.proxy.runpod.net/health

# Diarization
curl -X POST -F "audio=@audio.wav" https://your-pod-id-8000.proxy.runpod.net/diarize

# Speaker embedding
curl -X POST -F "audio=@speaker.wav" https://your-pod-id-8000.proxy.runpod.net/embedding

# Compare speakers
curl -X POST -F "audio1=@speaker1.wav" -F "audio2=@speaker2.wav" https://your-pod-id-8000.proxy.runpod.net/compare
```

## Features

✅ No HuggingFace token required (models pre-loaded)
✅ GPU acceleration ready
✅ Supports WAV, MP3, FLAC formats
✅ Automatic audio resampling
✅ Combined diarization + recognition in one container

## Models Included

- **pyannote/speaker-diarization-3.1** - Speaker diarization
- **speechbrain/spkrec-ecapa-voxceleb** - Speaker recognition
