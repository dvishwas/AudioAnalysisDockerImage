#!/usr/bin/env python3
import os
import platform
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition
from huggingface_hub import login

print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")

# Use build-time token to download models
token = os.environ.get("HF_TOKEN_BUILD")
if not token:
    raise ValueError("HF_TOKEN_BUILD environment variable is required for model download")

print("Logging into HuggingFace...")
login(token=token)

# Download and cache pyannote diarization model
print("Downloading pyannote speaker diarization model...")
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    print("✓ Pyannote model downloaded and cached successfully!")
except Exception as e:
    print(f"✗ Failed to download pyannote model: {e}")
    raise

# Download and cache SpeechBrain speaker recognition model
print("Downloading SpeechBrain speaker recognition model...")
try:
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_speechbrain"
    )
    print("✓ SpeechBrain model downloaded and cached successfully!")
except Exception as e:
    print(f"✗ Failed to download SpeechBrain model: {e}")
    raise

print("✓ All models downloaded and cached successfully!")
print("Models are now available offline without authentication.")
