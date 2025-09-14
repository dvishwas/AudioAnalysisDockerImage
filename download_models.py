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
if token:
    login(token=token)

# Download and cache pyannote diarization model
print("Downloading pyannote speaker diarization model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token
)
print("Pyannote model downloaded successfully!")

# Download and cache SpeechBrain speaker recognition model
print("Downloading SpeechBrain speaker recognition model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_speechbrain"
)
print("SpeechBrain model downloaded successfully!")

print("All models downloaded and cached successfully for Linux platform!")
