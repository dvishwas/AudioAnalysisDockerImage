from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition
import tempfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Analysis API", description="Combined speaker diarization and recognition")

# Global variables for models
diarization_pipeline = None
verification_model = None

def load_models():
    """Load both models"""
    global diarization_pipeline, verification_model
    
    try:
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load pyannote diarization model
        logger.info("Loading pyannote diarization model...")
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if torch.cuda.is_available():
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        logger.info("Pyannote model loaded successfully")
        
        # Load SpeechBrain verification model
        logger.info("Loading SpeechBrain verification model...")
        verification_model = SpeechBrain.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_speechbrain",
            run_opts={"device": device}
        )
        logger.info("SpeechBrain model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Load models on startup
load_models()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "diarization_model_loaded": diarization_pipeline is not None,
        "verification_model_loaded": verification_model is not None
    }

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    """Speaker diarization - who spoke when"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            # Run diarization
            diarization = diarization_pipeline(tmp_file.name)
            
            # Convert to list format
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker
                })
            
            return {"diarization_results": results}
            
        finally:
            os.unlink(tmp_file.name)

@app.post("/embedding")
async def get_embedding(file: UploadFile = File(...)):
    """Extract speaker embedding from audio"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            waveform, sr = torchaudio.load(tmp_file.name)
            # Resample if needed
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            embedding = verification_model.encode_batch(waveform).squeeze().tolist()
            return {"embedding": embedding}
        finally:
            os.unlink(tmp_file.name)

@app.post("/compare")
async def compare_speakers(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compare two audio files for speaker similarity"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file2:
        
        content1 = await file1.read()
        content2 = await file2.read()
        tmp_file1.write(content1)
        tmp_file2.write(content2)
        tmp_file1.flush()
        tmp_file2.flush()
        
        try:
            waveform1, sr1 = torchaudio.load(tmp_file1.name)
            waveform2, sr2 = torchaudio.load(tmp_file2.name)
            
            if sr1 != 16000:
                waveform1 = torchaudio.transforms.Resample(sr1, 16000)(waveform1)
            if sr2 != 16000:
                waveform2 = torchaudio.transforms.Resample(sr2, 16000)(waveform2)
                
            score, prediction = verification_model.verify_batch(waveform1, waveform2)
            return {"similarity_score": float(score), "same_speaker": bool(prediction)}
        finally:
            os.unlink(tmp_file1.name)
            os.unlink(tmp_file2.name)
