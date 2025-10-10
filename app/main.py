from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
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
    """Load both models from cached files"""
    global diarization_pipeline, verification_model
    
    try:
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load pyannote diarization model from cache (no auth needed)
        logger.info("Loading pyannote diarization model from cache...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=False  # Use cached model, no auth required
        )
        if torch.cuda.is_available():
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        logger.info("Pyannote model loaded successfully")
        
        # Load SpeechBrain verification model from cache
        logger.info("Loading SpeechBrain verification model from cache...")
        verification_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_speechbrain",
            run_opts={"device": device}
        )
        logger.info("SpeechBrain model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Models may not be properly cached. Rebuild with HF_TOKEN_BUILD.")
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
async def diarize_audio(
    file: UploadFile = File(...),
    min_speakers: int = None,
    max_speakers: int = None,
    num_speakers: int = None,
    min_duration: float = None
):
    """Speaker diarization - who spoke when"""
    import time
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            # Log file info
            waveform, sr = torchaudio.load(tmp_file.name)
            duration = waveform.shape[1] / sr
            logger.info(f"Processing audio: {file.filename}, duration: {duration:.2f}s, sample_rate: {sr}")
            
            # Build kwargs for pipeline
            kwargs = {}
            if num_speakers is not None:
                kwargs['num_speakers'] = num_speakers
            else:
                if min_speakers is not None:
                    kwargs['min_speakers'] = min_speakers
                if max_speakers is not None:
                    kwargs['max_speakers'] = max_speakers
            if min_duration is not None:
                kwargs['min_duration'] = min_duration
            
            # Run diarization
            diarization = diarization_pipeline(tmp_file.name, **kwargs)
            
            # Convert to list format and count segments
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker
                })
            
            processing_time = time.time() - start_time
            logger.info(f"Diarization complete: {len(results)} segments found in {processing_time:.2f}s")
            
            return {
                "diarization_results": results,
                "processing_time": processing_time,
                "audio_duration": duration,
                "segments_count": len(results)
            }
            
        finally:
            os.unlink(tmp_file.name)

@app.post("/embedding")
async def get_embedding(file: UploadFile = File(...), normalize: bool = True):
    """Extract speaker embedding from audio"""
    import torch.nn.functional as F
    import time
    
    start_time = time.time()
    logger.info(f"[/embedding] Processing file: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            waveform, sr = torchaudio.load(tmp_file.name)
            duration = waveform.shape[1] / sr
            logger.info(f"[/embedding] Audio loaded: duration={duration:.2f}s, sr={sr}")
            
            # Resample if needed
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                logger.info(f"[/embedding] Resampled to 16kHz")
            
            # Move waveform to GPU if available
            device = next(verification_model.mods.parameters()).device
            waveform = waveform.to(device)
            logger.info(f"[/embedding] Processing on device: {device}")
            
            embedding = verification_model.encode_batch(waveform).squeeze()
            
            # Normalize if requested (recommended for cosine similarity)
            if normalize:
                embedding = F.normalize(embedding, p=2, dim=0)
            
            processing_time = time.time() - start_time
            logger.info(f"[/embedding] Complete: {processing_time:.2f}s, embedding_shape={embedding.shape}")
            
            return {"embedding": embedding.cpu().tolist(), "normalized": normalize}
        finally:
            os.unlink(tmp_file.name)

@app.post("/compare")
async def compare_speakers(file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: float = 0.25):
    """Compare two audio files for speaker similarity"""
    import time
    
    start_time = time.time()
    logger.info(f"[/compare] Processing files: {file1.filename} vs {file2.filename}")
    
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
            logger.info(f"[/compare] Audio loaded: file1={waveform1.shape[1]/sr1:.2f}s, file2={waveform2.shape[1]/sr2:.2f}s")
            
            if sr1 != 16000:
                waveform1 = torchaudio.transforms.Resample(sr1, 16000)(waveform1)
            if sr2 != 16000:
                waveform2 = torchaudio.transforms.Resample(sr2, 16000)(waveform2)
            
            # Move waveforms to GPU if available
            device = next(verification_model.mods.parameters()).device
            waveform1 = waveform1.to(device)
            waveform2 = waveform2.to(device)
            logger.info(f"[/compare] Processing on device: {device}")
                
            score, prediction = verification_model.verify_batch(waveform1, waveform2)
            processing_time = time.time() - start_time
            logger.info(f"[/compare] Complete: {processing_time:.2f}s, score={float(score):.4f}, same_speaker={bool(prediction)}")
            
            return {
                "similarity_score": float(score),
                "same_speaker": bool(prediction),
                "threshold": threshold
            }
        finally:
            os.unlink(tmp_file1.name)
            os.unlink(tmp_file2.name)

@app.post("/compare_embedding")
async def compare_audio_to_embedding(
    audio: UploadFile = File(...),
    embedding: UploadFile = File(...),
    threshold: float = Form(0.25),
    segment_start_time: Optional[float] = Form(None),
    segment_end_time: Optional[float] = Form(None)
):
    """Compare audio file (or segment) against a pre-computed embedding from pkl file
    
    Args:
        audio: Audio file to verify
        embedding: Pickle file containing stored embedding
        threshold: Similarity threshold for same_speaker decision (default: 0.25)
        segment_start_time: Start time of segment in seconds (optional)
        segment_end_time: End time of segment in seconds (optional)
    """
    import torch.nn.functional as F
    import pickle
    import time
    
    start_time = time.time()
    
    # CRITICAL: Log received parameters FIRST
    logger.info(f"[/compare_embedding] ========== RECEIVED PARAMETERS ==========")
    logger.info(f"[/compare_embedding] audio.filename: {audio.filename}")
    logger.info(f"[/compare_embedding] embedding.filename: {embedding.filename}")
    logger.info(f"[/compare_embedding] threshold: {threshold}")
    logger.info(f"[/compare_embedding] segment_start_time: {segment_start_time} (is None: {segment_start_time is None})")
    logger.info(f"[/compare_embedding] segment_end_time: {segment_end_time} (is None: {segment_end_time is None})")
    logger.info(f"[/compare_embedding] WILL EXTRACT SEGMENT: {segment_start_time is not None and segment_end_time is not None}")
    logger.info(f"[/compare_embedding] ==========================================")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as emb_file:
        
        # Save audio file
        audio_content = await audio.read()
        audio_file.write(audio_content)
        audio_file.flush()
        
        # Save and load embedding pkl file
        emb_content = await embedding.read()
        emb_file.write(emb_content)
        emb_file.flush()
        
        try:
            # Load stored embedding from pkl
            with open(emb_file.name, 'rb') as f:
                stored_embedding_data = pickle.load(f)
            logger.info(f"[/compare_embedding] Loaded embedding from pkl: type={type(stored_embedding_data)}")
            
            # Convert to tensor (handle list or numpy array)
            if isinstance(stored_embedding_data, list):
                stored_embedding = torch.tensor(stored_embedding_data, dtype=torch.float32)
            else:
                stored_embedding = torch.tensor(stored_embedding_data, dtype=torch.float32)
            
            # Load and process audio
            waveform, sr = torchaudio.load(audio_file.name)
            duration = waveform.shape[1] / sr
            logger.info(f"[/compare_embedding] Audio loaded: duration={duration:.2f}s, sr={sr}")
            
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                logger.info(f"[/compare_embedding] Resampled to 16kHz")
            
            # Extract segment if times provided
            if segment_start_time is not None and segment_end_time is not None:
                start_sample = int(segment_start_time * 16000)
                end_sample = int(segment_end_time * 16000)
                
                logger.info(f"[/compare_embedding] Extracting segment: {segment_start_time:.2f}s to {segment_end_time:.2f}s")
                
                # Validate segment bounds
                if start_sample < 0 or end_sample > waveform.shape[1]:
                    logger.error(f"[/compare_embedding] Invalid segment bounds")
                    return {
                        "error": f"Invalid segment: segment_start_time={segment_start_time}, segment_end_time={segment_end_time}, audio_duration={waveform.shape[1]/16000:.2f}s"
                    }
                if start_sample >= end_sample:
                    logger.error(f"[/compare_embedding] start_time >= end_time")
                    return {"error": "segment_start_time must be less than segment_end_time"}
                
                waveform = waveform[:, start_sample:end_sample]
                logger.info(f"[/compare_embedding] Segment extracted: duration={(end_sample-start_sample)/16000:.2f}s")
            
            # Get device and move all tensors to GPU
            device = next(verification_model.mods.parameters()).device
            waveform = waveform.to(device)
            stored_embedding = stored_embedding.to(device)
            logger.info(f"[/compare_embedding] Processing on device: {device}")
            
            # Extract embedding from audio
            audio_embedding = verification_model.encode_batch(waveform).squeeze()
            logger.info(f"[/compare_embedding] Audio embedding extracted: shape={audio_embedding.shape}")
            
            # Normalize both embeddings for cosine similarity
            audio_embedding = F.normalize(audio_embedding, p=2, dim=0)
            stored_embedding = F.normalize(stored_embedding, p=2, dim=0)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(audio_embedding.unsqueeze(0), stored_embedding.unsqueeze(0))
            score = float(similarity)
            same_speaker = score > threshold
            
            processing_time = time.time() - start_time
            logger.info(f"[/compare_embedding] Complete: {processing_time:.2f}s, score={score:.4f}, same_speaker={same_speaker}, threshold={threshold}")
            
            return {
                "similarity_score": score,
                "same_speaker": same_speaker,
                "threshold": threshold
            }
        except Exception as e:
            logger.error(f"[/compare_embedding] Error: {str(e)}", exc_info=True)
            raise
        finally:
            os.unlink(audio_file.name)
            os.unlink(emb_file.name)
