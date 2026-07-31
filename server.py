"""
ajnano-voice — VoxCPM2 TTS Server
OpenAI-compatible /v1/audio/speech endpoint

DGX (ARM64): set TORCHDYNAMO_DISABLE=1 before running
x86_64: Triton JIT works natively, no env var needed
"""
import os as _os
if _os.getenv("TORCHDYNAMO_DISABLE"):
    import torch
    torch._dynamo.config.disable = True

import base64
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

MODEL = None
CONFIG = {
    "host": os.getenv("VOICE_HOST", "0.0.0.0"),
    "port": int(os.getenv("VOICE_PORT", "8808")),
    "model": os.getenv("VOICE_MODEL", "openbmb/VoxCPM2"),
    "device": os.getenv("VOICE_DEVICE", "cuda"),
    "default_preset": os.getenv("VOICE_DEFAULT_PRESET", "default"),
}

# ── Request models ──

class SpeechRequest(BaseModel):
    input: str
    voice: str = "default"
    speed: float = 1.0
    response_format: str = "wav"

class CloneRequest(BaseModel):
    input: str
    reference_audio: str  # base64 WAV
    reference_text: str | None = None  # for ultimate cloning

# ── Lifecycle ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print(f"[ajnano-voice] Loading {CONFIG['model']} on {CONFIG['device']}...")
    from voxcpm import VoxCPM
    MODEL = VoxCPM.from_pretrained(CONFIG["model"], load_denoiser=False)
    print(f"[ajnano-voice] Ready — sample_rate={MODEL.tts_model.sample_rate}")
    yield
    MODEL = None

app = FastAPI(title="ajnano-voice", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Helpers ──

def wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()

# ── Routes ──

@app.get("/health")
async def health():
    import torch
    return {
        "status": "ok",
        "model": CONFIG["model"],
        "device": CONFIG["device"],
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1) if torch.cuda.is_available() else None,
    }

@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(503, "Model not loaded yet")
    t0 = time.time()
    try:
        wav = MODEL.generate(
            text=req.input,
            cfg_value=2.0,
            inference_timesteps=10,
            seed=42,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
    elapsed = time.time() - t0
    duration = len(wav) / MODEL.tts_model.sample_rate
    audio = wav_bytes(wav, MODEL.tts_model.sample_rate)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{elapsed:.2f}",
            "X-RTF": f"{elapsed / duration:.3f}" if duration > 0 else "0",
        },
    )

@app.post("/v1/audio/speech/clone")
async def clone(req: CloneRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(503, "Model not loaded yet")
    # Decode base64 reference audio to temp file
    try:
        audio_data = base64.b64decode(req.reference_audio)
    except Exception:
        raise HTTPException(400, "Invalid base64 in reference_audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        ref_path = f.name

    try:
        kwargs = dict(
            text=req.input,
            reference_wav_path=ref_path,
            cfg_value=2.0,
            inference_timesteps=10,
            seed=42,
        )
        if req.reference_text:
            kwargs["prompt_wav_path"] = ref_path
            kwargs["prompt_text"] = req.reference_text

        wav = MODEL.generate(**kwargs)
    except Exception as e:
        raise HTTPException(500, f"Cloning failed: {e}")
    finally:
        os.unlink(ref_path)

    audio = wav_bytes(wav, MODEL.tts_model.sample_rate)
    return Response(content=audio, media_type="audio/wav")

# ── Static ──

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "web" / "index.html")

# ── Main ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
