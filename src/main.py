"""
FastAPI TTS server — Indic-Parler-TTS (Telugu, base model, no fine-tuning).

Endpoints
---------
POST /synthesize/stream  True async stream — 8 kHz μ-law PCM (production telephony)
POST /synthesize         Legacy sync stream — 8 kHz μ-law PCM
POST /synthesize/full    Complete 8 kHz PCM-16 WAV
POST /synthesize/hq      Complete 44100 Hz PCM-16 WAV (best quality)
GET  /health             Readiness + metrics
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.audio.processor import AudioProcessor, normalize
from src.config import config
from src.models.indic_tts_model import IndicTTSModel
from src.streaming.generator import StreamMetrics, generate_mulaw_stream
from src.utils.helpers import InferenceMetrics, Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialised on startup)
# ---------------------------------------------------------------------------
model = IndicTTSModel(config)
processor = AudioProcessor(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Indic-Parler-TTS model …")
    model.load()
    logger.info(
        f"Model ready — native sample rate: {model.sampling_rate} Hz  |  device: {model.device}"
    )
    yield


app = FastAPI(title="Telugu TTS", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Telugu text to synthesise")
    description: str | None = Field(
        None,
        description="Optional English speaker description (overrides default)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WAV_CHUNK_SIZE = 4096  # bytes per HTTP chunk for WAV streaming


def _iter_bytes(data: bytes, chunk_size: int = _WAV_CHUNK_SIZE) -> Generator[bytes, None, None]:
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


async def _stream_mulaw(
    text: str, description: str | None
) -> AsyncGenerator[bytes, None]:
    """
    Generator that drives the model in a background thread and yields
    20 ms μ-law chunks as they arrive.
    """
    metrics = InferenceMetrics(text=text)
    wall = Timer().start()
    first_chunk = True
    sample_count = 0

    for raw_chunk in model.generate_streaming(text, description):
        if first_chunk:
            metrics.ttfa_s = wall.elapsed()
            first_chunk = False
            logger.info(f"TTFA: {metrics.ttfa_s:.3f}s")

        resampled = processor.normalize_and_resample(raw_chunk, model.sampling_rate)
        sample_count += len(resampled)

        for telephony_chunk in processor.to_telephony_chunks(resampled):
            yield telephony_chunk

    metrics.total_inference_s = wall.elapsed()
    metrics.audio_duration_s = sample_count / config.audio.output_sample_rate
    metrics.compute_rtf()
    logger.info(f"Metrics: {metrics.as_dict()}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": config.model.model_id, "device": model.device}


@app.post(
    "/synthesize",
    summary="Stream 8 kHz μ-law PCM (telephony)",
    responses={200: {"content": {"audio/basic": {}}}},
)
async def synthesize_stream(req: SynthesizeRequest):
    """
    Returns a chunked stream of raw 8 kHz μ-law PCM bytes.
    Each chunk is 160 bytes = 20 ms of audio (standard G.711 packet size).

    Content-Type: audio/basic  (RFC 2046 — 8 kHz mono μ-law, no header)
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    return StreamingResponse(
        _stream_mulaw(req.text, req.description),
        media_type="audio/basic",
        headers={"X-Sample-Rate": "8000", "X-Encoding": "ulaw"},
    )


@app.post(
    "/synthesize/full",
    summary="Return complete 8 kHz PCM-16 WAV",
    responses={200: {"content": {"audio/wav": {}}}},
)
async def synthesize_full(req: SynthesizeRequest):
    """
    Runs full inference and returns a complete WAV file downsampled to 8 kHz.
    Suitable for offline evaluation and audio quality measurement.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    wall = Timer().start()
    audio = model.generate_full(req.text, req.description)
    elapsed = wall.elapsed()

    if audio.size == 0:
        raise HTTPException(status_code=500, detail="Model returned empty audio")

    wav_bytes = processor.to_wav_8k_bytes(audio, model.sampling_rate)
    audio_duration = len(processor.resample(audio, model.sampling_rate)) / config.audio.output_sample_rate

    logger.info(
        f"Full synthesis — duration: {audio_duration:.2f}s  "
        f"inference: {elapsed:.2f}s  RTF: {elapsed/audio_duration:.4f}"
    )

    return StreamingResponse(
        _iter_bytes(wav_bytes),
        media_type="audio/wav",
        headers={
            "X-RTF": f"{elapsed/audio_duration:.4f}",
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Inference-Time": f"{elapsed:.3f}",
            "X-Chunk-Size": str(_WAV_CHUNK_SIZE),
        },
    )


@app.post(
    "/synthesize/hq",
    summary="Return full-quality 44100 Hz WAV (best for speakers/headphones)",
    responses={200: {"content": {"audio/wav": {}}}},
)
async def synthesize_hq(req: SynthesizeRequest):
    """
    Returns audio at the model's native 44100 Hz sample rate — same quality as
    quick_test.py. Use this for evaluation; use /synthesize/full for telephony.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    wall = Timer().start()
    audio = model.generate_full(req.text, req.description)
    elapsed = wall.elapsed()

    if audio.size == 0:
        raise HTTPException(status_code=500, detail="Model returned empty audio")

    from src.audio.processor import normalize
    audio = normalize(audio)
    audio_duration = len(audio) / model.sampling_rate

    wav_bytes = processor.to_wav_bytes(audio, model.sampling_rate)

    logger.info(
        f"HQ synthesis — duration: {audio_duration:.2f}s  "
        f"inference: {elapsed:.2f}s  RTF: {elapsed/audio_duration:.4f}"
    )

    return StreamingResponse(
        _iter_bytes(wav_bytes),
        media_type="audio/wav",
        headers={
            "X-RTF": f"{elapsed/audio_duration:.4f}",
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Inference-Time": f"{elapsed:.3f}",
            "X-Sample-Rate": str(model.sampling_rate),
            "X-Chunk-Size": str(_WAV_CHUNK_SIZE),
        },
    )


@app.post(
    "/synthesize/stream",
    summary="True async stream — 8 kHz μ-law PCM (production telephony)",
    responses={200: {"content": {"audio/basic": {}}}},
)
async def synthesize_stream_async(req: SynthesizeRequest):
    """
    Production-grade streaming endpoint.

    - Runs TTS in a worker thread; event loop is never blocked.
    - Yields 160-byte G.711 μ-law chunks (20 ms) as they are generated.
    - Suitable for Twilio, SIP, IVR, and real-time AI voice agents.
    - TTFA and RTF are logged server-side on completion.

    Content-Type : audio/basic  (RFC 2046 — 8 kHz mono μ-law, no header)
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    metrics = StreamMetrics()

    async def _generate():
        async for chunk in generate_mulaw_stream(
            model, processor, req.text, req.description, metrics
        ):
            yield chunk
        logger.info(
            f"[/synthesize/stream] TTFA={metrics.ttfa_s:.3f}s  "
            f"RTF={metrics.rtf}  duration={metrics.audio_duration_s:.2f}s  "
            f"chunks={metrics.chunks_yielded}"
        )

    return StreamingResponse(
        _generate(),
        media_type="audio/basic",
        headers={
            "X-Sample-Rate": "8000",
            "X-Encoding": "ulaw",
            "X-Chunk-Size": "160",
            "X-Chunk-Duration-Ms": "20",
        },
    )
