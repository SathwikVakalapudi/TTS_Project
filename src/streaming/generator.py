"""
Async streaming generator for telephony-grade TTS output.

Architecture
------------
  [Worker Thread]  model.generate_full()   — blocking, GPU-bound
        ↓  complete float32 audio array at 44100 Hz
  [Event Loop]  normalize → resample 44100→8kHz → G.711 μ-law encode (all at once)
        ↓  160-byte chunks (20 ms @ 8 kHz)
  [Client]  continuous byte stream (Twilio / SIP / IVR compatible)

Why full-audio processing instead of per-chunk:
  - Per-chunk resampling creates edge artifacts at every boundary (resample_poly
    applies a polyphase filter that needs context samples at both ends).
  - Per-chunk normalization creates sudden level jumps between chunks.
  Both sound like crackling/noise. Processing the full audio once eliminates both.

Trade-off: TTFA = full generation time (no early bytes). For a base model on GPU
this is typically 3-8 s. Fine-tuning or a streaming codec would reduce this.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np

from src.audio.processor import AudioProcessor, float32_to_ulaw_bytes, normalize
from src.models.indic_tts_model import IndicTTSModel
from src.utils.helpers import Timer

_SENTINEL = object()


@dataclass
class StreamMetrics:
    ttfa_s: Optional[float] = None
    total_s: Optional[float] = None
    audio_duration_s: float = 0.0
    chunks_yielded: int = 0

    @property
    def rtf(self) -> Optional[float]:
        if self.total_s and self.audio_duration_s > 0:
            return round(self.total_s / self.audio_duration_s, 4)
        return None

    def as_dict(self) -> dict:
        return {
            "ttfa_s": round(self.ttfa_s, 3) if self.ttfa_s is not None else None,
            "total_s": round(self.total_s, 3) if self.total_s is not None else None,
            "audio_duration_s": round(self.audio_duration_s, 3),
            "chunks_yielded": self.chunks_yielded,
            "rtf": self.rtf,
        }


async def generate_mulaw_stream(
    model: IndicTTSModel,
    processor: AudioProcessor,
    text: str,
    description: Optional[str] = None,
    metrics: Optional[StreamMetrics] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator — yields exactly 160-byte G.711 μ-law chunks (20 ms @ 8 kHz).

    Generates full audio in a worker thread, processes it once (normalize +
    resample + μ-law encode), then streams the encoded bytes as 160-byte chunks.
    This guarantees clean audio with no resampling edge artifacts or level jumps.
    """
    if metrics is None:
        metrics = StreamMetrics()

    loop = asyncio.get_running_loop()
    timer = Timer().start()

    # ------------------------------------------------------------------
    # Step 1 — Generate full audio in thread pool (non-blocking)
    # ------------------------------------------------------------------
    audio_44k = await loop.run_in_executor(
        None,
        lambda: model.generate_full(text, description),
    )

    if audio_44k is None or len(audio_44k) == 0:
        return

    metrics.ttfa_s = timer.elapsed()   # TTFA = time until first byte is ready

    # ------------------------------------------------------------------
    # Step 2 — Process the full audio once (no per-chunk edge artifacts)
    # ------------------------------------------------------------------
    audio_norm = normalize(audio_44k)                            # peak normalize
    audio_8k   = processor.resample(audio_norm, model.sampling_rate)  # → 8 kHz

    # ------------------------------------------------------------------
    # Step 3 — Stream 160-byte μ-law chunks
    # ------------------------------------------------------------------
    CHUNK = processor.chunk_samples   # 160 samples = 20 ms @ 8 kHz
    n = len(audio_8k)

    for start in range(0, n, CHUNK):
        chunk = audio_8k[start : start + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))

        yield float32_to_ulaw_bytes(chunk)

        metrics.chunks_yielded += 1
        metrics.audio_duration_s += CHUNK / processor.target_sr

    metrics.total_s = timer.elapsed()
