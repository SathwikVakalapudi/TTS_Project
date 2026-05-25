import io
import struct
from fractions import Fraction
from typing import Generator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.config import Config, config as default_config

# G.711 μ-law segment table (used for strict ITU-T compliance)
_SEG_END = np.array([0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF])
_BIAS = 0x84
_CLIP = 32635
_MU = 255


def _float32_to_int16(samples: np.ndarray) -> np.ndarray:
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767).astype(np.int16)


def _int16_to_ulaw(samples: np.ndarray) -> np.ndarray:
    """
    Vectorised ITU-T G.711 μ-law encoder.
    Input : int16 array
    Output: uint8 array (one byte per sample)
    """
    s = samples.astype(np.int32)
    sign = np.where(s < 0, 0x80, 0x00).astype(np.uint8)
    s = np.abs(s)
    s = np.minimum(s, _CLIP)
    s += _BIAS

    # Find exponent (highest set bit in positions 3-10)
    exp = np.zeros(len(s), dtype=np.uint8)
    for e in range(7, 0, -1):
        mask = s >= (1 << (e + 3))
        exp = np.where(mask & (exp == 0), e, exp)

    mantissa = ((s >> (exp + 3)) & 0x0F).astype(np.uint8)
    ulaw = (~(sign | (exp << 4) | mantissa)).astype(np.uint8)
    return ulaw


def float32_to_ulaw_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] samples to raw G.711 μ-law bytes."""
    return _int16_to_ulaw(_float32_to_int16(samples)).tobytes()


def normalize(samples: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak-normalize audio so the loudest sample hits target_peak.
    Avoids distortion while maximising perceived loudness.
    """
    peak = np.max(np.abs(samples))
    if peak < 1e-6:
        return samples  # silence — don't divide by zero
    return (samples / peak * target_peak).astype(np.float32)


class AudioProcessor:
    def __init__(self, cfg: Config = default_config):
        self.cfg = cfg
        self.target_sr = cfg.audio.output_sample_rate  # 8000
        self.chunk_samples = (
            cfg.audio.output_sample_rate * cfg.audio.chunk_duration_ms // 1000
        )  # 160 samples @ 8 kHz / 20 ms

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(self, samples: np.ndarray, source_sr: int) -> np.ndarray:
        """Resample from source_sr → 8000 Hz using a polyphase filter."""
        if source_sr == self.target_sr:
            return samples
        frac = Fraction(self.target_sr, source_sr).limit_denominator(1000)
        return resample_poly(samples, frac.numerator, frac.denominator).astype(np.float32)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def to_telephony_chunks(
        self, samples_8k: np.ndarray
    ) -> Generator[bytes, None, None]:
        """
        Split resampled (8 kHz) float32 audio into fixed 20 ms μ-law chunks.
        Pads the final chunk with silence if needed.
        """
        n = len(samples_8k)
        for start in range(0, n, self.chunk_samples):
            chunk = samples_8k[start : start + self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            yield float32_to_ulaw_bytes(chunk)

    # ------------------------------------------------------------------
    # WAV export (for /synthesize/full)
    # ------------------------------------------------------------------

    def to_wav_bytes(self, samples: np.ndarray, sample_rate: int) -> bytes:
        """Return a standard PCM-16 WAV at the given sample_rate."""
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def to_wav_8k_bytes(self, samples: np.ndarray, source_sr: int) -> bytes:
        """Normalize, resample to 8 kHz, then return PCM-16 WAV."""
        samples = normalize(samples)
        resampled = self.resample(samples, source_sr)
        return self.to_wav_bytes(resampled, self.target_sr)

    def normalize_and_resample(self, samples: np.ndarray, source_sr: int) -> np.ndarray:
        """Normalize then resample. Only call this on full audio, not per-chunk."""
        return self.resample(normalize(samples), source_sr)
