"""Unit tests for AudioProcessor — no GPU or model required."""

import numpy as np
import pytest

from src.audio.processor import AudioProcessor, float32_to_ulaw_bytes
from src.config import Config


@pytest.fixture
def proc():
    return AudioProcessor(Config())


def _sine(freq=440, sr=44100, duration=0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ------------------------------------------------------------------
# μ-law encoding
# ------------------------------------------------------------------

def test_ulaw_output_dtype():
    samples = np.zeros(160, dtype=np.float32)
    result = float32_to_ulaw_bytes(samples)
    assert isinstance(result, bytes)
    assert len(result) == 160


def test_ulaw_output_length():
    samples = np.random.uniform(-1, 1, 1000).astype(np.float32)
    result = float32_to_ulaw_bytes(samples)
    assert len(result) == 1000


def test_ulaw_silence_not_all_same():
    """Silence (0.0) encodes to a specific μ-law value, not arbitrary bytes."""
    silence = np.zeros(10, dtype=np.float32)
    result = float32_to_ulaw_bytes(silence)
    # All samples should encode to the same value (μ-law of 0 → 0xFF)
    assert len(set(result)) == 1


# ------------------------------------------------------------------
# Resampling
# ------------------------------------------------------------------

def test_resample_44100_to_8000(proc):
    audio_44k = _sine(sr=44100, duration=1.0)
    audio_8k = proc.resample(audio_44k, 44100)
    assert abs(len(audio_8k) - 8000) <= 5   # allow rounding tolerance


def test_resample_identity(proc):
    audio = _sine(sr=8000, duration=0.5)
    result = proc.resample(audio, 8000)
    np.testing.assert_array_equal(audio, result)


# ------------------------------------------------------------------
# Telephony chunks
# ------------------------------------------------------------------

def test_chunk_size(proc):
    audio_8k = _sine(sr=8000, duration=1.0)
    chunks = list(proc.to_telephony_chunks(audio_8k))
    # 1 second @ 8000 Hz / 160 samples per chunk = 50 chunks
    assert len(chunks) == 50
    for chunk in chunks:
        assert len(chunk) == 160   # 20 ms × 8000 Hz = 160 bytes


def test_chunk_pads_last(proc):
    # 170 samples → 2 chunks: 160 + 10 (padded to 160)
    audio_8k = _sine(sr=8000, duration=170 / 8000)
    chunks = list(proc.to_telephony_chunks(audio_8k))
    assert len(chunks) == 2
    assert all(len(c) == 160 for c in chunks)


# ------------------------------------------------------------------
# WAV export
# ------------------------------------------------------------------

def test_to_wav_bytes_valid(proc):
    audio = _sine(sr=8000, duration=0.5)
    wav = proc.to_wav_bytes(audio, 8000)
    assert wav[:4] == b"RIFF"


def test_to_wav_8k_bytes(proc):
    audio_44k = _sine(sr=44100, duration=1.0)
    wav = proc.to_wav_8k_bytes(audio_44k, 44100)
    assert wav[:4] == b"RIFF"
