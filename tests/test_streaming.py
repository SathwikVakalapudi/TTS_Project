"""
Unit + integration tests for the async streaming generator.
Model is mocked — no GPU required.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.audio.processor import AudioProcessor
from src.config import Config
from src.streaming.generator import StreamMetrics, generate_mulaw_stream

CHUNK_SAMPLES = 160  # 20 ms @ 8 kHz


def _make_mock_model(duration_s: float = 1.0, sr: int = 44100):
    model = MagicMock()
    model.device = "cpu"
    model.sampling_rate = sr
    samples = (np.sin(np.linspace(0, 2 * np.pi * 440, int(sr * duration_s))) * 0.5).astype(np.float32)
    # Yield in 3 chunks to simulate streaming
    chunk_size = len(samples) // 3
    chunks = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]
    model.generate_streaming.return_value = iter(chunks)
    return model


@pytest.fixture
def proc():
    return AudioProcessor(Config())


# ------------------------------------------------------------------
# Chunk format
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_yields_bytes(proc):
    model = _make_mock_model()
    chunks = []
    async for chunk in generate_mulaw_stream(model, proc, "నమస్కారం."):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert all(isinstance(c, bytes) for c in chunks)


@pytest.mark.asyncio
async def test_stream_chunk_size(proc):
    model = _make_mock_model()
    async for chunk in generate_mulaw_stream(model, proc, "నమస్కారం."):
        assert len(chunk) == CHUNK_SAMPLES, f"Expected {CHUNK_SAMPLES}, got {len(chunk)}"


@pytest.mark.asyncio
async def test_stream_total_bytes_matches_duration(proc):
    sr = 44100
    duration = 1.0
    model = _make_mock_model(duration_s=duration, sr=sr)
    total_bytes = 0
    async for chunk in generate_mulaw_stream(model, proc, "నమస్కారం."):
        total_bytes += len(chunk)
    # At 8kHz, 1 second ≈ 8000 bytes (allow ±160 bytes rounding)
    assert abs(total_bytes - 8000) <= 160


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_populated(proc):
    model = _make_mock_model()
    metrics = StreamMetrics()
    async for _ in generate_mulaw_stream(model, proc, "నమస్కారం.", metrics=metrics):
        pass
    assert metrics.ttfa_s is not None
    assert metrics.total_s is not None
    assert metrics.audio_duration_s > 0
    assert metrics.chunks_yielded > 0


@pytest.mark.asyncio
async def test_ttfa_less_than_total(proc):
    model = _make_mock_model()
    metrics = StreamMetrics()
    async for _ in generate_mulaw_stream(model, proc, "నమస్కారం.", metrics=metrics):
        pass
    assert metrics.ttfa_s <= metrics.total_s


@pytest.mark.asyncio
async def test_rtf_is_positive(proc):
    model = _make_mock_model()
    metrics = StreamMetrics()
    async for _ in generate_mulaw_stream(model, proc, "నమస్కారం.", metrics=metrics):
        pass
    assert metrics.rtf is not None
    assert metrics.rtf > 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_model_output(proc):
    """If model yields nothing, generator should complete without error."""
    model = _make_mock_model()
    model.generate_streaming.return_value = iter([])
    chunks = []
    async for chunk in generate_mulaw_stream(model, proc, "నమస్కారం."):
        chunks.append(chunk)
    # No chunks expected — no error expected
    assert isinstance(chunks, list)
