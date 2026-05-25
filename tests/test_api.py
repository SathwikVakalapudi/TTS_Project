"""
API endpoint tests using TestClient (no live server required).
Model is mocked so these tests run without GPU.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Return a fake IndicTTSModel that yields a short sine-wave chunk."""
    m = MagicMock()
    m.device = "cpu"
    m.sampling_rate = 44100
    sine = (np.sin(np.linspace(0, 2 * np.pi, 44100)) * 0.5).astype(np.float32)
    m.generate_streaming.return_value = iter([sine])
    m.generate_full.return_value = sine
    return m


@pytest.fixture
def client(mock_model):
    with patch("src.main.model", mock_model):
        from src.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /synthesize (streaming μ-law)
# ---------------------------------------------------------------------------

def test_synthesize_stream_ok(client):
    r = client.post("/synthesize", json={"text": "నమస్కారం."})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/basic"
    assert len(r.content) > 0


def test_synthesize_stream_chunks_are_multiples_of_160(client):
    """Each μ-law chunk should be 160 bytes (20 ms @ 8 kHz)."""
    r = client.post("/synthesize", json={"text": "నమస్కారం."})
    assert r.status_code == 200
    assert len(r.content) % 160 == 0


def test_synthesize_empty_text_rejected(client):
    r = client.post("/synthesize", json={"text": ""})
    assert r.status_code == 422


def test_synthesize_whitespace_text_rejected(client):
    r = client.post("/synthesize", json={"text": "   "})
    assert r.status_code == 422


def test_synthesize_custom_description(client):
    r = client.post(
        "/synthesize",
        json={"text": "నమస్కారం.", "description": "A calm male voice."},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /synthesize/full (WAV)
# ---------------------------------------------------------------------------

def test_synthesize_full_ok(client):
    r = client.post("/synthesize/full", json={"text": "నమస్కారం."})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_synthesize_full_returns_rtf_header(client):
    r = client.post("/synthesize/full", json={"text": "నమస్కారం."})
    assert "x-rtf" in r.headers


def test_synthesize_full_empty_text_rejected(client):
    r = client.post("/synthesize/full", json={"text": ""})
    assert r.status_code == 422
