"""
Model integration tests — require GPU/CPU and the model to be downloaded.
Skipped automatically if the model is not cached.
"""

import numpy as np
import pytest

try:
    from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
    HAS_PARLER = True
except ImportError:
    HAS_PARLER = False

pytestmark = pytest.mark.skipif(not HAS_PARLER, reason="parler-tts not installed")

from src.config import Config
from src.models.indic_tts_model import IndicTTSModel

SHORT_TEXT = "నమస్కారం."   # "Hello." in Telugu


@pytest.fixture(scope="module")
def loaded_model():
    m = IndicTTSModel(Config())
    m.load()
    return m


def test_model_loads(loaded_model):
    assert loaded_model.model is not None
    assert loaded_model.sampling_rate is not None
    assert loaded_model.sampling_rate > 0


def test_generate_full_returns_array(loaded_model):
    audio = loaded_model.generate_full(SHORT_TEXT)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.size > 0


def test_generate_full_amplitude(loaded_model):
    audio = loaded_model.generate_full(SHORT_TEXT)
    # Audio should have meaningful amplitude, not silence
    assert np.max(np.abs(audio)) > 0.01


def test_streaming_yields_chunks(loaded_model):
    chunks = list(loaded_model.generate_streaming(SHORT_TEXT))
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.float32


def test_streaming_matches_full(loaded_model):
    """Concatenated streaming chunks should equal generate_full output."""
    streamed = np.concatenate(list(loaded_model.generate_streaming(SHORT_TEXT)))
    full = loaded_model.generate_full(SHORT_TEXT)
    # Allow minor floating-point variation
    np.testing.assert_allclose(streamed, full, rtol=1e-5, atol=1e-6)


def test_custom_description(loaded_model):
    desc = "A male speaker reads the text in a calm, clear Telugu voice."
    audio = loaded_model.generate_full(SHORT_TEXT, description=desc)
    assert audio.size > 0
