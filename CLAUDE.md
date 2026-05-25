# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (run from repo root in venv)
pip install -r requirements.txt

# Pre-download model weights once before first run
python scripts/download_model.py

# Start the FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run all tests (model tests auto-skip if parler-tts is not installed)
pytest tests/

# Run a single test file
pytest tests/test_audio.py -v

# Generate 20 held-out Telugu sample WAVs
python scripts/generate_samples.py --output_dir samples/

# Run benchmark (local model only)
python scripts/benchmark_cartesia.py

# Run benchmark with Cartesia comparison
CARTESIA_API_KEY=<key> python scripts/benchmark_cartesia.py --cartesia
```

## Architecture

**Goal**: Self-hosted Telugu TTS that streams 8 kHz μ-law PCM (telephony format, G.711) with sub-200 ms time-to-first-audio-chunk.

**Model**: `ai4bharat/indic-parler-tts` — a Parler-TTS variant fine-tuned by AI4Bharat for Indian languages. Uses *two* tokenizers:
- `prompt_tokenizer` (from the model itself): encodes Telugu text
- `desc_tokenizer` (from `model.config.text_encoder._name_or_path`): encodes the English speaker description

**Data flow**:
```
POST /synthesize
    │
    ▼
IndicTTSModel.generate_streaming()   ← yields float32 chunks at model's native SR (44100 Hz)
    │   (runs model.generate() in a background thread via ParlerTTSStreamer)
    ▼
AudioProcessor.resample()            ← polyphase downsample → 8000 Hz
    │
    ▼
AudioProcessor.to_telephony_chunks() ← split into 160-byte / 20 ms μ-law packets
    │
    ▼
StreamingResponse (audio/basic)      ← chunked HTTP to client
```

**Key design decisions**:
- `play_steps_in_s = 0.5` in `Config` controls how many seconds of audio are decoded per streaming iteration. Lower = lower TTFA but more overhead.
- μ-law encoding uses the vectorised ITU-T G.711 algorithm in `src/audio/processor.py:_int16_to_ulaw`. This is strict G.711, not the approximated log formula.
- `POST /synthesize/full` runs the same pipeline end-to-end and returns a standard PCM-16 WAV at 8 kHz — useful for offline quality evaluation.
- RTF and TTFA are logged server-side on every request; the full-synthesis endpoint also exposes them as response headers (`X-RTF`, `X-Audio-Duration`, `X-Inference-Time`).

**Module responsibilities**:
- `src/config.py` — single `Config` singleton; all tunable knobs live here
- `src/models/indic_tts_model.py` — model lifecycle (`load`) and inference (`generate_streaming`, `generate_full`)
- `src/audio/processor.py` — resampling, G.711 μ-law encoding, WAV export
- `src/utils/helpers.py` — `Timer` and `InferenceMetrics` dataclass
- `src/main.py` — FastAPI app; the only place that wires the pieces together
