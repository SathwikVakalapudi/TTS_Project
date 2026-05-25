# Telugu TTS — Real-Time Telephony Streaming

Self-hosted Telugu Text-to-Speech system using `ai4bharat/indic-parler-tts`.
Streams 8 kHz G.711 μ-law PCM audio for Twilio, SIP, IVR, and real-time AI voice agents.

**System:** NVIDIA RTX 5050 Laptop GPU (8 GB) · CUDA 12.8 · Python 3.13

---

## Setup

### 1. Clone and create virtualenv
```bash
git clone <repo-url>
cd TTS_system
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Linux/Mac
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model weights (~5 GB, one-time)
The model is gated. First request access at https://huggingface.co/ai4bharat/indic-parler-tts, then:
```bash
huggingface-cli login        # paste your HF token
python scripts/download_model.py
```

### 4. Start the server
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```
Wait for: `Model ready — native sample rate: 44100 Hz | device: cuda`

---

## API Endpoints

| Endpoint | Output | Use Case |
|---|---|---|
| `POST /synthesize/stream` | 8kHz μ-law streaming | **Production telephony** |
| `POST /synthesize` | 8kHz μ-law streaming | Legacy streaming |
| `POST /synthesize/full` | 8kHz PCM-16 WAV | Telephony evaluation |
| `POST /synthesize/hq` | 44100Hz PCM-16 WAV | Quality evaluation |
| `GET /health` | JSON | Health check |

### Request body (all POST endpoints)
```json
{
  "text": "నమస్కారం. ఇది తెలుగు వాయిస్ సిస్టమ్.",
  "description": null
}
```
`description` is optional. Default speaker: *Divya* (clear female Telugu voice).

---

## Usage Examples

### PowerShell (Windows)
```powershell
# High-quality WAV (best for testing)
$body = [System.Text.Encoding]::UTF8.GetBytes('{"text": "నమస్కారం."}')
Invoke-WebRequest -Uri "http://localhost:8000/synthesize/hq" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Body $body -OutFile test_hq.wav

# Telephony WAV (8kHz)
Invoke-WebRequest -Uri "http://localhost:8000/synthesize/full" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Body $body -OutFile test_8k.wav

# Raw μ-law stream
Invoke-WebRequest -Uri "http://localhost:8000/synthesize/stream" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Body $body -OutFile test.ulaw
```

### curl
```bash
# HQ WAV
curl -X POST http://localhost:8000/synthesize/hq \
  -H "Content-Type: application/json" \
  -d '{"text": "నమస్కారం."}' \
  --output test_hq.wav

# Stream μ-law and play with ffplay
curl -X POST http://localhost:8000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "నమస్కారం."}' | \
  ffplay -f mulaw -ar 8000 -ac 1 -

# Health check
curl http://localhost:8000/health
```

### Play μ-law file with ffplay
```bash
ffplay -f mulaw -ar 8000 -ac 1 test.ulaw
```

---

## Streaming Architecture

```
POST /synthesize/stream
        │
        ▼
  [Worker Thread]  model.generate_streaming()   ← GPU-bound, blocking
        │  threading.Queue (maxsize=64, backpressure)
        ▼
  [Event Loop]  normalize → resample 44100→8kHz → G.711 μ-law encode
        │  160-byte chunks (20 ms)
        ▼
  StreamingResponse  →  Twilio / SIP / IVR / AI voice agent
```

The asyncio event loop is **never blocked** — model runs in a worker thread,
chunks are passed via a thread-safe queue read with `run_in_executor`.

---

## Benchmark

Run the full 20-sentence benchmark (TTFA + RTF):
```bash
python scripts/benchmark.py
python scripts/benchmark.py --output results/benchmark.csv
```

Output includes:
- Time-to-First-Audio-Chunk (TTFA)
- Real-Time Factor (RTF = inference_time / audio_duration)
- Audio duration per sentence
- Separate stats for pure Telugu vs code-mixed

---

## Generate Audio Samples

```bash
# Generate all 20 samples (HQ + telephony formats)
python scripts/evaluate.py --output_dir samples/

# Quick test — 3 samples with different speaker descriptions
python scripts/quick_test.py
```

Output structure:
```
samples/
  hq_44k/          ← 44100 Hz WAV, best quality
  telephony_8k_ulaw/ ← raw G.711 μ-law, telephony-ready
```

---

## Run Tests

```bash
# All tests (model tests auto-skip if not installed)
pytest tests/ -v

# Audio processing only (no GPU)
pytest tests/test_audio.py tests/test_streaming.py -v

# Streaming tests only
pytest tests/test_streaming.py -v
```

---

## Telephony Compatibility

| System | Format | Endpoint |
|---|---|---|
| Twilio Media Streams | 8kHz μ-law, raw bytes | `/synthesize/stream` |
| Asterisk / FreePBX | 8kHz μ-law (G.711) | `/synthesize/stream` |
| SIP / RTP | 20ms packets, G.711 μ-law | `/synthesize/stream` |
| IVR platforms | 8kHz WAV or raw μ-law | `/synthesize/full` or `/synthesize/stream` |

Each streamed chunk is **160 bytes = 20 ms** — the standard G.711 RTP packet size.

---

## Project Structure

```
src/
  config.py                  — model, audio, Telugu settings
  main.py                    — FastAPI app + all endpoints
  models/
    indic_tts_model.py       — model loading + streaming inference
  audio/
    processor.py             — resample, G.711 μ-law encode, WAV export
  streaming/
    generator.py             — true async streaming generator
  utils/
    helpers.py               — Timer, InferenceMetrics

scripts/
  download_model.py          — pre-cache HF model weights
  quick_test.py              — fast offline audio test
  benchmark.py               — TTFA + RTF benchmark (20 sentences)
  evaluate.py                — generate + save 20 audio samples
  generate_samples.py        — bulk sample generation
  benchmark_cartesia.py      — compare with Cartesia API

tests/
  test_audio.py              — audio processing unit tests
  test_streaming.py          — async streaming tests
  test_api.py                — API endpoint tests (mocked model)
  test_model.py              — model integration tests
```

---

## Model Details

- **Model:** `ai4bharat/indic-parler-tts` (Parler-TTS architecture, no fine-tuning)
- **Languages:** Telugu (te) + code-mixed Telugu-English
- **Default speaker:** Divya — female, moderate pace, clear voice
- **Native output:** 44100 Hz float32 PCM
- **API output:** 8000 Hz G.711 μ-law (telephony) or 44100 Hz WAV (HQ)
- **Text encoder:** `google/flan-t5-large`
- **Audio codec:** DAC 44kHz (`ylacombe/dac_44khz`)
