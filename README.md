# 🎙️ Telugu Real-Time TTS for Telephony

A high-performance, self-hosted Text-to-Speech (TTS) engine specifically designed for real-time telephony applications like Twilio, SIP, IVR systems, and AI voice agents.

This project bridges the gap between modern generative AI models and legacy telephony infrastructure, allowing you to stream Telugu speech with minimal latency.

---

# 🚀 The Core Problem

Standard TTS models are often too slow for phone calls, leading to awkward silences and delayed responses.

This project solves that problem using a **Streaming Producer-Consumer Architecture**, ensuring the first syllable is heard by the caller as soon as the model generates it — instead of waiting for the entire sentence to finish processing.

---

# 🏗️ Architecture at a Glance

The system uses an asynchronous streaming workflow to maintain smooth real-time audio output.

## Flow Overview

1. **API Layer (FastAPI)**
   - Receives incoming synthesis requests
   - Starts the audio streaming pipeline

2. **Worker Thread**
   - Runs the `ai4bharat/indic-parler-tts` model in the background
   - Prevents blocking the main FastAPI event loop

3. **Processing Queue**
   - Buffers generated audio chunks
   - Uses a thread-safe queue for smooth streaming

4. **Audio Transformer Engine**
   - Converts high-quality model output into:
     - 8kHz audio
     - G.711 μ-law encoding
   - This is the standard format required by telephone networks

---

# 🛠️ Prerequisites

## Hardware
- NVIDIA GPU (CUDA-capable) strongly recommended for real-time performance

## Environment
- Python 3.13+
- CUDA 12.8

## Model Access
You must have access to:

```bash
ai4bharat/indic-parler-tts

on Hugging Face.

⚡ Quick Start
1️⃣ Clone the Repository
git clone https://github.com/SathwikVakalapudi/TTS_Project
cd TTS_Project
2️⃣ Create Virtual Environment
Linux / Mac
python -m venv venv
source venv/bin/activate
Windows
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Authenticate with Hugging Face
huggingface-cli login

Enter your Hugging Face token when prompted.

5️⃣ Download Model Weights
python scripts/download_model.py
6️⃣ Run the Server
uvicorn src.main:app --host 0.0.0.0 --port 8000

Server will start at:

http://localhost:8000
📡 API Endpoints
Endpoint	Purpose	Output Format
/synthesize/stream	Real-time streaming for calls	G.711 μ-law (8kHz)
/synthesize/full	Standard synthesis	WAV (8kHz)
/synthesize/hq	High-quality synthesis	WAV (44.1kHz)
/health	Server health check	JSON
🎯 Designed For
Twilio Voice Bots
SIP Telephony
IVR Systems
AI Call Centers
Conversational Voice Agents
Real-Time Telugu Assistants
WebRTC Audio Pipelines
🧠 Behind the Scenes
Why 8kHz G.711?

Modern AI TTS models typically generate audio at:

44.1kHz
48kHz

Traditional telephony systems only support:

8kHz mono audio

If high-quality audio is streamed directly into a phone network:

Audio may fail
Voice may sound robotic or metallic
Latency increases significantly

This project automatically:

Resamples audio
Encodes into G.711 μ-law
Streams telephony-compatible chunks in real time

Result:
✅ Low latency
✅ Smooth playback
✅ Natural phone-call audio quality

⚙️ Real-Time Streaming Architecture
Producer → Consumer Pipeline
TTS Model (Producer)
        ↓
Audio Queue
        ↓
Audio Processor
        ↓
G.711 Encoder
        ↓
Streaming API Response
        ↓
Telephony Client (Twilio/SIP)

This architecture allows:

Continuous audio generation
Chunk-by-chunk streaming
Reduced buffering delays
Near real-time interaction
📊 Real-Time Factor (RTF)

The project includes a benchmarking suite:

scripts/benchmark.py
What is RTF?
RTF < 1

✅ Audio is generated faster than playback speed

Ideal for real-time calls.

RTF > 1

❌ Generation is slower than playback

This may cause:

Choppy audio
Delays
Caller interruptions
📁 Project Structure
TTS_Project/
│
├── src/
│   ├── main.py
│   ├── api/
│   ├── audio/
│   ├── models/
│   └── streaming/
│
├── scripts/
│   ├── download_model.py
│   └── benchmark.py
│
├── requirements.txt
├── README.md
└── .gitignore
🔥 Performance Features
Real-time chunk streaming
GPU accelerated inference
Low-latency architecture
Thread-safe processing queues
Telephony-compatible audio pipeline
Async FastAPI server
Modular streaming engine
🌍 Future Improvements

Potential areas for contribution:

Opus codec support for WebRTC
TensorRT optimization
Multi-speaker Telugu voices
Additional Indic language support
Dynamic voice cloning
Streaming WebSocket API
Kubernetes deployment support
🤝 Contributing

Contributions are welcome!

If you'd like to improve the system:

Fork the repository
Create a feature branch
Submit a pull request

Ideas:

Faster inference optimization
Better streaming codecs
Improved buffering strategies
Additional language support
❤️ Acknowledgements

Built using:

FastAPI
Hugging Face Transformers
PyTorch
Indic Parler TTS
CUDA

Special thanks to the Telugu AI and open-source community.

📜 License

This project is licensed under the MIT License.

⭐ Support

If you found this project useful:

Star the repository
Share it with the community
Contribute improvements
🇮🇳 Built for the Future of Telugu Voice AI

Empowering real-time conversational AI systems with natural Telugu speech for telephony and beyond.
