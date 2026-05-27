# 🎙️ Telugu Real-Time TTS for Telephony

A high-performance, self-hosted Text-to-Speech (TTS) engine built specifically for:

- 📞 Twilio Voice Bots
- ☎️ SIP Calling Systems
- 🤖 AI Voice Agents
- 📢 IVR Systems
- 🌐 Real-Time Conversational AI

This project enables **low-latency Telugu speech streaming** over telephony networks by combining modern generative AI with traditional phone-call audio standards.

---

# 🚀 Why This Project Exists

Traditional TTS systems generate the **entire audio output first** before sending it to the caller.

That creates:

- ❌ Long pauses
- ❌ Delayed responses
- ❌ Poor call experience

For real-time phone conversations, this delay feels unnatural.

---

# ✅ The Solution

This project uses a **Streaming Producer-Consumer Architecture**.

Instead of waiting for the full sentence:

- Audio is generated chunk-by-chunk
- Processed immediately
- Streamed instantly to the caller

### Result

✅ Faster first response  
✅ Natural conversational flow  
✅ Real-time Telugu voice interaction

---

# 🏗️ System Architecture

## Real-Time Audio Pipeline

```text
Client Request
      ↓
FastAPI Server
      ↓
Background TTS Worker
      ↓
Audio Queue
      ↓
Audio Processor
      ↓
G.711 μ-law Encoder
      ↓
Streaming Response
      ↓
Telephony Client (Twilio / SIP / IVR)
```

---

# ⚡ Core Components

## 1️⃣ FastAPI API Layer

Handles:
- Incoming synthesis requests
- Streaming responses
- Health checks

---

## 2️⃣ Background Worker Thread

Runs the TTS model separately from the main API loop.

Benefits:
- Non-blocking execution
- Smooth streaming
- Better concurrency

---

## 3️⃣ Audio Queue

Acts as a buffer between:
- Audio generation
- Audio streaming

This ensures continuous playback without interruptions.

---

## 4️⃣ Audio Processing Engine

Converts model-generated audio into:

- 8kHz sample rate
- G.711 μ-law encoding

This format is required for:
- Telephone systems
- SIP calls
- Twilio media streams

---

# 🧠 Why Telephony Audio Needs Conversion

Modern AI models generate high-quality audio at:

- 44.1kHz
- 48kHz

But phone networks only support:

- 8kHz mono audio

Without proper conversion:
- Audio becomes robotic
- Streaming may fail
- Latency increases

This project automatically handles:
- Resampling
- Encoding
- Telephony optimization

---

# 🛠️ Prerequisites

## Hardware

### Recommended
- NVIDIA GPU with CUDA support

### Minimum
- CPU execution possible
- Real-time performance may be slower

---

## Software Requirements

| Requirement | Version |
|---|---|
| Python | 3.13+ |
| CUDA | 12.8 |
| OS | Windows / Linux |

---

## Hugging Face Access

You must have access to:

```text
ai4bharat/indic-parler-tts
```

---

# ⚡ Quick Start

# 1️⃣ Clone Repository

```bash
git clone https://github.com/SathwikVakalapudi/TTS_Project
cd TTS_Project
```

---

# 2️⃣ Create Virtual Environment

## Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

# 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Authenticate with Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted.

---

# 5️⃣ Download Model

```bash
python scripts/download_model.py
```

---

# 6️⃣ Start the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Server runs at:

```text
http://localhost:8000
```

---

# 📡 API Endpoints

| Endpoint | Description | Output |
|---|---|---|
| `/synthesize/stream` | Real-time streaming synthesis | G.711 μ-law (8kHz) |
| `/synthesize/full` | Standard synthesis | WAV (8kHz) |
| `/synthesize/hq` | High-quality synthesis | WAV (44.1kHz) |
| `/health` | Server health check | JSON |

---

# 🎯 Best Use Cases

This project is ideal for:

- AI Call Assistants
- Customer Support Bots
- Voice AI Agents
- Telugu IVR Systems
- SIP Audio Streaming
- Real-Time AI Conversations
- Twilio Media Streams
- Voice-enabled Automation

---

# 📊 Performance Benchmarking

A benchmarking utility is included:

```bash
python scripts/benchmark.py
```

---

# 📈 Understanding RTF (Real-Time Factor)

RTF measures how fast audio is generated compared to playback speed.

---

## ✅ RTF < 1

Audio generation is faster than playback.

This is ideal.

Example:
- 1 second audio generated in 0.5 seconds

---

## ❌ RTF > 1

Generation is slower than playback.

This may cause:
- Audio lag
- Choppy streaming
- Delayed responses

---

# 📁 Project Structure

```text
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
```

---

# 🔥 Key Features

✅ Real-time streaming  
✅ GPU accelerated inference  
✅ FastAPI backend  
✅ Low-latency architecture  
✅ Telephony-ready audio  
✅ Streaming queue system  
✅ Modular design  
✅ Telugu speech synthesis

---

# 🌍 Future Improvements

Planned enhancements:

- WebRTC support
- Opus codec streaming
- TensorRT optimization
- Multi-speaker voices
- Additional Indic languages
- Voice cloning
- Kubernetes deployment
- WebSocket streaming APIs

---

# 🤝 Contributing

Contributions are welcome.

## Steps

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

# ❤️ Built With

- FastAPI
- PyTorch
- Hugging Face Transformers
- CUDA
- Indic Parler TTS

---

# 📜 License

Licensed under the MIT License.

---

# ⭐ Support the Project

If this project helped you:

- ⭐ Star the repository
- 🍴 Fork the project
- 🧠 Contribute improvements
- 📢 Share with the community

---

# 🇮🇳 Built for the Future of Telugu Voice AI

Delivering natural Telugu speech for real-time telephony and conversational AI systems.
