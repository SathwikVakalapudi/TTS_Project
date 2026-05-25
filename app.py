"""
Streamlit demo — Telugu TTS audio format comparison.

Shows 3 outputs side by side for any Telugu text input:
  1. Original HQ WAV  (44100 Hz PCM-16)
  2. μ-law raw bytes  (8000 Hz G.711, telephony)
  3. Reconverted WAV  (μ-law decoded back via ffmpeg — telephony quality)

Run:
    streamlit run app.py
"""

import io
import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Telugu TTS Demo",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Telugu TTS — Audio Format Comparison")
st.markdown(
    "Enter Telugu text → get **3 audio outputs**: "
    "HQ original, telephony μ-law, and μ-law reconverted back to WAV."
)
st.divider()

# ------------------------------------------------------------------
# Model loading (cached — loads only once per session)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ai4bharat/indic-parler-tts model…")
def load_model():
    from src.audio.processor import AudioProcessor
    from src.config import Config
    from src.models.indic_tts_model import IndicTTSModel

    cfg = Config()
    tts = IndicTTSModel(cfg)
    tts.load()
    proc = AudioProcessor(cfg)
    return tts, proc


# ------------------------------------------------------------------
# Audio generation
# ------------------------------------------------------------------
def generate_formats(text: str, tts, proc):
    from src.audio.processor import float32_to_ulaw_bytes, normalize

    t0 = time.perf_counter()

    # 1 — Generate full audio at native 44100 Hz
    audio_44k = tts.generate_full(text)
    gen_time = time.perf_counter() - t0
    audio_norm = normalize(audio_44k)
    duration_s = len(audio_norm) / tts.sampling_rate

    # 2 — HQ WAV bytes (44100 Hz PCM-16)
    hq_buf = io.BytesIO()
    sf.write(hq_buf, audio_norm, tts.sampling_rate, format="WAV", subtype="PCM_16")
    hq_bytes = hq_buf.getvalue()

    # 3 — Resample to 8 kHz and encode to raw G.711 μ-law
    audio_8k = proc.resample(audio_norm, tts.sampling_rate)
    ulaw_bytes = float32_to_ulaw_bytes(audio_8k)

    # 4 — Reconvert μ-law → WAV using ffmpeg (shows telephony quality)
    reconverted_bytes = _ulaw_to_wav_via_ffmpeg(ulaw_bytes)

    rtf = gen_time / duration_s if duration_s > 0 else None
    chunk_size = proc.chunk_samples  # 160 bytes = 20 ms @ 8 kHz
    num_chunks = len(ulaw_bytes) // chunk_size

    return hq_bytes, ulaw_bytes, reconverted_bytes, duration_s, gen_time, rtf, num_chunks, chunk_size


def _ulaw_to_wav_via_ffmpeg(ulaw_bytes: bytes) -> bytes:
    """Convert raw G.711 μ-law bytes to WAV using ffmpeg subprocess."""
    with tempfile.TemporaryDirectory() as tmp:
        ulaw_path = os.path.join(tmp, "audio.ulaw")
        wav_path  = os.path.join(tmp, "audio_reconverted.wav")

        with open(ulaw_path, "wb") as f:
            f.write(ulaw_bytes)

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "mulaw", "-ar", "8000", "-ac", "1",
                "-i", ulaw_path,
                wav_path,
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            st.error(f"ffmpeg error: {result.stderr.decode()}")
            return b""

        with open(wav_path, "rb") as f:
            return f.read()


# ------------------------------------------------------------------
# UI — input
# ------------------------------------------------------------------
col_input, col_info = st.columns([2, 1])

with col_input:
    text = st.text_area(
        "Telugu Text",
        value="నమస్కారం. ఇది తెలుగు టెక్స్ట్ టు స్పీచ్ సిస్టమ్.",
        height=120,
        placeholder="Enter Telugu text here…",
    )

with col_info:
    st.markdown("**Example sentences**")
    examples = [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "హైదరాబాద్ తెలంగాణ రాజధాని.",
        "నేను office కి వెళ్ళాలి.",   # code-mixed
        "Please నాకు help చేయండి.",    # code-mixed
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["example_text"] = ex
            st.rerun()

# Apply example if selected
if "example_text" in st.session_state:
    text = st.session_state.pop("example_text")

generate = st.button("🎙️ Generate Audio", type="primary", use_container_width=False)

# ------------------------------------------------------------------
# UI — output
# ------------------------------------------------------------------
if generate:
    if not text.strip():
        st.warning("Please enter some Telugu text.")
        st.stop()

    tts, proc = load_model()

    with st.spinner(f"Generating on {tts.device}… (first run may take 10–30 s)"):
        hq_bytes, ulaw_bytes, reconv_bytes, duration, gen_time, rtf, num_chunks, chunk_size = generate_formats(
            text, tts, proc
        )

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Audio Duration", f"{duration:.2f} s")
    m2.metric("Generation Time", f"{gen_time:.2f} s")
    m3.metric("RTF", f"{rtf:.3f}" if rtf else "—")
    m4.metric("Telephony Chunks", str(num_chunks))
    m5.metric("Device", tts.device.upper())

    st.divider()

    # Three audio columns
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("① Original HQ")
        st.caption("44100 Hz · PCM-16 · WAV")
        st.audio(hq_bytes, format="audio/wav")
        st.download_button(
            "⬇ Download HQ WAV",
            hq_bytes,
            file_name="telugu_hq.wav",
            mime="audio/wav",
            use_container_width=True,
        )
        st.info("Full quality — best for speakers/headphones.")

    with c2:
        st.subheader("② Telephony μ-law")
        st.caption("8000 Hz · G.711 μ-law · raw bytes")
        st.download_button(
            "⬇ Download μ-law (.ulaw)",
            ulaw_bytes,
            file_name="telugu_telephony.ulaw",
            mime="application/octet-stream",
            use_container_width=True,
        )
        st.warning(
            "Raw G.711 μ-law cannot play in browser.\n\n"
            "This format is sent directly to Twilio / SIP / IVR systems.\n\n"
            "Use the reconverted WAV (column 3) to hear telephony quality."
        )

        # Chunk breakdown
        st.markdown("**Chunk breakdown**")
        ch1, ch2, ch3 = st.columns(3)
        ch1.metric("Chunks", str(num_chunks))
        ch2.metric("Bytes/chunk", str(chunk_size))
        ch3.metric("ms/chunk", "20")
        st.caption(
            f"Total: {len(ulaw_bytes):,} bytes  ·  "
            f"{num_chunks} × {chunk_size} B (G.711 20 ms packets)"
        )

        st.code(
            "ffmpeg -f mulaw -ar 8000 -ac 1 \\\n"
            "  -i telugu_telephony.ulaw output.wav",
            language="bash",
        )

    with c3:
        st.subheader("③ μ-law → WAV (Telephony Quality)")
        st.caption("8000 Hz · PCM-16 · WAV (decoded from μ-law via ffmpeg)")
        if reconv_bytes:
            st.audio(reconv_bytes, format="audio/wav")
            st.download_button(
                "⬇ Download Telephony WAV",
                reconv_bytes,
                file_name="telugu_telephony.wav",
                mime="audio/wav",
                use_container_width=True,
            )
            st.info(
                "This is exactly what a caller hears on Twilio/SIP.\n"
                "Slightly muffled (8 kHz bandwidth) — expected for phone calls."
            )
        else:
            st.error("ffmpeg not found. Install it with: winget install Gyan.FFmpeg")
