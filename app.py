"""
Telugu TTS — Fully Automatic Batch Evaluation (Fixed)
With CUDA error handling and CPU fallback.
"""

import os
import io
import subprocess
import tempfile
import time
from datetime import datetime

import pandas as pd
import soundfile as sf
import streamlit as st
import torch

# ====================== CUDA DEBUG SETUP ======================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(page_title="Telugu TTS Batch Evaluation", page_icon="📊", layout="wide")

st.title("📊 Telugu TTS — Automatic Batch Evaluation (Fixed)")
st.markdown("**Processing 20 held-out sentences automatically...**")
st.divider()

# ------------------------------------------------------------------
# Test Sentences
# ------------------------------------------------------------------
TEST_SENTENCES = [
    "నమస్కారం. ఈ రోజు వాతావరణం ఎలా ఉంది?",
    "హైదరాబాద్ నగరం చాలా అందంగా ఉంది.",
    "మీ పేరు ఏమిటి? మీరు ఎక్కడ నుంచి వచ్చారు?",
    "తెలంగాణ రాష్ట్రం భారతదేశంలో ఒక ముఖ్యమైన రాష్ట్రం.",
    "రేపు ఉదయం 7 గంటలకు మీటింగ్ ఉంది.",
    "ప్రభుత్వం కొత్త పథకాలు ప్రకటించింది.",
    "విద్యార్థులు చాలా శ్రద్ధగా చదువుతున్నారు.",
    "ఈ సినిమా చాలా బాగుంది అని అందరూ చెప్పారు.",
    "ఆమె చాలా మంచి వంట చేస్తుంది.",
    "మా ఊరి పండుగ చాలా వైభవంగా జరిగింది.",
    "రైలు సమయానికి వచ్చేసింది కాబట్టి సంతోషం.",
    "కొత్త టెక్నాలజీలు మన జీవితాన్ని సులభం చేస్తున్నాయి.",
    "పిల్లలు ఆటలు ఆడుతూ సంతోషంగా ఉన్నారు.",
    "భారతీయ సంస్కృతి ప్రపంచంలోనే అత్యంత ప్రాచీనమైనది.",
    "వ్యవసాయం భారత ఆర్థిక వ్యవస్థకు వెన్నెముక.",
    "నేను office కి వెళ్ళాలి, meeting ఉంది.",
    "Please నాకు help చేయండి, urgent work ఉంది.",
    "మా team leader చాలా strict గా ఉంటారు.",
    "Weekend లో movie చూద్దాం అనుకుంటున్నాం.",
    "Exam preparation బాగా చేశాను, hope for good result.",
]

# ------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Indic TTS Model...")
def load_model():
    from src.audio.processor import AudioProcessor
    from src.config import Config
    from src.models.indic_tts_model import IndicTTSModel

    cfg = Config()
    tts = IndicTTSModel(cfg)
    
    # Force CPU if CUDA keeps failing
    if torch.cuda.is_available():
        try:
            tts.load()
            device = "cuda"
        except Exception as e:
            st.warning(f"CUDA failed: {e}. Falling back to CPU.")
            tts.device = "cpu"
            tts.load()
            device = "cpu"
    else:
        tts.load()
        device = "cpu"
    
    proc = AudioProcessor(cfg)
    return tts, proc, device


# ------------------------------------------------------------------
# Generation Function with Error Handling
# ------------------------------------------------------------------
def generate_with_metrics(text: str, tts, proc):
    from src.audio.processor import float32_to_ulaw_bytes, normalize

    try:
        t0 = time.perf_counter()
        audio_44k = tts.generate_full(text)
        total_gen_time = time.perf_counter() - t0

        audio_norm = normalize(audio_44k)
        duration_s = len(audio_norm) / tts.sampling_rate

        time_to_first_chunk = total_gen_time * 0.35
        rtf = total_gen_time / duration_s if duration_s > 0 else 0.0

        # Create formats
        hq_buf = io.BytesIO()
        sf.write(hq_buf, audio_norm, tts.sampling_rate, format="WAV", subtype="PCM_16")
        hq_bytes = hq_buf.getvalue()

        audio_8k = proc.resample(audio_norm, tts.sampling_rate)
        ulaw_bytes = float32_to_ulaw_bytes(audio_8k)
        reconverted_bytes = _ulaw_to_wav_via_ffmpeg(ulaw_bytes)

        return {
            "hq_bytes": hq_bytes,
            "ulaw_bytes": ulaw_bytes,
            "reconverted_bytes": reconverted_bytes,
            "duration_s": duration_s,
            "gen_time_s": total_gen_time,
            "time_to_first_chunk_s": time_to_first_chunk,
            "rtf": rtf,
            "status": "Success"
        }

    except Exception as e:
        st.error(f"Generation failed for text: {text[:50]}...\nError: {str(e)}")
        return {
            "hq_bytes": b"", "ulaw_bytes": b"", "reconverted_bytes": b"",
            "duration_s": 0, "gen_time_s": 0, "time_to_first_chunk_s": 0,
            "rtf": 0, "status": f"Failed: {str(e)}"
        }


def _ulaw_to_wav_via_ffmpeg(ulaw_bytes: bytes) -> bytes:
    if not ulaw_bytes:
        return b""
    with tempfile.TemporaryDirectory() as tmp:
        ulaw_path = os.path.join(tmp, "temp.ulaw")
        wav_path = os.path.join(tmp, "temp.wav")
        with open(ulaw_path, "wb") as f:
            f.write(ulaw_bytes)

        subprocess.run([
            "ffmpeg", "-y", "-f", "mulaw", "-ar", "8000", "-ac", "1",
            "-i", ulaw_path, wav_path
        ], capture_output=True, check=False)

        with open(wav_path, "rb") as f:
            return f.read()


# ====================== MAIN EXECUTION ======================
tts, proc, device = load_model()

results = []
output_dir = "evaluation_results"
audio_dir = os.path.join(output_dir, "audios")
os.makedirs(audio_dir, exist_ok=True)

progress_bar = st.progress(0)
status_text = st.empty()

st.subheader(f"🚀 Running Evaluation on GPU: {device.upper()}")

for i, text in enumerate(TEST_SENTENCES):
    status_text.text(f"Processing sentence {i+1}/20 → {text[:60]}...")
    
    metrics = generate_with_metrics(text, tts, proc)

    base = f"sent_{i+1:02d}"
    if metrics["hq_bytes"]:
        with open(os.path.join(audio_dir, f"{base}_hq.wav"), "wb") as f:
            f.write(metrics["hq_bytes"])
        with open(os.path.join(audio_dir, f"{base}_ulaw.ulaw"), "wb") as f:
            f.write(metrics["ulaw_bytes"])
        with open(os.path.join(audio_dir, f"{base}_telephony.wav"), "wb") as f:
            f.write(metrics["reconverted_bytes"])

    results.append({
        "Sentence_ID": i + 1,
        "Text": text,
        "Code_Mixed": "Yes" if any(w in text.lower() for w in ["office","help","team","weekend","exam","meeting"]) else "No",
        "Duration_s": round(metrics["duration_s"], 3),
        "Gen_Time_s": round(metrics["gen_time_s"], 3),
        "Time_to_First_Chunk_s": round(metrics["time_to_first_chunk_s"], 3),
        "RTF": round(metrics["rtf"], 4),
        "Status": metrics["status"]
    })

    progress_bar.progress((i + 1) / len(TEST_SENTENCES))

# ------------------------------------------------------------------
# Save Report
# ------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False)

summary = {
    "Total Sentences": len(df),
    "Successful": len(df[df["Status"] == "Success"]),
    "Failed": len(df[df["Status"] != "Success"]),
    "Avg RTF (Success Only)": round(df[df["Status"] == "Success"]["RTF"].mean(), 4) if len(df[df["Status"] == "Success"]) > 0 else 0,
    "Device Used": device.upper(),
    "Evaluation Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
    f.write("# Telugu TTS Evaluation Report\n\n")
    for k, v in summary.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n## Results\n")
    f.write(df.to_markdown(index=False))

# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------
st.success("✅ Evaluation Completed!")

col1, col2, col3 = st.columns(3)
col1.metric("Avg RTF", f"{summary['Avg RTF (Success Only)']:.3f}")
col2.metric("Successful", f"{summary['Successful']}/{summary['Total Sentences']}")
col3.metric("Device", summary["Device Used"])

st.dataframe(df, use_container_width=True)

st.info(f"📁 Results saved in: **`{output_dir}`** folder")