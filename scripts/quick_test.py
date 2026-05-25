"""
Quick offline test — no server needed.
Generates Telugu audio and saves as WAV at native 44100 Hz.
Run this first to confirm the model is producing valid Telugu audio.

Usage:
    python scripts/quick_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
import numpy as np
from src.models.indic_tts_model import IndicTTSModel
from src.audio.processor import normalize
from src.config import Config

# Try multiple descriptions — different ones work better for different models
DESCRIPTIONS = [
    "Divya speaks at a moderate pace with a slightly high pitched voice delivering the message in Telugu. The recording is very clear, close-sounding, with no background noise.",
    "A female speaker delivers her words quite expressively, in a clear Telugu voice with a slightly faster-than-average pace, and an animated tone.",
    "Priya speaks slowly and clearly in Telugu with a calm, natural voice. The audio is clean with no background noise.",
]

TEST_TEXT = "నమస్కారం. ఇది ఒక పరీక్ష వాక్యం."

def main():
    print("Loading model ...")
    tts = IndicTTSModel(Config())
    tts.load()
    print(f"Model loaded | device={tts.device} | sample_rate={tts.sampling_rate}")

    for i, desc in enumerate(DESCRIPTIONS, 1):
        print(f"\n[{i}/3] Testing description: {desc[:60]}...")
        audio = tts.generate_full(TEST_TEXT, description=desc)

        duration = len(audio) / tts.sampling_rate
        peak = float(np.max(np.abs(audio)))
        print(f"      Duration: {duration:.2f}s  |  Peak amplitude: {peak:.4f}")

        if peak < 0.01:
            print("      WARNING: Audio is near-silent — wrong description or model issue")

        audio_norm = normalize(audio)
        out_path = f"quick_test_{i}.wav"
        sf.write(out_path, audio_norm, tts.sampling_rate)
        print(f"      Saved: {out_path}  (normalized peak: {float(np.max(np.abs(audio_norm))):.4f})")

    print("\nDone. Open the WAV files and listen — pick the best-sounding description.")

if __name__ == "__main__":
    main()
