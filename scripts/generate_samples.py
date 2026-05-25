"""
Generate 20 held-out Telugu audio samples and save them as 8 kHz WAV files.
Includes 5 code-mixed (Telugu + English) sentences.

Usage:
    python scripts/generate_samples.py --output_dir samples/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
from src.audio.processor import AudioProcessor
from src.config import config
from src.models.indic_tts_model import IndicTTSModel

TELUGU_SENTENCES = [
    # Pure Telugu (15 sentences)
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "ఈరోజు వాతావరణం చాలా బాగుంది.",
    "మీ పేరు ఏమిటి?",
    "నేను తెలుగు నేర్చుకుంటున్నాను.",
    "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని.",
    "భోజనం సిద్ధంగా ఉంది, రండి.",
    "పిల్లలు బడికి వెళ్ళారు.",
    "అతను చాలా మంచి వ్యక్తి.",
    "ఆకాశంలో నక్షత్రాలు వెలిగిపోతున్నాయి.",
    "నీళ్ళు తాగండి, ఆరోగ్యం బాగుంటుంది.",
    "తెలుగు సాహిత్యం చాలా గొప్పది.",
    "మా ఊరు చాలా అందంగా ఉంటుంది.",
    "సంగీతం వినడం నాకు చాలా ఇష్టం.",
    "రేపు సంత జరుగుతుంది.",
    "అమ్మ వంట చేస్తోంది.",
    # Code-mixed Telugu + English (5 sentences)
    "నేను office కి వెళ్ళాలి, meeting ఉంది.",
    "ఈ software చాలా useful గా ఉంది.",
    "Please నాకు help చేయండి.",
    "అతను engineer అయ్యాడు, చాలా proud గా ఉన్నాను.",
    "Mobile లో message పంపించు.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="samples", help="Directory to save WAV files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model …")
    tts = IndicTTSModel(config)
    tts.load()
    proc = AudioProcessor(config)

    for i, sentence in enumerate(TELUGU_SENTENCES, start=1):
        label = "codemix" if i > 15 else "telugu"
        filename = os.path.join(args.output_dir, f"sample_{i:02d}_{label}.wav")
        print(f"[{i:02d}/20] {sentence[:50]} …")

        audio = tts.generate_full(sentence)
        wav_bytes = proc.to_wav_8k_bytes(audio, tts.sampling_rate)

        with open(filename, "wb") as f:
            f.write(wav_bytes)

    print(f"\nSaved {len(TELUGU_SENTENCES)} samples to {args.output_dir}/")


if __name__ == "__main__":
    main()
