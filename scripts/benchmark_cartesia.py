"""
Benchmark this TTS system and (optionally) Cartesia's API side-by-side.

Measures for each sentence:
  - TTFA  : time-to-first-audio-chunk (seconds)
  - RTF   : real-time factor (inference_time / audio_duration)
  - Total : end-to-end wall time

Usage:
    # Local model only
    python scripts/benchmark_cartesia.py

    # With Cartesia comparison (requires CARTESIA_API_KEY env var)
    python scripts/benchmark_cartesia.py --cartesia
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.audio.processor import AudioProcessor
from src.config import config
from src.models.indic_tts_model import IndicTTSModel
from src.utils.helpers import Timer

BENCHMARK_SENTENCES = [
    "నమస్కారం, ఈ వ్యవస్థ పరీక్షిస్తున్నాను.",
    "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని.",
    "నేను office కి వెళ్ళాలి, meeting ఉంది.",
    "తెలుగు సాహిత్యం చాలా గొప్పది.",
    "Please నాకు help చేయండి.",
]


def benchmark_local(tts: IndicTTSModel, proc: AudioProcessor) -> list[dict]:
    results = []
    for sentence in BENCHMARK_SENTENCES:
        wall = Timer().start()
        ttfa = None
        sample_count = 0
        first = True

        for chunk in tts.generate_streaming(sentence):
            if first:
                ttfa = wall.elapsed()
                first = False
            resampled = proc.resample(chunk, tts.sampling_rate)
            sample_count += len(resampled)

        total = wall.elapsed()
        duration = sample_count / config.audio.output_sample_rate

        results.append({
            "sentence": sentence[:40],
            "ttfa_s": round(ttfa or 0, 3),
            "total_s": round(total, 3),
            "duration_s": round(duration, 3),
            "rtf": round(total / duration, 4) if duration else None,
        })

    return results


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Sentence':<42} {'TTFA':>6} {'RTF':>7} {'Dur':>6} {'Total':>7}")
    print(f"{'-'*70}")
    for r in rows:
        print(
            f"{r['sentence']:<42} {r['ttfa_s']:>6.3f} "
            f"{(r['rtf'] or 0):>7.4f} {r['duration_s']:>6.2f}s {r['total_s']:>6.2f}s"
        )

    ttfas = [r["ttfa_s"] for r in rows]
    rtfs = [r["rtf"] for r in rows if r["rtf"]]
    print(f"{'-'*70}")
    print(f"  Mean TTFA: {np.mean(ttfas):.3f}s   Mean RTF: {np.mean(rtfs):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartesia", action="store_true", help="Also benchmark Cartesia API")
    args = parser.parse_args()

    print("Loading local model …")
    tts = IndicTTSModel(config)
    tts.load()
    proc = AudioProcessor(config)

    local_results = benchmark_local(tts, proc)
    print_table(local_results, "Local — ai4bharat/indic-parler-tts (Telugu)")

    if args.cartesia:
        api_key = os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            print("\nCARTESIA_API_KEY not set — skipping Cartesia benchmark.")
            return
        print("\nCartesia benchmark not yet implemented.")


if __name__ == "__main__":
    main()
