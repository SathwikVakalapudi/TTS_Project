"""
Full benchmark — measures TTFA, RTF, audio quality for all 20 sentences.
Runs directly against the model (no HTTP overhead) for accurate timing.
Also saves the 5 submission audio samples (HQ WAV + telephony μ-law).

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --output_dir results/ --samples_dir samples/submission/
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf

from src.audio.processor import AudioProcessor, float32_to_ulaw_bytes, normalize
from src.config import Config
from src.models.indic_tts_model import IndicTTSModel
from src.utils.helpers import Timer

# ------------------------------------------------------------------
# 20 held-out sentences (15 pure Telugu + 5 code-mixed)
# ------------------------------------------------------------------
SENTENCES = [
    ("te", "01_telugu",  "నమస్కారం, మీరు ఎలా ఉన్నారు?"),
    ("te", "02_telugu",  "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని."),
    ("te", "03_telugu",  "ఈరోజు వాతావరణం చాలా బాగుంది."),
    ("te", "04_telugu",  "తెలుగు సాహిత్యం చాలా గొప్పది."),
    ("te", "05_telugu",  "నేను తెలుగు నేర్చుకుంటున్నాను."),
    ("te", "06_telugu",  "భోజనం సిద్ధంగా ఉంది, రండి."),
    ("te", "07_telugu",  "పిల్లలు బడికి వెళ్ళారు."),
    ("te", "08_telugu",  "ఆకాశంలో నక్షత్రాలు వెలిగిపోతున్నాయి."),
    ("te", "09_telugu",  "సంగీతం వినడం నాకు చాలా ఇష్టం."),
    ("te", "10_telugu",  "అమ్మ వంట చేస్తోంది."),
    ("te", "11_telugu",  "మా ఊరు చాలా అందంగా ఉంటుంది."),
    ("te", "12_telugu",  "నీళ్ళు తాగండి, ఆరోగ్యం బాగుంటుంది."),
    ("te", "13_telugu",  "రేపు సంత జరుగుతుంది."),
    ("te", "14_telugu",  "అతను చాలా మంచి వ్యక్తి."),
    ("te", "15_telugu",  "ఇది ఒక పరీక్ష వాక్యం."),
    ("cm", "16_codemix", "నేను office కి వెళ్ళాలి, meeting ఉంది."),
    ("cm", "17_codemix", "ఈ software చాలా useful గా ఉంది."),
    ("cm", "18_codemix", "Please నాకు help చేయండి."),
    ("cm", "19_codemix", "అతను engineer అయ్యాడు, చాలా proud గా ఉన్నాను."),
    ("cm", "20_codemix", "Mobile లో message పంపించు."),
]

# 5 submission samples (1-indexed): 2 Telugu + 1 longer Telugu + 2 code-mixed
SUBMISSION_IDS = {2, 4, 10, 16, 18}


# ------------------------------------------------------------------
# Quality metrics (reference-free, computed on the full HQ audio)
# ------------------------------------------------------------------

def compute_quality(audio_norm: np.ndarray, sr: int) -> dict:
    """
    Compute reference-free audio quality metrics on a normalized float32 array.

    rms_dbfs      : RMS signal level in dBFS (louder = closer to 0; speech: -18 to -6)
    peak_dbfs     : Peak level in dBFS (after peak-normalize at 0.95 → should be ~-0.45)
    silence_pct   : % of samples with |amplitude| < 0.01 (low = more voiced content)
    duration_s    : total duration in seconds
    """
    peak = float(np.max(np.abs(audio_norm)))
    rms  = float(np.sqrt(np.mean(audio_norm ** 2)))

    peak_dbfs     = 20 * np.log10(peak) if peak > 1e-9 else -96.0
    rms_dbfs      = 20 * np.log10(rms)  if rms  > 1e-9 else -96.0
    silence_pct   = float(np.mean(np.abs(audio_norm) < 0.01) * 100)
    duration_s    = len(audio_norm) / sr

    return {
        "duration_s":  round(duration_s, 3),
        "rms_dbfs":    round(rms_dbfs, 2),
        "peak_dbfs":   round(peak_dbfs, 2),
        "silence_pct": round(silence_pct, 1),
    }


# ------------------------------------------------------------------
# Single-sentence benchmark
# ------------------------------------------------------------------

def benchmark_one(tts: IndicTTSModel, proc: AudioProcessor, text: str) -> dict:
    """
    Measure TTFA via the streaming path, then run full generation for quality metrics.
    Returns a merged dict with timing + quality fields.
    """

    # --- TTFA: measured on the streaming path ---
    timer = Timer().start()
    ttfa = None
    stream_sample_count = 0

    for chunk in tts.generate_streaming(text):
        if ttfa is None:
            ttfa = timer.elapsed()
        stream_sample_count += len(proc.resample(chunk, tts.sampling_rate))

    total_s = timer.elapsed()
    audio_duration_s = stream_sample_count / proc.target_sr

    timing = {
        "ttfa_s":          round(ttfa or 0.0, 3),
        "total_s":         round(total_s, 3),
        "audio_duration_s": round(audio_duration_s, 3),
        "rtf":             round(total_s / audio_duration_s, 4) if audio_duration_s else None,
        "chars":           len(text),
    }

    # --- Quality: measured on full-generation audio (no chunk-boundary artifacts) ---
    audio_full = tts.generate_full(text)
    audio_norm = normalize(audio_full)
    quality = compute_quality(audio_norm, tts.sampling_rate)

    return {**timing, **quality}, audio_norm


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def print_ascii_table(rows: list[tuple]) -> None:
    W = 100
    print(f"\n{'=' * W}")
    print(f"  Telugu TTS Benchmark — ai4bharat/indic-parler-tts (base, no fine-tuning)")
    print(f"{'=' * W}")
    hdr = (
        f"{'#':>3}  {'T':<3}  {'Sentence':<35}  "
        f"{'TTFA':>6}  {'RTF':>7}  {'Dur':>5}  "
        f"{'RMS':>7}  {'Peak':>6}  {'Sil%':>5}"
    )
    print(hdr)
    print(f"{'-' * W}")

    for i, (lang, slug, text, r) in enumerate(rows, 1):
        label = "CM" if lang == "cm" else "TE"
        s = text[:33] + ".." if len(text) > 33 else text
        sub = " *" if i in SUBMISSION_IDS else ""
        print(
            f"{i:>3}  {label:<3}  {s:<35}  "
            f"{r['ttfa_s']:>6.3f}  {(r['rtf'] or 0):>7.4f}  {r['audio_duration_s']:>4.1f}s  "
            f"{r['rms_dbfs']:>6.2f}  {r['peak_dbfs']:>5.2f}  {r['silence_pct']:>4.1f}{sub}"
        )

    print(f"{'-' * W}")
    print("  * = submission sample")

    def _avg(lst, key):
        vals = [r[key] for _, _, _, r in lst if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    te  = [x for x in rows if x[0] == "te"]
    cm  = [x for x in rows if x[0] == "cm"]
    all_ = rows

    print(f"\n  Telugu (15):     Mean TTFA={_avg(te,'ttfa_s'):.3f}s  Mean RTF={_avg(te,'rtf'):.4f}  "
          f"Mean RMS={_avg(te,'rms_dbfs'):.2f} dBFS  Mean Silence={_avg(te,'silence_pct'):.1f}%")
    print(f"  Code-mixed (5):  Mean TTFA={_avg(cm,'ttfa_s'):.3f}s  Mean RTF={_avg(cm,'rtf'):.4f}  "
          f"Mean RMS={_avg(cm,'rms_dbfs'):.2f} dBFS  Mean Silence={_avg(cm,'silence_pct'):.1f}%")
    print(f"  Overall (20):    Mean TTFA={_avg(all_,'ttfa_s'):.3f}s  Mean RTF={_avg(all_,'rtf'):.4f}  "
          f"Mean RMS={_avg(all_,'rms_dbfs'):.2f} dBFS  Mean Silence={_avg(all_,'silence_pct'):.1f}%")
    print(f"{'=' * W}\n")


def save_csv(rows: list[tuple], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["id", "type", "slug", "sentence", "chars",
              "ttfa_s", "total_s", "audio_duration_s", "rtf",
              "rms_dbfs", "peak_dbfs", "silence_pct", "submission"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (lang, slug, text, r) in enumerate(rows, 1):
            w.writerow({
                "id": i,
                "type": "code-mixed" if lang == "cm" else "telugu",
                "slug": slug,
                "sentence": text,
                "submission": i in SUBMISSION_IDS,
                **r,
            })
    print(f"Saved CSV  → {path}")


def save_markdown_table(rows: list[tuple], path: str, device: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("## Benchmark Results\n")
    lines.append(f"**System:** NVIDIA RTX 5050 Laptop GPU (8 GB) · CUDA 12.8 · Python 3.13  ")
    lines.append(f"**Model:** `ai4bharat/indic-parler-tts` (base, no fine-tuning)  ")
    lines.append(f"**Device:** {device.upper()}\n")
    lines.append("| # | Type | Sentence | TTFA (s) | RTF | Duration (s) | RMS (dBFS) | Peak (dBFS) | Silence % |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

    for i, (lang, slug, text, r) in enumerate(rows, 1):
        label = "Code-mixed" if lang == "cm" else "Telugu"
        sub = " ★" if i in SUBMISSION_IDS else ""
        s = text[:40]
        lines.append(
            f"| {i:02d} | {label} | {s}{sub} | "
            f"{r['ttfa_s']:.3f} | {(r['rtf'] or 0):.4f} | "
            f"{r['audio_duration_s']:.2f} | "
            f"{r['rms_dbfs']:.2f} | {r['peak_dbfs']:.2f} | {r['silence_pct']:.1f} |"
        )

    lines.append("")

    def _avg(lst, key):
        vals = [r[key] for _, _, _, r in lst if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    te   = [x for x in rows if x[0] == "te"]
    cm   = [x for x in rows if x[0] == "cm"]
    all_ = rows

    lines.append("### Summary\n")
    lines.append("| Group | Count | Mean TTFA (s) | Mean RTF | Mean RMS (dBFS) | Mean Silence % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| Telugu | 15 | {_avg(te,'ttfa_s'):.3f} | {_avg(te,'rtf'):.4f} | {_avg(te,'rms_dbfs'):.2f} | {_avg(te,'silence_pct'):.1f} |")
    lines.append(f"| Code-mixed | 5 | {_avg(cm,'ttfa_s'):.3f} | {_avg(cm,'rtf'):.4f} | {_avg(cm,'rms_dbfs'):.2f} | {_avg(cm,'silence_pct'):.1f} |")
    lines.append(f"| **Overall** | **20** | **{_avg(all_,'ttfa_s'):.3f}** | **{_avg(all_,'rtf'):.4f}** | **{_avg(all_,'rms_dbfs'):.2f}** | **{_avg(all_,'silence_pct'):.1f}** |")
    lines.append("")
    lines.append("> ★ = submission sample  ")
    lines.append("> **RTF** = inference_time / audio_duration (lower is better; <1.0 = faster than real-time)  ")
    lines.append("> **RMS (dBFS)** = signal energy level; speech typically −18 to −6 dBFS  ")
    lines.append("> **Silence %** = fraction of near-silent samples; lower = more voiced content  ")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved MD   → {path}")


def save_submission_samples(
    rows: list[tuple],
    audio_map: dict,
    tts: IndicTTSModel,
    proc: AudioProcessor,
    samples_dir: str,
) -> None:
    hq_dir  = os.path.join(samples_dir, "hq_44k")
    tel_dir = os.path.join(samples_dir, "telephony_8k")
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(tel_dir, exist_ok=True)

    print("\nSaving 5 submission samples …")
    for i, (lang, slug, text, _) in enumerate(rows, 1):
        if i not in SUBMISSION_IDS:
            continue
        audio_norm = audio_map[i]

        # HQ WAV — 44100 Hz
        hq_path = os.path.join(hq_dir, f"{slug}.wav")
        sf.write(hq_path, audio_norm, tts.sampling_rate, subtype="PCM_16")

        # Telephony WAV — 8kHz PCM-16 (playable in browser, same content as μ-law)
        audio_8k = proc.resample(audio_norm, tts.sampling_rate)
        tel_path = os.path.join(tel_dir, f"{slug}_8k.wav")
        sf.write(tel_path, audio_8k, proc.target_sr, subtype="PCM_16")

        label = "[CM]" if lang == "cm" else "[TE]"
        print(f"  {label} {slug}  →  {hq_path}  |  {tel_path}")

    print(f"\n  HQ (44100 Hz)   : {hq_dir}/")
    print(f"  Telephony (8 kHz): {tel_dir}/")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="results",             help="Directory for CSV + Markdown table")
    parser.add_argument("--samples_dir", default="samples/submission",  help="Directory for 5 submission WAVs")
    parser.add_argument("--no_samples",  action="store_true",           help="Skip saving submission samples")
    args = parser.parse_args()

    print("Loading model …")
    cfg = Config()
    tts = IndicTTSModel(cfg)
    tts.load()
    proc = AudioProcessor(cfg)
    print(f"Device: {tts.device.upper()}  |  Sample rate: {tts.sampling_rate} Hz\n")

    rows = []
    audio_map = {}  # id → normalized float32 array (for saving samples)

    for i, (lang, slug, text) in enumerate(SENTENCES, 1):
        tag = "[CM]" if lang == "cm" else "[TE]"
        print(f"[{i:02d}/20] {tag} {text[:55]} …", end=" ", flush=True)

        result, audio_norm = benchmark_one(tts, proc, text)
        rows.append((lang, slug, text, result))
        audio_map[i] = audio_norm

        sub_marker = " ★" if i in SUBMISSION_IDS else ""
        print(
            f"TTFA={result['ttfa_s']:.2f}s  RTF={result['rtf']:.4f}  "
            f"Dur={result['audio_duration_s']:.1f}s  "
            f"RMS={result['rms_dbfs']:.1f}dBFS  Sil={result['silence_pct']:.1f}%{sub_marker}"
        )

    print_ascii_table(rows)

    csv_path = os.path.join(args.output_dir, "benchmark.csv")
    md_path  = os.path.join(args.output_dir, "benchmark.md")
    save_csv(rows, csv_path)
    save_markdown_table(rows, md_path, tts.device)

    if not args.no_samples:
        save_submission_samples(rows, audio_map, tts, proc, args.samples_dir)

    print(f"\nDone. Copy {md_path} content into README.md → 'Benchmark Results' section.")


if __name__ == "__main__":
    main()
