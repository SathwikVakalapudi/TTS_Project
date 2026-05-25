"""
Pre-download ai4bharat/indic-parler-tts weights to the HuggingFace cache.
Run once before starting the server so the first request is not slow.

Usage:
    python scripts/download_model.py
"""

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch

MODEL_ID = "ai4bharat/indic-parler-tts"


def main():
    print(f"Downloading {MODEL_ID} …")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )

    # Also download both tokenizers
    prompt_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    desc_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    print("All weights cached successfully.")
    print(f"  Native sample rate : {model.config.sampling_rate} Hz")
    print(f"  Text encoder       : {model.config.text_encoder._name_or_path}")


if __name__ == "__main__":
    main()
