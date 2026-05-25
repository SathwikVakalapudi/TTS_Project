import torch


class ModelConfig:
    model_id: str = "ai4bharat/indic-parler-tts"
    torch_dtype = torch.float16

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


class AudioConfig:
    output_sample_rate: int = 8000   # telephony target
    chunk_duration_ms: int = 20      # 20 ms → 160 bytes of μ-law at 8 kHz


class TeluguConfig:
    language_code: str = "te"
    # Speaker description steers voice quality; keep it in English.
    # Indic-Parler-TTS was trained with descriptions mentioning speaker name,
    # pace, pitch, and environment — more detail = better quality.
    speaker_description: str = (
        "Divya speaks at a moderate pace with a slightly high pitched voice "
        "delivering the message in Telugu. The recording is very clear, "
        "close-sounding, with no background noise."
    )


class Config:
    model = ModelConfig()
    audio = AudioConfig()
    telugu = TeluguConfig()
    # How many seconds of audio to buffer before yielding a chunk during streaming.
    play_steps_in_s: float = 0.5


config = Config()
