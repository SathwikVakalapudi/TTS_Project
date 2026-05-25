import threading
from typing import Generator, Optional

import numpy as np
import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer

from src.config import Config, config as default_config


class IndicTTSModel:
    """
    Wrapper around ai4bharat/indic-parler-tts.

    Two tokenizers are required:
      - prompt_tokenizer : encodes the target-language text (Telugu)
      - desc_tokenizer   : encodes the English speaker description
                           (google/flan-t5-large, the text encoder inside the model)
    """

    def __init__(self, cfg: Config = default_config):
        self.cfg = cfg
        self.device = cfg.model.device
        self.model: Optional[ParlerTTSForConditionalGeneration] = None
        self.prompt_tokenizer: Optional[AutoTokenizer] = None
        self.desc_tokenizer: Optional[AutoTokenizer] = None
        self.sampling_rate: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        model_id = self.cfg.model.model_id

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.cfg.model.torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.sampling_rate = self.model.config.sampling_rate

        # Prompt tokenizer: handles Telugu script
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Description tokenizer: must match the text encoder (flan-t5-large)
        desc_model_id = self.model.config.text_encoder._name_or_path
        self.desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_id)

    def _check_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Call load() before generating audio.")

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str, description: str):
        desc_ids = self.desc_tokenizer(
            description,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        prompt_ids = self.prompt_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        return desc_ids, prompt_ids

    # ------------------------------------------------------------------
    # Full synthesis — direct model.generate() for best quality
    # ------------------------------------------------------------------

    def generate_full(
        self,
        text: str,
        description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return complete audio as a 1-D float32 numpy array at model sample rate.
        Uses direct model.generate() (no streamer) to avoid chunk-boundary artifacts.
        """
        self._check_loaded()
        if description is None:
            description = self.cfg.telugu.speaker_description

        desc_ids, prompt_ids = self._tokenize(text, description)

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
                do_sample=True,
                temperature=1.0,
                min_new_tokens=10,
                max_new_tokens=2048,
            )

        # generation shape: (batch=1, samples) — squeeze to 1-D
        return generation.cpu().float().numpy().squeeze()

    # ------------------------------------------------------------------
    # Streaming synthesis — for the /synthesize endpoint
    # ------------------------------------------------------------------

    def _run_generate(self, kwargs: dict) -> None:
        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate_streaming(
        self,
        text: str,
        description: Optional[str] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield float32 audio chunks (model native sample rate) as they are decoded.
        Resample and encode in AudioProcessor before sending to clients.
        """
        self._check_loaded()
        if description is None:
            description = self.cfg.telugu.speaker_description

        desc_ids, prompt_ids = self._tokenize(text, description)

        frame_rate = self.model.audio_encoder.config.frame_rate
        play_steps = int(frame_rate * self.cfg.play_steps_in_s)

        streamer = ParlerTTSStreamer(
            self.model,
            device=self.device,
            play_steps=play_steps,
        )

        generation_kwargs = dict(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
            max_new_tokens=2048,
        )

        thread = threading.Thread(target=self._run_generate, args=(generation_kwargs,))
        thread.start()

        for chunk in streamer:
            if chunk is not None and chunk.shape[0] > 0:
                yield chunk.astype(np.float32)

        thread.join()
