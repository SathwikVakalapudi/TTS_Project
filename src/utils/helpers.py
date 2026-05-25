import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceMetrics:
    text: str
    ttfa_s: Optional[float] = None          # time-to-first-audio-chunk
    total_inference_s: Optional[float] = None
    audio_duration_s: Optional[float] = None
    rtf: Optional[float] = None             # real-time factor = inference / audio

    def compute_rtf(self) -> None:
        if self.total_inference_s and self.audio_duration_s:
            self.rtf = self.total_inference_s / self.audio_duration_s

    def as_dict(self) -> dict:
        return {
            "ttfa_s": round(self.ttfa_s, 3) if self.ttfa_s is not None else None,
            "total_inference_s": (
                round(self.total_inference_s, 3)
                if self.total_inference_s is not None
                else None
            ),
            "audio_duration_s": (
                round(self.audio_duration_s, 3)
                if self.audio_duration_s is not None
                else None
            ),
            "rtf": round(self.rtf, 4) if self.rtf is not None else None,
        }


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer not started.")
        return time.perf_counter() - self._start
