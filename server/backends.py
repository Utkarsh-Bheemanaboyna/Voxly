"""Whisper backends behind one small interface.

Every backend takes float32 mono 16 kHz audio and returns a list of Words with
start/end times in seconds, relative to the start of the audio it was given.
Keeping this interface tiny lets the streaming logic (streaming.py) stay
independent of which Whisper implementation runs underneath.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class Word:
    text: str      # Whisper words usually carry a leading space: " hello"
    start: float   # seconds
    end: float     # seconds
    prob: float = 1.0


class Backend:
    name = "base"

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> list[Word]:
        raise NotImplementedError


class FasterWhisperBackend(Backend):
    """CTranslate2 Whisper. Runs on CPU anywhere (int8), CUDA on NVIDIA."""

    def __init__(self, model_size: str = "base.en", device: str = "cpu",
                 compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{model_size}:{compute_type}"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> list[Word]:
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,                      # greedy: fastest, fine for dictation
            word_timestamps=True,             # needed for LocalAgreement + trimming
            initial_prompt=prompt or None,    # previous text as context
            condition_on_previous_text=False,
            vad_filter=False,                 # push-to-talk already bounds speech
        )
        words: list[Word] = []
        for seg in segments:
            if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
                continue  # likely hallucination on non-speech
            for w in seg.words or []:
                words.append(Word(w.word, w.start, w.end, w.probability))
        return words


class MLXWhisperBackend(Backend):
    """Apple-Silicon-native Whisper (runs on the Mac GPU via MLX)."""

    def __init__(self, repo: str = "mlx-community/whisper-base.en-mlx"):
        import mlx_whisper  # noqa: F401  (import check)

        self.name = f"mlx-whisper:{repo.split('/')[-1]}"
        self.repo = repo

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> list[Word]:
        import mlx_whisper

        out = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language="en",
            word_timestamps=True,
            initial_prompt=prompt or None,
            condition_on_previous_text=False,
            verbose=None,
        )
        words: list[Word] = []
        for seg in out.get("segments", []):
            for w in seg.get("words", []):
                words.append(Word(w["word"], w["start"], w["end"], w.get("probability", 1.0)))
        return words


def make_backend(kind: str, model: str) -> Backend:
    if kind == "faster-whisper":
        return FasterWhisperBackend(model)
    if kind == "mlx":
        return MLXWhisperBackend(model)
    raise ValueError(f"unknown backend {kind!r} (use 'faster-whisper' or 'mlx')")
