"""Streaming transcription on top of a non-streaming model (Whisper).

Whisper looks at a whole window of audio and then writes text, so it has no
native notion of "partial results". We fake streaming like this:

  1. Keep a rolling buffer of the audio received so far.
  2. Every `min_chunk_s` of new audio, re-transcribe the whole buffer.
  3. LocalAgreement-2: words on which two consecutive hypotheses agree are
     *committed* (they will never change). The rest is shown as *tentative*.
  4. When the buffer gets long, cut it at the end of the last committed word,
     so compute per step stays bounded instead of growing with utterance length.
  5. On stop (key released), transcribe what is left once more and return
     committed + remainder as the final text.

Idea follows Machacek et al., "Turning Whisper into Real-Time Transcription
System" (2023, whisper_streaming).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from backends import SAMPLE_RATE, Backend, Word

_PUNCT = re.compile(r"[^\w']+")


def norm(w: str) -> str:
    return _PUNCT.sub("", w.lower())


def join(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


@dataclass
class Partial:
    committed: str
    tentative: str
    audio_end_s: float  # absolute position (seconds) of audio this result covers


class StreamingTranscriber:
    def __init__(self, backend: Backend, min_chunk_s: float = 0.5,
                 max_buffer_s: float = 15.0, silence_rms: float = 0.004):
        self.backend = backend
        self.min_chunk_s = min_chunk_s
        self.max_buffer_s = max_buffer_s
        self.silence_rms = silence_rms
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_offset = 0.0           # absolute time of buffer[0]
        self.total_samples = 0             # all samples ever received
        self.processed_samples = 0         # total_samples at last process()
        self.committed: list[Word] = []
        self.prev_hyp: list[Word] = []     # last uncommitted hypothesis

    @property
    def audio_end_s(self) -> float:
        return self.total_samples / SAMPLE_RATE

    def insert_pcm16(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, samples])
        self.total_samples += len(samples)

    def ready(self) -> bool:
        return (self.total_samples - self.processed_samples) >= self.min_chunk_s * SAMPLE_RATE

    # ------------------------------------------------------------- helpers
    def _prompt(self) -> str:
        # last ~200 chars of committed text as context for the decoder
        return join(self.committed)[-200:]

    def _last_committed_end(self) -> float:
        return self.committed[-1].end if self.committed else 0.0

    def _run_model(self) -> list[Word]:
        if len(self.buffer) == 0:
            return []
        if float(np.sqrt(np.mean(self.buffer ** 2))) < self.silence_rms:
            return []  # silence: Whisper hallucinates ("Thank you.") on silence
        words = self.backend.transcribe(self.buffer, self._prompt())
        for w in words:  # make times absolute
            w.start += self.buffer_offset
            w.end += self.buffer_offset
        # drop words that lie in audio we already committed
        t = self._last_committed_end()
        words = [w for w in words if w.start > t - 0.1]
        # drop an n-gram repeated across the commit boundary (1..5 words)
        if self.committed and words:
            for n in range(min(5, len(self.committed), len(words)), 0, -1):
                tail = [norm(w.text) for w in self.committed[-n:]]
                head = [norm(w.text) for w in words[:n]]
                if tail == head:
                    words = words[n:]
                    break
        return words

    def _trim(self) -> None:
        if len(self.buffer) / SAMPLE_RATE <= self.max_buffer_s or not self.committed:
            return
        cut_t = self._last_committed_end()
        cut = int((cut_t - self.buffer_offset) * SAMPLE_RATE)
        if 0 < cut < len(self.buffer):
            self.buffer = self.buffer[cut:]
            self.buffer_offset = cut_t

    # ----------------------------------------------------------------- API
    def process(self) -> Partial:
        """One streaming step. Call when ready() is True."""
        self.processed_samples = self.total_samples
        end_s = self.audio_end_s
        new = self._run_model()

        # LocalAgreement-2: commit the longest common prefix of two hypotheses
        k = 0
        while (k < len(new) and k < len(self.prev_hyp)
               and norm(new[k].text) == norm(self.prev_hyp[k].text)):
            k += 1
        self.committed.extend(new[:k])
        self.prev_hyp = new[k:]
        self._trim()
        return Partial(join(self.committed), join(self.prev_hyp), end_s)

    def finalize(self) -> str:
        """Key released: transcribe the remainder once more, return final text."""
        rest = self._run_model()
        text = join(self.committed + rest)
        self.reset()
        return text
