"""Tests for the streaming logic and the WebSocket server, using a fake model.

The fake model mimics Whisper's relevant behaviour:
  * words come with timestamps (0.4 s per word here)
  * the last, partly-heard word is often wrong ("unstable tail")
Run:  cd voxly && python -m pytest tests -q
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from backends import SAMPLE_RATE, Backend, Word  # noqa: E402
from streaming import StreamingTranscriber  # noqa: E402

SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank".split()
SEC_PER_WORD = 0.4


class FakeWhisper(Backend):
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.offset = 0.0  # set by the test so the fake knows absolute time

    def transcribe(self, audio, prompt=""):
        self.calls += 1
        dur = len(audio) / SAMPLE_RATE
        out = []
        t = 0.0
        # which sentence word does buffer[0] correspond to?
        first = int(round(self.offset / SEC_PER_WORD))
        for i, w in enumerate(SENTENCE[first:], start=first):
            start, end = t, t + SEC_PER_WORD
            if end > dur + 1e-6:
                if start < dur:  # partly heard: unstable, wrong guess
                    out.append(Word(" " + w[:2] + "?", start, dur))
                break
            out.append(Word(" " + w, start, end))
            t = end
        return out


def tone(seconds):
    n = int(seconds * SAMPLE_RATE)
    x = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / SAMPLE_RATE)
    return (x * 32767).astype(np.int16).tobytes()


def run_stream(st, fake, total_s, chunk_s=0.1):
    partials = []
    for _ in range(int(round(total_s / chunk_s))):
        st.insert_pcm16(tone(chunk_s))
        if st.ready():
            fake.offset = st.buffer_offset
            partials.append(st.process())
    fake.offset = st.buffer_offset
    return partials, st.finalize()


def test_final_text_is_complete_and_has_no_duplicates():
    fake = FakeWhisper()
    st = StreamingTranscriber(fake, min_chunk_s=0.5, max_buffer_s=15)
    partials, final = run_stream(st, fake, len(SENTENCE) * SEC_PER_WORD)
    assert final == " ".join(SENTENCE)
    assert len(partials) >= 5


def test_committed_text_only_grows_and_never_contains_unstable_words():
    fake = FakeWhisper()
    st = StreamingTranscriber(fake, min_chunk_s=0.5)
    partials, _ = run_stream(st, fake, len(SENTENCE) * SEC_PER_WORD)
    prev = ""
    for p in partials:
        assert p.committed.startswith(prev)   # committed never changes
        assert "?" not in p.committed         # unstable guesses never committed
        prev = p.committed
    assert len(prev.split()) >= len(SENTENCE) - 3


def test_buffer_trimming_bounds_compute():
    fake = FakeWhisper()
    st = StreamingTranscriber(fake, min_chunk_s=0.5, max_buffer_s=2.0)
    _, final = run_stream(st, fake, len(SENTENCE) * SEC_PER_WORD)
    assert final == " ".join(SENTENCE)


def test_silence_is_not_transcribed():
    fake = FakeWhisper()
    st = StreamingTranscriber(fake)
    st.insert_pcm16(np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes())
    assert st.finalize() == ""
    assert fake.calls == 0


def test_websocket_server_end_to_end():
    import stt_server
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    class Args:
        min_chunk = 0.5
        max_buffer = 15.0

    fake = FakeWhisper()

    async def go():
        async with serve(lambda ws: stt_server.handle(ws, fake, Args), "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                assert json.loads(await ws.recv())["type"] == "ready"
                await ws.send(json.dumps({"type": "start"}))
                for _ in range(20):  # 2 s of audio in 100 ms chunks
                    await ws.send(tone(0.1))
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.2)
                await ws.send(json.dumps({"type": "stop"}))
                msgs = []
                while True:
                    m = json.loads(await ws.recv())
                    msgs.append(m)
                    if m["type"] == "final":
                        break
                return msgs

    msgs = asyncio.run(go())
    assert any(m["type"] == "partial" for m in msgs)
    assert msgs[-1]["text"] == " ".join(SENTENCE[:5])
