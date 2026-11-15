"""Latency + accuracy benchmark: stream WAV files to the STT server in real time.

This simulates a user holding the key while speaking: audio is sent in 50 ms
chunks at real-time pace, then "stop" is sent. We record:
  partial latency = time from sending the chunk that ends at position X
                    to receiving a partial that covers audio up to X
  final latency   = time from sending "stop" (end of speech) to the final text
  WER             = word error rate of the final text vs the reference .txt

Usage:
  python stream_wav.py data/my_clips --runs 3 --tag base.en
  python report.py results/*.csv
"""
import argparse
import asyncio
import csv
import json
import time
import wave
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

SR = 16000
CHUNK_MS = 50


def load_wav(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert sw == 2, "need 16-bit PCM WAV"
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:  # simple linear resample; fine for a benchmark
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x.astype(np.int16).tobytes()


def wer(ref: str, hyp: str) -> float:
    import jiwer

    tf = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                        jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                        jiwer.ReduceToListOfListOfWords()])
    return jiwer.wer(ref, hyp, reference_transform=tf, hypothesis_transform=tf)


async def stream_one(ws, pcm: bytes):
    step = SR * CHUNK_MS // 1000 * 2
    sent: list[tuple[int, float]] = []
    partials: list[float] = []
    final = {}

    async def receiver():
        async for raw in ws:
            m = json.loads(raw)
            now = time.perf_counter() * 1000
            if m["type"] == "partial":
                for ms, t in sent:
                    if ms >= m["audio_end_ms"]:
                        partials.append(now - t)
                        break
            elif m["type"] == "final":
                final.update(m, recv=now)
                return

    rx = asyncio.create_task(receiver())
    await ws.send(json.dumps({"type": "start"}))
    t0 = time.perf_counter()
    for i in range(0, len(pcm), step):
        await ws.send(pcm[i:i + step])
        audio_ms = (i + step) // 2 * 1000 // SR
        sent.append((audio_ms, time.perf_counter() * 1000))
        # real-time pacing
        await asyncio.sleep(max(0.0, t0 + audio_ms / 1000 - time.perf_counter()))
    t_stop = time.perf_counter() * 1000
    await ws.send(json.dumps({"type": "stop"}))
    await rx
    return partials, final["recv"] - t_stop, final


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="folder with NN.wav + NN.txt")
    ap.add_argument("--stt", default="ws://127.0.0.1:8765")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default="", help="label for this config, e.g. base.en")
    args = ap.parse_args()

    wavs = sorted(Path(args.data).glob("*.wav"))
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    async with connect(args.stt, max_size=2**20) as ws:
        model = json.loads(await ws.recv())["model"]
        tag = args.tag or model.replace(":", "_")
        path = out / f"latency_{tag}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "run", "kind", "latency_ms", "audio_ms", "compute_ms", "wer", "text"])
            for run in range(args.runs):
                for wav in wavs:
                    pcm = load_wav(wav)
                    partials, final_ms, final = await stream_one(ws, pcm)
                    ref_path = wav.with_suffix(".txt")
                    e = wer(ref_path.read_text(), final["text"]) if ref_path.exists() else ""
                    for p in partials:
                        w.writerow([wav.name, run, "partial", round(p, 1), "", "", "", ""])
                    w.writerow([wav.name, run, "final", round(final_ms, 1), final["audio_ms"],
                                final["compute_ms"], round(e, 4) if e != "" else "", final["text"]])
                    print(f"run{run} {wav.name}: final {final_ms:.0f} ms, "
                          f"{len(partials)} partials, WER {e if e == '' else round(e, 3)}")
    print(f"\nwrote {path}  ->  python report.py {path}")


if __name__ == "__main__":
    asyncio.run(main())
