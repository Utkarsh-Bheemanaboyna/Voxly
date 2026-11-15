"""Voxly STT server: WebSocket in (PCM16 audio), JSON out (partials + finals).

Protocol (one WebSocket connection per client):
  client -> server  text   {"type": "start"}                 begin an utterance
  client -> server  binary raw PCM16 little-endian, mono, 16 kHz
  client -> server  text   {"type": "stop"}                  key released
  server -> client  text   {"type": "ready", "model": ...}
  server -> client  text   {"type": "partial", "committed", "tentative",
                            "audio_end_ms", "compute_ms"}
  server -> client  text   {"type": "final", "text", "audio_ms", "compute_ms"}

Run:  python stt_server.py --backend faster-whisper --model base.en
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from websockets.asyncio.server import ServerConnection, serve

from backends import Backend, make_backend
from streaming import StreamingTranscriber

log = logging.getLogger("voxly.stt")

# One model shared by all connections. The lock makes sure only one inference
# runs at a time (Whisper on a laptop is compute-bound; parallel runs just slow
# each other down).
MODEL_LOCK = asyncio.Lock()


async def run_locked(fn):
    async with MODEL_LOCK:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(fn)  # keep the event loop free for I/O
        return result, (time.perf_counter() - t0) * 1000


async def handle(ws: ServerConnection, backend: Backend, args) -> None:
    st = StreamingTranscriber(backend, min_chunk_s=args.min_chunk,
                              max_buffer_s=args.max_buffer)
    active = False
    worker: asyncio.Task | None = None

    async def partial_loop():
        # Runs while the key is held: whenever enough new audio has arrived,
        # do one streaming step and push a partial to the client.
        # We stop it with the `active` flag, NOT task.cancel(): a cancelled
        # await does not stop the worker thread, which would then race finalize().
        while active:
            if st.ready():
                p, ms = await run_locked(st.process)
                if not active:
                    break  # key released mid-step; finalize() will cover it
                await ws.send(json.dumps({
                    "type": "partial", "committed": p.committed, "tentative": p.tentative,
                    "audio_end_ms": round(p.audio_end_s * 1000), "compute_ms": round(ms, 1),
                }))
            else:
                await asyncio.sleep(0.02)

    await ws.send(json.dumps({"type": "ready", "model": backend.name}))
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                if active:
                    st.insert_pcm16(msg)
                continue
            data = json.loads(msg)
            if data["type"] == "start":
                st.reset()
                active = True
                worker = asyncio.create_task(partial_loop())
            elif data["type"] == "stop" and active:
                active = False
                if worker:
                    # wait for an in-flight step so state is consistent
                    await asyncio.gather(worker, return_exceptions=True)
                audio_ms = round(st.audio_end_s * 1000)
                text, ms = await run_locked(st.finalize)
                await ws.send(json.dumps({
                    "type": "final", "text": text,
                    "audio_ms": audio_ms, "compute_ms": round(ms, 1),
                }))
                log.info("final (%d ms audio, %.0f ms compute): %s", audio_ms, ms, text)
    finally:
        if worker:
            worker.cancel()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="faster-whisper", choices=["faster-whisper", "mlx"])
    ap.add_argument("--model", default="base.en",
                    help="faster-whisper size (tiny.en/base.en/small.en/large-v3-turbo) "
                         "or MLX repo (mlx-community/whisper-base.en-mlx)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--min-chunk", type=float, default=0.5,
                    help="seconds of new audio between partial updates")
    ap.add_argument("--max-buffer", type=float, default=15.0,
                    help="trim the rolling buffer beyond this many seconds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    log.info("loading %s %s ...", args.backend, args.model)
    backend = make_backend(args.backend, args.model)
    # warm-up: the first inference is slow (allocations, kernel compile)
    import numpy as np
    backend.transcribe(np.zeros(16000, dtype=np.float32))
    log.info("ready on ws://%s:%d", args.host, args.port)

    async with serve(lambda ws: handle(ws, backend, args), args.host, args.port,
                     max_size=2**20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
