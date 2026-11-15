"""Voxly client (macOS): hold a key, speak, text appears at your cursor.

  Hold RIGHT OPTION  -> dictation: final transcript is typed at the cursor
  Hold RIGHT COMMAND -> command:   transcript goes to the LiveKit agent
                                   ("fix the grammar", "summarize what I just said")

Pipeline:
  mic (sounddevice, PCM16 16 kHz mono, 50 ms blocks)
    -> WebSocket -> STT server (streaming Whisper)  -> partials shown live
    -> final transcript
         dictation: Accessibility insert at cursor
         command:   LiveKit room -> agent (tool-calling LLM) -> RPC back to us
                    -> Accessibility insert

Every utterance is logged to ~/.voxly/ for the latency report (bench/report.py).

Run:  python voxly.py                 (dictation only)
      python voxly.py --agent         (also connect the LiveKit agent)
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard
from websockets.asyncio.client import connect

import insert

SAMPLE_RATE = 16_000
BLOCK_MS = 50                                  # smaller block = lower latency, more messages
BLOCK = SAMPLE_RATE * BLOCK_MS // 1000
PRE_ROLL_MS = 200                              # audio kept from *before* the key press
LOG_DIR = Path.home() / ".voxly"


def now_ms() -> float:
    return time.perf_counter() * 1000


class Logger:
    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        self.lat_path = LOG_DIR / "latency.csv"
        new = not self.lat_path.exists()
        self.lat = open(self.lat_path, "a", newline="")
        self.w = csv.writer(self.lat)
        if new:
            self.w.writerow(["ts", "kind", "latency_ms", "audio_ms", "compute_ms", "model"])
        self.utt = open(LOG_DIR / "utterances.jsonl", "a")

    def latency(self, kind, latency_ms, audio_ms, compute_ms, model):
        self.w.writerow([round(time.time(), 3), kind, round(latency_ms, 1), audio_ms,
                         compute_ms, model])
        self.lat.flush()

    def utterance(self, **row):
        self.utt.write(json.dumps({"ts": round(time.time(), 3), **row}) + "\n")
        self.utt.flush()


class Voxly:
    def __init__(self, args):
        self.args = args
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self.model = "?"
        self.log = Logger()
        self.recording = False
        self.mode = "dictate"
        self.pre_roll = collections.deque(maxlen=PRE_ROLL_MS // BLOCK_MS)
        self.audio_q: asyncio.Queue[bytes | str] = asyncio.Queue()
        # for partial latency: (cumulative audio ms sent, time it was sent)
        self.sent: list[tuple[int, float]] = []
        self.sent_ms = 0
        self.t_press = self.t_stop = 0.0
        self.history: list[str] = []          # recent dictated segments (agent tool)
        self.agent = None

    # ------------------------------------------------------------ audio
    def _mic_cb(self, indata, frames, t, status):
        # Runs on PortAudio's thread: hand bytes to the asyncio loop.
        self.loop.call_soon_threadsafe(self._on_block, bytes(indata))

    def _on_block(self, pcm: bytes):
        if self.recording:
            self.audio_q.put_nowait(pcm)
        else:
            self.pre_roll.append(pcm)

    async def _sender(self):
        # Control messages go through the same queue as audio, so "start" is
        # always sent before the first chunk and "stop" after the last one.
        while True:
            item = await self.audio_q.get()
            if isinstance(item, str):
                if '"stop"' in item:
                    self.t_stop = now_ms()   # latency clock starts at end of audio
                await self.ws.send(item)
                continue
            await self.ws.send(item)
            self.sent_ms += len(item) // 2 * 1000 // SAMPLE_RATE
            self.sent.append((self.sent_ms, now_ms()))

    # ---------------------------------------------------------- hotkeys
    def _key_down(self, key):
        mode = {keyboard.Key.alt_r: "dictate", keyboard.Key.cmd_r: "command"}.get(key)
        if mode and not self.recording:     # ignore key auto-repeat
            self.loop.call_soon_threadsafe(self._start, mode)

    def _key_up(self, key):
        if key in (keyboard.Key.alt_r, keyboard.Key.cmd_r) and self.recording:
            self.loop.call_soon_threadsafe(self._stop)

    def _start(self, mode):
        if self.recording:
            return
        if mode == "command" and not self.agent:
            print("\n[command mode needs --agent]")
            return
        self.mode, self.recording = mode, True
        self.t_press = now_ms()
        self.sent, self.sent_ms = [], 0
        self.audio_q.put_nowait(json.dumps({"type": "start"}))
        for pcm in self.pre_roll:            # don't clip the first syllable
            self.audio_q.put_nowait(pcm)
        self.pre_roll.clear()
        print(f"\n● {mode}…", end="", flush=True)

    def _stop(self):
        if not self.recording:
            return
        self.recording = False
        self.audio_q.put_nowait(json.dumps({"type": "stop"}))

    # ----------------------------------------------------- server events
    def _partial_latency(self, audio_end_ms: int) -> float | None:
        # time since the chunk containing the last processed audio was sent
        for ms, t in self.sent:
            if ms >= audio_end_ms:
                return now_ms() - t
        return None

    async def _receiver(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if m["type"] == "ready":
                self.model = m["model"]
                print(f"STT ready: {self.model}")
            elif m["type"] == "partial" and self.recording:
                lat = self._partial_latency(m["audio_end_ms"])
                if lat is not None:
                    self.log.latency("partial", lat, m["audio_end_ms"], m["compute_ms"], self.model)
                # live overlay (terminal): committed text + dim tentative tail
                sys.stdout.write(f"\r\033[K● {m['committed']} \033[2m{m['tentative']}\033[0m")
                sys.stdout.flush()
            elif m["type"] == "final":
                await self._on_final(m)

    async def _on_final(self, m):
        t_final = now_ms()
        final_lat = t_final - self.t_stop
        self.log.latency("final", final_lat, m["audio_ms"], m["compute_ms"], self.model)
        text = m["text"].strip()
        print(f"\r\033[K✓ {text}   \033[2m[final {final_lat:.0f} ms]\033[0m")
        if not text:
            return
        row = dict(mode=self.mode, text=text, hold_ms=round(self.t_stop - self.t_press),
                   final_latency_ms=round(final_lat), audio_ms=m["audio_ms"], model=self.model)
        if self.mode == "dictate":
            t0 = now_ms()
            method = await asyncio.to_thread(insert.insert_text, text + " ")
            row.update(insert_ms=round(now_ms() - t0, 1), method=method,
                       press_to_text_ms=round(now_ms() - self.t_press))
            self.history.append(text)
            self.history = self.history[-20:]
        else:
            row.update(sent_to_agent=True)
            await self.agent.send_command(text)
        self.log.utterance(**row)

    # -------------------------------------------------------------- main
    async def run(self):
        self.loop = asyncio.get_running_loop()
        if not insert.trusted():
            print("⚠ Accessibility permission missing: System Settings > Privacy & Security "
                  "> Accessibility > enable your terminal. Falling back to paste only.")
        if self.args.agent:
            from agent_link import AgentLink
            self.agent = AgentLink(self)
            await self.agent.connect()
        async with connect(self.args.stt, max_size=2**20) as ws:
            self.ws = ws
            stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                       blocksize=BLOCK, callback=self._mic_cb)
            listener = keyboard.Listener(on_press=self._key_down, on_release=self._key_up)
            with stream:
                listener.start()
                print("Hold RIGHT OPTION to dictate"
                      + (", RIGHT COMMAND for agent commands" if self.agent else "")
                      + ". Ctrl+C to quit.")
                await asyncio.gather(self._sender(), self._receiver())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default=os.getenv("VOXLY_STT", "ws://127.0.0.1:8765"))
    ap.add_argument("--agent", action="store_true", help="connect to the LiveKit agent")
    args = ap.parse_args()
    try:
        asyncio.run(Voxly(args).run())
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
