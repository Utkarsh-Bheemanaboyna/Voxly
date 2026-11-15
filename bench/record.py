"""Record a small dictation test set: one WAV + reference TXT per sentence.

Usage:  python record.py --out data/my_clips
For each sentence: press Enter, read it aloud, press Enter again.
"""
import argparse
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SENTENCES = [
    "Hey, can we move tomorrow's meeting to three thirty in the afternoon?",
    "I pushed the fix to the main branch, please review it when you get a chance.",
    "The model's validation loss stopped improving after the fourth epoch.",
    "Let's grab lunch near the library on Friday if you are free.",
    "Please send me the final version of the report before Monday morning.",
    "I think the latency problem comes from the network, not the model.",
    "Thanks for the feedback, I will update the slides tonight.",
    "Our team reduced the error rate by retraining on cleaner data.",
    "Can you share the link to the dataset and the evaluation script?",
    "I'm running about ten minutes late, start without me.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "We should cache the embeddings so we don't recompute them every request.",
    "Remind me to call the bank and pay the electricity bill.",
    "The interview is scheduled for Thursday at eleven in the morning.",
    "Honestly, the second approach is simpler and almost as accurate.",
    "Please add error handling for the case where the file is missing.",
    "I'll be presenting the poster in the afternoon session.",
    "Could you resend the invite, I can't find it in my inbox?",
    "The battery drains faster when the app keeps the microphone open.",
    "Let me know if you need anything else from my side.",
]


def record_until_enter(sr=16000) -> np.ndarray:
    chunks, stop = [], threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="int16", callback=cb):
        input()  # second Enter stops
    return np.concatenate(chunks)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/my_clips")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(SENTENCES):
        input(f"\n[{i+1}/{len(SENTENCES)}] Press Enter, then read:\n   {s}\n")
        print("   recording... press Enter when done")
        audio = record_until_enter()
        with wave.open(str(out / f"{i:02d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())
        (out / f"{i:02d}.txt").write_text(s)
    print(f"saved {len(SENTENCES)} clips to {out}")


if __name__ == "__main__":
    main()
