"""Measure dictation vs typing speed on the same passages (you are the user).

For each passage:
  TYPING:    press Enter, type the passage here, press Enter. Timed.
  DICTATION: with voxly.py running, click into any text box (e.g. TextEdit),
             hold Right Option and say the passage (you may use several holds),
             then come back here and press Enter. Timing comes from
             ~/.voxly/utterances.jsonl: first key press -> last text inserted.

Reports raw WPM and accuracy-adjusted WPM (= WPM x (1 - WER)) for both, and
the speed ratio. Correction time is NOT included for either method; mention
that as a limitation.
"""
import json
import time
from pathlib import Path

import jiwer
import numpy as np

PASSAGES = [
    "Hi team, quick update on the release. The latency fix is merged and the "
    "dashboard looks stable. I will run the full evaluation tonight and share "
    "the numbers tomorrow morning.",
    "Thanks for the interview invite. I am available on Thursday after two in "
    "the afternoon or any time on Friday. Please let me know which slot works "
    "best for you.",
    "The main issue was that the model kept hallucinating text during silence, "
    "so I added an energy check before transcription and the problem went away "
    "in most cases.",
]
LOG = Path.home() / ".voxly" / "utterances.jsonl"
TF = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                    jiwer.ReduceToListOfListOfWords()])


def wer(ref, hyp):
    return jiwer.wer(ref, hyp, reference_transform=TF, hypothesis_transform=TF)


def new_rows(offset):
    if not LOG.exists():
        return [], 0
    with open(LOG) as f:
        f.seek(offset)
        rows = [json.loads(l) for l in f if l.strip()]
        return rows, f.tell()


def main():
    results = []
    for i, ref in enumerate(PASSAGES):
        n_words = len(ref.split())
        print(f"\n=== Passage {i+1}/{len(PASSAGES)} ({n_words} words) ===\n{ref}\n")

        input("TYPING: press Enter, then type the passage and press Enter.")
        t0 = time.perf_counter()
        typed = input("> ")
        t_type = time.perf_counter() - t0
        e_type = wer(ref, typed)

        _, offset = new_rows(0)
        input("DICTATION: dictate it into a text box with Right Option, "
              "then press Enter here.")
        rows, _ = new_rows(offset)
        rows = [r for r in rows if r.get("mode") == "dictate"]
        if not rows:
            print("no dictation found in the log; skipping")
            continue
        start = rows[0]["ts"] - rows[0]["press_to_text_ms"] / 1000
        t_dict = rows[-1]["ts"] - start
        dictated = " ".join(r["text"] for r in rows)
        e_dict = wer(ref, dictated)

        r = dict(words=n_words, t_type=t_type, wer_type=e_type, t_dict=t_dict, wer_dict=e_dict)
        results.append(r)
        print(f"typing:    {60*n_words/t_type:5.1f} WPM, WER {e_type:.3f}")
        print(f"dictation: {60*n_words/t_dict:5.1f} WPM, WER {e_dict:.3f}")

    if not results:
        return
    W = sum(r["words"] for r in results)
    tt = sum(r["t_type"] for r in results)
    td = sum(r["t_dict"] for r in results)
    wt = np.mean([r["wer_type"] for r in results])
    wd = np.mean([r["wer_dict"] for r in results])
    raw_t, raw_d = 60 * W / tt, 60 * W / td
    adj_t, adj_d = raw_t * (1 - wt), raw_d * (1 - wd)
    print("\n=== Summary ===")
    print(f"typing     raw {raw_t:5.1f} WPM | WER {wt:.3f} | adjusted {adj_t:5.1f} WPM")
    print(f"dictation  raw {raw_d:5.1f} WPM | WER {wd:.3f} | adjusted {adj_d:5.1f} WPM")
    print(f"speed-up   raw {raw_d/raw_t:.2f}x | accuracy-adjusted {adj_d/adj_t:.2f}x")
    print(f"(n = {len(results)} passages, {W} words, one user; corrections not timed)")


if __name__ == "__main__":
    main()
