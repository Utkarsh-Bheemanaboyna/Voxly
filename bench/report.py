"""Summarize latency CSVs: n, mean, p50, p90, p95, p99 per kind; mean WER.

Works on bench/results/*.csv (from stream_wav.py) and on ~/.voxly/latency.csv
(logged by the live client while you use the app).

  python report.py results/latency_base.en.csv
  python report.py ~/.voxly/latency.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def summarize(path):
    lat, wers = defaultdict(list), []
    with open(path) as f:
        for row in csv.DictReader(f):
            lat[row["kind"]].append(float(row["latency_ms"]))
            if row.get("wer") not in (None, ""):
                wers.append(float(row["wer"]))
    print(f"\n{path}")
    print(f"{'kind':8} {'n':>5} {'mean':>7} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7}  (ms)")
    for kind, xs in sorted(lat.items()):
        a = np.array(xs)
        p = np.percentile(a, [50, 90, 95, 99])
        print(f"{kind:8} {len(a):5d} {a.mean():7.0f} {p[0]:7.0f} {p[1]:7.0f} {p[2]:7.0f} {p[3]:7.0f}")
    if wers:
        print(f"mean WER over {len(wers)} utterances: {np.mean(wers):.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
