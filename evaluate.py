#!/usr/bin/env python3
"""Evaluate video-time predictions against manually labeled ground truth.

Predictions CSV from --history-csv: time_s, station_id, predicted_state, reason
Truth CSV: station_id, start_s, end_s, state (free/occupied/unknown)

Sampling is approximately once per second per station. These are sampled
rather than duration-weighted metrics; report unknown coverage separately.
"""
import argparse
import csv
import json
from collections import defaultdict

STATES = {"free", "occupied", "unknown"}


def evaluate(prediction_rows, truth_rows):
    intervals = defaultdict(list)
    for row in truth_rows:
        station = row["station_id"]
        start, end, state = float(row["start_s"]), float(row["end_s"]), row["state"].lower()
        if state not in STATES or end <= start:
            raise ValueError(f"Invalid truth interval for station {station}: {row}")
        intervals[station].append((start, end, state))
    for st, entries in intervals.items():
        entries.sort()
        if any(entries[i][1] > entries[i+1][0] for i in range(len(entries)-1)):
            raise ValueError(f"Overlapping truth intervals for {st}")
    counts = defaultdict(lambda: defaultdict(int))
    for row in prediction_rows:
        st = row["station_id"]
        if st not in intervals:
            continue
        t, pred = float(row["time_s"]), row["predicted_state"].lower()
        if pred not in STATES:
            raise ValueError(f"Invalid prediction state: {pred}")
        true = next((s for start, end, s in intervals[st] if start <= t < end), None)
        if true in (None, "unknown"):
            continue
        counts[st][(true, pred)] += 1
    output = {}
    for st, pair_counts in counts.items():
        total = sum(pair_counts.values())
        tf = pair_counts[("free", "free")]
        fo = pair_counts[("occupied", "free")]  # dangerous false-free
        to = pair_counts[("occupied", "occupied")]
        occ_truth = sum(v for (truth, _), v in pair_counts.items() if truth == "occupied")
        unknown = sum(v for (_, pred), v in pair_counts.items() if pred == "unknown")
        def ratio(a, b):
            return round(a / b, 4) if b else None
        output[st] = {
            "labeled_samples": total,
            "unknown_fraction": ratio(unknown, total),
            "free_precision": ratio(tf, tf+fo),
            "false_free_rate_among_occupied": ratio(fo, occ_truth),
            "occupied_recall_including_unknown": ratio(to, occ_truth),
        }
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    with open(args.predictions, newline="", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    with open(args.truth, newline="", encoding="utf-8") as f:
        truth = list(csv.DictReader(f))
    print(json.dumps(evaluate(preds, truth), indent=2))


if __name__ == "__main__":
    main()
