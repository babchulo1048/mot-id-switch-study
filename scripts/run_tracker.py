"""Run the stage 0 tracker on one MOT17 video and save the result.

Example (from the repo folder):
    python -m scripts.run_tracker MOT17-09-FRCNN
"""
import argparse
from pathlib import Path

import numpy as np

from tracker.io_mot import load_detections, read_sequence_length, write_results
from tracker.stage0 import IoUTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", help="for example MOT17-09-FRCNN")
    parser.add_argument("--data", default="data/MOT17/train")
    parser.add_argument("--out", default="outputs/stage0")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=1)
    args = parser.parse_args()

    seq_dir = Path(args.data) / args.sequence
    detections = load_detections(seq_dir / "det" / "det.txt", min_score=args.min_score)
    n_frames = read_sequence_length(seq_dir / "seqinfo.ini")

    tracker = IoUTracker(iou_threshold=args.iou_threshold, max_age=args.max_age)
    no_boxes = np.zeros((0, 5))
    rows = []
    for frame in range(1, n_frames + 1):   # every frame, even frames without boxes
        boxes = detections.get(frame, no_boxes)[:, :4]
        for track_id, box in tracker.update(frame, boxes):
            rows.append((frame, track_id, *box))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sequence}.txt"
    write_results(out_path, rows)

    n_ids = len({row[1] for row in rows})
    print(f"{args.sequence}: {n_frames} frames, {len(rows)} boxes, {n_ids} track IDs -> {out_path}")


if __name__ == "__main__":
    main()
